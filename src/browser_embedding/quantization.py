"""Quantization primitives shared by training and export.

There is no module-swapping phase. Deployable matrix multiplications are
``TernaryLinear`` from model construction onward, so fp32 warmup, QAT,
checkpoint resume and export all see the same parameter tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QuantizedTensor:
    values: Tensor
    scale: Tensor


def ternary_quantize(weight: Tensor) -> QuantizedTensor:
    """AbsMean ternarization with a single scale per matrix."""
    scale = weight.detach().abs().mean().clamp(min=1e-8)
    values = (weight.detach() / scale).round().clamp(-1, 1).to(torch.int8)
    return QuantizedTensor(values=values, scale=scale)


class TernaryLinear(nn.Module):
    """Linear layer with a straight-through ternary forward path.

    ``quantization_strength`` is 0 during fp32 warmup and 1 during QAT. A
    scalar interpolation also supports gradual schedules later without
    changing checkpoint structure or the browser graph.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight: nn.Parameter = nn.Parameter(torch.empty(out_features, in_features))
        self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.quantization_strength: Tensor
        self.register_buffer("quantization_strength", torch.tensor(0.0))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        quantized = ternary_quantize(self.weight)
        dequantized = quantized.values.to(self.weight.dtype) * quantized.scale
        strength = self.quantization_strength.to(dtype=self.weight.dtype)
        effective = self.weight + strength * (dequantized - self.weight).detach()
        return F.linear(inputs, effective, self.bias)

    def set_quantization_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("quantization strength must be in [0, 1]")
        self.quantization_strength.fill_(strength)


def set_quantization_strength(module: nn.Module, strength: float) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, TernaryLinear):
            child.set_quantization_strength(strength)
            count += 1
    return count


def ternary_health(module: nn.Module) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, child in module.named_modules():
        if isinstance(child, TernaryLinear):
            values = ternary_quantize(child.weight).values
            result[name] = float((values == 0).float().mean())
    return result


def quantize_embedding_int4(weight: Tensor, padding_idx: int) -> tuple[Tensor, Tensor]:
    """Return signed int4 values and fp16 per-row scales."""
    scales = (weight.detach().abs().amax(dim=1) / 7.0).clamp(min=1e-8)
    values = (weight.detach() / scales[:, None]).round().clamp(-7, 7).to(torch.int8)
    values[padding_idx].zero_()
    scales[padding_idx] = 0
    return values, scales.to(torch.float16)


def pack_int4(values: Tensor) -> bytes:
    if values.ndim != 2 or values.shape[1] % 2:
        raise ValueError("int4 matrix must be rank two with an even row width")
    unsigned = (values.to(torch.int16) & 0x0F).to(torch.uint8)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    return packed.contiguous().cpu().numpy().tobytes()


def pack_ternary(values: Tensor) -> bytes:
    if values.numel() % 4:
        raise ValueError("ternary tensor element count must be divisible by four")
    flat = values.contiguous().view(-1)
    codes = torch.zeros_like(flat, dtype=torch.uint8)
    codes[flat > 0] = 1
    codes[flat < 0] = 2
    groups = codes.view(-1, 4)
    packed = groups[:, 0] | (groups[:, 1] << 2) | (groups[:, 2] << 4) | (groups[:, 3] << 6)
    return packed.cpu().numpy().tobytes()
