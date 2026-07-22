"""Export a checkpoint to the browser-native BEM2 binary format."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from browser_embedding.checkpoint import load_checkpoint
from browser_embedding.config import DeploymentConfig, ExperimentConfig
from browser_embedding.model import BrowserEncoder
from browser_embedding.quantization import (
    TernaryLinear,
    pack_int4,
    pack_ternary,
    quantize_embedding_int4,
    ternary_quantize,
)

MAGIC = b"BEM2"
FORMAT_VERSION = 2
HEADER_SIZE = 64
TRAILING_SHA256_SIZE = 32
EMBEDDING_FORMATS = {"int4": 1, "int8": 2, "fp16": 3}
_HEADER = struct.Struct("<4sHHIHHHHHHHBBIHB8H13s")
assert _HEADER.size == HEADER_SIZE


@dataclass(frozen=True)
class Section:
    name: str
    offset: int
    length: int
    encoding: str
    shape: tuple[int, ...]


def _fp16(tensor: Tensor) -> bytes:
    return tensor.detach().cpu().to(torch.float16).contiguous().numpy().tobytes()


def _ternary(layer: TernaryLinear) -> bytes:
    quantized = ternary_quantize(layer.weight)
    return pack_ternary(quantized.values) + struct.pack("<f", float(quantized.scale))


def _layer_norm(layer: nn.LayerNorm) -> bytes:
    assert layer.weight is not None and layer.bias is not None
    return _fp16(layer.weight) + _fp16(layer.bias)


def _embedding(weight: Tensor, deployment: DeploymentConfig, padding_idx: int) -> bytes:
    if deployment.embedding_format == "int4":
        values, scales = quantize_embedding_int4(weight, padding_idx)
        return pack_int4(values) + _fp16(scales)
    if deployment.embedding_format == "int8":
        scales = (weight.detach().abs().amax(dim=1) / 127).clamp(min=1e-8)
        values = (weight.detach() / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
        values[padding_idx].zero_()
        scales[padding_idx] = 0
        return values.cpu().contiguous().numpy().tobytes() + _fp16(scales)
    return _fp16(weight)


def _header(config: ExperimentConfig, body_length: int) -> bytes:
    model = config.model
    dimensions = list(model.matryoshka_dims)
    if len(dimensions) > 8:
        raise ValueError("BEM2 supports at most eight Matryoshka dimensions")
    padded_dimensions = dimensions + [0] * (8 - len(dimensions))
    flags = 0b0000_0011  # shared block + fp16 norm/head
    return _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        HEADER_SIZE,
        model.vocab_size,
        model.max_sequence_length,
        model.embedding_dim,
        model.hidden_dim,
        model.num_heads,
        model.num_repeats,
        model.ffn_dim,
        model.output_dim,
        EMBEDDING_FORMATS[config.deployment.embedding_format],
        flags,
        body_length,
        model.padding_idx,
        len(dimensions),
        *padded_dimensions,
        b"\0" * 13,
    )


def export_model(model: BrowserEncoder, config: ExperimentConfig, output: Path) -> dict[str, Any]:
    """Materialize deployable values and write one checksummed model file."""
    config.deployment.assert_fits(config.model)
    model = model.cpu().eval()
    parts: list[bytes] = []
    sections: list[Section] = []

    def add(name: str, encoding: str, shape: tuple[int, ...], encode: Callable[[], bytes]) -> None:
        value = encode()
        offset = HEADER_SIZE + sum(len(part) for part in parts)
        sections.append(Section(name, offset, len(value), encoding, shape))
        parts.append(value)

    architecture = config.model
    embeddings = model.embeddings
    block = model.shared_block
    add(
        "token_embedding",
        config.deployment.embedding_format,
        tuple(embeddings.token.weight.shape),
        lambda: _embedding(embeddings.token.weight, config.deployment, architecture.padding_idx),
    )
    add(
        "position_embedding",
        "fp16",
        tuple(embeddings.position.weight.shape),
        lambda: _fp16(embeddings.position.weight),
    )
    add(
        "embedding_norm",
        "fp16",
        (2, architecture.embedding_dim),
        lambda: _layer_norm(embeddings.norm),
    )
    add(
        "embedding_projection",
        "ternary2",
        tuple(embeddings.projection.weight.shape),
        lambda: _ternary(embeddings.projection),
    )
    add(
        "attention_norm",
        "fp16",
        (2, architecture.hidden_dim),
        lambda: _layer_norm(block.attention_norm),
    )
    add(
        "attention_qkv",
        "ternary2",
        tuple(block.attention.qkv.weight.shape),
        lambda: _ternary(block.attention.qkv),
    )
    add(
        "attention_output",
        "ternary2",
        tuple(block.attention.output.weight.shape),
        lambda: _ternary(block.attention.output),
    )
    add(
        "ffn_norm",
        "fp16",
        (2, architecture.hidden_dim),
        lambda: _layer_norm(block.ffn_norm),
    )
    add(
        "ffn_up",
        "ternary2",
        tuple(block.ffn_up.weight.shape),
        lambda: _ternary(block.ffn_up),
    )
    add(
        "ffn_down",
        "ternary2",
        tuple(block.ffn_down.weight.shape),
        lambda: _ternary(block.ffn_down),
    )
    add(
        "final_norm",
        "fp16",
        (2, architecture.hidden_dim),
        lambda: _layer_norm(model.final_norm),
    )
    add(
        "output_projection",
        "fp16",
        tuple(model.output_projection.weight.shape),
        lambda: _fp16(model.output_projection.weight) + _fp16(model.output_projection.bias),
    )

    body = b"".join(parts)
    header = _header(config, len(body))
    digest = sha256(header + body).digest()
    binary = header + body + digest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary)

    manifest = {
        "format": "BEM2",
        "format_version": FORMAT_VERSION,
        "sha256": sha256(binary).hexdigest(),
        "bytes": len(binary),
        "architecture": config.model.model_dump(mode="json"),
        "deployment": config.deployment.model_dump(mode="json"),
        "sections": [asdict(section) for section in sections],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    estimate = config.deployment.estimate(config.model).packed_model_bytes
    if len(binary) != estimate:
        raise AssertionError(f"budget estimate drift: estimated {estimate}, exported {len(binary)}")
    return manifest


def materialize_deployment_model(model: BrowserEncoder, config: ExperimentConfig) -> BrowserEncoder:
    """Build the exact dequantized PyTorch reference represented by BEM2."""
    deployed = deepcopy(model).cpu().eval()
    architecture = config.model
    with torch.no_grad():
        token_weight = deployed.embeddings.token.weight
        if config.deployment.embedding_format == "int4":
            values, scales = quantize_embedding_int4(token_weight, architecture.padding_idx)
            token_weight.copy_(values.float() * scales.float()[:, None])
        elif config.deployment.embedding_format == "int8":
            scales = (token_weight.abs().amax(dim=1) / 127).clamp(min=1e-8)
            values = (token_weight / scales[:, None]).round().clamp(-127, 127)
            values[architecture.padding_idx].zero_()
            scales[architecture.padding_idx] = 0
            token_weight.copy_(values * scales.half().float()[:, None])
        else:
            token_weight.copy_(token_weight.half().float())

        deployed.embeddings.position.weight.copy_(
            deployed.embeddings.position.weight.half().float()
        )
        for layer in deployed.modules():
            if isinstance(layer, TernaryLinear):
                quantized = ternary_quantize(layer.weight)
                layer.weight.copy_(quantized.values.float() * quantized.scale)
                layer.set_quantization_strength(0.0)
            elif isinstance(layer, nn.LayerNorm):
                assert layer.weight is not None and layer.bias is not None
                layer.weight.copy_(layer.weight.half().float())
                layer.bias.copy_(layer.bias.half().float())
        deployed.output_projection.weight.copy_(deployed.output_projection.weight.half().float())
        deployed.output_projection.bias.copy_(deployed.output_projection.bias.half().float())
    return deployed


def export_checkpoint(checkpoint_path: Path, output: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = ExperimentConfig.model_validate(checkpoint["experiment"])
    model = BrowserEncoder(config.model)
    model.load_state_dict(checkpoint["state"]["model"])
    return export_model(model, config, output)


def inspect_header(path: Path) -> dict[str, object]:
    binary = path.read_bytes()
    if len(binary) < HEADER_SIZE + TRAILING_SHA256_SIZE:
        raise ValueError("model file is truncated")
    values = _HEADER.unpack(binary[:HEADER_SIZE])
    if values[0] != MAGIC or values[1] != FORMAT_VERSION or values[2] != HEADER_SIZE:
        raise ValueError("unsupported model header")
    body_length = int(values[13])
    expected_size = HEADER_SIZE + body_length + TRAILING_SHA256_SIZE
    if len(binary) != expected_size:
        raise ValueError(f"model size mismatch: expected {expected_size}, got {len(binary)}")
    if sha256(binary[:-32]).digest() != binary[-32:]:
        raise ValueError("model checksum mismatch")
    matryoshka_count = int(values[15])
    return {
        "format": values[0].decode("ascii"),
        "format_version": values[1],
        "vocab_size": values[3],
        "max_sequence_length": values[4],
        "embedding_dim": values[5],
        "hidden_dim": values[6],
        "num_heads": values[7],
        "num_repeats": values[8],
        "ffn_dim": values[9],
        "output_dim": values[10],
        "embedding_format": values[11],
        "body_length": body_length,
        "padding_idx": values[14],
        "matryoshka_dims": list(values[16 : 16 + matryoshka_count]),
        "bytes": len(binary),
    }
