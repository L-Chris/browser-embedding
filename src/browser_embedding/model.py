"""Browser-native multilingual encoder.

The graph intentionally uses a small, fixed operator vocabulary that maps
directly to portable WASM kernels: embedding lookup, LayerNorm, ternary
linear, attention, tanh-GELU, masked mean pooling and L2 normalization.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from browser_embedding.config import ModelConfig
from browser_embedding.quantization import TernaryLinear


class FactorizedEmbedding(nn.Module):
    """INT4-friendly low-rank token table projected to the hidden width."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_sequence_length = config.max_sequence_length
        self.token = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.padding_idx
        )
        self.position = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.projection = TernaryLinear(config.embedding_dim, config.hidden_dim)
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token.weight[config.padding_idx].zero_()

    def forward(self, input_ids: Tensor) -> Tensor:
        sequence_length = input_ids.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds {self.max_sequence_length}"
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        embedded = self.token(input_ids) + self.position(positions)[None, :, :]
        projected: Tensor = self.projection(self.norm(embedded))
        return projected


class FusedSelfAttention(nn.Module):
    """Self attention with one QKV projection to reduce dispatch overhead."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim
        self.qkv = TernaryLinear(config.hidden_dim, 3 * config.hidden_dim)
        self.output = TernaryLinear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        batch, sequence, _ = inputs.shape
        qkv = self.qkv(inputs).view(batch, sequence, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~attention_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min
        )
        probabilities = self.dropout(F.softmax(scores, dim=-1))
        context = (probabilities @ value).transpose(1, 2).contiguous()
        output: Tensor = self.output(context.view(batch, sequence, self.hidden_dim))
        return output


class SharedEncoderBlock(nn.Module):
    """One pre-norm block whose parameters are reused for every depth step."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = FusedSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn_up = TernaryLinear(config.hidden_dim, config.ffn_dim)
        self.ffn_down = TernaryLinear(config.ffn_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = inputs + self.dropout(self.attention(self.attention_norm(inputs), attention_mask))
        feed_forward = self.ffn_down(F.gelu(self.ffn_up(self.ffn_norm(hidden)), approximate="tanh"))
        output: Tensor = hidden + self.dropout(feed_forward)
        return output


class BrowserEncoder(nn.Module):
    """Query/document tokens to an L2-normalized Matryoshka embedding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = FactorizedEmbedding(config)
        self.shared_block = SharedEncoderBlock(config)
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if not torch.all(attention_mask.sum(dim=1) > 0):
            raise ValueError("every sequence must contain at least one unmasked token")
        hidden = self.embeddings(input_ids)
        for _ in range(self.config.num_repeats):
            hidden = self.shared_block(hidden, attention_mask)
        hidden = self.final_norm(hidden)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(self.output_projection(pooled), dim=-1)

    @staticmethod
    def truncate(embeddings: Tensor, dimension: int) -> Tensor:
        if dimension <= 0 or dimension > embeddings.shape[-1]:
            raise ValueError(f"invalid embedding dimension {dimension}")
        return F.normalize(embeddings[..., :dimension], dim=-1)

    def parameter_counts(self) -> dict[str, int]:
        physical = sum(parameter.numel() for parameter in self.parameters())
        block = sum(parameter.numel() for parameter in self.shared_block.parameters())
        return {
            "physical": physical,
            "shared_block": block,
            "logical": physical + (self.config.num_repeats - 1) * block,
        }
