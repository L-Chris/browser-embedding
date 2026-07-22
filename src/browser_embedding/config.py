"""Strongly typed experiment config and browser deployment budget.

The deployment contract lives next to the model config on purpose: an
architecture that cannot fit the browser budget is rejected before a GPU run
starts, rather than discovered after export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

EmbeddingFormat = Literal["int4", "int8", "fp16"]
DataKind = Literal["synthetic", "memmap"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    vocab_size: int = Field(gt=8, le=1_000_000)
    max_sequence_length: int = Field(gt=0, le=512)
    embedding_dim: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_repeats: int = Field(gt=0, le=24)
    ffn_dim: int = Field(gt=0)
    output_dim: int = Field(gt=0)
    matryoshka_dims: tuple[int, ...]
    dropout: float = Field(ge=0.0, lt=1.0)
    padding_idx: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_shape(self) -> ModelConfig:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.embedding_dim % 4 or self.hidden_dim % 4 or self.ffn_dim % 4:
            raise ValueError("embedding_dim, hidden_dim and ffn_dim must be multiples of 4")
        if not self.matryoshka_dims:
            raise ValueError("matryoshka_dims cannot be empty")
        if tuple(sorted(set(self.matryoshka_dims))) != self.matryoshka_dims:
            raise ValueError("matryoshka_dims must be unique and strictly increasing")
        if self.matryoshka_dims[-1] != self.output_dim:
            raise ValueError("the last matryoshka dimension must equal output_dim")
        if self.padding_idx >= self.vocab_size:
            raise ValueError("padding_idx is outside the vocabulary")
        return self


@dataclass(frozen=True)
class DeploymentEstimate:
    packed_model_bytes: int
    estimated_bundle_bytes: int
    working_memory_bytes: int
    trainable_parameters: int
    shared_block_parameters: int
    logical_parameters: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class DeploymentConfig(StrictModel):
    embedding_format: EmbeddingFormat = "int4"
    max_model_bytes: int = Field(gt=0)
    max_bundle_bytes: int = Field(gt=0)
    max_working_memory_bytes: int = Field(gt=0)
    wasm_binary_budget_bytes: int = Field(gt=0)
    tokenizer_asset_budget_bytes: int = Field(gt=0)
    tokenizer_working_memory_bytes: int = Field(gt=0)
    enforce_budget: bool = True

    def estimate(self, model: ModelConfig) -> DeploymentEstimate:
        """Conservative byte estimate matching the BEM2 ordered sections.

        Ternary matrices use two bits per value plus one fp32 scale. Embedding
        scales and all fp16 sections use two bytes. Runtime memory assumes the
        shared ternary matrices are unpacked once to i8 for a branch-free hot
        loop and uses batch size one, the browser latency-critical path.
        """
        v, s, e, h, f, o = (
            model.vocab_size,
            model.max_sequence_length,
            model.embedding_dim,
            model.hidden_dim,
            model.ffn_dim,
            model.output_dim,
        )
        if self.embedding_format == "int4":
            token_embedding = v * (e // 2 + 2)
        elif self.embedding_format == "int8":
            token_embedding = v * (e + 2)
        else:
            token_embedding = v * e * 2

        def ternary(values: int) -> int:
            return (values + 3) // 4 + 4

        position_embedding = s * e * 2
        embedding_norm = e * 2 * 2
        embedding_projection = ternary(e * h)
        block_matrices = ternary(3 * h * h) + ternary(h * h) + ternary(h * f) + ternary(f * h)
        block_norms = 2 * h * 2 * 2
        final_norm = h * 2 * 2
        output_head = (o * h + o) * 2
        format_overhead = 64 + 32
        packed = (
            token_embedding
            + position_embedding
            + embedding_norm
            + embedding_projection
            + block_matrices
            + block_norms
            + final_norm
            + output_head
            + format_overhead
        )

        shared_block_parameters = 4 * h * h + 2 * h * f
        trainable = v * e + s * e + e * h + shared_block_parameters + (2 * e + 6 * h) + h * o + o
        logical = trainable + (model.num_repeats - 1) * shared_block_parameters
        unpacked_ternary = e * h + shared_block_parameters
        activations = 3 * s * max(h, f) * 4
        attention_scores = model.num_heads * s * s * 4
        bundle = packed + self.wasm_binary_budget_bytes + self.tokenizer_asset_budget_bytes
        # The JS-side source bytes and Rust-owned Vec can coexist. Include both
        # plus the tokenizer heap so the budget represents browser reality.
        working = (
            2 * packed
            + unpacked_ternary
            + activations
            + attention_scores
            + self.tokenizer_working_memory_bytes
        )
        return DeploymentEstimate(
            packed, bundle, working, trainable, shared_block_parameters, logical
        )

    def assert_fits(self, model: ModelConfig) -> DeploymentEstimate:
        estimate = self.estimate(model)
        violations: list[str] = []
        if estimate.packed_model_bytes > self.max_model_bytes:
            violations.append(
                f"packed model {estimate.packed_model_bytes:,} > {self.max_model_bytes:,} bytes"
            )
        if estimate.estimated_bundle_bytes > self.max_bundle_bytes:
            violations.append(
                f"bundle {estimate.estimated_bundle_bytes:,} > {self.max_bundle_bytes:,} bytes"
            )
        if estimate.working_memory_bytes > self.max_working_memory_bytes:
            violations.append(
                "working memory "
                f"{estimate.working_memory_bytes:,} > {self.max_working_memory_bytes:,} bytes"
            )
        if violations and self.enforce_budget:
            raise ValueError("deployment budget exceeded: " + "; ".join(violations))
        return estimate


class DataConfig(StrictModel):
    kind: DataKind
    cache_dir: Path | None = None
    teacher_key: str | None = None
    tokenizer_path: Path | None = None
    train_pairs: int = Field(default=64, gt=0)
    validation_pairs: int = Field(default=24, gt=0)
    variants: tuple[str, ...] = (
        "en_en",
        "zh_zh",
        "zh_en",
        "en_zh",
        "mixed_en",
        "mixed_zh",
    )

    @model_validator(mode="after")
    def validate_source(self) -> DataConfig:
        if self.kind == "memmap":
            missing = [
                name
                for name, value in (
                    ("cache_dir", self.cache_dir),
                    ("teacher_key", self.teacher_key),
                    ("tokenizer_path", self.tokenizer_path),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"memmap data requires {', '.join(missing)}")
        if not self.variants:
            raise ValueError("variants cannot be empty")
        return self


class ObjectiveConfig(StrictModel):
    pointwise_weight: float = Field(ge=0.0)
    relational_weight: float = Field(ge=0.0)
    retrieval_weight: float = Field(ge=0.0)
    temperature: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_weights(self) -> ObjectiveConfig:
        if self.pointwise_weight + self.relational_weight + self.retrieval_weight == 0:
            raise ValueError("at least one objective weight must be positive")
        return self


class QuantizationConfig(StrictModel):
    enabled: bool = True
    warmup_epochs: int = Field(default=1, ge=0)


class TrainingConfig(StrictModel):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=1)
    eval_batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    gradient_clip: float = Field(gt=0.0)
    num_workers: int = Field(ge=0)
    save_every: int = Field(gt=0)
    log_every: int = Field(gt=0)
    device: str = "auto"


class EvaluationConfig(StrictModel):
    multilingual_seed_enabled: bool = False
    seed_path: Path | None = None
    best_metric: str = "validation.ndcg_at_10"

    @model_validator(mode="after")
    def validate_seed(self) -> EvaluationConfig:
        if self.multilingual_seed_enabled and self.seed_path is None:
            raise ValueError("seed_path is required when multilingual seed evaluation is enabled")
        return self


class RunConfig(StrictModel):
    name: str = Field(min_length=1)
    output_dir: Path
    export_on_finish: bool = True


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    seed: int
    model: ModelConfig
    deployment: DeploymentConfig
    data: DataConfig
    objective: ObjectiveConfig
    quantization: QuantizationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    run: RunConfig

    @model_validator(mode="after")
    def validate_contract(self) -> ExperimentConfig:
        self.deployment.assert_fits(self.model)
        return self


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"cannot find pyproject.toml above {start}")


def load_experiment(path: Path) -> tuple[ExperimentConfig, Path]:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    config = ExperimentConfig.model_validate(raw)
    return config, find_project_root(path.parent)


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path
