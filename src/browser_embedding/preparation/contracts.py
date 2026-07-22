"""Strong schemas shared by offline preparation commands."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from browser_embedding.config import find_project_root

Variant = Literal["en_en", "zh_zh", "zh_en", "en_zh", "mixed_en", "mixed_zh"]
Split = Literal["train", "validation", "test"]
TokenizerInputFormat = Literal["text", "jsonl_text", "jsonl_pairs"]

SPECIAL_TOKEN_IDS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[QRY]": 4,
    "[DOC]": 5,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PairRecord(StrictModel):
    """Canonical source-neutral query/document record."""

    id: str | None = None
    group_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    document: str = Field(min_length=1)
    variant: Variant
    source: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    source_license: str = Field(min_length=1)
    split: Split | None = None

    @field_validator(
        "id",
        "group_id",
        "query",
        "document",
        "source",
        "source_revision",
        "source_license",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class SplitRatios(StrictModel):
    train: float = Field(default=0.9, gt=0.0, lt=1.0)
    validation: float = Field(default=0.05, gt=0.0, lt=1.0)
    test: float = Field(default=0.05, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> SplitRatios:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self


class CacheRecipe(StrictModel):
    schema_version: Literal[1]
    input_path: Path
    output_dir: Path
    tokenizer_path: Path
    seed: int = 42
    sample_limit: int | None = Field(default=None, gt=0)
    max_sequence_length: int = Field(default=128, gt=0, le=512)
    vocab_size: int = Field(default=32_000, gt=8)
    batch_size: int = Field(default=512, gt=0)
    split: SplitRatios = SplitRatios()


class TokenizerCorpus(StrictModel):
    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)


class TokenizerRecipe(StrictModel):
    schema_version: Literal[1]
    input_path: Path
    input_format: TokenizerInputFormat
    output_path: Path
    manifest_path: Path
    corpus: TokenizerCorpus
    vocab_size: int = Field(default=32_000, ge=512)
    asset_budget_bytes: int = Field(default=2_500_000, gt=0)


class TeacherRecipe(StrictModel):
    schema_version: Literal[1]
    cache_dir: Path
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str = Field(min_length=1)
    query_prefix: str = ""
    document_prefix: str = ""
    output_dimension: int = Field(default=384, gt=0)
    chunk_size: int = Field(default=256, gt=0)
    batch_size: int = Field(default=64, gt=0)
    trust_remote_code: bool = False


class LegacyTernlightRecipe(StrictModel):
    """Import the immutable bilingual pilot cache produced by ternlight."""

    schema_version: Literal[1]
    source_cache_dir: Path
    source_corpus_manifest: Path
    output_dir: Path
    tokenizer_path: Path
    source_teacher_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    teacher_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    teacher_license: str = Field(min_length=1)
    source_licenses: dict[str, str]
    link_mode: Literal["hardlink", "copy"] = "hardlink"
    verify_tokenizer: bool = True

    @field_validator("source_licenses")
    @classmethod
    def validate_source_licenses(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(
            not name.strip() or not license_name.strip() for name, license_name in value.items()
        ):
            raise ValueError("source_licenses must contain non-empty source and license names")
        return value


RecipeT = TypeVar("RecipeT", bound=BaseModel)


def load_recipe(path: Path, schema: type[RecipeT]) -> tuple[RecipeT, Path]:
    """Load a YAML recipe and return it with the project root."""
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return schema.model_validate(raw), find_project_root(path.parent)


def resolve_recipe_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
