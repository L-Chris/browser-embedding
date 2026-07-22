"""Pair-oriented data boundary for multilingual retrieval training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from browser_embedding.config import DataConfig, ModelConfig, TrainingConfig, resolve_path

SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


class PairDataset(Dataset[dict[str, Any]]):
    """Common query/document pair contract consumed by the runner."""

    variants: list[str]


class SyntheticPairDataset(PairDataset):
    """Deterministic, learnable pairs used only for tests and CI smoke runs."""

    def __init__(
        self,
        *,
        pairs: int,
        model: ModelConfig,
        variants: tuple[str, ...],
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        sequence = model.max_sequence_length
        minimum_length = min(6, sequence)
        lengths = torch.randint(minimum_length, sequence + 1, (pairs,), generator=generator)
        self.input_ids = torch.zeros(pairs, 2, sequence, dtype=torch.long)
        self.attention_mask = torch.zeros(pairs, 2, sequence, dtype=torch.long)
        self.teacher_embeddings = torch.empty(pairs, 2, model.output_dim)
        self.variants = [variants[index % len(variants)] for index in range(pairs)]

        for index, length_tensor in enumerate(lengths):
            length = int(length_tensor)
            shared = torch.randint(6, model.vocab_size, (length,), generator=generator)
            query = shared.clone()
            document = shared.clone()
            if length > 3:
                query[-1] = torch.randint(6, model.vocab_size, (), generator=generator)
                document[-2] = torch.randint(6, model.vocab_size, (), generator=generator)
            query[0] = 4  # [QRY]
            document[0] = 5  # [DOC]
            self.input_ids[index, 0, :length] = query
            self.input_ids[index, 1, :length] = document
            self.attention_mask[index, :, :length] = 1

            concept = torch.randn(model.output_dim, generator=generator)
            noise = 0.03 * torch.randn(2, model.output_dim, generator=generator)
            self.teacher_embeddings[index] = torch.nn.functional.normalize(
                concept[None, :] + noise, dim=-1
            )

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "row": index,
            "group_id": index,
            "variant": self.variants[index],
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "teacher_embeddings": self.teacher_embeddings[index],
        }


class MemmapPairDataset(PairDataset):
    """Zero-copy adapter for a versioned multilingual pair cache.

    Cache layout::

        manifest.json
        input_ids.npy          [pairs, 2, sequence]
        attention_mask.npy     [pairs, 2, sequence]
        split_codes.npy        [pairs]
        metadata.jsonl         {group_id, variant, ...}
        teachers/<key>.npy     [pairs, 2, output_dim]
    """

    def __init__(self, root: Path, teacher_key: str, split: str, model: ModelConfig) -> None:
        if split not in SPLIT_CODES:
            raise ValueError(f"unknown split {split!r}")
        manifest_path = root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported data cache schema in {manifest_path}")
        self.input_ids = np.load(root / "input_ids.npy", mmap_mode="r")
        self.attention_mask = np.load(root / "attention_mask.npy", mmap_mode="r")
        self.split_codes = np.load(root / "split_codes.npy", mmap_mode="r")
        self.teacher_embeddings = np.load(root / "teachers" / f"{teacher_key}.npy", mmap_mode="r")
        with (root / "metadata.jsonl").open(encoding="utf-8") as handle:
            self.metadata = [json.loads(line) for line in handle if line.strip()]
        self.indices = np.flatnonzero(self.split_codes == SPLIT_CODES[split])
        self.variants = [str(self.metadata[int(row)]["variant"]) for row in self.indices]
        self._validate_shapes(model)

    def _validate_shapes(self, model: ModelConfig) -> None:
        pairs = self.input_ids.shape[0]
        expected_tokens = (pairs, 2, model.max_sequence_length)
        if self.input_ids.shape != expected_tokens or self.attention_mask.shape != expected_tokens:
            raise ValueError(f"token cache shape must be {expected_tokens}")
        if self.teacher_embeddings.shape != (pairs, 2, model.output_dim):
            raise ValueError(f"teacher cache shape must be {(pairs, 2, model.output_dim)}")
        if len(self.split_codes) != pairs or len(self.metadata) != pairs:
            raise ValueError("cache arrays and metadata have different row counts")
        if int(np.max(self.input_ids, initial=0)) >= model.vocab_size:
            raise ValueError("cache contains token IDs outside model vocabulary")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.indices[index])
        metadata = self.metadata[row]
        group = str(metadata["group_id"])
        group_id = int(sha256(group.encode("utf-8")).hexdigest()[:15], 16)
        return {
            "row": row,
            "group_id": group_id,
            "variant": str(metadata["variant"]),
            "input_ids": torch.as_tensor(np.asarray(self.input_ids[row]).copy(), dtype=torch.long),
            "attention_mask": torch.as_tensor(
                np.asarray(self.attention_mask[row]).copy(), dtype=torch.long
            ),
            "teacher_embeddings": torch.as_tensor(
                np.asarray(self.teacher_embeddings[row]).copy(), dtype=torch.float32
            ),
        }


@dataclass(frozen=True)
class DataBundle:
    train: PairDataset
    validation: PairDataset
    train_loader: DataLoader[dict[str, Any]]
    validation_loader: DataLoader[dict[str, Any]]
    generator: torch.Generator
    provenance: dict[str, Any]


def build_data(
    data: DataConfig,
    model: ModelConfig,
    training: TrainingConfig,
    *,
    project_root: Path,
    seed: int,
    device: torch.device,
) -> DataBundle:
    if data.kind == "synthetic":
        train_dataset: PairDataset = SyntheticPairDataset(
            pairs=data.train_pairs, model=model, variants=data.variants, seed=seed
        )
        validation_dataset: PairDataset = SyntheticPairDataset(
            pairs=data.validation_pairs, model=model, variants=data.variants, seed=seed + 1
        )
        provenance: dict[str, Any] = {
            "kind": "synthetic",
            "seed": seed,
            "train_pairs": data.train_pairs,
            "validation_pairs": data.validation_pairs,
        }
    else:
        assert data.cache_dir is not None and data.teacher_key is not None
        cache_dir = resolve_path(project_root, data.cache_dir)
        train_dataset = MemmapPairDataset(cache_dir, data.teacher_key, "train", model)
        validation_dataset = MemmapPairDataset(cache_dir, data.teacher_key, "validation", model)
        provenance = {
            "kind": "memmap",
            "cache_dir": str(cache_dir),
            "teacher_key": data.teacher_key,
            "manifest": train_dataset.manifest,
        }

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=training.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.eval_batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=training.num_workers > 0,
    )
    return DataBundle(
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
        generator,
        provenance,
    )
