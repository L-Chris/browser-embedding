"""Build and validate the immutable memmap cache consumed by training."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from browser_embedding.preparation.contracts import (
    SPECIAL_TOKEN_IDS,
    CacheRecipe,
    PairRecord,
    Split,
    resolve_recipe_path,
)
from browser_embedding.preparation.io import (
    count_jsonl,
    iter_jsonl,
    sha256_file,
    stable_json,
    write_json,
)

SPLIT_CODES: dict[Split, int] = {"train": 0, "validation": 1, "test": 2}


def _clean_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value.replace("\u00a0", " "))
    return re.sub(r"\s+", " ", normalized).strip()


def _record_id(record: PairRecord) -> str:
    if record.id is not None:
        return record.id
    payload = "\0".join(
        (record.source, record.group_id, record.variant, record.query, record.document)
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def _split_for_group(record: PairRecord, recipe: CacheRecipe) -> Split:
    if record.split is not None:
        return record.split
    digest = sha256(f"{recipe.seed}:{record.group_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < recipe.split.train:
        return "train"
    if fraction < recipe.split.train + recipe.split.validation:
        return "validation"
    return "test"


def _load_records(path: Path, recipe: CacheRecipe) -> tuple[list[PairRecord], int]:
    records: list[PairRecord] = []
    seen: set[tuple[str, str, str]] = set()
    removed = 0
    group_splits: dict[str, Split] = {}
    for raw in iter_jsonl(path):
        parsed = PairRecord.model_validate(raw)
        parsed = parsed.model_copy(
            update={
                "id": _record_id(parsed),
                "query": _clean_text(parsed.query),
                "document": _clean_text(parsed.document),
            }
        )
        if not parsed.query or not parsed.document:
            raise ValueError(f"record {parsed.id} is empty after normalization")
        split = _split_for_group(parsed, recipe)
        existing_split = group_splits.setdefault(parsed.group_id, split)
        if existing_split != split:
            raise ValueError(f"group {parsed.group_id!r} spans multiple explicit splits")
        parsed = parsed.model_copy(update={"split": split})
        dedupe_key = (parsed.variant, parsed.query.casefold(), parsed.document.casefold())
        if dedupe_key in seen:
            removed += 1
            continue
        seen.add(dedupe_key)
        records.append(parsed)
    if not records:
        raise ValueError(f"{path} contains no usable pair records")
    return records, removed


def _select_records(records: list[PairRecord], limit: int | None, seed: int) -> list[PairRecord]:
    if limit is None or limit == len(records):
        selected = list(records)
    else:
        if limit > len(records):
            raise ValueError(f"requested {limit:,} samples from {len(records):,} records")
        by_variant: dict[str, list[PairRecord]] = defaultdict(list)
        for record in records:
            by_variant[record.variant].append(record)
        exact = {name: limit * len(values) / len(records) for name, values in by_variant.items()}
        quotas = {name: int(value) for name, value in exact.items()}
        remainder = limit - sum(quotas.values())
        for name in sorted(
            exact,
            key=lambda item: (exact[item] - quotas[item], item),
            reverse=True,
        ):
            if remainder == 0:
                break
            quotas[name] += 1
            remainder -= 1
        selected = []
        for name, values in by_variant.items():
            values.sort(key=lambda item: sha256(f"{seed}:select:{item.id}".encode()).digest())
            selected.extend(values[: quotas[name]])
    selected.sort(key=lambda item: sha256(f"{seed}:order:{item.id}".encode()).digest())
    return selected


def _selection_sha256(records: list[PairRecord]) -> str:
    digest = sha256()
    for record in records:
        digest.update(stable_json(record.model_dump(mode="json")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_manifest(records: list[PairRecord]) -> list[dict[str, Any]]:
    values: dict[str, tuple[str, str]] = {}
    for record in records:
        identity = (record.source_revision, record.source_license)
        existing = values.setdefault(record.source, identity)
        if existing != identity:
            raise ValueError(f"source {record.source!r} has inconsistent revision or license")
    counts = Counter(record.source for record in records)
    return [
        {
            "name": name,
            "revision": revision,
            "license": license_name,
            "rows": counts[name],
        }
        for name, (revision, license_name) in sorted(values.items())
    ]


def _validate_special_tokens(tokenizer: Any) -> dict[str, int]:
    actual = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKEN_IDS}
    if actual != SPECIAL_TOKEN_IDS:
        raise ValueError(f"tokenizer special token IDs differ: {actual}")
    return SPECIAL_TOKEN_IDS


def _write_records(output: Path, records: list[PairRecord]) -> None:
    with (
        (output / "texts.jsonl").open("w", encoding="utf-8", newline="\n") as texts,
        (output / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as metadata,
    ):
        for row, record in enumerate(records):
            texts.write(
                json.dumps(
                    {"id": record.id, "query": record.query, "document": record.document},
                    ensure_ascii=False,
                )
                + "\n"
            )
            metadata.write(
                json.dumps(
                    {
                        "row": row,
                        "id": record.id,
                        "group_id": record.group_id,
                        "variant": record.variant,
                        "split": record.split,
                        "source": record.source,
                        "source_revision": record.source_revision,
                        "source_license": record.source_license,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _split_code(record: PairRecord) -> int:
    if record.split is None:
        raise AssertionError("selected record has no split")
    return SPLIT_CODES[record.split]


def build_cache(recipe: CacheRecipe, project_root: Path) -> dict[str, Any]:
    """Build one immutable cache directory from canonical pair JSONL."""
    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    input_path = resolve_recipe_path(project_root, recipe.input_path)
    output = resolve_recipe_path(project_root, recipe.output_dir)
    tokenizer_path = resolve_recipe_path(project_root, recipe.tokenizer_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"cache directory is immutable and already populated: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records, duplicates_removed = _load_records(input_path, recipe)
    selected = _select_records(records, recipe.sample_limit, recipe.seed)
    _write_records(output, selected)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    special_ids = _validate_special_tokens(tokenizer)
    if tokenizer.get_vocab_size(with_added_tokens=True) != recipe.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary does not match recipe vocab_size={recipe.vocab_size}"
        )
    tokenizer.enable_truncation(max_length=recipe.max_sequence_length)
    tokenizer.enable_padding(
        length=recipe.max_sequence_length,
        pad_id=SPECIAL_TOKEN_IDS["[PAD]"],
        pad_token="[PAD]",
    )

    pairs = len(selected)
    shape = (pairs, 2, recipe.max_sequence_length)
    input_ids = np.lib.format.open_memmap(
        output / "input_ids.npy", mode="w+", dtype=np.uint32, shape=shape
    )
    attention_mask = np.lib.format.open_memmap(
        output / "attention_mask.npy", mode="w+", dtype=np.uint8, shape=shape
    )
    split_codes = np.lib.format.open_memmap(
        output / "split_codes.npy", mode="w+", dtype=np.uint8, shape=(pairs,)
    )
    unknown_tokens = 0
    truncated = {"query": 0, "document": 0}
    missing_roles = {"query": 0, "document": 0}
    for start in range(0, pairs, recipe.batch_size):
        end = min(start + recipe.batch_size, pairs)
        batch = selected[start:end]
        values = [f"[QRY] {record.query}" for record in batch] + [
            f"[DOC] {record.document}" for record in batch
        ]
        encodings = tokenizer.encode_batch(values, add_special_tokens=True)
        size = end - start
        for local, encoding in enumerate(encodings):
            role_index = 0 if local < size else 1
            row = start + (local if local < size else local - size)
            ids = encoding.ids
            mask = encoding.attention_mask
            input_ids[row, role_index] = ids
            attention_mask[row, role_index] = mask
            unknown_tokens += ids.count(SPECIAL_TOKEN_IDS["[UNK]"])
            role = "query" if role_index == 0 else "document"
            role_token = special_ids["[QRY]" if role_index == 0 else "[DOC]"]
            missing_roles[role] += int(role_token not in ids)
            truncated[role] += int(bool(encoding.overflowing))
        split_codes[start:end] = [_split_code(record) for record in batch]
    input_ids.flush()
    attention_mask.flush()
    split_codes.flush()

    files = {
        name: {"bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)}
        for name in (
            "input_ids.npy",
            "attention_mask.npy",
            "split_codes.npy",
            "metadata.jsonl",
            "texts.jsonl",
        )
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": output.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "samples": pairs,
        "input_sha256": sha256_file(input_path),
        "selection_sha256": _selection_sha256(selected),
        "seed": recipe.seed,
        "duplicates_removed": duplicates_removed,
        "max_sequence_length": recipe.max_sequence_length,
        "counts": {
            "split": dict(sorted(Counter(record.split for record in selected).items())),
            "variant": dict(sorted(Counter(record.variant for record in selected).items())),
            "source": dict(sorted(Counter(record.source for record in selected).items())),
        },
        "sources": _source_manifest(selected),
        "tokenizer": {
            "path": str(recipe.tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "bytes": tokenizer_path.stat().st_size,
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            "special_token_ids": special_ids,
            "unknown_tokens": unknown_tokens,
            "truncated": truncated,
            "missing_role_tokens": missing_roles,
        },
        "arrays": {
            "input_ids": {"shape": list(shape), "dtype": "uint32"},
            "attention_mask": {"shape": list(shape), "dtype": "uint8"},
            "split_codes": {"shape": [pairs], "dtype": "uint8"},
        },
        "files": files,
        "teachers": {},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_cache(cache_dir: Path, *, full_hash: bool = True) -> dict[str, Any]:
    """Validate shapes, row alignment, hashes, role tokens and teacher norms."""
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported cache schema in {manifest_path}")
    pairs = int(manifest["samples"])
    sequence = int(manifest["max_sequence_length"])
    ids = np.load(cache_dir / "input_ids.npy", mmap_mode="r")
    mask = np.load(cache_dir / "attention_mask.npy", mmap_mode="r")
    splits = np.load(cache_dir / "split_codes.npy", mmap_mode="r")
    if ids.shape != (pairs, 2, sequence) or ids.dtype != np.uint32:
        raise ValueError("input_ids.npy does not match manifest")
    if mask.shape != ids.shape or mask.dtype != np.uint8:
        raise ValueError("attention_mask.npy does not match manifest")
    if splits.shape != (pairs,) or splits.dtype != np.uint8:
        raise ValueError("split_codes.npy does not match manifest")
    if count_jsonl(cache_dir / "metadata.jsonl") != pairs:
        raise ValueError("metadata.jsonl row count differs from manifest")
    if count_jsonl(cache_dir / "texts.jsonl") != pairs:
        raise ValueError("texts.jsonl row count differs from manifest")
    if np.any(mask > 1):
        raise ValueError("attention mask contains values other than zero and one")
    if full_hash:
        for name, expected in manifest["files"].items():
            actual = sha256_file(cache_dir / name)
            if actual != expected["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {name}")

    teacher_reports: dict[str, Any] = {}
    for key, entry in manifest.get("teachers", {}).items():
        teacher_path = cache_dir / entry["path"]
        teacher_manifest_path = cache_dir / entry["manifest_path"]
        teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
        if teacher_manifest["selection_sha256"] != manifest["selection_sha256"]:
            raise ValueError(f"teacher {key} targets a different row selection")
        values = np.load(teacher_path, mmap_mode="r")
        expected_shape = (pairs, 2, int(teacher_manifest["output_dimension"]))
        if values.shape != expected_shape or values.dtype != np.float16:
            raise ValueError(f"teacher {key} shape or dtype differs from manifest")
        sample = np.asarray(values[: min(pairs, 10_000)], dtype=np.float32)
        norms = np.linalg.norm(sample, axis=-1)
        if not np.isfinite(sample).all() or float(np.max(np.abs(norms - 1.0))) > 0.01:
            raise ValueError(f"teacher {key} is not finite and L2-normalized")
        if full_hash and sha256_file(teacher_path) != teacher_manifest["sha256"]:
            raise ValueError(f"SHA-256 mismatch for teacher {key}")
        teacher_reports[key] = {
            "shape": list(values.shape),
            "norm_mean": float(norms.mean()),
            "norm_max_error": float(np.max(np.abs(norms - 1.0))),
        }
    return {
        "schema_version": 1,
        "cache": str(cache_dir),
        "samples": pairs,
        "full_hash": full_hash,
        "teachers": teacher_reports,
        "status": "ok",
    }
