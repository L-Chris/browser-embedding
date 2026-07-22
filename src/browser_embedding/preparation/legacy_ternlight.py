"""Import ternlight's bilingual pilot cache without recomputing large arrays."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, cast

import numpy as np

from browser_embedding.preparation.cache import validate_cache
from browser_embedding.preparation.contracts import (
    SPECIAL_TOKEN_IDS,
    LegacyTernlightRecipe,
    PairRecord,
    Split,
    Variant,
    resolve_recipe_path,
)
from browser_embedding.preparation.io import sha256_file, stable_json, write_json

_ARRAY_FILES = ("input_ids.npy", "attention_mask.npy", "split_codes.npy")


def _load_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def _close_memmap(value: np.ndarray) -> None:
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _legacy_record(
    raw: dict[str, Any], revisions: dict[str, str], licenses: dict[str, str]
) -> PairRecord:
    source = str(raw["source"])
    if source not in revisions:
        raise ValueError(f"legacy source {source!r} has no pinned revision")
    if source not in licenses:
        raise ValueError(f"legacy source {source!r} has no license audit value")
    return PairRecord(
        id=str(raw["id"]),
        group_id=str(raw["group_id"]),
        query=str(raw["query"]),
        document=str(raw["positive"]),
        variant=cast(Variant, str(raw["variant"])),
        source=source,
        source_revision=revisions[source],
        source_license=licenses[source],
        split=cast(Split, str(raw["split"])),
    )


def _verify_batch(
    tokenizer: Any,
    records: list[PairRecord],
    start: int,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> tuple[int, dict[str, int], dict[str, int]]:
    values = [f"[QRY] {record.query}" for record in records] + [
        f"[DOC] {record.document}" for record in records
    ]
    encodings = tokenizer.encode_batch(values, add_special_tokens=True)
    size = len(records)
    unknown_tokens = 0
    truncated = {"query": 0, "document": 0}
    missing_roles = {"query": 0, "document": 0}
    for local, encoding in enumerate(encodings):
        role_index = 0 if local < size else 1
        row = start + (local if local < size else local - size)
        expected_ids = np.asarray(encoding.ids, dtype=np.uint32)
        expected_mask = np.asarray(encoding.attention_mask, dtype=np.uint8)
        if not np.array_equal(expected_ids, input_ids[row, role_index]):
            raise ValueError(f"legacy input_ids differ from the target tokenizer at row {row}")
        if not np.array_equal(expected_mask, attention_mask[row, role_index]):
            raise ValueError(f"legacy attention_mask differs at row {row}")
        unknown_tokens += encoding.ids.count(SPECIAL_TOKEN_IDS["[UNK]"])
        role = "query" if role_index == 0 else "document"
        role_id = SPECIAL_TOKEN_IDS["[QRY]" if role_index == 0 else "[DOC]"]
        missing_roles[role] += int(role_id not in encoding.ids)
        truncated[role] += int(bool(encoding.overflowing))
    return unknown_tokens, truncated, missing_roles


def import_legacy_ternlight_cache(
    recipe: LegacyTernlightRecipe, project_root: Path
) -> dict[str, Any]:
    """Convert ternlight metadata and reuse its immutable token/teacher arrays."""
    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    source_cache = resolve_recipe_path(project_root, recipe.source_cache_dir)
    corpus_manifest_path = resolve_recipe_path(project_root, recipe.source_corpus_manifest)
    output = resolve_recipe_path(project_root, recipe.output_dir)
    tokenizer_path = resolve_recipe_path(project_root, recipe.tokenizer_path)
    shared = source_cache / "shared"
    source_teacher = source_cache / recipe.source_teacher_key
    selection_path = shared / "selection.jsonl"
    source_teacher_path = source_teacher / "embeddings.npy"
    source_teacher_manifest_path = source_teacher / "manifest.json"
    required = [
        selection_path,
        source_teacher_path,
        source_teacher_manifest_path,
        corpus_manifest_path,
        tokenizer_path,
        *(shared / name for name in _ARRAY_FILES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"legacy migration inputs are missing: {missing}")
    if output.exists():
        raise FileExistsError(f"cache output already exists: {output}")

    shared_manifest = _load_json(shared / "manifest.json")
    corpus_manifest = _load_json(corpus_manifest_path)
    source_teacher_manifest = _load_json(source_teacher_manifest_path)
    if shared_manifest.get("schema_version") != 1:
        raise ValueError("unsupported ternlight shared cache schema")
    if source_teacher_manifest.get("schema_version") != 1:
        raise ValueError("unsupported ternlight teacher cache schema")
    samples = int(shared_manifest["samples"])
    sequence = int(shared_manifest["max_length"])
    revisions = {
        str(name): str(revision)
        for name, revision in dict(corpus_manifest["source_revisions"]).items()
    }

    input_ids = np.load(shared / "input_ids.npy", mmap_mode="r")
    attention_mask = np.load(shared / "attention_mask.npy", mmap_mode="r")
    split_codes = np.load(shared / "split_codes.npy", mmap_mode="r")
    teacher_values = np.load(source_teacher_path, mmap_mode="r")
    if input_ids.shape != (samples, 2, sequence) or input_ids.dtype != np.uint32:
        raise ValueError("legacy input_ids shape or dtype differs from its manifest")
    if attention_mask.shape != input_ids.shape or attention_mask.dtype != np.uint8:
        raise ValueError("legacy attention_mask shape or dtype differs from its manifest")
    if split_codes.shape != (samples,) or split_codes.dtype != np.uint8:
        raise ValueError("legacy split_codes shape or dtype differs from its manifest")
    expected_teacher_shape = (samples, 2, int(source_teacher_manifest["shape"][2]))
    if teacher_values.shape != expected_teacher_shape or teacher_values.dtype != np.float16:
        raise ValueError("legacy teacher shape or dtype differs from its manifest")
    if source_teacher_manifest.get("selection_sha256") != shared_manifest.get("selection_sha256"):
        raise ValueError("legacy token and teacher caches have different row selections")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    actual_special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKEN_IDS}
    if actual_special_ids != SPECIAL_TOKEN_IDS:
        raise ValueError(f"target tokenizer special IDs differ: {actual_special_ids}")
    if tokenizer.get_vocab_size(with_added_tokens=True) != int(shared_manifest["vocab_size"]):
        raise ValueError("target tokenizer vocabulary size differs from the legacy cache")
    tokenizer.enable_truncation(max_length=sequence)
    tokenizer.enable_padding(length=sequence, pad_id=0, pad_token="[PAD]")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=f"{output.name}.tmp-", dir=output.parent))
    try:
        for name in _ARRAY_FILES:
            _link_or_copy(shared / name, temporary / name, recipe.link_mode)
        teacher_output = temporary / "teachers" / f"{recipe.teacher_key}.npy"
        _link_or_copy(source_teacher_path, teacher_output, recipe.link_mode)

        selection_digest = sha256()
        counts = {
            "split": Counter[str](),
            "variant": Counter[str](),
            "source": Counter[str](),
        }
        unknown_tokens = 0
        truncated = {"query": 0, "document": 0}
        missing_roles = {"query": 0, "document": 0}
        row_count = 0
        batch: list[PairRecord] = []
        batch_start = 0
        with (
            selection_path.open(encoding="utf-8") as source,
            (temporary / "texts.jsonl").open("w", encoding="utf-8", newline="\n") as texts,
            (temporary / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as metadata,
        ):
            for line in source:
                if not line.strip():
                    continue
                raw: object = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"legacy selection row {row_count + 1} is not an object")
                record = _legacy_record(raw, revisions, recipe.source_licenses)
                dumped = record.model_dump(mode="json")
                selection_digest.update(stable_json(dumped).encode("utf-8"))
                selection_digest.update(b"\n")
                counts["split"][str(record.split)] += 1
                counts["variant"][record.variant] += 1
                counts["source"][record.source] += 1
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
                            "row": row_count,
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
                batch.append(record)
                row_count += 1
                if recipe.verify_tokenizer and len(batch) == 512:
                    batch_unknown, batch_truncated, batch_missing = _verify_batch(
                        tokenizer, batch, batch_start, input_ids, attention_mask
                    )
                    unknown_tokens += batch_unknown
                    for role in truncated:
                        truncated[role] += batch_truncated[role]
                        missing_roles[role] += batch_missing[role]
                    batch_start += len(batch)
                    batch = []
            if recipe.verify_tokenizer and batch:
                batch_unknown, batch_truncated, batch_missing = _verify_batch(
                    tokenizer, batch, batch_start, input_ids, attention_mask
                )
                unknown_tokens += batch_unknown
                for role in truncated:
                    truncated[role] += batch_truncated[role]
                    missing_roles[role] += batch_missing[role]
        if row_count != samples:
            raise ValueError(f"legacy selection has {row_count} rows, expected {samples}")

        files = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": sha256_file(temporary / name),
            }
            for name in (*_ARRAY_FILES, "metadata.jsonl", "texts.jsonl")
        }
        teacher_sha = sha256_file(teacher_output)
        if teacher_sha != source_teacher_manifest["sha256"]:
            raise ValueError("legacy teacher SHA-256 differs from its manifest")
        norm_sample = np.asarray(teacher_values[: min(samples, 10_000)], dtype=np.float32)
        norms = np.linalg.norm(norm_sample, axis=-1)
        norm_max_error = float(np.max(np.abs(norms - 1.0)))
        if not np.isfinite(norm_sample).all() or norm_max_error > 0.01:
            raise ValueError("legacy teacher is not finite and L2-normalized")

        selection_sha = selection_digest.hexdigest()
        teacher_manifest: dict[str, Any] = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "teacher_key": recipe.teacher_key,
            "model_id": source_teacher_manifest["model_id"],
            "model_revision": source_teacher_manifest["model_revision"],
            "model_license": recipe.teacher_license,
            "query_prefix": source_teacher_manifest.get("query_prefix", ""),
            "document_prefix": source_teacher_manifest.get("document_prefix", ""),
            "selection_sha256": selection_sha,
            "samples": samples,
            "shape": list(teacher_values.shape),
            "dtype": "float16",
            "output_dimension": int(teacher_values.shape[2]),
            "device": source_teacher_manifest.get("device", "legacy"),
            "sentence_transformers_version": source_teacher_manifest.get(
                "sentence_transformers_version", "unknown"
            ),
            "elapsed_seconds_this_run": 0.0,
            "sample_norm_mean": float(norms.mean()),
            "sample_norm_max_error": norm_max_error,
            "sha256": teacher_sha,
            "migration": {
                "source_manifest_sha256": sha256_file(source_teacher_manifest_path),
                "source_generated_at": source_teacher_manifest.get("generated_at"),
            },
        }
        teacher_manifest_path = temporary / "teachers" / f"{recipe.teacher_key}.manifest.json"
        write_json(teacher_manifest_path, teacher_manifest)

        sources = [
            {
                "name": name,
                "revision": revisions[name],
                "license": recipe.source_licenses[name],
                "rows": count,
            }
            for name, count in sorted(counts["source"].items())
        ]
        tokenizer_sha = sha256_file(tokenizer_path)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact": output.name,
            "generated_at": datetime.now(UTC).isoformat(),
            "samples": samples,
            "input_sha256": sha256_file(selection_path),
            "selection_sha256": selection_sha,
            "seed": int(shared_manifest.get("seed", 42)),
            "duplicates_removed": 0,
            "max_sequence_length": sequence,
            "counts": {name: dict(sorted(values.items())) for name, values in counts.items()},
            "sources": sources,
            "tokenizer": {
                "path": str(recipe.tokenizer_path),
                "sha256": tokenizer_sha,
                "bytes": tokenizer_path.stat().st_size,
                "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
                "special_token_ids": SPECIAL_TOKEN_IDS,
                "unknown_tokens": unknown_tokens if recipe.verify_tokenizer else None,
                "truncated": truncated if recipe.verify_tokenizer else None,
                "missing_role_tokens": missing_roles if recipe.verify_tokenizer else None,
            },
            "arrays": {
                "input_ids": {"shape": list(input_ids.shape), "dtype": "uint32"},
                "attention_mask": {"shape": list(attention_mask.shape), "dtype": "uint8"},
                "split_codes": {"shape": list(split_codes.shape), "dtype": "uint8"},
            },
            "files": files,
            "teachers": {
                recipe.teacher_key: {
                    "path": f"teachers/{recipe.teacher_key}.npy",
                    "manifest_path": f"teachers/{recipe.teacher_key}.manifest.json",
                    "sha256": teacher_sha,
                }
            },
            "migration": {
                "source": "ternlight-bilingual-pilot-v1",
                "source_cache_manifest_sha256": sha256_file(shared / "manifest.json"),
                "source_corpus_manifest_sha256": sha256_file(corpus_manifest_path),
                "link_mode": recipe.link_mode,
                "tokenizer_cache_fully_verified": recipe.verify_tokenizer,
                "license_audit_required": any(
                    "UNKNOWN" in value.upper() for value in recipe.source_licenses.values()
                ),
            },
        }
        write_json(temporary / "manifest.json", manifest)

        for value in (input_ids, attention_mask, split_codes, teacher_values):
            _close_memmap(value)
        validation = validate_cache(temporary)
        temporary.replace(output)
        manifest["validation"] = validation
        return manifest
    except BaseException:
        for value in (input_ids, attention_mask, split_codes, teacher_values):
            _close_memmap(value)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
