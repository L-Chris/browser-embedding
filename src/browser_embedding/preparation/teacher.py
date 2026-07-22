"""Resumable offline teacher encoding into an immutable pair cache."""

from __future__ import annotations

import importlib
import json
import time
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from browser_embedding.preparation.contracts import TeacherRecipe, resolve_recipe_path
from browser_embedding.preparation.io import iter_jsonl, sha256_file, write_json


def _resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested teacher device {requested!r}, but CUDA is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("requested teacher device 'mps', but MPS is unavailable")
    return requested


def _load_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    raw: object = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"cache manifest in {cache_dir} must be a JSON object")
    value = cast(dict[str, Any], raw)
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported cache schema in {cache_dir}")
    return value


def encode_teacher(
    recipe: TeacherRecipe,
    project_root: Path,
    *,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Generate normalized float16 targets, resuming only an identical job."""
    cache_dir = resolve_recipe_path(project_root, recipe.cache_dir)
    cache_manifest = _load_cache_manifest(cache_dir)
    texts_path = cache_dir / "texts.jsonl"
    teachers_dir = cache_dir / "teachers"
    teachers_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = teachers_dir / f"{recipe.key}.npy"
    teacher_manifest_path = teachers_dir / f"{recipe.key}.manifest.json"
    progress_path = teachers_dir / f"{recipe.key}.progress.json"
    selection_sha = str(cache_manifest["selection_sha256"])
    samples = int(cache_manifest["samples"])
    expected_shape = [samples, 2, recipe.output_dimension]
    device = _resolve_device(requested_device)

    if teacher_manifest_path.is_file() and embeddings_path.is_file():
        raw_existing: object = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_existing, dict):
            raise ValueError(f"teacher manifest {teacher_manifest_path} must be a JSON object")
        existing = cast(dict[str, Any], raw_existing)
        if (
            existing.get("selection_sha256") == selection_sha
            and existing.get("model_revision") == recipe.revision
            and existing.get("shape") == expected_shape
            and existing.get("sha256") == sha256_file(embeddings_path)
        ):
            return existing
        raise FileExistsError(f"teacher key {recipe.key!r} already names a different artifact")

    sentence_transformers = importlib.import_module("sentence_transformers")
    model_class = sentence_transformers.SentenceTransformer
    model = model_class(
        recipe.model_id,
        revision=recipe.revision,
        device=device,
        trust_remote_code=recipe.trust_remote_code,
    )
    model_dimension = model.get_sentence_embedding_dimension()
    if model_dimension != recipe.output_dimension:
        raise ValueError(
            f"teacher output dimension {model_dimension} != recipe {recipe.output_dimension}"
        )

    completed = 0
    if progress_path.is_file() and embeddings_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {
            "selection_sha256": selection_sha,
            "model_id": recipe.model_id,
            "model_revision": recipe.revision,
            "shape": expected_shape,
            "device": device,
        }
        if any(progress.get(name) != value for name, value in expected.items()):
            raise ValueError("teacher progress belongs to a different recipe, selection, or device")
        completed = int(progress["completed_pairs"])
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
    elif embeddings_path.exists() or progress_path.exists():
        raise FileExistsError("incomplete teacher artifact is missing its matching progress file")
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float16,
            shape=(samples, 2, recipe.output_dimension),
        )
        write_json(
            progress_path,
            {
                "selection_sha256": selection_sha,
                "model_id": recipe.model_id,
                "model_revision": recipe.revision,
                "shape": expected_shape,
                "device": device,
                "completed_pairs": 0,
            },
        )

    started = time.perf_counter()
    text_records = iter_jsonl(texts_path)
    for _ in range(completed):
        try:
            next(text_records)
        except StopIteration as error:
            raise ValueError("texts.jsonl ended before teacher resume position") from error
    for start in range(completed, samples, recipe.chunk_size):
        end = min(start + recipe.chunk_size, samples)
        chunk = list(islice(text_records, end - start))
        if len(chunk) != end - start:
            raise ValueError("texts.jsonl ended before the cache manifest sample count")
        queries = [recipe.query_prefix + str(record["query"]) for record in chunk]
        documents = [recipe.document_prefix + str(record["document"]) for record in chunk]
        encoded = model.encode(
            queries + documents,
            batch_size=recipe.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = np.asarray(encoded, dtype=np.float32)
        size = end - start
        if vectors.shape != (2 * size, recipe.output_dimension):
            raise ValueError(f"teacher returned unexpected shape {vectors.shape}")
        embeddings[start:end, 0] = vectors[:size].astype(np.float16)
        embeddings[start:end, 1] = vectors[size:].astype(np.float16)
        embeddings.flush()
        write_json(
            progress_path,
            {
                "selection_sha256": selection_sha,
                "model_id": recipe.model_id,
                "model_revision": recipe.revision,
                "shape": expected_shape,
                "device": device,
                "completed_pairs": end,
            },
        )

    elapsed = time.perf_counter() - started
    sample = np.asarray(embeddings[: min(samples, 10_000)], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=-1)
    if not np.isfinite(sample).all() or float(np.max(np.abs(norms - 1.0))) > 0.01:
        raise ValueError("generated teacher vectors are not finite and L2-normalized")
    teacher_manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "teacher_key": recipe.key,
        "model_id": recipe.model_id,
        "model_revision": recipe.revision,
        "model_license": recipe.license,
        "query_prefix": recipe.query_prefix,
        "document_prefix": recipe.document_prefix,
        "selection_sha256": selection_sha,
        "samples": samples,
        "shape": expected_shape,
        "dtype": "float16",
        "output_dimension": recipe.output_dimension,
        "device": device,
        "sentence_transformers_version": sentence_transformers.__version__,
        "elapsed_seconds_this_run": elapsed,
        "sample_norm_mean": float(norms.mean()),
        "sample_norm_max_error": float(np.max(np.abs(norms - 1.0))),
        "sha256": sha256_file(embeddings_path),
    }
    write_json(teacher_manifest_path, teacher_manifest)
    teachers = dict(cache_manifest.get("teachers", {}))
    teachers[recipe.key] = {
        "path": f"teachers/{recipe.key}.npy",
        "manifest_path": f"teachers/{recipe.key}.manifest.json",
        "sha256": teacher_manifest["sha256"],
    }
    cache_manifest["teachers"] = teachers
    write_json(cache_dir / "manifest.json", cache_manifest)
    return teacher_manifest
