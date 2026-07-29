"""Score two frozen embedding manifests with exact full-corpus retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_LOCK = EVAL_DIR / "retrieval-frozen-v1.lock.json"
DEFAULT_CACHE = EVAL_DIR / "cache" / "retrieval-frozen-v1"
METRICS = (
    "ndcg_at_10",
    "map_at_100",
    "mrr_at_10",
    "recall_at_1",
    "recall_at_3",
    "recall_at_10",
    "recall_at_100",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def query_metrics(ranking: npt.ArrayLike, relevance: dict[int, int]) -> dict[str, float]:
    """Compute cutoff metrics for one exact top-100 ranking."""
    ranked = [int(value) for value in ranking]
    relevant = {index: score for index, score in relevance.items() if score > 0}
    if not relevant:
        raise ValueError("each frozen query must have at least one positive relevance judgment")

    hits = [index in relevant for index in ranked]
    relevant_count = len(relevant)
    recalls = {cutoff: sum(hits[:cutoff]) / relevant_count for cutoff in (1, 3, 10, 100)}
    reciprocal_rank = next(
        (1.0 / rank for rank, hit in enumerate(hits[:10], start=1) if hit),
        0.0,
    )

    precision_sum = 0.0
    hits_seen = 0
    for rank, hit in enumerate(hits[:100], start=1):
        if hit:
            hits_seen += 1
            precision_sum += hits_seen / rank
    average_precision = precision_sum / min(relevant_count, 100)

    dcg = sum(
        (2.0 ** relevant[index] - 1.0) / math.log2(rank + 1)
        for rank, index in enumerate(ranked[:10], start=1)
        if index in relevant
    )
    ideal_scores = sorted(relevant.values(), reverse=True)[:10]
    ideal_dcg = sum(
        (2.0**score - 1.0) / math.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1)
    )
    return {
        "ndcg_at_10": dcg / ideal_dcg,
        "map_at_100": average_precision,
        "mrr_at_10": reciprocal_rank,
        "recall_at_1": recalls[1],
        "recall_at_3": recalls[3],
        "recall_at_10": recalls[10],
        "recall_at_100": recalls[100],
    }


def verify_artifact(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"artifact size differs from frozen metadata: {path}")
    if sha256_path(path) != expected["sha256"]:
        raise ValueError(f"artifact SHA-256 differs from frozen metadata: {path}")


def load_frozen_dataset(
    cache: Path, dataset: dict[str, Any]
) -> tuple[list[str], list[str], list[dict[int, int]]]:
    artifact_paths = {
        kind: cache / dataset["artifacts"][kind]["path"] for kind in ("corpus", "queries", "qrels")
    }
    for kind, path in artifact_paths.items():
        verify_artifact(path, dataset["artifacts"][kind])
    corpus_rows = load_jsonl(artifact_paths["corpus"])
    query_rows = load_jsonl(artifact_paths["queries"])
    qrel_rows = load_jsonl(artifact_paths["qrels"])
    document_ids = [str(row["id"]) for row in corpus_rows]
    query_ids = [str(row["id"]) for row in query_rows]
    document_index = {document_id: index for index, document_id in enumerate(document_ids)}
    qrels: dict[str, dict[int, int]] = defaultdict(dict)
    for row in qrel_rows:
        qrels[str(row["query_id"])][document_index[str(row["corpus_id"])]] = int(row["relevance"])
    relevance = [qrels[query_id] for query_id in query_ids]
    if any(not judgments for judgments in relevance):
        raise ValueError(f"{dataset['id']} contains a query without positive qrels")
    return document_ids, query_ids, relevance


def load_matrix(manifest_dir: Path, specification: dict[str, Any]) -> npt.NDArray[np.float32]:
    path = manifest_dir / specification["path"]
    verify_artifact(path, specification)
    expected_bytes = specification["rows"] * specification["columns"] * np.dtype("<f4").itemsize
    if specification["bytes"] != expected_bytes:
        raise ValueError(f"matrix shape and byte count disagree: {path}")
    return np.memmap(
        path,
        dtype="<f4",
        mode="r",
        shape=(specification["rows"], specification["columns"]),
    )


def aggregate_metric_arrays(arrays: dict[str, npt.NDArray[np.float64]]) -> dict[str, float]:
    return {metric: float(np.mean(arrays[metric])) for metric in METRICS}


def exact_top_k(scores: npt.NDArray[np.float32], top_k: int) -> npt.NDArray[np.int64]:
    """Return exact score-descending top-k with document-index tie breaking."""
    if scores.ndim != 2 or not 1 <= top_k <= scores.shape[1]:
        raise ValueError("top_k must fit a non-empty two-dimensional score matrix")
    boundary_candidates = np.argpartition(scores, scores.shape[1] - top_k, axis=1)[
        :, -top_k:
    ]
    rankings = np.empty((scores.shape[0], top_k), dtype=np.int64)
    for row_index, candidates in enumerate(boundary_candidates):
        row = scores[row_index]
        boundary = float(np.min(row[candidates]))
        above = np.flatnonzero(row > boundary)
        tied = np.flatnonzero(row == boundary)[: top_k - len(above)]
        selected = np.concatenate((above, tied))
        order = np.lexsort((selected, -row[selected]))
        rankings[row_index] = selected[order]
    return rankings


def evaluate_engine(
    manifest_path: Path,
    lock: dict[str, Any],
    lock_sha256: str,
    cache: Path,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, npt.NDArray[np.float64]]]]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["suite_id"] != lock["suite_id"]:
        raise ValueError(f"embedding suite id differs from the frozen lock: {manifest_path}")
    if manifest["suite_lock_sha256"] != lock_sha256:
        raise ValueError(f"embedding suite lock hash differs: {manifest_path}")
    manifest_datasets = {dataset["id"]: dataset for dataset in manifest["datasets"]}
    internal: dict[str, dict[str, npt.NDArray[np.float64]]] = {}
    public_datasets: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    for dataset in lock["datasets"]:
        dataset_id = dataset["id"]
        embedded = manifest_datasets.get(dataset_id)
        if embedded is None:
            raise ValueError(f"embedding manifest is missing {dataset_id}")
        document_ids, query_ids, relevance = load_frozen_dataset(cache, dataset)
        corpus = load_matrix(manifest_path.parent, embedded["matrices"]["corpus"])
        queries = load_matrix(manifest_path.parent, embedded["matrices"]["queries"])
        if corpus.shape[0] != len(document_ids) or queries.shape[0] != len(query_ids):
            raise ValueError(f"embedding rows do not match frozen ids for {dataset_id}")
        if corpus.shape[1] != queries.shape[1]:
            raise ValueError(f"query and corpus dimensions differ for {dataset_id}")

        metric_arrays = {metric: np.empty(len(query_ids), dtype=np.float64) for metric in METRICS}
        top_k = min(100, len(document_ids))
        dataset_started = time.perf_counter()
        for start in range(0, len(query_ids), batch_size):
            end = min(start + batch_size, len(query_ids))
            scores = np.asarray(queries[start:end]) @ np.asarray(corpus).T
            if not np.isfinite(scores).all():
                raise ValueError(f"non-finite retrieval score in {dataset_id}")
            rankings = exact_top_k(scores, top_k)
            for offset, ranking in enumerate(rankings):
                query_index = start + offset
                values = query_metrics(ranking, relevance[query_index])
                for metric in METRICS:
                    metric_arrays[metric][query_index] = values[metric]
            if end == len(query_ids) or end % (batch_size * 8) == 0:
                elapsed = max(time.perf_counter() - dataset_started, 0.001)
                rate = end / elapsed
                print(
                    f"\r{manifest['engine']['id']} {dataset_id}: "
                    f"{end}/{len(query_ids)} ({rate:.1f} queries/s)",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        print(file=sys.stderr)
        internal[dataset_id] = metric_arrays
        public_datasets[dataset_id] = {
            "family": dataset["family"],
            "slice": dataset["slice"],
            "queries": len(query_ids),
            "documents": len(document_ids),
            "positive_qrels": dataset["positive_qrels"],
            "metrics": aggregate_metric_arrays(metric_arrays),
        }

    macro = {
        metric: float(np.mean([values[metric].mean() for values in internal.values()]))
        for metric in METRICS
    }
    micro = {
        metric: float(np.concatenate([values[metric] for values in internal.values()]).mean())
        for metric in METRICS
    }
    public = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "engine": manifest["engine"],
        "duration_ms": (time.perf_counter() - started) * 1000,
        "macro": macro,
        "micro": micro,
        "datasets": public_datasets,
    }
    return public, internal


def paired_bootstrap(
    left: dict[str, dict[str, npt.NDArray[np.float64]]],
    right: dict[str, dict[str, npt.NDArray[np.float64]]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    dataset_ids = sorted(left)
    if dataset_ids != sorted(right):
        raise ValueError("paired engines do not contain the same datasets")
    deltas = []
    for dataset_id in dataset_ids:
        left_values = left[dataset_id]["ndcg_at_10"]
        right_values = right[dataset_id]["ndcg_at_10"]
        if left_values.shape != right_values.shape:
            raise ValueError(f"paired query counts differ for {dataset_id}")
        deltas.append(left_values - right_values)

    random = np.random.default_rng(seed)
    samples = np.zeros(iterations, dtype=np.float64)
    chunk_size = 32
    for values in deltas:
        for start in range(0, iterations, chunk_size):
            end = min(start + chunk_size, iterations)
            indices = random.integers(0, len(values), size=(end - start, len(values)))
            samples[start:end] += values[indices].mean(axis=1) / len(deltas)
    estimate = float(np.mean([values.mean() for values in deltas]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    lower_value = float(lower)
    upper_value = float(upper)
    return {
        "metric": "macro_ndcg_at_10",
        "method": "paired stratified bootstrap by frozen dataset",
        "iterations": iterations,
        "seed": seed,
        "estimate": estimate,
        "confidence_level": 0.95,
        "lower": lower_value,
        "upper": upper_value,
        "non_inferiority_margin": -0.02,
        "verdict": (
            "superior"
            if lower_value > 0
            else "non_inferior"
            if lower_value >= -0.02
            else "regressed"
        ),
    }


def metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        scope: {metric: left[scope][metric] - right[scope][metric] for metric in METRICS}
        for scope in ("macro", "micro")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="left engine manifest.json")
    parser.add_argument("--right", type=Path, required=True, help="right engine manifest.json")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, help="write JSON report here instead of stdout")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_iterations < 1:
        parser.error("batch size and bootstrap iterations must be positive")
    return args


def render_summary(report: dict[str, Any]) -> str:
    left = report["results"]["left"]
    right = report["results"]["right"]
    bootstrap = report["comparison"]["bootstrap_macro_ndcg_at_10"]
    rows = []
    for metric in METRICS:
        delta = report["comparison"]["metric_deltas"]["macro"][metric]
        rows.append(
            f"{metric:<18}{left['macro'][metric] * 100:>10.2f}%"
            f"{right['macro'][metric] * 100:>12.2f}%{delta * 100:>+11.2f} pp"
        )
    return "\n".join(
        [
            "",
            f"Frozen retrieval: {left['engine']['id']} vs {right['engine']['id']}",
            f"{'metric':<18}{'left':>11}{'right':>13}{'delta':>14}",
            *rows,
            "",
            f"paired macro nDCG@10 delta 95% CI: "
            f"[{bootstrap['lower'] * 100:.2f}, {bootstrap['upper'] * 100:.2f}] pp "
            f"({bootstrap['verdict']})",
            "Formal output is aggregate-only; no per-query test diagnostics were emitted.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    lock_path = args.lock.resolve()
    cache = args.cache.resolve()
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    left_public, left_internal = evaluate_engine(
        args.left.resolve(), lock, lock_sha256, cache, args.batch_size
    )
    right_public, right_internal = evaluate_engine(
        args.right.resolve(), lock, lock_sha256, cache, args.batch_size
    )
    report = {
        "schema_version": 1,
        "suite": {
            "id": lock["suite_id"],
            "lock_path": str(lock_path),
            "lock_sha256": lock_sha256,
            "status": "frozen test; aggregate release evaluation only",
        },
        "protocol": {
            "similarity": "exact dot product over every document in each frozen corpus",
            "tie_break": "frozen corpus id order",
            "metrics": list(METRICS),
            "primary_metric": "macro nDCG@10 across frozen datasets",
            "test_diagnostics": "aggregate only; per-query results are not serialized",
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
        "results": {"left": left_public, "right": right_public},
        "comparison": {
            "metric_deltas": metric_deltas(left_public, right_public),
            "bootstrap_macro_ndcg_at_10": paired_bootstrap(
                left_internal,
                right_internal,
                iterations=args.bootstrap_iterations,
                seed=args.seed,
            ),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    sys.stderr.write(render_summary(report))


if __name__ == "__main__":
    main()
