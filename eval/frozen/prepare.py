"""Materialize and verify retrieval-frozen-v1 from immutable source revisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_SUITE = EVAL_DIR / "retrieval-frozen-v1.json"
DEFAULT_LOCK = EVAL_DIR / "retrieval-frozen-v1.lock.json"
DEFAULT_CACHE = EVAL_DIR / "cache" / "retrieval-frozen-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def compose_text(title: object, text: object) -> str:
    title_text = "" if title is None else str(title).strip()
    body_text = "" if text is None else str(text).strip()
    if title_text and body_text:
        return f"{title_text}\n{body_text}"
    return title_text or body_text


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for record in records:
            encoded = canonical_json_bytes(record)
            stream.write(encoded)
            digest.update(encoded)
            count += 1
    return count, digest.hexdigest(), path.stat().st_size


def download(repo_id: str, revision: str, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
        )
    )


def source_record(repo_id: str, revision: str, filename: str, path: Path) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "revision": revision,
        "filename": filename,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def unique_by_id(records: Iterable[dict[str, str]], kind: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for record in records:
        record_id = record["id"]
        if not record_id:
            raise ValueError(f"{kind} contains an empty id")
        if record_id in result:
            raise ValueError(f"{kind} contains duplicate id {record_id!r}")
        if not record["text"]:
            raise ValueError(f"{kind} {record_id!r} contains empty text")
        result[record_id] = record
    return result


def load_parquet_dataset(
    corpus_path: Path, queries_path: Path, qrels_path: Path
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[tuple[str, str], int]]:
    corpus_rows = pq.read_table(corpus_path).to_pylist()
    query_rows = pq.read_table(queries_path).to_pylist()
    qrel_rows = pq.read_table(qrels_path).to_pylist()
    corpus = unique_by_id(
        (
            {"id": str(row["id"]), "text": compose_text(row.get("title"), row.get("text"))}
            for row in corpus_rows
        ),
        "corpus",
    )
    queries = unique_by_id(
        ({"id": str(row["id"]), "text": str(row["text"]).strip()} for row in query_rows),
        "queries",
    )
    qrels: dict[tuple[str, str], int] = {}
    for row in qrel_rows:
        score = int(row["score"])
        if score > 0:
            key = (str(row["query-id"]), str(row["corpus-id"]))
            qrels[key] = max(score, qrels.get(key, 0))
    return corpus, queries, qrels


def load_beir_dataset(
    corpus_path: Path, queries_path: Path, qrels_path: Path
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[tuple[str, str], int]]:
    corpus_rows = pq.read_table(corpus_path).to_pylist()
    query_rows = pq.read_table(queries_path).to_pylist()
    corpus = unique_by_id(
        (
            {"id": str(row["_id"]), "text": compose_text(row.get("title"), row.get("text"))}
            for row in corpus_rows
        ),
        "corpus",
    )
    queries = unique_by_id(
        (
            {"id": str(row["_id"]), "text": compose_text(row.get("title"), row.get("text"))}
            for row in query_rows
        ),
        "queries",
    )
    qrels: dict[tuple[str, str], int] = {}
    with qrels_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                key = (str(row["query-id"]), str(row["corpus-id"]))
                qrels[key] = max(score, qrels.get(key, 0))
    return corpus, queries, qrels


def normalize_dataset(
    dataset: dict[str, Any], cache: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = dataset["source"]
    source_paths = {
        kind: download(source["repo_id"], source["revision"], source[kind])
        for kind in ("corpus", "queries")
    }
    qrels_source = dataset.get("qrels_source", source)
    source_paths["qrels"] = download(
        qrels_source["repo_id"], qrels_source["revision"], qrels_source["qrels"]
    )
    sources = [
        source_record(source["repo_id"], source["revision"], source[kind], source_paths[kind])
        for kind in ("corpus", "queries")
    ]
    sources.append(
        source_record(
            qrels_source["repo_id"],
            qrels_source["revision"],
            qrels_source["qrels"],
            source_paths["qrels"],
        )
    )

    if dataset["format"] == "mteb_parquet":
        corpus, queries, qrels = load_parquet_dataset(
            source_paths["corpus"], source_paths["queries"], source_paths["qrels"]
        )
    elif dataset["format"] == "beir_parquet_tsv":
        corpus, queries, qrels = load_beir_dataset(
            source_paths["corpus"], source_paths["queries"], source_paths["qrels"]
        )
    else:
        raise ValueError(f"unsupported dataset format: {dataset['format']}")

    judged_query_ids = {query_id for query_id, _ in qrels}
    missing_queries = judged_query_ids - queries.keys()
    missing_documents = {corpus_id for _, corpus_id in qrels} - corpus.keys()
    if missing_queries:
        raise ValueError(
            f"{dataset['id']} qrels reference missing queries: {sorted(missing_queries)[:5]}"
        )
    if missing_documents:
        raise ValueError(
            f"{dataset['id']} qrels reference missing documents: {sorted(missing_documents)[:5]}"
        )
    queries = {key: value for key, value in queries.items() if key in judged_query_ids}
    if not corpus or not queries or not qrels:
        raise ValueError(f"{dataset['id']} normalized to an empty component")

    dataset_dir = cache / "datasets" / dataset["id"]
    relative_dir = Path("datasets") / dataset["id"]
    artifact_specs = {
        "corpus": (
            dataset_dir / "corpus.jsonl",
            relative_dir / "corpus.jsonl",
            (corpus[key] for key in sorted(corpus)),
        ),
        "queries": (
            dataset_dir / "queries.jsonl",
            relative_dir / "queries.jsonl",
            (queries[key] for key in sorted(queries)),
        ),
        "qrels": (
            dataset_dir / "qrels.jsonl",
            relative_dir / "qrels.jsonl",
            (
                {
                    "query_id": query_id,
                    "corpus_id": corpus_id,
                    "relevance": qrels[(query_id, corpus_id)],
                }
                for query_id, corpus_id in sorted(qrels)
            ),
        ),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, (path, relative_path, records) in artifact_specs.items():
        count, digest, byte_count = write_jsonl(path, records)
        artifacts[name] = {
            "path": relative_path.as_posix(),
            "records": count,
            "bytes": byte_count,
            "sha256": digest,
        }

    return (
        {
            "id": dataset["id"],
            "family": dataset["family"],
            "slice": dataset["slice"],
            "license": dataset["license"],
            "artifacts": artifacts,
            "positive_qrels": len(qrels),
        },
        sources,
    )


def build_lock(suite_path: Path, cache: Path) -> dict[str, Any]:
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    normalized = []
    all_sources: list[dict[str, Any]] = []
    for dataset in suite["datasets"]:
        print(f"Preparing {dataset['id']}...", flush=True)
        record, sources = normalize_dataset(dataset, cache)
        normalized.append(record)
        all_sources.extend(sources)
    deduplicated_sources = {
        (record["repo_id"], record["revision"], record["filename"]): record
        for record in all_sources
    }
    return {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "suite_manifest_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "normalization_recipe_version": suite["normalization"]["recipe_version"],
        "sources": [deduplicated_sources[key] for key in sorted(deduplicated_sources)],
        "datasets": normalized,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--update-lock",
        action="store_true",
        help="intentionally replace the committed lock after reviewing a new suite id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual = build_lock(args.suite.resolve(), args.cache.resolve())
    encoded = json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.lock.exists() and not args.update_lock:
        expected = json.loads(args.lock.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit(
                "Frozen suite verification failed: source or normalized hashes differ from "
                f"{args.lock}. Use --update-lock only for an intentional new suite revision."
            )
        print(f"Verified {actual['suite_id']} against {args.lock}")
        return
    if not args.update_lock:
        raise SystemExit(
            f"Missing lock file {args.lock}; initialize it explicitly with --update-lock"
        )
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(encoded, encoding="utf-8")
    print(f"Wrote frozen lock {args.lock}")


if __name__ == "__main__":
    main()
