"""Retrieval evaluation independent from the training loop."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from browser_embedding.model import BrowserEncoder


def retrieval_metrics(
    queries: np.ndarray, documents: np.ndarray, relevant: np.ndarray | None = None
) -> dict[str, float | int]:
    if queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]:
        raise ValueError("queries and documents must be compatible rank-two matrices")
    if relevant is None:
        relevant = np.arange(queries.shape[0])
    scores = queries @ documents.T
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty(queries.shape[0], dtype=np.int64)
    for row, target in enumerate(relevant):
        ranks[row] = int(np.flatnonzero(order[row] == target)[0]) + 1
    return {
        "queries": int(queries.shape[0]),
        "documents": int(documents.shape[0]),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_3": float(np.mean(ranks <= 3)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr_at_10": float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0))),
        "ndcg_at_10": float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0))),
        "mean_rank": float(np.mean(ranks)),
    }


@torch.inference_mode()
def evaluate_pairs(
    model: BrowserEncoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    query_chunks: list[np.ndarray] = []
    document_chunks: list[np.ndarray] = []
    teacher_chunks: list[np.ndarray] = []
    variants: list[str] = []
    for batch in loader:
        token_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        pairs, roles, sequence = token_ids.shape
        output = model(
            token_ids.reshape(pairs * roles, sequence),
            mask.reshape(pairs * roles, sequence),
        ).reshape(pairs, roles, -1)
        query_chunks.append(output[:, 0].cpu().numpy())
        document_chunks.append(output[:, 1].cpu().numpy())
        teacher_chunks.append(batch["teacher_embeddings"].numpy())
        variants.extend(str(value) for value in batch["variant"])

    queries = np.concatenate(query_chunks)
    documents = np.concatenate(document_chunks)
    teachers = np.concatenate(teacher_chunks)
    result: dict[str, Any] = retrieval_metrics(queries, documents)
    result["pointwise_cosine"] = float(
        np.mean(
            np.concatenate(
                [
                    np.sum(queries * teachers[:, 0], axis=1),
                    np.sum(documents * teachers[:, 1], axis=1),
                ]
            )
        )
    )
    by_variant: dict[str, list[int]] = defaultdict(list)
    for index, variant in enumerate(variants):
        by_variant[variant].append(index)
    result["by_variant"] = {
        name: retrieval_metrics(queries[indices], documents[indices])
        for name, indices in sorted(by_variant.items())
    }
    return result


@torch.inference_mode()
def evaluate_multilingual_seed(
    model: BrowserEncoder,
    *,
    tokenizer_path: Path,
    seed_path: Path,
    max_sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate en/zh/cross-language/code-switched retrieval slices.

    ``transformers`` is imported lazily so the synthetic smoke path keeps a
    small dependency surface and never touches the network.
    """
    try:
        from transformers import (  # type: ignore[import-not-found]
            AutoTokenizer,
            PreTrainedTokenizerFast,
        )
    except ImportError as error:  # pragma: no cover - exercised in full env
        raise RuntimeError("multilingual seed evaluation requires the 'data' extra") from error

    records = [
        json.loads(line)
        for line in seed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"query_en", "query_zh", "query_mixed", "doc_en", "doc_zh"}
    if not records or any(required - record.keys() for record in records):
        raise ValueError(f"{seed_path} does not satisfy the multilingual seed contract")
    tokenizer = (
        PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
        if tokenizer_path.is_file()
        else AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    )

    def encode(texts: list[str], role: str) -> np.ndarray:
        prefix = "[QRY] " if role == "query" else "[DOC] "
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                [prefix + text for text in texts[start : start + batch_size]],
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            )
            chunks.append(
                model(
                    encoded["input_ids"].to(device),
                    encoded["attention_mask"].to(device),
                )
                .cpu()
                .numpy()
            )
        return np.concatenate(chunks)

    documents_en = encode([str(record["doc_en"]) for record in records], "document")
    documents_zh = encode([str(record["doc_zh"]) for record in records], "document")
    queries_en = encode([str(record["query_en"]) for record in records], "query")
    queries_zh = encode([str(record["query_zh"]) for record in records], "query")
    queries_mixed = encode([str(record["query_mixed"]) for record in records], "query")
    slices = {
        "en_en": retrieval_metrics(queries_en, documents_en),
        "zh_zh": retrieval_metrics(queries_zh, documents_zh),
        "zh_en": retrieval_metrics(queries_zh, documents_en),
        "en_zh": retrieval_metrics(queries_en, documents_zh),
        "mixed_en": retrieval_metrics(queries_mixed, documents_en),
        "mixed_zh": retrieval_metrics(queries_mixed, documents_zh),
    }
    metric_names = ("recall_at_1", "recall_at_3", "recall_at_10", "mrr_at_10", "ndcg_at_10")
    macro = {
        f"macro_{name}": float(np.mean([float(value[name]) for value in slices.values()]))
        for name in metric_names
    }
    return {**macro, "slices": slices}
