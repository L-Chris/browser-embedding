from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


prepare = load_module("frozen_prepare", ROOT / "eval" / "frozen" / "prepare.py")
score = load_module("frozen_score", ROOT / "eval" / "frozen" / "score.py")


def test_compose_text_has_a_frozen_title_body_separator() -> None:
    assert prepare.compose_text("  Title ", " Body  ") == "Title\nBody"
    assert prepare.compose_text("", " Body ") == "Body"
    assert prepare.compose_text(None, "Body") == "Body"


def test_multi_relevance_and_graded_ndcg_metrics() -> None:
    metrics = score.query_metrics(np.array([0, 1, 2, 3]), {0: 3, 2: 1})
    expected_dcg = 7.0 / math.log2(2) + 1.0 / math.log2(4)
    ideal_dcg = 7.0 / math.log2(2) + 1.0 / math.log2(3)

    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["recall_at_100"] == 1.0
    assert metrics["mrr_at_10"] == 1.0
    assert metrics["map_at_100"] == (1.0 + 2.0 / 3.0) / 2.0
    assert metrics["ndcg_at_10"] == expected_dcg / ideal_dcg


def test_cutoffs_do_not_credit_hits_beyond_the_cutoff() -> None:
    ranking = np.arange(100)
    metrics = score.query_metrics(ranking, {10: 1})

    assert metrics["recall_at_10"] == 0.0
    assert metrics["recall_at_100"] == 1.0
    assert metrics["mrr_at_10"] == 0.0
    assert metrics["map_at_100"] == 1.0 / 11.0
    assert metrics["ndcg_at_10"] == 0.0


def test_exact_top_k_resolves_boundary_ties_by_frozen_document_order() -> None:
    scores = np.array([[1.0, 0.5, 0.5, 0.5], [0.0, 2.0, 1.0, 3.0]], dtype=np.float32)

    assert score.exact_top_k(scores, 2).tolist() == [[0, 1], [3, 1]]
