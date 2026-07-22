from pathlib import Path

import torch
from torch.nn import functional as F

from browser_embedding.config import load_experiment
from browser_embedding.objectives import MatryoshkaObjective, multi_positive_infonce


def test_aligned_pairs_have_lower_retrieval_loss() -> None:
    queries = torch.eye(4)
    groups = torch.arange(4)
    aligned = multi_positive_infonce(queries, queries, groups, 0.1)
    permuted = multi_positive_infonce(queries, queries.roll(1, 0), groups, 0.1)
    assert aligned < permuted


def test_matryoshka_objective_backpropagates() -> None:
    config, _ = load_experiment(Path("configs/smoke.yaml"))
    objective = MatryoshkaObjective(config.model, config.objective)
    student = F.normalize(torch.randn(4, 2, config.model.output_dim), dim=-1).requires_grad_()
    teacher = F.normalize(torch.randn_like(student), dim=-1)
    result = objective(student, teacher, torch.arange(4))
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert set(result.components) == {"total", "pointwise", "relational", "retrieval"}
