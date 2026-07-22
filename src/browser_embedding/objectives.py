"""Composable Matryoshka distillation and retrieval objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from browser_embedding.config import ModelConfig, ObjectiveConfig


def cosine_distillation(student: Tensor, teacher: Tensor) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError(f"student/teacher shapes differ: {student.shape} vs {teacher.shape}")
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def relational_distillation(student: Tensor, teacher: Tensor) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError(f"student/teacher shapes differ: {student.shape} vs {teacher.shape}")
    if student.shape[0] < 2:
        return student.sum() * 0
    student_similarity = student @ student.T
    teacher_similarity = teacher @ teacher.T
    off_diagonal = ~torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
    return (student_similarity - teacher_similarity)[off_diagonal].square().mean()


def multi_positive_infonce(
    queries: Tensor,
    documents: Tensor,
    group_ids: Tensor,
    temperature: float,
) -> Tensor:
    if queries.shape != documents.shape:
        raise ValueError("query and document embedding shapes differ")
    logits = (queries @ documents.T) / temperature
    groups = group_ids.to(logits.device)
    positives = groups[:, None] == groups[None, :]
    negative_infinity = torch.finfo(logits.dtype).min

    def direction(scores: Tensor, mask: Tensor) -> Tensor:
        return (
            torch.logsumexp(scores, dim=1)
            - torch.logsumexp(scores.masked_fill(~mask, negative_infinity), dim=1)
        ).mean()

    return 0.5 * (direction(logits, positives) + direction(logits.T, positives.T))


@dataclass(frozen=True)
class ObjectiveResult:
    loss: Tensor
    components: dict[str, float]


class MatryoshkaObjective(nn.Module):
    """Apply the same semantic objectives at every supported output width."""

    def __init__(self, model: ModelConfig, config: ObjectiveConfig) -> None:
        super().__init__()
        self.dimensions = model.matryoshka_dims
        self.distillation_dimensions = config.distillation_dims or (model.output_dim,)
        unsupported = set(self.distillation_dimensions) - set(self.dimensions)
        if unsupported:
            raise ValueError(f"distillation dimensions are not Matryoshka outputs: {unsupported}")
        self.config = config

    def forward(self, student: Tensor, teacher: Tensor, group_ids: Tensor) -> ObjectiveResult:
        if student.ndim != 3 or student.shape[1] != 2:
            raise ValueError("student embeddings must have shape [pairs, 2, dimension]")
        if teacher.shape != student.shape:
            raise ValueError(f"teacher shape {teacher.shape} does not match {student.shape}")
        totals: dict[str, Tensor] = {
            "pointwise": student.sum() * 0,
            "relational": student.sum() * 0,
            "retrieval": student.sum() * 0,
        }
        for dimension in self.dimensions:
            student_view = F.normalize(student[..., :dimension], dim=-1)
            if dimension in self.distillation_dimensions:
                teacher_view = F.normalize(teacher[..., :dimension], dim=-1)
                flat_student = student_view.flatten(0, 1)
                flat_teacher = teacher_view.flatten(0, 1)
                totals["pointwise"] = totals["pointwise"] + cosine_distillation(
                    flat_student, flat_teacher
                )
                totals["relational"] = totals["relational"] + relational_distillation(
                    flat_student, flat_teacher
                )
            totals["retrieval"] = totals["retrieval"] + multi_positive_infonce(
                student_view[:, 0],
                student_view[:, 1],
                group_ids,
                self.config.temperature,
            )
        averaged = {
            "pointwise": totals["pointwise"] / float(len(self.distillation_dimensions)),
            "relational": totals["relational"] / float(len(self.distillation_dimensions)),
            "retrieval": totals["retrieval"] / float(len(self.dimensions)),
        }
        loss = (
            self.config.pointwise_weight * averaged["pointwise"]
            + self.config.relational_weight * averaged["relational"]
            + self.config.retrieval_weight * averaged["retrieval"]
        )
        return ObjectiveResult(
            loss=loss,
            components={
                "total": float(loss.detach()),
                **{name: float(value.detach()) for name, value in averaged.items()},
            },
        )
