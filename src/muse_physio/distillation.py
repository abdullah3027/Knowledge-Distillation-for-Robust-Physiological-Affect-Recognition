from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import TimeSeriesTransformer
from .training import supervised_loss


class DistillationStudent(nn.Module):
    """Student model with configured teacher/student relation-layer pairs."""

    def __init__(
        self,
        student: TimeSeriesTransformer,
        layer_pairs: list[list[int]],
    ) -> None:
        super().__init__()
        self.student = student
        self.layer_pairs = [(int(pair[0]), int(pair[1])) for pair in layer_pairs]

        for student_layer, _ in self.layer_pairs:
            if student_layer < 0 or student_layer >= student.num_layers:
                raise ValueError(f"Invalid student layer index: {student_layer}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        prediction, hidden_states = self.student(
            x,
            padding_mask,
            return_hidden_states=True,
        )
        return prediction, hidden_states


def temporal_relation_matrix(hidden: torch.Tensor) -> torch.Tensor:
    """Return pairwise cosine similarities between all timesteps."""
    normalized = F.normalize(hidden, p=2, dim=-1)
    return torch.bmm(normalized, normalized.transpose(1, 2))


def masked_relation_mse(
    student_relation: torch.Tensor,
    teacher_relation: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    if student_relation.shape != teacher_relation.shape:
        raise ValueError(
            "Student and teacher relation shapes differ: "
            f"{student_relation.shape} vs {teacher_relation.shape}"
        )
    if padding_mask.shape != student_relation.shape[:2]:
        raise ValueError(
            f"Padding mask {padding_mask.shape} does not match relation matrix "
            f"{student_relation.shape}"
        )

    valid_steps = ~padding_mask
    valid_pairs = valid_steps.unsqueeze(2) & valid_steps.unsqueeze(1)
    if not valid_pairs.any():
        raise ValueError("Relation loss received no valid timestep pairs")
    difference = student_relation[valid_pairs] - teacher_relation[valid_pairs]
    return torch.mean(difference**2)


def distillation_loss(
    wrapper: DistillationStudent,
    teacher: TimeSeriesTransformer,
    batch: dict[str, Any],
    config: dict[str, Any],
    *,
    student_x: torch.Tensor | None = None,
    teacher_x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_input = student_x if student_x is not None else batch["x"]
    teacher_input = teacher_x if teacher_x is not None else batch["x"]
    student_prediction, student_states = wrapper(
        student_input,
        batch["padding_mask"],
    )
    with torch.no_grad():
        _, teacher_states = teacher(
            teacher_input,
            batch["padding_mask"],
            return_hidden_states=True,
        )

    supervised, supervised_parts = supervised_loss(
        student_prediction,
        batch["y"],
        batch["target_mask"],
        ccc_weight=float(config.get("ccc_weight", 1.0)),
        mse_weight=float(config.get("mse_weight", 0.0)),
    )

    relation_losses: list[torch.Tensor] = []
    for student_layer, teacher_layer in wrapper.layer_pairs:
        if teacher_layer < 0 or teacher_layer >= len(teacher_states):
            raise ValueError(f"Invalid teacher layer index: {teacher_layer}")
        relation_losses.append(
            masked_relation_mse(
                temporal_relation_matrix(student_states[student_layer]),
                temporal_relation_matrix(teacher_states[teacher_layer]),
                batch["padding_mask"],
            )
        )

    relation_loss = (
        torch.stack(relation_losses).mean()
        if relation_losses
        else student_prediction.new_zeros(())
    )
    total = (
        float(config.get("supervised_weight", 1.0)) * supervised
        + float(config.get("relation_weight", 0.1)) * relation_loss
    )
    return total, {
        **supervised_parts,
        "supervised_loss": float(supervised.detach()),
        "relation_distillation_loss": float(relation_loss.detach()),
    }
