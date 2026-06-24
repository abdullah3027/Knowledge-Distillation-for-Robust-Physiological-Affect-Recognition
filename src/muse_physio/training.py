from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .metrics import masked_ccc, masked_ccc_loss, masked_mae, masked_mse, masked_pearson


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def supervised_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    ccc_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ccc_loss = masked_ccc_loss(prediction, target, target_mask)
    mse_loss = masked_mse(prediction, target, target_mask)
    total = ccc_weight * ccc_loss + mse_weight * mse_loss
    return total, {
        "ccc_loss": float(ccc_loss.detach()),
        "mse_loss": float(mse_loss.detach()),
    }


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: dict[str, Any],
    gradient_clip_norm: float | None,
    *,
    input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ccc_loss": 0.0, "mse_loss": 0.0}
    total_steps = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        model_input = (
            input_transform(batch["x"]) if input_transform is not None else batch["x"]
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(model_input, batch["padding_mask"])
        loss, parts = supervised_loss(
            prediction,
            batch["y"],
            batch["target_mask"],
            ccc_weight=float(loss_config.get("ccc_weight", 1.0)),
            mse_weight=float(loss_config.get("mse_weight", 0.0)),
        )
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        valid_steps = int(batch["target_mask"].sum())
        total_steps += valid_steps
        totals["loss"] += float(loss.detach()) * valid_steps
        totals["ccc_loss"] += parts["ccc_loss"] * valid_steps
        totals["mse_loss"] += parts["mse_loss"] * valid_steps

    if total_steps == 0:
        raise ValueError("Training epoch contained no valid target values")
    return {key: value / total_steps for key, value in totals.items()}


@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    participant_cccs: list[float] = []

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        model_input = (
            input_transform(batch["x"]) if input_transform is not None else batch["x"]
        )
        prediction = model(model_input, batch["padding_mask"])
        predictions.append(prediction.detach().cpu())
        targets.append(batch["y"].detach().cpu())
        masks.append(batch["target_mask"].detach().cpu())

        for index in range(prediction.shape[0]):
            sample_mask = batch["target_mask"][index]
            if sample_mask.any():
                participant_cccs.append(
                    float(
                        masked_ccc(
                            prediction[index],
                            batch["y"][index],
                            sample_mask,
                        ).detach()
                    )
                )

    prediction_all = torch.cat(predictions, dim=0)
    target_all = torch.cat(targets, dim=0)
    mask_all = torch.cat(masks, dim=0)
    if not mask_all.any():
        return {
            "ccc": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "mean_participant_ccc": float("nan"),
        }

    return {
        "ccc": float(masked_ccc(prediction_all, target_all, mask_all)),
        "mse": float(masked_mse(prediction_all, target_all, mask_all)),
        "mae": float(masked_mae(prediction_all, target_all, mask_all)),
        "pearson": float(masked_pearson(prediction_all, target_all, mask_all)),
        "mean_participant_ccc": float(np.mean(participant_cccs)),
    }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
