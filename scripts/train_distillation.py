from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from muse_physio.data import create_dataloader, load_manifest
from muse_physio.distillation import DistillationStudent, distillation_loss
from muse_physio.modalities import resolve_modality_selection, validate_model_input_dim
from muse_physio.model import TimeSeriesTransformer
from muse_physio.training import (
    evaluate_regression,
    load_yaml,
    move_batch,
    resolve_device,
    set_seed,
    write_history,
    write_json,
)


def train_distillation_epoch(
    wrapper: DistillationStudent,
    teacher: TimeSeriesTransformer,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: dict[str, Any],
    gradient_clip_norm: float | None,
    student_input_transform: Any,
    teacher_input_transform: Any,
) -> dict[str, float]:
    wrapper.train()
    teacher.eval()
    keys = [
        "loss",
        "supervised_loss",
        "ccc_loss",
        "mse_loss",
        "relation_distillation_loss",
    ]
    totals = {key: 0.0 for key in keys}
    total_steps = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        student_x = student_input_transform(batch["x"])
        teacher_x = teacher_input_transform(batch["x"])
        optimizer.zero_grad(set_to_none=True)
        loss, parts = distillation_loss(
            wrapper,
            teacher,
            batch,
            loss_config,
            student_x=student_x,
            teacher_x=teacher_x,
        )
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), gradient_clip_norm)
        optimizer.step()

        valid_steps = int(batch["target_mask"].sum())
        total_steps += valid_steps
        totals["loss"] += float(loss.detach()) * valid_steps
        for key in keys[1:]:
            totals[key] += parts[key] * valid_steps

    if total_steps == 0:
        raise ValueError("Training epoch contained no valid target values")
    return {key: value / total_steps for key, value in totals.items()}


def run(
    config_path: Path,
    teacher_override: Path | None = None,
    epochs_override: int | None = None,
    output_dir_override: Path | None = None,
) -> Path:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    data_config = config["data"]
    processed_dir = Path(data_config["processed_dir"])
    manifest = load_manifest(processed_dir)
    teacher_input_selection = resolve_modality_selection(manifest)
    student_input_selection = resolve_modality_selection(
        manifest,
        config.get("student_input"),
    )

    teacher_path = teacher_override or Path(config["teacher_checkpoint"])
    teacher_checkpoint = torch.load(
        teacher_path,
        map_location=device,
        weights_only=False,
    )
    teacher = TimeSeriesTransformer.from_config(
        teacher_checkpoint["model_config"]
    ).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student = TimeSeriesTransformer.from_config(config["model"]).to(device)
    layer_pairs = config["distillation"]["layer_pairs"]
    wrapper = DistillationStudent(
        student,
        layer_pairs,
    ).to(device)

    validate_model_input_dim(
        teacher_checkpoint["model_config"],
        teacher_input_selection,
        model_name="Teacher",
    )
    validate_model_input_dim(
        config["model"],
        student_input_selection,
        model_name="Student",
    )
    for _, teacher_layer in wrapper.layer_pairs:
        if teacher_layer < 0 or teacher_layer >= teacher.num_layers:
            raise ValueError(f"Invalid teacher layer index: {teacher_layer}")

    pin_memory = device.type == "cuda"
    train_loader = create_dataloader(
        processed_dir,
        "train",
        int(data_config["train_batch_size"]),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    devel_loader = create_dataloader(
        processed_dir,
        "devel",
        int(data_config["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        wrapper.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    scheduler_config = config.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 5)),
        min_lr=float(scheduler_config.get("min_learning_rate", 1e-6)),
    )

    training_config = config["training"]
    output_dir = output_dir_override or Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_ccc = float("-inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    epochs = epochs_override or int(training_config["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = train_distillation_epoch(
            wrapper,
            teacher,
            train_loader,
            optimizer,
            device,
            config["distillation"],
            float(training_config["gradient_clip_norm"])
            if training_config.get("gradient_clip_norm") is not None
            else None,
            student_input_selection.apply,
            teacher_input_selection.apply,
        )
        devel_metrics = evaluate_regression(
            wrapper.student,
            devel_loader,
            device,
            input_transform=student_input_selection.apply,
        )
        scheduler.step(devel_metrics["ccc"])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"devel_{key}": value for key, value in devel_metrics.items()},
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"devel_ccc={devel_metrics['ccc']:.6f}"
        )

        if devel_metrics["ccc"] > best_ccc:
            best_ccc = devel_metrics["ccc"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": wrapper.student.state_dict(),
                    "model_config": config["model"],
                    "training_config": config,
                    "teacher_checkpoint": str(teacher_path),
                    "model_input": student_input_selection.to_config(),
                    "teacher_input": teacher_input_selection.to_config(),
                    "manifest": manifest,
                    "epoch": epoch,
                    "devel_metrics": devel_metrics,
                },
                best_path,
            )
            write_json(output_dir / "best_metrics.json", devel_metrics)
        else:
            stale_epochs += 1

        if stale_epochs >= int(training_config.get("early_stopping_patience", 15)):
            print(f"Early stopping after epoch {epoch}.")
            break

    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a MuSe-Physio Transformer teacher into a student."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    checkpoint = run(
        arguments.config,
        arguments.teacher_checkpoint,
        arguments.epochs,
        arguments.output_dir,
    )
    print(f"Best checkpoint: {checkpoint}")
