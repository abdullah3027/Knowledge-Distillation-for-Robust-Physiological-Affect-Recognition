from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from muse_physio.data import create_dataloader, load_manifest
from muse_physio.modalities import resolve_modality_selection, validate_model_input_dim
from muse_physio.model import TimeSeriesTransformer, count_trainable_parameters
from muse_physio.training import (
    evaluate_regression,
    load_yaml,
    resolve_device,
    set_seed,
    train_supervised_epoch,
    write_history,
    write_json,
)


def validate_model_data_contract(
    model_config: dict[str, Any],
    manifest: dict[str, Any],
    selection: Any,
) -> None:
    validate_model_input_dim(model_config, selection)
    longest_sequence = max(
        summary["x_shape"][1] for summary in manifest["splits"].values()
    )
    if int(model_config["max_sequence_length"]) < longest_sequence:
        raise ValueError(
            "model.max_sequence_length is shorter than processed data: "
            f"{model_config['max_sequence_length']} < {longest_sequence}"
        )


def run(
    config_path: Path,
    epochs_override: int | None = None,
    output_dir_override: Path | None = None,
) -> Path:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))

    data_config = config["data"]
    processed_dir = Path(data_config["processed_dir"])
    manifest = load_manifest(processed_dir)
    input_selection = resolve_modality_selection(
        manifest,
        config.get("student_input"),
    )
    validate_model_data_contract(config["model"], manifest, input_selection)

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

    model = TimeSeriesTransformer.from_config(config["model"]).to(device)
    model_parameter_count = count_trainable_parameters(model)
    print(f"Model parameters: {model_parameter_count:,}")
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    history: list[dict[str, Any]] = []
    best_ccc = float("-inf")
    stale_epochs = 0
    best_path = output_dir / "best.pt"

    epochs = epochs_override or int(training_config["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = train_supervised_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config["loss"],
            float(training_config["gradient_clip_norm"])
            if training_config.get("gradient_clip_norm") is not None
            else None,
            input_transform=input_selection.apply,
        )
        devel_metrics = evaluate_regression(
            model,
            devel_loader,
            device,
            input_transform=input_selection.apply,
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
            f"devel_ccc={devel_metrics['ccc']:.6f} "
            f"devel_mse={devel_metrics['mse']:.6f}"
        )

        if devel_metrics["ccc"] > best_ccc:
            best_ccc = devel_metrics["ccc"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["model"],
                    "model_parameter_count": model_parameter_count,
                    "training_config": config,
                    "model_input": input_selection.to_config(),
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
        description="Train a MuSe-Physio Transformer teacher or student."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    checkpoint = run(
        arguments.config,
        arguments.epochs,
        arguments.output_dir,
    )
    print(f"Best checkpoint: {checkpoint}")
