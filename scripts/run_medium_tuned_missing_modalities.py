from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_distillation
import train_transformer
from muse_physio.distillation import resolve_distillation_weights
from muse_physio.model import count_parameters_from_config
from muse_physio.training import load_yaml


TEACHER_CHECKPOINT = Path("outputs/ablations/model_size_full_modalities/medium/teacher/best.pt")
TEACHER_CONFIG = Path("configs/ablations/model_size_full_modalities/medium/teacher.yaml")

MISSING_CONFIGS = {
    "missing_bpm": {
        "no_kd": Path("configs/training/student_missing_bpm.yaml"),
        "kd": Path("configs/training/student_kd_missing_bpm.yaml"),
    },
    "missing_ecg": {
        "no_kd": Path("configs/training/student_missing_ecg.yaml"),
        "kd": Path("configs/training/student_kd_missing_ecg.yaml"),
    },
    "missing_resp": {
        "no_kd": Path("configs/training/student_missing_resp.yaml"),
        "kd": Path("configs/training/student_kd_missing_resp.yaml"),
    },
}

MEDIUM_STUDENT_CAPACITY = {
    "d_model": 80,
    "num_heads": 2,
    "num_layers": 3,
    "dim_feedforward": 240,
}

TUNED_NO_KD_LOSS = {
    "ccc_weight": 1.0,
    "mse_weight": 0.3,
}

TUNED_KD_PROFILES = {
    "kd_rw5_mse1": {
        "supervised_weight": 1.0,
        "relation_weight": 5.0,
        "ccc_weight": 1.0,
        "mse_weight": 1.0,
    },
    "kd_rw10_mse1": {
        "supervised_weight": 1.0,
        "relation_weight": 10.0,
        "ccc_weight": 1.0,
        "mse_weight": 1.0,
    },
}


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def medium_layer_pairs() -> list[list[int]]:
    return [[0, 0], [1, 2], [2, 3]]


def apply_medium_student_settings(config: dict[str, Any]) -> None:
    config["model"].update(MEDIUM_STUDENT_CAPACITY)
    config["data"]["train_batch_size"] = 16
    config["data"]["evaluation_batch_size"] = 4


def planned_count(config: dict[str, Any]) -> int:
    return count_parameters_from_config(config["model"])


def infer_modalities(config: dict[str, Any]) -> tuple[str, str]:
    modalities = config.get("student_input", {}).get("available_modalities") or [
        "BPM",
        "ECG",
        "RESP",
    ]
    missing = sorted({"BPM", "ECG", "RESP"} - set(modalities))
    return "+".join(modalities), "+".join(missing) if missing else "none"


def make_no_kd_row(
    condition: str,
    base_path: Path,
    config_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(load_yaml(base_path))
    apply_medium_student_settings(config)
    config["loss"].update(TUNED_NO_KD_LOSS)
    output_dir = output_root / condition / "no_kd_mse0_3"
    config_path = config_root / condition / "no_kd_mse0_3.yaml"
    config["experiment_name"] = f"medium_tuned_{condition}_no_kd_mse0_3"
    config["training"]["output_dir"] = output_dir.as_posix()
    write_yaml(config_path, config)
    modalities, missing = infer_modalities(config)
    return {
        "condition": condition,
        "variant": "no_kd_mse0_3",
        "role": "student_no_kd",
        "modalities": modalities,
        "missing": missing,
        "config_path": config_path,
        "output_dir": output_dir,
        "teacher_checkpoint": "",
        "parameter_count_planned": planned_count(config),
        "parameter_millions_planned": planned_count(config) / 1_000_000,
        "supervised_weight": "",
        "relation_weight": "",
        "ccc_weight": config["loss"]["ccc_weight"],
        "mse_weight": config["loss"]["mse_weight"],
    }


def make_kd_row(
    condition: str,
    base_path: Path,
    variant: str,
    weights: dict[str, float],
    config_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(load_yaml(base_path))
    apply_medium_student_settings(config)
    config["teacher_checkpoint"] = TEACHER_CHECKPOINT.as_posix()
    config["distillation"].update(weights)
    config["distillation"]["layer_pairs"] = medium_layer_pairs()
    output_dir = output_root / condition / variant
    config_path = config_root / condition / f"{variant}.yaml"
    config["experiment_name"] = f"medium_tuned_{condition}_{variant}"
    config["training"]["output_dir"] = output_dir.as_posix()
    write_yaml(config_path, config)
    modalities, missing = infer_modalities(config)
    resolved = resolve_distillation_weights(config["distillation"])
    return {
        "condition": condition,
        "variant": variant,
        "role": "student_kd",
        "modalities": modalities,
        "missing": missing,
        "config_path": config_path,
        "output_dir": output_dir,
        "teacher_checkpoint": TEACHER_CHECKPOINT,
        "parameter_count_planned": planned_count(config),
        "parameter_millions_planned": planned_count(config) / 1_000_000,
        **resolved,
    }


def build_rows(config_root: Path, output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, paths in MISSING_CONFIGS.items():
        rows.append(make_no_kd_row(condition, paths["no_kd"], config_root, output_root))
        for variant, weights in TUNED_KD_PROFILES.items():
            rows.append(
                make_kd_row(
                    condition,
                    paths["kd"],
                    variant,
                    weights,
                    config_root,
                    output_root,
                )
            )
    return rows


def history_summary(output_dir: Path) -> dict[str, Any]:
    history_path = output_dir / "history.csv"
    if not history_path.exists():
        return {}
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    best_ccc = max(rows, key=lambda row: float(row["devel_ccc"]))
    best_mse = min(rows, key=lambda row: float(row["devel_mse"]))
    best_mae = min(rows, key=lambda row: float(row["devel_mae"]))
    result: dict[str, Any] = {
        "epochs_run": len(rows),
        "best_ccc_epoch": int(best_ccc["epoch"]),
        "best_mse_any_epoch": float(best_mse["devel_mse"]),
        "best_mse_epoch": int(best_mse["epoch"]),
        "best_mae_any_epoch": float(best_mae["devel_mae"]),
        "best_mae_epoch": int(best_mae["epoch"]),
    }
    if "train_weighted_relation_distillation_loss" in best_ccc:
        train_loss = float(best_ccc["train_loss"])
        weighted_relation = float(best_ccc["train_weighted_relation_distillation_loss"])
        result.update(
            {
                "train_relation_loss_at_best_ccc": float(
                    best_ccc["train_relation_distillation_loss"]
                ),
                "train_weighted_relation_loss_at_best_ccc": weighted_relation,
                "weighted_relation_pct_at_best_ccc": 100
                * weighted_relation
                / train_loss
                if train_loss
                else "",
            }
        )
    return result


def checkpoint_parameter_count(output_dir: Path) -> int | None:
    checkpoint_path = output_dir / "best.pt"
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_parameter_count") is not None:
        return int(checkpoint["model_parameter_count"])
    return int(sum(tensor.numel() for tensor in checkpoint["model_state"].values()))


def collect_result(row: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(row["output_dir"])
    result = {**row, "output_dir": output_dir.as_posix()}
    metrics_path = output_dir / "best_metrics.json"
    if not metrics_path.exists():
        result["status"] = "missing_metrics"
        return result
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    parameter_count = checkpoint_parameter_count(output_dir)
    result.update(
        {
            "status": "complete",
            "parameter_count": parameter_count,
            "parameter_millions": parameter_count / 1_000_000
            if parameter_count
            else "",
            "ccc": metrics.get("ccc"),
            "mse": metrics.get("mse"),
            "mae": metrics.get("mae"),
            "pearson": metrics.get("pearson"),
            "mean_participant_ccc": metrics.get("mean_participant_ccc"),
        }
    )
    result.update(history_summary(output_dir))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "condition",
        "variant",
        "role",
        "status",
        "missing",
        "relation_weight",
        "mse_weight",
        "ccc",
        "mse",
        "mae",
        "pearson",
        "best_mse_any_epoch",
        "best_mae_any_epoch",
        "weighted_relation_pct_at_best_ccc",
    ]
    lines = [
        "# Medium Tuned Missing-Modality Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def execute_rows(
    rows: list[dict[str, Any]],
    *,
    epochs: int | None,
    max_runs: int | None,
    rerun_complete: bool,
) -> None:
    if not TEACHER_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing medium teacher checkpoint required for KD: {TEACHER_CHECKPOINT}"
        )
    executed = 0
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        if max_runs is not None and executed >= max_runs:
            break
        output_dir = Path(row["output_dir"])
        if not rerun_complete and (output_dir / "best_metrics.json").exists():
            print(f"[{index}/{total}] skipping complete {row['condition']} {row['variant']}")
            continue
        if row["role"] == "student_no_kd":
            print(f"[{index}/{total}] running supervised {row['condition']} {row['variant']}")
            train_transformer.run(Path(row["config_path"]), epochs_override=epochs)
        else:
            print(f"[{index}/{total}] running KD {row['condition']} {row['variant']}")
            train_distillation.run(Path(row["config_path"]), epochs_override=epochs)
        executed += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tuned medium missing-modality KD/no-KD comparisons."
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/ablations/medium_tuned_missing_modalities"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/medium_tuned_missing_modalities"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/ablations/medium_tuned_missing_modalities_summary.csv"),
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        default=Path("outputs/ablations/medium_tuned_missing_modalities_summary.md"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--rerun-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()
    rows = build_rows(args.config_root, args.output_root)
    if args.execute:
        execute_rows(
            rows,
            epochs=args.epochs,
            max_runs=args.max_runs,
            rerun_complete=args.rerun_complete,
        )
    results = [collect_result(row) for row in rows]
    write_csv(args.summary_csv, results)
    write_markdown(args.summary_markdown, results)
    print(f"Wrote {len(rows)} rows under {args.config_root}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.summary_markdown}")
    if not args.execute:
        print("Add --execute to train generated configs.")


if __name__ == "__main__":
    main()
