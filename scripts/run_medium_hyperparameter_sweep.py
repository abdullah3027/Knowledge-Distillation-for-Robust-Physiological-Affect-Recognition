from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
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


MEDIUM_TEACHER_CONFIG = Path("configs/ablations/model_size_full_modalities/medium/teacher.yaml")
MEDIUM_STUDENT_CONFIG = Path("configs/ablations/model_size_full_modalities/medium/no_kd.yaml")
MEDIUM_KD_CONFIG = Path("configs/ablations/model_size_full_modalities/medium/kd.yaml")

MEDIUM_TEACHER_OUTPUT = Path("outputs/ablations/model_size_full_modalities/medium/teacher")
MEDIUM_STUDENT_OUTPUT = Path("outputs/ablations/model_size_full_modalities/medium/no_kd")
MEDIUM_KD_OUTPUT = Path("outputs/ablations/model_size_full_modalities/medium/kd")

TARGET_MSE_WEIGHTS = [0.0, 0.05, 0.1, 0.3, 1.0]
RELATION_WEIGHTS = [0.0, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]


def safe_float_name(prefix: str, value: float) -> str:
    return f"{prefix}_{str(value).replace('.', '_')}"


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def planned_count(config: dict[str, Any]) -> int:
    return count_parameters_from_config(config["model"])


def row_for_config(
    *,
    suite: str,
    variant: str,
    role: str,
    config_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    teacher_config_path: Path | None = None,
    teacher_output_dir: Path | None = None,
    teacher_checkpoint: Path | None = None,
) -> dict[str, Any]:
    weights = {}
    if role == "student_kd":
        weights = resolve_distillation_weights(config["distillation"])
    elif "loss" in config:
        weights = {
            "supervised_weight": "",
            "relation_weight": "",
            "ccc_weight": float(config["loss"].get("ccc_weight", 1.0)),
            "mse_weight": float(config["loss"].get("mse_weight", 0.0)),
        }

    teacher_count = ""
    student_count = ""
    if role == "teacher":
        teacher_count = planned_count(config)
    else:
        student_count = planned_count(config)

    row = {
        "suite": suite,
        "variant": variant,
        "role": role,
        "config_path": config_path,
        "output_dir": output_dir,
        "teacher_config_path": teacher_config_path or "",
        "teacher_output_dir": teacher_output_dir or "",
        "teacher_checkpoint": teacher_checkpoint or "",
        "teacher_parameter_count_planned": teacher_count,
        "teacher_parameter_millions_planned": teacher_count / 1_000_000
        if teacher_count
        else "",
        "student_parameter_count_planned": student_count,
        "student_parameter_millions_planned": student_count / 1_000_000
        if student_count
        else "",
        "supervised_weight": "",
        "relation_weight": "",
        "ccc_weight": "",
        "mse_weight": "",
    }
    row.update(weights)
    return row


def make_target_loss_rows(
    *,
    teacher_base: dict[str, Any],
    student_base: dict[str, Any],
    kd_base: dict[str, Any],
    config_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mse_weight in TARGET_MSE_WEIGHTS:
        variant = safe_float_name("mse", mse_weight)
        if mse_weight == 0.1:
            rows.append(
                row_for_config(
                    suite="target_loss",
                    variant=variant,
                    role="teacher",
                    config_path=MEDIUM_TEACHER_CONFIG,
                    output_dir=MEDIUM_TEACHER_OUTPUT,
                    config=teacher_base,
                )
            )
            rows.append(
                row_for_config(
                    suite="target_loss",
                    variant=variant,
                    role="student_no_kd",
                    config_path=MEDIUM_STUDENT_CONFIG,
                    output_dir=MEDIUM_STUDENT_OUTPUT,
                    config=student_base,
                )
            )
            rows.append(
                row_for_config(
                    suite="target_loss",
                    variant=variant,
                    role="student_kd",
                    config_path=MEDIUM_KD_CONFIG,
                    output_dir=MEDIUM_KD_OUTPUT,
                    config=kd_base,
                    teacher_config_path=MEDIUM_TEACHER_CONFIG,
                    teacher_output_dir=MEDIUM_TEACHER_OUTPUT,
                    teacher_checkpoint=MEDIUM_TEACHER_OUTPUT / "best.pt",
                )
            )
            continue

        teacher_config = copy.deepcopy(teacher_base)
        teacher_output = output_root / "target_loss" / variant / "teacher"
        teacher_config_path = config_root / "target_loss" / variant / "teacher.yaml"
        teacher_config["experiment_name"] = f"medium_teacher_target_{variant}"
        teacher_config["loss"]["mse_weight"] = float(mse_weight)
        teacher_config["training"]["output_dir"] = teacher_output.as_posix()
        write_yaml(teacher_config_path, teacher_config)

        student_config = copy.deepcopy(student_base)
        student_output = output_root / "target_loss" / variant / "student_no_kd"
        student_config_path = config_root / "target_loss" / variant / "student_no_kd.yaml"
        student_config["experiment_name"] = f"medium_student_no_kd_target_{variant}"
        student_config["loss"]["mse_weight"] = float(mse_weight)
        student_config["training"]["output_dir"] = student_output.as_posix()
        write_yaml(student_config_path, student_config)

        kd_config = copy.deepcopy(kd_base)
        kd_output = output_root / "target_loss" / variant / "student_kd"
        kd_config_path = config_root / "target_loss" / variant / "student_kd.yaml"
        kd_config["experiment_name"] = f"medium_student_kd_target_{variant}"
        kd_config["teacher_checkpoint"] = (teacher_output / "best.pt").as_posix()
        kd_config["distillation"]["mse_weight"] = float(mse_weight)
        kd_config["distillation"]["ccc_weight"] = 1.0
        kd_config["distillation"]["supervised_weight"] = 1.0
        kd_config["distillation"]["relation_weight"] = 0.1
        kd_config["training"]["output_dir"] = kd_output.as_posix()
        write_yaml(kd_config_path, kd_config)

        rows.append(
            row_for_config(
                suite="target_loss",
                variant=variant,
                role="teacher",
                config_path=teacher_config_path,
                output_dir=teacher_output,
                config=teacher_config,
            )
        )
        rows.append(
            row_for_config(
                suite="target_loss",
                variant=variant,
                role="student_no_kd",
                config_path=student_config_path,
                output_dir=student_output,
                config=student_config,
            )
        )
        rows.append(
            row_for_config(
                suite="target_loss",
                variant=variant,
                role="student_kd",
                config_path=kd_config_path,
                output_dir=kd_output,
                config=kd_config,
                teacher_config_path=teacher_config_path,
                teacher_output_dir=teacher_output,
                teacher_checkpoint=teacher_output / "best.pt",
            )
        )
    return rows


def make_relation_weight_rows(
    *,
    teacher_base: dict[str, Any],
    student_base: dict[str, Any],
    kd_base: dict[str, Any],
    config_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    rows = [
        row_for_config(
            suite="relation_weight",
            variant="baseline_anchor",
            role="teacher",
            config_path=MEDIUM_TEACHER_CONFIG,
            output_dir=MEDIUM_TEACHER_OUTPUT,
            config=teacher_base,
        ),
        row_for_config(
            suite="relation_weight",
            variant="baseline_anchor",
            role="student_no_kd",
            config_path=MEDIUM_STUDENT_CONFIG,
            output_dir=MEDIUM_STUDENT_OUTPUT,
            config=student_base,
        ),
    ]
    for relation_weight in RELATION_WEIGHTS:
        variant = safe_float_name("rw", relation_weight)
        if relation_weight == 0.1:
            rows.append(
                row_for_config(
                    suite="relation_weight",
                    variant=variant,
                    role="student_kd",
                    config_path=MEDIUM_KD_CONFIG,
                    output_dir=MEDIUM_KD_OUTPUT,
                    config=kd_base,
                    teacher_config_path=MEDIUM_TEACHER_CONFIG,
                    teacher_output_dir=MEDIUM_TEACHER_OUTPUT,
                    teacher_checkpoint=MEDIUM_TEACHER_OUTPUT / "best.pt",
                )
            )
            continue

        kd_config = copy.deepcopy(kd_base)
        kd_output = output_root / "relation_weight" / variant / "student_kd"
        kd_config_path = config_root / "relation_weight" / variant / "student_kd.yaml"
        kd_config["experiment_name"] = f"medium_student_kd_relation_{variant}"
        kd_config["teacher_checkpoint"] = (MEDIUM_TEACHER_OUTPUT / "best.pt").as_posix()
        kd_config["distillation"]["supervised_weight"] = 1.0
        kd_config["distillation"]["relation_weight"] = float(relation_weight)
        kd_config["distillation"]["ccc_weight"] = 1.0
        kd_config["distillation"]["mse_weight"] = 0.1
        kd_config["training"]["output_dir"] = kd_output.as_posix()
        write_yaml(kd_config_path, kd_config)
        rows.append(
            row_for_config(
                suite="relation_weight",
                variant=variant,
                role="student_kd",
                config_path=kd_config_path,
                output_dir=kd_output,
                config=kd_config,
                teacher_config_path=MEDIUM_TEACHER_CONFIG,
                teacher_output_dir=MEDIUM_TEACHER_OUTPUT,
                teacher_checkpoint=MEDIUM_TEACHER_OUTPUT / "best.pt",
            )
        )
    return rows


def build_rows(
    *,
    suite: str,
    config_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    teacher_base = load_yaml(MEDIUM_TEACHER_CONFIG)
    student_base = load_yaml(MEDIUM_STUDENT_CONFIG)
    kd_base = load_yaml(MEDIUM_KD_CONFIG)
    rows: list[dict[str, Any]] = []
    if suite in {"target_loss", "all"}:
        rows.extend(
            make_target_loss_rows(
                teacher_base=teacher_base,
                student_base=student_base,
                kd_base=kd_base,
                config_root=config_root,
                output_root=output_root,
            )
        )
    if suite in {"relation_weight", "all"}:
        rows.extend(
            make_relation_weight_rows(
                teacher_base=teacher_base,
                student_base=student_base,
                kd_base=kd_base,
                config_root=config_root,
                output_root=output_root,
            )
        )
    return rows


def history_summary(output_dir: Path) -> dict[str, Any]:
    history_path = output_dir / "history.csv"
    if not history_path.exists():
        return {}
    history = pd.read_csv(history_path)
    if history.empty:
        return {}
    result: dict[str, Any] = {"epochs_run": len(history)}
    if "devel_ccc" in history:
        best_ccc = history.loc[history["devel_ccc"].astype(float).idxmax()]
        result.update(
            {
                "best_ccc_epoch": int(best_ccc["epoch"]),
                "best_ccc_any_epoch": float(best_ccc["devel_ccc"]),
                "mse_at_best_ccc_epoch": float(best_ccc["devel_mse"]),
                "mae_at_best_ccc_epoch": float(best_ccc["devel_mae"]),
            }
        )
    if "devel_mse" in history:
        best_mse = history.loc[history["devel_mse"].astype(float).idxmin()]
        result.update(
            {
                "best_mse_epoch": int(best_mse["epoch"]),
                "best_mse_any_epoch": float(best_mse["devel_mse"]),
                "ccc_at_best_mse_epoch": float(best_mse["devel_ccc"]),
                "mae_at_best_mse_epoch": float(best_mse["devel_mae"]),
            }
        )
    if "devel_mae" in history:
        best_mae = history.loc[history["devel_mae"].astype(float).idxmin()]
        result.update(
            {
                "best_mae_epoch": int(best_mae["epoch"]),
                "best_mae_any_epoch": float(best_mae["devel_mae"]),
                "ccc_at_best_mae_epoch": float(best_mae["devel_ccc"]),
                "mse_at_best_mae_epoch": float(best_mae["devel_mse"]),
            }
        )
    if "train_weighted_relation_distillation_loss" in history and "train_loss" in history:
        best_ccc = history.loc[history["devel_ccc"].astype(float).idxmax()]
        train_loss = float(best_ccc["train_loss"])
        weighted_relation = float(best_ccc["train_weighted_relation_distillation_loss"])
        result.update(
            {
                "train_loss_at_best_ccc_epoch": train_loss,
                "train_supervised_loss_at_best_ccc_epoch": float(
                    best_ccc["train_supervised_loss"]
                ),
                "train_relation_loss_at_best_ccc_epoch": float(
                    best_ccc["train_relation_distillation_loss"]
                ),
                "train_weighted_relation_loss_at_best_ccc_epoch": weighted_relation,
                "weighted_relation_pct_at_best_ccc_epoch": 100
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
        "suite",
        "variant",
        "role",
        "status",
        "supervised_weight",
        "relation_weight",
        "ccc_weight",
        "mse_weight",
        "parameter_millions",
        "ccc",
        "mse",
        "mae",
        "pearson",
        "best_mse_any_epoch",
        "best_mae_any_epoch",
        "weighted_relation_pct_at_best_ccc_epoch",
    ]
    lines = [
        "# Medium Hyperparameter Sweep Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def train_teacher_if_missing(row: dict[str, Any], epochs: int | None) -> None:
    teacher_output_dir = Path(row["teacher_output_dir"])
    if (teacher_output_dir / "best_metrics.json").exists():
        return
    print(f"training required teacher {row['suite']} {row['variant']}")
    train_transformer.run(Path(row["teacher_config_path"]), epochs_override=epochs)


def execute_rows(
    rows: list[dict[str, Any]],
    *,
    epochs: int | None,
    max_runs: int | None,
    rerun_complete: bool,
) -> None:
    executed = 0
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        if max_runs is not None and executed >= max_runs:
            break
        output_dir = Path(row["output_dir"])
        if not rerun_complete and (output_dir / "best_metrics.json").exists():
            print(f"[{index}/{total}] skipping complete {row['suite']} {row['variant']} {row['role']}")
            continue
        if row["role"] in {"teacher", "student_no_kd"}:
            print(f"[{index}/{total}] running supervised {row['suite']} {row['variant']} {row['role']}")
            train_transformer.run(Path(row["config_path"]), epochs_override=epochs)
        else:
            train_teacher_if_missing(row, epochs)
            print(f"[{index}/{total}] running KD {row['suite']} {row['variant']}")
            train_distillation.run(Path(row["config_path"]), epochs_override=epochs)
        executed += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run medium-size target-loss and relation-weight hyperparameter sweeps."
    )
    parser.add_argument(
        "--suite",
        choices=["target_loss", "relation_weight", "all"],
        default="all",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/ablations/medium_hyperparameter_sweep"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/medium_hyperparameter_sweep"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/ablations/medium_hyperparameter_sweep_summary.csv"),
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        default=Path("outputs/ablations/medium_hyperparameter_sweep_summary.md"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-runs", type=int, help="Execute only the next N incomplete rows.")
    parser.add_argument("--rerun-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()
    rows = build_rows(
        suite=args.suite,
        config_root=args.config_root,
        output_root=args.output_root,
    )
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
