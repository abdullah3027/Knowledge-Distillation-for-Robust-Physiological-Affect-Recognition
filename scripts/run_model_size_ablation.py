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


DEFAULT_TEACHER_CONFIG = Path("configs/training/transformer_teacher.yaml")
DEFAULT_STUDENT_CONFIG = Path("configs/training/transformer_student.yaml")
DEFAULT_KD_STUDENT_CONFIG = Path("configs/training/transformer_student_relational_kd.yaml")

MODEL_SIZE_LEVELS = [
    (
        "base",
        {"d_model": 128, "num_heads": 4, "num_layers": 4, "dim_feedforward": 256},
        {"d_model": 64, "num_heads": 2, "num_layers": 2, "dim_feedforward": 128},
    ),
    (
        "medium",
        {"d_model": 192, "num_heads": 4, "num_layers": 4, "dim_feedforward": 384},
        {"d_model": 80, "num_heads": 2, "num_layers": 3, "dim_feedforward": 240},
    ),
    (
        "xlarge",
        {"d_model": 288, "num_heads": 4, "num_layers": 4, "dim_feedforward": 576},
        {"d_model": 120, "num_heads": 2, "num_layers": 4, "dim_feedforward": 240},
    ),
]


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def planned_count(config: dict[str, Any]) -> int:
    return count_parameters_from_config(config["model"])


def apply_capacity(
    config: dict[str, Any],
    capacity: dict[str, int],
    train_batch_size: int,
    evaluation_batch_size: int,
) -> None:
    config["model"].update(capacity)
    config["data"]["train_batch_size"] = train_batch_size
    config["data"]["evaluation_batch_size"] = evaluation_batch_size


def evenly_spaced_layer_pairs(student_layers: int, teacher_layers: int) -> list[list[int]]:
    if student_layers <= 0:
        raise ValueError("student_layers must be positive")
    if teacher_layers <= 0:
        raise ValueError("teacher_layers must be positive")
    if student_layers == 1:
        return [[0, teacher_layers - 1]]
    return [
        [student_layer, round(student_layer * (teacher_layers - 1) / (student_layers - 1))]
        for student_layer in range(student_layers)
    ]


def common_row(
    *,
    role: str,
    comparison: str,
    level: str,
    config_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    teacher_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    teacher_count = (
        planned_count(config)
        if role == "teacher"
        else int(teacher_row["teacher_parameter_count_planned"])
        if teacher_row is not None
        else None
    )
    student_count = planned_count(config) if role == "student" else None
    row = {
        "level": level,
        "role": role,
        "comparison": comparison,
        "condition": "full_modalities",
        "config_path": config_path,
        "output_dir": output_dir,
        "teacher_config_path": teacher_row["config_path"] if teacher_row else "",
        "teacher_output_dir": teacher_row["output_dir"] if teacher_row else "",
        "teacher_checkpoint": teacher_row["teacher_checkpoint"] if teacher_row else "",
        "teacher_parameter_count_planned": teacher_count,
        "teacher_parameter_millions_planned": teacher_count / 1_000_000 if teacher_count else None,
        "student_parameter_count_planned": student_count,
        "student_parameter_millions_planned": student_count / 1_000_000 if student_count else None,
        "teacher_student_parameter_ratio_planned": (
            teacher_count / student_count
            if teacher_count is not None and student_count is not None
            else None
        ),
        "supervised_weight": "",
        "relation_weight": "",
        "ccc_weight": "",
        "mse_weight": "",
    }
    if "distillation" in config:
        row.update(resolve_distillation_weights(config["distillation"]))
    if role == "teacher":
        row["teacher_checkpoint"] = (output_dir / "best.pt").as_posix()
    return row


def make_teacher_row(
    base_config: dict[str, Any],
    level: str,
    capacity: dict[str, int],
    config_root: Path,
    output_root: Path,
    train_batch_size: int,
    evaluation_batch_size: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    output_dir = output_root / level / "teacher"
    config_path = config_root / level / "teacher.yaml"
    config["experiment_name"] = f"teacher_model_size_{level}"
    config["training"]["output_dir"] = output_dir.as_posix()
    apply_capacity(config, capacity, train_batch_size, evaluation_batch_size)
    write_yaml(config_path, config)
    row = common_row(
        role="teacher",
        comparison="teacher",
        level=level,
        config_path=config_path,
        output_dir=output_dir,
        config=config,
    )
    row["teacher_num_layers"] = int(config["model"]["num_layers"])
    return row


def make_student_row(
    base_config: dict[str, Any],
    *,
    level: str,
    comparison: str,
    capacity: dict[str, int],
    teacher_row: dict[str, Any],
    config_root: Path,
    output_root: Path,
    train_batch_size: int,
    evaluation_batch_size: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    output_dir = output_root / level / comparison
    config_path = config_root / level / f"{comparison}.yaml"
    config["experiment_name"] = f"student_{comparison}_model_size_{level}"
    config["training"]["output_dir"] = output_dir.as_posix()
    apply_capacity(config, capacity, train_batch_size, evaluation_batch_size)

    if comparison == "kd":
        config["teacher_checkpoint"] = str(teacher_row["teacher_checkpoint"])
        config["distillation"]["layer_pairs"] = evenly_spaced_layer_pairs(
            int(config["model"]["num_layers"]),
            int(teacher_row["teacher_num_layers"]),
        )

    write_yaml(config_path, config)
    return common_row(
        role="student",
        comparison=comparison,
        level=level,
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        teacher_row=teacher_row,
    )


def build_runs(
    teacher_config_path: Path,
    student_config_path: Path,
    kd_student_config_path: Path,
    config_root: Path,
    output_root: Path,
    train_batch_size: int,
    evaluation_batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    teacher_base = load_yaml(teacher_config_path)
    student_base = load_yaml(student_config_path)
    kd_student_base = load_yaml(kd_student_config_path)

    for level, teacher_capacity, student_capacity in MODEL_SIZE_LEVELS:
        teacher_row = make_teacher_row(
            teacher_base,
            level,
            teacher_capacity,
            config_root,
            output_root,
            train_batch_size,
            evaluation_batch_size,
        )
        rows.append(teacher_row)
        rows.append(
            make_student_row(
                student_base,
                level=level,
                comparison="no_kd",
                capacity=student_capacity,
                teacher_row=teacher_row,
                config_root=config_root,
                output_root=output_root,
                train_batch_size=train_batch_size,
                evaluation_batch_size=evaluation_batch_size,
            )
        )
        rows.append(
            make_student_row(
                kd_student_base,
                level=level,
                comparison="kd",
                capacity=student_capacity,
                teacher_row=teacher_row,
                config_root=config_root,
                output_root=output_root,
                train_batch_size=train_batch_size,
                evaluation_batch_size=evaluation_batch_size,
            )
        )

    return rows


def history_summary(output_dir: Path) -> tuple[int | None, int | None]:
    history_path = output_dir / "history.csv"
    if not history_path.exists():
        return None, None
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None
    best = max(rows, key=lambda row: float(row["devel_ccc"]))
    return len(rows), int(best["epoch"])


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
    epochs_run, best_epoch = history_summary(output_dir)
    parameter_count = checkpoint_parameter_count(output_dir)
    result.update(
        {
            "status": "complete",
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "parameter_count": parameter_count,
            "parameter_millions": parameter_count / 1_000_000 if parameter_count else None,
            "ccc": metrics.get("ccc"),
            "mse": metrics.get("mse"),
            "mae": metrics.get("mae"),
            "pearson": metrics.get("pearson"),
            "mean_participant_ccc": metrics.get("mean_participant_ccc"),
        }
    )
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
        "level",
        "role",
        "comparison",
        "condition",
        "status",
        "teacher_parameter_millions_planned",
        "student_parameter_millions_planned",
        "teacher_student_parameter_ratio_planned",
        "supervised_weight",
        "relation_weight",
        "epochs_run",
        "best_epoch",
        "ccc",
        "mse",
        "mae",
        "pearson",
    ]
    lines = [
        "# Model Size Ablation Summary",
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
    print(f"training teacher for {row['level']}")
    train_transformer.run(Path(row["teacher_config_path"]), epochs_override=epochs)


def execute_runs(
    rows: list[dict[str, Any]],
    *,
    epochs: int | None,
    max_runs: int | None,
    rerun_complete: bool,
) -> None:
    executed = 0
    total_rows = len(rows)
    for index, row in enumerate(rows, start=1):
        if max_runs is not None and executed >= max_runs:
            break
        output_dir = Path(row["output_dir"])
        if not rerun_complete and (output_dir / "best_metrics.json").exists():
            print(f"[{index}/{total_rows}] skipping complete {row['level']} {row['comparison']}")
            continue
        if row["role"] == "teacher" or row["comparison"] == "no_kd":
            print(f"[{index}/{total_rows}] running supervised {row['level']} {row['comparison']}")
            train_transformer.run(Path(row["config_path"]), epochs_override=epochs)
        else:
            train_teacher_if_missing(row, epochs)
            print(f"[{index}/{total_rows}] running KD {row['level']}")
            train_distillation.run(Path(row["config_path"]), epochs_override=epochs)
        executed += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, run, and summarize the full-modality model-size ablation: "
            "teacher, student without KD, and student with relational KD."
        )
    )
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--student-config", type=Path, default=DEFAULT_STUDENT_CONFIG)
    parser.add_argument("--kd-student-config", type=Path, default=DEFAULT_KD_STUDENT_CONFIG)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/ablations/model_size_full_modalities"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/model_size_full_modalities"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/ablations/model_size_full_modalities_summary.csv"),
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        default=Path("outputs/ablations/model_size_full_modalities_summary.md"),
    )
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Train generated configs.")
    parser.add_argument("--epochs", type=int, help="Override epochs for executed runs.")
    parser.add_argument("--max-runs", type=int, help="Execute only the next N incomplete rows.")
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Rerun outputs that already have best_metrics.json.",
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()
    rows = build_runs(
        args.teacher_config,
        args.student_config,
        args.kd_student_config,
        args.config_root,
        args.output_root,
        args.train_batch_size,
        args.evaluation_batch_size,
    )
    if args.execute:
        execute_runs(
            rows,
            epochs=args.epochs,
            max_runs=args.max_runs,
            rerun_complete=args.rerun_complete,
        )
    results = [collect_result(row) for row in rows]
    write_csv(args.summary_csv, results)
    write_markdown(args.summary_markdown, results)
    print(f"Wrote {len(rows)} model-size rows under {args.config_root}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.summary_markdown}")
    if not args.execute:
        print("Add --execute to train the generated configs.")


if __name__ == "__main__":
    main()
