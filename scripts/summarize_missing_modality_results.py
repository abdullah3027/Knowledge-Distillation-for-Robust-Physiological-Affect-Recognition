from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


RUNS = [
    ("teacher_full", "teacher", "full", True, False, Path("outputs/transformer_teacher")),
    (
        "student_full_no_kd",
        "small_student",
        "full",
        False,
        False,
        Path("outputs/transformer_student"),
    ),
    (
        "student_full_kd",
        "small_student",
        "full",
        True,
        True,
        Path("outputs/transformer_student_relational_kd"),
    ),
    (
        "student_missing_bpm",
        "small_student",
        "missing_bpm",
        False,
        False,
        Path("outputs/student_missing_bpm"),
    ),
    (
        "student_missing_ecg",
        "small_student",
        "missing_ecg",
        False,
        False,
        Path("outputs/student_missing_ecg"),
    ),
    (
        "student_missing_resp",
        "small_student",
        "missing_resp",
        False,
        False,
        Path("outputs/student_missing_resp"),
    ),
    (
        "student_kd_missing_bpm",
        "small_student",
        "missing_bpm",
        True,
        True,
        Path("outputs/student_kd_missing_bpm"),
    ),
    (
        "student_kd_missing_ecg",
        "small_student",
        "missing_ecg",
        True,
        True,
        Path("outputs/student_kd_missing_ecg"),
    ),
    (
        "student_kd_missing_resp",
        "small_student",
        "missing_resp",
        True,
        True,
        Path("outputs/student_kd_missing_resp"),
    ),
    (
        "teacher_sized_student_kd_missing_bpm",
        "teacher_sized_student",
        "missing_bpm",
        True,
        True,
        Path("outputs/teacher_sized_student_kd_missing_bpm"),
    ),
    (
        "teacher_sized_student_kd_missing_ecg",
        "teacher_sized_student",
        "missing_ecg",
        True,
        True,
        Path("outputs/teacher_sized_student_kd_missing_ecg"),
    ),
    (
        "teacher_sized_student_kd_missing_resp",
        "teacher_sized_student",
        "missing_resp",
        True,
        True,
        Path("outputs/teacher_sized_student_kd_missing_resp"),
    ),
]


def load_history_summary(path: Path) -> tuple[int | None, int | None]:
    history_path = path / "history.csv"
    if not history_path.exists():
        return None, None

    with history_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None

    best = max(rows, key=lambda row: float(row["devel_ccc"]))
    return len(rows), int(best["epoch"])


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint_path = path / "best.pt"
    if not checkpoint_path.exists():
        return {}
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = checkpoint.get("model_config", {})
    model_input = checkpoint.get("model_input", {})
    teacher_input = checkpoint.get("teacher_input", {})
    return {
        "model_input_dim": model_config.get("input_dim"),
        "d_model": model_config.get("d_model"),
        "num_heads": model_config.get("num_heads"),
        "num_layers": model_config.get("num_layers"),
        "dim_feedforward": model_config.get("dim_feedforward"),
        "available_modalities": "|".join(model_input.get("available_modalities", [])),
        "source_modalities": "|".join(model_input.get("source_modalities", [])),
        "teacher_modalities": "|".join(teacher_input.get("available_modalities", [])),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, capacity, condition, uses_full_teacher, uses_kd, path in RUNS:
        metrics_path = path / "best_metrics.json"
        if not metrics_path.exists():
            rows.append(
                {
                    "run_name": name,
                    "capacity": capacity,
                    "condition": condition,
                    "uses_full_teacher": uses_full_teacher,
                    "uses_kd": uses_kd,
                    "output_dir": str(path),
                    "status": "missing_metrics",
                }
            )
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        epochs_run, best_epoch_from_history = load_history_summary(path)
        checkpoint_metadata = load_checkpoint_metadata(path)
        rows.append(
            {
                "run_name": name,
                "capacity": capacity,
                "condition": condition,
                "uses_full_teacher": uses_full_teacher,
                "uses_kd": uses_kd,
                "output_dir": str(path),
                "status": "complete",
                "epochs_run": epochs_run,
                "best_epoch": checkpoint_metadata.get(
                    "checkpoint_epoch",
                    best_epoch_from_history,
                ),
                "ccc": metrics.get("ccc"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "pearson": metrics.get("pearson"),
                "mean_participant_ccc": metrics.get("mean_participant_ccc"),
                **checkpoint_metadata,
            }
        )
    return rows


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
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "capacity",
        "condition",
        "uses_kd",
        "available_modalities",
        "epochs_run",
        "best_epoch",
        "ccc",
        "mse",
        "mae",
        "pearson",
        "mean_participant_ccc",
    ]
    lines = [
        "# Missing-Modality Result Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col)) for col in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect best metrics for full and missing-modality runs."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/missing_modality_results_summary.csv"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("outputs/missing_modality_results_summary.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows()
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
