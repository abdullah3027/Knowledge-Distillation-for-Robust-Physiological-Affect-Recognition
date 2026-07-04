from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


RUN_LABELS = {
    "teacher_full": "Teacher full",
    "student_full_no_kd": "Small full, no KD",
    "student_full_kd": "Small full, KD",
    "student_missing_bpm": "Small no KD, missing BPM",
    "student_missing_ecg": "Small no KD, missing ECG",
    "student_missing_resp": "Small no KD, missing RESP",
    "student_kd_missing_bpm": "Small KD, missing BPM",
    "student_kd_missing_ecg": "Small KD, missing ECG",
    "student_kd_missing_resp": "Small KD, missing RESP",
    "teacher_sized_student_kd_missing_bpm": "Teacher-sized KD, missing BPM",
    "teacher_sized_student_kd_missing_ecg": "Teacher-sized KD, missing ECG",
    "teacher_sized_student_kd_missing_resp": "Teacher-sized KD, missing RESP",
}

FAMILY_LABELS = {
    "teacher_full": "Teacher",
    "student_full_no_kd": "Small no KD",
    "student_full_kd": "Small KD",
    "student_missing_bpm": "Small no KD",
    "student_missing_ecg": "Small no KD",
    "student_missing_resp": "Small no KD",
    "student_kd_missing_bpm": "Small KD",
    "student_kd_missing_ecg": "Small KD",
    "student_kd_missing_resp": "Small KD",
    "teacher_sized_student_kd_missing_bpm": "Teacher-sized KD",
    "teacher_sized_student_kd_missing_ecg": "Teacher-sized KD",
    "teacher_sized_student_kd_missing_resp": "Teacher-sized KD",
}

FAMILY_COLORS = {
    "Teacher": "#34495e",
    "Small no KD": "#4c78a8",
    "Small KD": "#f58518",
    "Teacher-sized KD": "#54a24b",
}

CONDITION_ORDER = ["missing_bpm", "missing_ecg", "missing_resp"]
CONDITION_LABELS = {
    "full": "Full input",
    "missing_bpm": "Missing BPM",
    "missing_ecg": "Missing ECG",
    "missing_resp": "Missing RESP",
}
AVAILABLE_LABELS = {
    "full": "BPM + ECG + RESP",
    "missing_bpm": "ECG + RESP",
    "missing_ecg": "BPM + RESP",
    "missing_resp": "BPM + ECG",
}


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def modality_text(value: Any, condition: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.replace("|", " + ")
    return AVAILABLE_LABELS.get(condition, "")


def count_checkpoint_parameters(output_dir: Path) -> int | None:
    checkpoint_path = output_dir / "best.pt"
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    return int(sum(tensor.numel() for tensor in checkpoint["model_state"].values()))


def load_results(summary_path: Path) -> pd.DataFrame:
    data = pd.read_csv(summary_path)
    for column in ["ccc", "mse", "mae", "pearson", "mean_participant_ccc"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["display_name"] = data["run_name"].map(RUN_LABELS).fillna(data["run_name"])
    data["family"] = data["run_name"].map(FAMILY_LABELS).fillna(data["capacity"])
    data["condition_label"] = data["condition"].map(CONDITION_LABELS)
    data["available_modalities_text"] = [
        modality_text(value, condition)
        for value, condition in zip(data["available_modalities"], data["condition"])
    ]
    data["parameter_count"] = [
        count_checkpoint_parameters(Path(path)) if status == "complete" else None
        for path, status in zip(data["output_dir"], data["status"])
    ]
    data["parameter_millions"] = data["parameter_count"].astype(float) / 1_000_000
    return data


def load_histories(results: pd.DataFrame) -> pd.DataFrame:
    histories: list[pd.DataFrame] = []
    for row in results.itertuples(index=False):
        history_path = Path(row.output_dir) / "history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        history["run_name"] = row.run_name
        history["display_name"] = row.display_name
        history["family"] = row.family
        history["condition"] = row.condition
        history["condition_label"] = row.condition_label
        histories.append(history)
    return pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_pipeline_split_overview(manifest: dict[str, Any], output_path: Path) -> None:
    splits = pd.DataFrame(
        [
            {"split": split, **summary}
            for split, summary in manifest["splits"].items()
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(splits))

    axes[0].bar(x - 0.18, splits["participants"], width=0.36, label="Participants")
    axes[0].bar(x + 0.18, splits["samples"], width=0.36, label="Samples/windows")
    axes[0].set_xticks(x, splits["split"].str.title())
    axes[0].set_title("Processed split structure")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)

    axes[1].bar(x, splits["valid_target_steps"], label="Valid target steps")
    axes[1].bar(
        x,
        splits["padding_steps"],
        bottom=splits["valid_target_steps"],
        label="Padding steps",
    )
    axes[1].set_xticks(x, splits["split"].str.title())
    axes[1].set_title("Targets and padding by split")
    axes[1].set_ylabel("Timesteps")
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Pipeline EDA: train is windowed; devel/test are full participant sequences",
        fontsize=13,
        fontweight="bold",
    )
    savefig(output_path)


def plot_all_model_ccc(results: pd.DataFrame, output_path: Path) -> None:
    data = results.sort_values("ccc", ascending=True)
    colors = [FAMILY_COLORS.get(family, "#888888") for family in data["family"]]
    plt.figure(figsize=(11, 7.2))
    bars = plt.barh(data["display_name"], data["ccc"], color=colors)
    plt.axvline(0, color="#333333", linewidth=0.8)
    for bar, value in zip(bars, data["ccc"]):
        plt.text(
            value + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )
    plt.xlabel("Devel CCC, higher is better")
    plt.title("Best devel CCC across all completed models")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in FAMILY_COLORS.values()
    ]
    plt.legend(
        legend_handles,
        list(FAMILY_COLORS.keys()),
        frameon=False,
        loc="lower right",
    )
    savefig(output_path)


def plot_missing_modality_groups(results: pd.DataFrame, output_path: Path) -> None:
    data = results[results["condition"].isin(CONDITION_ORDER)].copy()
    family_order = ["Small no KD", "Small KD", "Teacher-sized KD"]
    width = 0.24
    x = np.arange(len(CONDITION_ORDER))
    plt.figure(figsize=(10.5, 5.6))
    for index, family in enumerate(family_order):
        values = []
        for condition in CONDITION_ORDER:
            match = data[(data["condition"] == condition) & (data["family"] == family)]
            values.append(float(match["ccc"].iloc[0]) if not match.empty else np.nan)
        offset = (index - 1) * width
        bars = plt.bar(
            x + offset,
            values,
            width=width,
            label=family,
            color=FAMILY_COLORS[family],
        )
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.xticks(x, [AVAILABLE_LABELS[condition] for condition in CONDITION_ORDER])
    plt.ylabel("Devel CCC")
    plt.xlabel("Available student modalities")
    plt.title("Missing-modality performance by model family")
    plt.legend(frameon=False)
    savefig(output_path)


def compute_small_kd_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("full", "student_full_no_kd", "student_full_kd"),
        ("missing_bpm", "student_missing_bpm", "student_kd_missing_bpm"),
        ("missing_ecg", "student_missing_ecg", "student_kd_missing_ecg"),
        ("missing_resp", "student_missing_resp", "student_kd_missing_resp"),
    ]
    for condition, no_kd_name, kd_name in pairs:
        no_kd = results.loc[results["run_name"] == no_kd_name].iloc[0]
        kd = results.loc[results["run_name"] == kd_name].iloc[0]
        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "available_modalities": AVAILABLE_LABELS[condition],
                "no_kd_ccc": no_kd["ccc"],
                "kd_ccc": kd["ccc"],
                "ccc_delta": kd["ccc"] - no_kd["ccc"],
                "no_kd_mse": no_kd["mse"],
                "kd_mse": kd["mse"],
                "mse_delta": kd["mse"] - no_kd["mse"],
                "no_kd_mae": no_kd["mae"],
                "kd_mae": kd["mae"],
                "mae_delta": kd["mae"] - no_kd["mae"],
            }
        )
    return pd.DataFrame(rows)


def plot_kd_deltas(delta: pd.DataFrame, output_path: Path) -> None:
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in delta["ccc_delta"]]
    plt.figure(figsize=(9.5, 5.2))
    bars = plt.bar(delta["available_modalities"], delta["ccc_delta"], color=colors)
    plt.axhline(0, color="#333333", linewidth=0.9)
    for bar, value in zip(bars, delta["ccc_delta"]):
        va = "bottom" if value >= 0 else "top"
        y = value + (0.002 if value >= 0 else -0.002)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=9,
        )
    plt.ylabel("CCC change from adding KD")
    plt.xlabel("Small-student input")
    plt.title("Does KD improve the small student?")
    savefig(output_path)


def compute_capacity_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        small = results[
            (results["condition"] == condition) & (results["family"] == "Small KD")
        ].iloc[0]
        large = results[
            (results["condition"] == condition)
            & (results["family"] == "Teacher-sized KD")
        ].iloc[0]
        rows.append(
            {
                "condition": condition,
                "available_modalities": AVAILABLE_LABELS[condition],
                "small_kd_ccc": small["ccc"],
                "teacher_sized_kd_ccc": large["ccc"],
                "ccc_delta": large["ccc"] - small["ccc"],
                "small_kd_mse": small["mse"],
                "teacher_sized_kd_mse": large["mse"],
                "mse_delta": large["mse"] - small["mse"],
            }
        )
    return pd.DataFrame(rows)


def plot_capacity_deltas(delta: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9.5, 5.2))
    bars = plt.bar(delta["available_modalities"], delta["ccc_delta"], color="#54a24b")
    for bar, value in zip(bars, delta["ccc_delta"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.axhline(0, color="#333333", linewidth=0.9)
    plt.ylabel("CCC gain")
    plt.xlabel("Available student modalities")
    plt.title("Teacher-sized capacity gain over small KD students")
    savefig(output_path)


def plot_training_curves(histories: pd.DataFrame, output_path: Path) -> None:
    if histories.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    family_order = ["Small no KD", "Small KD", "Teacher-sized KD"]
    for axis, condition in zip(axes, CONDITION_ORDER):
        subset = histories[histories["condition"] == condition]
        for family in family_order:
            run = subset[subset["family"] == family]
            if run.empty:
                continue
            axis.plot(
                run["epoch"],
                run["devel_ccc"],
                label=family,
                color=FAMILY_COLORS[family],
                linewidth=1.8,
            )
            best_index = run["devel_ccc"].idxmax()
            best = run.loc[best_index]
            axis.scatter(
                [best["epoch"]],
                [best["devel_ccc"]],
                color=FAMILY_COLORS[family],
                s=34,
                zorder=3,
            )
        axis.set_title(f"{CONDITION_LABELS[condition]}\n{AVAILABLE_LABELS[condition]}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Devel CCC")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Training dynamics for missing-modality runs", fontweight="bold")
    savefig(output_path)


def plot_metric_tradeoff(results: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9.6, 6.2))
    for family, group in results.groupby("family"):
        sizes = 180 + group["parameter_millions"].fillna(0.1) * 360
        plt.scatter(
            group["mse"],
            group["ccc"],
            s=sizes,
            color=FAMILY_COLORS.get(family, "#888888"),
            alpha=0.78,
            label=family,
            edgecolor="white",
            linewidth=0.9,
        )
    for _, row in results.iterrows():
        if row["run_name"] in {
            "teacher_full",
            "student_full_no_kd",
            "student_full_kd",
            "student_kd_missing_resp",
            "teacher_sized_student_kd_missing_resp",
        }:
            plt.text(
                row["mse"] + 0.008,
                row["ccc"] + 0.004,
                row["display_name"],
                fontsize=8,
            )
    plt.xlabel("Devel MSE, lower is better")
    plt.ylabel("Devel CCC, higher is better")
    plt.title("Metric tradeoff: agreement versus pointwise error")
    plt.legend(frameon=False)
    plt.grid(alpha=0.22)
    savefig(output_path)


def write_tables(
    results: pd.DataFrame,
    small_kd_delta: pd.DataFrame,
    capacity_delta: pd.DataFrame,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(tables_dir / "model_results_enriched.csv", index=False)
    small_kd_delta.to_csv(tables_dir / "small_student_kd_deltas.csv", index=False)
    capacity_delta.to_csv(tables_dir / "teacher_sized_capacity_deltas.csv", index=False)


def markdown_table(data: pd.DataFrame, columns: list[str], float_digits: int = 4) -> str:
    rows = data[columns].copy()
    for column in rows.columns:
        if pd.api.types.is_float_dtype(rows[column]):
            rows[column] = rows[column].map(lambda value: f"{value:.{float_digits}f}")

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in rows.iterrows():
        lines.append("| " + " | ".join(cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    manifest: dict[str, Any],
    norm_stats: pd.DataFrame,
    results: pd.DataFrame,
    small_kd_delta: pd.DataFrame,
    capacity_delta: pd.DataFrame,
) -> None:
    split_rows = []
    for split, summary in manifest["splits"].items():
        split_rows.append(
            {
                "split": split,
                "participants": summary["participants"],
                "samples": summary["samples"],
                "x_shape": str(summary["x_shape"]),
                "padding_steps": summary["padding_steps"],
                "valid_target_steps": summary["valid_target_steps"],
            }
        )
    split_table = pd.DataFrame(split_rows)

    ranked = results.sort_values("ccc", ascending=False).copy()
    ranked["modalities"] = ranked["available_modalities_text"]
    ranked["parameters"] = ranked["parameter_count"].map(
        lambda value: f"{int(value):,}" if pd.notna(value) else ""
    )
    ranked["kd"] = ranked["uses_kd"].map(lambda value: "yes" if value else "no")

    best = ranked.iloc[0]
    full_no_kd = results.loc[results["run_name"] == "student_full_no_kd"].iloc[0]
    full_kd = results.loc[results["run_name"] == "student_full_kd"].iloc[0]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Pipeline and Missing-Modality EDA Report",
                "",
                "## Scope",
                "",
                "This report summarizes the current MuSe-Physio preprocessing pipeline, "
                "the completed teacher/student/KD runs, and the missing-modality ablations. "
                "All conclusions here use the devel split and the currently saved seed/run "
                "artifacts. They should be treated as evidence, not final multi-seed claims.",
                "",
                "## Pipeline Summary",
                "",
                "The model target is frame-level `anno12_EDA`. Inputs are normalized "
                "`BPM`, `ECG`, and `RESP`. The teacher sees all three modalities. Missing-"
                "modality students receive true dropped-column inputs rather than mean-"
                "imputed zeros.",
                "",
                "```mermaid",
                'flowchart LR',
                '  A["Raw MuSe-Physio CSVs"] --> B["Timestamp alignment and validation"]',
                '  B --> C["Train-only feature normalization"]',
                '  C --> D["Train: sliding windows"]',
                '  C --> E["Devel/Test: full participant sequences"]',
                '  D --> F["PyTorch tensors and masks"]',
                '  E --> F',
                '  F --> G["Full-input teacher"]',
                '  F --> H["Selected-modality students"]',
                '  G --> I["Relational KD"]',
                '  H --> I',
                '  I --> J["Devel CCC/MSE/MAE/Pearson"]',
                "```",
                "",
                "![Pipeline split overview](figures/01_pipeline_split_overview.png)",
                "",
                "### Processed Split Contract",
                "",
                markdown_table(split_table, list(split_table.columns), 0),
                "",
                "### Normalization Statistics",
                "",
                markdown_table(
                    norm_stats[["modality", "mean", "std", "fit_split", "unique_frames_used"]],
                    ["modality", "mean", "std", "fit_split", "unique_frames_used"],
                    6,
                ),
                "",
                "## Primary Result Ranking",
                "",
                "Primary metric: devel CCC. Higher is better.",
                "",
                "![All model CCC](figures/02_devel_ccc_all_models.png)",
                "",
                markdown_table(
                    ranked[
                        [
                            "display_name",
                            "family",
                            "condition_label",
                            "modalities",
                            "kd",
                            "parameters",
                            "best_epoch",
                            "ccc",
                            "mse",
                            "mae",
                            "pearson",
                            "mean_participant_ccc",
                        ]
                    ],
                    [
                        "display_name",
                        "family",
                        "condition_label",
                        "modalities",
                        "kd",
                        "parameters",
                        "best_epoch",
                        "ccc",
                        "mse",
                        "mae",
                        "pearson",
                        "mean_participant_ccc",
                    ],
                    4,
                ),
                "",
                "## Objective Check",
                "",
                f"The strongest saved model is **{best['display_name']}** with "
                f"CCC **{best['ccc']:.4f}**. It uses **{best['modalities']}**.",
                "",
                "For the small full-input student, KD did **not** improve the primary "
                f"CCC metric: no-KD CCC was **{full_no_kd['ccc']:.4f}**, while full-input "
                f"KD CCC was **{full_kd['ccc']:.4f}**. KD did, however, reduce MSE and MAE "
                "for that full-input student.",
                "",
                "For missing-modality small students, KD is mixed: it helps when ECG or "
                "RESP is removed, but hurts slightly when BPM is removed. Teacher-sized "
                "students with KD improve strongly in every missing-modality case.",
                "",
                "Current verdict: **the project objective is partially supported**. The "
                "pipeline demonstrates missing-modality robustness experiments and shows "
                "that KD plus sufficient student capacity can perform well under missing "
                "modalities. But small-student KD is not consistently better than no-KD "
                "on CCC, so the claim that KD generally improves the compressed student "
                "still needs multi-seed confirmation and tuning.",
                "",
                "## KD Versus No KD",
                "",
                "Positive CCC delta means KD improved the small student. Negative MSE/MAE "
                "delta means KD reduced error.",
                "",
                "![KD deltas](figures/04_kd_delta_small_student.png)",
                "",
                markdown_table(
                    small_kd_delta[
                        [
                            "available_modalities",
                            "no_kd_ccc",
                            "kd_ccc",
                            "ccc_delta",
                            "mse_delta",
                            "mae_delta",
                        ]
                    ],
                    [
                        "available_modalities",
                        "no_kd_ccc",
                        "kd_ccc",
                        "ccc_delta",
                        "mse_delta",
                        "mae_delta",
                    ],
                    4,
                ),
                "",
                "## Missing-Modality Interpretation",
                "",
                "![Missing modality groups](figures/03_missing_modality_ccc_by_group.png)",
                "",
                "The two-modality pair **BPM + ECG** is consistently strongest. This means "
                "removing RESP hurts least in these runs. In contrast, **ECG + RESP** "
                "is weakest, suggesting BPM carries important information for EDA "
                "regression in this setup. **BPM + RESP** is in the middle.",
                "",
                "The fact that BPM + ECG sometimes beats the full three-modality models is "
                "important but should be interpreted carefully. It may indicate RESP noise, "
                "overfitting to RESP, optimization variance, or a true signal-quality issue. "
                "It should be checked with repeated seeds before making a final research "
                "claim.",
                "",
                "## Capacity Under Missing Modalities",
                "",
                "![Capacity deltas](figures/05_capacity_gain_under_missing_modalities.png)",
                "",
                markdown_table(
                    capacity_delta[
                        [
                            "available_modalities",
                            "small_kd_ccc",
                            "teacher_sized_kd_ccc",
                            "ccc_delta",
                            "mse_delta",
                        ]
                    ],
                    [
                        "available_modalities",
                        "small_kd_ccc",
                        "teacher_sized_kd_ccc",
                        "ccc_delta",
                        "mse_delta",
                    ],
                    4,
                ),
                "",
                "Capacity matters. Teacher-sized KD students gained between roughly "
                "0.039 and 0.117 CCC over the small KD students under missing-modality "
                "conditions.",
                "",
                "## Training Dynamics",
                "",
                "![Training curves](figures/06_training_curves_devel_ccc.png)",
                "",
                "The missing-modality histories are noisy, especially for teacher-sized "
                "students. This reinforces the need for repeated seeds and careful checkpoint "
                "selection using devel CCC only.",
                "",
                "## Metric Tradeoff",
                "",
                "![Metric tradeoff](figures/07_metric_tradeoff_mse_ccc.png)",
                "",
                "CCC and MSE do not always move together. For example, full-input KD reduced "
                "MSE/MAE versus the full-input no-KD student, but had lower CCC. Since CCC "
                "is the primary MuSe-style agreement metric, model selection should continue "
                "to prioritize devel CCC while reporting MSE/MAE as supporting evidence.",
                "",
                "## Recommended Next Steps",
                "",
                "1. Repeat the same experiment grid for at least seeds 42, 123, and 2026.",
                "2. Run relation-weight ablations for missing ECG and missing RESP, where KD "
                "already gives positive small-student CCC deltas.",
                "3. Investigate RESP by comparing full input versus BPM + ECG over multiple "
                "seeds.",
                "4. Export devel predictions for the top models and inspect participant-level "
                "failures.",
                "5. Do not use the test split for model selection because released test labels "
                "are unavailable.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EDA figures and report for MuSe-Physio pipeline results."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/missing_modality_results_summary.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Processed_dataset/muse_physio_baseline/manifest.json"),
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=Path(
            "Processed_dataset/muse_physio_baseline/feature_normalization_stats.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eda_pipeline_report"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    manifest = read_manifest(args.manifest)
    norm_stats = pd.read_csv(args.norm_stats)
    results = load_results(args.summary)
    histories = load_histories(results)
    small_kd_delta = compute_small_kd_deltas(results)
    capacity_delta = compute_capacity_deltas(results)

    plot_pipeline_split_overview(
        manifest,
        figures_dir / "01_pipeline_split_overview.png",
    )
    plot_all_model_ccc(results, figures_dir / "02_devel_ccc_all_models.png")
    plot_missing_modality_groups(
        results,
        figures_dir / "03_missing_modality_ccc_by_group.png",
    )
    plot_kd_deltas(small_kd_delta, figures_dir / "04_kd_delta_small_student.png")
    plot_capacity_deltas(
        capacity_delta,
        figures_dir / "05_capacity_gain_under_missing_modalities.png",
    )
    plot_training_curves(histories, figures_dir / "06_training_curves_devel_ccc.png")
    plot_metric_tradeoff(results, figures_dir / "07_metric_tradeoff_mse_ccc.png")

    write_tables(results, small_kd_delta, capacity_delta, tables_dir)
    write_report(
        output_dir / "pipeline_and_results_eda.md",
        manifest,
        norm_stats,
        results,
        small_kd_delta,
        capacity_delta,
    )

    print(f"Wrote report to {output_dir / 'pipeline_and_results_eda.md'}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote tables to {tables_dir}")


if __name__ == "__main__":
    main()
