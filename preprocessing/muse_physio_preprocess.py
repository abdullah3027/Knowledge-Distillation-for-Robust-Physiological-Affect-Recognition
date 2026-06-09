from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def resolve_existing_path(raw_path: str, config_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        config_path.parent / path,
        config_path.parent.parent / path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def resolve_output_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def validate_config(config: dict[str, Any]) -> None:
    for section in ["dataset", "modalities", "target", "windowing", "normalization", "output"]:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    modalities = config["modalities"]
    if not isinstance(modalities, list) or not modalities:
        raise ValueError("modalities must be a non-empty list")

    names = [modality["name"] for modality in modalities]
    if len(names) != len(set(names)):
        raise ValueError(f"Modality names must be unique: {names}")

    windowing = config["windowing"]
    if windowing.get("train_mode") != "sliding":
        raise ValueError("Only windowing.train_mode='sliding' is supported")
    if windowing.get("evaluation_mode") != "full_sequence":
        raise ValueError("Only windowing.evaluation_mode='full_sequence' is supported")
    if windowing.get("target_mode") != "full_sequence":
        raise ValueError("Transformer training requires target_mode='full_sequence'")
    if int(windowing["window_len"]) <= 0 or int(windowing["hop_len"]) <= 0:
        raise ValueError("window_len and hop_len must be positive")

    fit_split = config["normalization"]["features"].get("fit_split", "train")
    expected_splits = config["dataset"].get("expected_splits", [])
    if expected_splits and fit_split not in expected_splits:
        raise ValueError(f"Unknown normalization fit split: {fit_split}")


def read_partition(dataset_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    path = dataset_root / dataset_cfg["partition_file"]
    partition = pd.read_csv(path)
    require_columns(
        partition,
        [dataset_cfg["participant_column"], dataset_cfg["split_column"]],
        path,
    )

    partition = partition[
        [dataset_cfg["participant_column"], dataset_cfg["split_column"]]
    ].copy()
    partition.columns = ["participant_id", "split"]
    partition["participant_id"] = partition["participant_id"].astype(str)
    partition["split"] = partition["split"].astype(str)

    duplicates = partition.loc[
        partition["participant_id"].duplicated(), "participant_id"
    ].tolist()
    if duplicates:
        raise ValueError(f"Participants appear more than once in partition file: {duplicates}")
    return partition


def feature_path(
    dataset_root: Path,
    config: dict[str, Any],
    modality: dict[str, Any],
    participant_id: str,
) -> Path:
    feature_dir = config["dataset"]["feature_segments_dir"]
    folder = modality.get("folder", modality["name"])
    return dataset_root / feature_dir / folder / folder / f"{participant_id}.csv"


def target_path(
    dataset_root: Path,
    config: dict[str, Any],
    participant_id: str,
) -> Path:
    target_dir = config["dataset"]["label_segments_dir"]
    target_folder = config["target"]["folder"]
    return dataset_root / target_dir / target_folder / f"{participant_id}.csv"


def read_signal_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    require_columns(df, columns, path)
    return df[columns].copy()


def equal_with_nan(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def load_participant(
    dataset_root: Path,
    config: dict[str, Any],
    participant_id: str,
) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    timestamp_col = dataset_cfg["timestamp_column"]
    segment_col = dataset_cfg["segment_column"]
    merged: pd.DataFrame | None = None
    source_lengths: list[tuple[str, int]] = []
    segment_columns: list[str] = []

    for modality in config["modalities"]:
        name = modality["name"]
        source_column = modality["column"]
        path = feature_path(dataset_root, config, modality, participant_id)
        df = read_signal_csv(path, [timestamp_col, segment_col, source_column])
        df = df.rename(
            columns={
                segment_col: f"segment_id__{name}",
                source_column: name,
            }
        )
        if df[timestamp_col].duplicated().any():
            raise ValueError(f"{path} contains duplicate timestamps")

        source_lengths.append((name, len(df)))
        segment_columns.append(f"segment_id__{name}")
        merged = df if merged is None else merged.merge(df, on=timestamp_col, how="inner")

    target_cfg = config["target"]
    label_path = target_path(dataset_root, config, participant_id)
    label_df = read_signal_csv(
        label_path,
        [timestamp_col, segment_col, target_cfg["column"]],
    ).rename(
        columns={
            segment_col: f"segment_id__{target_cfg['name']}",
            target_cfg["column"]: "target",
        }
    )
    if label_df[timestamp_col].duplicated().any():
        raise ValueError(f"{label_path} contains duplicate timestamps")

    source_lengths.append((target_cfg["name"], len(label_df)))
    segment_columns.append(f"segment_id__{target_cfg['name']}")
    merged = merged.merge(label_df, on=timestamp_col, how="inner")

    expected_rows = source_lengths[0][1]
    if any(length != expected_rows for _, length in source_lengths):
        raise ValueError(f"Participant {participant_id} has unequal source lengths: {source_lengths}")
    if len(merged) != expected_rows:
        raise ValueError(
            f"Participant {participant_id} lost rows during timestamp alignment: "
            f"expected {expected_rows}, got {len(merged)}"
        )

    reference_segment = merged[segment_columns[0]]
    for column in segment_columns[1:]:
        if not equal_with_nan(reference_segment, merged[column]).all():
            raise ValueError(
                f"Participant {participant_id} has mismatched segment ids in {column}"
            )

    modality_names = [modality["name"] for modality in config["modalities"]]
    merged["segment_id"] = reference_segment
    merged = merged[
        [timestamp_col, "segment_id", *modality_names, "target"]
    ].sort_values(timestamp_col)

    feature_values = merged[modality_names].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError(f"Participant {participant_id} contains non-finite feature values")
    return merged.reset_index(drop=True)


def load_all_participants(
    dataset_root: Path,
    config: dict[str, Any],
    partition: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        row.participant_id: load_participant(
            dataset_root,
            config,
            row.participant_id,
        )
        for row in partition.itertuples(index=False)
    }


def fit_feature_normalization(
    participants: dict[str, pd.DataFrame],
    partition: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame | None:
    norm_cfg = config["normalization"]["features"]
    if not bool(norm_cfg.get("enabled", True)):
        return None

    fit_split = norm_cfg.get("fit_split", "train")
    modality_names = [modality["name"] for modality in config["modalities"]]
    participant_ids = partition.loc[
        partition["split"] == fit_split,
        "participant_id",
    ]
    fit_values = np.concatenate(
        [
            participants[participant_id][modality_names].to_numpy(dtype=np.float64)
            for participant_id in participant_ids
        ],
        axis=0,
    )
    mean = fit_values.mean(axis=0)
    std = fit_values.std(axis=0)
    epsilon = float(norm_cfg.get("epsilon", 1e-8))
    std = np.where(std < epsilon, 1.0, std)

    return pd.DataFrame(
        {
            "modality": modality_names,
            "mean": mean,
            "std": std,
            "fit_split": fit_split,
            "unique_frames_used": len(fit_values),
        }
    )


def apply_feature_normalization(
    participants: dict[str, pd.DataFrame],
    stats: pd.DataFrame | None,
    config: dict[str, Any],
) -> None:
    if stats is None:
        return

    modality_names = [modality["name"] for modality in config["modalities"]]
    mean = stats.set_index("modality").loc[modality_names, "mean"].to_numpy()
    std = stats.set_index("modality").loc[modality_names, "std"].to_numpy()
    for df in participants.values():
        df.loc[:, modality_names] = (
            df[modality_names].to_numpy(dtype=np.float64) - mean
        ) / std


def contiguous_ranges(segment_ids: pd.Series) -> list[tuple[int, int]]:
    if segment_ids.empty:
        return []
    changes = np.flatnonzero(
        segment_ids.to_numpy()[1:] != segment_ids.to_numpy()[:-1]
    ) + 1
    boundaries = np.concatenate(([0], changes, [len(segment_ids)]))
    return [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(len(boundaries) - 1)
    ]


def training_ranges(df: pd.DataFrame, config: dict[str, Any]) -> list[tuple[int, int]]:
    windowing = config["windowing"]
    window_len = int(windowing["window_len"])
    hop_len = int(windowing["hop_len"])
    include_incomplete = bool(
        windowing.get("include_incomplete_train_window", True)
    )

    base_ranges = (
        contiguous_ranges(df["segment_id"])
        if bool(windowing.get("respect_segment_boundaries", False))
        else [(0, len(df))]
    )

    ranges: list[tuple[int, int]] = []
    for base_start, base_end in base_ranges:
        for start in range(base_start, base_end, hop_len):
            end = min(start + window_len, base_end)
            if end - start < window_len and not include_incomplete:
                continue
            ranges.append((start, end))
            if end == base_end:
                break
    return ranges


def make_samples(
    df: pd.DataFrame,
    participant_id: str,
    split: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if split == "train":
        ranges = training_ranges(df, config)
        mode = config["windowing"]["train_mode"]
    else:
        ranges = [(0, len(df))]
        mode = config["windowing"]["evaluation_mode"]

    modality_names = [modality["name"] for modality in config["modalities"]]
    timestamp_col = config["dataset"]["timestamp_column"]
    samples: list[dict[str, Any]] = []

    for sample_index, (start, end) in enumerate(ranges):
        sample = df.iloc[start:end]
        segment_ids = sample["segment_id"].to_numpy(dtype=np.int64)
        samples.append(
            {
                "x": sample[modality_names].to_numpy(dtype=np.float64),
                "y": sample["target"].to_numpy(dtype=np.float64).reshape(-1, 1),
                "timestamps": sample[timestamp_col].to_numpy(dtype=np.int64),
                "segment_ids": segment_ids,
                "metadata": {
                    "participant_id": participant_id,
                    "split": split,
                    "sample_id": sample_index,
                    "mode": mode,
                    "row_start": start,
                    "row_end_exclusive": end,
                    "sequence_length": end - start,
                    "timestamp_start": int(sample[timestamp_col].iloc[0]),
                    "timestamp_end": int(sample[timestamp_col].iloc[-1]),
                    "segment_id_start": int(segment_ids[0]),
                    "segment_id_end": int(segment_ids[-1]),
                    "crosses_segment_boundary": bool(segment_ids[0] != segment_ids[-1]),
                },
            }
        )
    return samples


def stack_samples(
    samples: list[dict[str, Any]],
    split: str,
    config: dict[str, Any],
    dtype: np.dtype,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if not samples:
        raise ValueError(f"Split has no samples: {split}")

    output_cfg = config["output"]
    n_samples = len(samples)
    n_modalities = len(config["modalities"])
    max_length = (
        int(config["windowing"]["window_len"])
        if split == "train"
        else max(len(sample["x"]) for sample in samples)
    )

    x = np.full(
        (n_samples, max_length, n_modalities),
        float(output_cfg.get("feature_padding_value", 0.0)),
        dtype=dtype,
    )
    y = np.full(
        (n_samples, max_length, 1),
        float(output_cfg.get("target_padding_value", 0.0)),
        dtype=dtype,
    )
    padding_mask = np.ones((n_samples, max_length), dtype=bool)
    target_mask = np.zeros((n_samples, max_length, 1), dtype=bool)
    timestamps = np.full(
        (n_samples, max_length),
        int(output_cfg.get("timestamp_padding_value", -1)),
        dtype=np.int64,
    )
    segment_ids = np.full(
        (n_samples, max_length),
        int(output_cfg.get("segment_padding_value", -1)),
        dtype=np.int64,
    )

    metadata_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        length = len(sample["x"])
        if length > max_length:
            raise ValueError(
                f"Sample length {length} exceeds padded length {max_length} in {split}"
            )

        valid_target = np.isfinite(sample["y"])
        x[index, :length] = sample["x"].astype(dtype, copy=False)
        y[index, :length] = np.where(
            valid_target,
            sample["y"],
            float(output_cfg.get("target_padding_value", 0.0)),
        ).astype(dtype, copy=False)
        padding_mask[index, :length] = False
        target_mask[index, :length] = valid_target
        timestamps[index, :length] = sample["timestamps"]
        segment_ids[index, :length] = sample["segment_ids"]

        metadata = dict(sample["metadata"])
        metadata["padded_length"] = max_length
        metadata["padding_steps"] = max_length - length
        metadata["valid_target_steps"] = int(valid_target.sum())
        metadata_rows.append(metadata)

    arrays = {
        "x": x,
        "y": y,
        "padding_mask": padding_mask,
        "target_mask": target_mask,
        "timestamps": timestamps,
        "segment_ids": segment_ids,
    }
    return arrays, pd.DataFrame(metadata_rows)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")


def save_outputs(
    output_dir: Path,
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    metadata_by_split: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    used_config: dict[str, Any],
    normalization_stats: pd.DataFrame | None,
) -> None:
    for split, arrays in arrays_by_split.items():
        metadata = metadata_by_split[split]
        np.savez_compressed(
            output_dir / f"{split}.npz",
            **arrays,
            participant_id=metadata["participant_id"].to_numpy(dtype=str),
            sample_id=metadata["sample_id"].to_numpy(dtype=np.int64),
        )
        metadata.to_csv(output_dir / f"{split}_metadata.csv", index=False)

    if normalization_stats is not None:
        normalization_stats.to_csv(
            output_dir / "feature_normalization_stats.csv",
            index=False,
        )

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with (output_dir / "used_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(used_config, handle, sort_keys=False)


def build_manifest(
    config_path: Path,
    dataset_root: Path,
    config: dict[str, Any],
    partition: pd.DataFrame,
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    metadata_by_split: dict[str, pd.DataFrame],
    normalization_stats: pd.DataFrame | None,
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    for split, arrays in arrays_by_split.items():
        metadata = metadata_by_split[split]
        split_summaries[split] = {
            "participants": int(
                partition.loc[
                    partition["split"] == split,
                    "participant_id",
                ].nunique()
            ),
            "samples": int(arrays["x"].shape[0]),
            "x_shape": list(arrays["x"].shape),
            "y_shape": list(arrays["y"].shape),
            "padding_steps": int(arrays["padding_mask"].sum()),
            "valid_target_steps": int(arrays["target_mask"].sum()),
            "windows_crossing_segment_boundary": int(
                metadata["crosses_segment_boundary"].sum()
            ),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "modalities": [modality["name"] for modality in config["modalities"]],
        "target": config["target"]["name"],
        "windowing": config["windowing"],
        "normalization": {
            "enabled": normalization_stats is not None,
            "fit_split": (
                normalization_stats["fit_split"].iloc[0]
                if normalization_stats is not None
                else None
            ),
            "unique_frames_used": (
                int(normalization_stats["unique_frames_used"].iloc[0])
                if normalization_stats is not None
                else 0
            ),
        },
        "mask_semantics": {
            "padding_mask": "True means padded/ignored timestep",
            "target_mask": "True means target is available and may contribute to loss",
        },
        "splits": split_summaries,
    }


def preprocess(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)

    dataset_root = resolve_existing_path(config["dataset"]["root"], config_path)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dtype_name = config["output"].get("dtype", "float32")
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("output.dtype must be float32 or float64")
    dtype = np.dtype(dtype_name)

    partition = read_partition(dataset_root, config)
    participants = load_all_participants(dataset_root, config, partition)
    normalization_stats = fit_feature_normalization(
        participants,
        partition,
        config,
    )
    apply_feature_normalization(participants, normalization_stats, config)

    split_names = list(
        config["dataset"].get(
            "expected_splits",
            sorted(partition["split"].unique()),
        )
    )
    samples_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in split_names
    }
    for row in partition.itertuples(index=False):
        samples_by_split.setdefault(row.split, []).extend(
            make_samples(
                participants[row.participant_id],
                row.participant_id,
                row.split,
                config,
            )
        )

    arrays_by_split: dict[str, dict[str, np.ndarray]] = {}
    metadata_by_split: dict[str, pd.DataFrame] = {}
    for split, samples in samples_by_split.items():
        arrays, metadata = stack_samples(samples, split, config, dtype)
        arrays_by_split[split] = arrays
        metadata_by_split[split] = metadata

    output_dir = resolve_output_path(config["output"]["dir"])
    prepare_output_dir(
        output_dir,
        bool(config["output"].get("overwrite", True)),
    )
    manifest = build_manifest(
        config_path,
        dataset_root,
        config,
        partition,
        arrays_by_split,
        metadata_by_split,
        normalization_stats,
    )
    save_outputs(
        output_dir,
        arrays_by_split,
        metadata_by_split,
        manifest,
        config,
        normalization_stats,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MuSe-Physio for sequence-to-sequence training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/muse_physio_baseline.yaml"),
        help="YAML preprocessing config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = preprocess(args.config)
    print("Preprocessing complete.")
    for split, summary in manifest["splits"].items():
        print(
            f"{split}: participants={summary['participants']} "
            f"samples={summary['samples']} "
            f"x_shape={summary['x_shape']} "
            f"y_shape={summary['y_shape']} "
            f"padding_steps={summary['padding_steps']} "
            f"valid_target_steps={summary['valid_target_steps']}"
        )


if __name__ == "__main__":
    main()
