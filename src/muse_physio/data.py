from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_ARRAYS = {
    "x",
    "y",
    "padding_mask",
    "target_mask",
    "timestamps",
    "segment_ids",
    "participant_id",
    "sample_id",
}


class MusePhysioDataset(Dataset[dict[str, Any]]):
    """In-memory dataset backed by a processed split NPZ file."""

    def __init__(self, processed_dir: str | Path, split: str) -> None:
        self.processed_dir = Path(processed_dir)
        self.split = split
        split_path = self.processed_dir / f"{split}.npz"
        if not split_path.exists():
            raise FileNotFoundError(split_path)

        with np.load(split_path) as archive:
            missing = REQUIRED_ARRAYS.difference(archive.files)
            if missing:
                raise ValueError(f"{split_path} is missing arrays: {sorted(missing)}")
            self.x = torch.from_numpy(archive["x"].copy()).float()
            self.y = torch.from_numpy(archive["y"].copy()).float()
            self.padding_mask = torch.from_numpy(
                archive["padding_mask"].copy()
            ).bool()
            self.target_mask = torch.from_numpy(
                archive["target_mask"].copy()
            ).bool()
            self.timestamps = torch.from_numpy(
                archive["timestamps"].copy()
            ).long()
            self.segment_ids = torch.from_numpy(
                archive["segment_ids"].copy()
            ).long()
            self.participant_ids = archive["participant_id"].astype(str).tolist()
            self.sample_ids = archive["sample_id"].astype(np.int64).tolist()

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n_samples, sequence_length, _ = self.x.shape
        expected_sequence_shape = (n_samples, sequence_length)
        expected_target_shape = (n_samples, sequence_length, 1)

        if tuple(self.y.shape) != expected_target_shape:
            raise ValueError(
                f"Expected y shape {expected_target_shape}, got {tuple(self.y.shape)}"
            )
        if tuple(self.padding_mask.shape) != expected_sequence_shape:
            raise ValueError(
                "padding_mask shape does not match x: "
                f"{tuple(self.padding_mask.shape)} vs {expected_sequence_shape}"
            )
        if tuple(self.target_mask.shape) != expected_target_shape:
            raise ValueError(
                "target_mask shape does not match y: "
                f"{tuple(self.target_mask.shape)} vs {expected_target_shape}"
            )
        if tuple(self.timestamps.shape) != expected_sequence_shape:
            raise ValueError("timestamps shape does not match x")
        if tuple(self.segment_ids.shape) != expected_sequence_shape:
            raise ValueError("segment_ids shape does not match x")
        if len(self.participant_ids) != n_samples or len(self.sample_ids) != n_samples:
            raise ValueError("Sample identifiers do not match the number of samples")
        if self.target_mask.squeeze(-1)[self.padding_mask].any():
            raise ValueError("Padded timesteps cannot have valid targets")
        if not torch.isfinite(self.x).all() or not torch.isfinite(self.y).all():
            raise ValueError("Processed tensors must contain only finite values")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "padding_mask": self.padding_mask[index],
            "target_mask": self.target_mask[index],
            "timestamps": self.timestamps[index],
            "segment_ids": self.segment_ids[index],
            "participant_id": self.participant_ids[index],
            "sample_id": self.sample_ids[index],
        }


def load_manifest(processed_dir: str | Path) -> dict[str, Any]:
    path = Path(processed_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_dataloader(
    processed_dir: str | Path,
    split: str,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    dataset = MusePhysioDataset(processed_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
