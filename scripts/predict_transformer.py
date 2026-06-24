from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from muse_physio.data import create_dataloader, load_manifest
from muse_physio.modalities import resolve_modality_selection, validate_model_input_dim
from muse_physio.model import TimeSeriesTransformer
from muse_physio.training import resolve_device


@torch.no_grad()
def run(
    checkpoint_path: Path,
    processed_dir: Path,
    split: str,
    output_path: Path,
    device_name: str,
) -> None:
    device = resolve_device(device_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = TimeSeriesTransformer.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    manifest = checkpoint.get("manifest") or load_manifest(processed_dir)
    input_config = checkpoint.get("model_input")
    if input_config is None:
        input_config = checkpoint.get("training_config", {}).get("student_input")
    input_selection = resolve_modality_selection(manifest, input_config)
    validate_model_input_dim(checkpoint["model_config"], input_selection)

    loader = create_dataloader(
        processed_dir,
        split,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, object]] = []
    for batch in loader:
        model_input = input_selection.apply(batch["x"]).to(device)
        prediction = model(
            model_input,
            batch["padding_mask"].to(device),
        ).cpu()
        valid_sequence = ~batch["padding_mask"]
        for sample_index in range(prediction.shape[0]):
            participant_id = batch["participant_id"][sample_index]
            for timestep in torch.where(valid_sequence[sample_index])[0].tolist():
                target_available = bool(
                    batch["target_mask"][sample_index, timestep, 0]
                )
                rows.append(
                    {
                        "participant_id": participant_id,
                        "timestamp": int(batch["timestamps"][sample_index, timestep]),
                        "prediction": float(prediction[sample_index, timestep, 0]),
                        "target": float(batch["y"][sample_index, timestep, 0])
                        if target_available
                        else None,
                        "target_available": target_available,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frame-level predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "devel", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.checkpoint, args.processed_dir, args.split, args.output, args.device)
    print(f"Predictions saved to {args.output}")
