from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModalitySelection:
    """Column selection contract for model inputs."""

    source_modalities: tuple[str, ...]
    selected_modalities: tuple[str, ...]
    indices: tuple[int, ...]
    mode: str = "select"

    @property
    def input_dim(self) -> int:
        return len(self.indices)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [batch, time, features], got {x.shape}")
        if x.shape[-1] != len(self.source_modalities):
            raise ValueError(
                f"Expected {len(self.source_modalities)} source features, "
                f"got {x.shape[-1]}"
            )
        return x[..., list(self.indices)]

    def to_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_modalities": list(self.source_modalities),
            "available_modalities": list(self.selected_modalities),
            "indices": list(self.indices),
        }


def _normalise_names(values: list[Any] | tuple[Any, ...], field: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names:
        raise ValueError(f"{field} must contain at least one modality")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} contains duplicate modalities: {names}")
    return names


def resolve_modality_selection(
    manifest: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> ModalitySelection:
    """Resolve a config into a deterministic source-column selection."""

    source_modalities = _normalise_names(manifest["modalities"], "manifest modalities")
    if config is None:
        selected = source_modalities
        mode = "select"
    else:
        mode = str(config.get("mode", "select"))
        if mode != "select":
            raise ValueError(f"Unsupported student_input.mode: {mode}")
        has_available = "available_modalities" in config
        has_missing = "missing_modalities" in config
        if has_available and has_missing:
            raise ValueError(
                "Specify either available_modalities or missing_modalities, not both"
            )
        if has_available:
            selected = _normalise_names(
                config["available_modalities"],
                "available_modalities",
            )
        elif has_missing:
            missing = set(
                _normalise_names(
                    config["missing_modalities"],
                    "missing_modalities",
                )
            )
            selected = tuple(name for name in source_modalities if name not in missing)
        else:
            selected = source_modalities

    unknown = sorted(set(selected).difference(source_modalities))
    if unknown:
        raise ValueError(f"Unknown selected modalities: {unknown}")
    if not selected:
        raise ValueError("At least one modality must be selected")

    indices = tuple(source_modalities.index(name) for name in selected)
    return ModalitySelection(
        source_modalities=source_modalities,
        selected_modalities=selected,
        indices=indices,
        mode=mode,
    )


def validate_model_input_dim(
    model_config: dict[str, Any],
    selection: ModalitySelection,
    *,
    model_name: str = "Model",
) -> None:
    input_dim = int(model_config["input_dim"])
    if input_dim != selection.input_dim:
        raise ValueError(
            f"{model_name} expects {input_dim} input features, but selected "
            f"modalities require {selection.input_dim}: {selection.selected_modalities}"
        )
