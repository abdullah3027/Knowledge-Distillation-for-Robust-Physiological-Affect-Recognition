from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class TimeSeriesTransformer(nn.Module):
    """Bidirectional encoder-only Transformer for frame-level regression."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_sequence_length: int,
        *,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, max_sequence_length, d_model)
        )
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Linear(d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [batch, time, features], got {x.shape}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, got {x.shape[-1]}"
            )
        if x.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds configured maximum "
                f"{self.max_sequence_length}"
            )
        if padding_mask is not None and padding_mask.shape != x.shape[:2]:
            raise ValueError(
                f"padding_mask shape {padding_mask.shape} does not match {x.shape[:2]}"
            )

        hidden = self.input_projection(x) * math.sqrt(self.d_model)
        hidden = hidden + self.position_embedding[:, : x.shape[1]]
        hidden = self.input_dropout(hidden)

        hidden_states: list[torch.Tensor] = []
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
            if return_hidden_states:
                hidden_states.append(hidden)

        hidden = self.final_norm(hidden)
        prediction = self.regression_head(hidden)
        if return_hidden_states:
            return prediction, hidden_states
        return prediction

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TimeSeriesTransformer":
        return cls(
            input_dim=int(config["input_dim"]),
            d_model=int(config["d_model"]),
            num_heads=int(config["num_heads"]),
            num_layers=int(config["num_layers"]),
            dim_feedforward=int(config["dim_feedforward"]),
            dropout=float(config["dropout"]),
            max_sequence_length=int(config["max_sequence_length"]),
            activation=str(config.get("activation", "gelu")),
            norm_first=bool(config.get("norm_first", True)),
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def count_parameters_from_config(config: dict[str, Any]) -> int:
    model = TimeSeriesTransformer.from_config(config)
    return count_trainable_parameters(model)
