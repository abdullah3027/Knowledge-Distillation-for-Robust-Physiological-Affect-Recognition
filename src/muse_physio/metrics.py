from __future__ import annotations

import torch


def _masked_vectors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if mask.shape != target.shape:
        try:
            mask = mask.expand_as(target)
        except RuntimeError as error:
            raise ValueError(
                f"Mask shape {mask.shape} cannot match target {target.shape}"
            ) from error

    valid = mask.bool()
    if not valid.any():
        raise ValueError("Metric received no valid target values")
    return prediction[valid], target[valid]


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prediction_valid, target_valid = _masked_vectors(prediction, target, mask)
    return torch.mean((prediction_valid - target_valid) ** 2)


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prediction_valid, target_valid = _masked_vectors(prediction, target, mask)
    return torch.mean(torch.abs(prediction_valid - target_valid))


def masked_pearson(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    prediction_valid, target_valid = _masked_vectors(prediction, target, mask)
    prediction_centered = prediction_valid - prediction_valid.mean()
    target_centered = target_valid - target_valid.mean()
    denominator = torch.sqrt(
        torch.sum(prediction_centered**2) * torch.sum(target_centered**2)
    )
    return torch.sum(prediction_centered * target_centered) / (
        denominator + epsilon
    )


def masked_ccc(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    prediction_valid, target_valid = _masked_vectors(prediction, target, mask)
    prediction_mean = prediction_valid.mean()
    target_mean = target_valid.mean()
    prediction_centered = prediction_valid - prediction_mean
    target_centered = target_valid - target_mean
    covariance = torch.mean(prediction_centered * target_centered)
    prediction_variance = torch.mean(prediction_centered**2)
    target_variance = torch.mean(target_centered**2)
    mean_difference = (prediction_mean - target_mean) ** 2
    return (2.0 * covariance) / (
        prediction_variance + target_variance + mean_difference + epsilon
    )


def masked_ccc_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return 1.0 - masked_ccc(prediction, target, mask)
