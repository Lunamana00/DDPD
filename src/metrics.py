"""Path prediction metrics."""

from __future__ import annotations

import torch


def displacement_errors(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - target, dim=-1)


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return displacement_errors(pred, target).mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return displacement_errors(pred, target)[:, -1].mean()


def per_horizon_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return displacement_errors(pred, target).mean(dim=0)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    errors = displacement_errors(pred, target)
    return {
        "ADE": float(errors.mean().detach().cpu()),
        "FDE": float(errors[:, -1].mean().detach().cpu()),
    }
