"""Path prediction metrics."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def prediction_paths(pred: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(pred, torch.Tensor):
        return pred
    paths = pred.get("paths", pred.get("trajectories"))
    if paths is None:
        raise KeyError("Multimodal predictions must contain a 'paths' tensor")
    return paths


def select_best_trajectory(
    pred: torch.Tensor | Mapping[str, torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    paths = prediction_paths(pred)
    if paths.ndim == 3:
        return paths
    if paths.ndim != 4:
        raise ValueError(f"Expected prediction shape [B,H,2] or [B,K,H,2], got {tuple(paths.shape)}")
    errors = torch.linalg.norm(paths - target[:, None, :, :], dim=-1).mean(dim=-1)
    best_mode = errors.argmin(dim=1)
    batch_idx = torch.arange(paths.shape[0], device=paths.device)
    return paths[batch_idx, best_mode]


def displacement_errors(pred: torch.Tensor | Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    selected = select_best_trajectory(pred, target)
    return torch.linalg.norm(selected - target, dim=-1)


def ade(pred: torch.Tensor | Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    return displacement_errors(pred, target).mean()


def fde(pred: torch.Tensor | Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    return displacement_errors(pred, target)[:, -1].mean()


def per_horizon_error(pred: torch.Tensor | Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    return displacement_errors(pred, target).mean(dim=0)


def batch_metrics(pred: torch.Tensor | Mapping[str, torch.Tensor], target: torch.Tensor) -> dict[str, float]:
    errors = displacement_errors(pred, target)
    return {
        "ADE": float(errors.mean().detach().cpu()),
        "FDE": float(errors[:, -1].mean().detach().cpu()),
    }
