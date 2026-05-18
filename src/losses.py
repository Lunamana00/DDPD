"""Trajectory losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import torch
import torch.nn.functional as F


def _prediction_paths_and_logits(
    pred: torch.Tensor | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(pred, torch.Tensor):
        return pred, None
    paths = pred.get("paths", pred.get("trajectories"))
    if paths is None:
        raise KeyError("Multimodal predictions must contain a 'paths' tensor")
    logits = pred.get("logits", pred.get("mode_logits"))
    return paths, logits


def _coordinate_scale(
    coordinate_scale: float | Sequence[float] | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    scale = torch.as_tensor(
        coordinate_scale,
        dtype=reference.dtype,
        device=reference.device,
    ).clamp_min(1.0e-6)
    while scale.ndim < reference.ndim:
        scale = scale.unsqueeze(0)
    return scale


def trajectory_loss(
    pred: torch.Tensor | Mapping[str, torch.Tensor],
    target: torch.Tensor,
    loss_type: str = "huber",
    coordinate_scale: float | Sequence[float] | torch.Tensor = 1.0,
    multimodal_confidence_weight: float = 0.05,
) -> torch.Tensor:
    paths, logits = _prediction_paths_and_logits(pred)
    if paths.ndim == 3:
        scale = _coordinate_scale(coordinate_scale, paths)
        pred_for_loss = paths / scale
        target_for_loss = target / scale
        if loss_type == "huber":
            return F.smooth_l1_loss(pred_for_loss, target_for_loss)
        if loss_type in {"mse", "l2"}:
            return F.mse_loss(pred_for_loss, target_for_loss)
        raise ValueError(f"Unsupported trajectory loss: {loss_type}")

    if paths.ndim != 4:
        raise ValueError(f"Expected prediction shape [B,H,2] or [B,K,H,2], got {tuple(paths.shape)}")
    target_by_mode = target[:, None, :, :].expand_as(paths)
    scale = _coordinate_scale(coordinate_scale, paths)
    pred_for_loss = paths / scale
    target_for_loss = target_by_mode / scale
    if loss_type == "huber":
        per_coord = F.smooth_l1_loss(pred_for_loss, target_for_loss, reduction="none")
    elif loss_type in {"mse", "l2"}:
        per_coord = F.mse_loss(pred_for_loss, target_for_loss, reduction="none")
    else:
        raise ValueError(f"Unsupported trajectory loss: {loss_type}")

    mode_losses = per_coord.mean(dim=(-1, -2))
    best_mode = mode_losses.argmin(dim=1)
    loss = mode_losses.gather(1, best_mode[:, None]).mean()
    if logits is not None and multimodal_confidence_weight > 0.0:
        loss = loss + multimodal_confidence_weight * F.cross_entropy(logits, best_mode.detach())
    return loss
