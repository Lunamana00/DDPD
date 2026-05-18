"""Trajectory losses."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def trajectory_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "huber",
    coordinate_scale: float | Sequence[float] | torch.Tensor = 1.0,
) -> torch.Tensor:
    scale = torch.as_tensor(coordinate_scale, dtype=pred.dtype, device=pred.device).clamp_min(1.0e-6)
    while scale.ndim < pred.ndim:
        scale = scale.unsqueeze(0)
    pred_for_loss = pred / scale
    target_for_loss = target / scale
    if loss_type == "huber":
        return F.smooth_l1_loss(pred_for_loss, target_for_loss)
    if loss_type in {"mse", "l2"}:
        return F.mse_loss(pred_for_loss, target_for_loss)
    raise ValueError(f"Unsupported trajectory loss: {loss_type}")
