"""Motion priors shared by path prediction models."""

from __future__ import annotations

import torch


def constant_velocity_path(
    ego_history: torch.Tensor,
    future_steps: int,
    average_last: int = 5,
) -> torch.Tensor:
    """Project future local path from recent egocentric motion increments."""

    recent = ego_history[:, -min(average_last, ego_history.shape[1]) :, :2]
    velocity = recent.mean(dim=1)
    increments = velocity[:, None, :].repeat(1, future_steps, 1)
    return torch.cumsum(increments, dim=1)
