"""Baselines for WIT-VZ future local path prediction."""

from __future__ import annotations

import torch
from torch import nn

from .backbones import build_visual_encoder
from .motion import constant_velocity_path


class ConstantVelocityBaseline(nn.Module):
    def __init__(self, future_steps: int, average_last: int = 5) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.average_last = average_last

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return constant_velocity_path(batch["ego_history"], self.future_steps, self.average_last)


class EgoMotionOnlyModel(nn.Module):
    def __init__(self, future_steps: int, hidden_dim: int = 128, layers: int = 1) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_dim, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, future_steps * 2))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _out, h = self.gru(batch["ego_history"])
        pred = self.head(h[-1])
        return pred.view(pred.shape[0], self.future_steps, 2)


class LastFrameVisualBaseline(nn.Module):
    def __init__(
        self,
        future_steps: int,
        backbone_name: str = "small_cnn",
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = build_visual_encoder(backbone_name, hidden_dim, freeze_backbone)
        enc_dim = getattr(self.encoder, "out_dim", hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        frames = batch["rgb_history"][:, -1:, ...]
        tokens = self.encoder(frames)
        pooled = tokens.mean(dim=(1, 2))
        pred = self.head(pooled)
        return pred.view(pred.shape[0], self.future_steps, 2)


class VideoHistoryBaseline(nn.Module):
    def __init__(
        self,
        future_steps: int,
        backbone_name: str = "small_cnn",
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = build_visual_encoder(backbone_name, hidden_dim, freeze_backbone)
        enc_dim = getattr(self.encoder, "out_dim", hidden_dim)
        self.project = nn.Linear(enc_dim, hidden_dim) if enc_dim != hidden_dim else nn.Identity()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, future_steps * 2))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.encoder(batch["rgb_history"])
        per_frame = self.project(tokens.mean(dim=2))
        _out, h = self.gru(per_frame)
        pred = self.head(h[-1])
        return pred.view(pred.shape[0], self.future_steps, 2)
