"""Baselines for WIT-VZ future local path prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
    def __init__(self, future_steps: int, hidden_dim: int = 128, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_dim, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, future_steps * 2))

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


def _tokens_from_batch_or_encoder(
    batch: dict[str, torch.Tensor],
    encoder: nn.Module,
) -> torch.Tensor:
    if "visual_tokens" in batch:
        return batch["visual_tokens"].float()
    return encoder(batch["rgb_history"])


class VideoHistoryBaseline(nn.Module):
    def __init__(
        self,
        future_steps: int,
        backbone_name: str = "small_cnn",
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = build_visual_encoder(backbone_name, hidden_dim, freeze_backbone)
        enc_dim = getattr(self.encoder, "out_dim", hidden_dim)
        self.project = nn.Linear(enc_dim, hidden_dim) if enc_dim != hidden_dim else nn.Identity()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, future_steps * 2))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = _tokens_from_batch_or_encoder(batch, self.encoder)
        per_frame = self.project(tokens.mean(dim=2))
        _out, h = self.gru(per_frame)
        pred = self.head(h[-1])
        return pred.view(pred.shape[0], self.future_steps, 2)


class XuPixelsOnlyTrajectoryBaseline(nn.Module):
    """Trainable screen-only trajectory baseline inspired by Xu et al.

    The model deliberately excludes ego-motion and memory. It asks how far a
    compact pixels-only representation can go on this offline trajectory task.
    Cached DINO tokens can be supplied through batch["visual_tokens"].
    """

    def __init__(
        self,
        future_steps: int,
        backbone_name: str = "small_cnn",
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = build_visual_encoder(backbone_name, hidden_dim, freeze_backbone)
        enc_dim = getattr(self.encoder, "out_dim", hidden_dim)
        self.project = nn.Linear(enc_dim, hidden_dim) if enc_dim != hidden_dim else nn.Identity()
        self.token_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = _tokens_from_batch_or_encoder(batch, self.encoder)
        tokens = self.project(tokens)
        scores = self.token_score(tokens).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        per_frame = torch.sum(tokens * weights.unsqueeze(-1), dim=2)
        _out, h = self.temporal_gru(per_frame)
        pred = self.head(h[-1])
        return pred.view(pred.shape[0], self.future_steps, 2)


class KhalequeMotivatedExplorerBaseline(nn.Module):
    """Trainable ego-motion baseline inspired by exploratory-agent motivation.

    WIT-VZ does not store the live motivation/coverage/object state needed for
    an exact reproduction of Khaleque et al. This baseline therefore keeps the
    comparable part: a motion-only agent-state encoder plus learned motivation
    tokens that decode a future local rollout.
    """

    def __init__(
        self,
        future_steps: int,
        hidden_dim: int = 128,
        ego_layers: int = 1,
        num_motivation_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.future_steps = future_steps
        self.ego_encoder = nn.GRU(3, hidden_dim, num_layers=ego_layers, batch_first=True)
        self.motion_norm = nn.LayerNorm(hidden_dim)
        self.motivation_tokens = nn.Parameter(torch.randn(num_motivation_tokens, hidden_dim) * 0.02)
        self.horizon_queries = nn.Parameter(torch.randn(future_steps, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.step_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["ego_history"].shape[0]
        _out, h = self.ego_encoder(batch["ego_history"])
        motion = self.motion_norm(h[-1])
        motivation = self.motivation_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        memory = torch.cat([motion[:, None, :], motivation], dim=1)
        queries = self.horizon_queries.unsqueeze(0).expand(batch_size, -1, -1) + motion[:, None, :]
        decoded, _weights = self.cross_attention(queries, memory, memory)
        step_delta = self.step_head(decoded)
        return torch.cumsum(step_delta, dim=1)
