"""Visual token encoders."""

from __future__ import annotations

import importlib.util

import torch
from torch import nn


class SmallCNNTokenEncoder(nn.Module):
    """Small CNN fallback that returns spatial tokens.

    This is intended for tests and local CPU smoke runs. DINOv2 support is
    exposed through `DinoV2TokenEncoder`, but that path requires optional
    dependencies and locally available weights or network access.
    """

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W]
        batch, time, channels, height, width = frames.shape
        features = self.net(frames.reshape(batch * time, channels, height, width))
        tokens = features.flatten(2).transpose(1, 2)
        return tokens.reshape(batch, time, tokens.shape[1], self.out_dim)


class DinoV2TokenEncoder(nn.Module):
    """Optional DINOv2 token encoder via transformers.

    The implementation intentionally fails clearly when the optional dependency
    or weights are unavailable. This keeps the prototype runnable offline with
    `small_cnn` while preserving the intended DINOv2 path.
    """

    def __init__(self, model_name: str = "facebook/dinov2-small", freeze: bool = True) -> None:
        super().__init__()
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError(
                "DINOv2 requires the optional 'transformers' package and model weights. "
                "Install transformers and ensure weights are accessible, or use --backbone small_cnn."
            )
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = int(self.model.config.hidden_size)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        images = frames.reshape(batch * time, channels, height, width)
        outputs = self.model(pixel_values=images)
        tokens = outputs.last_hidden_state[:, 1:, :]
        return tokens.reshape(batch, time, tokens.shape[1], self.out_dim)


def build_visual_encoder(backbone_name: str, hidden_dim: int, freeze_backbone: bool = True) -> nn.Module:
    name = backbone_name.lower()
    if name in {"small_cnn", "cnn", "test_cnn"}:
        encoder = SmallCNNTokenEncoder(out_dim=hidden_dim)
        if freeze_backbone:
            for param in encoder.parameters():
                param.requires_grad = False
        return encoder
    if name in {"dinov2", "dino", "facebook/dinov2-small"}:
        return DinoV2TokenEncoder(freeze=freeze_backbone)
    raise ValueError(f"Unknown visual backbone: {backbone_name}")
