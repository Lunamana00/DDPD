"""Visual token encoders."""

from __future__ import annotations

import importlib.util
import os
import torch.nn.functional as F

import torch
from torch import nn


def _patch_restrictive_windows_mkdir() -> None:
    if os.name != "nt" or getattr(os.mkdir, "_wit_vz_patched", False):
        return
    original_mkdir = os.mkdir

    def mkdir_without_restrictive_mode(path, mode=0o777, *args, **kwargs):
        if mode == 0o700:
            mode = 0o777
        return original_mkdir(path, mode, *args, **kwargs)

    mkdir_without_restrictive_mode._wit_vz_patched = True
    os.mkdir = mkdir_without_restrictive_mode


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


class DinoV3TokenEncoder(nn.Module):
    """DINOv3 token encoder via Hugging Face Transformers.

    The default ConvNeXt-Tiny variant is the most practical DINOv3 candidate for
    this project: frozen, dense, and more locality-biased than a large ViT.
    """

    MODEL_ALIASES = {
        "dinov3": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "dinov3-convnext-tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "dinov3_convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "facebook/dinov3-convnext-tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "dinov3-convnext-small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        "dinov3_convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        "dinov3-vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    }

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        freeze: bool = True,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError(
                "DINOv3 requires 'transformers>=4.56' and accessible model weights. "
                "Install the optional dependency or use --backbone small_cnn."
            )
        from transformers import AutoModel

        resolved_model_name = self.MODEL_ALIASES.get(model_name.lower(), model_name)
        self.model = AutoModel.from_pretrained(resolved_model_name)
        self.model_name = resolved_model_name
        self.freeze = freeze
        self.image_size = image_size
        self.out_dim = self._resolve_output_dim()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def _resolve_output_dim(self) -> int:
        if hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)
        hidden_sizes = getattr(self.model.config, "hidden_sizes", None)
        if hidden_sizes:
            return int(hidden_sizes[-1])
        raise ValueError(f"Cannot infer output dim for {self.model_name}")

    def train(self, mode: bool = True) -> "DinoV3TokenEncoder":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (images - self.mean.to(images.dtype)) / self.std.to(images.dtype)

    def _outputs_to_tokens(self, outputs: object) -> torch.Tensor:
        hidden = outputs.last_hidden_state
        if hidden.ndim == 4:
            return hidden.flatten(2).transpose(1, 2)
        if hidden.ndim == 3:
            return hidden[:, 1:, :] if hidden.shape[1] > 1 else hidden
        raise ValueError(f"Unexpected DINOv3 hidden state shape: {tuple(hidden.shape)}")

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W], expected in [0, 1].
        batch, time, channels, height, width = frames.shape
        images = frames.reshape(batch * time, channels, height, width)
        images = self._prepare_images(images)
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(pixel_values=images)
        else:
            outputs = self.model(pixel_values=images)
        tokens = self._outputs_to_tokens(outputs)
        return tokens.reshape(batch, time, tokens.shape[1], tokens.shape[2])


class TimmDinoV3ConvNeXtTokenEncoder(nn.Module):
    """DINOv3 ConvNeXt feature encoder from timm/Hugging Face Hub."""

    MODEL_ALIASES = {
        "dinov3-convnext-tiny": "hf-hub:timm/convnext_tiny.dinov3_lvd1689m",
        "dinov3_convnext_tiny": "hf-hub:timm/convnext_tiny.dinov3_lvd1689m",
        "timm/convnext_tiny.dinov3_lvd1689m": "hf-hub:timm/convnext_tiny.dinov3_lvd1689m",
        "hf-hub:timm/convnext_tiny.dinov3_lvd1689m": "hf-hub:timm/convnext_tiny.dinov3_lvd1689m",
    }

    def __init__(
        self,
        model_name: str = "hf-hub:timm/convnext_tiny.dinov3_lvd1689m",
        freeze: bool = True,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        if importlib.util.find_spec("timm") is None:
            raise RuntimeError(
                "DINOv3 ConvNeXt via timm requires the optional 'timm' package. "
                "Install it or use --backbone small_cnn."
            )
        _patch_restrictive_windows_mkdir()
        import timm

        resolved_model_name = self.MODEL_ALIASES.get(model_name.lower(), model_name)
        self.model = timm.create_model(
            resolved_model_name,
            pretrained=True,
            features_only=True,
            out_indices=(-1,),
        )
        self.model_name = resolved_model_name
        self.freeze = freeze
        self.image_size = image_size
        self.out_dim = int(self.model.feature_info.channels()[-1])
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> "TimmDinoV3ConvNeXtTokenEncoder":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (images - self.mean.to(images.dtype)) / self.std.to(images.dtype)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W], expected in [0, 1].
        batch, time, channels, height, width = frames.shape
        images = frames.reshape(batch * time, channels, height, width)
        images = self._prepare_images(images)
        if self.freeze:
            with torch.no_grad():
                features = self.model(images)[-1]
        else:
            features = self.model(images)[-1]
        tokens = features.flatten(2).transpose(1, 2)
        return tokens.reshape(batch, time, tokens.shape[1], tokens.shape[2])


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
    if name in TimmDinoV3ConvNeXtTokenEncoder.MODEL_ALIASES:
        return TimmDinoV3ConvNeXtTokenEncoder(model_name=backbone_name, freeze=freeze_backbone)
    if name.startswith("dinov3") or "facebook/dinov3" in name:
        return DinoV3TokenEncoder(model_name=backbone_name, freeze=freeze_backbone)
    raise ValueError(f"Unknown visual backbone: {backbone_name}")
