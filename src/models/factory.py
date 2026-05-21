"""Model factory."""

from __future__ import annotations

from torch import nn

from .baselines import (
    ConstantVelocityBaseline,
    EgoMotionOnlyModel,
    LastFrameVisualBaseline,
    VideoHistoryBaseline,
)
from .cue_memory import TwoStreamEgocentricCueMemoryPathPredictor


def create_model(
    model_name: str,
    future_steps: int,
    backbone_name: str = "small_cnn",
    hidden_dim: int = 128,
    freeze_backbone: bool = True,
    **kwargs,
) -> nn.Module:
    name = model_name.lower()
    if name == "constant_velocity":
        return ConstantVelocityBaseline(future_steps=future_steps)
    if name == "ego_motion_only":
        return EgoMotionOnlyModel(
            future_steps=future_steps,
            hidden_dim=hidden_dim,
            layers=int(kwargs.get("ego_layers", 1)),
        )
    if name == "last_frame_dino":
        return LastFrameVisualBaseline(
            future_steps=future_steps,
            backbone_name=backbone_name,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
        )
    if name == "video_history_dino":
        return VideoHistoryBaseline(
            future_steps=future_steps,
            backbone_name=backbone_name,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
        )
    if name == "cue_memory_path_predictor":
        return TwoStreamEgocentricCueMemoryPathPredictor(
            future_steps=future_steps,
            backbone_name=backbone_name,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            use_bottleneck_adapters=bool(kwargs.get("use_bottleneck_adapters", True)),
            adapter_bottleneck_dim=int(kwargs.get("adapter_bottleneck_dim", 64)),
            temporal_type=str(kwargs.get("temporal_type", "transformer")),
            temporal_layers=int(kwargs.get("temporal_layers", 1)),
            num_cue_tokens=int(kwargs.get("num_cue_tokens", 8)),
            selector_layers=int(kwargs.get("selector_layers", 1)),
            selector_type=str(kwargs.get("selector_type", "query_attention")),
            memory_type=str(kwargs.get("memory_type", "gru_cell")),
            use_spatial_graph=bool(kwargs.get("use_spatial_graph", False)),
            spatial_graph_neighbors=int(kwargs.get("spatial_graph_neighbors", 8)),
            use_temporal_difference_conv=bool(kwargs.get("use_temporal_difference_conv", False)),
            use_temporal_shift=bool(kwargs.get("use_temporal_shift", False)),
            decoder_layers=int(kwargs.get("decoder_layers", 1)),
            dropout=float(kwargs.get("dropout", 0.1)),
            use_constant_velocity_residual=bool(kwargs.get("use_constant_velocity_residual", True)),
            residual_scale=float(kwargs.get("residual_scale", 1.0)),
            num_modes=int(kwargs.get("num_modes", 1)),
        )
    raise ValueError(f"Unknown model: {model_name}")


def needs_rgb(model_name: str) -> bool:
    return model_name.lower() in {
        "last_frame_dino",
        "video_history_dino",
        "cue_memory_path_predictor",
    }
