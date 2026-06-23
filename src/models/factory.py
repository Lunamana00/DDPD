"""Model factory."""

from __future__ import annotations

from torch import nn

from .baselines import (
    ConstantVelocityBaseline,
    EgoMotionOnlyModel,
    KhalequeMotivatedExplorerBaseline,
    LastFrameVisualBaseline,
    XuPixelsOnlyTrajectoryBaseline,
    VideoHistoryBaseline,
)
from .cue_memory import EpisodicLongTermCueMemoryPathPredictor, TwoStreamEgocentricCueMemoryPathPredictor


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
            dropout=float(kwargs.get("dropout", 0.0)),
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
            dropout=float(kwargs.get("dropout", 0.0)),
        )
    if name == "xu_pixels_only_baseline":
        return XuPixelsOnlyTrajectoryBaseline(
            future_steps=future_steps,
            backbone_name=backbone_name,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            dropout=float(kwargs.get("dropout", 0.1)),
        )
    if name == "khaleque_motivated_baseline":
        return KhalequeMotivatedExplorerBaseline(
            future_steps=future_steps,
            hidden_dim=hidden_dim,
            ego_layers=int(kwargs.get("ego_layers", 1)),
            num_motivation_tokens=int(kwargs.get("num_motivation_tokens", 4)),
            num_heads=int(kwargs.get("num_heads", 4)),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
    if name in {"cue_memory_path_predictor", "episodic_long_term_cue_memory_path_predictor", "episodic_cue_memory_path_predictor"}:
        model_cls = (
            EpisodicLongTermCueMemoryPathPredictor
            if name in {"episodic_long_term_cue_memory_path_predictor", "episodic_cue_memory_path_predictor"}
            else TwoStreamEgocentricCueMemoryPathPredictor
        )
        common_kwargs = dict(
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
            tokenlearner_pooling=str(kwargs.get("tokenlearner_pooling", "sigmoid")),
            memory_type=str(kwargs.get("memory_type", "gru_cell")),
            use_spatial_graph=bool(kwargs.get("use_spatial_graph", False)),
            spatial_graph_neighbors=int(kwargs.get("spatial_graph_neighbors", 8)),
            spatial_relation_type=kwargs.get("spatial_relation_type"),
            use_temporal_difference_conv=bool(kwargs.get("use_temporal_difference_conv", False)),
            use_temporal_shift=bool(kwargs.get("use_temporal_shift", False)),
            decoder_layers=int(kwargs.get("decoder_layers", 1)),
            decoder_type=str(kwargs.get("decoder_type", "horizon_query_decoder")),
            cue_temporal_layers=int(kwargs.get("cue_temporal_layers", 1)),
            dropout=float(kwargs.get("dropout", 0.1)),
            use_constant_velocity_residual=bool(kwargs.get("use_constant_velocity_residual", True)),
            residual_scale=float(kwargs.get("residual_scale", 1.0)),
            num_modes=int(kwargs.get("num_modes", 1)),
        )
        if model_cls is EpisodicLongTermCueMemoryPathPredictor:
            common_kwargs.update(
                long_memory_type=str(kwargs.get("long_memory_type", "gated_attention")),
                long_memory_slots=(
                    None
                    if kwargs.get("long_memory_slots", None) is None
                    else int(kwargs.get("long_memory_slots"))
                ),
                long_memory_use_ego=bool(kwargs.get("long_memory_use_ego", True)),
                detach_long_memory=bool(kwargs.get("detach_long_memory", True)),
            )
        return model_cls(**common_kwargs)
    raise ValueError(f"Unknown model: {model_name}")


def needs_rgb(model_name: str, backbone_name: str | None = None) -> bool:
    if backbone_name is not None and backbone_name.lower() in {"zero_tokens", "zero_visual", "no_visual"}:
        return False
    return model_name.lower() in {
        "last_frame_dino",
        "video_history_dino",
        "xu_pixels_only_baseline",
        "cue_memory_path_predictor",
        "episodic_long_term_cue_memory_path_predictor",
        "episodic_cue_memory_path_predictor",
    }
