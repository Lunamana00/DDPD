"""Cue-memory MSTP selector for Xu/VisualGuidance data."""

from __future__ import annotations

import torch
from torch import nn

from .backbones import build_visual_encoder
from .cue_memory import (
    AttentionCueMemoryBank,
    BottleneckAdapter,
    DynamicSpatialGraphAggregator,
    FullSpatialAttentionAggregator,
    LocalGridSpatialAggregator,
    SpatialPositionalEncoding,
    TokenLearnerCueTokenSelector,
)


def _spatial_relation_module(kind: str, dim: int, neighbors: int, dropout: float) -> nn.Module:
    normalized = kind.lower()
    if normalized in {"none", "identity"}:
        return nn.Identity()
    if normalized in {"topk_graph", "dynamic_graph"}:
        return DynamicSpatialGraphAggregator(dim, neighbors=neighbors, dropout=dropout)
    if normalized in {"full_attention", "full"}:
        return FullSpatialAttentionAggregator(dim, dropout=dropout)
    if normalized in {"local_grid", "local"}:
        return LocalGridSpatialAggregator(dim, dropout=dropout)
    raise ValueError(f"Unknown spatial relation type: {kind}")


def _token_centers(num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    height, width = SpatialPositionalEncoding._grid_size(num_tokens)
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.reshape(-1)[:num_tokens] / max(width - 1, 1)
    y = y.reshape(-1)[:num_tokens] / max(height - 1, 1)
    return torch.stack([x, y], dim=-1)


def _box_geometry(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    width = (x2 - x1).clamp_min(0.0)
    height = (y2 - y1).clamp_min(0.0)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    area = width * height
    aspect = width / height.clamp_min(1.0e-6)
    return torch.stack([x1, y1, x2, y2, center_x, center_y, width, height, area, aspect], dim=-1)


def _roi_pool_tokens(tokens: torch.Tensor, boxes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # tokens: [B, N, D], boxes: [B, C, 4], mask: [B, C]
    batch, num_tokens, dim = tokens.shape
    max_candidates = boxes.shape[1]
    centers = _token_centers(num_tokens, tokens.device, tokens.dtype)
    pooled = tokens.new_zeros((batch, max_candidates, dim))
    for b in range(batch):
        for c in range(max_candidates):
            if not bool(mask[b, c]):
                continue
            x1, y1, x2, y2 = boxes[b, c]
            inside = (
                (centers[:, 0] >= x1)
                & (centers[:, 0] <= x2)
                & (centers[:, 1] >= y1)
                & (centers[:, 1] <= y2)
            )
            if inside.any():
                pooled[b, c] = tokens[b, inside].mean(dim=0)
            else:
                center = boxes[b, c].view(2, 2).mean(dim=0)
                distances = torch.sum((centers - center[None, :]) ** 2, dim=-1)
                pooled[b, c] = tokens[b, int(distances.argmin())]
    return pooled


class CueMemoryMSTPSelector(nn.Module):
    """Apply the project's cue-memory visual reasoning to MSTP selection.

    Unlike the path predictor, this model is single-frame and has no ego-motion
    or future trajectory label. It encodes the screenshot into visual cue memory
    and scores each candidate STP box as the Main STP.
    """

    def __init__(
        self,
        backbone_name: str = "small_cnn",
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
        num_cue_tokens: int = 8,
        spatial_relation_type: str = "topk_graph",
        spatial_graph_neighbors: int = 8,
        dropout: float = 0.1,
        adapter_bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = build_visual_encoder(backbone_name, hidden_dim, freeze_backbone)
        enc_dim = getattr(self.encoder, "out_dim", hidden_dim)
        self.project = nn.Linear(enc_dim, hidden_dim) if enc_dim != hidden_dim else nn.Identity()
        self.spatial_position = SpatialPositionalEncoding(hidden_dim)
        self.adapter = BottleneckAdapter(hidden_dim, adapter_bottleneck_dim, dropout)
        self.spatial_relation = _spatial_relation_module(
            spatial_relation_type,
            hidden_dim,
            spatial_graph_neighbors,
            dropout,
        )
        self.cue_selector = TokenLearnerCueTokenSelector(
            hidden_dim,
            num_cue_tokens=num_cue_tokens,
            layers=1,
            dropout=dropout,
            pooling="sigmoid",
        )
        self.memory_bank = AttentionCueMemoryBank(hidden_dim, num_cue_tokens, dropout=dropout)
        self.box_embed = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch["image"]
        boxes = batch["candidate_boxes"].float()
        mask = batch["candidate_mask"].bool()
        tokens = self.encoder(images[:, None, ...])
        tokens = self.project(tokens)
        tokens = self.spatial_position(tokens)
        tokens = self.adapter(tokens)
        tokens = self.spatial_relation(tokens)
        frame_tokens = tokens[:, 0, :, :]
        cues = self.cue_selector(frame_tokens)
        zero_ego = frame_tokens.new_zeros((frame_tokens.shape[0], 1, 3))
        memory = self.memory_bank(cues[:, None, :, :], zero_ego)

        candidate_roi = _roi_pool_tokens(frame_tokens, boxes, mask)
        candidate_query = candidate_roi + self.box_embed(_box_geometry(boxes))
        attended, _weights = self.candidate_attention(candidate_query, memory, memory, need_weights=False)
        fused = torch.cat([candidate_query, attended], dim=-1)
        scores = self.score_head(fused).squeeze(-1)
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
