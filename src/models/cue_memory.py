"""Two-stream egocentric cue-memory path predictor."""

from __future__ import annotations

import math

import torch
from torch import nn

from .backbones import ZeroVisualTokenEncoder, build_visual_encoder
from .motion import constant_velocity_path


def _sinusoidal_encoding_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        return positions.new_zeros((positions.shape[0], 0))
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=positions.device, dtype=positions.dtype)
        * (-math.log(10000.0) / max(dim, 1))
    )
    scaled = positions[:, None] * div_term[None, :]
    encoding = positions.new_zeros((positions.shape[0], dim))
    encoding[:, 0::2] = torch.sin(scaled)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(scaled[:, : encoding[:, 1::2].shape[1]])
    return encoding


class SpatialPositionalEncoding(nn.Module):
    """Deterministic 2D positional encoding for visual token grids."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @staticmethod
    def _grid_size(num_tokens: int) -> tuple[int, int]:
        root = int(math.sqrt(num_tokens))
        for height in range(root, 0, -1):
            if num_tokens % height == 0:
                return height, num_tokens // height
        return 1, num_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        _batch, _time, num_tokens, dim = tokens.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {dim}")
        height, width = self._grid_size(num_tokens)
        y, x = torch.meshgrid(
            torch.arange(height, device=tokens.device, dtype=tokens.dtype),
            torch.arange(width, device=tokens.device, dtype=tokens.dtype),
            indexing="ij",
        )
        y = y.reshape(-1)[:num_tokens] / max(height - 1, 1)
        x = x.reshape(-1)[:num_tokens] / max(width - 1, 1)
        y_dim = dim // 2
        x_dim = dim - y_dim
        encoding = torch.cat(
            [
                _sinusoidal_encoding_1d(y, y_dim),
                _sinusoidal_encoding_1d(x, x_dim),
            ],
            dim=-1,
        )
        return tokens + encoding.view(1, 1, num_tokens, dim)


class TemporalPositionalEncoding(nn.Module):
    """Deterministic temporal positional encoding for frame histories."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        _batch, time, _num_tokens, dim = tokens.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {dim}")
        positions = torch.arange(time, device=tokens.device, dtype=tokens.dtype)
        encoding = _sinusoidal_encoding_1d(positions, dim)
        return tokens + encoding.view(1, time, 1, dim)


class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DynamicSpatialGraphAggregator(nn.Module):
    """Sparse dynamic graph reasoning over visual tokens within each frame."""

    def __init__(self, dim: int, neighbors: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        batch, time, num_tokens, dim = tokens.shape
        if num_tokens <= 1 or self.neighbors <= 0:
            return tokens
        flat = tokens.reshape(batch * time, num_tokens, dim)
        normalized = self.norm(flat)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(dim)
        eye = torch.eye(num_tokens, device=tokens.device, dtype=torch.bool)[None, :, :]
        scores = scores.masked_fill(eye, torch.finfo(scores.dtype).min)
        neighbor_count = min(self.neighbors, num_tokens - 1)
        top_scores, top_indices = torch.topk(scores, k=neighbor_count, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        expanded_value = value[:, None, :, :].expand(-1, num_tokens, -1, -1)
        gather_index = top_indices[..., None].expand(-1, -1, -1, dim)
        neighbors = torch.gather(expanded_value, dim=2, index=gather_index)
        context = (weights[..., None] * neighbors).sum(dim=2)
        aggregated = flat + self.out(context)
        return aggregated.reshape(batch, time, num_tokens, dim)


class FullSpatialAttentionAggregator(nn.Module):
    """Dense frame-local attention over all visual tokens except self."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        batch, time, num_tokens, dim = tokens.shape
        if num_tokens <= 1:
            return tokens
        flat = tokens.reshape(batch * time, num_tokens, dim)
        normalized = self.norm(flat)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(dim)
        eye = torch.eye(num_tokens, device=tokens.device, dtype=torch.bool)[None, :, :]
        scores = scores.masked_fill(eye, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)
        aggregated = flat + self.out(context)
        return aggregated.reshape(batch, time, num_tokens, dim)


class LocalGridSpatialAggregator(nn.Module):
    """Frame-local attention restricted to fixed 8-neighborhood grid edges."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    @staticmethod
    def _local_neighbor_mask(num_tokens: int, device: torch.device) -> torch.Tensor:
        height, width = SpatialPositionalEncoding._grid_size(num_tokens)
        mask = torch.zeros((num_tokens, num_tokens), device=device, dtype=torch.bool)
        for index in range(num_tokens):
            row, col = divmod(index, width)
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue
                    neighbor_row = row + row_offset
                    neighbor_col = col + col_offset
                    if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                        neighbor_index = neighbor_row * width + neighbor_col
                        if neighbor_index < num_tokens:
                            mask[index, neighbor_index] = True
        return mask

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        batch, time, num_tokens, dim = tokens.shape
        if num_tokens <= 1:
            return tokens
        flat = tokens.reshape(batch * time, num_tokens, dim)
        normalized = self.norm(flat)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(dim)
        local_mask = self._local_neighbor_mask(num_tokens, tokens.device)[None, :, :]
        scores = scores.masked_fill(~local_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)
        aggregated = flat + self.out(context)
        return aggregated.reshape(batch, time, num_tokens, dim)


class RelativeContrastHybridSpatialGraphAggregator(nn.Module):
    """Frame-local graph attention with relative, contrast, and local-grid priors."""

    def __init__(
        self,
        dim: int,
        neighbors: int = 8,
        dropout: float = 0.1,
        use_relative_position: bool = False,
        use_contrast: bool = False,
        include_local_prior: bool = False,
    ) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.use_relative_position = use_relative_position
        self.use_contrast = use_contrast
        self.include_local_prior = include_local_prior
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        if use_relative_position:
            rel_hidden = max(16, dim // 4)
            self.relative_bias = nn.Sequential(
                nn.Linear(4, rel_hidden),
                nn.GELU(),
                nn.Linear(rel_hidden, 1),
            )
        else:
            self.relative_bias = None
        if use_contrast:
            contrast_dim = max(8, min(32, dim // 8))
            contrast_hidden = max(16, dim // 4)
            self.contrast_pair_projection = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, contrast_dim),
            )
            self.contrast_score = nn.Sequential(
                nn.LayerNorm(contrast_dim),
                nn.Linear(contrast_dim, contrast_hidden),
                nn.GELU(),
                nn.Linear(contrast_hidden, 1),
            )
            self.contrast_score_chunk_size = 128
            self.contrast_message = nn.Sequential(
                nn.LayerNorm(contrast_dim),
                nn.Linear(contrast_dim, contrast_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(contrast_hidden, contrast_dim),
            )
            self.contrast_up = nn.Linear(contrast_dim, dim)
            self.contrast_message_scale = nn.Parameter(torch.tensor(1.0e-3))
        else:
            self.contrast_pair_projection = None
            self.contrast_score = None
            self.contrast_score_chunk_size = 128
            self.contrast_message = None
            self.contrast_up = None
            self.contrast_message_scale = None
        self.gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    @staticmethod
    def _relative_features(num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = SpatialPositionalEncoding._grid_size(num_tokens)
        rows = torch.arange(height, device=device, dtype=dtype).repeat_interleave(width)[:num_tokens]
        cols = torch.arange(width, device=device, dtype=dtype).repeat(height)[:num_tokens]
        dy = rows[None, :] - rows[:, None]
        dx = cols[None, :] - cols[:, None]
        dy = dy / max(height - 1, 1)
        dx = dx / max(width - 1, 1)
        return torch.stack([dy, dx, dy.abs(), dx.abs()], dim=-1)

    def _relative_position_bias(
        self,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.relative_bias is None:
            raise RuntimeError("relative position bias requested but not initialized")
        parameter = next(self.relative_bias.parameters())
        features = self._relative_features(num_tokens, device, parameter.dtype)
        return self.relative_bias(features).squeeze(-1).to(dtype=dtype)

    def _pairwise_contrast_bias(self, normalized: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.contrast_pair_projection is None or self.contrast_score is None:
            raise RuntimeError("pairwise contrast score requested but not initialized")
        contrast_features = self.contrast_pair_projection(normalized)
        chunk_size = max(1, self.contrast_score_chunk_size)
        biases = []
        for start in range(0, contrast_features.shape[0], chunk_size):
            chunk = contrast_features[start : start + chunk_size]
            pairwise_diff = chunk[:, None, :, :] - chunk[:, :, None, :]
            biases.append(self.contrast_score(pairwise_diff).squeeze(-1))
        return torch.cat(biases, dim=0).to(dtype=dtype)

    def _candidate_contrast_message(
        self,
        normalized: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_weights: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self.contrast_pair_projection is None
            or self.contrast_message is None
            or self.contrast_up is None
            or self.contrast_message_scale is None
        ):
            raise RuntimeError("candidate contrast message requested but not initialized")
        contrast_features = self.contrast_pair_projection(normalized)
        batch_time, num_tokens, contrast_dim = contrast_features.shape
        chunk_size = max(1, self.contrast_score_chunk_size)
        messages = []
        for start in range(0, batch_time, chunk_size):
            end = start + chunk_size
            chunk = contrast_features[start:end]
            indices = candidate_indices[start:end]
            weights = candidate_weights[start:end]
            expanded = chunk[:, None, :, :].expand(-1, num_tokens, -1, -1)
            gather_index = indices[..., None].expand(-1, -1, -1, contrast_dim)
            neighbors = torch.gather(expanded, dim=2, index=gather_index)
            center = chunk[:, :, None, :]
            edge_messages = self.contrast_message(neighbors - center)
            messages.append((weights[..., None] * edge_messages).sum(dim=2))
        contrast_context = torch.cat(messages, dim=0)
        return (self.contrast_message_scale * self.contrast_up(contrast_context)).to(dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        batch, time, num_tokens, dim = tokens.shape
        if num_tokens <= 1:
            return tokens
        if self.neighbors <= 0 and not self.include_local_prior:
            return tokens
        flat = tokens.reshape(batch * time, num_tokens, dim)
        normalized = self.norm(flat)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(dim)

        if self.use_relative_position:
            rel_bias = self._relative_position_bias(num_tokens, tokens.device, scores.dtype)
            scores = scores + rel_bias[None, :, :]

        if self.use_contrast:
            if (
                self.contrast_pair_projection is None
                or self.contrast_score is None
                or self.contrast_message is None
                or self.contrast_up is None
                or self.contrast_message_scale is None
            ):
                raise RuntimeError("contrast-aware graph requested but not initialized")
            scores = scores + self._pairwise_contrast_bias(normalized, scores.dtype)

        eye = torch.eye(num_tokens, device=tokens.device, dtype=torch.bool)[None, :, :]
        scores = scores.masked_fill(eye, torch.finfo(scores.dtype).min)
        neighbor_count = min(max(self.neighbors, 0), num_tokens - 1)
        candidate_mask = None
        if neighbor_count > 0:
            top_indices = torch.topk(scores, k=neighbor_count, dim=-1).indices
            candidate_mask = torch.zeros_like(scores, dtype=torch.bool)
            candidate_mask.scatter_(2, top_indices, True)
        if self.include_local_prior:
            local_mask = LocalGridSpatialAggregator._local_neighbor_mask(num_tokens, tokens.device)[None, :, :]
            candidate_mask = local_mask if candidate_mask is None else candidate_mask | local_mask
        if candidate_mask is None:
            return tokens

        candidate_mask = candidate_mask.expand_as(scores)
        max_candidates = int(candidate_mask.sum(dim=-1).max().detach().cpu().item())
        if max_candidates <= 0:
            return tokens
        masked_scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
        candidate_scores, candidate_indices = torch.topk(masked_scores, k=max_candidates, dim=-1)
        valid_candidates = torch.gather(candidate_mask, dim=2, index=candidate_indices)
        candidate_weights = torch.softmax(candidate_scores, dim=-1).masked_fill(~valid_candidates, 0.0)
        expanded_value = value[:, None, :, :].expand(-1, num_tokens, -1, -1)
        gather_index = candidate_indices[..., None].expand(-1, -1, -1, dim)
        selected_values = torch.gather(expanded_value, dim=2, index=gather_index)
        context = (candidate_weights[..., None] * selected_values).sum(dim=2)
        if self.use_contrast:
            context = context + self._candidate_contrast_message(
                normalized,
                candidate_indices,
                candidate_weights,
                context.dtype,
            )
        gate = self.gate(torch.cat([normalized, context], dim=-1))
        aggregated = flat + self.out(gate * context)
        return aggregated.reshape(batch, time, num_tokens, dim)


def build_spatial_relation_module(
    relation_type: str,
    dim: int,
    neighbors: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    normalized = relation_type.lower()
    if normalized in {"none", "identity"}:
        return nn.Identity()
    if normalized in {"topk_graph", "dynamic_graph"}:
        return DynamicSpatialGraphAggregator(dim, neighbors, dropout=dropout)
    if normalized in {"full_attention", "full"}:
        return FullSpatialAttentionAggregator(dim, dropout=dropout)
    if normalized in {"local_grid", "local"}:
        return LocalGridSpatialAggregator(dim, dropout=dropout)
    if normalized == "strnet_edge_message":
        return EdgeMessageDynamicSpatialGraphAggregator(dim, neighbors, dropout=dropout)
    if normalized in {"relpos_topk_graph", "relative_topk_graph", "relative_position_topk_graph"}:
        return RelativeContrastHybridSpatialGraphAggregator(
            dim,
            neighbors,
            dropout=dropout,
            use_relative_position=True,
        )
    if normalized in {"contrast_topk_graph", "contrast_graph"}:
        return RelativeContrastHybridSpatialGraphAggregator(
            dim,
            neighbors,
            dropout=dropout,
            use_contrast=True,
        )
    if normalized in {"hybrid_local_topk_graph", "hybrid_graph", "local_topk_graph"}:
        return RelativeContrastHybridSpatialGraphAggregator(
            dim,
            neighbors,
            dropout=dropout,
            include_local_prior=True,
        )
    if normalized in {"relpos_contrast_hybrid_graph", "relative_contrast_hybrid_graph"}:
        return RelativeContrastHybridSpatialGraphAggregator(
            dim,
            neighbors,
            dropout=dropout,
            use_relative_position=True,
            use_contrast=True,
            include_local_prior=True,
        )
    if normalized in {"relpos_contrast_topk_graph", "relative_contrast_topk_graph"}:
        return RelativeContrastHybridSpatialGraphAggregator(
            dim,
            neighbors,
            dropout=dropout,
            use_relative_position=True,
            use_contrast=True,
        )
    raise ValueError(f"Unknown spatial_relation_type: {relation_type}")


class EdgeMessageDynamicSpatialGraphAggregator(nn.Module):
    """Dynamic graph aggregation with STRNet-style edge messages."""

    def __init__(self, dim: int, neighbors: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.edge_gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        batch, time, num_tokens, dim = tokens.shape
        if num_tokens <= 1 or self.neighbors <= 0:
            return tokens
        flat = tokens.reshape(batch * time, num_tokens, dim)
        normalized = self.norm(flat)
        query = self.q(normalized)
        key = self.k(normalized)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(dim)
        eye = torch.eye(num_tokens, device=tokens.device, dtype=torch.bool)[None, :, :]
        scores = scores.masked_fill(eye, torch.finfo(scores.dtype).min)
        neighbor_count = min(self.neighbors, num_tokens - 1)
        top_scores, top_indices = torch.topk(scores, k=neighbor_count, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        expanded_tokens = normalized[:, None, :, :].expand(-1, num_tokens, -1, -1)
        gather_index = top_indices[..., None].expand(-1, -1, -1, dim)
        neighbors = torch.gather(expanded_tokens, dim=2, index=gather_index)
        center = normalized[:, :, None, :].expand(-1, -1, neighbor_count, -1)
        edge_features = torch.cat([center, neighbors - center], dim=-1)
        messages = self.edge_mlp(edge_features)
        context = (weights[..., None] * messages).sum(dim=2)
        gate = self.edge_gate(torch.cat([normalized, context], dim=-1))
        aggregated = flat + self.out(gate * context)
        return aggregated.reshape(batch, time, num_tokens, dim)


class TemporalDifferenceConv(nn.Module):
    """Multi-resolution temporal difference mixing for each spatial token."""

    def __init__(
        self,
        dim: int,
        dilations: tuple[int, ...] = (1, 2),
        dropout: float = 0.1,
        mode: str = "legacy",
    ) -> None:
        super().__init__()
        self.dilations = dilations
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.branches = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            dim,
                            dim,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                            groups=dim,
                        ),
                        nn.GELU(),
                        nn.Conv1d(dim, dim, kernel_size=1),
                    )
                    for dilation in dilations
                ]
            )
            if mode == "multi_resolution"
            else None
        )
        self.mix = nn.Sequential(
            nn.Conv1d(dim * (len(dilations) + 1), dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    @staticmethod
    def _difference(sequence: torch.Tensor, dilation: int) -> torch.Tensor:
        if sequence.shape[1] <= dilation:
            shifted = sequence[:, :1, :].expand(-1, sequence.shape[1], -1)
        else:
            prefix = sequence[:, :1, :].expand(-1, dilation, -1)
            shifted = torch.cat([prefix, sequence[:, :-dilation, :]], dim=1)
        return sequence - shifted

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [B*N, T, D]
        normalized = self.norm(sequence)
        features = [normalized.transpose(1, 2)]
        for index, dilation in enumerate(self.dilations):
            difference = self._difference(normalized, dilation).transpose(1, 2)
            if self.mode == "multi_resolution":
                if self.branches is None:
                    raise RuntimeError("multi_resolution mode requires temporal convolution branches")
                difference = self.branches[index](difference)
            features.append(difference)
        mixed = self.mix(torch.cat(features, dim=1)).transpose(1, 2)
        return sequence + mixed


class TemporalShift(nn.Module):
    """Lightweight temporal channel shift over each spatial token sequence."""

    def __init__(self, dim: int, shift_ratio: float = 0.25) -> None:
        super().__init__()
        self.fold = max(0, int(dim * shift_ratio / 2.0))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [B*N, T, D]
        if self.fold == 0 or sequence.shape[1] <= 1:
            return sequence
        shifted = sequence.clone()
        shifted[:, 1:, : self.fold] = sequence[:, :-1, : self.fold]
        shifted[:, :-1, self.fold : 2 * self.fold] = sequence[:, 1:, self.fold : 2 * self.fold]
        return shifted


class DividedSpaceTimeBlock(nn.Module):
    """TimeSformer-style temporal attention followed by spatial attention."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        batch, time, num_tokens, dim = tokens.shape
        temporal = tokens.permute(0, 2, 1, 3).reshape(batch * num_tokens, time, dim)
        attended, _weights = self.temporal_attn(
            self.temporal_norm(temporal),
            self.temporal_norm(temporal),
            self.temporal_norm(temporal),
            need_weights=False,
        )
        temporal = temporal + self.dropout(attended)
        tokens = temporal.reshape(batch, num_tokens, time, dim).permute(0, 2, 1, 3)

        spatial = tokens.reshape(batch * time, num_tokens, dim)
        attended, _weights = self.spatial_attn(
            self.spatial_norm(spatial),
            self.spatial_norm(spatial),
            self.spatial_norm(spatial),
            need_weights=False,
        )
        spatial = spatial + self.dropout(attended)
        spatial = spatial + self.dropout(self.ffn(self.ffn_norm(spatial)))
        return spatial.reshape(batch, time, num_tokens, dim)


class STRNetFusionBlock(nn.Module):
    """STRNet-style spatial graph plus temporal shift and difference-aware conv."""

    def __init__(self, dim: int, neighbors: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.spatial_graph = EdgeMessageDynamicSpatialGraphAggregator(dim, neighbors=neighbors, dropout=dropout)
        self.temporal_shift = TemporalShift(dim)
        self.difference_conv = TemporalDifferenceConv(
            dim,
            dilations=(1, 2, 4),
            dropout=dropout,
            mode="multi_resolution",
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        tokens = self.spatial_graph(tokens)
        batch, time, num_tokens, dim = tokens.shape
        sequence = tokens.permute(0, 2, 1, 3).reshape(batch * num_tokens, time, dim)
        sequence = self.temporal_shift(sequence)
        sequence = self.difference_conv(sequence)
        tokens = sequence.reshape(batch, num_tokens, time, dim).permute(0, 2, 1, 3)
        flat = tokens.reshape(batch * time * num_tokens, dim)
        flat = flat + self.dropout(self.ffn(self.ffn_norm(flat)))
        return flat.reshape(batch, time, num_tokens, dim)


class TemporalAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        temporal_type: str = "transformer",
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        use_difference_conv: bool = False,
        use_temporal_shift: bool = False,
        spatial_graph_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.temporal_type = temporal_type.lower()
        self.position = TemporalPositionalEncoding(dim)
        self.temporal_shift = TemporalShift(dim) if use_temporal_shift else nn.Identity()
        self.difference_conv = TemporalDifferenceConv(dim, dropout=dropout) if use_difference_conv else nn.Identity()
        if self.temporal_type in {"none", "identity"}:
            self.module = nn.Identity()
        elif self.temporal_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.module = nn.TransformerEncoder(layer, num_layers=layers)
        elif self.temporal_type == "gru":
            self.module = nn.GRU(dim, dim, num_layers=layers, batch_first=True)
        elif self.temporal_type == "timesformer":
            self.module = nn.ModuleList(
                [DividedSpaceTimeBlock(dim, heads=heads, dropout=dropout) for _ in range(layers)]
            )
        elif self.temporal_type == "strnet":
            self.module = nn.ModuleList(
                [
                    STRNetFusionBlock(dim, neighbors=spatial_graph_neighbors, dropout=dropout)
                    for _ in range(layers)
                ]
            )
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

    def _apply_sequence_mixers(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, time, num_tokens, dim = tokens.shape
        sequence = tokens.permute(0, 2, 1, 3).reshape(batch * num_tokens, time, dim)
        sequence = self.difference_conv(sequence)
        return sequence.reshape(batch, num_tokens, time, dim).permute(0, 2, 1, 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        if self.temporal_type in {"none", "identity"}:
            return tokens
        batch, time, num_tokens, dim = tokens.shape
        positioned = self.position(tokens)
        per_spatial_token = positioned.permute(0, 2, 1, 3).reshape(batch * num_tokens, time, dim)
        per_spatial_token = self.temporal_shift(per_spatial_token)
        positioned = per_spatial_token.reshape(batch, num_tokens, time, dim).permute(0, 2, 1, 3)
        if self.temporal_type == "timesformer":
            temporal = positioned
            for block in self.module:
                temporal = block(temporal)
            temporal = self._apply_sequence_mixers(temporal)
            return tokens + temporal
        if self.temporal_type == "strnet":
            temporal = positioned
            for block in self.module:
                temporal = block(temporal)
            return temporal
        if self.temporal_type == "gru":
            temporal, _h = self.module(per_spatial_token)
        else:
            temporal = self.module(per_spatial_token)
        temporal = self.difference_conv(temporal)
        temporal = temporal.reshape(batch, num_tokens, time, dim).permute(0, 2, 1, 3)
        return tokens + temporal


class LearnedCueTokenSelector(nn.Module):
    def __init__(
        self,
        dim: int,
        num_cue_tokens: int = 8,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_cue_tokens, dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                        "norm1": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        "dropout": nn.Dropout(dropout),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
                for _ in range(layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D]
        batch = tokens.shape[0]
        cues = self.queries[None, :, :].expand(batch, -1, -1)
        for layer in self.layers:
            attended, _weights = layer["attn"](cues, tokens, tokens, need_weights=False)
            cues = layer["norm1"](cues + layer["dropout"](attended))
            cues = layer["norm2"](cues + layer["dropout"](layer["ffn"](cues)))
        return cues


class TokenLearnerCueTokenSelector(nn.Module):
    """TokenLearner-style soft attention maps that mine S adaptive cue tokens."""

    def __init__(
        self,
        dim: int,
        num_cue_tokens: int = 8,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        hidden = max(dim // 2, 1)
        self.attention_maps = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_cue_tokens),
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                        "norm1": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        "dropout": nn.Dropout(dropout),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
                for _ in range(layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D]
        attention = self.attention_maps(tokens).transpose(1, 2)
        if self.pooling == "softmax":
            attention = torch.softmax(attention, dim=-1)
        elif self.pooling == "sigmoid":
            attention = torch.sigmoid(attention)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        else:
            raise ValueError(f"Unknown TokenLearner pooling mode: {self.pooling}")
        cues = torch.bmm(attention, tokens)
        for layer in self.layers:
            attended, _weights = layer["attn"](cues, cues, cues, need_weights=False)
            cues = layer["norm1"](cues + layer["dropout"](attended))
            cues = layer["norm2"](cues + layer["dropout"](layer["ffn"](cues)))
        return cues


class CueSpaceTimeTransformer(nn.Module):
    """Transformer over TokenLearner cue tokens across time."""

    def __init__(
        self,
        dim: int,
        num_cue_tokens: int,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.temporal_position = TemporalPositionalEncoding(dim)
        self.cue_position = nn.Parameter(torch.randn(num_cue_tokens, dim) * 0.02)
        if layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(dim)
        else:
            self.encoder = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, cues_over_time: torch.Tensor) -> torch.Tensor:
        # cues_over_time: [B, T, K, D]
        if self.layers <= 0:
            return cues_over_time
        batch, time, num_cues, dim = cues_over_time.shape
        positioned = self.temporal_position(cues_over_time)
        positioned = positioned + self.cue_position[:num_cues].view(1, 1, num_cues, dim)
        encoded = self.encoder(positioned.reshape(batch, time * num_cues, dim))
        encoded = self.norm(encoded)
        return encoded.reshape(batch, time, num_cues, dim)


class TopKTokenLearnerSelector(nn.Module):
    """Input-adaptive cue token mining with score-gated Top-K selection."""

    def __init__(
        self,
        dim: int,
        num_cue_tokens: int = 8,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_cue_tokens = num_cue_tokens
        hidden = max(dim // 2, 1)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                        "norm1": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        "dropout": nn.Dropout(dropout),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
                for _ in range(layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D]
        batch, num_tokens, dim = tokens.shape
        cue_count = min(self.num_cue_tokens, num_tokens)
        scores = self.score(tokens).squeeze(-1)
        top_scores, top_indices = torch.topk(scores, k=cue_count, dim=1)
        selected = torch.gather(tokens, dim=1, index=top_indices[..., None].expand(-1, -1, dim))
        selected = selected * torch.sigmoid(top_scores[..., None])
        for layer in self.layers:
            attended, _weights = layer["attn"](selected, selected, selected, need_weights=False)
            selected = layer["norm1"](selected + layer["dropout"](attended))
            selected = layer["norm2"](selected + layer["dropout"](layer["ffn"](selected)))
        if cue_count == self.num_cue_tokens:
            return selected
        padding = selected.new_zeros(batch, self.num_cue_tokens - cue_count, dim)
        return torch.cat([selected, padding], dim=1)


def build_cue_selector(
    selector_type: str,
    dim: int,
    num_cue_tokens: int,
    layers: int,
    heads: int = 4,
    dropout: float = 0.1,
    tokenlearner_pooling: str = "sigmoid",
) -> nn.Module:
    normalized = selector_type.lower()
    if normalized in {"query_attention", "learned_query", "query"}:
        return LearnedCueTokenSelector(dim, num_cue_tokens, layers, heads, dropout)
    if normalized in {"tokenlearner", "soft_tokenlearner"}:
        return TokenLearnerCueTokenSelector(
            dim,
            num_cue_tokens,
            layers,
            heads,
            dropout,
            pooling=tokenlearner_pooling,
        )
    if normalized in {"topk_tokenlearner", "topk"}:
        return TopKTokenLearnerSelector(dim, num_cue_tokens, layers, heads, dropout)
    raise ValueError(f"Unknown selector_type: {selector_type}")


class CueMemoryBank(nn.Module):
    def __init__(self, dim: int, num_slots: int, ego_dim: int = 3, use_ego: bool = True) -> None:
        super().__init__()
        self.use_ego = use_ego
        self.initial_memory = nn.Parameter(torch.zeros(num_slots, dim))
        self.ego_mlp = nn.Sequential(nn.Linear(ego_dim, dim), nn.GELU(), nn.Linear(dim, dim)) if use_ego else None
        self.gru_cell = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, cues_over_time: torch.Tensor, ego_history: torch.Tensor) -> torch.Tensor:
        # cues_over_time: [B, T, K, D], ego_history: [B, T, 3]
        batch, time, num_slots, dim = cues_over_time.shape
        memory = self.initial_memory[None, :, :].expand(batch, -1, -1)
        for t in range(time):
            if self.use_ego:
                if self.ego_mlp is None:
                    raise RuntimeError("ego_mlp is required when use_ego=True")
                ego = self.ego_mlp(ego_history[:, t, :])[:, None, :]
            else:
                ego = cues_over_time.new_zeros(batch, 1, dim)
            update_input = cues_over_time[:, t, :, :] + ego
            flat_in = update_input.reshape(batch * num_slots, dim)
            flat_memory = memory.reshape(batch * num_slots, dim)
            memory = self.gru_cell(flat_in, flat_memory).reshape(batch, num_slots, dim)
            memory = self.norm(memory)
        return memory


class LastCueMemoryBank(nn.Module):
    """No learned memory update: expose only the latest selected cue tokens."""

    def forward(self, cues_over_time: torch.Tensor, ego_history: torch.Tensor) -> torch.Tensor:
        del ego_history
        return cues_over_time[:, -1, :, :]


class MeanCueMemoryBank(nn.Module):
    """No learned memory update: average selected cue tokens over time."""

    def forward(self, cues_over_time: torch.Tensor, ego_history: torch.Tensor) -> torch.Tensor:
        del ego_history
        return cues_over_time.mean(dim=1)


class StaticMemoryBank(nn.Module):
    """Trainable memory slots with cue writes disabled."""

    def __init__(self, dim: int, num_slots: int) -> None:
        super().__init__()
        self.initial_memory = nn.Parameter(torch.zeros(num_slots, dim))

    def forward(self, cues_over_time: torch.Tensor, ego_history: torch.Tensor) -> torch.Tensor:
        del ego_history
        batch = cues_over_time.shape[0]
        return self.initial_memory[None, :, :].expand(batch, -1, -1)


class AttentionCueMemoryBank(nn.Module):
    """Content-addressed cue memory with attention writes over memory slots."""

    def __init__(
        self,
        dim: int,
        num_slots: int,
        ego_dim: int = 3,
        dropout: float = 0.1,
        use_ego: bool = True,
    ) -> None:
        super().__init__()
        self.use_ego = use_ego
        self.initial_memory = nn.Parameter(torch.zeros(num_slots, dim))
        self.ego_mlp = nn.Sequential(nn.Linear(ego_dim, dim), nn.GELU(), nn.Linear(dim, dim)) if use_ego else None
        self.cue_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.write_gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.write_candidate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, cues_over_time: torch.Tensor, ego_history: torch.Tensor) -> torch.Tensor:
        # cues_over_time: [B, T, K, D], ego_history: [B, T, 3]
        batch, time, _num_cues, dim = cues_over_time.shape
        memory = self.initial_memory[None, :, :].expand(batch, -1, -1)
        for t in range(time):
            if self.use_ego:
                if self.ego_mlp is None:
                    raise RuntimeError("ego_mlp is required when use_ego=True")
                ego = self.ego_mlp(ego_history[:, t, :])[:, None, :]
            else:
                ego = cues_over_time.new_zeros(batch, 1, dim)
            cues = self.cue_projection(cues_over_time[:, t, :, :] + ego)
            scores = torch.matmul(cues, self.norm(memory).transpose(1, 2)) / math.sqrt(dim)
            write_weights = torch.softmax(scores, dim=-1)
            slot_context = torch.einsum("bcm,bcd->bmd", write_weights, cues)
            slot_context = slot_context / write_weights.sum(dim=1).clamp_min(1.0e-6)[..., None]
            write_input = torch.cat([memory, slot_context], dim=-1)
            gate = self.write_gate(write_input)
            candidate = self.write_candidate(write_input)
            memory = self.norm(memory + gate * (candidate - memory))
        return memory


def build_cue_memory_bank(
    memory_type: str,
    dim: int,
    num_slots: int,
    dropout: float = 0.1,
) -> nn.Module:
    normalized = memory_type.lower()
    if normalized in {"gru_cell", "gru", "recurrent"}:
        return CueMemoryBank(dim, num_slots)
    if normalized in {"gru_no_ego", "recurrent_no_ego"}:
        return CueMemoryBank(dim, num_slots, use_ego=False)
    if normalized in {"attention", "memory_network", "content_addressed"}:
        return AttentionCueMemoryBank(dim, num_slots, dropout=dropout)
    if normalized in {"attention_no_ego", "no_ego_attention", "content_addressed_no_ego"}:
        return AttentionCueMemoryBank(dim, num_slots, dropout=dropout, use_ego=False)
    if normalized in {"last_cue", "last", "no_memory"}:
        return LastCueMemoryBank()
    if normalized in {"mean_cue", "mean", "temporal_mean"}:
        return MeanCueMemoryBank()
    if normalized in {"no_memory_update", "static_memory", "frozen_memory_slots"}:
        return StaticMemoryBank(dim, num_slots)
    raise ValueError(f"Unknown memory_type: {memory_type}")


class PathQueryDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        future_steps: int,
        heads: int = 4,
        layers: int = 1,
        num_modes: int = 1,
        dropout: float = 0.1,
        zero_init_output: bool = False,
        shared_horizon_query: bool = False,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.shared_horizon_query = shared_horizon_query
        num_horizon_queries = 1 if shared_horizon_query else future_steps
        self.horizon_queries = nn.Parameter(torch.randn(num_horizon_queries, dim) * 0.02)
        self.mode_queries = (
            nn.Parameter(torch.randn(num_modes, dim) * 0.02) if num_modes > 1 else None
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                        "norm1": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        "dropout": nn.Dropout(dropout),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
                for _ in range(layers)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        self.mode_score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        if zero_init_output:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.mode_score_head[-1].weight)
        nn.init.zeros_(self.mode_score_head[-1].bias)

    def forward(self, memory: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        batch = memory.shape[0]
        horizon_queries = self.horizon_queries
        if self.shared_horizon_query:
            horizon_queries = horizon_queries.expand(self.future_steps, -1)
        if self.num_modes == 1:
            query = horizon_queries[None, :, :].expand(batch, -1, -1)
        else:
            query = horizon_queries[None, :, :] + self.mode_queries[:, None, :]
            query = query.reshape(self.num_modes * self.future_steps, -1)
            query = query[None, :, :].expand(batch, -1, -1)
        for layer in self.layers:
            attended, _weights = layer["attn"](query, memory, memory, need_weights=False)
            query = layer["norm1"](query + layer["dropout"](attended))
            query = layer["norm2"](query + layer["dropout"](layer["ffn"](query)))
        if self.num_modes == 1:
            return self.head(query)
        query = query.view(batch, self.num_modes, self.future_steps, -1)
        paths = self.head(query)
        logits = self.mode_score_head(query.mean(dim=2)).squeeze(-1)
        return {"paths": paths, "logits": logits}


class SingleVectorMLPDecoder(nn.Module):
    """Decode the full future path from one pooled memory vector."""

    def __init__(
        self,
        dim: int,
        future_steps: int,
        num_modes: int = 1,
        dropout: float = 0.1,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.path_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, future_steps * num_modes * 2),
        )
        self.mode_score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_modes))
        if zero_init_output:
            nn.init.zeros_(self.path_head[-1].weight)
            nn.init.zeros_(self.path_head[-1].bias)
        nn.init.zeros_(self.mode_score_head[-1].weight)
        nn.init.zeros_(self.mode_score_head[-1].bias)

    def forward(self, memory: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        pooled = memory.mean(dim=1)
        paths = self.path_head(pooled)
        if self.num_modes == 1:
            return paths.view(memory.shape[0], self.future_steps, 2)
        paths = paths.view(memory.shape[0], self.num_modes, self.future_steps, 2)
        return {"paths": paths, "logits": self.mode_score_head(pooled)}


class AutoregressivePathDecoder(nn.Module):
    """Decode future steps sequentially from previous predicted offsets."""

    def __init__(
        self,
        dim: int,
        future_steps: int,
        dropout: float = 0.1,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.context = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.previous_step = nn.Linear(2, dim)
        self.cell = nn.GRUCell(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        if zero_init_output:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[0]
        context = self.context(memory.mean(dim=1))
        hidden = context
        previous = memory.new_zeros(batch, 2)
        outputs = []
        for _step in range(self.future_steps):
            step_input = context + self.previous_step(previous)
            hidden = self.cell(self.dropout(step_input), hidden)
            previous = self.head(hidden)
            outputs.append(previous)
        return torch.stack(outputs, dim=1)


def build_path_decoder(
    decoder_type: str,
    dim: int,
    future_steps: int,
    heads: int = 4,
    layers: int = 1,
    num_modes: int = 1,
    dropout: float = 0.1,
    zero_init_output: bool = False,
) -> nn.Module:
    normalized = decoder_type.lower()
    if normalized in {"horizon_query", "horizon_query_decoder", "path_query"}:
        return PathQueryDecoder(
            dim,
            future_steps,
            heads=heads,
            layers=layers,
            num_modes=num_modes,
            dropout=dropout,
            zero_init_output=zero_init_output,
        )
    if normalized in {"shared_query", "shared_query_decoder"}:
        return PathQueryDecoder(
            dim,
            future_steps,
            heads=heads,
            layers=layers,
            num_modes=num_modes,
            dropout=dropout,
            zero_init_output=zero_init_output,
            shared_horizon_query=True,
        )
    if normalized in {"single_vector", "single_vector_mlp", "single_vector_direct", "mlp"}:
        return SingleVectorMLPDecoder(
            dim,
            future_steps,
            num_modes=num_modes,
            dropout=dropout,
            zero_init_output=zero_init_output,
        )
    if normalized in {"autoregressive", "autoregressive_decoder", "ar"}:
        if num_modes != 1:
            raise ValueError("autoregressive_decoder currently supports num_modes=1 only")
        return AutoregressivePathDecoder(
            dim,
            future_steps,
            dropout=dropout,
            zero_init_output=zero_init_output,
        )
    raise ValueError(f"Unknown decoder_type: {decoder_type}")


class TwoStreamEgocentricCueMemoryPathPredictor(nn.Module):
    def __init__(
        self,
        future_steps: int,
        backbone_name: str = "small_cnn",
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
        use_bottleneck_adapters: bool = True,
        adapter_bottleneck_dim: int = 64,
        temporal_type: str = "transformer",
        temporal_layers: int = 1,
        num_cue_tokens: int = 8,
        selector_layers: int = 1,
        selector_type: str = "query_attention",
        tokenlearner_pooling: str = "sigmoid",
        memory_type: str = "gru_cell",
        use_spatial_graph: bool = False,
        spatial_graph_neighbors: int = 8,
        spatial_relation_type: str | None = None,
        use_temporal_difference_conv: bool = False,
        use_temporal_shift: bool = False,
        decoder_layers: int = 1,
        decoder_type: str = "horizon_query_decoder",
        cue_temporal_layers: int = 1,
        dropout: float = 0.1,
        use_constant_velocity_residual: bool = True,
        residual_scale: float = 1.0,
        num_modes: int = 1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.decoder_type = decoder_type
        self.use_constant_velocity_residual = use_constant_velocity_residual
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)), persistent=True)
        self.visual_encoder = build_visual_encoder(backbone_name, hidden_dim, freeze_backbone)
        enc_dim = getattr(self.visual_encoder, "out_dim", hidden_dim)
        self.input_projection = nn.Linear(enc_dim, hidden_dim) if enc_dim != hidden_dim else nn.Identity()
        self.spatial_position = SpatialPositionalEncoding(hidden_dim)
        self.adapter = (
            BottleneckAdapter(hidden_dim, adapter_bottleneck_dim, dropout=dropout)
            if use_bottleneck_adapters
            else nn.Identity()
        )
        strnet_temporal = temporal_type.lower() == "strnet"
        if spatial_relation_type is None:
            spatial_relation_type = "topk_graph" if use_spatial_graph else "none"
        self.spatial_relation_type = spatial_relation_type
        self.spatial_graph = (
            build_spatial_relation_module(
                spatial_relation_type,
                hidden_dim,
                spatial_graph_neighbors,
                dropout=dropout,
            )
            if not strnet_temporal
            else nn.Identity()
        )
        self.temporal = TemporalAdapter(
            hidden_dim,
            temporal_type=temporal_type,
            layers=temporal_layers,
            dropout=dropout,
            use_difference_conv=use_temporal_difference_conv,
            use_temporal_shift=use_temporal_shift,
            spatial_graph_neighbors=spatial_graph_neighbors,
        )
        self.selector = build_cue_selector(
            selector_type,
            hidden_dim,
            num_cue_tokens,
            selector_layers,
            dropout=dropout,
            tokenlearner_pooling=tokenlearner_pooling,
        )
        self.cue_temporal = CueSpaceTimeTransformer(
            hidden_dim,
            num_cue_tokens,
            layers=cue_temporal_layers,
            dropout=dropout,
        )
        self.memory = build_cue_memory_bank(memory_type, hidden_dim, num_cue_tokens, dropout=dropout)
        self.decoder = build_path_decoder(
            decoder_type,
            hidden_dim,
            future_steps,
            layers=decoder_layers,
            num_modes=num_modes,
            dropout=dropout,
            zero_init_output=use_constant_velocity_residual,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if "visual_tokens" in batch:
            tokens = batch["visual_tokens"].float()
        elif isinstance(self.visual_encoder, ZeroVisualTokenEncoder) and "rgb_history" not in batch:
            ego_history = batch["ego_history"]
            batch_size, time = ego_history.shape[:2]
            tokens = ego_history.new_zeros(
                (batch_size, time, self.visual_encoder.token_count, self.visual_encoder.out_dim)
            )
        else:
            tokens = self.visual_encoder(batch["rgb_history"])
        tokens = self.input_projection(tokens)
        tokens = self.spatial_position(tokens)
        tokens = self.adapter(tokens)
        tokens = self.spatial_graph(tokens)
        tokens = self.temporal(tokens)
        cues = []
        for t in range(tokens.shape[1]):
            cues.append(self.selector(tokens[:, t, :, :]))
        cues_over_time = torch.stack(cues, dim=1)
        cues_over_time = self.cue_temporal(cues_over_time)
        memory = self.memory(cues_over_time, batch["ego_history"])
        decoded = self.decoder(memory)
        if not self.use_constant_velocity_residual:
            return decoded
        base = constant_velocity_path(batch["ego_history"], self.future_steps)
        if isinstance(decoded, dict):
            paths = base[:, None, :, :] + decoded["paths"] * self.residual_scale.to(decoded["paths"].dtype)
            return {"paths": paths, "logits": decoded["logits"]}
        return base + decoded * self.residual_scale.to(decoded.dtype)
