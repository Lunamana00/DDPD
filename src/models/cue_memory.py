"""Two-stream egocentric cue-memory path predictor."""

from __future__ import annotations

import math

import torch
from torch import nn

from .backbones import build_visual_encoder
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


class TemporalAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        temporal_type: str = "transformer",
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temporal_type = temporal_type
        self.position = TemporalPositionalEncoding(dim)
        if temporal_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.module = nn.TransformerEncoder(layer, num_layers=layers)
        elif temporal_type == "gru":
            self.module = nn.GRU(dim, dim, num_layers=layers, batch_first=True)
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        batch, time, num_tokens, dim = tokens.shape
        positioned = self.position(tokens)
        per_spatial_token = positioned.permute(0, 2, 1, 3).reshape(batch * num_tokens, time, dim)
        if self.temporal_type == "gru":
            temporal, _h = self.module(per_spatial_token)
        else:
            temporal = self.module(per_spatial_token)
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


class CueMemoryBank(nn.Module):
    def __init__(self, dim: int, num_slots: int, ego_dim: int = 3) -> None:
        super().__init__()
        self.initial_memory = nn.Parameter(torch.zeros(num_slots, dim))
        self.ego_mlp = nn.Sequential(nn.Linear(ego_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gru_cell = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, cues_over_time: torch.Tensor, ego_history: torch.Tensor) -> torch.Tensor:
        # cues_over_time: [B, T, K, D], ego_history: [B, T, 3]
        batch, time, num_slots, dim = cues_over_time.shape
        memory = self.initial_memory[None, :, :].expand(batch, -1, -1)
        for t in range(time):
            ego = self.ego_mlp(ego_history[:, t, :])[:, None, :]
            update_input = cues_over_time[:, t, :, :] + ego
            flat_in = update_input.reshape(batch * num_slots, dim)
            flat_memory = memory.reshape(batch * num_slots, dim)
            memory = self.gru_cell(flat_in, flat_memory).reshape(batch, num_slots, dim)
            memory = self.norm(memory)
        return memory


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
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.horizon_queries = nn.Parameter(torch.randn(future_steps, dim) * 0.02)
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
        if self.num_modes == 1:
            query = self.horizon_queries[None, :, :].expand(batch, -1, -1)
        else:
            query = self.horizon_queries[None, :, :] + self.mode_queries[:, None, :]
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
        decoder_layers: int = 1,
        dropout: float = 0.1,
        use_constant_velocity_residual: bool = True,
        residual_scale: float = 1.0,
        num_modes: int = 1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
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
        self.temporal = TemporalAdapter(
            hidden_dim,
            temporal_type=temporal_type,
            layers=temporal_layers,
            dropout=dropout,
        )
        self.selector = LearnedCueTokenSelector(
            hidden_dim,
            num_cue_tokens,
            selector_layers,
            dropout=dropout,
        )
        self.memory = CueMemoryBank(hidden_dim, num_cue_tokens)
        self.decoder = PathQueryDecoder(
            hidden_dim,
            future_steps,
            layers=decoder_layers,
            num_modes=num_modes,
            dropout=dropout,
            zero_init_output=use_constant_velocity_residual,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        tokens = self.visual_encoder(batch["rgb_history"])
        tokens = self.input_projection(tokens)
        tokens = self.spatial_position(tokens)
        tokens = self.adapter(tokens)
        tokens = self.temporal(tokens)
        cues = []
        for t in range(tokens.shape[1]):
            cues.append(self.selector(tokens[:, t, :, :]))
        cues_over_time = torch.stack(cues, dim=1)
        memory = self.memory(cues_over_time, batch["ego_history"])
        decoded = self.decoder(memory)
        if not self.use_constant_velocity_residual:
            return decoded
        base = constant_velocity_path(batch["ego_history"], self.future_steps)
        if isinstance(decoded, dict):
            paths = base[:, None, :, :] + decoded["paths"] * self.residual_scale.to(decoded["paths"].dtype)
            return {"paths": paths, "logits": decoded["logits"]}
        return base + decoded * self.residual_scale.to(decoded.dtype)
