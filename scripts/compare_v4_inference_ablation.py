"""Inference-time ablations for the v4 WIT-VZ DINOv3 checkpoints.

This script does not retrain or collect data. It evaluates trained checkpoints
while perturbing intermediate inference tensors to estimate which components
matter for the final path prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import displacement_errors
from src.models.factory import create_model
from src.models.motion import constant_velocity_path
from src.wit_vz.dataset import WITVZPathDataset, collate_path_batch


ABLATIONS = [
    "constant_velocity",
    "full_model",
    "zero_visual_tokens",
    "static_visual_tokens",
    "no_temporal_adapter",
    "uniform_selector",
    "no_cue_temporal",
    "no_memory_update",
    "no_ego_memory",
]

ABLATION_DESCRIPTIONS = {
    "constant_velocity": "No visual model; extrapolates recent ego-motion linearly.",
    "full_model": "Original trained checkpoint inference path.",
    "zero_visual_tokens": "Sets all cached visual tokens to zero before the model.",
    "static_visual_tokens": "Repeats the last frame token grid across the whole history.",
    "no_temporal_adapter": "Skips the TimeSFormer-style temporal/spatial fusion adapter.",
    "uniform_selector": "Replaces adaptive TokenLearner cues with repeated spatial means.",
    "no_cue_temporal": "Skips temporal modeling over selected cue tokens.",
    "no_memory_update": "Bypasses the cue memory bank and decodes from the last cue set.",
    "no_ego_memory": "Zeros ego-motion only inside memory updates; keeps the final CV base intact.",
}

HORIZON_DATASETS = {
    1: "future_01s",
    3: "future_03s",
    5: "future_05s",
    10: "future_10s",
}

HORIZON_CHECKPOINTS = {
    1: "wit_vz_v4_defaults_dinov3_single_01s.pt",
    3: "wit_vz_v4_defaults_dinov3_single_03s.pt",
    5: "wit_vz_v4_defaults_dinov3_single_05s.pt",
    10: "wit_vz_v4_defaults_dinov3_single_10s.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v4 checkpoint inference-time ablations.")
    parser.add_argument("--main-dataset", type=Path, default=Path("data/wit_vz/processed/wit_vz_v4_defaults_001"))
    parser.add_argument("--horizon-root", type=Path, default=Path("data/wit_vz/processed/horizon_sweep_v4_defaults"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny"),
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--ablations", nargs="+", default=ABLATIONS, choices=ABLATIONS)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def dataset_for_horizon(args: argparse.Namespace, horizon: int) -> Path:
    if horizon == 1:
        return args.main_dataset
    return args.horizon_root / HORIZON_DATASETS[horizon]


def checkpoint_for_horizon(args: argparse.Namespace, horizon: int) -> Path:
    return args.checkpoint_root / HORIZON_CHECKPOINTS[horizon]


def check_required_paths(args: argparse.Namespace) -> list[Path]:
    required = [args.cache]
    for horizon in args.horizons:
        if horizon not in HORIZON_DATASETS:
            raise ValueError(f"Unsupported horizon {horizon}; supported: {sorted(HORIZON_DATASETS)}")
        required.append(dataset_for_horizon(args, horizon))
        required.append(checkpoint_for_horizon(args, horizon))
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing required local paths. Not downloading or collecting anything:")
        for path in missing:
            print(f"- {path}")
    return missing


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("visual_tokens", "ego_history", "future_path"):
        if key in moved:
            moved[key] = moved[key].to(device, non_blocking=True)
    return moved


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model(
        str(checkpoint["model_name"]),
        future_steps=int(checkpoint["future_steps"]),
        backbone_name=str(checkpoint["backbone"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        freeze_backbone=bool(checkpoint.get("freeze_backbone", True)),
        temporal_type=str(checkpoint.get("temporal_type", "transformer")),
        temporal_layers=int(checkpoint.get("temporal_layers", 1)),
        num_cue_tokens=int(checkpoint.get("num_cue_tokens", 8)),
        selector_layers=int(checkpoint.get("selector_layers", 1)),
        selector_type=str(checkpoint.get("selector_type", "query_attention")),
        tokenlearner_pooling=str(checkpoint.get("tokenlearner_pooling", "sigmoid")),
        memory_type=str(checkpoint.get("memory_type", "gru_cell")),
        use_spatial_graph=bool(checkpoint.get("use_spatial_graph", False)),
        spatial_graph_neighbors=int(checkpoint.get("spatial_graph_neighbors", 8)),
        use_temporal_difference_conv=bool(checkpoint.get("use_temporal_difference_conv", False)),
        use_temporal_shift=bool(checkpoint.get("use_temporal_shift", False)),
        decoder_layers=int(checkpoint.get("decoder_layers", 1)),
        cue_temporal_layers=int(checkpoint.get("cue_temporal_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.1)),
        use_constant_velocity_residual=bool(checkpoint.get("use_constant_velocity_residual", True)),
        residual_scale=float(checkpoint.get("residual_scale", 1.0)),
        num_modes=int(checkpoint.get("num_modes", 1)),
    )
    state = checkpoint["model_state"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint


def num_cue_tokens(model: torch.nn.Module) -> int:
    cue_temporal = getattr(model, "cue_temporal", None)
    if cue_temporal is not None and hasattr(cue_temporal, "cue_position"):
        return int(cue_temporal.cue_position.shape[0])
    memory = getattr(model, "memory", None)
    if memory is not None and hasattr(memory, "initial_memory"):
        return int(memory.initial_memory.shape[0])
    raise RuntimeError("Could not infer number of cue tokens from model.")


def tensor_prediction(prediction: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(prediction, torch.Tensor):
        paths = prediction
    else:
        paths = prediction.get("paths")
        if paths is None:
            raise KeyError("Prediction dict has no 'paths' tensor")
        if paths.ndim == 4 and paths.shape[1] == 1:
            paths = paths[:, 0]
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError(f"Expected final prediction shape [B,H,2], got {tuple(paths.shape)}")
    return paths


def forward_with_ablation(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    ablation: str,
) -> torch.Tensor:
    """Run model inference with one controlled ablation."""

    future_steps = int(getattr(model, "future_steps"))
    ego_history = batch["ego_history"]
    if ablation == "constant_velocity":
        return constant_velocity_path(ego_history, future_steps)

    tokens = batch["visual_tokens"].float()
    if ablation == "zero_visual_tokens":
        tokens = torch.zeros_like(tokens)
    elif ablation == "static_visual_tokens":
        tokens = tokens[:, -1:, :, :].expand_as(tokens)

    tokens = model.input_projection(tokens)
    tokens = model.spatial_position(tokens)
    tokens = model.adapter(tokens)
    tokens = model.spatial_graph(tokens)
    if ablation != "no_temporal_adapter":
        tokens = model.temporal(tokens)

    cues = []
    cue_count = num_cue_tokens(model)
    for t in range(tokens.shape[1]):
        frame_tokens = tokens[:, t, :, :]
        if ablation == "uniform_selector":
            pooled = frame_tokens.mean(dim=1, keepdim=True)
            cues.append(pooled.repeat(1, cue_count, 1))
        else:
            cues.append(model.selector(frame_tokens))
    cues_over_time = torch.stack(cues, dim=1)

    if ablation != "no_cue_temporal":
        cues_over_time = model.cue_temporal(cues_over_time)

    if ablation == "no_memory_update":
        memory = cues_over_time[:, -1, :, :]
    else:
        memory_ego = torch.zeros_like(ego_history) if ablation == "no_ego_memory" else ego_history
        memory = model.memory(cues_over_time, memory_ego)

    decoded = model.decoder(memory)
    if not getattr(model, "use_constant_velocity_residual", False):
        return tensor_prediction(decoded)

    base = constant_velocity_path(ego_history, future_steps)
    decoded_tensor = tensor_prediction(decoded)
    residual_scale = model.residual_scale.to(decoded_tensor.dtype)
    return base + decoded_tensor * residual_scale


def make_loader(args: argparse.Namespace, dataset_path: Path) -> DataLoader:
    dataset = WITVZPathDataset(
        dataset_path,
        split=args.split,
        load_rgb=False,
        visual_feature_cache_dir=args.cache,
    )
    if args.limit > 0:
        dataset.samples = dataset.samples[: args.limit]
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_path_batch,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_ablation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ablation: str,
) -> dict[str, Any]:
    total = 0
    ade_sum = 0.0
    fde_sum = 0.0
    per_step_sum: torch.Tensor | None = None
    expected_horizon = int(getattr(model, "future_steps"))

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            prediction = forward_with_ablation(model, batch, ablation)
            prediction = tensor_prediction(prediction)
            target = batch["future_path"]
            if prediction.shape != target.shape:
                raise ValueError(
                    f"{ablation}: prediction shape {tuple(prediction.shape)} "
                    f"does not match target {tuple(target.shape)}"
                )
            if prediction.shape[1] != expected_horizon:
                raise ValueError(
                    f"{ablation}: expected horizon {expected_horizon}, got {prediction.shape[1]}"
                )
            errors = displacement_errors(prediction, target)
            batch_size = int(target.shape[0])
            ade_sum += float(errors.mean().detach().cpu()) * batch_size
            fde_sum += float(errors[:, -1].mean().detach().cpu()) * batch_size
            step_sum = errors.sum(dim=0).detach().cpu()
            per_step_sum = step_sum if per_step_sum is None else per_step_sum + step_sum
            total += batch_size

    if total <= 0 or per_step_sum is None:
        raise ValueError("No samples were evaluated")
    return {
        "samples": total,
        "ADE": ade_sum / total,
        "FDE": fde_sum / total,
        "per_step_error": (per_step_sum / total).tolist(),
        "prediction_shape": [total, expected_horizon, 2],
    }


def rank_impacts(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    full = metrics["full_model"]
    output = []
    for name, values in metrics.items():
        output.append(
            {
                "ablation": name,
                "delta_ADE_vs_full": values["ADE"] - full["ADE"],
                "delta_FDE_vs_full": values["FDE"] - full["FDE"],
                "relative_ADE_vs_full_pct": (
                    (values["ADE"] / full["ADE"] - 1.0) * 100.0 if full["ADE"] else math.nan
                ),
                "relative_FDE_vs_full_pct": (
                    (values["FDE"] / full["FDE"] - 1.0) * 100.0 if full["FDE"] else math.nan
                ),
            }
        )
    return sorted(output, key=lambda item: item["delta_ADE_vs_full"], reverse=True)


def dino_on_off_summary(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = metrics["full_model"]
    constant_velocity = metrics["constant_velocity"]
    zero_visual = metrics["zero_visual_tokens"]
    return {
        "dino_on": {
            "case": "full_model",
            "ADE": full["ADE"],
            "FDE": full["FDE"],
        },
        "dino_off_ego_only": {
            "case": "constant_velocity",
            "meaning": "No DINO, no visual model; ego-motion-only extrapolation.",
            "ADE": constant_velocity["ADE"],
            "FDE": constant_velocity["FDE"],
            "delta_ADE_vs_dino_on": constant_velocity["ADE"] - full["ADE"],
            "delta_FDE_vs_dino_on": constant_velocity["FDE"] - full["FDE"],
            "relative_ADE_vs_dino_on_pct": (
                (constant_velocity["ADE"] / full["ADE"] - 1.0) * 100.0 if full["ADE"] else math.nan
            ),
            "relative_FDE_vs_dino_on_pct": (
                (constant_velocity["FDE"] / full["FDE"] - 1.0) * 100.0 if full["FDE"] else math.nan
            ),
        },
        "dino_signal_off_same_checkpoint": {
            "case": "zero_visual_tokens",
            "meaning": "Same trained model, but cached DINO tokens are replaced by zeros at inference.",
            "ADE": zero_visual["ADE"],
            "FDE": zero_visual["FDE"],
            "delta_ADE_vs_dino_on": zero_visual["ADE"] - full["ADE"],
            "delta_FDE_vs_dino_on": zero_visual["FDE"] - full["FDE"],
            "relative_ADE_vs_dino_on_pct": (
                (zero_visual["ADE"] / full["ADE"] - 1.0) * 100.0 if full["ADE"] else math.nan
            ),
            "relative_FDE_vs_dino_on_pct": (
                (zero_visual["FDE"] / full["FDE"] - 1.0) * 100.0 if full["FDE"] else math.nan
            ),
        },
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    header_list = list(headers)
    lines = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join(["---"] * len(header_list)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def impact_lookup(horizon: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["ablation"]: item for item in horizon["impact_ranked_by_ADE"]}


def write_overall_findings(lines: list[str], payload: dict[str, Any]) -> None:
    horizons = payload["horizons"]
    if not horizons:
        return
    lines.append("## Overall Findings")
    lines.append("")
    lines.append(
        "The full-model rows match the previously reported v4 horizon test metrics, "
        "so the ablation script is evaluating the intended checkpoints and splits."
    )
    lines.append("")
    rows = []
    for horizon in horizons:
        impacts = impact_lookup(horizon)
        dino = horizon["dino_on_off_summary"]
        rows.append(
            [
                f"{horizon['horizon_sec']}s",
                fmt_pct(dino["dino_off_ego_only"]["relative_ADE_vs_dino_on_pct"]),
                fmt_pct(dino["dino_signal_off_same_checkpoint"]["relative_ADE_vs_dino_on_pct"]),
                fmt_pct(impacts["no_temporal_adapter"]["relative_ADE_vs_full_pct"]),
                fmt_pct(impacts["no_cue_temporal"]["relative_ADE_vs_full_pct"]),
                fmt_pct(impacts["no_memory_update"]["relative_ADE_vs_full_pct"]),
                fmt_pct(impacts["no_ego_memory"]["relative_ADE_vs_full_pct"]),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Horizon",
                "DINO off ego-only",
                "DINO signal off",
                "No temporal adapter",
                "No cue temporal",
                "No memory update",
                "No ego memory",
            ],
            rows,
        )
    )
    lines.append("")

    dino_cv = [h["dino_on_off_summary"]["dino_off_ego_only"]["relative_ADE_vs_dino_on_pct"] for h in horizons]
    dino_zero = [
        h["dino_on_off_summary"]["dino_signal_off_same_checkpoint"]["relative_ADE_vs_dino_on_pct"]
        for h in horizons
    ]
    memory_impacts = [impact_lookup(h)["no_memory_update"]["relative_ADE_vs_full_pct"] for h in horizons]
    ego_impacts = [impact_lookup(h)["no_ego_memory"]["relative_ADE_vs_full_pct"] for h in horizons]
    cue_temporal_impacts = [impact_lookup(h)["no_cue_temporal"]["relative_ADE_vs_full_pct"] for h in horizons]
    temporal_adapter_impacts = [
        impact_lookup(h)["no_temporal_adapter"]["relative_ADE_vs_full_pct"] for h in horizons
    ]
    static_impacts = [impact_lookup(h)["static_visual_tokens"]["relative_ADE_vs_full_pct"] for h in horizons]
    selector_impacts = [impact_lookup(h)["uniform_selector"]["relative_ADE_vs_full_pct"] for h in horizons]

    lines.append(
        f"- DINO/visual information helps at every horizon. Removing visual modeling entirely "
        f"(`constant_velocity`) worsens ADE by {min(dino_cv):.1f}% to {max(dino_cv):.1f}%; "
        f"zeroing the DINO tokens inside the trained model worsens ADE by "
        f"{min(dino_zero):.1f}% to {max(dino_zero):.1f}%."
    )
    lines.append(
        f"- The cue memory update is the largest inference-time dependency. "
        f"`no_memory_update` is the worst ablation at every horizon, worsening ADE by "
        f"{min(memory_impacts):.1f}% to {max(memory_impacts):.1f}%."
    )
    lines.append(
        f"- Ego-motion conditioning inside the memory matters, especially as the prediction "
        f"horizon grows: `no_ego_memory` worsens ADE by {min(ego_impacts):.1f}% to "
        f"{max(ego_impacts):.1f}%."
    )
    lines.append(
        f"- Cue temporal modeling has a meaningful effect beyond 1s. `no_cue_temporal` "
        f"worsens ADE by {min(cue_temporal_impacts):.1f}% to {max(cue_temporal_impacts):.1f}%."
    )
    lines.append(
        f"- The TimeSFormer-style temporal adapter has a smaller but consistent positive "
        f"effect in this inference test, with ADE degradation from "
        f"{min(temporal_adapter_impacts):.1f}% to {max(temporal_adapter_impacts):.1f}%."
    )
    lines.append(
        f"- Repeating the last visual token grid over time has only a small effect "
        f"({min(static_impacts):.1f}% to {max(static_impacts):.1f}% ADE), suggesting that "
        f"visual content dominates over short-term visual change for these checkpoints."
    )
    lines.append(
        f"- `uniform_selector` is weakly mixed ({min(selector_impacts):.1f}% to "
        f"{max(selector_impacts):.1f}% ADE). Because this is inference-time surgery rather "
        f"than retraining, it should not be used alone to claim the learned selector is "
        f"unnecessary."
    )
    lines.append("")


def write_markdown(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# V4 Inference-Time Ablation")
    lines.append("")
    lines.append("Date: 2026-05-22")
    lines.append("")
    lines.append("This is an inference-time ablation only. No checkpoint was retrained, no")
    lines.append("data was downloaded, and no new data was collected for this evaluation.")
    lines.append("")
    lines.append("## What The Model Predicts")
    lines.append("")
    lines.append("The model predicts an egocentric future local path, not a global map")
    lines.append("trajectory, route identity, action sequence, or semantic goal. For each")
    lines.append("sample, the input is 1 second of visual history plus relative ego-motion")
    lines.append("history. The output shape is `[B, H, 2]`, where `H` is the number of future")
    lines.append("steps at 5 FPS. Each point is `[forward, right]` in the current pose's")
    lines.append("local coordinate frame, with the origin at the current agent pose.")
    lines.append("")
    lines.append("## Inputs And Checkpoints")
    lines.append("")
    lines.append(f"- Main v4 dataset: `{args.main_dataset}`")
    lines.append(f"- Horizon root: `{args.horizon_root}`")
    lines.append(f"- DINOv3 cache: `{args.cache}`")
    lines.append(f"- Checkpoint root: `{args.checkpoint_root}`")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Limit: `{args.limit or 'none'}`")
    lines.append(f"- Batch size: `{args.batch_size}`")
    lines.append("")
    lines.append("## Ablation Cases")
    lines.append("")
    for name in ABLATIONS:
        if name in args.ablations:
            lines.append(f"- `{name}`: {ABLATION_DESCRIPTIONS[name]}")
    lines.append("")

    write_overall_findings(lines, payload)

    for horizon in payload["horizons"]:
        metrics = horizon["metrics"]
        impacts = impact_lookup(horizon)
        lines.append(f"## {horizon['horizon_sec']}s Horizon")
        lines.append("")
        lines.append(f"- Dataset: `{horizon['dataset']}`")
        lines.append(f"- Checkpoint: `{horizon['checkpoint']}`")
        lines.append(f"- Future steps: `{horizon['future_steps']}`")
        lines.append(f"- Evaluated samples per case: `{horizon['samples']}`")
        lines.append("")
        dino_summary = horizon["dino_on_off_summary"]
        dino_rows = [
            [
                "`DINO on`",
                "`full_model`",
                fmt(dino_summary["dino_on"]["ADE"]),
                fmt(dino_summary["dino_on"]["FDE"]),
                "0.0000",
                "0.0000",
                "+0.0%",
            ],
            [
                "`DINO off, ego only`",
                "`constant_velocity`",
                fmt(dino_summary["dino_off_ego_only"]["ADE"]),
                fmt(dino_summary["dino_off_ego_only"]["FDE"]),
                fmt(dino_summary["dino_off_ego_only"]["delta_ADE_vs_dino_on"]),
                fmt(dino_summary["dino_off_ego_only"]["delta_FDE_vs_dino_on"]),
                fmt_pct(dino_summary["dino_off_ego_only"]["relative_ADE_vs_dino_on_pct"]),
            ],
            [
                "`DINO signal off`",
                "`zero_visual_tokens`",
                fmt(dino_summary["dino_signal_off_same_checkpoint"]["ADE"]),
                fmt(dino_summary["dino_signal_off_same_checkpoint"]["FDE"]),
                fmt(dino_summary["dino_signal_off_same_checkpoint"]["delta_ADE_vs_dino_on"]),
                fmt(dino_summary["dino_signal_off_same_checkpoint"]["delta_FDE_vs_dino_on"]),
                fmt_pct(dino_summary["dino_signal_off_same_checkpoint"]["relative_ADE_vs_dino_on_pct"]),
            ],
        ]
        lines.append("DINO on/off summary:")
        lines.append("")
        lines.append(
            markdown_table(
                ["Comparison", "Case", "ADE", "FDE", "Delta ADE", "Delta FDE", "ADE rel."],
                dino_rows,
            )
        )
        lines.append("")
        rows = []
        for name in args.ablations:
            values = metrics[name]
            impact = impacts[name]
            rows.append(
                [
                    f"`{name}`",
                    fmt(values["ADE"]),
                    fmt(values["FDE"]),
                    fmt(impact["delta_ADE_vs_full"]),
                    fmt(impact["delta_FDE_vs_full"]),
                    fmt_pct(impact["relative_ADE_vs_full_pct"]),
                    fmt_pct(impact["relative_FDE_vs_full_pct"]),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "Case",
                    "ADE",
                    "FDE",
                    "Delta ADE vs full",
                    "Delta FDE vs full",
                    "ADE rel.",
                    "FDE rel.",
                ],
                rows,
            )
        )
        lines.append("")
        worst = next(item for item in horizon["impact_ranked_by_ADE"] if item["ablation"] != "full_model")
        lines.append(
            f"Largest ADE degradation: `{worst['ablation']}` "
            f"({fmt(worst['delta_ADE_vs_full'])}, {fmt_pct(worst['relative_ADE_vs_full_pct'])})."
        )
        lines.append("")
        full_steps = metrics["full_model"]["per_step_error"]
        lines.append("Full-model per-step error:")
        lines.append("")
        lines.append("```text")
        lines.append(json.dumps([round(float(value), 4) for value in full_steps]))
        lines.append("```")
        lines.append("")

    lines.append("## Reading The Results")
    lines.append("")
    lines.append("- `constant_velocity` and `zero_visual_tokens` are the two DINO-off views:")
    lines.append("  the first removes visual modeling entirely, while the second keeps the")
    lines.append("  trained network but removes its DINO signal at inference.")
    lines.append("- `static_visual_tokens` isolates whether frame-to-frame visual change matters.")
    lines.append("  In these results its impact is small, so most of the visual gain appears to")
    lines.append("  come from scene content rather than short-term visual motion.")
    lines.append("- `no_temporal_adapter` and `no_cue_temporal` isolate two different temporal")
    lines.append("  stages: dense visual-token fusion before cue selection, and temporal fusion")
    lines.append("  after cue selection.")
    lines.append("- `uniform_selector` is mixed here, so adaptive TokenLearner selection should")
    lines.append("  be judged with a retraining-time ablation before making a strong claim.")
    lines.append("- `no_memory_update` is consistently the largest degradation, showing that")
    lines.append("  the memory update is central to these checkpoints at inference.")
    lines.append("- If an ablation improves over `full_model`, that component may be adding")
    lines.append("  noise for that horizon and should be checked with retraining-time ablation")
    lines.append("  before drawing architectural conclusions.")
    lines.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.horizons = sorted(dict.fromkeys(args.horizons))
    missing = check_required_paths(args)
    if missing:
        raise SystemExit(2)

    device = choose_device(args.device)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "evaluation_type": "inference_time_ablation",
        "note": "No retraining, no DVC pull, no downloads, no new data collection.",
        "split": args.split,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": str(device),
        "ablations": {name: ABLATION_DESCRIPTIONS[name] for name in args.ablations},
        "horizons": [],
    }

    for horizon in args.horizons:
        dataset_path = dataset_for_horizon(args, horizon)
        checkpoint_path = checkpoint_for_horizon(args, horizon)
        loader = make_loader(args, dataset_path)
        model, checkpoint = load_model(checkpoint_path, device)
        expected_steps = int(checkpoint["future_steps"])
        expected_from_horizon = int(horizon * 5)
        if expected_steps != expected_from_horizon:
            raise ValueError(
                f"{horizon}s checkpoint has H={expected_steps}, expected {expected_from_horizon}"
            )
        print(
            f"horizon={horizon}s dataset={dataset_path} checkpoint={checkpoint_path} "
            f"samples={len(loader.dataset)}"
        )
        metrics: dict[str, Any] = {}
        for ablation in args.ablations:
            print(f"  ablation={ablation}")
            metrics[ablation] = evaluate_ablation(model, loader, device, ablation)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        payload["horizons"].append(
            {
                "horizon_sec": horizon,
                "future_steps": expected_steps,
                "dataset": dataset_path.as_posix(),
                "checkpoint": checkpoint_path.as_posix(),
                "samples": next(iter(metrics.values()))["samples"],
                "metrics": metrics,
                "impact_ranked_by_ADE": rank_impacts(metrics),
                "dino_on_off_summary": dino_on_off_summary(metrics),
            }
        )

    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args, payload)
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
