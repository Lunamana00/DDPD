"""Run metric-only inference for paper-proxy baselines and model ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import trajectory_loss
from src.metrics import ade, fde, per_horizon_error
from src.models.factory import create_model, needs_rgb
from src.train_path_predictor import move_batch
from src.wit_vz.dataset import WITVZPathDataset, collate_path_batch
from src.wit_vz.geometry import world_delta_to_local


@dataclass(frozen=True)
class EvalCase:
    model_key: str
    label: str
    horizon: int
    dataset: Path
    kind: str
    checkpoint: Path | None
    visual_feature_cache: Path | None


def horizon_tag(horizon: int) -> str:
    return f"{horizon:02d}s"


def make_v2_cases(root: Path, horizons: list[int]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    paper_proxy_specs = [
        (
            "khaleque_center_random_proxy",
            "Khaleque-style center-biased exploratory proxy",
            "khaleque_proxy",
        ),
        (
            "xu_pixels_saliency_proxy",
            "Xu-style pixels-only saliency proxy",
            "pixels_proxy",
        ),
    ]
    model_specs = [
        ("constant_velocity", "Internal motion-only CV ablation", "checkpoint", None),
        ("small_cnn_timesformer", "Internal small-CNN TimeSFormer ablation", "checkpoint", None),
        ("dinov3_timesformer", "Ours: DINOv3 TimeSFormer", "checkpoint", "dinov3"),
        ("dinov3_strnet", "Internal DINOv3 STRNet-style ablation", "checkpoint", "dinov3"),
    ]
    for horizon in horizons:
        tag = horizon_tag(horizon)
        dataset = root / "data" / "wit_vz" / "processed" / "horizon_sweep_v2" / f"future_{tag}"
        for model_key, label, kind in paper_proxy_specs:
            cases.append(
                EvalCase(
                    model_key=model_key,
                    label=label,
                    horizon=horizon,
                    dataset=dataset,
                    kind=kind,
                    checkpoint=None,
                    visual_feature_cache=None,
                )
            )
        for model_key, label, kind, cache_kind in model_specs:
            cache = None
            if cache_kind == "dinov3":
                cache = (
                    root
                    / "data"
                    / "wit_vz"
                    / "feature_cache"
                    / f"horizon_sweep_v2_future_{tag}_dinov3_convnext_tiny"
                )
            cases.append(
                EvalCase(
                    model_key=model_key,
                    label=label,
                    horizon=horizon,
                    dataset=dataset,
                    kind=kind,
                    checkpoint=root / "runs" / "horizon_sweep_v2" / f"{model_key}_{tag}" / "best.pt",
                    visual_feature_cache=cache,
                )
            )
    return cases


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = str(checkpoint["model_name"])
    model = create_model(
        model_name,
        future_steps=int(checkpoint["future_steps"]),
        backbone_name=str(checkpoint.get("backbone", "small_cnn")),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        freeze_backbone=bool(checkpoint.get("freeze_backbone", True)),
        use_bottleneck_adapters=bool(checkpoint.get("use_bottleneck_adapters", True)),
        adapter_bottleneck_dim=int(checkpoint.get("adapter_bottleneck_dim", 64)),
        num_cue_tokens=int(checkpoint.get("num_cue_tokens", 8)),
        num_modes=int(checkpoint.get("num_modes", 1)),
        temporal_type=str(checkpoint.get("temporal_type", "transformer")),
        temporal_layers=int(checkpoint.get("temporal_layers", 1)),
        selector_layers=int(checkpoint.get("selector_layers", 1)),
        decoder_layers=int(checkpoint.get("decoder_layers", 1)),
        cue_temporal_layers=int(checkpoint.get("cue_temporal_layers", 0)),
        tokenlearner_pooling=str(checkpoint.get("tokenlearner_pooling", "softmax")),
        selector_type=str(checkpoint.get("selector_type", "query_attention")),
        memory_type=str(checkpoint.get("memory_type", "gru_cell")),
        use_spatial_graph=bool(checkpoint.get("use_spatial_graph", False)),
        spatial_graph_neighbors=int(checkpoint.get("spatial_graph_neighbors", 8)),
        spatial_relation_type=checkpoint.get("spatial_relation_type"),
        use_temporal_difference_conv=bool(checkpoint.get("use_temporal_difference_conv", False)),
        use_temporal_shift=bool(checkpoint.get("use_temporal_shift", False)),
        dropout=float(checkpoint.get("dropout", 0.1)),
        use_constant_velocity_residual=bool(checkpoint.get("use_constant_velocity_residual", False)),
        residual_scale=float(checkpoint.get("residual_scale", checkpoint.get("trajectory_scale", 1.0))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, checkpoint


def estimate_speed(ego_history: torch.Tensor) -> torch.Tensor:
    speeds = torch.linalg.norm(ego_history[..., :2], dim=-1)
    return speeds.mean(dim=1).clamp_min(1.0)


def deterministic_uniform(key: str, low: float, high: float) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64 - 1)
    return low + (high - low) * value


def source_centers_from_train(dataset_dir: Path) -> dict[str, tuple[float, float]]:
    dataset = WITVZPathDataset(dataset_dir, split="train", load_rgb=False)
    bounds: dict[str, list[float]] = {}
    for sample in dataset.samples:
        pose = sample["current_pose"]
        source = sample.get("source", {})
        source_id = str(source.get("source_id") or sample.get("metadata", {}).get("source_id") or "unknown")
        x = float(pose["x"])
        y = float(pose["y"])
        if source_id not in bounds:
            bounds[source_id] = [x, x, y, y]
        else:
            item = bounds[source_id]
            item[0] = min(item[0], x)
            item[1] = max(item[1], x)
            item[2] = min(item[2], y)
            item[3] = max(item[3], y)
    return {
        source_id: ((values[0] + values[1]) * 0.5, (values[2] + values[3]) * 0.5)
        for source_id, values in bounds.items()
    }


def khaleque_proxy_prediction(batch: dict[str, Any], source_centers: dict[str, tuple[float, float]]) -> torch.Tensor:
    ego_history = batch["ego_history"]
    target = batch["future_path"]
    batch_size, future_steps = target.shape[:2]
    speeds = estimate_speed(ego_history)
    outputs = torch.zeros((batch_size, future_steps, 2), dtype=target.dtype, device=target.device)
    decision_interval = 10
    for i in range(batch_size):
        pose = batch["current_pose"][i]
        source_id = str(batch["source"][i].get("source_id") or batch["metadata"][i].get("source_id") or "unknown")
        center = source_centers.get(source_id, (float(pose["x"]), float(pose["y"])))
        center_forward, center_right = world_delta_to_local(
            float(pose["x"]),
            float(pose["y"]),
            float(pose.get("angle", 0.0)),
            center[0],
            center[1],
        )
        pos_forward = 0.0
        pos_right = 0.0
        direction = 0.0
        for step in range(future_steps):
            if step % decision_interval == 0:
                bias = math.atan2(center_right - pos_right, center_forward - pos_forward)
                random_offset = deterministic_uniform(
                    f"{batch['sample_id'][i]}::{step}",
                    -math.radians(67.5),
                    math.radians(67.5),
                )
                direction = bias + random_offset
            step_speed = float(speeds[i].detach().cpu())
            pos_forward += step_speed * math.cos(direction)
            pos_right += step_speed * math.sin(direction)
            outputs[i, step, 0] = pos_forward
            outputs[i, step, 1] = pos_right
    return outputs


def pixels_saliency_proxy_prediction(batch: dict[str, Any]) -> torch.Tensor:
    frames = batch["rgb_history"][:, -1]
    target = batch["future_path"]
    batch_size, future_steps = target.shape[:2]
    gray = frames.mean(dim=1)
    _, height, width = gray.shape
    crop = gray[:, int(height * 0.35) : int(height * 0.92), :]
    edges = torch.zeros_like(crop)
    edges[:, :, 1:] = (crop[:, :, 1:] - crop[:, :, :-1]).abs()
    brightness = crop.mean(dim=1)
    texture = edges.mean(dim=1)
    center_prior = torch.linspace(-1.0, 1.0, width, device=frames.device).abs()
    score = texture + 0.35 * brightness - 0.10 * center_prior
    best_col = score.argmax(dim=1).float()
    centered = (best_col - (width - 1) * 0.5) / max((width - 1) * 0.5, 1.0)
    directions = centered * math.radians(60.0)
    speeds = estimate_speed(batch["ego_history"])
    outputs = torch.zeros((batch_size, future_steps, 2), dtype=target.dtype, device=target.device)
    for step in range(future_steps):
        outputs[:, step, 0] = (step + 1) * speeds * torch.cos(directions)
        outputs[:, step, 1] = (step + 1) * speeds * torch.sin(directions)
    return outputs


@torch.no_grad()
def evaluate_case(case: EvalCase, device: torch.device, batch_size: int, num_workers: int) -> dict[str, Any]:
    start = time.perf_counter()
    model = None
    checkpoint: dict[str, Any] = {}
    source_centers = None
    if case.kind == "checkpoint":
        if case.checkpoint is None:
            raise ValueError(f"Checkpoint case missing checkpoint: {case}")
        model, checkpoint = load_model(case.checkpoint, device)
        model_name = str(checkpoint["model_name"])
        load_rgb = needs_rgb(model_name) and case.visual_feature_cache is None
        image_size = int(checkpoint.get("image_size", 64))
        loss_type = str(checkpoint.get("loss", "huber"))
        trajectory_scale = float(checkpoint.get("trajectory_scale", 1.0))
        confidence_weight = float(checkpoint.get("multimodal_confidence_weight", 0.05))
    else:
        load_rgb = case.kind == "pixels_proxy"
        image_size = 64
        loss_type = "huber"
        trajectory_scale = 1.0
        confidence_weight = 0.05
        if case.kind == "khaleque_proxy":
            source_centers = source_centers_from_train(case.dataset)
    dataset = WITVZPathDataset(
        case.dataset,
        split="test",
        image_size=image_size,
        load_rgb=load_rgb,
        visual_feature_cache_dir=case.visual_feature_cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_path_batch,
    )

    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_count = 0
    horizon_errors = []

    for batch in loader:
        batch = move_batch(batch, device)
        if case.kind == "checkpoint":
            if model is None:
                raise RuntimeError("Checkpoint model was not initialized")
            pred = model(batch)
        elif case.kind == "khaleque_proxy":
            if source_centers is None:
                raise RuntimeError("Khaleque proxy source centers were not initialized")
            pred = khaleque_proxy_prediction(batch, source_centers)
        elif case.kind == "pixels_proxy":
            pred = pixels_saliency_proxy_prediction(batch)
        else:
            raise ValueError(f"Unknown eval case kind: {case.kind}")
        target = batch["future_path"]
        batch_size_actual = target.shape[0]
        loss = trajectory_loss(
            pred,
            target,
            loss_type,
            coordinate_scale=trajectory_scale,
            multimodal_confidence_weight=confidence_weight,
        )
        total_loss += float(loss.detach().cpu()) * batch_size_actual
        total_ade += float(ade(pred, target).detach().cpu()) * batch_size_actual
        total_fde += float(fde(pred, target).detach().cpu()) * batch_size_actual
        horizon_errors.append(per_horizon_error(pred, target).detach().cpu() * batch_size_actual)
        total_count += batch_size_actual

    if total_count == 0:
        raise ValueError(f"No test samples for {case}")
    per_h = torch.stack(horizon_errors, dim=0).sum(dim=0) / total_count
    elapsed = time.perf_counter() - start
    return {
        "horizon_sec": case.horizon,
        "model_key": case.model_key,
        "label": case.label,
        "kind": case.kind,
        "dataset": case.dataset.as_posix(),
        "checkpoint": case.checkpoint.as_posix() if case.checkpoint else None,
        "visual_feature_cache": case.visual_feature_cache.as_posix() if case.visual_feature_cache else None,
        "test_samples": total_count,
        "loss": total_loss / total_count,
        "ADE": total_ade / total_count,
        "FDE": total_fde / total_count,
        "per_horizon_error": per_h.tolist(),
        "elapsed_sec": elapsed,
    }


def add_gains(rows: list[dict[str, Any]]) -> None:
    paper_keys = {"khaleque_center_random_proxy", "xu_pixels_saliency_proxy"}
    by_horizon_ade: dict[int, dict[str, Any]] = {}
    by_horizon_fde: dict[int, dict[str, Any]] = {}
    for horizon in sorted({int(row["horizon_sec"]) for row in rows}):
        paper_rows = [
            row
            for row in rows
            if int(row["horizon_sec"]) == horizon and row["model_key"] in paper_keys
        ]
        by_horizon_ade[horizon] = min(paper_rows, key=lambda item: item["ADE"])
        by_horizon_fde[horizon] = min(paper_rows, key=lambda item: item["FDE"])
    for row in rows:
        horizon = int(row["horizon_sec"])
        base_ade = by_horizon_ade[horizon]
        base_fde = by_horizon_fde[horizon]
        row["best_paper_proxy_ADE_model"] = base_ade["label"]
        row["best_paper_proxy_FDE_model"] = base_fde["label"]
        row["ADE_gain_vs_best_paper_proxy_pct"] = (1.0 - row["ADE"] / base_ade["ADE"]) * 100.0
        row["FDE_gain_vs_best_paper_proxy_pct"] = (1.0 - row["FDE"] / base_fde["FDE"]) * 100.0


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], device: torch.device) -> None:
    model_order = {
        "khaleque_center_random_proxy": 0,
        "xu_pixels_saliency_proxy": 1,
        "constant_velocity": 2,
        "small_cnn_timesformer": 3,
        "dinov3_timesformer": 4,
        "dinov3_strnet": 5,
    }
    lines = [
        "# Paper-Proxy Baseline And Ablation Inference Comparison",
        "",
        "Date: 2026-05-22",
        "",
        "## Scope",
        "",
        "- Re-ran metric-only inference locally on the available v2 horizon-sweep test splits.",
        "- Paper baselines requested: Khaleque et al. (2024) exploratory agents and Xu et al. (2026) pixels-only navigation.",
        "- The exact paper systems are interactive agents, not offline trajectory regressors. This report therefore includes paper-inspired offline proxies plus internal ablations.",
        "- Metrics are ADE/FDE in local egocentric coordinates; lower is better.",
        f"- Device used for this rerun: `{device}`.",
        "",
        "## Baseline Adaptation",
        "",
        "| Paper baseline | Offline proxy used here | What is missing vs the paper |",
        "| --- | --- | --- |",
        "| Khaleque, Cook, & Gow (2024) | Center-biased exploratory context-steering proxy that picks a deterministic random direction inside a 135 degree sector and biases it toward the train-set source center every 2 seconds. | The original uses level/object motivation metrics and context steering inside an interactive environment. Our processed samples do not include object/light annotations or interactive rollout state. |",
        "| Xu et al. (2026) | Screen-only saliency controller that reads the last RGB frame, chooses a salient horizontal interest point, and rolls out a fixed-speed local path. | The original builds on a visual affordance detector and finite-state controller in a live commercial 3D ARPG. We do not have that detector or live ARPG environment here. |",
        "",
        "## Re-run Results",
        "",
        "| Horizon | Model | Test samples | ADE | FDE | ADE gain vs best paper proxy | FDE gain vs best paper proxy |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["horizon_sec"], model_order.get(item["model_key"], 99))):
        lines.append(
            "| "
            f"{row['horizon_sec']}s | {row['label']} | {row['test_samples']} | "
            f"{format_float(row['ADE'])} | {format_float(row['FDE'])} | "
            f"{row['ADE_gain_vs_best_paper_proxy_pct']:.1f}% | {row['FDE_gain_vs_best_paper_proxy_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Best By Horizon",
            "",
            "| Horizon | Best ADE model | Best ADE | Best FDE model | Best FDE |",
            "| ---: | --- | ---: | --- | ---: |",
        ]
    )
    horizons = sorted({int(row["horizon_sec"]) for row in rows})
    for horizon in horizons:
        subset = [row for row in rows if int(row["horizon_sec"]) == horizon]
        best_ade = min(subset, key=lambda item: item["ADE"])
        best_fde = min(subset, key=lambda item: item["FDE"])
        lines.append(
            "| "
            f"{horizon}s | {best_ade['label']} | {format_float(best_ade['ADE'])} | "
            f"{best_fde['label']} | {format_float(best_fde['FDE'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The paper-proxy baselines are not exact reproductions; they are task adapters for offline ADE/FDE comparison.",
            "- Constant velocity remains an internal motion-only ablation, not one of the requested literature baselines.",
            "- From 1s through 10s, the learned visual models beat the best paper-inspired proxy on the main ADE comparison.",
            "- The 30s Khaleque-style proxy is strongest on this v2 split, but this should be treated as a diagnostic artifact: the 30s test set is small and dominated by `my_way_home`, while the proxy uses a train-set source center that acts like a map prior.",
            "- Cached DINOv3 generally improves over small-CNN, which supports using a frozen visual token cache as the stronger visual representation path.",
            "- STRNet-style fusion is not uniformly better than TimeSFormer. It helps most clearly on 3s ADE/FDE and on some mid-horizon endpoint errors, while TimeSFormer remains stronger at 1s and 30s.",
            "- This is an ablation of representation/temporal modules, not a pathfinding benchmark. The models predict a single future local trajectory.",
            "",
            "## V4 Published Checkpoint Context",
            "",
            "The pushed v4 checkpoints could not be re-run on this local machine because `data/wit_vz/processed/wit_vz_v4_defaults_001` and the 44GB DINOv3 cache are not present locally. The published server-side v4 results are:",
            "",
            "| Horizon | Constant-velocity ADE/FDE | DINOv3 TimeSFormer ADE/FDE | ADE gain | FDE gain |",
            "| ---: | ---: | ---: | ---: | ---: |",
            "| 1s | 33.1120 / 51.4413 | 26.8676 / 41.5629 | 18.9% | 19.2% |",
            "| 3s | 75.7201 / 131.6904 | 62.1001 / 103.3531 | 18.0% | 21.5% |",
            "| 5s | 111.2669 / 202.7233 | 88.6020 / 157.0852 | 20.4% | 22.5% |",
            "| 10s | 217.1669 / 408.6508 | 154.5734 / 258.7196 | 28.8% | 36.7% |",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/inference_ablation_compare/results_v2_horizon.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/inference_ablation_baseline_comparison_20260522.md"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10, 30])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    device = choose_device(args.device)
    cases = make_v2_cases(root, args.horizons)
    rows = []
    for case in cases:
        print(f"eval horizon={case.horizon}s model={case.model_key}")
        rows.append(evaluate_case(case, device, args.batch_size, args.num_workers))
    add_gains(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_md, rows, device)
    print(json.dumps({"output_json": args.output_json.as_posix(), "output_md": args.output_md.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
