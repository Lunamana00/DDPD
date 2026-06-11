"""Evaluate paper-adapted baselines on WIT-VZ v4 horizon datasets.

The cited papers do not expose the same offline trajectory-prediction task as
WIT-VZ. This script therefore evaluates explicit adapters:

- Khaleque et al. (2024): center-biased exploratory context-steering rollout.
- Xu et al. (2026): pixels-only saliency steering rollout.

Both adapters output future local paths so they can be scored with ADE/FDE on
the same test split as the proposed model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from src.models.baselines import ConstantVelocityBaseline
from src.models.factory import create_model, needs_rgb
from src.models.paper_proxies import (
    khaleque_center_random_prediction,
    source_centers_from_train,
    xu_pixels_saliency_prediction,
)
from src.train_path_predictor import move_batch
from src.wit_vz.dataset import WITVZPathDataset, collate_path_batch


@dataclass(frozen=True)
class EvalCase:
    model_key: str
    label: str
    horizon: int
    dataset: Path
    kind: str
    checkpoint: Path | None = None
    visual_feature_cache: Path | None = None


def horizon_tag(horizon: int) -> str:
    return f"{horizon:02d}s"


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def v4_dataset_path(root: Path, horizon: int) -> Path:
    return root / "data" / "wit_vz" / "processed" / "horizon_sweep_v4_defaults" / f"future_{horizon_tag(horizon)}"


def v4_single_checkpoint(root: Path, horizon: int) -> Path:
    return root / "checkpoints" / f"wit_vz_v4_defaults_dinov3_single_{horizon_tag(horizon)}.pt"


def make_v4_cases(
    root: Path,
    horizons: list[int],
    include_constant_velocity: bool,
    include_ours: bool,
    visual_feature_cache: Path | None,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for horizon in horizons:
        dataset = v4_dataset_path(root, horizon)
        cases.extend(
            [
                EvalCase(
                    model_key="khaleque_center_random_proxy",
                    label="Khaleque-style exploratory proxy",
                    horizon=horizon,
                    dataset=dataset,
                    kind="khaleque_proxy",
                ),
                EvalCase(
                    model_key="xu_pixels_saliency_proxy",
                    label="Xu-style pixels-only saliency proxy",
                    horizon=horizon,
                    dataset=dataset,
                    kind="pixels_proxy",
                ),
            ]
        )
        if include_constant_velocity:
            cases.append(
                EvalCase(
                    model_key="constant_velocity",
                    label="Internal motion-only constant velocity",
                    horizon=horizon,
                    dataset=dataset,
                    kind="constant_velocity",
                )
            )
        if include_ours:
            checkpoint = v4_single_checkpoint(root, horizon)
            cases.append(
                EvalCase(
                    model_key="ours_dinov3_single",
                    label="Ours: cached DINOv3 trajectory predictor",
                    horizon=horizon,
                    dataset=dataset,
                    kind="checkpoint",
                    checkpoint=checkpoint,
                    visual_feature_cache=visual_feature_cache,
                )
            )
    return cases


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = str(checkpoint["model_name"])
    model = create_model(
        model_name,
        future_steps=int(checkpoint["future_steps"]),
        backbone_name=str(checkpoint.get("backbone", "small_cnn")),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        freeze_backbone=bool(checkpoint.get("freeze_backbone", True)),
        num_motivation_tokens=int(checkpoint.get("num_motivation_tokens", 4)),
        num_heads=int(checkpoint.get("num_heads", 4)),
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


def missing_case_result(case: EvalCase, reason: str) -> dict[str, Any]:
    return {
        "horizon_sec": case.horizon,
        "model_key": case.model_key,
        "label": case.label,
        "kind": case.kind,
        "dataset": case.dataset.as_posix(),
        "checkpoint": case.checkpoint.as_posix() if case.checkpoint else None,
        "visual_feature_cache": case.visual_feature_cache.as_posix() if case.visual_feature_cache else None,
        "available": False,
        "missing_reason": reason,
    }


def override_dataset_raw_root(dataset: WITVZPathDataset, raw_root: Path | None) -> None:
    """Redirect raw frame roots for worktrees that only symlink processed data."""
    if raw_root is None:
        return
    root = raw_root.expanduser()
    dataset.raw_dirs = {
        source_id: root / Path(raw_dir).name
        for source_id, raw_dir in dataset.raw_dirs.items()
    }
    dataset.raw_dir = next(iter(dataset.raw_dirs.values()))


@torch.no_grad()
def evaluate_case(
    case: EvalCase,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    if not case.dataset.exists():
        return missing_case_result(case, f"missing dataset: {case.dataset.as_posix()}")

    model: torch.nn.Module | None = None
    checkpoint: dict[str, Any] = {}
    source_centers = None
    load_rgb = False
    image_size = 64
    loss_type = "huber"
    trajectory_scale = 1.0
    confidence_weight = 0.05

    if case.kind == "checkpoint":
        if case.checkpoint is None or not case.checkpoint.exists():
            return missing_case_result(case, f"missing checkpoint: {case.checkpoint}")
        raw_checkpoint = torch.load(case.checkpoint, map_location="cpu")
        backbone_name = str(raw_checkpoint.get("backbone", "small_cnn")).lower()
        if backbone_name.startswith("cached") and case.visual_feature_cache is None:
            return missing_case_result(
                case,
                "checkpoint uses cached visual tokens but visual feature cache is missing",
            )
        model, checkpoint = load_checkpoint_model(case.checkpoint, device)
        model_name = str(checkpoint["model_name"])
        load_rgb = needs_rgb(model_name) and case.visual_feature_cache is None
        image_size = int(checkpoint.get("image_size", 64))
        loss_type = str(checkpoint.get("loss", "huber"))
        trajectory_scale = float(checkpoint.get("trajectory_scale", 1.0))
        confidence_weight = float(checkpoint.get("multimodal_confidence_weight", 0.05))
    elif case.kind == "constant_velocity":
        manifest = json.loads((case.dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
        future_steps = int(manifest.get("future_steps") or round(float(manifest["future_seconds"]) * 5))
        model = ConstantVelocityBaseline(future_steps=future_steps).to(device)
        model.eval()
    elif case.kind == "pixels_proxy":
        load_rgb = True
    elif case.kind == "khaleque_proxy":
        source_centers = source_centers_from_train(case.dataset)
    else:
        raise ValueError(f"Unknown eval case kind: {case.kind}")

    dataset = WITVZPathDataset(
        case.dataset,
        split="test",
        image_size=image_size,
        load_rgb=load_rgb,
        visual_feature_cache_dir=case.visual_feature_cache,
    )
    override_dataset_raw_root(dataset, raw_root)
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
        if case.kind in {"checkpoint", "constant_velocity"}:
            if model is None:
                raise RuntimeError(f"Model was not initialized for {case}")
            pred = model(batch)
        elif case.kind == "khaleque_proxy":
            if source_centers is None:
                raise RuntimeError("Khaleque proxy source centers were not initialized")
            pred = khaleque_center_random_prediction(batch, source_centers)
        elif case.kind == "pixels_proxy":
            pred = xu_pixels_saliency_prediction(batch)
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
        return missing_case_result(case, "no test samples")
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
        "available": True,
        "test_samples": total_count,
        "loss": total_loss / total_count,
        "ADE": total_ade / total_count,
        "FDE": total_fde / total_count,
        "per_horizon_error": per_h.tolist(),
        "elapsed_sec": elapsed,
    }


def add_gains(rows: list[dict[str, Any]]) -> None:
    paper_keys = {"khaleque_center_random_proxy", "xu_pixels_saliency_proxy"}
    available = [row for row in rows if row.get("available")]
    for horizon in sorted({int(row["horizon_sec"]) for row in available}):
        paper_rows = [
            row
            for row in available
            if int(row["horizon_sec"]) == horizon and row["model_key"] in paper_keys
        ]
        if not paper_rows:
            continue
        base_ade = min(paper_rows, key=lambda item: float(item["ADE"]))
        base_fde = min(paper_rows, key=lambda item: float(item["FDE"]))
        for row in available:
            if int(row["horizon_sec"]) != horizon:
                continue
            row["best_paper_proxy_ADE_model"] = base_ade["label"]
            row["best_paper_proxy_FDE_model"] = base_fde["label"]
            row["ADE_gain_vs_best_paper_proxy_pct"] = (1.0 - row["ADE"] / base_ade["ADE"]) * 100.0
            row["FDE_gain_vs_best_paper_proxy_pct"] = (1.0 - row["FDE"] / base_fde["FDE"]) * 100.0


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], device: torch.device) -> None:
    order = {
        "khaleque_center_random_proxy": 0,
        "xu_pixels_saliency_proxy": 1,
        "constant_velocity": 2,
        "ours_dinov3_single": 3,
    }
    lines = [
        "# V4 Paper-Adapted Baseline Evaluation",
        "",
        "## Scope",
        "",
        "- Dataset family: `horizon_sweep_v4_defaults`.",
        "- Metrics: ADE/FDE in local egocentric coordinates; lower is better.",
        "- The paper baselines are adapters, not exact reproductions of the original interactive systems.",
        f"- Device: `{device}`.",
        "",
        "## Baseline Adaptation",
        "",
        "| Paper | Adapter used here | Missing vs original paper |",
        "| --- | --- | --- |",
        "| Khaleque, Cook, & Gow (2024) | Deterministic center-biased exploratory context-steering rollout. | No live object/motivation/coverage state is stored in WIT-VZ processed samples. |",
        "| Xu et al. (2026) | Last-frame pixels-only saliency steering rollout. | No live ARPG controller, no trained STP/MSTP detector for ViZDoom. |",
        "",
        "## Results",
        "",
        "| Horizon | Model | Available | Test samples | ADE | FDE | ADE gain vs best paper proxy | FDE gain vs best paper proxy |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["horizon_sec"], order.get(item["model_key"], 99))):
        if row.get("available"):
            lines.append(
                "| "
                f"{row['horizon_sec']}s | {row['label']} | yes | {row['test_samples']} | "
                f"{fmt(row['ADE'])} | {fmt(row['FDE'])} | "
                f"{fmt(row.get('ADE_gain_vs_best_paper_proxy_pct'))} | "
                f"{fmt(row.get('FDE_gain_vs_best_paper_proxy_pct'))} |"
            )
        else:
            lines.append(
                "| "
                f"{row['horizon_sec']}s | {row['label']} | no | - | - | - | - | - |"
            )

    missing = [row for row in rows if not row.get("available")]
    if missing:
        lines.extend(["", "## Missing Cases", ""])
        for row in missing:
            lines.append(f"- {row['horizon_sec']}s `{row['label']}`: {row.get('missing_reason')}")

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Treat these rows as paper-adapted offline trajectory proxies.",
            "- Do not claim exact reproduction unless the original interactive environment, model checkpoints, and control loop are available.",
            "- These baselines are useful for answering whether a simple paper-inspired decision rule can match the learned WIT-VZ trajectory predictor under the same ADE/FDE protocol.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Optional raw frame root used when processed manifests point at a different checkout.",
    )
    parser.add_argument(
        "--visual-feature-cache",
        type=Path,
        default=Path("data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny"),
    )
    parser.add_argument("--no-constant-velocity", action="store_true")
    parser.add_argument("--no-ours", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/paper_baselines_v4/results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/paper_baselines_v4.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    device = choose_device(args.device)
    cache = args.visual_feature_cache
    if not cache.exists():
        cache = None
    cases = make_v4_cases(
        root,
        args.horizons,
        include_constant_velocity=not args.no_constant_velocity,
        include_ours=not args.no_ours,
        visual_feature_cache=cache,
    )

    rows = []
    for case in cases:
        print(f"eval horizon={case.horizon}s model={case.model_key}", flush=True)
        rows.append(evaluate_case(case, device, args.batch_size, args.num_workers, raw_root=args.raw_root))
    add_gains(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(args.output_md, rows, device)
    print(json.dumps({"output_json": args.output_json.as_posix(), "output_md": args.output_md.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
