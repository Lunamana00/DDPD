"""Evaluate a trained path predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .models.factory import create_model, needs_rgb
from .train_path_predictor import evaluate_loader, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a WIT-VZ path predictor.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--visual-feature-cache", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
    visual_feature_cache = args.visual_feature_cache
    if visual_feature_cache is None and checkpoint.get("visual_feature_cache"):
        visual_feature_cache = Path(str(checkpoint["visual_feature_cache"]))
    model = create_model(
        model_name,
        future_steps=int(checkpoint["future_steps"]),
        backbone_name=str(checkpoint.get("backbone", "small_cnn")),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        freeze_backbone=bool(checkpoint.get("freeze_backbone", True)),
        num_motivation_tokens=int(checkpoint.get("num_motivation_tokens", 4)),
        num_heads=int(checkpoint.get("num_heads", 4)),
        num_cue_tokens=int(checkpoint.get("num_cue_tokens", 8)),
        num_modes=int(checkpoint.get("num_modes", 1)),
        temporal_type=str(checkpoint.get("temporal_type", "transformer")),
        temporal_layers=int(checkpoint.get("temporal_layers", 1)),
        selector_layers=int(checkpoint.get("selector_layers", 1)),
        decoder_layers=int(checkpoint.get("decoder_layers", 1)),
        decoder_type=str(checkpoint.get("decoder_type", "horizon_query_decoder")),
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

    loader_args = SimpleNamespace(
        dataset=args.dataset,
        visual_feature_cache=visual_feature_cache,
        image_size=int(checkpoint.get("image_size", 64)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history_frame_mode=str(checkpoint.get("history_frame_mode", "full")),
        train_frame_order="normal",
    )
    loader = make_loader(
        loader_args,
        args.split,
        needs_rgb(model_name, str(checkpoint.get("backbone", "small_cnn"))) and visual_feature_cache is None,
    )
    metrics, predictions = evaluate_loader(
        model,
        loader,
        device,
        str(checkpoint.get("loss", "huber")),
        float(checkpoint.get("trajectory_scale", 1.0)),
        float(checkpoint.get("multimodal_confidence_weight", 0.05)),
    )
    metrics["model"] = model_name
    metrics["split"] = args.split
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
