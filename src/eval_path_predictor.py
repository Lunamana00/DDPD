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
    model = create_model(
        model_name,
        future_steps=int(checkpoint["future_steps"]),
        backbone_name=str(checkpoint.get("backbone", "small_cnn")),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        freeze_backbone=bool(checkpoint.get("freeze_backbone", True)),
        num_cue_tokens=int(checkpoint.get("num_cue_tokens", 8)),
        temporal_type=str(checkpoint.get("temporal_type", "transformer")),
        use_constant_velocity_residual=bool(checkpoint.get("use_constant_velocity_residual", False)),
        residual_scale=float(checkpoint.get("residual_scale", checkpoint.get("trajectory_scale", 1.0))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    loader_args = SimpleNamespace(
        dataset=args.dataset,
        image_size=int(checkpoint.get("image_size", 64)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = make_loader(loader_args, args.split, needs_rgb(model_name))
    metrics, predictions = evaluate_loader(
        model,
        loader,
        device,
        str(checkpoint.get("loss", "huber")),
        float(checkpoint.get("trajectory_scale", 1.0)),
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
