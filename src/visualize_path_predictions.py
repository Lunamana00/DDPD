"""Visualize predicted future local paths against ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from .wit_vz.io import load_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize WIT-VZ path predictions.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    return parser.parse_args()


def resolve_raw_path(dataset_dir: Path, rel_path: str) -> Path:
    manifest = load_json(dataset_dir / "dataset_manifest.json")
    raw_dir = Path(manifest["raw_dir"])
    if not raw_dir.is_absolute():
        candidate = dataset_dir / raw_dir
        raw_dir = candidate if candidate.exists() else raw_dir
    return raw_dir / rel_path


def path_panel(pred: list[list[float]], gt: list[list[float]], size: int = 360) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    margin = 34
    cx = size // 2
    cy = size - margin
    draw.line((cx, cy, cx, margin), fill=(40, 40, 40), width=2)
    draw.line((cx, cy, size - margin, cy), fill=(80, 80, 80), width=2)
    draw.text((cx + 4, margin), "forward", fill=(40, 40, 40))
    draw.text((size - 90, cy + 4), "right", fill=(80, 80, 80))
    all_points = pred + gt
    max_abs = max([max(abs(x), abs(y)) for x, y in all_points] + [1.0])
    scale = (size - 2 * margin) / max(max_abs * 2.2, 1.0)

    def convert(path: list[list[float]]) -> list[tuple[float, float]]:
        return [(cx + y * scale, cy - x * scale) for x, y in path]

    gt_points = convert(gt)
    pred_points = convert(pred)
    if len(gt_points) > 1:
        draw.line(gt_points, fill=(20, 140, 45), width=4)
    if len(pred_points) > 1:
        draw.line(pred_points, fill=(210, 55, 45), width=4)
    for px, py in gt_points:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(20, 140, 45))
    for px, py in pred_points:
        draw.rectangle((px - 3, py - 3, px + 3, py + 3), fill=(210, 55, 45))
    return image


def ade_fde(pred: list[list[float]], gt: list[list[float]]) -> tuple[float, float]:
    errors = [math.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(pred, gt)]
    return sum(errors) / len(errors), errors[-1]


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = {sample["sample_id"]: sample for sample in read_jsonl(args.dataset / "samples.jsonl")}
    predictions = []
    with args.predictions.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    figure_paths = []
    for i, pred_item in enumerate(predictions[: args.num_samples], start=1):
        sample = samples[pred_item["sample_id"]]
        frame_path = resolve_raw_path(args.dataset, sample["rgb_history_paths"][-1])
        frame = Image.open(frame_path).convert("RGB").resize((360, 270))
        pred = pred_item["prediction"]
        gt = sample["future_local_path"]
        panel = path_panel(pred, gt)
        canvas = Image.new("RGB", (760, 410), "white")
        canvas.paste(frame, (20, 52))
        canvas.paste(panel, (390, 25))
        draw = ImageDraw.Draw(canvas)
        sample_ade, sample_fde = ade_fde(pred, gt)
        draw.text((20, 18), pred_item["sample_id"], fill=(0, 0, 0))
        draw.text((20, 335), f"GT: green   Pred: red   ADE={sample_ade:.3f} FDE={sample_fde:.3f}", fill=(0, 0, 0))
        out_path = args.out / f"prediction_{i:04d}.png"
        canvas.save(out_path)
        figure_paths.append(out_path)

    if figure_paths:
        thumbs = [Image.open(path).resize((380, 205)) for path in figure_paths[: min(8, len(figure_paths))]]
        cols = 2
        rows = math.ceil(len(thumbs) / cols)
        montage = Image.new("RGB", (cols * 380, rows * 205), "white")
        for idx, thumb in enumerate(thumbs):
            montage.paste(thumb, ((idx % cols) * 380, (idx // cols) * 205))
        montage.save(args.out / "montage.png")
    print(f"Wrote {len(figure_paths)} figures to: {args.out}")


if __name__ == "__main__":
    main()
