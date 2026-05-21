"""Render animated ViZDoom prediction replays from saved path predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .visualize_path_predictions import ade_fde, resolve_raw_path
from .wit_vz.io import load_json, read_jsonl


PRED_COLOR = (35, 105, 220)
GT_COLOR = (25, 145, 65)
GRID_COLOR = (220, 225, 230)
TEXT_COLOR = (25, 30, 35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an animated ViZDoom replay with predicted and GT future paths."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--frame-width", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=420)
    parser.add_argument("--save-frames", action="store_true")
    return parser.parse_args()


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_replay_items(
    samples: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    episode_id: str | None,
    num_frames: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    available: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pred in predictions:
        sample = samples.get(str(pred.get("sample_id")))
        if sample is None:
            continue
        if episode_id is not None and str(sample.get("episode_id")) != episode_id:
            continue
        available.append((sample, pred))

    if not available:
        raise ValueError("No predictions matched the processed dataset and episode filter.")

    if episode_id is None:
        by_episode: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for sample, pred in available:
            by_episode[str(sample.get("episode_id"))].append((sample, pred))
        available = max(
            by_episode.values(),
            key=lambda items: (len(items), -min(int(item[0].get("center_step", 0)) for item in items)),
        )

    available.sort(key=lambda item: int(item[0].get("center_step", 0)))
    return available[: max(1, num_frames)]


def _path_bounds(*paths: list[list[float]]) -> float:
    values = [
        max(abs(float(point[0])), abs(float(point[1])))
        for path in paths
        for point in path
        if len(point) >= 2
    ]
    return max(values + [1.0])


def _to_panel_points(
    path: list[list[float]],
    *,
    cx: float,
    cy: float,
    scale: float,
) -> list[tuple[float, float]]:
    return [(cx + float(y) * scale, cy - float(x) * scale) for x, y in path]


def draw_path_panel(
    prediction: list[list[float]],
    gt: list[list[float]],
    candidate_predictions: list[list[list[float]]] | None = None,
    size: int = 420,
) -> Image.Image:
    image = Image.new("RGB", (size, size), (250, 251, 252))
    draw = ImageDraw.Draw(image)
    margin = 44
    cx = size // 2
    cy = size - margin
    max_abs = _path_bounds(prediction, gt, *(candidate_predictions or []))
    scale = (size - 2 * margin) / max(max_abs * 2.25, 1.0)

    for offset in range(-4, 5):
        x = cx + offset * scale
        y = cy - offset * scale
        draw.line((x, margin, x, cy), fill=GRID_COLOR, width=1)
        draw.line((margin, y, size - margin, y), fill=GRID_COLOR, width=1)

    draw.line((cx, cy, cx, margin), fill=(50, 55, 60), width=2)
    draw.line((margin, cy, size - margin, cy), fill=(80, 85, 90), width=2)
    draw.polygon([(cx, cy - 12), (cx - 8, cy + 8), (cx + 8, cy + 8)], fill=(35, 35, 35))
    draw.text((cx + 6, margin - 24), "forward", fill=TEXT_COLOR, font=_font(13))
    draw.text((size - 92, cy + 8), "right", fill=TEXT_COLOR, font=_font(13))

    if candidate_predictions:
        for candidate in candidate_predictions:
            points = _to_panel_points(candidate, cx=cx, cy=cy, scale=scale)
            if len(points) > 1:
                draw.line(points, fill=(155, 190, 245), width=2)

    gt_points = _to_panel_points(gt, cx=cx, cy=cy, scale=scale)
    pred_points = _to_panel_points(prediction, cx=cx, cy=cy, scale=scale)
    if len(gt_points) > 1:
        draw.line(gt_points, fill=GT_COLOR, width=4)
    if len(pred_points) > 1:
        draw.line(pred_points, fill=PRED_COLOR, width=4)
    for px, py in gt_points:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=GT_COLOR)
    for px, py in pred_points:
        draw.rectangle((px - 4, py - 4, px + 4, py + 4), fill=PRED_COLOR)
    return image


def render_frame(
    dataset_dir: Path,
    sample: dict[str, Any],
    pred_item: dict[str, Any],
    frame_width: int,
    panel_size: int,
) -> Image.Image:
    source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
    frame_path = resolve_raw_path(dataset_dir, sample["rgb_history_paths"][-1], source_id)
    frame = Image.open(frame_path).convert("RGB")
    frame_height = max(1, round(frame.height * (frame_width / frame.width)))
    frame = frame.resize((frame_width, frame_height))

    pred = pred_item["prediction"]
    gt = sample["future_local_path"]
    panel = draw_path_panel(pred, gt, pred_item.get("candidate_predictions"), panel_size)
    sample_ade, sample_fde = ade_fde(pred, gt)

    margin = 24
    header_h = 70
    legend_h = 48
    content_h = max(frame_height, panel_size)
    canvas = Image.new(
        "RGB",
        (frame_width + panel_size + margin * 3, header_h + content_h + legend_h + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    body_font = _font(15)
    small_font = _font(13)

    draw.text((margin, 18), "ViZDoom Future Path Prediction Replay", fill=TEXT_COLOR, font=title_font)
    draw.text(
        (margin, 44),
        f"{sample['sample_id']}  step={sample.get('center_step')}  ADE={sample_ade:.3f}  FDE={sample_fde:.3f}",
        fill=(75, 80, 85),
        font=small_font,
    )
    frame_y = header_h
    panel_x = frame_width + margin * 2
    panel_y = header_h
    canvas.paste(frame, (margin, frame_y))
    canvas.paste(panel, (panel_x, panel_y))

    legend_y = header_h + content_h + 14
    draw.rectangle((margin, legend_y + 4, margin + 18, legend_y + 16), fill=GT_COLOR)
    draw.text((margin + 26, legend_y), "ground truth future path", fill=TEXT_COLOR, font=body_font)
    pred_x = margin + 230
    draw.rectangle((pred_x, legend_y + 4, pred_x + 18, legend_y + 16), fill=PRED_COLOR)
    draw.text((pred_x + 26, legend_y), "model prediction", fill=TEXT_COLOR, font=body_font)
    draw.text(
        (panel_x, legend_y),
        "local coordinates: x=forward, y=right",
        fill=(75, 80, 85),
        font=small_font,
    )
    return canvas


def resolve_output_path(out: Path) -> Path:
    if out.suffix.lower() == ".gif":
        return out
    return out / "vizdoom_path_replay.gif"


def save_replay(
    frames: list[Image.Image],
    gif_path: Path,
    fps: float,
    save_frames: bool = False,
) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000.0 / max(fps, 0.1)))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    if save_frames:
        frame_dir = gif_path.parent / f"{gif_path.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames, start=1):
            frame.save(frame_dir / f"frame_{idx:04d}.png")


def main() -> None:
    args = parse_args()
    manifest = load_json(args.dataset / "dataset_manifest.json")
    samples = {sample["sample_id"]: sample for sample in read_jsonl(args.dataset / "samples.jsonl")}
    predictions = load_predictions(args.predictions)
    items = select_replay_items(samples, predictions, args.episode_id, args.num_frames)

    frames = [
        render_frame(args.dataset, sample, pred, args.frame_width, args.panel_size)
        for sample, pred in items
    ]
    out_path = resolve_output_path(args.out)
    save_replay(frames, out_path, args.fps, args.save_frames)

    episode = items[0][0].get("episode_id")
    future_sec = manifest.get("future_sec", "unknown")
    print(
        f"Wrote {len(frames)} replay frames from episode={episode} "
        f"future_sec={future_sec} to: {out_path}"
    )


if __name__ == "__main__":
    main()
