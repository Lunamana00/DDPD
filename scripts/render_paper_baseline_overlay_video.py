"""Render moving CV / Xu paper baseline / GT / model overlay demo video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render_triptych_demo_video import (  # noqa: E402
    BG_COLOR,
    CV_COLOR,
    GT_COLOR,
    MUTED_COLOR,
    PRED_COLOR,
    TEXT_COLOR,
    DemoSource,
    axis_scale,
    clipped_path,
    custom_sources,
    default_sources,
    draw_path_plot,
    existing,
    font,
    load_raw_dirs,
    load_rgb,
    metadata_label,
    moving_frame_paths,
    paste_center,
    path_error,
    read_jsonl,
    resolve_frame_path,
    selected_ids,
    wrap,
)
from src.models.paper_proxies import xu_pixels_saliency_prediction  # noqa: E402
from src.wit_vz.dataset import load_rgb_tensor  # noqa: E402


XU_COLOR = (135, 76, 188)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/demo/presentation_sequence/demo_paper_baseline_overlay_sequence.mp4"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hold-last-frames", type=int, default=6)
    parser.add_argument("--max-frames-per-item", type=int, default=0)
    parser.add_argument("--max-items-per-source", type=int, default=0)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Custom source as name|summary|dataset|predictions|raw_root_base.",
    )
    return parser.parse_args()


def compute_xu_path(sample: dict[str, Any], raw_dirs: dict[str, Path], source_id: str | None, target: list[list[float]]) -> list[list[float]]:
    frames = []
    for rel_path in sample["rgb_history_paths"]:
        path = resolve_frame_path(raw_dirs, rel_path, source_id)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing RGB frame for Xu proxy: {rel_path}")
        frames.append(load_rgb_tensor(path, 64))
    batch = {
        "sample_id": [sample["sample_id"]],
        "ego_history": torch.tensor([sample["relative_egomotion_history"]], dtype=torch.float32),
        "future_path": torch.tensor([target], dtype=torch.float32),
        "rgb_history": torch.stack(frames, dim=0).unsqueeze(0),
    }
    with torch.no_grad():
        return xu_pixels_saliency_prediction(batch)[0].cpu().tolist()


def load_source_items(source: DemoSource, max_items_per_source: int) -> list[dict[str, Any]]:
    dataset = existing(source.dataset_candidates)
    predictions = existing(source.prediction_candidates)
    if dataset is None or predictions is None or not source.summary.exists():
        print(f"skip {source.name}: missing dataset/predictions/summary")
        return []

    samples = {str(row["sample_id"]): row for row in read_jsonl(dataset / "samples.jsonl")}
    prediction_map = {str(row["sample_id"]): row for row in read_jsonl(predictions)}
    raw_dirs = load_raw_dirs(dataset, source.raw_root_bases)
    ids = selected_ids(source.summary, max_items_per_source or source.max_items)
    items = []
    for sample_id, label, case in ids:
        sample = samples.get(sample_id)
        prediction = prediction_map.get(sample_id)
        if sample is None or prediction is None:
            continue
        target = prediction.get("target") or sample.get("future_local_path") or []
        pred_path = prediction.get("prediction") or []
        cv_path = prediction.get("constant_velocity_prediction") or []
        source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
        xu_path = compute_xu_path(sample, raw_dirs, source_id, target)
        horizon_length = min(len(target), len(pred_path) or len(target), len(cv_path) or len(target), len(xu_path))
        frame_path = resolve_frame_path(raw_dirs, sample["rgb_history_paths"][-1], source_id)
        frame_paths = moving_frame_paths(sample, raw_dirs, source_id, horizon_length)
        ade, fde = path_error(pred_path, target)
        cv_ade, cv_fde = path_error(cv_path, target) if cv_path else (float("nan"), float("nan"))
        xu_ade, xu_fde = path_error(xu_path, target)
        items.append(
            {
                "source": source.name,
                "label": metadata_label(sample, label),
                "case": case,
                "sample_id": sample_id,
                "frame_path": frame_path,
                "frame_paths": frame_paths,
                "target": target,
                "prediction": pred_path,
                "cv": cv_path,
                "xu": xu_path,
                "ADE": float(prediction.get("ADE", ade)),
                "FDE": float(prediction.get("FDE", fde)),
                "cv_ADE": float(prediction.get("constant_velocity_ADE", cv_ade)),
                "cv_FDE": float(prediction.get("constant_velocity_FDE", cv_fde)),
                "xu_ADE": xu_ade,
                "xu_FDE": xu_fde,
            }
        )
    return items


def draw_column(
    canvas: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    subtitle: str,
    rgb: Image.Image,
    path: list[list[float]],
    full_path: list[list[float]],
    color: tuple[int, int, int],
    max_abs: float,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill="white", outline=(216, 220, 226), width=2)
    draw.text((x + 16, y + 16), title, fill=color, font=font(24, bold=True))
    draw.text((x + 16, y + 48), subtitle, fill=MUTED_COLOR, font=font(14))
    frame_box = (x + 18, y + 78, x + w - 18, y + 346)
    paste_center(canvas, rgb, frame_box)
    plot = draw_path_plot(path, (w - 36, h - 382), color, max_abs, full_path)
    canvas.paste(plot, (x + 18, y + 364))


def render_item_frame(
    item: dict[str, Any],
    order: int,
    total: int,
    progress: int,
    frame_path: Path | None,
    width: int,
    height: int,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    header = f"{order:02d}/{total:02d}  {item['source']} / {item['label']} / {item['case']}"
    draw.text((34, 26), header, fill=TEXT_COLOR, font=font(32, bold=True))
    horizon = max(len(item["target"]), len(item["prediction"]), len(item["cv"]), len(item["xu"]))
    metrics = (
        f"sample={item['sample_id']}    t={min(progress + 1, horizon):02d}/{horizon:02d}    "
        f"ours ADE/FDE={item['ADE']:.2f}/{item['FDE']:.2f}    "
        f"CV={item['cv_ADE']:.2f}/{item['cv_FDE']:.2f}    "
        f"Xu proxy={item['xu_ADE']:.2f}/{item['xu_FDE']:.2f}"
    )
    draw.text((34, 72), wrap(metrics, 145), fill=MUTED_COLOR, font=font(18))
    col_gap = 16
    margin_x = 34
    top = 134
    col_w = (width - 2 * margin_x - 3 * col_gap) // 4
    col_h = height - top - 34
    rgb = load_rgb(frame_path, (col_w - 40, 268))
    max_abs = axis_scale(item["cv"], item["xu"], item["target"], item["prediction"])
    shown = progress + 1
    specs = [
        ("CV baseline", "recent-motion extrapolation", "cv", CV_COLOR),
        ("Xu paper baseline", "pixels-only saliency steering", "xu", XU_COLOR),
        ("GT", "future local path label", "target", GT_COLOR),
        ("Ours", "visual cue-memory output", "prediction", PRED_COLOR),
    ]
    for idx, (title, subtitle, key, color) in enumerate(specs):
        x = margin_x + idx * (col_w + col_gap)
        draw_column(
            canvas,
            x,
            top,
            col_w,
            col_h,
            title,
            subtitle,
            rgb,
            clipped_path(item[key], shown),
            item[key],
            color,
            max_abs,
        )
    return canvas


def write_video(items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")
    try:
        total = len(items)
        for index, item in enumerate(items, start=1):
            frame_paths = item["frame_paths"] or [item["frame_path"]]
            if args.max_frames_per_item > 0:
                frame_paths = frame_paths[: args.max_frames_per_item]
            last_frame_bgr: np.ndarray | None = None
            for progress, frame_path in enumerate(frame_paths):
                frame = render_item_frame(item, index, total, progress, frame_path, args.width, args.height)
                last_frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                writer.write(last_frame_bgr)
            if last_frame_bgr is not None:
                for _ in range(max(0, args.hold_last_frames)):
                    writer.write(last_frame_bgr)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    sources = custom_sources(args.source) if args.source else default_sources()
    all_items: list[dict[str, Any]] = []
    for source in sources:
        items = load_source_items(source, args.max_items_per_source)
        print(f"{source.name}: {len(items)} items")
        all_items.extend(items)
    if not all_items:
        raise RuntimeError("No renderable demo items found.")
    write_video(all_items, args)
    moving_frames = sum(
        min(len(item["frame_paths"]), args.max_frames_per_item) if args.max_frames_per_item > 0 else len(item["frame_paths"])
        for item in all_items
    )
    missing = sum(1 for item in all_items for path in item["frame_paths"] if path is None or not path.exists())
    manifest = {
        "output": str(args.output),
        "mode": "moving_recorded_scene_with_paper_baseline_overlay",
        "columns": ["CV baseline", "Xu paper baseline", "GT", "Ours"],
        "fps": args.fps,
        "moving_frames": moving_frames,
        "missing_frame_count": missing,
        "items": [
            {
                "source": item["source"],
                "label": item["label"],
                "case": item["case"],
                "sample_id": item["sample_id"],
                "frames_rendered": (
                    min(len(item["frame_paths"]), args.max_frames_per_item)
                    if args.max_frames_per_item > 0
                    else len(item["frame_paths"])
                ),
                "ADE": item["ADE"],
                "FDE": item["FDE"],
                "cv_ADE": item["cv_ADE"],
                "cv_FDE": item["cv_FDE"],
                "xu_ADE": item["xu_ADE"],
                "xu_FDE": item["xu_FDE"],
            }
            for item in all_items
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(all_items)} items")


if __name__ == "__main__":
    main()
