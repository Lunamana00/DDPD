"""Render 5-column navigation oracle baseline overlay video.

Columns:
- CV baseline
- PointNav/DD-PPO goal oracle
- A* pose-graph oracle
- GT
- Ours

PointNav and A* are privileged baselines. They use the GT future endpoint, and
A* additionally uses a recorded pose graph. This video is intended to visualize
that upper-bound comparison, not to present them as input-matched competitors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
    axis_scale,
    clipped_path,
    draw_path_plot,
    font,
    load_raw_dirs,
    load_rgb,
    metadata_label,
    moving_frame_paths,
    paste_center,
    path_error,
    read_jsonl,
    resolve_frame_path,
    wrap,
)


POINTNAV_COLOR = (135, 76, 188)
ASTAR_COLOR = (34, 139, 150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/wit_vz/processed/horizon_sweep_v4_defaults/future_05s"))
    parser.add_argument(
        "--ours-predictions",
        type=Path,
        default=Path("runs/episodic_memory_ablation_v4/seed_7/05s/long_attention_no_ego/predictions.jsonl"),
    )
    parser.add_argument(
        "--oracle-predictions",
        type=Path,
        default=Path("outputs/navigation_oracle_baselines_v4/predictions_05s.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/demo/presentation_sequence/demo_navigation_oracle_overlay_05s.mp4"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hold-last-frames", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--max-frames-per-item", type=int, default=0)
    parser.add_argument(
        "--min-target-extent",
        type=float,
        default=20.0,
        help="Skip near-stationary samples whose GT path never moves this far from the origin.",
    )
    parser.add_argument("--raw-root-base", action="append", default=[])
    return parser.parse_args()


def average_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Read prediction rows, averaging duplicate sample ids if present."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[str(row["sample_id"])].append(row)
    output: dict[str, dict[str, Any]] = {}
    for sample_id, rows in grouped.items():
        if len(rows) == 1:
            output[sample_id] = rows[0]
            continue
        first = dict(rows[0])
        predictions = [torch.tensor(row["prediction"], dtype=torch.float32) for row in rows if "prediction" in row]
        if predictions:
            first["prediction"] = torch.stack(predictions, dim=0).mean(dim=0).tolist()
        if "target" in first:
            target = torch.tensor(first["target"], dtype=torch.float32)
            pred = torch.tensor(first["prediction"], dtype=torch.float32)
            errors = torch.linalg.norm(pred - target, dim=-1)
            first["ADE"] = float(errors.mean().item())
            first["FDE"] = float(errors[-1].item())
        output[sample_id] = first
    return output


def scenario_key(sample: dict[str, Any], fallback: str = "") -> str:
    metadata = sample.get("metadata", {})
    return str(metadata.get("scenario") or metadata.get("source_dataset") or fallback or "unknown")


def choose_diverse(rows: list[dict[str, Any]], count: int, used_scenarios: set[str]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for row in rows:
        scenario = row["scenario"]
        if scenario in used_scenarios and len(chosen) < max(1, count - 1):
            continue
        chosen.append(row)
        used_scenarios.add(scenario)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for row in rows:
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= count:
                break
    return chosen


def select_demo_rows(candidates: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    per_case = max(1, math.ceil(max_samples / 3))
    used: set[str] = set()

    easy_pool = sorted(candidates, key=lambda row: (row["cv_ADE"], row["ours_ADE"]))
    hard_pool = sorted(
        candidates,
        key=lambda row: (row["cv_ADE"] - row["ours_ADE"], row["cv_ADE"]),
        reverse=True,
    )
    failure_pool = sorted(
        candidates,
        key=lambda row: (row["ours_ADE"] - row["cv_ADE"], row["ours_ADE"]),
        reverse=True,
    )

    easy = choose_diverse(easy_pool, per_case, used)
    hard = choose_diverse(hard_pool, per_case, used)
    failure = choose_diverse(failure_pool, per_case, used)

    selected: list[dict[str, Any]] = []
    for idx in range(max(len(easy), len(hard), len(failure))):
        for case, rows in (("easy", easy), ("hard", hard), ("failure", failure)):
            if idx < len(rows) and rows[idx] not in selected:
                item = dict(rows[idx])
                item["case"] = case
                selected.append(item)
            if len(selected) >= max_samples:
                return selected
    return selected[:max_samples]


def load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = {str(row["sample_id"]): row for row in read_jsonl(args.dataset / "samples.jsonl")}
    ours = average_prediction_rows(args.ours_predictions)
    oracle = {str(row["sample_id"]): row for row in read_jsonl(args.oracle_predictions)}
    raw_dirs = load_raw_dirs(args.dataset, [ROOT, *[Path(item) for item in args.raw_root_base]])

    candidates: list[dict[str, Any]] = []
    for sample_id, sample in samples.items():
        ours_row = ours.get(sample_id)
        oracle_row = oracle.get(sample_id)
        if ours_row is None or oracle_row is None:
            continue
        target = oracle_row.get("target") or ours_row.get("target") or sample.get("future_local_path") or []
        pred_path = ours_row.get("prediction") or []
        cv_path = oracle_row.get("constant_velocity_prediction") or ours_row.get("constant_velocity_prediction") or []
        pointnav_path = oracle_row.get("pointnav_goal_oracle_prediction") or []
        astar_path = oracle_row.get("astar_oracle_prediction") or []
        if not all([target, pred_path, cv_path, pointnav_path, astar_path]):
            continue
        horizon_length = min(len(target), len(pred_path), len(cv_path), len(pointnav_path), len(astar_path))
        if horizon_length <= 0:
            continue
        target_extent = max(math.hypot(float(point[0]), float(point[1])) for point in target[:horizon_length])
        if target_extent < float(args.min_target_extent):
            continue
        ours_ade, ours_fde = path_error(pred_path, target)
        cv_ade, cv_fde = path_error(cv_path, target)
        pointnav_ade, pointnav_fde = path_error(pointnav_path, target)
        astar_ade, astar_fde = path_error(astar_path, target)
        candidates.append(
            {
                "sample_id": sample_id,
                "sample": sample,
                "source": "ViZDoom 5s",
                "label": metadata_label(sample, scenario_key(sample)),
                "scenario": scenario_key(sample),
                "target_extent": target_extent,
                "target": target[:horizon_length],
                "prediction": pred_path[:horizon_length],
                "cv": cv_path[:horizon_length],
                "pointnav": pointnav_path[:horizon_length],
                "astar": astar_path[:horizon_length],
                "ours_ADE": float(ours_row.get("ADE", ours_ade)),
                "ours_FDE": float(ours_row.get("FDE", ours_fde)),
                "cv_ADE": float(oracle_row.get("constant_velocity_ADE", cv_ade)),
                "cv_FDE": float(oracle_row.get("constant_velocity_FDE", cv_fde)),
                "pointnav_ADE": float(oracle_row.get("pointnav_goal_oracle_ADE", pointnav_ade)),
                "pointnav_FDE": float(oracle_row.get("pointnav_goal_oracle_FDE", pointnav_fde)),
                "astar_ADE": float(oracle_row.get("astar_oracle_ADE", astar_ade)),
                "astar_FDE": float(oracle_row.get("astar_oracle_FDE", astar_fde)),
            }
        )

    selected = select_demo_rows(candidates, args.max_samples)
    items: list[dict[str, Any]] = []
    for row in selected:
        sample = row["sample"]
        source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
        frame_path = resolve_frame_path(raw_dirs, sample["rgb_history_paths"][-1], source_id)
        frame_paths = moving_frame_paths(sample, raw_dirs, source_id, len(row["target"]))
        row = dict(row)
        row["frame_path"] = frame_path
        row["frame_paths"] = frame_paths
        items.append(row)
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
    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill="white", outline=(216, 220, 226), width=2)
    draw.text((x + 12, y + 12), title, fill=color, font=font(20, bold=True))
    draw.text((x + 12, y + 40), wrap(subtitle, 31), fill=MUTED_COLOR, font=font(12))
    frame_box = (x + 12, y + 84, x + w - 12, y + 322)
    paste_center(canvas, rgb, frame_box)
    plot = draw_path_plot(path, (w - 24, h - 356), color, max_abs, full_path)
    canvas.paste(plot, (x + 12, y + 338))


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
    header = f"{order:02d}/{total:02d}  Privileged navigation baselines vs visual trajectory prediction / {item['label']} / {item['case']}"
    draw.text((34, 24), header, fill=TEXT_COLOR, font=font(28, bold=True))
    horizon = max(len(item["target"]), len(item["prediction"]), len(item["cv"]), len(item["pointnav"]), len(item["astar"]))
    metrics = (
        f"sample={item['sample_id']}    t={min(progress + 1, horizon):02d}/{horizon:02d}    "
        f"CV {item['cv_ADE']:.1f}/{item['cv_FDE']:.1f}    "
        f"PointNav oracle {item['pointnav_ADE']:.1f}/{item['pointnav_FDE']:.1f}    "
        f"A* oracle {item['astar_ADE']:.1f}/{item['astar_FDE']:.1f}    "
        f"Ours {item['ours_ADE']:.1f}/{item['ours_FDE']:.1f}"
    )
    draw.text((34, 62), wrap(metrics, 155), fill=MUTED_COLOR, font=font(16))
    warning = "PointNav and A* use privileged information: GT endpoint; A* also uses a recorded pose graph. Ours uses RGB history + ego-motion only."
    draw.text((34, 96), warning, fill=(142, 82, 22), font=font(15, bold=True))

    col_gap = 12
    margin_x = 28
    top = 132
    col_w = (width - 2 * margin_x - 4 * col_gap) // 5
    col_h = height - top - 28
    rgb = load_rgb(frame_path, (col_w - 28, 238))
    max_abs = axis_scale(item["cv"], item["pointnav"], item["astar"], item["target"], item["prediction"])
    shown = progress + 1
    specs = [
        ("CV baseline", "recent motion only", "cv", CV_COLOR),
        ("PointNav oracle", "given GT endpoint", "pointnav", POINTNAV_COLOR),
        ("A* oracle", "pose graph + GT endpoint", "astar", ASTAR_COLOR),
        ("GT", "recorded future path", "target", GT_COLOR),
        ("Ours", "RGB history + ego-motion", "prediction", PRED_COLOR),
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


def write_poster(video_path: Path) -> Path | None:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    poster = video_path.with_suffix(".png")
    cv2.imwrite(str(poster), frame)
    return poster


def main() -> None:
    args = parse_args()
    items = load_items(args)
    if not items:
        raise RuntimeError("No renderable navigation oracle overlay items found.")
    write_video(items, args)
    poster = write_poster(args.output)
    moving_frames = sum(
        min(len(item["frame_paths"]), args.max_frames_per_item) if args.max_frames_per_item > 0 else len(item["frame_paths"])
        for item in items
    )
    missing = sum(1 for item in items for path in item["frame_paths"] if path is None or not path.exists())
    manifest = {
        "output": str(args.output),
        "poster": str(poster) if poster else None,
        "mode": "navigation_oracle_5column_recorded_scene_overlay",
        "columns": ["CV baseline", "PointNav oracle", "A* oracle", "GT", "Ours"],
        "fps": args.fps,
        "moving_frames": moving_frames,
        "missing_frame_count": missing,
        "items": [
            {
                "sample_id": item["sample_id"],
                "label": item["label"],
                "case": item["case"],
                "scenario": item["scenario"],
                "target_extent": item["target_extent"],
                "frames_rendered": (
                    min(len(item["frame_paths"]), args.max_frames_per_item)
                    if args.max_frames_per_item > 0
                    else len(item["frame_paths"])
                ),
                "cv_ADE": item["cv_ADE"],
                "cv_FDE": item["cv_FDE"],
                "pointnav_ADE": item["pointnav_ADE"],
                "pointnav_FDE": item["pointnav_FDE"],
                "astar_ADE": item["astar_ADE"],
                "astar_FDE": item["astar_FDE"],
                "ours_ADE": item["ours_ADE"],
                "ours_FDE": item["ours_FDE"],
            }
            for item in items
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "poster": str(poster) if poster else None, "items": len(items)}, indent=2))


if __name__ == "__main__":
    main()
