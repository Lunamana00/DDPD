"""Render real ViZDoom counterfactual rollouts for navigation oracle baselines.

Unlike recorded-scene overlays, each column in this video is a separate
simulator branch from the same start pose. The branches follow:

- CV baseline
- PointNav/DD-PPO goal oracle
- A* pose-graph oracle
- GT
- Ours

PointNav and A* remain privileged baselines because their planned paths use the
GT future endpoint, and A* additionally uses a recorded pose graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render_navigation_oracle_overlay_video import (  # noqa: E402
    ASTAR_COLOR,
    POINTNAV_COLOR,
    load_items,
)
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
    paste_center,
    wrap,
)
from render_vizdoom_counterfactual_rollouts import (  # noqa: E402
    fit_image,
    load_raw_episode,
    rollout_method,
)


METHODS = [
    ("cv", "CV baseline", "recent motion only", CV_COLOR),
    ("pointnav", "PointNav oracle", "given GT endpoint", POINTNAV_COLOR),
    ("astar", "A* oracle", "pose graph + GT endpoint", ASTAR_COLOR),
    ("target", "GT", "recorded future path", GT_COLOR),
    ("prediction", "Ours", "RGB history + ego-motion", PRED_COLOR),
]


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
    parser.add_argument("--output", type=Path, default=Path("reports/demo/presentation_sequence/demo_navigation_oracle_counterfactual_rollout_05s.mp4"))
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--min-target-extent", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--screen-width", type=int, default=320)
    parser.add_argument("--screen-height", type=int, default=240)
    parser.add_argument("--hold-last-frames", type=int, default=8)
    parser.add_argument("--position-threshold", type=float, default=5.0)
    parser.add_argument("--angle-threshold", type=float, default=10.0)
    parser.add_argument("--raw-root-base", action="append", default=[])
    parser.add_argument("--write-branch-videos", action="store_true")
    return parser.parse_args()


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
    frame = fit_image(rgb, (w - 24, 238))
    paste_center(canvas, frame, (x + 12, y + 84, x + w - 12, y + 322))
    plot = draw_path_plot(path, (w - 24, h - 356), color, max_abs, full_path)
    canvas.paste(plot, (x + 12, y + 338))


def render_composite_frame(
    item: dict[str, Any],
    rollouts: dict[str, dict[str, Any]],
    order: int,
    total: int,
    progress: int,
    args: argparse.Namespace,
) -> Image.Image:
    canvas = Image.new("RGB", (args.width, args.height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    header = f"{order:02d}/{total:02d}  Real ViZDoom counterfactual rollout / {item['label']} / {item['case']}"
    draw.text((34, 24), header, fill=TEXT_COLOR, font=font(28, bold=True))
    metrics = (
        f"sample={item['sample_id']}    t={progress + 1:02d}    "
        f"CV {item['cv_ADE']:.1f}/{item['cv_FDE']:.1f}    "
        f"PointNav oracle {item['pointnav_ADE']:.1f}/{item['pointnav_FDE']:.1f}    "
        f"A* oracle {item['astar_ADE']:.1f}/{item['astar_FDE']:.1f}    "
        f"Ours {item['ours_ADE']:.1f}/{item['ours_FDE']:.1f}"
    )
    draw.text((34, 62), wrap(metrics, 155), fill=MUTED_COLOR, font=font(16))
    warning = "Each column is a separate simulator branch. PointNav/A* are privileged upper bounds; Ours uses RGB history + ego-motion only."
    draw.text((34, 96), warning, fill=(142, 82, 22), font=font(15, bold=True))

    col_gap = 12
    margin_x = 28
    top = 132
    col_w = (args.width - 2 * margin_x - 4 * col_gap) // 5
    col_h = args.height - top - 28
    max_abs = axis_scale(*(item[key] for key, *_ in METHODS))
    for idx, (key, title, subtitle, color) in enumerate(METHODS):
        frames = rollouts[key]["frames"]
        rgb = frames[min(progress, len(frames) - 1)] if frames else Image.new("RGB", (320, 240), (230, 234, 238))
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
            clipped_path(item[key], progress + 1),
            item[key],
            color,
            max_abs,
        )
    return canvas


def render_branch_frame(
    item: dict[str, Any],
    rollouts: dict[str, dict[str, Any]],
    method: tuple[str, str, str, tuple[int, int, int]],
    order: int,
    total: int,
    progress: int,
    args: argparse.Namespace,
) -> Image.Image:
    key, title, subtitle, color = method
    canvas = Image.new("RGB", (args.width, args.height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    header = f"{order:02d}/{total:02d}  {title} / real ViZDoom branch / {item['label']} / {item['case']}"
    draw.text((34, 26), header, fill=color, font=font(30, bold=True))
    draw.text((34, 72), f"{subtitle}    sample={item['sample_id']}    t={progress + 1:02d}", fill=MUTED_COLOR, font=font(18))
    frames = rollouts[key]["frames"]
    rgb = frames[min(progress, len(frames) - 1)] if frames else Image.new("RGB", (320, 240), (230, 234, 238))
    frame = fit_image(rgb, (args.width - 80, args.height - 150))
    paste_center(canvas, frame, (40, 120, args.width - 40, args.height - 30))
    return canvas


def render_item_rollouts(item: dict[str, Any], raw_dirs: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    _raw_root, manifest, steps = load_raw_episode(item["sample"], raw_dirs)
    rollouts = {}
    for key, _title, _subtitle, _color in METHODS:
        rollouts[key] = rollout_method(manifest, steps, item["sample"], item[key], args)
    return {"manifest": manifest, "rollouts": rollouts}


def write_video(rendered_items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")
    branch_writers: dict[str, Any] = {}
    if args.write_branch_videos:
        for key, *_ in METHODS:
            path = args.output.with_name(f"{args.output.stem}_{key}.mp4")
            branch_writers[key] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
            if not branch_writers[key].isOpened():
                raise RuntimeError(f"Could not open writer: {path}")
    try:
        for order, payload in enumerate(rendered_items, start=1):
            item = payload["item"]
            rollouts = payload["rollouts"]
            max_frames = max(len(rollouts[key]["frames"]) for key, *_ in METHODS)
            last_frame = None
            last_branch: dict[str, np.ndarray] = {}
            for progress in range(max_frames):
                frame = render_composite_frame(item, rollouts, order, len(rendered_items), progress, args)
                last_frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                writer.write(last_frame)
                for method in METHODS:
                    key = method[0]
                    if key in branch_writers:
                        branch = render_branch_frame(item, rollouts, method, order, len(rendered_items), progress, args)
                        last_branch[key] = cv2.cvtColor(np.asarray(branch), cv2.COLOR_RGB2BGR)
                        branch_writers[key].write(last_branch[key])
            if last_frame is not None:
                for _ in range(max(0, args.hold_last_frames)):
                    writer.write(last_frame)
                    for key, branch_writer in branch_writers.items():
                        if key in last_branch:
                            branch_writer.write(last_branch[key])
    finally:
        writer.release()
        for branch_writer in branch_writers.values():
            branch_writer.release()


def write_poster(video_path: Path) -> Path | None:
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, max(0, frame_count // 2)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    poster = video_path.with_suffix(".png")
    cv2.imwrite(str(poster), frame)
    return poster


def main() -> None:
    args = parse_args()
    raw_dirs = load_raw_dirs(args.dataset, [ROOT, *[Path(item) for item in args.raw_root_base]])
    selected = load_items(args)
    rendered = []
    skipped = []
    for item in selected:
        try:
            payload = render_item_rollouts(item, raw_dirs, args)
        except Exception as exc:
            skipped.append({"sample_id": item["sample_id"], "reason": repr(exc)})
            print(f"skip {item['sample_id']}: {exc}")
            continue
        rendered.append({"item": item, **payload})
    if len(rendered) < 3:
        raise RuntimeError(f"Counterfactual rollout needs at least 3 successful samples, got {len(rendered)}")
    write_video(rendered, args)
    poster = write_poster(args.output)
    manifest = {
        "output": str(args.output),
        "poster": str(poster) if poster else None,
        "mode": "navigation_oracle_real_vizdoom_counterfactual_rollout",
        "columns": [title for _key, title, _subtitle, _color in METHODS],
        "branch_videos": {
            key: str(args.output.with_name(f"{args.output.stem}_{key}.mp4")) for key, *_ in METHODS if args.write_branch_videos
        },
        "success_count": len(rendered),
        "skipped": skipped,
        "items": [
            {
                "sample_id": payload["item"]["sample_id"],
                "label": payload["item"]["label"],
                "case": payload["item"]["case"],
                "scenario": payload["manifest"].get("scenario"),
                "metrics": {
                    "cv_ADE": payload["item"]["cv_ADE"],
                    "cv_FDE": payload["item"]["cv_FDE"],
                    "pointnav_ADE": payload["item"]["pointnav_ADE"],
                    "pointnav_FDE": payload["item"]["pointnav_FDE"],
                    "astar_ADE": payload["item"]["astar_ADE"],
                    "astar_FDE": payload["item"]["astar_FDE"],
                    "ours_ADE": payload["item"]["ours_ADE"],
                    "ours_FDE": payload["item"]["ours_FDE"],
                },
                "rollouts": {
                    key: {
                        "frames": len(payload["rollouts"][key]["frames"]),
                        "start_info": payload["rollouts"][key]["start_info"],
                    }
                    for key, *_ in METHODS
                },
            }
            for payload in rendered
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "poster": str(poster) if poster else None,
                "success_count": len(rendered),
                "skipped": len(skipped),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
