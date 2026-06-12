"""Render main 5-baseline multi-scenario ViZDoom counterfactual rollout video.

Unlike recorded-scene overlays, each column in this video is a separate
ViZDoom branch from the same selected sample pose.

Columns:
- CV baseline
- PointNav/DD-PPO endpoint oracle
- A* pose-graph oracle
- GT
- Ours

The sequence follows the same sample selection as
`demo_main_5baseline_multiscenario_05s`, but renders each method by actually
following its planned local path in ViZDoom. Human-action replay GT and V4
recorded policy GT are explicitly marked as different GT sources.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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

from render_main_5baseline_multiscenario_video import (  # noqa: E402
    ASTAR_COLOR,
    HUMAN_BLOCK_COLOR,
    POINTNAV_COLOR,
    V4_BLOCK_COLOR,
    average_prediction_rows,
    make_oracle_row,
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
    read_json,
    read_jsonl,
    wrap,
)
from render_vizdoom_counterfactual_rollouts import fit_image, load_raw_episode, rollout_method  # noqa: E402
from src.models.navigation_oracles import PoseGraphAStarPlanner  # noqa: E402


METHODS = [
    ("cv", "CV", "recent motion only", CV_COLOR),
    ("pointnav", "PointNav", "GT endpoint oracle", POINTNAV_COLOR),
    ("astar", "A*", "pose graph + endpoint oracle", ASTAR_COLOR),
    ("target", "GT", "future path label", GT_COLOR),
    ("prediction", "Ours", "RGB history + ego-motion", PRED_COLOR),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dataset", type=Path, default=Path("data/wit_vz/processed/wit_vz_sauerkrautlm_human_replay_001"))
    parser.add_argument(
        "--human-ours-predictions",
        type=Path,
        default=Path("reports/demo/human_action_replay_gt_comparison_05s/eval_test/predictions.jsonl"),
    )
    parser.add_argument("--v4-dataset", type=Path, default=Path("data/wit_vz/processed/horizon_sweep_v4_defaults/future_05s"))
    parser.add_argument(
        "--v4-ours-predictions",
        type=Path,
        default=Path("runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_05s/predictions.jsonl"),
    )
    parser.add_argument(
        "--v4-oracle-predictions",
        type=Path,
        default=Path("outputs/navigation_oracle_baselines_v4/predictions_05s.jsonl"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("reports/demo/main_5baseline_multiscenario_05s/selection.json"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/demo/main_5baseline_multiscenario_counterfactual_05s"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/demo/presentation_sequence/demo_main_5baseline_multiscenario_counterfactual_05s.mp4"),
    )
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--screen-width", type=int, default=320)
    parser.add_argument("--screen-height", type=int, default=240)
    parser.add_argument("--hold-last-frames", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--position-threshold", type=float, default=80.0)
    parser.add_argument("--angle-threshold", type=float, default=12.0)
    parser.add_argument("--raw-root-base", action="append", default=[])
    parser.add_argument("--write-branch-videos", action="store_true")
    return parser.parse_args()


def source_id(sample: dict[str, Any]) -> str | None:
    return sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")


def load_selection(path: Path, max_items: int) -> list[dict[str, Any]]:
    data = read_json(path)
    rows = data.get("items", data if isinstance(data, list) else [])
    if max_items > 0:
        rows = rows[:max_items]
    return rows


def v4_row_from_predictions(
    sample: dict[str, Any],
    ours_row: dict[str, Any],
    oracle_row: dict[str, Any],
) -> dict[str, Any]:
    target = oracle_row.get("target") or ours_row.get("target") or sample.get("future_local_path") or []
    prediction = ours_row.get("prediction") or []
    cv_path = oracle_row.get("constant_velocity_prediction") or ours_row.get("constant_velocity_prediction") or []
    pointnav_path = oracle_row.get("pointnav_goal_oracle_prediction") or []
    astar_path = oracle_row.get("astar_oracle_prediction") or []
    horizon = min(len(target), len(prediction), len(cv_path), len(pointnav_path), len(astar_path))
    if horizon <= 0:
        raise ValueError(f"empty paths for sample={sample['sample_id']}")
    target = target[:horizon]
    prediction = prediction[:horizon]
    cv_path = cv_path[:horizon]
    pointnav_path = pointnav_path[:horizon]
    astar_path = astar_path[:horizon]
    return {
        "sample_id": sample["sample_id"],
        "target": target,
        "prediction": prediction,
        "cv": cv_path,
        "pointnav": pointnav_path,
        "astar": astar_path,
        "cv_ADE": float(oracle_row.get("constant_velocity_ADE", 0.0)),
        "cv_FDE": float(oracle_row.get("constant_velocity_FDE", 0.0)),
        "pointnav_ADE": float(oracle_row.get("pointnav_goal_oracle_ADE", 0.0)),
        "pointnav_FDE": float(oracle_row.get("pointnav_goal_oracle_FDE", 0.0)),
        "astar_ADE": float(oracle_row.get("astar_oracle_ADE", 0.0)),
        "astar_FDE": float(oracle_row.get("astar_oracle_FDE", 0.0)),
        "ours_ADE": float(ours_row.get("ADE", 0.0)),
        "ours_FDE": float(ours_row.get("FDE", 0.0)),
    }


def build_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    selection = load_selection(args.selection, args.max_items)
    human_samples = {str(row["sample_id"]): row for row in read_jsonl(args.human_dataset / "samples.jsonl")}
    human_ours = average_prediction_rows(args.human_ours_predictions)
    human_planner = PoseGraphAStarPlanner.from_samples(human_samples.values())

    v4_samples = {str(row["sample_id"]): row for row in read_jsonl(args.v4_dataset / "samples.jsonl")}
    v4_ours = average_prediction_rows(args.v4_ours_predictions)
    v4_oracle = {str(row["sample_id"]): row for row in read_jsonl(args.v4_oracle_predictions)}

    output = []
    for row in selection:
        sample_id = str(row["sample_id"])
        block = str(row.get("block", ""))
        if block.startswith("Human"):
            sample = human_samples[sample_id]
            ours = human_ours[sample_id]
            target = ours.get("target") or sample.get("future_local_path") or []
            prediction = ours.get("prediction") or []
            built = make_oracle_row(sample, target, prediction, ours.get("constant_velocity_prediction"), human_planner)
        else:
            sample = v4_samples[sample_id]
            built = v4_row_from_predictions(sample, v4_ours[sample_id], v4_oracle[sample_id])
        output.append(
            {
                **built,
                "sample": sample,
                "block": block,
                "gt_source": row.get("gt_source", ""),
                "label": row.get("label", ""),
                "case": row.get("case", ""),
            }
        )
    return output


def raw_dirs_for_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    bases = [ROOT, *[Path(value) for value in args.raw_root_base]]
    dataset = args.human_dataset if item["block"].startswith("Human") else args.v4_dataset
    return load_raw_dirs(dataset, bases)


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
    draw.text((x + 16, y + 14), title, fill=color, font=font(27, bold=True))
    draw.text((x + 16, y + 50), wrap(subtitle, 38), fill=MUTED_COLOR, font=font(15))
    frame = fit_image(rgb, (w - 36, 288))
    paste_center(canvas, frame, (x + 18, y + 94, x + w - 18, y + 382))
    plot = draw_path_plot(path, (w - 36, h - 430), color, max_abs, full_path)
    canvas.paste(plot, (x + 18, y + 406))


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
    block_color = HUMAN_BLOCK_COLOR if item["block"].startswith("Human") else V4_BLOCK_COLOR
    header = f"{order:02d}/{total:02d}  REAL rollout / {item['block']} / {item['label']} / {item['case']}"
    draw.text((38, 24), header, fill=TEXT_COLOR, font=font(34, bold=True))
    horizon = max(len(item["target"]), len(item["prediction"]), len(item["cv"]), len(item["pointnav"]), len(item["astar"]))
    metrics = (
        f"sample={item['sample_id']}    t={min(progress + 1, horizon):02d}/{horizon:02d}    "
        f"CV {item['cv_ADE']:.1f}/{item['cv_FDE']:.1f}    "
        f"PointNav {item['pointnav_ADE']:.1f}/{item['pointnav_FDE']:.1f}    "
        f"A* {item['astar_ADE']:.1f}/{item['astar_FDE']:.1f}    "
        f"Ours {item['ours_ADE']:.1f}/{item['ours_FDE']:.1f}"
    )
    draw.text((38, 70), wrap(metrics, 190), fill=MUTED_COLOR, font=font(18))
    warning = (
        f"GT source: {item['gt_source']}. "
        "Each column is a separate ViZDoom branch from the same selected pose. "
        "PointNav/A* are privileged endpoint upper bounds."
    )
    draw.text((38, 108), wrap(warning, 190), fill=block_color, font=font(17, bold=True))

    col_gap = 14
    margin_x = 34
    top = 154
    col_w = (args.width - 2 * margin_x - 4 * col_gap) // 5
    col_h = args.height - top - 34
    max_abs = axis_scale(item["cv"], item["pointnav"], item["astar"], item["target"], item["prediction"])
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


def render_item_rollouts(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw_dirs = raw_dirs_for_item(item, args)
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
                for key, title, subtitle, color in METHODS:
                    if key not in branch_writers:
                        continue
                    branch = render_composite_frame(
                        {**item, "case": f"{item['case']} / branch={title}"},
                        {key: rollouts[key], **rollouts},
                        order,
                        len(rendered_items),
                        progress,
                        args,
                    )
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


def write_reports(args: argparse.Namespace, rendered: list[dict[str, Any]], skipped: list[dict[str, Any]], poster: Path | None) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output": str(args.output),
        "poster": str(poster) if poster else None,
        "mode": "main_5baseline_multiscenario_real_vizdoom_counterfactual_rollout",
        "columns": [title for _key, title, _subtitle, _color in METHODS],
        "success_count": len(rendered),
        "skipped": skipped,
        "items": [
            {
                "sample_id": payload["item"]["sample_id"],
                "block": payload["item"]["block"],
                "gt_source": payload["item"]["gt_source"],
                "label": payload["item"]["label"],
                "case": payload["item"]["case"],
                "scenario": payload["manifest"].get("scenario"),
                "metrics": {
                    "cv_ADE": payload["item"]["cv_ADE"],
                    "pointnav_ADE": payload["item"]["pointnav_ADE"],
                    "astar_ADE": payload["item"]["astar_ADE"],
                    "ours_ADE": payload["item"]["ours_ADE"],
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
    (args.report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Main 5-Baseline Multi-Scenario Counterfactual Video (5s)",
        "",
        "- This is the corrected video where each column is a separate ViZDoom branch.",
        "- Columns: CV, PointNav oracle, A* oracle, GT, Ours.",
        "- Human block GT: human-action replay-derived trajectory.",
        "- V4 block GT: recorded WIT-VZ ViZDoom policy trajectory.",
        "- PointNav and A* are privileged endpoint upper-bound baselines.",
        "",
        f"- Success count: {len(rendered)}",
        f"- Skipped count: {len(skipped)}",
        "",
        "| order | block | label | case | sample_id |",
        "|---:|---|---|---|---|",
    ]
    for idx, payload in enumerate(rendered, start=1):
        item = payload["item"]
        lines.append(f"| {idx} | {item['block']} | {item['label']} | {item['case']} | `{item['sample_id']}` |")
    if skipped:
        lines.extend(["", "## Skipped", "", "| sample_id | reason |", "|---|---|"])
        for row in skipped:
            lines.append(f"| `{row['sample_id']}` | {row['reason']} |")
    (args.report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected = build_items(args)
    rendered = []
    skipped = []
    for item in selected:
        try:
            payload = render_item_rollouts(item, args)
        except Exception as exc:
            skipped.append({"sample_id": item["sample_id"], "block": item["block"], "reason": repr(exc)})
            print(f"skip {item['sample_id']}: {exc}")
            continue
        rendered.append({"item": item, **payload})
    if len(rendered) < 3:
        raise RuntimeError(f"Counterfactual rollout needs at least 3 successful samples, got {len(rendered)}")
    write_video(rendered, args)
    poster = write_poster(args.output)
    write_reports(args, rendered, skipped, poster)
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
