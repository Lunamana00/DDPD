"""Render all selected baselines against human-action replay-derived GT.

Columns:
- CV baseline
- Xu-style pixels-only proxy
- Khaleque-style exploratory proxy
- PointNav/DD-PPO goal oracle
- A* pose-graph oracle
- GT (human replay)
- Ours

The GT path is produced by replaying public human action labels in ViZDoom. It
is not a recovered original human pose trajectory. PointNav and A* are
privileged baselines because they receive the GT future endpoint; A* also uses a
recorded pose graph.
"""

from __future__ import annotations

import argparse
import json
import math
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
    axis_scale,
    clipped_path,
    draw_path_plot,
    font,
    load_raw_dirs,
    load_rgb,
    moving_frame_paths,
    paste_center,
    path_error,
    read_json,
    read_jsonl,
    resolve_frame_path,
    wrap,
)
from src.models.motion import constant_velocity_path  # noqa: E402
from src.models.navigation_oracles import (  # noqa: E402
    PoseGraphAStarPlanner,
    astar_oracle_prediction,
    pointnav_goal_oracle_prediction,
)
from src.models.paper_proxies import (  # noqa: E402
    khaleque_center_random_prediction,
    source_centers_from_train,
    xu_pixels_saliency_prediction,
)
from src.wit_vz.dataset import load_rgb_tensor  # noqa: E402
from src.wit_vz.io import load_json  # noqa: E402


XU_COLOR = (135, 76, 188)
KHALEQUE_COLOR = (214, 133, 38)
POINTNAV_COLOR = (99, 90, 190)
ASTAR_COLOR = (34, 139, 150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/wit_vz/processed/wit_vz_sauerkrautlm_human_replay_001"))
    parser.add_argument(
        "--ours-predictions",
        type=Path,
        default=Path("reports/demo/human_action_replay_gt_comparison_05s/eval_test/predictions.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/demo/human_action_replay_gt_comparison_05s/summary.json"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/demo/human_action_replay_all_baselines_05s"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/demo/presentation_sequence/demo_human_action_replay_all_baselines_05s.mp4"),
    )
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hold-last-frames", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--max-frames-per-item", type=int, default=0)
    parser.add_argument("--astar-cell-size", type=float, default=16.0)
    parser.add_argument("--raw-root-base", action="append", default=[])
    return parser.parse_args()


def mean_metric(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def split_sample_ids(dataset: Path, split: str) -> set[str]:
    splits = load_json(dataset / "splits.json")
    return set(str(item) for item in splits.get(split, []))


def load_selected_ids(summary: Path, limit: int) -> list[tuple[str, str]]:
    data = read_json(summary)
    rows = data if isinstance(data, list) else data.get("selected", [])
    output: list[tuple[str, str]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        output.append((sample_id, str(row.get("case", ""))))
        if 0 < limit <= len(output):
            break
    return output


def build_batch(
    sample: dict[str, Any],
    target: list[list[float]],
    raw_dirs: dict[str, Path],
) -> dict[str, Any]:
    source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
    frames = []
    for rel_path in sample["rgb_history_paths"]:
        frame_path = resolve_frame_path(raw_dirs, rel_path, source_id)
        if frame_path is None or not frame_path.exists():
            raise FileNotFoundError(f"Missing RGB frame for sample={sample['sample_id']}: {rel_path}")
        frames.append(load_rgb_tensor(frame_path, 64))
    return {
        "sample_id": [sample["sample_id"]],
        "ego_history": torch.tensor([sample["relative_egomotion_history"]], dtype=torch.float32),
        "future_path": torch.tensor([target], dtype=torch.float32),
        "current_pose": [sample["current_pose"]],
        "metadata": [sample.get("metadata", {})],
        "source": [sample.get("source", {})],
        "rgb_history": torch.stack(frames, dim=0).unsqueeze(0),
    }


def compute_all_rows(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    samples = {str(row["sample_id"]): row for row in read_jsonl(args.dataset / "samples.jsonl")}
    test_ids = split_sample_ids(args.dataset, "test")
    ours_rows = {str(row["sample_id"]): row for row in read_jsonl(args.ours_predictions)}
    raw_dirs = load_raw_dirs(args.dataset, [ROOT, *[Path(item) for item in args.raw_root_base]])
    source_centers = source_centers_from_train(args.dataset)
    planner = PoseGraphAStarPlanner.from_samples(samples.values(), cell_size=args.astar_cell_size)

    rows: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, list[float]] = {
        "cv_ADE": [],
        "cv_FDE": [],
        "xu_ADE": [],
        "xu_FDE": [],
        "khaleque_ADE": [],
        "khaleque_FDE": [],
        "pointnav_ADE": [],
        "pointnav_FDE": [],
        "astar_ADE": [],
        "astar_FDE": [],
        "ours_ADE": [],
        "ours_FDE": [],
    }

    for sample_id in sorted(test_ids):
        sample = samples.get(sample_id)
        ours = ours_rows.get(sample_id)
        if sample is None or ours is None:
            continue
        target = ours.get("target") or sample.get("future_local_path") or []
        if not target:
            continue
        batch = build_batch(sample, target, raw_dirs)
        with torch.no_grad():
            cv_path = ours.get("constant_velocity_prediction")
            if not cv_path:
                cv_path = constant_velocity_path(batch["ego_history"], len(target))[0].cpu().tolist()
            xu_path = xu_pixels_saliency_prediction(batch)[0].cpu().tolist()
            khaleque_path = khaleque_center_random_prediction(batch, source_centers)[0].cpu().tolist()
            pointnav_path = pointnav_goal_oracle_prediction(batch)[0].cpu().tolist()
            astar_path = astar_oracle_prediction(sample, planner)

        pred_path = ours.get("prediction") or []
        horizon = min(
            len(target),
            len(pred_path),
            len(cv_path),
            len(xu_path),
            len(khaleque_path),
            len(pointnav_path),
            len(astar_path),
        )
        if horizon <= 0:
            continue
        target = target[:horizon]
        pred_path = pred_path[:horizon]
        cv_path = cv_path[:horizon]
        xu_path = xu_path[:horizon]
        khaleque_path = khaleque_path[:horizon]
        pointnav_path = pointnav_path[:horizon]
        astar_path = astar_path[:horizon]

        cv_ade, cv_fde = path_error(cv_path, target)
        xu_ade, xu_fde = path_error(xu_path, target)
        khaleque_ade, khaleque_fde = path_error(khaleque_path, target)
        pointnav_ade, pointnav_fde = path_error(pointnav_path, target)
        astar_ade, astar_fde = path_error(astar_path, target)
        ours_ade, ours_fde = path_error(pred_path, target)

        row = {
            "sample_id": sample_id,
            "target": target,
            "prediction": pred_path,
            "constant_velocity_prediction": cv_path,
            "xu_pixels_only_prediction": xu_path,
            "khaleque_exploratory_prediction": khaleque_path,
            "pointnav_goal_oracle_prediction": pointnav_path,
            "astar_oracle_prediction": astar_path,
            "cv_ADE": cv_ade,
            "cv_FDE": cv_fde,
            "xu_ADE": xu_ade,
            "xu_FDE": xu_fde,
            "khaleque_ADE": khaleque_ade,
            "khaleque_FDE": khaleque_fde,
            "pointnav_ADE": pointnav_ade,
            "pointnav_FDE": pointnav_fde,
            "astar_ADE": astar_ade,
            "astar_FDE": astar_fde,
            "ours_ADE": float(ours.get("ADE", ours_ade)),
            "ours_FDE": float(ours.get("FDE", ours_fde)),
        }
        rows[sample_id] = row
        for key in aggregate:
            aggregate[key].append(float(row[key]))

    metrics = {
        "dataset": args.dataset.as_posix(),
        "split": "test",
        "test_samples": len(rows),
        "horizon_sec": 5,
        "future_steps": len(next(iter(rows.values()))["target"]) if rows else 0,
        "gt_note": "GT is generated by replaying public human action labels in ViZDoom, not by recovering original human pose trajectories.",
        "baseline_notes": {
            "xu_pixels_only": "Paper-adapted screen-only proxy, not exact reproduction.",
            "khaleque_exploratory": "Paper-adapted exploratory proxy, not exact reproduction.",
            "pointnav_goal_oracle": "Privileged: receives GT future endpoint.",
            "astar_oracle": "Privileged: uses GT future endpoint and a recorded pose graph.",
        },
        "metrics": {
            "constant_velocity": {"ADE": mean_metric(aggregate["cv_ADE"]), "FDE": mean_metric(aggregate["cv_FDE"])},
            "xu_pixels_only": {"ADE": mean_metric(aggregate["xu_ADE"]), "FDE": mean_metric(aggregate["xu_FDE"])},
            "khaleque_exploratory": {"ADE": mean_metric(aggregate["khaleque_ADE"]), "FDE": mean_metric(aggregate["khaleque_FDE"])},
            "pointnav_goal_oracle": {"ADE": mean_metric(aggregate["pointnav_ADE"]), "FDE": mean_metric(aggregate["pointnav_FDE"])},
            "astar_oracle": {"ADE": mean_metric(aggregate["astar_ADE"]), "FDE": mean_metric(aggregate["astar_FDE"])},
            "ours": {"ADE": mean_metric(aggregate["ours_ADE"]), "FDE": mean_metric(aggregate["ours_FDE"])},
        },
    }
    return rows, metrics


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
    draw.text((x + 12, y + 12), title, fill=color, font=font(21, bold=True))
    draw.text((x + 12, y + 42), wrap(subtitle, 30), fill=MUTED_COLOR, font=font(12))
    frame_box = (x + 12, y + 86, x + w - 12, y + 338)
    paste_center(canvas, rgb, frame_box)
    plot = draw_path_plot(path, (w - 24, h - 374), color, max_abs, full_path)
    canvas.paste(plot, (x + 12, y + 356))


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
    header = f"{order:02d}/{total:02d}  Human replay GT vs all baselines / defend_the_center / {item['case']}"
    draw.text((34, 24), header, fill=TEXT_COLOR, font=font(32, bold=True))
    horizon = max(len(item["target"]), len(item["prediction"]), len(item["cv"]))
    metrics = (
        f"sample={item['sample_id']}    t={min(progress + 1, horizon):02d}/{horizon:02d}    "
        f"CV {item['cv_ADE']:.1f}    Xu {item['xu_ADE']:.1f}    Khaleque {item['khaleque_ADE']:.1f}    "
        f"PointNav {item['pointnav_ADE']:.1f}    A* {item['astar_ADE']:.1f}    Ours {item['ours_ADE']:.1f}"
    )
    draw.text((34, 68), wrap(metrics, 210), fill=MUTED_COLOR, font=font(17))
    warning = (
        "GT = human-action replay-derived future path. "
        "PointNav/A* are privileged endpoint baselines; Xu/Khaleque are paper-adapted proxies."
    )
    draw.text((34, 104), warning, fill=(142, 82, 22), font=font(16, bold=True))

    col_gap = 10
    margin_x = 24
    top = 142
    col_w = (width - 2 * margin_x - 6 * col_gap) // 7
    col_h = height - top - 28
    rgb = load_rgb(frame_path, (col_w - 28, 252))
    max_abs = axis_scale(
        item["cv"],
        item["xu"],
        item["khaleque"],
        item["pointnav"],
        item["astar"],
        item["target"],
        item["prediction"],
    )
    shown = progress + 1
    specs = [
        ("CV", "recent motion only", "cv", CV_COLOR),
        ("Xu-style", "pixels-only proxy", "xu", XU_COLOR),
        ("Khaleque", "exploratory proxy", "khaleque", KHALEQUE_COLOR),
        ("PointNav", "GT endpoint oracle", "pointnav", POINTNAV_COLOR),
        ("A*", "pose graph oracle", "astar", ASTAR_COLOR),
        ("GT", "human replay path", "target", GT_COLOR),
        ("Ours", "RGB history + ego", "prediction", PRED_COLOR),
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


def write_poster(video_path: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read poster frame from {video_path}")
    poster = video_path.with_suffix(".png")
    cv2.imwrite(str(poster), frame)
    return poster


def write_reports(args: argparse.Namespace, rows: dict[str, dict[str, Any]], metrics: dict[str, Any], items: list[dict[str, Any]]) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.report_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(rows):
            handle.write(json.dumps(rows[sample_id], ensure_ascii=False) + "\n")
    (args.report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metric_rows = [
        ("CV", "constant_velocity"),
        ("Xu-style", "xu_pixels_only"),
        ("Khaleque-style", "khaleque_exploratory"),
        ("PointNav/DD-PPO oracle", "pointnav_goal_oracle"),
        ("A* oracle", "astar_oracle"),
        ("Ours", "ours"),
    ]
    lines = [
        "# Human Replay GT All Baselines (5s)",
        "",
        "- Dataset: `data/wit_vz/processed/wit_vz_sauerkrautlm_human_replay_001`",
        "- Split: test",
        "- GT: ViZDoom trajectory generated by replaying SauerkrautLM public human action labels.",
        "- Caveat: replay-derived GT, not recovered original human pose trajectory.",
        "- PointNav/DD-PPO and A* are privileged endpoint baselines.",
        "- Xu-style and Khaleque-style are paper-adapted proxies, not exact reproductions.",
        "",
        "## Test Metrics",
        "",
        "| method | ADE | FDE | note |",
        "|---|---:|---:|---|",
    ]
    notes = {
        "constant_velocity": "recent ego-motion only",
        "xu_pixels_only": "screen-only paper proxy",
        "khaleque_exploratory": "exploratory paper proxy",
        "pointnav_goal_oracle": "uses GT endpoint",
        "astar_oracle": "uses GT endpoint + pose graph",
        "ours": "RGB history + ego-motion",
    }
    for label, key in metric_rows:
        item = metrics["metrics"][key]
        lines.append(f"| {label} | {item['ADE']:.2f} | {item['FDE']:.2f} | {notes[key]} |")
    lines.extend(["", "## Video Samples", "", "| case | sample_id | Ours ADE | CV ADE | Xu ADE | Khaleque ADE | PointNav ADE | A* ADE |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for item in items:
        lines.append(
            f"| {item['case']} | `{item['sample_id']}` | {item['ours_ADE']:.2f} | {item['cv_ADE']:.2f} | "
            f"{item['xu_ADE']:.2f} | {item['khaleque_ADE']:.2f} | {item['pointnav_ADE']:.2f} | {item['astar_ADE']:.2f} |"
        )
    (args.report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, metrics = compute_all_rows(args)
    if not rows:
        raise RuntimeError("No predictions could be computed.")

    samples = {str(row["sample_id"]): row for row in read_jsonl(args.dataset / "samples.jsonl")}
    raw_dirs = load_raw_dirs(args.dataset, [ROOT, *[Path(item) for item in args.raw_root_base]])
    selected = load_selected_ids(args.summary, args.max_samples)
    if not selected:
        selected = [(sample_id, "selected") for sample_id in list(rows)[: args.max_samples]]

    items: list[dict[str, Any]] = []
    for sample_id, case in selected:
        sample = samples.get(sample_id)
        row = rows.get(sample_id)
        if sample is None or row is None:
            continue
        source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
        frame_path = resolve_frame_path(raw_dirs, sample["rgb_history_paths"][-1], source_id)
        frame_paths = moving_frame_paths(sample, raw_dirs, source_id, len(row["target"]))
        items.append(
            {
                "sample_id": sample_id,
                "case": case or "selected",
                "frame_path": frame_path,
                "frame_paths": frame_paths,
                "target": row["target"],
                "prediction": row["prediction"],
                "cv": row["constant_velocity_prediction"],
                "xu": row["xu_pixels_only_prediction"],
                "khaleque": row["khaleque_exploratory_prediction"],
                "pointnav": row["pointnav_goal_oracle_prediction"],
                "astar": row["astar_oracle_prediction"],
                "cv_ADE": row["cv_ADE"],
                "cv_FDE": row["cv_FDE"],
                "xu_ADE": row["xu_ADE"],
                "xu_FDE": row["xu_FDE"],
                "khaleque_ADE": row["khaleque_ADE"],
                "khaleque_FDE": row["khaleque_FDE"],
                "pointnav_ADE": row["pointnav_ADE"],
                "pointnav_FDE": row["pointnav_FDE"],
                "astar_ADE": row["astar_ADE"],
                "astar_FDE": row["astar_FDE"],
                "ours_ADE": row["ours_ADE"],
                "ours_FDE": row["ours_FDE"],
            }
        )
    if not items:
        raise RuntimeError("No renderable items found.")

    write_video(items, args)
    poster = write_poster(args.output)
    write_reports(args, rows, metrics, items)
    moving_frames = sum(
        min(len(item["frame_paths"]), args.max_frames_per_item) if args.max_frames_per_item > 0 else len(item["frame_paths"])
        for item in items
    )
    missing = sum(1 for item in items for path in item["frame_paths"] if path is None or not path.exists())
    manifest = {
        "output": str(args.output),
        "poster": str(poster),
        "mode": "human_replay_all_baselines_recorded_scene_overlay",
        "columns": ["CV", "Xu-style", "Khaleque", "PointNav", "A*", "GT", "Ours"],
        "fps": args.fps,
        "moving_frames": moving_frames,
        "missing_frame_count": missing,
        "metrics": metrics["metrics"],
        "items": [
            {
                "sample_id": item["sample_id"],
                "case": item["case"],
                "frames_rendered": (
                    min(len(item["frame_paths"]), args.max_frames_per_item)
                    if args.max_frames_per_item > 0
                    else len(item["frame_paths"])
                ),
                "cv_ADE": item["cv_ADE"],
                "xu_ADE": item["xu_ADE"],
                "khaleque_ADE": item["khaleque_ADE"],
                "pointnav_ADE": item["pointnav_ADE"],
                "astar_ADE": item["astar_ADE"],
                "ours_ADE": item["ours_ADE"],
            }
            for item in items
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "poster": str(poster), "items": len(items)}, indent=2))


if __name__ == "__main__":
    main()
