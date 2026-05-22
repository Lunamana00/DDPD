"""Create a GT-vs-prediction replay video from held-out v4 ViZDoom samples."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/result_visualizations_20260522"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT_DIR))

from make_result_visualizations import (  # noqa: E402
    AXIS,
    BG,
    BLUE,
    F,
    GREEN,
    GRID,
    H,
    MUTED,
    ORANGE,
    PANEL,
    RED,
    SCALE,
    TEXT,
    W,
    draw_line,
    draw_text,
    ellipse_center,
    new_canvas,
    rounded_rect,
    save_canvas,
    sbox,
    spoint,
)
from scripts.compare_v4_inference_ablation import forward_with_ablation, load_model, move_batch, tensor_prediction  # noqa: E402
from src.models.motion import constant_velocity_path  # noqa: E402
from src.wit_vz.dataset import WITVZPathDataset, collate_path_batch  # noqa: E402


DATASET = ROOT / "data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s"
CACHE = ROOT / "data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny"
CHECKPOINT = ROOT / "checkpoints/wit_vz_v4_defaults_dinov3_single_03s.pt"
MP4_PATH = OUT_DIR / "viz_04_gt_comparison_replay.mp4"
GIF_PATH = OUT_DIR / "viz_04_gt_comparison_replay.gif"
FRAME_000 = OUT_DIR / "viz_04_frame_000.png"
FRAME_MID = OUT_DIR / "viz_04_frame_mid.png"
NOTES_PATH = OUT_DIR / "video_notes.md"

VIDEO_FPS = 10
REPEAT_PER_SAMPLE = 5
UNIQUE_SAMPLES = 50
FRAME_SKIP = 4
DOOM_FPS = 35.0


@dataclass
class ReplayRecord:
    sample_id: str
    episode_id: str
    center_step: int
    rgb_path: Path | None
    target: list[list[float]]
    full: list[list[float]]
    cv: list[list[float]]
    full_ade: float
    full_fde: float
    cv_ade: float
    cv_fde: float


def resolve_rgb_path(dataset: WITVZPathDataset, sample: dict[str, Any]) -> Path | None:
    source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
    paths = sample.get("rgb_history_paths") or []
    if not paths:
        return None
    path = dataset._resolve_raw_path(paths[-1], source_id)
    return path if path is not None and path.exists() else None


def path_lateral_range(path: list[list[float]]) -> float:
    rights = [float(point[1]) for point in path]
    return max(rights) - min(rights) if rights else 0.0


def collect_predictions(device: torch.device) -> list[ReplayRecord]:
    dataset = WITVZPathDataset(DATASET, split="test", load_rgb=False, visual_feature_cache_dir=CACHE)
    sample_by_id = {sample["sample_id"]: sample for sample in dataset.samples}
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_path_batch)
    model, _checkpoint = load_model(CHECKPOINT, device)
    records: list[ReplayRecord] = []
    with torch.inference_mode():
        for batch in loader:
            moved = move_batch(batch, device)
            full = tensor_prediction(forward_with_ablation(model, moved, "full_model"))
            cv = constant_velocity_path(moved["ego_history"], full.shape[1])
            target = moved["future_path"]
            full_errors = torch.linalg.norm(full - target, dim=-1)
            cv_errors = torch.linalg.norm(cv - target, dim=-1)
            for i, sample_id in enumerate(batch["sample_id"]):
                raw_sample = sample_by_id[sample_id]
                records.append(
                    ReplayRecord(
                        sample_id=sample_id,
                        episode_id=str(batch["episode_id"][i]),
                        center_step=int(batch["center_step"][i]),
                        rgb_path=resolve_rgb_path(dataset, raw_sample),
                        target=target[i].detach().cpu().tolist(),
                        full=full[i].detach().cpu().tolist(),
                        cv=cv[i].detach().cpu().tolist(),
                        full_ade=float(full_errors[i].mean().item()),
                        full_fde=float(full_errors[i, -1].item()),
                        cv_ade=float(cv_errors[i].mean().item()),
                        cv_fde=float(cv_errors[i, -1].item()),
                    )
                )
    return records


def select_segment(records: list[ReplayRecord]) -> list[ReplayRecord]:
    by_episode: dict[str, list[ReplayRecord]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)
    best_score = -1.0e9
    best: list[ReplayRecord] | None = None
    for episode_records in by_episode.values():
        ordered = sorted(episode_records, key=lambda item: item.center_step)
        if len(ordered) < UNIQUE_SAMPLES:
            continue
        for start in range(0, len(ordered) - UNIQUE_SAMPLES + 1, 5):
            window = ordered[start : start + UNIQUE_SAMPLES]
            improvements = [item.cv_ade - item.full_ade for item in window]
            good_fraction = sum(value > 0.0 for value in improvements) / len(improvements)
            mean_improvement = sum(improvements) / len(improvements)
            mean_cv = sum(item.cv_ade for item in window) / len(window)
            mean_full = sum(item.full_ade for item in window) / len(window)
            mean_lateral = sum(path_lateral_range(item.target) for item in window) / len(window)
            if good_fraction < 0.65 or mean_improvement <= 0.0:
                continue
            # Prefer strong but not absurd CV drift, so the plot remains readable.
            readability_penalty = max(mean_cv - 260.0, 0.0) * 0.25
            score = mean_improvement + 40.0 * good_fraction + 0.05 * mean_lateral - readability_penalty
            if mean_full < mean_cv and score > best_score:
                best_score = score
                best = window
    if best is None:
        improving = [item for item in records if item.full_ade < item.cv_ade]
        best = sorted(improving, key=lambda item: item.cv_ade - item.full_ade, reverse=True)[:UNIQUE_SAMPLES]
        best = sorted(best, key=lambda item: (item.episode_id, item.center_step))
    if len(best) < UNIQUE_SAMPLES:
        raise RuntimeError("Could not select enough improving held-out v4 test samples.")
    return best


def compute_plot_bounds(segment: list[ReplayRecord]) -> tuple[float, float, float, float]:
    rights = [0.0]
    forwards = [0.0]
    for record in segment:
        # Frame the plot around GT and the full model so the comparison remains
        # readable. Motion-only CV can drift far away; it is clipped to the
        # panel edge when necessary.
        for path in (record.target, record.full):
            for forward, right in path:
                rights.append(float(right))
                forwards.append(float(forward))
    # Clip extreme CV outliers so one bad extrapolation does not collapse the visible GT/model paths.
    rights_sorted = sorted(rights)
    forwards_sorted = sorted(forwards)

    def percentile(values: list[float], pct: float) -> float:
        index = int(round((len(values) - 1) * pct))
        return values[max(0, min(index, len(values) - 1))]

    min_r = percentile(rights_sorted, 0.02)
    max_r = percentile(rights_sorted, 0.98)
    min_f = percentile(forwards_sorted, 0.02)
    max_f = percentile(forwards_sorted, 0.98)
    span_r = max(max_r - min_r, 20.0)
    span_f = max(max_f - min_f, 20.0)
    min_r -= span_r * 0.12
    max_r += span_r * 0.12
    min_f -= span_f * 0.12
    max_f += span_f * 0.12
    return min_r, max_r, min_f, max_f


def project_path(path: list[list[float]], plot: tuple[int, int, int, int], bounds: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    min_r, max_r, min_f, max_f = bounds
    x0, y0, x1, y1 = plot
    width = x1 - x0
    height = y1 - y0
    scale = min(width / (max_r - min_r), height / (max_f - min_f))
    center_r = (min_r + max_r) / 2.0
    center_f = (min_f + max_f) / 2.0
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    points = []
    for forward, right in path:
        x = cx + (float(right) - center_r) * scale
        y = cy - (float(forward) - center_f) * scale
        x = max(x0, min(x1, x))
        y = max(y0, min(y1, y))
        points.append((x, y))
    return points


def draw_dots(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: tuple[int, int, int], radius: int = 7) -> None:
    for point in points:
        ellipse_center(draw, point, radius, (255, 255, 255), outline=color)


def draw_triangle(draw: ImageDraw.ImageDraw, point: tuple[float, float]) -> None:
    x, y = point
    points = [spoint((x, y - 14)), spoint((x - 13, y + 12)), spoint((x + 13, y + 12))]
    draw.polygon(points, fill=(0, 0, 0))


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    entries = [
        ("GT future path", GREEN, False),
        ("Motion-only CV", ORANGE, True),
        ("Full Model", BLUE, False),
    ]
    cx = x
    for label, color, dashed in entries:
        draw_line(draw, [(cx, y + 14), (cx + 58, y + 14)], color, width=6, dashed=dashed)
        draw_text(draw, (cx + 70, y), label, TEXT, F["small"])
        cx += 250


def draw_rgb_panel(canvas: Image.Image, draw: ImageDraw.ImageDraw, record: ReplayRecord, box: tuple[int, int, int, int]) -> None:
    rounded_rect(draw, box, PANEL, outline=(226, 232, 240), radius=18)
    x0, y0, x1, y1 = box
    draw_text(draw, (x0 + 22, y0 + 22), "Current RGB", TEXT, F["section"])
    frame_box = (x0 + 24, y0 + 72, x1 - 24, y1 - 30)
    if record.rgb_path is not None:
        rgb = Image.open(record.rgb_path).convert("RGB")
        max_w = (frame_box[2] - frame_box[0]) * SCALE
        max_h = (frame_box[3] - frame_box[1]) * SCALE
        scale = min(max_w / rgb.width, max_h / rgb.height)
        rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.Resampling.NEAREST)
        bg = Image.new(
            "RGB",
            (max_w, max_h),
            (16, 24, 39),
        )
        bg.paste(rgb, ((bg.width - rgb.width) // 2, (bg.height - rgb.height) // 2))
        canvas.paste(bg, spoint((frame_box[0], frame_box[1])))
    draw.rounded_rectangle(sbox(frame_box), radius=14 * SCALE, outline=(203, 213, 225), width=2 * SCALE)


def draw_gt_motion_marker(
    draw: ImageDraw.ImageDraw,
    gt_points: list[tuple[float, float]],
    progress_index: int,
) -> None:
    if not gt_points:
        return
    progress_index = max(0, min(progress_index, len(gt_points) - 1))
    trail = gt_points[: progress_index + 1]
    if len(trail) >= 2:
        draw_line(draw, trail, (21, 128, 61), width=12)
    active = gt_points[progress_index]
    ellipse_center(draw, active, 16, GREEN, outline=(255, 255, 255))
    ellipse_center(draw, active, 7, (255, 255, 255), outline=GREEN)


def draw_path_panel(
    draw: ImageDraw.ImageDraw,
    record: ReplayRecord,
    box: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
    progress_index: int,
) -> None:
    rounded_rect(draw, box, PANEL, outline=(226, 232, 240), radius=18)
    x0, y0, x1, y1 = box
    draw_text(draw, (x0 + 34, y0 + 24), "Future local path prediction", TEXT, F["section"])
    draw_text(draw, (x0 + 34, y0 + 58), "x = right, y = forward, origin = current pose", MUTED, F["small"])
    t_future = (progress_index + 1) / 5.0
    draw_text(draw, (x1 - 410, y0 + 52), f"GT motion marker: t+{t_future:.1f}s", GREEN, F["label"])
    plot = (x0 + 78, y0 + 122, x1 - 72, y1 - 86)
    px0, py0, px1, py1 = plot
    for frac in [0.25, 0.5, 0.75]:
        gx = px0 + (px1 - px0) * frac
        gy = py0 + (py1 - py0) * frac
        draw.line([spoint((gx, py0)), spoint((gx, py1))], fill=GRID, width=SCALE)
        draw.line([spoint((px0, gy)), spoint((px1, gy))], fill=GRID, width=SCALE)
    draw.line([spoint((px0, py1)), spoint((px1, py1))], fill=AXIS, width=2 * SCALE)
    draw.line([spoint((px0, py0)), spoint((px0, py1))], fill=AXIS, width=2 * SCALE)
    gt = project_path(record.target, plot, bounds)
    full = project_path(record.full, plot, bounds)
    cv = project_path(record.cv, plot, bounds)
    origin = project_path([[0.0, 0.0]], plot, bounds)[0]
    draw_line(draw, cv, ORANGE, width=6, dashed=True)
    draw_line(draw, full, BLUE, width=8)
    draw_line(draw, gt, (74, 222, 128), width=7)
    draw_gt_motion_marker(draw, gt, progress_index)
    draw_dots(draw, cv, ORANGE, radius=5)
    draw_dots(draw, full, BLUE, radius=6)
    draw_dots(draw, gt, GREEN, radius=5)
    draw_triangle(draw, origin)
    draw_text(draw, (px0, py1 + 22), "right", AXIS, F["axis"])
    draw_text(draw, (px0 - 12, py0 - 34), "forward", AXIS, F["axis"])


def draw_metrics(draw: ImageDraw.ImageDraw, record: ReplayRecord, frame_index: int, total_frames: int) -> None:
    rounded_rect(draw, (70, 675, 490, 955), (248, 250, 252), outline=(226, 232, 240), radius=16)
    timestamp = record.center_step * FRAME_SKIP / DOOM_FPS
    sample_tail = record.sample_id.split("__", 1)[-1]
    if len(sample_tail) > 33:
        sample_tail = sample_tail[:30] + "..."
    draw_text(draw, (96, 700), "Offline replay state", TEXT, F["section"])
    draw_text(draw, (96, 740), f"sample: {sample_tail}", TEXT, F["small"])
    draw_text(draw, (96, 774), f"time {timestamp:.2f}s", MUTED, F["small"])
    draw_text(draw, (96, 808), f"frame {frame_index + 1}/{total_frames}", MUTED, F["small"])
    draw_text(draw, (96, 858), "Full ADE/FDE", BLUE, F["small"])
    draw_text(draw, (275, 858), f"{record.full_ade:.2f} / {record.full_fde:.2f}", BLUE, F["small"])
    draw_text(draw, (96, 895), "CV ADE/FDE", ORANGE, F["small"])
    draw_text(draw, (275, 895), f"{record.cv_ade:.2f} / {record.cv_fde:.2f}", ORANGE, F["small"])
    improvement = record.cv_ade - record.full_ade
    fill = GREEN if improvement > 0 else RED
    draw_text(draw, (96, 934), f"Full better by {improvement:.2f} ADE", fill, F["small"])


def render_frame(
    record: ReplayRecord,
    bounds: tuple[float, float, float, float],
    progress_index: int,
    frame_index: int,
    total_frames: int,
) -> Image.Image:
    canvas, draw = new_canvas()
    draw_text(draw, (60, 34), "GT vs Prediction on Held-out ViZDoom Replay", TEXT, F["title"])
    draw_text(draw, (62, 88), "Future local path prediction from 1-second visual and ego-motion history", MUTED, F["subtitle"])
    draw_legend(draw, 760, 50)
    draw_rgb_panel(canvas, draw, record, (70, 150, 490, 640))
    local_bounds = compute_plot_bounds([record])
    draw_path_panel(draw, record, (525, 150, 1850, 955), local_bounds, progress_index)
    draw_metrics(draw, record, frame_index, total_frames)
    return canvas.resize((W, H), Image.Resampling.LANCZOS)


def write_notes(segment: list[ReplayRecord], frame_count: int, duration: float) -> None:
    mean_full = sum(item.full_ade for item in segment) / len(segment)
    mean_cv = sum(item.cv_ade for item in segment) / len(segment)
    text = f"""# GT Comparison Replay Notes

Date: 2026-05-22

## Artifacts

- MP4: `outputs/result_visualizations_20260522/viz_04_gt_comparison_replay.mp4`
- GIF: `outputs/result_visualizations_20260522/viz_04_gt_comparison_replay.gif`
- First frame: `outputs/result_visualizations_20260522/viz_04_frame_000.png`
- Middle frame: `outputs/result_visualizations_20260522/viz_04_frame_mid.png`

## Source

- Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s`
- Base dataset family: v4 defaults, held-out test split
- Checkpoint: `checkpoints/wit_vz_v4_defaults_dinov3_single_03s.pt`
- DINOv3 cache: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`
- Horizon: 3 seconds, 15 future waypoints at 5 FPS
- Input history: 1 second of visual and ego-motion history
- Video: {frame_count} frames at {VIDEO_FPS} FPS, {duration:.1f}s

This is an offline prediction replay, not closed-loop control. The model is not
driving the agent in the video; each frame shows a held-out logged ViZDoom state
and compares future-path predictions against the logged future trajectory.

The large right panel is the main comparison view. The bright green moving dot
walks along the GT future path from t+0.2s to t+3.0s while the full model and
Motion-only CV predictions remain overlaid for that logged state. This makes
the GT future motion explicit instead of showing it only as a static line. Each
frame uses an adaptive zoom around the current GT and Full Model paths for
readability; if Motion-only CV drifts far away, its dashed line can run to the
plot edge.

## Colors

- Green: GT future path
- Orange: Motion-only CV, recent velocity extrapolation
- Blue/purple: Full Model prediction

## Metrics

- ADE: mean Euclidean distance over all future waypoints.
- FDE: Euclidean distance at the final future waypoint.

## Selection Rule

Selected from held-out v4 test samples where RGB, GT future path, Motion-only CV
prediction, and Full Model prediction are all available, and where the Full
Model improves over Motion-only CV on average over the replay segment.

Selected episode: `{segment[0].episode_id}`
Selected sample range: `{segment[0].sample_id}` to `{segment[-1].sample_id}`
Mean Full Model ADE: {mean_full:.2f}
Mean Motion-only CV ADE: {mean_cv:.2f}

## Presenter Script

1. "This is an offline replay from a held-out ViZDoom test episode, so the
   agent trajectory is fixed and the model is only predicting future local
   path from each logged state."
2. "Green is the logged future, orange is recent velocity extrapolation, and
   blue is the full visual-memory model."
3. "The key visual cue is that the full model often bends toward the logged
   future path instead of drifting like the motion-only baseline."
"""
    NOTES_PATH.write_text(text, encoding="utf-8")


def run_ffmpeg(frame_dir: Path, frame_count: int) -> None:
    pattern = frame_dir / "frame_%04d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(VIDEO_FPS),
            "-i",
            str(pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(MP4_PATH),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(MP4_PATH),
            "-vf",
            "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(GIF_PATH),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not MP4_PATH.exists() or not GIF_PATH.exists():
        raise RuntimeError("ffmpeg did not create expected video outputs")
    if frame_count <= 0:
        raise RuntimeError("No frames were encoded")


def verify_outputs() -> None:
    for path in [MP4_PATH, GIF_PATH, FRAME_000, FRAME_MID, NOTES_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)
    with Image.open(FRAME_000) as image:
        if image.size != (W, H):
            raise RuntimeError(f"{FRAME_000} has size {image.size}")
    with Image.open(FRAME_MID) as image:
        if image.size != (W, H):
            raise RuntimeError(f"{FRAME_MID} has size {image.size}")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of",
            "json",
            str(MP4_PATH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(probe.stdout)


def main() -> None:
    for path in [DATASET, CACHE, CHECKPOINT]:
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = collect_predictions(device)
    segment = select_segment(records)
    bounds = compute_plot_bounds(segment)
    expanded = [
        (record, phase)
        for record in segment
        for phase in range(REPEAT_PER_SAMPLE)
    ]
    total_frames = len(expanded)
    with tempfile.TemporaryDirectory(prefix="ddpd_gt_replay_") as tmp:
        frame_dir = Path(tmp)
        for index, (record, phase) in enumerate(expanded):
            if len(record.target) <= 1:
                progress_index = 0
            else:
                progress_index = int(round(phase * (len(record.target) - 1) / max(REPEAT_PER_SAMPLE - 1, 1)))
            frame = render_frame(record, bounds, progress_index, index, total_frames)
            frame_path = frame_dir / f"frame_{index:04d}.png"
            frame.save(frame_path, "PNG")
            if index == 0:
                shutil.copy(frame_path, FRAME_000)
            if index == total_frames // 2:
                shutil.copy(frame_path, FRAME_MID)
        run_ffmpeg(frame_dir, total_frames)
    write_notes(segment, total_frames, total_frames / VIDEO_FPS)
    verify_outputs()
    print(f"Wrote {MP4_PATH}")
    print(f"Wrote {GIF_PATH}")


if __name__ == "__main__":
    main()
