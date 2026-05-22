"""Create presentation-ready result visualizations for v4 DDPD results.

The script intentionally uses only local repo artifacts and PIL drawing so it
does not require matplotlib or any data download.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_v4_inference_ablation import forward_with_ablation, load_model, move_batch, tensor_prediction
from src.models.motion import constant_velocity_path
from src.wit_vz.dataset import WITVZPathDataset, collate_path_batch


OUT_DIR = ROOT / "outputs/result_visualizations_20260522"
MAIN_DATASET = ROOT / "data/wit_vz/processed/wit_vz_v4_defaults_001"
HORIZON_ROOT = ROOT / "data/wit_vz/processed/horizon_sweep_v4_defaults"
CACHE = ROOT / "data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny"
CKPT_10S = ROOT / "checkpoints/wit_vz_v4_defaults_dinov3_single_10s.pt"
ABLATION_JSON = ROOT / "outputs/v4_inference_ablation/results.json"

W, H = 1920, 1080
SCALE = 2
BG = (248, 250, 252)
PANEL = (255, 255, 255)
GRID = (218, 226, 236)
AXIS = (72, 83, 99)
TEXT = (15, 23, 42)
MUTED = (100, 116, 139)
BLUE = (79, 70, 229)
PURPLE = (126, 58, 242)
GREEN = (22, 163, 74)
ORANGE = (234, 88, 12)
GRAY = (107, 114, 128)
RED = (220, 38, 38)
LIGHT_BLUE = (199, 210, 254)


@dataclass
class OverlaySample:
    sample_id: str
    rgb_path: Path | None
    target: list[list[float]]
    full: list[list[float]]
    cv: list[list[float]]
    full_ade: float
    cv_ade: float
    score: float


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F = {
    "title": font(40, True),
    "subtitle": font(21),
    "section": font(25, True),
    "label": font(20),
    "small": font(17),
    "tiny": font(14),
    "axis": font(16),
    "bar": font(18, True),
}


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W * SCALE, H * SCALE), BG)
    return image, ImageDraw.Draw(image)


def save_canvas(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((W, H), Image.Resampling.LANCZOS)
    image.save(path, "PNG")


def sbox(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(int(v * SCALE) for v in box)  # type: ignore[return-value]


def spoint(point: tuple[float, float]) -> tuple[int, int]:
    return int(point[0] * SCALE), int(point[1] * SCALE)


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int] = TEXT,
    fnt: ImageFont.ImageFont | None = None,
    anchor: str | None = None,
) -> None:
    draw.text(spoint(xy), text, fill=fill, font=fnt or F["label"], anchor=anchor)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    radius: int = 16,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(sbox(box), radius=radius * SCALE, fill=fill, outline=outline, width=width * SCALE)


def draw_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int = 4,
    dashed: bool = False,
) -> None:
    if len(points) < 2:
        return
    scaled = [spoint(p) for p in points]
    if not dashed:
        draw.line(scaled, fill=color, width=width * SCALE, joint="curve")
        return
    dash = 14 * SCALE
    gap = 10 * SCALE
    for p0, p1 in zip(scaled[:-1], scaled[1:]):
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        dist = 0.0
        while dist < length:
            end = min(dist + dash, length)
            a = dist / length
            b = end / length
            draw.line(
                [
                    (int(x0 + dx * a), int(y0 + dy * a)),
                    (int(x0 + dx * b), int(y0 + dy * b)),
                ],
                fill=color,
                width=width * SCALE,
            )
            dist += dash + gap


def ellipse_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    r: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
) -> None:
    x, y = xy
    draw.ellipse(sbox((int(x - r), int(y - r), int(x + r), int(y + r))), fill=fill, outline=outline)


def transform_points(
    paths: list[list[list[float]]],
    plot_box: tuple[int, int, int, int],
    pad_ratio: float = 0.12,
) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float]]:
    all_forward = [0.0]
    all_right = [0.0]
    for path in paths:
        for forward, right in path:
            all_forward.append(float(forward))
            all_right.append(float(right))
    min_r, max_r = min(all_right), max(all_right)
    min_f, max_f = min(all_forward), max(all_forward)
    span_r = max(max_r - min_r, 1.0)
    span_f = max(max_f - min_f, 1.0)
    min_r -= span_r * pad_ratio
    max_r += span_r * pad_ratio
    min_f -= span_f * pad_ratio
    max_f += span_f * pad_ratio
    x0, y0, x1, y1 = plot_box
    width = x1 - x0
    height = y1 - y0
    scale = min(width / (max_r - min_r), height / (max_f - min_f))
    center_r = (min_r + max_r) / 2
    center_f = (min_f + max_f) / 2
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    def project(path: list[list[float]]) -> list[tuple[float, float]]:
        return [
            (cx + (float(right) - center_r) * scale, cy - (float(forward) - center_f) * scale)
            for forward, right in path
        ]

    return [project(path) for path in paths], (center_r, center_f, scale, scale)


def resolve_rgb_path(dataset: WITVZPathDataset, sample: dict[str, Any]) -> Path | None:
    if not sample.get("rgb_history_paths"):
        return None
    source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
    raw_path = dataset._resolve_raw_path(sample["rgb_history_paths"][-1], source_id)
    return raw_path if raw_path is not None and raw_path.exists() else None


def load_overlay_samples(device: torch.device, max_batches: int = 0) -> list[OverlaySample]:
    dataset = WITVZPathDataset(
        HORIZON_ROOT / "future_10s",
        split="test",
        load_rgb=False,
        visual_feature_cache_dir=CACHE,
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_path_batch)
    model, _checkpoint = load_model(CKPT_10S, device)
    candidates: list[OverlaySample] = []
    sample_by_id = {sample["sample_id"]: sample for sample in dataset.samples}

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            moved = move_batch(batch, device)
            full = tensor_prediction(forward_with_ablation(model, moved, "full_model"))
            cv = constant_velocity_path(moved["ego_history"], full.shape[1])
            target = moved["future_path"]
            full_err = torch.linalg.norm(full - target, dim=-1).mean(dim=1)
            cv_err = torch.linalg.norm(cv - target, dim=-1).mean(dim=1)
            improvement = cv_err - full_err
            for i, sample_id in enumerate(batch["sample_id"]):
                if improvement[i].item() <= 10.0:
                    continue
                gt = target[i].detach().cpu()
                deltas = gt[1:] - gt[:-1]
                if deltas.shape[0] >= 2:
                    angles = torch.atan2(deltas[:, 1], deltas[:, 0])
                    angle_diff = torch.atan2(torch.sin(angles[1:] - angles[:-1]), torch.cos(angles[1:] - angles[:-1]))
                    turn = float(angle_diff.abs().sum().item())
                else:
                    turn = 0.0
                lateral = float((gt[:, 1].max() - gt[:, 1].min()).item())
                score = float(improvement[i].item() + 20.0 * min(turn, 3.0) + 0.08 * lateral)
                raw_sample = sample_by_id[sample_id]
                candidates.append(
                    OverlaySample(
                        sample_id=sample_id,
                        rgb_path=resolve_rgb_path(dataset, raw_sample),
                        target=gt.tolist(),
                        full=full[i].detach().cpu().tolist(),
                        cv=cv[i].detach().cpu().tolist(),
                        full_ade=float(full_err[i].item()),
                        cv_ade=float(cv_err[i].item()),
                        score=score,
                    )
                )
    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: list[OverlaySample] = []
    seen_sources: set[str] = set()
    for item in candidates:
        source = item.sample_id.split("__", 1)[0]
        if source in seen_sources and len(selected) < 2:
            continue
        selected.append(item)
        seen_sources.add(source)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        selected = candidates[:3]
    if len(selected) < 3:
        raise RuntimeError("Could not find three overlay samples where Full Model improves over CV.")
    return selected


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    entries = [("GT future path", GREEN, False), ("Full Model", BLUE, False), ("Motion-only CV", ORANGE, True)]
    cx = x
    for label, color, dashed in entries:
        draw_line(draw, [(cx, y + 11), (cx + 48, y + 11)], color, width=5, dashed=dashed)
        draw_text(draw, (cx + 60, y), label, TEXT, F["small"])
        tw, _ = text_size(draw, label, F["small"])
        cx += 60 + tw // SCALE + 42


def draw_panel_overlay(draw: ImageDraw.ImageDraw, canvas: Image.Image, sample: OverlaySample, panel: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = panel
    rounded_rect(draw, panel, PANEL, outline=(226, 232, 240), radius=18)
    title = sample.sample_id.replace("wit_vz_v4_default_", "")
    if len(title) > 58:
        title = title[:55] + "..."
    draw_text(draw, (x0 + 26, y0 + 22), title, TEXT, F["small"])
    draw_text(
        draw,
        (x0 + 26, y0 + 52),
        f"Full ADE {sample.full_ade:.1f} vs CV {sample.cv_ade:.1f}",
        MUTED,
        F["tiny"],
    )

    plot_box = (x0 + 54, y0 + 132, x1 - 44, y1 - 64)
    if sample.rgb_path is not None:
        try:
            rgb = Image.open(sample.rgb_path).convert("RGB")
            rgb.thumbnail((150 * SCALE, 112 * SCALE), Image.Resampling.LANCZOS)
            inset = Image.new("RGB", (160 * SCALE, 122 * SCALE), (15, 23, 42))
            inset.paste(rgb, ((inset.width - rgb.width) // 2, (inset.height - rgb.height) // 2))
            canvas.paste(inset, spoint((x1 - 190, y0 + 32)))
            draw.rounded_rectangle(sbox((x1 - 190, y0 + 32, x1 - 30, y0 + 154)), radius=10 * SCALE, outline=(203, 213, 225), width=2 * SCALE)
            draw_text(draw, (x1 - 185, y0 + 160), "last RGB frame", MUTED, F["tiny"])
            plot_box = (x0 + 54, y0 + 150, x1 - 44, y1 - 64)
        except Exception:
            pass

    px0, py0, px1, py1 = plot_box
    for frac in [0.25, 0.5, 0.75]:
        gx = px0 + (px1 - px0) * frac
        gy = py0 + (py1 - py0) * frac
        draw.line([spoint((gx, py0)), spoint((gx, py1))], fill=GRID, width=SCALE)
        draw.line([spoint((px0, gy)), spoint((px1, gy))], fill=GRID, width=SCALE)
    projected, _meta = transform_points([sample.target, sample.cv, sample.full, [[0.0, 0.0]]], plot_box)
    gt_pts, cv_pts, full_pts, origin_pts = projected
    origin = origin_pts[0]
    draw_line(draw, gt_pts, GREEN, width=6)
    draw_line(draw, cv_pts, ORANGE, width=5, dashed=True)
    draw_line(draw, full_pts, BLUE, width=6)
    ellipse_center(draw, origin, 8, (255, 255, 255), outline=AXIS)
    draw_text(draw, (px0, py1 + 18), "right", AXIS, F["axis"])
    draw_text(draw, (px0 - 4, py0 - 25), "forward", AXIS, F["axis"])


def make_trajectory_overlay(device: torch.device) -> list[OverlaySample]:
    samples = load_overlay_samples(device)
    image, draw = new_canvas()
    draw_text(draw, (60, 42), "Trajectory Overlay: Actual v4 10s Test Samples", TEXT, F["title"])
    draw_text(
        draw,
        (62, 96),
        "Egocentric local coordinates: x-axis = right, y-axis = forward. Motion-only CV uses recent velocity extrapolation.",
        MUTED,
        F["subtitle"],
    )
    draw_legend(draw, 1180, 56)
    panels = [(50, 150, 640, 1015), (665, 150, 1255, 1015), (1280, 150, 1870, 1015)]
    for sample, panel in zip(samples, panels):
        draw_panel_overlay(draw, image, sample, panel)
    save_canvas(image, OUT_DIR / "viz_01_trajectory_overlay.png")
    return samples


def load_ablation_payload() -> dict[str, Any]:
    return json.loads(ABLATION_JSON.read_text(encoding="utf-8"))


def y_scale(values: list[float], plot: tuple[int, int, int, int], ymin: float = 0.0, ymax: float | None = None):
    x0, y0, x1, y1 = plot
    ymax = max(values) * 1.08 if ymax is None else ymax
    if ymax <= ymin:
        ymax = ymin + 1.0

    def project_x(t: float, max_t: float) -> float:
        return x0 + (x1 - x0) * (t / max_t)

    def project_y(v: float) -> float:
        return y1 - (y1 - y0) * ((v - ymin) / (ymax - ymin))

    return project_x, project_y, ymax


def draw_axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    ymax: float,
    x_max: float,
    x_label: str,
    y_label: str,
    show_x_ticks: bool = True,
) -> None:
    x0, y0, x1, y1 = plot
    draw.line([spoint((x0, y1)), spoint((x1, y1))], fill=AXIS, width=2 * SCALE)
    draw.line([spoint((x0, y0)), spoint((x0, y1))], fill=AXIS, width=2 * SCALE)
    for i in range(1, 6):
        value = ymax * i / 5
        y = y1 - (y1 - y0) * i / 5
        draw.line([spoint((x0, y)), spoint((x1, y))], fill=GRID, width=SCALE)
        draw_text(draw, (x0 - 62, int(y - 10)), f"{value:.0f}", MUTED, F["axis"])
    if show_x_ticks:
        for t in range(0, int(x_max) + 1, 2):
            x = x0 + (x1 - x0) * (t / x_max)
            draw.line([spoint((x, y1)), spoint((x, y1 + 8))], fill=AXIS, width=2 * SCALE)
            draw_text(draw, (int(x - 8), y1 + 16), str(t), MUTED, F["axis"])
    if x_label:
        draw_text(draw, ((x0 + x1) // 2 - 82, y1 + 55), x_label, AXIS, F["label"])
    draw_text(draw, (x0 - 86, y0 - 35), y_label, AXIS, F["label"])


def draw_summary_bars(draw: ImageDraw.ImageDraw, payload: dict[str, Any], box: tuple[int, int, int, int]) -> None:
    rounded_rect(draw, box, (250, 252, 255), outline=(226, 232, 240), radius=14)
    x0, y0, x1, y1 = box
    draw_text(draw, (x0 + 22, y0 + 20), "ADE by Horizon", TEXT, F["section"])
    horizons = [h["horizon_sec"] for h in payload["horizons"]]
    cv = [h["metrics"]["constant_velocity"]["ADE"] for h in payload["horizons"]]
    full = [h["metrics"]["full_model"]["ADE"] for h in payload["horizons"]]
    max_v = max(cv) * 1.15
    base_y = y1 - 58
    plot_h = y1 - y0 - 120
    group_w = (x1 - x0 - 70) / len(horizons)
    for i, horizon in enumerate(horizons):
        gx = x0 + 42 + group_w * i
        bar_w = 25
        cv_h = plot_h * cv[i] / max_v
        full_h = plot_h * full[i] / max_v
        draw.rectangle(sbox((int(gx), int(base_y - cv_h), int(gx + bar_w), base_y)), fill=ORANGE)
        draw.rectangle(sbox((int(gx + bar_w + 8), int(base_y - full_h), int(gx + 2 * bar_w + 8), base_y)), fill=BLUE)
        draw_text(draw, (int(gx + 4), base_y + 12), f"{horizon}s", MUTED, F["tiny"])
        gain = (1.0 - full[i] / cv[i]) * 100.0
        draw_text(draw, (int(gx - 4), int(base_y - max(cv_h, full_h) - 26)), f"{gain:.0f}%", GREEN, F["tiny"])
    draw_text(draw, (x0 + 30, y1 - 30), "orange=Motion-only CV, blue=Full, green=gain", MUTED, F["tiny"])


def make_horizon_error_growth() -> None:
    payload = load_ablation_payload()
    ten = next(h for h in payload["horizons"] if h["horizon_sec"] == 10)
    cv = ten["metrics"]["constant_velocity"]["per_step_error"]
    full = ten["metrics"]["full_model"]["per_step_error"]
    times = [(i + 1) / 5.0 for i in range(len(full))]

    image, draw = new_canvas()
    draw_text(draw, (60, 42), "Horizon-wise Error Growth", TEXT, F["title"])
    draw_text(
        draw,
        (62, 96),
        "Full Model slows long-horizon error growth compared with Motion-only CV",
        MUTED,
        F["subtitle"],
    )
    plot = (110, 170, 1325, 880)
    project_x, project_y, ymax = y_scale(cv + full, plot, ymax=440)
    draw_axes(draw, plot, ymax, 10.0, "future time (seconds)", "per-step error")
    cv_pts = [(project_x(t, 10.0), project_y(v)) for t, v in zip(times, cv)]
    full_pts = [(project_x(t, 10.0), project_y(v)) for t, v in zip(times, full)]
    draw_line(draw, cv_pts, ORANGE, width=7, dashed=True)
    draw_line(draw, full_pts, BLUE, width=7)
    draw_text(draw, (1070, 270), f"Motion-only CV FDE {cv[-1]:.1f}", ORANGE, F["label"])
    draw_text(draw, (1000, 515), f"Full Model FDE {full[-1]:.1f}", BLUE, F["label"])
    draw_text(draw, (170, 910), "Per-step errors are from the v4 10s test split at 5 FPS.", MUTED, F["small"])
    draw_summary_bars(draw, payload, (1390, 170, 1855, 880))
    rounded_rect(draw, (110, 945, 1855, 1025), (239, 246, 255), outline=(199, 210, 254), radius=14)
    draw_text(
        draw,
        (135, 965),
        "Required v4 metrics: 1s 26.87/41.56 vs CV 33.11/51.44 | 3s 62.10/103.35 vs CV 75.72/131.69 | 5s 88.60/157.09 vs CV 111.27/202.72 | 10s 154.57/258.72 vs CV 217.17/408.65",
        (49, 46, 129),
        F["tiny"],
    )
    save_canvas(image, OUT_DIR / "viz_02_horizon_error_growth.png")


def wrap_label(label: str) -> list[str]:
    replacements = {
        "Motion-only CV": ["Motion-only", "CV"],
        "Zero Visual Tokens": ["Zero Visual", "Tokens"],
        "No Temporal Adapter": ["No Temporal", "Adapter"],
        "No Cue Temporal": ["No Cue", "Temporal"],
        "No Cue Memory Update": ["No Cue Memory", "Update"],
        "No Ego-motion in Memory": ["No Ego-motion", "in Memory"],
    }
    return replacements.get(label, [label])


def make_ablation_bar_chart() -> None:
    values = [
        ("Full Model", 154.57, "+0.0%", BLUE),
        ("Motion-only CV", 217.17, "+40.5%", ORANGE),
        ("Zero Visual Tokens", 167.83, "+8.6%", (139, 92, 246)),
        ("No Temporal Adapter", 161.07, "+4.2%", (148, 163, 184)),
        ("No Cue Temporal", 180.12, "+16.5%", (124, 58, 237)),
        ("No Cue Memory Update", 313.74, "+103.0%", RED),
        ("No Ego-motion in Memory", 221.97, "+43.6%", (100, 116, 139)),
    ]
    image, draw = new_canvas()
    draw_text(draw, (60, 42), "10s Inference-time Ablation: ADE", TEXT, F["title"])
    draw_text(draw, (62, 96), "Full Model baseline highlighted; lower ADE is better.", MUTED, F["subtitle"])
    plot = (115, 170, 1810, 850)
    x0, y0, x1, y1 = plot
    ymax = 340
    draw_axes(draw, plot, ymax, 7.0, "", "ADE", show_x_ticks=False)
    bar_gap = 26
    bar_w = int((x1 - x0 - bar_gap * (len(values) - 1)) / len(values))
    baseline_y = y1 - (y1 - y0) * (154.57 / ymax)
    draw.line([spoint((x0, baseline_y)), spoint((x1, baseline_y))], fill=LIGHT_BLUE, width=4 * SCALE)
    draw_text(draw, (x1 - 210, int(baseline_y - 32)), "Full Model ADE baseline", BLUE, F["small"])
    for i, (label, value, pct, color) in enumerate(values):
        bx0 = x0 + i * (bar_w + bar_gap)
        bx1 = bx0 + bar_w
        by0 = y1 - (y1 - y0) * (value / ymax)
        draw.rounded_rectangle(sbox((bx0, int(by0), bx1, y1)), radius=8 * SCALE, fill=color)
        draw_text(draw, (bx0 + 12, int(by0 - 56)), f"{value:.2f}", TEXT, F["bar"])
        draw_text(draw, (bx0 + 12, int(by0 - 30)), pct, RED if value > 250 else MUTED, F["small"])
        label_lines = wrap_label(label)
        for j, line in enumerate(label_lines):
            draw_text(draw, (bx0 + 2, y1 + 24 + j * 23), line, TEXT, F["small"])
    rounded_rect(draw, (1120, 900, 1810, 1008), (255, 247, 247), outline=(254, 202, 202), radius=14)
    draw_text(draw, (1145, 922), "Key takeaway", RED, F["section"])
    draw_text(draw, (1145, 960), "Removing cue memory update causes the largest 10s ADE degradation.", TEXT, F["small"])
    draw_text(draw, (115, 930), "Values are v4 10s test ADE from inference-time ablation; no retraining.", MUTED, F["small"])
    save_canvas(image, OUT_DIR / "viz_03_ablation_10s_ade.png")


def write_notes(samples: list[OverlaySample]) -> None:
    sample_lines = "\n".join(
        f"- `{s.sample_id}`: Full ADE {s.full_ade:.2f}, Motion-only CV ADE {s.cv_ade:.2f}, RGB inset={'yes' if s.rgb_path else 'no'}"
        for s in samples
    )
    text = f"""# Result Visualization Notes

Date: 2026-05-22

All three figures use v4 defaults results. No presentation slide file is
created or committed.

## viz_01_trajectory_overlay.png

Shows actual v4-derived 10s test samples from
`data/wit_vz/processed/horizon_sweep_v4_defaults/future_10s`. Each subplot
compares GT future local path, Full Model, and Motion-only CV. The coordinate
system is egocentric local coordinates with x-axis = right and y-axis =
forward. RGB insets use the last history frame when available.

Selected actual v4 samples:

{sample_lines}

Presenter script: "These are real v4 test samples, not schematic paths. The
orange dashed path is recent velocity extrapolation, while the full model uses
visual DINOv3 cues and memory. The examples were chosen because the full model
improves over motion-only extrapolation, especially when the future path bends
or changes direction."

## viz_02_horizon_error_growth.png

Shows v4 per-step error growth for the 10s test split, with a horizon summary
inset for 1s, 3s, 5s, and 10s ADE. Values come from
`outputs/v4_inference_ablation/results.json` and match the v4 test metrics in
the reports.

Presenter script: "The model does not just improve the final number; it slows
the growth of error across future time. The advantage is especially visible at
10 seconds, where Motion-only CV drifts much faster."

## viz_03_ablation_10s_ade.png

Shows v4 10s inference-time ablation ADE. Full Model is the baseline,
Motion-only CV and visual-token ablations show the effect of DINO/visual
information, and No Cue Memory Update is highlighted in red.

Presenter script: "The largest degradation comes from removing the cue memory
update, so the memory mechanism is a central contributor at inference time.
DINO/visual information also matters: both Motion-only CV and Zero Visual
Tokens are worse than the full model."
"""
    (OUT_DIR / "result_visualization_notes.md").write_text(text, encoding="utf-8")


def verify_images() -> None:
    for name in [
        "viz_01_trajectory_overlay.png",
        "viz_02_horizon_error_growth.png",
        "viz_03_ablation_10s_ade.png",
    ]:
        path = OUT_DIR / name
        with Image.open(path) as image:
            if image.size != (W, H):
                raise RuntimeError(f"{path} has size {image.size}, expected {(W, H)}")
            image.verify()


def main() -> None:
    for path in [MAIN_DATASET, HORIZON_ROOT / "future_10s", CACHE, CKPT_10S, ABLATION_JSON]:
        if not path.exists():
            raise FileNotFoundError(f"Required local artifact is missing: {path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = make_trajectory_overlay(device)
    make_horizon_error_growth()
    make_ablation_bar_chart()
    write_notes(samples)
    verify_images()
    print(f"Wrote visualizations to {OUT_DIR}")


if __name__ == "__main__":
    main()
