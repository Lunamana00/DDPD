"""Render scenario-level ViZDoom demo panels from saved path predictions.

The script is designed for presentation demos. It selects easy/hard/failure
examples inside each requested ViZDoom scenario, renders a still panel for each
example, optionally renders short GIF replays, and writes a compact summary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


GT_COLOR = (28, 145, 72)
PRED_COLOR = (38, 100, 215)
CV_COLOR = (205, 74, 55)
GRID_COLOR = (222, 226, 230)
TEXT_COLOR = (24, 28, 33)
MUTED_COLOR = (96, 104, 112)


DEFAULT_SCENARIOS = (
    "basic",
    "simpler_basic",
    "my_way_home",
    "deadly_corridor",
    "health_gathering_supreme",
    "take_cover",
    "deathmatch",
    "rocket_basic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--cases", nargs="+", default=["easy", "hard", "failure"])
    parser.add_argument("--samples-per-case", type=int, default=1)
    parser.add_argument(
        "--raw-root-base",
        action="append",
        default=[],
        help="Fallback repo/data roots used when raw frames are not present next to the processed dataset.",
    )
    parser.add_argument("--frame-width", type=int, default=480)
    parser.add_argument("--panel-size", type=int, default=420)
    parser.add_argument("--contact-cols", type=int, default=3)
    parser.add_argument("--make-gifs", action="store_true")
    parser.add_argument("--gif-frames", type=int, default=48)
    parser.add_argument("--gif-fps", type=float, default=8.0)
    parser.add_argument("--max-predictions", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def resolve_raw_root(dataset_dir: Path, raw_dir: str | Path, raw_root_bases: list[Path]) -> Path:
    path = Path(raw_dir)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                dataset_dir / path,
                Path.cwd() / path,
                path,
            ]
        )
        for base in raw_root_bases:
            candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def raw_dirs_from_manifest(dataset_dir: Path, raw_root_bases: list[Path]) -> dict[str, Path]:
    manifest = load_json(dataset_dir / "dataset_manifest.json")
    raw_entries = manifest.get("raw_dirs")
    if raw_entries is None:
        raw_entries = {"default": manifest.get("raw_dir", "")}
    return {
        str(source_id): resolve_raw_root(dataset_dir, raw_dir, raw_root_bases)
        for source_id, raw_dir in raw_entries.items()
    }


def resolve_raw_path(
    dataset_dir: Path,
    raw_dirs: dict[str, Path],
    rel_path: str,
    source_id: str | None,
) -> Path | None:
    selected_source = source_id
    rel = rel_path
    if "::" in rel_path:
        selected_source, rel = rel_path.split("::", 1)
    candidates = []
    path = Path(rel)
    if path.is_absolute():
        candidates.append(path)
    if selected_source is not None and selected_source in raw_dirs:
        candidates.append(raw_dirs[selected_source] / path)
    candidates.extend(raw_root / path for raw_root in raw_dirs.values())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def displacement_errors(prediction: list[list[float]], target: list[list[float]]) -> list[float]:
    return [
        math.hypot(float(pred[0]) - float(gt[0]), float(pred[1]) - float(gt[1]))
        for pred, gt in zip(prediction, target, strict=True)
    ]


def prediction_metrics(prediction: list[list[float]], target: list[list[float]]) -> tuple[float, float]:
    errors = displacement_errors(prediction, target)
    return float(sum(errors) / len(errors)), float(errors[-1])


def prediction_row(pred: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    gt = sample["future_local_path"]
    ade, fde = prediction_metrics(pred["prediction"], gt)
    cv_prediction = pred.get("constant_velocity_prediction")
    if cv_prediction is None:
        cv_ade = float(pred.get("constant_velocity_ADE", 0.0))
        cv_fde = float(pred.get("constant_velocity_FDE", 0.0))
    else:
        cv_ade, cv_fde = prediction_metrics(cv_prediction, gt)
    metadata = sample.get("metadata", {})
    return {
        "sample_id": str(pred["sample_id"]),
        "episode_id": str(sample.get("episode_id", "")),
        "center_step": int(sample.get("center_step", 0)),
        "scenario": str(metadata.get("scenario", "unknown")),
        "source_id": str(sample.get("source", {}).get("source_id") or metadata.get("source_id") or ""),
        "policy": str(metadata.get("policy", "")),
        "prediction": pred["prediction"],
        "cv_prediction": cv_prediction,
        "target": gt,
        "ADE": ade,
        "FDE": fde,
        "cv_ADE": cv_ade,
        "cv_FDE": cv_fde,
        "model_minus_cv_ADE": ade - cv_ade,
        "sample": sample,
        "prediction_item": pred,
    }


def load_rows(dataset_dir: Path, predictions_path: Path, max_predictions: int = 0) -> list[dict[str, Any]]:
    samples = {
        str(sample["sample_id"]): sample
        for sample in read_jsonl(dataset_dir / "samples.jsonl")
    }
    rows = []
    for pred in read_jsonl(predictions_path, limit=max_predictions):
        sample = samples.get(str(pred.get("sample_id")))
        if sample is not None:
            rows.append(prediction_row(pred, sample))
    return rows


def select_case_rows(rows: list[dict[str, Any]], case: str, count: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    if case == "easy":
        ordered = sorted(rows, key=lambda row: (row["cv_ADE"], row["ADE"]))
    elif case == "hard":
        candidates = [row for row in rows if row["cv_ADE"] >= row["ADE"]]
        if not candidates:
            candidates = rows
        ordered = sorted(candidates, key=lambda row: (-(row["cv_ADE"] - row["ADE"]), -row["cv_ADE"]))
    elif case == "failure":
        candidates = [row for row in rows if row["ADE"] > row["cv_ADE"]]
        if not candidates:
            candidates = rows
        ordered = sorted(candidates, key=lambda row: (-(row["ADE"] - row["cv_ADE"]), -row["ADE"]))
    else:
        raise ValueError(f"Unknown case: {case}")
    return ordered[:count]


def bounds(*paths: list[list[float]] | None) -> float:
    values = [
        max(abs(float(point[0])), abs(float(point[1])))
        for path in paths
        if path
        for point in path
        if len(point) >= 2
    ]
    return max(values + [1.0])


def to_panel_points(
    path: list[list[float]],
    cx: float,
    cy: float,
    scale: float,
) -> list[tuple[float, float]]:
    return [(cx + float(right) * scale, cy - float(forward) * scale) for forward, right in path]


def draw_path_panel(
    prediction: list[list[float]],
    target: list[list[float]],
    cv_prediction: list[list[float]] | None,
    size: int,
) -> Image.Image:
    image = Image.new("RGB", (size, size), (250, 251, 252))
    draw = ImageDraw.Draw(image)
    margin = 44
    cx = size // 2
    cy = size - margin
    max_abs = bounds(prediction, target, cv_prediction)
    scale = (size - 2 * margin) / max(max_abs * 2.25, 1.0)
    for offset in range(-4, 5):
        x = cx + offset * scale
        y = cy - offset * scale
        draw.line((x, margin, x, cy), fill=GRID_COLOR, width=1)
        draw.line((margin, y, size - margin, y), fill=GRID_COLOR, width=1)
    draw.line((cx, cy, cx, margin), fill=(50, 55, 60), width=2)
    draw.line((margin, cy, size - margin, cy), fill=(80, 85, 90), width=2)
    draw.polygon([(cx, cy - 12), (cx - 8, cy + 8), (cx + 8, cy + 8)], fill=(35, 35, 35))
    draw.text((cx + 6, margin - 24), "forward", fill=TEXT_COLOR, font=font(13))
    draw.text((size - 92, cy + 8), "right", fill=TEXT_COLOR, font=font(13))
    if cv_prediction:
        cv_points = to_panel_points(cv_prediction, cx, cy, scale)
        if len(cv_points) > 1:
            draw.line(cv_points, fill=CV_COLOR, width=3)
        for px, py in cv_points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=CV_COLOR, width=2)
    target_points = to_panel_points(target, cx, cy, scale)
    pred_points = to_panel_points(prediction, cx, cy, scale)
    if len(target_points) > 1:
        draw.line(target_points, fill=GT_COLOR, width=4)
    if len(pred_points) > 1:
        draw.line(pred_points, fill=PRED_COLOR, width=4)
    for px, py in target_points:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=GT_COLOR)
    for px, py in pred_points:
        draw.rectangle((px - 4, py - 4, px + 4, py + 4), fill=PRED_COLOR)
    return image


def load_frame(
    dataset_dir: Path,
    raw_dirs: dict[str, Path],
    row: dict[str, Any],
    frame_width: int,
) -> Image.Image:
    sample = row["sample"]
    source_id = row["source_id"]
    frame_path = resolve_raw_path(dataset_dir, raw_dirs, sample["rgb_history_paths"][-1], source_id)
    if frame_path is None:
        image = Image.new("RGB", (frame_width, round(frame_width * 0.75)), (235, 238, 242))
        draw = ImageDraw.Draw(image)
        draw.text((18, 18), "RGB frame not found", fill=TEXT_COLOR, font=font(18))
        draw.text((18, 46), source_id, fill=MUTED_COLOR, font=font(13))
        return image
    frame = Image.open(frame_path).convert("RGB")
    frame_height = max(1, round(frame.height * (frame_width / frame.width)))
    return frame.resize((frame_width, frame_height))


def render_case_panel(
    dataset_dir: Path,
    raw_dirs: dict[str, Path],
    row: dict[str, Any],
    case: str,
    frame_width: int,
    panel_size: int,
) -> Image.Image:
    frame = load_frame(dataset_dir, raw_dirs, row, frame_width)
    path_panel = draw_path_panel(row["prediction"], row["target"], row["cv_prediction"], panel_size)
    margin = 24
    header_h = 92
    legend_h = 54
    content_h = max(frame.height, panel_size)
    canvas = Image.new(
        "RGB",
        (frame_width + panel_size + margin * 3, header_h + content_h + legend_h + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title = f"{row['scenario']} - {case}"
    subtitle = (
        f"step={row['center_step']}  ADE={row['ADE']:.2f}  FDE={row['FDE']:.2f}  "
        f"CV ADE={row['cv_ADE']:.2f}  Δ={row['model_minus_cv_ADE']:.2f}"
    )
    draw.text((margin, 16), title, fill=TEXT_COLOR, font=font(24))
    draw.text((margin, 48), subtitle, fill=MUTED_COLOR, font=font(15))
    draw.text((margin, 70), row["sample_id"], fill=MUTED_COLOR, font=font(12))
    canvas.paste(frame, (margin, header_h))
    panel_x = frame_width + margin * 2
    canvas.paste(path_panel, (panel_x, header_h))
    legend_y = header_h + content_h + 14
    x = margin
    for label, color in (
        ("GT future path", GT_COLOR),
        ("model prediction", PRED_COLOR),
        ("constant velocity", CV_COLOR),
    ):
        draw.rectangle((x, legend_y + 4, x + 18, legend_y + 16), fill=color)
        draw.text((x + 26, legend_y), label, fill=TEXT_COLOR, font=font(14))
        x += 190
    draw.text((panel_x, legend_y), "local coords: x=forward, y=right", fill=MUTED_COLOR, font=font(13))
    return canvas


def build_contact_sheet(images: list[tuple[str, Path]], cols: int) -> Image.Image:
    thumbs = []
    for label, path in images:
        image = Image.open(path).convert("RGB")
        image.thumbnail((460, 300))
        tile = Image.new("RGB", (480, 340), "white")
        tile.paste(image, ((480 - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((12, 310), label[:70], fill=TEXT_COLOR, font=font(14))
        thumbs.append(tile)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 480, rows * 340), "white")
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % cols) * 480, (index // cols) * 340))
    return sheet


def matching_episode_rows(rows: list[dict[str, Any]], selected: dict[str, Any], max_frames: int) -> list[dict[str, Any]]:
    episode_rows = [
        row for row in rows
        if row["episode_id"] == selected["episode_id"] and row["scenario"] == selected["scenario"]
    ]
    episode_rows.sort(key=lambda row: row["center_step"])
    selected_index = 0
    for index, row in enumerate(episode_rows):
        if row["sample_id"] == selected["sample_id"]:
            selected_index = index
            break
    half = max_frames // 2
    start = max(0, selected_index - half)
    end = min(len(episode_rows), start + max_frames)
    start = max(0, end - max_frames)
    return episode_rows[start:end]


def save_gif(
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    raw_dirs: dict[str, Path],
    out_path: Path,
    frame_width: int,
    panel_size: int,
    fps: float,
) -> None:
    frames = [
        render_case_panel(dataset_dir, raw_dirs, row, "replay", frame_width, panel_size)
        for row in rows
    ]
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000.0 / max(fps, 0.1))),
        loop=0,
        optimize=False,
    )


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw_root_bases = [Path(item) for item in args.raw_root_base]
    raw_dirs = raw_dirs_from_manifest(args.dataset, raw_root_bases)
    rows = load_rows(args.dataset, args.predictions, args.max_predictions)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    selected: list[dict[str, Any]] = []
    figure_paths: list[tuple[str, Path]] = []
    for scenario in args.scenarios:
        scenario_rows = by_scenario.get(scenario, [])
        scenario_dir = args.out / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for case in args.cases:
            for case_index, row in enumerate(select_case_rows(scenario_rows, case, args.samples_per_case), start=1):
                row = dict(row)
                row["case"] = case
                row["case_index"] = case_index
                selected.append(row)
                stem = f"{scenario}_{case}_{case_index:02d}"
                image = render_case_panel(args.dataset, raw_dirs, row, case, args.frame_width, args.panel_size)
                png_path = scenario_dir / f"{stem}.png"
                image.save(png_path)
                figure_paths.append((f"{scenario}/{case}", png_path))
                if args.make_gifs:
                    replay_rows = matching_episode_rows(scenario_rows, row, args.gif_frames)
                    save_gif(
                        replay_rows,
                        args.dataset,
                        raw_dirs,
                        scenario_dir / f"{stem}.gif",
                        args.frame_width,
                        args.panel_size,
                        args.gif_fps,
                    )

    if figure_paths:
        contact = build_contact_sheet(figure_paths, max(1, args.contact_cols))
        contact.save(args.out / "contact_sheet.png")

    summary = {
        "dataset": str(args.dataset),
        "predictions": str(args.predictions),
        "scenarios": args.scenarios,
        "cases": args.cases,
        "raw_dirs": {key: str(value) for key, value in raw_dirs.items()},
        "selected": [
            {
                key: row[key]
                for key in (
                    "scenario",
                    "case",
                    "case_index",
                    "sample_id",
                    "episode_id",
                    "center_step",
                    "policy",
                    "ADE",
                    "FDE",
                    "cv_ADE",
                    "cv_FDE",
                    "model_minus_cv_ADE",
                )
            }
            for row in selected
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# ViZDoom Multi-Scenario Demo Selection",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Predictions: `{args.predictions}`",
        f"- Selected examples: `{len(selected)}`",
        "",
        "| Scenario | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    row["case"],
                    row["sample_id"],
                    f"{row['ADE']:.3f}",
                    f"{row['FDE']:.3f}",
                    f"{row['cv_ADE']:.3f}",
                    f"{row['model_minus_cv_ADE']:.3f}",
                ]
            )
            + " |"
        )
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(selected)} demo examples to: {args.out}")


if __name__ == "__main__":
    main()
