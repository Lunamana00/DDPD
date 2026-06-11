"""Render sample-by-sample demo video with CV / GT / model triptychs."""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CV_COLOR = (214, 72, 60)
GT_COLOR = (32, 150, 80)
PRED_COLOR = (42, 108, 218)
GRID_COLOR = (220, 224, 230)
TEXT_COLOR = (24, 28, 33)
MUTED_COLOR = (92, 100, 110)
BG_COLOR = (248, 249, 251)


@dataclass(frozen=True)
class DemoSource:
    name: str
    summary: Path
    dataset_candidates: tuple[Path, ...]
    prediction_candidates: tuple[Path, ...]
    raw_root_bases: tuple[Path, ...] = ()
    max_items: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/demo/presentation_sequence/demo_triptych_sequence.mp4"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds-per-item", type=float, default=2.2)
    parser.add_argument(
        "--max-items-per-source",
        type=int,
        default=0,
        help="Optional cap applied after reading each summary selection. 0 keeps all selected rows.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Custom source as name|summary|dataset|predictions|raw_root_base. "
            "May be repeated. Empty raw_root_base is allowed."
        ),
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_sources() -> list[DemoSource]:
    root = repo_root()
    server_ddpd = Path("/home/taehyun/projects/DDPD")
    server_episodic = Path("/home/taehyun/projects/DDPD_episodic_memory_v4")
    return [
        DemoSource(
            name="ViZDoom 3s",
            summary=root / "reports/demo/vizdoom_multi_scenario_03s/summary.json",
            dataset_candidates=(
                root / "data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s",
                server_ddpd / "data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s",
            ),
            prediction_candidates=(
                root / "runs/episodic_memory_ablation_v4/seed_7/03s/long_attention_no_ego/predictions.jsonl",
                server_episodic / "runs/episodic_memory_ablation_v4/seed_7/03s/long_attention_no_ego/predictions.jsonl",
            ),
            raw_root_bases=(root, server_ddpd),
        ),
        DemoSource(
            name="ViZDoom 10s",
            summary=root / "reports/demo/vizdoom_multi_scenario_10s/summary.json",
            dataset_candidates=(
                root / "data/wit_vz/processed/horizon_sweep_v4_defaults/future_10s",
                server_ddpd / "data/wit_vz/processed/horizon_sweep_v4_defaults/future_10s",
            ),
            prediction_candidates=(
                root / "runs/episodic_memory_ablation_v4/seed_7/10s/long_attention_no_ego/predictions.jsonl",
                server_episodic / "runs/episodic_memory_ablation_v4/seed_7/10s/long_attention_no_ego/predictions.jsonl",
            ),
            raw_root_bases=(root, server_ddpd),
        ),
        DemoSource(
            name="MiniWorld",
            summary=root / "reports/demo/external_miniworld_zero_shot_03s/contact_by_env/summary.json",
            dataset_candidates=(root / "data/wit_vz/processed/miniworld_demo_001_03s",),
            prediction_candidates=(root / "reports/demo/external_miniworld_zero_shot_03s/eval_all/predictions.jsonl",),
            raw_root_bases=(root,),
        ),
        DemoSource(
            name="AI2-THOR",
            summary=root / "reports/demo/external_ai2thor_zero_shot_03s/contact_by_scene/summary.json",
            dataset_candidates=(root / "data/wit_vz/processed/ai2thor_demo_001_03s",),
            prediction_candidates=(root / "reports/demo/external_ai2thor_zero_shot_03s/eval_all/predictions.jsonl",),
            raw_root_bases=(root,),
        ),
        DemoSource(
            name="ProcTHOR",
            summary=root / "reports/demo/external_procthor_zero_shot_03s/contact_by_house/summary.json",
            dataset_candidates=(root / "data/wit_vz/processed/procthor_demo_001_03s",),
            prediction_candidates=(root / "reports/demo/external_procthor_zero_shot_03s/eval_all/predictions.jsonl",),
            raw_root_bases=(root,),
        ),
        DemoSource(
            name="DeepMind Lab",
            summary=root / "reports/demo/external_deepmind_lab_zero_shot_03s/contact_by_level/summary.json",
            dataset_candidates=(root / "data/wit_vz/processed/deepmind_lab_demo_001_03s",),
            prediction_candidates=(root / "reports/demo/external_deepmind_lab_zero_shot_03s/eval_all/predictions.jsonl",),
            raw_root_bases=(root,),
        ),
        DemoSource(
            name="Habitat",
            summary=root / "reports/demo/external_habitat_zero_shot_03s/contact_by_scene/summary.json",
            dataset_candidates=(root / "data/wit_vz/processed/habitat_demo_001_03s",),
            prediction_candidates=(root / "reports/demo/external_habitat_zero_shot_03s/eval_all/predictions.jsonl",),
            raw_root_bases=(root,),
        ),
        DemoSource(
            name="MineDojo",
            summary=root / "reports/demo/external_minedojo_zero_shot_03s/contact_by_biome/summary.json",
            dataset_candidates=(root / "data/wit_vz/processed/minedojo_demo_001_03s",),
            prediction_candidates=(root / "reports/demo/external_minedojo_zero_shot_03s/eval_all/predictions.jsonl",),
            raw_root_bases=(root,),
        ),
    ]


def custom_sources(values: list[str]) -> list[DemoSource]:
    sources = []
    for raw in values:
        parts = raw.split("|")
        if len(parts) not in {4, 5}:
            raise ValueError("--source expects name|summary|dataset|predictions|raw_root_base")
        raw_root = (Path(parts[4]),) if len(parts) == 5 and parts[4] else ()
        sources.append(
            DemoSource(
                name=parts[0],
                summary=Path(parts[1]),
                dataset_candidates=(Path(parts[2]),),
                prediction_candidates=(Path(parts[3]),),
                raw_root_bases=raw_root,
            )
        )
    return sources


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def path_error(path_a: list[list[float]], path_b: list[list[float]]) -> tuple[float, float]:
    errors = [math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])) for a, b in zip(path_a, path_b)]
    if not errors:
        return 0.0, 0.0
    return float(sum(errors) / len(errors)), float(errors[-1])


def resolve_raw_root(dataset_dir: Path, raw_dir: str | Path, bases: Iterable[Path]) -> Path:
    path = Path(raw_dir)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([dataset_dir / path, Path.cwd() / path, path])
        for base in bases:
            candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_raw_dirs(dataset_dir: Path, bases: Iterable[Path]) -> dict[str, Path]:
    manifest = read_json(dataset_dir / "dataset_manifest.json")
    entries = manifest.get("raw_dirs") or {"default": manifest.get("raw_dir", "")}
    return {str(key): resolve_raw_root(dataset_dir, value, bases) for key, value in entries.items()}


def resolve_frame_path(raw_dirs: dict[str, Path], rel_path: str, source_id: str | None) -> Path | None:
    selected_source = source_id
    rel = rel_path
    if "::" in rel_path:
        selected_source, rel = rel_path.split("::", 1)
    path = Path(rel)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    if selected_source and selected_source in raw_dirs:
        candidates.append(raw_dirs[selected_source] / path)
    candidates.extend(raw_root / path for raw_root in raw_dirs.values())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def selected_ids(summary_path: Path, max_items: int) -> list[tuple[str, str, str]]:
    summary = read_json(summary_path)
    if isinstance(summary, list):
        rows = summary
    else:
        rows = summary.get("selected", [])
    selected = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        label = str(row.get("scenario") or row.get("group") or row.get("level") or row.get("scene") or "")
        case = str(row.get("case", ""))
        selected.append((sample_id, label, case))
    if max_items > 0:
        selected = selected[:max_items]
    return selected


def metadata_label(sample: dict[str, Any], fallback: str) -> str:
    metadata = sample.get("metadata", {})
    return str(
        fallback
        or metadata.get("scenario")
        or metadata.get("source_dataset")
        or metadata.get("env_id")
        or metadata.get("scene")
        or metadata.get("level")
        or metadata.get("biome")
        or "unknown"
    )


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
        ade, fde = path_error(pred_path, target)
        cv_ade, cv_fde = path_error(cv_path, target) if cv_path else (float("nan"), float("nan"))
        source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
        frame_rel = sample["rgb_history_paths"][-1]
        frame_path = resolve_frame_path(raw_dirs, frame_rel, source_id)
        items.append(
            {
                "source": source.name,
                "label": metadata_label(sample, label),
                "case": case,
                "sample_id": sample_id,
                "frame_path": frame_path,
                "target": target,
                "prediction": pred_path,
                "cv": cv_path,
                "ADE": float(prediction.get("ADE", ade)),
                "FDE": float(prediction.get("FDE", fde)),
                "cv_ADE": float(prediction.get("constant_velocity_ADE", cv_ade)),
                "cv_FDE": float(prediction.get("constant_velocity_FDE", cv_fde)),
            }
        )
    return items


def fit_image(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    max_w, max_h = max_size
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def axis_scale(*paths: list[list[float]]) -> float:
    max_abs = 1.0
    for path in paths:
        for point in path or []:
            if len(point) >= 2:
                max_abs = max(max_abs, abs(float(point[0])), abs(float(point[1])))
    return max_abs


def draw_path_plot(path: list[list[float]], size: tuple[int, int], color: tuple[int, int, int], max_abs: float) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    margin = 38
    cx = width // 2
    cy = height - margin
    scale = min((width - 2 * margin), (height - 2 * margin)) / max(max_abs * 2.2, 1.0)
    for offset in range(-4, 5):
        x = cx + offset * scale
        y = cy - offset * scale
        draw.line((x, margin, x, cy), fill=GRID_COLOR, width=1)
        draw.line((margin, y, width - margin, y), fill=GRID_COLOR, width=1)
    draw.line((cx, cy, cx, margin), fill=(35, 40, 45), width=2)
    draw.line((margin, cy, width - margin, cy), fill=(60, 65, 70), width=2)
    draw.text((cx + 4, margin - 22), "forward", fill=TEXT_COLOR, font=font(12))
    draw.text((width - 84, cy + 8), "right", fill=TEXT_COLOR, font=font(12))
    points = [(cx + float(right) * scale, cy - float(forward) * scale) for forward, right in path or []]
    if len(points) > 1:
        draw.line(points, fill=color, width=5)
    for px, py in points:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(30, 30, 30))
    return image


def load_rgb(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path and path.exists():
        image = Image.open(path).convert("RGB")
    else:
        image = Image.new("RGB", (320, 180), (230, 234, 238))
        draw = ImageDraw.Draw(image)
        draw.text((18, 18), "RGB frame missing", fill=TEXT_COLOR, font=font(18))
    return fit_image(image, size)


def paste_center(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    x = x1 + (x2 - x1 - image.width) // 2
    y = y1 + (y2 - y1 - image.height) // 2
    canvas.paste(image, (x, y))


def wrap(text: str, chars: int) -> str:
    return "\n".join(textwrap.wrap(text, width=chars, break_long_words=False))


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
    color: tuple[int, int, int],
    max_abs: float,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill="white", outline=(216, 220, 226), width=2)
    draw.text((x + 22, y + 18), title, fill=color, font=font(30, bold=True))
    draw.text((x + 22, y + 56), subtitle, fill=MUTED_COLOR, font=font(17))
    frame_box = (x + 24, y + 92, x + w - 24, y + 392)
    paste_center(canvas, rgb, frame_box)
    plot = draw_path_plot(path, (w - 48, h - 430), color, max_abs)
    canvas.paste(plot, (x + 24, y + 408))


def render_item(item: dict[str, Any], order: int, total: int, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    header = f"{order:02d}/{total:02d}  {item['source']} / {item['label']} / {item['case']}"
    draw.text((46, 28), header, fill=TEXT_COLOR, font=font(34, bold=True))
    metrics = (
        f"sample={item['sample_id']}    "
        f"model ADE/FDE={item['ADE']:.2f}/{item['FDE']:.2f}    "
        f"CV ADE/FDE={item['cv_ADE']:.2f}/{item['cv_FDE']:.2f}"
    )
    draw.text((46, 76), wrap(metrics, 142), fill=MUTED_COLOR, font=font(20))
    col_gap = 24
    margin_x = 46
    top = 142
    col_w = (width - 2 * margin_x - 2 * col_gap) // 3
    col_h = height - top - 44
    rgb = load_rgb(item["frame_path"], (col_w - 52, 300))
    max_abs = axis_scale(item["cv"], item["target"], item["prediction"])
    draw_column(canvas, margin_x, top, col_w, col_h, "CV baseline", "recent-motion extrapolation", rgb, item["cv"], CV_COLOR, max_abs)
    draw_column(canvas, margin_x + col_w + col_gap, top, col_w, col_h, "GT", "future local path label", rgb, item["target"], GT_COLOR, max_abs)
    draw_column(canvas, margin_x + 2 * (col_w + col_gap), top, col_w, col_h, "Prediction", "visual cue-memory output", rgb, item["prediction"], PRED_COLOR, max_abs)
    return canvas


def write_video(items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")
    repeats = max(1, round(args.seconds_per_item * args.fps))
    try:
        total = len(items)
        for index, item in enumerate(items, start=1):
            frame = render_item(item, index, total, args.width, args.height)
            frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            for _ in range(repeats):
                writer.write(frame_bgr)
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
    manifest = {
        "output": str(args.output),
        "fps": args.fps,
        "seconds_per_item": args.seconds_per_item,
        "items": [
            {
                "source": item["source"],
                "label": item["label"],
                "case": item["case"],
                "sample_id": item["sample_id"],
                "ADE": item["ADE"],
                "FDE": item["FDE"],
                "cv_ADE": item["cv_ADE"],
                "cv_FDE": item["cv_FDE"],
            }
            for item in all_items
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(all_items)} items")


if __name__ == "__main__":
    main()
