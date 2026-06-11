"""Render external-domain WIT-VZ zero-shot demo panels.

This script complements ``render_vizdoom_scenario_demo.py``. It is intended
for converted external domains such as MiniWorld and AI2-THOR, where the
processed WIT-VZ sample metadata may have a single generic scenario but the raw
manifest still records per-episode environment IDs or scene names.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group labels to render.")
    parser.add_argument("--cases", nargs="+", default=["easy", "hard", "failure"])
    parser.add_argument("--samples-per-case", type=int, default=1)
    parser.add_argument("--contact-cols", type=int, default=3)
    parser.add_argument("--panel-size", type=int, default=360)
    parser.add_argument("--frame-width", type=int, default=320)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_raw_root(dataset_dir: Path, raw_dir: str | Path) -> Path:
    path = Path(raw_dir)
    if path.is_absolute():
        return path
    candidate = (dataset_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def load_raw_context(dataset_dir: Path) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    manifest = read_json(dataset_dir / "dataset_manifest.json")
    raw_entries = manifest.get("raw_dirs", {"default": manifest["raw_dir"]})
    raw_dirs = {str(source_id): resolve_raw_root(dataset_dir, raw_dir) for source_id, raw_dir in raw_entries.items()}
    episode_labels: dict[str, dict[str, str]] = {}
    for source_id, raw_root in raw_dirs.items():
        raw_manifest_path = raw_root / "manifest.json"
        if not raw_manifest_path.exists():
            continue
        raw_manifest = read_json(raw_manifest_path)
        for summary in raw_manifest.get("episode_summaries", []):
            episode_id = str(summary.get("episode_id", ""))
            label = (
                summary.get("env_id")
                or summary.get("scene")
                or summary.get("level")
                or summary.get("biome")
                or raw_manifest.get("scenario")
                or raw_manifest.get("env_name")
                or source_id
            )
            episode_labels[f"{source_id}::{episode_id}"] = {
                "group": str(label),
                "source_dataset": str(raw_manifest.get("source_dataset", "")),
            }
    return raw_dirs, episode_labels


def resolve_raw_path(raw_dirs: dict[str, Path], rel_path: str, source_id: str | None) -> Path:
    selected_source_id = source_id
    rel = rel_path
    if "::" in rel_path:
        selected_source_id, rel = rel_path.split("::", 1)
    path = Path(rel)
    if path.is_absolute():
        return path
    if selected_source_id and selected_source_id in raw_dirs:
        return raw_dirs[selected_source_id] / path
    return next(iter(raw_dirs.values())) / path


def group_label(sample: dict[str, Any], episode_labels: dict[str, dict[str, str]]) -> str:
    source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
    raw_episode_id = sample.get("metadata", {}).get("raw_episode_id")
    label = episode_labels.get(f"{source_id}::{raw_episode_id}", {}).get("group")
    if label:
        return label
    metadata = sample.get("metadata", {})
    return str(metadata.get("scenario") or metadata.get("source_dataset") or source_id or "unknown")


def path_error(path_a: list[list[float]], path_b: list[list[float]]) -> tuple[float, float]:
    errors = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path_a, path_b)]
    return sum(errors) / max(len(errors), 1), errors[-1] if errors else 0.0


def enrich_prediction(prediction: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    pred = prediction["prediction"]
    target = prediction.get("target") or sample["future_local_path"]
    ade, fde = path_error(pred, target)
    cv_pred = prediction.get("constant_velocity_prediction")
    if cv_pred is None:
        cv_ade = float("nan")
        cv_fde = float("nan")
    else:
        cv_ade, cv_fde = path_error(cv_pred, target)
    return {
        "sample_id": prediction["sample_id"],
        "prediction": pred,
        "target": target,
        "constant_velocity_prediction": cv_pred,
        "ADE": float(prediction.get("ADE", ade)),
        "FDE": float(prediction.get("FDE", fde)),
        "constant_velocity_ADE": float(prediction.get("constant_velocity_ADE", cv_ade)),
        "constant_velocity_FDE": float(prediction.get("constant_velocity_FDE", cv_fde)),
        "sample": sample,
    }


def select_cases(items: list[dict[str, Any]], cases: list[str], count: int) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for case in cases:
        if case == "easy":
            ranked = sorted(items, key=lambda item: (item["constant_velocity_ADE"], item["ADE"]))
        elif case == "hard":
            ranked = sorted(
                items,
                key=lambda item: (item["constant_velocity_ADE"] - item["ADE"], item["constant_velocity_ADE"]),
                reverse=True,
            )
        elif case == "failure":
            ranked = sorted(items, key=lambda item: (item["ADE"] - item["constant_velocity_ADE"], item["ADE"]), reverse=True)
        else:
            raise ValueError(f"Unsupported case: {case}")
        for item in ranked[:count]:
            selected.append((case, item))
    return selected


def font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_path_panel(item: dict[str, Any], size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    margin = 34
    cx = size // 2
    cy = size - margin
    draw.line((cx, cy, cx, margin), fill=(40, 40, 40), width=2)
    draw.line((cx, cy, size - margin, cy), fill=(80, 80, 80), width=2)
    draw.text((cx + 4, margin), "forward", fill=(40, 40, 40), font=font(11))
    draw.text((size - 80, cy + 5), "right", fill=(80, 80, 80), font=font(11))

    paths = [item["target"], item["prediction"]]
    if item.get("constant_velocity_prediction") is not None:
        paths.append(item["constant_velocity_prediction"])
    max_abs = max([max(abs(x), abs(y)) for path in paths for x, y in path] + [1.0])
    scale = (size - 2 * margin) / max(max_abs * 2.2, 1.0)

    def convert(path: list[list[float]]) -> list[tuple[float, float]]:
        return [(cx + y * scale, cy - x * scale) for x, y in path]

    def draw_path(path: list[list[float]], color: tuple[int, int, int], marker: str) -> None:
        points = convert(path)
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for px, py in points:
            if marker == "square":
                draw.rectangle((px - 3, py - 3, px + 3, py + 3), fill=color)
            else:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)

    draw_path(item["target"], (20, 140, 45), "circle")
    if item.get("constant_velocity_prediction") is not None:
        draw_path(item["constant_velocity_prediction"], (70, 110, 220), "circle")
    draw_path(item["prediction"], (210, 55, 45), "square")
    return image


def render_panel(
    raw_dirs: dict[str, Path],
    group: str,
    case: str,
    item: dict[str, Any],
    frame_width: int,
    path_size: int,
) -> Image.Image:
    sample = item["sample"]
    source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
    frame_path = resolve_raw_path(raw_dirs, sample["rgb_history_paths"][-1], source_id)
    frame = Image.open(frame_path).convert("RGB")
    frame_height = max(1, round(frame.height * frame_width / frame.width))
    frame = frame.resize((frame_width, frame_height))
    path_panel = draw_path_panel(item, path_size)
    width = frame_width + path_size + 44
    height = max(frame_height, path_size) + 92
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{group} / {case}", fill=(0, 0, 0), font=font(16))
    draw.text(
        (12, 32),
        f"ADE {item['ADE']:.3f}  FDE {item['FDE']:.3f}  CV ADE {item['constant_velocity_ADE']:.3f}",
        fill=(50, 50, 50),
        font=font(12),
    )
    draw.text((12, 52), item["sample_id"], fill=(90, 90, 90), font=font(10))
    canvas.paste(frame, (12, 82))
    canvas.paste(path_panel, (frame_width + 32, 72))
    return canvas


def save_contact_sheet(panels: list[Image.Image], out_path: Path, cols: int) -> None:
    if not panels:
        return
    cols = max(1, cols)
    rows = math.ceil(len(panels) / cols)
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for index, panel in enumerate(panels):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        sheet.paste(panel, (x, y))
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = {sample["sample_id"]: sample for sample in read_jsonl(args.dataset / "samples.jsonl")}
    raw_dirs, episode_labels = load_raw_context(args.dataset)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for prediction in read_jsonl(args.predictions):
        sample = samples.get(prediction["sample_id"])
        if sample is None:
            continue
        label = group_label(sample, episode_labels)
        if args.groups and label not in set(args.groups):
            continue
        grouped.setdefault(label, []).append(enrich_prediction(prediction, sample))

    panels = []
    summary_rows = []
    for group in sorted(grouped):
        selected = select_cases(grouped[group], args.cases, args.samples_per_case)
        group_dir = args.out / group.replace("/", "_").replace(" ", "_")
        group_dir.mkdir(parents=True, exist_ok=True)
        for ordinal, (case, item) in enumerate(selected, start=1):
            panel = render_panel(raw_dirs, group, case, item, args.frame_width, args.panel_size)
            filename = f"{case}_{ordinal:02d}.png"
            panel.save(group_dir / filename)
            panels.append(panel)
            summary_rows.append(
                {
                    "group": group,
                    "case": case,
                    "sample_id": item["sample_id"],
                    "ADE": item["ADE"],
                    "FDE": item["FDE"],
                    "constant_velocity_ADE": item["constant_velocity_ADE"],
                    "constant_velocity_FDE": item["constant_velocity_FDE"],
                    "image": (group_dir / filename).relative_to(args.out).as_posix(),
                }
            )

    save_contact_sheet(panels, args.out / "contact_sheet.png", args.contact_cols)
    (args.out / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    lines = [
        "# External Generalization Demo Selection",
        "",
        f"- Dataset: `{args.dataset.as_posix()}`",
        f"- Predictions: `{args.predictions.as_posix()}`",
        f"- Selected examples: `{len(summary_rows)}`",
        "",
        "| Group | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {case} | {sample_id} | {ADE:.3f} | {FDE:.3f} | {constant_velocity_ADE:.3f} | {delta:.3f} |".format(
                **row,
                delta=row["ADE"] - row["constant_velocity_ADE"],
            )
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary_rows)} external demo examples to: {args.out}")


if __name__ == "__main__":
    main()
