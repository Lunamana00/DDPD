"""Plot WIT-VZ horizon sweep metrics without matplotlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "cv": (80, 115, 200),
    "model": (220, 90, 70),
    "improvement": (35, 150, 95),
    "axis": (45, 45, 45),
    "grid": (220, 225, 230),
    "text": (25, 25, 25),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PNG figures from horizon_summary.json.")
    parser.add_argument("--summary", type=Path, default=Path("runs/horizon_sweep/horizon_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/horizon_sweep/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ade_path = args.out_dir / "horizon_ade_fde.png"
    improvement_path = args.out_dir / "horizon_improvement.png"
    combined_path = args.out_dir / "horizon_summary.png"

    ade_fde = Image.new("RGB", (1280, 620), "white")
    draw = ImageDraw.Draw(ade_fde)
    font = _font(18)
    title_font = _font(28)
    _draw_line_plot(
        draw,
        rows,
        box=(70, 85, 610, 535),
        title="ADE by Future Horizon",
        y_label="ADE",
        series=[("CV", "cv_ADE", COLORS["cv"]), ("Cue-memory residual", "model_ADE", COLORS["model"])],
        font=font,
        title_font=title_font,
    )
    _draw_line_plot(
        draw,
        rows,
        box=(720, 85, 1210, 535),
        title="FDE by Future Horizon",
        y_label="FDE",
        series=[("CV", "cv_FDE", COLORS["cv"]), ("Cue-memory residual", "model_FDE", COLORS["model"])],
        font=font,
        title_font=title_font,
    )
    ade_fde.save(ade_path)

    improvement = Image.new("RGB", (1280, 620), "white")
    draw = ImageDraw.Draw(improvement)
    improvement_rows = []
    for row in rows:
        updated = dict(row)
        updated["ADE_improvement"] = row["cv_ADE"] - row["model_ADE"]
        updated["FDE_improvement"] = row["cv_FDE"] - row["model_FDE"]
        updated["ADE_improvement_pct"] = 100.0 * updated["ADE_improvement"] / max(row["cv_ADE"], 1.0e-8)
        updated["FDE_improvement_pct"] = 100.0 * updated["FDE_improvement"] / max(row["cv_FDE"], 1.0e-8)
        improvement_rows.append(updated)
    _draw_bar_plot(
        draw,
        improvement_rows,
        box=(70, 85, 610, 535),
        title="ADE Improvement over CV",
        y_label="ADE points",
        key="ADE_improvement",
        color=COLORS["improvement"],
        font=font,
        title_font=title_font,
    )
    _draw_bar_plot(
        draw,
        improvement_rows,
        box=(720, 85, 1210, 535),
        title="FDE Improvement over CV",
        y_label="FDE points",
        key="FDE_improvement",
        color=COLORS["improvement"],
        font=font,
        title_font=title_font,
    )
    improvement.save(improvement_path)

    combined = Image.new("RGB", (1280, 1180), "white")
    combined.paste(ade_fde, (0, 0))
    combined.paste(improvement, (0, 560))
    ImageDraw.Draw(combined).text((70, 20), "WIT-VZ Horizon Sweep", fill=COLORS["text"], font=_font(34))
    combined.save(combined_path)
    print(f"wrote {ade_path}")
    print(f"wrote {improvement_path}")
    print(f"wrote {combined_path}")


def _draw_line_plot(
    draw: ImageDraw.ImageDraw,
    rows: list[dict],
    box: tuple[int, int, int, int],
    title: str,
    y_label: str,
    series: list[tuple[str, str, tuple[int, int, int]]],
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    plot_left = x0 + 70
    plot_top = y0 + 60
    plot_right = x1 - 25
    plot_bottom = y1 - 65
    horizons = [row["horizon_sec"] for row in rows]
    values = [row[key] for _, key, _ in series for row in rows]
    ymax = _nice_max(max(values))
    ymin = 0.0

    draw.text((x0, y0), title, fill=COLORS["text"], font=title_font)
    _draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom, font, ymin, ymax, y_label)
    for name, key, color in series:
        points = []
        for row in rows:
            x = _scale(row["horizon_sec"], min(horizons), max(horizons), plot_left, plot_right)
            y = _scale(row[key], ymin, ymax, plot_bottom, plot_top)
            points.append((x, y))
        draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    _draw_x_ticks(draw, horizons, plot_left, plot_right, plot_bottom, font)
    _draw_legend(draw, series, x0 + 80, y1 - 40, font)


def _draw_bar_plot(
    draw: ImageDraw.ImageDraw,
    rows: list[dict],
    box: tuple[int, int, int, int],
    title: str,
    y_label: str,
    key: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    plot_left = x0 + 70
    plot_top = y0 + 60
    plot_right = x1 - 25
    plot_bottom = y1 - 65
    horizons = [row["horizon_sec"] for row in rows]
    values = [row[key] for row in rows]
    ymin = min(0.0, min(values))
    ymax = _nice_max(max(values))

    draw.text((x0, y0), title, fill=COLORS["text"], font=title_font)
    _draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom, font, ymin, ymax, y_label)
    zero_y = _scale(0.0, ymin, ymax, plot_bottom, plot_top)
    bar_gap = 8
    slot = (plot_right - plot_left) / len(rows)
    for idx, row in enumerate(rows):
        cx = plot_left + slot * idx + slot / 2
        bar_w = max(14, slot - bar_gap)
        y = _scale(row[key], ymin, ymax, plot_bottom, plot_top)
        top = min(y, zero_y)
        bottom = max(y, zero_y)
        draw.rectangle((cx - bar_w / 2, top, cx + bar_w / 2, bottom), fill=color)
        draw.text((cx - 10, plot_bottom + 12), str(row["horizon_sec"]), fill=COLORS["text"], font=font)
    draw.text((plot_left, plot_bottom + 42), "Future horizon (seconds)", fill=COLORS["text"], font=font)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
    font: ImageFont.ImageFont,
    ymin: float,
    ymax: float,
    y_label: str,
) -> None:
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=2)
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=2)
    for idx in range(6):
        value = ymin + (ymax - ymin) * idx / 5
        y = _scale(value, ymin, ymax, bottom, top)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((left - 64, y - 9), f"{value:.0f}", fill=COLORS["text"], font=font)
    draw.text((left - 62, top - 30), y_label, fill=COLORS["text"], font=font)


def _draw_x_ticks(
    draw: ImageDraw.ImageDraw,
    horizons: list[int],
    left: int,
    right: int,
    bottom: int,
    font: ImageFont.ImageFont,
) -> None:
    for horizon in horizons:
        x = _scale(horizon, min(horizons), max(horizons), left, right)
        draw.line((x, bottom, x, bottom + 6), fill=COLORS["axis"], width=2)
        draw.text((x - 8, bottom + 12), str(horizon), fill=COLORS["text"], font=font)
    draw.text((left, bottom + 42), "Future horizon (seconds)", fill=COLORS["text"], font=font)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    series: list[tuple[str, str, tuple[int, int, int]]],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    cursor = x
    for name, _key, color in series:
        draw.line((cursor, y + 10, cursor + 28, y + 10), fill=color, width=4)
        draw.text((cursor + 36, y), name, fill=COLORS["text"], font=font)
        cursor += 220


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    return ((int(value / magnitude) + 1) * magnitude)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
