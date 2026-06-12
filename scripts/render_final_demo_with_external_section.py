"""Append external-domain demo material to the main counterfactual video.

The final video keeps the comparison semantics separated:

1. Main ViZDoom section:
   CV / PointNav / A* / GT / Ours real counterfactual rollout.
2. External-domain section:
   MiniWorld and AI2-THOR real 3-way rollouts, followed by overview cards for
   the remaining converted external datasets.

PointNav/A* are not shown in the external section because those oracle adapters
were defined for the ViZDoom/WIT-VZ pose-graph setting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/demo/presentation_sequence/demo_final_main_with_external_05s.mp4"),
    )
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--title-seconds", type=float, default=2.0)
    parser.add_argument("--metrics-seconds", type=float, default=5.0)
    parser.add_argument("--overview-seconds", type=float, default=2.0)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, width: int, height: int, fill: tuple[int, int, int] = (248, 249, 251)) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def title_card(width: int, height: int, title: str, subtitle: str, bullets: list[str]) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 110), title, fill=(24, 28, 33), font=font(66, bold=True))
    draw.text((92, 210), subtitle, fill=(76, 84, 94), font=font(34))
    y = 330
    for bullet in bullets:
        draw.rounded_rectangle((92, y + 3, 110, y + 21), radius=5, fill=(42, 108, 218))
        draw.text((132, y - 6), bullet, fill=(34, 40, 46), font=font(32))
        y += 70
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def image_card(path: Path, width: int, height: int, title: str, subtitle: str) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 42), title, fill=(24, 28, 33), font=font(44, bold=True))
    draw.text((72, 100), subtitle, fill=(76, 84, 94), font=font(24))
    image = Image.open(path).convert("RGB")
    fitted = fit_image(image, width - 120, height - 170, fill=(248, 249, 251))
    canvas.paste(fitted.crop((0, 0, width - 120, height - 170)), (60, 150))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_metric(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def draw_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    columns: list[tuple[str, int]],
    rows: list[list[str]],
    header_fill: tuple[int, int, int] = (232, 237, 244),
) -> int:
    row_h = 62
    table_w = sum(width for _, width in columns)
    draw.rounded_rectangle((x, y, x + table_w, y + row_h * (len(rows) + 1)), radius=14, fill=(255, 255, 255), outline=(214, 221, 230), width=2)
    draw.rounded_rectangle((x, y, x + table_w, y + row_h), radius=14, fill=header_fill)
    cur_x = x
    for label, width in columns:
        draw.text((cur_x + 20, y + 16), label, fill=(36, 44, 55), font=font(26, bold=True))
        cur_x += width
    for row_idx, row in enumerate(rows):
        row_y = y + row_h * (row_idx + 1)
        draw.line((x, row_y, x + table_w, row_y), fill=(226, 231, 237), width=1)
        cur_x = x
        for cell, (_, width) in zip(row, columns):
            draw.text((cur_x + 20, row_y + 16), cell, fill=(36, 44, 55), font=font(25))
            cur_x += width
    return y + row_h * (len(rows) + 1)


def metrics_card_main(metrics_path: Path, width: int, height: int) -> np.ndarray:
    metrics = read_json(metrics_path)
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 82), "Quantitative Result: ViZDoom 5s Counterfactual Demo", fill=(24, 28, 33), font=font(58, bold=True))
    draw.text((92, 162), "ADE/FDE are trajectory errors; lower is better. PointNav and A* use privileged GT endpoint information.", fill=(76, 84, 94), font=font(28))

    label_map = {
        "constant_velocity": "CV",
        "pointnav_goal_oracle": "PointNav oracle",
        "astar_oracle": "A* oracle",
        "ours": "Ours",
    }
    columns = [("Block", 410), ("Samples", 190), ("Method", 380), ("ADE", 170), ("FDE", 170), ("Interpretation", 850)]
    rows: list[list[str]] = []
    for block_key, block in metrics["blocks"].items():
        block_label = "Human replay GT" if block_key == "human_action_replay" else "V4 multi-scenario"
        for method_key in ["constant_velocity", "pointnav_goal_oracle", "astar_oracle", "ours"]:
            item = block[method_key]
            if method_key == "ours":
                cv_ade = block["constant_velocity"]["ADE"]
                rel = (cv_ade - item["ADE"]) / cv_ade * 100.0
                interpretation = f"{rel:+.1f}% ADE vs CV"
            elif method_key in {"pointnav_goal_oracle", "astar_oracle"}:
                interpretation = "Privileged upper-bound baseline"
            else:
                interpretation = "Recent-motion extrapolation"
            rows.append([
                block_label,
                str(block["samples"]),
                label_map[method_key],
                fmt_metric(item["ADE"]),
                fmt_metric(item["FDE"]),
                interpretation,
            ])
    y_end = draw_table(draw, 90, 250, columns, rows)
    notes = [
        "Human replay GT: public human action labels are replayed in ViZDoom to derive future paths.",
        "V4 multi-scenario GT: recorded WIT-VZ ViZDoom trajectories, not human-play GT.",
        "Use these numbers for in-domain demo comparison only; they are not comparable to external-domain scales.",
    ]
    y = y_end + 48
    for note in notes:
        draw.rounded_rectangle((92, y + 8, 110, y + 26), radius=5, fill=(42, 108, 218))
        draw.text((132, y), note, fill=(64, 72, 82), font=font(27))
        y += 54
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def metrics_card_external(root: Path, width: int, height: int) -> np.ndarray:
    paths = [
        ("MiniWorld", root / "reports/demo/external_miniworld_zero_shot_03s/eval_all/metrics.json"),
        ("AI2-THOR", root / "reports/demo/external_ai2thor_zero_shot_03s/eval_all/metrics.json"),
        ("ProcTHOR", root / "reports/demo/external_procthor_zero_shot_03s/eval_all/metrics.json"),
        ("DeepMind Lab", root / "reports/demo/external_deepmind_lab_zero_shot_03s/eval_all/metrics.json"),
        ("Habitat", root / "reports/demo/external_habitat_zero_shot_03s/eval_all/metrics.json"),
        ("MineDojo", root / "reports/demo/external_minedojo_zero_shot_03s/eval_all/metrics.json"),
    ]
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 82), "Quantitative Result: External Zero-shot 3s Sanity Checks", fill=(24, 28, 33), font=font(58, bold=True))
    draw.text((92, 162), "Same predictor is evaluated without retraining. Coordinate scale differs by environment; compare each row internally.", fill=(76, 84, 94), font=font(28))

    columns = [("Dataset", 360), ("CV ADE", 160), ("Ours ADE", 170), ("Ours FDE", 170), ("Ours vs CV", 210), ("Takeaway", 930)]
    rows: list[list[str]] = []
    for name, path in paths:
        item = read_json(path)
        cv_ade = float(item["cv_baseline"]["ADE"])
        ours_ade = float(item["ADE"])
        ours_fde = float(item["FDE"])
        rel = (cv_ade - ours_ade) / cv_ade * 100.0 if cv_ade else 0.0
        if rel > 0:
            takeaway = "Positive sanity case"
        elif ours_ade > cv_ade * 20:
            takeaway = "Strong domain/scale shift"
        else:
            takeaway = "Zero-shot transfer is weak"
        rows.append([name, fmt_metric(cv_ade), fmt_metric(ours_ade), fmt_metric(ours_fde), f"{rel:+.1f}%", takeaway])
    y_end = draw_table(draw, 90, 250, columns, rows)
    notes = [
        "External section tests whether the data format and inference path run outside ViZDoom.",
        "Large Ours-vs-CV gaps here mainly show domain and coordinate-scale mismatch, not model failure inside ViZDoom.",
        "DeepMind Lab is the current positive small sanity case; other rows motivate domain adaptation or retraining.",
    ]
    y = y_end + 56
    for note in notes:
        draw.rounded_rectangle((92, y + 8, 110, y + 26), radius=5, fill=(42, 108, 218))
        draw.text((132, y), note, fill=(64, 72, 82), font=font(27))
        y += 54
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def write_video_frames(
    writer: cv2.VideoWriter,
    path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    cap.release()
    return {"path": str(path), "fps": src_fps, "input_frames": src_frames, "written_frames": written}


def write_still(writer: cv2.VideoWriter, frame: np.ndarray, count: int) -> None:
    for _ in range(count):
        writer.write(frame)


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


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir = root / "reports/demo/final_main_with_external_05s"
    report_dir.mkdir(parents=True, exist_ok=True)

    main_video = root / "reports/demo/presentation_sequence/demo_main_5baseline_multiscenario_counterfactual_05s.mp4"
    external_videos = [
        root / "reports/demo/presentation_sequence/demo_miniworld_counterfactual_rollout.mp4",
        root / "reports/demo/presentation_sequence/demo_ai2thor_counterfactual_rollout.mp4",
    ]
    main_metrics_path = root / "reports/demo/main_5baseline_multiscenario_05s/metrics.json"
    external_metric_paths = [
        root / "reports/demo/external_miniworld_zero_shot_03s/eval_all/metrics.json",
        root / "reports/demo/external_ai2thor_zero_shot_03s/eval_all/metrics.json",
        root / "reports/demo/external_procthor_zero_shot_03s/eval_all/metrics.json",
        root / "reports/demo/external_deepmind_lab_zero_shot_03s/eval_all/metrics.json",
        root / "reports/demo/external_habitat_zero_shot_03s/eval_all/metrics.json",
        root / "reports/demo/external_minedojo_zero_shot_03s/eval_all/metrics.json",
    ]
    external_overviews = [
        ("MiniWorld overview", "Zero-shot formulation transfer; CV/GT/Ours available as real rollout above.", root / "reports/demo/presentation_sequence/06_miniworld_external_overview.png"),
        ("AI2-THOR overview", "Object-rich Unity-domain sanity check; CV/GT/Ours available as real rollout above.", root / "reports/demo/presentation_sequence/07_ai2thor_external_overview.png"),
        ("ProcTHOR overview", "Procedural Unity-house conversion; shown as static external-domain evidence.", root / "reports/demo/presentation_sequence/08_procthor_external_overview.png"),
        ("DeepMind Lab overview", "Game-like external-domain conversion; useful positive sanity case.", root / "reports/demo/presentation_sequence/09_deepmind_lab_external_overview.png"),
        ("Habitat overview", "Photorealistic embodied-navigation conversion; strong scale/domain shift.", root / "reports/demo/presentation_sequence/10_habitat_external_overview.png"),
        ("MineDojo overview", "Minecraft-style formulation gate and domain-gap failure case.", root / "reports/demo/presentation_sequence/11_minedojo_external_overview.png"),
    ]

    missing = [path for path in [main_video, *external_videos, main_metrics_path, *external_metric_paths, *[item[2] for item in external_overviews]] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(str(path) for path in missing))

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {output}")

    manifest: dict[str, Any] = {
        "output": str(output),
        "mode": "final_main_counterfactual_plus_external_section",
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "sections": [],
    }
    try:
        manifest["sections"].append({"type": "video", "name": "main_vizdoom_counterfactual", **write_video_frames(writer, main_video, args.width, args.height)})

        metrics_frames = max(1, round(args.metrics_seconds * args.fps))
        main_metrics = metrics_card_main(main_metrics_path, args.width, args.height)
        write_still(writer, main_metrics, metrics_frames)
        manifest["sections"].append({"type": "metrics_card", "name": "main_vizdoom_metrics_05s", "path": str(main_metrics_path), "written_frames": metrics_frames})

        title = title_card(
            args.width,
            args.height,
            "External Dataset Sanity Checks",
            "Same WIT-VZ formulation, shown as domain-shift evidence rather than generalization proof.",
            [
                "MiniWorld / AI2-THOR: real simulator branch rollouts, columns = CV / GT / Ours",
                "ProcTHOR / DeepMind Lab / Habitat / MineDojo: converted external-domain overview cards",
                "PointNav/A* oracles are ViZDoom pose-graph baselines, so they are excluded here.",
            ],
        )
        title_frames = max(1, round(args.title_seconds * args.fps))
        write_still(writer, title, title_frames)
        manifest["sections"].append({"type": "title_card", "name": "external_section_intro", "written_frames": title_frames})

        for path in external_videos:
            manifest["sections"].append({"type": "video", "name": path.stem, **write_video_frames(writer, path, args.width, args.height)})

        external_metrics = metrics_card_external(root, args.width, args.height)
        write_still(writer, external_metrics, metrics_frames)
        manifest["sections"].append({"type": "metrics_card", "name": "external_zero_shot_metrics_03s", "paths": [str(path) for path in external_metric_paths], "written_frames": metrics_frames})

        overview_frames = max(1, round(args.overview_seconds * args.fps))
        for title_text, subtitle, path in external_overviews:
            frame = image_card(path, args.width, args.height, title_text, subtitle)
            write_still(writer, frame, overview_frames)
            manifest["sections"].append({"type": "overview_card", "name": title_text, "path": str(path), "written_frames": overview_frames})
    finally:
        writer.release()

    poster = write_poster(output)
    manifest["poster"] = str(poster)
    manifest["total_frames"] = sum(int(section.get("written_frames", 0)) for section in manifest["sections"])
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = [
        "# Final Main Demo With External Section",
        "",
        "- First section: real ViZDoom 5-baseline counterfactual rollout.",
        "- Numeric card 1: ViZDoom 5s ADE/FDE for CV, PointNav, A*, and ours.",
        "- Final section: external-domain sanity checks.",
        "- MiniWorld and AI2-THOR are real simulator counterfactual rollouts with CV / GT / Ours.",
        "- Numeric card 2: external zero-shot 3s ADE/FDE and CV comparison.",
        "- ProcTHOR, DeepMind Lab, Habitat, and MineDojo are static overview cards.",
        "- External section is formulation/domain-shift evidence, not broad generalization proof.",
        "",
        f"- Output: `{output.as_posix()}`",
        f"- Poster: `{poster.as_posix()}`",
        f"- Total frames: {manifest['total_frames']}",
        f"- Duration: {manifest['total_frames'] / args.fps:.1f} seconds at {args.fps} FPS",
    ]
    (report_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "poster": str(poster), "total_frames": manifest["total_frames"]}, indent=2))


if __name__ == "__main__":
    main()
