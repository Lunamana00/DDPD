"""Render the presentation demo sequence into a single MP4 video."""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence


TABLE_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("reports/demo/presentation_sequence"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--static-seconds", type=float, default=4.0)
    parser.add_argument("--gif-max-seconds", type=float, default=7.0)
    return parser.parse_args()


def load_sequence(sequence_dir: Path) -> list[tuple[int, Path, str]]:
    readme = sequence_dir / "README.md"
    items: list[tuple[int, Path, str]] = []
    for line in readme.read_text(encoding="utf-8").splitlines():
        match = TABLE_RE.match(line.strip())
        if match:
            order = int(match.group(1))
            file_name = match.group(2)
            message = match.group(3).strip()
            items.append((order, sequence_dir / file_name, message))
    if not items:
        raise RuntimeError(f"No sequence table items found in {readme}")
    return sorted(items, key=lambda item: item[0])


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


def wrap_text(text: str, max_chars: int) -> str:
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False))


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(max_width / image.width, max_height / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def slide_frame(
    image: Image.Image,
    order: int,
    total: int,
    file_path: Path,
    message: str,
    canvas_size: tuple[int, int],
) -> Image.Image:
    width, height = canvas_size
    canvas = Image.new("RGB", canvas_size, (248, 249, 251))
    draw = ImageDraw.Draw(canvas)

    title_font = font(34, bold=True)
    body_font = font(24)
    small_font = font(20)

    title = f"{order:02d}/{total:02d}  {file_path.name}"
    draw.text((54, 34), title, fill=(22, 28, 36), font=title_font)
    wrapped = wrap_text(message, 108)
    draw.multiline_text((54, 83), wrapped, fill=(60, 68, 78), font=body_font, spacing=6)

    top = 160
    bottom = height - 70
    content = fit_image(image, width - 120, bottom - top)
    x = (width - content.width) // 2
    y = top + max(0, (bottom - top - content.height) // 2)
    shadow = Image.new("RGBA", (content.width + 10, content.height + 10), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle((8, 8, content.width + 8, content.height + 8), fill=(0, 0, 0, 30))
    canvas.paste(shadow.convert("RGB"), (x - 5, y - 5))
    canvas.paste(content, (x, y))

    footer = "GT path: green | model prediction: red | constant-velocity baseline: blue"
    draw.text((54, height - 48), footer, fill=(90, 96, 106), font=small_font)
    return canvas


def load_media_frames(path: Path, fps: int, static_seconds: float, gif_max_seconds: float) -> Iterable[Image.Image]:
    if path.suffix.lower() != ".gif":
        image = Image.open(path).convert("RGB")
        repeat = max(1, round(static_seconds * fps))
        for _ in range(repeat):
            yield image
        return

    gif = Image.open(path)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(gif)]
    if not frames:
        return
    durations = [max(1, frame.info.get("duration", gif.info.get("duration", 100))) for frame in ImageSequence.Iterator(gif)]
    expanded: list[Image.Image] = []
    for frame, duration_ms in zip(frames, durations):
        frame_count = max(1, round((duration_ms / 1000.0) * fps))
        expanded.extend([frame] * frame_count)
    max_frames = max(1, round(gif_max_seconds * fps))
    if len(expanded) > max_frames:
        step = len(expanded) / max_frames
        expanded = [expanded[math.floor(i * step)] for i in range(max_frames)]
    for frame in expanded:
        yield frame


def write_video(items: list[tuple[int, Path, str]], args: argparse.Namespace) -> Path:
    output = args.output or (args.sequence_dir / "demo_full_sequence.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")

    total = len(items)
    try:
        for order, path, message in items:
            if not path.exists():
                raise FileNotFoundError(path)
            for media in load_media_frames(path, args.fps, args.static_seconds, args.gif_max_seconds):
                slide = slide_frame(media, order, total, path, message, (args.width, args.height))
                writer.write(cv2.cvtColor(np.asarray(slide), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output


def main() -> None:
    args = parse_args()
    items = load_sequence(args.sequence_dir)
    output = write_video(items, args)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
