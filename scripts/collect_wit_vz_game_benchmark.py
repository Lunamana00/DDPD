"""Collect a larger multi-source ViZDoom game benchmark and build samples."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/wit_vz_game_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count per run.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per run.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not config.get("runs"):
        raise ValueError(f"No runs configured in {path}")
    return config


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def bool_flag(enabled: bool, positive: str, negative: str) -> str:
    return positive if enabled else negative


def collect_run(
    run: dict[str, Any],
    raw_root: Path,
    args: argparse.Namespace,
) -> Path:
    run_id = str(run["run_id"])
    command = [
        sys.executable,
        "-m",
        "src.wit_vz.collect",
        "--scenario",
        str(run["scenario"]),
        "--map",
        str(run.get("map", "map01")),
        "--run-id",
        run_id,
        "--out-root",
        raw_root.as_posix(),
        "--episodes",
        str(args.episodes or run.get("episodes", 20)),
        "--max-steps",
        str(args.max_steps or run.get("max_steps", 600)),
        "--frame-skip",
        str(run.get("frame_skip", 4)),
        "--seed",
        str(run.get("seed", 7)),
        "--mode",
        str(run.get("mode", "scripted")),
        "--policy",
        str(run.get("policy", "corridor")),
        "--screen-width",
        str(run.get("screen_width", 160)),
        "--screen-height",
        str(run.get("screen_height", 120)),
        bool_flag(bool(run.get("save_rgb", True)), "--save-rgb", "--no-save-rgb"),
    ]

    if bool(run.get("save_depth", False)):
        command.append("--save-depth")
    if bool(run.get("save_labels", False)):
        command.append("--save-labels")
    if bool(run.get("save_automap", False)):
        command.append("--save-automap")
    if args.visible:
        command.append("--visible")
    if args.overwrite:
        command.append("--overwrite")

    run_command(command, args.dry_run)
    return raw_root / run_id


def build_samples(raw_dirs: list[Path], config: dict[str, Any], args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "src.wit_vz.build_samples",
        "--raw",
        *[path.as_posix() for path in raw_dirs],
        "--out",
        str(config["processed_out"]),
        "--history-sec",
        str(config.get("history_sec", 1.0)),
        "--future-sec",
        str(config.get("future_sec", 3.0)),
        "--sample-fps",
        str(config.get("sample_fps", 5.0)),
        "--stride",
        str(config.get("stride", 1)),
        "--split",
        str(config.get("split", "episode")),
        "--seed",
        str(config.get("seed", 301)),
    ]
    run_command(command, args.dry_run)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_root = Path(config.get("raw_root", "data/wit_vz/raw"))
    runs = config["runs"][: args.limit_runs] if args.limit_runs else config["runs"]

    raw_dirs: list[Path] = [raw_root / str(run["run_id"]) for run in runs]
    if not args.build_only:
        raw_dirs = [collect_run(run, raw_root, args) for run in runs]

    if not args.collect_only:
        build_samples(raw_dirs, config, args)


if __name__ == "__main__":
    main()
