"""Record a human ViZDoom session and optionally build WIT-VZ samples."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="deadly_corridor")
    parser.add_argument("--map", default="map01")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--player-id", default="player_001")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--raw-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--processed-out", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--screen-width", type=int, default=320)
    parser.add_argument("--screen-height", type=int, default=240)
    parser.add_argument("--human-countdown-sec", type=float, default=5.0)
    parser.add_argument("--history-sec", type=float, default=1.0)
    parser.add_argument("--future-sec", type=float, default=3.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--split", choices=["episode", "map", "source"], default="episode")
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def default_run_id() -> str:
    return "wit_vz_human_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    args = parse_args()
    run_id = args.run_id or default_run_id()
    session_id = args.session_id or run_id
    processed_out = args.processed_out or Path("data/wit_vz/processed") / run_id
    raw_dir = args.raw_root / run_id

    if not args.build_only:
        command = [
            sys.executable,
            "-m",
            "src.wit_vz.collect",
            "--scenario",
            args.scenario,
            "--map",
            args.map,
            "--run-id",
            run_id,
            "--out-root",
            args.raw_root.as_posix(),
            "--episodes",
            str(args.episodes),
            "--max-steps",
            str(args.max_steps),
            "--frame-skip",
            str(args.frame_skip),
            "--mode",
            "human",
            "--visible",
            "--player-id",
            args.player_id,
            "--session-id",
            session_id,
            "--screen-width",
            str(args.screen_width),
            "--screen-height",
            str(args.screen_height),
            "--human-countdown-sec",
            str(args.human_countdown_sec),
        ]
        if args.overwrite:
            command.append("--overwrite")
        run(command)

    if not args.collect_only:
        command = [
            sys.executable,
            "-m",
            "src.wit_vz.build_samples",
            "--raw",
            raw_dir.as_posix(),
            "--out",
            processed_out.as_posix(),
            "--history-sec",
            str(args.history_sec),
            "--future-sec",
            str(args.future_sec),
            "--sample-fps",
            str(args.sample_fps),
            "--stride",
            str(args.stride),
            "--split",
            args.split,
            "--seed",
            str(args.seed),
        ]
        run(command)


if __name__ == "__main__":
    main()
