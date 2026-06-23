"""Collect DeepMind Lab rollouts into the WIT-VZ raw schema.

DeepMind Lab is a game-like first-person navigation domain. This collector
records RGB frames plus debug pose observations, then writes the same raw
``manifest.json`` and per-episode ``steps.jsonl`` format used by the other
WIT-VZ demo collectors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wit_vz.geometry import compute_relative_egomotion
from src.wit_vz.io import write_json, write_jsonl


DEFAULT_LEVELS = [
    "nav_maze_static_01",
    "nav_maze_random_goal_01",
    "seekavoid_arena_01",
    "lt_chasm",
]

OBSERVATIONS = [
    "RGB_INTERLEAVED",
    "DEBUG.POS.TRANS",
    "DEBUG.POS.ROT",
    "VEL.TRANS",
    "VEL.ROT",
]

ACTION_POLICY: list[tuple[str, list[int]]] = [
    ("forward", [0, 0, 0, 1, 0, 0, 0]),
    ("forward", [0, 0, 0, 1, 0, 0, 0]),
    ("forward", [0, 0, 0, 1, 0, 0, 0]),
    ("strafe_left", [0, 0, -1, 0, 0, 0, 0]),
    ("strafe_right", [0, 0, 1, 0, 0, 0, 0]),
    ("look_left", [-20, 0, 0, 0, 0, 0, 0]),
    ("look_right", [20, 0, 0, 0, 0, 0, 0]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--run-id", default="deepmind_lab_demo_001")
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--episodes-per-level", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Nominal sampling rate written to the WIT-VZ manifest. One DeepMind Lab step is treated as one frame.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_deepmind_lab() -> Any:
    try:
        import deepmind_lab
    except ImportError as exc:
        raise RuntimeError(
            "DeepMind Lab is not installed in this Python environment. Build and "
            "install its wheel first, then rerun this collector."
        ) from exc
    return deepmind_lab


def make_lab(deepmind_lab: Any, level: str, width: int, height: int, fps: float) -> Any:
    return deepmind_lab.Lab(
        level,
        OBSERVATIONS,
        {
            "fps": str(int(round(fps))),
            "width": str(width),
            "height": str(height),
        },
    )


def pose_from_observations(observations: dict[str, np.ndarray]) -> dict[str, float]:
    position = np.asarray(observations["DEBUG.POS.TRANS"], dtype=float).reshape(-1)
    rotation = np.asarray(observations["DEBUG.POS.ROT"], dtype=float).reshape(-1)
    if position.size < 3 or rotation.size < 2:
        raise RuntimeError(
            "DeepMind Lab DEBUG.POS observations did not expose expected xyz/yaw values."
        )
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "angle": float(rotation[1]),
    }


def save_rgb(observations: dict[str, np.ndarray], path: Path) -> None:
    frame = observations["RGB_INTERLEAVED"]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).convert("RGB").save(path)


def choose_action(rng: random.Random) -> tuple[str, np.ndarray]:
    name, values = rng.choice(ACTION_POLICY)
    return name, np.asarray(values, dtype=np.intc)


def scalar_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def collect_episode(
    deepmind_lab: Any,
    run_dir: Path,
    level: str,
    episode_index: int,
    max_steps: int,
    width: int,
    height: int,
    fps: float,
    seed: int,
    rng: random.Random,
) -> dict[str, Any]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    lab = make_lab(deepmind_lab, level, width, height, fps)
    lab.reset(seed=seed)

    records: list[dict[str, Any]] = []
    previous_pose: dict[str, float] | None = None
    total_reward = 0.0
    reason_end = "max_steps"

    for step in range(max_steps):
        if not lab.is_running():
            reason_end = "terminated"
            break
        observations = lab.observations()
        pose = pose_from_observations(observations)
        egomotion = compute_relative_egomotion(previous_pose, pose)
        action_name, action = choose_action(rng)

        frame_rel = Path("episodes") / episode_id / "frames" / f"{step:06d}.png"
        save_rgb(observations, run_dir / frame_rel)
        reward = float(lab.step(action, num_steps=1))
        total_reward += reward

        records.append(
            {
                "sample_id": f"{episode_id}_{step:06d}",
                "episode_id": episode_id,
                "step": step,
                "global_step": step,
                "timestamp": float(step) / max(fps, 1e-6),
                "frame_path": frame_rel.as_posix(),
                "pose": pose,
                "relative_egomotion_from_prev": egomotion,
                "action": {
                    "action_name": action_name,
                    "action_vector": action.astype(int).tolist(),
                },
                "reward": reward,
                "done": not bool(lab.is_running()),
                "metadata": {
                    "level": level,
                    "source_dataset": "deepmind_lab",
                    "episode_seed": seed,
                    "vel_trans": scalar_metadata(observations["VEL.TRANS"]),
                    "vel_rot": scalar_metadata(observations["VEL.ROT"]),
                },
            }
        )
        previous_pose = pose
        if not lab.is_running():
            reason_end = "terminated"
            break

    summary = {
        "episode_id": episode_id,
        "num_steps": len(records),
        "level": level,
        "seed": seed,
        "total_reward": total_reward,
        "start_pose": records[0]["pose"] if records else None,
        "final_pose": records[-1]["pose"] if records else None,
        "reason_end": reason_end,
    }
    write_jsonl(episode_dir / "steps.jsonl", records)
    write_json(episode_dir / "summary.json", summary)
    lab.close()
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.out_root / args.run_id
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    deepmind_lab = import_deepmind_lab()
    rng = random.Random(args.seed)
    summaries: list[dict[str, Any]] = []
    episode_index = 1
    for level in args.levels:
        for _ in range(args.episodes_per_level):
            episode_seed = rng.randrange(0, 2**31 - 1)
            summary = collect_episode(
                deepmind_lab,
                run_dir,
                level,
                episode_index,
                args.max_steps,
                args.width,
                args.height,
                args.fps,
                episode_seed,
                rng,
            )
            summaries.append(summary)
            print(
                f"{summary['episode_id']}: level={level} "
                f"seed={episode_seed} steps={summary['num_steps']}"
            )
            episode_index += 1

    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "deepmind_lab",
        "env_name": "deepmind_lab",
        "scenario": "deepmind_lab",
        "map": "multi_level",
        "levels": args.levels,
        "fps": args.fps,
        "frame_skip": 1,
        "width": args.width,
        "height": args.height,
        "num_episodes": len(summaries),
        "total_steps": int(sum(item["num_steps"] for item in summaries)),
        "collector": Path(__file__).name,
        "notes": (
            "DeepMind Lab debug pose observations are used to generate local "
            "future path labels. This is an external-domain demo, not a "
            "training-set source for the ViZDoom checkpoint."
        ),
        "episode_summaries": summaries,
        "episodes": [
            {
                "episode_id": item["episode_id"],
                "steps_path": f"episodes/{item['episode_id']}/steps.jsonl",
                "num_steps": item["num_steps"],
                "level": item["level"],
                "seed": item["seed"],
            }
            for item in summaries
        ],
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote DeepMind Lab raw run: {run_dir}")


if __name__ == "__main__":
    main()
