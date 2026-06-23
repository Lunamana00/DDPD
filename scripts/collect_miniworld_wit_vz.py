"""Collect MiniWorld rollouts into the WIT-VZ raw schema.

MiniWorld is useful as a light out-of-domain sanity check: it is still
first-person RGB navigation, but it is visually and dynamically different from
ViZDoom. The resulting raw run can be passed to ``src.wit_vz.build_samples``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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


DEFAULT_ENVS = [
    "MiniWorld-Hallway-v0",
    "MiniWorld-Maze-v0",
    "MiniWorld-WallGap-v0",
    "MiniWorld-ThreeRooms-v0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--run-id", default="miniworld_nav_001")
    parser.add_argument("--env-ids", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--episodes-per-env", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=951)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Nominal sampling rate written to the WIT-VZ manifest. One MiniWorld step is treated as one frame.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_gymnasium() -> Any:
    try:
        import gymnasium as gym
        import miniworld  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MiniWorld is not installed. Install it first, for example: "
            "uv pip install miniworld gymnasium"
        ) from exc
    return gym


def observation_to_frame(observation: Any, env: Any) -> np.ndarray:
    if isinstance(observation, dict):
        for key in ("image", "rgb", "observation"):
            value = observation.get(key)
            if isinstance(value, np.ndarray) and value.ndim == 3:
                return value
        for value in observation.values():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                return value
    if isinstance(observation, np.ndarray) and observation.ndim == 3:
        return observation
    rendered = env.render()
    if isinstance(rendered, np.ndarray) and rendered.ndim == 3:
        return rendered
    raise RuntimeError("Could not extract an RGB frame from MiniWorld observation/render output.")


def resize_frame(frame: np.ndarray, width: int, height: int) -> Image.Image:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame).convert("RGB").resize((width, height))


def pose_from_env(env: Any) -> dict[str, float]:
    unwrapped = getattr(env, "unwrapped", env)
    agent = getattr(unwrapped, "agent", None)
    if agent is None:
        raise RuntimeError("MiniWorld env does not expose unwrapped.agent; cannot compute trajectory labels.")
    pos = getattr(agent, "pos", None)
    direction = getattr(agent, "dir", None)
    if pos is None or direction is None:
        raise RuntimeError("MiniWorld agent does not expose pos/dir; cannot compute trajectory labels.")
    pos_array = np.asarray(pos, dtype=float).reshape(-1)
    if pos_array.size < 3:
        raise RuntimeError(f"Unexpected MiniWorld agent.pos shape: {pos_array.shape}")
    return {
        "x": float(pos_array[0]),
        "y": float(pos_array[2]),
        "z": float(pos_array[1]),
        "angle": math.degrees(float(direction)),
    }


def action_candidates(env: Any) -> list[int]:
    unwrapped = getattr(env, "unwrapped", env)
    actions = getattr(unwrapped, "actions", None)
    names = ("move_forward", "move_forward", "move_forward", "turn_left", "turn_right")
    candidates = []
    for name in names:
        if actions is not None and hasattr(actions, name):
            candidates.append(int(getattr(actions, name)))
    if candidates:
        return candidates
    if hasattr(env, "action_space") and getattr(env.action_space, "n", 0) >= 3:
        return [0, 0, 0, 1, 2]
    raise RuntimeError("Could not infer MiniWorld navigation action ids.")


def action_name(env: Any, action: int) -> str:
    unwrapped = getattr(env, "unwrapped", env)
    actions = getattr(unwrapped, "actions", None)
    if actions is not None:
        for name in dir(actions):
            if name.startswith("_"):
                continue
            try:
                if int(getattr(actions, name)) == int(action):
                    return name
            except (TypeError, ValueError):
                continue
    return str(action)


def env_kwargs(env_id: str, width: int, height: int) -> dict[str, Any]:
    # Farama MiniWorld supports render_mode through Gymnasium. Width/height are
    # not accepted by every environment version, so resize frames after render.
    return {"render_mode": "rgb_array"}


def collect_episode(
    env: Any,
    run_dir: Path,
    env_id: str,
    episode_index: int,
    max_steps: int,
    width: int,
    height: int,
    fps: float,
    rng: random.Random,
) -> dict[str, Any]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    seed = rng.randrange(0, 2**31 - 1)
    observation, _info = env.reset(seed=seed)
    actions = action_candidates(env)

    records = []
    previous_pose: dict[str, float] | None = None
    terminated = False
    truncated = False
    total_reward = 0.0
    for step in range(max_steps):
        pose = pose_from_env(env)
        egomotion = compute_relative_egomotion(previous_pose, pose)
        frame = observation_to_frame(observation, env)
        frame_rel = Path("episodes") / episode_id / "frames" / f"{step:06d}.png"
        (run_dir / frame_rel).parent.mkdir(parents=True, exist_ok=True)
        resize_frame(frame, width, height).save(run_dir / frame_rel)

        action = rng.choice(actions)
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
                "action": {"action_id": int(action), "action_name": action_name(env, action)},
                "reward": 0.0,
                "done": False,
                "metadata": {
                    "env_id": env_id,
                    "source_dataset": "miniworld",
                    "episode_seed": seed,
                },
            }
        )
        previous_pose = pose
        observation, reward, terminated, truncated, info = env.step(action)
        records[-1]["reward"] = float(reward)
        records[-1]["done"] = bool(terminated or truncated)
        records[-1]["metadata"]["step_info"] = {
            key: value
            for key, value in dict(info or {}).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        total_reward += float(reward)
        if terminated or truncated:
            break

    summary = {
        "episode_id": episode_id,
        "num_steps": len(records),
        "env_id": env_id,
        "seed": seed,
        "total_reward": total_reward,
        "start_pose": records[0]["pose"] if records else None,
        "final_pose": records[-1]["pose"] if records else None,
        "reason_end": "terminated" if terminated else "truncated" if truncated else "max_steps",
    }
    write_jsonl(episode_dir / "steps.jsonl", records)
    write_json(episode_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.out_root / args.run_id
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gym = import_gymnasium()
    rng = random.Random(args.seed)
    summaries: list[dict[str, Any]] = []
    episode_index = 1
    for env_id in args.env_ids:
        env = gym.make(env_id, **env_kwargs(env_id, args.width, args.height))
        try:
            for _ in range(args.episodes_per_env):
                summary = collect_episode(
                    env,
                    run_dir,
                    env_id,
                    episode_index,
                    args.max_steps,
                    args.width,
                    args.height,
                    args.fps,
                    rng,
                )
                summaries.append(summary)
                episode_index += 1
                print(f"{summary['episode_id']}: env={env_id} steps={summary['num_steps']}")
        finally:
            env.close()

    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "miniworld",
        "env_name": "miniworld",
        "scenario": "miniworld_navigation",
        "map": "multi_env",
        "fps": args.fps,
        "frame_skip": 1,
        "episode_count": len(summaries),
        "max_steps": args.max_steps,
        "generation_mode": "miniworld_scripted_navigation",
        "policy": "random_walk",
        "seed": args.seed,
        "enabled_buffers": {"rgb": True, "depth": False, "labels": False, "automap": False},
        "episodes": [
            {
                "episode_id": item["episode_id"],
                "steps_path": f"episodes/{item['episode_id']}/steps.jsonl",
                "summary_path": f"episodes/{item['episode_id']}/summary.json",
            }
            for item in summaries
        ],
        "episode_summaries": summaries,
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote MiniWorld WIT-VZ raw run to: {run_dir}")


if __name__ == "__main__":
    main()
