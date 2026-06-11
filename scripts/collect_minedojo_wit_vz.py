"""Collect MineDojo Minecraft rollouts into the WIT-VZ raw schema.

MineDojo exposes first-person RGB plus privileged location statistics. This
collector uses those privileged pose observations only to build evaluation
labels; the downstream predictor still receives RGB history and ego-motion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wit_vz.geometry import compute_relative_egomotion, wrap_degrees
from src.wit_vz.io import write_json, write_jsonl


DEFAULT_BIOMES = ["plains", "forest", "desert", "taiga"]
ACTION_POLICY = [
    "forward",
    "forward",
    "forward",
    "forward_sprint",
    "strafe_left",
    "strafe_right",
    "look_left",
    "look_right",
    "forward_look_left",
    "forward_look_right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--run-id", default="minedojo_demo_001")
    parser.add_argument("--biomes", nargs="+", default=DEFAULT_BIOMES)
    parser.add_argument("--episodes-per-biome", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--start-time", type=int, default=6000)
    parser.add_argument("--flat-world", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_minedojo() -> Any:
    try:
        import minedojo
    except ImportError as exc:
        raise RuntimeError(
            "MineDojo is not installed in this Python environment. Install a "
            "Java 8 MineDojo environment first, then rerun this collector."
        ) from exc
    return minedojo


def make_env(args: argparse.Namespace, minedojo: Any, biome: str, seed: int) -> Any:
    world_kwargs: dict[str, Any]
    if args.flat_world:
        world_kwargs = {"generate_world_type": "flat"}
    else:
        world_kwargs = {
            "generate_world_type": "specified_biome",
            "specified_biome": biome,
            "world_seed": seed,
        }
    return minedojo.make(
        "open-ended",
        image_size=(args.height, args.width),
        allow_mob_spawn=False,
        allow_time_passage=False,
        start_time=args.start_time,
        seed=seed,
        **world_kwargs,
    )


def pose_from_observation(obs: dict[str, Any]) -> dict[str, float]:
    stats = obs.get("location_stats") or {}
    pos = np.asarray(stats["pos"], dtype=float).reshape(-1)
    yaw = float(np.asarray(stats["yaw"], dtype=float).reshape(-1)[0])
    if pos.size < 3:
        raise RuntimeError("MineDojo location_stats.pos did not expose expected xyz values.")
    # Minecraft yaw 0 faces +Z. WIT-VZ expects angle 0 to face +world-X.
    # Use world-x := minecraft z and world-y := minecraft x, then invert yaw.
    return {
        "x": float(pos[2]),
        "y": float(pos[0]),
        "z": float(pos[1]),
        "angle": wrap_degrees(-yaw),
    }


def save_rgb(obs: dict[str, Any], path: Path) -> None:
    frame = np.asarray(obs["rgb"])
    if frame.ndim != 3:
        raise RuntimeError(f"Expected MineDojo rgb to be 3D, got shape {frame.shape}")
    if frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).convert("RGB").save(path)


def scalar_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): scalar_metadata(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.astype(float).reshape(-1).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_action(env: Any, action_name: str) -> np.ndarray:
    action = np.asarray(env.action_space.no_op()).copy()
    yaw_center = int(action[4])
    if action_name in {"forward", "forward_sprint", "forward_look_left", "forward_look_right"}:
        action[0] = 1
    if action_name == "forward_sprint":
        action[2] = 3
    if action_name == "strafe_left":
        action[1] = 1
    elif action_name == "strafe_right":
        action[1] = 2
    if action_name in {"look_left", "forward_look_left"}:
        action[4] = max(0, yaw_center - 1)
    elif action_name in {"look_right", "forward_look_right"}:
        action[4] = min(int(env.action_space.nvec[4]) - 1, yaw_center + 1)
    return action


def collect_episode(
    env: Any,
    run_dir: Path,
    biome: str,
    episode_index: int,
    max_steps: int,
    fps: float,
    rng: random.Random,
) -> dict[str, Any]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    obs = env.reset()

    records: list[dict[str, Any]] = []
    previous_pose: dict[str, float] | None = None
    total_reward = 0.0
    reason_end = "max_steps"
    for step in range(max_steps):
        pose = pose_from_observation(obs)
        egomotion = compute_relative_egomotion(previous_pose, pose)
        action_name = rng.choice(ACTION_POLICY)
        action = make_action(env, action_name)

        frame_rel = Path("episodes") / episode_id / "frames" / f"{step:06d}.png"
        save_rgb(obs, run_dir / frame_rel)

        record = {
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
            "reward": 0.0,
            "done": False,
            "metadata": {
                "biome": biome,
                "source_dataset": "minedojo",
                "location_stats": scalar_metadata(obs.get("location_stats", {})),
            },
        }

        previous_pose = pose
        obs, reward, done, _info = env.step(action)
        total_reward += float(reward)
        record["reward"] = float(reward)
        record["done"] = bool(done)
        records.append(record)
        if done:
            reason_end = "terminated"
            break

    summary = {
        "episode_id": episode_id,
        "num_steps": len(records),
        "biome": biome,
        "total_reward": total_reward,
        "start_pose": records[0]["pose"] if records else None,
        "final_pose": records[-1]["pose"] if records else None,
        "reason_end": reason_end,
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

    minedojo = import_minedojo()
    rng = random.Random(args.seed)
    summaries: list[dict[str, Any]] = []
    episode_index = 1
    for biome in args.biomes:
        env_seed = rng.randrange(0, 2**31 - 1)
        env = make_env(args, minedojo, biome, env_seed)
        try:
            for _ in range(args.episodes_per_biome):
                summary = collect_episode(
                    env,
                    run_dir,
                    biome,
                    episode_index,
                    args.max_steps,
                    args.fps,
                    rng,
                )
                summaries.append(summary)
                print(
                    f"{summary['episode_id']}: biome={biome} "
                    f"steps={summary['num_steps']} seed={env_seed}"
                )
                episode_index += 1
        finally:
            env.close()

    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "minedojo",
        "env_name": "minedojo",
        "scenario": "minedojo_open_ended",
        "map": "minecraft_world",
        "fps": args.fps,
        "frame_skip": 1,
        "width": args.width,
        "height": args.height,
        "num_episodes": len(summaries),
        "total_steps": int(sum(item["num_steps"] for item in summaries)),
        "collector": Path(__file__).name,
        "notes": (
            "MineDojo privileged location_stats are used only to create pose and "
            "future-path supervision. RGB and ego-motion remain the predictor inputs."
        ),
        "episode_summaries": summaries,
        "episodes": [
            {
                "episode_id": item["episode_id"],
                "steps_path": f"episodes/{item['episode_id']}/steps.jsonl",
                "num_steps": item["num_steps"],
                "biome": item["biome"],
            }
            for item in summaries
        ],
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote {run_dir} with {manifest['total_steps']} steps")


if __name__ == "__main__":
    main()
