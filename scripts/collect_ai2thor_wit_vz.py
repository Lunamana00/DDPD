"""Collect AI2-THOR navigation rollouts into the WIT-VZ raw schema."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wit_vz.geometry import compute_relative_egomotion
from src.wit_vz.io import write_json, write_jsonl


DEFAULT_SCENES = ["FloorPlan1", "FloorPlan2", "FloorPlan201", "FloorPlan301"]
ACTION_POLICY = ["MoveAhead", "MoveAhead", "MoveAhead", "RotateLeft", "RotateRight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--run-id", default="ai2thor_nav_001")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--episodes-per-scene", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--rotate-step-degrees", type=float, default=45.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_ai2thor() -> Any:
    try:
        from ai2thor.controller import Controller
    except ImportError as exc:
        raise RuntimeError(
            "AI2-THOR is not installed. Install it in this environment before "
            "running this collector, for example: uv pip install ai2thor"
        ) from exc
    return Controller


def save_frame(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def pose_from_event(event: Any) -> dict[str, float]:
    agent = event.metadata["agent"]
    position = agent["position"]
    rotation = agent["rotation"]
    return {
        "x": float(position["z"]),
        "y": float(position["x"]),
        "z": float(position["y"]),
        "angle": float(rotation["y"]),
    }


def choose_action(rng: random.Random, failed_count: int) -> str:
    if failed_count >= 2:
        return rng.choice(["RotateLeft", "RotateRight"])
    return rng.choice(ACTION_POLICY)


def reset_to_random_reachable_pose(controller: Any, rng: random.Random) -> Any:
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata.get("actionReturn") or []
    if not positions:
        return event
    position = rng.choice(positions)
    rotation = {"x": 0.0, "y": float(rng.choice([0, 45, 90, 135, 180, 225, 270, 315])), "z": 0.0}
    return controller.step(action="Teleport", position=position, rotation=rotation, horizon=0.0, standing=True)


def collect_episode(
    controller: Any,
    run_dir: Path,
    scene: str,
    episode_index: int,
    max_steps: int,
    rng: random.Random,
) -> dict[str, Any]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    event = reset_to_random_reachable_pose(controller, rng)

    records: list[dict[str, Any]] = []
    previous_pose: dict[str, float] | None = None
    failed_count = 0
    for step in range(max_steps):
        pose = pose_from_event(event)
        egomotion = compute_relative_egomotion(previous_pose, pose)
        frame_rel = Path("episodes") / episode_id / "frames" / f"{step:06d}.png"
        save_frame(event.frame, run_dir / frame_rel)

        action = choose_action(rng, failed_count)
        records.append(
            {
                "sample_id": f"{episode_id}_{step:06d}",
                "episode_id": episode_id,
                "step": step,
                "global_step": step,
                "timestamp": float(step),
                "frame_path": frame_rel.as_posix(),
                "pose": pose,
                "relative_egomotion_from_prev": egomotion,
                "action": {"action_name": action},
                "reward": 0.0,
                "done": False,
                "metadata": {
                    "scene": scene,
                    "source_dataset": "ai2thor",
                    "last_action_success": bool(event.metadata.get("lastActionSuccess", True)),
                },
            }
        )

        previous_pose = pose
        event = controller.step(action=action)
        if bool(event.metadata.get("lastActionSuccess", True)):
            failed_count = 0
        else:
            failed_count += 1

    summary = {
        "episode_id": episode_id,
        "num_steps": len(records),
        "scene": scene,
        "start_pose": records[0]["pose"] if records else None,
        "final_pose": records[-1]["pose"] if records else None,
        "reason_end": "max_steps",
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

    Controller = import_ai2thor()
    controller = Controller(
        width=args.width,
        height=args.height,
        gridSize=args.grid_size,
        rotateStepDegrees=args.rotate_step_degrees,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )

    rng = random.Random(args.seed)
    summaries: list[dict[str, Any]] = []
    episode_index = 1
    try:
        for scene in args.scenes:
            controller.reset(scene=scene)
            for _ in range(args.episodes_per_scene):
                summary = collect_episode(controller, run_dir, scene, episode_index, args.max_steps, rng)
                summaries.append(summary)
                episode_index += 1
                print(f"{summary['episode_id']}: scene={scene} steps={summary['num_steps']}")
    finally:
        controller.stop()

    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "ai2thor",
        "env_name": "ai2thor",
        "scenario": "ithor_navigation",
        "map": "multi_scene",
        "fps": 1.0,
        "frame_skip": 1,
        "episode_count": len(summaries),
        "max_steps": args.max_steps,
        "generation_mode": "ai2thor_scripted_navigation",
        "policy": "random_walk_with_failure_recovery",
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
    print(f"Wrote AI2-THOR WIT-VZ raw run to: {run_dir}")


if __name__ == "__main__":
    main()
