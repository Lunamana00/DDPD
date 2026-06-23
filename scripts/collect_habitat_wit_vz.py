"""Collect Habitat-Sim rollouts into the WIT-VZ raw schema."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


ACTION_POLICY = ["move_forward", "move_forward", "move_forward", "turn_left", "turn_right"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--run-id", default="habitat_demo_001")
    parser.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="Path to a Habitat scene asset, e.g. habitat-test-scenes/skokloster-castle.glb.",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--sensor-height", type=float, default=1.5)
    parser.add_argument("--forward-step", type=float, default=0.25)
    parser.add_argument("--turn-degrees", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_habitat_sim() -> Any:
    try:
        import habitat_sim
    except ImportError as exc:
        raise RuntimeError(
            "Habitat-Sim is not installed in this Python environment. Install "
            "a headless habitat-sim environment first."
        ) from exc
    return habitat_sim


def make_sim(args: argparse.Namespace, habitat_sim: Any) -> Any:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(args.scene.expanduser().resolve())
    sim_cfg.enable_physics = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color_sensor"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [args.height, args.width]
    sensor.position = [0.0, args.sensor_height, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=args.forward_step)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=args.turn_degrees)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=args.turn_degrees)
        ),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def rotation_matrix(rotation: Any) -> np.ndarray:
    try:
        import quaternion
        return quaternion.as_rotation_matrix(rotation)
    except Exception:
        coeffs = np.asarray([rotation.x, rotation.y, rotation.z, rotation.w], dtype=float)
        x, y, z, w = coeffs
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )


def pose_from_state(state: Any) -> dict[str, float]:
    position = np.asarray(state.position, dtype=float).reshape(-1)
    # Habitat's default forward direction is negative z. Convert to a 2D plane
    # where x=world x and y=-world z, then compute yaw in that same plane.
    rot = rotation_matrix(state.rotation)
    forward_world = rot @ np.array([0.0, 0.0, -1.0])
    forward_2d = np.array([forward_world[0], -forward_world[2]], dtype=float)
    yaw = math.degrees(math.atan2(float(forward_2d[1]), float(forward_2d[0])))
    return {
        "x": float(position[0]),
        "y": float(-position[2]),
        "z": float(position[1]),
        "angle": yaw,
    }


def reset_agent(sim: Any, rng: random.Random) -> None:
    agent = sim.get_agent(0)
    state = agent.get_state()
    point = sim.pathfinder.get_random_navigable_point()
    if not np.isfinite(point).all():
        point = state.position
    state.position = point
    yaw = math.radians(rng.uniform(-180.0, 180.0))
    try:
        import quaternion
        state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
    except Exception:
        pass
    agent.set_state(state)


def save_frame(observations: dict[str, np.ndarray], path: Path) -> None:
    frame = observations["color_sensor"]
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).convert("RGB").save(path)


def collect_episode(
    sim: Any,
    run_dir: Path,
    scene_name: str,
    episode_index: int,
    max_steps: int,
    fps: float,
    rng: random.Random,
) -> dict[str, Any]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    reset_agent(sim, rng)
    observations = sim.get_sensor_observations()

    records: list[dict[str, Any]] = []
    previous_pose: dict[str, float] | None = None
    for step in range(max_steps):
        state = sim.get_agent(0).get_state()
        pose = pose_from_state(state)
        egomotion = compute_relative_egomotion(previous_pose, pose)
        action = rng.choice(ACTION_POLICY)

        frame_rel = Path("episodes") / episode_id / "frames" / f"{step:06d}.png"
        save_frame(observations, run_dir / frame_rel)

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
                "action": {"action_name": action},
                "reward": 0.0,
                "done": False,
                "metadata": {
                    "scene": scene_name,
                    "source_dataset": "habitat_sim",
                    "collided": bool(getattr(sim, "previous_step_collided", False)),
                },
            }
        )
        previous_pose = pose
        observations = sim.step(action)

    summary = {
        "episode_id": episode_id,
        "num_steps": len(records),
        "scene": scene_name,
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

    habitat_sim = import_habitat_sim()
    rng = random.Random(args.seed)
    sim = make_sim(args, habitat_sim)
    try:
        scene_name = args.scene.stem
        summaries = [
            collect_episode(sim, run_dir, scene_name, idx, args.max_steps, args.fps, rng)
            for idx in range(1, args.episodes + 1)
        ]
    finally:
        sim.close()

    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "habitat_sim",
        "env_name": "habitat_sim",
        "scenario": "habitat_sim",
        "map": args.scene.stem,
        "scene": str(args.scene),
        "fps": args.fps,
        "frame_skip": 1,
        "width": args.width,
        "height": args.height,
        "num_episodes": len(summaries),
        "total_steps": int(sum(item["num_steps"] for item in summaries)),
        "collector": Path(__file__).name,
        "notes": (
            "Habitat-Sim agent state is converted to WIT-VZ local coordinates. "
            "This is an external photorealistic navigation demo."
        ),
        "episode_summaries": summaries,
        "episodes": [
            {
                "episode_id": item["episode_id"],
                "steps_path": f"episodes/{item['episode_id']}/steps.jsonl",
                "num_steps": item["num_steps"],
                "scene": item["scene"],
            }
            for item in summaries
        ],
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote Habitat raw run: {run_dir}")


if __name__ == "__main__":
    main()
