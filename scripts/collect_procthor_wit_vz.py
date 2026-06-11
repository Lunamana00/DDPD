"""Collect ProcTHOR procedural-house rollouts into the WIT-VZ raw schema.

ProcTHOR is sensitive to the AI2-THOR build it is paired with. For the current
demo, use a source checkout of https://github.com/allenai/procthor through
``--procthor-source-root`` so the house schema matches AI2-THOR 5 CloudRendering.
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
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


ACTION_POLICY = ["MoveAhead", "MoveAhead", "MoveAhead", "RotateLeft", "RotateRight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--run-id", default="procthor_nav_001")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--rotate-step-degrees", type=float, default=90.0)
    parser.add_argument("--platform", default="CloudRendering")
    parser.add_argument("--gpu-device", type=int, default=None)
    parser.add_argument(
        "--procthor-source-root",
        type=Path,
        default=None,
        help="Optional local checkout of github.com/allenai/procthor to prepend to PYTHONPATH.",
    )
    parser.add_argument(
        "--vulkan-library",
        type=Path,
        default=None,
        help="Optional libvulkan.so path for rootless CloudRendering setups.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def patch_vulkan_find_library(vulkan_library: Path | None) -> None:
    if vulkan_library is None:
        return
    resolved = vulkan_library.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Requested --vulkan-library does not exist: {resolved}")
    original_find_library = ctypes.util.find_library

    def find_library_with_vulkan(name: str) -> str | None:
        if name == "vulkan":
            return str(resolved)
        return original_find_library(name)

    ctypes.util.find_library = find_library_with_vulkan
    lib_dir = str(resolved.parent)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir


def maybe_prepend_procthor_source(path: Path | None) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    if not (resolved / "procthor").is_dir():
        raise FileNotFoundError(f"--procthor-source-root must contain a procthor package: {resolved}")
    sys.path.insert(0, str(resolved))


def resolve_platform(name: str) -> Any:
    if name.lower() in {"", "auto", "none"}:
        return None
    import ai2thor.platform as platform_module

    if not hasattr(platform_module, name):
        valid = ["auto", "CloudRendering", "Linux64", "Windows64", "OSXIntel64"]
        raise ValueError(f"Unsupported AI2-THOR platform {name!r}. Expected one of: {', '.join(valid)}")
    return getattr(platform_module, name)


def save_frame(frame: Any, path: Path) -> None:
    if frame is None:
        raise RuntimeError("AI2-THOR returned no RGB frame during ProcTHOR collection.")
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


def create_controller(args: argparse.Namespace) -> Any:
    from ai2thor.controller import Controller
    from procthor.constants import PROCTHOR_INITIALIZATION

    return Controller(
        width=args.width,
        height=args.height,
        platform=resolve_platform(args.platform),
        gpu_device=args.gpu_device,
        quality="Low",
        gridSize=args.grid_size,
        rotateStepDegrees=args.rotate_step_degrees,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        **PROCTHOR_INITIALIZATION,
    )


def create_house(controller: Any, seed: int) -> Any:
    from procthor.generation import HouseGenerator
    from procthor.generation.room_specs import PROCTHOR10K_ROOM_SPEC_SAMPLER

    generator = HouseGenerator(
        split="train",
        seed=seed,
        controller=controller,
        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
    )
    house, _ = generator.sample()
    event = controller.step(action="CreateHouse", house=house.data)
    if not bool(event.metadata.get("lastActionSuccess", False)):
        message = event.metadata.get("errorMessage", "")
        raise RuntimeError(f"CreateHouse failed for seed={seed}: {message}")
    return house


def teleport_to_house_start(controller: Any, house: Any) -> Any:
    pose = house.choose_agent_pose()
    position = pose["position"]
    rotation = pose.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
    horizon = float(pose.get("horizon", 0.0))
    event = controller.step(
        action="Teleport",
        position=position,
        rotation=rotation,
        horizon=horizon,
        standing=bool(pose.get("standing", True)),
    )
    if not bool(event.metadata.get("lastActionSuccess", False)):
        raise RuntimeError(f"Teleport to ProcTHOR start pose failed: {event.metadata.get('errorMessage', '')}")
    return event


def collect_episode(
    controller: Any,
    run_dir: Path,
    episode_index: int,
    house_seed: int,
    max_steps: int,
    fps: float,
    rng: random.Random,
) -> dict[str, Any]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    house = create_house(controller, house_seed)
    event = teleport_to_house_start(controller, house)

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
                "timestamp": float(step) / max(fps, 1e-6),
                "frame_path": frame_rel.as_posix(),
                "pose": pose,
                "relative_egomotion_from_prev": egomotion,
                "action": {"action_name": action},
                "reward": 0.0,
                "done": False,
                "metadata": {
                    "scene": f"procthor_seed_{house_seed}",
                    "source_dataset": "procthor",
                    "house_seed": house_seed,
                    "rooms": len(house.rooms),
                    "objects": len(house.data.get("objects", [])),
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
        "scene": f"procthor_seed_{house_seed}",
        "house_seed": house_seed,
        "rooms": len(house.rooms),
        "objects": len(house.data.get("objects", [])),
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

    maybe_prepend_procthor_source(args.procthor_source_root)
    patch_vulkan_find_library(args.vulkan_library)

    rng = random.Random(args.seed)
    controller = create_controller(args)
    summaries: list[dict[str, Any]] = []
    try:
        for episode_index in range(1, args.episodes + 1):
            house_seed = args.seed + episode_index
            summary = collect_episode(controller, run_dir, episode_index, house_seed, args.max_steps, args.fps, rng)
            summaries.append(summary)
            print(
                f"{summary['episode_id']}: seed={house_seed} "
                f"rooms={summary['rooms']} objects={summary['objects']} steps={summary['num_steps']}"
            )
    finally:
        controller.stop()

    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "procthor",
        "env_name": "procthor",
        "scenario": "procedural_house_navigation",
        "map": "procedural_houses",
        "fps": args.fps,
        "frame_skip": 1,
        "episode_count": len(summaries),
        "max_steps": args.max_steps,
        "generation_mode": "procthor_scripted_navigation",
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
    print(f"Wrote ProcTHOR WIT-VZ raw run to: {run_dir}")


if __name__ == "__main__":
    main()
