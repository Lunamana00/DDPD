"""Prepare external datasets into a WIT-VZ-like normalized schema.

The supported datasets require licenses, credentials, or separate simulators in
many environments, so this module provides explicit placeholders that fail
gracefully while documenting the expected output schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED = {
    "ai2thor",
    "alfred",
    "deepmind_lab",
    "habitat",
    "minerl",
    "procthor",
    "rxr",
}

DATASET_RECIPES = {
    "ai2thor": {
        "env_name": "ai2thor",
        "priority": 1,
        "requires": ["AI2-THOR install", "local simulator rendering"],
        "split_keys": ["scene", "house_id", "episode_id"],
        "pose_fields": ["position.x", "position.z", "rotation.y"],
        "notes": "Best first external source because RGB, agent pose, and ProcTHOR houses share the same tooling.",
    },
    "procthor": {
        "env_name": "ai2thor_procthor",
        "priority": 2,
        "requires": ["AI2-THOR install", "ProcTHOR generated house metadata"],
        "split_keys": ["house_id", "room_type", "episode_id"],
        "pose_fields": ["position.x", "position.z", "rotation.y"],
        "notes": "Use house-disjoint splits to prove generalization beyond memorized layouts.",
    },
    "habitat": {
        "env_name": "habitat",
        "priority": 3,
        "requires": ["Habitat-Sim install", "licensed scene assets such as HM3D or Replica"],
        "split_keys": ["scene_id", "episode_id"],
        "pose_fields": ["agent_state.position", "agent_state.rotation"],
        "notes": "Strong paper signal, but asset licensing and renderer setup are heavier.",
    },
    "deepmind_lab": {
        "env_name": "deepmind_lab",
        "priority": 4,
        "requires": ["DeepMind Lab build", "custom trajectory logger"],
        "split_keys": ["level", "episode_id"],
        "pose_fields": ["position", "yaw"],
        "notes": "Useful for game-like visual diversity after AI2-THOR/Habitat are in place.",
    },
    "alfred": {
        "env_name": "alfred",
        "priority": 5,
        "requires": ["ALFRED download", "AI2-THOR-compatible replay"],
        "split_keys": ["scene", "task_type", "episode_id"],
        "pose_fields": ["planner_action pose or replayed AI2-THOR agent pose"],
        "notes": "Task trajectories are semantically rich, but replay/pose normalization needs care.",
    },
    "rxr": {
        "env_name": "rxr_matterport",
        "priority": 6,
        "requires": ["RxR metadata", "Matterport access"],
        "split_keys": ["scan_id", "path_id"],
        "pose_fields": ["viewpoint graph pose"],
        "notes": "Good language-navigation diversity, not first choice for dense egocentric future paths.",
    },
    "minerl": {
        "env_name": "minerl",
        "priority": 7,
        "requires": ["MineRL install", "trajectory export"],
        "split_keys": ["task", "episode_id"],
        "pose_fields": ["agent position/yaw if exported"],
        "notes": "Visual domain is very different; use after indoor simulators to test robustness.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare external path datasets.")
    parser.add_argument("--dataset", choices=sorted(SUPPORTED), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recipe = DATASET_RECIPES[args.dataset]
    schema = {
        "source_dataset": args.dataset,
        "status": "manual_preparation_required",
        "limit": args.limit,
        "recipe": recipe,
        "normalized_schema": {
            "source": {
                "source_id": "stable source/run identifier",
                "env_name": "simulator or dataset family",
                "source_dataset": "dataset name",
                "raw_run_id": "raw collection or original trajectory id",
            },
            "rgb_history_paths": "list[str] or video frame paths",
            "relative_egomotion_history": "list[[dx_forward, dy_right, dyaw]] if available",
            "future_local_path": "list[[dx_forward, dy_right]] if pose/path is available",
            "episode_id": "str",
            "metadata": "dict",
        },
        "required_sample_fields": [
            "sample_id",
            "episode_id",
            "center_step",
            "rgb_history_paths",
            "relative_egomotion_history",
            "future_local_path",
            "future_world_path",
            "current_pose",
            "source",
            "metadata",
        ],
        "recommended_metadata": {
            "scenario": "scene/house/level identifier",
            "map_id": "specific map, house, scan, or scene id",
            "policy": "random_walk, shortest_path, goal_directed, expert, noisy_policy, etc.",
            "split_group": "scene-disjoint or house-disjoint group key",
        },
        "coordinate_convention": "local x=forward, local y=right, origin=current pose",
    }
    instructions = {
        "ai2thor": "Install AI2-THOR, export reachable navigation rollouts with RGB frames and agent pose, then convert pose windows to local future paths.",
        "alfred": "Download ALFRED according to its official license and export trajectories with pose.",
        "deepmind_lab": "Build DeepMind Lab on a supported Linux host, record first-person RGB and agent pose/actions, then export WIT-VZ-compatible windows.",
        "habitat": "Obtain Habitat scene assets such as HM3D/Replica under their licenses, render PointNav-style trajectories, and export RGB plus agent poses.",
        "rxr": "Prepare RxR/Matterport path metadata after obtaining Matterport access.",
        "minerl": "Install MineRL and export episodes with frames and agent pose/action metadata.",
        "procthor": "Install AI2-THOR/ProcTHOR, generate procedural houses, and export navigation rollouts with RGB and pose metadata.",
    }
    schema["instructions"] = instructions[args.dataset]
    (args.out / "manifest.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (args.out / "samples.jsonl").write_text("", encoding="utf-8")
    readme = [
        f"# {args.dataset} External Dataset Stub",
        "",
        "This directory is intentionally a normalized-contract stub.",
        "Fill `samples.jsonl` with WIT-VZ-compatible records after obtaining the required simulator/assets.",
        "",
        "Minimum acceptance checks:",
        "- RGB frame paths resolve from `dataset_manifest.json` raw roots.",
        "- `future_local_path` uses local x-forward/y-right coordinates.",
        "- Evaluation includes scene/house/source-disjoint splits.",
        "- `metadata.policy` is populated so policy-balanced training can be enabled.",
        "",
    ]
    (args.out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(
        f"{args.dataset} direct download was not attempted. "
        f"Wrote normalized schema placeholder to {args.out}"
    )


if __name__ == "__main__":
    main()
