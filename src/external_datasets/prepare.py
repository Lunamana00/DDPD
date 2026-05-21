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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare external path datasets.")
    parser.add_argument("--dataset", choices=sorted(SUPPORTED), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    schema = {
        "source_dataset": args.dataset,
        "status": "manual_preparation_required",
        "limit": args.limit,
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
    print(
        f"{args.dataset} direct download was not attempted. "
        f"Wrote normalized schema placeholder to {args.out}"
    )


if __name__ == "__main__":
    main()
