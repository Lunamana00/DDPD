"""Prepare external datasets into a WIT-VZ-like normalized schema.

The supported datasets require licenses, credentials, or separate simulators in
many environments, so this module provides explicit placeholders that fail
gracefully while documenting the expected output schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED = {"alfred", "rxr", "minerl"}


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
            "rgb_history_paths": "list[str] or video frame paths",
            "relative_egomotion_history": "list[[dx_forward, dy_right, dyaw]] if available",
            "future_local_path": "list[[dx_forward, dy_right]] if pose/path is available",
            "episode_id": "str",
            "metadata": "dict",
        },
    }
    instructions = {
        "alfred": "Download ALFRED according to its official license and export trajectories with pose.",
        "rxr": "Prepare RxR/Matterport path metadata after obtaining Matterport access.",
        "minerl": "Install MineRL and export episodes with frames and agent pose/action metadata.",
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
