"""Build supervised WIT-VZ path-prediction samples from raw episodes."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .geometry import egomotion_history_from_records, world_future_path_to_local
from .io import load_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw WIT-VZ episodes to supervised samples.")
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-sec", type=float, default=3.0)
    parser.add_argument("--future-sec", type=float, default=3.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=["episode", "map", "source"], default="episode")
    parser.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


def safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    normalized = normalized.strip("._-")
    return normalized or "source"


def raw_sample_period(manifest: dict[str, Any]) -> float:
    fps = float(manifest.get("fps", 35.0))
    frame_skip = float(manifest.get("frame_skip", 1.0))
    return frame_skip / fps


def sample_every_raw_step(manifest: dict[str, Any], sample_fps: float) -> int:
    raw_rate = 1.0 / raw_sample_period(manifest)
    return max(1, round(raw_rate / sample_fps))


def draw_preview(path: Path, future_local_path: list[list[float]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 360
    margin = 36
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    cx = size // 2
    cy = size - margin
    draw.line((cx, cy, cx, margin), fill=(40, 40, 40), width=2)
    draw.line((cx, cy, size - margin, cy), fill=(80, 80, 80), width=2)
    draw.text((8, 8), title, fill=(0, 0, 0))
    if future_local_path:
        max_abs = max(max(abs(x), abs(y)) for x, y in future_local_path)
        scale = (size - 2 * margin) / max(max_abs * 2.2, 1.0)
        points = [(cx + y * scale, cy - x * scale) for x, y in future_local_path]
        draw.line(points, fill=(20, 130, 40), width=3)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(20, 130, 40))
    image.save(path)


def make_group_splits(
    samples: list[dict[str, Any]],
    seed: int,
    strategy: str = "episode",
) -> tuple[dict[str, list[str]], str]:
    def episode_key(sample: dict[str, Any]) -> str:
        return str(sample["episode_id"])

    def map_key(sample: dict[str, Any]) -> str:
        metadata = sample.get("metadata", {})
        return "::".join(
            [
                str(metadata.get("env_name", "unknown_env")),
                str(metadata.get("scenario", "unknown_scenario")),
                str(metadata.get("map_id", "unknown_map")),
            ]
        )

    def source_key(sample: dict[str, Any]) -> str:
        source = sample.get("source", {})
        return str(source.get("source_id") or sample.get("metadata", {}).get("source_id") or sample["episode_id"])

    key_fn = {"episode": episode_key, "map": map_key, "source": source_key}[strategy]
    group_ids = sorted({key_fn(sample) for sample in samples})
    resolved_strategy = strategy
    if len(group_ids) < 3 and strategy != "episode":
        key_fn = episode_key
        group_ids = sorted({key_fn(sample) for sample in samples})
        resolved_strategy = "episode"
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    if len(group_ids) == 1:
        groups = {"train": group_ids, "val": group_ids, "test": group_ids}
    else:
        holdout_slots = 2 if len(group_ids) >= 3 else 1
        n_train = min(max(1, int(round(len(group_ids) * 0.7))), len(group_ids) - holdout_slots)
        n_val = max(1, int(round(len(group_ids) * 0.15))) if len(group_ids) >= 3 else 1
        n_val = min(n_val, len(group_ids) - n_train - 1)
        train_groups = group_ids[:n_train]
        val_groups = group_ids[n_train : n_train + n_val]
        test_groups = group_ids[n_train + n_val :]
        if not val_groups:
            val_groups = train_groups[-1:]
        if not test_groups:
            test_groups = val_groups
        groups = {"train": train_groups, "val": val_groups, "test": test_groups}

    splits = {name: [] for name in groups}
    for sample in samples:
        for split_name, split_episodes in groups.items():
            if key_fn(sample) in split_episodes:
                splits[split_name].append(sample["sample_id"])
    return splits, resolved_strategy


def source_descriptor(raw_dir: Path, manifest: dict[str, Any], index: int) -> dict[str, str]:
    run_id = str(manifest.get("run_id") or raw_dir.name)
    source_id = safe_id(run_id)
    env_name = str(manifest.get("env_name") or ("vizdoom" if manifest.get("scenario") else "unknown"))
    return {
        "source_id": source_id,
        "env_name": env_name,
        "source_dataset": str(manifest.get("source_dataset") or "wit_vz"),
        "raw_run_id": run_id,
        "scenario": str(manifest.get("scenario") or "unknown"),
        "map_id": str(manifest.get("map") or "unknown"),
        "index": str(index),
    }


def build_samples(
    raw_dirs: list[Path],
    out_dir: Path,
    history_sec: float,
    future_sec: float,
    sample_fps: float,
    stride: int,
    seed: int,
    split_strategy: str,
    preview_count: int,
) -> list[dict[str, Any]]:
    if not raw_dirs:
        raise ValueError("At least one raw dataset directory is required")
    history_frames = max(1, round(history_sec * sample_fps))
    future_steps = max(1, round(future_sec * sample_fps))
    samples: list[dict[str, Any]] = []
    raw_dir_entries: dict[str, str] = {}
    source_summaries: dict[str, dict[str, Any]] = {}
    used_source_ids: set[str] = set()

    for raw_index, raw_dir in enumerate(raw_dirs, start=1):
        manifest = load_json(raw_dir / "manifest.json")
        source = source_descriptor(raw_dir, manifest, raw_index)
        base_source_id = source["source_id"]
        if base_source_id in used_source_ids:
            source["source_id"] = f"{base_source_id}_{raw_index:02d}"
        used_source_ids.add(source["source_id"])
        source_id = source["source_id"]
        raw_dir_entries[source_id] = raw_dir.as_posix()
        step_gap = sample_every_raw_step(manifest, sample_fps)
        source_sample_count = 0

        for episode in manifest["episodes"]:
            raw_episode_id = episode["episode_id"]
            episode_id = f"{source_id}__{raw_episode_id}"
            records = read_jsonl(raw_dir / episode["steps_path"])
            if len(records) < history_frames * step_gap + future_steps * step_gap + 1:
                continue

            first_center = (history_frames - 1) * step_gap
            last_center = len(records) - future_steps * step_gap - 1
            for center_idx in range(first_center, last_center + 1, max(1, stride)):
                history_indices = [
                    center_idx - (history_frames - 1 - i) * step_gap for i in range(history_frames)
                ]
                future_indices = [center_idx + (i + 1) * step_gap for i in range(future_steps)]
                history_records = [records[i] for i in history_indices]
                future_records = [records[i] for i in future_indices]
                center_record = records[center_idx]
                future_world = [
                    {"x": item["pose"]["x"], "y": item["pose"]["y"]} for item in future_records
                ]
                future_local = world_future_path_to_local(center_record["pose"], future_world)
                sample_id = f"{source_id}__{raw_episode_id}_t{center_record['step']:06d}"
                sample_source = {
                    "source_id": source_id,
                    "env_name": source["env_name"],
                    "source_dataset": source["source_dataset"],
                    "raw_run_id": source["raw_run_id"],
                }
                sample = {
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "center_step": center_record["step"],
                    "rgb_history_paths": [
                        f"{source_id}::{item['frame_path']}" for item in history_records
                    ],
                    "relative_egomotion_history": egomotion_history_from_records(history_records),
                    "future_local_path": future_local,
                    "future_world_path": future_world,
                    "current_pose": center_record["pose"],
                    "depth_history_paths": [
                        f"{source_id}::{item.get('depth_path')}" if item.get("depth_path") else None
                        for item in history_records
                    ],
                    "labels_history_paths": [
                        f"{source_id}::{item.get('labels_path')}" if item.get("labels_path") else None
                        for item in history_records
                    ],
                    "source": sample_source,
                    "metadata": {
                        "env_name": source["env_name"],
                        "source_dataset": source["source_dataset"],
                        "source_id": source_id,
                        "map_id": source["map_id"],
                        "scenario": source["scenario"],
                        "raw_run_id": source["raw_run_id"],
                        "raw_episode_id": raw_episode_id,
                        "sample_fps": sample_fps,
                        "step_gap": step_gap,
                        "history_frames": history_frames,
                        "future_steps": future_steps,
                        "raw_episode_path": episode["steps_path"],
                    },
                }
                samples.append(sample)
                source_sample_count += 1

        source_summaries[source_id] = {
            "env_name": source["env_name"],
            "source_dataset": source["source_dataset"],
            "raw_run_id": source["raw_run_id"],
            "scenario": source["scenario"],
            "map_id": source["map_id"],
            "raw_dir": raw_dir.as_posix(),
            "step_gap": step_gap,
            "num_samples": source_sample_count,
            "num_episodes": len(manifest.get("episodes", [])),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "samples.jsonl", samples)
    splits, resolved_split_strategy = make_group_splits(samples, seed, split_strategy)
    write_json(out_dir / "splits.json", splits)
    step_gaps = sorted({summary["step_gap"] for summary in source_summaries.values()})
    dataset_manifest = {
        "dataset_id": out_dir.name,
        "raw_dir": next(iter(raw_dir_entries.values())),
        "raw_dirs": raw_dir_entries,
        "sources": source_summaries,
        "num_samples": len(samples),
        "history_sec": history_sec,
        "future_sec": future_sec,
        "sample_fps": sample_fps,
        "stride": stride,
        "step_gap": step_gaps[0] if len(step_gaps) == 1 else None,
        "step_gap_by_source": {
            source_id: summary["step_gap"] for source_id, summary in source_summaries.items()
        },
        "history_frames": history_frames,
        "future_steps": future_steps,
        "split_strategy": f"{resolved_split_strategy}-disjoint",
        "splits": {key: len(value) for key, value in splits.items()},
        "target": "future_local_path",
        "coordinate_convention": "local x=forward, local y=right, origin=current pose",
    }
    write_json(out_dir / "dataset_manifest.json", dataset_manifest)

    for i, sample in enumerate(samples[:preview_count], start=1):
        draw_preview(
            out_dir / "preview" / f"sample_{i:06d}_path.png",
            sample["future_local_path"],
            sample["sample_id"],
        )
    return samples


def main() -> None:
    args = parse_args()
    samples = build_samples(
        raw_dirs=args.raw,
        out_dir=args.out,
        history_sec=args.history_sec,
        future_sec=args.future_sec,
        sample_fps=args.sample_fps,
        stride=args.stride,
        seed=args.seed,
        split_strategy=args.split,
        preview_count=args.preview_count,
    )
    print(f"Wrote {len(samples)} processed samples to: {args.out}")


if __name__ == "__main__":
    main()
