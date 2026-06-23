import json
from pathlib import Path

from PIL import Image

import torch

from src.wit_vz.dataset import (
    WITVZEpisodicChunkDataset,
    WITVZPathDataset,
    collate_episodic_path_batch,
    collate_path_batch,
    sample_group_key,
)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_processed_dataset_sample_shapes(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "run"
    frame_dir = raw / "episodes" / "episode_000001" / "frames"
    frame_dir.mkdir(parents=True)
    for idx in range(3):
        Image.new("RGB", (32, 24), (idx * 40, 0, 0)).save(frame_dir / f"{idx:06d}.png")

    processed = tmp_path / "processed" / "ds"
    write_json(
        processed / "dataset_manifest.json",
        {
            "raw_dir": raw.as_posix(),
            "history_frames": 3,
            "future_steps": 2,
            "num_samples": 1,
        },
    )
    write_json(processed / "splits.json", {"train": ["s1"], "val": ["s1"], "test": ["s1"]})
    write_jsonl(
        processed / "samples.jsonl",
        [
            {
                "sample_id": "s1",
                "episode_id": "episode_000001",
                "center_step": 2,
                "rgb_history_paths": [
                    "episodes/episode_000001/frames/000000.png",
                    "episodes/episode_000001/frames/000001.png",
                    "episodes/episode_000001/frames/000002.png",
                ],
                "relative_egomotion_history": [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                "future_local_path": [[1, 0], [2, 0]],
                "future_world_path": [{"x": 1, "y": 0}, {"x": 2, "y": 0}],
                "current_pose": {"x": 0, "y": 0, "angle": 0},
                "metadata": {
                    "env_name": "vizdoom",
                    "source_id": "source_a",
                    "scenario": "deadly_corridor",
                    "map_id": "map01",
                    "policy": "corridor",
                },
            }
        ],
    )
    dataset = WITVZPathDataset(processed, split="train", image_size=16)
    item = dataset[0]
    assert item["rgb_history"].shape == (3, 3, 16, 16)
    assert item["ego_history"].shape == (3, 3)
    assert item["future_path"].shape == (2, 2)
    assert item["balance"]["source"] == "source_a"
    assert item["balance"]["scenario"] == "vizdoom::deadly_corridor"
    assert item["balance"]["source_policy"] == "source_a::corridor"
    assert sample_group_key(dataset.samples[0], "policy") == "corridor"

    cache = tmp_path / "feature_cache"
    feature_dir = cache / "features"
    feature_dir.mkdir(parents=True)
    visual_tokens = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    torch.save({"visual_tokens": visual_tokens}, feature_dir / "s1.pt")

    cached_dataset = WITVZPathDataset(
        processed,
        split="train",
        image_size=16,
        load_rgb=False,
        visual_feature_cache_dir=cache,
    )
    cached_item = cached_dataset[0]
    assert "rgb_history" not in cached_item
    assert cached_item["visual_tokens"].shape == (3, 4, 8)

    batch = collate_path_batch([cached_item, cached_item])
    assert batch["visual_tokens"].shape == (2, 3, 4, 8)
    assert batch["balance"][0]["policy"] == "corridor"

    last_frame_dataset = WITVZPathDataset(
        processed,
        split="train",
        image_size=16,
        load_rgb=False,
        visual_feature_cache_dir=cache,
        history_frame_mode="last_frame_only",
    )
    last_frame_item = last_frame_dataset[0]
    assert last_frame_item["rgb_history_paths"] == ["episodes/episode_000001/frames/000002.png"]
    assert last_frame_item["ego_history"].tolist() == [[2.0, 0.0, 0.0]]
    assert torch.equal(last_frame_item["visual_tokens"], visual_tokens[-1:])

    monkeypatch.setattr(torch, "randperm", lambda length: torch.tensor([2, 0, 1]))
    shuffled_dataset = WITVZPathDataset(
        processed,
        split="train",
        image_size=16,
        load_rgb=False,
        visual_feature_cache_dir=cache,
        frame_order="shuffle",
    )
    shuffled_item = shuffled_dataset[0]
    assert shuffled_item["rgb_history_paths"] == [
        "episodes/episode_000001/frames/000002.png",
        "episodes/episode_000001/frames/000000.png",
        "episodes/episode_000001/frames/000001.png",
    ]
    assert shuffled_item["ego_history"][:, 0].tolist() == [2.0, 0.0, 1.0]
    assert torch.equal(shuffled_item["visual_tokens"], visual_tokens[[2, 0, 1]])


def test_processed_dataset_resolves_source_prefixed_raw_paths(tmp_path):
    raw = tmp_path / "raw" / "source_a"
    frame_dir = raw / "episodes" / "episode_000001" / "frames"
    frame_dir.mkdir(parents=True)
    for idx in range(2):
        Image.new("RGB", (32, 24), (idx * 60, 0, 0)).save(frame_dir / f"{idx:06d}.png")

    processed = tmp_path / "processed" / "multi"
    write_json(
        processed / "dataset_manifest.json",
        {
            "raw_dir": raw.as_posix(),
            "raw_dirs": {"source_a": raw.as_posix()},
            "history_frames": 2,
            "future_steps": 1,
            "num_samples": 1,
        },
    )
    write_json(processed / "splits.json", {"train": ["source_a__s1"], "val": [], "test": []})
    write_jsonl(
        processed / "samples.jsonl",
        [
            {
                "sample_id": "source_a__s1",
                "episode_id": "source_a__episode_000001",
                "center_step": 1,
                "source": {"source_id": "source_a", "env_name": "vizdoom"},
                "rgb_history_paths": [
                    "source_a::episodes/episode_000001/frames/000000.png",
                    "source_a::episodes/episode_000001/frames/000001.png",
                ],
                "relative_egomotion_history": [[0, 0, 0], [1, 0, 0]],
                "future_local_path": [[1, 0]],
                "future_world_path": [{"x": 1, "y": 0}],
                "current_pose": {"x": 0, "y": 0, "angle": 0},
                "metadata": {"source_id": "source_a"},
            }
        ],
    )

    dataset = WITVZPathDataset(processed, split="train", image_size=16)
    item = dataset[0]
    assert item["source"]["source_id"] == "source_a"
    assert item["rgb_history"].shape == (2, 3, 16, 16)


def test_episodic_chunk_dataset_uses_episode_order_and_cached_tokens(tmp_path):
    raw = tmp_path / "raw" / "run"
    frame_dir = raw / "episodes" / "episode_000001" / "frames"
    frame_dir.mkdir(parents=True)
    for idx in range(6):
        Image.new("RGB", (32, 24), (idx * 20, 0, 0)).save(frame_dir / f"{idx:06d}.png")

    processed = tmp_path / "processed" / "episodic"
    sample_ids = [f"s{idx}" for idx in range(5)]
    write_json(
        processed / "dataset_manifest.json",
        {
            "raw_dir": raw.as_posix(),
            "history_frames": 2,
            "future_steps": 2,
            "num_samples": 5,
        },
    )
    write_json(processed / "splits.json", {"train": sample_ids, "val": sample_ids, "test": sample_ids})
    rows = []
    for idx, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "episode_id": "episode_000001",
                "center_step": idx + 1,
                "rgb_history_paths": [
                    f"episodes/episode_000001/frames/{idx:06d}.png",
                    f"episodes/episode_000001/frames/{idx + 1:06d}.png",
                ],
                "relative_egomotion_history": [[1, 0, 0], [1, 0, 0]],
                "future_local_path": [[1, 0], [2, 0]],
                "future_world_path": [{"x": 1, "y": 0}, {"x": 2, "y": 0}],
                "current_pose": {"x": idx, "y": 0, "angle": 0},
                "metadata": {
                    "env_name": "vizdoom",
                    "source_id": "source_a",
                    "scenario": "basic",
                    "map_id": "map01",
                    "policy": "scripted",
                },
            }
        )
    write_jsonl(processed / "samples.jsonl", rows)

    cache = tmp_path / "feature_cache"
    feature_dir = cache / "features"
    feature_dir.mkdir(parents=True)
    for idx, sample_id in enumerate(sample_ids):
        visual_tokens = torch.full((2, 4, 8), float(idx))
        torch.save({"visual_tokens": visual_tokens}, feature_dir / f"{sample_id}.pt")

    dataset = WITVZEpisodicChunkDataset(
        processed,
        split="train",
        load_rgb=False,
        visual_feature_cache_dir=cache,
        chunk_length=3,
        chunk_stride=2,
    )
    assert len(dataset) == 2
    item = dataset[0]
    assert item["sample_id"] == ["s0", "s1", "s2"]
    assert item["center_step"].tolist() == [1, 2, 3]
    assert item["visual_tokens"].shape == (3, 2, 4, 8)
    assert item["ego_history"].shape == (3, 2, 3)

    batch = collate_episodic_path_batch([dataset[0], dataset[1]])
    assert batch["visual_tokens"].shape == (2, 3, 2, 4, 8)
    assert batch["future_path"].shape == (2, 3, 2, 2)
    assert batch["balance"][0][0]["source_policy"] == "source_a::scripted"
