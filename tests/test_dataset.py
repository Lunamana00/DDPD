import json
from pathlib import Path

from PIL import Image

from src.wit_vz.dataset import WITVZPathDataset


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_processed_dataset_sample_shapes(tmp_path):
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
                "relative_egomotion_history": [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
                "future_local_path": [[1, 0], [2, 0]],
                "future_world_path": [{"x": 1, "y": 0}, {"x": 2, "y": 0}],
                "current_pose": {"x": 0, "y": 0, "angle": 0},
                "metadata": {},
            }
        ],
    )
    dataset = WITVZPathDataset(processed, split="train", image_size=16)
    item = dataset[0]
    assert item["rgb_history"].shape == (3, 3, 16, 16)
    assert item["ego_history"].shape == (3, 3)
    assert item["future_path"].shape == (2, 2)
