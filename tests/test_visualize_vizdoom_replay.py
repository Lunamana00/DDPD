import json
from pathlib import Path

from PIL import Image

from src.visualize_vizdoom_replay import (
    render_frame,
    save_replay,
    select_replay_items,
)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_processed_dataset(tmp_path: Path) -> tuple[Path, list[dict], list[dict]]:
    raw = tmp_path / "raw" / "run"
    frame_dir = raw / "episodes" / "episode_000001" / "frames"
    frame_dir.mkdir(parents=True)
    for idx in range(4):
        Image.new("RGB", (80, 45), (30 + idx * 30, 20, 80)).save(frame_dir / f"{idx:06d}.png")

    processed = tmp_path / "processed" / "ds"
    write_json(
        processed / "dataset_manifest.json",
        {
            "raw_dir": raw.as_posix(),
            "history_frames": 2,
            "future_steps": 2,
            "future_sec": 1.0,
            "num_samples": 2,
        },
    )
    write_json(processed / "splits.json", {"test": ["s1", "s2"]})
    samples = [
        {
            "sample_id": "s1",
            "episode_id": "episode_000001",
            "center_step": 1,
            "rgb_history_paths": [
                "episodes/episode_000001/frames/000000.png",
                "episodes/episode_000001/frames/000001.png",
            ],
            "relative_egomotion_history": [[0, 0, 0], [1, 0, 0]],
            "future_local_path": [[1, 0], [2, 0]],
            "current_pose": {"x": 0, "y": 0, "angle": 0},
            "metadata": {},
        },
        {
            "sample_id": "s2",
            "episode_id": "episode_000001",
            "center_step": 2,
            "rgb_history_paths": [
                "episodes/episode_000001/frames/000001.png",
                "episodes/episode_000001/frames/000002.png",
            ],
            "relative_egomotion_history": [[0, 0, 0], [1, 0, 0]],
            "future_local_path": [[1, 0.25], [2, 0.5]],
            "current_pose": {"x": 1, "y": 0, "angle": 0},
            "metadata": {},
        },
    ]
    predictions = [
        {"sample_id": "s2", "prediction": [[1, 0.2], [2, 0.4]]},
        {"sample_id": "s1", "prediction": [[1, 0.1], [2, 0.2]]},
    ]
    write_jsonl(processed / "samples.jsonl", samples)
    return processed, samples, predictions


def test_select_replay_items_sorts_by_center_step(tmp_path):
    _, samples, predictions = make_processed_dataset(tmp_path)
    by_id = {sample["sample_id"]: sample for sample in samples}

    items = select_replay_items(by_id, predictions, episode_id=None, num_frames=10)

    assert [sample["sample_id"] for sample, _ in items] == ["s1", "s2"]


def test_render_and_save_replay_gif(tmp_path):
    processed, samples, predictions = make_processed_dataset(tmp_path)
    frame = render_frame(processed, samples[0], predictions[1], frame_width=160, panel_size=140)
    assert frame.width > 160
    assert frame.height > 140

    gif_path = tmp_path / "out" / "replay.gif"
    save_replay([frame, frame.copy()], gif_path, fps=4.0, save_frames=True)

    assert gif_path.exists()
    assert (tmp_path / "out" / "replay_frames" / "frame_0001.png").exists()
