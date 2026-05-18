"""PyTorch dataset for WIT-VZ processed path prediction samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .io import load_json, read_jsonl


def load_rgb_tensor(path: Path, image_size: int | tuple[int, int]) -> torch.Tensor:
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = (int(image_size[0]), int(image_size[1]))
    image = Image.open(path).convert("RGB").resize(size)
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


class WITVZPathDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        image_size: int | tuple[int, int] = 64,
        load_rgb: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest = load_json(self.dataset_dir / "dataset_manifest.json")
        self.raw_dir = Path(self.manifest["raw_dir"])
        if not self.raw_dir.is_absolute():
            self.raw_dir = (self.dataset_dir / self.raw_dir).resolve()
            if not self.raw_dir.exists():
                self.raw_dir = Path(self.manifest["raw_dir"]).resolve()
        self.image_size = image_size
        self.load_rgb = load_rgb

        all_samples = read_jsonl(self.dataset_dir / "samples.jsonl")
        splits = load_json(self.dataset_dir / "splits.json")
        wanted = set(splits.get(split, []))
        self.samples = [sample for sample in all_samples if not wanted or sample["sample_id"] in wanted]
        if not self.samples:
            raise ValueError(f"No samples found for split={split} in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_raw_path(self, rel_path: str | None) -> Path | None:
        if rel_path is None:
            return None
        path = Path(rel_path)
        if path.is_absolute():
            return path
        return self.raw_dir / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        item: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "episode_id": sample["episode_id"],
            "center_step": sample["center_step"],
            "ego_history": torch.tensor(sample["relative_egomotion_history"], dtype=torch.float32),
            "future_path": torch.tensor(sample["future_local_path"], dtype=torch.float32),
            "metadata": sample.get("metadata", {}),
            "current_pose": sample.get("current_pose"),
            "rgb_history_paths": sample["rgb_history_paths"],
        }
        if self.load_rgb:
            frames = [
                load_rgb_tensor(self._resolve_raw_path(path), self.image_size)
                for path in sample["rgb_history_paths"]
            ]
            item["rgb_history"] = torch.stack(frames, dim=0)
        return item


def collate_path_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "sample_id": [item["sample_id"] for item in batch],
        "episode_id": [item["episode_id"] for item in batch],
        "center_step": torch.tensor([item["center_step"] for item in batch], dtype=torch.long),
        "ego_history": torch.stack([item["ego_history"] for item in batch], dim=0),
        "future_path": torch.stack([item["future_path"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
        "current_pose": [item["current_pose"] for item in batch],
        "rgb_history_paths": [item["rgb_history_paths"] for item in batch],
    }
    if "rgb_history" in batch[0]:
        output["rgb_history"] = torch.stack([item["rgb_history"] for item in batch], dim=0)
    return output
