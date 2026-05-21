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
        visual_feature_cache_dir: str | Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest = load_json(self.dataset_dir / "dataset_manifest.json")
        self.raw_dirs = self._load_raw_dirs()
        self.raw_dir = next(iter(self.raw_dirs.values()))
        self.image_size = image_size
        self.load_rgb = load_rgb
        self.visual_feature_cache_dir = (
            Path(visual_feature_cache_dir) if visual_feature_cache_dir is not None else None
        )

        all_samples = read_jsonl(self.dataset_dir / "samples.jsonl")
        splits = load_json(self.dataset_dir / "splits.json")
        wanted = set(splits.get(split, []))
        self.samples = [sample for sample in all_samples if not wanted or sample["sample_id"] in wanted]
        if not self.samples:
            raise ValueError(f"No samples found for split={split} in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_raw_root(self, raw_dir: str | Path) -> Path:
        path = Path(raw_dir)
        if path.is_absolute():
            return path
        candidate = (self.dataset_dir / path).resolve()
        if candidate.exists():
            return candidate
        return path.resolve()

    def _load_raw_dirs(self) -> dict[str, Path]:
        if "raw_dirs" in self.manifest:
            return {
                str(source_id): self._resolve_raw_root(raw_dir)
                for source_id, raw_dir in self.manifest["raw_dirs"].items()
            }
        return {"default": self._resolve_raw_root(self.manifest["raw_dir"])}

    def _resolve_raw_path(self, rel_path: str | None, source_id: str | None = None) -> Path | None:
        if rel_path is None:
            return None
        selected_source_id = source_id
        rel = rel_path
        if "::" in rel_path:
            selected_source_id, rel = rel_path.split("::", 1)
        path = Path(rel)
        if path.is_absolute():
            return path
        if selected_source_id is not None and selected_source_id in self.raw_dirs:
            return self.raw_dirs[selected_source_id] / path
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
            "source": sample.get("source", {}),
            "current_pose": sample.get("current_pose"),
            "rgb_history_paths": sample["rgb_history_paths"],
        }
        source_id = (
            item["source"].get("source_id")
            or item["metadata"].get("source_id")
            or None
        )
        if self.load_rgb:
            frames = [
                load_rgb_tensor(self._resolve_raw_path(path, source_id), self.image_size)
                for path in sample["rgb_history_paths"]
            ]
            item["rgb_history"] = torch.stack(frames, dim=0)
        if self.visual_feature_cache_dir is not None:
            feature_path = self.visual_feature_cache_dir / "features" / f"{sample['sample_id']}.pt"
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing cached visual feature file: {feature_path}")
            cached = torch.load(feature_path, map_location="cpu")
            if isinstance(cached, dict):
                tokens = cached["visual_tokens"]
            else:
                tokens = cached
            item["visual_tokens"] = tokens.float()
        return item


def collate_path_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "sample_id": [item["sample_id"] for item in batch],
        "episode_id": [item["episode_id"] for item in batch],
        "center_step": torch.tensor([item["center_step"] for item in batch], dtype=torch.long),
        "ego_history": torch.stack([item["ego_history"] for item in batch], dim=0),
        "future_path": torch.stack([item["future_path"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
        "source": [item["source"] for item in batch],
        "current_pose": [item["current_pose"] for item in batch],
        "rgb_history_paths": [item["rgb_history_paths"] for item in batch],
    }
    if "rgb_history" in batch[0]:
        output["rgb_history"] = torch.stack([item["rgb_history"] for item in batch], dim=0)
    if "visual_tokens" in batch[0]:
        output["visual_tokens"] = torch.stack([item["visual_tokens"] for item in batch], dim=0)
    return output
