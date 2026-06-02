"""Dataset utilities for Xu/VisualGuidance STP-MSTP selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_image_index(root: str | Path) -> dict[str, Path]:
    root_path = Path(root)
    if root_path.is_file():
        return {root_path.name: root_path}
    index: dict[str, Path] = {}
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name, path)
    return index


def annotation_to_selector_record(record: dict[str, Any]) -> dict[str, Any]:
    if "candidates" in record and "gt_index" in record:
        return record
    candidates = []
    gt_index = -1
    if record.get("MSTP"):
        candidates.append(record["MSTP"])
        gt_index = 0
    candidates.extend(record.get("STP") or [])
    return {
        "image_id": record["image_id"],
        "candidates": candidates,
        "gt_index": gt_index,
    }


def load_rgb_tensor(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return tensor, original_size


class XuMSTPSelectionDataset(Dataset):
    """VisualGuidance candidate-STP to MSTP-index dataset.

    The dataset returns one screenshot, a variable number of candidate STP
    boxes, and the ground-truth candidate index for the MSTP. Candidate boxes
    are normalized to `[0, 1]` in original image coordinates.
    """

    def __init__(
        self,
        annotations_file: str | Path,
        image_root: str | Path,
        image_size: int = 128,
    ) -> None:
        self.annotations_file = Path(annotations_file)
        self.image_root = Path(image_root)
        self.image_size = image_size
        records = [annotation_to_selector_record(item) for item in _load_records(self.annotations_file)]
        self.records = [
            item
            for item in records
            if item.get("candidates") and int(item.get("gt_index", -1)) >= 0
        ]
        if not self.records:
            raise ValueError(f"No valid MSTP selector samples found in {annotations_file}")
        self.image_index = build_image_index(self.image_root)

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, image_id: str) -> Path:
        path = self.image_index.get(image_id)
        if path is None:
            raise FileNotFoundError(f"Could not find image_id={image_id} under {self.image_root}")
        return path

    @staticmethod
    def _normalize_boxes(boxes: list[list[float]], width: int, height: int) -> torch.Tensor:
        tensor = torch.tensor(boxes, dtype=torch.float32)
        scale = tensor.new_tensor([width, height, width, height]).clamp_min(1.0)
        tensor = (tensor / scale).clamp(0.0, 1.0)
        x1 = torch.minimum(tensor[:, 0], tensor[:, 2])
        y1 = torch.minimum(tensor[:, 1], tensor[:, 3])
        x2 = torch.maximum(tensor[:, 0], tensor[:, 2])
        y2 = torch.maximum(tensor[:, 1], tensor[:, 3])
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_id = str(record["image_id"])
        image, (width, height) = load_rgb_tensor(self._image_path(image_id), self.image_size)
        candidates = self._normalize_boxes(record["candidates"], width, height)
        gt_index = int(record["gt_index"])
        if gt_index >= candidates.shape[0]:
            raise ValueError(f"gt_index={gt_index} out of range for {image_id}")
        return {
            "image_id": image_id,
            "image": image,
            "candidate_boxes": candidates,
            "candidate_mask": torch.ones(candidates.shape[0], dtype=torch.bool),
            "gt_index": torch.tensor(gt_index, dtype=torch.long),
        }


def collate_xu_mstp_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_candidates = max(item["candidate_boxes"].shape[0] for item in batch)
    candidate_boxes = []
    candidate_mask = []
    for item in batch:
        boxes = item["candidate_boxes"]
        mask = item["candidate_mask"]
        pad_count = max_candidates - boxes.shape[0]
        if pad_count > 0:
            boxes = torch.cat([boxes, boxes.new_zeros((pad_count, 4))], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_count, dtype=torch.bool)], dim=0)
        candidate_boxes.append(boxes)
        candidate_mask.append(mask)
    return {
        "image_id": [item["image_id"] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "candidate_boxes": torch.stack(candidate_boxes, dim=0),
        "candidate_mask": torch.stack(candidate_mask, dim=0),
        "gt_index": torch.stack([item["gt_index"] for item in batch], dim=0),
    }


class XuRouteTargetDataset(Dataset):
    """VisualGuidance screenshot dataset for direct MSTP box prediction."""

    def __init__(
        self,
        annotations_file: str | Path,
        image_root: str | Path,
        image_size: int = 128,
    ) -> None:
        self.annotations_file = Path(annotations_file)
        self.image_root = Path(image_root)
        self.image_size = image_size
        records = _load_records(self.annotations_file)
        self.records = [
            item
            for item in records
            if item.get("MSTP") or ("candidates" in item and int(item.get("gt_index", -1)) >= 0)
        ]
        if not self.records:
            raise ValueError(f"No valid route target samples found in {annotations_file}")
        self.image_index = build_image_index(self.image_root)

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, image_id: str) -> Path:
        path = self.image_index.get(image_id)
        if path is None:
            raise FileNotFoundError(f"Could not find image_id={image_id} under {self.image_root}")
        return path

    @staticmethod
    def _target_box(record: dict[str, Any]) -> list[float]:
        if record.get("MSTP"):
            return record["MSTP"]
        candidates = record.get("candidates") or []
        gt_index = int(record.get("gt_index", -1))
        if gt_index < 0 or gt_index >= len(candidates):
            raise ValueError(f"Invalid route target record: {record}")
        return candidates[gt_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_id = str(record["image_id"])
        image, (width, height) = load_rgb_tensor(self._image_path(image_id), self.image_size)
        target_box = XuMSTPSelectionDataset._normalize_boxes([self._target_box(record)], width, height)[0]
        return {
            "image_id": image_id,
            "image": image,
            "target_box": target_box,
        }


def collate_xu_route_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_id": [item["image_id"] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "target_box": torch.stack([item["target_box"] for item in batch], dim=0),
    }
