"""STP detection dataset, model factory, and metrics for VisualGuidance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .dataset import build_image_index


def load_detection_records(path: str | Path) -> list[dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [record for record in records if detection_boxes_from_record(record)]


def detection_boxes_from_record(record: dict[str, Any]) -> list[list[float]]:
    if "candidates" in record:
        return [list(map(float, box)) for box in (record.get("candidates") or [])]
    boxes = []
    if record.get("MSTP"):
        boxes.append(list(map(float, record["MSTP"])))
    boxes.extend(list(map(float, box)) for box in (record.get("STP") or []))
    return boxes


def mstp_box_from_record(record: dict[str, Any]) -> list[float] | None:
    if record.get("MSTP"):
        return list(map(float, record["MSTP"]))
    if "candidates" in record and int(record.get("gt_index", -1)) >= 0:
        candidates = record.get("candidates") or []
        gt_index = int(record["gt_index"])
        if gt_index < len(candidates):
            return list(map(float, candidates[gt_index]))
    return None


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def sanitize_absolute_boxes(boxes: list[list[float]], width: int, height: int) -> torch.Tensor:
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    tensor = torch.tensor(boxes, dtype=torch.float32)
    x1 = torch.minimum(tensor[:, 0], tensor[:, 2]).clamp(0, max(width - 1, 1))
    y1 = torch.minimum(tensor[:, 1], tensor[:, 3]).clamp(0, max(height - 1, 1))
    x2 = torch.maximum(tensor[:, 0], tensor[:, 2]).clamp(1, max(width, 1))
    y2 = torch.maximum(tensor[:, 1], tensor[:, 3]).clamp(1, max(height, 1))
    x2 = torch.maximum(x2, x1 + 1.0)
    y2 = torch.maximum(y2, y1 + 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


class XuSTPDetectionDataset(Dataset):
    """VisualGuidance screenshot dataset for STP object detection."""

    def __init__(
        self,
        annotations_file: str | Path,
        image_root: str | Path,
    ) -> None:
        self.annotations_file = Path(annotations_file)
        self.image_root = Path(image_root)
        self.records = load_detection_records(self.annotations_file)
        if not self.records:
            raise ValueError(f"No detection samples found in {annotations_file}")
        self.image_index = build_image_index(self.image_root)

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, image_id: str) -> Path:
        path = self.image_index.get(image_id)
        if path is None:
            raise FileNotFoundError(f"Could not find image_id={image_id} under {self.image_root}")
        return path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        record = self.records[index]
        image_id = str(record["image_id"])
        image = Image.open(self._image_path(image_id)).convert("RGB")
        width, height = image.size
        boxes = sanitize_absolute_boxes(detection_boxes_from_record(record), width, height)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target: dict[str, torch.Tensor | str] = {
            "boxes": boxes,
            "labels": torch.ones((boxes.shape[0],), dtype=torch.long),
            "image_id": torch.tensor([index], dtype=torch.long),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.long),
            "image_id_str": image_id,
        }
        return _to_tensor(image), target


def stp_detection_collate(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    return tuple(zip(*batch))


def create_fasterrcnn_stp_detector(
    pretrained: bool = True,
    min_size: int = 640,
    max_size: int = 960,
    num_classes: int = 2,
) -> torch.nn.Module:
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except Exception as exc:  # pragma: no cover - depends on optional torchvision install
        raise RuntimeError("STP detector training requires torchvision") from exc

    weights = None
    if pretrained:
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        except Exception:
            weights = None
    try:
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None if weights is None else None,
            min_size=min_size,
            max_size=max_size,
        )
    except Exception:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp_min(1.0e-6)


def greedy_match_count(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_threshold: float) -> int:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    matches = 0
    for pred_idx in range(pred_boxes.shape[0]):
        best_iou, best_gt = ious[pred_idx].max(dim=0)
        best_gt_idx = int(best_gt.item())
        if float(best_iou.item()) >= iou_threshold and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            matches += 1
    return matches
