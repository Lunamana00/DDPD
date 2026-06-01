import json
from pathlib import Path

from PIL import Image
import torch

from src.xu_mstp.stp_detection import (
    XuSTPDetectionDataset,
    box_iou,
    detection_boxes_from_record,
    greedy_match_count,
    mstp_box_from_record,
)


def test_detection_record_helpers_handle_visualguidance_format():
    record = {
        "image_id": "frame.jpg",
        "MSTP": [10, 20, 30, 40],
        "STP": [[40, 50, 70, 80]],
    }
    assert detection_boxes_from_record(record) == [[10.0, 20.0, 30.0, 40.0], [40.0, 50.0, 70.0, 80.0]]
    assert mstp_box_from_record(record) == [10.0, 20.0, 30.0, 40.0]


def test_detection_dataset_returns_torchvision_targets(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 80), color=(12, 34, 56)).save(image_root / "frame.jpg")
    annotations = tmp_path / "stp.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "image_id": "frame.jpg",
                    "MSTP": [10, 20, 30, 40],
                    "STP": [[40, 50, 70, 80]],
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset = XuSTPDetectionDataset(annotations, image_root)
    image, target = dataset[0]
    assert image.shape == (3, 80, 100)
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].tolist() == [1, 1]
    assert target["image_id_str"] == "frame.jpg"


def test_iou_and_greedy_matching():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [19.0, 19.0, 31.0, 31.0]])
    iou = box_iou(pred, gt)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0))
    assert greedy_match_count(pred, gt, iou_threshold=0.5) == 2
