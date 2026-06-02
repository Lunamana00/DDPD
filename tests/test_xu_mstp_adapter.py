import json
from pathlib import Path

from PIL import Image
import torch

from src.models.xu_mstp import (
    CueMemoryDirectRoutePredictor,
    CueMemoryMSTPSelector,
    VisualGuidanceMSTPSelectorBaseline,
)
from src.xu_mstp.dataset import (
    XuMSTPSelectionDataset,
    XuRouteTargetDataset,
    collate_xu_mstp_batch,
    collate_xu_route_batch,
)


def make_tiny_visualguidance_dataset(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for image_id, color in [("a.jpg", (255, 0, 0)), ("b.jpg", (0, 255, 0))]:
        Image.new("RGB", (64, 48), color=color).save(image_root / image_id)
    records = [
        {
            "image_id": "a.jpg",
            "candidates": [[0, 0, 32, 24], [32, 24, 64, 48]],
            "gt_index": 1,
        },
        {
            "image_id": "b.jpg",
            "candidates": [[8, 8, 48, 40]],
            "gt_index": 0,
        },
    ]
    annotations = tmp_path / "model2.json"
    annotations.write_text(json.dumps(records), encoding="utf-8")
    return annotations, image_root


def test_xu_mstp_dataset_collates_variable_candidate_counts(tmp_path):
    annotations, image_root = make_tiny_visualguidance_dataset(tmp_path)
    dataset = XuMSTPSelectionDataset(annotations, image_root, image_size=32)
    batch = collate_xu_mstp_batch([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["candidate_boxes"].shape == (2, 2, 4)
    assert batch["candidate_mask"].tolist() == [[True, True], [True, False]]
    assert batch["gt_index"].tolist() == [1, 0]


def test_cue_memory_mstp_selector_outputs_candidate_scores(tmp_path):
    annotations, image_root = make_tiny_visualguidance_dataset(tmp_path)
    dataset = XuMSTPSelectionDataset(annotations, image_root, image_size=32)
    batch = collate_xu_mstp_batch([dataset[0], dataset[1]])
    model = CueMemoryMSTPSelector(
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        spatial_relation_type="topk_graph",
        spatial_graph_neighbors=2,
        dropout=0.0,
        adapter_bottleneck_dim=16,
    )
    scores = model(batch)
    assert scores.shape == (2, 2)
    assert torch.isfinite(scores[0]).all()
    assert torch.isfinite(scores[1, 0])
    assert scores[1, 1] < -1.0e20


def test_xu_route_target_dataset_and_direct_predictor(tmp_path):
    annotations, image_root = make_tiny_visualguidance_dataset(tmp_path)
    dataset = XuRouteTargetDataset(annotations, image_root, image_size=32)
    batch = collate_xu_route_batch([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["target_box"].shape == (2, 4)

    model = CueMemoryDirectRoutePredictor(
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        spatial_relation_type="topk_graph",
        spatial_graph_neighbors=2,
        dropout=0.0,
        adapter_bottleneck_dim=16,
    )
    boxes = model(batch)
    assert boxes.shape == (2, 4)
    assert torch.isfinite(boxes).all()
    assert torch.all((boxes >= 0.0) & (boxes <= 1.0))
    assert torch.all(boxes[:, 0] <= boxes[:, 2])
    assert torch.all(boxes[:, 1] <= boxes[:, 3])


def test_visualguidance_mstp_selector_baseline_outputs_scores(tmp_path):
    annotations, image_root = make_tiny_visualguidance_dataset(tmp_path)
    dataset = XuMSTPSelectionDataset(annotations, image_root, image_size=64)
    batch = collate_xu_mstp_batch([dataset[0], dataset[1]])
    model = VisualGuidanceMSTPSelectorBaseline(
        hidden_dim=16,
        bottleneck_dim=16,
        crop_size=32,
        global_size=32,
        pretrained=False,
    )
    scores = model(batch)
    assert scores.shape == (2, 2)
    assert torch.isfinite(scores[0]).all()
    assert torch.isfinite(scores[1, 0])
    assert scores[1, 1] < -1.0e20
