"""Evaluate image -> STP detector -> cue-memory MSTP selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.xu_mstp import CueMemoryMSTPSelector
from src.train_path_predictor import load_flat_config
from src.xu_mstp.dataset import build_image_index, load_rgb_tensor
from src.xu_mstp.stp_detection import (
    box_iou,
    create_fasterrcnn_stp_detector,
    detection_boxes_from_record,
    load_detection_records,
    mstp_box_from_record,
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _remaining = config_parser.parse_known_args()
    defaults = load_flat_config(config_args.config)

    def default(key: str, fallback: Any) -> Any:
        return defaults.get(key, fallback)

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--annotations", type=Path, default=default("annotations", None), required="annotations" not in defaults)
    parser.add_argument("--image-root", type=Path, default=default("image_root", None), required="image_root" not in defaults)
    parser.add_argument("--detector-checkpoint", type=Path, default=default("detector_checkpoint", None), required="detector_checkpoint" not in defaults)
    parser.add_argument("--selector-checkpoint", type=Path, default=default("selector_checkpoint", None), required="selector_checkpoint" not in defaults)
    parser.add_argument("--selector-metrics", type=Path, default=default("selector_metrics", None))
    parser.add_argument("--output-json", type=Path, default=default("output_json", Path("outputs/xu_full_pipeline/results.json")))
    parser.add_argument("--output-md", type=Path, default=default("output_md", Path("reports/xu_full_pipeline.md")))
    parser.add_argument("--score-threshold", type=float, default=default("score_threshold", 0.3))
    parser.add_argument("--iou-threshold", type=float, default=default("iou_threshold", 0.5))
    parser.add_argument("--max-detections", type=int, default=default("max_detections", 10))
    parser.add_argument("--device", default=default("device", "auto"))
    return parser.parse_args()


def _image_to_detection_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def _normalize_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    scale = boxes.new_tensor([width, height, width, height]).clamp_min(1.0)
    out = (boxes / scale).clamp(0.0, 1.0)
    x1 = torch.minimum(out[:, 0], out[:, 2])
    y1 = torch.minimum(out[:, 1], out[:, 3])
    x2 = torch.maximum(out[:, 0], out[:, 2])
    y2 = torch.maximum(out[:, 1], out[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def load_detector(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model = create_fasterrcnn_stp_detector(
        pretrained=False,
        min_size=int(checkpoint.get("min_size", 640)),
        max_size=int(checkpoint.get("max_size", 960)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_selector(path: Path, device: torch.device) -> tuple[CueMemoryMSTPSelector, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model = CueMemoryMSTPSelector(
        backbone_name=str(checkpoint.get("backbone", "small_cnn")),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        freeze_backbone=bool(checkpoint.get("freeze_backbone", True)),
        num_cue_tokens=int(checkpoint.get("num_cue_tokens", 8)),
        spatial_relation_type=str(checkpoint.get("spatial_relation_type", "topk_graph")),
        spatial_graph_neighbors=int(checkpoint.get("spatial_graph_neighbors", 8)),
        adapter_bottleneck_dim=int(checkpoint.get("adapter_bottleneck_dim", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def selector_batch_for_image(
    image_path: Path,
    boxes_abs: torch.Tensor,
    selector_image_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image_tensor, (width, height) = load_rgb_tensor(image_path, selector_image_size)
    boxes = _normalize_boxes(boxes_abs, width, height)
    return {
        "image": image_tensor[None, ...].to(device),
        "candidate_boxes": boxes[None, ...].to(device),
        "candidate_mask": torch.ones((1, boxes.shape[0]), dtype=torch.bool, device=device),
    }


@torch.no_grad()
def evaluate_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    device = choose_device(args.device)
    detector = load_detector(args.detector_checkpoint, device)
    selector, selector_checkpoint = load_selector(args.selector_checkpoint, device)
    selector_image_size = int(selector_checkpoint.get("image_size", 128))
    image_index = build_image_index(args.image_root)
    records = load_detection_records(args.annotations)

    total = 0
    no_detection = 0
    detector_mstp_recall = 0
    selected_mstp_iou_hits = 0
    top3_mstp_iou_hits = 0
    oracle_mstp_available = 0
    pred_count = 0
    gt_count = 0
    rows = []

    for record in records:
        image_id = str(record["image_id"])
        image_path = image_index[image_id]
        mstp = mstp_box_from_record(record)
        if mstp is None:
            continue
        gt_boxes = torch.tensor(detection_boxes_from_record(record), dtype=torch.float32)
        gt_mstp = torch.tensor([mstp], dtype=torch.float32)
        det_input = _image_to_detection_tensor(image_path).to(device)
        output = detector([det_input])[0]
        scores = output["scores"].detach().cpu()
        keep = torch.nonzero(scores >= args.score_threshold, as_tuple=False).squeeze(1)
        if keep.numel() > args.max_detections:
            keep = keep[: args.max_detections]
        boxes_abs = output["boxes"].detach().cpu()[keep]
        total += 1
        pred_count += boxes_abs.shape[0]
        gt_count += gt_boxes.shape[0]
        oracle_mstp_available += 1

        if boxes_abs.numel() == 0:
            no_detection += 1
            rows.append({"image_id": image_id, "num_detections": 0, "selected_iou": 0.0})
            continue

        mstp_ious = box_iou(boxes_abs, gt_mstp).squeeze(1)
        best_detector_iou = float(mstp_ious.max().item())
        if best_detector_iou >= args.iou_threshold:
            detector_mstp_recall += 1

        batch = selector_batch_for_image(image_path, boxes_abs, selector_image_size, device)
        logits = selector(batch).detach().cpu()[0]
        selected_idx = int(logits.argmax().item())
        selected_iou = float(mstp_ious[selected_idx].item())
        if selected_iou >= args.iou_threshold:
            selected_mstp_iou_hits += 1
        top_k = min(3, logits.shape[0])
        top_indices = logits.topk(k=top_k).indices
        top3_iou = float(mstp_ious[top_indices].max().item())
        if top3_iou >= args.iou_threshold:
            top3_mstp_iou_hits += 1
        rows.append(
            {
                "image_id": image_id,
                "num_detections": int(boxes_abs.shape[0]),
                "best_detector_iou": best_detector_iou,
                "selected_index": selected_idx,
                "selected_iou": selected_iou,
                "top3_iou": top3_iou,
            }
        )

    metrics = {
        "samples": total,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "max_detections": args.max_detections,
        "mean_detections": pred_count / max(total, 1),
        "mean_gt_boxes": gt_count / max(total, 1),
        "no_detection_rate": no_detection / max(total, 1),
        "detector_mstp_recall_at_iou": detector_mstp_recall / max(total, 1),
        "end_to_end_mstp_accuracy_at_iou": selected_mstp_iou_hits / max(total, 1),
        "end_to_end_mstp_top3_at_iou": top3_mstp_iou_hits / max(total, 1),
        "oracle_candidate_mstp_available": oracle_mstp_available / max(total, 1),
    }
    if args.selector_metrics and args.selector_metrics.exists():
        selector_metrics = json.loads(args.selector_metrics.read_text(encoding="utf-8"))
        metrics["oracle_candidate_selector_accuracy"] = selector_metrics.get("test", {}).get("accuracy")
        metrics["oracle_candidate_selector_top3_accuracy"] = selector_metrics.get("test", {}).get("top3_accuracy")
    return {"metrics": metrics, "per_sample": rows}


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(path: Path, result: dict[str, Any], args: argparse.Namespace) -> None:
    m = result["metrics"]
    lines = [
        "# Xu Full STP-to-MSTP Pipeline",
        "",
        "## Scope",
        "",
        "- Dataset: VisualGuidance `test_dataset.json` screenshots.",
        "- Pipeline: image -> Faster R-CNN STP detector -> cue-memory MSTP selector.",
        "- End-to-end success: selected detected box has IoU >= threshold with GT MSTP.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Samples | {fmt(m.get('samples'))} |",
        f"| Mean detections | {fmt(m.get('mean_detections'))} |",
        f"| No detection rate | {fmt(m.get('no_detection_rate'))} |",
        f"| Detector MSTP recall@IoU | {fmt(m.get('detector_mstp_recall_at_iou'))} |",
        f"| End-to-end MSTP accuracy@IoU | {fmt(m.get('end_to_end_mstp_accuracy_at_iou'))} |",
        f"| End-to-end MSTP top-3@IoU | {fmt(m.get('end_to_end_mstp_top3_at_iou'))} |",
        f"| Oracle-candidate selector accuracy | {fmt(m.get('oracle_candidate_selector_accuracy'))} |",
        f"| Oracle-candidate selector top-3 | {fmt(m.get('oracle_candidate_selector_top3_accuracy'))} |",
        "",
        "## Interpretation",
        "",
        "- `Detector MSTP recall@IoU` is the upper bound for the selector once detector boxes replace oracle candidates.",
        "- `End-to-end MSTP accuracy@IoU` is the actual full pipeline result.",
        "- The gap from oracle-candidate selector accuracy measures how much performance is lost by replacing GT candidate boxes with detected boxes.",
        "",
        "## Config",
        "",
        "```text",
        f"detector_checkpoint: {args.detector_checkpoint}",
        f"selector_checkpoint: {args.selector_checkpoint}",
        f"score_threshold: {args.score_threshold}",
        f"iou_threshold: {args.iou_threshold}",
        f"max_detections: {args.max_detections}",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = evaluate_pipeline(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(args.output_md, result, args)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
