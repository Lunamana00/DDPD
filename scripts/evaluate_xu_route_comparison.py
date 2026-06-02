"""Compare direct route-target prediction with Xu/VisualGuidance MSTP baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.xu_mstp import CueMemoryDirectRoutePredictor, VisualGuidanceMSTPSelectorBaseline
from src.train_path_predictor import load_flat_config
from src.xu_mstp.dataset import (
    XuMSTPSelectionDataset,
    XuRouteTargetDataset,
    build_image_index,
    collate_xu_mstp_batch,
    collate_xu_route_batch,
    load_rgb_tensor,
)
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
    parser.add_argument("--selector-annotations", type=Path, default=default("selector_annotations", None), required="selector_annotations" not in defaults)
    parser.add_argument("--image-root", type=Path, default=default("image_root", None), required="image_root" not in defaults)
    parser.add_argument("--direct-checkpoint", type=Path, default=default("direct_checkpoint", None), required="direct_checkpoint" not in defaults)
    parser.add_argument("--selector-checkpoint", type=Path, default=default("selector_checkpoint", None), required="selector_checkpoint" not in defaults)
    parser.add_argument("--detector-checkpoint", type=Path, default=default("detector_checkpoint", None))
    parser.add_argument("--direct-metrics", type=Path, default=default("direct_metrics", None))
    parser.add_argument("--selector-metrics", type=Path, default=default("selector_metrics", None))
    parser.add_argument("--output-json", type=Path, default=default("output_json", Path("outputs/xu_route_comparison/results.json")))
    parser.add_argument("--output-md", type=Path, default=default("output_md", Path("reports/xu_route_comparison.md")))
    parser.add_argument("--score-threshold", type=float, default=default("score_threshold", 0.3))
    parser.add_argument("--iou-threshold", type=float, default=default("iou_threshold", 0.5))
    parser.add_argument("--center-threshold", type=float, default=default("center_threshold", 0.1))
    parser.add_argument("--max-detections", type=int, default=default("max_detections", 3))
    parser.add_argument("--batch-size", type=int, default=default("batch_size", 32))
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


def center_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_center = (pred[:, :2] + pred[:, 2:]) * 0.5
    target_center = (target[:, :2] + target[:, 2:]) * 0.5
    return torch.linalg.norm(pred_center - target_center, dim=-1)


def load_direct_predictor(path: Path, device: torch.device) -> tuple[CueMemoryDirectRoutePredictor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model = CueMemoryDirectRoutePredictor(
        backbone_name=str(checkpoint.get("backbone", "small_cnn")),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        freeze_backbone=bool(checkpoint.get("freeze_backbone", False)),
        num_cue_tokens=int(checkpoint.get("num_cue_tokens", 8)),
        spatial_relation_type=str(checkpoint.get("spatial_relation_type", "topk_graph")),
        spatial_graph_neighbors=int(checkpoint.get("spatial_graph_neighbors", 8)),
        adapter_bottleneck_dim=int(checkpoint.get("adapter_bottleneck_dim", 64)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_visualguidance_selector(
    path: Path,
    device: torch.device,
) -> tuple[VisualGuidanceMSTPSelectorBaseline, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model = VisualGuidanceMSTPSelectorBaseline(
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        bottleneck_dim=int(checkpoint.get("bottleneck_dim", 256)),
        crop_size=int(checkpoint.get("crop_size", 224)),
        global_size=int(checkpoint.get("global_size", 64)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


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


def _selector_batch_for_image(
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
def evaluate_direct(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    image_size: int,
) -> dict[str, Any]:
    dataset = XuRouteTargetDataset(args.annotations, args.image_root, image_size=image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_xu_route_batch)
    total = 0
    iou_sum = 0.0
    center_sum = 0.0
    hit_iou = 0
    hit_center = 0
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        target = batch["target_box"].to(device)
        pred = model({"image": images}).detach().cpu()
        target_cpu = target.detach().cpu()
        ious = box_iou(pred, target_cpu).diag()
        distances = center_distance(pred, target_cpu)
        total += target_cpu.shape[0]
        iou_sum += float(ious.sum().item())
        center_sum += float(distances.sum().item())
        hit_iou += int((ious >= args.iou_threshold).sum().item())
        hit_center += int((distances <= args.center_threshold).sum().item())
        for image_id, iou, distance in zip(batch["image_id"], ious.tolist(), distances.tolist()):
            rows.append({"image_id": image_id, "iou": iou, "center_error": distance})
    return {
        "samples": total,
        "mean_iou": iou_sum / max(total, 1),
        "center_error": center_sum / max(total, 1),
        "hit_iou": hit_iou / max(total, 1),
        "hit_center": hit_center / max(total, 1),
        "per_sample": rows,
    }


@torch.no_grad()
def evaluate_oracle_selector(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    image_size: int,
) -> dict[str, Any]:
    dataset = XuMSTPSelectionDataset(args.selector_annotations, args.image_root, image_size=image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_xu_mstp_batch)
    total = 0
    correct = 0
    top3 = 0
    iou_hits = 0
    top3_iou_hits = 0
    candidate_sum = 0
    for batch in loader:
        moved = {
            "image": batch["image"].to(device),
            "candidate_boxes": batch["candidate_boxes"].to(device),
            "candidate_mask": batch["candidate_mask"].to(device),
        }
        logits = model(moved).detach().cpu()
        target = batch["gt_index"]
        boxes = batch["candidate_boxes"]
        pred = logits.argmax(dim=1)
        k = min(3, logits.shape[1])
        top = logits.topk(k=k, dim=1).indices
        for idx in range(target.shape[0]):
            target_box = boxes[idx, target[idx]][None, :]
            selected_iou = box_iou(boxes[idx, pred[idx]][None, :], target_box)[0, 0]
            top_iou = box_iou(boxes[idx, top[idx]], target_box).max()
            iou_hits += int(float(selected_iou.item()) >= args.iou_threshold)
            top3_iou_hits += int(float(top_iou.item()) >= args.iou_threshold)
        total += target.shape[0]
        correct += int((pred == target).sum().item())
        top3 += int((top == target[:, None]).any(dim=1).sum().item())
        candidate_sum += int(batch["candidate_mask"].sum().item())
    return {
        "samples": total,
        "mean_candidates": candidate_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3 / max(total, 1),
        "selected_iou_hit": iou_hits / max(total, 1),
        "top3_iou_hit": top3_iou_hits / max(total, 1),
    }


@torch.no_grad()
def evaluate_detected_selector(
    detector: torch.nn.Module,
    selector: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    selector_image_size: int,
) -> dict[str, Any]:
    image_index = build_image_index(args.image_root)
    records = load_detection_records(args.annotations)
    total = 0
    no_detection = 0
    pred_count = 0
    gt_count = 0
    detector_mstp_recall = 0
    selected_hits = 0
    top3_hits = 0
    rows = []
    for record in records:
        image_id = str(record["image_id"])
        image_path = image_index[image_id]
        mstp = mstp_box_from_record(record)
        if mstp is None:
            continue
        gt_boxes = torch.tensor(detection_boxes_from_record(record), dtype=torch.float32)
        gt_mstp = torch.tensor([mstp], dtype=torch.float32)
        output = detector([_image_to_detection_tensor(image_path).to(device)])[0]
        scores = output["scores"].detach().cpu()
        keep = torch.nonzero(scores >= args.score_threshold, as_tuple=False).squeeze(1)
        if keep.numel() > args.max_detections:
            keep = keep[: args.max_detections]
        boxes_abs = output["boxes"].detach().cpu()[keep]
        total += 1
        pred_count += boxes_abs.shape[0]
        gt_count += gt_boxes.shape[0]
        if boxes_abs.numel() == 0:
            no_detection += 1
            rows.append({"image_id": image_id, "num_detections": 0, "selected_iou": 0.0})
            continue
        mstp_ious = box_iou(boxes_abs, gt_mstp).squeeze(1)
        best_detector_iou = float(mstp_ious.max().item())
        detector_mstp_recall += int(best_detector_iou >= args.iou_threshold)
        batch = _selector_batch_for_image(image_path, boxes_abs, selector_image_size, device)
        logits = selector(batch).detach().cpu()[0]
        selected_idx = int(logits.argmax().item())
        selected_iou = float(mstp_ious[selected_idx].item())
        selected_hits += int(selected_iou >= args.iou_threshold)
        top_k = min(3, logits.shape[0])
        top_iou = float(mstp_ious[logits.topk(k=top_k).indices].max().item())
        top3_hits += int(top_iou >= args.iou_threshold)
        rows.append(
            {
                "image_id": image_id,
                "num_detections": int(boxes_abs.shape[0]),
                "best_detector_iou": best_detector_iou,
                "selected_iou": selected_iou,
                "top3_iou": top_iou,
            }
        )
    return {
        "samples": total,
        "mean_detections": pred_count / max(total, 1),
        "mean_gt_boxes": gt_count / max(total, 1),
        "no_detection_rate": no_detection / max(total, 1),
        "detector_mstp_recall": detector_mstp_recall / max(total, 1),
        "end_to_end_hit_iou": selected_hits / max(total, 1),
        "end_to_end_top3_hit_iou": top3_hits / max(total, 1),
        "per_sample": rows,
    }


def _read_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(path: Path, result: dict[str, Any], args: argparse.Namespace) -> None:
    direct = result["direct_route_target"]
    oracle = result["visualguidance_oracle_candidates"]
    detected = result["visualguidance_detected_candidates"]
    direct_meta = result.get("direct_model", {})
    selector_meta = result.get("selector_model", {})
    lines = [
        "# Route Target Comparison on VisualGuidance",
        "",
        "## Corrected Comparison",
        "",
        "This report compares final screen-space route-target quality, not a selector-stage substitution.",
        "",
        "- Ours: image -> cue-memory direct route-target predictor -> MSTP-like box.",
        "- Xu/VisualGuidance oracle: image + GT STP candidates -> MSTP selector -> selected candidate.",
        "- Xu/VisualGuidance detected: image -> Faster R-CNN STP detector -> MSTP selector -> selected detected candidate.",
        "",
        "VisualGuidance does not provide ego-motion or future trajectory labels, so this is not an ADE/FDE trajectory benchmark. The common target is the GT MSTP bounding box.",
        "",
        "The direct route model is a screen-space adaptation of the cue-memory architecture. It uses the configured visual encoder from its checkpoint and does not use explicit STP candidate boxes.",
        "",
        "## Test Metrics",
        "",
        f"IoU hit threshold: `{args.iou_threshold}`. Center hit threshold: `{args.center_threshold}` normalized image distance.",
        "",
        "| Method | Samples | Main metric | Mean IoU | Center error | Top-3 / upper-bound |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            "| Ours: direct cue-memory route target "
            f"| {fmt(direct.get('samples'))} "
            f"| hit@IoU {fmt(direct.get('hit_iou'))} "
            f"| {fmt(direct.get('mean_iou'))} "
            f"| {fmt(direct.get('center_error'))} "
            f"| center-hit {fmt(direct.get('hit_center'))} |"
        ),
        (
            "| Xu baseline: oracle STP candidates + MSTP selector "
            f"| {fmt(oracle.get('samples'))} "
            f"| acc {fmt(oracle.get('accuracy'))} "
            f"| selected-hit {fmt(oracle.get('selected_iou_hit'))} "
            "| - "
            f"| top-3 {fmt(oracle.get('top3_accuracy'))} |"
        ),
        (
            "| Xu baseline: detected STP candidates + MSTP selector "
            f"| {fmt(detected.get('samples'))} "
            f"| hit@IoU {fmt(detected.get('end_to_end_hit_iou'))} "
            "| - "
            "| - "
            f"| detector recall {fmt(detected.get('detector_mstp_recall'))} |"
        ),
        "",
        "## Interpretation Guide",
        "",
        "- If direct cue-memory hit@IoU is close to or above the detected Xu pipeline, then our method is competitive at finding a route target without explicit STP proposals.",
        "- If oracle-candidate Xu is much higher than detected Xu, the bottleneck is STP detection rather than MSTP selection.",
        "- If direct cue-memory is low but center error is reasonable, the model is roughly pointing toward the right area but not producing accurate boxes.",
        "- This comparison is a proxy for screen-only route finding. It does not prove full trajectory navigation because VisualGuidance has no temporal movement labels.",
        "",
        "## Config",
        "",
        "```text",
        f"annotations: {args.annotations}",
        f"selector_annotations: {args.selector_annotations}",
        f"direct_backbone: {direct_meta.get('backbone')}",
        f"direct_hidden_dim: {direct_meta.get('hidden_dim')}",
        f"direct_spatial_relation_type: {direct_meta.get('spatial_relation_type')}",
        f"selector_baseline: VisualGuidance local-crop/global-context selector",
        f"selector_image_size: {selector_meta.get('image_size')}",
        f"direct_checkpoint: {args.direct_checkpoint}",
        f"selector_checkpoint: {args.selector_checkpoint}",
        f"detector_checkpoint: {args.detector_checkpoint}",
        f"score_threshold: {args.score_threshold}",
        f"max_detections: {args.max_detections}",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    direct_model, direct_checkpoint = load_direct_predictor(args.direct_checkpoint, device)
    selector_model, selector_checkpoint = load_visualguidance_selector(args.selector_checkpoint, device)
    direct_image_size = int(direct_checkpoint.get("image_size", 128))
    selector_image_size = int(selector_checkpoint.get("image_size", 128))

    result = {
        "config": {
            "annotations": args.annotations.as_posix(),
            "selector_annotations": args.selector_annotations.as_posix(),
            "image_root": args.image_root.as_posix(),
            "direct_checkpoint": args.direct_checkpoint.as_posix(),
            "selector_checkpoint": args.selector_checkpoint.as_posix(),
            "detector_checkpoint": args.detector_checkpoint.as_posix() if args.detector_checkpoint else None,
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "center_threshold": args.center_threshold,
            "max_detections": args.max_detections,
        },
        "direct_model": {
            "backbone": direct_checkpoint.get("backbone", "small_cnn"),
            "hidden_dim": direct_checkpoint.get("hidden_dim", 128),
            "num_cue_tokens": direct_checkpoint.get("num_cue_tokens", 8),
            "spatial_relation_type": direct_checkpoint.get("spatial_relation_type", "topk_graph"),
            "image_size": direct_image_size,
        },
        "selector_model": {
            "name": selector_checkpoint.get("model_name", "visualguidance_mstp_selector_baseline"),
            "hidden_dim": selector_checkpoint.get("hidden_dim", 256),
            "image_size": selector_image_size,
            "crop_size": selector_checkpoint.get("crop_size", 224),
            "global_size": selector_checkpoint.get("global_size", 64),
        },
        "direct_training_metrics": _read_metrics(args.direct_metrics),
        "selector_training_metrics": _read_metrics(args.selector_metrics),
        "direct_route_target": evaluate_direct(direct_model, args, device, direct_image_size),
        "visualguidance_oracle_candidates": evaluate_oracle_selector(
            selector_model,
            args,
            device,
            selector_image_size,
        ),
    }
    if args.detector_checkpoint and args.detector_checkpoint.exists():
        detector = load_detector(args.detector_checkpoint, device)
        result["visualguidance_detected_candidates"] = evaluate_detected_selector(
            detector,
            selector_model,
            args,
            device,
            selector_image_size,
        )
    else:
        result["visualguidance_detected_candidates"] = {"available": False}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(args.output_md, result, args)
    print(json.dumps({k: v for k, v in result.items() if k.endswith("candidates") or k == "direct_route_target"}, indent=2))


if __name__ == "__main__":
    main()
