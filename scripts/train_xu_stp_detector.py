"""Train and evaluate a VisualGuidance STP detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_path_predictor import load_flat_config, set_seed
from src.xu_mstp.stp_detection import (
    XuSTPDetectionDataset,
    box_iou,
    create_fasterrcnn_stp_detector,
    greedy_match_count,
    mstp_box_from_record,
    sanitize_absolute_boxes,
    stp_detection_collate,
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
    parser.add_argument("--train-annotations", type=Path, default=default("train_annotations", None), required="train_annotations" not in defaults)
    parser.add_argument("--val-annotations", type=Path, default=default("val_annotations", None))
    parser.add_argument("--test-annotations", type=Path, default=default("test_annotations", None), required="test_annotations" not in defaults)
    parser.add_argument("--image-root", type=Path, default=default("image_root", None), required="image_root" not in defaults)
    parser.add_argument("--output-dir", type=Path, default=default("output_dir", None), required="output_dir" not in defaults)
    parser.add_argument("--epochs", type=int, default=default("epochs", 12))
    parser.add_argument("--batch-size", type=int, default=default("batch_size", 2))
    parser.add_argument("--lr", type=float, default=default("lr", 0.005))
    parser.add_argument("--momentum", type=float, default=default("momentum", 0.9))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay", 0.0005))
    parser.add_argument("--min-size", type=int, default=default("min_size", 640))
    parser.add_argument("--max-size", type=int, default=default("max_size", 960))
    parser.add_argument("--pretrained", action="store_true", default=bool(default("pretrained", True)))
    parser.add_argument("--random-init", dest="pretrained", action="store_false")
    parser.add_argument("--score-threshold", type=float, default=default("score_threshold", 0.3))
    parser.add_argument("--iou-threshold", type=float, default=default("iou_threshold", 0.5))
    parser.add_argument("--max-detections", type=int, default=default("max_detections", 10))
    parser.add_argument("--early-stopping-patience", type=int, default=default("early_stopping_patience", 5))
    parser.add_argument("--grad-clip-norm", type=float, default=default("grad_clip_norm", 5.0))
    parser.add_argument("--seed", type=int, default=default("seed", 7))
    parser.add_argument("--device", default=default("device", "auto"))
    parser.add_argument("--num-workers", type=int, default=default("num_workers", 0))
    return parser.parse_args()


def make_loader(annotations: Path, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = XuSTPDetectionDataset(annotations, args.image_root)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=stp_detection_collate,
    )


def move_targets(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        moved.append({
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
            if key != "image_id_str"
        })
    return moved


@torch.no_grad()
def evaluate_detector(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> dict[str, float]:
    model.eval()
    total_gt = 0
    total_pred = 0
    matched = 0
    mstp_recalled = 0
    image_count = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            scores = output["scores"].detach().cpu()
            keep = torch.nonzero(scores >= score_threshold, as_tuple=False).squeeze(1)
            if keep.numel() > max_detections:
                keep = keep[:max_detections]
            pred_boxes = output["boxes"].detach().cpu()[keep]
            gt_boxes = target["boxes"].detach().cpu()
            total_gt += gt_boxes.shape[0]
            total_pred += pred_boxes.shape[0]
            matched += greedy_match_count(pred_boxes, gt_boxes, iou_threshold)
            image_count += 1

            image_id = str(target["image_id_str"])
            record = next((item for item in loader.dataset.records if str(item["image_id"]) == image_id), None)
            if record is None:
                continue
            mstp = mstp_box_from_record(record)
            if mstp is None:
                continue
            if pred_boxes.numel() > 0:
                width_height = pred_boxes.new_tensor([10**9, 10**9])
                _ = width_height  # keeps lint-free intent: boxes are already absolute.
                mstp_box = torch.tensor([mstp], dtype=torch.float32)
                if float(box_iou(pred_boxes, mstp_box).max().item()) >= iou_threshold:
                    mstp_recalled += 1
    precision = matched / max(total_pred, 1)
    recall = matched / max(total_gt, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1.0e-6)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_boxes": matched,
        "gt_boxes": total_gt,
        "pred_boxes": total_pred,
        "images": image_count,
        "mean_detections": total_pred / max(image_count, 1),
        "mstp_recall": mstp_recalled / max(image_count, 1),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
    }


def save_checkpoint(path: Path, model: torch.nn.Module, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "fasterrcnn_stp_detector",
            "min_size": args.min_size,
            "max_size": args.max_size,
            "pretrained": args.pretrained,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    val_annotations = args.val_annotations or args.test_annotations
    train_loader = make_loader(args.train_annotations, args, shuffle=True)
    val_loader = make_loader(val_annotations, args, shuffle=False)
    test_loader = make_loader(args.test_annotations, args, shuffle=False)

    model = create_fasterrcnn_stp_detector(
        pretrained=args.pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)
    best_recall = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    config = vars(args).copy()
    for key in ("train_annotations", "val_annotations", "test_annotations", "image_root", "output_dir", "config"):
        if config.get(key) is not None:
            config[key] = Path(config[key]).as_posix()
    config["device"] = str(device)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            moved_targets = move_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, moved_targets)
            loss = sum(value for value in loss_dict.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite detector loss at epoch {epoch}")
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        val_metrics = evaluate_detector(
            model,
            val_loader,
            device,
            args.score_threshold,
            args.iou_threshold,
            args.max_detections,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "val": val_metrics,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} val_mstp_recall={val_metrics['mstp_recall']:.4f}"
        )
        score = val_metrics["mstp_recall"]
        if score > best_recall:
            best_recall = score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(args.output_dir / "best.pt", model, args, row)
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping=epoch={epoch} best_epoch={best_epoch} best_mstp_recall={best_recall:.4f}")
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics = evaluate_detector(
        model,
        val_loader,
        device,
        args.score_threshold,
        args.iou_threshold,
        args.max_detections,
    )
    test_metrics = evaluate_detector(
        model,
        test_loader,
        device,
        args.score_threshold,
        args.iou_threshold,
        args.max_detections,
    )
    metrics = {
        "model": "fasterrcnn_stp_detector",
        "best_epoch": best_epoch,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_checkpoint(args.output_dir / "best.pt", model, args, metrics)
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
