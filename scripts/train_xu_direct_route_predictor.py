"""Train cue-memory direct route target prediction on VisualGuidance data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.xu_mstp import CueMemoryDirectRoutePredictor
from src.train_path_predictor import load_flat_config, set_seed
from src.xu_mstp.dataset import XuRouteTargetDataset, collate_xu_route_batch
from src.xu_mstp.stp_detection import box_iou


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
    parser.add_argument("--test-annotations", type=Path, default=default("test_annotations", None), required="test_annotations" not in defaults)
    parser.add_argument("--image-root", type=Path, default=default("image_root", None), required="image_root" not in defaults)
    parser.add_argument("--output-dir", type=Path, default=default("output_dir", None), required="output_dir" not in defaults)
    parser.add_argument("--backbone", default=default("backbone", "small_cnn"))
    parser.add_argument("--hidden-dim", type=int, default=default("hidden_dim", 128))
    parser.add_argument("--num-cue-tokens", type=int, default=default("num_cue_tokens", 8))
    parser.add_argument("--spatial-relation-type", default=default("spatial_relation_type", "topk_graph"))
    parser.add_argument("--spatial-graph-neighbors", type=int, default=default("spatial_graph_neighbors", 8))
    parser.add_argument("--adapter-bottleneck-dim", type=int, default=default("adapter_bottleneck_dim", 64))
    parser.add_argument("--image-size", type=int, default=default("image_size", 128))
    parser.add_argument("--epochs", type=int, default=default("epochs", 40))
    parser.add_argument("--batch-size", type=int, default=default("batch_size", 32))
    parser.add_argument("--lr", type=float, default=default("lr", 5e-4))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay", 1e-3))
    parser.add_argument("--dropout", type=float, default=default("dropout", 0.2))
    parser.add_argument("--grad-clip-norm", type=float, default=default("grad_clip_norm", 1.0))
    parser.add_argument("--early-stopping-patience", type=int, default=default("early_stopping_patience", 10))
    parser.add_argument("--freeze-backbone", action="store_true", default=bool(default("freeze_backbone", False)))
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--seed", type=int, default=default("seed", 7))
    parser.add_argument("--device", default=default("device", "auto"))
    parser.add_argument("--num-workers", type=int, default=default("num_workers", 0))
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["image"] = moved["image"].to(device)
    moved["target_box"] = moved["target_box"].to(device)
    return moved


def make_loader(annotations: Path, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = XuRouteTargetDataset(annotations, args.image_root, image_size=args.image_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_xu_route_batch,
    )


def center_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_center = (pred[:, :2] + pred[:, 2:]) * 0.5
    target_center = (target[:, :2] + target[:, 2:]) * 0.5
    return torch.linalg.norm(pred_center - target_center, dim=-1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    center_sum = 0.0
    iou_sum = 0.0
    hit_05 = 0
    hit_center_01 = 0
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        target = batch["target_box"]
        loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
        distances = center_distance(pred, target)
        ious = box_iou(pred.detach().cpu(), target.detach().cpu()).diag()
        batch_size = target.shape[0]
        total += batch_size
        loss_sum += float(loss.sum().detach().cpu())
        center_sum += float(distances.sum().detach().cpu())
        iou_sum += float(ious.sum())
        hit_05 += int((ious >= 0.5).sum().item())
        hit_center_01 += int((distances.detach().cpu() <= 0.1).sum().item())
    return {
        "loss": loss_sum / max(total, 1),
        "center_error": center_sum / max(total, 1),
        "mean_iou": iou_sum / max(total, 1),
        "hit_iou_0_5": hit_05 / max(total, 1),
        "hit_center_0_1": hit_center_01 / max(total, 1),
        "samples": total,
    }


def save_checkpoint(path: Path, model: torch.nn.Module, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "cue_memory_direct_route_predictor",
            "backbone": args.backbone,
            "hidden_dim": args.hidden_dim,
            "num_cue_tokens": args.num_cue_tokens,
            "spatial_relation_type": args.spatial_relation_type,
            "spatial_graph_neighbors": args.spatial_graph_neighbors,
            "adapter_bottleneck_dim": args.adapter_bottleneck_dim,
            "image_size": args.image_size,
            "freeze_backbone": args.freeze_backbone,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    train_loader = make_loader(args.train_annotations, args, shuffle=True)
    test_loader = make_loader(args.test_annotations, args, shuffle=False)
    model = CueMemoryDirectRoutePredictor(
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        freeze_backbone=args.freeze_backbone,
        num_cue_tokens=args.num_cue_tokens,
        spatial_relation_type=args.spatial_relation_type,
        spatial_graph_neighbors=args.spatial_graph_neighbors,
        dropout=args.dropout,
        adapter_bottleneck_dim=args.adapter_bottleneck_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_hit = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    config = vars(args).copy()
    for key in ("train_annotations", "test_annotations", "image_root", "output_dir", "config"):
        if config.get(key) is not None:
            config[key] = Path(config[key]).as_posix()
    config["device"] = str(device)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            target = batch["target_box"]
            box_loss = F.smooth_l1_loss(pred, target)
            center_loss = center_distance(pred, target).mean()
            loss = box_loss + 0.5 * center_loss
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * target.shape[0]
            total += target.shape[0]
        test_metrics = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "test": test_metrics,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"test_hit={test_metrics['hit_iou_0_5']:.4f} center={test_metrics['center_error']:.4f}"
        )
        score = test_metrics["hit_iou_0_5"]
        if score > best_hit:
            best_hit = score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(args.output_dir / "best.pt", model, args, row)
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping=epoch={epoch} best_epoch={best_epoch} best_hit={best_hit:.4f}")
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    metrics = {
        "model": "cue_memory_direct_route_predictor",
        "best_epoch": best_epoch,
        "history": history,
        "test": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_checkpoint(args.output_dir / "best.pt", model, args, metrics)
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
