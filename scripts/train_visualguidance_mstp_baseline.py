"""Train a VisualGuidance-style MSTP selector baseline."""

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

from src.models.xu_mstp import VisualGuidanceMSTPSelectorBaseline
from src.train_path_predictor import load_flat_config, set_seed
from src.xu_mstp.dataset import XuMSTPSelectionDataset, collate_xu_mstp_batch


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
    parser.add_argument("--image-size", type=int, default=default("image_size", 128))
    parser.add_argument("--epochs", type=int, default=default("epochs", 30))
    parser.add_argument("--batch-size", type=int, default=default("batch_size", 16))
    parser.add_argument("--lr", type=float, default=default("lr", 5e-4))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay", 1e-3))
    parser.add_argument("--hidden-dim", type=int, default=default("hidden_dim", 256))
    parser.add_argument("--bottleneck-dim", type=int, default=default("bottleneck_dim", 256))
    parser.add_argument("--crop-size", type=int, default=default("crop_size", 224))
    parser.add_argument("--global-size", type=int, default=default("global_size", 64))
    parser.add_argument("--pretrained", action="store_true", default=bool(default("pretrained", True)))
    parser.add_argument("--random-init", dest="pretrained", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=default("early_stopping_patience", 8))
    parser.add_argument("--grad-clip-norm", type=float, default=default("grad_clip_norm", 1.0))
    parser.add_argument("--seed", type=int, default=default("seed", 7))
    parser.add_argument("--device", default=default("device", "auto"))
    parser.add_argument("--num-workers", type=int, default=default("num_workers", 0))
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("image", "candidate_boxes", "candidate_mask", "gt_index"):
        moved[key] = moved[key].to(device)
    return moved


def make_loader(annotations: Path, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = XuMSTPSelectionDataset(annotations, args.image_root, image_size=args.image_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_xu_mstp_batch,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    top3 = 0
    loss_sum = 0.0
    candidate_sum = 0
    criterion = torch.nn.CrossEntropyLoss()
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch)
        target = batch["gt_index"]
        loss = criterion(logits, target)
        pred = logits.argmax(dim=1)
        k = min(3, logits.shape[1])
        top = logits.topk(k=k, dim=1).indices
        total += target.shape[0]
        correct += int((pred == target).sum().item())
        top3 += int((top == target[:, None]).any(dim=1).sum().item())
        loss_sum += float(loss.detach().cpu()) * target.shape[0]
        candidate_sum += int(batch["candidate_mask"].sum().item())
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3 / max(total, 1),
        "samples": total,
        "mean_candidates": candidate_sum / max(total, 1),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "visualguidance_mstp_selector_baseline",
            "hidden_dim": args.hidden_dim,
            "bottleneck_dim": args.bottleneck_dim,
            "crop_size": args.crop_size,
            "global_size": args.global_size,
            "image_size": args.image_size,
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
    train_loader = make_loader(args.train_annotations, args, shuffle=True)
    test_loader = make_loader(args.test_annotations, args, shuffle=False)
    model = VisualGuidanceMSTPSelectorBaseline(
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        crop_size=args.crop_size,
        global_size=args.global_size,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = -1.0
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
        correct = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch["gt_index"])
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch["gt_index"].shape[0]
            total += batch["gt_index"].shape[0]
            correct += int((logits.argmax(dim=1) == batch["gt_index"]).sum().item())
        test_metrics = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "test": test_metrics,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_acc={row['train_accuracy']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} top3={test_metrics['top3_accuracy']:.4f}"
        )
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(args.output_dir / "best.pt", model, args, row)
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping=epoch={epoch} best_epoch={best_epoch} best_acc={best_acc:.4f}")
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    metrics = {
        "model": "visualguidance_mstp_selector_baseline",
        "best_epoch": best_epoch,
        "history": history,
        "test": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_checkpoint(args.output_dir / "best.pt", model, args, metrics)
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
