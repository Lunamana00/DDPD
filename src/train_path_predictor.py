"""Train WIT-VZ path prediction models and baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .losses import trajectory_loss
from .metrics import ade, fde, per_horizon_error, select_best_trajectory
from .models.factory import create_model, needs_rgb
from .wit_vz.dataset import WITVZPathDataset, collate_path_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a WIT-VZ path predictor.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", default="small_cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-patience", type=int, default=25)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--loss", choices=["huber", "mse", "l2"], default="huber")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--num-cue-tokens", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--multimodal-confidence-weight", type=float, default=0.05)
    parser.add_argument("--temporal-type", choices=["transformer", "gru"], default="transformer")
    parser.add_argument(
        "--trajectory-scale",
        default="auto",
        help="Coordinate scale for normalized loss. Use 'auto' to estimate from train targets.",
    )
    parser.add_argument(
        "--residual-scale",
        default="auto",
        help="Scale for learned residual path. Use 'auto' to reuse the resolved trajectory scale.",
    )
    parser.add_argument("--no-cv-residual", dest="use_constant_velocity_residual", action="store_false")
    parser.set_defaults(use_constant_velocity_residual=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("rgb_history", "ego_history", "future_path"):
        if key in moved:
            moved[key] = moved[key].to(device)
    return moved


def estimate_trajectory_scale(dataset: WITVZPathDataset) -> float:
    values = []
    for sample in dataset.samples:
        values.append(torch.tensor(sample["future_local_path"], dtype=torch.float32))
    if not values:
        return 1.0
    stacked = torch.stack(values, dim=0)
    return float(torch.sqrt((stacked ** 2).mean()).clamp_min(1.0).item())


def resolve_scale(value: str | float, dataset: WITVZPathDataset | None = None, fallback: float = 1.0) -> float:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "auto":
            if dataset is None:
                return fallback
            return estimate_trajectory_scale(dataset)
        if normalized in {"none", "off"}:
            return 1.0
    return float(value)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    coordinate_scale: float = 1.0,
    multimodal_confidence_weight: float = 0.05,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_count = 0
    horizon_errors = []
    predictions = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        target = batch["future_path"]
        loss = trajectory_loss(
            pred,
            target,
            loss_type,
            coordinate_scale=coordinate_scale,
            multimodal_confidence_weight=multimodal_confidence_weight,
        )
        selected_pred = select_best_trajectory(pred, target)
        batch_size = target.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_ade += float(ade(pred, target).detach().cpu()) * batch_size
        total_fde += float(fde(pred, target).detach().cpu()) * batch_size
        horizon_errors.append(per_horizon_error(pred, target).detach().cpu() * batch_size)
        total_count += batch_size
        for i, sample_id in enumerate(batch["sample_id"]):
            item = {
                "sample_id": sample_id,
                "prediction": selected_pred[i].detach().cpu().tolist(),
                "target": target[i].detach().cpu().tolist(),
                "ADE": float(torch.linalg.norm(selected_pred[i] - target[i], dim=-1).mean().detach().cpu()),
                "FDE": float(torch.linalg.norm(selected_pred[i, -1] - target[i, -1]).detach().cpu()),
            }
            if isinstance(pred, dict):
                item["candidate_predictions"] = pred["paths"][i].detach().cpu().tolist()
                if "logits" in pred:
                    item["mode_logits"] = pred["logits"][i].detach().cpu().tolist()
            predictions.append(item)
    if total_count == 0:
        raise ValueError("Evaluation loader produced no samples")
    per_h = torch.stack(horizon_errors, dim=0).sum(dim=0) / total_count
    return {
        "loss": total_loss / total_count,
        "ADE": total_ade / total_count,
        "FDE": total_fde / total_count,
        "per_horizon_error": per_h.tolist(),
    }, predictions


def make_loader(args: argparse.Namespace, split: str, load_rgb: bool) -> DataLoader:
    dataset = WITVZPathDataset(
        args.dataset,
        split=split,
        image_size=args.image_size,
        load_rgb=load_rgb,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        collate_fn=collate_path_batch,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataset_manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": args.model,
            "backbone": args.backbone,
            "hidden_dim": args.hidden_dim,
            "image_size": args.image_size,
            "future_steps": dataset_manifest["future_steps"],
            "history_frames": dataset_manifest["history_frames"],
            "freeze_backbone": args.freeze_backbone,
            "num_cue_tokens": args.num_cue_tokens,
            "num_modes": args.num_modes,
            "multimodal_confidence_weight": args.multimodal_confidence_weight,
            "temporal_type": args.temporal_type,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "min_lr": args.min_lr,
            "trajectory_scale": float(getattr(args, "resolved_trajectory_scale", 1.0)),
            "residual_scale": float(getattr(args, "resolved_residual_scale", 1.0)),
            "use_constant_velocity_residual": bool(getattr(args, "use_constant_velocity_residual", True)),
            "loss": args.loss,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    dataset_manifest = json.loads((args.dataset / "dataset_manifest.json").read_text())
    load_rgb = needs_rgb(args.model)

    train_loader = make_loader(args, "train", load_rgb)
    val_loader = make_loader(args, "val", load_rgb)
    test_loader = make_loader(args, "test", load_rgb)
    trajectory_scale = resolve_scale(args.trajectory_scale, train_loader.dataset)
    residual_scale = resolve_scale(args.residual_scale, None, fallback=trajectory_scale)
    args.resolved_trajectory_scale = trajectory_scale
    args.resolved_residual_scale = residual_scale

    model = create_model(
        args.model,
        future_steps=int(dataset_manifest["future_steps"]),
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        freeze_backbone=args.freeze_backbone,
        num_cue_tokens=args.num_cue_tokens,
        num_modes=args.num_modes,
        temporal_type=args.temporal_type,
        dropout=args.dropout,
        use_constant_velocity_residual=args.use_constant_velocity_residual,
        residual_scale=residual_scale,
    ).to(device)

    config = vars(args).copy()
    config["dataset"] = args.dataset.as_posix()
    config["output_dir"] = args.output_dir.as_posix()
    config["device"] = str(device)
    config["resolved_trajectory_scale"] = trajectory_scale
    config["resolved_residual_scale"] = residual_scale
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.model == "constant_velocity":
        val_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            args.loss,
            trajectory_scale,
            args.multimodal_confidence_weight,
        )
        test_metrics, predictions = evaluate_loader(
            model,
            test_loader,
            device,
            args.loss,
            trajectory_scale,
            args.multimodal_confidence_weight,
        )
        metrics = {"val": val_metrics, "test": test_metrics, "model": args.model}
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        save_checkpoint(args.output_dir / "best.pt", model, args, dataset_manifest, metrics)
        with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        print(json.dumps(metrics, indent=2))
        return

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler_patience > 0 and 0.0 < args.lr_scheduler_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.min_lr,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ade = 0.0
        total_count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.mixed_precision and device.type == "cuda"):
                pred = model(batch)
                loss = trajectory_loss(
                    pred,
                    batch["future_path"],
                    args.loss,
                    coordinate_scale=trajectory_scale,
                    multimodal_confidence_weight=args.multimodal_confidence_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=args.grad_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            batch_size = batch["future_path"].shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_ade += float(ade(pred, batch["future_path"]).detach().cpu()) * batch_size
            total_count += batch_size

        train_loss = total_loss / max(total_count, 1)
        train_ade = total_ade / max(total_count, 1)
        val_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            args.loss,
            trajectory_scale,
            args.multimodal_confidence_weight,
        )
        if scheduler is not None:
            scheduler.step(val_metrics["ADE"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ADE": train_ade,
            "val_train_ADE_gap": val_metrics["ADE"] - train_ade,
            "lr": current_lr,
            "val": val_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_ADE={train_ade:.4f} val_ADE={val_metrics['ADE']:.4f} lr={current_lr:.2e}"
        )
        if val_metrics["ADE"] < best_val - args.early_stopping_min_delta:
            best_val = val_metrics["ADE"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(args.output_dir / "best.pt", model, args, dataset_manifest, row)
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                "early_stopping="
                f"epoch={epoch} best_epoch={best_epoch} best_val_ADE={best_val:.4f}"
            )
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics, _ = evaluate_loader(
        model,
        val_loader,
        device,
        args.loss,
        trajectory_scale,
        args.multimodal_confidence_weight,
    )
    test_metrics, predictions = evaluate_loader(
        model,
        test_loader,
        device,
        args.loss,
        trajectory_scale,
        args.multimodal_confidence_weight,
    )
    metrics = {"model": args.model, "history": history, "val": val_metrics, "test": test_metrics}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_checkpoint(args.output_dir / "best.pt", model, args, dataset_manifest, metrics)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
