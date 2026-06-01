"""Train WIT-VZ path prediction models and baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .losses import trajectory_loss
from .metrics import ade, fde, per_horizon_error, select_best_trajectory
from .models.factory import create_model, needs_rgb
from .wit_vz.dataset import WITVZPathDataset, collate_path_batch, sample_group_key


def _parse_config_scalar(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return ""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def load_flat_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line {line_number} in {path}: {line}")
        key, value = stripped.split(":", 1)
        config[key.strip()] = _parse_config_scalar(value)
    return config


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for key in ("dataset", "visual_feature_cache", "output_dir", "config"):
        value = getattr(args, key, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, key, Path(value))
    if args.spatial_relation_type is None:
        args.spatial_relation_type = "topk_graph" if args.use_spatial_graph else "none"
    args.use_spatial_graph = args.spatial_relation_type != "none"
    return args


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _remaining = config_parser.parse_known_args()
    defaults = load_flat_config(config_args.config)

    def default(key: str, fallback: Any) -> Any:
        return defaults.get(key, fallback)

    parser = argparse.ArgumentParser(
        description="Train a WIT-VZ path predictor.",
        parents=[config_parser],
    )
    parser.add_argument("--dataset", type=Path, default=default("dataset", None), required="dataset" not in defaults)
    parser.add_argument("--visual-feature-cache", type=Path, default=default("visual_feature_cache", None))
    parser.add_argument("--model", default=default("model", None), required="model" not in defaults)
    parser.add_argument("--backbone", default=default("backbone", "small_cnn"))
    parser.add_argument("--epochs", type=int, default=default("epochs", 5))
    parser.add_argument("--batch-size", type=int, default=default("batch_size", 4))
    parser.add_argument("--lr", type=float, default=default("lr", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay", 1e-4))
    parser.add_argument("--dropout", type=float, default=default("dropout", 0.1))
    parser.add_argument("--grad-clip-norm", type=float, default=default("grad_clip_norm", 1.0))
    parser.add_argument("--early-stopping-patience", type=int, default=default("early_stopping_patience", 100))
    parser.add_argument("--early-stopping-min-delta", type=float, default=default("early_stopping_min_delta", 0.0))
    parser.add_argument("--lr-scheduler-patience", type=int, default=default("lr_scheduler_patience", 25))
    parser.add_argument("--lr-scheduler-factor", type=float, default=default("lr_scheduler_factor", 0.5))
    parser.add_argument("--min-lr", type=float, default=default("min_lr", 1e-6))
    parser.add_argument("--hidden-dim", type=int, default=default("hidden_dim", 128))
    parser.add_argument("--image-size", type=int, default=default("image_size", 64))
    parser.add_argument("--loss", choices=["huber", "mse", "l2"], default=default("loss", "huber"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default("output_dir", None),
        required="output_dir" not in defaults,
    )
    parser.add_argument("--seed", type=int, default=default("seed", 7))
    parser.add_argument("--device", default=default("device", "auto"))
    parser.add_argument("--num-workers", type=int, default=default("num_workers", 0))
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        default=bool(default("data_parallel", False)),
        help="Wrap the model with torch.nn.DataParallel when multiple CUDA devices are visible.",
    )
    parser.add_argument(
        "--balance-key",
        choices=[
            "none",
            "source",
            "scenario",
            "map",
            "policy",
            "episode",
            "source_scenario",
            "source_map",
            "source_policy",
        ],
        default=default("balance_key", "none"),
        help="Metadata group used for train-set balancing.",
    )
    parser.add_argument(
        "--balance-mode",
        choices=["none", "sampler", "loss", "both"],
        default=default("balance_mode", "none"),
        help="Apply balancing via WeightedRandomSampler, per-sample loss weights, or both.",
    )
    parser.add_argument(
        "--balance-exponent",
        type=float,
        default=default("balance_exponent", 1.0),
        help="Inverse-frequency exponent. 1.0 fully balances groups; 0.5 is softer.",
    )
    parser.add_argument("--freeze-backbone", action="store_true", default=bool(default("freeze_backbone", True)))
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--mixed-precision", action="store_true", default=bool(default("mixed_precision", False)))
    parser.add_argument("--num-cue-tokens", type=int, default=default("num_cue_tokens", 8))
    parser.add_argument("--num-modes", type=int, default=default("num_modes", 1))
    parser.add_argument(
        "--multimodal-confidence-weight",
        type=float,
        default=default("multimodal_confidence_weight", 0.05),
    )
    parser.add_argument(
        "--temporal-type",
        choices=["transformer", "gru", "timesformer", "strnet"],
        default=default("temporal_type", "transformer"),
    )
    parser.add_argument("--temporal-layers", type=int, default=default("temporal_layers", 1))
    parser.add_argument(
        "--selector-type",
        choices=["query_attention", "tokenlearner", "topk_tokenlearner"],
        default=default("selector_type", "query_attention"),
    )
    parser.add_argument("--selector-layers", type=int, default=default("selector_layers", 1))
    parser.add_argument(
        "--tokenlearner-pooling",
        choices=["sigmoid", "softmax"],
        default=default("tokenlearner_pooling", "sigmoid"),
    )
    parser.add_argument(
        "--memory-type",
        choices=["gru_cell", "attention"],
        default=default("memory_type", "gru_cell"),
    )
    parser.add_argument("--use-spatial-graph", action="store_true", default=bool(default("use_spatial_graph", False)))
    parser.add_argument("--spatial-graph-neighbors", type=int, default=default("spatial_graph_neighbors", 8))
    parser.add_argument(
        "--spatial-relation-type",
        choices=["topk_graph", "none", "full_attention", "local_grid"],
        default=default("spatial_relation_type", None),
    )
    parser.add_argument(
        "--use-temporal-difference-conv",
        action="store_true",
        default=bool(default("use_temporal_difference_conv", False)),
    )
    parser.add_argument("--use-temporal-shift", action="store_true", default=bool(default("use_temporal_shift", False)))
    parser.add_argument("--decoder-layers", type=int, default=default("decoder_layers", 1))
    parser.add_argument("--cue-temporal-layers", type=int, default=default("cue_temporal_layers", 1))
    parser.add_argument(
        "--trajectory-scale",
        default=default("trajectory_scale", "auto"),
        help="Coordinate scale for normalized loss. Use 'auto' to estimate from train targets.",
    )
    parser.add_argument(
        "--residual-scale",
        default=default("residual_scale", "auto"),
        help="Scale for learned residual path. Use 'auto' to reuse the resolved trajectory scale.",
    )
    parser.add_argument(
        "--no-cv-residual",
        dest="use_constant_velocity_residual",
        action="store_false",
        default=bool(default("use_constant_velocity_residual", True)),
    )
    return normalize_args(parser.parse_args())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("rgb_history", "visual_tokens", "ego_history", "future_path"):
        if key in moved:
            moved[key] = moved[key].to(device)
    return moved


def make_grad_scaler(device: torch.device, enabled: bool) -> torch.cuda.amp.GradScaler | torch.amp.GradScaler:
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def autocast_kwargs(device: torch.device, enabled: bool) -> dict[str, Any]:
    return {"enabled": enabled and device.type == "cuda"}


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


def compute_balance_weights(
    samples: list[dict[str, Any]],
    key: str,
    exponent: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if key == "none":
        weights = torch.ones(len(samples), dtype=torch.float32)
        return weights, {"enabled": False, "key": key, "groups": {}}

    groups = [sample_group_key(sample, key) for sample in samples]
    counts = Counter(groups)
    raw_weights = torch.tensor(
        [float(counts[group]) ** (-max(exponent, 0.0)) for group in groups],
        dtype=torch.float32,
    )
    weights = raw_weights * (len(raw_weights) / raw_weights.sum().clamp_min(1.0e-6))
    group_weights = {
        group: float(float(count) ** (-max(exponent, 0.0)))
        for group, count in counts.items()
    }
    mean_raw_weight = sum(group_weights[group] * count for group, count in counts.items()) / max(len(groups), 1)
    normalized_group_weights = {
        group: weight / max(mean_raw_weight, 1.0e-6)
        for group, weight in group_weights.items()
    }
    return weights, {
        "enabled": True,
        "key": key,
        "exponent": exponent,
        "groups": {
            group: {
                "count": int(count),
                "weight": float(normalized_group_weights[group]),
            }
            for group, count in sorted(counts.items())
        },
    }


def make_loss_weight_lookup(samples: list[dict[str, Any]], key: str, exponent: float) -> dict[str, float]:
    if key == "none":
        return {}
    _weights, stats = compute_balance_weights(samples, key, exponent)
    return {
        group: float(values["weight"])
        for group, values in stats.get("groups", {}).items()
    }


def batch_sample_weights(
    batch: dict[str, Any],
    key: str,
    weight_lookup: dict[str, float],
    device: torch.device,
) -> torch.Tensor | None:
    if key == "none" or not weight_lookup:
        return None
    weights = []
    for item in batch.get("balance", []):
        group = item[key]
        weights.append(weight_lookup.get(group, 1.0))
    if not weights:
        return None
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
        visual_feature_cache_dir=getattr(args, "visual_feature_cache", None),
    )
    sampler = None
    shuffle = split == "train"
    balance_key = getattr(args, "balance_key", "none")
    balance_mode = getattr(args, "balance_mode", "none")
    balance_exponent = getattr(args, "balance_exponent", 1.0)
    if (
        split == "train"
        and balance_key != "none"
        and balance_mode in {"sampler", "both"}
    ):
        weights, _stats = compute_balance_weights(
            dataset.samples,
            balance_key,
            balance_exponent,
        )
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
            "visual_feature_cache": (
                args.visual_feature_cache.as_posix() if args.visual_feature_cache is not None else None
            ),
            "hidden_dim": args.hidden_dim,
            "image_size": args.image_size,
            "future_steps": dataset_manifest["future_steps"],
            "history_frames": dataset_manifest["history_frames"],
            "freeze_backbone": args.freeze_backbone,
            "num_cue_tokens": args.num_cue_tokens,
            "num_modes": args.num_modes,
            "temporal_layers": args.temporal_layers,
            "selector_layers": args.selector_layers,
            "decoder_layers": args.decoder_layers,
            "cue_temporal_layers": args.cue_temporal_layers,
            "tokenlearner_pooling": args.tokenlearner_pooling,
            "selector_type": args.selector_type,
            "memory_type": args.memory_type,
            "use_spatial_graph": args.use_spatial_graph,
            "spatial_graph_neighbors": args.spatial_graph_neighbors,
            "spatial_relation_type": args.spatial_relation_type,
            "use_temporal_difference_conv": args.use_temporal_difference_conv,
            "use_temporal_shift": args.use_temporal_shift,
            "multimodal_confidence_weight": args.multimodal_confidence_weight,
            "temporal_type": args.temporal_type,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "balance_key": args.balance_key,
            "balance_mode": args.balance_mode,
            "balance_exponent": args.balance_exponent,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "min_lr": args.min_lr,
            "trajectory_scale": float(getattr(args, "resolved_trajectory_scale", 1.0)),
            "residual_scale": float(getattr(args, "resolved_residual_scale", 1.0)),
            "use_constant_velocity_residual": bool(getattr(args, "use_constant_velocity_residual", True)),
            "data_parallel": bool(getattr(args, "data_parallel", False)),
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
    load_rgb = needs_rgb(args.model) and args.visual_feature_cache is None

    train_loader = make_loader(args, "train", load_rgb)
    val_loader = make_loader(args, "val", load_rgb)
    test_loader = make_loader(args, "test", load_rgb)
    trajectory_scale = resolve_scale(args.trajectory_scale, train_loader.dataset)
    residual_scale = resolve_scale(args.residual_scale, None, fallback=trajectory_scale)
    args.resolved_trajectory_scale = trajectory_scale
    args.resolved_residual_scale = residual_scale
    balance_loss_weights = make_loss_weight_lookup(
        train_loader.dataset.samples,
        args.balance_key,
        args.balance_exponent,
    )
    _balance_sampler_weights, balance_stats = compute_balance_weights(
        train_loader.dataset.samples,
        args.balance_key,
        args.balance_exponent,
    )

    model = create_model(
        args.model,
        future_steps=int(dataset_manifest["future_steps"]),
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        freeze_backbone=args.freeze_backbone,
        num_cue_tokens=args.num_cue_tokens,
        num_modes=args.num_modes,
        temporal_type=args.temporal_type,
        temporal_layers=args.temporal_layers,
        selector_layers=args.selector_layers,
        decoder_layers=args.decoder_layers,
        cue_temporal_layers=args.cue_temporal_layers,
        tokenlearner_pooling=args.tokenlearner_pooling,
        selector_type=args.selector_type,
        memory_type=args.memory_type,
        use_spatial_graph=args.use_spatial_graph,
        spatial_graph_neighbors=args.spatial_graph_neighbors,
        spatial_relation_type=args.spatial_relation_type,
        use_temporal_difference_conv=args.use_temporal_difference_conv,
        use_temporal_shift=args.use_temporal_shift,
        dropout=args.dropout,
        use_constant_velocity_residual=args.use_constant_velocity_residual,
        residual_scale=residual_scale,
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    config = vars(args).copy()
    config["dataset"] = args.dataset.as_posix()
    if args.config is not None:
        config["config"] = args.config.as_posix()
    if args.visual_feature_cache is not None:
        config["visual_feature_cache"] = args.visual_feature_cache.as_posix()
    config["output_dir"] = args.output_dir.as_posix()
    config["device"] = str(device)
    config["resolved_trajectory_scale"] = trajectory_scale
    config["resolved_residual_scale"] = residual_scale
    config["balance_stats"] = balance_stats
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
        save_checkpoint(args.output_dir / "best.pt", unwrap_model(model), args, dataset_manifest, metrics)
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
    scaler = make_grad_scaler(device, args.mixed_precision)
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss = 0.0
        total_ade = 0.0
        total_count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            sample_weight = None
            if args.balance_mode in {"loss", "both"}:
                sample_weight = batch_sample_weights(
                    batch,
                    args.balance_key,
                    balance_loss_weights,
                    device,
                )
            with torch.cuda.amp.autocast(**autocast_kwargs(device, args.mixed_precision)):
                pred = model(batch)
                loss = trajectory_loss(
                    pred,
                    batch["future_path"],
                    args.loss,
                    coordinate_scale=trajectory_scale,
                    multimodal_confidence_weight=args.multimodal_confidence_weight,
                    sample_weight=sample_weight,
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
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if device.type == "cuda":
            row["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_ADE={train_ade:.4f} val_ADE={val_metrics['ADE']:.4f} lr={current_lr:.2e}"
        )
        if val_metrics["ADE"] < best_val - args.early_stopping_min_delta:
            best_val = val_metrics["ADE"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(args.output_dir / "best.pt", unwrap_model(model), args, dataset_manifest, row)
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                "early_stopping="
                f"epoch={epoch} best_epoch={best_epoch} best_val_ADE={best_val:.4f}"
            )
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
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
    save_checkpoint(args.output_dir / "best.pt", unwrap_model(model), args, dataset_manifest, metrics)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
