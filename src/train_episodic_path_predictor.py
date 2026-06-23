"""Train episodic WIT-VZ path predictors with across-window memory."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .losses import trajectory_loss
from .metrics import ade, fde, per_horizon_error, select_best_trajectory
from .models.factory import create_model, needs_rgb
from .models.motion import constant_velocity_path
from .train_path_predictor import (
    autocast_kwargs,
    choose_device,
    compute_balance_weights,
    load_flat_config,
    make_grad_scaler,
    resolve_scale,
    set_seed,
    unwrap_model,
)
from .wit_vz.dataset import WITVZEpisodicChunkDataset, collate_episodic_path_batch


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for key in ("dataset", "visual_feature_cache", "output_dir", "config"):
        value = getattr(args, key, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, key, Path(value))
    if args.spatial_relation_type is None:
        args.spatial_relation_type = "topk_graph" if args.use_spatial_graph else "none"
    args.use_spatial_graph = args.spatial_relation_type != "none"
    if args.eval_chunk_stride <= 0:
        args.eval_chunk_stride = max(args.chunk_length - args.burn_in, 1)
    return args


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _remaining = config_parser.parse_known_args()
    defaults = load_flat_config(config_args.config)

    def default(key: str, fallback: Any) -> Any:
        return defaults.get(key, fallback)

    parser = argparse.ArgumentParser(
        description="Train a WIT-VZ episodic path predictor.",
        parents=[config_parser],
    )
    parser.add_argument("--dataset", type=Path, default=default("dataset", None), required="dataset" not in defaults)
    parser.add_argument("--visual-feature-cache", type=Path, default=default("visual_feature_cache", None))
    parser.add_argument("--model", default=default("model", "episodic_long_term_cue_memory_path_predictor"))
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
    parser.add_argument("--history-frame-mode", choices=["full", "last_frame_only"], default=default("history_frame_mode", "full"))
    parser.add_argument("--train-frame-order", choices=["normal", "shuffle"], default=default("train_frame_order", "normal"))
    parser.add_argument("--loss", choices=["huber", "mse", "l2"], default=default("loss", "huber"))
    parser.add_argument("--output-dir", type=Path, default=default("output_dir", None), required="output_dir" not in defaults)
    parser.add_argument("--seed", type=int, default=default("seed", 7))
    parser.add_argument("--device", default=default("device", "auto"))
    parser.add_argument("--num-workers", type=int, default=default("num_workers", 0))
    parser.add_argument("--data-parallel", action="store_true", default=_as_bool(default("data_parallel", False)))
    parser.add_argument(
        "--balance-key",
        choices=["none", "source", "scenario", "map", "policy", "episode", "source_scenario", "source_map", "source_policy"],
        default=default("balance_key", "none"),
    )
    parser.add_argument("--balance-mode", choices=["none", "sampler", "loss", "both"], default=default("balance_mode", "none"))
    parser.add_argument("--balance-exponent", type=float, default=default("balance_exponent", 1.0))
    parser.add_argument("--freeze-backbone", action="store_true", default=_as_bool(default("freeze_backbone", True)))
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--mixed-precision", action="store_true", default=_as_bool(default("mixed_precision", False)))
    parser.add_argument("--num-cue-tokens", type=int, default=default("num_cue_tokens", 8))
    parser.add_argument("--num-modes", type=int, default=default("num_modes", 1))
    parser.add_argument("--multimodal-confidence-weight", type=float, default=default("multimodal_confidence_weight", 0.05))
    parser.add_argument("--temporal-type", choices=["transformer", "gru", "timesformer", "strnet", "none", "identity"], default=default("temporal_type", "transformer"))
    parser.add_argument("--temporal-layers", type=int, default=default("temporal_layers", 1))
    parser.add_argument("--selector-type", choices=["query_attention", "tokenlearner", "topk_tokenlearner"], default=default("selector_type", "query_attention"))
    parser.add_argument("--selector-layers", type=int, default=default("selector_layers", 1))
    parser.add_argument("--tokenlearner-pooling", choices=["sigmoid", "softmax"], default=default("tokenlearner_pooling", "sigmoid"))
    parser.add_argument(
        "--memory-type",
        choices=["gru_cell", "gru_no_ego", "attention", "attention_no_ego", "last_cue", "no_memory", "mean_cue", "no_memory_update"],
        default=default("memory_type", "attention"),
    )
    parser.add_argument("--long-memory-type", default=default("long_memory_type", "gated_attention"))
    parser.add_argument("--long-memory-slots", type=int, default=default("long_memory_slots", None))
    parser.add_argument("--long-memory-use-ego", action="store_true", default=_as_bool(default("long_memory_use_ego", True)))
    parser.add_argument("--no-long-memory-ego", dest="long_memory_use_ego", action="store_false")
    parser.add_argument("--detach-long-memory", action="store_true", default=_as_bool(default("detach_long_memory", True)))
    parser.add_argument("--no-detach-long-memory", dest="detach_long_memory", action="store_false")
    parser.add_argument("--chunk-length", type=int, default=default("chunk_length", 16))
    parser.add_argument("--chunk-stride", type=int, default=default("chunk_stride", 8))
    parser.add_argument("--eval-chunk-stride", type=int, default=default("eval_chunk_stride", 0))
    parser.add_argument("--burn-in", type=int, default=default("burn_in", 8))
    parser.add_argument("--include-tail", action="store_true", default=_as_bool(default("include_tail", True)))
    parser.add_argument("--drop-tail", dest="include_tail", action="store_false")
    parser.add_argument("--use-spatial-graph", action="store_true", default=_as_bool(default("use_spatial_graph", False)))
    parser.add_argument("--spatial-graph-neighbors", type=int, default=default("spatial_graph_neighbors", 8))
    parser.add_argument(
        "--spatial-relation-type",
        choices=[
            "topk_graph",
            "none",
            "full_attention",
            "local_grid",
            "strnet_edge_message",
            "relpos_topk_graph",
            "contrast_topk_graph",
            "hybrid_local_topk_graph",
            "relpos_contrast_topk_graph",
            "relpos_contrast_hybrid_graph",
        ],
        default=default("spatial_relation_type", None),
    )
    parser.add_argument("--use-temporal-difference-conv", action="store_true", default=_as_bool(default("use_temporal_difference_conv", False)))
    parser.add_argument("--use-temporal-shift", action="store_true", default=_as_bool(default("use_temporal_shift", False)))
    parser.add_argument("--decoder-layers", type=int, default=default("decoder_layers", 1))
    parser.add_argument(
        "--decoder-type",
        choices=["horizon_query_decoder", "single_vector_mlp", "shared_query_decoder", "autoregressive_decoder"],
        default=default("decoder_type", "horizon_query_decoder"),
    )
    parser.add_argument("--cue-temporal-layers", type=int, default=default("cue_temporal_layers", 1))
    parser.add_argument("--trajectory-scale", default=default("trajectory_scale", "auto"))
    parser.add_argument("--residual-scale", default=default("residual_scale", "auto"))
    parser.add_argument("--no-cv-residual", dest="use_constant_velocity_residual", action="store_false", default=_as_bool(default("use_constant_velocity_residual", True)))
    return normalize_args(parser.parse_args())


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("rgb_history", "visual_tokens", "ego_history", "future_path"):
        if key in moved:
            moved[key] = moved[key].to(device)
    return moved


def _loss_window_target(batch: dict[str, torch.Tensor], burn_in: int) -> torch.Tensor:
    return batch["future_path"][:, burn_in:, :, :]


def flatten_loss_windows(
    pred: torch.Tensor | Mapping[str, torch.Tensor],
    target: torch.Tensor,
    burn_in: int,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
    target_window = target[:, burn_in:, :, :]
    batch, windows, horizon, coords = target_window.shape
    flat_target = target_window.reshape(batch * windows, horizon, coords)
    if isinstance(pred, torch.Tensor):
        flat_pred = pred[:, burn_in:, :, :].reshape(batch * windows, horizon, coords)
        return flat_pred, flat_target
    paths = pred["paths"][:, burn_in:, :, :, :]
    modes = paths.shape[2]
    flat_paths = paths.reshape(batch * windows, modes, horizon, coords)
    flat_logits = pred["logits"][:, burn_in:, :].reshape(batch * windows, modes)
    return {"paths": flat_paths, "logits": flat_logits}, flat_target


def flatten_ego_windows(ego_history: torch.Tensor, burn_in: int) -> torch.Tensor:
    selected = ego_history[:, burn_in:, :, :]
    batch, windows, time, dim = selected.shape
    return selected.reshape(batch * windows, time, dim)


def episodic_batch_sample_weights(
    batch: dict[str, Any],
    key: str,
    weight_lookup: dict[str, float],
    burn_in: int,
    device: torch.device,
) -> torch.Tensor | None:
    if key == "none" or not weight_lookup:
        return None
    weights = []
    for chunk_balance in batch.get("balance", []):
        for item in chunk_balance[burn_in:]:
            weights.append(weight_lookup.get(item[key], 1.0))
    if not weights:
        return None
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loss_weight_lookup(samples: list[dict[str, Any]], key: str, exponent: float) -> dict[str, float]:
    if key == "none":
        return {}
    _weights, stats = compute_balance_weights(samples, key, exponent)
    return {
        group: float(values["weight"])
        for group, values in stats.get("groups", {}).items()
    }


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    burn_in: int,
    coordinate_scale: float = 1.0,
    multimodal_confidence_weight: float = 0.05,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_count = 0
    horizon_errors = []
    cv_horizon_errors = []
    model_errors_for_subset = []
    cv_errors_for_subset = []
    predictions = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        flat_pred, flat_target = flatten_loss_windows(pred, batch["future_path"], burn_in)
        loss = trajectory_loss(
            flat_pred,
            flat_target,
            loss_type,
            coordinate_scale=coordinate_scale,
            multimodal_confidence_weight=multimodal_confidence_weight,
        )
        selected_pred = select_best_trajectory(flat_pred, flat_target)
        model_errors = torch.linalg.norm(selected_pred - flat_target, dim=-1).detach().cpu()
        flat_ego = flatten_ego_windows(batch["ego_history"], burn_in)
        cv_pred = constant_velocity_path(flat_ego, flat_target.shape[1])
        cv_errors = torch.linalg.norm(cv_pred - flat_target, dim=-1).detach().cpu()
        count = flat_target.shape[0]
        total_loss += float(loss.detach().cpu()) * count
        total_ade += float(ade(flat_pred, flat_target).detach().cpu()) * count
        total_fde += float(fde(flat_pred, flat_target).detach().cpu()) * count
        horizon_errors.append(per_horizon_error(flat_pred, flat_target).detach().cpu() * count)
        cv_horizon_errors.append(cv_errors.sum(dim=0))
        model_errors_for_subset.append(model_errors)
        cv_errors_for_subset.append(cv_errors)
        total_count += count
        flat_index = 0
        for b, chunk_ids in enumerate(batch["sample_id"]):
            for local_t, sample_id in enumerate(chunk_ids[burn_in:], start=burn_in):
                item = {
                    "sample_id": sample_id,
                    "chunk_id": batch["chunk_id"][b],
                    "episode_id": batch["episode_id"][b],
                    "center_step": int(batch["center_step"][b, local_t].detach().cpu()),
                    "prediction": selected_pred[flat_index].detach().cpu().tolist(),
                    "target": flat_target[flat_index].detach().cpu().tolist(),
                    "ADE": float(model_errors[flat_index].mean()),
                    "FDE": float(model_errors[flat_index, -1]),
                    "constant_velocity_prediction": cv_pred[flat_index].detach().cpu().tolist(),
                    "constant_velocity_ADE": float(cv_errors[flat_index].mean()),
                    "constant_velocity_FDE": float(cv_errors[flat_index, -1]),
                }
                if isinstance(flat_pred, dict):
                    item["candidate_predictions"] = flat_pred["paths"][flat_index].detach().cpu().tolist()
                    item["mode_logits"] = flat_pred["logits"][flat_index].detach().cpu().tolist()
                predictions.append(item)
                flat_index += 1
    if total_count == 0:
        raise ValueError("Evaluation loader produced no loss windows")
    per_h = torch.stack(horizon_errors, dim=0).sum(dim=0) / total_count
    cv_per_h = torch.stack(cv_horizon_errors, dim=0).sum(dim=0) / total_count
    all_model_errors = torch.cat(model_errors_for_subset, dim=0)
    all_cv_errors = torch.cat(cv_errors_for_subset, dim=0)
    cv_ade_per_sample = all_cv_errors.mean(dim=1)
    hard_threshold = torch.quantile(cv_ade_per_sample, 0.75)
    hard_mask = cv_ade_per_sample >= hard_threshold
    hard_model_errors = all_model_errors[hard_mask]
    hard_cv_errors = all_cv_errors[hard_mask]
    metrics = {
        "loss": total_loss / total_count,
        "ADE": total_ade / total_count,
        "FDE": total_fde / total_count,
        "evaluated_windows": int(total_count),
        "burn_in": int(burn_in),
        "per_horizon_error": per_h.tolist(),
        "cv_baseline": {
            "ADE": float(all_cv_errors.mean()),
            "FDE": float(all_cv_errors[:, -1].mean()),
            "per_horizon_error": cv_per_h.tolist(),
        },
        "cv_hard_subset": {
            "definition": "top_25pct_by_constant_velocity_ADE_within_evaluated_windows",
            "cv_ade_quantile": 0.75,
            "cv_ade_threshold": float(hard_threshold),
            "samples": int(hard_mask.sum().item()),
            "ADE": float(hard_model_errors.mean()),
            "FDE": float(hard_model_errors[:, -1].mean()),
            "per_horizon_error": hard_model_errors.mean(dim=0).tolist(),
            "cv_ADE": float(hard_cv_errors.mean()),
            "cv_FDE": float(hard_cv_errors[:, -1].mean()),
        },
    }
    return metrics, predictions


def make_loader(args: argparse.Namespace, split: str, load_rgb: bool) -> DataLoader:
    stride = args.chunk_stride if split == "train" else args.eval_chunk_stride
    dataset = WITVZEpisodicChunkDataset(
        args.dataset,
        split=split,
        image_size=args.image_size,
        load_rgb=load_rgb,
        visual_feature_cache_dir=args.visual_feature_cache,
        history_frame_mode=args.history_frame_mode,
        frame_order=args.train_frame_order if split == "train" else "normal",
        chunk_length=args.chunk_length,
        chunk_stride=stride,
        include_tail=args.include_tail,
    )
    sampler = None
    shuffle = split == "train"
    if split == "train" and args.balance_key != "none" and args.balance_mode in {"sampler", "both"}:
        chunk_samples = [
            dataset.chunk_balance_sample(index, burn_in=args.burn_in)
            for index in range(len(dataset))
        ]
        weights, _stats = compute_balance_weights(chunk_samples, args.balance_key, args.balance_exponent)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_episodic_path_batch,
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
            "visual_feature_cache": args.visual_feature_cache.as_posix() if args.visual_feature_cache is not None else None,
            "hidden_dim": args.hidden_dim,
            "image_size": args.image_size,
            "future_steps": dataset_manifest["future_steps"],
            "history_frames": dataset_manifest["history_frames"],
            "history_frame_mode": args.history_frame_mode,
            "train_frame_order": args.train_frame_order,
            "freeze_backbone": args.freeze_backbone,
            "num_cue_tokens": args.num_cue_tokens,
            "num_modes": args.num_modes,
            "temporal_layers": args.temporal_layers,
            "selector_layers": args.selector_layers,
            "decoder_layers": args.decoder_layers,
            "decoder_type": args.decoder_type,
            "cue_temporal_layers": args.cue_temporal_layers,
            "tokenlearner_pooling": args.tokenlearner_pooling,
            "selector_type": args.selector_type,
            "memory_type": args.memory_type,
            "long_memory_type": args.long_memory_type,
            "long_memory_slots": args.long_memory_slots,
            "long_memory_use_ego": args.long_memory_use_ego,
            "detach_long_memory": args.detach_long_memory,
            "chunk_length": args.chunk_length,
            "chunk_stride": args.chunk_stride,
            "eval_chunk_stride": args.eval_chunk_stride,
            "burn_in": args.burn_in,
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
            "trajectory_scale": float(getattr(args, "resolved_trajectory_scale", 1.0)),
            "residual_scale": float(getattr(args, "resolved_residual_scale", 1.0)),
            "use_constant_velocity_residual": bool(args.use_constant_velocity_residual),
            "data_parallel": bool(args.data_parallel),
            "loss": args.loss,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.burn_in >= args.chunk_length:
        raise ValueError("burn_in must be smaller than chunk_length")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    dataset_manifest = json.loads((args.dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    load_rgb = needs_rgb(args.model, args.backbone) and args.visual_feature_cache is None

    train_loader = make_loader(args, "train", load_rgb)
    val_loader = make_loader(args, "val", load_rgb)
    test_loader = make_loader(args, "test", load_rgb)
    trajectory_scale = resolve_scale(args.trajectory_scale, train_loader.dataset.base)
    residual_scale = resolve_scale(args.residual_scale, None, fallback=trajectory_scale)
    args.resolved_trajectory_scale = trajectory_scale
    args.resolved_residual_scale = residual_scale

    balance_samples = [
        train_loader.dataset.chunk_balance_sample(index, burn_in=args.burn_in)
        for index in range(len(train_loader.dataset))
    ]
    balance_loss_weights = make_loss_weight_lookup(balance_samples, args.balance_key, args.balance_exponent)
    _balance_sampler_weights, balance_stats = compute_balance_weights(
        balance_samples,
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
        decoder_type=args.decoder_type,
        cue_temporal_layers=args.cue_temporal_layers,
        tokenlearner_pooling=args.tokenlearner_pooling,
        selector_type=args.selector_type,
        memory_type=args.memory_type,
        long_memory_type=args.long_memory_type,
        long_memory_slots=args.long_memory_slots,
        long_memory_use_ego=args.long_memory_use_ego,
        detach_long_memory=args.detach_long_memory,
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
    for key in ("dataset", "visual_feature_cache", "output_dir", "config"):
        value = config.get(key)
        if isinstance(value, Path):
            config[key] = value.as_posix()
    config["device"] = str(device)
    config["resolved_trajectory_scale"] = trajectory_scale
    config["resolved_residual_scale"] = residual_scale
    config["balance_stats"] = balance_stats
    config["train_chunks"] = len(train_loader.dataset)
    config["val_chunks"] = len(val_loader.dataset)
    config["test_chunks"] = len(test_loader.dataset)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

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
                sample_weight = episodic_batch_sample_weights(
                    batch,
                    args.balance_key,
                    balance_loss_weights,
                    args.burn_in,
                    device,
                )
            with torch.cuda.amp.autocast(**autocast_kwargs(device, args.mixed_precision)):
                pred = model(batch)
                flat_pred, flat_target = flatten_loss_windows(pred, batch["future_path"], args.burn_in)
                loss = trajectory_loss(
                    flat_pred,
                    flat_target,
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
            count = flat_target.shape[0]
            total_loss += float(loss.detach().cpu()) * count
            total_ade += float(ade(flat_pred, flat_target).detach().cpu()) * count
            total_count += count

        train_loss = total_loss / max(total_count, 1)
        train_ade = total_ade / max(total_count, 1)
        val_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            args.loss,
            args.burn_in,
            trajectory_scale,
            args.multimodal_confidence_weight,
        )
        if scheduler is not None:
            scheduler.step(val_metrics["ADE"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ADE": train_ade,
            "val_train_ADE_gap": val_metrics["ADE"] - train_ade,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val": val_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if device.type == "cuda":
            row["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_ADE={train_ade:.4f} val_ADE={val_metrics['ADE']:.4f} lr={row['lr']:.2e}"
        )
        if val_metrics["ADE"] < best_val - args.early_stopping_min_delta:
            best_val = val_metrics["ADE"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(args.output_dir / "best.pt", unwrap_model(model), args, dataset_manifest, row)
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping=epoch={epoch} best_epoch={best_epoch} best_val_ADE={best_val:.4f}")
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    val_metrics, _ = evaluate_loader(
        model,
        val_loader,
        device,
        args.loss,
        args.burn_in,
        trajectory_scale,
        args.multimodal_confidence_weight,
    )
    test_metrics, predictions = evaluate_loader(
        model,
        test_loader,
        device,
        args.loss,
        args.burn_in,
        trajectory_scale,
        args.multimodal_confidence_weight,
    )
    metrics = {
        "model": args.model,
        "best_epoch": best_epoch,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_checkpoint(args.output_dir / "best.pt", unwrap_model(model), args, dataset_manifest, metrics)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
