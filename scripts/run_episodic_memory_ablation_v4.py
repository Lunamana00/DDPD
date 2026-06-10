"""Queue episodic long-memory ablations on available GPUs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


VARIANTS: dict[str, dict[str, Any]] = {
    "current_short_window": {
        "trainer": "single",
        "model": "cue_memory_path_predictor",
    },
    "episodic_short_only": {
        "trainer": "episodic",
        "model": "episodic_long_term_cue_memory_path_predictor",
        "long_memory_type": "none",
    },
    "long_mean_memory": {
        "trainer": "episodic",
        "model": "episodic_long_term_cue_memory_path_predictor",
        "long_memory_type": "mean",
    },
    "long_attention_no_ego": {
        "trainer": "episodic",
        "model": "episodic_long_term_cue_memory_path_predictor",
        "long_memory_type": "attention",
        "long_memory_use_ego": False,
    },
    "long_attention_ego": {
        "trainer": "episodic",
        "model": "episodic_long_term_cue_memory_path_predictor",
        "long_memory_type": "attention",
        "long_memory_use_ego": True,
    },
    "long_gated_ego": {
        "trainer": "episodic",
        "model": "episodic_long_term_cue_memory_path_predictor",
        "long_memory_type": "gated_attention",
        "long_memory_use_ego": True,
    },
    "long_gated_forget_ego": {
        "trainer": "episodic",
        "model": "episodic_long_term_cue_memory_path_predictor",
        "long_memory_type": "gated_forget",
        "long_memory_use_ego": True,
    },
}


@dataclass(frozen=True)
class Task:
    horizon: str
    variant: str
    trainer: str
    output_dir: Path
    config_path: Path
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/wit_vz/processed/horizon_sweep_v4_defaults"))
    parser.add_argument(
        "--visual-feature-cache",
        type=Path,
        default=Path("data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs/episodic_memory_ablation_v4"))
    parser.add_argument("--config-root", type=Path, default=Path("configs/episodic_memory_ablation_v4"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/episodic_memory_ablation_v4"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/episodic_memory_ablation_v4/results.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/episodic_memory_ablation_v4.md"))
    parser.add_argument("--horizons", nargs="+", default=["01s", "03s", "05s", "10s"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--max-gpu-memory-mb", type=int, default=2000)
    parser.add_argument("--max-gpu-util-percent", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--lr-scheduler-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--single-batch-size", type=int, default=512)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--chunk-stride", type=int, default=8)
    parser.add_argument("--eval-chunk-stride", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--long-memory-slots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--reset-run-dirs", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    return parser.parse_args()


def write_flat_config(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "none"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def base_config(args: argparse.Namespace, task: Task) -> dict[str, Any]:
    dataset = args.processed_root / f"future_{task.horizon}"
    return {
        "dataset": dataset.as_posix(),
        "visual_feature_cache": args.visual_feature_cache.as_posix(),
        "model": VARIANTS[task.variant]["model"],
        "backbone": "cached_dinov3_convnext_tiny",
        "epochs": args.epochs,
        "batch_size": args.single_batch_size if task.trainer == "single" else args.batch_size,
        "lr": 0.0005,
        "weight_decay": 0.001,
        "dropout": 0.2,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": 0.0,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": 0.5,
        "min_lr": 0.000001,
        "hidden_dim": 128,
        "image_size": 256,
        "output_dir": task.output_dir.as_posix(),
        "seed": args.seed,
        "device": "auto",
        "num_workers": args.num_workers,
        "freeze_backbone": True,
        "mixed_precision": True,
        "balance_key": "source_policy",
        "balance_mode": "both",
        "balance_exponent": 1.0,
        "loss": "huber",
        "num_cue_tokens": 8,
        "num_modes": 1,
        "temporal_type": "timesformer",
        "temporal_layers": 1,
        "selector_type": "tokenlearner",
        "selector_layers": 1,
        "tokenlearner_pooling": "sigmoid",
        "memory_type": "attention",
        "spatial_graph_neighbors": 8,
        "spatial_relation_type": "relpos_contrast_hybrid_graph",
        "use_temporal_difference_conv": False,
        "use_temporal_shift": False,
        "decoder_layers": 1,
        "decoder_type": "horizon_query_decoder",
        "cue_temporal_layers": 1,
        "trajectory_scale": "auto",
        "residual_scale": "auto",
        "use_constant_velocity_residual": True,
        "multimodal_confidence_weight": 0.05,
        "history_frame_mode": "full",
        "train_frame_order": "normal",
    }


def common_config(args: argparse.Namespace, task: Task) -> dict[str, Any]:
    config = base_config(args, task)
    variant = VARIANTS[task.variant]
    if task.trainer == "episodic":
        config.update(
            {
                "chunk_length": args.chunk_length,
                "chunk_stride": args.chunk_stride,
                "eval_chunk_stride": args.eval_chunk_stride,
                "burn_in": args.burn_in,
                "include_tail": True,
                "long_memory_type": variant.get("long_memory_type", "gated_attention"),
                "long_memory_slots": args.long_memory_slots,
                "long_memory_use_ego": variant.get("long_memory_use_ego", True),
                "detach_long_memory": True,
            }
        )
    return config


def make_tasks(args: argparse.Namespace) -> list[Task]:
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    tasks = []
    for horizon in args.horizons:
        for variant in args.variants:
            trainer = str(VARIANTS[variant]["trainer"])
            output_dir = args.run_root / f"seed_{args.seed}" / horizon / variant
            config_path = args.config_root / f"train_episodic_memory_{horizon}_{variant}.yaml"
            log_path = args.log_root / f"{horizon}_{variant}.log"
            tasks.append(Task(horizon, variant, trainer, output_dir, config_path, log_path))
    return tasks


def run_complete(task: Task) -> bool:
    return (
        (task.output_dir / "config.json").exists()
        and (task.output_dir / "best.pt").exists()
        and (task.output_dir / "metrics.json").exists()
        and (task.output_dir / "predictions.jsonl").exists()
    )


def gpu_stats() -> dict[str, dict[str, int]]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(query, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    usage = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        index, memory, utilization = [part.strip() for part in line.split(",", 2)]
        usage[index] = {
            "memory_mb": int(memory),
            "utilization_pct": int(utilization),
        }
    return usage


def launch(task: Task, gpu: str, args: argparse.Namespace) -> subprocess.Popen:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(args.torch_threads)
    env["MKL_NUM_THREADS"] = str(args.torch_threads)
    module = "src.train_path_predictor" if task.trainer == "single" else "src.train_episodic_path_predictor"
    command = [sys.executable, "-m", module, "--config", task.config_path.as_posix()]
    log_handle = task.log_path.open("a", encoding="utf-8")
    log_handle.write(f"\n[launch] gpu={gpu} command={' '.join(command)}\n")
    log_handle.flush()
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    process._ddpd_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def close_process_log(process: subprocess.Popen) -> None:
    handle = getattr(process, "_ddpd_log_handle", None)
    if handle is not None:
        handle.close()


def summarize(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/summarize_episodic_memory_ablation_v4.py",
        "--run-root",
        args.run_root.as_posix(),
        "--output-json",
        args.output_json.as_posix(),
        "--report",
        args.report.as_posix(),
        "--seed",
        str(args.seed),
        "--horizons",
        *args.horizons,
        "--variants",
        *args.variants,
    ]
    subprocess.check_call(command)


def main() -> None:
    args = parse_args()
    tasks = make_tasks(args)
    for task in tasks:
        if args.reset_run_dirs and task.output_dir.exists():
            shutil.rmtree(task.output_dir)
        write_flat_config(task.config_path, common_config(args, task))

    pending = [task for task in tasks if not (args.skip_existing and run_complete(task))]
    print(
        json.dumps(
            {
                "total_tasks": len(tasks),
                "pending_tasks": len(pending),
                "gpus": args.gpus,
                "run_root": args.run_root.as_posix(),
            },
            indent=2,
        )
    )
    if args.dry_run:
        for task in pending:
            print(f"DRYRUN {task.horizon} {task.variant} trainer={task.trainer} config={task.config_path}")
        return

    running: dict[str, tuple[Task, subprocess.Popen]] = {}
    failures: list[tuple[Task, int]] = []
    while pending or running:
        for gpu, (task, process) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            close_process_log(process)
            del running[gpu]
            if code != 0:
                failures.append((task, code))
                print(f"[failed] gpu={gpu} horizon={task.horizon} variant={task.variant} code={code}")
            else:
                print(f"[done] gpu={gpu} horizon={task.horizon} variant={task.variant}")

        if failures:
            failed_task, code = failures[0]
            raise SystemExit(
                f"Task failed: horizon={failed_task.horizon} variant={failed_task.variant} "
                f"code={code} log={failed_task.log_path}"
            )

        usage = gpu_stats()
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            if usage:
                stats = usage.get(gpu, {"memory_mb": 10**9, "utilization_pct": 100})
                if stats["memory_mb"] > args.max_gpu_memory_mb:
                    continue
                if stats["utilization_pct"] > args.max_gpu_util_percent:
                    continue
            task = pending.pop(0)
            process = launch(task, gpu, args)
            running[gpu] = (task, process)
            print(f"[started] gpu={gpu} horizon={task.horizon} variant={task.variant}")

        if pending or running:
            status = {
                "pending": len(pending),
                "running": {
                    gpu: {"horizon": task.horizon, "variant": task.variant, "pid": process.pid}
                    for gpu, (task, process) in running.items()
                },
                "gpu_stats": usage,
            }
            print(json.dumps(status, indent=2))
            time.sleep(max(args.poll_seconds, 5))

    if not args.no_summarize:
        summarize(args)


if __name__ == "__main__":
    main()
