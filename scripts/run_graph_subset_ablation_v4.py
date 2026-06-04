"""Queue graph spatial-relation ablations across available GPUs."""

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


VARIANTS = {
    "no_graph": "none",
    "topk_graph": "topk_graph",
    "relpos_graph": "relpos_topk_graph",
    "contrast_graph": "contrast_topk_graph",
    "local_topk_graph": "hybrid_local_topk_graph",
    "relpos_contrast_local_graph": "relpos_contrast_hybrid_graph",
}


@dataclass(frozen=True)
class Task:
    horizon: str
    variant: str
    spatial_relation_type: str
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
    parser.add_argument("--run-root", type=Path, default=Path("runs/graph_subset_ablation_v4_10s"))
    parser.add_argument("--config-root", type=Path, default=Path("configs/graph_subset_ablation_v4_10s"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/graph_subset_ablation_v4_10s"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/graph_subset_ablation_v4_10s/results.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/graph_subset_ablation_v4_10s.md"))
    parser.add_argument("--horizons", nargs="+", default=["10s"])
    parser.add_argument("--prefixes", nargs="+", default=["01s", "03s", "05s", "10s"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--max-gpu-memory-mb", type=int, default=12000)
    parser.add_argument("--max-gpu-util-percent", type=int, default=95)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
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


def common_config(args: argparse.Namespace, task: Task) -> dict[str, Any]:
    dataset = args.processed_root / f"future_{task.horizon}"
    return {
        "dataset": dataset.as_posix(),
        "visual_feature_cache": args.visual_feature_cache.as_posix(),
        "model": "cue_memory_path_predictor",
        "backbone": "cached_dinov3_convnext_tiny",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": 0.0005,
        "weight_decay": 0.001,
        "dropout": 0.2,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": args.epochs,
        "lr_scheduler_patience": 25,
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
        "spatial_relation_type": task.spatial_relation_type,
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


def make_tasks(args: argparse.Namespace) -> list[Task]:
    tasks = []
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    for horizon in args.horizons:
        for variant in args.variants:
            output_dir = args.run_root / f"seed_{args.seed}" / horizon / variant
            config_path = args.config_root / f"train_graph_subset_{horizon}_{variant}.yaml"
            log_path = args.log_root / f"{horizon}_{variant}.log"
            tasks.append(
                Task(
                    horizon=horizon,
                    variant=variant,
                    spatial_relation_type=VARIANTS[variant],
                    output_dir=output_dir,
                    config_path=config_path,
                    log_path=log_path,
                )
            )
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
    command = [sys.executable, "-m", "src.train_path_predictor", "--config", task.config_path.as_posix()]
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
    if len(args.horizons) != 1:
        raise ValueError("Prefix summarization expects exactly one train horizon, e.g. --horizons 10s")
    command = [
        sys.executable,
        "scripts/summarize_graph_subset_ablation_v4.py",
        "--run-root",
        args.run_root.as_posix(),
        "--output-json",
        args.output_json.as_posix(),
        "--report",
        args.report.as_posix(),
        "--seed",
        str(args.seed),
        "--train-horizon",
        args.horizons[0],
        "--prefixes",
        *args.prefixes,
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
            print(f"DRYRUN {task.horizon} {task.variant} {task.config_path}")
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
