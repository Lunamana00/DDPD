"""Build and train WIT-VZ path predictors over multiple future horizons."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run path-prediction horizon sweep.")
    parser.add_argument("--raw", type=Path, default=Path("data/wit_vz/raw/wit_vz_basic_10s"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/wit_vz/processed/horizon_sweep"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/horizon_sweep"))
    parser.add_argument("--min-sec", type=int, default=1)
    parser.add_argument("--max-sec", type=int, default=10)
    parser.add_argument("--history-sec", type=float, default=1.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.processed_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon_sec in range(args.min_sec, args.max_sec + 1):
        dataset_dir = args.processed_root / f"future_{horizon_sec:02d}s"
        if not args.summarize_only:
            run(
                [
                    sys.executable,
                    "-m",
                    "src.wit_vz.build_samples",
                    "--raw",
                    str(args.raw),
                    "--out",
                    str(dataset_dir),
                    "--history-sec",
                    str(args.history_sec),
                    "--future-sec",
                    str(float(horizon_sec)),
                    "--sample-fps",
                    str(args.sample_fps),
                    "--stride",
                    str(args.stride),
                    "--seed",
                    str(args.seed),
                    "--preview-count",
                    "0",
                ]
            )
        manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        if int(manifest["num_samples"]) == 0:
            rows.append({"horizon_sec": horizon_sec, "num_samples": 0, "skipped": True})
            continue

        cv_dir = args.runs_root / f"constant_velocity_{horizon_sec:02d}s"
        if not args.summarize_only:
            run(
                [
                    sys.executable,
                    "-m",
                    "src.train_path_predictor",
                    "--dataset",
                    str(dataset_dir),
                    "--model",
                    "constant_velocity",
                    "--epochs",
                    "1",
                    "--batch-size",
                    str(args.batch_size),
                    "--output-dir",
                    str(cv_dir),
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--trajectory-scale",
                    "auto",
                ]
            )
        cv_metrics = json.loads((cv_dir / "metrics.json").read_text(encoding="utf-8"))["test"]
        row = {
            "horizon_sec": horizon_sec,
            "num_samples": int(manifest["num_samples"]),
            "future_steps": int(manifest["future_steps"]),
            "cv_ADE": float(cv_metrics["ADE"]),
            "cv_FDE": float(cv_metrics["FDE"]),
        }

        if not args.skip_model:
            model_dir = args.runs_root / f"cue_memory_residual_{horizon_sec:02d}s"
            if not args.summarize_only:
                run(
                    [
                        sys.executable,
                        "-m",
                        "src.train_path_predictor",
                        "--dataset",
                        str(dataset_dir),
                        "--model",
                        "cue_memory_path_predictor",
                        "--backbone",
                        "small_cnn",
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--lr",
                        "0.001",
                        "--weight-decay",
                        "0.0001",
                        "--hidden-dim",
                        str(args.hidden_dim),
                        "--image-size",
                        "64",
                        "--loss",
                        "huber",
                        "--output-dir",
                        str(model_dir),
                        "--seed",
                        str(args.seed),
                        "--device",
                        args.device,
                        "--num-cue-tokens",
                        "4",
                        "--temporal-type",
                        "gru",
                        "--train-backbone",
                        "--trajectory-scale",
                        "auto",
                        "--residual-scale",
                        "auto",
                    ]
                )
            model_metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["test"]
            row.update(
                {
                    "model_ADE": float(model_metrics["ADE"]),
                    "model_FDE": float(model_metrics["FDE"]),
                }
            )
        rows.append(row)

    summary_path = args.runs_root / "horizon_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    markdown = [
        "# Horizon Sweep",
        "",
        "| Horizon | Samples | Steps | CV ADE | CV FDE | Model ADE | Model FDE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        markdown.append(
            "| {horizon}s | {samples} | {steps} | {cv_ade:.4f} | {cv_fde:.4f} | {model_ade:.4f} | {model_fde:.4f} |".format(
                horizon=row["horizon_sec"],
                samples=row["num_samples"],
                steps=row.get("future_steps", 0),
                cv_ade=row.get("cv_ADE", float("nan")),
                cv_fde=row.get("cv_FDE", float("nan")),
                model_ade=row.get("model_ADE", float("nan")),
                model_fde=row.get("model_FDE", float("nan")),
            )
        )
    report_path = args.runs_root / "horizon_summary.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
