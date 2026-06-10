"""Summarize episodic memory ablation runs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_VARIANTS = (
    "current_short_window",
    "episodic_short_only",
    "long_mean_memory",
    "long_attention_no_ego",
    "long_attention_ego",
    "long_gated_ego",
    "long_gated_forget_ego",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/episodic_memory_ablation_v4"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/episodic_memory_ablation_v4/results.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/episodic_memory_ablation_v4.md"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizons", nargs="+", default=["01s", "03s", "05s", "10s"])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def displacement_errors(prediction: list[list[float]], target: list[list[float]]) -> list[float]:
    return [
        math.hypot(float(pred[0]) - float(gt[0]), float(pred[1]) - float(gt[1]))
        for pred, gt in zip(prediction, target, strict=True)
    ]


def path_features(target: list[list[float]]) -> dict[str, float]:
    points = [(0.0, 0.0)] + [(float(x), float(y)) for x, y in target]
    vectors = [
        (points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        for index in range(len(points) - 1)
    ]
    headings = [math.atan2(right, forward) for forward, right in vectors if math.hypot(forward, right) > 1.0e-6]
    heading_changes = []
    for previous, current in zip(headings, headings[1:], strict=False):
        delta = (current - previous + math.pi) % (2 * math.pi) - math.pi
        heading_changes.append(abs(delta))
    max_abs_right = max(abs(point[1]) for point in points[1:]) if len(points) > 1 else 0.0
    mean_abs_right = mean(abs(point[1]) for point in points[1:]) if len(points) > 1 else 0.0
    return {
        "curvature": float(sum(heading_changes)),
        "max_abs_right": float(max_abs_right),
        "mean_abs_right": float(mean_abs_right),
        "final_abs_right": float(abs(points[-1][1] if points else 0.0)),
        "final_forward": float(points[-1][0] if points else 0.0),
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def mean_or_none(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def prediction_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in predictions:
        prediction = item["prediction"]
        target = item["target"]
        errors = displacement_errors(prediction, target)
        cv_prediction = item.get("constant_velocity_prediction")
        if cv_prediction is None:
            cv_ade = float(item.get("constant_velocity_ADE", 0.0))
            cv_fde = float(item.get("constant_velocity_FDE", 0.0))
        else:
            cv_errors = displacement_errors(cv_prediction, target)
            cv_ade = float(sum(cv_errors) / len(cv_errors))
            cv_fde = float(cv_errors[-1])
        rows.append(
            {
                "sample_id": item["sample_id"],
                "episode_id": item.get("episode_id"),
                "ADE": float(sum(errors) / len(errors)),
                "FDE": float(errors[-1]),
                "cv_ADE": cv_ade,
                "cv_FDE": cv_fde,
                "features": path_features(target),
            }
        )
    return rows


def subset_masks(rows: list[dict[str, Any]]) -> dict[str, list[bool]]:
    curvature = [row["features"]["curvature"] for row in rows]
    max_abs_right = [row["features"]["max_abs_right"] for row in rows]
    mean_abs_right = [row["features"]["mean_abs_right"] for row in rows]
    cv_ade = [row["cv_ADE"] for row in rows]
    curvature_high = percentile(curvature, 0.75)
    curvature_low = percentile(curvature, 0.25)
    max_right_high = percentile(max_abs_right, 0.75)
    mean_right_high = percentile(mean_abs_right, 0.75)
    cv_high = percentile(cv_ade, 0.75)
    return {
        "all": [True for _row in rows],
        "high_curvature_path": [value >= curvature_high for value in curvature],
        "turn_scene": [
            row["features"]["max_abs_right"] >= max_right_high
            or row["features"]["curvature"] >= curvature_high
            for row in rows
        ],
        "cv_baseline_error_high": [value >= cv_high for value in cv_ade],
        "left_right_asymmetric_layout": [
            row["features"]["mean_abs_right"] >= mean_right_high
            or row["features"]["max_abs_right"] >= max_right_high
            for row in rows
        ],
        "corridor_like_scene": [
            row["features"]["curvature"] <= curvature_low
            and row["features"]["max_abs_right"] < max_right_high
            for row in rows
        ],
        "front_blocked_or_obstacle_proxy": [
            row["cv_ADE"] >= cv_high and row["features"]["curvature"] >= curvature_high
            for row in rows
        ],
    }


def subset_metrics(rows: list[dict[str, Any]], masks: dict[str, list[bool]]) -> dict[str, dict[str, Any]]:
    output = {}
    for name, mask in masks.items():
        selected = [row for row, keep in zip(rows, mask, strict=True) if keep]
        output[name] = {
            "samples": len(selected),
            "ADE": mean_or_none(row["ADE"] for row in selected),
            "FDE": mean_or_none(row["FDE"] for row in selected),
            "cv_ADE": mean_or_none(row["cv_ADE"] for row in selected),
            "cv_FDE": mean_or_none(row["cv_FDE"] for row in selected),
        }
    return output


def training_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    history = metrics.get("history", [])
    if not history:
        return {}
    best = min(history, key=lambda row: row.get("val", {}).get("ADE", float("inf")))
    return {
        "best_epoch": int(metrics.get("best_epoch") or best.get("epoch", 0)),
        "best_val_ADE": best.get("val", {}).get("ADE"),
        "train_ADE_at_best": best.get("train_ADE"),
        "val_train_ADE_gap_at_best": best.get("val_train_ADE_gap"),
        "mean_epoch_seconds": mean_or_none(float(row.get("epoch_seconds", 0.0)) for row in history),
        "peak_cuda_memory_mb": max(
            (float(row.get("cuda_peak_memory_mb", 0.0)) for row in history),
            default=None,
        ),
        "epochs_ran": len(history),
    }


def collect_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.jsonl"
    best_path = run_dir / "best.pt"
    config_path = run_dir / "config.json"
    missing = [
        path.name
        for path in (metrics_path, predictions_path, best_path, config_path)
        if not path.exists()
    ]
    if missing:
        return {"complete": False, "missing": missing}
    metrics = load_json(metrics_path)
    predictions = load_predictions(predictions_path)
    config = load_json(config_path)
    rows = prediction_rows(predictions)
    masks = subset_masks(rows)
    return {
        "complete": True,
        "config": config,
        "val": metrics.get("val", {}),
        "test": metrics.get("test", {}),
        "training": training_summary(metrics),
        "overall": {
            "samples": len(rows),
            "ADE": mean_or_none(row["ADE"] for row in rows),
            "FDE": mean_or_none(row["FDE"] for row in rows),
            "cv_baseline": {
                "ADE": mean_or_none(row["cv_ADE"] for row in rows),
                "FDE": mean_or_none(row["cv_FDE"] for row in rows),
            },
        },
        "subsets": subset_metrics(rows, masks),
    }


def improvement(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return (baseline - value) / baseline * 100.0


def add_relative_improvements(horizon_results: dict[str, Any]) -> None:
    current = horizon_results.get("current_short_window", {})
    short = horizon_results.get("episodic_short_only", {})
    for _variant, payload in horizon_results.items():
        if not payload.get("complete"):
            continue
        payload["overall"]["ADE_improvement_vs_current_short_pct"] = improvement(
            payload["overall"].get("ADE"),
            current.get("overall", {}).get("ADE"),
        )
        payload["overall"]["ADE_improvement_vs_episodic_short_pct"] = improvement(
            payload["overall"].get("ADE"),
            short.get("overall", {}).get("ADE"),
        )
        for subset_name, subset in payload.get("subsets", {}).items():
            subset["ADE_improvement_vs_current_short_pct"] = improvement(
                subset.get("ADE"),
                current.get("subsets", {}).get(subset_name, {}).get("ADE"),
            )
            subset["ADE_improvement_vs_episodic_short_pct"] = improvement(
                subset.get("ADE"),
                short.get("subsets", {}).get(subset_name, {}).get("ADE"),
            )


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(results: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Episodic Long-Term Cue Memory Ablation",
        "",
        "All trainable variants are retrained from scratch under the same v4 DINO-cache setup.",
        "The main controls are `current_short_window` and `episodic_short_only`; improvements over the latter isolate the long-memory update beyond chunked training itself.",
        "",
    ]
    for horizon, horizon_results in results["horizons"].items():
        lines.extend([f"## Horizon {horizon}", ""])
        lines.append("| Variant | Complete | ADE | FDE | CV ADE | Best epoch | Gap | ADE vs current | ADE vs episodic short |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for variant, payload in horizon_results.items():
            if not payload.get("complete"):
                lines.append(f"| {variant} | no | - | - | - | - | - | - | - |")
                continue
            overall = payload["overall"]
            training = payload.get("training", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        "yes",
                        format_float(overall.get("ADE")),
                        format_float(overall.get("FDE")),
                        format_float(overall.get("cv_baseline", {}).get("ADE")),
                        str(training.get("best_epoch", "-")),
                        format_float(training.get("val_train_ADE_gap_at_best")),
                        format_float(overall.get("ADE_improvement_vs_current_short_pct"), 2),
                        format_float(overall.get("ADE_improvement_vs_episodic_short_pct"), 2),
                    ]
                )
                + " |"
            )
        lines.append("")
        subset_names = [
            "cv_baseline_error_high",
            "high_curvature_path",
            "turn_scene",
            "front_blocked_or_obstacle_proxy",
            "corridor_like_scene",
        ]
        for subset_name in subset_names:
            lines.extend([f"### {subset_name}", ""])
            lines.append("| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for variant, payload in horizon_results.items():
                if not payload.get("complete"):
                    continue
                subset = payload.get("subsets", {}).get(subset_name, {})
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            variant,
                            str(subset.get("samples", "-")),
                            format_float(subset.get("ADE")),
                            format_float(subset.get("FDE")),
                            format_float(subset.get("cv_ADE")),
                            format_float(subset.get("ADE_improvement_vs_episodic_short_pct"), 2),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results: dict[str, Any] = {
        "seed": args.seed,
        "run_root": args.run_root.as_posix(),
        "horizons": {},
    }
    for horizon in args.horizons:
        horizon_results = {}
        for variant in args.variants:
            run_dir = args.run_root / f"seed_{args.seed}" / horizon / variant
            horizon_results[variant] = collect_run(run_dir)
        add_relative_improvements(horizon_results)
        results["horizons"][horizon] = horizon_results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, args.report)
    print(json.dumps({"output_json": args.output_json.as_posix(), "report": args.report.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
