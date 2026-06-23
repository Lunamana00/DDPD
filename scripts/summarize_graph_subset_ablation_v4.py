"""Summarize graph spatial-relation ablations with 10s prefix metrics."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_VARIANTS = (
    "no_graph",
    "topk_graph",
    "relpos_graph",
    "contrast_graph",
    "local_topk_graph",
    "relpos_contrast_local_graph",
)

DEFAULT_PREFIX_STEPS = {
    "01s": 5,
    "03s": 15,
    "05s": 25,
    "10s": 50,
}


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


def trim_path(path: list[list[float]], steps: int) -> list[list[float]]:
    if len(path) < steps:
        raise ValueError(f"Path has {len(path)} steps, but {steps} were requested")
    return path[:steps]


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
    curvature = sum(heading_changes)
    final_forward, final_right = points[-1]
    max_abs_right = max(abs(point[1]) for point in points[1:]) if len(points) > 1 else 0.0
    mean_abs_right = mean(abs(point[1]) for point in points[1:]) if len(points) > 1 else 0.0
    return {
        "curvature": curvature,
        "final_abs_right": abs(final_right),
        "max_abs_right": max_abs_right,
        "mean_abs_right": mean_abs_right,
        "final_forward": final_forward,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def subset_masks(rows: list[dict[str, Any]]) -> dict[str, list[bool]]:
    curvature = [row["features"]["curvature"] for row in rows]
    final_abs_right = [row["features"]["final_abs_right"] for row in rows]
    max_abs_right = [row["features"]["max_abs_right"] for row in rows]
    mean_abs_right = [row["features"]["mean_abs_right"] for row in rows]
    cv_ade = [row["cv_ADE"] for row in rows]

    curvature_high = percentile(curvature, 0.75)
    curvature_low = percentile(curvature, 0.25)
    final_right_high = percentile(final_abs_right, 0.75)
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
            row["features"]["final_abs_right"] >= final_right_high
            or row["features"]["mean_abs_right"] >= mean_right_high
            for row in rows
        ],
        "corridor_like_scene": [
            row["features"]["curvature"] <= curvature_low
            and row["features"]["final_abs_right"] <= final_right_high
            for row in rows
        ],
        "front_blocked_or_obstacle_proxy": [
            row["cv_ADE"] >= cv_high and row["features"]["curvature"] >= curvature_high
            for row in rows
        ],
    }


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


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


def overall_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "ADE": mean_or_none(row["ADE"] for row in rows),
        "FDE": mean_or_none(row["FDE"] for row in rows),
        "cv_baseline": {
            "ADE": mean_or_none(row["cv_ADE"] for row in rows),
            "FDE": mean_or_none(row["cv_FDE"] for row in rows),
        },
    }


def training_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    history = metrics.get("history", [])
    if not history:
        return {}
    best = min(history, key=lambda row: row.get("val", {}).get("ADE", float("inf")))
    return {
        "best_epoch": int(best.get("epoch", 0)),
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


def prediction_rows_for_prefix(predictions: list[dict[str, Any]], steps: int) -> list[dict[str, Any]]:
    rows = []
    for item in predictions:
        prediction = trim_path(item["prediction"], steps)
        target = trim_path(item["target"], steps)
        errors = displacement_errors(prediction, target)
        cv_prediction = item.get("constant_velocity_prediction")
        if cv_prediction is None:
            cv_ade = float(item.get("constant_velocity_ADE", 0.0))
            cv_fde = float(item.get("constant_velocity_FDE", 0.0))
        else:
            cv_errors = displacement_errors(trim_path(cv_prediction, steps), target)
            cv_ade = float(sum(cv_errors) / len(cv_errors))
            cv_fde = float(cv_errors[-1])
        rows.append(
            {
                "sample_id": item["sample_id"],
                "ADE": float(sum(errors) / len(errors)),
                "FDE": float(errors[-1]),
                "cv_ADE": cv_ade,
                "cv_FDE": cv_fde,
                "features": path_features(target),
            }
        )
    return rows


def collect_run(run_dir: Path, prefix_steps: dict[str, int]) -> dict[str, Any]:
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
    training = training_summary(metrics)
    prefixes = {}
    for prefix_name, steps in prefix_steps.items():
        rows = prediction_rows_for_prefix(predictions, steps)
        masks = subset_masks(rows)
        prefixes[prefix_name] = {
            "complete": True,
            "config": config,
            "overall": overall_metrics(rows),
            "val": metrics.get("val", {}),
            "training": training,
            "subsets": subset_metrics(rows, masks),
        }
    return {
        "complete": True,
        "config": config,
        "train_horizon_overall": metrics.get("test", {}),
        "val": metrics.get("val", {}),
        "training": training,
        "prefixes": prefixes,
    }


def improvement(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return (baseline - value) / baseline * 100.0


def add_relative_improvements(horizon_results: dict[str, Any]) -> None:
    no_graph = horizon_results.get("no_graph", {})
    topk = horizon_results.get("topk_graph", {})
    for _variant, payload in horizon_results.items():
        if not payload.get("complete"):
            continue
        payload["overall"]["ADE_improvement_vs_no_graph_pct"] = improvement(
            payload.get("overall", {}).get("ADE"),
            no_graph.get("overall", {}).get("ADE"),
        )
        payload["overall"]["ADE_improvement_vs_topk_graph_pct"] = improvement(
            payload.get("overall", {}).get("ADE"),
            topk.get("overall", {}).get("ADE"),
        )
        for subset_name, subset in payload.get("subsets", {}).items():
            base_no = no_graph.get("subsets", {}).get(subset_name, {})
            base_topk = topk.get("subsets", {}).get(subset_name, {})
            subset["ADE_improvement_vs_no_graph_pct"] = improvement(subset.get("ADE"), base_no.get("ADE"))
            subset["ADE_improvement_vs_topk_graph_pct"] = improvement(subset.get("ADE"), base_topk.get("ADE"))


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_float(value) if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(results: dict[str, Any]) -> str:
    lines = [
        "# Graph Subset Ablation v4 10s Prefix Evaluation",
        "",
        "Compares spatial-relation modules with the same v4 cached DINOv3 setup.",
        "",
        f"Train horizon: {results['train_horizon']}."
        " Metrics for 1s/3s/5s/10s are computed by slicing the same 10s prediction.",
        "",
        "Variants: no_graph, topk_graph, relpos_graph, contrast_graph, local_topk_graph, relpos_contrast_local_graph.",
        "",
    ]
    for prefix, prefix_results in results["prefix_results"].items():
        lines.extend([f"## Prefix {prefix}", ""])
        overall_rows = []
        for variant in results["variants"]:
            payload = prefix_results.get(variant, {})
            if not payload.get("complete"):
                overall_rows.append([variant, "incomplete", "-", "-", "-", "-", "-"])
                continue
            training = payload.get("training", {})
            overall = payload.get("overall", {})
            cv_baseline = overall.get("cv_baseline", {})
            overall_rows.append(
                [
                    variant,
                    "complete",
                    overall.get("samples"),
                    overall.get("ADE"),
                    overall.get("FDE"),
                    cv_baseline.get("ADE"),
                    training.get("best_epoch"),
                ]
            )
        lines.extend([table(["variant", "status", "N", "ADE", "FDE", "CV ADE", "best_epoch"], overall_rows), ""])

        subset_names = [
            "high_curvature_path",
            "turn_scene",
            "cv_baseline_error_high",
            "left_right_asymmetric_layout",
            "corridor_like_scene",
            "front_blocked_or_obstacle_proxy",
        ]
        for subset_name in subset_names:
            subset_rows = []
            for variant in results["variants"]:
                payload = prefix_results.get(variant, {})
                subset = payload.get("subsets", {}).get(subset_name, {})
                subset_rows.append(
                    [
                        variant,
                        subset.get("samples", "-"),
                        subset.get("ADE"),
                        subset.get("FDE"),
                        subset.get("cv_ADE"),
                        subset.get("ADE_improvement_vs_no_graph_pct"),
                        subset.get("ADE_improvement_vs_topk_graph_pct"),
                    ]
                )
            lines.extend(
                [
                    f"### {subset_name}",
                    "",
                    table(
                        [
                            "variant",
                            "N",
                            "ADE",
                            "FDE",
                            "CV ADE",
                            "ADE imp vs no_graph %",
                            "ADE imp vs topk %",
                        ],
                        subset_rows,
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def prefix_steps_from_args(prefixes: list[str]) -> dict[str, int]:
    unknown = [name for name in prefixes if name not in DEFAULT_PREFIX_STEPS]
    if unknown:
        raise ValueError(f"Unknown prefixes: {unknown}. Known prefixes: {sorted(DEFAULT_PREFIX_STEPS)}")
    return {name: DEFAULT_PREFIX_STEPS[name] for name in prefixes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/graph_subset_ablation_v4_10s"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/graph_subset_ablation_v4_10s/results.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/graph_subset_ablation_v4_10s.md"))
    parser.add_argument("--train-horizon", default="10s")
    parser.add_argument("--horizons", nargs="+", default=None, help="Backward-compatible alias; first value is used as train horizon.")
    parser.add_argument("--prefixes", nargs="+", default=["01s", "03s", "05s", "10s"])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_horizon = args.train_horizon
    if args.horizons:
        train_horizon = args.horizons[0]
    prefix_steps = prefix_steps_from_args(args.prefixes)
    results: dict[str, Any] = {
        "run_root": args.run_root.as_posix(),
        "seed": args.seed,
        "train_horizon": train_horizon,
        "prefix_steps": prefix_steps,
        "prefix_results": {prefix: {} for prefix in prefix_steps},
        "variants": args.variants,
        "subset_definitions": {
            "high_curvature_path": "top 25 percent by summed heading changes in GT future path for each prefix",
            "turn_scene": "top 25 percent by lateral displacement or high curvature for each prefix",
            "cv_baseline_error_high": "top 25 percent by constant-velocity ADE for each prefix",
            "left_right_asymmetric_layout": "top 25 percent by final or mean absolute right displacement for each prefix",
            "corridor_like_scene": "low curvature and low final lateral displacement control subset for each prefix",
            "front_blocked_or_obstacle_proxy": "high CV error and high curvature proxy; no explicit obstacle label is used",
        },
    }

    for variant in args.variants:
        run_dir = args.run_root / f"seed_{args.seed}" / train_horizon / variant
        run_payload = collect_run(run_dir, prefix_steps)
        if not run_payload.get("complete"):
            for prefix in prefix_steps:
                results["prefix_results"][prefix][variant] = run_payload
            continue
        for prefix in prefix_steps:
            results["prefix_results"][prefix][variant] = run_payload["prefixes"][prefix]

    for prefix in prefix_steps:
        add_relative_improvements(results["prefix_results"][prefix])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.report.write_text(build_report(results), encoding="utf-8")
    print(json.dumps({"output_json": args.output_json.as_posix(), "report": args.report.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
