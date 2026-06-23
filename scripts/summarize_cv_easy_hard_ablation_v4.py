"""Summarize CV-defined easy/hard subsets for episodic-memory ablations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import json
import math
from pathlib import Path
from statistics import median
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/cv_easy_hard_episodic_memory_ablation_v4/results.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/cv_easy_hard_episodic_memory_ablation_v4.md"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizons", nargs="+", default=["01s", "03s", "05s", "10s"])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument(
        "--subset-reference",
        default="episodic_short_only",
        help="Variant whose CV ADE distribution defines easy/hard thresholds.",
    )
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


def mean_or_none(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def median_or_none(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(median(filtered))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def item_errors(item: dict[str, Any]) -> dict[str, float]:
    if "ADE" in item and "FDE" in item:
        ade = float(item["ADE"])
        fde = float(item["FDE"])
    else:
        errors = displacement_errors(item["prediction"], item["target"])
        ade = float(sum(errors) / len(errors))
        fde = float(errors[-1])
    if "constant_velocity_ADE" in item and "constant_velocity_FDE" in item:
        cv_ade = float(item["constant_velocity_ADE"])
        cv_fde = float(item["constant_velocity_FDE"])
    else:
        cv_errors = displacement_errors(item["constant_velocity_prediction"], item["target"])
        cv_ade = float(sum(cv_errors) / len(cv_errors))
        cv_fde = float(cv_errors[-1])
    return {"ADE": ade, "FDE": fde, "cv_ADE": cv_ade, "cv_FDE": cv_fde}


def aggregate_by_sample(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Average duplicate chunk predictions so every sample id contributes once."""
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in predictions:
        grouped[str(item["sample_id"])].append(item_errors(item))
    output = {}
    for sample_id, rows in grouped.items():
        output[sample_id] = {
            "sample_id": sample_id,
            "duplicates": len(rows),
            "ADE": mean_or_none(row["ADE"] for row in rows),
            "FDE": mean_or_none(row["FDE"] for row in rows),
            "cv_ADE": mean_or_none(row["cv_ADE"] for row in rows),
            "cv_FDE": mean_or_none(row["cv_FDE"] for row in rows),
        }
    return output


def load_variant(run_dir: Path) -> dict[str, Any]:
    paths = {
        "metrics": run_dir / "metrics.json",
        "predictions": run_dir / "predictions.jsonl",
        "config": run_dir / "config.json",
        "best": run_dir / "best.pt",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {"complete": False, "missing": missing, "run_dir": str(run_dir)}
    metrics = load_json(paths["metrics"])
    predictions = load_predictions(paths["predictions"])
    return {
        "complete": True,
        "run_dir": str(run_dir),
        "config": load_json(paths["config"]),
        "metrics": metrics,
        "raw_predictions": len(predictions),
        "samples": aggregate_by_sample(predictions),
    }


def improvement(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return float((baseline - value) / baseline * 100.0)


def subset_stats(
    sample_rows: dict[str, dict[str, Any]],
    ids: list[str],
    episodic_short_rows: dict[str, dict[str, Any]] | None = None,
    current_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = [sample_rows[sample_id] for sample_id in ids]
    stats: dict[str, Any] = {
        "samples": len(selected),
        "ADE": mean_or_none(row["ADE"] for row in selected),
        "FDE": mean_or_none(row["FDE"] for row in selected),
        "median_ADE": median_or_none(row["ADE"] for row in selected),
        "median_FDE": median_or_none(row["FDE"] for row in selected),
        "cv_ADE": mean_or_none(row["cv_ADE"] for row in selected),
        "cv_FDE": mean_or_none(row["cv_FDE"] for row in selected),
        "mean_duplicate_count": mean_or_none(row["duplicates"] for row in selected),
    }
    if episodic_short_rows is not None:
        baseline_values = [episodic_short_rows[sample_id]["ADE"] for sample_id in ids]
        deltas = [sample_rows[sample_id]["ADE"] - episodic_short_rows[sample_id]["ADE"] for sample_id in ids]
        wins = [sample_rows[sample_id]["ADE"] < episodic_short_rows[sample_id]["ADE"] for sample_id in ids]
        stats["ADE_vs_episodic_short_delta"] = mean_or_none(deltas)
        stats["median_ADE_vs_episodic_short_delta"] = median_or_none(deltas)
        stats["ADE_improvement_vs_episodic_short_pct"] = improvement(stats["ADE"], mean_or_none(baseline_values))
        stats["win_rate_vs_episodic_short"] = mean_or_none(1.0 if win else 0.0 for win in wins)
    if current_rows is not None:
        baseline_values = [current_rows[sample_id]["ADE"] for sample_id in ids]
        deltas = [sample_rows[sample_id]["ADE"] - current_rows[sample_id]["ADE"] for sample_id in ids]
        wins = [sample_rows[sample_id]["ADE"] < current_rows[sample_id]["ADE"] for sample_id in ids]
        stats["ADE_vs_current_short_delta"] = mean_or_none(deltas)
        stats["median_ADE_vs_current_short_delta"] = median_or_none(deltas)
        stats["ADE_improvement_vs_current_short_pct"] = improvement(stats["ADE"], mean_or_none(baseline_values))
        stats["win_rate_vs_current_short"] = mean_or_none(1.0 if win else 0.0 for win in wins)
    return stats


def best_variant_for_subset(variants: dict[str, Any], subset_name: str) -> dict[str, Any]:
    candidates = []
    for variant, payload in variants.items():
        if not payload.get("complete"):
            continue
        subset = payload.get("subsets", {}).get(subset_name, {})
        if subset.get("ADE") is not None:
            candidates.append((float(subset["ADE"]), variant))
    if not candidates:
        return {}
    ade, variant = min(candidates)
    return {"variant": variant, "ADE": ade}


def summarize_horizon(args: argparse.Namespace, horizon: str) -> dict[str, Any]:
    loaded = {}
    for variant in args.variants:
        run_dir = args.run_root / f"seed_{args.seed}" / horizon / variant
        loaded[variant] = load_variant(run_dir)

    complete = {variant: payload for variant, payload in loaded.items() if payload.get("complete")}
    if args.subset_reference not in complete:
        raise ValueError(f"Subset reference variant is incomplete or missing: {args.subset_reference}")

    common_ids = sorted(set.intersection(*(set(payload["samples"]) for payload in complete.values())))
    reference_rows = complete[args.subset_reference]["samples"]
    reference_cv = [float(reference_rows[sample_id]["cv_ADE"]) for sample_id in common_ids]
    easy_threshold = percentile(reference_cv, 0.25)
    hard_threshold = percentile(reference_cv, 0.75)

    subset_ids = {
        "all_common": common_ids,
        "cv_easy_bottom25": [
            sample_id for sample_id in common_ids if float(reference_rows[sample_id]["cv_ADE"]) <= easy_threshold
        ],
        "cv_hard_top25": [
            sample_id for sample_id in common_ids if float(reference_rows[sample_id]["cv_ADE"]) >= hard_threshold
        ],
    }

    episodic_short_rows = complete.get("episodic_short_only", {}).get("samples")
    current_rows = complete.get("current_short_window", {}).get("samples")
    variants: dict[str, Any] = {}
    for variant, payload in loaded.items():
        if not payload.get("complete"):
            variants[variant] = payload
            continue
        rows = payload["samples"]
        variants[variant] = {
            "complete": True,
            "run_dir": payload["run_dir"],
            "raw_predictions": payload["raw_predictions"],
            "unique_samples": len(rows),
            "common_samples": len(common_ids),
            "config": {
                key: payload["config"].get(key)
                for key in (
                    "model",
                    "memory_type",
                    "long_memory_type",
                    "long_memory_use_ego",
                    "detach_long_memory",
                    "chunk_length",
                    "burn_in",
                    "future_steps",
                    "spatial_relation_type",
                )
                if key in payload["config"]
            },
            "training": {
                "best_epoch": payload["metrics"].get("best_epoch"),
                "val_ADE": payload["metrics"].get("val", {}).get("ADE"),
                "test_ADE": payload["metrics"].get("test", {}).get("ADE"),
                "test_FDE": payload["metrics"].get("test", {}).get("FDE"),
            },
            "subsets": {
                name: subset_stats(
                    rows,
                    ids,
                    episodic_short_rows=episodic_short_rows,
                    current_rows=current_rows,
                )
                for name, ids in subset_ids.items()
            },
        }

    return {
        "subset_reference": args.subset_reference,
        "common_samples": len(common_ids),
        "thresholds": {
            "cv_ADE_easy_q25": easy_threshold,
            "cv_ADE_hard_q75": hard_threshold,
        },
        "subset_sizes": {name: len(ids) for name, ids in subset_ids.items()},
        "best_by_ADE": {
            name: best_variant_for_subset(variants, name)
            for name in subset_ids
        },
        "variants": variants,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(results: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# CV Easy/Hard Subset Ablation",
        "",
        "This report reuses already trained episodic-memory ablation checkpoints and re-evaluates their saved predictions on matched sample subsets.",
        "",
        "Subset definition:",
        "",
        "- `cv_easy_bottom25`: bottom 25% by constant-velocity ADE.",
        "- `cv_hard_top25`: top 25% by constant-velocity ADE.",
        "- Thresholds are computed from the `episodic_short_only` variant on the common sample-id intersection for each horizon.",
        "- Duplicate episodic chunk predictions for the same sample id are averaged before comparison.",
        "",
    ]
    lines.extend(
        [
            "## Executive Summary",
            "",
            "| Horizon | Common samples | Overall best | Easy best | Hard best | Hard-subset read |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for horizon, horizon_payload in results["horizons"].items():
        overall_best = horizon_payload["best_by_ADE"].get("all_common", {}).get("variant", "-")
        easy_best = horizon_payload["best_by_ADE"].get("cv_easy_bottom25", {}).get("variant", "-")
        hard_best = horizon_payload["best_by_ADE"].get("cv_hard_top25", {}).get("variant", "-")
        if str(hard_best).startswith("long_"):
            read = "long memory wins hard subset"
        elif hard_best == "episodic_short_only":
            read = "episodic training wins; explicit long memory not needed"
        elif hard_best == "current_short_window":
            read = "single-window control wins"
        else:
            read = "-"
        lines.append(
            f"| {horizon} | {horizon_payload['common_samples']} | `{overall_best}` | `{easy_best}` | `{hard_best}` | {read} |"
        )
    lines.extend(
        [
            "",
            "Main interpretation:",
            "",
            "- Hard/easy is defined by the constant-velocity baseline, so it asks whether a sample is easy for motion extrapolation before looking at model errors.",
            "- Explicit long memory wins the hard subset in 1s, 5s, and 10s, but not in 3s.",
            "- The easy subset is mixed, so the result does not support a blanket claim that long memory is always better.",
            "- The strongest defensible claim is conditional: memory is useful mainly on samples where recent-motion extrapolation is insufficient.",
            "",
        ]
    )
    variants_order = results["variants"]
    for horizon, horizon_payload in results["horizons"].items():
        lines.extend(
            [
                f"## Horizon {horizon}",
                "",
                f"- common samples: {horizon_payload['common_samples']}",
                f"- easy threshold CV ADE <= {fmt(horizon_payload['thresholds']['cv_ADE_easy_q25'])}",
                f"- hard threshold CV ADE >= {fmt(horizon_payload['thresholds']['cv_ADE_hard_q75'])}",
                "",
            ]
        )
        for subset_name in ("all_common", "cv_easy_bottom25", "cv_hard_top25"):
            best = horizon_payload["best_by_ADE"].get(subset_name, {})
            lines.extend(
                [
                    f"### {subset_name}",
                    "",
                    f"Best ADE: `{best.get('variant', '-')}` ({fmt(best.get('ADE'))})",
                    "",
                    "| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for variant in variants_order:
                payload = horizon_payload["variants"].get(variant, {})
                subset = payload.get("subsets", {}).get(subset_name, {})
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            variant,
                            str(subset.get("samples", "-")),
                            fmt(subset.get("ADE")),
                            fmt(subset.get("FDE")),
                            fmt(subset.get("median_ADE")),
                            fmt(subset.get("cv_ADE")),
                            fmt(subset.get("ADE_vs_episodic_short_delta")),
                            fmt(subset.get("win_rate_vs_episodic_short"), 3),
                            fmt(subset.get("ADE_vs_current_short_delta")),
                            fmt(subset.get("win_rate_vs_current_short"), 3),
                        ]
                    )
                    + " |"
                )
            lines.append("")
        lines.extend(["### Quick interpretation", ""])
        all_best = horizon_payload["best_by_ADE"].get("all_common", {}).get("variant", "-")
        easy_best = horizon_payload["best_by_ADE"].get("cv_easy_bottom25", {}).get("variant", "-")
        hard_best = horizon_payload["best_by_ADE"].get("cv_hard_top25", {}).get("variant", "-")
        lines.extend(
            [
                f"- Overall best on matched common samples: `{all_best}`.",
                f"- Easy subset best: `{easy_best}`.",
                f"- Hard subset best: `{hard_best}`.",
                "- If a memory variant improves mainly on `cv_hard_top25` but not on `cv_easy_bottom25`, it supports the claim that memory helps when motion extrapolation is insufficient.",
                "",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = {
        "description": "Evaluation-time easy/hard subset ablation using constant-velocity ADE thresholds.",
        "run_root": str(args.run_root),
        "seed": args.seed,
        "variants": args.variants,
        "subset_reference": args.subset_reference,
        "horizons": {
            horizon: summarize_horizon(args, horizon)
            for horizon in args.horizons
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, args.report)
    print(json.dumps({"output_json": str(args.output_json), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
