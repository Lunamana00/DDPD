"""Summarize 3s DINO visual-backbone ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = [
    ("constant_velocity", "Motion-only constant velocity baseline; no visual branch training."),
    ("zero_visual", "Cue-memory model with zero image-content tokens plus the same positional scaffold."),
    ("small_cnn", "Cue-memory model with a trainable small CNN visual encoder from RGB."),
    ("cached_dinov3", "Cue-memory model with frozen cached DINOv3 ConvNeXt-Tiny dense visual tokens."),
]

PREFIX_STEPS = {
    "1s": 5,
    "2s": 10,
    "3s": 15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DINO ablation metrics.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/dino_ablation_v4_03s"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/dino_ablation_v4_03s/results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/dino_ablation_v4_03s.md"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_history_row(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return min(history, key=lambda row: float(row.get("val", {}).get("ADE", float("inf"))))


def prefix_metrics(per_horizon_error: list[float]) -> dict[str, dict[str, float]]:
    prefixes: dict[str, dict[str, float]] = {}
    for label, steps in PREFIX_STEPS.items():
        if len(per_horizon_error) < steps:
            continue
        prefix = per_horizon_error[:steps]
        prefixes[label] = {
            "ADE": sum(float(value) for value in prefix) / steps,
            "FDE": float(per_horizon_error[steps - 1]),
        }
    return prefixes


def summarize_variant(runs_root: Path, name: str, description: str) -> dict[str, Any]:
    run_dir = runs_root / name
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    row: dict[str, Any] = {
        "variant": name,
        "description": description,
        "run_dir": run_dir.as_posix(),
        "available": metrics_path.exists(),
    }
    if not metrics_path.exists():
        row["missing"] = metrics_path.as_posix()
        return row

    metrics = load_json(metrics_path)
    config = load_json(config_path) if config_path.exists() else {}
    history = metrics.get("history", [])
    best = best_history_row(history)
    test = metrics.get("test", {})
    val = metrics.get("val", {})
    test_per_horizon = test.get("per_horizon_error", [])
    row.update(
        {
            "model": metrics.get("model", config.get("model")),
            "backbone": config.get("backbone"),
            "epochs_completed": len(history),
            "best_epoch": best.get("epoch") if best else None,
            "best_val_ADE": best.get("val", {}).get("ADE") if best else val.get("ADE"),
            "best_val_FDE": best.get("val", {}).get("FDE") if best else val.get("FDE"),
            "train_ADE_at_best": best.get("train_ADE") if best else None,
            "val_train_ADE_gap_at_best": best.get("val_train_ADE_gap") if best else None,
            "test_ADE": test.get("ADE"),
            "test_FDE": test.get("FDE"),
            "test_per_horizon_error": test_per_horizon,
            "test_prefix_metrics": prefix_metrics(test_per_horizon),
            "avg_epoch_seconds": (
                sum(float(item.get("epoch_seconds", 0.0)) for item in history) / len(history)
                if history
                else None
            ),
            "peak_cuda_memory_mb": max(
                (float(item.get("cuda_peak_memory_mb", 0.0)) for item in history),
                default=None,
            ),
        }
    )
    return row


def percent_gain(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference == 0:
        return None
    return (1.0 - candidate / reference) * 100.0


def add_pairwise_gains(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["variant"]: row for row in rows if row.get("available")}
    dino = by_name.get("cached_dinov3", {})
    comparisons: dict[str, Any] = {}
    for baseline in ("constant_velocity", "zero_visual", "small_cnn"):
        base = by_name.get(baseline)
        if not base:
            continue
        comparisons[f"cached_dinov3_vs_{baseline}"] = {
            "test_ADE_gain_percent": percent_gain(base.get("test_ADE"), dino.get("test_ADE")),
            "test_FDE_gain_percent": percent_gain(base.get("test_FDE"), dino.get("test_FDE")),
        }
        for label in PREFIX_STEPS:
            base_prefix = base.get("test_prefix_metrics", {}).get(label, {})
            dino_prefix = dino.get("test_prefix_metrics", {}).get(label, {})
            comparisons[f"cached_dinov3_vs_{baseline}"][f"{label}_ADE_gain_percent"] = percent_gain(
                base_prefix.get("ADE"), dino_prefix.get("ADE")
            )
            comparisons[f"cached_dinov3_vs_{baseline}"][f"{label}_FDE_gain_percent"] = percent_gain(
                base_prefix.get("FDE"), dino_prefix.get("FDE")
            )
    return comparisons


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], comparisons: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DINO Ablation v4 3s",
        "",
        "Retraining-time ablation over the v4 3s WIT-VZ split.",
        "All trainable variants use the same cue-memory path predictor, TimeSFormer-style temporal adapter, "
        "TokenLearner selector, attention cue memory, horizon query decoder, constant-velocity residual prior, "
        "seed, optimizer, and source-policy balancing. The changed factor is the visual evidence source.",
        "",
        "Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s`.",
        "DINO cache: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`.",
        "",
        "| Variant | Backbone | Available | Epochs | Best epoch | Test ADE | Test FDE | Best Val ADE | "
        "Train-Val ADE Gap | Avg epoch sec | Peak CUDA MB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {backbone} | {available} | {epochs} | {best_epoch} | {test_ADE} | {test_FDE} | "
            "{best_val_ADE} | {gap} | {epoch_sec} | {peak_mem} |".format(
                variant=row["variant"],
                backbone=fmt(row.get("backbone")),
                available="yes" if row.get("available") else "no",
                epochs=fmt(row.get("epochs_completed")),
                best_epoch=fmt(row.get("best_epoch")),
                test_ADE=fmt(row.get("test_ADE")),
                test_FDE=fmt(row.get("test_FDE")),
                best_val_ADE=fmt(row.get("best_val_ADE")),
                gap=fmt(row.get("val_train_ADE_gap_at_best")),
                epoch_sec=fmt(row.get("avg_epoch_seconds"), 2),
                peak_mem=fmt(row.get("peak_cuda_memory_mb"), 1),
            )
        )

    lines.extend(
        [
            "",
            "## Prefix Metrics",
            "",
            "The 3s model predicts 15 future points at 5 FPS. Prefix metrics reuse the same prediction and score "
            "the first 5/10/15 points as 1s/2s/3s evidence.",
            "",
            "| Variant | 1s ADE | 1s FDE | 2s ADE | 2s FDE | 3s ADE | 3s FDE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        prefixes = row.get("test_prefix_metrics", {})
        lines.append(
            "| {variant} | {ade1} | {fde1} | {ade2} | {fde2} | {ade3} | {fde3} |".format(
                variant=row["variant"],
                ade1=fmt(prefixes.get("1s", {}).get("ADE")),
                fde1=fmt(prefixes.get("1s", {}).get("FDE")),
                ade2=fmt(prefixes.get("2s", {}).get("ADE")),
                fde2=fmt(prefixes.get("2s", {}).get("FDE")),
                ade3=fmt(prefixes.get("3s", {}).get("ADE")),
                fde3=fmt(prefixes.get("3s", {}).get("FDE")),
            )
        )

    lines.extend(["", "## DINO Gain"])
    if comparisons:
        for name, values in comparisons.items():
            lines.append(
                "- `{name}`: ADE gain {ade}, FDE gain {fde}.".format(
                    name=name,
                    ade=fmt(values.get("test_ADE_gain_percent"), 2) + "%",
                    fde=fmt(values.get("test_FDE_gain_percent"), 2) + "%",
                )
            )
    else:
        lines.append("- Pairwise DINO comparisons are unavailable until completed metrics exist.")

    available = [row for row in rows if row.get("available") and row.get("test_ADE") is not None]
    if available:
        best = min(available, key=lambda row: float(row["test_ADE"]))
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- Best test ADE: `{best['variant']}` ({float(best['test_ADE']):.4f}).",
                "- If `cached_dinov3` beats `small_cnn`, pretrained dense visual tokens add useful game-navigation signal "
                "beyond learning a small RGB encoder from this dataset alone.",
                "- If `cached_dinov3` only beats `zero_visual` but not `constant_velocity`, the visual branch is learning "
                "some image-conditioned residuals but the task is still dominated by recent ego-motion.",
                "- If `small_cnn` is competitive with DINO, the v4 ViZDoom visual domain may be simple enough that task-specific "
                "RGB features match frozen foundation features for this horizon.",
            ]
        )
    else:
        lines.extend(["", "## Interpretation", "", "- No completed variant metrics found yet."])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [summarize_variant(args.runs_root, name, description) for name, description in VARIANTS]
    comparisons = add_pairwise_gains(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"variants": rows, "comparisons": comparisons}, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_md, rows, comparisons)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
