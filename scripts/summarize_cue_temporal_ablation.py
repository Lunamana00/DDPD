"""Summarize 3s cue temporal transformer retraining ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = [
    ("cue_temporal_on", "Full model with cue temporal transformer"),
    ("cue_temporal_off", "Cue temporal transformer disabled"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cue temporal ablation metrics.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/cue_temporal_ablation_v4_03s"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/cue_temporal_ablation_v4_03s/results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/cue_temporal_ablation_v4_03s.md"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_history_row(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return min(history, key=lambda row: float(row.get("val", {}).get("ADE", float("inf"))))


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
    row.update(
        {
            "cue_temporal_layers": config.get("cue_temporal_layers"),
            "epochs_completed": len(history),
            "best_epoch": best.get("epoch") if best else None,
            "best_val_ADE": best.get("val", {}).get("ADE") if best else val.get("ADE"),
            "best_val_FDE": best.get("val", {}).get("FDE") if best else val.get("FDE"),
            "train_ADE_at_best": best.get("train_ADE") if best else None,
            "val_train_ADE_gap_at_best": best.get("val_train_ADE_gap") if best else None,
            "test_ADE": test.get("ADE"),
            "test_FDE": test.get("FDE"),
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


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def pct_change(new_value: float, base_value: float) -> float:
    return (new_value / base_value - 1.0) * 100.0


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cue Temporal Transformer Ablation v4 3s",
        "",
        "Retraining-time ablation over the 3s v4 DINOv3 TimeSFormer path predictor.",
        "",
        "| Variant | Cue temporal layers | Available | Best epoch | Test ADE | Test FDE | Best Val ADE | Train-Val ADE Gap | Avg epoch sec | Peak CUDA MB |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {cue_temporal_layers} | {available} | {best_epoch} | {test_ADE} | {test_FDE} | "
            "{best_val_ADE} | {val_train_ADE_gap_at_best} | {avg_epoch_seconds} | {peak_cuda_memory_mb} |".format(
                variant=row["variant"],
                cue_temporal_layers=fmt(row.get("cue_temporal_layers")),
                available="yes" if row.get("available") else "no",
                best_epoch=fmt(row.get("best_epoch")),
                test_ADE=fmt(row.get("test_ADE")),
                test_FDE=fmt(row.get("test_FDE")),
                best_val_ADE=fmt(row.get("best_val_ADE")),
                val_train_ADE_gap_at_best=fmt(row.get("val_train_ADE_gap_at_best")),
                avg_epoch_seconds=fmt(row.get("avg_epoch_seconds")),
                peak_cuda_memory_mb=fmt(row.get("peak_cuda_memory_mb")),
            )
        )

    available = {row["variant"]: row for row in rows if row.get("available") and row.get("test_ADE") is not None}
    if {"cue_temporal_on", "cue_temporal_off"}.issubset(available):
        on = available["cue_temporal_on"]
        off = available["cue_temporal_off"]
        ade_delta = float(off["test_ADE"]) - float(on["test_ADE"])
        fde_delta = float(off["test_FDE"]) - float(on["test_FDE"])
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- Disabling cue temporal changes test ADE by `{ade_delta:.4f}` "
                f"({pct_change(float(off['test_ADE']), float(on['test_ADE'])):+.2f}% vs enabled).",
                f"- Disabling cue temporal changes test FDE by `{fde_delta:.4f}` "
                f"({pct_change(float(off['test_FDE']), float(on['test_FDE'])):+.2f}% vs enabled).",
                "- Interpret this as a 3s-only retraining ablation; rerun other horizons before making horizon-general claims.",
            ]
        )
    else:
        lines.extend(["", "- No complete enabled/disabled pair found yet."])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [summarize_variant(args.runs_root, name, description) for name, description in VARIANTS]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"variants": rows}, indent=2), encoding="utf-8")
    write_markdown(args.output_md, rows)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
