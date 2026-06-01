"""Summarize trainable paper-inspired baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "khaleque_motivated_trainable": {
        "label": "Khaleque-inspired trainable ego-motion baseline",
        "path": Path("runs/paper_trainable_baselines_v4_03s/khaleque_motivated"),
    },
    "xu_pixels_only_trainable": {
        "label": "Xu-inspired trainable pixels-only baseline",
        "path": Path("runs/paper_trainable_baselines_v4_03s/xu_pixels_only"),
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def best_history_row(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return min(history, key=lambda row: float(row.get("val", {}).get("ADE", float("inf"))))


def run_row(model_key: str, label: str, run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if not metrics_path.exists():
        return {
            "model_key": model_key,
            "label": label,
            "kind": "trainable",
            "available": False,
            "run_dir": run_dir.as_posix(),
            "missing_reason": f"missing metrics: {metrics_path.as_posix()}",
        }
    metrics = load_json(metrics_path)
    config = load_json(config_path) if config_path.exists() else {}
    best = best_history_row(metrics.get("history", []))
    test = metrics.get("test", {})
    val = metrics.get("val", {})
    row = {
        "model_key": model_key,
        "label": label,
        "kind": "trainable",
        "available": True,
        "run_dir": run_dir.as_posix(),
        "dataset": config.get("dataset"),
        "visual_feature_cache": config.get("visual_feature_cache"),
        "test_ADE": test.get("ADE"),
        "test_FDE": test.get("FDE"),
        "val_ADE": val.get("ADE"),
        "val_FDE": val.get("FDE"),
        "best_epoch": best.get("epoch") if best else None,
        "best_val_ADE": best.get("val", {}).get("ADE") if best else None,
        "best_train_ADE": best.get("train_ADE") if best else None,
        "best_train_val_ADE_gap": best.get("val_train_ADE_gap") if best else None,
        "best_epoch_seconds": best.get("epoch_seconds") if best else None,
        "best_cuda_peak_memory_mb": best.get("cuda_peak_memory_mb") if best else None,
        "epochs_ran": len(metrics.get("history", [])),
        "lr": config.get("lr"),
        "batch_size": config.get("batch_size"),
        "seed": config.get("seed"),
    }
    return row


def load_reference_rows(path: Path, horizon: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for item in load_json(path):
        if int(item.get("horizon_sec", -1)) != horizon or not item.get("available"):
            continue
        rows.append(
            {
                "model_key": item.get("model_key"),
                "label": item.get("label"),
                "kind": item.get("kind"),
                "available": True,
                "test_ADE": item.get("ADE"),
                "test_FDE": item.get("FDE"),
                "test_samples": item.get("test_samples"),
                "source": path.as_posix(),
            }
        )
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def add_gains(rows: list[dict[str, Any]]) -> None:
    trainable = [row for row in rows if row.get("kind") == "trainable" and row.get("available")]
    if not trainable:
        return
    best_train_ade = min(trainable, key=lambda row: float(row["test_ADE"]))
    best_train_fde = min(trainable, key=lambda row: float(row["test_FDE"]))
    for row in rows:
        if not row.get("available"):
            continue
        if row.get("test_ADE") is not None:
            row["ADE_gain_vs_best_trainable_paper_pct"] = (
                1.0 - float(row["test_ADE"]) / float(best_train_ade["test_ADE"])
            ) * 100.0
        if row.get("test_FDE") is not None:
            row["FDE_gain_vs_best_trainable_paper_pct"] = (
                1.0 - float(row["test_FDE"]) / float(best_train_fde["test_FDE"])
            ) * 100.0


def write_markdown(path: Path, rows: list[dict[str, Any]], horizon: int) -> None:
    order = {
        "khaleque_center_random_proxy": 0,
        "xu_pixels_saliency_proxy": 1,
        "khaleque_motivated_trainable": 2,
        "xu_pixels_only_trainable": 3,
        "constant_velocity": 4,
        "ours_dinov3_single": 5,
    }
    lines = [
        f"# Trainable Paper-Inspired Baselines V4 {horizon}s",
        "",
        "## Scope",
        "",
        f"- Dataset: `horizon_sweep_v4_defaults/future_{horizon:02d}s`.",
        "- Task: predict future local path `[forward, right]` from the same test split.",
        "- Metrics: ADE/FDE in local egocentric coordinates; lower is better.",
        "- These are trainable adapters for WIT-VZ, not exact reproductions of the original interactive systems.",
        "",
        "## Models",
        "",
        "- Khaleque-inspired trainable baseline: ego-motion history encoder plus learned motivation tokens; no RGB.",
        "- Xu-inspired trainable baseline: pixels-only visual history encoder using cached DINO tokens; no ego-motion.",
        "",
        "## Results",
        "",
        "| Model | Kind | Available | ADE | FDE | Best epoch | Train/val ADE gap | Epoch sec | Peak VRAM MB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: order.get(str(item.get("model_key")), 99)):
        if row.get("available"):
            lines.append(
                "| "
                f"{row['label']} | {row.get('kind')} | yes | "
                f"{fmt(row.get('test_ADE'))} | {fmt(row.get('test_FDE'))} | "
                f"{fmt(row.get('best_epoch'))} | {fmt(row.get('best_train_val_ADE_gap'))} | "
                f"{fmt(row.get('best_epoch_seconds'))} | {fmt(row.get('best_cuda_peak_memory_mb'))} |"
            )
        else:
            lines.append(
                "| "
                f"{row['label']} | {row.get('kind')} | no | - | - | - | - | - | - |"
            )

    missing = [row for row in rows if not row.get("available")]
    if missing:
        lines.extend(["", "## Missing", ""])
        for row in missing:
            lines.append(f"- `{row['model_key']}`: {row.get('missing_reason')}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If a trainable paper baseline still trails the proposed model, the gap is less likely to be caused only by an unfair hand-coded proxy.",
            "- If the Xu-inspired trainable baseline is strong, screen-only visual history already carries meaningful navigation signal.",
            "- If the Khaleque-inspired trainable baseline is close to constant velocity, ego-motion alone explains much of the local short-horizon behavior.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--reference-json", type=Path, default=Path("outputs/paper_baselines_v4/results.json"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/trainable_paper_baselines_v4_03s/results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/trainable_paper_baselines_v4_03s.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_reference_rows(args.reference_json, args.horizon)
    for model_key, spec in DEFAULT_RUNS.items():
        rows.append(run_row(model_key, spec["label"], spec["path"]))
    add_gains(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(args.output_md, rows, args.horizon)
    print(json.dumps({"output_json": args.output_json.as_posix(), "output_md": args.output_md.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
