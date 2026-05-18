"""Generate a Markdown comparison table from run metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare path prediction model runs.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def find_metrics(run_dir: Path) -> dict:
    candidates = [run_dir / "eval" / "metrics.json", run_dir / "metrics.json"]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No metrics.json found under {run_dir}")


def main() -> None:
    args = parse_args()
    rows = []
    for run in args.runs:
        metrics = find_metrics(run)
        test = metrics.get("test", metrics)
        rows.append(
            {
                "run": run.name,
                "model": metrics.get("model", run.name),
                "ADE": float(test.get("ADE", float("nan"))),
                "FDE": float(test.get("FDE", float("nan"))),
            }
        )
    lines = [
        "# Path Prediction Comparison",
        "",
        "| Run | Model | ADE | FDE | Notes |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['model']} | {row['ADE']:.4f} | {row['FDE']:.4f} |  |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison report to: {args.out}")


if __name__ == "__main__":
    main()
