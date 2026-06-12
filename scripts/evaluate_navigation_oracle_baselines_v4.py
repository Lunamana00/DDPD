"""Evaluate privileged navigation/pathfinding baselines on WIT-VZ v4.

The baselines in this file intentionally use information that the proposed
RGB-history predictor does not receive:

- PointNav/DD-PPO goal oracle: uses the GT future endpoint as the PointGoal.
- Pose-graph A* oracle: uses recorded pose/future world positions as a
  privileged traversability map and plans to the GT future endpoint.

Use these rows as upper-bound/context baselines, not as fair input-matched
competitors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.motion import constant_velocity_path
from src.models.navigation_oracles import (
    PoseGraphAStarPlanner,
    astar_oracle_prediction,
    pointnav_goal_oracle_prediction,
)
from src.wit_vz.io import load_json, read_jsonl


def horizon_tag(horizon: int) -> str:
    return f"{horizon:02d}s"


def v4_dataset_path(root: Path, horizon: int) -> Path:
    return root / "data" / "wit_vz" / "processed" / "horizon_sweep_v4_defaults" / f"future_{horizon_tag(horizon)}"


def path_errors(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = torch.linalg.norm(prediction - target, dim=-1)
    return distances.mean(dim=-1), distances[:, -1], distances.mean(dim=0)


def metric_summary(predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> dict[str, Any]:
    if not predictions:
        return {"available": False, "missing_reason": "no predictions"}
    pred = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    ade, fde, per_h = path_errors(pred, target)
    return {
        "available": True,
        "test_samples": int(target.shape[0]),
        "ADE": float(ade.mean().item()),
        "FDE": float(fde.mean().item()),
        "per_horizon_error": [float(value) for value in per_h.tolist()],
    }


def split_samples(dataset: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    samples = read_jsonl(dataset / "samples.jsonl")
    splits = load_json(dataset / "splits.json")
    manifest = load_json(dataset / "dataset_manifest.json")
    wanted = set(splits.get(split, []))
    if wanted:
        selected = [sample for sample in samples if sample["sample_id"] in wanted]
    else:
        selected = samples
    return samples, selected, manifest


def planner_samples_for_mode(
    all_samples: list[dict[str, Any]],
    dataset: Path,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "all":
        return all_samples
    splits = load_json(dataset / "splits.json")
    if mode == "train":
        wanted = set(splits.get("train", []))
    elif mode == "train_val":
        wanted = set(splits.get("train", [])) | set(splits.get("val", []))
    else:
        raise ValueError(f"Unsupported planner map source: {mode}")
    return [sample for sample in all_samples if sample["sample_id"] in wanted]


def evaluate_dataset(
    dataset: Path,
    horizon: int,
    split: str,
    astar_cell_size: float,
    planner_map_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start = time.perf_counter()
    all_samples, samples, manifest = split_samples(dataset, split)
    if not samples:
        return [
            {
                "horizon_sec": horizon,
                "dataset": dataset.as_posix(),
                "available": False,
                "missing_reason": f"no samples for split={split}",
            }
        ], []

    map_samples = planner_samples_for_mode(all_samples, dataset, planner_map_source)
    planner = PoseGraphAStarPlanner.from_samples(map_samples, cell_size=astar_cell_size)

    targets: list[torch.Tensor] = []
    cv_predictions: list[torch.Tensor] = []
    pointnav_predictions: list[torch.Tensor] = []
    astar_predictions: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []

    for sample in samples:
        target = torch.tensor(sample["future_local_path"], dtype=torch.float32).unsqueeze(0)
        ego_history = torch.tensor(sample["relative_egomotion_history"], dtype=torch.float32).unsqueeze(0)
        batch = {
            "sample_id": [sample["sample_id"]],
            "ego_history": ego_history,
            "future_path": target,
            "current_pose": [sample["current_pose"]],
            "metadata": [sample.get("metadata", {})],
            "source": [sample.get("source", {})],
        }
        cv_pred = constant_velocity_path(ego_history, target.shape[1])
        pointnav_pred = pointnav_goal_oracle_prediction(batch)
        astar_pred = torch.tensor(astar_oracle_prediction(sample, planner), dtype=torch.float32).unsqueeze(0)

        targets.append(target)
        cv_predictions.append(cv_pred)
        pointnav_predictions.append(pointnav_pred)
        astar_predictions.append(astar_pred)

        def one_error(pred: torch.Tensor) -> tuple[float, float]:
            ade, fde, _ = path_errors(pred, target)
            return float(ade.item()), float(fde.item())

        cv_ade, cv_fde = one_error(cv_pred)
        pn_ade, pn_fde = one_error(pointnav_pred)
        astar_ade, astar_fde = one_error(astar_pred)
        records.append(
            {
                "sample_id": sample["sample_id"],
                "episode_id": sample.get("episode_id"),
                "center_step": sample.get("center_step"),
                "metadata": sample.get("metadata", {}),
                "source": sample.get("source", {}),
                "target": sample["future_local_path"],
                "constant_velocity_prediction": cv_pred.squeeze(0).tolist(),
                "constant_velocity_ADE": cv_ade,
                "constant_velocity_FDE": cv_fde,
                "pointnav_goal_oracle_prediction": pointnav_pred.squeeze(0).tolist(),
                "pointnav_goal_oracle_ADE": pn_ade,
                "pointnav_goal_oracle_FDE": pn_fde,
                "astar_oracle_prediction": astar_pred.squeeze(0).tolist(),
                "astar_oracle_ADE": astar_ade,
                "astar_oracle_FDE": astar_fde,
            }
        )

    elapsed = time.perf_counter() - start
    rows = []
    base = {
        "horizon_sec": horizon,
        "dataset": dataset.as_posix(),
        "split": split,
        "future_steps": int(manifest.get("future_steps", len(samples[0]["future_local_path"]))),
        "planner_map_source": planner_map_source,
        "astar_cell_size": astar_cell_size,
        "astar_occupied_cells": len(planner.occupied or set()),
        "elapsed_sec": elapsed,
    }
    for key, label, preds, note in [
        (
            "constant_velocity",
            "Internal motion-only constant velocity",
            cv_predictions,
            "Uses only recent ego-motion. Included for context.",
        ),
        (
            "pointnav_ddppo_goal_oracle",
            "PointNav/DD-PPO goal-oracle adapter",
            pointnav_predictions,
            "Privileged: receives the GT future endpoint as PointGoal.",
        ),
        (
            "astar_pose_graph_oracle",
            "Classical A* pose-graph oracle",
            astar_predictions,
            "Privileged: uses recorded pose graph and GT future endpoint.",
        ),
    ]:
        summary = metric_summary(preds, targets)
        summary.update(base)
        summary.update({"model_key": key, "label": label, "note": note})
        rows.append(summary)
    return rows, records


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], predictions_dir: Path) -> None:
    order = {
        "constant_velocity": 0,
        "pointnav_ddppo_goal_oracle": 1,
        "astar_pose_graph_oracle": 2,
    }
    lines = [
        "# Navigation Oracle Baselines on WIT-VZ v4",
        "",
        "## Scope",
        "",
        "- Task: predict WIT-VZ future local trajectory `[forward, right]` and evaluate with ADE/FDE.",
        "- These are privileged navigation/pathfinding adapters, not input-matched competitors.",
        "- `PointNav/DD-PPO goal-oracle` receives the GT future endpoint as the PointGoal.",
        "- `A* pose-graph oracle` builds a traversability graph from recorded world poses and plans to the GT future endpoint.",
        f"- Prediction JSONL files: `{predictions_dir.as_posix()}`.",
        "",
        "## Results",
        "",
        "| Horizon | Model | Test samples | ADE | FDE | Notes |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: (item.get("horizon_sec", 0), order.get(item.get("model_key"), 99))):
        if row.get("available"):
            lines.append(
                "| "
                f"{row['horizon_sec']}s | {row['label']} | {row['test_samples']} | "
                f"{fmt(row['ADE'])} | {fmt(row['FDE'])} | {row['note']} |"
            )
        else:
            lines.append(
                "| "
                f"{row.get('horizon_sec', '-')}s | {row.get('label', 'missing')} | - | - | - | "
                f"{row.get('missing_reason', '')} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If PointNav has very low FDE, that is expected: it is given the true endpoint.",
            "- If A* performs well, it shows the value of map/pose/goal privileges, not that the RGB-only problem is solved.",
            "- Use these rows to frame the gap between local visual prediction and classical goal/map-based navigation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--split", default="test")
    parser.add_argument("--astar-cell-size", type=float, default=16.0)
    parser.add_argument("--planner-map-source", choices=["all", "train", "train_val"], default="all")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/navigation_oracle_baselines_v4/results.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/navigation_oracle_baselines_v4"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/navigation_oracle_baselines_v4.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    all_rows: list[dict[str, Any]] = []
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        dataset = v4_dataset_path(root, horizon)
        print(f"eval navigation oracles horizon={horizon}s dataset={dataset}", flush=True)
        if not dataset.exists():
            all_rows.append(
                {
                    "horizon_sec": horizon,
                    "dataset": dataset.as_posix(),
                    "available": False,
                    "missing_reason": f"missing dataset: {dataset.as_posix()}",
                }
            )
            continue
        rows, records = evaluate_dataset(
            dataset=dataset,
            horizon=horizon,
            split=args.split,
            astar_cell_size=args.astar_cell_size,
            planner_map_source=args.planner_map_source,
        )
        all_rows.extend(rows)
        pred_path = args.predictions_dir / f"predictions_{horizon_tag(horizon)}.jsonl"
        with pred_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    write_markdown(args.output_md, all_rows, args.predictions_dir)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "output_md": args.output_md.as_posix(),
                "predictions_dir": args.predictions_dir.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
