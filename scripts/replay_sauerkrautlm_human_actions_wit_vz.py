"""Replay SauerkrautLM human action labels in ViZDoom and build WIT-VZ samples.

The Hugging Face dataset stores human gameplay action scores, not original
pose trajectories. This script therefore creates a replay-derived human-action
trajectory: it applies the public human action sequence to a fresh ViZDoom
episode, records RGB/pose/action rows, and optionally converts the replayed raw
run to the WIT-VZ supervised path-prediction schema.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wit_vz.build_samples import build_samples
from src.wit_vz.collect import (
    BUTTON_NAMES,
    GAME_VARIABLE_NAMES,
    action_id_for_vector,
    get_screen_resolution,
    labels_to_dicts,
    pose_from_variables,
    scenario_path,
)
from src.wit_vz.geometry import compute_relative_egomotion
from src.wit_vz.io import save_npz, save_png, to_jsonable, write_json, write_jsonl


DATASET_NAME = "VAGOsolutions/SauerkrautLM-Doom-MultiVec-31k"
ACTION_SCORE_ORDER = ("shoot", "move_forward", "turn_left", "turn_right")
POLICY_NAME = "sauerkrautlm_human_action_replay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--run-id", default="wit_vz_sauerkrautlm_human_replay_001")
    parser.add_argument("--out-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--processed-out", type=Path, default=None)
    parser.add_argument("--scenario", default="defend_the_center")
    parser.add_argument("--map", default="map01")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--episode-gap", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--screen-width", type=int, default=160)
    parser.add_argument("--screen-height", type=int, default=120)
    parser.add_argument("--action-threshold", type=float, default=0.5)
    parser.add_argument(
        "--argmax-if-empty",
        action="store_true",
        help="If no action score reaches the threshold, activate the highest-scoring action.",
    )
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--save-labels", action="store_true")
    parser.add_argument("--save-automap", action="store_true")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-samples", action="store_true")
    parser.add_argument("--history-sec", type=float, default=1.0)
    parser.add_argument("--future-sec", type=float, default=5.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sample-split", choices=["episode", "map", "source"], default="episode")
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


def scores_to_action_vector(
    scores: list[float],
    threshold: float,
    argmax_if_empty: bool = False,
) -> tuple[list[int], str, list[str]]:
    if len(scores) != len(ACTION_SCORE_ORDER):
        raise ValueError(f"Expected {len(ACTION_SCORE_ORDER)} action scores, got {len(scores)}")
    active = [score >= threshold for score in scores]
    if argmax_if_empty and not any(active):
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        active[best_idx] = True

    shoot, move_forward, turn_left, turn_right = active
    vector = [
        int(shoot),
        int(move_forward),
        0,
        0,
        int(turn_right),
        int(turn_left),
    ]
    labels = [name for name, is_active in zip(ACTION_SCORE_ORDER, active) if is_active]
    return vector, "+".join(labels) if labels else "pause", labels


def episode_row_ranges(
    total_rows: int,
    episodes: int,
    episode_steps: int,
    start_index: int,
    episode_gap: int,
) -> list[range]:
    ranges: list[range] = []
    cursor = start_index
    for _episode_idx in range(episodes):
        end = min(cursor + episode_steps, total_rows)
        if cursor >= end:
            break
        ranges.append(range(cursor, end))
        cursor = end + max(0, episode_gap)
    return ranges


def load_human_action_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "This script needs Hugging Face datasets. Run with: "
            "uv run --with datasets python scripts/replay_sauerkrautlm_human_actions_wit_vz.py ..."
        ) from exc
    dataset = load_dataset(dataset_name, split=split)
    return [dict(item) for item in dataset]


def build_game(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("ViZDoom is required. Install with: uv pip install vizdoom pillow") from exc

    game = vzd.DoomGame()
    game.set_doom_scenario_path(scenario_path(vzd, args.scenario))
    game.set_doom_map(args.map)
    game.set_window_visible(bool(args.visible))
    if hasattr(game, "set_sound_enabled"):
        game.set_sound_enabled(False)
    if hasattr(game, "set_music_enabled"):
        game.set_music_enabled(False)
    game.set_screen_resolution(get_screen_resolution(vzd, args.screen_width, args.screen_height))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(bool(args.save_depth))
    game.set_labels_buffer_enabled(bool(args.save_labels))
    game.set_automap_buffer_enabled(bool(args.save_automap))
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_episode_timeout(args.episode_steps * args.frame_skip)
    game.set_episode_start_time(10)
    game.set_living_reward(0.0)
    game.set_mode(vzd.Mode.PLAYER)

    buttons = []
    button_names = []
    for name in BUTTON_NAMES:
        if hasattr(vzd.Button, name):
            buttons.append(getattr(vzd.Button, name))
            button_names.append(name)
    variables = []
    variable_names = []
    for name in GAME_VARIABLE_NAMES:
        if hasattr(vzd.GameVariable, name):
            variables.append(getattr(vzd.GameVariable, name))
            variable_names.append(name)
    game.set_available_buttons(buttons)
    game.set_available_game_variables(variables)
    game.init()
    return game, {
        "button_order": button_names,
        "game_variable_names": variable_names,
        "doom_tics_per_second": 35.0,
    }


def collect_replay_episode(
    game: Any,
    game_context: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    episode_index: int,
    row_indices: range,
    action_rows: list[dict[str, Any]],
    global_step_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    global_step = global_step_start
    prev_pose = None

    game.new_episode()
    for step, row_idx in enumerate(row_indices):
        if game.is_episode_finished():
            break
        state = game.get_state()
        if state is None:
            break

        variables = {
            name: to_jsonable(value)
            for name, value in zip(game_context["game_variable_names"], state.game_variables)
        }
        pose = pose_from_variables(variables)
        egomotion = compute_relative_egomotion(prev_pose, pose)

        frame_name = f"{step:06d}"
        frame_rel = Path("episodes") / episode_id / "frames" / f"{frame_name}.png"
        depth_rel = Path("episodes") / episode_id / "depth" / f"{frame_name}.npz"
        labels_rel = Path("episodes") / episode_id / "labels" / f"{frame_name}.npz"
        automap_rel = Path("episodes") / episode_id / "automap" / f"{frame_name}.png"

        save_png(state.screen_buffer, run_dir / frame_rel)
        depth_buffer = getattr(state, "depth_buffer", None)
        labels_buffer = getattr(state, "labels_buffer", None)
        if args.save_depth and depth_buffer is None:
            raise RuntimeError("Depth buffer was requested but ViZDoom returned no depth_buffer")
        if args.save_labels and labels_buffer is None:
            raise RuntimeError("Labels buffer was requested but ViZDoom returned no labels_buffer")
        saved_depth = save_npz(depth_buffer if args.save_depth else None, run_dir / depth_rel)
        saved_labels = save_npz(labels_buffer if args.save_labels else None, run_dir / labels_rel)
        automap_path = None
        automap_buffer = getattr(state, "automap_buffer", None)
        if args.save_automap and automap_buffer is None:
            raise RuntimeError("Automap buffer was requested but ViZDoom returned no automap_buffer")
        if args.save_automap and automap_buffer is not None:
            save_png(automap_buffer, run_dir / automap_rel)
            automap_path = automap_rel.as_posix()

        scores = [float(value) for value in action_rows[row_idx]["scores"]]
        action_vector, decoded_name, active_labels = scores_to_action_vector(
            scores,
            threshold=args.action_threshold,
            argmax_if_empty=args.argmax_if_empty,
        )
        reward = float(game.make_action(action_vector, args.frame_skip))
        done = bool(game.is_episode_finished())
        total_reward += reward

        records.append(
            {
                "sample_id": f"{episode_id}_{step:06d}",
                "episode_id": episode_id,
                "step": step,
                "global_step": global_step,
                "timestamp": step * args.frame_skip / game_context["doom_tics_per_second"],
                "frame_path": frame_rel.as_posix(),
                "depth_path": depth_rel.as_posix() if saved_depth else None,
                "labels_path": labels_rel.as_posix() if saved_labels else None,
                "automap_path": automap_path,
                "pose": pose,
                "relative_egomotion_from_prev": egomotion,
                "action": {
                    "action_id": action_id_for_vector(action_vector),
                    "action_name": decoded_name,
                    "action_vector": action_vector,
                    "button_order": game_context["button_order"],
                    "policy": POLICY_NAME,
                    "external_dataset": args.dataset_name,
                    "external_split": args.split,
                    "external_row_index": row_idx,
                    "score_order": list(ACTION_SCORE_ORDER),
                    "scores": scores,
                    "active_score_labels": active_labels,
                    "threshold": args.action_threshold,
                },
                "reward": reward,
                "done": done,
                "visible_labels": labels_to_dicts(getattr(state, "labels", None)),
                "game_variables": variables,
            }
        )
        prev_pose = pose
        global_step += 1

    summary = {
        "episode_id": episode_id,
        "num_steps": len(records),
        "total_reward": total_reward,
        "policy": POLICY_NAME,
        "external_dataset": args.dataset_name,
        "external_split": args.split,
        "external_row_start": row_indices.start,
        "external_row_stop": row_indices.stop,
        "start_pose": records[0]["pose"] if records else None,
        "final_pose": records[-1]["pose"] if records else None,
        "done": records[-1]["done"] if records else False,
        "reason_end": "done" if records and records[-1]["done"] else "max_steps_or_no_state",
    }
    write_jsonl(episode_dir / "steps.jsonl", records)
    write_json(episode_dir / "summary.json", summary)
    return records, summary, global_step


def replay_dataset(args: argparse.Namespace) -> Path:
    run_dir = args.out_root / args.run_id
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    action_rows = load_human_action_rows(args.dataset_name, args.split)
    row_ranges = episode_row_ranges(
        total_rows=len(action_rows),
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        start_index=args.start_index,
        episode_gap=args.episode_gap,
    )
    if not row_ranges:
        raise RuntimeError("No replay episode ranges were selected")

    game, game_context = build_game(args)
    summaries = []
    global_step = 0
    try:
        for episode_index, row_range in enumerate(row_ranges, start=1):
            _records, summary, global_step = collect_replay_episode(
                game,
                game_context,
                args,
                run_dir,
                episode_index,
                row_range,
                action_rows,
                global_step,
            )
            summaries.append(summary)
            print(
                f"{summary['episode_id']}: rows={row_range.start}:{row_range.stop} "
                f"steps={summary['num_steps']} reward={summary['total_reward']:.3f}"
            )
    finally:
        game.close()

    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "sauerkrautlm_doom_multivec_31k_replay",
        "source_dataset_url": "https://huggingface.co/datasets/VAGOsolutions/SauerkrautLM-Doom-MultiVec-31k",
        "external_dataset": args.dataset_name,
        "external_split": args.split,
        "env_name": "vizdoom",
        "scenario": args.scenario,
        "map": args.map,
        "fps": game_context["doom_tics_per_second"],
        "frame_skip": args.frame_skip,
        "episode_count": len(summaries),
        "max_steps": args.episode_steps,
        "buttons": game_context["button_order"],
        "action_score_order": list(ACTION_SCORE_ORDER),
        "action_threshold": args.action_threshold,
        "argmax_if_empty": bool(args.argmax_if_empty),
        "enabled_buffers": {
            "rgb": True,
            "depth": bool(args.save_depth),
            "labels": bool(args.save_labels),
            "automap": bool(args.save_automap),
        },
        "generation_mode": "human_action_replay",
        "policy": POLICY_NAME,
        "player_id": "external_human",
        "session_id": args.run_id,
        "replay_limitations": [
            "The public dataset provides human action scores but not original pose trajectories.",
            "This run replays those actions from fresh ViZDoom episode starts, so poses are replay-derived GT, not recovered original human poses.",
            "Rows are chunked sequentially because the public dataset split does not expose original episode IDs.",
        ],
        "episodes": [
            {
                "episode_id": item["episode_id"],
                "steps_path": f"episodes/{item['episode_id']}/steps.jsonl",
                "summary_path": f"episodes/{item['episode_id']}/summary.json",
            }
            for item in summaries
        ],
        "episode_summaries": summaries,
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote replay-derived WIT-VZ raw run to: {run_dir}")
    return run_dir


def main() -> None:
    args = parse_args()
    raw_dir = replay_dataset(args)
    if args.build_samples:
        processed_out = args.processed_out or Path("data/wit_vz/processed") / args.run_id
        samples = build_samples(
            raw_dirs=[raw_dir],
            out_dir=processed_out,
            history_sec=args.history_sec,
            future_sec=args.future_sec,
            sample_fps=args.sample_fps,
            stride=args.stride,
            seed=args.seed,
            split_strategy=args.sample_split,
            preview_count=args.preview_count,
        )
        print(f"Wrote {len(samples)} replay-derived processed samples to: {processed_out}")


if __name__ == "__main__":
    main()
