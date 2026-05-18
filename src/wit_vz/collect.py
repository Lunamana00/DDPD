"""Collect WIT-VZ raw path-prediction episodes from ViZDoom."""

from __future__ import annotations

import argparse
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .geometry import compute_relative_egomotion
from .io import save_npz, save_png, to_jsonable, write_json, write_jsonl


BUTTON_NAMES = [
    "ATTACK",
    "MOVE_FORWARD",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "TURN_RIGHT",
    "TURN_LEFT",
]

GAME_VARIABLE_NAMES = [
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "ANGLE",
    "HEALTH",
    "ARMOR",
    "AMMO2",
    "KILLCOUNT",
    "DEATHCOUNT",
    "HITCOUNT",
    "DAMAGECOUNT",
    "SELECTED_WEAPON_AMMO",
]


ACTION_SPACE = [
    ("TURN_LEFT", [0, 0, 0, 0, 0, 1]),
    ("TURN_RIGHT", [0, 0, 0, 0, 1, 0]),
    ("MOVE_RIGHT", [0, 0, 1, 0, 0, 0]),
    ("MOVE_RIGHT+TURN_LEFT", [0, 0, 1, 0, 0, 1]),
    ("MOVE_RIGHT+TURN_RIGHT", [0, 0, 1, 0, 1, 0]),
    ("MOVE_LEFT", [0, 0, 0, 1, 0, 0]),
    ("MOVE_LEFT+TURN_LEFT", [0, 0, 0, 1, 0, 1]),
    ("MOVE_LEFT+TURN_RIGHT", [0, 0, 0, 1, 1, 0]),
    ("MOVE_FORWARD", [0, 1, 0, 0, 0, 0]),
    ("MOVE_FORWARD+TURN_LEFT", [0, 1, 0, 0, 0, 1]),
    ("MOVE_FORWARD+TURN_RIGHT", [0, 1, 0, 0, 1, 0]),
    ("MOVE_FORWARD+MOVE_RIGHT", [0, 1, 1, 0, 0, 0]),
    ("MOVE_FORWARD+MOVE_RIGHT+TURN_LEFT", [0, 1, 1, 0, 0, 1]),
    ("MOVE_FORWARD+MOVE_RIGHT+TURN_RIGHT", [0, 1, 1, 0, 1, 0]),
    ("MOVE_FORWARD+MOVE_LEFT", [0, 1, 0, 1, 0, 0]),
    ("MOVE_FORWARD+MOVE_LEFT+TURN_LEFT", [0, 1, 0, 1, 0, 1]),
    ("MOVE_FORWARD+MOVE_LEFT+TURN_RIGHT", [0, 1, 0, 1, 1, 0]),
    ("ATTACK", [1, 0, 0, 0, 0, 0]),
    ("PAUSE", [0, 0, 0, 0, 0, 0]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect WIT-VZ raw ViZDoom path data.")
    parser.add_argument("--scenario", default="deadly_corridor")
    parser.add_argument("--map", default="map01")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", type=Path, default=Path("data/wit_vz/raw"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["scripted", "human"], default="scripted")
    parser.add_argument("--policy", choices=["corridor", "random"], default="corridor")
    parser.add_argument("--screen-width", type=int, default=160)
    parser.add_argument("--screen-height", type=int, default=120)
    parser.add_argument("--save-rgb", action="store_true", default=True)
    parser.add_argument("--no-save-rgb", dest="save_rgb", action="store_false")
    parser.add_argument("--save-depth", action="store_true", default=False)
    parser.add_argument("--save-labels", action="store_true", default=False)
    parser.add_argument("--save-automap", action="store_true", default=False)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scenario_path(vzd: Any, scenario: str) -> str:
    path = Path(scenario)
    if path.suffix.lower() == ".wad" or path.exists():
        return str(path)
    return str(Path(vzd.scenarios_path) / f"{scenario}.wad")


def get_screen_resolution(vzd: Any, width: int, height: int) -> Any:
    preferred = f"RES_{width}X{height}"
    if hasattr(vzd.ScreenResolution, preferred):
        return getattr(vzd.ScreenResolution, preferred)
    for name in ("RES_160X120", "RES_320X240", "RES_640X480"):
        if hasattr(vzd.ScreenResolution, name):
            return getattr(vzd.ScreenResolution, name)
    raise RuntimeError("No supported ViZDoom screen resolution enum found")


def labels_to_dicts(labels: Any) -> list[dict[str, Any]]:
    attrs = [
        "value",
        "object_name",
        "object_position_x",
        "object_position_y",
        "object_position_z",
        "object_angle",
        "object_velocity_x",
        "object_velocity_y",
        "object_velocity_z",
    ]
    output = []
    for label in labels or []:
        output.append({attr: to_jsonable(getattr(label, attr)) for attr in attrs if hasattr(label, attr)})
    return output


def build_game(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("ViZDoom is required. Install with: uv pip install vizdoom pillow") from exc

    if args.mode == "human":
        raise NotImplementedError(
            "Human recording mode is scaffolded but not implemented for this environment. "
            "Use --mode scripted for real data collection."
        )

    game = vzd.DoomGame()
    game.set_doom_scenario_path(scenario_path(vzd, args.scenario))
    game.set_doom_map(args.map)
    game.set_window_visible(bool(args.visible))
    game.set_screen_resolution(get_screen_resolution(vzd, args.screen_width, args.screen_height))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(bool(args.save_depth))
    game.set_labels_buffer_enabled(bool(args.save_labels))
    game.set_automap_buffer_enabled(bool(args.save_automap))
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_episode_timeout(args.max_steps * args.frame_skip)
    game.set_episode_start_time(10)
    game.set_living_reward(0.0)
    game.set_mode(vzd.Mode.PLAYER)

    buttons = [getattr(vzd.Button, name) for name in BUTTON_NAMES if hasattr(vzd.Button, name)]
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
        "button_order": BUTTON_NAMES[: len(buttons)],
        "game_variable_names": variable_names,
        "doom_tics_per_second": 35.0,
    }


def choose_scripted_action(rng: random.Random, policy: str, step: int) -> tuple[int, str, list[int]]:
    if policy == "random":
        action_id = rng.randrange(len(ACTION_SPACE))
        name, vector = ACTION_SPACE[action_id]
        return action_id, name, vector

    # Corridor policy: mostly moves forward, occasionally attacks, strafes, or
    # turns. The randomness is intentional so the future path target is not a
    # trivial straight line.
    if step % rng.randint(9, 18) == 0 and rng.random() < 0.55:
        candidates = [17]  # ATTACK
    else:
        candidates = [8, 9, 10, 11, 14, 15, 16, 18]
        if rng.random() < 0.20:
            candidates.extend([3, 4, 6, 7])
        if rng.random() < 0.08:
            candidates.append(18)  # PAUSE
    action_id = rng.choice(candidates)
    name, vector = ACTION_SPACE[action_id]
    return action_id, name, vector


def pose_from_variables(variable_values: dict[str, float]) -> dict[str, float]:
    return {
        "x": float(variable_values.get("POSITION_X", 0.0)),
        "y": float(variable_values.get("POSITION_Y", 0.0)),
        "z": float(variable_values.get("POSITION_Z", 0.0)),
        "angle": float(variable_values.get("ANGLE", 0.0)),
    }


def collect_episode(
    game: Any,
    game_context: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    episode_index: int,
    rng: random.Random,
    global_step_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    global_step = global_step_start
    prev_pose = None

    game.new_episode()
    step = 0
    while not game.is_episode_finished() and step < args.max_steps:
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

        frame_path = None
        if args.save_rgb:
            save_png(state.screen_buffer, run_dir / frame_rel)
            frame_path = frame_rel.as_posix()

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

        action_id, action_name, action_vector = choose_scripted_action(rng, args.policy, step)
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
                "frame_path": frame_path,
                "depth_path": depth_rel.as_posix() if saved_depth else None,
                "labels_path": labels_rel.as_posix() if saved_labels else None,
                "automap_path": automap_path,
                "pose": pose,
                "relative_egomotion_from_prev": egomotion,
                "action": {
                    "action_id": action_id,
                    "action_name": action_name,
                    "action_vector": action_vector,
                    "button_order": game_context["button_order"],
                },
                "reward": reward,
                "done": done,
                "visible_labels": labels_to_dicts(getattr(state, "labels", None)),
                "game_variables": variables,
            }
        )
        prev_pose = pose
        step += 1
        global_step += 1

    summary = {
        "episode_id": episode_id,
        "num_steps": len(records),
        "total_reward": total_reward,
        "start_pose": records[0]["pose"] if records else None,
        "final_pose": records[-1]["pose"] if records else None,
        "done": records[-1]["done"] if records else False,
        "reason_end": "done" if records and records[-1]["done"] else "max_steps_or_no_state",
    }
    write_jsonl(episode_dir / "steps.jsonl", records)
    write_json(episode_dir / "summary.json", summary)
    return records, summary, global_step


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("wit_vz_%Y%m%dT%H%M%SZ")
    run_dir = args.out_root / run_id
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    game, game_context = build_game(args)
    summaries = []
    global_step = 0
    try:
        for episode_index in range(1, args.episodes + 1):
            episode_rng = random.Random(args.seed + episode_index * 1009)
            _records, summary, global_step = collect_episode(
                game, game_context, args, run_dir, episode_index, episode_rng, global_step
            )
            summaries.append(summary)
            print(
                f"{summary['episode_id']}: steps={summary['num_steps']} "
                f"reward={summary['total_reward']:.3f}"
            )
    finally:
        game.close()

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "map": args.map,
        "fps": game_context["doom_tics_per_second"],
        "frame_skip": args.frame_skip,
        "episode_count": len(summaries),
        "max_steps": args.max_steps,
        "buttons": game_context["button_order"],
        "action_space": [
            {"action_id": i, "action_name": name, "action_vector": vector}
            for i, (name, vector) in enumerate(ACTION_SPACE)
        ],
        "enabled_buffers": {
            "rgb": bool(args.save_rgb),
            "depth": bool(args.save_depth),
            "labels": bool(args.save_labels),
            "automap": bool(args.save_automap),
        },
        "generation_mode": args.mode,
        "policy": args.policy,
        "seed": args.seed,
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
    print(f"Wrote WIT-VZ raw run to: {run_dir}")


if __name__ == "__main__":
    main()
