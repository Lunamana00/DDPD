"""Collect WIT-VZ raw path-prediction episodes from ViZDoom."""

from __future__ import annotations

import argparse
import math
import random
import shutil
import time
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
    parser.add_argument(
        "--policy",
        choices=[
            "corridor",
            "random",
            "random_walk",
            "noisy_corridor",
            "goal_directed",
            "obstacle_avoidance",
            "mixed",
        ],
        default="corridor",
    )
    parser.add_argument(
        "--policy-mix",
        nargs="+",
        default=["corridor", "random_walk", "noisy_corridor", "goal_directed", "obstacle_avoidance"],
        choices=["corridor", "random", "random_walk", "noisy_corridor", "goal_directed", "obstacle_avoidance"],
        help="Episode-level policy choices used when --policy mixed.",
    )
    parser.add_argument("--policy-noise", type=float, default=0.05)
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument(
        "--start-random-steps",
        type=int,
        default=0,
        help="Random warmup actions before recording each episode to diversify the initial pose.",
    )
    parser.add_argument(
        "--start-random-jitter",
        type=int,
        default=0,
        help="Additional random warmup steps sampled uniformly from [0, jitter].",
    )
    parser.add_argument("--player-id", default=None, help="Optional anonymized player id for human sessions.")
    parser.add_argument("--session-id", default=None, help="Optional human recording session id.")
    parser.add_argument(
        "--human-countdown-sec",
        type=float,
        default=3.0,
        help="Seconds to wait after opening the ViZDoom window in human mode.",
    )
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

    game = vzd.DoomGame()
    game.set_doom_scenario_path(scenario_path(vzd, args.scenario))
    game.set_doom_map(args.map)
    game.set_window_visible(bool(args.visible or args.mode == "human"))
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
    game.set_episode_timeout(args.max_steps * args.frame_skip)
    game.set_episode_start_time(10)
    game.set_living_reward(0.0)
    game.set_mode(vzd.Mode.SPECTATOR if args.mode == "human" else vzd.Mode.PLAYER)

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
    if args.mode == "human" and args.human_countdown_sec > 0:
        print(
            f"Human mode: focus the ViZDoom window. "
            f"Recording starts in {args.human_countdown_sec:.1f}s."
        )
        time.sleep(args.human_countdown_sec)
    return game, {
        "button_order": button_names,
        "game_variable_names": variable_names,
        "doom_tics_per_second": 35.0,
    }


def _angle_delta_degrees(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _default_goal_for_episode(episode_index: int) -> tuple[float, float]:
    goals = [
        (384.0, 0.0),
        (-384.0, 0.0),
        (0.0, 384.0),
        (0.0, -384.0),
        (256.0, 256.0),
        (-256.0, -256.0),
    ]
    return goals[(episode_index - 1) % len(goals)]


def _maybe_noisy_action(rng: random.Random, policy_noise: float) -> tuple[int, str, list[int]] | None:
    if policy_noise > 0.0 and rng.random() < policy_noise:
        action_id = rng.randrange(len(ACTION_SPACE))
        name, vector = ACTION_SPACE[action_id]
        return action_id, name, vector
    return None


def choose_scripted_action(
    rng: random.Random,
    policy: str,
    step: int,
    pose: dict[str, float] | None = None,
    goal: tuple[float, float] | None = None,
    policy_noise: float = 0.0,
) -> tuple[int, str, list[int]]:
    noisy = _maybe_noisy_action(rng, policy_noise)
    if noisy is not None:
        action_id, name, vector = noisy
        return action_id, f"{name}[noise]", vector

    if policy == "random":
        action_id = rng.randrange(len(ACTION_SPACE))
        name, vector = ACTION_SPACE[action_id]
        return action_id, name, vector

    if policy == "random_walk":
        action_id = rng.choice([0, 1, 3, 4, 6, 7, 8, 9, 10, 18])
        name, vector = ACTION_SPACE[action_id]
        return action_id, name, vector

    if policy == "goal_directed" and pose is not None and goal is not None:
        dx = goal[0] - float(pose.get("x", 0.0))
        dy = goal[1] - float(pose.get("y", 0.0))
        target_angle = math.degrees(math.atan2(dy, dx))
        delta = _angle_delta_degrees(target_angle, float(pose.get("angle", 0.0)))
        if abs(delta) > 55.0:
            action_id = 5 if delta > 0.0 else 4
        elif abs(delta) > 18.0:
            action_id = 9 if delta > 0.0 else 10
        else:
            action_id = rng.choice([8, 11, 14])
        name, vector = ACTION_SPACE[action_id]
        return action_id, f"{name}[goal_directed]", vector

    if policy == "obstacle_avoidance":
        phase = step % 24
        if phase in {0, 1, 2}:
            action_id = rng.choice([9, 10, 15, 16])
        elif phase in {8, 9, 10}:
            action_id = rng.choice([3, 4, 6, 7])
        elif rng.random() < 0.12:
            action_id = rng.choice([0, 1, 17, 18])
        else:
            action_id = rng.choice([8, 9, 10, 11, 14])
        name, vector = ACTION_SPACE[action_id]
        return action_id, f"{name}[obstacle_avoidance]", vector

    if policy == "noisy_corridor":
        policy_noise = max(policy_noise, 0.18)
        noisy = _maybe_noisy_action(rng, policy_noise)
        if noisy is not None:
            action_id, name, vector = noisy
            return action_id, f"{name}[noise]", vector

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


def action_id_for_vector(action_vector: list[int]) -> int | None:
    for action_id, (_name, vector) in enumerate(ACTION_SPACE):
        if vector == action_vector:
            return action_id
    return None


def action_name_from_vector(action_vector: list[int], button_order: list[str]) -> str:
    active = [name for name, value in zip(button_order, action_vector) if value]
    return "+".join(active) if active else "PAUSE"


def step_environment(
    game: Any,
    game_context: dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
    step: int,
    policy: str | None = None,
    pose: dict[str, float] | None = None,
    goal: tuple[float, float] | None = None,
) -> tuple[int | None, str, list[int], float]:
    if args.mode == "human":
        game.advance_action(args.frame_skip)
        raw_action = game.get_last_action()
        action_vector = [int(round(float(value))) for value in raw_action]
        action_name = action_name_from_vector(action_vector, game_context["button_order"])
        return action_id_for_vector(action_vector), action_name, action_vector, float(game.get_last_reward())

    action_id, action_name, action_vector = choose_scripted_action(
        rng,
        policy or args.policy,
        step,
        pose=pose,
        goal=goal,
        policy_noise=args.policy_noise,
    )
    reward = float(game.make_action(action_vector, args.frame_skip))
    return action_id, action_name, action_vector, reward


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
    episode_policy = args.policy
    if args.policy == "mixed":
        episode_policy = rng.choice(args.policy_mix)
    episode_goal = None
    if args.goal_x is not None and args.goal_y is not None:
        episode_goal = (float(args.goal_x), float(args.goal_y))
    elif episode_policy == "goal_directed":
        episode_goal = _default_goal_for_episode(episode_index)
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    global_step = global_step_start
    prev_pose = None

    game.new_episode()
    warmup_steps = max(0, args.start_random_steps)
    if args.start_random_jitter > 0:
        warmup_steps += rng.randrange(args.start_random_jitter + 1)
    for warmup_step in range(warmup_steps):
        if game.is_episode_finished():
            break
        _action_id, _action_name, action_vector = choose_scripted_action(
            rng,
            "random_walk",
            warmup_step,
            policy_noise=max(args.policy_noise, 0.15),
        )
        game.make_action(action_vector, args.frame_skip)

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

        action_id, action_name, action_vector, reward = step_environment(
            game,
            game_context,
            args,
            rng,
            step,
            policy=episode_policy,
            pose=pose,
            goal=episode_goal,
        )
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
                    "policy": episode_policy,
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
        "policy": episode_policy,
        "goal": {"x": episode_goal[0], "y": episode_goal[1]} if episode_goal else None,
        "warmup_steps": warmup_steps,
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
        "source_dataset": "wit_vz",
        "env_name": "vizdoom",
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
        "player_id": args.player_id,
        "session_id": args.session_id,
        "policy_mix": args.policy_mix if args.policy == "mixed" else None,
        "policy_noise": args.policy_noise,
        "goal": {"x": args.goal_x, "y": args.goal_y} if args.goal_x is not None and args.goal_y is not None else None,
        "start_random_steps": args.start_random_steps,
        "start_random_jitter": args.start_random_jitter,
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
