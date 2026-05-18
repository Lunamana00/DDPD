"""Collect route-conditioned ViZDoom rollout data.

The collector writes one episode directory per rollout. Each step contains an
RGB frame plus JSON metadata with pose, action, reward, route distances, and
route labels derived from a route specification file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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


@dataclass(frozen=True)
class Route:
    route_id: str
    waypoints: list[tuple[float, float]]
    description: str = ""


@dataclass(frozen=True)
class Action:
    action_id: int
    name: str
    vector: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect route-conditioned ViZDoom rollouts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/route_vizdoom/runs"),
        help="Directory where the run folder will be written.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--route-spec",
        type=Path,
        default=Path("configs/vizdoom_route_specs/deadly_corridor_three_lanes.json"),
        help="Route specification JSON file.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help=(
            "Built-in scenario name such as deadly_corridor, basic, or a .wad path. "
            "Defaults to route_spec.scenario."
        ),
    )
    parser.add_argument("--map", type=str, default=None, help="Doom map name.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--policy",
        choices=["random", "forward_bias", "attack_bias"],
        default="forward_bias",
    )
    parser.add_argument("--visible", action="store_true", help="Show game window.")
    parser.add_argument(
        "--screen-width",
        type=int,
        default=160,
        help="Requested screen width. ViZDoom will use the closest enum value.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=120,
        help="Requested screen height. ViZDoom will use the closest enum value.",
    )
    parser.add_argument("--save-depth", action="store_true", default=True)
    parser.add_argument("--no-save-depth", dest="save_depth", action="store_false")
    parser.add_argument("--save-labels", action="store_true", default=True)
    parser.add_argument("--no-save-labels", dest="save_labels", action="store_false")
    parser.add_argument("--save-automap", action="store_true", default=True)
    parser.add_argument("--no-save-automap", dest="save_automap", action="store_false")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the run directory if it already exists.",
    )
    return parser.parse_args()


def load_route_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if "routes" not in spec or not spec["routes"]:
        raise ValueError(f"Route spec has no routes: {path}")
    return spec


def parse_routes(spec: dict[str, Any]) -> list[Route]:
    routes: list[Route] = []
    for raw in spec["routes"]:
        waypoints = [tuple(map(float, point[:2])) for point in raw["waypoints"]]
        if len(waypoints) < 2:
            raise ValueError(f"Route {raw.get('id')} must have at least two waypoints")
        routes.append(
            Route(
                route_id=str(raw["id"]),
                waypoints=waypoints,
                description=str(raw.get("description", "")),
            )
        )
    return routes


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def ensure_rgb_hwc(array: np.ndarray) -> np.ndarray:
    frame = np.asarray(array)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (
        1,
        3,
        4,
    ):
        frame = np.moveaxis(frame, 0, -1)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    return np.clip(frame, 0, 255).astype(np.uint8)


def save_png(array: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save PNG frames: uv pip install pillow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_rgb_hwc(array)).save(path)


def save_npz(array: np.ndarray | None, path: Path) -> str | None:
    if array is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=np.asarray(array))
    return path.as_posix()


def point_segment_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> tuple[float, float]:
    px, py = point
    sx, sy = start
    ex, ey = end
    vx, vy = ex - sx, ey - sy
    wx, wy = px - sx, py - sy
    length_sq = vx * vx + vy * vy
    if length_sq == 0.0:
        return math.hypot(px - sx, py - sy), 0.0
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    proj_x = sx + t * vx
    proj_y = sy + t * vy
    return math.hypot(px - proj_x, py - proj_y), t


def route_length(waypoints: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(waypoints[:-1], waypoints[1:])
    )


def route_metric(point: tuple[float, float], route: Route) -> dict[str, Any]:
    best_distance = float("inf")
    best_progress = 0.0
    traversed = 0.0
    total_length = max(route_length(route.waypoints), 1e-6)

    for start, end in zip(route.waypoints[:-1], route.waypoints[1:]):
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        distance, segment_t = point_segment_distance(point, start, end)
        progress = traversed + segment_t * segment_length
        if distance < best_distance:
            best_distance = distance
            best_progress = progress
        traversed += segment_length

    return {
        "route_id": route.route_id,
        "distance": best_distance,
        "progress": best_progress,
        "progress_norm": best_progress / total_length,
    }


def route_metrics(point: tuple[float, float], routes: list[Route]) -> list[dict[str, Any]]:
    metrics = [route_metric(point, route) for route in routes]
    return sorted(metrics, key=lambda item: item["distance"])


def choose_episode_route(
    records: list[dict[str, Any]], ignore_first_steps: int, max_distance: float | None
) -> str | None:
    counts: dict[str, int] = {}
    final_nearest: str | None = None
    for record in records:
        route_id = record.get("nearest_route_id")
        if not route_id:
            continue
        final_nearest = route_id
        if record["step"] < ignore_first_steps:
            continue
        distance = record.get("nearest_route_distance")
        if max_distance is not None and distance is not None and distance > max_distance:
            continue
        counts[route_id] = counts.get(route_id, 0) + 1
    if counts:
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return final_nearest


def make_actions() -> list[Action]:
    # Vectors follow BUTTON_NAMES order.
    names_and_vectors = [
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
    ]
    return [Action(i, name, vector) for i, (name, vector) in enumerate(names_and_vectors)]


def choose_action(actions: list[Action], policy: str, rng: random.Random) -> Action:
    if policy == "random":
        return rng.choice(actions)

    if policy == "attack_bias" and rng.random() < 0.20:
        return actions[-1]

    if policy in {"forward_bias", "attack_bias"}:
        weights = []
        for action in actions:
            weight = 1.0
            if "MOVE_FORWARD" in action.name:
                weight += 4.0
            if "TURN" in action.name:
                weight += 1.5
            if action.name == "ATTACK":
                weight += 1.0 if policy == "attack_bias" else 0.25
            weights.append(weight)
        return rng.choices(actions, weights=weights, k=1)[0]

    raise ValueError(f"Unknown policy: {policy}")


def get_screen_resolution(vzd: Any, width: int, height: int) -> Any:
    preferred = f"RES_{width}X{height}"
    if hasattr(vzd.ScreenResolution, preferred):
        return getattr(vzd.ScreenResolution, preferred)

    candidates = [
        "RES_160X120",
        "RES_320X240",
        "RES_640X480",
        "RES_800X600",
    ]
    for name in candidates:
        if hasattr(vzd.ScreenResolution, name):
            return getattr(vzd.ScreenResolution, name)
    raise RuntimeError("No supported ViZDoom screen resolution enum found")


def scenario_path(vzd: Any, scenario: str) -> str:
    path = Path(scenario)
    if path.suffix.lower() == ".wad" or path.exists():
        return str(path)
    return str(Path(vzd.scenarios_path) / f"{scenario}.wad")


def build_game(
    args: argparse.Namespace, spec: dict[str, Any], available_actions: list[Action]
) -> tuple[Any, dict[str, Any]]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("ViZDoom is required: uv pip install vizdoom") from exc

    game = vzd.DoomGame()
    selected_scenario = args.scenario or spec.get("scenario") or "deadly_corridor"
    selected_map = args.map or spec.get("map") or "map01"

    game.set_doom_scenario_path(scenario_path(vzd, selected_scenario))
    game.set_doom_map(selected_map)
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

    buttons = []
    for name in BUTTON_NAMES:
        button = getattr(vzd.Button, name, None)
        if button is not None:
            buttons.append(button)
    game.set_available_buttons(buttons)

    variables = []
    variable_names: list[str] = []
    for name in GAME_VARIABLE_NAMES:
        variable = getattr(vzd.GameVariable, name, None)
        if variable is not None:
            variables.append(variable)
            variable_names.append(name)
    game.set_available_game_variables(variables)
    game.init()
    context = {
        "variable_names": variable_names,
        "scenario": selected_scenario,
        "map": selected_map,
        "buttons": BUTTON_NAMES[: len(buttons)],
        "actions": available_actions,
    }
    return game, context


def labels_to_dicts(labels: Any) -> list[dict[str, Any]]:
    output = []
    attrs = [
        "value",
        "object_name",
        "object_position_x",
        "object_position_y",
        "object_position_z",
        "object_angle",
        "object_pitch",
        "object_roll",
        "object_velocity_x",
        "object_velocity_y",
        "object_velocity_z",
    ]
    for label in labels or []:
        output.append({attr: to_jsonable(getattr(label, attr)) for attr in attrs if hasattr(label, attr)})
    return output


def collect_episode(
    game: Any,
    game_context: dict[str, Any],
    episode_index: int,
    global_step_start: int,
    run_dir: Path,
    routes: list[Route],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = run_dir / "episodes" / episode_id
    frame_dir = episode_dir / "frames"
    depth_dir = episode_dir / "depth"
    labels_dir = episode_dir / "labels"
    automap_dir = episode_dir / "automap"
    frame_dir.mkdir(parents=True, exist_ok=True)

    actions = game_context["actions"]
    variable_names = game_context["variable_names"]
    records: list[dict[str, Any]] = []
    global_step = global_step_start

    game.new_episode()
    step = 0
    while not game.is_episode_finished() and step < args.max_steps:
        state = game.get_state()
        if state is None:
            break

        frame_name = f"{step:06d}"
        frame_rel = Path("episodes") / episode_id / "frames" / f"{frame_name}.png"
        depth_rel = Path("episodes") / episode_id / "depth" / f"{frame_name}.npz"
        labels_rel = Path("episodes") / episode_id / "labels" / f"{frame_name}.npz"
        automap_rel = Path("episodes") / episode_id / "automap" / f"{frame_name}.png"

        save_png(state.screen_buffer, run_dir / frame_rel)
        saved_depth_path = save_npz(
            getattr(state, "depth_buffer", None) if args.save_depth else None,
            run_dir / depth_rel,
        )
        saved_labels_path = save_npz(
            getattr(state, "labels_buffer", None) if args.save_labels else None,
            run_dir / labels_rel,
        )
        depth_path = depth_rel.as_posix() if saved_depth_path else None
        labels_path = labels_rel.as_posix() if saved_labels_path else None
        automap_path = None
        automap_buffer = getattr(state, "automap_buffer", None)
        if args.save_automap and automap_buffer is not None:
            save_png(automap_buffer, run_dir / automap_rel)
            automap_path = automap_rel.as_posix()

        variable_values = {
            name: to_jsonable(value)
            for name, value in zip(variable_names, state.game_variables)
        }
        pose = {
            "x": variable_values.get("POSITION_X"),
            "y": variable_values.get("POSITION_Y"),
            "z": variable_values.get("POSITION_Z"),
            "angle": variable_values.get("ANGLE"),
        }

        metrics: list[dict[str, Any]] = []
        nearest_route_id = None
        nearest_distance = None
        if pose["x"] is not None and pose["y"] is not None:
            metrics = route_metrics((float(pose["x"]), float(pose["y"])), routes)
            nearest_route_id = metrics[0]["route_id"]
            nearest_distance = metrics[0]["distance"]

        action = choose_action(actions, args.policy, rng)
        reward = game.make_action(action.vector, args.frame_skip)
        done = game.is_episode_finished()

        records.append(
            {
                "sample_id": f"{episode_id}_{step:06d}",
                "episode_id": episode_id,
                "episode_index": episode_index,
                "step": step,
                "global_step": global_step,
                "frame_path": frame_rel.as_posix(),
                "depth_path": depth_path,
                "labels_path": labels_path,
                "automap_path": automap_path,
                "pose": pose,
                "game_variables": variable_values,
                "visible_labels": labels_to_dicts(getattr(state, "labels", None)),
                "candidate_route_ids": [route.route_id for route in routes],
                "route_metrics": metrics,
                "nearest_route_id": nearest_route_id,
                "nearest_route_distance": nearest_distance,
                "action": {
                    "action_id": action.action_id,
                    "action_name": action.name,
                    "action_vector": action.vector,
                    "button_order": BUTTON_NAMES,
                },
                "reward": float(reward),
                "done": bool(done),
                "chosen_route_id": None,
            }
        )
        step += 1
        global_step += 1

    return records, global_step


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    spec = load_route_spec(args.route_spec)
    routes = parse_routes(spec)
    actions = make_actions()

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "route_spec.json", spec)

    game, game_context = build_game(args, spec, actions)
    global_step = 0
    all_episode_summaries = []
    label_cfg = spec.get("labeling", {})
    ignore_first_steps = int(label_cfg.get("ignore_first_steps", 10))
    max_route_distance = label_cfg.get("max_route_distance")
    if max_route_distance is not None:
        max_route_distance = float(max_route_distance)

    try:
        for episode_index in range(1, args.episodes + 1):
            records, global_step = collect_episode(
                game=game,
                game_context=game_context,
                episode_index=episode_index,
                global_step_start=global_step,
                run_dir=run_dir,
                routes=routes,
                args=args,
                rng=rng,
            )
            chosen_route_id = choose_episode_route(records, ignore_first_steps, max_route_distance)
            for record in records:
                record["chosen_route_id"] = chosen_route_id
                record["route_labeling"] = {
                    "strategy": "nearest-route-majority",
                    "ignore_first_steps": ignore_first_steps,
                    "max_route_distance": max_route_distance,
                }

            episode_dir = run_dir / "episodes" / f"episode_{episode_index:06d}"
            write_jsonl(episode_dir / "steps.jsonl", records)
            summary = {
                "episode_id": f"episode_{episode_index:06d}",
                "num_steps": len(records),
                "total_reward": sum(record["reward"] for record in records),
                "chosen_route_id": chosen_route_id,
                "done": records[-1]["done"] if records else False,
                "last_step": records[-1]["step"] if records else None,
            }
            write_json(episode_dir / "summary.json", summary)
            all_episode_summaries.append(summary)
            print(
                f"{summary['episode_id']}: steps={summary['num_steps']} "
                f"reward={summary['total_reward']:.3f} chosen_route={chosen_route_id}"
            )
    finally:
        game.close()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ViZDoom rollout",
        "scenario": game_context["scenario"],
        "map": game_context["map"],
        "route_spec_path": args.route_spec.as_posix(),
        "run_dir": run_dir.as_posix(),
        "episodes": all_episode_summaries,
        "num_episodes": len(all_episode_summaries),
        "num_steps": global_step,
        "button_order": BUTTON_NAMES,
        "game_variable_names": game_context["variable_names"],
        "observation_files": {
            "rgb": "episodes/<episode_id>/frames/<step>.png",
            "depth": "episodes/<episode_id>/depth/<step>.npz",
            "labels": "episodes/<episode_id>/labels/<step>.npz",
            "automap": "episodes/<episode_id>/automap/<step>.png",
            "metadata": "episodes/<episode_id>/steps.jsonl",
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote route-conditioned dataset run to: {run_dir}")


if __name__ == "__main__":
    main()
