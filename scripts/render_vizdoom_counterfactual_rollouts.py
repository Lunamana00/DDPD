"""Render demo-grade ViZDoom counterfactual rollouts for predicted paths."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render_paper_baseline_overlay_video import XU_COLOR, compute_xu_path  # noqa: E402
from render_triptych_demo_video import (  # noqa: E402
    BG_COLOR,
    CV_COLOR,
    GT_COLOR,
    MUTED_COLOR,
    PRED_COLOR,
    TEXT_COLOR,
    axis_scale,
    clipped_path,
    draw_path_plot,
    font,
    load_raw_dirs,
    paste_center,
    path_error,
    read_json,
    read_jsonl,
    selected_ids,
    wrap,
)
from src.wit_vz.collect import BUTTON_NAMES, GAME_VARIABLE_NAMES, get_screen_resolution, scenario_path  # noqa: E402
from src.wit_vz.geometry import world_delta_to_local, wrap_degrees  # noqa: E402


METHODS = [
    ("cv", "CV baseline", "recent-motion extrapolation", CV_COLOR),
    ("xu", "Xu paper baseline", "pixels-only saliency steering", XU_COLOR),
    ("target", "GT", "future local path label", GT_COLOR),
    ("prediction", "Ours", "visual cue-memory output", PRED_COLOR),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("runs/episodic_memory_ablation_v4/seed_7/03s/long_attention_no_ego/predictions.jsonl"),
    )
    parser.add_argument("--summary", type=Path, default=Path("reports/demo/vizdoom_multi_scenario_03s/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/demo/presentation_sequence/demo_vizdoom_counterfactual_rollout.mp4"))
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--screen-width", type=int, default=320)
    parser.add_argument("--screen-height", type=int, default=240)
    parser.add_argument("--hold-last-frames", type=int, default=8)
    parser.add_argument("--position-threshold", type=float, default=5.0)
    parser.add_argument("--angle-threshold", type=float, default=10.0)
    parser.add_argument("--raw-root-base", action="append", default=[])
    return parser.parse_args()


def source_id(sample: dict[str, Any]) -> str | None:
    return sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")


def load_selected_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_dirs = load_raw_dirs(args.dataset, [ROOT, *[Path(item) for item in args.raw_root_base]])
    samples = {str(row["sample_id"]): row for row in read_jsonl(args.dataset / "samples.jsonl")}
    predictions = {str(row["sample_id"]): row for row in read_jsonl(args.predictions)}
    selected = selected_ids(args.summary, 0)

    by_case: dict[str, list[tuple[str, str, str]]] = {"easy": [], "hard": [], "failure": []}
    for sample_id, label, case in selected:
        if case in by_case:
            by_case[case].append((sample_id, label, case))

    ordered: list[tuple[str, str, str]] = []
    rounds = max(len(rows) for rows in by_case.values())
    for idx in range(rounds):
        for case in ("easy", "hard", "failure"):
            if idx < len(by_case[case]):
                ordered.append(by_case[case][idx])
            if len(ordered) >= args.max_samples:
                break
        if len(ordered) >= args.max_samples:
            break

    items = []
    for sample_id, label, case in ordered:
        sample = samples.get(sample_id)
        prediction = predictions.get(sample_id)
        if sample is None or prediction is None:
            continue
        target = prediction.get("target") or sample.get("future_local_path") or []
        pred_path = prediction.get("prediction") or []
        cv_path = prediction.get("constant_velocity_prediction") or []
        sid = source_id(sample)
        xu_path = compute_xu_path(sample, raw_dirs, sid, target)
        ours_ade, ours_fde = path_error(pred_path, target)
        cv_ade, cv_fde = path_error(cv_path, target)
        xu_ade, xu_fde = path_error(xu_path, target)
        items.append(
            {
                "sample_id": sample_id,
                "label": label,
                "case": case,
                "sample": sample,
                "prediction_item": prediction,
                "target": target,
                "prediction": pred_path,
                "cv": cv_path,
                "xu": xu_path,
                "ADE": float(prediction.get("ADE", ours_ade)),
                "FDE": float(prediction.get("FDE", ours_fde)),
                "cv_ADE": float(prediction.get("constant_velocity_ADE", cv_ade)),
                "cv_FDE": float(prediction.get("constant_velocity_FDE", cv_fde)),
                "xu_ADE": xu_ade,
                "xu_FDE": xu_fde,
            }
        )
    return items


def load_raw_episode(sample: dict[str, Any], raw_dirs: dict[str, Path]) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    sid = source_id(sample)
    if sid is None or sid not in raw_dirs:
        raise KeyError(f"Missing raw dir for source_id={sid}")
    raw_root = raw_dirs[sid]
    manifest = read_json(raw_root / "manifest.json")
    steps_path = raw_root / str(sample["metadata"]["raw_episode_path"])
    return raw_root, manifest, read_jsonl(steps_path)


def build_game(manifest: dict[str, Any], screen_width: int, screen_height: int):
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.set_doom_scenario_path(scenario_path(vzd, str(manifest["scenario"])))
    game.set_doom_map(str(manifest.get("map", "map01")))
    game.set_window_visible(False)
    if hasattr(game, "set_sound_enabled"):
        game.set_sound_enabled(False)
    if hasattr(game, "set_music_enabled"):
        game.set_music_enabled(False)
    game.set_screen_resolution(get_screen_resolution(vzd, screen_width, screen_height))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_available_buttons([getattr(vzd.Button, name) for name in manifest.get("buttons", BUTTON_NAMES) if hasattr(vzd.Button, name)])
    game.set_available_game_variables([getattr(vzd.GameVariable, name) for name in GAME_VARIABLE_NAMES if hasattr(vzd.GameVariable, name)])
    game.set_episode_timeout(int(manifest.get("max_steps", 300)) * int(manifest.get("frame_skip", 4)))
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    game.new_episode()
    return game


def pose_from_game(game: Any) -> dict[str, float]:
    state = game.get_state()
    if state is None:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "angle": 0.0}
    names = [name for name in GAME_VARIABLE_NAMES if name in {"POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "HEALTH", "ARMOR", "AMMO2", "KILLCOUNT", "DEATHCOUNT", "HITCOUNT", "DAMAGECOUNT", "SELECTED_WEAPON_AMMO"}]
    values = {name: float(value) for name, value in zip(names, state.game_variables)}
    return {
        "x": values.get("POSITION_X", 0.0),
        "y": values.get("POSITION_Y", 0.0),
        "z": values.get("POSITION_Z", 0.0),
        "angle": values.get("ANGLE", 0.0),
    }


def pose_error(got: dict[str, float], want: dict[str, float]) -> tuple[float, float]:
    pos = math.hypot(float(got["x"]) - float(want["x"]), float(got["y"]) - float(want["y"]))
    angle = abs(wrap_degrees(float(got.get("angle", 0.0)) - float(want.get("angle", 0.0))))
    return pos, angle


def turn_vector(direction: str) -> list[int]:
    if direction == "right":
        return [0, 0, 0, 0, 1, 0]
    return [0, 0, 0, 0, 0, 1]


def align_by_warp_and_turn(game: Any, target_pose: dict[str, float]) -> None:
    game.send_game_command(f"warp {float(target_pose['x'])} {float(target_pose['y'])}")
    game.advance_action(1)
    for _ in range(240):
        current = pose_from_game(game)
        delta = wrap_degrees(float(target_pose.get("angle", 0.0)) - float(current.get("angle", 0.0)))
        if abs(delta) <= 3.6:
            break
        # In ViZDoom's ANGLE convention, TURN_LEFT increases ANGLE and
        # TURN_RIGHT decreases it. A two-tic action gives a small nonzero turn.
        game.make_action(turn_vector("left" if delta > 0.0 else "right"), 2)


def replay_to_center(game: Any, steps: list[dict[str, Any]], center_step: int, frame_skip: int) -> None:
    for row in steps[:center_step]:
        action = row.get("action", {}).get("action_vector")
        if action is None:
            break
        if game.is_episode_finished():
            break
        game.make_action([int(value) for value in action], frame_skip)


def prepare_start(
    manifest: dict[str, Any],
    steps: list[dict[str, Any]],
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    game = build_game(manifest, args.screen_width, args.screen_height)
    target_pose = sample["current_pose"]
    center_step = int(sample.get("center_step", 0))
    frame_skip = int(manifest.get("frame_skip", 4))
    replay_to_center(game, steps, center_step, frame_skip)
    replay_pose = pose_from_game(game)
    replay_pos, replay_angle = pose_error(replay_pose, target_pose)
    start_mode = "action_replay"
    if replay_pos > args.position_threshold or replay_angle > args.angle_threshold:
        start_mode = "warp_align_after_replay_mismatch"
        if game.is_episode_finished():
            game.close()
            game = build_game(manifest, args.screen_width, args.screen_height)
        align_by_warp_and_turn(game, target_pose)
    final_pose = pose_from_game(game)
    final_pos, final_angle = pose_error(final_pose, target_pose)
    if final_pos > args.position_threshold or final_angle > args.angle_threshold:
        game.close()
        raise RuntimeError(
            "could not align start pose: "
            f"pos_error={final_pos:.2f}, angle_error={final_angle:.2f}, mode={start_mode}"
        )
    info = {
        "start_mode": start_mode,
        "replay_position_error": replay_pos,
        "replay_angle_error": replay_angle,
        "final_position_error": final_pos,
        "final_angle_error": final_angle,
        "target_pose": target_pose,
        "start_pose": final_pose,
    }
    return game, info


def local_path_to_world(start_pose: dict[str, float], path: list[list[float]]) -> list[tuple[float, float]]:
    yaw = math.radians(float(start_pose.get("angle", 0.0)))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    origin_x = float(start_pose["x"])
    origin_y = float(start_pose["y"])
    world = []
    for forward, right in path:
        dx = cos_yaw * float(forward) - sin_yaw * float(right)
        dy = sin_yaw * float(forward) + cos_yaw * float(right)
        world.append((origin_x + dx, origin_y + dy))
    return world


def action_toward(current_pose: dict[str, float], target_xy: tuple[float, float]) -> list[int]:
    dx = float(target_xy[0]) - float(current_pose["x"])
    dy = float(target_xy[1]) - float(current_pose["y"])
    target_angle = math.degrees(math.atan2(dy, dx))
    delta = wrap_degrees(target_angle - float(current_pose.get("angle", 0.0)))
    action = [0, 0, 0, 0, 0, 0]
    if abs(delta) < 75.0:
        action[1] = 1
    if delta > 18.0:
        action[4] = 1
    elif delta < -18.0:
        action[5] = 1
    if action == [0, 0, 0, 0, 0, 0]:
        action[1] = 1
    return action


def capture_frame(game: Any) -> Image.Image:
    state = game.get_state()
    if state is None:
        return Image.new("RGB", (320, 240), (230, 234, 238))
    return Image.fromarray(np.asarray(state.screen_buffer, dtype=np.uint8)).convert("RGB")


def rollout_method(
    manifest: dict[str, Any],
    steps: list[dict[str, Any]],
    sample: dict[str, Any],
    path: list[list[float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    game, start_info = prepare_start(manifest, steps, sample, args)
    frames: list[Image.Image] = []
    realized: list[list[float]] = []
    start_pose = pose_from_game(game)
    waypoints = local_path_to_world(start_pose, path)
    frame_skip = int(manifest.get("frame_skip", 4))
    try:
        for step, waypoint in enumerate(waypoints):
            if game.is_episode_finished():
                break
            frames.append(capture_frame(game))
            current = pose_from_game(game)
            forward, right = world_delta_to_local(start_pose["x"], start_pose["y"], start_pose["angle"], current["x"], current["y"])
            realized.append([forward, right])
            action = action_toward(current, waypoint)
            game.make_action(action, frame_skip)
    finally:
        game.close()
    return {
        "frames": frames,
        "planned_path": path,
        "realized_path": realized,
        "start_info": start_info,
    }


def fit_image(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    max_w, max_h = max_size
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def draw_column(
    canvas: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    subtitle: str,
    rgb: Image.Image,
    path: list[list[float]],
    full_path: list[list[float]],
    color: tuple[int, int, int],
    max_abs: float,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill="white", outline=(216, 220, 226), width=2)
    draw.text((x + 16, y + 16), title, fill=color, font=font(24, bold=True))
    draw.text((x + 16, y + 48), subtitle, fill=MUTED_COLOR, font=font(14))
    frame = fit_image(rgb, (w - 36, 268))
    paste_center(canvas, frame, (x + 18, y + 78, x + w - 18, y + 346))
    plot = draw_path_plot(path, (w - 36, h - 382), color, max_abs, full_path)
    canvas.paste(plot, (x + 18, y + 364))


def render_composite_frame(
    item: dict[str, Any],
    rollouts: dict[str, dict[str, Any]],
    order: int,
    total: int,
    progress: int,
    args: argparse.Namespace,
) -> Image.Image:
    canvas = Image.new("RGB", (args.width, args.height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    header = f"{order:02d}/{total:02d}  ViZDoom counterfactual / {item['label']} / {item['case']}"
    draw.text((34, 26), header, fill=TEXT_COLOR, font=font(32, bold=True))
    metrics = (
        f"sample={item['sample_id']}    t={progress + 1:02d}    "
        f"ours={item['ADE']:.2f}/{item['FDE']:.2f}    "
        f"CV={item['cv_ADE']:.2f}/{item['cv_FDE']:.2f}    "
        f"Xu={item['xu_ADE']:.2f}/{item['xu_FDE']:.2f}"
    )
    draw.text((34, 72), wrap(metrics, 145), fill=MUTED_COLOR, font=font(18))
    col_gap = 16
    margin_x = 34
    top = 134
    col_w = (args.width - 2 * margin_x - 3 * col_gap) // 4
    col_h = args.height - top - 34
    max_abs = axis_scale(item["cv"], item["xu"], item["target"], item["prediction"])
    for idx, (key, title, subtitle, color) in enumerate(METHODS):
        rollout = rollouts[key]
        frames = rollout["frames"]
        rgb = frames[min(progress, len(frames) - 1)] if frames else Image.new("RGB", (320, 240), (230, 234, 238))
        x = margin_x + idx * (col_w + col_gap)
        draw_column(
            canvas,
            x,
            top,
            col_w,
            col_h,
            title,
            subtitle,
            rgb,
            clipped_path(item[key], progress + 1),
            item[key],
            color,
            max_abs,
        )
    return canvas


def render_item_rollouts(item: dict[str, Any], raw_dirs: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    _raw_root, manifest, steps = load_raw_episode(item["sample"], raw_dirs)
    rollouts = {}
    for key, _title, _subtitle, _color in METHODS:
        rollouts[key] = rollout_method(manifest, steps, item["sample"], item[key], args)
    return {"manifest": manifest, "rollouts": rollouts}


def write_video(rendered_items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")
    try:
        for order, payload in enumerate(rendered_items, start=1):
            item = payload["item"]
            rollouts = payload["rollouts"]
            max_frames = max(len(rollouts[key]["frames"]) for key, *_ in METHODS)
            last_frame = None
            for progress in range(max_frames):
                frame = render_composite_frame(item, rollouts, order, len(rendered_items), progress, args)
                last_frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                writer.write(last_frame)
            if last_frame is not None:
                for _ in range(max(0, args.hold_last_frames)):
                    writer.write(last_frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    raw_dirs = load_raw_dirs(args.dataset, [ROOT, *[Path(item) for item in args.raw_root_base]])
    selected = load_selected_items(args)
    rendered = []
    skipped = []
    for item in selected:
        try:
            payload = render_item_rollouts(item, raw_dirs, args)
        except Exception as exc:  # keep demo generation resilient across scenarios
            skipped.append({"sample_id": item["sample_id"], "reason": repr(exc)})
            print(f"skip {item['sample_id']}: {exc}")
            continue
        rendered.append({"item": item, **payload})
    if len(rendered) < 3:
        raise RuntimeError(f"Counterfactual rollout needs at least 3 successful samples, got {len(rendered)}")
    write_video(rendered, args)
    manifest = {
        "output": str(args.output),
        "mode": "vizdoom_counterfactual_rollout_demo_grade",
        "columns": [title for _key, title, _subtitle, _color in METHODS],
        "success_count": len(rendered),
        "skipped": skipped,
        "items": [
            {
                "sample_id": payload["item"]["sample_id"],
                "label": payload["item"]["label"],
                "case": payload["item"]["case"],
                "scenario": payload["manifest"].get("scenario"),
                "metrics": {
                    "ours_ADE": payload["item"]["ADE"],
                    "ours_FDE": payload["item"]["FDE"],
                    "cv_ADE": payload["item"]["cv_ADE"],
                    "cv_FDE": payload["item"]["cv_FDE"],
                    "xu_ADE": payload["item"]["xu_ADE"],
                    "xu_FDE": payload["item"]["xu_FDE"],
                },
                "rollouts": {
                    key: {
                        "frames": len(payload["rollouts"][key]["frames"]),
                        "start_info": payload["rollouts"][key]["start_info"],
                    }
                    for key, *_ in METHODS
                },
            }
            for payload in rendered
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "success_count": len(rendered), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
