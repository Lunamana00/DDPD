"""Render real simulator counterfactual rollouts for external WIT-VZ demos.

This script uses the WIT-VZ external demo samples and zero-shot predictions,
then restarts each supported simulator at the sample's current pose and follows
three separate paths: constant velocity, ground truth, and model prediction.

Supported local environments:
- AI2-THOR
- MiniWorld
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    wrap,
)
from src.wit_vz.geometry import world_delta_to_local, wrap_degrees  # noqa: E402


METHODS = [
    ("cv", "CV baseline", "recent-motion extrapolation", CV_COLOR),
    ("target", "GT", "future local path label", GT_COLOR),
    ("prediction", "Ours", "visual cue-memory output", PRED_COLOR),
]


@dataclass(frozen=True)
class SourceSpec:
    key: str
    name: str
    dataset: Path
    predictions: Path
    summary: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/demo/presentation_sequence/demo_external_counterfactual_rollout_suite.mp4"),
    )
    parser.add_argument("--sources", nargs="+", default=["ai2thor", "miniworld"])
    parser.add_argument("--max-samples-per-env", type=int, default=2)
    parser.add_argument("--cases", nargs="+", default=["easy", "hard", "failure"])
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--screen-width", type=int, default=320)
    parser.add_argument("--screen-height", type=int, default=240)
    parser.add_argument("--hold-last-frames", type=int, default=8)
    parser.add_argument("--raw-root-base", action="append", default=[])
    parser.add_argument("--write-branch-videos", action="store_true")
    parser.add_argument("--ai2thor-platform", default="auto", help="AI2-THOR platform, e.g. auto or CloudRendering.")
    parser.add_argument("--ai2thor-gpu-device", type=int, default=None)
    parser.add_argument("--ai2thor-quality", default="Low")
    parser.add_argument("--vulkan-library", type=Path, default=None)
    return parser.parse_args()


def patch_vulkan_find_library(vulkan_library: Path | None) -> None:
    if vulkan_library is None:
        return
    resolved = vulkan_library.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Requested --vulkan-library does not exist: {resolved}")
    original_find_library = ctypes.util.find_library

    def find_library_with_vulkan(name: str) -> str | None:
        if name == "vulkan":
            return str(resolved)
        return original_find_library(name)

    ctypes.util.find_library = find_library_with_vulkan
    lib_dir = str(resolved.parent)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir


def resolve_ai2thor_platform(name: str) -> Any:
    if name.lower() in {"", "auto", "none"}:
        return None
    import ai2thor.platform as platform_module

    if not hasattr(platform_module, name):
        valid = ["auto", "CloudRendering", "Linux64", "Windows64", "OSXIntel64"]
        raise ValueError(f"Unsupported AI2-THOR platform {name!r}. Expected one of: {', '.join(valid)}")
    return getattr(platform_module, name)


def default_sources() -> dict[str, SourceSpec]:
    return {
        "ai2thor": SourceSpec(
            key="ai2thor",
            name="AI2-THOR",
            dataset=ROOT / "data/wit_vz/processed/ai2thor_demo_001_03s",
            predictions=ROOT / "reports/demo/external_ai2thor_zero_shot_03s/eval_all/predictions.jsonl",
            summary=ROOT / "reports/demo/external_ai2thor_zero_shot_03s/contact_by_scene/summary.json",
        ),
        "miniworld": SourceSpec(
            key="miniworld",
            name="MiniWorld",
            dataset=ROOT / "data/wit_vz/processed/miniworld_demo_001_03s",
            predictions=ROOT / "reports/demo/external_miniworld_zero_shot_03s/eval_all/predictions.jsonl",
            summary=ROOT / "reports/demo/external_miniworld_zero_shot_03s/contact_by_env/summary.json",
        ),
    }


def source_id(sample: dict[str, Any]) -> str | None:
    return sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")


def resolve_raw_path(raw_dirs: dict[str, Path], rel_path: str, sid: str | None) -> Path:
    selected = sid
    rel = rel_path
    if "::" in rel_path:
        selected, rel = rel_path.split("::", 1)
    path = Path(rel)
    if path.is_absolute():
        return path
    if selected and selected in raw_dirs:
        return raw_dirs[selected] / path
    return next(iter(raw_dirs.values())) / path


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


def pose_error(got: dict[str, float], want: dict[str, float]) -> tuple[float, float]:
    pos = math.hypot(float(got["x"]) - float(want["x"]), float(got["y"]) - float(want["y"]))
    angle = abs(wrap_degrees(float(got.get("angle", 0.0)) - float(want.get("angle", 0.0))))
    return pos, angle


def load_items(spec: SourceSpec, args: argparse.Namespace) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    raw_dirs = load_raw_dirs(spec.dataset, [ROOT, *[Path(item) for item in args.raw_root_base]])
    samples = {str(row["sample_id"]): row for row in read_jsonl(spec.dataset / "samples.jsonl")}
    predictions = {str(row["sample_id"]): row for row in read_jsonl(spec.predictions)}
    summary = read_json(spec.summary)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in args.cases:
        for row in summary:
            if row.get("case") != case:
                continue
            sample_id = str(row["sample_id"])
            if sample_id in seen or sample_id not in samples or sample_id not in predictions:
                continue
            seen.add(sample_id)
            sample = samples[sample_id]
            pred = predictions[sample_id]
            target = pred.get("target") or sample["future_local_path"]
            prediction = pred.get("prediction") or []
            cv_path = pred.get("constant_velocity_prediction") or []
            ade, fde = path_error(prediction, target)
            cv_ade, cv_fde = path_error(cv_path, target)
            selected.append(
                {
                    "env_key": spec.key,
                    "env_name": spec.name,
                    "case": str(row.get("case", "demo")),
                    "group": str(row.get("group") or sample.get("metadata", {}).get("scene") or spec.name),
                    "sample_id": sample_id,
                    "sample": sample,
                    "prediction": prediction,
                    "target": target,
                    "cv": cv_path,
                    "ADE": float(pred.get("ADE", ade)),
                    "FDE": float(pred.get("FDE", fde)),
                    "cv_ADE": float(pred.get("constant_velocity_ADE", cv_ade)),
                    "cv_FDE": float(pred.get("constant_velocity_FDE", cv_fde)),
                }
            )
            if len(selected) >= args.max_samples_per_env:
                return raw_dirs, selected
    return raw_dirs, selected


class AI2ThorAdapter:
    def __init__(self, args: argparse.Namespace) -> None:
        from ai2thor.controller import Controller

        patch_vulkan_find_library(args.vulkan_library)
        self.Controller = Controller
        self.args = args

    def pose_from_event(self, event: Any) -> dict[str, float]:
        agent = event.metadata["agent"]
        position = agent["position"]
        rotation = agent["rotation"]
        return {
            "x": float(position["z"]),
            "y": float(position["x"]),
            "z": float(position["y"]),
            "angle": float(rotation["y"]),
        }

    def frame_from_event(self, event: Any) -> Image.Image:
        if event.frame is None:
            return Image.new("RGB", (self.args.screen_width, self.args.screen_height), (225, 228, 232))
        return Image.fromarray(event.frame).convert("RGB")

    def start_controller(self, sample: dict[str, Any], raw_dirs: dict[str, Path]) -> tuple[Any, Any, dict[str, Any]]:
        sid = source_id(sample)
        steps = read_jsonl(resolve_raw_path(raw_dirs, sample["metadata"]["raw_episode_path"], sid))
        center_step = int(sample.get("center_step", 0))
        center_row = steps[min(center_step, len(steps) - 1)]
        scene = center_row.get("metadata", {}).get("scene") or sample.get("metadata", {}).get("scene") or "FloorPlan1"
        controller = self.Controller(
            width=self.args.screen_width,
            height=self.args.screen_height,
            platform=resolve_ai2thor_platform(self.args.ai2thor_platform),
            gpu_device=self.args.ai2thor_gpu_device,
            quality=self.args.ai2thor_quality,
            gridSize=0.25,
            rotateStepDegrees=90.0,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )
        controller.reset(scene=scene)
        pose = sample["current_pose"]
        event = controller.step(
            action="Teleport",
            position={"x": float(pose["y"]), "y": float(pose.get("z", 0.9)), "z": float(pose["x"])},
            rotation={"x": 0.0, "y": float(pose.get("angle", 0.0)), "z": 0.0},
            horizon=0.0,
            standing=True,
            forceAction=True,
        )
        start_pose = self.pose_from_event(event)
        pos_error, angle_error = pose_error(start_pose, pose)
        return controller, event, {
            "start_mode": "ai2thor_teleport",
            "scene": scene,
            "final_position_error": pos_error,
            "final_angle_error": angle_error,
            "target_pose": pose,
            "start_pose": start_pose,
        }

    def action_toward(self, pose: dict[str, float], target_xy: tuple[float, float]) -> str:
        dx = float(target_xy[0]) - float(pose["x"])
        dy = float(target_xy[1]) - float(pose["y"])
        if math.hypot(dx, dy) < 0.05:
            return "Pass"
        target_angle = math.degrees(math.atan2(dy, dx))
        delta = wrap_degrees(target_angle - float(pose.get("angle", 0.0)))
        if delta > 45.0:
            return "RotateLeft"
        if delta < -45.0:
            return "RotateRight"
        return "MoveAhead"

    def rollout(
        self, sample: dict[str, Any], raw_dirs: dict[str, Path], path: list[list[float]]
    ) -> dict[str, Any]:
        controller, event, start_info = self.start_controller(sample, raw_dirs)
        frames: list[Image.Image] = []
        realized: list[list[float]] = []
        start_pose = self.pose_from_event(event)
        waypoints = local_path_to_world(start_pose, path)
        try:
            for waypoint in waypoints:
                frames.append(self.frame_from_event(event))
                current = self.pose_from_event(event)
                forward, right = world_delta_to_local(
                    start_pose["x"], start_pose["y"], start_pose["angle"], current["x"], current["y"]
                )
                realized.append([forward, right])
                action = self.action_toward(current, waypoint)
                event = controller.step(action=action)
                if action == "MoveAhead" and not bool(event.metadata.get("lastActionSuccess", True)):
                    event = controller.step(action="RotateRight")
        finally:
            controller.stop()
        return {"frames": frames, "planned_path": path, "realized_path": realized, "start_info": start_info}


class MiniWorldAdapter:
    def __init__(self, args: argparse.Namespace) -> None:
        import gymnasium as gym
        import miniworld  # noqa: F401

        self.gym = gym
        self.args = args

    def pose_from_env(self, env: Any) -> dict[str, float]:
        agent = env.unwrapped.agent
        pos = np.asarray(agent.pos, dtype=float).reshape(-1)
        return {
            "x": float(pos[0]),
            "y": float(pos[2]),
            "z": float(pos[1]),
            "angle": math.degrees(float(agent.dir)),
        }

    def set_pose(self, env: Any, pose: dict[str, float]) -> None:
        agent = env.unwrapped.agent
        agent.pos = np.array([float(pose["x"]), float(pose.get("z", 0.0)), float(pose["y"])], dtype=float)
        agent.dir = math.radians(float(pose.get("angle", 0.0)))

    def frame_from_env(self, env: Any) -> Image.Image:
        frame = env.render()
        if frame is None:
            frame = np.zeros((self.args.screen_height, self.args.screen_width, 3), dtype=np.uint8)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return Image.fromarray(frame[..., :3]).convert("RGB").resize((self.args.screen_width, self.args.screen_height))

    def actions(self, env: Any) -> dict[str, int]:
        actions = getattr(env.unwrapped, "actions", None)
        if actions is not None:
            return {
                "move_forward": int(getattr(actions, "move_forward")),
                "turn_left": int(getattr(actions, "turn_left")),
                "turn_right": int(getattr(actions, "turn_right")),
            }
        return {"move_forward": 0, "turn_left": 1, "turn_right": 2}

    def action_toward(self, env: Any, pose: dict[str, float], target_xy: tuple[float, float]) -> int | None:
        dx = float(target_xy[0]) - float(pose["x"])
        dy = float(target_xy[1]) - float(pose["y"])
        if math.hypot(dx, dy) < 0.04:
            return None
        target_angle = math.degrees(math.atan2(dy, dx))
        delta = wrap_degrees(target_angle - float(pose.get("angle", 0.0)))
        choices = self.actions(env)
        if delta > 20.0:
            return choices["turn_left"]
        if delta < -20.0:
            return choices["turn_right"]
        return choices["move_forward"]

    def rollout(
        self, sample: dict[str, Any], raw_dirs: dict[str, Path], path: list[list[float]]
    ) -> dict[str, Any]:
        sid = source_id(sample)
        steps = read_jsonl(resolve_raw_path(raw_dirs, sample["metadata"]["raw_episode_path"], sid))
        center_step = int(sample.get("center_step", 0))
        center_row = steps[min(center_step, len(steps) - 1)]
        env_id = center_row.get("metadata", {}).get("env_id") or "MiniWorld-Hallway-v0"
        seed = int(center_row.get("metadata", {}).get("episode_seed", 0))
        env = self.gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        self.set_pose(env, sample["current_pose"])
        start_pose = self.pose_from_env(env)
        pos_error, angle_error = pose_error(start_pose, sample["current_pose"])
        start_info = {
            "start_mode": "miniworld_set_agent_state",
            "env_id": env_id,
            "episode_seed": seed,
            "final_position_error": pos_error,
            "final_angle_error": angle_error,
            "target_pose": sample["current_pose"],
            "start_pose": start_pose,
        }
        frames: list[Image.Image] = []
        realized: list[list[float]] = []
        waypoints = local_path_to_world(start_pose, path)
        try:
            for waypoint in waypoints:
                frames.append(self.frame_from_env(env))
                current = self.pose_from_env(env)
                forward, right = world_delta_to_local(
                    start_pose["x"], start_pose["y"], start_pose["angle"], current["x"], current["y"]
                )
                realized.append([forward, right])
                action = self.action_toward(env, current, waypoint)
                if action is not None:
                    env.step(action)
        finally:
            env.close()
        return {"frames": frames, "planned_path": path, "realized_path": realized, "start_info": start_info}


def adapter_for(env_key: str, args: argparse.Namespace) -> Any:
    if env_key == "ai2thor":
        return AI2ThorAdapter(args)
    if env_key == "miniworld":
        return MiniWorldAdapter(args)
    raise ValueError(f"Unsupported source for this script: {env_key}")


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
    inner_w = max(80, w - 36)
    plot_h = max(96, min(260, h // 3))
    frame_h = max(96, h - plot_h - 126)
    frame_top = y + 78
    frame = fit_image(rgb, (inner_w, frame_h))
    paste_center(canvas, frame, (x + 18, frame_top, x + w - 18, frame_top + frame_h))
    plot_top = frame_top + frame_h + 18
    plot = draw_path_plot(path, (inner_w, plot_h), color, max_abs, full_path)
    canvas.paste(plot, (x + 18, plot_top))


def render_composite_frame(payload: dict[str, Any], order: int, total: int, progress: int, args: argparse.Namespace) -> Image.Image:
    item = payload["item"]
    rollouts = payload["rollouts"]
    canvas = Image.new("RGB", (args.width, args.height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    header = f"{order:02d}/{total:02d}  Real counterfactual rollout / {item['env_name']} / {item['group']} / {item['case']}"
    draw.text((34, 26), header, fill=TEXT_COLOR, font=font(30, bold=True))
    metrics = (
        f"sample={item['sample_id']}    t={progress + 1:02d}    "
        f"ours={item['ADE']:.2f}/{item['FDE']:.2f}    CV={item['cv_ADE']:.2f}/{item['cv_FDE']:.2f}"
    )
    draw.text((34, 72), wrap(metrics, 150), fill=MUTED_COLOR, font=font(18))
    col_gap = 18
    margin_x = 34
    top = 134
    col_w = (args.width - 2 * margin_x - (len(METHODS) - 1) * col_gap) // len(METHODS)
    col_h = args.height - top - 34
    max_abs = axis_scale(item["cv"], item["target"], item["prediction"])
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


def render_branch_frame(payload: dict[str, Any], method_key: str, order: int, total: int, progress: int, args: argparse.Namespace) -> Image.Image:
    item = payload["item"]
    rollout = payload["rollouts"][method_key]
    spec = {key: (title, subtitle, color) for key, title, subtitle, color in METHODS}[method_key]
    title, subtitle, color = spec
    canvas = Image.new("RGB", (args.width, args.height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    draw.text((34, 26), f"{order:02d}/{total:02d}  {title} / {item['env_name']} / {item['group']}", fill=color, font=font(32, bold=True))
    draw.text((34, 72), f"{subtitle}    sample={item['sample_id']}    t={progress + 1:02d}", fill=MUTED_COLOR, font=font(18))
    frames = rollout["frames"]
    rgb = frames[min(progress, len(frames) - 1)] if frames else Image.new("RGB", (320, 240), (230, 234, 238))
    frame = fit_image(rgb, (args.width - 80, args.height - 150))
    paste_center(canvas, frame, (40, 120, args.width - 40, args.height - 30))
    return canvas


def render_video(rendered: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")
    branch_writers: dict[str, Any] = {}
    if args.write_branch_videos:
        for key, *_ in METHODS:
            path = args.output.with_name(f"{args.output.stem}_{key}.mp4")
            branch_writers[key] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
            if not branch_writers[key].isOpened():
                raise RuntimeError(f"Could not open writer: {path}")
    try:
        for order, payload in enumerate(rendered, start=1):
            max_frames = max(len(payload["rollouts"][key]["frames"]) for key, *_ in METHODS)
            last_composite = None
            last_branch: dict[str, np.ndarray] = {}
            for progress in range(max_frames):
                frame = render_composite_frame(payload, order, len(rendered), progress, args)
                last_composite = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                writer.write(last_composite)
                for key, *_ in METHODS:
                    if key in branch_writers:
                        branch = render_branch_frame(payload, key, order, len(rendered), progress, args)
                        last_branch[key] = cv2.cvtColor(np.asarray(branch), cv2.COLOR_RGB2BGR)
                        branch_writers[key].write(last_branch[key])
            for _ in range(max(0, args.hold_last_frames)):
                if last_composite is not None:
                    writer.write(last_composite)
                for key, branch_writer in branch_writers.items():
                    if key in last_branch:
                        branch_writer.write(last_branch[key])
    finally:
        writer.release()
        for branch_writer in branch_writers.values():
            branch_writer.release()


def render_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rendered: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    specs = default_sources()
    for key in args.sources:
        if key not in specs:
            skipped.append({"source": key, "reason": "unsupported source"})
            continue
        spec = specs[key]
        try:
            raw_dirs, items = load_items(spec, args)
            adapter = adapter_for(key, args)
        except Exception as exc:
            skipped.append({"source": key, "reason": repr(exc)})
            continue
        for item in items:
            try:
                rollouts = {
                    method_key: adapter.rollout(item["sample"], raw_dirs, item[method_key])
                    for method_key, *_ in METHODS
                }
                rendered.append({"item": item, "rollouts": rollouts})
            except Exception as exc:
                skipped.append({"source": key, "sample_id": item["sample_id"], "reason": repr(exc)})
                print(f"skip {key} {item['sample_id']}: {exc}")
    return rendered, skipped


def main() -> None:
    args = parse_args()
    rendered, skipped = render_items(args)
    if not rendered:
        raise RuntimeError("Expected at least 1 successful external counterfactual item, got 0")
    render_video(rendered, args)
    manifest = {
        "output": str(args.output),
        "mode": "external_real_counterfactual_rollout",
        "columns": [title for _key, title, _subtitle, _color in METHODS],
        "branch_videos": {
            key: str(args.output.with_name(f"{args.output.stem}_{key}.mp4")) for key, *_ in METHODS if args.write_branch_videos
        },
        "success_count": len(rendered),
        "skipped": skipped,
        "items": [
            {
                "env_name": payload["item"]["env_name"],
                "group": payload["item"]["group"],
                "case": payload["item"]["case"],
                "sample_id": payload["item"]["sample_id"],
                "metrics": {
                    "ours_ADE": payload["item"]["ADE"],
                    "ours_FDE": payload["item"]["FDE"],
                    "cv_ADE": payload["item"]["cv_ADE"],
                    "cv_FDE": payload["item"]["cv_FDE"],
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
