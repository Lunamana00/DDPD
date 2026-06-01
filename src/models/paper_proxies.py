"""Paper-adapted offline trajectory proxies.

These are not exact reproductions of the cited interactive systems. They adapt
the papers' decision styles to WIT-VZ's offline future-local-path metric so the
baselines can be evaluated with ADE/FDE on the same processed samples.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import math
from typing import Any

import torch

from src.wit_vz.dataset import WITVZPathDataset
from src.wit_vz.geometry import world_delta_to_local


def estimate_ego_speed(ego_history: torch.Tensor, min_speed: float = 1.0) -> torch.Tensor:
    """Estimate rollout speed from recent egocentric forward/right motion."""
    speeds = torch.linalg.norm(ego_history[..., :2], dim=-1)
    return speeds.mean(dim=1).clamp_min(float(min_speed))


def deterministic_uniform(key: str, low: float, high: float) -> float:
    """Stable pseudo-random scalar for deterministic offline rollouts."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64 - 1)
    return low + (high - low) * value


def source_centers_from_train(dataset_dir: str | Path) -> dict[str, tuple[float, float]]:
    """Estimate a simple per-source map prior from train poses.

    The Khaleque-style proxy needs an exploration attractor in an offline
    dataset. Since WIT-VZ does not store object/motivation fields, we use the
    center of train-set pose bounds for each source as a deterministic context
    steering target.
    """
    dataset = WITVZPathDataset(dataset_dir, split="train", load_rgb=False)
    bounds: dict[str, list[float]] = {}
    for sample in dataset.samples:
        pose = sample["current_pose"]
        source = sample.get("source", {})
        metadata = sample.get("metadata", {})
        source_id = str(source.get("source_id") or metadata.get("source_id") or "unknown")
        x = float(pose["x"])
        y = float(pose["y"])
        if source_id not in bounds:
            bounds[source_id] = [x, x, y, y]
            continue
        item = bounds[source_id]
        item[0] = min(item[0], x)
        item[1] = max(item[1], x)
        item[2] = min(item[2], y)
        item[3] = max(item[3], y)
    return {
        source_id: ((values[0] + values[1]) * 0.5, (values[2] + values[3]) * 0.5)
        for source_id, values in bounds.items()
    }


def khaleque_center_random_prediction(
    batch: dict[str, Any],
    source_centers: dict[str, tuple[float, float]],
    sector_degrees: float = 135.0,
    decision_interval: int = 10,
) -> torch.Tensor:
    """Khaleque-style center-biased exploratory context-steering proxy.

    The original work studies motivated exploratory agents in an interactive
    environment. WIT-VZ samples do not contain the live motivation/object state,
    so this proxy periodically chooses a deterministic exploratory direction in
    a sector around a per-source center prior and rolls out at recent ego speed.
    """
    ego_history = batch["ego_history"]
    target = batch["future_path"]
    batch_size, future_steps = target.shape[:2]
    speeds = estimate_ego_speed(ego_history)
    outputs = torch.zeros((batch_size, future_steps, 2), dtype=target.dtype, device=target.device)
    half_sector = math.radians(float(sector_degrees) * 0.5)

    for i in range(batch_size):
        pose = batch["current_pose"][i]
        metadata = batch["metadata"][i]
        source = batch["source"][i]
        source_id = str(source.get("source_id") or metadata.get("source_id") or "unknown")
        center = source_centers.get(source_id, (float(pose["x"]), float(pose["y"])))
        center_forward, center_right = world_delta_to_local(
            float(pose["x"]),
            float(pose["y"]),
            float(pose.get("angle", 0.0)),
            center[0],
            center[1],
        )
        pos_forward = 0.0
        pos_right = 0.0
        direction = 0.0
        for step in range(future_steps):
            if step % int(decision_interval) == 0:
                bias = math.atan2(center_right - pos_right, center_forward - pos_forward)
                random_offset = deterministic_uniform(
                    f"{batch['sample_id'][i]}::{step}",
                    -half_sector,
                    half_sector,
                )
                direction = bias + random_offset
            step_speed = float(speeds[i].detach().cpu())
            pos_forward += step_speed * math.cos(direction)
            pos_right += step_speed * math.sin(direction)
            outputs[i, step, 0] = pos_forward
            outputs[i, step, 1] = pos_right
    return outputs


def xu_pixels_saliency_prediction(
    batch: dict[str, Any],
    max_angle_degrees: float = 60.0,
    brightness_weight: float = 0.35,
    center_penalty: float = 0.10,
) -> torch.Tensor:
    """Xu-style pixels-only visual interest proxy.

    This offline proxy reads only the last RGB frame, scores horizontal columns
    by texture/brightness with a mild center prior, converts the best column to
    a steering angle, then rolls out a fixed-speed local path.
    """
    if "rgb_history" not in batch:
        raise KeyError("xu_pixels_saliency_prediction requires batch['rgb_history']")
    frames = batch["rgb_history"][:, -1]
    target = batch["future_path"]
    batch_size, future_steps = target.shape[:2]
    gray = frames.mean(dim=1)
    _, height, width = gray.shape

    crop = gray[:, int(height * 0.35) : int(height * 0.92), :]
    edges = torch.zeros_like(crop)
    edges[:, :, 1:] = (crop[:, :, 1:] - crop[:, :, :-1]).abs()
    brightness = crop.mean(dim=1)
    texture = edges.mean(dim=1)
    center_prior = torch.linspace(-1.0, 1.0, width, device=frames.device).abs()
    score = texture + float(brightness_weight) * brightness - float(center_penalty) * center_prior

    best_col = score.argmax(dim=1).float()
    centered = (best_col - (width - 1) * 0.5) / max((width - 1) * 0.5, 1.0)
    directions = centered * math.radians(float(max_angle_degrees))
    speeds = estimate_ego_speed(batch["ego_history"])

    outputs = torch.zeros((batch_size, future_steps, 2), dtype=target.dtype, device=target.device)
    for step in range(future_steps):
        outputs[:, step, 0] = (step + 1) * speeds * torch.cos(directions)
        outputs[:, step, 1] = (step + 1) * speeds * torch.sin(directions)
    return outputs
