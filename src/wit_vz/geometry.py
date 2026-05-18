"""Geometry utilities for egocentric local path prediction.

Coordinate convention:
- World coordinates use ViZDoom POSITION_X / POSITION_Y.
- Doom ANGLE is interpreted in degrees, with 0 deg facing +X.
- Local x is forward.
- Local y is right.
"""

from __future__ import annotations

import math
from typing import Iterable


def wrap_degrees(angle: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def world_delta_to_local(
    origin_x: float, origin_y: float, origin_angle_deg: float, x: float, y: float
) -> tuple[float, float]:
    """Transform a world point displacement into origin-local forward/right axes."""
    dx = x - origin_x
    dy = y - origin_y
    yaw = math.radians(origin_angle_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward = cos_yaw * dx + sin_yaw * dy
    right = -sin_yaw * dx + cos_yaw * dy
    return forward, right


def compute_relative_egomotion(
    prev_pose: dict[str, float] | None, curr_pose: dict[str, float]
) -> dict[str, float]:
    """Compute current pose relative to previous pose.

    The translational delta is expressed in the previous pose's local frame.
    `dyaw` is stored in radians for model input, with `dyaw_deg` retained for
    debugging and inspection.
    """
    if prev_pose is None:
        return {"dx_forward": 0.0, "dy_right": 0.0, "dyaw": 0.0, "dyaw_deg": 0.0}

    dx_forward, dy_right = world_delta_to_local(
        float(prev_pose["x"]),
        float(prev_pose["y"]),
        float(prev_pose.get("angle", 0.0)),
        float(curr_pose["x"]),
        float(curr_pose["y"]),
    )
    dyaw_deg = wrap_degrees(float(curr_pose.get("angle", 0.0)) - float(prev_pose.get("angle", 0.0)))
    return {
        "dx_forward": dx_forward,
        "dy_right": dy_right,
        "dyaw": math.radians(dyaw_deg),
        "dyaw_deg": dyaw_deg,
    }


def world_future_path_to_local(
    current_pose: dict[str, float], future_world_positions: Iterable[dict[str, float] | tuple[float, float]]
) -> list[list[float]]:
    """Convert future world positions into the current egocentric local frame.

    Output points are `[dx_forward, dy_right]`.
    """
    local_path: list[list[float]] = []
    for point in future_world_positions:
        if isinstance(point, dict):
            x = float(point["x"])
            y = float(point["y"])
        else:
            x = float(point[0])
            y = float(point[1])
        forward, right = world_delta_to_local(
            float(current_pose["x"]),
            float(current_pose["y"]),
            float(current_pose.get("angle", 0.0)),
            x,
            y,
        )
        local_path.append([forward, right])
    return local_path


def egomotion_history_from_records(records: list[dict]) -> list[list[float]]:
    """Return model-ready `[dx_forward, dy_right, dyaw]` history."""
    history = []
    for record in records:
        ego = record["relative_egomotion_from_prev"]
        history.append(
            [
                float(ego["dx_forward"]),
                float(ego["dy_right"]),
                float(ego["dyaw"]),
            ]
        )
    return history
