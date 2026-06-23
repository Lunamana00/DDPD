"""Privileged navigation baselines adapted to WIT-VZ trajectory metrics.

These adapters are intentionally oracle-style baselines:

- PointNav/DD-PPO-style goal oracle receives the GT future endpoint as the
  PointGoal and rolls out a straight local path toward it.
- Pose-graph A* receives a privileged traversability graph built from recorded
  poses and the GT future endpoint, then converts the planned path back to the
  WIT-VZ local [forward, right] target frame.

They are useful as upper-bound/context baselines, not as fair input-matched
competitors for the RGB-history path predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Iterable

import torch

from src.wit_vz.geometry import world_delta_to_local


Point2D = tuple[float, float]
GridCell = tuple[int, int]


def local_to_world_point(origin_pose: dict[str, Any], local_point: Iterable[float]) -> Point2D:
    """Convert one local [forward, right] point to world x/y coordinates."""

    forward, right = [float(value) for value in local_point]
    yaw = math.radians(float(origin_pose.get("angle", 0.0)))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    dx = cos_yaw * forward - sin_yaw * right
    dy = sin_yaw * forward + cos_yaw * right
    return float(origin_pose["x"]) + dx, float(origin_pose["y"]) + dy


def pointnav_goal_oracle_prediction(batch: dict[str, Any]) -> torch.Tensor:
    """Idealized PointNav/DD-PPO adapter using the GT endpoint as goal.

    PointGoal navigation policies are goal-conditioned: the agent is given a
    distance/bearing to the target. WIT-VZ does not have an external goal, so
    this adapter uses the last GT future point as the privileged goal and emits
    a straight local path to that endpoint.
    """

    target = batch["future_path"]
    future_steps = int(target.shape[1])
    endpoint = target[:, -1:, :]
    fractions = torch.linspace(
        1.0 / float(future_steps),
        1.0,
        future_steps,
        dtype=target.dtype,
        device=target.device,
    ).view(1, future_steps, 1)
    return endpoint * fractions


def _dedupe_consecutive(points: list[Point2D], eps: float = 1e-6) -> list[Point2D]:
    deduped: list[Point2D] = []
    for x, y in points:
        if not deduped or math.hypot(x - deduped[-1][0], y - deduped[-1][1]) > eps:
            deduped.append((float(x), float(y)))
    return deduped


def resample_polyline(points: list[Point2D], num_points: int) -> list[Point2D]:
    """Uniformly resample a world polyline into exactly num_points positions."""

    if num_points <= 0:
        return []
    points = _dedupe_consecutive(points)
    if not points:
        return [(0.0, 0.0) for _ in range(num_points)]
    if len(points) == 1:
        return [points[0] for _ in range(num_points)]

    segment_lengths = [
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    ]
    total = sum(segment_lengths)
    if total <= 1e-6:
        return [points[-1] for _ in range(num_points)]

    requested = [total * (i + 1) / float(num_points) for i in range(num_points)]
    output: list[Point2D] = []
    seg_index = 0
    traversed = 0.0
    for distance in requested:
        while (
            seg_index < len(segment_lengths) - 1
            and traversed + segment_lengths[seg_index] < distance
        ):
            traversed += segment_lengths[seg_index]
            seg_index += 1
        seg_len = max(segment_lengths[seg_index], 1e-6)
        ratio = min(max((distance - traversed) / seg_len, 0.0), 1.0)
        x0, y0 = points[seg_index]
        x1, y1 = points[seg_index + 1]
        output.append((x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio))
    return output


def sample_world_points(sample: dict[str, Any]) -> list[Point2D]:
    """Extract privileged world path support from a processed WIT-VZ sample."""

    points: list[Point2D] = []
    pose = sample.get("current_pose")
    if pose is not None:
        points.append((float(pose["x"]), float(pose["y"])))
    for point in sample.get("future_world_path", []) or []:
        points.append((float(point["x"]), float(point["y"])))
    return points


@dataclass
class PoseGraphAStarPlanner:
    """A lightweight A* planner over a privileged recorded-pose occupancy grid."""

    cell_size: float = 16.0
    occupied: set[GridCell] | None = None

    def __post_init__(self) -> None:
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if self.occupied is None:
            self.occupied = set()

    @classmethod
    def from_samples(cls, samples: Iterable[dict[str, Any]], cell_size: float = 16.0) -> "PoseGraphAStarPlanner":
        planner = cls(cell_size=cell_size)
        for sample in samples:
            for point in sample_world_points(sample):
                planner.add_world_point(point)
        return planner

    def world_to_cell(self, point: Point2D) -> GridCell:
        return (
            int(round(float(point[0]) / self.cell_size)),
            int(round(float(point[1]) / self.cell_size)),
        )

    def cell_to_world(self, cell: GridCell) -> Point2D:
        return (float(cell[0]) * self.cell_size, float(cell[1]) * self.cell_size)

    def add_world_point(self, point: Point2D) -> GridCell:
        cell = self.world_to_cell(point)
        assert self.occupied is not None
        self.occupied.add(cell)
        return cell

    def _neighbors(self, cell: GridCell) -> Iterable[tuple[GridCell, float]]:
        assert self.occupied is not None
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in self.occupied:
                    yield neighbor, math.hypot(dx, dy)

    @staticmethod
    def _heuristic(a: GridCell, b: GridCell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def astar_cells(self, start_world: Point2D, goal_world: Point2D) -> list[GridCell]:
        """Run A* between two world points after injecting their cells."""

        start = self.add_world_point(start_world)
        goal = self.add_world_point(goal_world)
        if start == goal:
            return [start]

        frontier: list[tuple[float, GridCell]] = [(0.0, start)]
        came_from: dict[GridCell, GridCell | None] = {start: None}
        cost_so_far: dict[GridCell, float] = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for neighbor, edge_cost in self._neighbors(current):
                new_cost = cost_so_far[current] + edge_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return [start, goal]

        path = [goal]
        current = goal
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[assignment]
            path.append(current)
        path.reverse()
        return path

    def plan_world_path(self, start_world: Point2D, goal_world: Point2D) -> list[Point2D]:
        cells = self.astar_cells(start_world, goal_world)
        worlds = [start_world]
        worlds.extend(self.cell_to_world(cell) for cell in cells[1:-1])
        worlds.append(goal_world)
        return _dedupe_consecutive(worlds)


def astar_oracle_prediction(sample: dict[str, Any], planner: PoseGraphAStarPlanner) -> list[list[float]]:
    """Plan to the GT endpoint and return a local WIT-VZ path."""

    pose = sample["current_pose"]
    target = sample["future_local_path"]
    future_steps = len(target)
    if future_steps == 0:
        return []
    start_world = (float(pose["x"]), float(pose["y"]))
    future_world = sample.get("future_world_path") or []
    if future_world:
        last = future_world[-1]
        goal_world = (float(last["x"]), float(last["y"]))
    else:
        goal_world = local_to_world_point(pose, target[-1])
    world_path = planner.plan_world_path(start_world, goal_world)
    resampled = resample_polyline(world_path, future_steps)
    return [
        list(world_delta_to_local(float(pose["x"]), float(pose["y"]), float(pose.get("angle", 0.0)), x, y))
        for x, y in resampled
    ]
