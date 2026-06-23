import torch

from src.models.navigation_oracles import (
    PoseGraphAStarPlanner,
    astar_oracle_prediction,
    local_to_world_point,
    pointnav_goal_oracle_prediction,
    resample_polyline,
)


def test_pointnav_goal_oracle_interpolates_to_gt_endpoint():
    batch = {
        "future_path": torch.tensor(
            [
                [[1.0, 0.0], [2.0, 1.0], [3.0, 3.0]],
                [[0.0, 1.0], [0.0, 2.0], [0.0, 6.0]],
            ],
            dtype=torch.float32,
        )
    }
    pred = pointnav_goal_oracle_prediction(batch)
    assert pred.shape == batch["future_path"].shape
    assert torch.allclose(pred[:, -1], batch["future_path"][:, -1])
    assert torch.allclose(pred[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(pred[1, 1], torch.tensor([0.0, 4.0]))


def test_local_to_world_point_matches_project_coordinate_convention():
    pose = {"x": 10.0, "y": 20.0, "angle": 90.0}
    x, y = local_to_world_point(pose, [5.0, 2.0])
    assert abs(x - 8.0) < 1e-5
    assert abs(y - 25.0) < 1e-5


def test_resample_polyline_returns_requested_count_and_endpoint():
    points = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
    out = resample_polyline(points, 4)
    assert len(out) == 4
    assert out[-1] == (2.0, 2.0)


def test_astar_oracle_prediction_returns_local_path_on_pose_graph():
    sample = {
        "sample_id": "demo",
        "current_pose": {"x": 0.0, "y": 0.0, "angle": 0.0},
        "future_world_path": [
            {"x": 0.0, "y": 1.0},
            {"x": 0.0, "y": 2.0},
            {"x": 1.0, "y": 2.0},
            {"x": 2.0, "y": 2.0},
        ],
        "future_local_path": [
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ],
    }
    planner = PoseGraphAStarPlanner.from_samples([sample], cell_size=1.0)
    pred = astar_oracle_prediction(sample, planner)
    assert len(pred) == 4
    assert torch.isfinite(torch.tensor(pred)).all()
    assert torch.allclose(torch.tensor(pred[-1]), torch.tensor([2.0, 2.0]), atol=1e-5)
