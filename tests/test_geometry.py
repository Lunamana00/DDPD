import math

from src.wit_vz.geometry import compute_relative_egomotion, world_future_path_to_local


def test_world_future_path_to_local_angle_zero():
    pose = {"x": 0.0, "y": 0.0, "angle": 0.0}
    out = world_future_path_to_local(pose, [(10.0, 0.0), (10.0, 5.0)])
    assert out == [[10.0, 0.0], [10.0, 5.0]]


def test_world_future_path_to_local_angle_ninety():
    pose = {"x": 0.0, "y": 0.0, "angle": 90.0}
    out = world_future_path_to_local(pose, [(0.0, 10.0), (5.0, 10.0)])
    assert abs(out[0][0] - 10.0) < 1e-5
    assert abs(out[0][1]) < 1e-5
    assert abs(out[1][0] - 10.0) < 1e-5
    assert abs(out[1][1] + 5.0) < 1e-5


def test_relative_egomotion():
    prev = {"x": 0.0, "y": 0.0, "angle": 0.0}
    curr = {"x": 3.0, "y": 4.0, "angle": 10.0}
    ego = compute_relative_egomotion(prev, curr)
    assert ego["dx_forward"] == 3.0
    assert ego["dy_right"] == 4.0
    assert abs(ego["dyaw"] - math.radians(10.0)) < 1e-6
