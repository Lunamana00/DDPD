import torch

from src.models.paper_proxies import (
    khaleque_center_random_prediction,
    xu_pixels_saliency_prediction,
)


def make_proxy_batch(batch_size=2, future_steps=4):
    rgb = torch.zeros(batch_size, 3, 3, 16, 16)
    rgb[:, -1, :, :, 12:] = 1.0
    return {
        "sample_id": [f"sample_{idx}" for idx in range(batch_size)],
        "ego_history": torch.tensor(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
            ],
            dtype=torch.float32,
        )[:batch_size],
        "future_path": torch.zeros(batch_size, future_steps, 2),
        "rgb_history": rgb,
        "current_pose": [
            {"x": 0.0, "y": 0.0, "angle": 0.0},
            {"x": 3.0, "y": -2.0, "angle": 0.2},
        ][:batch_size],
        "metadata": [{"source_id": "source_a"} for _ in range(batch_size)],
        "source": [{"source_id": "source_a"} for _ in range(batch_size)],
    }


def test_khaleque_proxy_is_deterministic_and_path_shaped():
    batch = make_proxy_batch()
    centers = {"source_a": (10.0, 0.0)}
    first = khaleque_center_random_prediction(batch, centers)
    second = khaleque_center_random_prediction(batch, centers)
    assert first.shape == batch["future_path"].shape
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_xu_pixels_saliency_proxy_uses_last_frame_and_rolls_out_rightward():
    batch = make_proxy_batch(batch_size=1, future_steps=3)
    pred = xu_pixels_saliency_prediction(batch)
    assert pred.shape == (1, 3, 2)
    assert torch.isfinite(pred).all()
    assert torch.all(pred[0, :, 0] > 0)
    assert torch.all(pred[0, :, 1] > 0)
