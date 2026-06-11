import torch

from src.models.factory import create_model, needs_rgb


def make_batch(batch_size=2, time=5, future_steps=15):
    return {
        "visual_tokens": torch.randn(batch_size, time, 64, 768),
        "rgb_history": torch.randn(batch_size, time, 3, 32, 32),
        "ego_history": torch.randn(batch_size, time, 3),
        "future_path": torch.randn(batch_size, future_steps, 2),
    }


def test_xu_pixels_only_baseline_accepts_cached_tokens():
    model = create_model(
        "xu_pixels_only_baseline",
        future_steps=15,
        backbone_name="cached_dinov3_convnext_tiny",
        hidden_dim=32,
        dropout=0.0,
    )
    pred = model(make_batch(future_steps=15))
    assert pred.shape == (2, 15, 2)
    assert torch.isfinite(pred).all()
    assert needs_rgb("xu_pixels_only_baseline")


def test_khaleque_motivated_baseline_is_motion_only():
    model = create_model(
        "khaleque_motivated_baseline",
        future_steps=15,
        hidden_dim=32,
        num_motivation_tokens=3,
        num_heads=4,
        dropout=0.0,
    )
    pred = model(make_batch(future_steps=15))
    assert pred.shape == (2, 15, 2)
    assert torch.isfinite(pred).all()
    assert not needs_rgb("khaleque_motivated_baseline")
