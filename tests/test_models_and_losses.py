import torch

from src.losses import trajectory_loss
from src.metrics import ade, fde
from src.models.baselines import ConstantVelocityBaseline, EgoMotionOnlyModel
from src.models.cue_memory import TwoStreamEgocentricCueMemoryPathPredictor
from src.models.motion import constant_velocity_path


def make_batch(batch=2, history=4, future=3):
    return {
        "rgb_history": torch.rand(batch, history, 3, 32, 32),
        "ego_history": torch.rand(batch, history, 3),
        "future_path": torch.rand(batch, future, 2),
    }


def test_constant_velocity_forward():
    model = ConstantVelocityBaseline(future_steps=3)
    batch = {"ego_history": torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])}
    out = model(batch)
    assert out.shape == (1, 3, 2)
    assert torch.allclose(out[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_cue_memory_model_forward_shape():
    batch = make_batch()
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="gru",
        freeze_backbone=False,
    )
    out = model(batch)
    assert out.shape == (2, 3, 2)


def test_cue_memory_residual_initializes_to_constant_velocity():
    batch = make_batch(batch=2, history=5, future=3)
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="gru",
        freeze_backbone=False,
        use_constant_velocity_residual=True,
        residual_scale=25.0,
    )
    out = model(batch)
    expected = constant_velocity_path(batch["ego_history"], future_steps=3)
    assert torch.allclose(out, expected, atol=1e-6)


def test_loss_and_metrics():
    pred = torch.zeros(2, 3, 2)
    target = torch.ones(2, 3, 2)
    loss = trajectory_loss(pred, target, "huber")
    scaled_loss = trajectory_loss(pred, target, "huber", coordinate_scale=10.0)
    assert loss.item() > 0
    assert 0 < scaled_loss.item() < loss.item()
    assert ade(pred, target).item() > 0
    assert fde(pred, target).item() > 0


def test_tiny_overfit_reduces_loss():
    torch.manual_seed(1)
    batch = make_batch(batch=4, history=5, future=2)
    batch["future_path"] = torch.cumsum(batch["ego_history"][:, -2:, :2], dim=1)
    model = EgoMotionOnlyModel(future_steps=2, hidden_dim=32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.03)
    with torch.no_grad():
        initial = trajectory_loss(model(batch), batch["future_path"]).item()
    for _ in range(40):
        opt.zero_grad()
        loss = trajectory_loss(model(batch), batch["future_path"])
        loss.backward()
        opt.step()
    final = trajectory_loss(model(batch), batch["future_path"]).item()
    assert final < initial
