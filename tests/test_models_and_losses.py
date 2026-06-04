import torch

from src.losses import trajectory_loss
from src.metrics import ade, fde, select_best_trajectory
from src.models.baselines import ConstantVelocityBaseline, EgoMotionOnlyModel
from src.models.cue_memory import (
    LastCueMemoryBank,
    MeanCueMemoryBank,
    RelativeContrastHybridSpatialGraphAggregator,
    StaticMemoryBank,
    TwoStreamEgocentricCueMemoryPathPredictor,
)
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


def test_cue_memory_transformer_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="transformer",
        freeze_backbone=False,
    )
    out = model(batch)
    assert out.shape == (2, 3, 2)


def test_cue_memory_no_temporal_adapter_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="none",
        freeze_backbone=False,
        selector_type="tokenlearner",
        memory_type="attention",
        spatial_relation_type="topk_graph",
        spatial_graph_neighbors=3,
    )
    out = model(batch)
    assert out.shape == (2, 3, 2)


def test_cue_memory_zero_visual_backbone_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="zero_tokens",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="timesformer",
        freeze_backbone=True,
        selector_type="tokenlearner",
        memory_type="attention",
    )
    out = model(batch)
    assert out.shape == (2, 3, 2)


def test_cue_memory_zero_visual_backbone_without_rgb_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    batch.pop("rgb_history")
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="zero_tokens",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="timesformer",
        freeze_backbone=True,
        selector_type="tokenlearner",
        memory_type="attention",
    )
    out = model(batch)
    assert out.shape == (2, 3, 2)


def test_cue_memory_tokenlearner_graph_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="timesformer",
        freeze_backbone=False,
        selector_type="tokenlearner",
        memory_type="attention",
        use_spatial_graph=True,
        spatial_graph_neighbors=3,
        use_temporal_difference_conv=True,
        use_temporal_shift=True,
    )
    out = model(batch)
    assert out.shape == (2, 3, 2)


def test_cue_memory_spatial_relation_variants_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    for relation_type in (
        "topk_graph",
        "none",
        "full_attention",
        "local_grid",
        "strnet_edge_message",
        "relpos_topk_graph",
        "contrast_topk_graph",
        "hybrid_local_topk_graph",
        "relpos_contrast_hybrid_graph",
    ):
        model = TwoStreamEgocentricCueMemoryPathPredictor(
            future_steps=3,
            backbone_name="small_cnn",
            hidden_dim=32,
            num_cue_tokens=4,
            temporal_type="timesformer",
            freeze_backbone=False,
            selector_type="tokenlearner",
            memory_type="attention",
            spatial_relation_type=relation_type,
            spatial_graph_neighbors=3,
        )
        out = model(batch)
        assert out.shape == (2, 3, 2)


def test_pairwise_contrast_bias_shape():
    aggregator = RelativeContrastHybridSpatialGraphAggregator(
        dim=16,
        neighbors=2,
        use_contrast=True,
    )
    normalized = torch.randn(3, 4, 16)
    bias = aggregator._pairwise_contrast_bias(normalized, normalized.dtype)
    assert bias.shape == (3, 4, 4)


def test_candidate_contrast_message_shape():
    aggregator = RelativeContrastHybridSpatialGraphAggregator(
        dim=16,
        neighbors=2,
        use_contrast=True,
    )
    normalized = torch.randn(3, 4, 16)
    candidate_indices = torch.tensor(
        [
            [[1, 2], [0, 2], [1, 3], [2, 1]],
            [[1, 3], [0, 3], [0, 1], [1, 2]],
            [[2, 3], [2, 0], [3, 1], [0, 1]],
        ]
    )
    candidate_weights = torch.full((3, 4, 2), 0.5)
    message = aggregator._candidate_contrast_message(
        normalized,
        candidate_indices,
        candidate_weights,
        normalized.dtype,
    )
    assert message.shape == (3, 4, 16)


def test_cue_memory_bank_ablation_variants_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    for memory_type in (
        "attention",
        "attention_no_ego",
        "gru_cell",
        "last_cue",
        "no_memory",
        "mean_cue",
        "no_memory_update",
    ):
        model = TwoStreamEgocentricCueMemoryPathPredictor(
            future_steps=3,
            backbone_name="small_cnn",
            hidden_dim=32,
            num_cue_tokens=4,
            temporal_type="timesformer",
            freeze_backbone=False,
            selector_type="tokenlearner",
            memory_type=memory_type,
            spatial_relation_type="topk_graph",
            spatial_graph_neighbors=3,
        )
        out = model(batch)
        assert out.shape == (2, 3, 2)


def test_cue_memory_decoder_ablation_variants_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    for decoder_type in (
        "horizon_query_decoder",
        "single_vector_mlp",
        "shared_query_decoder",
        "autoregressive_decoder",
    ):
        model = TwoStreamEgocentricCueMemoryPathPredictor(
            future_steps=3,
            backbone_name="small_cnn",
            hidden_dim=32,
            num_cue_tokens=4,
            temporal_type="timesformer",
            freeze_backbone=False,
            selector_type="tokenlearner",
            memory_type="attention",
            spatial_relation_type="topk_graph",
            spatial_graph_neighbors=3,
            decoder_type=decoder_type,
        )
        out = model(batch)
        assert out.shape == (2, 3, 2)


def test_no_update_memory_banks_are_deterministic_pooling_ops():
    cues = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    ego = torch.randn(2, 3, 3)
    assert torch.equal(LastCueMemoryBank()(cues, ego), cues[:, -1])
    assert torch.equal(MeanCueMemoryBank()(cues, ego), cues.mean(dim=1))
    static_memory = StaticMemoryBank(dim=5, num_slots=4)
    assert static_memory(cues, ego).shape == (2, 4, 5)


def test_cue_memory_strnet_tokenlearner_forward_shape():
    batch = make_batch(batch=2, history=4, future=3)
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="strnet",
        freeze_backbone=False,
        selector_type="tokenlearner",
        memory_type="attention",
        spatial_graph_neighbors=3,
        cue_temporal_layers=1,
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


def test_cue_memory_no_cv_residual_forward_shape():
    batch = make_batch(batch=2, history=5, future=3)
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="gru",
        freeze_backbone=False,
        use_constant_velocity_residual=False,
    )
    out = model(batch)
    assert out.shape == (2, 3, 2)


def test_cue_memory_multimodal_forward_shape():
    batch = make_batch(batch=2, history=5, future=3)
    model = TwoStreamEgocentricCueMemoryPathPredictor(
        future_steps=3,
        backbone_name="small_cnn",
        hidden_dim=32,
        num_cue_tokens=4,
        temporal_type="gru",
        freeze_backbone=False,
        num_modes=3,
    )
    out = model(batch)
    assert set(out) == {"paths", "logits"}
    assert out["paths"].shape == (2, 3, 3, 2)
    assert out["logits"].shape == (2, 3)


def test_cue_memory_multimodal_residual_initializes_to_constant_velocity():
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
        num_modes=3,
    )
    out = model(batch)
    expected = constant_velocity_path(batch["ego_history"], future_steps=3)
    assert torch.allclose(out["paths"], expected[:, None, :, :].expand_as(out["paths"]), atol=1e-6)


def test_loss_and_metrics():
    pred = torch.zeros(2, 3, 2)
    target = torch.ones(2, 3, 2)
    loss = trajectory_loss(pred, target, "huber")
    scaled_loss = trajectory_loss(pred, target, "huber", coordinate_scale=10.0)
    assert loss.item() > 0
    assert 0 < scaled_loss.item() < loss.item()
    assert ade(pred, target).item() > 0
    assert fde(pred, target).item() > 0


def test_trajectory_loss_accepts_sample_weights():
    pred = torch.zeros(2, 2, 2)
    target = torch.stack([torch.zeros(2, 2), torch.ones(2, 2) * 10.0], dim=0)
    unweighted = trajectory_loss(pred, target, "mse")
    first_only = trajectory_loss(pred, target, "mse", sample_weight=torch.tensor([1.0, 0.0]))
    second_only = trajectory_loss(pred, target, "mse", sample_weight=torch.tensor([0.0, 1.0]))
    assert first_only.item() == 0.0
    assert second_only.item() > unweighted.item()


def test_multimodal_loss_and_metrics_choose_best_candidate():
    target = torch.zeros(2, 3, 2)
    pred = {
        "paths": torch.stack(
            [
                torch.ones(2, 3, 2) * 10.0,
                target.clone(),
                torch.ones(2, 3, 2) * -10.0,
            ],
            dim=1,
        ),
        "logits": torch.zeros(2, 3),
    }
    loss = trajectory_loss(pred, target, multimodal_confidence_weight=0.0)
    selected = select_best_trajectory(pred, target)
    assert loss.item() == 0.0
    assert torch.allclose(selected, target)
    assert ade(pred, target).item() == 0.0
    assert fde(pred, target).item() == 0.0


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
