import torch

from src.train_path_predictor import batch_sample_weights, compute_balance_weights


def test_compute_balance_weights_upweights_minority_source():
    samples = [
        {"sample_id": "a1", "source": {"source_id": "source_a"}, "metadata": {}},
        {"sample_id": "a2", "source": {"source_id": "source_a"}, "metadata": {}},
        {"sample_id": "a3", "source": {"source_id": "source_a"}, "metadata": {}},
        {"sample_id": "b1", "source": {"source_id": "source_b"}, "metadata": {}},
    ]

    weights, stats = compute_balance_weights(samples, "source", exponent=1.0)

    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert weights[-1] > weights[0]
    assert stats["groups"]["source_a"]["count"] == 3
    assert stats["groups"]["source_b"]["count"] == 1


def test_batch_sample_weights_reads_collated_balance_metadata():
    batch = {
        "balance": [
            {"source_policy": "source_a::corridor"},
            {"source_policy": "source_b::random_walk"},
        ]
    }
    weights = batch_sample_weights(
        batch,
        "source_policy",
        {"source_a::corridor": 0.5, "source_b::random_walk": 2.0},
        torch.device("cpu"),
    )

    assert torch.allclose(weights, torch.tensor([0.5, 2.0]))
