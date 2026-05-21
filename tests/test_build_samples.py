from src.wit_vz.build_samples import make_group_splits, summarize_split_diagnostics


def make_sample(sample_id, source_id, scenario, map_id, episode_id, policy):
    return {
        "sample_id": sample_id,
        "episode_id": episode_id,
        "source": {"source_id": source_id, "env_name": "vizdoom"},
        "metadata": {
            "source_id": source_id,
            "env_name": "vizdoom",
            "scenario": scenario,
            "map_id": map_id,
            "policy": policy,
        },
    }


def test_source_disjoint_split_diagnostics_report_group_overlap():
    samples = [
        make_sample("a1", "source_a", "deadly_corridor", "map01", "ea1", "corridor"),
        make_sample("a2", "source_a", "deadly_corridor", "map01", "ea2", "random_walk"),
        make_sample("b1", "source_b", "take_cover", "map01", "eb1", "goal_directed"),
        make_sample("c1", "source_c", "predict_position", "map01", "ec1", "obstacle_avoidance"),
    ]

    splits, resolved = make_group_splits(samples, seed=1, strategy="source")
    diagnostics = summarize_split_diagnostics(samples, splits)

    assert resolved == "source"
    assert diagnostics["by_split"]["train"]["num_samples"] >= 1
    assert diagnostics["leakage"]["source"]["train_val_overlap"] == []
    assert diagnostics["leakage"]["source"]["train_test_overlap"] == []
    assert "policy" in diagnostics["by_split"]["train"]["groups"]
