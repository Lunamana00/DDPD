from scripts.replay_sauerkrautlm_human_actions_wit_vz import (
    episode_row_ranges,
    scores_to_action_vector,
)


def test_scores_to_action_vector_maps_human_scores_to_vizdoom_buttons():
    vector, name, labels = scores_to_action_vector([0.0, 0.8, 0.75, 0.0], threshold=0.5)

    assert vector == [0, 1, 0, 0, 0, 1]
    assert name == "move_forward+turn_left"
    assert labels == ["move_forward", "turn_left"]


def test_scores_to_action_vector_can_use_argmax_when_no_score_crosses_threshold():
    vector, name, labels = scores_to_action_vector([0.1, 0.4, 0.2, 0.3], threshold=0.5, argmax_if_empty=True)

    assert vector == [0, 1, 0, 0, 0, 0]
    assert name == "move_forward"
    assert labels == ["move_forward"]


def test_episode_row_ranges_chunks_external_rows():
    ranges = episode_row_ranges(
        total_rows=25,
        episodes=4,
        episode_steps=6,
        start_index=2,
        episode_gap=1,
    )

    assert [(item.start, item.stop) for item in ranges] == [(2, 8), (9, 15), (16, 22), (23, 25)]
