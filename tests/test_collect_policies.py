import random

from src.wit_vz.collect import ACTION_SPACE, choose_scripted_action


def test_scripted_policy_variants_return_valid_actions():
    rng = random.Random(7)
    policies = [
        "corridor",
        "random",
        "random_walk",
        "noisy_corridor",
        "goal_directed",
        "obstacle_avoidance",
    ]

    for step, policy in enumerate(policies):
        action_id, action_name, action_vector = choose_scripted_action(
            rng,
            policy,
            step,
            pose={"x": 0.0, "y": 0.0, "angle": 0.0},
            goal=(128.0, 0.0),
            policy_noise=0.0,
        )
        assert 0 <= action_id < len(ACTION_SPACE)
        assert action_name
        assert len(action_vector) == 6


def test_policy_noise_can_override_policy_action():
    action_id, action_name, action_vector = choose_scripted_action(
        random.Random(3),
        "corridor",
        0,
        policy_noise=1.0,
    )

    assert 0 <= action_id < len(ACTION_SPACE)
    assert action_name.endswith("[noise]")
    assert len(action_vector) == 6
