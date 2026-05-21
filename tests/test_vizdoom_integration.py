import os
import subprocess
import sys

import pytest

from src.wit_vz.collect import action_id_for_vector, action_name_from_vector


def test_human_action_vector_is_named_and_matched():
    vector = [0, 1, 0, 0, 1, 0]

    assert action_name_from_vector(
        vector,
        ["ATTACK", "MOVE_FORWARD", "MOVE_RIGHT", "MOVE_LEFT", "TURN_RIGHT", "TURN_LEFT"],
    ) == "MOVE_FORWARD+TURN_RIGHT"
    assert action_id_for_vector(vector) == 10


def test_pause_action_vector_name():
    assert action_name_from_vector([0, 0, 0], ["ATTACK", "MOVE_FORWARD", "TURN_RIGHT"]) == "PAUSE"


@pytest.mark.skipif(
    os.environ.get("RUN_VIZDOOM_INTEGRATION") != "1",
    reason="Set RUN_VIZDOOM_INTEGRATION=1 to run ViZDoom collection integration test.",
)
def test_vizdoom_collection_smoke(tmp_path):
    pytest.importorskip("vizdoom")
    out_root = tmp_path / "raw"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.wit_vz.collect",
            "--episodes",
            "1",
            "--max-steps",
            "5",
            "--run-id",
            "itest",
            "--out-root",
            str(out_root),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote WIT-VZ raw run" in result.stdout
    assert (out_root / "itest" / "manifest.json").exists()
