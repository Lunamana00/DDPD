import os
import subprocess
import sys

import pytest


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
