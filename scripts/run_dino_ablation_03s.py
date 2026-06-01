"""Run the 3s DINO visual-backbone ablation end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    Path("configs/ablations/train_dino_ablation_cv_03s.yaml"),
    Path("configs/ablations/train_dino_ablation_zero_visual_03s.yaml"),
    Path("configs/ablations/train_dino_ablation_small_cnn_03s.yaml"),
    Path("configs/ablations/train_dino_ablation_cached_dinov3_03s.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3s DINO ablation training and summarization.")
    parser.add_argument("--configs", type=Path, nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--skip-existing", action="store_true", help="Skip configs whose output metrics already exist.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocesses.")
    return parser.parse_args()


def read_output_dir(config_path: Path) -> Path:
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("output_dir:"):
            return Path(stripped.split(":", 1)[1].strip())
    raise ValueError(f"No output_dir field found in {config_path}")


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    for config in args.configs:
        if not config.exists():
            raise FileNotFoundError(config)
        output_dir = read_output_dir(config)
        metrics_path = output_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"skip existing metrics: {metrics_path}", flush=True)
            continue
        run([args.python, "-m", "src.train_path_predictor", "--config", str(config)])

    run([args.python, "scripts/summarize_dino_ablation_03s.py"])


if __name__ == "__main__":
    main()
