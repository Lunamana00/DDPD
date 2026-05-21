# Current Progress

Updated: 2026-05-18

## Project Scope

This repository currently contains a ViZDoom-based research prototype for egocentric future path prediction.

The current task is not route classification, next-action prediction, or occupancy prediction. The supervised target is a continuous future local path:

```text
future_local_path[t+1:t+H] = [[dx_forward, dy_right], ...]
```

The model predicts future local path waypoints from:

- egocentric RGB history
- relative ego-motion history
- optional learned visual cue memory

Route IDs such as `chosen_route_id` and `nearest_route_id` are not used as the main learning target in the current prototype.

## Implemented Code

Core package:

- `src/wit_vz/collect.py`: WIT-VZ ViZDoom data collection.
- `src/wit_vz/build_samples.py`: raw episode records to processed path-prediction samples.
- `src/wit_vz/dataset.py`: PyTorch dataset and collate logic.
- `src/wit_vz/geometry.py`: egocentric coordinate conversion and relative ego-motion.
- `src/wit_vz/io.py`: JSON/JSONL/image IO helpers.

Training and evaluation:

- `src/train_path_predictor.py`: model/baseline training loop.
- `src/eval_path_predictor.py`: checkpoint evaluation.
- `src/compare_models.py`: Markdown comparison table generation.
- `src/visualize_path_predictions.py`: prediction-vs-ground-truth visualization.
- `src/visualize_vizdoom_replay.py`: animated ViZDoom RGB replay with predicted and GT local paths, without Unity.
- `src/metrics.py`: ADE, FDE, per-horizon error.
- `src/losses.py`: trajectory losses.

Models:

- `constant_velocity`: motion-prior baseline.
- `ego_motion_only`: GRU over ego-motion only.
- `cue_memory_path_predictor`: visual cue-memory path predictor.
- `cue_memory_residual`: cue-memory model with constant-velocity residual path head.
- `small_cnn`: offline smoke-test visual backbone.
- `dinov2`: optional DINOv2 path via `transformers`.

Scripts:

- `scripts/collect_route_vizdoom.py`: earlier route-aware ViZDoom collection prototype.
- `scripts/collect_wit_vz_game_benchmark.py`: multi-scenario ViZDoom scripted data expansion.
- `scripts/record_vizdoom_human_session.py`: interactive ViZDoom human recording and sample building.
- `scripts/run_horizon_sweep.py`: build/train/evaluate multiple future horizons.
- `scripts/plot_horizon_sweep.py`: plot horizon sweep metrics using PIL.

## Data State

Local data exists under `data/`, but it is intentionally ignored by Git.

Known local datasets:

- `data/doomframe_sample`
- `data/route_vizdoom/runs/route_demo_3ep`
- `data/wit_vz/raw/wit_vz_mini_001`
- `data/wit_vz/raw/wit_vz_horizon_10s`
- `data/wit_vz/raw/wit_vz_basic_10s`
- `data/wit_vz/processed/wit_vz_mini_001`
- `data/wit_vz/processed/horizon_sweep/future_01s` through `future_10s`

The Git repository tracks code, configs, documentation, metrics, and selected figures. It does not track raw data, processed samples, checkpoints, or prediction dumps.

## Current Mini Results

The latest useful mini comparison is:

| Run | Model | ADE | FDE | Notes |
| --- | --- | ---: | ---: | --- |
| `constant_velocity` | `constant_velocity` | 56.2430 | 96.1106 | Motion prior baseline |
| `ego_motion_only` | `ego_motion_only` | 86.5778 | 144.3438 | GRU over egomotion only |
| `cue_memory_mini` | `cue_memory_path_predictor` | 85.2724 | 143.2239 | Failed direct-regression smoke run; tiny prediction collapse |
| `cue_memory_residual` | `cue_memory_path_predictor` | 51.5144 | 91.0359 | CV residual + auto scale + trainable small CNN |

The useful current model is `cue_memory_residual`. It improves over `constant_velocity` on the mini test split.

## Horizon Sweep Results

The horizon sweep trains/evaluates from 1 to 10 seconds on `data/wit_vz/raw/wit_vz_basic_10s`.

| Horizon | Samples | Steps | CV ADE | CV FDE | Model ADE | Model FDE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | 594 | 5 | 39.2304 | 60.1868 | 37.9153 | 58.8077 |
| 2s | 549 | 10 | 62.9874 | 105.9262 | 63.0989 | 100.0550 |
| 3s | 504 | 15 | 86.3891 | 151.7404 | 79.3597 | 125.6612 |
| 4s | 459 | 20 | 110.7841 | 200.3845 | 92.6877 | 141.9131 |
| 5s | 414 | 25 | 134.8382 | 252.0663 | 111.1487 | 165.9537 |
| 6s | 369 | 30 | 161.3506 | 288.3283 | 129.8960 | 183.4811 |
| 7s | 324 | 35 | 191.0831 | 340.2279 | 144.1681 | 221.7033 |
| 8s | 279 | 40 | 221.1716 | 416.9505 | 162.9982 | 268.3496 |
| 9s | 234 | 45 | 255.2140 | 502.8008 | 156.4600 | 272.7639 |
| 10s | 189 | 50 | 315.8391 | 602.4938 | 178.9178 | 297.9203 |

Main observation:

- The learned residual model becomes more useful as the prediction horizon grows.
- At 10 seconds, `cue_memory_residual` reaches ADE `178.9178` and FDE `297.9203`, while `constant_velocity` reaches ADE `315.8391` and FDE `602.4938`.

## Documentation Already Present

Main documents:

- `README.md`
- `docs/path_prediction.md`
- `reports/path_prediction_comparison.md`
- `runs/horizon_sweep/horizon_summary.md`

Earlier dataset/collection notes:

- `VIZDOOM_DATASET_STRUCTURE.md`
- `ROUTE_VIZDOOM_COLLECTION.md`

## Test Status

Current test command:

```bash
uv run pytest -q
```

Latest result:

```text
9 passed, 1 skipped
```

The skipped test is `tests/test_vizdoom_integration.py`. It requires:

```bash
RUN_VIZDOOM_INTEGRATION=1
```

That test performs a live ViZDoom collection smoke test and is intentionally skipped by default.

## Git Cleanup Completed

Initial Git cleanup was completed.

Current commit:

```text
dc1d6f0 Initial path prediction prototype
```

Added repository hygiene:

- `.gitignore`
- `.gitattributes`
- `pyproject.toml`
- refreshed `uv.lock`

Ignored local artifacts:

- `.venv/`
- `.pytest_cache/`
- `__pycache__/`
- `data/`
- `outputs/`
- `_vizdoom.ini`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `runs/**/predictions.jsonl`

Tracked artifacts:

- source code
- configs
- tests
- documentation
- metrics JSON
- selected PNG figures
- horizon summary reports

## Current Repository State

Before adding this document, the repository was clean on branch `main` after the initial prototype commit.

```text
## main
```

This progress note is intended to be tracked as a separate documentation commit after the initial prototype commit.

## Remaining Work

Near-term engineering work:

- Add clearer reproduction commands to `README.md`.
- Decide whether `requirements-vizdoom.txt` should be kept or replaced by `pyproject.toml` extras.
- Add a lightweight script or Makefile for common commands.
- Add data download/generation instructions because `data/` is intentionally not tracked.

Research work:

- Run larger-scale data collection.
- Repeat experiments across seeds.
- Add ablations for cue memory, residual scaling, backbone, and temporal module.
- Compare against stronger learned baselines.
- Decide final target framing for presentation or paper.

Presentation/report work:

- Turn `docs/path_prediction.md` into a concise capstone report section.
- Use `runs/horizon_sweep/figures/` for result slides.
- Add qualitative examples from `runs/cue_memory_residual/figures/`.
