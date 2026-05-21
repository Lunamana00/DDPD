# Egocentric Cue-Memory Path Prediction Critic

This prototype predicts a future local path from egocentric ViZDoom video history and past relative ego-motion.

The main target is not route classification, next action, or future occupancy. The supervised target is:

```text
future_local_path[t+1:t+H] = [[dx_forward, dy_right], ...]
```

The local coordinate frame uses the current player pose as origin:

- local x = forward
- local y = right
- yaw comes from ViZDoom `ANGLE`

## Architecture

The MVP model is `TwoStreamEgocentricCueMemoryPathPredictor`.

Inputs:

- RGB frame history: `I[t-L:t]`
- relative ego-motion history: `dx_forward`, `dy_right`, `dyaw`

Core components:

- visual token encoder
- bottleneck adapter after selected visual features
- optional dynamic spatial graph aggregation within each frame
- temporal adapter over visual history, either GRU/temporal Transformer or TimeSformer-style divided space-time attention, with optional temporal shift and multi-resolution temporal difference mixing
- learned cue token selector, either query-attention pooling, TokenLearner-style soft attention-map token mining, or score-gated Top-K token mining
- cue memory bank conditioned on ego-motion, either GRU-cell slots or content-addressed attention memory
- path query decoder
- constant-velocity motion prior plus learned residual future local path head

The learned head predicts a residual over a constant-velocity trajectory. Residual output is zero-initialized, so the proposed model starts from the motion baseline instead of first needing to relearn straight-line egomotion. Training uses an auto-estimated coordinate scale for the loss and residual de-normalization. Raw ADE/FDE metrics remain in local map units.

Architecture alignment note: the paper-aligned experimental path uses `selector_type=tokenlearner`, `temporal_type=timesformer`, `use_temporal_shift=true`, `use_temporal_difference_conv=true`, `use_spatial_graph=true`, and `memory_type=attention`. This implements TokenLearner-style soft spatial attention maps for cue mining, TimeSformer-style divided temporal/spatial attention, STRNet-inspired dynamic spatial graph and temporal shift/difference mixing, and Memory-Network-inspired content-addressed memory writes. It is still not a full reproduction of STRNet because the project predicts future local paths rather than goal-conditioned navigation policies and does not include STRNet's goal-observation fusion stack.

The implementation supports a DINOv2 path through optional `transformers` dependencies. In this local environment those dependencies/weights are not available, so smoke runs use `--backbone small_cnn`. This fallback is for tests and CPU prototypes, not the intended final foundation backbone.

Adapter note: the MVP implements the bottleneck adapter as a clean post-token adapter around selected visual features. This approximates "adapter after selected frozen visual blocks"; it is not an exact in-block ViT adapter insertion yet.

For DINOv3 experiments, the preferred training path is to cache frozen visual tokens first:

```text
scripts/cache_visual_features.py
  -> data/wit_vz/feature_cache/<dataset>_dinov3_convnext_tiny/
  -> train with --backbone cached_dinov3_convnext_tiny --visual-feature-cache <cache>
```

This avoids rerunning DINOv3 on every epoch and keeps the paper-aligned temporal/memory head training stable.

## Dataset Schema

Raw WIT-VZ data is stored as:

```text
data/wit_vz/raw/<run_id>/
  manifest.json
  episodes/
    episode_000001/
      frames/*.png
      depth/*.npz
      labels/*.npz
      automap/*.png
      steps.jsonl
      summary.json
```

Each `steps.jsonl` row contains:

- `frame_path`
- `pose`: `x`, `y`, `z`, `angle`
- `relative_egomotion_from_prev`
- `action`
- `reward`
- `done`
- optional `depth_path`, `labels_path`, `automap_path`
- optional `visible_labels`
- `game_variables`

Processed samples are stored as:

```text
data/wit_vz/processed/<dataset_id>/
  dataset_manifest.json
  samples.jsonl
  splits.json
  preview/*.png
```

Each sample contains:

- `rgb_history_paths`
- `relative_egomotion_history`
- `future_local_path`
- `future_world_path` for debugging only
- `current_pose`
- metadata

## Generate Mini Data

```powershell
uv run python -m src.wit_vz.collect `
  --scenario deadly_corridor `
  --run-id wit_vz_mini_001 `
  --episodes 10 `
  --max-steps 600 `
  --save-rgb `
  --save-depth `
  --save-labels `
  --save-automap `
  --mode scripted `
  --overwrite
```

The scripted controller runs real ViZDoom episodes with controlled randomness. It records action but does not use action as the main prediction target.

Human recording is scaffolded but not implemented for this environment. Use scripted mode for real generated data.

## Build Samples

```powershell
uv run python -m src.wit_vz.build_samples `
  --raw data/wit_vz/raw/wit_vz_mini_001 `
  --out data/wit_vz/processed/wit_vz_mini_001 `
  --history-sec 1.0 `
  --future-sec 1.0 `
  --sample-fps 5 `
  --stride 1
```

Splits are episode-disjoint by default to avoid leaking near-identical sliding windows across train/test.

## Train Baselines

```powershell
uv run python -m src.train_path_predictor `
  --dataset data/wit_vz/processed/wit_vz_mini_001 `
  --model constant_velocity `
  --epochs 1 `
  --batch-size 8 `
  --output-dir runs/constant_velocity

uv run python -m src.train_path_predictor `
  --dataset data/wit_vz/processed/wit_vz_mini_001 `
  --model ego_motion_only `
  --epochs 5 `
  --batch-size 8 `
  --output-dir runs/ego_motion_only
```

## Train Proposed Model

Offline smoke run:

```powershell
uv run python -m src.train_path_predictor `
  --dataset data/wit_vz/processed/wit_vz_mini_001 `
  --model cue_memory_path_predictor `
  --backbone small_cnn `
  --epochs 20 `
  --batch-size 8 `
  --output-dir runs/cue_memory_residual `
  --train-backbone `
  --trajectory-scale auto `
  --residual-scale auto
```

The older `runs/cue_memory_mini` smoke run used a frozen random `small_cnn` and direct coordinate regression. It collapsed to tiny predictions and should be treated as a failed run.

Intended DINOv2 run, after installing dependencies and making weights available:

```powershell
uv run python -m src.train_path_predictor `
  --dataset data/wit_vz/processed/wit_vz_mini_001 `
  --model cue_memory_path_predictor `
  --backbone dinov2 `
  --epochs 5 `
  --batch-size 4 `
  --output-dir runs/cue_memory_dinov2 `
  --trajectory-scale auto `
  --residual-scale auto
```

## Evaluate

```powershell
uv run python -m src.eval_path_predictor `
  --dataset data/wit_vz/processed/wit_vz_mini_001 `
  --checkpoint runs/cue_memory_residual/best.pt `
  --output-dir runs/cue_memory_residual/eval
```

Metrics:

- ADE: average displacement error
- FDE: final displacement error
- per-horizon displacement error

## Visualize

```powershell
uv run python -m src.visualize_path_predictions `
  --dataset data/wit_vz/processed/wit_vz_mini_001 `
  --predictions runs/cue_memory_residual/eval/predictions.jsonl `
  --out runs/cue_memory_residual/figures `
  --num-samples 20
```

Each figure shows:

- current egocentric frame
- GT future local path
- predicted future local path
- forward/right axes
- ADE/FDE

## Compare Runs

```powershell
uv run python -m src.compare_models `
  --runs runs/constant_velocity runs/ego_motion_only runs/cue_memory_mini runs/cue_memory_residual `
  --out reports/path_prediction_comparison.md
```

Current mini comparison:

```text
constant_velocity:    ADE=56.2430 FDE=96.1106
ego_motion_only:      ADE=86.5778 FDE=144.3438
cue_memory_mini:      ADE=85.2724 FDE=143.2239  failed direct-regression smoke run
cue_memory_residual:  ADE=51.5144 FDE=91.0359
```

## Horizon Sweep

The original `deadly_corridor` mini run is too short for long horizons: with `sample_fps=5`, it yields samples through about 3 seconds and zero samples from 4 to 10 seconds. For the 1-10 second sweep, collect a longer timeout-style run with `basic`:

```powershell
uv run python -m src.wit_vz.collect `
  --scenario basic `
  --run-id wit_vz_basic_10s `
  --episodes 12 `
  --max-steps 150 `
  --frame-skip 4 `
  --seed 13 `
  --mode scripted `
  --policy random `
  --overwrite
```

Then build one processed dataset per horizon and train/evaluate both CV and cue-memory residual runs:

```powershell
uv run python scripts/run_horizon_sweep.py `
  --raw data/wit_vz/raw/wit_vz_basic_10s `
  --processed-root data/wit_vz/processed/horizon_sweep `
  --runs-root runs/horizon_sweep `
  --min-sec 1 `
  --max-sec 10 `
  --epochs 8 `
  --batch-size 8 `
  --device cpu
```

Summary files:

- `runs/horizon_sweep/horizon_summary.json`
- `runs/horizon_sweep/horizon_summary.md`

Generate PNG figures:

```powershell
uv run python scripts/plot_horizon_sweep.py `
  --summary runs/horizon_sweep/horizon_summary.json `
  --out-dir runs/horizon_sweep/figures
```

Figure outputs:

- `runs/horizon_sweep/figures/horizon_ade_fde.png`
- `runs/horizon_sweep/figures/horizon_improvement.png`
- `runs/horizon_sweep/figures/horizon_summary.png`

Current 1-10 second result:

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

## Why Future Local Path?

Future local path directly captures where the player actually moves in the next few seconds. This better matches design critique questions like readability, hesitation, and movement intent than discrete route labels. A route ID may hide within-route behavior, while local path supervision keeps the target continuous and grounded in actual trajectory data.

Action is recorded for analysis and ablations, but it is not the main target. Future occupancy is left for a later extension.

Depth and labels are auxiliary or upper-bound metadata. The main inference path is RGB history plus relative ego-motion.

## Current Limitations

- Scripted trajectories are not large-scale human trajectories.
- ViZDoom is a limited domain.
- DINOv2/adapter choices need ablation.
- The current DINOv2 path requires optional dependencies and model weights.
- Future occupancy distribution is future work.
- Human recording mode needs environment-specific input handling.
