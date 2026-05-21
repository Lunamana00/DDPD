# Dataset And Training Method Summary

Date: 2026-05-21

This note summarizes the dataset actually used for the current pushed
checkpoints and the corresponding training procedure.

## Scope

The current main result uses the WIT-VZ v4 default-scenario dataset:

- Raw root pattern: `data/wit_vz/raw/wit_vz_v4_default_*_001`
- Main processed dataset: `data/wit_vz/processed/wit_vz_v4_defaults_001`
- Source-disjoint split: `data/wit_vz/processed/wit_vz_v4_defaults_source_disjoint_001`
- Map-disjoint split: `data/wit_vz/processed/wit_vz_v4_defaults_map_disjoint_001`
- DINOv3 cache: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`
- Published checkpoints: `checkpoints/wit_vz_v4_defaults_dinov3_single_*.pt`

Older v2 runs were kept as development baselines, but the pushed v4
checkpoints are trained from the v4 default-scenario data.

## Raw Data Collection

The dataset was collected from ViZDoom with scripted agents. Each raw run saves
RGB frames plus per-step metadata: pose, game variables, action, reward,
episode termination state, and relative ego-motion from the previous step.

Common collection settings:

- Environment: ViZDoom
- Map: `map01`
- Episodes per scenario: `40`
- Maximum recorded steps per episode: `300`
- ViZDoom frame skip: `4`
- Doom tick rate recorded in manifests: `35 Hz`
- Effective raw sampling interval: `4 / 35 = 0.114s`
- RGB resolution: `160x120`
- RGB saved: yes
- Depth, labels, automap: disabled for the v4 default run
- Random seed range: `402` through `416`

Collected scenarios:

| Scenario | Episodes | Samples | Policy family |
| --- | ---: | ---: | --- |
| `basic` | 40 | 10,279 | `random_walk` |
| `basic_audio` | 40 | 10,315 | `random_walk` |
| `basic_notifications` | 40 | 10,554 | `random_walk` |
| `deadly_corridor` | 40 | 240 | mixed |
| `deathmatch` | 40 | 6,584 | mixed |
| `defend_the_center` | 40 | 2,594 | mixed |
| `defend_the_line` | 40 | 2,587 | mixed |
| `health_gathering` | 40 | 3,858 | mixed |
| `health_gathering_supreme` | 40 | 2,682 | mixed |
| `multi_deathmatch` | 40 | 10,493 | mixed |
| `my_way_home` | 40 | 9,951 | mixed |
| `predict_position` | 40 | 2,436 | mixed |
| `rocket_basic` | 40 | 10,054 | `random_walk` |
| `simpler_basic` | 40 | 10,276 | `random_walk` |
| `take_cover` | 40 | 500 | mixed |

Total: `15` runnable scenarios, `600` raw episodes, `93,403` supervised
samples for the 1-second future target.

The mixed policy chooses an episode-level policy from:

- `random_walk`
- `noisy_corridor`
- `goal_directed`
- `obstacle_avoidance`

For mixed-policy scenarios, collection used `policy_noise=0.08`,
`start_random_steps=5`, and `start_random_jitter=25` to diversify initial pose
and short-term movement. `goal_directed` uses rotating default goals when no
explicit goal is passed.

Three installed default WADs were smoke-tested but not collected because they
segfaulted during initialization on this host:

- `cig`
- `cig_with_unknown`
- `multi_duel`

## Supervised Sample Construction

Raw episodes were converted into path-prediction samples by
`src/wit_vz/build_samples.py`.

Main 1-second dataset settings:

- History window: `1.0s`
- Future window: `1.0s`
- Sample FPS: `5.0`
- History frames: `5`
- Future steps: `5`
- Raw step gap: `2`
- Sliding-window stride: `1`
- Target: `future_local_path`
- Coordinate convention: local `x = forward`, local `y = right`, origin at the
  current pose.

Each sample contains:

- `rgb_history_paths`: the 5 RGB frames leading up to the current time
- `relative_egomotion_history`: `[dx_forward, dy_right, dyaw]` for each history
  frame
- `future_local_path`: future positions transformed from world coordinates into
  the current egocentric frame
- `future_world_path` and `current_pose` for inspection/debugging
- metadata for `source`, `scenario`, `map`, `policy`, and `episode`

The local target transform uses the current Doom pose as the origin. Doom
`POSITION_X`, `POSITION_Y`, and `ANGLE` are converted so that future points are
expressed as `[forward, right]` relative to the agent's current heading.

## Splits

The primary training split is episode-disjoint:

| Split | Samples |
| --- | ---: |
| Train | 64,620 |
| Val | 13,410 |
| Test | 15,373 |

Two stricter evaluation datasets were also built from the same raw data:

| Split type | Train | Val | Test | Purpose |
| --- | ---: | ---: | ---: | --- |
| Source-disjoint | 56,567 | 12,387 | 24,449 | Hold out source/scenario-style groups |
| Map-disjoint | 59,243 | 20,833 | 13,327 | Hold out map group keys |

The processed manifests include leakage diagnostics over source, scenario, map,
policy, episode, source-scenario, source-map, and source-policy keys. The
strict source/map builds showed empty overlap for the intended held-out group
keys.

## Horizon Datasets

For time-horizon analysis, additional datasets were generated from the same v4
raw runs with the same 1-second history and `5 Hz` sample rate, changing only
the future window:

| Horizon | Future steps | Samples | Train | Val | Test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | 5 | 93,403 | 64,620 | 13,410 | 15,373 |
| 3s | 15 | 82,848 | 58,317 | 12,647 | 11,884 |
| 5s | 25 | 73,149 | 51,173 | 11,682 | 10,294 |
| 10s | 50 | 52,765 | 36,601 | 8,730 | 7,434 |
| 30s | 150 | 0 | 0 | 0 | 0 |

The 30-second horizon is not trainable from the current raw episodes because
the episodes do not contain enough future context after the history window.
Longer raw episodes are required before training a 30-second predictor.

## Visual Feature Cache

Instead of training directly on RGB for the final runs, frozen DINOv3 tokens
were precomputed for every processed sample.

Cache settings:

- Backbone: `dinov3-convnext-tiny`
- Input image size: `256`
- History frames per sample: `5`
- Stored payload: tensor-only `.pt`
- Token shape per sample: `[5, 64, 768]`
- Cached files: `93,403`
- Cache size: about `44G`
- Generation: 3 shards on 3 RTX 3090 GPUs
- Combined manifest: `feature_manifest.json`

The dataset loader then uses `load_rgb=False` and loads cached visual tokens
from `features/{sample_id}.pt`, reducing repeated DINOv3 forward cost during
training.

## Training Setup

Final model family:

- Model entrypoint: `cue_memory_path_predictor`
- Visual input: cached DINOv3 ConvNeXt-Tiny tokens
- Temporal block: TimeSFormer-style divided temporal/spatial attention
- Cue selector: TokenLearner-style selector
- Cue tokens: `8`
- Memory: attention memory
- Decoder: learned future-step queries with cross-attention into cue memory
- Output: one deterministic future trajectory
- Motion prior: constant-velocity residual enabled

Final 1-second training config:

- Dataset: `data/wit_vz/processed/wit_vz_v4_defaults_001`
- Cache: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`
- Epoch budget: `80`
- Actual best epoch: `10`
- Early stopping epoch: `24`
- Batch size: `512`
- Hardware: 3-GPU `torch.nn.DataParallel`
- Mixed precision: enabled
- Optimizer: AdamW
- LR: `5e-4`
- LR scheduler: ReduceLROnPlateau, factor `0.5`, patience `6`, min LR `1e-6`
- LR reductions observed: `2.5e-4` at epoch 17, `1.25e-4` at epoch 24
- Weight decay: `0.001`
- Dropout: `0.2`
- Loss: Huber trajectory loss
- Gradient clipping: `1.0`
- Coordinate scaling: automatic from train targets
- Residual scale: automatic
- Balance key: `source_policy`
- Balance mode: `both`
- Balance exponent: `0.5`

Balancing is applied in two places:

- Sampler balancing: `WeightedRandomSampler` samples inverse-frequency
  source-policy groups with exponent `0.5`.
- Loss balancing: per-sample loss weights use the same group weights.

This soft balancing reduces domination by high-volume sources without making
rare, short-episode scenarios fully overpower the training distribution.

## Evaluation Metrics

The main metrics are:

- ADE: average Euclidean distance over all predicted future steps
- FDE: Euclidean distance at the final predicted future step
- Per-horizon error: Euclidean error at each future step

For the final 1-second checkpoint:

| Evaluation | ADE | FDE |
| --- | ---: | ---: |
| Episode-disjoint test | 26.8676 | 41.5629 |
| Source-disjoint test | 25.3887 | 38.5235 |
| Map-disjoint test | 22.1936 | 33.7372 |

Per-step error for the 1-second episode-disjoint test:

```text
[10.4377, 19.7209, 27.7245, 34.8919, 41.5629]
```

## Horizon Results

Horizon-specific models reused the same v4 raw data and DINOv3 cache strategy.
All trainable horizons used the same single-output model family, 3-GPU
DataParallel, mixed precision, `batch_size=512`, and `source_policy` balancing.

| Horizon | Constant-velocity ADE/FDE | Model ADE/FDE | ADE gain | FDE gain |
| ---: | ---: | ---: | ---: | ---: |
| 1s | 33.1120 / 51.4413 | 26.8676 / 41.5629 | 18.9% | 19.2% |
| 3s | 75.7201 / 131.6904 | 62.1001 / 103.3531 | 18.0% | 21.5% |
| 5s | 111.2669 / 202.7233 | 88.6020 / 157.0852 | 20.4% | 22.5% |
| 10s | 217.1669 / 408.6508 | 154.5734 / 258.7196 | 28.8% | 36.7% |

Published checkpoint mapping:

| File | Horizon | Source run |
| --- | ---: | --- |
| `checkpoints/wit_vz_v4_defaults_dinov3_single_01s.pt` | 1s | `runs/wit_vz_v4_defaults_dinov3_timesformer_balanced_dp_single_bs512` |
| `checkpoints/wit_vz_v4_defaults_dinov3_single_03s.pt` | 3s | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_03s` |
| `checkpoints/wit_vz_v4_defaults_dinov3_single_05s.pt` | 5s | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_05s` |
| `checkpoints/wit_vz_v4_defaults_dinov3_single_10s.pt` | 10s | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_10s` |

## Important Limitations

- `deadly_corridor` and `take_cover` are underrepresented because episodes
  often ended quickly under the current policy mix.
- The strict source/map test sets are not necessarily harder than the main
  episode-disjoint test under this generated-policy distribution.
- The current dataset predicts egocentric local future path, not global route
  identity or map-level navigation goal.
- Data and feature caches are too large for normal Git tracking. The pushed Git
  repo contains code, configs, reports, and lightweight selected checkpoints;
  raw data, processed samples, DINOv3 caches, and prediction dumps remain
  server-local or should be handled through DVC/object storage.

## Reproduction Pointers

Primary configs:

- Collection: `configs/wit_vz_collect_defaults_v4.yaml`
- Main build: `configs/wit_vz_build_v4_defaults.yaml`
- Source-disjoint build: `configs/wit_vz_build_v4_defaults_source_disjoint.yaml`
- Map-disjoint build: `configs/wit_vz_build_v4_defaults_map_disjoint.yaml`
- Training: `configs/train_wit_vz_v4_defaults_dinov3_balanced_timesformer.yaml`

Related reports:

- `reports/vizdoom_default_scenarios_dataset_20260521.md`
- `reports/v4_dinov3_cache_training_20260521.md`
- `reports/horizon_sweep_v4_defaults_single_output_20260521.md`
