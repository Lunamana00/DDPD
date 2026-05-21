# V4 Default Horizon Sweep Single-Output Results

Date: 2026-05-21

## Setup

- Raw data: v4 default ViZDoom scenarios, 15 runnable scenario sources.
- Processed root: `data/wit_vz/processed/horizon_sweep_v4_defaults`
- Run root: `runs/horizon_sweep_v4_defaults`
- Visual cache reused from: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`
- Cache coverage: missing cache files `0` for 1s, 3s, 5s, and 10s datasets.
- History window: 1 second, 5 frames.
- Sample FPS: 5.0.
- Split: episode-disjoint.
- Model: single deterministic future path output, cached DINOv3 ConvNeXt-Tiny tokens, TimeSFormer-style temporal block, TokenLearner cue selection, attention memory.
- Training: 3-GPU DataParallel, mixed precision, `batch_size=512`, `source_policy` balancing.

The 30-second horizon produced `0` samples from the current v4 raw episodes, so it was not trainable without collecting longer episodes.

## Dataset Sizes

| Horizon | Future steps | Samples | Train | Val | Test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | 5 | 93,403 | 64,620 | 13,410 | 15,373 |
| 3s | 15 | 82,848 | 58,317 | 12,647 | 11,884 |
| 5s | 25 | 73,149 | 51,173 | 11,682 | 10,294 |
| 10s | 50 | 52,765 | 36,601 | 8,730 | 7,434 |
| 30s | 150 | 0 | 0 | 0 | 0 |

## Test Results

| Horizon | CV ADE | CV FDE | Model ADE | Model FDE | ADE gain | FDE gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | 33.1120 | 51.4413 | 26.8676 | 41.5629 | 18.9% | 19.2% |
| 3s | 75.7201 | 131.6904 | 62.1001 | 103.3531 | 18.0% | 21.5% |
| 5s | 111.2669 | 202.7233 | 88.6020 | 157.0852 | 20.4% | 22.5% |
| 10s | 217.1669 | 408.6508 | 154.5734 | 258.7196 | 28.8% | 36.7% |

## Run Paths

| Horizon | Model run |
| ---: | --- |
| 1s | `runs/wit_vz_v4_defaults_dinov3_timesformer_balanced_dp_single_bs512` |
| 3s | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_03s` |
| 5s | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_05s` |
| 10s | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_10s` |

Aggregate local summaries:

- `runs/horizon_sweep_v4_defaults/horizon_summary_with_model.json`
- `runs/horizon_sweep_v4_defaults/horizon_summary_with_model.md`

## Takeaways

- The single-output visual model beats constant velocity at every trainable horizon.
- Relative improvement grows at longer horizons, especially FDE at 10 seconds.
- Validation curves show earlier overfitting as the horizon gets longer, so stronger regularization or more long episodes should be prioritized before extending beyond 10 seconds.
- To train 30-second prediction, collect longer raw episodes first; the current v4 default scenario episodes do not contain enough future context.
