# WIT-VZ V2 Training - 2026-05-21

## Setup

- Dataset: `data/wit_vz/processed/wit_vz_v2_multi_source_001`
- Samples: 37,070
- Split: episode-disjoint
  - train: 25,420
  - val: 5,629
  - test: 6,021
- GPU: NVIDIA GeForce RTX 4070 SUPER
- Baseline run: `runs/wit_vz_v2_constant_velocity`
- Model run: `runs/wit_vz_v2_paper_aligned_small_cnn`

## Model

Trained the current paper-aligned cue-memory architecture with the lightweight visual path:

- Backbone: trainable `small_cnn`
- Hidden dim: 128
- Temporal adapter: `timesformer`
- Cue selector: `tokenlearner`
- Memory: `attention`
- Spatial graph: enabled, 8 neighbors
- Temporal shift: enabled
- Temporal difference conv: enabled
- Decoder: horizon query decoder
- Output: constant-velocity path + learned residual
- Loss: Huber
- Dropout: 0.2
- Weight decay: 0.001
- Mixed precision: enabled

The planned 60-epoch run was stopped after a usable checkpoint because PNG loading from the uncached RGB dataset was the bottleneck. The best checkpoint was saved at epoch 6.

Best validation checkpoint:

- Epoch: 6
- Train ADE: 18.8197
- Validation ADE: 20.3080
- Validation FDE: 31.2134

## Results

| Model | Val ADE | Val FDE | Test ADE | Test FDE |
|---|---:|---:|---:|---:|
| constant_velocity | 26.4165 | 41.5123 | 23.6033 | 37.5514 |
| paper-aligned small_cnn | 20.3080 | 31.2134 | 18.4290 | 28.4636 |

Relative to constant velocity on test:

- ADE improvement: 21.9%
- FDE improvement: 24.2%

Per-horizon test error for the trained model:

| Future Step | Error |
|---:|---:|
| 1 | 8.1038 |
| 2 | 13.6603 |
| 3 | 18.5175 |
| 4 | 23.3998 |
| 5 | 28.4636 |

## Per-Source Test Breakdown

| Source | Samples | CV ADE | CV FDE | Model ADE | Model FDE |
|---|---:|---:|---:|---:|---:|
| `wit_vz_v2_deadly_corridor_001` | 159 | 64.5676 | 105.9390 | 44.6280 | 74.7098 |
| `wit_vz_v2_health_gathering_001` | 848 | 50.4484 | 81.3543 | 34.6900 | 56.1905 |
| `wit_vz_v2_my_way_home_001` | 5,014 | 17.7640 | 27.9745 | 14.8480 | 22.3077 |

The model improves over the motion prior on every source. The overall metric is heavily influenced by `my_way_home` because it contributes most of the test samples.

## Takeaways

- The new v2 dataset is learnable with the current architecture.
- Visual/contextual modeling improves over recent-motion extrapolation, so the model is not only copying the constant-velocity prior.
- Uncached RGB training is slow. The next serious run should use a cached visual feature path or a lighter cached CNN token format.
- Full DINOv3 cache for v2 is expected to be large, roughly 18 GB based on the previous cache ratio, so it should be generated intentionally rather than as a quick smoke run.
