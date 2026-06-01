# DINO Ablation v4 3s

Retraining-time ablation over the v4 3s WIT-VZ split.
All trainable variants use the same cue-memory path predictor, TimeSFormer-style temporal adapter, TokenLearner selector, attention cue memory, horizon query decoder, constant-velocity residual prior, seed, optimizer, and source-policy balancing. The changed factor is the visual evidence source.

Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s`.
DINO cache: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`.

| Variant | Backbone | Available | Epochs | Best epoch | Test ADE | Test FDE | Best Val ADE | Train-Val ADE Gap | Avg epoch sec | Peak CUDA MB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | small_cnn | yes | 0 | - | 75.7201 | 131.6904 | 70.5372 | - | - | - |
| zero_visual | zero_tokens | yes | 21 | 7 | 70.2319 | 119.1072 | 66.2466 | -3.2618 | 25.91 | 3632.3 |
| small_cnn | small_cnn | yes | 34 | 20 | 65.2136 | 108.7450 | 59.8070 | 10.9139 | 59.04 | 4413.6 |
| cached_dinov3 | cached_dinov3_convnext_tiny | yes | 17 | 3 | 61.8506 | 103.5420 | 57.8149 | -2.3777 | 54.74 | 4433.4 |

## Prefix Metrics

The 3s model predicts 15 future points at 5 FPS. Prefix metrics reuse the same prediction and score the first 5/10/15 points as 1s/2s/3s evidence.

| Variant | 1s ADE | 1s FDE | 2s ADE | 2s FDE | 3s ADE | 3s FDE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 34.2716 | 52.6049 | 55.5224 | 92.6405 | 75.7201 | 131.6904 |
| zero_visual | 32.4757 | 49.7272 | 52.2112 | 86.2128 | 70.2319 | 119.1072 |
| small_cnn | 31.2997 | 47.4951 | 49.1859 | 79.4106 | 65.2136 | 108.7450 |
| cached_dinov3 | 31.2709 | 44.4001 | 46.8191 | 74.3256 | 61.8506 | 103.5420 |

## DINO Gain
- `cached_dinov3_vs_constant_velocity`: ADE gain 18.32%, FDE gain 21.37%.
- `cached_dinov3_vs_zero_visual`: ADE gain 11.93%, FDE gain 13.07%.
- `cached_dinov3_vs_small_cnn`: ADE gain 5.16%, FDE gain 4.78%.

## Interpretation

- Best test ADE: `cached_dinov3` (61.8506).
- If `cached_dinov3` beats `small_cnn`, pretrained dense visual tokens add useful game-navigation signal beyond learning a small RGB encoder from this dataset alone.
- If `cached_dinov3` only beats `zero_visual` but not `constant_velocity`, the visual branch is learning some image-conditioned residuals but the task is still dominated by recent ego-motion.
- If `small_cnn` is competitive with DINO, the v4 ViZDoom visual domain may be simple enough that task-specific RGB features match frozen foundation features for this horizon.
