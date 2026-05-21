# V4 Inference-Time Ablation

Date: 2026-05-22

This is an inference-time ablation only. No checkpoint was retrained, no
data was downloaded, and no new data was collected for this evaluation.

## What The Model Predicts

The model predicts an egocentric future local path, not a global map
trajectory, route identity, action sequence, or semantic goal. For each
sample, the input is 1 second of visual history plus relative ego-motion
history. The output shape is `[B, H, 2]`, where `H` is the number of future
steps at 5 FPS. Each point is `[forward, right]` in the current pose's
local coordinate frame, with the origin at the current agent pose.

## Inputs And Checkpoints

- Main v4 dataset: `data/wit_vz/processed/wit_vz_v4_defaults_001`
- Horizon root: `data/wit_vz/processed/horizon_sweep_v4_defaults`
- DINOv3 cache: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`
- Checkpoint root: `checkpoints`
- Split: `test`
- Limit: `none`
- Batch size: `512`

## Ablation Cases

- `constant_velocity`: No visual model; extrapolates recent ego-motion linearly.
- `full_model`: Original trained checkpoint inference path.
- `zero_visual_tokens`: Sets all cached visual tokens to zero before the model.
- `static_visual_tokens`: Repeats the last frame token grid across the whole history.
- `no_temporal_adapter`: Skips the TimeSFormer-style temporal/spatial fusion adapter.
- `uniform_selector`: Replaces adaptive TokenLearner cues with repeated spatial means.
- `no_cue_temporal`: Skips temporal modeling over selected cue tokens.
- `no_memory_update`: Bypasses the cue memory bank and decodes from the last cue set.
- `no_ego_memory`: Zeros ego-motion only inside memory updates; keeps the final CV base intact.

## Overall Findings

The full-model rows match the previously reported v4 horizon test metrics, so the ablation script is evaluating the intended checkpoints and splits.

| Horizon | DINO off ego-only | DINO signal off | No temporal adapter | No cue temporal | No memory update | No ego memory |
| --- | --- | --- | --- | --- | --- | --- |
| 1s | +23.2% | +24.9% | +4.6% | +5.2% | +209.5% | +15.5% |
| 3s | +21.9% | +17.1% | +6.2% | +19.3% | +114.6% | +24.2% |
| 5s | +25.6% | +37.9% | +13.7% | +48.5% | +150.0% | +28.3% |
| 10s | +40.5% | +8.6% | +4.2% | +16.5% | +103.0% | +43.6% |

- DINO/visual information helps at every horizon. Removing visual modeling entirely (`constant_velocity`) worsens ADE by 21.9% to 40.5%; zeroing the DINO tokens inside the trained model worsens ADE by 8.6% to 37.9%.
- The cue memory update is the largest inference-time dependency. `no_memory_update` is the worst ablation at every horizon, worsening ADE by 103.0% to 209.5%.
- Ego-motion conditioning inside the memory matters, especially as the prediction horizon grows: `no_ego_memory` worsens ADE by 15.5% to 43.6%.
- Cue temporal modeling has a meaningful effect beyond 1s. `no_cue_temporal` worsens ADE by 5.2% to 48.5%.
- The TimeSFormer-style temporal adapter has a smaller but consistent positive effect in this inference test, with ADE degradation from 4.2% to 13.7%.
- Repeating the last visual token grid over time has only a small effect (1.5% to 4.6% ADE), suggesting that visual content dominates over short-term visual change for these checkpoints.
- `uniform_selector` is weakly mixed (-0.5% to 4.0% ADE). Because this is inference-time surgery rather than retraining, it should not be used alone to claim the learned selector is unnecessary.

## 1s Horizon

- Dataset: `data/wit_vz/processed/wit_vz_v4_defaults_001`
- Checkpoint: `checkpoints/wit_vz_v4_defaults_dinov3_single_01s.pt`
- Future steps: `5`
- Evaluated samples per case: `15373`

DINO on/off summary:

| Comparison | Case | ADE | FDE | Delta ADE | Delta FDE | ADE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `DINO on` | `full_model` | 26.8676 | 41.5629 | 0.0000 | 0.0000 | +0.0% |
| `DINO off, ego only` | `constant_velocity` | 33.1120 | 51.4413 | 6.2445 | 9.8785 | +23.2% |
| `DINO signal off` | `zero_visual_tokens` | 33.5479 | 52.5510 | 6.6804 | 10.9881 | +24.9% |

| Case | ADE | FDE | Delta ADE vs full | Delta FDE vs full | ADE rel. | FDE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `constant_velocity` | 33.1120 | 51.4413 | 6.2445 | 9.8785 | +23.2% | +23.8% |
| `full_model` | 26.8676 | 41.5629 | 0.0000 | 0.0000 | +0.0% | +0.0% |
| `zero_visual_tokens` | 33.5479 | 52.5510 | 6.6804 | 10.9881 | +24.9% | +26.4% |
| `static_visual_tokens` | 27.2839 | 42.3604 | 0.4164 | 0.7976 | +1.5% | +1.9% |
| `no_temporal_adapter` | 28.1008 | 43.6583 | 1.2332 | 2.0954 | +4.6% | +5.0% |
| `uniform_selector` | 26.7233 | 41.4492 | -0.1443 | -0.1136 | -0.5% | -0.3% |
| `no_cue_temporal` | 28.2575 | 44.0660 | 1.3900 | 2.5031 | +5.2% | +6.0% |
| `no_memory_update` | 83.1597 | 113.9139 | 56.2921 | 72.3511 | +209.5% | +174.1% |
| `no_ego_memory` | 31.0426 | 48.7539 | 4.1750 | 7.1910 | +15.5% | +17.3% |

Largest ADE degradation: `no_memory_update` (56.2921, +209.5%).

Full-model per-step error:

```text
[10.4377, 19.7209, 27.7245, 34.8919, 41.5629]
```

## 3s Horizon

- Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s`
- Checkpoint: `checkpoints/wit_vz_v4_defaults_dinov3_single_03s.pt`
- Future steps: `15`
- Evaluated samples per case: `11884`

DINO on/off summary:

| Comparison | Case | ADE | FDE | Delta ADE | Delta FDE | ADE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `DINO on` | `full_model` | 62.1001 | 103.3531 | 0.0000 | 0.0000 | +0.0% |
| `DINO off, ego only` | `constant_velocity` | 75.7201 | 131.6904 | 13.6200 | 28.3373 | +21.9% |
| `DINO signal off` | `zero_visual_tokens` | 72.7168 | 120.4932 | 10.6167 | 17.1401 | +17.1% |

| Case | ADE | FDE | Delta ADE vs full | Delta FDE vs full | ADE rel. | FDE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `constant_velocity` | 75.7201 | 131.6904 | 13.6200 | 28.3373 | +21.9% | +27.4% |
| `full_model` | 62.1001 | 103.3531 | 0.0000 | 0.0000 | +0.0% | +0.0% |
| `zero_visual_tokens` | 72.7168 | 120.4932 | 10.6167 | 17.1401 | +17.1% | +16.6% |
| `static_visual_tokens` | 63.6982 | 106.0563 | 1.5981 | 2.7032 | +2.6% | +2.6% |
| `no_temporal_adapter` | 65.9331 | 109.2122 | 3.8330 | 5.8591 | +6.2% | +5.7% |
| `uniform_selector` | 64.1482 | 107.4330 | 2.0481 | 4.0798 | +3.3% | +3.9% |
| `no_cue_temporal` | 74.0695 | 127.1839 | 11.9694 | 23.8308 | +19.3% | +23.1% |
| `no_memory_update` | 133.2666 | 259.3934 | 71.1665 | 156.0403 | +114.6% | +151.0% |
| `no_ego_memory` | 77.1363 | 139.3291 | 15.0362 | 35.9760 | +24.2% | +34.8% |

Largest ADE degradation: `no_memory_update` (71.1665, +114.6%).

Full-model per-step error:

```text
[15.9004, 23.1418, 31.1501, 38.425, 45.2682, 51.6146, 57.662, 63.482, 69.3304, 75.1149, 80.8034, 86.4321, 92.0808, 97.7429, 103.3531]
```

## 5s Horizon

- Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_05s`
- Checkpoint: `checkpoints/wit_vz_v4_defaults_dinov3_single_05s.pt`
- Future steps: `25`
- Evaluated samples per case: `10294`

DINO on/off summary:

| Comparison | Case | ADE | FDE | Delta ADE | Delta FDE | ADE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `DINO on` | `full_model` | 88.6020 | 157.0852 | 0.0000 | 0.0000 | +0.0% |
| `DINO off, ego only` | `constant_velocity` | 111.2669 | 202.7233 | 22.6648 | 45.6381 | +25.6% |
| `DINO signal off` | `zero_visual_tokens` | 122.2235 | 199.7509 | 33.6214 | 42.6657 | +37.9% |

| Case | ADE | FDE | Delta ADE vs full | Delta FDE vs full | ADE rel. | FDE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `constant_velocity` | 111.2669 | 202.7233 | 22.6648 | 45.6381 | +25.6% | +29.1% |
| `full_model` | 88.6020 | 157.0852 | 0.0000 | 0.0000 | +0.0% | +0.0% |
| `zero_visual_tokens` | 122.2235 | 199.7509 | 33.6214 | 42.6657 | +37.9% | +27.2% |
| `static_visual_tokens` | 92.7088 | 163.5263 | 4.1067 | 6.4411 | +4.6% | +4.1% |
| `no_temporal_adapter` | 100.7022 | 178.6896 | 12.1001 | 21.6044 | +13.7% | +13.8% |
| `uniform_selector` | 92.1746 | 163.4462 | 3.5726 | 6.3610 | +4.0% | +4.0% |
| `no_cue_temporal` | 131.5405 | 244.5854 | 42.9385 | 87.5002 | +48.5% | +55.7% |
| `no_memory_update` | 221.4888 | 247.5626 | 132.8868 | 90.4774 | +150.0% | +57.6% |
| `no_ego_memory` | 113.6956 | 215.5797 | 25.0935 | 58.4945 | +28.3% | +37.2% |

Largest ADE degradation: `no_memory_update` (132.8868, +150.0%).

Full-model per-step error:

```text
[15.5379, 22.1366, 29.6666, 36.7942, 43.4334, 49.5769, 55.6054, 61.4864, 67.1163, 72.6741, 78.2446, 83.8356, 89.3529, 94.8597, 100.4037, 105.9116, 111.5543, 117.2592, 122.8808, 128.4462, 134.2139, 139.8449, 145.7222, 151.4087, 157.0852]
```

## 10s Horizon

- Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_10s`
- Checkpoint: `checkpoints/wit_vz_v4_defaults_dinov3_single_10s.pt`
- Future steps: `50`
- Evaluated samples per case: `7434`

DINO on/off summary:

| Comparison | Case | ADE | FDE | Delta ADE | Delta FDE | ADE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `DINO on` | `full_model` | 154.5734 | 258.7196 | 0.0000 | 0.0000 | +0.0% |
| `DINO off, ego only` | `constant_velocity` | 217.1669 | 408.6508 | 62.5934 | 149.9312 | +40.5% |
| `DINO signal off` | `zero_visual_tokens` | 167.8308 | 265.4715 | 13.2573 | 6.7519 | +8.6% |

| Case | ADE | FDE | Delta ADE vs full | Delta FDE vs full | ADE rel. | FDE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `constant_velocity` | 217.1669 | 408.6508 | 62.5934 | 149.9312 | +40.5% | +58.0% |
| `full_model` | 154.5734 | 258.7196 | 0.0000 | 0.0000 | +0.0% | +0.0% |
| `zero_visual_tokens` | 167.8308 | 265.4715 | 13.2573 | 6.7519 | +8.6% | +2.6% |
| `static_visual_tokens` | 157.1395 | 261.0352 | 2.5660 | 2.3156 | +1.7% | +0.9% |
| `no_temporal_adapter` | 161.0679 | 263.9448 | 6.4945 | 5.2252 | +4.2% | +2.0% |
| `uniform_selector` | 156.9201 | 263.3926 | 2.3467 | 4.6730 | +1.5% | +1.8% |
| `no_cue_temporal` | 180.1199 | 310.6571 | 25.5465 | 51.9375 | +16.5% | +20.1% |
| `no_memory_update` | 313.7379 | 560.3830 | 159.1645 | 301.6634 | +103.0% | +116.6% |
| `no_ego_memory` | 221.9740 | 412.5122 | 67.4006 | 153.7926 | +43.6% | +59.4% |

Largest ADE degradation: `no_memory_update` (159.1645, +103.0%).

Full-model per-step error:

```text
[24.4804, 27.6541, 33.0559, 39.7708, 46.7109, 53.7137, 60.1843, 66.3117, 72.1931, 77.7037, 83.0474, 88.2503, 93.6338, 99.0333, 104.3054, 109.7688, 115.2495, 120.5841, 126.3086, 131.597, 136.7529, 142.1785, 147.6799, 153.3994, 159.0627, 164.6735, 170.0871, 175.3017, 180.4888, 185.4588, 190.0124, 194.5227, 198.758, 202.9556, 207.1485, 211.0908, 215.085, 219.0511, 222.8189, 226.6991, 230.2771, 233.9094, 237.3595, 240.8555, 244.1153, 247.3409, 250.2558, 253.099, 255.9566, 258.7196]
```

## Reading The Results

- `constant_velocity` and `zero_visual_tokens` are the two DINO-off views:
  the first removes visual modeling entirely, while the second keeps the
  trained network but removes its DINO signal at inference.
- `static_visual_tokens` isolates whether frame-to-frame visual change matters.
  In these results its impact is small, so most of the visual gain appears to
  come from scene content rather than short-term visual motion.
- `no_temporal_adapter` and `no_cue_temporal` isolate two different temporal
  stages: dense visual-token fusion before cue selection, and temporal fusion
  after cue selection.
- `uniform_selector` is mixed here, so adaptive TokenLearner selection should
  be judged with a retraining-time ablation before making a strong claim.
- `no_memory_update` is consistently the largest degradation, showing that
  the memory update is central to these checkpoints at inference.
- If an ablation improves over `full_model`, that component may be adding
  noise for that horizon and should be checked with retraining-time ablation
  before drawing architectural conclusions.
