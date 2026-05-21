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
- Limit: `256`
- Batch size: `256`

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
| 1s | +21.9% | +19.7% | +4.9% | +2.9% | +184.4% | +13.2% |

- DINO/visual information helps at every horizon. Removing visual modeling entirely (`constant_velocity`) worsens ADE by 21.9% to 21.9%; zeroing the DINO tokens inside the trained model worsens ADE by 19.7% to 19.7%.
- The cue memory update is the largest inference-time dependency. `no_memory_update` is the worst ablation at every horizon, worsening ADE by 184.4% to 184.4%.
- Ego-motion conditioning inside the memory matters, especially as the prediction horizon grows: `no_ego_memory` worsens ADE by 13.2% to 13.2%.
- Cue temporal modeling has a meaningful effect beyond 1s. `no_cue_temporal` worsens ADE by 2.9% to 2.9%.
- The TimeSFormer-style temporal adapter has a smaller but consistent positive effect in this inference test, with ADE degradation from 4.9% to 4.9%.
- Repeating the last visual token grid over time has only a small effect (0.0% to 0.0% ADE), suggesting that visual content dominates over short-term visual change for these checkpoints.
- `uniform_selector` is weakly mixed (-0.2% to -0.2% ADE). Because this is inference-time surgery rather than retraining, it should not be used alone to claim the learned selector is unnecessary.

## 1s Horizon

- Dataset: `data/wit_vz/processed/wit_vz_v4_defaults_001`
- Checkpoint: `checkpoints/wit_vz_v4_defaults_dinov3_single_01s.pt`
- Future steps: `5`
- Evaluated samples per case: `256`

DINO on/off summary:

| Comparison | Case | ADE | FDE | Delta ADE | Delta FDE | ADE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `DINO on` | `full_model` | 30.3352 | 46.0432 | 0.0000 | 0.0000 | +0.0% |
| `DINO off, ego only` | `constant_velocity` | 36.9725 | 56.5070 | 6.6372 | 10.4637 | +21.9% |
| `DINO signal off` | `zero_visual_tokens` | 36.3057 | 56.4940 | 5.9705 | 10.4508 | +19.7% |

| Case | ADE | FDE | Delta ADE vs full | Delta FDE vs full | ADE rel. | FDE rel. |
| --- | --- | --- | --- | --- | --- | --- |
| `constant_velocity` | 36.9725 | 56.5070 | 6.6372 | 10.4637 | +21.9% | +22.7% |
| `full_model` | 30.3352 | 46.0432 | 0.0000 | 0.0000 | +0.0% | +0.0% |
| `zero_visual_tokens` | 36.3057 | 56.4940 | 5.9705 | 10.4508 | +19.7% | +22.7% |
| `static_visual_tokens` | 30.3437 | 45.7748 | 0.0085 | -0.2684 | +0.0% | -0.6% |
| `no_temporal_adapter` | 31.8297 | 48.9705 | 1.4944 | 2.9273 | +4.9% | +6.4% |
| `uniform_selector` | 30.2611 | 45.6297 | -0.0742 | -0.4136 | -0.2% | -0.9% |
| `no_cue_temporal` | 31.2107 | 47.2084 | 0.8754 | 1.1652 | +2.9% | +2.5% |
| `no_memory_update` | 86.2590 | 132.1525 | 55.9238 | 86.1092 | +184.4% | +187.0% |
| `no_ego_memory` | 34.3512 | 52.1324 | 4.0160 | 6.0891 | +13.2% | +13.2% |

Largest ADE degradation: `no_memory_update` (55.9238, +184.4%).

Full-model per-step error:

```text
[11.692, 22.5426, 31.7754, 39.623, 46.0432]
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
