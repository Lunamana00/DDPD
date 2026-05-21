# DINOv3 Feature Cache Training - 2026-05-21

## Feature Cache

- Dataset: `data/wit_vz/processed/wit_vz_expanded_001`
- Backbone: `dinov3-convnext-tiny`
- Cache path: `data/wit_vz/feature_cache/wit_vz_expanded_001_dinov3_convnext_tiny`
- Cached samples: 1,895
- Feature shape per sample: `[5, 64, 768]`
- Cache dtype: `float16`
- Cache size on disk: about 0.93 GB

The feature cache stores frozen DINOv3 ConvNeXt-Tiny visual tokens per processed sample. Training then uses `backbone=cached_dinov3_convnext_tiny`, so DINOv3 is not rerun every epoch.

## Training Run

- Output: `runs/cue_memory_paper_aligned_dinov3_cached`
- Model: `cue_memory_path_predictor`
- Visual input: cached DINOv3 ConvNeXt-Tiny tokens
- Trainable part: projection, adapter, spatial graph, TimeSformer-style temporal adapter, TokenLearner selector, attention memory, decoder
- Batch size: 64
- Max epochs: 100
- Early stopping: patience 20, min delta 0.01
- Mixed precision: enabled

Architecture settings:

- `selector_type=tokenlearner`
- `temporal_type=timesformer`
- `memory_type=attention`
- `use_spatial_graph=true`
- `spatial_graph_neighbors=8`
- `use_temporal_difference_conv=true`
- `use_temporal_shift=true`
- `use_constant_velocity_residual=true`

## Result

Early stopping triggered at epoch 28.

Best validation checkpoint:

- Best epoch: 8
- Validation ADE: `35.0428`
- Validation FDE: `51.9829`
- Test ADE: `36.8649`
- Test FDE: `56.7751`

Per-horizon test error:

| Step | Error |
| --- | ---: |
| 1 | 18.7551 |
| 2 | 25.8397 |
| 3 | 36.2459 |
| 4 | 46.7086 |
| 5 | 56.7751 |

## Comparison

| Run | Val ADE | Val FDE | Test ADE | Test FDE |
| --- | ---: | ---: | ---: | ---: |
| paper-aligned small-CNN | 41.2311 | 63.3457 | 40.6838 | 64.3525 |
| paper-aligned cached DINOv3 | 35.0428 | 51.9829 | 36.8649 | 56.7751 |

Cached DINOv3 improves the paper-aligned model on both validation and test metrics. Overfitting is still visible after the best epoch: train ADE continues downward after epoch 8 while validation ADE worsens, so the next step should focus on augmentation, stronger regularization, or expanding the dataset further.
