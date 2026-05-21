# STRNet TokenLearner Validation - 2026-05-21

## Architecture Change

Implemented a closer paper-aligned visual representation path in `src/models/cue_memory.py`.

- Added STRNet-specific edge-message dynamic graph aggregation while keeping the existing value-attention spatial graph for older checkpoints: top-k neighbors are selected per visual token, edge features are formed as `[center, neighbor - center]`, and gated messages update the token.
- Reworked temporal difference modeling into multi-resolution depthwise temporal convolutions over temporal differences at dilations 1, 2, and 4.
- Added `temporal_type=strnet`, a fusion block that applies spatial graph reasoning inside each frame, temporal shift, and difference-aware temporal convolution.
- Updated `TokenLearnerCueTokenSelector` to use sigmoid spatial attention maps followed by normalized weighted pooling, matching the TokenLearner mechanism more directly than a plain softmax selector.
- Added `CueSpaceTimeTransformer`, which runs a transformer over the `T * K` cue tokens after per-frame TokenLearner selection.

This is still adapted to the project task: our dataset has egocentric history and future local path labels, but it does not have STRNet's goal image, progress estimation, or navigation policy targets. The implemented part is the representation mechanism, not STRNet's full goal-conditioned controller.

## Validation Setup

- Dataset: `data/wit_vz/processed/wit_vz_expanded_001`
- Samples: 1,895
- Split: train 1,255 / val 291 / test 349
- Input: frozen cached DINOv3 ConvNeXt-Tiny visual tokens
- Feature cache: `data/wit_vz/feature_cache/wit_vz_expanded_001_dinov3_convnext_tiny`
- Feature shape per sample: `[5, 64, 768]`
- Model output: future local path `[5, 2]`
- Metrics: ADE and FDE in local egocentric coordinates

Run configuration:

- `backbone=cached_dinov3_convnext_tiny`
- `temporal_type=strnet`
- `temporal_layers=2`
- `selector_type=tokenlearner`
- `cue_temporal_layers=2`
- `memory_type=attention`
- `num_cue_tokens=8`
- `use_constant_velocity_residual=true`
- `dropout=0.2`
- `weight_decay=0.001`
- `grad_clip_norm=1.0`
- `early_stopping_patience=8`

## Test Result

- Unit/integration tests: `16 passed, 1 skipped`
- Checkpoint compatibility smoke test: previous cached-DINOv3 checkpoint re-evaluates at Test ADE/FDE `36.8649 / 56.7751`
- Training completed with early stopping at epoch 24
- Best checkpoint: epoch 16

| Split | ADE | FDE |
| --- | ---: | ---: |
| Validation | 35.2393 | 53.2370 |
| Test | 33.0923 | 52.5272 |

Per-horizon test error:

| Future step | Error |
| --- | ---: |
| 1 | 14.2254 |
| 2 | 22.8930 |
| 3 | 32.9843 |
| 4 | 42.8317 |
| 5 | 52.5272 |

## Comparison To Previous Cached DINOv3 Run

| Run | Val ADE | Val FDE | Test ADE | Test FDE |
| --- | ---: | ---: | ---: | ---: |
| TimeSformer-style cached DINOv3 | 35.0428 | 51.9829 | 36.8649 | 56.7751 |
| STRNet + TokenLearner cached DINOv3 | 35.2393 | 53.2370 | 33.0923 | 52.5272 |

The new STRNet-style path is slightly worse on validation ADE/FDE but better on test ADE/FDE for this split. The train ADE kept decreasing after the best validation epoch, so the current bottleneck is still data diversity and regularization rather than just module expressiveness.
