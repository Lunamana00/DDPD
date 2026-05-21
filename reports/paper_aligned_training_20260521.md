# Paper-Aligned Cue-Memory Training - 2026-05-21

## Run

- Dataset: `data/wit_vz/processed/wit_vz_expanded_001`
- Split size: train 1255, val 291, test 349
- Model: `cue_memory_path_predictor`
- Backbone: `small_cnn`, trainable
- Device: CUDA, mixed precision
- Output: `runs/cue_memory_paper_aligned_small_cnn_expanded`

## Architecture Settings

- `selector_type=tokenlearner`
- `temporal_type=timesformer`
- `memory_type=attention`
- `use_spatial_graph=true`
- `spatial_graph_neighbors=8`
- `use_temporal_difference_conv=true`
- `use_temporal_shift=true`
- `use_constant_velocity_residual=true`
- `trajectory_scale=auto`, resolved to `91.74687957763672`

## Training Settings

- Max epochs: 100
- Batch size: 32
- Learning rate: 0.0005
- Weight decay: 0.001
- Dropout: 0.2
- Gradient clipping: 1.0
- LR scheduler: ReduceLROnPlateau, patience 8, factor 0.5
- Early stopping: patience 20, min delta 0.01

## Result

Early stopping triggered at epoch 33.

Best validation checkpoint:

- Best epoch: 13
- Validation ADE: `41.2311`
- Validation FDE: `63.3457`
- Test ADE: `40.6838`
- Test FDE: `64.3525`

Per-horizon test error:

| Step | Error |
| --- | ---: |
| 1 | 16.6182 |
| 2 | 29.0268 |
| 3 | 40.8077 |
| 4 | 52.6140 |
| 5 | 64.3525 |

## Interpretation

The paper-aligned path trains successfully and improves over the old regularized small-CNN run on test ADE/FDE in the existing local runs. The training curve shows clear overfitting after the best validation epoch: train ADE continues from `38.7449` at epoch 13 down to `24.0900` at epoch 33, while validation ADE worsens from `41.2311` to `48.3511`.

This suggests the new modules are functional, but their capacity is high for the current expanded dataset. The next robust training step should either use stronger augmentation/regularization or precompute frozen DINOv3 features before training the paper-aligned head, instead of repeatedly running the DINOv3 backbone end-to-end.
