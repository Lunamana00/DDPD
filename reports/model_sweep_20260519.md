# Model Sweep Results - 2026-05-19

## Setup

- Dataset: `data/wit_vz/processed/wit_vz_mini_001`
- Seed: `7`
- Training budget: `1000` epochs with early stopping
- Optimizer settings: AdamW, `lr=1e-3`, `weight_decay=1e-3`, gradient clipping `1.0`
- Scheduler: ReduceLROnPlateau, patience `25`, factor `0.5`, min LR `1e-6`
- Loss: Huber with automatic trajectory coordinate scaling
- GPU: CUDA via local `.venv`

The sweep added new runs under `runs/model_sweep_20260519/`. The two full cue-memory reference runs were reused from the latest comparable regularized experiments:

- `runs/cue_memory_regularized_1000ep_gpu`
- `runs/cue_memory_dinov3_convnext_tiny_frozen`

## Results

Sorted by test ADE.

| Model | Best Epoch | Epochs Run | Val ADE | Val FDE | Test ADE | Test FDE |
|---|---:|---:|---:|---:|---:|---:|
| cue_memory_small_cnn_transformer | 19 | 119 | 43.76 | 67.09 | 44.95 | 74.26 |
| cue_memory_dinov3_frozen | 64 | 164 | 30.32 | 45.50 | 46.31 | 85.86 |
| cue_memory_small_cnn_gru | 36 | 136 | 35.50 | 47.99 | 52.57 | 91.79 |
| video_history_dinov3_frozen | 43 | 143 | 33.96 | 49.60 | 53.86 | 82.72 |
| ego_motion_only_gru64 | 90 | 190 | 46.38 | 68.73 | 53.90 | 92.00 |
| constant_velocity | - | 0 | 61.44 | 87.35 | 56.24 | 96.11 |
| last_frame_dinov3_frozen | 9 | 109 | 49.97 | 77.68 | 63.70 | 101.81 |
| last_frame_small_cnn | 8 | 108 | 58.26 | 79.97 | 73.03 | 124.69 |
| video_history_small_cnn | 26 | 126 | 56.95 | 77.49 | 73.33 | 128.09 |

## Readout

Test ADE 기준으로는 `cue_memory_small_cnn_transformer`가 가장 좋았다. DINOv3 ConvNeXt frozen full cue-memory는 validation ADE가 가장 낮았지만 test ADE/FDE는 small_cnn transformer보다 나빴다. 현재 mini split에서는 DINOv3 frozen이 일반성을 자동으로 보장한다기보다 split sensitivity와 overfitting risk가 크다.

단순 visual baseline은 약했다. `last_frame_*`는 small_cnn과 DINOv3 모두 좋지 않았고, frame 하나만으로 미래 path를 직접 회귀하는 구조는 이 데이터에서 부족하다.

DINOv3는 temporal aggregation을 붙이면 좋아졌다. `video_history_dinov3_frozen`은 validation ADE 33.96까지 내려갔지만 test ADE 53.86으로, full cue-memory나 small_cnn transformer를 넘지는 못했다.

`cue_memory_small_cnn_gru`는 단순 baseline보다 강하지만 transformer temporal mode보다 test 성능이 낮았다. 지금 구조에서는 spatial-token temporal modeling + transformer decoder 조합이 가장 견고하다.

## Next Experiments

1. 같은 모델을 seed 3개 이상으로 반복해 validation/test 괴리를 확인한다.
2. DINOv3 계열은 더 작은 head LR, stronger dropout, feature-cache 기반 학습으로 안정성을 본다.
3. 단순 visual baseline에는 constant-velocity residual을 붙인 버전을 추가한다.
4. mini split이 작으므로 larger processed dataset에서 같은 sweep을 다시 돈다.
