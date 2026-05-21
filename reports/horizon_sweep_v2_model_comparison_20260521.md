# WIT-VZ V2 Horizon Model Comparison - 2026-05-21

## Setup

- Dataset root: `data/wit_vz/processed/horizon_sweep_v2`
- Raw sources: `deadly_corridor`, `health_gathering`, `my_way_home`
- Horizons: 1, 3, 5, 10, 30 seconds
- Sample FPS: 5.0
- History window: 1 second
- Split: episode-disjoint per horizon
- Metric: ADE/FDE in local egocentric coordinates
- Device: RTX 4070 SUPER

All learned models use the same cue-memory trajectory head family:

- 2D spatial positional encoding
- temporal modeling over spatial tokens
- TokenLearner cue selector
- attention cue memory
- horizon query decoder
- constant-velocity residual path
- Huber loss, dropout 0.2, weight decay 0.001, gradient clipping

## Compared Models

| Name | Visual input | Temporal module | Epochs |
| --- | --- | --- | ---: |
| `constant_velocity` | none | none | 0 |
| `small_cnn_timesformer` | trainable small CNN over RGB | TimeSformer-style | 6 |
| `dinov3_timesformer` | frozen cached DINOv3 ConvNeXt-Tiny tokens | TimeSformer-style | 6 |
| `dinov3_strnet` | frozen cached DINOv3 ConvNeXt-Tiny tokens | STRNet-style graph/temporal fusion | 6 |

DINOv3 features were cached for every horizon dataset under:

```text
data/wit_vz/feature_cache/horizon_sweep_v2_future_XXs_dinov3_convnext_tiny
```

Each cached sample stores `[5, 64, 768]` frozen visual tokens.

## Test Results

Lower is better.

| Horizon | CV ADE | CV FDE | small-CNN ADE | small-CNN FDE | DINOv3 TimeSFormer ADE | DINOv3 TimeSFormer FDE | DINOv3 STRNet ADE | DINOv3 STRNet FDE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | 27.38 | 43.71 | 23.18 | 36.09 | 18.39 | 28.91 | 19.73 | 30.80 |
| 3s | 63.89 | 116.04 | 55.53 | 102.71 | 50.76 | 90.78 | 48.07 | 88.34 |
| 5s | 106.38 | 197.45 | 88.52 | 155.99 | 77.51 | 135.70 | 77.73 | 130.88 |
| 10s | 145.36 | 281.98 | 104.84 | 199.12 | 110.40 | 205.46 | 113.00 | 191.74 |
| 30s | 391.12 | 694.28 | 289.01 | 501.62 | 245.80 | 425.97 | 255.86 | 429.14 |

## Best Model By Horizon

| Horizon | Best ADE | ADE improvement vs CV | Best FDE | FDE improvement vs CV |
| ---: | --- | ---: | --- | ---: |
| 1s | DINOv3 TimeSFormer | 32.8% | DINOv3 TimeSFormer | 33.9% |
| 3s | DINOv3 STRNet | 24.8% | DINOv3 STRNet | 23.9% |
| 5s | DINOv3 TimeSFormer | 27.1% | DINOv3 STRNet | 33.7% |
| 10s | small-CNN TimeSFormer | 27.9% | DINOv3 STRNet | 32.0% |
| 30s | DINOv3 TimeSFormer | 37.2% | DINOv3 TimeSFormer | 38.6% |

## Takeaways

- Every learned model beats constant velocity at every horizon.
- DINOv3 is generally stronger than small-CNN, especially at 1, 3, 5, and 30 seconds.
- The 10-second ADE exception favors small-CNN, while 10-second FDE still favors DINOv3 STRNet. This suggests the models differ in average-path fit versus endpoint fit.
- STRNet is not uniformly better than TimeSFormer. It helps most clearly at 3-second ADE/FDE and 5/10-second FDE.
- Long-horizon deterministic ADE/FDE is harsh. At 10 and 30 seconds, route uncertainty grows, so these results should be interpreted with horizon-specific endpoint and curvature diagnostics.

## Practical Interpretation

For a single deterministic path predictor, the current strongest default is:

- Short horizon 1s: DINOv3 TimeSFormer
- Mid horizon 3s: DINOv3 STRNet
- 5s: DINOv3 TimeSFormer for ADE, DINOv3 STRNet for FDE
- 10s: small-CNN for ADE, DINOv3 STRNet for FDE
- 30s: DINOv3 TimeSFormer

The robust next step is a multi-horizon deterministic model instead of training separate heads for every horizon.
