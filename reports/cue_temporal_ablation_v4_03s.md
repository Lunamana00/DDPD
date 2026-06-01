# Cue Temporal Transformer Ablation v4 3s

Retraining-time ablation over the 3s v4 DINOv3 TimeSFormer path predictor.

Note: `cue_temporal_on` reuses the completed config-equivalent `spatial_relation_ablation_v4_03s/topk_graph` control run. `cue_temporal_off` was retrained on `gpu3090` for this ablation.

| Variant | Cue temporal layers | Available | Best epoch | Test ADE | Test FDE | Best Val ADE | Train-Val ADE Gap | Avg epoch sec | Peak CUDA MB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cue_temporal_on | 1 | yes | 3 | 61.8506 | 103.5420 | 57.8149 | -2.3777 | 81.4622 | 4433.4219 |
| cue_temporal_off | 0 | yes | 3 | 61.7153 | 102.9285 | 57.1840 | -3.0167 | 615.4696 | 4430.3809 |

## Interpretation

- Disabling cue temporal changes test ADE by `-0.1353` (-0.22% vs enabled).
- Disabling cue temporal changes test FDE by `-0.6135` (-0.59% vs enabled).
- Interpret this as a 3s-only retraining ablation; rerun other horizons before making horizon-general claims.
