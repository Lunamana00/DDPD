# Spatial Relation Ablation v4 3s

Retraining-time ablation over the 3s v4 DINOv3 TimeSFormer path predictor.

| Variant | Available | Best epoch | Test ADE | Test FDE | Best Val ADE | Train-Val ADE Gap | Avg epoch sec | Peak CUDA MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| topk_graph | yes | 3 | 61.8506 | 103.5420 | 57.8149 | -2.3777 | 81.4622 | 4433.4219 |
| no_graph | yes | 3 | 62.1850 | 103.8032 | 57.5455 | -3.3700 | 67.2126 | 3716.2842 |
| full_attention | yes | 5 | 63.1742 | 104.9694 | 58.2653 | 3.9910 | 68.9472 | 4158.4219 |
| local_grid | yes | 5 | 61.9014 | 104.1554 | 57.2378 | 2.7250 | 69.0817 | 4158.4219 |

- Best test ADE: `topk_graph` (61.8506).
- Interpret this as a 3s-only retraining ablation; rerun other horizons before making horizon-general claims.

## Interpretation

- `topk_graph` is the best test-ADE variant, but the gain over `no_graph` is small: ADE improves by 0.3344 (about 0.54%) and FDE improves by 0.2612 (about 0.25%).
- `full_attention` is worse than `topk_graph`: ADE is higher by 1.3236 and FDE is higher by 1.4274, suggesting that dense 64-token mixing may add noise or over-smoothing in this 3s setting.
- `local_grid` is very close to `topk_graph` in ADE (+0.0508 worse), but its FDE is worse (+0.6134), so the fixed local 8-neighbor prior is competitive but not clearly better.
- Presentation wording should be conservative: the Dynamic Spatial Graph is better described as a spatial context refinement module, not as a proven level graph or pathfinding graph.
