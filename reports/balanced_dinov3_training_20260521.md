# Balanced DINOv3 Training - 2026-05-21

## Run

Completed a full balanced cached-DINOv3 training run on the host GPUs.

- Dataset: `data/wit_vz/processed/wit_vz_v2_multi_source_001`
- Feature cache: `data/wit_vz/feature_cache/wit_vz_v2_multi_source_001_dinov3_convnext_tiny`
- Output: `runs/wit_vz_v2_dinov3_timesformer_balanced_dp`
- GPUs: `3 x RTX 3090`
- Driver: `575.64.03`
- PyTorch: `2.11.0+cu128`
- Training device: `cuda`
- Parallelism: `torch.nn.DataParallel`
- Batch size: `192`
- Model: `cue_memory_path_predictor`
- Backbone: `cached_dinov3_convnext_tiny`
- Temporal block: `timesformer`
- Selector: `tokenlearner`
- Modes: `3`
- Balance: `--balance-key source --balance-mode both --balance-exponent 0.5`

The DINOv3 cache was not regenerated because the v2 cache was already complete
and validated. Reusing it avoided rewriting another 18 GB of feature files.

## Training Outcome

Early stopping stopped the run at epoch `31`.

Best checkpoint:

- Epoch: `19`
- Best val ADE: `12.0712`
- Path: `runs/wit_vz_v2_dinov3_timesformer_balanced_dp/best.pt`

Final episode-disjoint metrics from the best checkpoint:

| Split | Loss | ADE | FDE |
| --- | ---: | ---: | ---: |
| val | 0.1070 | 12.0712 | 18.4385 |
| test | 0.0974 | 10.9387 | 16.6070 |

Per-horizon test error:

```text
[6.0182, 8.9167, 10.5133, 12.6381, 16.6070]
```

## Strict Split Evaluation

Evaluated the same best checkpoint against the strict v2 datasets.

| Dataset | Test group | ADE | FDE |
| --- | --- | ---: | ---: |
| `wit_vz_v2_source_disjoint_001` | `my_way_home` | 8.8873 | 13.2048 |
| `wit_vz_v2_map_disjoint_001` | `health_gathering/map01` | 14.6593 | 22.3121 |

Run directories:

- `runs/wit_vz_v2_source_disjoint_balanced_dp_eval`
- `runs/wit_vz_v2_map_disjoint_balanced_dp_eval`

## Notes

- GPU memory stayed low, around `1.7-1.8 GB` per GPU during training.
- The workload is partly limited by cached feature file I/O and DataParallel
  gather overhead, so GPU utilization is not saturated even though all 3 GPUs
  participate.
- Full DDP would likely use the 3 GPUs more efficiently, but DataParallel was
  enough to complete this run cleanly without changing the training loop shape
  too much.
