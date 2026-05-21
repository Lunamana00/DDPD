# V4 Default ViZDoom DINOv3 Cache And Training

Date: 2026-05-21

## Dataset And Cache

- Processed dataset: `data/wit_vz/processed/wit_vz_v4_defaults_001`
- Strict split variants:
  - Source-disjoint: `data/wit_vz/processed/wit_vz_v4_defaults_source_disjoint_001`
  - Map-disjoint: `data/wit_vz/processed/wit_vz_v4_defaults_map_disjoint_001`
- Samples: `93,403`
- Main split sizes: train `64,620`, val `13,410`, test `15,373`
- Source-disjoint split sizes: train `56,567`, val `12,387`, test `24,449`
- Map-disjoint split sizes: train `59,243`, val `20,833`, test `13,327`

DINOv3 feature cache was regenerated on 3 RTX 3090 GPUs with 3 shards:

- Cache path: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`
- Backbone: `dinov3-convnext-tiny`
- Image size: `256`
- Payload: tensor-only `.pt`
- Token shape: `[5, 64, 768]`
- Cached files: `93,403`
- Cache size: about `44G`
- Combined manifest: `feature_manifest.json`

Validation:

- `WITVZPathDataset(..., load_rgb=False, visual_feature_cache_dir=...)` loaded the cache successfully.
- First train sample token shape: `(5, 64, 768)`.
- Test suite after cache code changes: `22 passed, 1 skipped`.

## Batch Size Notes

The final paper-facing run uses a single deterministic trajectory output. A
`batch_size=1024` probe was stable but validation flattened early. The final
single-output run used `batch_size=512`, which kept about `3.5-4.1G` VRAM per
GPU in use and gave slightly better validation performance.

Interrupted probe runs:

- `runs/wit_vz_v4_defaults_dinov3_timesformer_balanced_dp_single_bs1024`
  - `batch_size=1024`
  - Interrupted after epoch 8.
  - Best observed val ADE: `27.3858`

## Final Training Run

- Output: `runs/wit_vz_v4_defaults_dinov3_timesformer_balanced_dp_single_bs512`
- Checkpoint: `runs/wit_vz_v4_defaults_dinov3_timesformer_balanced_dp_single_bs512/best.pt`
- Model: `cue_memory_path_predictor`
- Visual backbone: `cached_dinov3_convnext_tiny`
- Temporal block: `timesformer`
- Selector: `tokenlearner`
- Memory: `attention`
- Output head: single deterministic future trajectory
- Batch size: `512`
- Data parallel: enabled on 3 GPUs
- Mixed precision: enabled
- Balance: `source_policy`, mode `both`, exponent `0.5`
- LR: `5e-4`, reduced to `2.5e-4` at epoch 17 and `1.25e-4` at epoch 24
- Early stopping: epoch `24`
- Best epoch: `10`

Main split metrics from best checkpoint:

| Split | ADE | FDE |
| --- | ---: | ---: |
| Val | 27.1727 | 41.4551 |
| Test | 26.8676 | 41.5629 |

Main test per-horizon error:

```text
[10.4377, 19.7209, 27.7245, 34.8919, 41.5629]
```

## Strict Split Evaluation

Same best checkpoint evaluated with the v4 cache:

| Evaluation | Output | ADE | FDE |
| --- | --- | ---: | ---: |
| Source-disjoint test | `runs/wit_vz_v4_defaults_dinov3_timesformer_balanced_dp_single_bs512_eval_source_disjoint` | 25.3887 | 38.5235 |
| Map-disjoint test | `runs/wit_vz_v4_defaults_dinov3_timesformer_balanced_dp_single_bs512_eval_map_disjoint` | 22.1936 | 33.7372 |

Source-disjoint per-horizon error:

```text
[10.4671, 19.2214, 26.2642, 32.4675, 38.5235]
```

Map-disjoint per-horizon error:

```text
[9.2901, 16.9373, 22.9373, 28.0662, 33.7372]
```

## Notes

- The v4 default-scenario dataset is much broader than v2, so metrics are not
  directly comparable to the v2 three-source run.
- The strict split results being better than the main episode split suggests
  the held-out source/map sets are not necessarily harder under the current
  generated-policy distribution.
- Horizon-specific single-output results are summarized in
  `reports/horizon_sweep_v4_defaults_single_output_20260521.md`.
- Free disk after the later horizon sweep was about `5.6G` on `/home/taehyun`; avoid
  starting another large cache without freeing space or moving artifacts.
- DVC CLI is not installed in the current environment, so the cache was not
  registered with DVC in this session.
