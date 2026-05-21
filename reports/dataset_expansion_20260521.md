# Dataset Expansion - 2026-05-21

## Summary

Created a larger WIT-VZ dataset from newly collected ViZDoom episodes.

| Item | Value |
|---|---:|
| Raw run | `data/wit_vz/raw/wit_vz_expanded_001` |
| Processed dataset | `data/wit_vz/processed/wit_vz_expanded_001` |
| Scenario | `deadly_corridor` |
| Map | `map01` |
| Policy | scripted `corridor` |
| Episodes collected | 120 |
| Processed samples | 1,895 |
| Previous mini samples | 133 |
| Increase | ~14.2x |
| Raw size | ~59 MB |
| Processed metadata size | ~2.8 MB |

The raw and processed data directories are intentionally ignored by Git. The collection and build settings are tracked in:

- `configs/wit_vz_collect_expanded.yaml`
- `configs/wit_vz_build_expanded.yaml`

## Processed Dataset

The dataset keeps the same prediction setup used by the earlier mini experiments:

- History: 1.0 sec
- Future horizon: 1.0 sec
- Sampling rate: 5 FPS
- History frames: 5
- Future steps: 5
- Step gap: 2 raw steps
- Split strategy: episode-disjoint
- Target: `future_local_path`
- Coordinate convention: local `x=forward`, local `y=right`, origin at current pose

Split counts:

| Split | Episodes | Samples |
|---|---:|---:|
| train | 80 | 1,255 |
| val | 17 | 291 |
| test | 18 | 349 |

## Validation

Checks performed:

- Loaded train/val/test splits with `WITVZPathDataset`.
- Verified tensor shapes:
  - `rgb_history`: `[5, 3, 64, 64]`
  - `ego_history`: `[5, 3]`
  - `future_path`: `[5, 2]`
- Verified episode-disjoint split:
  - train/val overlap: false
  - train/test overlap: false
  - val/test overlap: false
- Ran dataset unit test: `1 passed`
- Ran a 2-epoch cue-memory smoke training on CUDA.

Smoke training result on `wit_vz_expanded_001`:

| Epochs | Val ADE | Val FDE | Test ADE | Test FDE |
|---:|---:|---:|---:|---:|
| 2 | 32.16 | 48.20 | 31.07 | 49.77 |

This smoke result is only a pipeline sanity check, not a final model comparison.

## Next Step

Run the full model sweep again on `data/wit_vz/processed/wit_vz_expanded_001`. The larger validation/test splits should make model comparisons less sensitive to individual episode luck than the previous 133-sample mini dataset.
