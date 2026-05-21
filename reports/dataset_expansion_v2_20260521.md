# Dataset Expansion V2 - 2026-05-21

## Summary

Expanded the WIT-VZ data beyond the previous single `deadly_corridor` run and added source-aware dataset support.

| Item | Value |
|---|---:|
| Processed dataset | `data/wit_vz/processed/wit_vz_v2_multi_source_001` |
| Total processed samples | 37,070 |
| Previous expanded samples | 1,895 |
| Increase | ~19.6x |
| Split strategy | episode-disjoint |
| History / future | 1.0 sec / 1.0 sec |
| Sample FPS | 5 |
| History frames / future steps | 5 / 5 |

## Raw Sources

| Source | Scenario | Episodes | Processed Samples | Raw Size |
|---|---|---:|---:|---:|
| `wit_vz_v2_deadly_corridor_001` | `deadly_corridor` | 60 | 1,138 | ~30.40 MB |
| `wit_vz_v2_health_gathering_001` | `health_gathering` | 60 | 6,120 | ~61.21 MB |
| `wit_vz_v2_my_way_home_001` | `my_way_home` | 60 | 29,812 | ~122.69 MB |

## Splits

| Split | Samples |
|---|---:|
| train | 25,420 |
| val | 5,629 |
| test | 6,021 |

Each split contains samples from all three ViZDoom sources while keeping episodes disjoint. Every sample now carries a `source` field and source-aware metadata:

- `source.source_id`
- `source.env_name`
- `source.source_dataset`
- `source.raw_run_id`
- `metadata.scenario`
- `metadata.map_id`

## Code Changes

- `src/wit_vz/build_samples.py`
  - accepts multiple `--raw` directories
  - writes `raw_dirs` and `sources` into `dataset_manifest.json`
  - prefixes sample IDs with source IDs to avoid collisions
  - supports `episode`, `map`, and `source` split strategies
- `src/wit_vz/dataset.py`
  - resolves source-prefixed frame paths like `source_id::episodes/...`
  - exposes `source` metadata in loaded items and batches
- `scripts/cache_visual_features.py`
  - resolves source-aware raw paths for future DINOv3 feature cache generation
- `src/visualize_path_predictions.py`
  - supports source-aware frame paths
- `src/external_datasets/prepare.py`
  - documents normalized schema placeholders for AI2-THOR, Habitat, ProcTHOR, and DeepMind Lab

## DVC

Created DVC pointer files for the new raw and processed data:

- `data/wit_vz/raw/wit_vz_v2_deadly_corridor_001.dvc`
- `data/wit_vz/raw/wit_vz_v2_health_gathering_001.dvc`
- `data/wit_vz/raw/wit_vz_v2_my_way_home_001.dvc`
- `data/wit_vz/processed/wit_vz_v2_multi_source_001.dvc`

The data is in the local DVC cache. Because the Codex sandbox cannot write directly to `G:\내 드라이브`, the Drive Desktop sync folder still needs a manual cache copy after commit.

## Validation

- Loaded train/val/test with `WITVZPathDataset`
- Verified tensor shapes:
  - `rgb_history`: `[5, 3, 64, 64]`
  - `ego_history`: `[5, 3]`
  - `future_path`: `[5, 2]`
- Ran full test suite: `15 passed, 1 skipped`

## Next Step

Generate DINOv3 ConvNeXt-Tiny feature cache for `wit_vz_v2_multi_source_001` before training the current paper-aligned model on this larger dataset. Full cache size is expected to be much larger than the previous 0.93 GB cache because the dataset is ~19.6x larger.
