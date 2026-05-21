# Diversity and Balance Expansion - 2026-05-21

## Summary

This pass added the knobs needed to strengthen WIT-VZ beyond the current
episode-disjoint v2 setup:

- source/policy-balanced training through sampler and per-sample loss weights
- richer ViZDoom scripted policies and per-episode policy mixing
- random warmup starts and goal-directed policy variation
- source/map/policy split diagnostics in processed dataset manifests
- v2 source-disjoint and map-disjoint processed datasets for immediate strict evaluation
- v3 collection/build/train configs for broader ViZDoom scenarios
- external embodied dataset stubs for AI2-THOR, ProcTHOR, and Habitat

## Current Data Balance

`wit_vz_v2_multi_source_001` was regenerated with the same sample IDs and new
metadata/diagnostics, so the existing DINOv3 cache remains valid.

| Source | Samples | Policy |
| --- | ---: | --- |
| `wit_vz_v2_deadly_corridor_001` | 1,138 | corridor |
| `wit_vz_v2_health_gathering_001` | 6,120 | corridor |
| `wit_vz_v2_my_way_home_001` | 29,812 | corridor |

This confirms the issue: `my_way_home` dominates the current dataset. The new
training flags compensate without deleting data:

```bash
--balance-key source --balance-mode both --balance-exponent 0.5
```

For the train split, the resolved loss weights were:

| Source | Train samples | Weight |
| --- | ---: | ---: |
| `wit_vz_v2_deadly_corridor_001` | 634 | 4.2979 |
| `wit_vz_v2_health_gathering_001` | 4,552 | 1.6040 |
| `wit_vz_v2_my_way_home_001` | 20,234 | 0.7608 |

## Strict Splits

Generated local processed datasets:

- `data/wit_vz/processed/wit_vz_v2_source_disjoint_001`
- `data/wit_vz/processed/wit_vz_v2_map_disjoint_001`

Both manifests now include `split_diagnostics.leakage`. For the generated v2
strict splits, source/map train-val-test overlap is empty.

| Evaluation | Test group | CV ADE/FDE | Balanced DINOv3 smoke ADE/FDE |
| --- | --- | ---: | ---: |
| source-disjoint | `my_way_home` | 19.5806 / 30.8031 | 13.5975 / 19.8896 |
| map-disjoint | `health_gathering/map01` | 48.9435 / 78.5330 | 24.6862 / 37.8282 |

The DINOv3 smoke checkpoint was trained for 1 epoch on CPU because the GPU
driver is currently unavailable.

## New Configs

- `configs/wit_vz_collect_diverse_v3.yaml`
  - adds `defend_the_center`, `take_cover`, `predict_position`, `defend_the_line`
  - reduces `my_way_home`
  - uses `mixed` policy, random warmup, jitter, and policy noise
- `configs/wit_vz_build_v3_diverse.yaml`
- `configs/wit_vz_build_v3_source_disjoint.yaml`
- `configs/wit_vz_build_v3_map_disjoint.yaml`
- `configs/wit_vz_build_v2_source_disjoint.yaml`
- `configs/wit_vz_build_v2_map_disjoint.yaml`
- `configs/train_wit_vz_v2_dinov3_balanced_timesformer.yaml`
- `configs/train_wit_vz_v3_dinov3_balanced_timesformer.yaml`
- `configs/external_sources_v2.yaml`

## External Dataset Prep

Prepared local normalized-schema stubs:

- `data/external/ai2thor_stub`
- `data/external/procthor_stub`
- `data/external/habitat_stub`

These intentionally do not download assets. AI2-THOR/ProcTHOR/Habitat require
separate simulator installs and/or licensed scenes. Each stub documents required
sample fields, coordinate convention, recommended metadata, and strict split
keys.

## Environment Notes

- Disk: `/home/taehyun` has about `52G` free and is `98%` used.
- Data sizes:
  - raw WIT-VZ: `396M`
  - processed WIT-VZ: `393M`
  - DINOv3 feature cache: `18G`
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- Host CUDA availability outside the Codex sandbox: `True`
- GPUs: `3 x NVIDIA GeForce RTX 3090`, driver `575.64.03`
- Codex sandbox note: regular sandboxed commands do not expose `/dev/nvidia*`,
  so `nvidia-smi` and `torch.cuda.is_available()` fail inside the sandbox. Run
  GPU commands outside the sandbox when needed.
- DINOv3 CUDA smoke passed with `scripts.cache_visual_features --limit 2`.
- ViZDoom diversified collection smoke passed outside the Codex sandbox for
  `defend_the_center`, `take_cover`, and `predict_position` with `policy=mixed`.
- DVC CLI is not installed in the current uv environment, so newly generated
  strict processed directories are local only until DVC is installed and added.

## Verification

Commands completed:

```bash
uv run python -m pytest
uv run python -m src.wit_vz.build_samples ... --out data/wit_vz/processed/wit_vz_v2_multi_source_001 --split episode
uv run python -m src.wit_vz.build_samples ... --out data/wit_vz/processed/wit_vz_v2_source_disjoint_001 --split source
uv run python -m src.wit_vz.build_samples ... --out data/wit_vz/processed/wit_vz_v2_map_disjoint_001 --split map
uv run python -m src.train_path_predictor ... --balance-key source --balance-mode both --epochs 1
uv run python -m src.eval_path_predictor ... --dataset data/wit_vz/processed/wit_vz_v2_source_disjoint_001
uv run python -m src.eval_path_predictor ... --dataset data/wit_vz/processed/wit_vz_v2_map_disjoint_001
```

Final test result:

```text
22 passed, 1 skipped
```

Follow-up balanced 3-GPU training was completed after this expansion pass:

- Report: `reports/balanced_dinov3_training_20260521.md`
- Run: `runs/wit_vz_v2_dinov3_timesformer_balanced_dp`
- Episode-disjoint test ADE/FDE: `10.9387 / 16.6070`
- Source-disjoint test ADE/FDE: `8.8873 / 13.2048`
- Map-disjoint test ADE/FDE: `14.6593 / 22.3121`

## Next Execution Order

1. Restore GPU driver so `nvidia-smi` and `torch.cuda.is_available()` work.
2. Run the v3 collection plan from `configs/wit_vz_collect_diverse_v3.yaml`.
3. Build `wit_vz_v3_diverse_001`, `wit_vz_v3_source_disjoint_001`, and
   `wit_vz_v3_map_disjoint_001`.
4. Generate the v3 DINOv3 cache:

```bash
uv run python -m scripts.cache_visual_features \
  --dataset data/wit_vz/processed/wit_vz_v3_diverse_001 \
  --output-dir data/wit_vz/feature_cache/wit_vz_v3_diverse_001_dinov3_convnext_tiny \
  --backbone dinov3-convnext-tiny \
  --image-size 256 \
  --batch-size 16 \
  --device cuda \
  --dtype float16 \
  --mixed-precision
```

5. Train with `configs/train_wit_vz_v3_dinov3_balanced_timesformer.yaml`.
6. Add strict-split v3 evaluations to the paper table.
