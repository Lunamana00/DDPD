# ViZDoom Default Scenarios Dataset - 2026-05-21

## Summary

Created a new WIT-VZ v4 dataset from every default ViZDoom scenario that runs
reliably in this server environment.

- Raw dataset prefix: `data/wit_vz/raw/wit_vz_v4_default_*_001`
- Processed episode split: `data/wit_vz/processed/wit_vz_v4_defaults_001`
- Processed source-disjoint split:
  `data/wit_vz/processed/wit_vz_v4_defaults_source_disjoint_001`
- Processed map-disjoint split:
  `data/wit_vz/processed/wit_vz_v4_defaults_map_disjoint_001`
- Collection scale: `15` scenarios x `40` episodes = `600` raw episodes
- Processed supervised samples: `93,403`

Three installed WADs were not collected because ViZDoom segfaulted during
initialization in this host environment:

- `cig`
- `cig_with_unknown`
- `multi_duel`

## Scenario Counts

| Scenario | Samples | Episodes | Policy mix |
| --- | ---: | ---: | --- |
| `basic` | 10,279 | 40 | random_walk |
| `basic_audio` | 10,315 | 40 | random_walk |
| `basic_notifications` | 10,554 | 40 | random_walk |
| `deadly_corridor` | 240 | 40 | mixed |
| `deathmatch` | 6,584 | 40 | mixed |
| `defend_the_center` | 2,594 | 40 | mixed |
| `defend_the_line` | 2,587 | 40 | mixed |
| `health_gathering` | 3,858 | 40 | mixed |
| `health_gathering_supreme` | 2,682 | 40 | mixed |
| `multi_deathmatch` | 10,493 | 40 | mixed |
| `my_way_home` | 9,951 | 40 | mixed |
| `predict_position` | 2,436 | 40 | mixed |
| `rocket_basic` | 10,054 | 40 | random_walk |
| `simpler_basic` | 10,276 | 40 | random_walk |
| `take_cover` | 500 | 40 | mixed |

`deadly_corridor` and `take_cover` are underrepresented because episodes often
ended quickly with the current mixed policy. They should be topped up in the
next collection pass with safer/no-attack movement policies or longer timeout
settings if we need stronger balance without relying on sampler/loss weights.

## Splits

Episode-disjoint:

```text
train=64,620
val=13,410
test=15,373
```

Source-disjoint:

```text
train=56,567
val=12,387
test=24,449
```

Map-disjoint:

```text
train=59,243
val=20,833
test=13,327
```

For source-disjoint and map-disjoint builds, manifest leakage diagnostics show
empty train/val/test overlap for source and map groups.

## Disk

After collection/build:

- Free space on `/home/taehyun`: about `50G`
- Raw v4 default runs: roughly `1.3G`
- Each processed v4 split directory: about `194M`
- Existing v2 DINOv3 cache: `18G`

No DINOv3 cache was generated for v4 yet. A v4 cache is estimated to require
roughly `45G` because it has `93,403` samples versus `37,070` in v2. That is too
close to the current free disk budget, so free disk or use a larger cache target
before generating it.

## Verification

Completed:

```bash
uv run python -m src.wit_vz.build_samples ... --out data/wit_vz/processed/wit_vz_v4_defaults_001 --split episode
uv run python -m src.wit_vz.build_samples ... --out data/wit_vz/processed/wit_vz_v4_defaults_source_disjoint_001 --split source
uv run python -m src.wit_vz.build_samples ... --out data/wit_vz/processed/wit_vz_v4_defaults_map_disjoint_001 --split map
uv run python -m pytest
```

Final test result:

```text
22 passed, 1 skipped
```

Dataset loader smoke passed for all three v4 processed datasets.

## Next

1. Top up `deadly_corridor` and `take_cover` with movement-only policy data.
2. Free disk before generating `wit_vz_v4_defaults_001_dinov3_convnext_tiny`.
3. Train with `configs/train_wit_vz_v4_defaults_dinov3_balanced_timesformer.yaml`.
