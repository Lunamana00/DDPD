# WIT-VZ V2 Horizon Sweep Baseline - 2026-05-21

## Setup

- Raw sources:
  - `data/wit_vz/raw/wit_vz_v2_deadly_corridor_001`
  - `data/wit_vz/raw/wit_vz_v2_health_gathering_001`
  - `data/wit_vz/raw/wit_vz_v2_my_way_home_001`
- Processed root: `data/wit_vz/processed/horizon_sweep_v2`
- Run root: `runs/horizon_sweep_v2`
- History window: 1.0 second
- Sample FPS: 5.0
- Horizons: 1, 3, 5, 10, 30 seconds
- Split strategy: episode-disjoint
- Stride: 2

The sweep uses the same raw sources and same sample-generation rule for every horizon. The sample count naturally decreases at longer horizons because each sample needs a longer future segment after the current frame.

## What The Baseline Does

The baseline run here is `constant_velocity`.

It does not use RGB frames and does not train. It takes the recent ego-motion history, averages the last 5 local motion increments, repeats that average velocity for every future step, and cumulatively sums those increments into a future path.

Formula:

```text
v = mean(e_{t-4:t, x:y})
p_hat_{t+k} = sum_{i=1..k} v
```

where `x` is forward and `y` is right in the local egocentric frame.

Other baselines exist in code but were not run in this sweep:

- `ego_motion_only`: GRU over ego-motion only
- `last_frame_dino`: visual feature from only the last frame
- `video_history_dino`: frame-pooled visual history through a GRU

The earlier v2 training report compared the small-CNN cue-memory model against `constant_velocity` at 1 second. This new sweep first extends the motion-prior baseline to longer horizons; model runs should be trained on these exact horizon datasets next.

## Dataset Sizes

| Horizon | Future steps | Samples | Train | Val | Test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | 5 | 18,554 | 13,071 | 3,078 | 2,405 |
| 3s | 15 | 16,940 | 11,341 | 3,763 | 1,836 |
| 5s | 25 | 15,683 | 10,698 | 3,578 | 1,407 |
| 10s | 50 | 12,799 | 8,973 | 2,250 | 1,576 |
| 30s | 150 | 6,894 | 5,087 | 1,034 | 773 |

## Constant Velocity Results

| Horizon | CV ADE | CV FDE |
| ---: | ---: | ---: |
| 1s | 27.3805 | 43.7101 |
| 3s | 63.8882 | 116.0432 |
| 5s | 106.3786 | 197.4482 |
| 10s | 145.3607 | 281.9835 |
| 30s | 391.1181 | 694.2850 |

## Takeaway

The long-horizon baseline gets much worse as expected. This is useful: if visual models actually learn map structure, object cues, and route context, their improvement over constant velocity should become more visible at 3 seconds and beyond.

For 10 and 30 seconds, deterministic single-path ADE/FDE will become increasingly harsh because multiple future paths may be plausible. Those horizons should eventually include multi-modal metrics such as `minADE`, `minFDE`, and mode confidence.
