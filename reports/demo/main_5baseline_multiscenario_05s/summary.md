# Main 5-Baseline Multi-Scenario Video (5s)

- Columns: CV, PointNav oracle, A* oracle, GT, Ours.
- Human block GT: human-action replay-derived trajectory.
- V4 block GT: recorded WIT-VZ ViZDoom policy trajectory.
- PointNav and A* are privileged upper-bound baselines using the GT endpoint.
- Xu-style and Khaleque-style proxies are intentionally excluded from this main video.
- V4 aggregate metrics use samples with target extent >= 20.0.

## Metrics

### human_action_replay

- Samples evaluated: 64

| method | ADE | FDE |
|---|---:|---:|
| CV | 161.91 | 280.42 |
| PointNav oracle | 73.73 | 0.00 |
| A* oracle | 69.10 | 0.00 |
| Ours | 147.87 | 259.02 |

### v4_multi_scenario

- Samples evaluated: 9448

| method | ADE | FDE |
|---|---:|---:|
| CV | 119.35 | 217.38 |
| PointNav oracle | 50.26 | 0.00 |
| A* oracle | 50.95 | 0.00 |
| Ours | 93.32 | 164.84 |

## Video Sequence

| order | block | label | case | sample_id | Ours ADE | CV ADE |
|---:|---|---|---|---|---:|---:|
| 1 | Human-action replay GT | defend_the_center | easy | `wit_vz_sauerkrautlm_human_replay_001__episode_000002_t000061` | 19.55 | 66.33 |
| 2 | Human-action replay GT | defend_the_center | CV-hard / ours better | `wit_vz_sauerkrautlm_human_replay_001__episode_000002_t000009` | 161.54 | 297.53 |
| 3 | Human-action replay GT | defend_the_center | ours worse than CV | `wit_vz_sauerkrautlm_human_replay_001__episode_000002_t000071` | 420.54 | 98.13 |
| 4 | V4 multi-scenario GT | my_way_home | CV-hard / Ours better | `wit_vz_v4_default_my_way_home_001__episode_000005_t000154` | 26.46 | 322.22 |
| 5 | V4 multi-scenario GT | rocket_basic | mixed | `wit_vz_v4_default_rocket_basic_001__episode_000021_t000196` | 180.21 | 106.22 |
| 6 | V4 multi-scenario GT | basic | mixed | `wit_vz_v4_default_basic_001__episode_000039_t000158` | 174.13 | 174.15 |
| 7 | V4 multi-scenario GT | health_gathering | CV-hard / Ours better | `wit_vz_v4_default_health_gathering_001__episode_000038_t000008` | 128.17 | 462.13 |
| 8 | V4 multi-scenario GT | health_gathering_supreme | Ours failure | `wit_vz_v4_default_health_gathering_supreme_001__episode_000040_t000033` | 507.92 | 301.56 |
| 9 | V4 multi-scenario GT | deadly_corridor | easy | `wit_vz_v4_default_deadly_corridor_001__episode_000014_t000013` | 47.12 | 56.04 |
| 10 | V4 multi-scenario GT | deathmatch | CV-hard / Ours better | `wit_vz_v4_default_deathmatch_001__episode_000025_t000026` | 90.36 | 396.77 |
| 11 | V4 multi-scenario GT | multi_deathmatch | Ours failure | `wit_vz_v4_default_multi_deathmatch_001__episode_000024_t000098` | 306.41 | 98.14 |
| 12 | V4 multi-scenario GT | predict_position | mixed | `wit_vz_v4_default_predict_position_001__episode_000017_t000091` | 158.03 | 158.03 |
