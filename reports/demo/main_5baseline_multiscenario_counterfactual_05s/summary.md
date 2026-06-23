# Main 5-Baseline Multi-Scenario Counterfactual Video (5s)

- This is the corrected video where each column is a separate ViZDoom branch.
- Columns: CV, PointNav oracle, A* oracle, GT, Ours.
- Human block GT: human-action replay-derived trajectory.
- V4 block GT: recorded WIT-VZ ViZDoom policy trajectory.
- PointNav and A* are privileged endpoint upper-bound baselines.

- Success count: 12
- Skipped count: 0

| order | block | label | case | sample_id |
|---:|---|---|---|---|
| 1 | Human-action replay GT | defend_the_center | easy | `wit_vz_sauerkrautlm_human_replay_001__episode_000002_t000061` |
| 2 | Human-action replay GT | defend_the_center | CV-hard / ours better | `wit_vz_sauerkrautlm_human_replay_001__episode_000002_t000009` |
| 3 | Human-action replay GT | defend_the_center | ours worse than CV | `wit_vz_sauerkrautlm_human_replay_001__episode_000002_t000071` |
| 4 | V4 multi-scenario GT | my_way_home | CV-hard / Ours better | `wit_vz_v4_default_my_way_home_001__episode_000005_t000154` |
| 5 | V4 multi-scenario GT | rocket_basic | mixed | `wit_vz_v4_default_rocket_basic_001__episode_000021_t000196` |
| 6 | V4 multi-scenario GT | basic | mixed | `wit_vz_v4_default_basic_001__episode_000039_t000158` |
| 7 | V4 multi-scenario GT | health_gathering | CV-hard / Ours better | `wit_vz_v4_default_health_gathering_001__episode_000038_t000008` |
| 8 | V4 multi-scenario GT | health_gathering_supreme | Ours failure | `wit_vz_v4_default_health_gathering_supreme_001__episode_000040_t000033` |
| 9 | V4 multi-scenario GT | deadly_corridor | easy | `wit_vz_v4_default_deadly_corridor_001__episode_000014_t000013` |
| 10 | V4 multi-scenario GT | deathmatch | CV-hard / Ours better | `wit_vz_v4_default_deathmatch_001__episode_000025_t000026` |
| 11 | V4 multi-scenario GT | multi_deathmatch | Ours failure | `wit_vz_v4_default_multi_deathmatch_001__episode_000024_t000098` |
| 12 | V4 multi-scenario GT | predict_position | mixed | `wit_vz_v4_default_predict_position_001__episode_000017_t000091` |
