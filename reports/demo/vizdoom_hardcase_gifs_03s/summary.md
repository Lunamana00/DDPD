# ViZDoom Multi-Scenario Demo Selection

- Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s`
- Predictions: `runs/episodic_memory_ablation_v4/seed_7/03s/long_attention_no_ego/predictions.jsonl`
- Selected examples: `6`

| Scenario | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |
|---|---|---|---:|---:|---:|---:|
| basic | hard | wit_vz_v4_default_basic_001__episode_000038_t000056 | 76.314 | 139.918 | 188.421 | -112.108 |
| my_way_home | hard | wit_vz_v4_default_my_way_home_001__episode_000025_t000190 | 22.502 | 25.884 | 178.481 | -155.978 |
| health_gathering | hard | wit_vz_v4_default_health_gathering_001__episode_000037_t000021 | 44.341 | 58.253 | 221.632 | -177.291 |
| defend_the_line | hard | wit_vz_v4_default_defend_the_line_001__episode_000032_t000017 | 75.414 | 134.066 | 237.511 | -162.098 |
| predict_position | hard | wit_vz_v4_default_predict_position_001__episode_000029_t000133 | 44.070 | 8.517 | 214.766 | -170.696 |
| deathmatch | hard | wit_vz_v4_default_deathmatch_001__episode_000018_t000034 | 47.064 | 47.601 | 229.704 | -182.640 |