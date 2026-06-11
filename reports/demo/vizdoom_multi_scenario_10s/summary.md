# ViZDoom Multi-Scenario Demo Selection

- Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_10s`
- Predictions: `runs/episodic_memory_ablation_v4/seed_7/10s/long_attention_no_ego/predictions.jsonl`
- Selected examples: `18`

| Scenario | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |
|---|---|---|---:|---:|---:|---:|
| basic | easy | wit_vz_v4_default_basic_001__episode_000008_t000162 | 95.969 | 90.125 | 30.898 | 65.070 |
| basic | hard | wit_vz_v4_default_basic_001__episode_000006_t000049 | 189.457 | 116.759 | 529.256 | -339.800 |
| basic | failure | wit_vz_v4_default_basic_001__episode_000008_t000112 | 315.512 | 501.787 | 94.683 | 220.829 |
| simpler_basic | easy | wit_vz_v4_default_simpler_basic_001__episode_000025_t000031 | 57.284 | 84.269 | 27.998 | 29.286 |
| simpler_basic | hard | wit_vz_v4_default_simpler_basic_001__episode_000034_t000099 | 83.464 | 363.875 | 501.304 | -417.840 |
| simpler_basic | failure | wit_vz_v4_default_simpler_basic_001__episode_000019_t000054 | 253.284 | 339.298 | 72.816 | 180.468 |
| my_way_home | easy | wit_vz_v4_default_my_way_home_001__episode_000027_t000141 | 84.796 | 149.853 | 0.243 | 84.553 |
| my_way_home | hard | wit_vz_v4_default_my_way_home_001__episode_000021_t000094 | 78.170 | 31.190 | 650.261 | -572.091 |
| my_way_home | failure | wit_vz_v4_default_my_way_home_001__episode_000034_t000069 | 310.726 | 675.228 | 67.052 | 243.674 |
| health_gathering_supreme | easy | wit_vz_v4_default_health_gathering_supreme_001__episode_000004_t000048 | 597.117 | 955.040 | 373.243 | 223.874 |
| health_gathering_supreme | hard | wit_vz_v4_default_health_gathering_supreme_001__episode_000004_t000039 | 422.183 | 594.897 | 489.106 | -66.923 |
| health_gathering_supreme | failure | wit_vz_v4_default_health_gathering_supreme_001__episode_000004_t000044 | 761.469 | 1228.627 | 376.640 | 384.829 |
| deathmatch | easy | wit_vz_v4_default_deathmatch_001__episode_000030_t000132 | 66.166 | 111.943 | 20.912 | 45.253 |
| deathmatch | hard | wit_vz_v4_default_deathmatch_001__episode_000026_t000022 | 797.950 | 1109.798 | 1220.583 | -422.633 |
| deathmatch | failure | wit_vz_v4_default_deathmatch_001__episode_000030_t000037 | 446.162 | 620.926 | 110.213 | 335.949 |
| rocket_basic | easy | wit_vz_v4_default_rocket_basic_001__episode_000034_t000150 | 164.626 | 417.567 | 39.535 | 125.090 |
| rocket_basic | hard | wit_vz_v4_default_rocket_basic_001__episode_000034_t000083 | 116.227 | 71.553 | 420.848 | -304.621 |
| rocket_basic | failure | wit_vz_v4_default_rocket_basic_001__episode_000034_t000160 | 362.131 | 713.570 | 62.338 | 299.793 |