# ViZDoom Multi-Scenario Demo Selection

- Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s`
- Predictions: `runs/episodic_memory_ablation_v4/seed_7/03s/long_attention_no_ego/predictions.jsonl`
- Selected examples: `30`

| Scenario | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |
|---|---|---|---:|---:|---:|---:|
| basic | easy | wit_vz_v4_default_basic_001__episode_000038_t000226 | 13.255 | 18.042 | 12.116 | 1.140 |
| basic | hard | wit_vz_v4_default_basic_001__episode_000038_t000056 | 76.314 | 139.918 | 188.421 | -112.108 |
| basic | failure | wit_vz_v4_default_basic_001__episode_000010_t000201 | 115.954 | 152.781 | 41.078 | 74.876 |
| my_way_home | easy | wit_vz_v4_default_my_way_home_001__episode_000037_t000136 | 5.139 | 6.575 | 1.367 | 3.772 |
| my_way_home | hard | wit_vz_v4_default_my_way_home_001__episode_000025_t000190 | 22.502 | 25.884 | 178.481 | -155.978 |
| my_way_home | failure | wit_vz_v4_default_my_way_home_001__episode_000037_t000190 | 144.940 | 267.568 | 47.646 | 97.295 |
| health_gathering | easy | wit_vz_v4_default_health_gathering_001__episode_000007_t000065 | 39.201 | 66.212 | 7.925 | 31.276 |
| health_gathering | hard | wit_vz_v4_default_health_gathering_001__episode_000037_t000021 | 44.341 | 58.253 | 221.632 | -177.291 |
| health_gathering | failure | wit_vz_v4_default_health_gathering_001__episode_000004_t000127 | 248.914 | 476.583 | 47.870 | 201.044 |
| health_gathering_supreme | easy | wit_vz_v4_default_health_gathering_supreme_001__episode_000033_t000021 | 8.100 | 10.830 | 6.850 | 1.251 |
| health_gathering_supreme | hard | wit_vz_v4_default_health_gathering_supreme_001__episode_000020_t000042 | 23.886 | 35.816 | 68.163 | -44.278 |
| health_gathering_supreme | failure | wit_vz_v4_default_health_gathering_supreme_001__episode_000020_t000016 | 164.046 | 343.418 | 54.781 | 109.265 |
| defend_the_center | easy | wit_vz_v4_default_defend_the_center_001__episode_000011_t000039 | 41.810 | 36.045 | 5.876 | 35.934 |
| defend_the_center | hard | wit_vz_v4_default_defend_the_center_001__episode_000039_t000041 | 22.220 | 75.577 | 214.568 | -192.348 |
| defend_the_center | failure | wit_vz_v4_default_defend_the_center_001__episode_000002_t000016 | 321.711 | 532.959 | 189.865 | 131.846 |
| defend_the_line | easy | wit_vz_v4_default_defend_the_line_001__episode_000004_t000043 | 15.242 | 10.713 | 17.633 | -2.392 |
| defend_the_line | hard | wit_vz_v4_default_defend_the_line_001__episode_000032_t000017 | 75.414 | 134.066 | 237.511 | -162.098 |
| defend_the_line | failure | wit_vz_v4_default_defend_the_line_001__episode_000006_t000037 | 134.648 | 238.662 | 59.304 | 75.344 |
| predict_position | easy | wit_vz_v4_default_predict_position_001__episode_000032_t000049 | 46.796 | 27.602 | 12.955 | 33.841 |
| predict_position | hard | wit_vz_v4_default_predict_position_001__episode_000029_t000133 | 44.070 | 8.517 | 214.766 | -170.696 |
| predict_position | failure | wit_vz_v4_default_predict_position_001__episode_000030_t000017 | 307.631 | 565.176 | 119.478 | 188.154 |
| deathmatch | easy | wit_vz_v4_default_deathmatch_001__episode_000026_t000083 | 8.474 | 7.351 | 1.018 | 7.456 |
| deathmatch | hard | wit_vz_v4_default_deathmatch_001__episode_000018_t000034 | 47.064 | 47.601 | 229.704 | -182.640 |
| deathmatch | failure | wit_vz_v4_default_deathmatch_001__episode_000012_t000029 | 207.007 | 515.750 | 50.161 | 156.847 |
| multi_deathmatch | easy | wit_vz_v4_default_multi_deathmatch_001__episode_000034_t000096 | 4.384 | 6.602 | 1.635 | 2.749 |
| multi_deathmatch | hard | wit_vz_v4_default_multi_deathmatch_001__episode_000021_t000057 | 30.896 | 79.159 | 192.902 | -162.006 |
| multi_deathmatch | failure | wit_vz_v4_default_multi_deathmatch_001__episode_000025_t000023 | 160.417 | 355.510 | 37.737 | 122.679 |
| rocket_basic | easy | wit_vz_v4_default_rocket_basic_001__episode_000007_t000029 | 26.240 | 48.831 | 14.231 | 12.009 |
| rocket_basic | hard | wit_vz_v4_default_rocket_basic_001__episode_000007_t000182 | 36.546 | 32.744 | 161.267 | -124.721 |
| rocket_basic | failure | wit_vz_v4_default_rocket_basic_001__episode_000011_t000109 | 163.971 | 335.727 | 18.485 | 145.486 |