# External Generalization Demo Selection

- Dataset: `data/wit_vz/processed/deepmind_lab_demo_001_03s`
- Predictions: `reports/demo/external_deepmind_lab_zero_shot_03s/eval_all/predictions.jsonl`
- Selected examples: `12`

| Group | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |
|---|---|---|---:|---:|---:|---:|
| lt_chasm | easy | deepmind_lab_demo_001__episode_000004_t000034 | 117.524 | 201.211 | 0.000 | 117.524 |
| lt_chasm | hard | deepmind_lab_demo_001__episode_000004_t000016 | 238.037 | 416.413 | 422.918 | -184.881 |
| lt_chasm | failure | deepmind_lab_demo_001__episode_000004_t000031 | 133.508 | 221.960 | 0.000 | 133.508 |
| nav_maze_random_goal_01 | easy | deepmind_lab_demo_001__episode_000002_t000023 | 26.774 | 32.150 | 38.800 | -12.026 |
| nav_maze_random_goal_01 | hard | deepmind_lab_demo_001__episode_000002_t000008 | 244.224 | 329.456 | 354.570 | -110.346 |
| nav_maze_random_goal_01 | failure | deepmind_lab_demo_001__episode_000002_t000004 | 149.193 | 158.077 | 58.732 | 90.461 |
| nav_maze_static_01 | easy | deepmind_lab_demo_001__episode_000001_t000029 | 84.576 | 65.329 | 53.143 | 31.433 |
| nav_maze_static_01 | hard | deepmind_lab_demo_001__episode_000001_t000020 | 299.635 | 556.408 | 518.997 | -219.362 |
| nav_maze_static_01 | failure | deepmind_lab_demo_001__episode_000001_t000034 | 114.946 | 148.628 | 64.324 | 50.622 |
| seekavoid_arena_01 | easy | deepmind_lab_demo_001__episode_000003_t000034 | 95.358 | 182.894 | 76.377 | 18.981 |
| seekavoid_arena_01 | hard | deepmind_lab_demo_001__episode_000003_t000025 | 407.222 | 774.484 | 565.542 | -158.320 |
| seekavoid_arena_01 | failure | deepmind_lab_demo_001__episode_000003_t000013 | 466.826 | 656.558 | 336.788 | 130.038 |
