# External Generalization Demo Selection

- Dataset: `data/wit_vz/processed/miniworld_demo_001_03s`
- Predictions: `reports/demo/external_miniworld_zero_shot_03s/eval_all/predictions.jsonl`
- Selected examples: `12`

| Group | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |
|---|---|---|---:|---:|---:|---:|
| MiniWorld-Hallway-v0 | easy | miniworld_demo_001__episode_000002_t000033 | 4.050 | 6.952 | 0.000 | 4.050 |
| MiniWorld-Hallway-v0 | hard | miniworld_demo_001__episode_000001_t000005 | 3.677 | 7.425 | 0.754 | 2.923 |
| MiniWorld-Hallway-v0 | failure | miniworld_demo_001__episode_000002_t000004 | 124.124 | 201.517 | 0.453 | 123.671 |
| MiniWorld-Maze-v0 | easy | miniworld_demo_001__episode_000004_t000013 | 4.044 | 1.963 | 0.000 | 4.044 |
| MiniWorld-Maze-v0 | hard | miniworld_demo_001__episode_000004_t000013 | 4.044 | 1.963 | 0.000 | 4.044 |
| MiniWorld-Maze-v0 | failure | miniworld_demo_001__episode_000004_t000058 | 146.233 | 247.832 | 0.000 | 146.233 |
| MiniWorld-ThreeRooms-v0 | easy | miniworld_demo_001__episode_000007_t000056 | 2.788 | 6.222 | 0.000 | 2.788 |
| MiniWorld-ThreeRooms-v0 | hard | miniworld_demo_001__episode_000008_t000042 | 1.549 | 2.521 | 0.284 | 1.264 |
| MiniWorld-ThreeRooms-v0 | failure | miniworld_demo_001__episode_000008_t000025 | 58.623 | 100.560 | 0.138 | 58.485 |
| MiniWorld-WallGap-v0 | easy | miniworld_demo_001__episode_000006_t000060 | 10.387 | 19.640 | 0.000 | 10.387 |
| MiniWorld-WallGap-v0 | hard | miniworld_demo_001__episode_000006_t000047 | 3.001 | 2.087 | 0.578 | 2.423 |
| MiniWorld-WallGap-v0 | failure | miniworld_demo_001__episode_000005_t000063 | 135.231 | 240.340 | 0.406 | 134.825 |
