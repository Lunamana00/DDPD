# External Generalization Demo Selection

- Dataset: `data/wit_vz/processed/ai2thor_demo_001_03s`
- Predictions: `reports/demo/external_ai2thor_zero_shot_03s/eval_all/predictions.jsonl`
- Selected examples: `6`

| Group | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |
|---|---|---|---:|---:|---:|---:|
| FloorPlan1 | easy | ai2thor_demo_001__episode_000001_t000030 | 35.002 | 55.021 | 0.230 | 34.772 |
| FloorPlan1 | hard | ai2thor_demo_001__episode_000001_t000004 | 11.669 | 20.649 | 1.022 | 10.647 |
| FloorPlan1 | failure | ai2thor_demo_001__episode_000001_t000017 | 144.570 | 235.055 | 1.445 | 143.125 |
| FloorPlan201 | easy | ai2thor_demo_001__episode_000002_t000034 | 77.495 | 137.124 | 0.000 | 77.495 |
| FloorPlan201 | hard | ai2thor_demo_001__episode_000002_t000019 | 6.698 | 5.613 | 1.543 | 5.155 |
| FloorPlan201 | failure | ai2thor_demo_001__episode_000002_t000030 | 114.734 | 198.587 | 1.697 | 113.037 |
