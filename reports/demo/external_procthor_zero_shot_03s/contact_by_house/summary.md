# External Generalization Demo Selection

- Dataset: `data/wit_vz/processed/procthor_demo_001_03s`
- Predictions: `reports/demo/external_procthor_zero_shot_03s/eval_all/predictions.jsonl`
- Selected examples: `6`

| Group | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |
|---|---|---|---:|---:|---:|---:|
| procthor_seed_1202 | easy | procthor_demo_001__episode_000001_t000033 | 110.206 | 187.225 | 0.463 | 109.744 |
| procthor_seed_1202 | hard | procthor_demo_001__episode_000001_t000007 | 18.967 | 29.774 | 1.062 | 17.905 |
| procthor_seed_1202 | failure | procthor_demo_001__episode_000001_t000005 | 147.371 | 234.853 | 1.536 | 145.834 |
| procthor_seed_1203 | easy | procthor_demo_001__episode_000002_t000029 | 72.097 | 114.808 | 0.344 | 71.753 |
| procthor_seed_1203 | hard | procthor_demo_001__episode_000002_t000025 | 17.449 | 28.070 | 3.284 | 14.166 |
| procthor_seed_1203 | failure | procthor_demo_001__episode_000002_t000032 | 125.554 | 206.911 | 1.186 | 124.368 |
