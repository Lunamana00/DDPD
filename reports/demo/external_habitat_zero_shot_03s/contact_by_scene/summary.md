# External Generalization Demo Selection

- Dataset: `data/wit_vz/processed/habitat_demo_001_03s`
- Predictions: `reports/demo/external_habitat_zero_shot_03s/eval_all/predictions.jsonl`
- Selected examples: `3`

| Group | Case | Sample | ADE | FDE | CV ADE | Model-CV ADE |
|---|---|---|---:|---:|---:|---:|
| skokloster-castle | easy | habitat_demo_001__episode_000001_t000014 | 74.342 | 122.923 | 0.160 | 74.183 |
| skokloster-castle | hard | habitat_demo_001__episode_000003_t000022 | 3.537 | 5.395 | 0.564 | 2.973 |
| skokloster-castle | failure | habitat_demo_001__episode_000002_t000006 | 139.186 | 264.930 | 0.606 | 138.580 |
