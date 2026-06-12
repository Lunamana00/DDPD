# Navigation Oracle Baselines on WIT-VZ v4

## Scope

- Task: predict WIT-VZ future local trajectory `[forward, right]` and evaluate with ADE/FDE.
- These are privileged navigation/pathfinding adapters, not input-matched competitors.
- `PointNav/DD-PPO goal-oracle` receives the GT future endpoint as the PointGoal.
- `A* pose-graph oracle` builds a traversability graph from recorded world poses and plans to the GT future endpoint.
- Prediction JSONL files: `outputs/navigation_oracle_baselines_v4`.

## Results

| Horizon | Model | Test samples | ADE | FDE | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1s | Internal motion-only constant velocity | 15373 | 33.1120 | 51.4413 | Uses only recent ego-motion. Included for context. |
| 1s | PointNav/DD-PPO goal-oracle adapter | 15373 | 10.8429 | 0.0000 | Privileged: receives the GT future endpoint as PointGoal. |
| 1s | Classical A* pose-graph oracle | 15373 | 11.5689 | 0.0000 | Privileged: uses recorded pose graph and GT future endpoint. |
| 3s | Internal motion-only constant velocity | 11884 | 75.7201 | 131.6904 | Uses only recent ego-motion. Included for context. |
| 3s | PointNav/DD-PPO goal-oracle adapter | 11884 | 31.6851 | 0.0000 | Privileged: receives the GT future endpoint as PointGoal. |
| 3s | Classical A* pose-graph oracle | 11884 | 32.5039 | 0.0000 | Privileged: uses recorded pose graph and GT future endpoint. |
| 5s | Internal motion-only constant velocity | 10294 | 111.2669 | 202.7233 | Uses only recent ego-motion. Included for context. |
| 5s | PointNav/DD-PPO goal-oracle adapter | 10294 | 46.3397 | 0.0000 | Privileged: receives the GT future endpoint as PointGoal. |
| 5s | Classical A* pose-graph oracle | 10294 | 46.9705 | 0.0000 | Privileged: uses recorded pose graph and GT future endpoint. |
| 10s | Internal motion-only constant velocity | 7434 | 217.1669 | 408.6508 | Uses only recent ego-motion. Included for context. |
| 10s | PointNav/DD-PPO goal-oracle adapter | 7434 | 87.2023 | 0.0000 | Privileged: receives the GT future endpoint as PointGoal. |
| 10s | Classical A* pose-graph oracle | 7434 | 87.4048 | 0.0000 | Privileged: uses recorded pose graph and GT future endpoint. |

## Interpretation

- If PointNav has very low FDE, that is expected: it is given the true endpoint.
- If A* performs well, it shows the value of map/pose/goal privileges, not that the RGB-only problem is solved.
- Use these rows to frame the gap between local visual prediction and classical goal/map-based navigation.
