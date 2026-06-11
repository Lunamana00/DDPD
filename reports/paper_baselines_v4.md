# V4 Paper-Adapted Baseline Evaluation

## Scope

- Dataset family: `horizon_sweep_v4_defaults`.
- Metrics: ADE/FDE in local egocentric coordinates; lower is better.
- The paper baselines are adapters, not exact reproductions of the original interactive systems.
- Device: `cuda`.

## Baseline Adaptation

| Paper | Adapter used here | Missing vs original paper |
| --- | --- | --- |
| Khaleque, Cook, & Gow (2024) | Deterministic center-biased exploratory context-steering rollout. | No live object/motivation/coverage state is stored in WIT-VZ processed samples. |
| Xu et al. (2026) | Last-frame pixels-only saliency steering rollout. | No live ARPG controller, no trained STP/MSTP detector for ViZDoom. |

## Results

| Horizon | Model | Available | Test samples | ADE | FDE | ADE gain vs best paper proxy | FDE gain vs best paper proxy |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1s | Khaleque-style exploratory proxy | yes | 15373 | 46.8594 | 73.1665 | -28.2306 | -27.9678 |
| 1s | Xu-style pixels-only saliency proxy | yes | 15373 | 36.5431 | 57.1757 | 0.0000 | 0.0000 |
| 1s | Internal motion-only constant velocity | yes | 15373 | 33.1120 | 51.4413 | 9.3891 | 10.0294 |
| 1s | Ours: cached DINOv3 trajectory predictor | yes | 15373 | 26.8676 | 41.5629 | 26.4771 | 27.3068 |
| 3s | Khaleque-style exploratory proxy | yes | 11884 | 101.7428 | 163.3355 | -18.0144 | -6.2985 |
| 3s | Xu-style pixels-only saliency proxy | yes | 11884 | 86.2122 | 153.6574 | 0.0000 | 0.0000 |
| 3s | Internal motion-only constant velocity | yes | 11884 | 75.7201 | 131.6904 | 12.1700 | 14.2960 |
| 3s | Ours: cached DINOv3 trajectory predictor | yes | 11884 | 62.1001 | 103.3531 | 27.9683 | 32.7379 |
| 5s | Khaleque-style exploratory proxy | yes | 10294 | 139.2382 | 222.0337 | -11.3883 | 0.0000 |
| 5s | Xu-style pixels-only saliency proxy | yes | 10294 | 125.0026 | 234.8750 | 0.0000 | -5.7835 |
| 5s | Internal motion-only constant velocity | yes | 10294 | 111.2669 | 202.7233 | 10.9883 | 8.6971 |
| 5s | Ours: cached DINOv3 trajectory predictor | yes | 10294 | 88.6020 | 157.0852 | 29.1198 | 29.2516 |
| 10s | Khaleque-style exploratory proxy | yes | 7434 | 198.3879 | 286.8605 | 0.0000 | 0.0000 |
| 10s | Xu-style pixels-only saliency proxy | yes | 7434 | 254.5895 | 495.8117 | -28.3291 | -72.8407 |
| 10s | Internal motion-only constant velocity | yes | 7434 | 217.1668 | 408.6508 | -9.4658 | -42.4563 |
| 10s | Ours: cached DINOv3 trajectory predictor | yes | 7434 | 154.5734 | 258.7196 | 22.0853 | 9.8100 |

## Interpretation Guardrails

- Treat these rows as paper-adapted offline trajectory proxies.
- Do not claim exact reproduction unless the original interactive environment, model checkpoints, and control loop are available.
- These baselines are useful for answering whether a simple paper-inspired decision rule can match the learned WIT-VZ trajectory predictor under the same ADE/FDE protocol.
