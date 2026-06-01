# Trainable Paper-Inspired Baselines V4 3s

## Scope

- Dataset: `horizon_sweep_v4_defaults/future_03s`.
- Task: predict future local path `[forward, right]` from the same test split.
- Metrics: ADE/FDE in local egocentric coordinates; lower is better.
- These are trainable adapters for WIT-VZ, not exact reproductions of the original interactive systems.

## Models

- Khaleque-inspired trainable baseline: ego-motion history encoder plus learned motivation tokens; no RGB.
- Xu-inspired trainable baseline: pixels-only visual history encoder using cached DINO tokens; no ego-motion.

## Results

| Model | Kind | Available | ADE | FDE | Best epoch | Train/val ADE gap | Epoch sec | Peak VRAM MB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Khaleque-style exploratory proxy | khaleque_proxy | yes | 101.7428 | 163.3355 | - | - | - | - |
| Xu-style pixels-only saliency proxy | pixels_proxy | yes | 86.2122 | 153.6574 | - | - | - | - |
| Khaleque-inspired trainable ego-motion baseline | trainable | yes | 67.9502 | 116.9806 | 7 | -5.9008 | 4.5222 | 53.0571 |
| Xu-inspired trainable pixels-only baseline | trainable | yes | 64.2422 | 103.1896 | 6 | 0.0454 | 70.2527 | 1202.6777 |
| Internal motion-only constant velocity | constant_velocity | yes | 75.7201 | 131.6904 | - | - | - | - |
| Ours: cached DINOv3 trajectory predictor | checkpoint | yes | 62.1001 | 103.3532 | - | - | - | - |

## Interpretation

- If a trainable paper baseline still trails the proposed model, the gap is less likely to be caused only by an unfair hand-coded proxy.
- If the Xu-inspired trainable baseline is strong, screen-only visual history already carries meaningful navigation signal.
- If the Khaleque-inspired trainable baseline is close to constant velocity, ego-motion alone explains much of the local short-horizon behavior.
