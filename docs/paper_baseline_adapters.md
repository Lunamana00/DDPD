# Paper Baseline Adapters

This project predicts future local trajectory from WIT-VZ samples:

```text
RGB history + ego-motion history -> future local path [forward, right]
```

The requested literature baselines do not expose the same offline prediction
task, so the repository implements explicit paper-adapted trajectory proxies.

## Codebase Audit

| Paper | Public code status | Directly usable for WIT-VZ ADE/FDE? | Notes |
| --- | --- | --- | --- |
| Khaleque, Cook, & Gow (2024), *Experiments in Motivating Exploratory Agents* | A referenced GitHub repository was not accessible during implementation. | No | The paper concerns interactive exploratory agents with motivation/context state, not offline trajectory regression. |
| Xu et al. (2026), *How Far Can We Go with Pixels Alone?* | No direct agent repository was found during implementation. | No | It is closer conceptually because it uses screen-only navigation, but it still depends on an interactive controller. |
| Xu et al. / VisualGuidance | `https://github.com/Nortrom1213/VisualGuidance` is public. | No | It provides STP/MSTP screenshot detection code/data, not ego-motion or future local paths. |

## Implemented Adapters

### Khaleque-style Exploratory Proxy

Implemented in `src/models/paper_proxies.py` as
`khaleque_center_random_prediction`.

Adaptation:

```text
train-set source pose bounds -> per-source center prior
recent ego speed -> rollout speed
deterministic sector random direction around center bias -> local path
```

This approximates context-steering exploration without claiming to reproduce
the original live motivation system.

### Xu-style Pixels-only Proxy

Implemented in `src/models/paper_proxies.py` as
`xu_pixels_saliency_prediction`.

Adaptation:

```text
last RGB frame -> texture/brightness horizontal saliency
best column -> steering angle
recent ego speed -> local path rollout
```

This approximates a screen-only visual-interest controller under the WIT-VZ
offline ADE/FDE protocol.

## V4 Evaluation Script

Run on the GPU server where v4 processed datasets and cached features exist:

```bash
python3 scripts/evaluate_paper_baselines_v4.py \
  --horizons 1 3 5 10 \
  --device cuda \
  --batch-size 256
```

Outputs:

```text
outputs/paper_baselines_v4/results.json
reports/paper_baselines_v4.md
```

The script evaluates:

```text
Khaleque-style exploratory proxy
Xu-style pixels-only saliency proxy
Internal constant-velocity baseline
Ours: cached DINOv3 trajectory predictor, when the matching checkpoint exists
```

## Reporting Rule

Use the phrase `paper-adapted offline proxy baseline`, not `exact
reproduction`, unless the original interactive environment, original model
weights, and original control loop are available.
