# Paper Baseline Adapters

This project predicts future local trajectory from WIT-VZ samples:

```text
RGB history + ego-motion history -> future local path [forward, right]
```

The requested literature baselines do not expose the same offline prediction
task, so the repository implements two comparable forms:

```text
1. paper-adapted offline proxy baselines
2. trainable paper-inspired trajectory baselines
```

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

## Trainable Paper-Inspired Baselines

These baselines are trained with the same `src.train_path_predictor` loop,
same WIT-VZ split, and same ADE/FDE evaluation protocol as the proposed model.

### Khaleque-inspired Trainable Baseline

Implemented as `khaleque_motivated_baseline`.

```text
ego-motion history [B,T,3]
-> GRU agent-state encoder
-> learned motivation tokens
-> horizon queries with cross-attention
-> cumulative local path [B,H,2]
```

This is motion-only. It does not use RGB, DINO tokens, cue memory, or map state.
It is meant to answer whether a trainable exploratory-agent style motion prior
can explain the WIT-VZ local trajectory labels.

### Xu-inspired Trainable Pixels-only Baseline

Implemented as `xu_pixels_only_baseline`.

```text
RGB visual history or cached visual tokens [B,T,N,C]
-> learned token scoring / spatial pooling
-> temporal GRU
-> future local path [B,H,2]
```

This is screen-only. It does not use ego-motion history or cue memory. In the
v4 run it uses the same cached DINOv3 ConvNeXt-Tiny visual tokens as the main
model, but with a much simpler pooling-and-GRU trajectory head.

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

## V4 3s Trainable Baseline Run

```bash
python3 -m src.train_path_predictor \
  --config configs/baselines/train_khaleque_motivated_v4_03s.yaml

python3 -m src.train_path_predictor \
  --config configs/baselines/train_xu_pixels_only_v4_03s.yaml

python3 scripts/summarize_trainable_paper_baselines.py
```

Outputs:

```text
outputs/trainable_paper_baselines_v4_03s/results.json
reports/trainable_paper_baselines_v4_03s.md
```

## Reporting Rule

Use the phrase `paper-adapted offline proxy baseline`, not `exact
reproduction`, unless the original interactive environment, original model
weights, and original control loop are available.
