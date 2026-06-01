# Xu / VisualGuidance Adaptation Plan

## Why This Is A Different Task

The original WIT-VZ model predicts a future local trajectory:

```text
(RGB history, ego-motion history) -> future local path [H, 2]
```

Xu / VisualGuidance defines a different screen-understanding task:

```text
I_t -> S_t = {candidate STP boxes}
S_t -> m_t = selected MSTP index
```

So applying our method to Xu data should not be reported with ADE/FDE. It
should be reported with MSTP selection metrics such as accuracy and top-k
accuracy.

## Implemented First Step

This repo now includes a cue-memory MSTP selector:

```text
image
-> visual token encoder
-> projection + 2D spatial positional encoding
-> bottleneck adapter
-> spatial relation module
-> TokenLearner cue selection
-> cue memory
-> candidate box query + ROI token pooling
-> cross-attention to cue memory
-> score each candidate STP
```

The model assumes candidate STP boxes are already available from the Xu dataset
or a trained STP detector. This directly targets the second stage of the
VisualGuidance pipeline: MSTP selection.

## Notation

```text
I_i: screenshot image
B_i = {b_i1, ..., b_iN}: candidate STP boxes
y_i: ground-truth MSTP candidate index

f_theta(I_i, B_i) -> scores z_i in R^N
loss = CrossEntropy(z_i, y_i)
prediction = argmax_j z_ij
```

## Run

```bash
git clone https://github.com/Nortrom1213/VisualGuidance.git .tmp_external/VisualGuidance

python scripts/train_xu_mstp_selector.py \
  --config configs/xu_mstp/train_cue_memory_mstp_visualguidance.yaml
```

## Metrics

```text
accuracy      = whether top-1 predicted candidate equals gt_index
top3_accuracy = whether gt_index is within top-3 candidates
mean_candidates = average number of STP candidates per frame
```

## Full Pipeline

The STP detector stage is implemented with a VisualGuidance-aligned Faster
R-CNN detector:

```bash
python scripts/train_xu_stp_detector.py \
  --config configs/xu_mstp/train_stp_detector_visualguidance.yaml

python scripts/evaluate_xu_full_pipeline.py \
  --config configs/xu_mstp/evaluate_full_pipeline_visualguidance.yaml
```

The comparison now has two levels:

```text
oracle candidate boxes -> cue-memory MSTP selector
detected candidate boxes -> cue-memory MSTP selector
```

The second line is the actual end-to-end Xu-style pipeline. Detector
MSTP-recall is reported because it is an upper bound on what the selector can
recover when GT candidate boxes are removed.
