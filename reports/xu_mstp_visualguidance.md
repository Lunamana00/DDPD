# Xu / VisualGuidance MSTP Selection With Cue-Memory

## Scope

- External dataset/code: `Nortrom1213/VisualGuidance`.
- Task: given a screenshot and candidate STP boxes, select the MSTP candidate index.
- This is not ADE/FDE trajectory prediction; it is Xu-style STP/MSTP selection.
- Current implementation uses provided candidate boxes from `model2_*_dataset.json`, so it evaluates the MSTP selector stage, not the full STP detector + selector pipeline.

## Model

```text
image
-> visual token encoder
-> projection + 2D spatial positional encoding
-> bottleneck adapter
-> dynamic spatial graph
-> TokenLearner cue selector
-> cue memory bank
-> candidate box query + ROI token pooling
-> cross-attention to cue memory
-> candidate scores
```

## Training Setup

```text
train annotations: .tmp_external/VisualGuidance/data/processed/model2_train_dataset.json
test annotations:  .tmp_external/VisualGuidance/data/processed/model2_test_dataset.json
image root:        .tmp_external/VisualGuidance/data/datasets
backbone:          small_cnn
image_size:        128
hidden_dim:        128
num_cue_tokens:    8
spatial relation:  topk_graph, k=8
epochs:            30 with early stopping patience 8
optimizer:         AdamW, lr=5e-4, weight_decay=1e-3
GPU:               RTX 3090
```

## Result

```text
best epoch:       5
epochs run:       13
MSTP accuracy:    0.8316
Top-3 accuracy:   1.0000
samples:          95
mean candidates:  1.6526
loss:             0.3920
```

## Interpretation

This confirms that our visual cue-memory machinery can be transferred from trajectory prediction to Xu's MSTP selection setting, but the current result should be described precisely:

```text
Cue-memory MSTP selector on VisualGuidance candidate boxes
```

It should not yet be described as a full Xu reproduction, because the first-stage STP detector is not trained/evaluated in this run. Full reproduction requires:

```text
image -> STP detector -> candidate boxes -> cue-memory MSTP selector -> MSTP
```
