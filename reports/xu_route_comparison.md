# Route Target Comparison on VisualGuidance

## Corrected Comparison

This report compares final screen-space route-target quality, not a selector-stage substitution.

- Ours: image -> cue-memory direct route-target predictor -> MSTP-like box.
- Xu/VisualGuidance oracle: image + GT STP candidates -> MSTP selector -> selected candidate.
- Xu/VisualGuidance detected: image -> Faster R-CNN STP detector -> MSTP selector -> selected detected candidate.

VisualGuidance does not provide ego-motion or future trajectory labels, so this is not an ADE/FDE trajectory benchmark. The common target is the GT MSTP bounding box.

The direct route model is a screen-space adaptation of the cue-memory architecture. It uses the configured visual encoder from its checkpoint and does not use explicit STP candidate boxes.

## Test Metrics

IoU hit threshold: `0.5`. Center hit threshold: `0.1` normalized image distance.

| Method | Samples | Main metric | Mean IoU | Center error | Top-3 / upper-bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours: direct cue-memory route target | 95 | hit@IoU 0.0737 | 0.1651 | 0.2011 | center-hit 0.2947 |
| Xu baseline: oracle STP candidates + MSTP selector | 95 | acc 0.7684 | selected-hit 0.7684 | - | top-3 1.0000 |
| Xu baseline: detected STP candidates + MSTP selector | 95 | hit@IoU 0.3053 | - | - | detector recall 0.5263 |

## Interpretation Guide

- If direct cue-memory hit@IoU is close to or above the detected Xu pipeline, then our method is competitive at finding a route target without explicit STP proposals.
- If oracle-candidate Xu is much higher than detected Xu, the bottleneck is STP detection rather than MSTP selection.
- If direct cue-memory is low but center error is reasonable, the model is roughly pointing toward the right area but not producing accurate boxes.
- This comparison is a proxy for screen-only route finding. It does not prove full trajectory navigation because VisualGuidance has no temporal movement labels.

## Config

```text
annotations: .tmp_external/VisualGuidance/data/processed/test_dataset.json
selector_annotations: .tmp_external/VisualGuidance/data/processed/model2_test_dataset.json
direct_backbone: small_cnn
direct_hidden_dim: 128
direct_spatial_relation_type: topk_graph
selector_baseline: VisualGuidance local-crop/global-context selector
selector_image_size: 128
direct_checkpoint: runs/xu_route/direct_cue_memory_visualguidance/best.pt
selector_checkpoint: runs/xu_mstp/visualguidance_baseline/best.pt
detector_checkpoint: runs/xu_stp/detector_visualguidance/best.pt
score_threshold: 0.3
max_detections: 3
```
