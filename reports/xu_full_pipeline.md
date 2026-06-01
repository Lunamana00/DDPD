# Xu Full STP-to-MSTP Pipeline

## Scope

- Dataset: VisualGuidance `test_dataset.json` screenshots.
- Pipeline: image -> Faster R-CNN STP detector -> cue-memory MSTP selector.
- End-to-end success: selected detected box has IoU >= threshold with GT MSTP.

## Metrics

### Detector

The detector was trained with Faster R-CNN ResNet50-FPN initialized from COCO
weights. Early stopping selected epoch 3.

| Metric | Value |
| --- | ---: |
| Test images | 95 |
| GT boxes | 157 |
| Pred boxes @ score 0.3, top 10 | 370 |
| Precision@IoU 0.5 | 0.2243 |
| Recall@IoU 0.5 | 0.5287 |
| F1@IoU 0.5 | 0.3150 |
| MSTP recall@IoU 0.5 | 0.6105 |

### End-To-End

| Metric | Value |
| --- | ---: |
| Samples | 95 |
| Mean detections | 2.4947 |
| No detection rate | 0.0316 |
| Detector MSTP recall@IoU | 0.5263 |
| End-to-end MSTP accuracy@IoU | 0.3684 |
| End-to-end MSTP top-3@IoU | 0.5263 |
| Oracle-candidate selector accuracy | 0.8316 |
| Oracle-candidate selector top-3 | 1.0000 |

### Threshold Sweep

| Score threshold | Max detections | Mean detections | Detector MSTP recall | End-to-end top-1 | End-to-end top-3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 10 | 9.8105 | 0.7368 | 0.1789 | 0.4632 |
| 0.10 | 10 | 8.8526 | 0.7368 | 0.2211 | 0.5158 |
| 0.20 | 10 | 5.9263 | 0.6632 | 0.2632 | 0.5579 |
| 0.30 | 10 | 3.8526 | 0.6105 | 0.3474 | 0.5684 |
| 0.50 | 10 | 2.2211 | 0.4316 | 0.3368 | 0.4105 |
| 0.30 | 3 | 2.4947 | 0.5263 | 0.3684 | 0.5263 |

## Interpretation

- `Detector MSTP recall@IoU` is the upper bound for the selector once detector boxes replace oracle candidates.
- `End-to-end MSTP accuracy@IoU` is the actual full pipeline result.
- The gap from oracle-candidate selector accuracy measures how much performance is lost by replacing GT candidate boxes with detected boxes.
- Lower thresholds improve detector recall but introduce too many false-positive candidates, which hurts the selector. The best top-1 operating point here was score 0.3 with at most 3 detections.

## Config

```text
detector_checkpoint: runs/xu_stp/detector_visualguidance/best.pt
selector_checkpoint: runs/xu_mstp/cue_memory_visualguidance/best.pt
score_threshold: 0.3
iou_threshold: 0.5
max_detections: 3
```
