# Real Counterfactual Rollout Demo

This demo renders actual simulator branches instead of drawing different paths
on the same recorded RGB frames.

## Deliverables

| File | Meaning |
|---|---|
| `reports/demo/presentation_sequence/demo_real_counterfactual_rollout_suite.mp4` | Composite side-by-side comparison: CV, GT, and ours. |
| `reports/demo/presentation_sequence/demo_real_counterfactual_rollout_suite_cv.mp4` | First-person branch video when the controller follows the CV path. |
| `reports/demo/presentation_sequence/demo_real_counterfactual_rollout_suite_target.mp4` | First-person branch video when the controller follows the GT path. |
| `reports/demo/presentation_sequence/demo_real_counterfactual_rollout_suite_prediction.mp4` | First-person branch video when the controller follows the model prediction path. |
| `reports/demo/presentation_sequence/demo_vizdoom_counterfactual_rollout_3way.mp4` | ViZDoom-only composite branch rollout. |
| `reports/demo/presentation_sequence/demo_miniworld_counterfactual_rollout.mp4` | MiniWorld-only composite branch rollout. |
| `reports/demo/presentation_sequence/demo_ai2thor_counterfactual_rollout.mp4` | AI2-THOR-only composite branch rollout. |

## Rendered Environments

| Environment | Samples rendered | Start reset method | Max start position error | Max start angle error |
|---|---:|---|---:|---:|
| ViZDoom | 6 | action replay, then strict warp/turn alignment when replay drifts | 3.72 Doom units | 2.13 deg |
| MiniWorld | 2 | direct agent state reset from WIT-VZ current pose | 0.00 | 0.00 deg |
| AI2-THOR | 2 | CloudRendering Teleport to WIT-VZ current pose | 0.00 | 0.00 deg |

## Method

Each selected sample has the same starting pose and three local future paths:

```text
CV path         = recent-motion extrapolation
GT path         = future_local_path label
Prediction path = model output
```

For each path, the renderer creates an independent simulator branch:

```text
same current pose
-> convert local [forward, right] waypoints to world waypoints
-> steer with environment-specific discrete actions
-> render first-person RGB frames
```

The composite video places these independently rendered branches side by side.
The branch-only videos keep only one method at a time, proving that CV, GT, and
prediction were not just overlays on the same RGB recording.

## Verification

All final videos were opened with OpenCV after rendering.

| Video | Frames | FPS | Resolution |
|---|---:|---:|---|
| `demo_real_counterfactual_rollout_suite.mp4` | 230 | 8 | 1920x1080 |
| `demo_real_counterfactual_rollout_suite_cv.mp4` | 230 | 8 | 1920x1080 |
| `demo_real_counterfactual_rollout_suite_target.mp4` | 230 | 8 | 1920x1080 |
| `demo_real_counterfactual_rollout_suite_prediction.mp4` | 230 | 8 | 1920x1080 |

## Limitations

This is a demo-grade counterfactual rollout. It is not a perfect physics replay
of the original episode. ViZDoom samples whose start pose could not be aligned
within the strict threshold were skipped. The path follower is a simple
waypoint controller, so it approximates each trajectory with the simulator's
available discrete actions.
