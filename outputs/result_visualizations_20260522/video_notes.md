# GT Comparison Replay Notes

Date: 2026-05-22

## Artifacts

- MP4: `outputs/result_visualizations_20260522/viz_04_gt_comparison_replay.mp4`
- GIF: `outputs/result_visualizations_20260522/viz_04_gt_comparison_replay.gif`
- First frame: `outputs/result_visualizations_20260522/viz_04_frame_000.png`
- Middle frame: `outputs/result_visualizations_20260522/viz_04_frame_mid.png`

## Source

- Dataset: `data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s`
- Base dataset family: v4 defaults, held-out test split
- Checkpoint: `checkpoints/wit_vz_v4_defaults_dinov3_single_03s.pt`
- DINOv3 cache: `data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny`
- Horizon: 3 seconds, 15 future waypoints at 5 FPS
- Input history: 1 second of visual and ego-motion history
- Video: 250 frames at 10 FPS, 25.0s

This is an offline prediction replay, not closed-loop control. The model is not
driving the agent in the video; each frame shows a held-out logged ViZDoom state
and compares future-path predictions against the logged future trajectory.

The large right panel is the main comparison view. The bright green moving dot
walks along the GT future path from t+0.2s to t+3.0s while the full model and
Motion-only CV predictions remain overlaid for that logged state. This makes
the GT future motion explicit instead of showing it only as a static line. Each
frame uses an adaptive zoom around the current GT and Full Model paths for
readability; if Motion-only CV drifts far away, its dashed line can run to the
plot edge.

## Colors

- Green: GT future path
- Orange: Motion-only CV, recent velocity extrapolation
- Blue/purple: Full Model prediction

## Metrics

- ADE: mean Euclidean distance over all future waypoints.
- FDE: Euclidean distance at the final future waypoint.

## Selection Rule

Selected from held-out v4 test samples where RGB, GT future path, Motion-only CV
prediction, and Full Model prediction are all available, and where the Full
Model improves over Motion-only CV on average over the replay segment.

Selected episode: `wit_vz_v4_default_health_gathering_001__episode_000020`
Selected sample range: `wit_vz_v4_default_health_gathering_001__episode_000020_t000008` to `wit_vz_v4_default_health_gathering_001__episode_000020_t000057`
Mean Full Model ADE: 60.30
Mean Motion-only CV ADE: 138.99

## Presenter Script

1. "This is an offline replay from a held-out ViZDoom test episode, so the
   agent trajectory is fixed and the model is only predicting future local
   path from each logged state."
2. "Green is the logged future, orange is recent velocity extrapolation, and
   blue is the full visual-memory model."
3. "The key visual cue is that the full model often bends toward the logged
   future path instead of drifting like the motion-only baseline."
