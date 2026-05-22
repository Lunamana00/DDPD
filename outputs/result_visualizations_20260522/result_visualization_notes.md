# Result Visualization Notes

Date: 2026-05-22

All three figures use v4 defaults results. No presentation slide file is
created or committed.

## viz_01_trajectory_overlay.png

Shows actual v4-derived 10s test samples from
`data/wit_vz/processed/horizon_sweep_v4_defaults/future_10s`. Each subplot
compares GT future local path, Full Model, and Motion-only CV. The coordinate
system is egocentric local coordinates with x-axis = right and y-axis =
forward. RGB insets use the last history frame when available.

Selected actual v4 samples:

- `wit_vz_v4_default_multi_deathmatch_001__episode_000013_t000017`: Full ADE 88.18, Motion-only CV ADE 719.44, RGB inset=yes
- `wit_vz_v4_default_predict_position_001__episode_000018_t000158`: Full ADE 164.31, Motion-only CV ADE 753.03, RGB inset=yes
- `wit_vz_v4_default_predict_position_001__episode_000018_t000159`: Full ADE 184.07, Motion-only CV ADE 768.55, RGB inset=yes

Presenter script: "These are real v4 test samples, not schematic paths. The
orange dashed path is recent velocity extrapolation, while the full model uses
visual DINOv3 cues and memory. The examples were chosen because the full model
improves over motion-only extrapolation, especially when the future path bends
or changes direction."

## viz_02_horizon_error_growth.png

Shows v4 per-step error growth for the 10s test split, with a horizon summary
inset for 1s, 3s, 5s, and 10s ADE. Values come from
`outputs/v4_inference_ablation/results.json` and match the v4 test metrics in
the reports.

Presenter script: "The model does not just improve the final number; it slows
the growth of error across future time. The advantage is especially visible at
10 seconds, where Motion-only CV drifts much faster."

## viz_03_ablation_10s_ade.png

Shows v4 10s inference-time ablation ADE. Full Model is the baseline,
Motion-only CV and visual-token ablations show the effect of DINO/visual
information, and No Cue Memory Update is highlighted in red.

Presenter script: "The largest degradation comes from removing the cue memory
update, so the memory mechanism is a central contributor at inference time.
DINO/visual information also matters: both Motion-only CV and Zero Visual
Tokens are worse than the full model."
