# Result Visualization Notes

Date: 2026-05-22

All three figures use v4 defaults results. No presentation slide file is
created or committed.

## viz_01_trajectory_overlay.png

Shows six actual v4-derived 10s test samples from
`data/wit_vz/processed/horizon_sweep_v4_defaults/future_10s`. Each subplot
compares GT future local path, Full Model, and Motion-only CV. The coordinate
system is egocentric local coordinates with x-axis = right and y-axis =
forward. RGB insets use the last history frame when available. The selection
prioritizes held-out samples with different source/episode ids, visible
curve/lateral motion, and Full Model improvement over Motion-only CV.

Selected actual v4 samples:

- `wit_vz_v4_default_deathmatch_001__episode_000034_t000028`: source `wit_vz_v4_default_deathmatch_001`, episode `wit_vz_v4_default_deathmatch_001__episode_000034`, Full ADE 340.62, Motion-only CV ADE 672.61, gain 331.98, RGB inset=yes
- `wit_vz_v4_default_health_gathering_001__episode_000016_t000011`: source `wit_vz_v4_default_health_gathering_001`, episode `wit_vz_v4_default_health_gathering_001__episode_000016`, Full ADE 305.63, Motion-only CV ADE 631.82, gain 326.19, RGB inset=yes
- `wit_vz_v4_default_basic_audio_001__episode_000016_t000009`: source `wit_vz_v4_default_basic_audio_001`, episode `wit_vz_v4_default_basic_audio_001__episode_000016`, Full ADE 259.43, Motion-only CV ADE 656.21, gain 396.77, RGB inset=yes
- `wit_vz_v4_default_defend_the_center_001__episode_000026_t000015`: source `wit_vz_v4_default_defend_the_center_001`, episode `wit_vz_v4_default_defend_the_center_001__episode_000026`, Full ADE 339.06, Motion-only CV ADE 635.61, gain 296.55, RGB inset=yes
- `wit_vz_v4_default_multi_deathmatch_001__episode_000012_t000010`: source `wit_vz_v4_default_multi_deathmatch_001`, episode `wit_vz_v4_default_multi_deathmatch_001__episode_000012`, Full ADE 237.72, Motion-only CV ADE 515.76, gain 278.05, RGB inset=yes
- `wit_vz_v4_default_basic_001__episode_000006_t000087`: source `wit_vz_v4_default_basic_001`, episode `wit_vz_v4_default_basic_001__episode_000006`, Full ADE 217.91, Motion-only CV ADE 468.96, gain 251.05, RGB inset=yes

Presenter script: "These are real v4 test samples, not schematic paths. The
orange dashed path is recent velocity extrapolation, while the full model uses
visual DINOv3 cues and memory. I selected a diverse set across source and
episode ids so the comparison is not just three neighboring moments from the
same scene."

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
