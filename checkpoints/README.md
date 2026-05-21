# Curated Checkpoints

Date: 2026-05-21

These are the lightweight published checkpoints for the v4 default ViZDoom
DINOv3 single-output path prediction models. Training run directories and
prediction dumps are intentionally not tracked in Git.

| File | Source run | Horizon | Test ADE | Test FDE | SHA256 |
| --- | --- | ---: | ---: | ---: | --- |
| `wit_vz_v4_defaults_dinov3_single_01s.pt` | `runs/wit_vz_v4_defaults_dinov3_timesformer_balanced_dp_single_bs512/best.pt` | 1s | 26.8676 | 41.5629 | `0f947382dfeb29487537a87bf4693f229e5969747955af811ba30e30003aa8d8` |
| `wit_vz_v4_defaults_dinov3_single_03s.pt` | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_03s/best.pt` | 3s | 62.1001 | 103.3531 | `7271d4a04268733d3ba531b4452f3b0f072db00638f6517c2ab9d45abfc91ed7` |
| `wit_vz_v4_defaults_dinov3_single_05s.pt` | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_05s/best.pt` | 5s | 88.6020 | 157.0852 | `a58b54155c053141a5dd82fa866ef8186c0273e135d29aae8335c2db2545a6b2` |
| `wit_vz_v4_defaults_dinov3_single_10s.pt` | `runs/horizon_sweep_v4_defaults/dinov3_timesformer_single_10s/best.pt` | 10s | 154.5734 | 258.7196 | `2e8141b9ed0c4c5d78e2450be3ff40477ccfae314a5007951ab862b658f9d5e3` |

See `reports/horizon_sweep_v4_defaults_single_output_20260521.md` for the
full horizon sweep summary.
