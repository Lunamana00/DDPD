# 2026 Capstone

This repository currently contains a ViZDoom-based research prototype for egocentric future path prediction.

Main document:

- [Egocentric Cue-Memory Path Prediction Critic](docs/path_prediction.md)

Earlier dataset inspection notes:

- [ViZDoom / DoomFrameDataset structure](VIZDOOM_DATASET_STRUCTURE.md)
- [Route-aware ViZDoom collection notes](ROUTE_VIZDOOM_COLLECTION.md)

The current path-prediction prototype does not use `chosen_route_id` or `nearest_route_id` as the learning target. It predicts future local path waypoints from RGB history and relative ego-motion.

Current mini result:

```text
cue_memory_residual: ADE=51.5144 FDE=91.0359
constant_velocity:   ADE=56.2430 FDE=96.1106
```

The earlier `cue_memory_mini` run collapsed to tiny direct-regression predictions. The active proposed run uses a constant-velocity residual head, auto coordinate scaling, and a trainable `small_cnn` smoke-test backbone.

Long-horizon smoke results are available in `runs/horizon_sweep/horizon_summary.md`. The current sweep trains horizons from 1 to 10 seconds on `data/wit_vz/raw/wit_vz_basic_10s`; at 10 seconds, `cue_memory_residual` reaches `ADE=178.9178` and `FDE=297.9203` versus `constant_velocity` at `ADE=315.8391` and `FDE=602.4938`.
