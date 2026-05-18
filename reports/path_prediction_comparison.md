# Path Prediction Comparison

| Run | Model | ADE | FDE | Notes |
| --- | --- | ---: | ---: | --- |
| constant_velocity | constant_velocity | 56.2430 | 96.1106 | Motion prior baseline |
| ego_motion_only | ego_motion_only | 86.5778 | 144.3438 | GRU over egomotion only |
| cue_memory_mini | cue_memory_path_predictor | 85.2724 | 143.2239 | Failed direct-regression smoke run; tiny prediction collapse |
| cue_memory_residual | cue_memory_path_predictor | 51.5144 | 91.0359 | CV residual + auto scale + trainable small CNN |
