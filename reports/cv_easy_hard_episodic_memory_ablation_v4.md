# CV Easy/Hard Subset Ablation

This report reuses already trained episodic-memory ablation checkpoints and re-evaluates their saved predictions on matched sample subsets.

Subset definition:

- `cv_easy_bottom25`: bottom 25% by constant-velocity ADE.
- `cv_hard_top25`: top 25% by constant-velocity ADE.
- Thresholds are computed from the `episodic_short_only` variant on the common sample-id intersection for each horizon.
- Duplicate episodic chunk predictions for the same sample id are averaged before comparison.

## Executive Summary

| Horizon | Common samples | Overall best | Easy best | Hard best | Hard-subset read |
|---|---:|---|---|---|---|
| 01s | 14706 | `current_short_window` | `long_mean_memory` | `long_gated_forget_ego` | long memory wins hard subset |
| 03s | 11285 | `long_attention_no_ego` | `long_gated_ego` | `episodic_short_only` | episodic training wins; explicit long memory not needed |
| 05s | 9720 | `episodic_short_only` | `long_attention_ego` | `long_gated_ego` | long memory wins hard subset |
| 10s | 7003 | `long_gated_ego` | `episodic_short_only` | `long_attention_no_ego` | long memory wins hard subset |

Main interpretation:

- Hard/easy is defined by the constant-velocity baseline, so it asks whether a sample is easy for motion extrapolation before looking at model errors.
- Explicit long memory wins the hard subset in 1s, 5s, and 10s, but not in 3s.
- The easy subset is mixed, so the result does not support a blanket claim that long memory is always better.
- The strongest defensible claim is conditional: memory is useful mainly on samples where recent-motion extrapolation is insufficient.

## Horizon 01s

- common samples: 14706
- easy threshold CV ADE <= 13.039
- hard threshold CV ADE >= 44.608

### all_common

Best ADE: `current_short_window` (27.051)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 14706 | 27.051 | 41.828 | 21.840 | 31.971 | -0.816 | 0.499 | 0.000 | 0.000 |
| episodic_short_only | 14706 | 27.867 | 43.074 | 22.255 | 31.971 | 0.000 | 0.000 | 0.816 | 0.501 |
| long_mean_memory | 14706 | 27.951 | 42.986 | 22.448 | 31.971 | 0.084 | 0.480 | 0.900 | 0.478 |
| long_attention_no_ego | 14706 | 27.523 | 42.894 | 22.226 | 31.971 | -0.344 | 0.480 | 0.472 | 0.475 |
| long_attention_ego | 14706 | 27.803 | 42.507 | 22.129 | 31.971 | -0.064 | 0.496 | 0.752 | 0.499 |
| long_gated_ego | 14706 | 27.682 | 42.554 | 22.493 | 31.971 | -0.185 | 0.458 | 0.631 | 0.456 |
| long_gated_forget_ego | 14706 | 27.282 | 42.118 | 21.768 | 31.971 | -0.585 | 0.514 | 0.231 | 0.509 |

### cv_easy_bottom25

Best ADE: `long_mean_memory` (9.484)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 3677 | 10.038 | 14.716 | 8.476 | 6.541 | -1.245 | 0.501 | 0.000 | 0.000 |
| episodic_short_only | 3677 | 11.283 | 16.362 | 8.170 | 6.541 | 0.000 | 0.000 | 1.245 | 0.499 |
| long_mean_memory | 3677 | 9.484 | 13.295 | 7.796 | 6.541 | -1.799 | 0.532 | -0.554 | 0.534 |
| long_attention_no_ego | 3677 | 9.693 | 13.856 | 8.110 | 6.541 | -1.590 | 0.494 | -0.345 | 0.485 |
| long_attention_ego | 3677 | 12.412 | 17.596 | 8.869 | 6.541 | 1.129 | 0.405 | 2.375 | 0.411 |
| long_gated_ego | 3677 | 11.048 | 15.382 | 9.013 | 6.541 | -0.235 | 0.389 | 1.010 | 0.404 |
| long_gated_forget_ego | 3677 | 11.048 | 16.145 | 8.481 | 6.541 | -0.236 | 0.476 | 1.010 | 0.471 |

### cv_hard_top25

Best ADE: `long_gated_forget_ego` (48.362)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 3677 | 49.514 | 78.523 | 45.632 | 67.860 | -0.761 | 0.504 | 0.000 | 0.000 |
| episodic_short_only | 3677 | 50.275 | 80.786 | 45.408 | 67.860 | 0.000 | 0.000 | 0.761 | 0.496 |
| long_mean_memory | 3677 | 52.057 | 82.446 | 48.603 | 67.860 | 1.782 | 0.441 | 2.543 | 0.435 |
| long_attention_no_ego | 3677 | 50.078 | 80.573 | 47.393 | 67.860 | -0.197 | 0.489 | 0.564 | 0.473 |
| long_attention_ego | 3677 | 49.135 | 78.166 | 44.565 | 67.860 | -1.140 | 0.554 | -0.379 | 0.546 |
| long_gated_ego | 3677 | 49.201 | 78.859 | 46.201 | 67.860 | -1.075 | 0.505 | -0.314 | 0.491 |
| long_gated_forget_ego | 3677 | 48.362 | 77.372 | 44.324 | 67.860 | -1.913 | 0.558 | -1.152 | 0.546 |

### Quick interpretation

- Overall best on matched common samples: `current_short_window`.
- Easy subset best: `long_mean_memory`.
- Hard subset best: `long_gated_forget_ego`.
- If a memory variant improves mainly on `cv_hard_top25` but not on `cv_easy_bottom25`, it supports the claim that memory helps when motion extrapolation is insufficient.

## Horizon 03s

- common samples: 11285
- easy threshold CV ADE <= 35.078
- hard threshold CV ADE >= 99.620

### all_common

Best ADE: `long_attention_no_ego` (63.254)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 11285 | 65.245 | 109.274 | 51.407 | 73.856 | 1.052 | 0.478 | 0.000 | 0.000 |
| episodic_short_only | 11285 | 64.194 | 107.932 | 50.769 | 73.856 | 0.000 | 0.000 | -1.052 | 0.522 |
| long_mean_memory | 11285 | 65.608 | 109.101 | 51.855 | 73.856 | 1.414 | 0.464 | 0.363 | 0.482 |
| long_attention_no_ego | 11285 | 63.254 | 107.004 | 50.705 | 73.856 | -0.940 | 0.494 | -1.992 | 0.517 |
| long_attention_ego | 11285 | 64.244 | 108.318 | 50.466 | 73.856 | 0.051 | 0.486 | -1.001 | 0.513 |
| long_gated_ego | 11285 | 63.520 | 106.916 | 49.809 | 73.856 | -0.674 | 0.496 | -1.725 | 0.522 |
| long_gated_forget_ego | 11285 | 64.286 | 109.208 | 50.700 | 73.856 | 0.092 | 0.497 | -0.960 | 0.519 |

### cv_easy_bottom25

Best ADE: `long_gated_ego` (27.667)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 2822 | 30.669 | 50.911 | 22.843 | 21.419 | -0.773 | 0.494 | 0.000 | 0.000 |
| episodic_short_only | 2822 | 31.443 | 51.621 | 23.694 | 21.419 | 0.000 | 0.000 | 0.773 | 0.506 |
| long_mean_memory | 2822 | 28.831 | 46.698 | 23.287 | 21.419 | -2.611 | 0.509 | -1.838 | 0.508 |
| long_attention_no_ego | 2822 | 27.994 | 45.730 | 22.087 | 21.419 | -3.449 | 0.560 | -2.676 | 0.556 |
| long_attention_ego | 2822 | 28.460 | 44.910 | 22.761 | 21.419 | -2.983 | 0.522 | -2.210 | 0.525 |
| long_gated_ego | 2822 | 27.667 | 44.656 | 21.726 | 21.419 | -3.775 | 0.542 | -3.002 | 0.542 |
| long_gated_forget_ego | 2822 | 30.896 | 50.911 | 23.521 | 21.419 | -0.546 | 0.502 | 0.227 | 0.502 |

### cv_hard_top25

Best ADE: `episodic_short_only` (108.299)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 2822 | 112.900 | 189.464 | 101.978 | 147.538 | 4.601 | 0.444 | 0.000 | 0.000 |
| episodic_short_only | 2822 | 108.299 | 184.367 | 96.860 | 147.538 | 0.000 | 0.000 | -4.601 | 0.556 |
| long_mean_memory | 2822 | 115.885 | 194.763 | 106.086 | 147.538 | 7.586 | 0.401 | 2.984 | 0.465 |
| long_attention_no_ego | 2822 | 109.663 | 186.771 | 98.113 | 147.538 | 1.364 | 0.446 | -3.238 | 0.516 |
| long_attention_ego | 2822 | 111.388 | 192.346 | 100.462 | 147.538 | 3.089 | 0.449 | -1.513 | 0.522 |
| long_gated_ego | 2822 | 109.449 | 187.278 | 96.750 | 147.538 | 1.150 | 0.477 | -3.452 | 0.530 |
| long_gated_forget_ego | 2822 | 108.672 | 187.482 | 96.529 | 147.538 | 0.373 | 0.490 | -4.228 | 0.556 |

### Quick interpretation

- Overall best on matched common samples: `long_attention_no_ego`.
- Easy subset best: `long_gated_ego`.
- Hard subset best: `episodic_short_only`.
- If a memory variant improves mainly on `cv_hard_top25` but not on `cv_easy_bottom25`, it supports the claim that memory helps when motion extrapolation is insufficient.

## Horizon 05s

- common samples: 9720
- easy threshold CV ADE <= 50.472
- hard threshold CV ADE >= 150.312

### all_common

Best ADE: `episodic_short_only` (86.695)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 9720 | 87.536 | 154.486 | 71.642 | 107.841 | 0.840 | 0.465 | 0.000 | 0.000 |
| episodic_short_only | 9720 | 86.695 | 153.714 | 70.489 | 107.841 | 0.000 | 0.000 | -0.840 | 0.535 |
| long_mean_memory | 9720 | 89.676 | 158.365 | 72.309 | 107.841 | 2.981 | 0.456 | 2.140 | 0.491 |
| long_attention_no_ego | 9720 | 87.037 | 153.437 | 70.620 | 107.841 | 0.342 | 0.490 | -0.499 | 0.517 |
| long_attention_ego | 9720 | 88.327 | 153.024 | 72.957 | 107.841 | 1.632 | 0.460 | 0.791 | 0.490 |
| long_gated_ego | 9720 | 88.418 | 155.643 | 71.246 | 107.841 | 1.722 | 0.480 | 0.882 | 0.503 |
| long_gated_forget_ego | 9720 | 87.621 | 154.469 | 69.999 | 107.841 | 0.926 | 0.483 | 0.085 | 0.514 |

### cv_easy_bottom25

Best ADE: `long_attention_ego` (28.417)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 2430 | 37.796 | 64.365 | 27.950 | 24.898 | 0.251 | 0.417 | 0.000 | 0.000 |
| episodic_short_only | 2430 | 37.545 | 66.102 | 27.112 | 24.898 | 0.000 | 0.000 | -0.251 | 0.583 |
| long_mean_memory | 2430 | 35.237 | 63.023 | 27.933 | 24.898 | -2.307 | 0.549 | -2.558 | 0.588 |
| long_attention_no_ego | 2430 | 35.958 | 62.715 | 28.889 | 24.898 | -1.587 | 0.526 | -1.838 | 0.574 |
| long_attention_ego | 2430 | 28.417 | 47.644 | 23.823 | 24.898 | -9.128 | 0.630 | -9.379 | 0.685 |
| long_gated_ego | 2430 | 42.962 | 74.981 | 34.430 | 24.898 | 5.417 | 0.407 | 5.166 | 0.474 |
| long_gated_forget_ego | 2430 | 41.706 | 76.368 | 32.009 | 24.898 | 4.161 | 0.445 | 3.910 | 0.495 |

### cv_hard_top25

Best ADE: `long_gated_ego` (145.778)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 2431 | 149.771 | 260.936 | 140.910 | 215.602 | 1.183 | 0.470 | 0.000 | 0.000 |
| episodic_short_only | 2431 | 148.588 | 263.247 | 138.285 | 215.602 | 0.000 | 0.000 | -1.183 | 0.530 |
| long_mean_memory | 2431 | 153.209 | 266.337 | 142.908 | 215.602 | 4.621 | 0.438 | 3.438 | 0.490 |
| long_attention_no_ego | 2431 | 149.214 | 261.802 | 135.472 | 215.602 | 0.626 | 0.501 | -0.557 | 0.526 |
| long_attention_ego | 2431 | 160.754 | 274.498 | 154.581 | 215.602 | 12.166 | 0.358 | 10.983 | 0.389 |
| long_gated_ego | 2431 | 145.778 | 256.346 | 131.261 | 215.602 | -2.811 | 0.542 | -3.994 | 0.547 |
| long_gated_forget_ego | 2431 | 146.183 | 254.199 | 133.592 | 215.602 | -2.405 | 0.517 | -3.588 | 0.556 |

### Quick interpretation

- Overall best on matched common samples: `episodic_short_only`.
- Easy subset best: `long_attention_ego`.
- Hard subset best: `long_gated_ego`.
- If a memory variant improves mainly on `cv_hard_top25` but not on `cv_easy_bottom25`, it supports the claim that memory helps when motion extrapolation is insufficient.

## Horizon 10s

- common samples: 7003
- easy threshold CV ADE <= 107.960
- hard threshold CV ADE >= 280.318

### all_common

Best ADE: `long_gated_ego` (158.058)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 7003 | 166.158 | 275.745 | 145.461 | 209.866 | 7.695 | 0.475 | 0.000 | 0.000 |
| episodic_short_only | 7003 | 158.463 | 263.649 | 137.662 | 209.866 | 0.000 | 0.000 | -7.695 | 0.525 |
| long_mean_memory | 7003 | 158.409 | 266.604 | 136.738 | 209.866 | -0.054 | 0.493 | -7.749 | 0.530 |
| long_attention_no_ego | 7003 | 158.394 | 267.533 | 134.658 | 209.866 | -0.069 | 0.506 | -7.764 | 0.566 |
| long_attention_ego | 7003 | 162.207 | 269.640 | 138.991 | 209.866 | 3.744 | 0.469 | -3.951 | 0.535 |
| long_gated_ego | 7003 | 158.058 | 266.766 | 135.090 | 209.866 | -0.406 | 0.490 | -8.100 | 0.541 |
| long_gated_forget_ego | 7003 | 161.669 | 265.679 | 139.327 | 209.866 | 3.205 | 0.448 | -4.489 | 0.513 |

### cv_easy_bottom25

Best ADE: `episodic_short_only` (71.296)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 1751 | 102.463 | 168.564 | 94.521 | 55.074 | 31.168 | 0.304 | 0.000 | 0.000 |
| episodic_short_only | 1751 | 71.296 | 127.587 | 57.592 | 55.074 | 0.000 | 0.000 | -31.168 | 0.696 |
| long_mean_memory | 1751 | 72.417 | 136.027 | 59.135 | 55.074 | 1.121 | 0.458 | -30.047 | 0.662 |
| long_attention_no_ego | 1751 | 90.203 | 167.103 | 79.579 | 55.074 | 18.907 | 0.306 | -12.261 | 0.600 |
| long_attention_ego | 1751 | 90.867 | 158.352 | 84.691 | 55.074 | 19.571 | 0.319 | -11.597 | 0.620 |
| long_gated_ego | 1751 | 79.896 | 141.271 | 68.476 | 55.074 | 8.600 | 0.336 | -22.568 | 0.638 |
| long_gated_forget_ego | 1751 | 77.960 | 133.256 | 66.102 | 55.074 | 6.665 | 0.371 | -24.503 | 0.650 |

### cv_hard_top25

Best ADE: `long_attention_no_ego` (237.463)

| Variant | Samples | ADE | FDE | Median ADE | CV ADE | ADE delta vs episodic | Win vs episodic | ADE delta vs current | Win vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | 1751 | 250.182 | 413.874 | 210.122 | 404.236 | 4.437 | 0.525 | 0.000 | 0.000 |
| episodic_short_only | 1751 | 245.746 | 402.387 | 216.039 | 404.236 | 0.000 | 0.000 | -4.437 | 0.475 |
| long_mean_memory | 1751 | 251.105 | 406.272 | 222.718 | 404.236 | 5.360 | 0.463 | 0.923 | 0.460 |
| long_attention_no_ego | 1751 | 237.463 | 392.856 | 197.962 | 404.236 | -8.282 | 0.584 | -12.719 | 0.584 |
| long_attention_ego | 1751 | 240.422 | 393.289 | 207.354 | 404.236 | -5.324 | 0.541 | -9.761 | 0.558 |
| long_gated_ego | 1751 | 244.949 | 403.135 | 212.773 | 404.236 | -0.797 | 0.508 | -5.233 | 0.515 |
| long_gated_forget_ego | 1751 | 252.056 | 411.003 | 220.589 | 404.236 | 6.310 | 0.434 | 1.874 | 0.481 |

### Quick interpretation

- Overall best on matched common samples: `long_gated_ego`.
- Easy subset best: `episodic_short_only`.
- Hard subset best: `long_attention_no_ego`.
- If a memory variant improves mainly on `cv_hard_top25` but not on `cv_easy_bottom25`, it supports the claim that memory helps when motion extrapolation is insufficient.
