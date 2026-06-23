# Episodic Long-Term Cue Memory Ablation

All trainable variants are retrained from scratch under the same v4 DINO-cache setup.
The main controls are `current_short_window` and `episodic_short_only`; improvements over the latter isolate the long-memory update beyond chunked training itself.

## Horizon 01s

| Variant | Complete | ADE | FDE | CV ADE | Best epoch | Gap | ADE vs current | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | yes | 27.773 | 43.065 | 33.112 | 11 | 8.986 | 0.00 | 0.62 |
| episodic_short_only | yes | 27.946 | 43.183 | 31.972 | 25 | 11.797 | -0.62 | 0.00 |
| long_mean_memory | yes | 28.020 | 43.061 | 31.972 | 16 | 7.798 | -0.89 | -0.27 |
| long_attention_no_ego | yes | 27.565 | 42.928 | 31.972 | 6 | 3.140 | 0.75 | 1.36 |
| long_attention_ego | yes | 27.889 | 42.643 | 31.972 | 43 | 15.382 | -0.42 | 0.21 |
| long_gated_ego | yes | 27.718 | 42.596 | 31.972 | 7 | 4.104 | 0.20 | 0.82 |
| long_gated_forget_ego | yes | 27.373 | 42.249 | 31.972 | 34 | 14.200 | 1.44 | 2.05 |

### cv_baseline_error_high

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 3844 | 50.931 | 81.334 | 70.430 | -0.87 |
| episodic_short_only | 3745 | 50.492 | 81.098 | 67.822 | 0.00 |
| long_mean_memory | 3745 | 52.201 | 82.573 | 67.822 | -3.39 |
| long_attention_no_ego | 3745 | 50.201 | 80.729 | 67.822 | 0.58 |
| long_attention_ego | 3745 | 49.293 | 78.341 | 67.822 | 2.38 |
| long_gated_ego | 3745 | 49.261 | 78.940 | 67.822 | 2.44 |
| long_gated_forget_ego | 3745 | 48.491 | 77.528 | 67.822 | 3.96 |

### high_curvature_path

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 3844 | 16.911 | 24.431 | 18.367 | -0.01 |
| episodic_short_only | 3745 | 16.909 | 23.999 | 17.993 | 0.00 |
| long_mean_memory | 3745 | 17.021 | 24.219 | 17.993 | -0.66 |
| long_attention_no_ego | 3745 | 17.033 | 24.591 | 17.993 | -0.73 |
| long_attention_ego | 3745 | 17.433 | 24.770 | 17.993 | -3.09 |
| long_gated_ego | 3745 | 17.629 | 24.734 | 17.993 | -4.26 |
| long_gated_forget_ego | 3745 | 17.413 | 24.892 | 17.993 | -2.98 |

### turn_scene

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 7553 | 32.204 | 51.457 | 38.451 | 1.65 |
| episodic_short_only | 7362 | 32.745 | 52.139 | 37.519 | 0.00 |
| long_mean_memory | 7362 | 32.934 | 51.958 | 37.519 | -0.58 |
| long_attention_no_ego | 7362 | 31.748 | 50.884 | 37.519 | 3.05 |
| long_attention_ego | 7362 | 32.494 | 51.627 | 37.519 | 0.77 |
| long_gated_ego | 7362 | 32.005 | 50.504 | 37.519 | 2.26 |
| long_gated_forget_ego | 7362 | 32.023 | 51.321 | 37.519 | 2.20 |

### front_blocked_or_obstacle_proxy

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 156 | 33.632 | 45.346 | 56.522 | -10.21 |
| episodic_short_only | 157 | 30.516 | 41.656 | 54.711 | 0.00 |
| long_mean_memory | 157 | 34.090 | 46.414 | 54.711 | -11.71 |
| long_attention_no_ego | 157 | 33.731 | 48.595 | 54.711 | -10.54 |
| long_attention_ego | 157 | 29.121 | 37.275 | 54.711 | 4.57 |
| long_gated_ego | 157 | 33.748 | 46.513 | 54.711 | -10.59 |
| long_gated_forget_ego | 157 | 31.816 | 43.130 | 54.711 | -4.26 |

### corridor_like_scene

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 1993 | 24.272 | 39.272 | 31.701 | -1.19 |
| episodic_short_only | 1947 | 23.987 | 39.158 | 28.749 | 0.00 |
| long_mean_memory | 1947 | 23.150 | 37.248 | 28.749 | 3.49 |
| long_attention_no_ego | 1947 | 22.936 | 37.848 | 28.749 | 4.38 |
| long_attention_ego | 1947 | 24.290 | 38.494 | 28.749 | -1.27 |
| long_gated_ego | 1947 | 23.217 | 37.729 | 28.749 | 3.21 |
| long_gated_forget_ego | 1947 | 22.649 | 36.437 | 28.749 | 5.58 |

## Horizon 03s

| Variant | Complete | ADE | FDE | CV ADE | Best epoch | Gap | ADE vs current | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | yes | 66.882 | 111.544 | 75.720 | 18 | 21.337 | 0.00 | -3.60 |
| episodic_short_only | yes | 64.561 | 108.573 | 73.894 | 22 | 22.094 | 3.47 | 0.00 |
| long_mean_memory | yes | 65.807 | 109.291 | 73.894 | 23 | 18.461 | 1.61 | -1.93 |
| long_attention_no_ego | yes | 63.458 | 107.426 | 73.894 | 20 | 20.578 | 5.12 | 1.71 |
| long_attention_ego | yes | 64.537 | 108.820 | 73.894 | 20 | 18.473 | 3.51 | 0.04 |
| long_gated_ego | yes | 63.856 | 107.492 | 73.894 | 15 | 17.619 | 4.52 | 1.09 |
| long_gated_forget_ego | yes | 64.627 | 109.809 | 73.894 | 25 | 23.669 | 3.37 | -0.10 |

### cv_baseline_error_high

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 2972 | 116.639 | 194.222 | 151.737 | -7.14 |
| episodic_short_only | 2883 | 108.869 | 185.670 | 147.423 | 0.00 |
| long_mean_memory | 2883 | 116.243 | 195.392 | 147.423 | -6.77 |
| long_attention_no_ego | 2883 | 110.128 | 187.717 | 147.423 | -1.16 |
| long_attention_ego | 2883 | 111.986 | 193.492 | 147.423 | -2.86 |
| long_gated_ego | 2883 | 110.026 | 188.433 | 147.423 | -1.06 |
| long_gated_forget_ego | 2883 | 109.293 | 188.781 | 147.423 | -0.39 |

### high_curvature_path

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 2972 | 40.331 | 61.676 | 45.043 | -1.36 |
| episodic_short_only | 2883 | 39.790 | 61.067 | 44.615 | 0.00 |
| long_mean_memory | 2883 | 40.403 | 62.292 | 44.615 | -1.54 |
| long_attention_no_ego | 2883 | 37.068 | 57.559 | 44.615 | 6.84 |
| long_attention_ego | 2883 | 38.619 | 58.533 | 44.615 | 2.94 |
| long_gated_ego | 2883 | 37.777 | 57.560 | 44.615 | 5.06 |
| long_gated_forget_ego | 2883 | 38.644 | 58.595 | 44.615 | 2.88 |

### turn_scene

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 5872 | 76.509 | 134.876 | 82.711 | -3.29 |
| episodic_short_only | 5699 | 74.073 | 131.874 | 81.137 | 0.00 |
| long_mean_memory | 5699 | 74.467 | 130.420 | 81.137 | -0.53 |
| long_attention_no_ego | 5699 | 72.782 | 130.198 | 81.137 | 1.74 |
| long_attention_ego | 5699 | 73.429 | 131.585 | 81.137 | 0.87 |
| long_gated_ego | 5699 | 73.404 | 131.146 | 81.137 | 0.90 |
| long_gated_forget_ego | 5699 | 72.756 | 130.257 | 81.137 | 1.78 |

### front_blocked_or_obstacle_proxy

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 175 | 73.420 | 104.869 | 132.958 | -6.47 |
| episodic_short_only | 184 | 68.958 | 98.291 | 129.650 | 0.00 |
| long_mean_memory | 184 | 77.534 | 108.497 | 129.650 | -12.44 |
| long_attention_no_ego | 184 | 67.608 | 97.605 | 129.650 | 1.96 |
| long_attention_ego | 184 | 72.183 | 104.475 | 129.650 | -4.68 |
| long_gated_ego | 184 | 71.034 | 103.339 | 129.650 | -3.01 |
| long_gated_forget_ego | 184 | 62.807 | 85.981 | 129.650 | 8.92 |

### corridor_like_scene

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 1370 | 68.846 | 113.971 | 88.278 | -7.82 |
| episodic_short_only | 1329 | 63.850 | 104.849 | 82.076 | 0.00 |
| long_mean_memory | 1329 | 66.905 | 109.720 | 82.076 | -4.78 |
| long_attention_no_ego | 1329 | 66.255 | 109.565 | 82.076 | -3.77 |
| long_attention_ego | 1329 | 66.356 | 110.147 | 82.076 | -3.93 |
| long_gated_ego | 1329 | 64.174 | 106.488 | 82.076 | -0.51 |
| long_gated_forget_ego | 1329 | 67.054 | 112.328 | 82.076 | -5.02 |

## Horizon 05s

| Variant | Complete | ADE | FDE | CV ADE | Best epoch | Gap | ADE vs current | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | yes | 90.129 | 158.468 | 111.267 | 11 | 22.744 | 0.00 | -4.08 |
| episodic_short_only | yes | 86.594 | 153.440 | 107.407 | 31 | 31.338 | 3.92 | 0.00 |
| long_mean_memory | yes | 89.567 | 158.106 | 107.407 | 30 | 29.796 | 0.62 | -3.43 |
| long_attention_no_ego | yes | 87.070 | 153.321 | 107.407 | 30 | 30.189 | 3.39 | -0.55 |
| long_attention_ego | yes | 88.094 | 152.509 | 107.407 | 7 | 3.988 | 2.26 | -1.73 |
| long_gated_ego | yes | 88.405 | 155.486 | 107.407 | 33 | 36.661 | 1.91 | -2.09 |
| long_gated_forget_ego | yes | 87.577 | 154.297 | 107.407 | 34 | 35.311 | 2.83 | -1.13 |

### cv_baseline_error_high

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 2574 | 154.554 | 268.381 | 222.323 | -4.05 |
| episodic_short_only | 2483 | 148.535 | 262.835 | 215.085 | 0.00 |
| long_mean_memory | 2483 | 153.290 | 266.084 | 215.085 | -3.20 |
| long_attention_no_ego | 2483 | 149.347 | 261.336 | 215.085 | -0.55 |
| long_attention_ego | 2483 | 160.523 | 273.718 | 215.085 | -8.07 |
| long_gated_ego | 2483 | 145.665 | 255.523 | 215.085 | 1.93 |
| long_gated_forget_ego | 2483 | 146.189 | 253.741 | 215.085 | 1.58 |

### high_curvature_path

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 2574 | 53.553 | 88.108 | 75.544 | -3.84 |
| episodic_short_only | 2483 | 51.572 | 84.057 | 73.578 | 0.00 |
| long_mean_memory | 2483 | 55.666 | 91.863 | 73.578 | -7.94 |
| long_attention_no_ego | 2483 | 51.415 | 84.458 | 73.578 | 0.30 |
| long_attention_ego | 2483 | 50.375 | 78.612 | 73.578 | 2.32 |
| long_gated_ego | 2483 | 55.830 | 92.965 | 73.578 | -8.26 |
| long_gated_forget_ego | 2483 | 53.412 | 89.698 | 73.578 | -3.57 |

### turn_scene

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 5068 | 103.174 | 193.044 | 126.131 | -4.04 |
| episodic_short_only | 4889 | 99.164 | 188.713 | 122.520 | 0.00 |
| long_mean_memory | 4889 | 104.116 | 194.609 | 122.520 | -4.99 |
| long_attention_no_ego | 4889 | 100.531 | 189.242 | 122.520 | -1.38 |
| long_attention_ego | 4889 | 103.070 | 190.216 | 122.520 | -3.94 |
| long_gated_ego | 4889 | 101.340 | 188.816 | 122.520 | -2.19 |
| long_gated_forget_ego | 4889 | 100.034 | 187.133 | 122.520 | -0.88 |

### front_blocked_or_obstacle_proxy

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 235 | 95.917 | 149.100 | 205.082 | -5.34 |
| episodic_short_only | 239 | 91.052 | 141.213 | 198.057 | 0.00 |
| long_mean_memory | 239 | 101.375 | 164.077 | 198.057 | -11.34 |
| long_attention_no_ego | 239 | 90.515 | 144.457 | 198.057 | 0.59 |
| long_attention_ego | 239 | 99.401 | 153.841 | 198.057 | -9.17 |
| long_gated_ego | 239 | 85.504 | 137.850 | 198.057 | 6.09 |
| long_gated_forget_ego | 239 | 85.699 | 134.066 | 198.057 | 5.88 |

### corridor_like_scene

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 1271 | 90.454 | 145.276 | 113.162 | -4.32 |
| episodic_short_only | 1237 | 86.706 | 142.371 | 107.298 | 0.00 |
| long_mean_memory | 1237 | 85.505 | 138.231 | 107.298 | 1.38 |
| long_attention_no_ego | 1237 | 85.990 | 137.627 | 107.298 | 0.83 |
| long_attention_ego | 1237 | 84.201 | 131.228 | 107.298 | 2.89 |
| long_gated_ego | 1237 | 86.137 | 141.307 | 107.298 | 0.66 |
| long_gated_forget_ego | 1237 | 90.014 | 148.428 | 107.298 | -3.82 |

## Horizon 10s

| Variant | Complete | ADE | FDE | CV ADE | Best epoch | Gap | ADE vs current | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_short_window | yes | 169.190 | 280.091 | 217.167 | 19 | 56.008 | 0.00 | -6.12 |
| episodic_short_only | yes | 159.437 | 265.095 | 210.728 | 11 | 28.136 | 5.76 | 0.00 |
| long_mean_memory | yes | 159.045 | 267.350 | 210.728 | 16 | 38.310 | 6.00 | 0.25 |
| long_attention_no_ego | yes | 159.164 | 268.978 | 210.728 | 35 | 65.208 | 5.93 | 0.17 |
| long_attention_ego | yes | 163.208 | 271.314 | 210.728 | 21 | 50.016 | 3.54 | -2.37 |
| long_gated_ego | yes | 159.048 | 268.344 | 210.728 | 17 | 42.778 | 5.99 | 0.24 |
| long_gated_forget_ego | yes | 162.646 | 267.250 | 210.728 | 9 | 30.797 | 3.87 | -2.01 |

### cv_baseline_error_high

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 1859 | 258.275 | 424.352 | 420.636 | -3.94 |
| episodic_short_only | 1801 | 248.493 | 406.020 | 406.147 | 0.00 |
| long_mean_memory | 1801 | 252.868 | 407.877 | 406.147 | -1.76 |
| long_attention_no_ego | 1801 | 239.839 | 396.836 | 406.147 | 3.48 |
| long_attention_ego | 1801 | 243.092 | 397.251 | 406.147 | 2.17 |
| long_gated_ego | 1801 | 247.461 | 406.361 | 406.147 | 0.42 |
| long_gated_forget_ego | 1801 | 254.660 | 414.705 | 406.147 | -2.48 |

### high_curvature_path

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 1859 | 116.622 | 188.955 | 154.601 | -10.16 |
| episodic_short_only | 1801 | 105.866 | 173.732 | 148.887 | 0.00 |
| long_mean_memory | 1801 | 104.975 | 175.405 | 148.887 | 0.84 |
| long_attention_no_ego | 1801 | 114.630 | 192.809 | 148.887 | -8.28 |
| long_attention_ego | 1801 | 116.280 | 187.416 | 148.887 | -9.84 |
| long_gated_ego | 1801 | 108.040 | 180.945 | 148.887 | -2.05 |
| long_gated_forget_ego | 1801 | 109.165 | 176.878 | 148.887 | -3.12 |

### turn_scene

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 3596 | 189.490 | 321.185 | 223.636 | -3.80 |
| episodic_short_only | 3493 | 182.553 | 307.035 | 216.161 | 0.00 |
| long_mean_memory | 3493 | 183.257 | 309.400 | 216.161 | -0.39 |
| long_attention_no_ego | 3493 | 182.826 | 311.631 | 216.161 | -0.15 |
| long_attention_ego | 3493 | 185.678 | 312.566 | 216.161 | -1.71 |
| long_gated_ego | 3493 | 183.819 | 314.740 | 216.161 | -0.69 |
| long_gated_forget_ego | 3493 | 186.414 | 309.819 | 216.161 | -2.12 |

### front_blocked_or_obstacle_proxy

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 200 | 181.501 | 296.332 | 379.294 | -4.13 |
| episodic_short_only | 194 | 174.310 | 301.033 | 361.455 | 0.00 |
| long_mean_memory | 194 | 176.567 | 287.440 | 361.455 | -1.30 |
| long_attention_no_ego | 194 | 162.340 | 287.665 | 361.455 | 6.87 |
| long_attention_ego | 194 | 191.983 | 330.175 | 361.455 | -10.14 |
| long_gated_ego | 194 | 184.553 | 315.095 | 361.455 | -5.88 |
| long_gated_forget_ego | 194 | 195.554 | 342.617 | 361.455 | -12.19 |

### corridor_like_scene

| Variant | Samples | ADE | FDE | CV ADE | ADE vs episodic short |
|---|---:|---:|---:|---:|---:|
| current_short_window | 1093 | 159.882 | 258.046 | 250.443 | -16.02 |
| episodic_short_only | 1059 | 137.802 | 225.759 | 240.557 | 0.00 |
| long_mean_memory | 1059 | 134.526 | 222.991 | 240.557 | 2.38 |
| long_attention_no_ego | 1059 | 136.681 | 226.541 | 240.557 | 0.81 |
| long_attention_ego | 1059 | 144.529 | 230.140 | 240.557 | -4.88 |
| long_gated_ego | 1059 | 135.164 | 219.528 | 240.557 | 1.91 |
| long_gated_forget_ego | 1059 | 140.576 | 225.699 | 240.557 | -2.01 |
