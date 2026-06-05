# Graph Subset Ablation v4 10s Prefix Evaluation

Compares spatial-relation modules with the same v4 cached DINOv3 setup.

Train horizon: 10s. Metrics for 1s/3s/5s/10s are computed by slicing the same 10s prediction.

Variants: no_graph, topk_graph, relpos_graph, contrast_graph, local_topk_graph, relpos_contrast_local_graph.

## Prefix 01s

| variant | status | N | ADE | FDE | CV ADE | best_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | complete | 7434 | 38.4808 | 53.8339 | 33.0348 | 21 |
| topk_graph | complete | 7434 | 32.2773 | 47.2246 | 33.0348 | 51 |
| relpos_graph | complete | 7434 | 36.7874 | 51.8258 | 33.0348 | 46 |
| contrast_graph | complete | 7434 | 36.6996 | 52.9787 | 33.0348 | 46 |
| local_topk_graph | complete | 7434 | 34.7949 | 50.7310 | 33.0348 | 44 |
| relpos_contrast_local_graph | complete | 7434 | 34.6756 | 48.9218 | 33.0348 | 61 |

### high_curvature_path

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1859 | 31.4805 | 40.0164 | 19.5500 | 0.0000 | -32.2204 |
| topk_graph | 1859 | 23.8091 | 31.9269 | 19.5500 | 24.3687 | 0.0000 |
| relpos_graph | 1859 | 29.1579 | 37.9547 | 19.5500 | 7.3780 | -22.4652 |
| contrast_graph | 1859 | 29.4659 | 39.7199 | 19.5500 | 6.3994 | -23.7591 |
| local_topk_graph | 1859 | 27.4141 | 37.3474 | 19.5500 | 12.9173 | -15.1411 |
| relpos_contrast_local_graph | 1859 | 27.9804 | 35.8932 | 19.5500 | 11.1182 | -17.5200 |

### turn_scene

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 3637 | 44.2839 | 64.6339 | 38.9894 | 0.0000 | -19.8891 |
| topk_graph | 3637 | 36.9374 | 55.5122 | 38.9894 | 16.5896 | 0.0000 |
| relpos_graph | 3637 | 41.8593 | 61.3781 | 38.9894 | 5.4753 | -13.3248 |
| contrast_graph | 3637 | 40.9658 | 60.7632 | 38.9894 | 7.4929 | -10.9060 |
| local_topk_graph | 3637 | 39.4232 | 59.2684 | 38.9894 | 10.9762 | -6.7299 |
| relpos_contrast_local_graph | 3637 | 39.6425 | 58.1523 | 38.9894 | 10.4811 | -7.3234 |

### cv_baseline_error_high

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1859 | 57.7048 | 88.3990 | 68.7355 | 0.0000 | -7.5729 |
| topk_graph | 1859 | 53.6425 | 82.8259 | 68.7355 | 7.0398 | 0.0000 |
| relpos_graph | 1859 | 54.4723 | 83.3475 | 68.7355 | 5.6018 | -1.5469 |
| contrast_graph | 1859 | 54.3439 | 83.5043 | 68.7355 | 5.8243 | -1.3076 |
| local_topk_graph | 1859 | 53.1033 | 80.5055 | 68.7355 | 7.9741 | 1.0051 |
| relpos_contrast_local_graph | 1859 | 53.0707 | 80.4149 | 68.7355 | 8.0306 | 1.0658 |

### left_right_asymmetric_layout

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 2188 | 54.5933 | 83.7507 | 55.3155 | 0.0000 | -13.3699 |
| topk_graph | 2188 | 48.1550 | 74.9265 | 55.3155 | 11.7931 | 0.0000 |
| relpos_graph | 2188 | 52.0713 | 79.4205 | 55.3155 | 4.6195 | -8.1327 |
| contrast_graph | 2188 | 50.2293 | 76.9714 | 55.3155 | 7.9936 | -4.3075 |
| local_topk_graph | 2188 | 49.2422 | 76.4852 | 55.3155 | 9.8017 | -2.2577 |
| relpos_contrast_local_graph | 2188 | 48.9127 | 75.4600 | 55.3155 | 10.4052 | -1.5735 |

### corridor_like_scene

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 958 | 32.4813 | 44.4121 | 28.4159 | 0.0000 | -16.2797 |
| topk_graph | 958 | 27.9338 | 41.4994 | 28.4159 | 14.0005 | 0.0000 |
| relpos_graph | 958 | 29.5883 | 40.1782 | 28.4159 | 8.9066 | -5.9231 |
| contrast_graph | 958 | 30.8529 | 43.9481 | 28.4159 | 5.0132 | -10.4503 |
| local_topk_graph | 958 | 28.0327 | 41.0880 | 28.4159 | 13.6960 | -0.3540 |
| relpos_contrast_local_graph | 958 | 29.7041 | 41.2258 | 28.4159 | 8.5501 | -6.3376 |

### front_blocked_or_obstacle_proxy

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 78 | 46.9865 | 57.7730 | 53.9170 | 0.0000 | -3.3280 |
| topk_graph | 78 | 45.4731 | 59.0754 | 53.9170 | 3.2208 | 0.0000 |
| relpos_graph | 78 | 43.2647 | 55.3600 | 53.9170 | 7.9209 | 4.8565 |
| contrast_graph | 78 | 46.0155 | 59.3640 | 53.9170 | 2.0666 | -1.1926 |
| local_topk_graph | 78 | 43.4100 | 49.1332 | 53.9170 | 7.6118 | 4.5371 |
| relpos_contrast_local_graph | 78 | 45.8239 | 56.4417 | 53.9170 | 2.4743 | -0.7714 |

## Prefix 03s

| variant | status | N | ADE | FDE | CV ADE | best_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | complete | 7434 | 70.6455 | 113.6350 | 76.1686 | 21 |
| topk_graph | complete | 7434 | 66.1293 | 110.6484 | 76.1686 | 51 |
| relpos_graph | complete | 7434 | 67.5253 | 109.0093 | 76.1686 | 46 |
| contrast_graph | complete | 7434 | 69.4383 | 112.2044 | 76.1686 | 46 |
| local_topk_graph | complete | 7434 | 68.5594 | 111.2445 | 76.1686 | 44 |
| relpos_contrast_local_graph | complete | 7434 | 66.3595 | 108.3127 | 76.1686 | 61 |

### high_curvature_path

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1859 | 50.0023 | 74.8030 | 47.4450 | 0.0000 | -10.0085 |
| topk_graph | 1859 | 45.4531 | 72.2087 | 47.4450 | 9.0979 | 0.0000 |
| relpos_graph | 1859 | 48.6837 | 74.1082 | 47.4450 | 2.6371 | -7.1074 |
| contrast_graph | 1859 | 49.5805 | 75.1160 | 47.4450 | 0.8435 | -9.0806 |
| local_topk_graph | 1859 | 49.0629 | 75.4879 | 47.4450 | 1.8786 | -7.9418 |
| relpos_contrast_local_graph | 1859 | 47.5908 | 73.4181 | 47.4450 | 4.8228 | -4.7030 |

### turn_scene

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 3651 | 80.7663 | 136.8627 | 87.8451 | 0.0000 | -7.8575 |
| topk_graph | 3651 | 74.8824 | 133.6259 | 87.8451 | 7.2850 | 0.0000 |
| relpos_graph | 3651 | 77.1970 | 133.1538 | 87.8451 | 4.4193 | -3.0909 |
| contrast_graph | 3651 | 76.8723 | 131.2089 | 87.8451 | 4.8214 | -2.6573 |
| local_topk_graph | 3651 | 76.2266 | 130.9396 | 87.8451 | 5.6208 | -1.7950 |
| relpos_contrast_local_graph | 3651 | 74.8734 | 129.8379 | 87.8451 | 7.2963 | 0.0121 |

### cv_baseline_error_high

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1859 | 116.3277 | 186.4971 | 156.5978 | 0.0000 | -6.1832 |
| topk_graph | 1859 | 109.5538 | 185.1903 | 156.5978 | 5.8231 | 0.0000 |
| relpos_graph | 1859 | 108.1546 | 178.3294 | 156.5978 | 7.0259 | 1.2771 |
| contrast_graph | 1859 | 111.2765 | 181.1933 | 156.5978 | 4.3422 | -1.5725 |
| local_topk_graph | 1859 | 109.1159 | 178.7113 | 156.5978 | 6.1995 | 0.3997 |
| relpos_contrast_local_graph | 1859 | 104.9549 | 173.4503 | 156.5978 | 9.7765 | 4.1978 |

### left_right_asymmetric_layout

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 2322 | 106.0229 | 183.6788 | 120.3533 | 0.0000 | -6.0470 |
| topk_graph | 2322 | 99.9773 | 180.5223 | 120.3533 | 5.7022 | 0.0000 |
| relpos_graph | 2322 | 100.9675 | 177.1603 | 120.3533 | 4.7683 | -0.9904 |
| contrast_graph | 2322 | 99.6116 | 173.3066 | 120.3533 | 6.0471 | 0.3658 |
| local_topk_graph | 2322 | 98.6032 | 172.1541 | 120.3533 | 6.9982 | 1.3744 |
| relpos_contrast_local_graph | 2322 | 97.9110 | 172.0656 | 120.3533 | 7.6511 | 2.0668 |

### corridor_like_scene

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1041 | 77.8436 | 107.1773 | 89.5225 | 0.0000 | -6.2256 |
| topk_graph | 1041 | 73.2814 | 109.2597 | 89.5225 | 5.8608 | 0.0000 |
| relpos_graph | 1041 | 72.5815 | 104.7913 | 89.5225 | 6.7599 | 0.9551 |
| contrast_graph | 1041 | 76.2719 | 111.9356 | 89.5225 | 2.0191 | -4.0809 |
| local_topk_graph | 1041 | 76.5088 | 110.9724 | 89.5225 | 1.7148 | -4.4041 |
| relpos_contrast_local_graph | 1041 | 72.1052 | 103.4687 | 89.5225 | 7.3718 | 1.6051 |

### front_blocked_or_obstacle_proxy

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 124 | 78.7136 | 102.3595 | 119.8509 | 0.0000 | -7.0867 |
| topk_graph | 124 | 73.5045 | 105.8748 | 119.8509 | 6.6177 | 0.0000 |
| relpos_graph | 124 | 69.3136 | 93.8295 | 119.8509 | 11.9420 | 5.7017 |
| contrast_graph | 124 | 78.4090 | 97.6036 | 119.8509 | 0.3869 | -6.6723 |
| local_topk_graph | 124 | 68.5636 | 93.1918 | 119.8509 | 12.8948 | 6.7219 |
| relpos_contrast_local_graph | 124 | 65.7932 | 90.1320 | 119.8509 | 16.4145 | 10.4910 |

## Prefix 05s

| variant | status | N | ADE | FDE | CV ADE | best_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | complete | 7434 | 101.1153 | 173.8761 | 116.3795 | 21 |
| topk_graph | complete | 7434 | 96.6943 | 166.9312 | 116.3795 | 51 |
| relpos_graph | complete | 7434 | 97.5282 | 169.4531 | 116.3795 | 46 |
| contrast_graph | complete | 7434 | 99.8654 | 172.1716 | 116.3795 | 46 |
| local_topk_graph | complete | 7434 | 97.7953 | 168.4063 | 116.3795 | 44 |
| relpos_contrast_local_graph | complete | 7434 | 96.1196 | 168.2712 | 116.3795 | 61 |

### high_curvature_path

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1859 | 68.4213 | 111.0930 | 76.6073 | 0.0000 | -6.8515 |
| topk_graph | 1859 | 64.0340 | 105.5911 | 76.6073 | 6.4122 | 0.0000 |
| relpos_graph | 1859 | 67.6588 | 113.5680 | 76.6073 | 1.1144 | -5.6608 |
| contrast_graph | 1859 | 71.7976 | 117.2372 | 76.6073 | -4.9346 | -12.1242 |
| local_topk_graph | 1859 | 67.8149 | 109.8236 | 76.6073 | 0.8863 | -5.9045 |
| relpos_contrast_local_graph | 1859 | 66.3134 | 111.8964 | 76.6073 | 3.0807 | -3.5598 |

### turn_scene

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 3625 | 116.3808 | 212.7688 | 130.7631 | 0.0000 | -3.9324 |
| topk_graph | 3625 | 111.9774 | 207.4023 | 130.7631 | 3.7836 | 0.0000 |
| relpos_graph | 3625 | 112.7543 | 208.3839 | 130.7631 | 3.1160 | -0.6938 |
| contrast_graph | 3625 | 113.8920 | 208.8577 | 130.7631 | 2.1384 | -1.7099 |
| local_topk_graph | 3625 | 109.9622 | 203.2191 | 130.7631 | 5.5152 | 1.7996 |
| relpos_contrast_local_graph | 3625 | 110.6111 | 206.4820 | 130.7631 | 4.9576 | 1.2201 |

### cv_baseline_error_high

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1859 | 166.4863 | 283.4944 | 233.2185 | 0.0000 | -2.9594 |
| topk_graph | 1859 | 161.7010 | 280.5818 | 233.2185 | 2.8743 | 0.0000 |
| relpos_graph | 1859 | 158.9168 | 274.5368 | 233.2185 | 4.5466 | 1.7218 |
| contrast_graph | 1859 | 161.6583 | 277.0877 | 233.2185 | 2.8999 | 0.0264 |
| local_topk_graph | 1859 | 157.3319 | 272.5207 | 233.2185 | 5.4986 | 2.7019 |
| relpos_contrast_local_graph | 1859 | 155.6804 | 272.3107 | 233.2185 | 6.4906 | 3.7233 |

### left_right_asymmetric_layout

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 2316 | 153.6552 | 285.7054 | 171.9782 | 0.0000 | -2.5476 |
| topk_graph | 2316 | 149.8380 | 281.6996 | 171.9782 | 2.4843 | 0.0000 |
| relpos_graph | 2316 | 147.8319 | 276.9172 | 171.9782 | 3.7898 | 1.3388 |
| contrast_graph | 2316 | 145.6695 | 273.5919 | 171.9782 | 5.1971 | 2.7820 |
| local_topk_graph | 2316 | 142.9408 | 270.8728 | 171.9782 | 6.9730 | 4.6031 |
| relpos_contrast_local_graph | 2316 | 144.5615 | 273.7642 | 171.9782 | 5.9183 | 3.5215 |

### corridor_like_scene

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1019 | 95.0338 | 141.1548 | 129.6660 | 0.0000 | -6.5137 |
| topk_graph | 1019 | 89.2222 | 134.9942 | 129.6660 | 6.1154 | 0.0000 |
| relpos_graph | 1019 | 90.0115 | 136.1151 | 129.6660 | 5.2847 | -0.8847 |
| contrast_graph | 1019 | 92.2235 | 139.7649 | 129.6660 | 2.9572 | -3.3639 |
| local_topk_graph | 1019 | 93.3762 | 143.1163 | 129.6660 | 1.7442 | -4.6559 |
| relpos_contrast_local_graph | 1019 | 89.8854 | 137.8799 | 129.6660 | 5.4175 | -0.7434 |

### front_blocked_or_obstacle_proxy

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 149 | 94.9014 | 143.6993 | 193.9694 | 0.0000 | 3.2358 |
| topk_graph | 149 | 98.0748 | 152.1059 | 193.9694 | -3.3440 | 0.0000 |
| relpos_graph | 149 | 94.5387 | 148.9497 | 193.9694 | 0.3821 | 3.6055 |
| contrast_graph | 149 | 100.3315 | 146.6857 | 193.9694 | -5.7219 | -2.3010 |
| local_topk_graph | 149 | 92.9836 | 140.4745 | 193.9694 | 2.0208 | 5.1912 |
| relpos_contrast_local_graph | 149 | 93.2395 | 149.4976 | 193.9694 | 1.7512 | 4.9303 |

## Prefix 10s

| variant | status | N | ADE | FDE | CV ADE | best_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | complete | 7434 | 167.3324 | 277.6008 | 217.1669 | 21 |
| topk_graph | complete | 7434 | 159.7269 | 263.4599 | 217.1669 | 51 |
| relpos_graph | complete | 7434 | 163.2365 | 270.5631 | 217.1669 | 46 |
| contrast_graph | complete | 7434 | 165.2452 | 271.8678 | 217.1669 | 46 |
| local_topk_graph | complete | 7434 | 163.7404 | 273.4510 | 217.1669 | 44 |
| relpos_contrast_local_graph | complete | 7434 | 164.7210 | 279.1571 | 217.1669 | 61 |

### high_curvature_path

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1859 | 119.6885 | 191.6841 | 154.6012 | 0.0000 | -7.4319 |
| topk_graph | 1859 | 111.4087 | 176.9102 | 154.6012 | 6.9178 | 0.0000 |
| relpos_graph | 1859 | 120.0329 | 195.1914 | 154.6012 | -0.2878 | -7.7411 |
| contrast_graph | 1859 | 124.5120 | 199.5016 | 154.6012 | -4.0301 | -11.7615 |
| local_topk_graph | 1859 | 116.8342 | 189.7747 | 154.6012 | 2.3848 | -4.8699 |
| relpos_contrast_local_graph | 1859 | 121.6507 | 205.1790 | 154.6012 | -1.6395 | -9.1932 |

### turn_scene

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 3596 | 189.7722 | 318.9143 | 223.6355 | 0.0000 | -3.7677 |
| topk_graph | 3596 | 182.8817 | 306.4601 | 223.6355 | 3.6309 | 0.0000 |
| relpos_graph | 3596 | 187.2157 | 315.9064 | 223.6355 | 1.3471 | -2.3698 |
| contrast_graph | 3596 | 185.6597 | 309.7239 | 223.6355 | 2.1671 | -1.5190 |
| local_topk_graph | 3596 | 182.6923 | 313.2872 | 223.6355 | 3.7308 | 0.1036 |
| relpos_contrast_local_graph | 3596 | 185.9706 | 321.4456 | 223.6355 | 2.0033 | -1.6890 |

### cv_baseline_error_high

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1859 | 255.5405 | 416.8501 | 420.6365 | 0.0000 | -1.6890 |
| topk_graph | 1859 | 251.2961 | 406.3191 | 420.6365 | 1.6610 | 0.0000 |
| relpos_graph | 1859 | 252.0484 | 407.6643 | 420.6365 | 1.3666 | -0.2994 |
| contrast_graph | 1859 | 250.2865 | 401.2192 | 420.6365 | 2.0560 | 0.4018 |
| local_topk_graph | 1859 | 249.0534 | 405.2693 | 420.6365 | 2.5386 | 0.8924 |
| relpos_contrast_local_graph | 1859 | 249.0317 | 410.8231 | 420.6365 | 2.5471 | 0.9011 |

### left_right_asymmetric_layout

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 2357 | 246.7760 | 421.9072 | 282.3569 | 0.0000 | -2.3480 |
| topk_graph | 2357 | 241.1147 | 411.8899 | 282.3569 | 2.2941 | 0.0000 |
| relpos_graph | 2357 | 240.7784 | 410.4568 | 282.3569 | 2.4304 | 0.1394 |
| contrast_graph | 2357 | 234.5005 | 396.8908 | 282.3569 | 4.9743 | 2.7431 |
| local_topk_graph | 2357 | 236.5762 | 412.1598 | 282.3569 | 4.1332 | 1.8823 |
| relpos_contrast_local_graph | 2357 | 236.6992 | 410.6948 | 282.3569 | 4.0834 | 1.8313 |

### corridor_like_scene

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 1200 | 163.0326 | 237.6796 | 258.5344 | 0.0000 | -4.8208 |
| topk_graph | 1200 | 155.5345 | 231.6370 | 258.5344 | 4.5991 | 0.0000 |
| relpos_graph | 1200 | 162.6907 | 241.5864 | 258.5344 | 0.2097 | -4.6010 |
| contrast_graph | 1200 | 158.7448 | 237.0126 | 258.5344 | 2.6300 | -2.0640 |
| local_topk_graph | 1200 | 166.9648 | 247.4553 | 258.5344 | -2.4120 | -7.3490 |
| relpos_contrast_local_graph | 1200 | 162.5783 | 249.2789 | 258.5344 | 0.2786 | -4.5287 |

### front_blocked_or_obstacle_proxy

| variant | N | ADE | FDE | CV ADE | ADE imp vs no_graph % | ADE imp vs topk % |
| --- | --- | --- | --- | --- | --- | --- |
| no_graph | 200 | 178.0242 | 295.7110 | 379.2938 | 0.0000 | -3.0242 |
| topk_graph | 200 | 172.7984 | 279.6860 | 379.2938 | 2.9355 | 0.0000 |
| relpos_graph | 200 | 179.3992 | 298.8637 | 379.2938 | -0.7724 | -3.8200 |
| contrast_graph | 200 | 179.0348 | 295.1902 | 379.2938 | -0.5677 | -3.6091 |
| local_topk_graph | 200 | 168.8567 | 275.9015 | 379.2938 | 5.1496 | 2.2811 |
| relpos_contrast_local_graph | 200 | 171.3253 | 283.3151 | 379.2938 | 3.7629 | 0.8525 |
