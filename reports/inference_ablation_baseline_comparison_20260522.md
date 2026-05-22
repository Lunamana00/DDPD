# Paper-Proxy Baseline And Ablation Inference Comparison

Date: 2026-05-22

## Scope

- Re-ran metric-only inference locally on the available v2 horizon-sweep test splits.
- Paper baselines requested: Khaleque et al. (2024) exploratory agents and Xu et al. (2026) pixels-only navigation.
- The exact paper systems are interactive agents, not offline trajectory regressors. This report therefore includes paper-inspired offline proxies plus internal ablations.
- Metrics are ADE/FDE in local egocentric coordinates; lower is better.
- Device used for this rerun: `cpu`.

## Baseline Adaptation

| Paper baseline | Offline proxy used here | What is missing vs the paper |
| --- | --- | --- |
| Khaleque, Cook, & Gow (2024) | Center-biased exploratory context-steering proxy that picks a deterministic random direction inside a 135 degree sector and biases it toward the train-set source center every 2 seconds. | The original uses level/object motivation metrics and context steering inside an interactive environment. Our processed samples do not include object/light annotations or interactive rollout state. |
| Xu et al. (2026) | Screen-only saliency controller that reads the last RGB frame, chooses a salient horizontal interest point, and rolls out a fixed-speed local path. | The original builds on a visual affordance detector and finite-state controller in a live commercial 3D ARPG. We do not have that detector or live ARPG environment here. |

## Re-run Results

| Horizon | Model | Test samples | ADE | FDE | ADE gain vs best paper proxy | FDE gain vs best paper proxy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1s | Internal motion-only CV ablation | 2405 | 27.3805 | 43.7101 | 12.4% | 12.2% |
| 1s | Internal DINOv3 STRNet-style ablation | 2405 | 19.7297 | 30.8009 | 36.9% | 38.1% |
| 1s | Ours: DINOv3 TimeSFormer | 2405 | 18.3886 | 28.9105 | 41.2% | 41.9% |
| 1s | Khaleque-style center-biased exploratory proxy | 2405 | 39.7888 | 63.2691 | -27.3% | -27.1% |
| 1s | Internal small-CNN TimeSFormer ablation | 2405 | 23.1751 | 36.0909 | 25.8% | 27.5% |
| 1s | Xu-style pixels-only saliency proxy | 2405 | 31.2498 | 49.7632 | 0.0% | 0.0% |
| 3s | Internal motion-only CV ablation | 1836 | 63.8882 | 116.0432 | 13.0% | 13.6% |
| 3s | Internal DINOv3 STRNet-style ablation | 1836 | 48.0674 | 88.3438 | 34.5% | 34.2% |
| 3s | Ours: DINOv3 TimeSFormer | 1836 | 50.7555 | 90.7791 | 30.8% | 32.4% |
| 3s | Khaleque-style center-biased exploratory proxy | 1836 | 93.8668 | 157.8634 | -27.9% | -17.5% |
| 3s | Internal small-CNN TimeSFormer ablation | 1836 | 55.5294 | 102.7056 | 24.3% | 23.6% |
| 3s | Xu-style pixels-only saliency proxy | 1836 | 73.3959 | 134.3571 | 0.0% | 0.0% |
| 5s | Internal motion-only CV ablation | 1407 | 106.3786 | 197.4482 | 13.3% | 13.2% |
| 5s | Internal DINOv3 STRNet-style ablation | 1407 | 77.7281 | 130.8785 | 36.7% | 42.5% |
| 5s | Ours: DINOv3 TimeSFormer | 1407 | 77.5128 | 135.7005 | 36.8% | 40.3% |
| 5s | Khaleque-style center-biased exploratory proxy | 1407 | 152.5831 | 240.8973 | -24.4% | -5.9% |
| 5s | Internal small-CNN TimeSFormer ablation | 1407 | 88.5169 | 155.9891 | 27.9% | 31.4% |
| 5s | Xu-style pixels-only saliency proxy | 1407 | 122.7029 | 227.4763 | 0.0% | 0.0% |
| 10s | Internal motion-only CV ablation | 1576 | 145.3607 | 281.9835 | 3.4% | -14.8% |
| 10s | Internal DINOv3 STRNet-style ablation | 1576 | 112.9948 | 191.7430 | 24.9% | 22.0% |
| 10s | Ours: DINOv3 TimeSFormer | 1576 | 110.4049 | 205.4597 | 26.6% | 16.4% |
| 10s | Khaleque-style center-biased exploratory proxy | 1576 | 150.4949 | 245.6877 | 0.0% | 0.0% |
| 10s | Internal small-CNN TimeSFormer ablation | 1576 | 104.8426 | 199.1186 | 30.3% | 19.0% |
| 10s | Xu-style pixels-only saliency proxy | 1576 | 172.2527 | 337.6668 | -14.5% | -37.4% |
| 30s | Internal motion-only CV ablation | 773 | 391.1181 | 694.2851 | -59.5% | -119.3% |
| 30s | Internal DINOv3 STRNet-style ablation | 773 | 255.8596 | 429.1458 | -4.3% | -35.6% |
| 30s | Ours: DINOv3 TimeSFormer | 773 | 245.8012 | 425.9725 | -0.2% | -34.6% |
| 30s | Khaleque-style center-biased exploratory proxy | 773 | 245.2857 | 316.5833 | 0.0% | 0.0% |
| 30s | Internal small-CNN TimeSFormer ablation | 773 | 289.0119 | 501.6175 | -17.8% | -58.4% |
| 30s | Xu-style pixels-only saliency proxy | 773 | 452.3900 | 830.0411 | -84.4% | -162.2% |

## Best By Horizon

| Horizon | Best ADE model | Best ADE | Best FDE model | Best FDE |
| ---: | --- | ---: | --- | ---: |
| 1s | Ours: DINOv3 TimeSFormer | 18.3886 | Ours: DINOv3 TimeSFormer | 28.9105 |
| 3s | Internal DINOv3 STRNet-style ablation | 48.0674 | Internal DINOv3 STRNet-style ablation | 88.3438 |
| 5s | Ours: DINOv3 TimeSFormer | 77.5128 | Internal DINOv3 STRNet-style ablation | 130.8785 |
| 10s | Internal small-CNN TimeSFormer ablation | 104.8426 | Internal DINOv3 STRNet-style ablation | 191.7430 |
| 30s | Khaleque-style center-biased exploratory proxy | 245.2857 | Khaleque-style center-biased exploratory proxy | 316.5833 |

## Interpretation

- The paper-proxy baselines are not exact reproductions; they are task adapters for offline ADE/FDE comparison.
- Constant velocity remains an internal motion-only ablation, not one of the requested literature baselines.
- From 1s through 10s, the learned visual models beat the best paper-inspired proxy on the main ADE comparison.
- The 30s Khaleque-style proxy is strongest on this v2 split, but this should be treated as a diagnostic artifact: the 30s test set is small and dominated by `my_way_home`, while the proxy uses a train-set source center that acts like a map prior.
- Cached DINOv3 generally improves over small-CNN, which supports using a frozen visual token cache as the stronger visual representation path.
- STRNet-style fusion is not uniformly better than TimeSFormer. It helps most clearly on 3s ADE/FDE and on some mid-horizon endpoint errors, while TimeSFormer remains stronger at 1s and 30s.
- This is an ablation of representation/temporal modules, not a pathfinding benchmark. The models predict a single future local trajectory.

## V4 Published Checkpoint Context

The pushed v4 checkpoints could not be re-run on this local machine because `data/wit_vz/processed/wit_vz_v4_defaults_001` and the 44GB DINOv3 cache are not present locally. The published server-side v4 results are:

| Horizon | Constant-velocity ADE/FDE | DINOv3 TimeSFormer ADE/FDE | ADE gain | FDE gain |
| ---: | ---: | ---: | ---: | ---: |
| 1s | 33.1120 / 51.4413 | 26.8676 / 41.5629 | 18.9% | 19.2% |
| 3s | 75.7201 / 131.6904 | 62.1001 / 103.3531 | 18.0% | 21.5% |
| 5s | 111.2669 / 202.7233 | 88.6020 / 157.0852 | 20.4% | 22.5% |
| 10s | 217.1669 / 408.6508 | 154.5734 / 258.7196 | 28.8% | 36.7% |

## References

- Khaleque, B., Cook, M., & Gow, J. (2024). Experiments in Motivating Exploratory Agents. FDG 2024. https://doi.org/10.1145/3649921.3659850
- Xu, K., Bugti, M., & Verbrugge, C. (2026). How Far Can We Go with Pixels Alone? A Pilot Study on Screen-Only Navigation in Commercial 3D ARPGs. https://arxiv.org/abs/2602.18981
