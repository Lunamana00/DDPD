# DDPD: Egocentric Visual-History Path Prediction

DDPD is a ViZDoom-based capstone research project for predicting a future
egocentric local path from first-person visual history and recent ego-motion.

- GitHub: <https://github.com/Lunamana00/DDPD>
- SSH remote: `ssh://git@github.com/Lunamana00/DDPD.git`
- Final presentation/video Drive folder:
  <https://drive.google.com/drive/folders/1k8lbIFUdT6LeuC4LRM7S6oAxGhE-kfnq>

## Research Question

The project asks whether a model can predict where an agent will move next in a
3D game scene without using a global map, full pose, depth labels, semantic
labels, or object annotations.

Input:

- 1 second RGB history from the egocentric game screen
- 1 second ego-motion history: `[delta_forward, delta_right, delta_yaw]`

Output:

- Future local trajectory in the current egocentric frame
- Each waypoint is represented as `[forward, right]`

## Method Summary

The current model predicts a visual residual on top of a motion prior:

```text
P_pred = P_cv + Delta P_visual
```

Main pipeline:

```text
cached DINOv3 ConvNeXt-Tiny visual tokens
-> linear projection + 2D spatial positional encoding
-> spatial relation refinement
-> TimeSFormer-style temporal adapter
-> TokenLearner cue selection
-> cue temporal transformer
-> cue memory bank with ego-motion conditioning
-> horizon query decoder
-> future local path
```

The DINOv3 backbone is frozen. Training updates the downstream projection,
spatial/temporal modules, cue selector, memory module, and decoder.

## Dataset

The main dataset is WIT-VZ v4, built from ViZDoom default scenarios.

```text
Dataset: data/wit_vz/processed/wit_vz_v4_defaults_001
Samples: 93,403
Scenarios: 15
History: 1 second, 5 FPS, 5 RGB frames
Future horizons: 1s, 3s, 5s, 10s
Target: future local path [forward, right]
```

Cached DINOv3 features:

```text
data/wit_vz/feature_cache/wit_vz_v4_defaults_001_dinov3_convnext_tiny
visual token shape per sample: [5, 64, 768]
```

## Key Results

Lower ADE/FDE is better. The table uses the v4 test split and non-privileged
baselines.

| Horizon | Model | ADE | FDE | Test samples |
|---:|---|---:|---:|---:|
| 1s | Ours: cached DINOv3 trajectory predictor | 26.87 | 41.56 | 15,373 |
| 1s | Constant velocity baseline | 33.11 | 51.44 | 15,373 |
| 1s | Xu-style pixels-only saliency proxy | 36.54 | 57.18 | 15,373 |
| 1s | Khaleque-style exploratory proxy | 46.86 | 73.17 | 15,373 |
| 3s | Ours: cached DINOv3 trajectory predictor | 62.10 | 103.35 | 11,884 |
| 3s | Constant velocity baseline | 75.72 | 131.69 | 11,884 |
| 3s | Xu-style pixels-only saliency proxy | 86.21 | 153.66 | 11,884 |
| 3s | Khaleque-style exploratory proxy | 101.74 | 163.34 | 11,884 |
| 5s | Ours: cached DINOv3 trajectory predictor | 88.60 | 157.09 | 10,294 |
| 5s | Constant velocity baseline | 111.27 | 202.72 | 10,294 |
| 5s | Xu-style pixels-only saliency proxy | 125.00 | 234.87 | 10,294 |
| 5s | Khaleque-style exploratory proxy | 139.24 | 222.03 | 10,294 |
| 10s | Ours: cached DINOv3 trajectory predictor | 154.57 | 258.72 | 7,434 |
| 10s | Constant velocity baseline | 217.17 | 408.65 | 7,434 |
| 10s | Xu-style pixels-only saliency proxy | 254.59 | 495.81 | 7,434 |
| 10s | Khaleque-style exploratory proxy | 198.39 | 286.86 | 7,434 |

Cue-memory ablation at 3s horizon:

| Variant | Test ADE | Test FDE |
|---|---:|---:|
| Attention cue memory + ego-motion | 62.98 | 105.85 |
| Attention memory without ego-motion | 70.85 | 123.10 |
| Latest cue only, no learned memory update | 70.02 | 119.22 |
| Mean cue only, no learned memory update | 70.84 | 120.90 |
| Slot-wise GRU memory | 62.38 | 105.81 |

## Reports And Deliverables

Core documentation:

- [Current research A to Z](docs/research_a_to_z_20260621.md)
- [Dataset and training method](reports/dataset_and_training_method_20260521.md)
- [Paper-adapted baselines](reports/paper_baselines_v4.md)
- [Cue memory ablation](reports/episodic_memory_ablation_v4.md)
- [Demo sequence notes](reports/demo/presentation_sequence/README.md)

Final presentation/video deliverables:

- Drive folder:
  <https://drive.google.com/drive/folders/1k8lbIFUdT6LeuC4LRM7S6oAxGhE-kfnq>
- TTS video:
  <https://drive.google.com/file/d/1ZQDnm_AA9yzy1GjAUNCk56Sxoro_HGd4/view?usp=drivesdk>
- Presentation PPTX:
  <https://docs.google.com/presentation/d/1vj3SYwIgIGbweFySrISUTHOEaGhK75zy/edit?usp=drivesdk&ouid=110532140100191797409&rtpof=true&sd=true>
- Submission ZIP:
  <https://drive.google.com/file/d/1wTzpsCrKHDllz8wF0IN02WZXpqSAS-5l/view?usp=drivesdk>

Local generated presentation/video files are under:

```text
report/작성본/presentation_video_20260623/
```

## Repository Layout

```text
src/       model, dataset, training, and evaluation code
scripts/   experiment, rendering, and dataset utilities
configs/   training and ablation configs
reports/   experiment reports and demo summaries
docs/      architecture notes and research planning documents
client/    Unity visualization prototype
server/    inference server prototype
```

Large raw datasets, feature caches, and training outputs are intentionally not
expected to be fully tracked in Git. They are managed as local/Drive artifacts.

## Quick Start

Install dependencies:

```powershell
uv sync
```

Run tests:

```powershell
uv run pytest
```

Example evaluation/reporting scripts are in `scripts/` and experiment configs
are in `configs/`.

## Notes

- The current model is a supervised trajectory forecaster, not a full game
  policy.
- Paper baselines in this repository are paper-adapted WIT-VZ baselines, not
  exact reproductions of the original papers.
- Oracle-style baselines such as PointNav/A* use privileged target or map
  information and should be interpreted as reference comparisons, not fair
  non-privileged baselines.
