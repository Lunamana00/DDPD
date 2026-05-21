# DDPD Handoff For SSH Codex

Last updated: 2026-05-21

This file is the compact handoff context for continuing the project from
VSCode Remote SSH with the Codex extension.

## Current State

- Repository: `ssh://git@github.com/Lunamana00/DDPD.git`
- Active branch: `main`
- Latest code state before this handoff document: `c220b24 Record v2 horizon model comparison`
- Working focus: egocentric future local path prediction from visual history and relative ego-motion.
- Local code has already been pushed through `c220b24`.

The model predicts a future local path in the agent's egocentric coordinate
frame. It does not currently predict global map position or route ID.

## Core Task

Given:

- RGB history: `[B, T, 3, H_img, W_img]`
- or cached visual tokens: `[B, T, N, C]`
- relative ego-motion history: `[B, T, 3]`

Predict:

- future local path: `[B, H, 2]`

Notation:

- `B`: batch size
- `T`: number of past/history frames
- `N`: number of visual tokens per frame
- `C`: visual token feature dimension
- `H`: number of future prediction steps
- output coordinate: `[forward, right]` in local egocentric frame

## Important Files

- `src/models/cue_memory.py`: main cue-memory path predictor architecture.
- `src/models/backbones.py`: small-CNN, DINOv2, DINOv3, and cached-token backbone logic.
- `src/train_path_predictor.py`: training entrypoint.
- `src/eval_path_predictor.py`: checkpoint evaluation entrypoint.
- `scripts/cache_visual_features.py`: DINOv3 feature cache generation.
- `scripts/run_horizon_sweep.py`: builds/evaluates horizon datasets and baselines.
- `src/wit_vz/build_samples.py`: processed dataset builder.
- `src/wit_vz/dataset.py`: PyTorch dataset loader.
- `reports/horizon_sweep_v2_model_comparison_20260521.md`: latest model comparison.
- `reports/strnet_tokenlearner_validation_20260521.md`: STRNet/TokenLearner validation details.
- `docs/model_architecture_paper_alignment.md`: architecture notes.

## Current Model Architecture

The active model family is implemented in `src/models/cue_memory.py`.

Pipeline:

1. Visual backbone
   - `small_cnn`: trainable lightweight CNN for RGB.
   - `cached_dinov3_convnext_tiny`: frozen cached DINOv3 ConvNeXt-Tiny visual tokens.

2. Positional encoding
   - 2D spatial positional encoding for visual tokens.
   - temporal positional encoding for history frames.

3. Temporal/spatial fusion
   - `temporal_type=timesformer`: TimeSFormer-style divided attention.
     It applies temporal self-attention per spatial token, then spatial
     self-attention per frame.
   - `temporal_type=strnet`: STRNet-style adapted representation path.
     It applies dynamic spatial graph aggregation, temporal shift, and
     multi-resolution temporal difference convolution.

4. Cue selection
   - TokenLearner-style cue token selector.
   - It learns attention maps over visual tokens and produces a smaller set of
     cue tokens.

5. Cue memory
   - Attention-based cue memory bank updated with cue tokens and ego-motion.

6. Decoder
   - Horizon query decoder.
   - Uses learned future-step queries and cross-attention into cue memory.
   - Outputs `[B, H, 2]`.

7. Motion prior
   - Constant-velocity residual path is enabled in recent experiments.
   - The learned model predicts a correction on top of a simple motion prior.

Regularization currently used in training:

- Huber loss
- dropout `0.2`
- weight decay `0.001`
- gradient clipping `1.0`
- early stopping in longer runs

There is multi-modal head support in code, but the latest reported horizon
comparison used deterministic single-path outputs.

## Paper Alignment Notes

The architecture is inspired by TokenLearner, STRNet, and TimeSFormer-style
space-time modeling, but it is adapted to this project.

Important caveat:

- The STRNet implementation here is not the full original goal-conditioned
  visual navigation controller.
- The implemented STRNet-like part is the representation mechanism:
  dynamic spatial graph aggregation, edge messages, temporal shift, and
  temporal difference convolution.
- The dataset does not currently include STRNet's goal image, progress target,
  or navigation policy target.

## Current Dataset

Main expanded v2 dataset:

- Processed dataset: `data/wit_vz/processed/wit_vz_v2_multi_source_001`
- Raw sources:
  - `deadly_corridor`
  - `health_gathering`
  - `my_way_home`

Horizon sweep datasets:

- Processed root: `data/wit_vz/processed/horizon_sweep_v2`
- Horizons: `1, 3, 5, 10, 30` seconds
- Sample FPS: `5.0`
- History window: `1` second
- Split: episode-disjoint per horizon

DINOv3 feature caches:

```text
data/wit_vz/feature_cache/horizon_sweep_v2_future_XXs_dinov3_convnext_tiny
```

Each cached sample stores frozen tokens shaped `[5, 64, 768]`.

Data and feature caches are not normal Git assets. They should be handled with
DVC, Google Drive, rclone, or server-local storage.

## Google Drive / DVC Notes

Local Windows setup used Google Drive Desktop:

- `.drive-ddpd` is a local junction to `G:\...DDPD`
- `.dvc/config.local` has a local Google Drive Desktop remote named
  `gdrive_desktop`

On SSH/Linux, do not assume Google Drive Desktop exists. Prefer one of:

- `rclone` mount/copy for Google Drive
- DVC remote configured against a server-accessible remote
- direct dataset archive download/unpack

Recommended SSH approach:

1. Clone the repo.
2. Sync/download raw and processed data into the same relative paths under
   `data/wit_vz/...`.
3. Regenerate DINOv3 caches on the server if needed.
4. Keep large checkpoints, datasets, and caches out of Git.

## Latest Horizon Comparison

From `reports/horizon_sweep_v2_model_comparison_20260521.md`.

Lower is better.

| Horizon | CV ADE | CV FDE | small-CNN ADE | small-CNN FDE | DINOv3 TimeSFormer ADE | DINOv3 TimeSFormer FDE | DINOv3 STRNet ADE | DINOv3 STRNet FDE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | 27.38 | 43.71 | 23.18 | 36.09 | 18.39 | 28.91 | 19.73 | 30.80 |
| 3s | 63.89 | 116.04 | 55.53 | 102.71 | 50.76 | 90.78 | 48.07 | 88.34 |
| 5s | 106.38 | 197.45 | 88.52 | 155.99 | 77.51 | 135.70 | 77.73 | 130.88 |
| 10s | 145.36 | 281.98 | 104.84 | 199.12 | 110.40 | 205.46 | 113.00 | 191.74 |
| 30s | 391.12 | 694.28 | 289.01 | 501.62 | 245.80 | 425.97 | 255.86 | 429.14 |

Best observed choices:

- 1s: DINOv3 TimeSFormer
- 3s: DINOv3 STRNet
- 5s: DINOv3 TimeSFormer for ADE, DINOv3 STRNet for FDE
- 10s: small-CNN TimeSFormer for ADE, DINOv3 STRNet for FDE
- 30s: DINOv3 TimeSFormer

Takeaway:

- DINOv3 usually helps versus small-CNN.
- STRNet is useful but not uniformly better than TimeSFormer.
- Long horizons need multi-modal metrics and probably a multi-modal output
  head because deterministic ADE/FDE becomes harsh when many futures are valid.

## Environment Setup On SSH

Basic clone:

```bash
git clone git@github.com:Lunamana00/DDPD.git
cd DDPD
```

Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dinov3,vizdoom]" pytest
```

If using `uv`:

```bash
uv sync --extra dinov3 --extra vizdoom --group dev
```

For DINOv3:

```bash
huggingface-cli login
```

Make sure CUDA-compatible PyTorch is installed for the server GPU. The project
itself only declares a generic `torch>=2.0`; the correct CUDA wheel depends on
the server driver/CUDA setup.

## Useful Commands

Run tests:

```bash
python -m pytest
```

Generate DINOv3 cache for a horizon dataset:

```bash
python -m scripts.cache_visual_features \
  --dataset data/wit_vz/processed/horizon_sweep_v2/future_01s \
  --output-dir data/wit_vz/feature_cache/horizon_sweep_v2_future_01s_dinov3_convnext_tiny \
  --backbone dinov3-convnext-tiny \
  --device cuda
```

Train DINOv3 TimeSFormer:

```bash
python -m src.train_path_predictor \
  --dataset data/wit_vz/processed/horizon_sweep_v2/future_01s \
  --visual-feature-cache data/wit_vz/feature_cache/horizon_sweep_v2_future_01s_dinov3_convnext_tiny \
  --model cue_memory_residual \
  --backbone cached_dinov3_convnext_tiny \
  --temporal-type timesformer \
  --selector-type tokenlearner \
  --memory-type attention \
  --output-dir runs/horizon_sweep_v2/dinov3_timesformer_01s \
  --epochs 6 \
  --batch-size 16 \
  --device cuda
```

Train DINOv3 STRNet:

```bash
python -m src.train_path_predictor \
  --dataset data/wit_vz/processed/horizon_sweep_v2/future_01s \
  --visual-feature-cache data/wit_vz/feature_cache/horizon_sweep_v2_future_01s_dinov3_convnext_tiny \
  --model cue_memory_residual \
  --backbone cached_dinov3_convnext_tiny \
  --temporal-type strnet \
  --selector-type tokenlearner \
  --memory-type attention \
  --output-dir runs/horizon_sweep_v2/dinov3_strnet_01s \
  --epochs 6 \
  --batch-size 16 \
  --device cuda
```

Evaluate:

```bash
python -m src.eval_path_predictor \
  --checkpoint runs/horizon_sweep_v2/dinov3_strnet_01s/best.pt \
  --dataset data/wit_vz/processed/horizon_sweep_v2/future_01s \
  --split test \
  --device cuda
```

## Recommended Next Work

1. Move heavy training to the SSH GPU server.
2. Sync or regenerate the v2 horizon datasets and DINOv3 caches.
3. Re-run comparisons with more epochs and early stopping, not only 6 epochs.
4. Add multi-modal trajectory metrics:
   - `minADE`
   - `minFDE`
   - mode confidence / calibration
5. Train one multi-horizon model instead of separate deterministic models for
   every horizon.
6. Increase dataset diversity beyond Doom-only if possible.
7. Keep reports in `reports/` and commit after each completed work sector.

## Notes For Future Codex Sessions

- Read this file first, then inspect `reports/`.
- Do not commit datasets, feature caches, or checkpoints directly to Git.
- Use Git for code, configs, reports, and small metadata.
- Use DVC/Drive/rclone/server storage for large artifacts.
- The user prefers Korean explanations.
- The user wants a commit after each completed sector.
