# GPU Server DINOv3 Cache Regeneration - 2026-05-21

## Environment

- Project root: `/home/taehyun/projects/DDPD`
- Python environment: `uv` project `.venv`
- Python: `3.10.12`
- PyTorch: `2.11.0+cu128`
- Torch CUDA runtime: `12.8`
- TorchVision: `0.26.0+cu128`
- DINOv3 dependencies:
  - `timm==1.0.27`
  - `transformers==5.8.1`
- ViZDoom: `1.3.0`

GPU check:

| GPU | Device | Memory |
|---:|---|---:|
| 0 | NVIDIA GeForce RTX 3090 | 24 GB |
| 1 | NVIDIA GeForce RTX 3090 | 24 GB |
| 2 | NVIDIA GeForce RTX 3090 | 24 GB |

The Codex sandbox does not expose `/dev/nvidia*` by default, but escalated
commands can access the host GPU devices. PyTorch CUDA validation succeeded
with `torch.cuda.is_available() == True` and `torch.cuda.device_count() == 3`.

## Data Paths

- Processed dataset:
  `data/wit_vz/processed/wit_vz_v2_multi_source_001`
- Raw sources:
  - `data/wit_vz/raw/wit_vz_v2_deadly_corridor_001`
  - `data/wit_vz/raw/wit_vz_v2_health_gathering_001`
  - `data/wit_vz/raw/wit_vz_v2_my_way_home_001`

Dataset loader validation succeeded on the train split:

- Train samples: `25,420`
- RGB history shape: `[5, 3, 64, 64]`
- Ego history shape: `[5, 3]`
- Future path shape: `[5, 2]`

## Cache

- Cache path:
  `data/wit_vz/feature_cache/wit_vz_v2_multi_source_001_dinov3_convnext_tiny`
- Dataset samples cached: `37,070 / 37,070`
- Backbone: `dinov3-convnext-tiny`
- Image size: `256`
- Save dtype: `float16`
- Per-sample token shape: `[5, 64, 768]`
- Final cache size: about `18 GB`

Command:

```bash
uv run python -m scripts.cache_visual_features \
  --dataset data/wit_vz/processed/wit_vz_v2_multi_source_001 \
  --output-dir data/wit_vz/feature_cache/wit_vz_v2_multi_source_001_dinov3_convnext_tiny \
  --backbone dinov3-convnext-tiny \
  --device cuda \
  --batch-size 128 \
  --mixed-precision \
  --overwrite
```

## Validation

Cached-token dataset loading succeeded:

- Cached visual token shape: `[5, 64, 768]`
- Loader converts cached tokens to `float32` for training.

Test suite:

```text
16 passed, 1 skipped
```

## Smoke Training

Ran a 1-epoch cached DINOv3 TimeSFormer smoke training job to validate the new
cache path end-to-end.

- Run:
  `runs/wit_vz_v2_dinov3_timesformer_cached_smoke`
- Model: `cue_memory_path_predictor`
- Backbone: `cached_dinov3_convnext_tiny`
- Temporal type: `timesformer`
- Selector: `tokenlearner`
- Memory: `attention`
- Batch size: `128`
- Mixed precision: enabled

Result:

| Split | ADE | FDE |
|---|---:|---:|
| Val | 20.8838 | 32.2943 |
| Test | 18.8148 | 28.9961 |

This was only a pipeline smoke check, not a final training run. The next
substantive step is a longer cached-DINOv3 v2 training run with early stopping,
then a comparison against the existing v2 small-CNN result.
