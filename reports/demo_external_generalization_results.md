# External Generalization Demo Results

This report tracks what has actually been run outside ViZDoom.

## Summary

| Domain | Status | Evidence | Interpretation |
|---|---|---|---|
| MiniWorld | Completed | `reports/demo/external_miniworld_zero_shot_03s/` | The WIT-VZ input/output formulation transfers, but the ViZDoom checkpoint does not zero-shot generalize well. |
| AI2-THOR | Completed | `reports/demo/external_ai2thor_zero_shot_03s/` | The WIT-VZ schema also runs on object-rich Unity indoor scenes, but the ViZDoom checkpoint fails under this domain shift. |
| ProcTHOR | Smoke-tested but blocked | `reports/demo_external_execution_gates.md` | CloudRendering starts, but generated houses repeatedly fail at `CreateHouse`; needs a pinned ProcTHOR environment before demo use. |
| DeepMind Lab | Environment gate not satisfied | `reports/demo_external_execution_gates.md` | Good game-like future extension, but requires Bazel/source build or a known working wheel/container. |
| Habitat | Environment gate not satisfied | `reports/demo_external_execution_gates.md` | Useful robotics-style domain shift, but current server lacks conda/mamba and `habitat-sim`. |
| MineRL / MineDojo | Environment gate not satisfied | `reports/demo_external_execution_gates.md` | Game-like, but current server lacks Java and pose-to-WIT-VZ conversion must be verified. |

## MiniWorld Zero-Shot Demo

Data collection:

```bash
uv run python scripts/collect_miniworld_wit_vz.py \
  --out-root data/wit_vz/raw \
  --run-id miniworld_demo_001 \
  --env-ids MiniWorld-Hallway-v0 MiniWorld-Maze-v0 MiniWorld-WallGap-v0 MiniWorld-ThreeRooms-v0 \
  --episodes-per-env 2 \
  --max-steps 80 \
  --fps 5 \
  --overwrite
```

Processed WIT-VZ samples:

```text
dataset: data/wit_vz/processed/miniworld_demo_001_03s
episodes: 8
samples: 488
history: 1s, 5 frames
future: 3s, 15 waypoints
split: episode-disjoint
```

Zero-shot evaluation command:

```bash
uv run python -m src.eval_path_predictor \
  --dataset data/wit_vz/processed/miniworld_demo_001_03s \
  --checkpoint checkpoints/wit_vz_v4_defaults_dinov3_single_03s.pt \
  --output-dir reports/demo/external_miniworld_zero_shot_03s/eval_all \
  --split all \
  --batch-size 8 \
  --device auto \
  --backbone-override dinov3_convnext_tiny \
  --image-size-override 256 \
  --ignore-checkpoint-visual-cache
```

Result on all MiniWorld demo samples:

| Model | ADE | FDE |
|---|---:|---:|
| ViZDoom-trained DINOv3 cue-memory checkpoint, zero-shot MiniWorld | 42.156 | 71.734 |
| Constant-velocity baseline | 0.250 | 0.456 |

Hard-CV subset:

| Model | ADE | FDE |
|---|---:|---:|
| ViZDoom-trained checkpoint | 36.124 | 59.406 |
| Constant-velocity baseline | 0.615 | 1.122 |

Visualization:

```text
reports/demo/external_miniworld_zero_shot_03s/contact_by_env/contact_sheet.png
reports/demo/external_miniworld_zero_shot_03s/figures/montage.png
```

Interpretation:

```text
This is not evidence of successful zero-shot generalization.
It is evidence that the WIT-VZ formulation can be applied to a different
first-person navigation domain, while the ViZDoom-trained checkpoint is not
calibrated to MiniWorld's visual style, coordinate scale, or simple random-walk
dynamics. In MiniWorld, recent-motion extrapolation is almost perfect because
the trajectories are locally simple, so the learned visual residual hurts.
```

## AI2-THOR Zero-Shot Demo

Server setup:

```bash
# rootless Vulkan workaround on gpuserver3090
apt-get download libvulkan1
dpkg-deb -x libvulkan1_*_amd64.deb ~/local_libs/vulkan
ln -sf libvulkan.so.1.3.204 ~/local_libs/vulkan/usr/lib/x86_64-linux-gnu/libvulkan.so

apt-get download vulkan-tools
dpkg-deb -x vulkan-tools_*_amd64.deb ~/local_libs/vulkan-tools
```

Data collection:

```bash
python scripts/collect_ai2thor_wit_vz.py \
  --out-root data/wit_vz/raw \
  --run-id ai2thor_demo_001 \
  --scenes FloorPlan1 FloorPlan201 \
  --episodes-per-scene 1 \
  --max-steps 50 \
  --fps 5 \
  --width 160 \
  --height 120 \
  --platform CloudRendering \
  --gpu-device 0 \
  --vulkan-library /home/taehyun/local_libs/vulkan/usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
  --overwrite
```

Important execution note:

```text
Do not pass --headless for visual collection.
With AI2-THOR 5.0 CloudRendering, headless=True returned metadata but no RGB frame.
```

Processed WIT-VZ samples:

```text
dataset: data/wit_vz/processed/ai2thor_demo_001_03s
episodes: 2
frames: 100
samples: 62
history: 1s, 5 frames
future: 3s, 15 waypoints
split: episode-disjoint
```

Zero-shot evaluation command:

```bash
uv run python -m src.eval_path_predictor \
  --dataset data/wit_vz/processed/ai2thor_demo_001_03s \
  --checkpoint checkpoints/wit_vz_v4_defaults_dinov3_single_03s.pt \
  --output-dir reports/demo/external_ai2thor_zero_shot_03s/eval_all \
  --split all \
  --batch-size 8 \
  --device auto \
  --num-workers 0 \
  --backbone-override dinov3_convnext_tiny \
  --image-size-override 256 \
  --ignore-checkpoint-visual-cache
```

Result on all AI2-THOR demo samples:

| Model | ADE | FDE |
|---|---:|---:|
| ViZDoom-trained DINOv3 cue-memory checkpoint, zero-shot AI2-THOR | 51.372 | 83.158 |
| Constant-velocity baseline | 1.028 | 1.922 |

Hard-CV subset:

| Model | ADE | FDE |
|---|---:|---:|
| ViZDoom-trained checkpoint | 49.574 | 80.641 |
| Constant-velocity baseline | 1.737 | 3.250 |

Visualization:

```text
reports/demo/external_ai2thor_zero_shot_03s/contact_by_scene/contact_sheet.png
reports/demo/presentation_sequence/07_ai2thor_external_overview.png
```

Interpretation:

```text
This is also not evidence of successful zero-shot generalization.
It is evidence that the WIT-VZ data schema and inference stack can be ported
to Unity indoor scenes. The checkpoint itself is strongly miscalibrated outside
ViZDoom: coordinate scale, visual appearance, and simple scripted movement make
constant velocity much stronger than the learned visual residual.
```

## Demo Claim

## Remaining External Candidates

Detailed execution gates are recorded in:

```text
reports/demo_external_execution_gates.md
```

Short version:

```text
ProcTHOR was the closest next candidate because it reuses AI2-THOR, but the
current procthor==0.0.1.dev2 + ai2thor==5.0.0 setup repeatedly failed at
CreateHouse. DeepMind Lab, Habitat-Sim, and MineRL/MineDojo require separate
simulator environments before they can become fair WIT-VZ demos.
```

Use the demos in this order:

```text
1. ViZDoom 3s in-domain overview.
2. ViZDoom hard-case GIFs.
3. ViZDoom 10s long-horizon failure.
4. MiniWorld zero-shot failure as domain-shift evidence.
5. AI2-THOR zero-shot failure as a more object-rich Unity-domain limitation demo.
```

The correct claim is:

```text
The current pipeline is portable to non-ViZDoom WIT-VZ-style data, but the
learned checkpoint is not yet broadly domain-general. External domains require
scale calibration, adapter tuning, or external-domain training.
```
