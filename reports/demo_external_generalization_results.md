# External Generalization Demo Results

This report tracks what has actually been run outside ViZDoom.

## Summary

| Domain | Status | Evidence | Interpretation |
|---|---|---|---|
| MiniWorld | Completed | `reports/demo/external_miniworld_zero_shot_03s/` | The WIT-VZ input/output formulation transfers, but the ViZDoom checkpoint does not zero-shot generalize well. |
| AI2-THOR | Completed | `reports/demo/external_ai2thor_zero_shot_03s/` | The WIT-VZ schema also runs on object-rich Unity indoor scenes, but the ViZDoom checkpoint fails under this domain shift. |
| ProcTHOR | Completed with source checkout | `reports/demo/external_procthor_zero_shot_03s/` | Source ProcTHOR runs through the same WIT-VZ path; the ViZDoom checkpoint fails even more strongly under procedural-house domain shift. |
| DeepMind Lab | Completed with source build | `reports/demo/external_deepmind_lab_zero_shot_03s/` | Source DeepMind Lab runs through the same WIT-VZ path; on this small game-like demo the model improves over CV, but the result is not yet a broad generalization claim. |
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

## ProcTHOR Zero-Shot Demo

ProcTHOR required the source checkout rather than the old PyPI package:

```bash
git clone https://github.com/allenai/procthor.git ~/projects/external_sources/procthor
```

The old PyPI package paired poorly with AI2-THOR 5.0:

```text
procthor==0.0.1.dev2 + ai2thor==5.0.0
CreateHouse failed with a material schema mismatch.
```

The source checkout worked with:

```text
procthor source commit: 53d5bd4
PROCTHOR_INITIALIZATION: branch=main, scene=Procedural
```

Data collection:

```bash
python scripts/collect_procthor_wit_vz.py \
  --out-root data/wit_vz/raw \
  --run-id procthor_demo_001 \
  --episodes 2 \
  --max-steps 50 \
  --fps 5 \
  --width 160 \
  --height 120 \
  --platform CloudRendering \
  --gpu-device 0 \
  --procthor-source-root /home/taehyun/projects/external_sources/procthor \
  --vulkan-library /home/taehyun/local_libs/vulkan/usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
  --overwrite
```

Processed WIT-VZ samples:

```text
dataset: data/wit_vz/processed/procthor_demo_001_03s
episodes: 2
frames: 100
samples: 62
history: 1s, 5 frames
future: 3s, 15 waypoints
split: episode-disjoint
```

Result on all ProcTHOR demo samples:

| Model | ADE | FDE |
|---|---:|---:|
| ViZDoom-trained DINOv3 cue-memory checkpoint, zero-shot ProcTHOR | 79.794 | 134.560 |
| Constant-velocity baseline | 1.158 | 2.288 |

Hard-CV subset:

| Model | ADE | FDE |
|---|---:|---:|
| ViZDoom-trained checkpoint | 75.393 | 126.362 |
| Constant-velocity baseline | 1.936 | 3.685 |

Visualization:

```text
reports/demo/external_procthor_zero_shot_03s/contact_by_house/contact_sheet.png
reports/demo/presentation_sequence/08_procthor_external_overview.png
```

Interpretation:

```text
This is the strongest external-domain failure demo in the current package.
The same WIT-VZ formulation runs on generated Unity houses, but the learned
visual residual is badly miscalibrated. CV remains strong because the scripted
rollouts are locally simple, while the ViZDoom-trained visual head overreacts
to unfamiliar indoor colors, geometry, and scale.
```

## DeepMind Lab Zero-Shot Demo

DeepMind Lab required a source build rather than a direct pip install:

```text
source: https://github.com/google-deepmind/lab
source commit: b1db91a
build route: Bazelisk + Bazel 6.5.0 + local user-space SDL2/OSMesa/libffi/gettext dev packages
runtime env: separate ~/projects/dmlab_env with numpy<2
```

Why a separate runtime env was used:

```text
The DeepMind Lab native module was built against the NumPy 1.x ABI.
The main DDPD venv uses NumPy 2.2.6, so the collector runs in a separate
DeepMind Lab venv and the resulting raw WIT-VZ data is evaluated by the normal
DDPD stack afterward.
```

Data collection:

```bash
python scripts/collect_deepmind_lab_wit_vz.py \
  --out-root data/wit_vz/raw \
  --run-id deepmind_lab_demo_001 \
  --levels nav_maze_static_01 nav_maze_random_goal_01 seekavoid_arena_01 lt_chasm \
  --episodes-per-level 1 \
  --max-steps 50 \
  --fps 5 \
  --width 160 \
  --height 120 \
  --seed 1101 \
  --overwrite
```

Processed WIT-VZ samples:

```text
dataset: data/wit_vz/processed/deepmind_lab_demo_001_03s
levels: 4
episodes: 4
frames: 200
samples: 124
history: 1s, 5 frames
future: 3s, 15 waypoints
split: episode-disjoint
```

Result on all DeepMind Lab demo samples:

| Model | ADE | FDE |
|---|---:|---:|
| ViZDoom-trained DINOv3 cue-memory checkpoint, zero-shot DeepMind Lab | 155.288 | 239.752 |
| Constant-velocity baseline | 180.822 | 306.231 |

Hard-CV subset:

| Model | ADE | FDE |
|---|---:|---:|
| ViZDoom-trained checkpoint | 256.349 | 423.052 |
| Constant-velocity baseline | 345.598 | 619.168 |

Visualization:

```text
reports/demo/external_deepmind_lab_zero_shot_03s/contact_by_level/contact_sheet.png
reports/demo/presentation_sequence/09_deepmind_lab_external_overview.png
```

Interpretation:

```text
This is the first external zero-shot demo where the ViZDoom-trained checkpoint
beats constant velocity on the collected samples. The likely reason is that
DeepMind Lab has more game-like first-person visual structure and more
turn/maze-like trajectory changes than MiniWorld, AI2-THOR, and ProcTHOR.

However, this is still only a small demonstration run: 4 levels, 4 episodes,
124 samples. It should be presented as "the formulation and learned visual
residual can transfer to another game-like domain in some cases", not as proof
of broad zero-shot game generalization.
```

## Demo Claim

## Remaining External Candidates

Detailed execution gates are recorded in:

```text
reports/demo_external_execution_gates.md
```

Short version:

```text
ProcTHOR and DeepMind Lab are now completed using source/checkouted simulator
routes. Habitat-Sim and MineRL/MineDojo still require separate simulator
environments before they can become fair WIT-VZ demos.
```

Use the demos in this order:

```text
1. ViZDoom 3s in-domain overview.
2. ViZDoom hard-case GIFs.
3. ViZDoom 10s long-horizon failure.
4. MiniWorld zero-shot failure as domain-shift evidence.
5. AI2-THOR zero-shot failure as a more object-rich Unity-domain limitation demo.
6. ProcTHOR zero-shot failure as a procedural Unity-house limitation demo.
7. DeepMind Lab zero-shot as a small game-like external-domain positive sanity case.
```

The correct claim is:

```text
The current pipeline is portable to non-ViZDoom WIT-VZ-style data, but the
learned checkpoint is not yet broadly domain-general. DeepMind Lab suggests
that game-like visual/trajectory structure can help transfer, while MiniWorld,
AI2-THOR, and ProcTHOR show that different dynamics, coordinate scale, and
visual style still require scale calibration, adapter tuning, or external-domain
training.
```
