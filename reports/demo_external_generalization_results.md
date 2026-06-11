# External Generalization Demo Results

This report tracks what has actually been run outside ViZDoom.

## Summary

| Domain | Status | Evidence | Interpretation |
|---|---|---|---|
| MiniWorld | Completed | `reports/demo/external_miniworld_zero_shot_03s/` | The WIT-VZ input/output formulation transfers, but the ViZDoom checkpoint does not zero-shot generalize well. |
| AI2-THOR / ProcTHOR | Environment blocked | Local Windows has no matching AI2-THOR build; GPU server lacks `libvulkan1` for CloudRendering. | Collector is ready, but execution requires system-level Unity rendering dependency setup. |
| DeepMind Lab | Planned only | Official candidate in `reports/demo_generalization_plan.md` | Good game-like future extension, heavier install/export work. |
| Habitat | Planned only | Official candidate in `reports/demo_generalization_plan.md` | Useful robotics-style domain shift, less game-like. |
| MineRL / MineDojo | Planned only | Candidate in `reports/demo_generalization_plan.md` | Game-like but action logs must be converted into local future path labels. |

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

## AI2-THOR / ProcTHOR Status

The collector is implemented:

```text
scripts/collect_ai2thor_wit_vz.py
```

Local Windows run failed before scene collection:

```text
ValueError: Invalid commit_id ... no build exists for arch=Windows
```

GPU server Linux CloudRendering run also failed before scene collection:

```text
Platform CloudRendering failed validation with the following errors:
Vulkan API driver missing.
CloudRendering requires libvulkan1.
```

Next required server setup:

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1
```

Then rerun:

```bash
python scripts/collect_ai2thor_wit_vz.py \
  --out-root data/wit_vz/raw \
  --run-id ai2thor_demo_001 \
  --scenes FloorPlan1 FloorPlan2 FloorPlan201 FloorPlan301 \
  --episodes-per-scene 2 \
  --max-steps 120 \
  --fps 5 \
  --platform CloudRendering \
  --headless \
  --overwrite
```

## Demo Claim

Use the demos in this order:

```text
1. ViZDoom 3s in-domain overview.
2. ViZDoom hard-case GIFs.
3. ViZDoom 10s long-horizon failure.
4. MiniWorld zero-shot failure as domain-shift evidence.
5. AI2-THOR as a prepared but system-dependency-blocked next demo.
```

The correct claim is:

```text
The current pipeline is portable to non-ViZDoom WIT-VZ-style data, but the
learned checkpoint is not yet broadly domain-general. External domains require
scale calibration, adapter tuning, or external-domain training.
```
