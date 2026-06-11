# Demo and Generalization Plan

This plan separates the demo into two evidence levels:

1. **ViZDoom multi-scenario demo**: same task/domain family, multiple scenario styles.
2. **External generalization demo**: different visual domains converted into the same WIT-VZ path-prediction schema.

## 1. ViZDoom Multi-Scenario Demo

Goal:

```text
Show that the model is not demonstrated on one cherry-picked scenario only.
Use the same input/output definition across several ViZDoom scenarios:
RGB history + ego-motion -> future local path [forward, right].
```

Generated artifacts:

```text
reports/demo/vizdoom_multi_scenario_03s/
reports/demo/vizdoom_multi_scenario_10s/
reports/demo/vizdoom_hardcase_gifs_03s/
reports/demo/presentation_sequence/
```

Recommended presentation sequence:

| Demo | Scenario | Why it is useful |
|---|---|---|
| Easy/basic | `basic`, `simpler_basic` | Simple movement; CV baseline is often already competitive. |
| Navigation | `my_way_home` | Shows route-like egocentric movement. |
| Object/avoidance | `health_gathering`, `health_gathering_supreme` | Movement is affected by pickups, obstacles, and scene layout. |
| Turn/defense | `defend_the_center`, `defend_the_line` | Shows rotation-heavy and lateral-motion cases. |
| Noisy/gameplay | `deathmatch`, `multi_deathmatch`, `rocket_basic` | Useful as limitation examples; scene can be visually busy and target path can be noisy. |

Selection rule:

```text
easy    = lowest constant-velocity ADE inside the scenario
hard    = high constant-velocity ADE where the model improves over CV
failure = high model error or model worse than CV
```

Main script:

```bash
python scripts/render_vizdoom_scenario_demo.py \
  --dataset data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s \
  --predictions runs/episodic_memory_ablation_v4/seed_7/03s/long_attention_no_ego/predictions.jsonl \
  --out reports/demo/vizdoom_multi_scenario_03s \
  --raw-root-base /home/taehyun/projects/DDPD \
  --scenarios basic my_way_home health_gathering health_gathering_supreme defend_the_center defend_the_line predict_position deathmatch multi_deathmatch rocket_basic \
  --cases easy hard failure
```

GIF subset:

```bash
python scripts/render_vizdoom_scenario_demo.py \
  --dataset data/wit_vz/processed/horizon_sweep_v4_defaults/future_03s \
  --predictions runs/episodic_memory_ablation_v4/seed_7/03s/long_attention_no_ego/predictions.jsonl \
  --out reports/demo/vizdoom_hardcase_gifs_03s \
  --raw-root-base /home/taehyun/projects/DDPD \
  --scenarios basic my_way_home health_gathering defend_the_line predict_position deathmatch \
  --cases hard \
  --make-gifs \
  --gif-frames 36 \
  --gif-fps 8
```

Interpretation:

```text
If CV is already good, the model does not need much visual correction.
If CV is poor and the model path is closer to GT, the visual/history modules are doing useful work.
If both CV and model fail, it is a limitation or label-noise/domain-complexity example.
```

## 2. External Generalization Demo

External datasets should not be treated as the same claim as ViZDoom in-domain performance.
They should answer a weaker but important question:

```text
Can the same input/output formulation be applied outside ViZDoom, and where does it break?
```

### 2.1 MiniWorld

Role:

```text
Lightweight synthetic 3D first-person navigation sanity check.
Visually simpler than ViZDoom, but still RGB + pose trajectory.
```

Collector:

```text
scripts/collect_miniworld_wit_vz.py
```

Example commands:

```bash
uv pip install miniworld gymnasium

uv run python scripts/collect_miniworld_wit_vz.py \
  --out-root data/wit_vz/raw \
  --run-id miniworld_nav_001 \
  --env-ids MiniWorld-Hallway-v0 MiniWorld-Maze-v0 MiniWorld-WallGap-v0 MiniWorld-ThreeRooms-v0 \
  --episodes-per-env 8 \
  --max-steps 240 \
  --fps 5 \
  --overwrite
```

Then build WIT-VZ samples:

```bash
uv run python -m src.wit_vz.build_samples \
  --raw data/wit_vz/raw/miniworld_nav_001 \
  --out data/wit_vz/processed/miniworld_nav_001 \
  --history-sec 1.0 \
  --future-sec 3.0 \
  --sample-fps 5 \
  --stride 1 \
  --split episode \
  --seed 951
```

Evaluation modes:

```text
zero-shot: use the ViZDoom checkpoint directly
adapter-tuned: freeze DINO and train only downstream/adapters on MiniWorld
in-domain: train the full downstream predictor on MiniWorld train split
```

Current demo result:

```text
reports/demo/external_miniworld_zero_shot_03s/

MiniWorld zero-shot was run on 488 samples from 4 MiniWorld envs.
The pipeline runs, but the ViZDoom checkpoint fails to generalize:
ADE 42.156 / FDE 71.734, while constant velocity is ADE 0.250 / FDE 0.456.
This is a useful domain-shift limitation demo, not a success claim.
```

### 2.2 AI2-THOR / ProcTHOR

Role:

```text
Object-rich Unity indoor domain.
Good for showing a more realistic cross-domain setting than MiniWorld.
```

AI2-THOR collector:

```text
scripts/collect_ai2thor_wit_vz.py
```

Demo command used on `gpuserver3090`:

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

Then build WIT-VZ samples:

```bash
uv run python -m src.wit_vz.build_samples \
  --raw data/wit_vz/raw/ai2thor_demo_001 \
  --out data/wit_vz/processed/ai2thor_demo_001_03s \
  --history-sec 1.0 \
  --future-sec 3.0 \
  --sample-fps 5 \
  --stride 1 \
  --split episode \
  --seed 901
```

Current demo result:

```text
reports/demo/external_ai2thor_zero_shot_03s/

AI2-THOR zero-shot was run on 62 samples from 2 AI2-THOR scenes.
The pipeline runs, but the ViZDoom checkpoint again fails to generalize:
ADE 51.372 / FDE 83.158, while constant velocity is ADE 1.028 / FDE 1.922.
This is a stronger object-rich domain-shift limitation demo than MiniWorld.
It should not be presented as a successful zero-shot result.
```

Execution note:

```text
CloudRendering worked on gpuserver3090 with a rootless libvulkan1/vulkan-tools setup.
Do not pass --headless for visual collection because it can return metadata without RGB frames.
ProcTHOR is still a planned extension, not collected in this pass.
```

### 2.3 Later Candidates

| Candidate | Use | Risk |
|---|---|---|
| DeepMind Lab | Most game-like external environment after ViZDoom. | Heavier install and trajectory export work. |
| MineRL / MineDojo | Minecraft game data and human demonstrations. | Converting action logs to local future path is non-trivial. |
| Habitat-Sim | Photorealistic indoor navigation. | More robotics than game. |
| WorldCam-50h | Real gameplay video + camera pose. | Dataset-style extension, not a quick playable simulator demo. |

## 3. Recommended Demo Package

Use this order for presentation:

```text
1. ViZDoom 3s contact sheet: many scenarios, easy/hard/failure.
2. ViZDoom hard-case GIFs: six short playable-looking examples.
3. ViZDoom 10s contact sheet: long-horizon limitation and trajectory drift.
4. MiniWorld external zero-shot: quick cross-domain sanity check.
5. AI2-THOR external zero-shot: object-rich Unity-domain limitation demo.
6. ProcTHOR / DeepMind Lab / MineRL / Habitat: future expansion candidates.
```

Claim boundary:

```text
ViZDoom multi-scenario results support in-domain scenario diversity.
MiniWorld/AI2-THOR results support external formulation transfer and domain-shift analysis.
They should not be oversold as proven broad game generalization unless retrained and evaluated with matched splits.
```

## 4. Reference Links

| Candidate | Official source | Why it was included |
|---|---|---|
| MiniWorld | https://miniworld.farama.org/ | Lightweight first-person Gymnasium navigation; fastest external sanity check. |
| AI2-THOR | https://ai2thor.allenai.org/ | Unity-based interactive indoor scenes with RGB observations and agent state. |
| ProcTHOR | https://procthor.allenai.org/ | Procedurally generated AI2-THOR houses for larger domain variation. |
| DeepMind Lab | https://github.com/google-deepmind/lab | Game-like first-person 3D navigation and puzzle-solving testbed. |
| Habitat | https://aihabitat.org/ | Photorealistic embodied AI simulator; useful as a later robotics-style domain shift. |
