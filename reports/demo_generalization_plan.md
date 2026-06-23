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
ProcTHOR was completed after this using a source checkout because the old PyPI
package mismatched the AI2-THOR 5.0 procedural material schema.
See reports/demo_external_execution_gates.md and
reports/demo/external_procthor_zero_shot_03s/.
```

### 2.3 ProcTHOR

Role:

```text
Procedurally generated Unity indoor houses.
Good for showing that the same WIT-VZ path can run beyond fixed AI2-THOR scenes.
```

Collector:

```text
scripts/collect_procthor_wit_vz.py
```

Execution note:

```text
Use a source checkout of github.com/allenai/procthor.
The PyPI procthor==0.0.1.dev2 package failed against AI2-THOR 5.0 because
the generated material schema was stale.
```

Current demo result:

```text
reports/demo/external_procthor_zero_shot_03s/

ProcTHOR zero-shot was run on 62 samples from 2 generated houses.
The pipeline runs, but the ViZDoom checkpoint again fails to generalize:
ADE 79.794 / FDE 134.560, while constant velocity is ADE 1.158 / FDE 2.288.
This is a procedural Unity-house domain-shift limitation demo.
```

### 2.4 DeepMind Lab

Role:

```text
Game-like external first-person navigation domain.
Compared with MiniWorld/AI2-THOR/ProcTHOR, it is visually closer to a game
and contains more maze/turn-style movement.
```

Collector:

```text
scripts/collect_deepmind_lab_wit_vz.py
```

Execution note:

```text
DeepMind Lab was built from source on gpuserver3090 using Bazelisk/Bazel 6.5.0
and user-space SDL2/OSMesa/libffi/gettext dev packages. Collection runs in a
separate dmlab_env because the native module needs NumPy 1.x ABI compatibility.
```

Current demo result:

```text
reports/demo/external_deepmind_lab_zero_shot_03s/

DeepMind Lab zero-shot was run on 124 samples from 4 levels:
nav_maze_static_01, nav_maze_random_goal_01, seekavoid_arena_01, lt_chasm.
The pipeline runs, and unlike the other external demos the model improves over
constant velocity:
ADE 155.288 / FDE 239.752, while constant velocity is ADE 180.822 / FDE 306.231.
This is a small positive external-domain sanity demo, not broad generalization proof.
```

### 2.5 Habitat-Sim

Role:

```text
Photorealistic embodied-navigation domain.
This is less game-like than ViZDoom or DeepMind Lab, but useful for testing
whether the same RGB-history + ego-motion -> local future path formulation can
run on realistic indoor scenes with true agent pose.
```

Collector:

```text
scripts/collect_habitat_wit_vz.py
```

Execution note:

```text
Habitat-Sim was installed in a user-space micromamba env:
/home/taehyun/projects/habitat_env
package: habitat-sim 0.3.3 headless, Python 3.9
test scene: habitat-test-scenes/skokloster-castle.glb
```

Current demo result:

```text
reports/demo/external_habitat_zero_shot_03s/

Habitat-Sim zero-shot was run on 124 samples from the skokloster-castle test scene.
The pipeline runs, but the ViZDoom checkpoint fails to generalize:
ADE 44.765 / FDE 74.896, while constant velocity is ADE 0.571 / FDE 1.080.
This is a photorealistic domain-shift limitation demo.
```

### 2.6 MineDojo

Role:

```text
Minecraft-style sandbox first-person domain.
Useful for checking whether the WIT-VZ schema can use a very different game
visual style while still obtaining privileged pose labels for evaluation.
```

Collector:

```text
scripts/collect_minedojo_wit_vz.py
```

Execution note:

```text
MineDojo was installed in a user-space micromamba env with Java 8:
/home/taehyun/projects/minedojo_env

Compatibility fixes were needed:
- pip/setuptools/wheel were pinned for gym==0.21.0 metadata compatibility.
- numpy was pinned to <2 because MineDojo uses removed np.unicode_ aliases.
- the old Malmo Gradle dependency com.github.SpongePowered:MixinGradle:dcfaf61
  was provided through a local Maven cache because the original remote artifact
  no longer resolved reliably.

One plains episode was collected. Additional biome launches were too slow on
the shared server, so this should be treated as a formulation gate and failure
case, not as a broad Minecraft evaluation.
```

Current demo result:

```text
reports/demo/external_minedojo_zero_shot_03s/

MineDojo zero-shot was run on 31 samples from one plains episode.
The pipeline runs with RGB and privileged pose labels, but the ViZDoom
checkpoint fails badly while constant velocity remains strong:
ADE 89.338 / FDE 161.605, while constant velocity is ADE 0.447 / FDE 0.837.
This is a Minecraft-style domain-gap failure demo.
```

### 2.7 Later Candidates

| Candidate | Use | Risk |
|---|---|---|
| MineRL human demonstrations | Minecraft human trajectories / videos. | Standard MineRL observations may not expose pose labels needed for ADE/FDE without extra instrumentation. |
| WorldCam-50h | Real gameplay video + camera pose. | Dataset-style extension, not a quick playable simulator demo. |

## 3. Recommended Demo Package

Use this order for presentation:

```text
1. ViZDoom 3s contact sheet: many scenarios, easy/hard/failure.
2. ViZDoom hard-case GIFs: six short playable-looking examples.
3. ViZDoom 10s contact sheet: long-horizon limitation and trajectory drift.
4. MiniWorld external zero-shot: quick cross-domain sanity check.
5. AI2-THOR external zero-shot: object-rich Unity-domain limitation demo.
6. ProcTHOR external zero-shot: procedural Unity-house limitation demo.
7. DeepMind Lab external zero-shot: small game-like positive sanity demo.
8. Habitat-Sim external zero-shot: photorealistic embodied-navigation failure demo.
9. MineDojo external zero-shot: Minecraft-style formulation gate and domain-gap failure demo.
```

Claim boundary:

```text
ViZDoom multi-scenario results support in-domain scenario diversity.
MiniWorld/AI2-THOR/ProcTHOR/DeepMind Lab/Habitat-Sim/MineDojo results support external formulation
transfer and domain-shift analysis. DMLab is currently the only small external
demo where the zero-shot checkpoint beats CV, but it should not be oversold as
proven broad game generalization unless retrained and evaluated with matched splits.
```

## 4. Reference Links

| Candidate | Official source | Why it was included |
|---|---|---|
| MiniWorld | https://miniworld.farama.org/ | Lightweight first-person Gymnasium navigation; fastest external sanity check. |
| AI2-THOR | https://ai2thor.allenai.org/ | Unity-based interactive indoor scenes with RGB observations and agent state. |
| ProcTHOR | https://procthor.allenai.org/ | Procedurally generated AI2-THOR houses for larger domain variation. |
| DeepMind Lab | https://github.com/google-deepmind/lab | Game-like first-person 3D navigation and puzzle-solving testbed. |
| Habitat | https://aihabitat.org/ | Photorealistic embodied AI simulator; useful as a later robotics-style domain shift. |
| MineDojo | https://docs.minedojo.org/ | Minecraft-style first-person environment with RGB and privileged location stats. |
