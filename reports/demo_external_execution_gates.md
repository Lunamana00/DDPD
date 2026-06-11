# External Demo Execution Gates

This report records the current execution state of the external demo candidates
after the completed MiniWorld, AI2-THOR, ProcTHOR, and DeepMind Lab zero-shot
demos.

## Current Status

| Candidate | Status | Evidence | Demo Decision |
|---|---|---|---|
| MiniWorld | Completed | `reports/demo/external_miniworld_zero_shot_03s/` | Use as a lightweight external-domain failure/sanity demo. |
| AI2-THOR | Completed | `reports/demo/external_ai2thor_zero_shot_03s/` | Use as the object-rich Unity-domain failure/sanity demo. |
| ProcTHOR | Completed with source checkout | `reports/demo/external_procthor_zero_shot_03s/` | Use as a procedural Unity-house domain-shift failure demo. |
| DeepMind Lab | Completed with source build | `reports/demo/external_deepmind_lab_zero_shot_03s/` | Use as the small game-like external-domain positive sanity case. |
| Habitat-Sim | Environment gate not satisfied | No `habitat-sim` pip candidate in this venv check; no conda/mamba installed on `gpuserver3090`. | Keep as future robotics/photorealistic extension. |
| MineRL / MineDojo | Environment gate not satisfied | `minerl` and `minedojo` exist on PyPI, but Java is absent on `gpuserver3090`; WIT-VZ pose conversion is not direct. | Keep as future Minecraft-style extension. |

## Server State

Checked on `gpuserver3090` under `/home/taehyun/projects/DDPD/.venv`.

```text
ai2thor==5.0.0
procthor==0.0.1.dev2 (PyPI package; incompatible with AI2-THOR 5.0 procedural material schema)
attrs==26.1.0
pandas==2.3.3
shapely==2.1.2
python-fcl==0.7.0.11
scipy==1.15.3
moviepy==1.0.3
minerl: not-installed
minedojo: not-installed
habitat-sim: not-installed
deepmind_lab: installed in separate /home/taehyun/projects/dmlab_env
```

System tools:

```text
xvfb-run: available
git/gcc/g++: available
bazel: installed user-space through Bazelisk at ~/bin/bazel
conda/mamba: missing
java: missing
home disk: 624G free
AI2-THOR release cache: 3.1G
```

## ProcTHOR Smoke Test And Completed Demo

ProcTHOR was selected as the next most practical candidate because it is
compatible with AI2-THOR and should reuse the existing CloudRendering and WIT-VZ
collection path.

Installed packages:

```bash
pip install procthor attrs pandas shapely "moviepy<2" python-fcl scipy
git clone https://github.com/allenai/procthor.git ~/projects/external_sources/procthor
```

Important finding:

```text
The PyPI procthor==0.0.1.dev2 package is too old for the current AI2-THOR 5.0
procedural house schema. It failed with MaterialProperties JSON conversion
errors. The source checkout at commit 53d5bd4 uses branch=main and works.
```

Rootless rendering setup reused from the AI2-THOR demo:

```bash
PATH=$HOME/local_libs/vulkan-tools/usr/bin:$PATH
LD_LIBRARY_PATH=$HOME/local_libs/vulkan/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Smoke-test structure:

```python
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from procthor.constants import PROCTHOR_INITIALIZATION
from procthor.generation import HouseGenerator
from procthor.generation.room_specs import PROCTHOR10K_ROOM_SPEC_SAMPLER

controller = Controller(
    width=160,
    height=120,
    platform=CloudRendering,
    gpu_device=0,
    quality="Low",
    **PROCTHOR_INITIALIZATION,
)

generator = HouseGenerator(
    split="train",
    seed=seed,
    controller=controller,
    room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
)
house, _ = generator.sample()
controller.step(action="CreateHouse", house=house.data)
```

Initial PyPI-package result:

```text
seeds 1-10: AssertionError Unable to CreateHouse!
after disabling small-object placement: CreateHouse failed with
Newtonsoft.Json.JsonSerializationException converting string floorMaterial to
Thor.Procedural.Data.MaterialProperties.
```

Source-checkout result:

```text
PYTHONPATH=/home/taehyun/projects/external_sources/procthor
PROCTHOR_INITIALIZATION: branch=main, scene=Procedural
CreateHouse: success
Teleport to choose_agent_pose: success
MoveAhead / RotateRight / MoveAhead: success
RGB frame shape: [120, 160, 3]
```

Completed WIT-VZ run:

```text
collector: scripts/collect_procthor_wit_vz.py
raw: data/wit_vz/raw/procthor_demo_001
processed: data/wit_vz/processed/procthor_demo_001_03s
episodes: 2
frames: 100
samples: 62
zero-shot output: reports/demo/external_procthor_zero_shot_03s/
ADE/FDE: 79.794 / 134.560
CV ADE/FDE: 1.158 / 2.288
```

## DeepMind Lab Source Build And Completed Demo

Why it is still useful:

```text
DeepMind Lab is the most game-like candidate after ViZDoom: first-person 3D
levels, navigation tasks, RGB observations, and action-based movement.
```

Build route:

```text
Bazelisk installed at ~/bin/bazel
DeepMind Lab source checkout: ~/projects/external_sources/deepmind_lab
source commit: b1db91a
Bazel version used: 6.5.0
local user-space dev packages: SDL2, OSMesa, libffi, gettext
wheel: /tmp/dmlab_pkg/deepmind_lab-1.0-py3-none-any.whl
runtime venv: ~/projects/dmlab_env with numpy<2, dm_env, six, pillow
```

Important build notes:

```text
1. Bazel 9 failed because the upstream WORKSPACE/BUILD files are not bzlmod-ready.
2. Bazel 6.5.0 worked after pinning Abseil to 20230125.3.
3. The source tree needed user-space SDL2/OSMesa/libffi/gettext dev packages
   because sudo apt install is not available.
4. The DDPD venv has NumPy 2.2.6, but the DMLab native module needs NumPy 1.x
   ABI compatibility, so collection runs in a separate venv.
5. DeepMind Lab exposes RGB plus `DEBUG.POS.TRANS` and `DEBUG.POS.ROT`, so
   WIT-VZ future local path labels can be built without velocity integration.
```

Smoke-test result:

```text
level: nav_maze_static_01
observations: RGB_INTERLEAVED, DEBUG.POS.TRANS, DEBUG.POS.ROT, VEL.TRANS, VEL.ROT
RGB shape: [120, 160, 3]
step: success
```

Completed WIT-VZ run:

```text
collector: scripts/collect_deepmind_lab_wit_vz.py
raw: data/wit_vz/raw/deepmind_lab_demo_001
processed: data/wit_vz/processed/deepmind_lab_demo_001_03s
levels: nav_maze_static_01, nav_maze_random_goal_01, seekavoid_arena_01, lt_chasm
episodes: 4
frames: 200
samples: 124
zero-shot output: reports/demo/external_deepmind_lab_zero_shot_03s/
ADE/FDE: 155.288 / 239.752
CV ADE/FDE: 180.822 / 306.231
```

## Habitat-Sim Gate

Why it is still useful:

```text
Habitat is less game-like, but it is a strong photorealistic embodied-navigation
domain-shift candidate with RGB, agent pose, and standard trajectory APIs.
```

Current blocker:

```text
conda/mamba: missing on gpuserver3090
habitat-sim: no direct pip candidate in the current venv check
```

Required route:

```text
1. Install mamba/conda or use a container.
2. Install habitat-sim headless from conda-forge/aihabitat.
3. Download habitat_test_scenes.
4. Run a scripted agent and export RGB frames plus agent state.
5. Convert to WIT-VZ raw schema.
```

## MineRL / MineDojo Gate

Why it is still useful:

```text
Minecraft is game-like and visually very different from ViZDoom. It would be a
strong stress test for broad visual-domain generalization.
```

Current blocker:

```text
minerl and minedojo are available on PyPI.
java is missing on gpuserver3090.
xvfb-run is available.
```

Additional modeling issue:

```text
MineRL/MineDojo expose first-person RGB observations, but ADE/FDE evaluation
requires local future path labels. The conversion is not just video playback:
we need reliable agent position/yaw or action-to-pose integration to produce
[forward, right] future waypoints.
```

Required route:

```text
1. Install Java/JDK and MineRL or MineDojo in a separate environment.
2. Run under xvfb-run for headless rendering.
3. Verify RGB observation and agent position/yaw availability.
4. Implement WIT-VZ raw export only if pose is reliable.
```

## Presentation Recommendation

Use completed demos only:

```text
1. ViZDoom in-domain diversity.
2. ViZDoom hard-case GIFs.
3. ViZDoom 10s long-horizon limitation.
4. MiniWorld external zero-shot failure.
5. AI2-THOR external zero-shot failure.
6. ProcTHOR external zero-shot failure.
7. DeepMind Lab external zero-shot positive sanity case.
```

Mention the remaining candidates as future work:

```text
ProcTHOR and DeepMind Lab are now completed with source-checkout/build routes.
Habitat and MineRL/MineDojo need separate simulator environments before they
can become fair WIT-VZ demos.
```

## Source Pointers

- ProcTHOR repository: https://github.com/allenai/procthor
- AI2-THOR CloudRendering docs: https://ai2thor.allenai.org/ithor/documentation/
- DeepMind Lab repository: https://github.com/google-deepmind/lab
- Habitat-Sim repository: https://github.com/facebookresearch/habitat-sim
- MineRL first-agent docs: https://minerl.readthedocs.io/en/latest/tutorials/first_agent.html
- MineDojo repository: https://github.com/MineDojo/MineDojo
