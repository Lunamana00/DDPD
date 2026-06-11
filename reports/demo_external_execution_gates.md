# External Demo Execution Gates

This report records the current execution state of the external demo candidates
after the completed MiniWorld and AI2-THOR zero-shot demos.

## Current Status

| Candidate | Status | Evidence | Demo Decision |
|---|---|---|---|
| MiniWorld | Completed | `reports/demo/external_miniworld_zero_shot_03s/` | Use as a lightweight external-domain failure/sanity demo. |
| AI2-THOR | Completed | `reports/demo/external_ai2thor_zero_shot_03s/` | Use as the object-rich Unity-domain failure/sanity demo. |
| ProcTHOR | Smoke-tested but blocked | `HouseGenerator` starts, CloudRendering starts, but generated houses repeatedly fail at `CreateHouse`. | Do not present as a completed demo yet. Keep as next engineering item. |
| DeepMind Lab | Environment gate not satisfied | No `deepmind_lab` pip candidate in this venv check; no `bazel` installed on `gpuserver3090`. | Keep as future game-like extension. |
| Habitat-Sim | Environment gate not satisfied | No `habitat-sim` pip candidate in this venv check; no conda/mamba installed on `gpuserver3090`. | Keep as future robotics/photorealistic extension. |
| MineRL / MineDojo | Environment gate not satisfied | `minerl` and `minedojo` exist on PyPI, but Java is absent on `gpuserver3090`; WIT-VZ pose conversion is not direct. | Keep as future Minecraft-style extension. |

## Server State

Checked on `gpuserver3090` under `/home/taehyun/projects/DDPD/.venv`.

```text
ai2thor==5.0.0
procthor==0.0.1.dev2
attrs==26.1.0
pandas==2.3.3
shapely==2.1.2
python-fcl==0.7.0.11
scipy==1.15.3
moviepy==1.0.3
minerl: not-installed
minedojo: not-installed
habitat-sim: not-installed
deepmind_lab: not-installed
```

System tools:

```text
xvfb-run: available
git/gcc/g++: available
bazel: missing
conda/mamba: missing
java: missing
home disk: 624G free
AI2-THOR release cache: 3.1G
```

## ProcTHOR Smoke Test

ProcTHOR was selected as the next most practical candidate because it is
compatible with AI2-THOR and should reuse the existing CloudRendering and WIT-VZ
collection path.

Installed packages:

```bash
pip install procthor attrs pandas shapely "moviepy<2" python-fcl scipy
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

Observed result:

```text
seeds 1-10: AssertionError Unable to CreateHouse!
```

Interpretation:

```text
This is not a GPU/Vulkan failure anymore. CloudRendering starts, and the
ProcTHOR generator reaches the procedural house creation path. The blocker is
inside ProcTHOR/AI2-THOR procedural house compatibility. The likely issue is
version skew between the older procthor PyPI package and the AI2-THOR 5.0
CloudRendering build/branch used by the current environment.
```

Next practical route:

```text
1. Create a separate pinned ProcTHOR environment rather than reusing the current venv.
2. Use the ProcTHOR repository's expected AI2-THOR branch/build.
3. Verify one generated house with CreateHouse.
4. Only then add a dedicated collect_procthor_wit_vz.py wrapper.
```

## DeepMind Lab Gate

Why it is still useful:

```text
DeepMind Lab is the most game-like candidate after ViZDoom: first-person 3D
levels, navigation tasks, RGB observations, and action-based movement.
```

Current blocker:

```text
bazel: missing on gpuserver3090
deepmind_lab: no direct pip candidate in the current venv check
```

Required route:

```text
1. Install Bazel and DeepMind Lab system build dependencies.
2. Build DeepMind Lab from source or use a known wheel/Docker image.
3. Run a random-agent level with RGB and velocity/pose observations.
4. Convert frames plus pose deltas to WIT-VZ raw schema.
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
```

Mention the remaining candidates as future work:

```text
ProcTHOR is the nearest next demo but currently blocked by procedural
CreateHouse compatibility. DeepMind Lab, Habitat, and MineRL/MineDojo need
separate simulator environments before they can become fair WIT-VZ demos.
```

## Source Pointers

- ProcTHOR repository: https://github.com/allenai/procthor
- AI2-THOR CloudRendering docs: https://ai2thor.allenai.org/ithor/documentation/
- DeepMind Lab repository: https://github.com/google-deepmind/lab
- Habitat-Sim repository: https://github.com/facebookresearch/habitat-sim
- MineRL first-agent docs: https://minerl.readthedocs.io/en/latest/tutorials/first_agent.html
- MineDojo repository: https://github.com/MineDojo/MineDojo
