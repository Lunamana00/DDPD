# Game Data Strategy

The primary benchmark should stay close to real first-person game dynamics.
Synthetic Unity data is useful only when it deliberately tests visual-context
dependence rather than replacing ViZDoom.

## Track A: ViZDoom Game Benchmark

Use the capstone augmented multi-source config:

```powershell
uv run python scripts/collect_wit_vz_game_benchmark.py `
  --config configs/wit_vz_capstone_augmented.json `
  --overwrite
```

This collects scripted runs from:

- `deadly_corridor` with corridor-biased control
- `deadly_corridor` with random control
- `health_gathering`
- `health_gathering_supreme`
- `my_way_home`
- `take_cover`
- `predict_position`

Then it builds:

```text
data/wit_vz/processed/wit_vz_capstone_augmented_001
```

The default processed set uses a 1-second future horizon so short combat
episodes still contribute samples. For 3/5/10/30-second analysis, rebuild from
the same raw runs with `scripts/run_horizon_sweep.py`.

Use a smoke run first:

```powershell
uv run python scripts/collect_wit_vz_game_benchmark.py `
  --config configs/wit_vz_capstone_augmented.json `
  --dry-run `
  --limit-runs 1 `
  --episodes 2
```

The older smaller benchmark is still available:

```powershell
uv run python scripts/collect_wit_vz_game_benchmark.py --overwrite
```

This collects:

- `deadly_corridor` with corridor-biased control
- `deadly_corridor` with random control
- `health_gathering`
- `my_way_home`
- `take_cover`

Then it builds:

```text
data/wit_vz/processed/wit_vz_game_benchmark_001
```

Use a smoke run first:

```powershell
uv run python scripts/collect_wit_vz_game_benchmark.py --dry-run --limit-runs 1 --episodes 2
```

## Track B: Human ViZDoom Hold-Out

Human data should be used as the main held-out validation source, not mixed
blindly into every training split. The point is to test whether the model
learned visual context beyond a scripted controller.

Record a short human session:

```powershell
uv run python scripts/record_vizdoom_human_session.py `
  --scenario deadly_corridor `
  --run-id wit_vz_human_p001_s001 `
  --player-id p001 `
  --episodes 5 `
  --max-steps 900 `
  --overwrite
```

The script opens a visible ViZDoom window in `SPECTATOR` mode, records RGB,
pose, relative ego-motion, reward, and the last human action vector, then builds:

```text
data/wit_vz/processed/wit_vz_human_p001_s001
```

Recommended split discipline:

- train: scripted augmented runs
- validation: held-out scripted episodes or scenarios
- final test: human sessions held out by player/session
- optional mixed training: add only some human sessions after reporting the
  scripted-only result

## Track C: Game-Like Unity Synthetic Data

Unity synthetic data should be treated as a controlled ablation source. Its job
is to answer whether visual cues change future-path prediction when recent
ego-motion is ambiguous.

The Unity generator now defaults to a game-like branch corridor profile:

- darker enclosed dungeon-like corridors
- wall panels and ceiling bands
- branch doors and false doors
- color-coded turn cue panels before junctions
- pickups, crates, hazards, and enemy silhouettes
- local point lights and stronger visual contrast

Generate it with:

```powershell
.\scripts\generate_unity_dataset.ps1 -RunId unity_game_synthetic_001 -Episodes 40 -FramesPerEpisode 260 -CaptureSize 160
```

## External Dataset Candidates

| Source | Fit | Notes |
|---|---:|---|
| AI2-THOR / ProcTHOR | High | Unity-backed indoor embodied navigation. AI2-THOR returns egocentric RGB frames and agent pose metadata, so it can be converted to the WIT-VZ schema. |
| DeepMind Lab | High | Game-like first-person 3D navigation, but setup is more natural on Linux than Windows. |
| Habitat / HM3D / HSSD | Medium | Strong for navigation and pose-rendered trajectories, but less game-like and often requires separate scene asset access. |
| MineRL Navigate | Medium | Real human Minecraft demonstrations with video/actions, but pose/local trajectory availability is weaker than simulator-generated data. |
| CARLA | Medium | Excellent RGB plus pose/transform support, but it is a driving domain rather than FPS navigation. |

For this project, the best next external adapter is AI2-THOR/ProcTHOR because it
is Unity-based, controllable, and provides both RGB and agent pose.

An AI2-THOR collector scaffold is available:

```powershell
uv pip install ai2thor
uv run python scripts/collect_ai2thor_wit_vz.py --run-id ai2thor_nav_001 --overwrite
```

Then convert it with the existing builder:

```powershell
uv run python -m src.wit_vz.build_samples `
  --raw data/wit_vz/raw/ai2thor_nav_001 `
  --out data/wit_vz/processed/ai2thor_nav_001 `
  --history-sec 1 `
  --future-sec 3 `
  --sample-fps 1 `
  --stride 1 `
  --split episode
```

References:

- AI2-THOR event metadata exposes egocentric RGB frames and agent pose:
  https://ai2thor.allenai.org/ithor/documentation/environment-state/
- ProcTHOR provides open-source procedural house generation and the ProcTHOR-10K
  houses dataset: https://procthor.allenai.org/
- Habitat can render observations at agent states and compute shortest paths:
  https://aihabitat.org/docs/habitat-lab/habitat.core.simulator.Simulator.html
- DeepMind Lab is a first-person game-like 3D platform:
  https://deepmind.google/blog/open-sourcing-deepmind-lab/
- MineRL contains Minecraft human demonstrations with video feed and actions:
  https://zenodo.org/records/12659939
- CARLA RGB sensors provide sensor transforms with each measurement:
  https://carla.readthedocs.io/en/0.9.7/cameras_and_sensors/
