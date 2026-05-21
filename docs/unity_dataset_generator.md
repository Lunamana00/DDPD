# Unity Procedural Dataset Generator

The Unity client now includes a procedural raw-dataset generator for visual path
prediction experiments. It is separate from the lightweight visualization demo.

## What It Generates

The generator writes a WIT-VZ-compatible raw run:

```text
manifest.json
episodes/
  episode_0001/
    rgb/frame_000000.png
    ...
    steps.jsonl
    summary.json
```

Each `steps.jsonl` record includes:

- `step`
- `time_sec`
- `frame_path`
- `pose.x`, `pose.y`, `pose.angle`
- `relative_egomotion_from_prev.dx_forward`
- `relative_egomotion_from_prev.dy_right`
- `relative_egomotion_from_prev.dyaw`

That is the schema expected by `src.wit_vz.build_samples`.

## Scene Content

Each episode is seed-generated with:

- enclosed dungeon-like corridors with darker walls and ceiling ribs
- branch doors, false doors, and color-coded turn cue panels
- floor arrows before turn decisions
- pickups, crates, hazard-like props, and enemy silhouettes
- a route-following camera agent
- no explicit ground-truth route line unless `Show Debug Route Line` is enabled

The point is not photorealism. The point is to make RGB frames contain
route-relevant visual cues before the ego-motion history alone reveals the
upcoming turn, while keeping labels reproducible.

## Editor Usage

1. Open `client/unity_path_client` in Unity `6000.3.11f1`.
2. Use the menu item:

```text
DDPD > Generate Unity Raw Dataset
```

This writes directly to:

```text
data/wit_vz/raw/unity_game_synthetic_001
```

Manual setup is still possible:

1. Create an empty scene.
2. Add an empty GameObject named `UnityDatasetGenerator`.
3. Attach `UnityDatasetGenerator.cs`.
4. Configure:
   - `Run Id`
   - `Episode Count`
   - `Frames Per Episode`
   - `Sample Fps`
   - `Capture Width` / `Capture Height`
5. Enable `Generate On Start`, then press Play.

By default the raw run is written under Unity's `Application.persistentDataPath`
inside:

```text
DDPDUnityDataset/<run_id>
```

The Unity console prints the exact output path when generation completes.

## Generate And Convert

From the repo root on Windows:

```powershell
.\scripts\generate_unity_dataset.ps1
```

This runs Unity in batch mode, writes the raw run to
`data/wit_vz/raw/unity_game_synthetic_001`, checks `manifest.json`, then runs:

```bash
python -m src.wit_vz.build_samples \
  --raw data/wit_vz/raw/unity_game_synthetic_001 \
  --out data/wit_vz/processed/unity_game_synthetic_001 \
  --history-sec 1 \
  --future-sec 3 \
  --sample-fps 5 \
  --stride 2 \
  --split episode
```

The script accepts overrides:

```powershell
.\scripts\generate_unity_dataset.ps1 `
  -RunId unity_game_synthetic_002 `
  -Episodes 12 `
  -FramesPerEpisode 260 `
  -CaptureSize 160
```

## Batch Mode

The Unity project includes an editor command:

```text
DDPDUnityDatasetBatch.GenerateDefaultAndQuit
```

It creates an empty scene, creates a `DDPD Unity Dataset Generator` GameObject,
attaches `UnityDatasetGenerator`, generates the raw run, and exits.
