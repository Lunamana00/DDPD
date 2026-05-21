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

- high-contrast wall panels and floor cues
- colored landmarks, beacons, and crates
- obstacles placed near but not exactly on the route
- a route-following camera agent
- no explicit ground-truth route line unless `Show Debug Route Line` is enabled

The point is to make RGB frames contain route-relevant visual cues while keeping
labels reproducible.

## Editor Usage

1. Open `client/unity_path_client` in Unity `6000.3.11f1`.
2. Create an empty scene.
3. Add an empty GameObject named `UnityDatasetGenerator`.
4. Attach `UnityDatasetGenerator.cs`.
5. Configure:
   - `Run Id`
   - `Episode Count`
   - `Frames Per Episode`
   - `Sample Fps`
   - `Capture Width` / `Capture Height`
6. Enable `Generate On Start`, then press Play.

By default the raw run is written under Unity's `Application.persistentDataPath`
inside:

```text
DDPDUnityDataset/<run_id>
```

The Unity console prints the exact output path when generation completes.

## Convert To Processed Samples

Copy the generated raw run into the repo, for example:

```text
data/wit_vz/raw/unity_procedural_001
```

Then run:

```bash
python -m src.wit_vz.build_samples \
  --raw data/wit_vz/raw/unity_procedural_001 \
  --out data/wit_vz/processed/unity_procedural_001 \
  --history-sec 1 \
  --future-sec 3 \
  --sample-fps 5 \
  --stride 2 \
  --split episode
```

After that, the existing training scripts can consume the processed dataset.

## Batch Mode

The project includes a command-line hook:

```text
--ddpd-generate-dataset
```

When Unity is launched with that argument, the bootstrap creates a generator
object. This is intended for later automation from a Unity batch-mode command.
For now, the Editor workflow is the safer path because generator settings still
come from serialized fields.
