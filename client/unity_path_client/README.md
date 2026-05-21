# Unity Path Client

This is a lightweight Unity client scaffold for the DDPD GPU inference server.

The Unity side does not load PyTorch or DINO. It collects recent ego-motion,
optionally captures recent RGB frames, sends them to the GPU server over HTTP,
and renders the returned future local path with a `LineRenderer`.

## Open In Unity

Open this folder as a Unity project:

```text
client/unity_path_client
```

Fast demo setup:

1. Open any empty Unity scene.
2. Press Play.

If the scene does not already contain a `PathPredictionClient`, the bootstrap
script creates a complete demo setup automatically:

- high-contrast floor, walls, grid, obstacles, and a green reference route
- colored wall panels, floor bands, beacons, crates, and decision gates
- moving capsule agent
- first-person camera
- directional light
- predicted path line and waypoint markers

The client starts in `Demo Mode`, so it draws a local constant-velocity path
from recent motion history without the GPU server.

The visual cues are intentionally high contrast. They are there so the camera
stream contains route-relevant information instead of a blank corridor.

Dataset generation:

For training data, use `UnityDatasetGenerator.cs` instead of the debug demo.
It exports WIT-VZ-compatible raw episodes with RGB frames, poses, and
ego-motion labels. See:

```text
../../docs/unity_dataset_generator.md
```

GPU server setup:

1. Disable `Demo Mode`.
2. Keep `Fallback To Local Motion` enabled while testing connectivity.
3. Assign the camera to `Source Camera` if RGB frames should be sent.
4. Set `Server URL` to the GPU server endpoint, for example:

```text
http://GPU_SERVER_IP:8000/predict
```

For an ego-motion-only smoke test, leave `Send RGB Frames` disabled. For the
visual model, enable `Send RGB Frames` and run the server with a live RGB
backbone.

If the GPU server is reachable only through SSH, forward the port locally:

```bash
ssh -L 8000:127.0.0.1:8000 USER@GPU_SERVER
```

Then use:

```text
http://127.0.0.1:8000/predict
```
