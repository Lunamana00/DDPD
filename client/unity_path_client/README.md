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

Recommended scene setup:

1. Add an empty GameObject named `PathPredictionClient`.
2. Attach `PathPredictionClient.cs`.
3. Assign the player/agent transform to `Agent Root`.
4. Assign the camera to `Source Camera` if RGB frames should be sent.
5. Add a `LineRenderer` and assign it to `Predicted Path Line`.
6. Set `Server URL` to the GPU server endpoint, for example:

```text
http://GPU_SERVER_IP:8000/predict
```

For an ego-motion-only smoke test, leave `Send RGB Frames` disabled. For the
visual model, enable `Send RGB Frames` and run the server with a live RGB
backbone.
