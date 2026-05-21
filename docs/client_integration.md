# Unity Client And GPU Inference Server

This branch adds a thin Unity client and a Python inference server.

## Intended Runtime Split

- GPU server: loads the PyTorch checkpoint, runs DINO/model inference, exposes HTTP.
- Unity client: captures recent ego-motion and optional RGB frames, calls the GPU server, renders the returned path.

Unity should not load PyTorch directly for the first prototype.

## Start The Server

Install the server dependencies on the SSH GPU server:

```bash
python -m pip install -e ".[server,dinov3]" pytest
```

Run an ego-motion-only or small-CNN checkpoint:

```bash
python -m server.inference_api \
  --checkpoint runs/path_to_checkpoint/best.pt \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

For a checkpoint trained with cached DINOv3 tokens, the default
`--backbone-override auto` maps the cached backbone to live
`dinov3-convnext-tiny` so the server can accept Unity RGB frames:

```bash
python -m server.inference_api \
  --checkpoint runs/horizon_sweep_v2/dinov3_strnet_01s/best.pt \
  --device cuda \
  --backbone-override auto \
  --host 0.0.0.0 \
  --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ego-motion smoke request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ego_history":[[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0]]}'
```

Visual-model requests should include `rgb_frames`, a list of base64 PNG frames.
The Unity scaffold does this when `Send RGB Frames` is enabled.

## Unity Setup

Open this folder in Unity:

```text
client/unity_path_client
```

Create an empty GameObject and attach:

```text
Assets/Scripts/PathPredictionClient.cs
```

Assign:

- `Agent Root`: the player/camera rig transform
- `Source Camera`: the camera used for RGB capture
- `Predicted Path Line`: a `LineRenderer`
- `Server URL`: `http://GPU_SERVER_IP:8000/predict`

For initial connectivity testing, leave `Send RGB Frames` disabled and use a
constant-velocity or ego-motion-only checkpoint. For the actual visual path
model, enable `Send RGB Frames`.

## Network Notes

If the GPU server is behind SSH only, forward the port from the local machine:

```bash
ssh -L 8000:127.0.0.1:8000 USER@GPU_SERVER
```

Then set Unity's `Server URL` to:

```text
http://127.0.0.1:8000/predict
```

This is usually safer than opening port `8000` to the network.
