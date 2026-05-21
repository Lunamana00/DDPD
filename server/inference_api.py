"""HTTP inference API for the egocentric path predictor.

The Unity client is intentionally thin: it sends recent ego-motion and,
optionally, recent RGB frames. The GPU server owns model loading and inference.
"""

from __future__ import annotations

import argparse
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.factory import create_model


_CACHED_TO_LIVE_BACKBONE = {
    "cached_dinov3": "dinov3-convnext-tiny",
    "cached_dinov3_convnext_tiny": "dinov3-convnext-tiny",
    "cached_dinov3-convnext-tiny": "dinov3-convnext-tiny",
    "cached_timm_dinov3_convnext_tiny": "dinov3-convnext-tiny",
    "cached_features": "dinov3-convnext-tiny",
}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_backbone(checkpoint: dict[str, Any], override: str | None) -> str:
    checkpoint_backbone = str(checkpoint.get("backbone", "small_cnn"))
    if override is None or override == "checkpoint":
        return checkpoint_backbone
    if override == "auto":
        return _CACHED_TO_LIVE_BACKBONE.get(checkpoint_backbone.lower(), checkpoint_backbone)
    return override


def create_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    backbone_override: str | None = "auto",
) -> torch.nn.Module:
    model_name = str(checkpoint["model_name"])
    backbone_name = resolve_backbone(checkpoint, backbone_override)
    return create_model(
        model_name,
        future_steps=int(checkpoint["future_steps"]),
        backbone_name=backbone_name,
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        freeze_backbone=bool(checkpoint.get("freeze_backbone", True)),
        num_cue_tokens=int(checkpoint.get("num_cue_tokens", 8)),
        num_modes=int(checkpoint.get("num_modes", 1)),
        temporal_type=str(checkpoint.get("temporal_type", "transformer")),
        temporal_layers=int(checkpoint.get("temporal_layers", 1)),
        selector_layers=int(checkpoint.get("selector_layers", 1)),
        decoder_layers=int(checkpoint.get("decoder_layers", 1)),
        cue_temporal_layers=int(checkpoint.get("cue_temporal_layers", 0)),
        tokenlearner_pooling=str(checkpoint.get("tokenlearner_pooling", "softmax")),
        selector_type=str(checkpoint.get("selector_type", "query_attention")),
        memory_type=str(checkpoint.get("memory_type", "gru_cell")),
        use_spatial_graph=bool(checkpoint.get("use_spatial_graph", False)),
        spatial_graph_neighbors=int(checkpoint.get("spatial_graph_neighbors", 8)),
        use_temporal_difference_conv=bool(checkpoint.get("use_temporal_difference_conv", False)),
        use_temporal_shift=bool(checkpoint.get("use_temporal_shift", False)),
        dropout=float(checkpoint.get("dropout", 0.1)),
        use_constant_velocity_residual=bool(checkpoint.get("use_constant_velocity_residual", False)),
        residual_scale=float(checkpoint.get("residual_scale", checkpoint.get("trajectory_scale", 1.0))),
    )


def _strip_data_url_prefix(encoded: str) -> str:
    if "," in encoded and encoded.lower().startswith("data:"):
        return encoded.split(",", 1)[1]
    return encoded


def decode_rgb_frame(encoded: str, image_size: int) -> torch.Tensor:
    raw = base64.b64decode(_strip_data_url_prefix(encoded))
    image = Image.open(io.BytesIO(raw)).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def ego_history_to_tensor(ego_history: Any, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(ego_history, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError("ego_history must have shape [T, 3] or [B, T, 3]")
    return tensor


def rgb_frames_to_tensor(frames: list[str], image_size: int, device: torch.device) -> torch.Tensor:
    if not frames:
        raise ValueError("rgb_frames cannot be empty")
    tensors = [decode_rgb_frame(frame, image_size) for frame in frames]
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device)


def visual_tokens_to_tensor(tokens: Any, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(tokens, dtype=torch.float32, device=device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("visual_tokens must have shape [T, N, C] or [B, T, N, C]")
    return tensor


def tensor_path_to_points(path: torch.Tensor) -> list[dict[str, float]]:
    return [
        {"forward": float(point[0]), "right": float(point[1])}
        for point in path.detach().cpu()
    ]


def tensor_path_to_xy(path: torch.Tensor) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in path.detach().cpu()]


@dataclass
class PathPredictorService:
    model: torch.nn.Module
    device: torch.device
    image_size: int = 64
    checkpoint_path: Path | None = None
    backbone: str | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        image_size: int | None = None,
        backbone_override: str | None = "auto",
    ) -> "PathPredictorService":
        resolved_device = choose_device(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=resolved_device)
        model = create_model_from_checkpoint(checkpoint, backbone_override=backbone_override).to(resolved_device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        model.eval()
        resolved_image_size = int(image_size or checkpoint.get("image_size", 64))
        return cls(
            model=model,
            device=resolved_device,
            image_size=resolved_image_size,
            checkpoint_path=Path(checkpoint_path),
            backbone=resolve_backbone(checkpoint, backbone_override),
        )

    def build_batch(self, payload: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "ego_history" not in payload:
            raise ValueError("Missing required field: ego_history")
        batch: dict[str, torch.Tensor] = {
            "ego_history": ego_history_to_tensor(payload["ego_history"], self.device)
        }
        if "visual_tokens" in payload and payload["visual_tokens"] is not None:
            batch["visual_tokens"] = visual_tokens_to_tensor(payload["visual_tokens"], self.device)
        elif "rgb_frames" in payload and payload["rgb_frames"] is not None:
            batch["rgb_history"] = rgb_frames_to_tensor(
                list(payload["rgb_frames"]),
                int(payload.get("image_size", self.image_size)),
                self.device,
            )
        return batch

    @torch.inference_mode()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        batch = self.build_batch(payload)
        output = self.model(batch)
        response: dict[str, Any] = {
            "device": str(self.device),
            "image_size": self.image_size,
        }
        if self.checkpoint_path is not None:
            response["checkpoint"] = self.checkpoint_path.as_posix()
        if self.backbone is not None:
            response["backbone"] = self.backbone

        if isinstance(output, dict):
            paths = output["paths"][0]
            logits = output["logits"][0]
            confidence = torch.softmax(logits, dim=-1)
            selected_index = int(torch.argmax(confidence).detach().cpu())
            selected_path = paths[selected_index]
            response.update(
                {
                    "future_steps": int(selected_path.shape[0]),
                    "selected_mode": selected_index,
                    "mode_confidences": [float(value) for value in confidence.detach().cpu()],
                    "path": tensor_path_to_points(selected_path),
                    "path_xy": tensor_path_to_xy(selected_path),
                    "candidates": [tensor_path_to_xy(candidate) for candidate in paths],
                }
            )
            return response

        path = output[0]
        response.update(
            {
                "future_steps": int(path.shape[0]),
                "path": tensor_path_to_points(path),
                "path_xy": tensor_path_to_xy(path),
            }
        )
        return response


def create_app(service: PathPredictorService) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "FastAPI is required to run the HTTP server. "
            'Install it with: python -m pip install -e ".[server]"'
        ) from exc

    app = FastAPI(title="DDPD Path Prediction API")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "device": str(service.device),
            "image_size": service.image_size,
            "checkpoint": service.checkpoint_path.as_posix() if service.checkpoint_path else None,
            "backbone": service.backbone,
        }

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return service.predict(payload)
        except Exception as exc:  # pragma: no cover - FastAPI transport wrapper
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DDPD path prediction HTTP API.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--backbone-override",
        default="auto",
        help="'auto' maps cached DINOv3 checkpoints to live DINOv3 RGB inference.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - command-line dependency guard
        raise RuntimeError(
            "uvicorn is required to run the HTTP server. "
            'Install it with: python -m pip install -e ".[server]"'
        ) from exc

    service = PathPredictorService.from_checkpoint(
        args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        backbone_override=args.backbone_override,
    )
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
