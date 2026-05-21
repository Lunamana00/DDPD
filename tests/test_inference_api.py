import torch

from server.inference_api import PathPredictorService, resolve_backbone
from src.models.baselines import ConstantVelocityBaseline


def test_resolve_backbone_maps_cached_dinov3_to_live_backbone():
    checkpoint = {"backbone": "cached_dinov3_convnext_tiny"}
    assert resolve_backbone(checkpoint, "auto") == "dinov3-convnext-tiny"
    assert resolve_backbone(checkpoint, "checkpoint") == "cached_dinov3_convnext_tiny"
    assert resolve_backbone(checkpoint, "small_cnn") == "small_cnn"


def test_service_predicts_constant_velocity_payload():
    service = PathPredictorService(
        model=ConstantVelocityBaseline(future_steps=3),
        device=torch.device("cpu"),
    )
    response = service.predict(
        {
            "ego_history": [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
            ]
        }
    )

    assert response["future_steps"] == 3
    assert response["path_xy"] == [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]
    assert response["path"][0] == {"forward": 1.0, "right": 0.5}
