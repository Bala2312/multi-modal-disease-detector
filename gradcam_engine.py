"""Grad-CAM generation for the capstone's DenseNet and EfficientNet models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from vision_engine import Modality, build_image_transform


DEFAULT_TARGET_LAYERS = {
    Modality.CHEST_XRAY: "features.denseblock4",
    Modality.SKIN_LESION: "features.7",
}


@dataclass(frozen=True)
class GradCAMResult:
    """Generated Grad-CAM overlay and its source heatmap."""

    output_path: Path
    heatmap: np.ndarray
    target_class_index: int
    target_score: float


class GradCAM:
    """Create Grad-CAM overlays for a model and a convolutional target layer.

    The forward hook stores activations and attaches a PyTorch tensor backward
    hook to the target layer output. This captures the gradient flowing into
    that output during backpropagation without a module backward hook's
    interaction with torchvision's in-place activations.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str | nn.Module,
        *,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        device: str | torch.device | None = None,
        output_dir: str | Path = "outputs/heatmaps",
        alpha: float = 0.40,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        self.model = model
        self.device = torch.device(device) if device is not None else self._model_device(model)
        self.model.to(self.device).eval()
        self.transform = transform or build_image_transform()
        self.output_dir = Path(output_dir)
        self.alpha = alpha
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.target_layer = self._resolve_target_layer(target_layer)
        self._forward_handle = self.target_layer.register_forward_hook(self._capture_forward)

    @staticmethod
    def _model_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _resolve_target_layer(self, target_layer: str | nn.Module) -> nn.Module:
        if isinstance(target_layer, nn.Module):
            return target_layer
        modules = dict(self.model.named_modules())
        if target_layer not in modules:
            supported = ", ".join(DEFAULT_TARGET_LAYERS.values())
            raise ValueError(
                f"Target layer {target_layer!r} was not found in the model. "
                f"Example torchvision paths: {supported}"
            )
        return modules[target_layer]

    def _capture_forward(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError("Grad-CAM target layer must output a 4D feature tensor (N, C, H, W)")
        self.activations = output

        def capture_gradient(gradient: torch.Tensor) -> None:
            self.gradients = gradient

        output.register_hook(capture_gradient)

    def generate(
        self,
        image: Image.Image,
        target_class_index: int,
        *,
        output_path: str | Path | None = None,
    ) -> GradCAMResult:
        """Generate and save an overlay for one output class.

        The class score is the selected model logit. For sigmoid multi-label
        heads, its gradient has the same sign as the corresponding sigmoid
        probability's gradient, without sigmoid saturation reducing gradient
        magnitude.
        """
        source = image.convert("RGB")
        input_tensor = self.transform(source).unsqueeze(0).to(self.device).requires_grad_(True)
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("Expected model output with shape (1, num_labels)")
        if not 0 <= target_class_index < logits.shape[1]:
            raise IndexError(
                f"target_class_index {target_class_index} is outside [0, {logits.shape[1] - 1}]"
            )

        target_score = logits[0, target_class_index]
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Target-layer activations or gradients were not captured")

        # Global-average-pool gradients over height and width, then weight each
        # activation channel and sum the channels to form the class map.
        channel_weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (channel_weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(source.height, source.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam.detach().cpu().numpy()
        maximum = float(cam.max()) if cam.size else 0.0
        if maximum > 0:
            cam = cam / maximum
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        heatmap_u8 = np.uint8(np.clip(cam * 255.0, 0, 255))
        # OpenCV's colormap returns BGR; convert to RGB before blending with PIL.
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        original_rgb = np.asarray(source, dtype=np.uint8)
        overlay = cv2.addWeighted(original_rgb, 1.0 - self.alpha, heatmap_rgb, self.alpha, 0)

        destination = Path(output_path) if output_path is not None else self._new_output_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not save Grad-CAM overlay to {destination}")

        return GradCAMResult(
            output_path=destination,
            heatmap=cam,
            target_class_index=target_class_index,
            target_score=float(target_score.detach().cpu()),
        )

    def _new_output_path(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.output_dir / f"gradcam_{timestamp}_{uuid4().hex[:8]}.jpg"

    def close(self) -> None:
        """Remove the registered hook when this Grad-CAM object is no longer used."""
        self._forward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def default_target_layer(modality: Modality | str) -> str:
    """Return the recommended final convolutional feature layer for a modality."""
    return DEFAULT_TARGET_LAYERS[Modality(modality)]
