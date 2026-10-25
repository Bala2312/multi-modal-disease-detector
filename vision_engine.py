"""Modality routing and multi-label vision inference for the capstone prototype.

The default modality detector is a lightweight image-statistics heuristic. It is
not a validated medical-image classifier; production callers should supply the
modality explicitly or replace the heuristic with a trained router.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import torch
from PIL import Image, ImageStat
from torch import nn
from torchvision import models, transforms


class Modality(str, Enum):
    """Supported image modalities."""

    CHEST_XRAY = "chest_xray"
    SKIN_LESION = "skin_lesion"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = (224, 224)


def build_image_transform() -> transforms.Compose:
    """Create the shared resize and ImageNet normalization transform."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(
    modality: Modality | str,
    num_labels: int,
    *,
    pretrained: bool = False,
) -> nn.Module:
    """Build the requested torchvision backbone with a multi-label logit head.

    Sigmoid is applied by :class:`VisionEngine` at inference time, so the model
    returns logits suitable for ``BCEWithLogitsLoss`` during training.

    ``pretrained=True`` initializes the backbone with ImageNet weights only;
    the replacement disease-classification head is always newly initialized.
    """
    modality = Modality(modality)
    if num_labels <= 0:
        raise ValueError("num_labels must be a positive integer")

    if modality is Modality.CHEST_XRAY:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_labels)
        return model

    weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_labels)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    """Load a model state dict from a checkpoint saved by this project.

    The checkpoint may be either a raw state dict or a mapping containing a
    ``model_state_dict`` entry. Loading is strict so incompatible label heads
    or architectures fail visibly instead of producing misleading outputs.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # ``weights_only`` is absent in older supported PyTorch builds.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint must contain a model state dict")
    model.load_state_dict(state_dict, strict=True)
    return model


def detect_modality(image: Image.Image) -> Modality:
    """Estimate modality from grayscale/saturation statistics.

    This rule treats grayscale or low-saturation images as chest X-rays and
    other images as skin lesions. It is intentionally simple for scaffolding
    and can fail on colorized radiographs, grayscale dermoscopy, screenshots,
    or atypical acquisition pipelines. Pass an explicit modality to inference
    whenever it is known.
    """
    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    mean_saturation = ImageStat.Stat(hsv.getchannel("S")).mean[0]
    # A low mean saturation is a useful rough cue, not a clinical modality test.
    if mean_saturation < 28.0:
        return Modality.CHEST_XRAY
    return Modality.SKIN_LESION


@dataclass(frozen=True)
class VisionPrediction:
    """Per-image output from one modality-specific model."""

    modality: Modality
    probabilities: dict[str, float]
    decision: str | None = None

    @property
    def top_label(self) -> str:
        """Return thresholded binary decision or highest multi-label score."""
        return self.decision or max(self.probabilities, key=self.probabilities.get)

    @property
    def top_score(self) -> float:
        """Highest independent sigmoid probability."""
        return self.probabilities[self.top_label]


class VisionEngine:
    """Run a modality-specific sigmoid multi-label or softmax binary model.

    ``labels`` must describe the trained checkpoint output order for each
    modality. ``output_modes`` is normally inferred from checkpoint metadata by
    the application; omitted modes retain the original sigmoid behavior.
    """

    def __init__(
        self,
        xray_model: nn.Module,
        skin_model: nn.Module,
        labels: Mapping[Modality | str, Sequence[str]],
        *,
        device: str | torch.device | None = None,
        output_modes: Mapping[Modality | str, str] | None = None,
        decision_thresholds: Mapping[Modality | str, float] | None = None,
        positive_labels: Mapping[Modality | str, str] | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.models = {
            Modality.CHEST_XRAY: xray_model.to(self.device).eval(),
            Modality.SKIN_LESION: skin_model.to(self.device).eval(),
        }
        self.labels = {Modality(key): tuple(value) for key, value in labels.items()}
        self.output_modes = {key: "multilabel" for key in self.labels}
        if output_modes:
            for key, value in output_modes.items():
                if value not in {"multilabel", "binary"}:
                    raise ValueError(f"Unsupported output mode for {key}: {value}")
                self.output_modes[Modality(key)] = value
        self.decision_thresholds = {key: 0.5 for key in self.labels}
        if decision_thresholds:
            for key, value in decision_thresholds.items():
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"Decision threshold for {key} must be within [0, 1]")
                self.decision_thresholds[Modality(key)] = float(value)
        self.positive_labels = {Modality(key): value for key, value in (positive_labels or {}).items()}
        for modality, model in self.models.items():
            if modality not in self.labels or not self.labels[modality]:
                raise ValueError(f"Provide at least one trained output label for {modality.value}")
            output_layer = self._classifier_layer(model, modality)
            if output_layer.out_features != len(self.labels[modality]):
                raise ValueError(
                    f"{modality.value} model has {output_layer.out_features} outputs, "
                    f"but {len(self.labels[modality])} labels were provided"
                )
            if self.output_modes[modality] == "binary" and output_layer.out_features != 2:
                raise ValueError(f"Binary checkpoint for {modality.value} must have exactly two outputs")
            if self.output_modes[modality] == "binary":
                if modality not in self.positive_labels:
                    # Preserve sensible defaults for direct callers; serving
                    # loads the exact label and threshold from checkpoint metadata.
                    preferred = next((name for name in ("Pneumonia", "MEL")
                                      if name in self.labels[modality]), self.labels[modality][-1])
                    self.positive_labels[modality] = preferred
                if self.positive_labels[modality] not in self.labels[modality]:
                    raise ValueError(f"Binary positive label is absent from the {modality.value} label order")

        self.transform = build_image_transform()

    @staticmethod
    def _classifier_layer(model: nn.Module, modality: Modality) -> nn.Linear:
        if modality is Modality.CHEST_XRAY and hasattr(model, "classifier"):
            layer = model.classifier
        elif modality is Modality.SKIN_LESION and hasattr(model, "classifier"):
            classifier = model.classifier
            layer = classifier[-1] if isinstance(classifier, nn.Sequential) else classifier
        else:
            raise TypeError(f"Model does not expose a supported {modality.value} classifier")
        if not isinstance(layer, nn.Linear):
            raise TypeError("The model's final classifier must be torch.nn.Linear")
        return layer

    def predict(
        self,
        image: Image.Image,
        modality: Modality | str | None = None,
    ) -> VisionPrediction:
        """Return independent per-label probabilities for one image.

        ``modality`` can be supplied by the caller to bypass the heuristic.
        Input images are converted to RGB before preprocessing so grayscale
        images are represented consistently with the ImageNet backbone.
        """
        selected_modality = Modality(modality) if modality is not None else detect_modality(image)
        model = self.models[selected_modality]
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = model(batch)
            probability_tensor = (
                torch.softmax(logits, dim=1) if self.output_modes[selected_modality] == "binary"
                else torch.sigmoid(logits)
            )
            probabilities = probability_tensor[0].detach().cpu().tolist()

        probabilities_by_label = {
            label: float(score)
            for label, score in zip(self.labels[selected_modality], probabilities, strict=True)
        }
        decision = None
        if self.output_modes[selected_modality] == "binary":
            positive_label = self.positive_labels[selected_modality]
            negative_label = next(label for label in self.labels[selected_modality] if label != positive_label)
            decision = (positive_label if probabilities_by_label[positive_label] >=
                        self.decision_thresholds[selected_modality] else negative_label)
        return VisionPrediction(
            modality=selected_modality,
            probabilities=probabilities_by_label,
            decision=decision,
        )
