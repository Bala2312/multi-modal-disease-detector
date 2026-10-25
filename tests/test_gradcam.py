import numpy as np
import torch
from PIL import Image
from torch import nn

from gradcam_engine import GradCAM


class _TinyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1), nn.ReLU())
        self.classifier = nn.Linear(4, 2)

    def forward(self, image):
        features = self.features(image)
        return self.classifier(features.mean(dim=(2, 3)))


def test_gradcam_saves_original_heatmap_and_overlay_at_input_resolution(tmp_path):
    image = Image.new("RGB", (37, 29), color=(70, 120, 160))
    model = _TinyVisionModel()
    with GradCAM(model, "features.0", device="cpu", output_dir=tmp_path) as cam:
        result = cam.generate(image, target_class_index=1)

    assert result.output_path.is_file()
    assert result.original_path.is_file()
    assert result.heatmap_path.is_file()
    assert result.heatmap.shape == (29, 37)
    assert np.isfinite(result.heatmap).all()
    with Image.open(result.output_path) as overlay:
        assert overlay.size == (37, 29)
    with Image.open(result.heatmap_path) as heatmap:
        assert heatmap.size == (37, 29)
