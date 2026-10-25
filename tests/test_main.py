import pytest
from fastapi import HTTPException

from main import _decode_uploaded_image, _probability_uncertainty, _validate_modality, app, health


def test_health_endpoint_is_registered_and_lightweight() -> None:
    assert any(getattr(route, "path", None) == "/health" for route in app.routes)
    assert health()["status"] == "ok"


def test_analyze_rejects_unknown_modality() -> None:
    with pytest.raises(HTTPException) as error:
        _validate_modality("mri")
    assert error.value.status_code == 422


def test_analyze_rejects_non_image_upload() -> None:
    with pytest.raises(HTTPException) as error:
        _decode_uploaded_image(b"not an image")
    assert error.value.status_code == 415


def test_uncertainty_is_normalized_entropy() -> None:
    assert _probability_uncertainty({"positive": 0.5, "negative": 0.5}) == pytest.approx(1.0)
    assert _probability_uncertainty({"positive": 0.999999, "negative": 0.000001}) < 0.001
