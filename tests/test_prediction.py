"""Unit tests for single-image prediction logic and label consistency."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.model import build_three_layer_cnn
from src.predict import DogCatPredictor


@pytest.fixture
def trained_dummy_model_file():
    """Create a temporary model file saved to disk for predictor testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.keras"
        model = build_three_layer_cnn(input_shape=(128, 128, 3))
        model.save(model_path)
        yield str(model_path)


def test_prediction_output_structure(trained_dummy_model_file):
    """Verify inference returns the exact expected dictionary fields."""
    predictor = DogCatPredictor(model_path=trained_dummy_model_file)

    synthetic_image = Image.new("RGB", (128, 128), color=(100, 150, 200))
    result = predictor.predict(synthetic_image)

    assert "predicted_class" in result
    assert result["predicted_class"] in ["Cat", "Dog"]
    assert "predicted_label" in result
    assert result["predicted_label"] in [0, 1]

    # Check label to class mapping consistency
    if result["predicted_class"] == "Cat":
        assert result["predicted_label"] == 0
    else:
        assert result["predicted_label"] == 1

    assert "confidence" in result
    assert 0.5 <= result["confidence"] <= 1.0

    assert "probabilities" in result
    prob_sum = result["probabilities"]["Cat"] + result["probabilities"]["Dog"]
    assert abs(prob_sum - 1.0) < 1e-3


def test_predictor_handles_pil_and_bytes(trained_dummy_model_file):
    """Ensure predictor accepts both PIL Image objects and byte strings."""
    predictor = DogCatPredictor(model_path=trained_dummy_model_file)

    img = Image.new("RGB", (64, 64), color=(50, 60, 70))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    result_bytes = predictor.predict(img_bytes)
    assert result_bytes["predicted_class"] in ["Cat", "Dog"]
