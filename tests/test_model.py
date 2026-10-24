"""Unit tests for 3-Layer CNN Architecture."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from src.model import (
    build_three_layer_cnn,
    count_conv2d_layers,
    export_model_metadata,
    validate_conv2d_count,
)


@pytest.fixture
def cnn_model():
    """Build a standard 3-layer CNN model for testing."""
    return build_three_layer_cnn(
        input_shape=(128, 128, 3),
        conv1_filters=32,
        conv2_filters=64,
        conv3_filters=128,
        dense_units=128,
        dropout_rate=0.5,
        learning_rate=0.001,
    )


def test_model_has_exactly_three_conv2d_layers(cnn_model):
    """Verify strictly three Conv2D layers exist in the architecture."""
    num_conv2d = count_conv2d_layers(cnn_model)
    assert num_conv2d == 3, f"Expected 3 Conv2D layers, but found {num_conv2d}"
    assert validate_conv2d_count(cnn_model, expected_count=3) is True


def test_model_fails_validation_if_conv_count_mismatches():
    """Ensure validator raises ValueError if Conv2D count is altered."""
    inputs = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Conv2D(16, (3, 3))(inputs)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    bad_model = keras.Model(inputs=inputs, outputs=outputs)

    with pytest.raises(ValueError, match="Architecture violation"):
        validate_conv2d_count(bad_model, expected_count=3)


def test_model_input_and_output_shapes(cnn_model):
    """Check input and output dimensions."""
    assert cnn_model.input_shape == (None, 128, 128, 3)
    assert cnn_model.output_shape == (None, 1)


def test_model_forward_pass_range(cnn_model):
    """Ensure forward pass output is a valid probability bounded in [0, 1]."""
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 128, 128, 3)).astype(np.float32)
    predictions = cnn_model.predict(dummy_input, verbose=0)

    assert predictions.shape == (2, 1)
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


def test_model_save_and_load(cnn_model):
    """Ensure model can be saved to native .keras format and reloaded cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_path = Path(tmp_dir) / "test_model.keras"
        cnn_model.save(tmp_model_path)
        assert tmp_model_path.is_file()

        loaded_model = keras.models.load_model(tmp_model_path)
        assert count_conv2d_layers(loaded_model) == 3

        dummy_input = np.random.uniform(0.0, 1.0, size=(1, 128, 128, 3)).astype(np.float32)
        orig_pred = cnn_model.predict(dummy_input, verbose=0)
        loaded_pred = loaded_model.predict(dummy_input, verbose=0)
        np.testing.assert_allclose(orig_pred, loaded_pred, rtol=1e-5, atol=1e-5)


def test_export_model_metadata(cnn_model):
    """Verify metadata generation produces correct structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_json = Path(tmp_dir) / "meta.json"
        meta = export_model_metadata(cnn_model, out_json)

        assert meta["conv2d_layer_count"] == 3
        assert meta["total_parameters"] > 0
        assert out_json.is_file()
