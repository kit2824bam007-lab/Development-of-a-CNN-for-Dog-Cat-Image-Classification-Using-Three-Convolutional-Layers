"""Unit tests for preprocessing and image validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data_preparation import validate_and_convert_image
from src.predict import preprocess_image_for_inference


def test_preprocess_synthetic_image():
    """Verify inference preprocessing produces exact shape and [0, 1] range."""
    # Create 200x150 synthetic RGB image
    img = Image.new("RGB", (200, 150), color=(120, 80, 40))

    tensor = preprocess_image_for_inference(img, target_height=128, target_width=128)

    assert tensor.shape == (1, 128, 128, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocess_grayscale_conversion():
    """Verify that a grayscale image is correctly converted to 3-channel RGB."""
    gray_img = Image.new("L", (100, 100), color=128)

    tensor = preprocess_image_for_inference(gray_img, target_height=128, target_width=128)

    assert tensor.shape == (1, 128, 128, 3)
    # Check all three channels are identical and equal to 128/255
    expected_val = 128.0 / 255.0
    np.testing.assert_allclose(tensor[0, :, :, 0], tensor[0, :, :, 1], atol=1e-3)
    np.testing.assert_allclose(tensor[0, :, :, 0], tensor[0, :, :, 2], atol=1e-3)
    assert abs(tensor[0, 50, 50, 0] - expected_val) < 0.01


def test_preprocess_nonexistent_file_raises():
    """Ensure FileNotFoundError is raised when file path is invalid."""
    with pytest.raises(FileNotFoundError):
        preprocess_image_for_inference("non_existent_image_path_12345.jpg")


def test_preprocess_zero_byte_file_raises():
    """Ensure ValueError is raised on 0-byte corrupt image file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="0 bytes"):
            preprocess_image_for_inference(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_validate_and_convert_corrupt_file():
    """Ensure corrupt header bytes are caught by validate_and_convert_image."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bad_file = Path(tmp_dir) / "corrupt.jpg"
        out_file = Path(tmp_dir) / "out.jpg"

        # Write invalid binary data
        with open(bad_file, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_AT_ALL")

        is_valid, msg = validate_and_convert_image(bad_file, out_file)
        assert is_valid is False
        assert "Corrupted or unsupported" in msg
        assert not out_file.exists()
