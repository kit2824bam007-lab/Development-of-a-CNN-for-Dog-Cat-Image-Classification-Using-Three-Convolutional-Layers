"""Inference and single-image prediction module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Union

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow import keras

from src.config import AppConfig, load_config
from src.model import validate_conv2d_count
from src.utils import setup_logger

logger = setup_logger("predict")


def preprocess_image_for_inference(
    image_input: Union[str, Path, Image.Image, bytes],
    target_height: int = 128,
    target_width: int = 128,
) -> np.ndarray:
    """Read, validate, convert to RGB, resize, and normalize an input image.

    Returns:
        np.ndarray of shape (1, target_height, target_width, 3) with float32 values in [0, 1].
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        if p.stat().st_size == 0:
            raise ValueError(f"Image file is 0 bytes: {p}")
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Cannot decode image file {p}: {str(e)}")
    elif isinstance(image_input, bytes):
        import io
        try:
            with Image.open(io.BytesIO(image_input)) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot decode raw image bytes: {str(e)}")
    elif isinstance(image_input, Image.Image):
        img_rgb = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Resize using high-quality bilinear interpolation
    img_resized = img_rgb.resize((target_width, target_height), Image.Resampling.BILINEAR)

    # Convert to float array normalized to [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    # Expand batch dimension: (1, height, width, 3)
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor


class DogCatPredictor:
    """Production-ready inference wrapper for the Three-Layer CNN model."""

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        config: AppConfig | None = None,
    ):
        self.config = config if config else load_config()
        self.model_path = Path(model_path) if model_path else self.config.paths.model_checkpoint

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model checkpoint not found at: {self.model_path}. "
                "Please train the model before running inference."
            )

        logger.info(f"Loading model checkpoint from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        validate_conv2d_count(self.model)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
    ) -> Dict[str, Any]:
        """Perform classification inference on a single image.

        Returns:
            Dictionary with prediction class, confidence score, raw output, and per-class probabilities.
        """
        processed_tensor = preprocess_image_for_inference(
            image_input=image_input,
            target_height=self.config.image.height,
            target_width=self.config.image.width,
        )

        raw_pred = self.model.predict(processed_tensor, verbose=0)
        prob_dog = float(raw_pred[0][0])
        prob_cat = 1.0 - prob_dog

        # Threshold: < 0.5 is Cat (0), >= 0.5 is Dog (1)
        if prob_dog >= 0.5:
            predicted_class = "Dog"
            predicted_label = 1
            confidence = prob_dog
        else:
            predicted_class = "Cat"
            predicted_label = 0
            confidence = prob_cat

        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": {
                "Cat": round(prob_cat, 4),
                "Dog": round(prob_dog, 4),
            },
            "raw_output": prob_dog,
        }


def main() -> None:
    """CLI prediction entrypoint."""
    parser = argparse.ArgumentParser(description="Classify an image as Cat or Dog using the 3-Layer CNN.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model checkpoint.")
    args = parser.parse_args()

    predictor = DogCatPredictor(model_path=args.model)
    result = predictor.predict(args.image)

    print("\n" + "=" * 50)
    print("           PREDICTION RESULT")
    print("=" * 50)
    print(f"Predicted Class:        {result['predicted_class']}")
    print(f"Confidence:             {result['confidence_percentage']}%")
    print(f"Probability (Cat):      {result['probabilities']['Cat']:.4f}")
    print(f"Probability (Dog):      {result['probabilities']['Dog']:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
