"""Three-Layer Convolutional Neural Network (CNN) Model Architecture.

Enforces strictly THREE Conv2D layers for binary Dog-Cat image classification.
Compatible with TensorFlow 2.16+ and Keras 3.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import AppConfig, ModelConfig, load_config
from src.utils import save_json, setup_logger

logger = setup_logger("model")


def build_three_layer_cnn(
    input_shape: tuple = (128, 128, 3),
    conv1_filters: int = 32,
    conv2_filters: int = 64,
    conv3_filters: int = 128,
    dense_units: int = 128,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Construct and compile the required 3-Layer Convolutional Neural Network.

    Layer Architecture:
    - Input: Explicit tensor (128, 128, 3) representing normalized RGB images.
    - Layer 1 (Low-level features):
        * Conv2D(32, (3, 3), padding='same', relu) -> extracts basic edges, textures, color gradients.
        * MaxPooling2D((2, 2)) -> downsamples spatial dimensions from 128x128 to 64x64.
    - Layer 2 (Mid-level features):
        * Conv2D(64, (3, 3), padding='same', relu) -> detects corners, curves, fur textures, contours.
        * MaxPooling2D((2, 2)) -> downsamples spatial dimensions from 64x64 to 32x32.
    - Layer 3 (High-level semantic features):
        * Conv2D(128, (3, 3), padding='same', relu) -> captures complex patterns (ears, snout, eyes, facial structures).
        * MaxPooling2D((2, 2)) -> downsamples spatial dimensions from 32x32 to 16x16.
    - Flatten: Converts 3D feature maps (16x16x128 = 32,768 elements) into 1D vector.
    - Dense(128, relu): Non-linear representation combining localized visual features.
    - Dropout(0.5): Regularization layer preventing co-adaptation and reducing overfitting.
    - Dense(1, sigmoid): Output probability in range [0, 1].
        * Probability < 0.5 -> Cat (Class 0)
        * Probability >= 0.5 -> Dog (Class 1)
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Convolutional Block 1
    x = layers.Conv2D(
        filters=conv1_filters,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_1",
    )(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1")(x)

    # Convolutional Block 2
    x = layers.Conv2D(
        filters=conv2_filters,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_2",
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2")(x)

    # Convolutional Block 3
    x = layers.Conv2D(
        filters=conv3_filters,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_3",
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_3")(x)

    # Fully-Connected Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(units=dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(rate=dropout_rate, name="dropout")(x)
    outputs = layers.Dense(units=1, activation="sigmoid", name="output_dense")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DogCat_3Layer_CNN")

    # Compile with Adam optimizer, binary cross-entropy, and accuracy
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    validate_conv2d_count(model)
    return model


def validate_conv2d_count(model: keras.Model, expected_count: int = 3) -> bool:
    """Strictly assert that the model contains exactly the specified number of Conv2D layers."""
    conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    actual_count = len(conv_layers)
    if actual_count != expected_count:
        raise ValueError(
            f"Architecture violation: Expected exactly {expected_count} Conv2D layers, "
            f"found {actual_count} Conv2D layers ({[l.name for l in conv_layers]})."
        )
    return True


def count_conv2d_layers(model: keras.Model) -> int:
    """Return the total number of Conv2D layers present in the model."""
    return sum(1 for layer in model.layers if isinstance(layer, layers.Conv2D))


def get_model_summary_string(model: keras.Model) -> str:
    """Return the string representation of model.summary()."""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


def export_model_metadata(model: keras.Model, output_path: Path | str) -> Dict[str, Any]:
    """Inspect model parameters, layer breakdown, and save metadata JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    metadata = {
        "model_name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(keras.backend.count_params(w) for w in model.trainable_weights)),
        "non_trainable_parameters": int(sum(keras.backend.count_params(w) for w in model.non_trainable_weights)),
        "conv2d_layer_count": len(conv_layers),
        "conv2d_layers": [
            {
                "name": l.name,
                "filters": l.filters,
                "kernel_size": list(l.kernel_size),
                "activation": l.activation.__name__,
                "padding": l.padding,
            }
            for l in conv_layers
        ],
        "all_layers": [
            {
                "name": l.name,
                "type": l.__class__.__name__,
                "output_shape": str(getattr(l, "output_shape", getattr(l, "shape", "unknown"))),
            }
            for l in model.layers
        ],
    }

    save_json(metadata, p)
    return metadata
