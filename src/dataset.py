"""Data loading and tf.data pipeline module.

Constructs high-performance, deterministic tf.data.Dataset pipelines
with explicit static shapes to eliminate 'Cannot take the length of shape with unknown rank'.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

from src.config import AppConfig, load_config
from src.utils import setup_logger

logger = setup_logger("dataset")


def load_manifest(manifest_csv_path: Path | str) -> Tuple[List[str], List[int]]:
    """Parse a split manifest CSV into image paths and corresponding integer labels."""
    csv_path = Path(manifest_csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

    image_paths: List[str] = []
    labels: List[int] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.append(row["image_path"])
            labels.append(int(row["label"]))

    if not image_paths:
        raise ValueError(f"Manifest {csv_path} contains zero records.")

    return image_paths, labels


def parse_and_preprocess_image(
    file_path: tf.Tensor,
    label: tf.Tensor,
    target_height: int = 128,
    target_width: int = 128,
    is_training: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read an image file from disk, decode, resize, normalize, and assign static tensor shapes.

    Critical Fix: Explicitly calling set_shape guarantees known rank, preventing
    the Keras error: 'Cannot take the length of shape with unknown rank'.
    """
    img_bytes = tf.io.read_file(file_path)
    # decode_jpeg with channels=3 ensures 3 color channels even if image is Grayscale
    image = tf.io.decode_jpeg(img_bytes, channels=3)
    image = tf.image.resize(image, [target_height, target_width])
    image = tf.cast(image, tf.float32) / 255.0  # Normalized [0, 1]

    if is_training:
        image = tf.image.random_flip_left_right(image)

    # Enforce static shape definitions so element_spec has known rank
    image.set_shape([target_height, target_width, 3])
    label = tf.cast(label, tf.float32)
    label.set_shape([])

    return image, label


def create_dataset_from_manifest(
    manifest_csv_path: Path | str,
    target_height: int = 128,
    target_width: int = 128,
    batch_size: int = 32,
    is_training: bool = False,
    shuffle_buffer_size: int = 1024,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from a split manifest file.

    Features:
    - Guaranteed isolation (only loads files assigned to this specific split).
    - Fully deterministic with fixed seed when shuffling.
    - Zero unknown rank errors with explicit set_shape in parse_and_preprocess_image.
    - Autotuned prefetching and non-blocking background I/O.
    """
    paths, labels = load_manifest(manifest_csv_path)

    paths_tensor = tf.constant(paths, dtype=tf.string)
    labels_tensor = tf.constant(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), shuffle_buffer_size), seed=seed)

    dataset = dataset.map(
        lambda p, l: parse_and_preprocess_image(
            p, l, target_height=target_height, target_width=target_width, is_training=is_training
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_data_splits(
    config: AppConfig | None = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Convenience helper to retrieve train, val, and test tf.data.Datasets."""
    if config is None:
        config = load_config()

    train_ds = create_dataset_from_manifest(
        manifest_csv_path=config.dataset.manifest_dir / "train_manifest.csv",
        target_height=config.image.height,
        target_width=config.image.width,
        batch_size=config.training.batch_size,
        is_training=True,
        seed=config.random_seed,
    )

    val_ds = create_dataset_from_manifest(
        manifest_csv_path=config.dataset.manifest_dir / "val_manifest.csv",
        target_height=config.image.height,
        target_width=config.image.width,
        batch_size=config.training.batch_size,
        is_training=False,
    )

    test_ds = create_dataset_from_manifest(
        manifest_csv_path=config.dataset.manifest_dir / "test_manifest.csv",
        target_height=config.image.height,
        target_width=config.image.width,
        batch_size=config.training.batch_size,
        is_training=False,
    )

    return train_ds, val_ds, test_ds
