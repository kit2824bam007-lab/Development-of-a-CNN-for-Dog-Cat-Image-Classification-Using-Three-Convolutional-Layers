"""Model training pipeline module for Dog-Cat 3-Layer CNN."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import keras

from src.config import AppConfig, load_config
from src.dataset import create_dataset_from_manifest
from src.model import build_three_layer_cnn, export_model_metadata, validate_conv2d_count
from src.utils import configure_device, plot_training_history, save_json, set_seed, setup_logger

logger = setup_logger("train")


def train_model(
    config: AppConfig | None = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    resume_checkpoint: bool = False,
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute end-to-end training of the Three-Layer CNN model."""
    if config is None:
        config = load_config()

    set_seed(config.random_seed)
    hw_info = configure_device()
    logger.info(f"Hardware configuration: {hw_info}")

    epochs = epochs if epochs is not None else config.training.epochs
    batch_size = batch_size if batch_size is not None else config.training.batch_size

    # Validate manifest availability
    train_manifest = config.dataset.manifest_dir / "train_manifest.csv"
    val_manifest = config.dataset.manifest_dir / "val_manifest.csv"

    if not train_manifest.is_file() or not val_manifest.is_file():
        raise FileNotFoundError(
            f"Train/Val manifests not found at {config.dataset.manifest_dir}. "
            "Please run 'python scripts/prepare_data.py' first."
        )

    logger.info("Loading training and validation datasets...")
    train_ds = create_dataset_from_manifest(
        manifest_csv_path=train_manifest,
        target_height=config.image.height,
        target_width=config.image.width,
        batch_size=batch_size,
        is_training=True,
        seed=config.random_seed,
    )

    val_ds = create_dataset_from_manifest(
        manifest_csv_path=val_manifest,
        target_height=config.image.height,
        target_width=config.image.width,
        batch_size=batch_size,
        is_training=False,
    )

    # Checkpoint setup
    checkpoint_path = config.paths.model_checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if resume_checkpoint and checkpoint_path.is_file():
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
        validate_conv2d_count(model)
    else:
        logger.info("Initializing new Three-Layer CNN...")
        model = build_three_layer_cnn(
            input_shape=config.model.input_shape,
            conv1_filters=config.model.conv1_filters,
            conv2_filters=config.model.conv2_filters,
            conv3_filters=config.model.conv3_filters,
            dense_units=config.model.dense_units,
            dropout_rate=config.model.dropout_rate,
            learning_rate=config.training.learning_rate,
        )

    # Export model architecture details
    arch_json = config.paths.reports_dir / "model_architecture.json"
    export_model_metadata(model, arch_json)
    logger.info(f"Architecture metadata exported to {arch_json}")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config.training.early_stopping_patience and "val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.training.reduce_lr_factor,
            patience=config.training.reduce_lr_patience,
            min_lr=config.training.min_lr,
            verbose=1,
        ),
    ]

    logger.info(f"Commencing training: epochs={epochs}, batch_size={batch_size}...")
    start_time = time.time()

    fit_kwargs: Dict[str, Any] = {
        "x": train_ds,
        "validation_data": val_ds,
        "epochs": epochs,
        "callbacks": callbacks,
    }
    if steps_per_epoch is not None:
        fit_kwargs["steps_per_epoch"] = steps_per_epoch
    if validation_steps is not None:
        fit_kwargs["validation_steps"] = validation_steps

    history = model.fit(**fit_kwargs)
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

    # Convert history values to float for JSON serializability
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    history_dict["training_duration_seconds"] = float(elapsed_time)
    history_dict["epochs_trained"] = len(history_dict.get("accuracy", []))

    save_json(history_dict, config.paths.training_history)
    logger.info(f"Training history saved to {config.paths.training_history}")

    # Generate accuracy and loss visualization curves
    acc_plot, loss_plot = plot_training_history(history_dict, config.paths.reports_dir)
    logger.info(f"Plots generated: {acc_plot}, {loss_plot}")

    return {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "history": history_dict,
        "plots": {"accuracy": str(acc_plot), "loss": str(loss_plot)},
    }
