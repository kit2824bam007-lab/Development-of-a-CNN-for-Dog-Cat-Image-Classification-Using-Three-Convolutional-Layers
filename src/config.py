"""Configuration management module for Dog-Cat CNN classification."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass
class DatasetConfig:
    url: str
    archive_filename: str
    raw_dir: Path
    processed_dir: Path
    manifest_dir: Path
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class ImageConfig:
    height: int = 128
    width: int = 128
    channels: int = 3
    normalization_scale: float = 1.0 / 255.0
    classes: List[str] = field(default_factory=lambda: ["Cat", "Dog"])
    class_indices: Dict[str, int] = field(default_factory=lambda: {"Cat": 0, "Dog": 1})


@dataclass
class ModelConfig:
    name: str = "ThreeLayerCNN"
    input_shape: tuple = (128, 128, 3)
    conv1_filters: int = 32
    conv2_filters: int = 64
    conv3_filters: int = 128
    kernel_size: tuple = (3, 3)
    pool_size: tuple = (2, 2)
    dense_units: int = 128
    dropout_rate: float = 0.5
    output_units: int = 1
    output_activation: str = "sigmoid"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5


@dataclass
class PathConfig:
    project_root: Path
    artifacts_dir: Path
    reports_dir: Path
    model_checkpoint: Path
    training_history: Path
    eval_metrics: Path
    data_report: Path


@dataclass
class AppConfig:
    random_seed: int
    dataset: DatasetConfig
    image: ImageConfig
    model: ModelConfig
    training: TrainingConfig
    paths: PathConfig


def get_project_root() -> Path:
    """Return the absolute root path of the project."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file and environment variables."""
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    project_sec = raw.get("project", {})
    dataset_sec = raw.get("dataset", {})
    splits_sec = dataset_sec.get("splits", {})
    image_sec = raw.get("image", {})
    model_sec = raw.get("model", {})
    train_sec = raw.get("training", {})
    paths_sec = raw.get("paths", {})

    random_seed = int(project_sec.get("random_seed", 42))

    dataset_cfg = DatasetConfig(
        url=dataset_sec.get("url", ""),
        archive_filename=dataset_sec.get("archive_filename", "kagglecatsanddogs_5340.zip"),
        raw_dir=root / dataset_sec.get("raw_dir", "data/raw"),
        processed_dir=root / dataset_sec.get("processed_dir", "data/processed"),
        manifest_dir=root / dataset_sec.get("manifest_dir", "data/manifests"),
        train_split=float(splits_sec.get("train", 0.70)),
        val_split=float(splits_sec.get("val", 0.15)),
        test_split=float(splits_sec.get("test", 0.15)),
    )

    image_cfg = ImageConfig(
        height=int(image_sec.get("height", 128)),
        width=int(image_sec.get("width", 128)),
        channels=int(image_sec.get("channels", 3)),
        normalization_scale=float(image_sec.get("normalization_scale", 1.0 / 255.0)),
        classes=[c["name"] for c in image_sec.get("classes", [{"name": "Cat"}, {"name": "Dog"}])],
        class_indices={c["name"]: int(c["index"]) for c in image_sec.get("classes", [{"name": "Cat", "index": 0}, {"name": "Dog", "index": 1}])},
    )

    conv_layers = model_sec.get("conv_layers", [])
    c1 = conv_layers[0]["filters"] if len(conv_layers) > 0 else 32
    c2 = conv_layers[1]["filters"] if len(conv_layers) > 1 else 64
    c3 = conv_layers[2]["filters"] if len(conv_layers) > 2 else 128

    model_cfg = ModelConfig(
        name=model_sec.get("name", "ThreeLayerCNN"),
        input_shape=tuple(model_sec.get("input_shape", [128, 128, 3])),
        conv1_filters=int(c1),
        conv2_filters=int(c2),
        conv3_filters=int(c3),
        dense_units=int(model_sec.get("dense_units", 128)),
        dropout_rate=float(model_sec.get("dropout_rate", 0.5)),
        output_units=int(model_sec.get("output_units", 1)),
        output_activation=model_sec.get("output_activation", "sigmoid"),
    )

    es_sec = train_sec.get("early_stopping", {})
    rlr_sec = train_sec.get("reduce_lr", {})

    training_cfg = TrainingConfig(
        batch_size=int(train_sec.get("batch_size", 32)),
        epochs=int(train_sec.get("epochs", 20)),
        learning_rate=float(train_sec.get("learning_rate", 0.001)),
        early_stopping_patience=int(es_sec.get("patience", 5)),
        reduce_lr_patience=int(rlr_sec.get("patience", 3)),
        reduce_lr_factor=float(rlr_sec.get("factor", 0.5)),
        min_lr=float(rlr_sec.get("min_lr", 1e-5)),
    )

    paths_cfg = PathConfig(
        project_root=root,
        artifacts_dir=root / paths_sec.get("artifacts_dir", "artifacts"),
        reports_dir=root / paths_sec.get("reports_dir", "reports"),
        model_checkpoint=root / paths_sec.get("model_checkpoint", "artifacts/best_model.keras"),
        training_history=root / paths_sec.get("training_history", "reports/training_history.json"),
        eval_metrics=root / paths_sec.get("eval_metrics", "reports/evaluation_metrics.json"),
        data_report=root / paths_sec.get("data_report", "reports/data_preparation_report.json"),
    )

    # Ensure output directories exist
    paths_cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths_cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg.manifest_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        random_seed=random_seed,
        dataset=dataset_cfg,
        image=image_cfg,
        model=model_cfg,
        training=training_cfg,
        paths=paths_cfg,
    )
