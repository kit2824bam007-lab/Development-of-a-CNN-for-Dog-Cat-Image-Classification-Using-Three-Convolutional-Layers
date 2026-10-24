"""Utility functions for logging, reproducibility, hardware detection, and plotting."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None


def setup_logger(name: str = "dog_cat_cnn", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def configure_device() -> Dict[str, Any]:
    """Detect and configure CPU and GPU hardware for TensorFlow."""
    info = {"gpu_available": False, "gpu_count": 0, "devices": []}
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["gpu_available"] = True
            info["gpu_count"] = len(gpus)
            info["devices"] = [gpu.name for gpu in gpus]
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
        else:
            cpus = tf.config.list_physical_devices("CPU")
            info["devices"] = [cpu.name for cpu in cpus]
    except Exception as e:
        info["error"] = str(e)
    return info


def save_json(data: Any, path: Path | str) -> None:
    """Save serializable data to a formatted JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path | str) -> Any:
    """Load and parse JSON file."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_history(
    history_data: Dict[str, List[float]],
    output_dir: Path | str,
) -> Tuple[Path, Path]:
    """Generate and save professional training vs validation curves for Accuracy and Loss."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = history_data.get("accuracy", [])
    val_acc = history_data.get("val_accuracy", [])
    loss = history_data.get("loss", [])
    val_loss = history_data.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    # 1. Accuracy Plot
    acc_path = out_dir / "accuracy_curve.png"
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#1976D2", linewidth=2)
    if val_acc:
        plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#E64A19", linewidth=2, linestyle="--")
    plt.title("Model Accuracy over Epochs", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    # 2. Loss Plot
    loss_path = out_dir / "loss_curve.png"
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs_range, loss, label="Training Loss", color="#1976D2", linewidth=2)
    if val_loss:
        plt.plot(epochs_range, val_loss, label="Validation Loss", color="#E64A19", linewidth=2, linestyle="--")
    plt.title("Binary Cross-Entropy Loss over Epochs", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    return acc_path, loss_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path | str,
) -> Path:
    """Generate and save a confusion matrix plot using seaborn or pure matplotlib."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5), dpi=300)
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            annot_kws={"size": 14, "weight": "bold"},
        )
    else:
        im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=11)
        plt.yticks(tick_marks, class_names, fontsize=11)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14,
                    fontweight="bold",
                )
    plt.title("Confusion Matrix (Isolated Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    output_path: Path | str,
) -> Path:
    """Generate and save ROC curve plot."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(fpr, tpr, color="#0288D1", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9E9E9E", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
