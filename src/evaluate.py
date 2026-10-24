"""Model evaluation and metrics reporting module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import tensorflow as tf
from tensorflow import keras

from src.config import AppConfig, load_config
from src.dataset import create_dataset_from_manifest, load_manifest
from src.model import validate_conv2d_count
from src.utils import plot_confusion_matrix, plot_roc_curve, save_json, setup_logger

logger = setup_logger("evaluate")


def evaluate_model_on_test_set(
    config: AppConfig | None = None,
    model_path: Optional[Path | str] = None,
    test_manifest_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Perform rigorous scientific evaluation on the isolated test set manifest."""
    if config is None:
        config = load_config()

    m_path = Path(model_path) if model_path else config.paths.model_checkpoint
    t_manifest = Path(test_manifest_path) if test_manifest_path else config.dataset.manifest_dir / "test_manifest.csv"

    if not m_path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found at: {m_path}")
    if not t_manifest.is_file():
        raise FileNotFoundError(f"Test manifest not found at: {t_manifest}")

    logger.info(f"Loading trained model from {m_path}...")
    model = keras.models.load_model(m_path)
    validate_conv2d_count(model)

    logger.info(f"Loading isolated test set from {t_manifest}...")
    test_paths, y_true = load_manifest(t_manifest)
    y_true = np.array(y_true, dtype=np.int32)

    test_ds = create_dataset_from_manifest(
        manifest_csv_path=t_manifest,
        target_height=config.image.height,
        target_width=config.image.width,
        batch_size=config.training.batch_size,
        is_training=False,
    )

    logger.info(f"Evaluating loss and accuracy across {len(y_true)} test samples...")
    eval_results = model.evaluate(test_ds, verbose=1)
    test_loss = float(eval_results[0])
    test_acc_from_eval = float(eval_results[1])

    logger.info("Computing test predictions and probability distributions...")
    raw_preds = model.predict(test_ds, verbose=1)
    y_scores = raw_preds.flatten()
    y_pred = (y_scores >= 0.5).astype(np.int32)

    # Scikit-learn rigorous metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc_score = float(roc_auc_score(y_true, y_scores))

    cm = confusion_matrix(y_true, y_pred)
    class_names = config.image.classes
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Save Plots
    cm_plot_path = config.paths.reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_plot_path = config.paths.reports_dir / "roc_curve.png"
    plot_roc_curve(fpr, tpr, auc_score, roc_plot_path)

    cat_count = int(np.sum(y_true == 0))
    dog_count = int(np.sum(y_true == 1))

    results = {
        "dataset_split": "isolated_test_set",
        "sample_counts": {
            "total": len(y_true),
            "Cat": cat_count,
            "Dog": dog_count,
        },
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "test_accuracy_keras": test_acc_from_eval,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc_score,
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "plots": {
            "confusion_matrix": str(cm_plot_path),
            "roc_curve": str(roc_plot_path),
        },
    }

    save_json(results, config.paths.eval_metrics)
    logger.info(f"Evaluation complete. Metrics saved to {config.paths.eval_metrics}")
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}% | F1-Score: {f1:.4f} | ROC AUC: {auc_score:.4f}")

    return results
