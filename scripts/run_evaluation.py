#!/usr/bin/env python
"""Command-line script to evaluate the trained 3-Layer CNN model on the isolated test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.evaluate import evaluate_model_on_test_set
from src.utils import setup_logger

logger = setup_logger("run_evaluation_script")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained 3-Layer CNN on isolated test set.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model checkpoint.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to test_manifest.csv.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = evaluate_model_on_test_set(
        config=cfg,
        model_path=args.model,
        test_manifest_path=args.manifest,
    )

    print("\n" + "=" * 55)
    print("        ISOLATED TEST SET EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Total Test Samples:      {results['sample_counts']['total']}")
    print(f"Test Loss:               {results['metrics']['test_loss']:.4f}")
    print(f"Test Accuracy:           {results['metrics']['test_accuracy'] * 100:.2f}%")
    print(f"Precision:               {results['metrics']['precision']:.4f}")
    print(f"Recall:                  {results['metrics']['recall']:.4f}")
    print(f"F1-Score:                {results['metrics']['f1_score']:.4f}")
    print(f"ROC AUC:                 {results['metrics']['roc_auc']:.4f}")
    print("=" * 55)
    print(f"Confusion Matrix:        {results['confusion_matrix']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
