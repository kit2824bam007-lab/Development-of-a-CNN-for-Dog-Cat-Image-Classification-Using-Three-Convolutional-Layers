#!/usr/bin/env python
"""Command-line script to download, clean, validate, and split the Cats vs Dogs dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data_preparation import run_data_preparation_pipeline
from src.utils import setup_logger

logger = setup_logger("prepare_data_script")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Cats vs Dogs dataset with integrity verification and splitting.")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config.yaml.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per class (for testing/smoke runs).")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of dataset archive.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = run_data_preparation_pipeline(
        config=cfg,
        max_samples_per_class=args.max_samples,
        force_download=args.force_download,
    )

    logger.info("Dataset preparation complete.")
    logger.info(f"Total Scanned:  {report['total_scanned']}")
    logger.info(f"Total Valid:    {report['total_valid']}")
    logger.info(f"Total Rejected: {report['total_rejected']}")


if __name__ == "__main__":
    main()
