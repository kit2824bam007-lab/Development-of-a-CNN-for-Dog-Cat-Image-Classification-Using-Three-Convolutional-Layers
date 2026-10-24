#!/usr/bin/env python
"""Command-line script to train the 3-Layer CNN on preprocessed manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import train_model
from src.utils import setup_logger

logger = setup_logger("run_training_script")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3-Layer CNN on dog-cat classification.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing model checkpoint.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 1-epoch smoke test with limited steps.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke_test:
        logger.info("Executing SMOKE TEST mode (1 epoch, limited steps)...")
        results = train_model(
            config=cfg,
            epochs=1,
            batch_size=min(16, cfg.training.batch_size),
            steps_per_epoch=2,
            validation_steps=2,
        )
    else:
        results = train_model(
            config=cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume_checkpoint=args.resume,
        )

    logger.info("Training pipeline finished successfully.")
    logger.info(f"Model saved to: {results['checkpoint']}")


if __name__ == "__main__":
    main()
