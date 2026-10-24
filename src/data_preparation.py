"""Dataset preparation module for Microsoft Cats vs Dogs.

Handles downloading, zip extraction, corrupt image filtering, RGB conversion,
stratified train/val/test splitting, and manifest generation.
"""

from __future__ import annotations

import csv
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from src.config import AppConfig, load_config
from src.utils import save_json, setup_logger

logger = setup_logger("data_preparation")


def download_dataset(
    url: str,
    zip_dest: Path | str,
    extract_to: Path | str,
    force_download: bool = False,
) -> Path:
    """Download the dataset zip archive if not already cached and extract it."""
    zip_path = Path(zip_dest)
    extract_path = Path(extract_to)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    # Check if raw PetImages folder already extracted
    pet_images = extract_path / "PetImages"
    if not pet_images.exists():
        # Check subdirectories
        pet_images_alt = extract_path
        if (pet_images_alt / "Cat").is_dir() and (pet_images_alt / "Dog").is_dir():
            pet_images = pet_images_alt

    if pet_images.is_dir() and any((pet_images / "Cat").iterdir()) and not force_download:
        logger.info(f"Dataset already extracted at: {pet_images}")
        return pet_images

    # Check if zip exists
    if not zip_path.is_file() or force_download or zip_path.stat().st_size == 0:
        logger.info(f"Downloading dataset from {url} to {zip_path}...")
        try:
            # Custom user agent to prevent 403 Forbidden from download servers
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            logger.info(f"Download complete: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
        except Exception as e:
            logger.error(f"Failed to download dataset from {url}: {e}")
            raise

    # Extract zip archive
    logger.info(f"Extracting {zip_path} to {extract_path}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_path)
    logger.info("Extraction complete.")

    if (extract_path / "PetImages").is_dir():
        return extract_path / "PetImages"
    return extract_path


def validate_and_convert_image(
    input_path: Path,
    output_path: Path,
) -> Tuple[bool, str]:
    """Validate image integrity, convert to RGB, and save cleanly.

    Returns:
        (is_valid, reason_or_success_message)
    """
    if not input_path.is_file():
        return False, "File does not exist"

    if input_path.stat().st_size == 0:
        return False, "Zero-byte file"

    try:
        # Phase 1: verify integrity
        with Image.open(input_path) as img:
            img.verify()

        # Phase 2: verify re-reading and conversion to RGB (strips invalid EXIF / palette)
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_img.save(output_path, format="JPEG", quality=95)
        return True, "Valid"
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return False, f"Corrupted or unsupported image: {str(e)}"
    except Exception as e:
        return False, f"Unexpected read error: {str(e)}"


def clean_and_process_dataset(
    raw_source_dir: Path,
    processed_dir: Path,
    classes: List[str] = ("Cat", "Dog"),
    max_samples_per_class: int | None = None,
) -> Dict[str, Any]:
    """Inspect all raw images, filter out corrupted files, convert valid images to RGB."""
    report: Dict[str, Any] = {
        "classes": {},
        "total_scanned": 0,
        "total_valid": 0,
        "total_rejected": 0,
        "rejection_log": [],
    }

    # Locate class folders inside raw_source_dir or raw_source_dir/PetImages
    root_to_check = raw_source_dir
    if (raw_source_dir / "PetImages").is_dir():
        root_to_check = raw_source_dir / "PetImages"

    for class_name in classes:
        class_raw_dir = root_to_check / class_name
        class_out_dir = processed_dir / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        if not class_raw_dir.is_dir():
            logger.warning(f"Class folder not found: {class_raw_dir}")
            continue

        raw_files = sorted(
            [f for f in class_raw_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
        )
        if max_samples_per_class:
            raw_files = raw_files[:max_samples_per_class]

        valid_count = 0
        rejected_count = 0

        logger.info(f"Scanning and cleaning class '{class_name}' ({len(raw_files)} files)...")

        for fpath in raw_files:
            # Skip thumbs.db or non-image files
            if fpath.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                rejected_count += 1
                report["rejection_log"].append(
                    {"file": str(fpath.name), "class": class_name, "reason": "Non-image extension"}
                )
                continue

            target_path = class_out_dir / f"{class_name}_{fpath.stem}.jpg"

            # Check if target already processed and valid
            if target_path.is_file() and target_path.stat().st_size > 0:
                valid_count += 1
                continue

            is_valid, msg = validate_and_convert_image(fpath, target_path)
            if is_valid:
                valid_count += 1
            else:
                rejected_count += 1
                report["rejection_log"].append(
                    {"file": str(fpath.name), "class": class_name, "reason": msg}
                )

        report["classes"][class_name] = {
            "scanned": len(raw_files),
            "valid": valid_count,
            "rejected": rejected_count,
        }
        report["total_scanned"] += len(raw_files)
        report["total_valid"] += valid_count
        report["total_rejected"] += rejected_count

        logger.info(
            f"Class '{class_name}': {valid_count} valid, {rejected_count} rejected "
            f"out of {len(raw_files)} scanned."
        )

    return report


def create_stratified_manifests(
    processed_dir: Path,
    manifest_dir: Path,
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    classes: Tuple[str, ...] = ("Cat", "Dog"),
    class_indices: Dict[str, int] = None,
    seed: int = 42,
) -> Dict[str, Path]:
    """Create reproducible stratified train, validation, and test split manifests."""
    if class_indices is None:
        class_indices = {"Cat": 0, "Dog": 1}

    manifest_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []

    for class_name in classes:
        c_dir = processed_dir / class_name
        if not c_dir.is_dir():
            continue
        files = sorted([f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])
        label = class_indices[class_name]
        for f in files:
            all_records.append(
                {
                    "relative_path": str(f.relative_to(processed_dir.parent.parent)).replace("\\", "/"),
                    "absolute_path": str(f.resolve()).replace("\\", "/"),
                    "filename": f.name,
                    "class_name": class_name,
                    "label": label,
                }
            )

    if not all_records:
        raise ValueError(f"No processed image files found in {processed_dir}")

    paths = np.array([r["absolute_path"] for r in all_records])
    labels = np.array([r["label"] for r in all_records])

    train_pct, val_pct, test_pct = splits
    total_pct = train_pct + val_pct + test_pct
    assert abs(total_pct - 1.0) < 1e-4, f"Splits must sum to 1.0, got {total_pct}"

    temp_pct = val_pct + test_pct
    # First split: Train vs (Val + Test)
    indices = np.arange(len(paths))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_pct,
        stratify=labels,
        random_state=seed,
    )

    # Second split: Val vs Test
    val_relative_pct = val_pct / temp_pct
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_relative_pct,
        stratify=labels[temp_idx],
        random_state=seed,
    )

    # Verify zero data leakage across splits
    train_set = set(paths[train_idx])
    val_set = set(paths[val_idx])
    test_set = set(paths[test_idx])

    assert len(train_set.intersection(val_set)) == 0, "Data leakage detected between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Data leakage detected between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Data leakage detected between val and test!"

    manifest_paths = {
        "train": manifest_dir / "train_manifest.csv",
        "val": manifest_dir / "val_manifest.csv",
        "test": manifest_dir / "test_manifest.csv",
    }

    def write_csv(manifest_path: Path, split_name: str, split_indices: np.ndarray) -> None:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label", "class_name", "split"])
            for idx in split_indices:
                rec = all_records[idx]
                writer.writerow([rec["absolute_path"], rec["label"], rec["class_name"], split_name])

    write_csv(manifest_paths["train"], "train", train_idx)
    write_csv(manifest_paths["val"], "val", val_idx)
    write_csv(manifest_paths["test"], "test", test_idx)

    logger.info(
        f"Manifests successfully generated:\n"
        f"  Train: {len(train_idx)} samples ({len(train_idx) / len(paths) * 100:.2f}%)\n"
        f"  Val:   {len(val_idx)} samples ({len(val_idx) / len(paths) * 100:.2f}%)\n"
        f"  Test:  {len(test_idx)} samples ({len(test_idx) / len(paths) * 100:.2f}%)"
    )

    return manifest_paths


def run_data_preparation_pipeline(
    config: AppConfig | None = None,
    max_samples_per_class: int | None = None,
    force_download: bool = False,
) -> Dict[str, Any]:
    """Execute complete data preparation workflow and write final report."""
    if config is None:
        config = load_config()

    logger.info("Starting dataset preparation pipeline...")

    # 1. Download / verify raw files
    raw_pet_images = download_dataset(
        url=config.dataset.url,
        zip_dest=config.dataset.raw_dir / config.dataset.archive_filename,
        extract_to=config.dataset.raw_dir,
        force_download=force_download,
    )

    # 2. Clean and convert to RGB
    report = clean_and_process_dataset(
        raw_source_dir=raw_pet_images,
        processed_dir=config.dataset.processed_dir,
        classes=config.image.classes,
        max_samples_per_class=max_samples_per_class,
    )

    # 3. Create stratified manifests
    manifest_paths = create_stratified_manifests(
        processed_dir=config.dataset.processed_dir,
        manifest_dir=config.dataset.manifest_dir,
        splits=(
            config.dataset.train_split,
            config.dataset.val_split,
            config.dataset.test_split,
        ),
        classes=tuple(config.image.classes),
        class_indices=config.image.class_indices,
        seed=config.random_seed,
    )

    report["manifests"] = {k: str(v) for k, v in manifest_paths.items()}
    save_json(report, config.paths.data_report)
    logger.info(f"Data preparation report saved to {config.paths.data_report}")
    return report
