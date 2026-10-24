# Data Directory Documentation

This directory contains dataset resources and manifests for the Dog–Cat Image Classification project.

## Directory Structure
- `raw/`: Unzipped genuine Microsoft Cats vs Dogs dataset (`kagglecatsanddogs_5340.zip`). Excluded from Git.
- `processed/`: Validated, RGB-converted images cleaned of corrupted headers and zero-byte files. Excluded from Git.
- `manifests/`: Stratified split manifests (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) detailing file paths, labels, and split assignments without data leakage.

## Dataset Details
- **Dataset**: Microsoft Research Cats vs Dogs
- **Classes**: Cat (Label 0), Dog (Label 1)
- **Split Ratios**: 70% Train, 15% Validation, 15% Test
- **Download Source**: [Microsoft Cats and Dogs Official Archive](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip)

Run `python scripts/prepare_data.py` to automatically download, clean, and generate manifests.
