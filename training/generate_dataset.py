"""Dataset Generator — converts raw PE binaries to heatmap tensors for training.

Usage:
    python training/generate_dataset.py
    python training/generate_dataset.py --raw-dir /path/to/extracted/samples
    python training/generate_dataset.py --max-samples 2000
"""

import argparse
import random
import sys
from pathlib import Path

import torch

# Add project root to path so we can import the app's heatmap generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components.heatmap import EntropyHeatmapGenerator
from training.config import (
    CLASS_LABELS,
    DATASET_DIR,
    MAX_SAMPLES_PER_CLASS,
    RAW_SAMPLES_DIR,
    SEED,
    VAL_SPLIT,
)


def find_class_folders(raw_dir: Path) -> dict[str, Path]:
    """Locate folders matching our class labels within the raw samples directory.

    Handles known folder name aliases (e.g., "Agensla" → "AgentTesla").
    Searches recursively in case the zip extracts with nested structure.
    Matching is case-insensitive.
    """
    # Known aliases: folder_name_lower → canonical class label
    FOLDER_ALIASES = {
        "agensla": "AgentTesla",
        "agenttesla": "AgentTesla",
        "remcos": "Remcos",
        "dcrat": "DCRat",
        "androm": "Androm",
        "snakelogger": "SnakeLogger",
        "mokes": "Mokes",
        "benign": "Benign",
    }

    class_folders: dict[str, Path] = {}
    label_lower_map = {label.lower(): label for label in CLASS_LABELS}

    # First try direct children
    for item in raw_dir.iterdir():
        if item.is_dir():
            name_lower = item.name.lower()
            # Check alias first, then exact match
            if name_lower in FOLDER_ALIASES:
                label = FOLDER_ALIASES[name_lower]
                if label in CLASS_LABELS and label not in class_folders:
                    class_folders[label] = item
            elif name_lower in label_lower_map:
                class_folders[label_lower_map[name_lower]] = item

    # If we didn't find all classes, search one level deeper
    if len(class_folders) < len(CLASS_LABELS):
        for subdir in raw_dir.iterdir():
            if subdir.is_dir():
                for item in subdir.iterdir():
                    if item.is_dir():
                        name_lower = item.name.lower()
                        if name_lower in FOLDER_ALIASES:
                            label = FOLDER_ALIASES[name_lower]
                            if label in CLASS_LABELS and label not in class_folders:
                                class_folders[label] = item
                        elif name_lower in label_lower_map and label_lower_map[name_lower] not in class_folders:
                            class_folders[label_lower_map[name_lower]] = item

    return class_folders


def get_pe_files(folder: Path) -> list[Path]:
    """Get all files in a folder (PE binaries may not have standard extensions)."""
    files = []
    for f in folder.iterdir():
        if f.is_file() and f.stat().st_size > 0:
            files.append(f)
    # Also check one level of subdirectories
    for sub in folder.iterdir():
        if sub.is_dir():
            for f in sub.iterdir():
                if f.is_file() and f.stat().st_size > 0:
                    files.append(f)
    return files


def generate_dataset(raw_dir: Path, output_dir: Path, max_samples: int) -> None:
    """Convert raw PE binaries to heatmap tensors, organized into train/val splits."""
    random.seed(SEED)
    generator = EntropyHeatmapGenerator()

    # Find class folders
    print(f"Scanning {raw_dir} for class folders...")
    class_folders = find_class_folders(raw_dir)

    if not class_folders:
        print(f"ERROR: No class folders found in {raw_dir}")
        print(f"Expected folder names (case-insensitive): {CLASS_LABELS}")
        print(f"Contents of {raw_dir}:")
        for item in raw_dir.iterdir():
            print(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
        sys.exit(1)

    print(f"Found {len(class_folders)}/{len(CLASS_LABELS)} classes:")
    for label, path in class_folders.items():
        print(f"  {label}: {path}")

    missing = set(CLASS_LABELS) - set(class_folders.keys())
    if missing:
        print(f"\nWARNING: Missing classes: {missing}")
        print("Proceeding with available classes.\n")

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    # Process each class
    total_generated = 0
    class_counts: dict[str, dict[str, int]] = {}

    for label, folder_path in class_folders.items():
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print(f"{'='*60}")

        # Get all PE files
        all_files = get_pe_files(folder_path)
        print(f"  Found {len(all_files)} files")

        # Cap at max_samples
        if len(all_files) > max_samples:
            random.shuffle(all_files)
            all_files = all_files[:max_samples]
            print(f"  Capped to {max_samples} samples")

        # Shuffle and split into train/val
        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - VAL_SPLIT))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        # Create class directories
        (train_dir / label).mkdir(parents=True, exist_ok=True)
        (val_dir / label).mkdir(parents=True, exist_ok=True)

        # Process training files
        train_count = 0
        for i, pe_path in enumerate(train_files):
            try:
                file_bytes = pe_path.read_bytes()
                if len(file_bytes) < 64:  # Skip tiny/corrupt files
                    continue

                heatmap_tensor = generator.generate(file_bytes)
                output_path = train_dir / label / f"{i:05d}.pt"
                torch.save(heatmap_tensor, output_path)
                train_count += 1

                if (i + 1) % 100 == 0:
                    print(f"  [train] {i + 1}/{len(train_files)} processed")

            except Exception as e:
                print(f"  [SKIP] {pe_path.name}: {e}")
                continue

        # Process validation files
        val_count = 0
        for i, pe_path in enumerate(val_files):
            try:
                file_bytes = pe_path.read_bytes()
                if len(file_bytes) < 64:
                    continue

                heatmap_tensor = generator.generate(file_bytes)
                output_path = val_dir / label / f"{i:05d}.pt"
                torch.save(heatmap_tensor, output_path)
                val_count += 1

                if (i + 1) % 100 == 0:
                    print(f"  [val] {i + 1}/{len(val_files)} processed")

            except Exception as e:
                print(f"  [SKIP] {pe_path.name}: {e}")
                continue

        class_counts[label] = {"train": train_count, "val": val_count}
        total_generated += train_count + val_count
        print(f"  Done: {train_count} train + {val_count} val = {train_count + val_count} total")

    # Print summary
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples generated: {total_generated}")
    print(f"Output directory: {output_dir}")
    print(f"\nPer-class breakdown:")
    print(f"  {'Class':<20} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'-'*46}")
    for label in CLASS_LABELS:
        if label in class_counts:
            c = class_counts[label]
            print(f"  {label:<20} {c['train']:>8} {c['val']:>8} {c['train']+c['val']:>8}")

    print(f"\nNext step: python training/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmap dataset from raw PE binaries")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_SAMPLES_DIR,
        help=f"Path to extracted PE samples (default: {RAW_SAMPLES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Output directory for tensor dataset (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_SAMPLES_PER_CLASS,
        help=f"Max samples per class (default: {MAX_SAMPLES_PER_CLASS})",
    )
    args = parser.parse_args()

    generate_dataset(args.raw_dir, args.output_dir, args.max_samples)
