"""Collect Benign PE Samples — gathers legitimate PE files for the Benign training class.

This script collects verified-clean Windows PE binaries (.exe, .dll) from local
system directories. These are legitimate, signed Microsoft binaries that serve
as ground truth for the "Benign" class.

Alternatively, if you have the Assemblage_PE dataset from HuggingFace downloaded
and extracted, point --source-dir to that folder instead.

Usage:
    python training/collect_benign.py
    python training/collect_benign.py --count 2000
    python training/collect_benign.py --source-dir "D:/assemblage_pe/binaries"
    python training/collect_benign.py --source-dir "C:/Windows/System32" --count 1500
"""

import argparse
import random
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import MAX_SAMPLES_PER_CLASS, PROJECT_ROOT, SEED

# Default Windows directories with legitimate PE binaries
DEFAULT_SYSTEM_DIRS = [
    Path("C:/Windows/System32"),
    Path("C:/Windows/SysWOW64"),
    Path("C:/Program Files/Common Files"),
    Path("C:/Program Files (x86)/Common Files"),
]

# Minimum file size (skip tiny stubs)
MIN_FILE_SIZE = 4096  # 4 KB

# Maximum file size (skip huge binaries that would slow heatmap generation)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

OUTPUT_DIR = PROJECT_ROOT / "raw_samples" / "Benign"


def is_valid_pe(file_path: Path) -> bool:
    """Quick check if file is a valid PE binary (checks MZ header and PE signature).

    Does NOT execute the file — only reads the first few bytes.
    """
    try:
        with open(file_path, "rb") as f:
            # Check MZ magic bytes
            mz_magic = f.read(2)
            if mz_magic != b"MZ":
                return False

            # Read PE header offset from DOS header at offset 0x3C
            f.seek(0x3C)
            pe_offset_bytes = f.read(4)
            if len(pe_offset_bytes) < 4:
                return False

            pe_offset = struct.unpack("<I", pe_offset_bytes)[0]

            # Sanity check offset
            if pe_offset > 1024:
                return False

            # Check PE signature
            f.seek(pe_offset)
            pe_sig = f.read(4)
            if pe_sig != b"PE\x00\x00":
                return False

        return True
    except (OSError, struct.error):
        return False


def collect_from_directory(source_dir: Path, extensions: set[str]) -> list[Path]:
    """Recursively find PE files in a directory.

    Only reads file headers for validation — does NOT execute anything.
    """
    pe_files = []

    if not source_dir.exists():
        print(f"  Skipping (not found): {source_dir}")
        return pe_files

    print(f"  Scanning: {source_dir}")
    try:
        for file_path in source_dir.rglob("*"):
            try:
                # Skip directories and symlinks
                if not file_path.is_file():
                    continue

                # Check extension
                if file_path.suffix.lower() not in extensions:
                    continue

                # Check file size bounds
                file_size = file_path.stat().st_size
                if file_size < MIN_FILE_SIZE or file_size > MAX_FILE_SIZE:
                    continue

                # Validate PE header (reads first few bytes only)
                if is_valid_pe(file_path):
                    pe_files.append(file_path)

            except (OSError, PermissionError):
                # Skip files we can't access
                continue

    except (OSError, PermissionError):
        print(f"  Permission denied: {source_dir}")

    print(f"  Found {len(pe_files)} valid PE files")
    return pe_files


def main(args: argparse.Namespace) -> None:
    """Collect benign PE samples into raw_samples/Benign/."""
    random.seed(SEED)

    target_count = args.count
    print(f"Target: {target_count} benign PE samples")
    print(f"Output: {OUTPUT_DIR}\n")

    # Determine source directories
    if args.source_dir:
        source_dirs = [Path(args.source_dir)]
    else:
        source_dirs = DEFAULT_SYSTEM_DIRS

    # Collect PE file paths from all sources
    all_pe_files: list[Path] = []
    extensions = {".exe", ".dll", ".sys", ".ocx", ".cpl"}

    print("Scanning for PE binaries (read-only, no execution)...")
    for source_dir in source_dirs:
        found = collect_from_directory(source_dir, extensions)
        all_pe_files.extend(found)

    print(f"\nTotal valid PE files found: {len(all_pe_files)}")

    if len(all_pe_files) < target_count:
        print(f"WARNING: Only found {len(all_pe_files)} files, less than target {target_count}")
        print("Consider adding more source directories or lowering --count")
        target_count = len(all_pe_files)

    # Random sample
    random.shuffle(all_pe_files)
    selected = all_pe_files[:target_count]

    # Copy to output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {len(selected)} files to {OUTPUT_DIR}...")
    copied = 0
    for i, src_path in enumerate(selected):
        try:
            # Use index-based naming to avoid filename collisions
            dst_name = f"benign_{i:05d}{src_path.suffix.lower()}"
            dst_path = OUTPUT_DIR / dst_name
            shutil.copy2(src_path, dst_path)
            copied += 1

            if (i + 1) % 200 == 0:
                print(f"  Copied {i + 1}/{len(selected)}")

        except (OSError, PermissionError) as e:
            print(f"  [SKIP] {src_path.name}: {e}")
            continue

    print(f"\nDone! Copied {copied} benign PE files to {OUTPUT_DIR}")
    print(f"\nYour raw_samples/ should now have all 7 classes:")
    print(f"  AgentTesla (Agensla), Remcos, DCRat, Androm, SnakeLogger, Mokes, Benign")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect benign PE samples for training (read-only, no execution)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=MAX_SAMPLES_PER_CLASS,
        help=f"Number of benign samples to collect (default: {MAX_SAMPLES_PER_CLASS})",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Custom source directory (default: scans Windows system dirs)",
    )
    args = parser.parse_args()
    main(args)
