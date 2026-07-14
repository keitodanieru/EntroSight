"""Verify that downloaded PE samples are usable for heatmap generation and training."""
import sys
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.components.heatmap import EntropyHeatmapGenerator

RAW_DIR = Path("raw_samples")
generator = EntropyHeatmapGenerator()


def is_valid_pe(file_bytes: bytes) -> bool:
    """Check MZ header and PE signature."""
    if len(file_bytes) < 64:
        return False
    if file_bytes[:2] != b"MZ":
        return False
    try:
        pe_offset = struct.unpack("<I", file_bytes[0x3C:0x40])[0]
        if pe_offset + 4 > len(file_bytes):
            return False
        if file_bytes[pe_offset:pe_offset+4] != b"PE\x00\x00":
            return False
    except (struct.error, IndexError):
        return False
    return True


def main():
    print("Verifying samples in raw_samples/...\n")

    for family_dir in sorted(RAW_DIR.iterdir()):
        if not family_dir.is_dir():
            continue

        files = list(family_dir.iterdir())
        if not files:
            print(f"{family_dir.name}: EMPTY")
            continue

        total = len(files)
        valid_pe = 0
        heatmap_ok = 0
        errors = []

        # Test up to 5 samples per family
        test_count = min(5, total)
        for f in files[:test_count]:
            if not f.is_file():
                continue

            try:
                data = f.read_bytes()
            except OSError:
                # Handle long path names on Windows
                long_path = "\\\\?\\" + str(f.resolve())
                try:
                    with open(long_path, "rb") as fh:
                        data = fh.read()
                except OSError as e:
                    errors.append(f"{f.name[:16]}: can't read: {e}")
                    continue

            # Check PE validity
            if is_valid_pe(data):
                valid_pe += 1
            else:
                errors.append(f"{f.name[:16]}: not valid PE")
                continue

            # Try generating heatmap
            try:
                tensor = generator.generate(data)
                if tensor.shape == (3, 256, 256):
                    heatmap_ok += 1
                else:
                    errors.append(f"{f.name[:16]}: wrong shape {tensor.shape}")
            except Exception as e:
                errors.append(f"{f.name[:16]}: heatmap error: {e}")

        status = "✓" if heatmap_ok == test_count else "⚠"
        print(f"{family_dir.name}: {total} files | tested {test_count} → "
              f"PE valid: {valid_pe}/{test_count}, heatmap OK: {heatmap_ok}/{test_count} {status}")

        for err in errors:
            print(f"  {err}")

    print("\nDone! If all show ✓, samples are ready for generate_dataset.py")


if __name__ == "__main__":
    main()
