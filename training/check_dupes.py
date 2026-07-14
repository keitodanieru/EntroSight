"""Check duplicate situation in AgentTesla folder."""
import hashlib
from pathlib import Path
from collections import Counter

folder = Path("raw_samples/AgentTesla")
files = list(folder.iterdir())
print(f"Total files: {len(files)}")

# Check file sizes
sizes = [f.stat().st_size for f in files if f.is_file()]
size_counts = Counter(sizes)
print(f"\nUnique file sizes: {len(size_counts)}")
print(f"Most common sizes:")
for size, count in size_counts.most_common(10):
    print(f"  {size:>10} bytes: {count} files")

# Check SHA-256 hashes
print(f"\nHashing all files...")
hashes = {}
for f in files:
    if not f.is_file():
        continue
    try:
        long_path = "\\\\?\\" + str(f.resolve())
        with open(long_path, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        if h not in hashes:
            hashes[h] = []
        hashes[h].append(f.name[:20])
    except Exception as e:
        print(f"  Error: {f.name[:20]}: {e}")

print(f"\nUnique hashes: {len(hashes)}")
print(f"Duplicate groups (same content, different filename):")
dupes = {h: names for h, names in hashes.items() if len(names) > 1}
print(f"  {len(dupes)} groups with duplicates")
for h, names in list(dupes.items())[:5]:
    print(f"  Hash {h[:16]}... has {len(names)} copies")
