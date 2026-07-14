"""Combined Downloader — gets hashes from MalwareBazaar, downloads from MalShare.

Strategy:
- MalwareBazaar: list family-tagged sample hashes (no rate limit on listing)
- MalShare: download actual PE binaries by hash (2,000 calls/day, no aggressive throttling)

Usage:
    python training/download_combined.py
    python training/download_combined.py --limit 500
    python training/download_combined.py --families AgentTesla Remcos
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import MAX_SAMPLES_PER_CLASS, PROJECT_ROOT

# === API Configuration ===
BAZAAR_API_URL = "https://mb-api.abuse.ch/api/v1/"
BAZAAR_API_KEY = "f93c6fa5f5db6c37988038e9757789828e138eefbc7054f4"

MALSHARE_API_URL = "https://malshare.com/api.php"
MALSHARE_API_KEY = "4417b70dd1c0d08200a965d8db0fec23f917dc9e61ca7bf010b01f42e8a533a5"

# MalwareBazaar tags for each family
FAMILY_TAGS = {
    "AgentTesla": "AgentTesla",
    "Remcos": "Remcos",
    "DCRat": "DCRat",
    "FormBook": "Formbook",
    "RedLine": "RedLine",
    "AsyncRAT": "AsyncRAT",
}

OUTPUT_DIR = PROJECT_ROOT / "raw_samples"

# Delay between MalShare downloads (seconds)
DOWNLOAD_DELAY = 2


def get_family_hashes(tag: str, limit: int = 1000) -> list[str]:
    """Get PE file hashes from MalwareBazaar for a given family tag.

    Returns list of SHA-256 hashes for exe/dll files.
    """
    hashes = []
    try:
        with httpx.Client(
            headers={"Auth-Key": BAZAAR_API_KEY},
            timeout=60,
        ) as client:
            response = client.post(
                BAZAAR_API_URL,
                data={
                    "query": "get_taginfo",
                    "tag": tag,
                    "limit": min(limit, 1000),
                },
            )
            response.raise_for_status()
            result = response.json()

            if result.get("query_status") != "ok":
                print(f"  Bazaar returned: {result.get('query_status')}")
                return hashes

            data = result.get("data", [])
            for sample in data:
                file_type = sample.get("file_type", "").lower()
                # Only PE files
                if any(t in file_type for t in ["exe", "dll", "pe32", "pe64"]):
                    hashes.append(sample["sha256_hash"])

    except Exception as e:
        print(f"  [bazaar-error] {e}")

    return hashes


def download_from_malshare(sha256: str) -> bytes | None:
    """Download a single sample from MalShare by SHA-256 hash.

    Returns PE bytes or None if download fails.
    """
    try:
        response = httpx.get(
            f"{MALSHARE_API_URL}?api_key={MALSHARE_API_KEY}&action=getfile&hash={sha256}",
            timeout=60,
            follow_redirects=True,
        )

        if response.status_code != 200:
            return None

        content = response.content

        # Verify it's a PE file (MZ header)
        if len(content) > 2 and content[:2] == b"MZ":
            return content

        # Check for error messages
        if b"Sample not found" in content or b"error" in content.lower():
            return None

        # Some PE files might be in different format, accept if > 1KB
        if len(content) > 1024:
            return content

        return None

    except (httpx.TimeoutException, httpx.ConnectError):
        return None


def download_family(
    family_name: str,
    tag: str,
    output_dir: Path,
    limit: int,
) -> int:
    """Download samples for one family using both APIs.

    Returns number of successfully downloaded samples.
    """
    family_dir = output_dir / family_name
    family_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = set(f.name for f in family_dir.iterdir() if f.is_file())
    existing_count = len(existing)

    if existing_count >= limit:
        print(f"  Already have {existing_count} samples, skipping.")
        return existing_count

    remaining = limit - existing_count
    print(f"  Target: {limit} (have {existing_count}, need {remaining})")

    # Step 1: Get hashes from MalwareBazaar
    print(f"  Fetching hashes from MalwareBazaar (tag: {tag})...")
    all_hashes = get_family_hashes(tag, limit=1000)
    print(f"  Got {len(all_hashes)} PE hashes from Bazaar")

    if not all_hashes:
        print(f"  No hashes found for tag '{tag}'")
        return existing_count

    # Filter out already downloaded
    new_hashes = [h for h in all_hashes if h not in existing]
    print(f"  {len(new_hashes)} new hashes to try")

    # Step 2: Download from MalShare
    downloaded = 0
    not_found = 0
    max_not_found = 200  # Allow many misses — MalShare only has ~10% of Bazaar hashes

    for i, sha256 in enumerate(new_hashes):
        if downloaded >= remaining:
            break

        if not_found >= max_not_found:
            print(f"  Too many misses ({max_not_found}), stopping.")
            break

        pe_bytes = download_from_malshare(sha256)

        if pe_bytes:
            output_path = family_dir / sha256
            output_path.write_bytes(pe_bytes)
            downloaded += 1
            not_found = 0  # Reset miss counter on success

            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded}/{remaining} ({i+1} attempted)")
        else:
            not_found += 1

        # Rate limiting
        time.sleep(DOWNLOAD_DELAY)

    total = existing_count + downloaded
    print(f"  Done: {downloaded} new (total: {total})")
    return total


def main(args: argparse.Namespace) -> None:
    """Download samples for all families."""
    print(f"Combined Downloader (Bazaar hashes + MalShare downloads)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target per family: {args.limit}")
    print(f"Download delay: {DOWNLOAD_DELAY}s")
    print(f"{'='*50}\n")

    if args.families:
        families = {k: v for k, v in FAMILY_TAGS.items() if k in args.families}
    else:
        families = FAMILY_TAGS

    results: dict[str, int] = {}

    for family_name, tag in families.items():
        print(f"\n[{family_name}] (tag: {tag})")
        count = download_family(
            family_name=family_name,
            tag=tag,
            output_dir=OUTPUT_DIR,
            limit=args.limit,
        )
        results[family_name] = count

    # Summary
    print(f"\n{'='*50}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    print(f"{'Family':<20} {'Samples':>10}")
    print(f"{'-'*30}")
    for family, count in results.items():
        status = "✓" if count >= args.limit else "⚠"
        print(f"{family:<20} {count:>10} {status}")

    total = sum(results.values())
    print(f"\nTotal malware samples: {total}")
    print(f"\nNext step: python training/generate_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download malware (Bazaar + MalShare)")
    parser.add_argument("--limit", type=int, default=MAX_SAMPLES_PER_CLASS)
    parser.add_argument("--families", nargs="+", default=None)
    args = parser.parse_args()
    main(args)
