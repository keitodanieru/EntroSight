"""Download malware samples from MalwareBazaar by family tag.

Downloads unique PE samples from MalwareBazaar's API, organized by family.
Each sample has a guaranteed unique SHA-256 hash.

MalwareBazaar API: https://bazaar.abuse.ch/api/

Usage:
    python training/download_bazaar.py
    python training/download_bazaar.py --limit 1500
    python training/download_bazaar.py --families AgentTesla Remcos DCRat
"""

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import MAX_SAMPLES_PER_CLASS, PROJECT_ROOT

BAZAAR_API_URL = "https://mb-api.abuse.ch/api/v1/"
BAZAAR_API_V2 = "https://mb-api.abuse.ch/v2/"
BAZAAR_API_KEY = "f93c6fa5f5db6c37988038e9757789828e138eefbc7054f4"
ZIP_PASSWORD = b"infected"  # MalwareBazaar's standard zip password

# Target families and their MalwareBazaar tags
FAMILY_TAGS = {
    "AgentTesla": "AgentTesla",
    "Remcos": "Remcos",
    "DCRat": "DCRat",
    "FormBook": "Formbook",
    "RedLine": "RedLine",
    "AsyncRAT": "AsyncRAT",
}

OUTPUT_DIR = PROJECT_ROOT / "raw_samples"


def download_family_samples(
    family_name: str,
    tag: str,
    output_dir: Path,
    limit: int,
    batch_size: int = 100,
) -> int:
    """Download samples for a single family from MalwareBazaar.

    Args:
        family_name: Display name for the family (folder name).
        tag: MalwareBazaar tag to query.
        output_dir: Base output directory (raw_samples/).
        limit: Maximum samples to download.
        batch_size: Samples per API request (max 1000).

    Returns:
        Number of successfully downloaded samples.
    """
    family_dir = output_dir / family_name
    family_dir.mkdir(parents=True, exist_ok=True)

    # Check how many we already have
    existing = list(family_dir.glob("*"))
    existing_count = len(existing)
    if existing_count >= limit:
        print(f"  Already have {existing_count} samples, skipping.")
        return existing_count

    remaining = limit - existing_count
    downloaded = 0
    seen_hashes: set[str] = set()

    # Collect hashes of existing files to avoid re-downloading
    for f in existing:
        seen_hashes.add(f.stem)  # Files are named by SHA256

    print(f"  Target: {limit} samples (have {existing_count}, need {remaining})")

    # MalwareBazaar returns samples in pages, but the API doesn't have traditional pagination.
    # We use "get_taginfo" to get recent samples by tag.
    with httpx.Client(
        timeout=httpx.Timeout(120.0, connect=30.0),
        headers={"Auth-Key": BAZAAR_API_KEY},
    ) as client:
        while downloaded < remaining:
            try:
                # Query for samples by tag
                response = client.post(
                    BAZAAR_API_URL,
                    data={
                        "query": "get_taginfo",
                        "tag": tag,
                        "limit": batch_size,
                    },
                )
                response.raise_for_status()
                result = response.json()

                status = result.get("query_status")
                if status != "ok":
                    print(f"  API returned: {status}")
                    break

                data = result.get("data", [])
                if not data:
                    print(f"  No more samples available for tag '{tag}'")
                    break

                # Process each sample
                for sample in data:
                    if downloaded >= remaining:
                        break

                    sha256 = sample.get("sha256_hash", "")
                    file_type = sample.get("file_type", "")

                    # Debug: print first few file types to see what's available
                    if downloaded == 0 and len(seen_hashes) < 5:
                        print(f"  [debug] file_type='{file_type}' sha256={sha256[:16]}...")

                    # Accept PE files — MalwareBazaar uses various type strings
                    file_type_lower = file_type.lower()
                    is_pe = any(t in file_type_lower for t in ["exe", "dll", "pe32", "pe64", "msi", "windows"])
                    if not is_pe and file_type_lower:
                        continue

                    # Skip already downloaded
                    if sha256 in seen_hashes:
                        continue

                    # Download the sample (comes as password-protected zip)
                    try:
                        print(f"  [downloading] {sha256[:16]}...")
                        dl_response = client.post(
                            BAZAAR_API_URL,
                            data={
                                "query": "get_file",
                                "sha256_hash": sha256,
                            },
                        )

                        if dl_response.status_code != 200:
                            print(f"  [dl-error] status={dl_response.status_code} for {sha256[:16]}")
                            continue

                        # Check if response is a zip file
                        content = dl_response.content
                        if content[:2] != b"PK":
                            # Check if rate limited
                            if b"ratelimited" in content:
                                print(f"  [rate-limited] Waiting 60s...")
                                time.sleep(60)
                                # Retry this sample
                                dl_response = client.post(
                                    BAZAAR_API_URL,
                                    data={
                                        "query": "get_file",
                                        "sha256_hash": sha256,
                                    },
                                )
                                content = dl_response.content
                                if content[:2] != b"PK":
                                    print(f"  [dl-error] Still not a zip after retry")
                                    continue
                            else:
                                print(f"  [dl-error] Not a zip for {sha256[:16]}, got: {content[:100]}")
                                continue

                        # Extract from password-protected zip
                        try:
                            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                                names = zf.namelist()
                                if not names:
                                    continue
                                # Extract the first (and usually only) file
                                pe_bytes = zf.read(names[0], pwd=ZIP_PASSWORD)
                        except (zipfile.BadZipFile, RuntimeError):
                            continue

                        # Save the PE binary named by its SHA256
                        output_path = family_dir / sha256
                        output_path.write_bytes(pe_bytes)
                        seen_hashes.add(sha256)
                        downloaded += 1

                        if downloaded % 10 == 0:
                            print(f"  Downloaded {downloaded}/{remaining}")

                        # Rate limit: wait between downloads to avoid throttling
                        time.sleep(3)

                    except (httpx.TimeoutException, httpx.ConnectError) as e:
                        print(f"  [timeout] {e}. Retrying in 10s...")
                        time.sleep(10)
                        continue

                # MalwareBazaar rate limit: be polite
                time.sleep(1)

                # If we got fewer samples than requested, we've exhausted the tag
                if len(data) < batch_size:
                    print(f"  Exhausted available samples for tag '{tag}'")
                    break

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                print(f"  [error] {e}. Retrying in 10s...")
                time.sleep(10)
                continue
            except Exception as e:
                print(f"  [error] Unexpected: {e}")
                break

    total = existing_count + downloaded
    print(f"  Done: {downloaded} new downloads (total: {total})")
    return total


def main(args: argparse.Namespace) -> None:
    """Download samples for all target families."""
    print(f"MalwareBazaar Sample Downloader")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target per family: {args.limit}")
    print(f"{'='*50}\n")

    # Filter families if specified
    if args.families:
        families = {k: v for k, v in FAMILY_TAGS.items() if k in args.families}
    else:
        families = FAMILY_TAGS

    results: dict[str, int] = {}

    for family_name, tag in families.items():
        print(f"\n[{family_name}] (tag: {tag})")
        count = download_family_samples(
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

    print(f"\nNext step: python training/generate_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download malware from MalwareBazaar")
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_SAMPLES_PER_CLASS,
        help=f"Samples per family (default: {MAX_SAMPLES_PER_CLASS})",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Specific families to download (default: all)",
    )
    args = parser.parse_args()
    main(args)
