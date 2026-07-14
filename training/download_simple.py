"""Simple MalwareBazaar downloader — minimal, proven approach.

Downloads PE samples one at a time with full error reporting.
Based on the one-liner that confirmed working.

Usage:
    python -u training/download_simple.py AgentTesla 500
    python -u training/download_simple.py Remcos 500
"""

import io
import os
import sys
import time
import pyzipper

import httpx

BAZAAR_API = "https://mb-api.abuse.ch/api/v1/"
BAZAAR_KEY = "f93c6fa5f5db6c37988038e9757789828e138eefbc7054f4"
OUTPUT_BASE = "raw_samples"
DELAY = 5  # seconds between downloads


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_simple.py <FamilyTag> [limit]")
        print("  e.g: python download_simple.py AgentTesla 500")
        sys.exit(1)

    tag = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    out_dir = f"{OUTPUT_BASE}/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    # Get existing files
    existing = set(os.listdir(out_dir))
    print(f"Family: {tag}")
    print(f"Target: {limit}, Already have: {len(existing)}")
    print(f"Output: {out_dir}")
    print(f"Delay: {DELAY}s between downloads\n")

    if len(existing) >= limit:
        print("Already done!")
        return

    # Step 1: Get hashes
    print("Fetching sample list from MalwareBazaar...")
    client = httpx.Client(
        headers={"Auth-Key": BAZAAR_KEY},
        timeout=httpx.Timeout(120.0, connect=30.0),
    )

    r = client.post(BAZAAR_API, data={"query": "get_taginfo", "tag": tag, "limit": 1000})
    result = r.json()

    if result.get("query_status") != "ok":
        print(f"API error: {result.get('query_status')}")
        return

    all_samples = result.get("data", [])
    # Filter to PE only
    pe_samples = [s for s in all_samples if s.get("file_type", "").lower() in ("exe", "dll")]
    print(f"Total samples: {len(all_samples)}, PE files: {len(pe_samples)}")

    # Filter already downloaded
    to_download = [s for s in pe_samples if s["sha256_hash"] not in existing]
    print(f"New to download: {len(to_download)}\n")

    need = limit - len(existing)
    downloaded = 0

    for i, sample in enumerate(to_download):
        if downloaded >= need:
            break

        sha = sample["sha256_hash"]
        print(f"[{downloaded+1}/{need}] Downloading {sha[:20]}... ", end="", flush=True)

        try:
            r = client.post(BAZAAR_API, data={"query": "get_file", "sha256_hash": sha})

            if r.status_code != 200:
                print(f"HTTP {r.status_code}")
                time.sleep(DELAY)
                continue

            content = r.content

            # Check for rate limit
            if b"ratelimited" in content:
                print("RATE LIMITED - waiting 90s")
                time.sleep(90)
                continue

            # Check for zip
            if content[:2] != b"PK":
                print(f"Not zip ({len(content)}b): {content[:40]}")
                time.sleep(DELAY)
                continue

            # Extract
            zf = pyzipper.AESZipFile(io.BytesIO(content))
            names = zf.namelist()
            try:
                pe_bytes = zf.read(names[0], pwd=b"infected")
            except Exception as e:
                print(f"Extract failed: {e}")
                time.sleep(DELAY)
                continue

            # Save
            filepath = os.path.join(out_dir, sha)
            with open(filepath, "wb") as f:
                f.write(pe_bytes)

            downloaded += 1
            print(f"OK ({len(pe_bytes)} bytes)")

        except httpx.TimeoutException:
            print("TIMEOUT")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(DELAY)

    print(f"\nDone! Downloaded {downloaded} new files.")
    print(f"Total in {out_dir}: {len(os.listdir(out_dir))}")


if __name__ == "__main__":
    main()
