"""End-to-end smoke test: upload a synthetic PE file and poll for the result."""

from __future__ import annotations

import io
import os
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def _multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = "----entrosightsmoke"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


def main() -> None:
    # Synthetic PE: MZ signature + random-ish bytes
    payload = b"MZ" + os.urandom(4096)
    body, boundary = _multipart("file", "sample.exe", payload)

    req = urllib.request.Request(
        f"{BASE}/api/scan",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        import json

        submit = json.loads(resp.read())
    print("SUBMIT", resp.status, submit)

    scan_id = submit["scan_id"]
    for _ in range(60):
        with urllib.request.urlopen(f"{BASE}/api/scan/{scan_id}/status", timeout=30) as r:
            import json

            status = json.loads(r.read())
        print("STATUS", status.get("status"), status.get("progress_stage"))
        if status.get("status") in ("complete", "error"):
            print("FINAL", json.dumps(status, indent=2, default=str))
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
