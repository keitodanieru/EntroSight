"""Simulate the HTMX browser flow: expect HTML fragments + HX-Redirect on complete."""

from __future__ import annotations

import json
import os
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
HX = {"HX-Request": "true"}


def _multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = "----entrosighthtmx"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


def main() -> None:
    payload = b"MZ" + os.urandom(4096)
    body, boundary = _multipart("file", "htmx_sample.exe", payload)

    req = urllib.request.Request(
        f"{BASE}/api/scan",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **HX},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode()
    print("POST status:", resp.status)
    print("POST is HTML fragment:", "scan-status" in html and "hx-get" in html)
    print("POST contains raw JSON:", html.strip().startswith("{"))

    # Extract scan_id from the polling URL in the fragment
    import re

    m = re.search(r"/api/scan/([0-9a-f-]+)/status", html)
    scan_id = m.group(1)
    print("scan_id:", scan_id)

    for _ in range(60):
        req = urllib.request.Request(f"{BASE}/api/scan/{scan_id}/status", headers=HX)
        # Don't auto-follow; inspect HX-Redirect header on completion
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(req, timeout=30) as r:
            status_code = r.status
            hx_redirect = r.headers.get("HX-Redirect")
            frag = r.read().decode()
        if hx_redirect:
            print("POLL -> HX-Redirect:", hx_redirect)
            # Fetch the result page
            with urllib.request.urlopen(f"{BASE}{hx_redirect}", timeout=30) as rp:
                page = rp.read().decode()
            print("RESULT page status:", rp.status)
            print("RESULT has verdict badge:", "verdict-badge" in page)
            print("RESULT has heatmap img src:", "/static/heatmaps/" in page)
            break
        print("POLL fragment status:", status_code, "| has steps:",
              "step__label" in frag)
        time.sleep(1)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401
        return None


if __name__ == "__main__":
    main()
