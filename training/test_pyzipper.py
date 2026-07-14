"""Test pyzipper extraction from MalwareBazaar."""
import httpx
import pyzipper
import io

c = httpx.Client(
    headers={"Auth-Key": "f93c6fa5f5db6c37988038e9757789828e138eefbc7054f4"},
    timeout=120,
)

r = c.post(
    "https://mb-api.abuse.ch/api/v1/",
    data={"query": "get_taginfo", "tag": "AgentTesla", "limit": 100},
)
exes = [s for s in r.json().get("data", []) if s.get("file_type") == "exe"]
sha = exes[0]["sha256_hash"]
print(f"Downloading {sha[:20]}...")

r2 = c.post(
    "https://mb-api.abuse.ch/api/v1/",
    data={"query": "get_file", "sha256_hash": sha},
)
print(f"Download size: {len(r2.content)} bytes")

zf = pyzipper.AESZipFile(io.BytesIO(r2.content))
data = zf.read(zf.namelist()[0], pwd=b"infected")
print(f"SUCCESS! Extracted: {len(data)} bytes, MZ header: {data[:2] == b'MZ'}")

path = f"raw_samples/AgentTesla/{sha}"
with open(path, "wb") as f:
    f.write(data)
print(f"Saved to {path}")
