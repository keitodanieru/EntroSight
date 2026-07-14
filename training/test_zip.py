"""Quick test to figure out MalwareBazaar zip password/format."""
import httpx
import zipfile
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
print(f"Hash: {sha}")

r2 = c.post(
    "https://mb-api.abuse.ch/api/v1/",
    data={"query": "get_file", "sha256_hash": sha},
)
print(f"Download size: {len(r2.content)} bytes")
print(f"Starts with PK: {r2.content[:2] == b'PK'}")

zf = zipfile.ZipFile(io.BytesIO(r2.content))
print(f"Files in zip: {zf.namelist()}")
info = zf.infolist()[0]
print(f"Compress type: {info.compress_type}")
print(f"Flag bits: {info.flag_bits}")
print(f"Compress size: {info.compress_size}")
print(f"File size: {info.file_size}")

passwords = [b"infected", b"infected!", b"bazaar", b"malware", b"virus", b"zip", b"sample", b"dangerous", b"password", None]
for pwd in passwords:
    try:
        data = zf.read(info.filename, pwd=pwd)
        print(f"\nSUCCESS! Password: {pwd}")
        print(f"Extracted size: {len(data)} bytes")
        print(f"First 4 bytes: {data[:4]}")
        break
    except RuntimeError as e:
        print(f"  {pwd}: {e}")
    except Exception as e:
        print(f"  {pwd}: {type(e).__name__}: {e}")
