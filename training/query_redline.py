"""Query RedLine tag variations."""
import httpx, time

BAZAAR_API = "https://mb-api.abuse.ch/api/v1/"
BAZAAR_KEY = "f93c6fa5f5db6c37988038e9757789828e138eefbc7054f4"
client = httpx.Client(headers={"Auth-Key": BAZAAR_KEY}, timeout=60)

tags = ["RedLine", "redline", "RedLineStealer", "redlinestealer", "Redline", "RedLine Stealer"]

for tag in tags:
    r = client.post(BAZAAR_API, data={"query": "get_taginfo", "tag": tag, "limit": 1000})
    result = r.json()
    if result.get("query_status") == "ok":
        data = result.get("data", [])
        pe = sum(1 for s in data if s.get("file_type", "").lower() in ("exe", "dll"))
        print(f"{tag:<20} total={len(data):>5}  PE={pe:>5}")
    else:
        print(f"{tag:<20} {result.get('query_status')}")
    time.sleep(1)
