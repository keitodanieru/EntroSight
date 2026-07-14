"""Query MalwareBazaar with multiple tag variations per family to find more PE samples."""
import httpx
import time

BAZAAR_API = "https://mb-api.abuse.ch/api/v1/"
BAZAAR_KEY = "f93c6fa5f5db6c37988038e9757789828e138eefbc7054f4"

# Multiple tag variations per family
FAMILY_TAGS = {
    "AgentTesla": ["AgentTesla", "agenttesla", "Agent Tesla", "agent_tesla", "AgenTesla"],
    "Remcos": ["Remcos", "remcos", "RemcosRAT", "remcos_rat"],
    "DCRat": ["DCRat", "dcrat", "DarkCrystal", "DarkCrystalRAT", "dc_rat"],
    "FormBook": ["Formbook", "formbook", "FormBook", "XLoader", "xloader"],
    "RedLine": ["RedLine", "redline", "RedLineStealer", "redlinestealer", "RedLine Stealer"],
    "AsyncRAT": ["AsyncRAT", "asyncrat", "Async RAT", "async_rat"],
}

client = httpx.Client(
    headers={"Auth-Key": BAZAAR_KEY},
    timeout=60,
)


def query_tag(tag: str) -> dict:
    """Query a single tag and return PE count + total."""
    try:
        r = client.post(BAZAAR_API, data={"query": "get_taginfo", "tag": tag, "limit": 1000})
        result = r.json()
        if result.get("query_status") != "ok":
            return {"tag": tag, "total": 0, "pe": 0, "status": result.get("query_status")}
        
        data = result.get("data", [])
        pe_count = sum(1 for s in data if s.get("file_type", "").lower() in ("exe", "dll"))
        return {"tag": tag, "total": len(data), "pe": pe_count, "status": "ok"}
    except Exception as e:
        return {"tag": tag, "total": 0, "pe": 0, "status": str(e)}


print("Querying MalwareBazaar for tag variations...\n")
print(f"{'Family':<15} {'Tag':<20} {'Total':>8} {'PE files':>10} {'Status'}")
print(f"{'-'*65}")

best_tags = {}

for family, tags in FAMILY_TAGS.items():
    best_pe = 0
    best_tag = tags[0]
    
    for tag in tags:
        result = query_tag(tag)
        marker = ""
        if result["pe"] > best_pe:
            best_pe = result["pe"]
            best_tag = tag
            marker = " ◄ best"
        print(f"{family:<15} {tag:<20} {result['total']:>8} {result['pe']:>10} {result['status']}{marker}")
        time.sleep(1)
    
    best_tags[family] = {"tag": best_tag, "pe": best_pe}
    print()

print(f"\n{'='*65}")
print(f"BEST TAG PER FAMILY")
print(f"{'='*65}")
print(f"{'Family':<15} {'Best Tag':<20} {'PE samples':>12}")
print(f"{'-'*50}")
for family, info in best_tags.items():
    print(f"{family:<15} {info['tag']:<20} {info['pe']:>12}")
