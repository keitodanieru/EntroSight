"""Check MalwareBazaar for available sample counts per family tag.

MalwareBazaar API docs: https://bazaar.abuse.ch/api/
This script queries the API to see how many samples are tagged for each family.
Does NOT download any files — just checks availability.

Usage:
    python training/check_bazaar.py
"""

import httpx
import sys

# Families to check — includes both our current and potential alternatives
FAMILIES_TO_CHECK = [
    "AgentTesla",
    "Remcos",
    "DCRat",
    "AsyncRAT",
    "RedLineStealer",
    "Formbook",
    "SnakeKeylogger",
    "Andromeda",
    "Mokes",
    # Additional from RawMal-TF dataset
    "Stealerc",
]

BAZAAR_API_URL = "https://mb-api.abuse.ch/api/v1/"


def check_family(family: str) -> dict:
    """Query MalwareBazaar for samples tagged with this family."""
    try:
        response = httpx.post(
            BAZAAR_API_URL,
            data={
                "query": "get_taginfo",
                "tag": family.lower(),
                "limit": 1,
            },
            timeout=30.0,
        )
        result = response.json()
        status = result.get("query_status", "unknown")

        if status == "ok":
            count = result.get("count", 0)
            return {"family": family, "status": "found", "count": count}
        elif status == "tag_not_found":
            return {"family": family, "status": "not_found", "count": 0}
        else:
            return {"family": family, "status": status, "count": 0}

    except Exception as e:
        return {"family": family, "status": f"error: {e}", "count": 0}


def main():
    """Check all families against MalwareBazaar."""
    print("Querying MalwareBazaar API for sample availability...\n")
    print(f"{'Family':<20} {'Status':<15} {'Samples':>10}")
    print(f"{'-'*45}")

    results = []
    for family in FAMILIES_TO_CHECK:
        result = check_family(family)
        results.append(result)
        print(f"{result['family']:<20} {result['status']:<15} {result['count']:>10}")

    print(f"\n{'='*45}")
    print("Families with enough samples (>= 1000):")
    viable = [r for r in results if r["count"] >= 1000]
    for r in sorted(viable, key=lambda x: x["count"], reverse=True):
        print(f"  {r['family']}: {r['count']}")

    if not viable:
        print("  None found with >= 1000. Try different tag names.")
        print("\nTip: MalwareBazaar tags are case-sensitive and may use")
        print("different naming than expected (e.g., 'AgentTesla' vs 'agenttesla')")


if __name__ == "__main__":
    main()
