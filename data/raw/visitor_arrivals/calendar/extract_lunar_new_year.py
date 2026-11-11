"""Cache HKO conversion tables and derive a monthly Lunar New Year control.

Run from any directory with the project's Python environment. Existing raw tables
are reused. Delete a raw table deliberately to refresh that year.
"""
from __future__ import annotations

import calendar
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "data/processed/visitor_arrivals/lunar_new_year.csv"
YEARS = list(range(2008, 2027))
BASE = "https://www.hko.gov.hk/en/gts/time/calendar/text/files/"


def fetch(year: int) -> dict:
    path = HERE / f"T{year}e.txt"
    url = BASE + path.name
    old_manifest = HERE / "manifest.json"
    old = json.loads(old_manifest.read_text()).get(path.name, {}) if old_manifest.exists() else {}
    retrieved = old.get("retrieved_utc")
    if not path.exists():
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Singapore visitor arrivals research)"}, timeout=45)
        response.raise_for_status()
        path.write_bytes(response.content)
        retrieved = datetime.now(timezone.utc).isoformat()
    raw = path.read_bytes()
    # Date and lunar-month labels are ASCII even in older non-UTF8 files.
    if retrieved is None:
        retrieved = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    matches = re.findall(rb"(?mi)^\s*(\d{4})/(\d{1,2})/(\d{1,2})\s+1st Lunar Month\s", raw)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one lunar new year in {path.name}: {matches}")
    day = date(*map(int, matches[0]))
    if day.year != year or day.month not in (1, 2):
        raise ValueError(f"Unexpected lunar new year date: {day}")
    gregorian_rows = re.findall(rb"(?m)^\s*(\d{4})/(\d{1,2})/(\d{1,2})\s", raw)
    parsed = [date(*map(int, parts)) for parts in gregorian_rows]
    expected = [date(year, 1, 1) + timedelta(days=i) for i in range(366 if calendar.isleap(year) else 365)]
    if parsed != expected:
        raise ValueError(f"Incomplete or reordered Gregorian year in {path.name}")
    return {"year": year, "lunar_new_year_date": day.isoformat(), "url": url,
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "retrieved_utc": retrieved, "gregorian_days_verified": len(parsed)}


def main() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, YEARS))
    monthly = []
    for row in records:
        new_year = date.fromisoformat(row["lunar_new_year_date"])
        window = [new_year + timedelta(days=i) for i in range(-7, 8)]
        for month in range(1, 13):
            days = sum(d.year == row["year"] and d.month == month for d in window)
            monthly.append({"month": f"{row['year']}-{month:02d}", "year": row["year"],
                            "lunar_new_year_date": new_year.isoformat(), "lny_window_days": days})
        assert sum(x["lny_window_days"] for x in monthly if x["year"] == row["year"]) == 15
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(monthly[0]))
        writer.writeheader()
        writer.writerows(monthly)
    (HERE / "lunar_new_year_dates.json").write_text(json.dumps(records, indent=2) + "\n")
    (HERE / "manifest.json").write_text(json.dumps({f"T{x['year']}e.txt": x for x in records}, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "monthly_rows": len(monthly),
                      "years": len(records), "dates": {x["year"]: x["lunar_new_year_date"] for x in records}}, indent=2))


if __name__ == "__main__":
    main()
