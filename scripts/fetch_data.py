"""Cache official source files and record their provenance; refresh only on request."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
TABLES = {"va_nsa": "M015661", "va_sa": "M015662",
          "iip_nsa": "M355381", "iip_sa": "M355382", "iip_quarterly_nsa": "M355391"}
WEATHER = {
    "temperature": ("d_755290a24afe70c8f9e8bcbf9f251573", "mean_temp"),
    "rainfall": ("d_b16d06b83473fdfcc92ed9d37b66ba58", "total_rainfall"),
    "humidity": ("d_e3b1111a26b2aefef4f386c068d4ca5d", "mean_rh"),
}


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (Singapore heat and real VA research)"
    retries = Retry(total=6, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def fetch(refresh: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    s = session()

    def save(name: str, content: bytes, provenance: dict) -> None:
        path = RAW / name
        path.write_bytes(content)
        manifest[name] = {
            **provenance,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {name}: {len(content):,} bytes", flush=True)

    for name, table in TABLES.items():
        for kind in ("metadata", "tabledata"):
            filename = f"{name}_{kind}.json"
            if (RAW / filename).exists() and not refresh:
                continue
            url = f"https://tablebuilder.singstat.gov.sg/api/table/{kind}/{table}"
            r = s.get(url, params={"limit": 5000, "offset": 0} if kind == "tabledata" else None, timeout=90)
            r.raise_for_status()
            payload = r.json()
            if payload.get("StatusCode") != 200 or not payload.get("Data"):
                raise ValueError(f"Unexpected SingStat response for {table}/{kind}")
            d = payload["Data"]
            info = d.get("records", d)
            if info.get("id") != table:
                raise ValueError(f"Wrong table returned for {table}")
            provenance = {
                "source": info.get("datasource", info.get("dataSource", "Singapore Department of Statistics")),
                "table_id": table, "url": r.url,
                "landing_url": f"https://tablebuilder.singstat.gov.sg/table/TS/{table}",
                "title": info["title"], "source_last_updated": info["dataLastUpdated"],
            }
            content = r.content
            if kind == "tabledata":
                meta = json.loads((RAW / f"{name}_metadata.json").read_text())["Data"]["records"]
                expected = meta["total"]
                cells = sum(len(row["columns"]) for row in d["row"])
                if expected > cells:
                    pages = []
                    merged = {row["seriesNo"]: {**row, "columns": list(row["columns"])} for row in d["row"]}
                    page_name = f"{name}_tabledata_page_00000.json"
                    save(page_name, r.content, provenance)
                    pages.append(page_name)
                    while cells < expected:
                        response = s.get(url, params={"limit": 5000, "offset": cells}, timeout=90)
                        response.raise_for_status()
                        page = response.json()
                        page_rows = page["Data"]["row"]
                        count = sum(len(row["columns"]) for row in page_rows)
                        if not count:
                            raise ValueError(f"Empty SingStat page at {cells}/{expected}")
                        page_name = f"{name}_tabledata_page_{cells:05d}.json"
                        save(page_name, response.content, {**provenance, "url": response.url})
                        pages.append(page_name)
                        for row in page_rows:
                            if row["seriesNo"] in merged:
                                merged[row["seriesNo"]]["columns"].extend(row["columns"])
                            else:
                                merged[row["seriesNo"]] = {**row, "columns": list(row["columns"])}
                        cells += count
                    d["row"] = list(merged.values())
                    provenance["assembled_from"] = pages
                    content = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
                if cells != expected:
                    raise ValueError(f"Incomplete table {table}: {cells}/{expected} cells")
            save(filename, content, provenance)

    for name, (dataset, column) in WEATHER.items():
        metadata_name = f"{name}_metadata.json"
        if not (RAW / metadata_name).exists() or refresh:
            metadata_url = f"https://api-open.data.gov.sg/v2/public/api/datasets/{dataset}/metadata"
            r = s.get(metadata_url, timeout=90)
            r.raise_for_status()
            info = r.json()["data"]
            save(metadata_name, r.content, {
                "source": "National Environment Agency", "dataset_id": dataset,
                "url": metadata_url, "source_last_updated": info.get("lastUpdatedAt"),
            })
            time.sleep(13)
        filename = f"{name}_monthly.csv"
        if (RAW / filename).exists() and not refresh:
            continue
        # The API is rate limited and the signed download URL must be used immediately.
        url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset}/poll-download"
        for attempt in range(8):
            r = s.get(url, timeout=90)
            r.raise_for_status()
            payload = r.json()
            signed = payload.get("data", {}).get("url")
            if signed:
                break
            time.sleep(min(3 * (attempt + 1), 20))
        else:
            raise RuntimeError(f"Download not ready for {dataset}: {payload}")
        download = s.get(signed, timeout=90)
        download.raise_for_status()
        header = download.content.decode("utf-8-sig").splitlines()[0].strip().split(",")
        if header != ["month", column]:
            raise ValueError(f"Unexpected weather file for {dataset}: {header}")
        save(filename, download.content, {
            "source": "National Environment Agency (Changi Climate Station)",
            "dataset_id": dataset, "url": url,
            "landing_url": f"https://data.gov.sg/datasets/{dataset}/view",
            "value_column": column,
        })
        time.sleep(13)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Replace cached sources with current vintages")
    fetch(parser.parse_args().refresh)
