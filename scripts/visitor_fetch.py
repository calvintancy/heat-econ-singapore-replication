"""Cache current official visitor-arrival sources without changing economic inputs."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/visitor_arrivals"
DATASETS = {"nsa": "d_d1c33009b674dcf70b8e8c790b793f28",
            "sa": "d_c6d3f73a65d1fa3a92f1f4e70e619a66"}


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))

    def cache(name: str, url: str, **info) -> bytes:
        path = RAW / name
        if path.exists():
            content = path.read_bytes()
            assert hashlib.sha256(content).hexdigest() == manifest[name]["sha256"]
            return content
        response = session.get(url, timeout=90)
        response.raise_for_status()
        content = response.content
        path.write_bytes(content)
        manifest[name] = {"url": url, **info, "bytes": len(content),
                          "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                          "sha256": hashlib.sha256(content).hexdigest()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Cached {name}: {len(content)} bytes", flush=True)
        return content

    for adjustment, dataset in DATASETS.items():
        landing = f"https://data.gov.sg/datasets/{dataset}/view"
        metadata = json.loads(cache(f"{adjustment}_metadata.json",
            f"https://api-open.data.gov.sg/v2/public/api/datasets/{dataset}/metadata"))["data"]
        if not (RAW / f"{adjustment}.csv").exists():
            time.sleep(2)
            endpoint = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset}/poll-download"
            for attempt in range(10):
                response = session.get(endpoint, timeout=90)
                response.raise_for_status()
                signed = response.json().get("data", {}).get("url")
                if signed:
                    break
                time.sleep(min(2 * (attempt + 1), 15))
            else:
                raise RuntimeError(f"Download not ready: {dataset}")
            cache(f"{adjustment}.csv", signed, landing_url=landing,
                  dataset_id=dataset, source_updated=metadata["lastUpdatedAt"],
                  download_endpoint=endpoint)
            # Signed URLs expire and are not a reproducible source reference.
            manifest[f"{adjustment}.csv"]["url"] = endpoint
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        else:
            cache(f"{adjustment}.csv", landing)
        table = "M550001" if adjustment == "nsa" else "M550002"
        cache(f"{adjustment}_singstat_metadata.json",
              f"https://tablebuilder.singstat.gov.sg/api/table/metadata/{table}",
              landing_url=f"https://tablebuilder.singstat.gov.sg/table/TS/{table}")


if __name__ == "__main__":
    fetch()
