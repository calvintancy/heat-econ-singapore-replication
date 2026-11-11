"""Build one portable HTML report directly from the saved analysis outputs."""
from __future__ import annotations

import base64
import csv
import json
from datetime import datetime
from pathlib import Path

from regression_tables import table_fragment
import pandas as pd
from sector_readout import sector_readout

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def records(name: str) -> list[dict]:
    numeric = {"coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "r_squared",
               "n_obs", "n_sectors", "n_quarters", "hac_lags", "inference_df",
               "temp_residual", "growth_residual"}
    with (OUT / name).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row.keys() & numeric:
            row[key] = float(row[key])
    return rows


def build_html() -> Path:
    manifest = json.loads((ROOT / "data/raw/manifest.json").read_text())
    retrieved = max(datetime.fromisoformat(item["retrieved_utc"]) for item in manifest.values())
    data = {
        "results": records("regression_results.csv"),
        "sectors": records("sector_slopes.csv"),
        "quarters": records("partial_regression_quarter_means.csv"),
        "lags": records("distributed_lag_results.csv"),
        "lag_sum": json.loads((OUT / "distributed_lag_sum.json").read_text())["sector_cluster"],
        "checks": json.loads((OUT / "estimation_checks.json").read_text()),
        "verification": json.loads((OUT / "verification.json").read_text()),
        "data_checks": json.loads((ROOT / "data/processed/data_checks.json").read_text()),
        "snapshot": retrieved.strftime("%d %B %Y").lstrip("0"),
        "snapshot_short": retrieved.strftime("%B %Y"),
        "va_updated": manifest["va_sa_tabledata.json"]["source_last_updated"],
        "downloads": {},
    }
    for name in ["regression_results.csv", "sector_slopes.csv", "distributed_lag_results.csv", "verification.json"]:
        mime = "application/json" if name.endswith(".json") else "text/csv;charset=utf-8"
        encoded = base64.b64encode((OUT / name).read_bytes()).decode()
        data["downloads"][name] = f"data:{mime};base64,{encoded}"
    data["downloads"]["source_manifest.json"] = "data:application/json;base64," + base64.b64encode(
        (ROOT / "data/raw/manifest.json").read_bytes()).decode()
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (ROOT / "templates/results.html").read_text()
    va_table = table_fragment(OUT / "regression_results.csv", "QoQ real VA log growth (SA)", "quarterly",
                             ["S1", "S2", "S3", "S4", "S5"],
                             ["Temperature\nonly", "+ Rainfall", "+ Humidity\nBaseline", "Before\nCOVID", "Exclude\n2020–2022"], suffix="va_main")
    html = template.replace("__REPORT_DATA__", payload).replace("__VA_REGRESSION_TABLE__", va_table)
    html = html.replace("__VA_SECTOR_READOUT__", sector_readout(pd.read_csv(OUT/"sector_slopes.csv", dtype={"sector_id":str}), OUT/"regression_results.csv"))
    assert "__REPORT_DATA__" not in html
    path = OUT / "heat_econ_results.html"
    path.write_text(html)
    print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes, self-contained)")
    return path


if __name__ == "__main__":
    build_html()
