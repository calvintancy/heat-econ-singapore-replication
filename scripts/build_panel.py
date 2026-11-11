"""Build non-overlapping broad and detailed sector panels from cached official data."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED = ROOT / "data/raw", ROOT / "data/processed"
BROAD = [f"1.1.{i}" for i in range(1, 5)] + [f"1.2.{i}" for i in range(1, 8)]


def quarter_period(label: str) -> pd.Period:
    year, quarter = re.fullmatch(r"(\d{4}) ([1-4])Q", label).groups()
    return pd.Period(f"{year}Q{quarter}", freq="Q")


def parse_va(name: str) -> pd.DataFrame:
    data = json.loads((RAW / f"{name}_tabledata.json").read_text())["Data"]
    metadata = json.loads((RAW / f"{name}_metadata.json").read_text())["Data"]["records"]
    rows = []
    for r in data["row"]:
        for c in r["columns"]:
            value = c["value"].replace(",", "")
            rows.append({"sector_id": r["seriesNo"], "sector": r["rowText"],
                         "quarter": quarter_period(c["key"]),
                         name: np.nan if value.lower() in {"na", "-", ""} else float(value)})
    df = pd.DataFrame(rows).sort_values(["sector_id", "quarter"])
    assert not df.duplicated(["sector_id", "quarter"]).any(), "Duplicate VA observations"
    assert len(df) == metadata["total"], "Truncated VA response; SingStat may require pagination"
    assert df.quarter.max() == quarter_period(metadata["endPeriod"]), "Truncated VA response"
    for sector_id, group in df.groupby("sector_id"):
        expected = pd.period_range(group.quarter.min(), group.quarter.max(), freq="Q")
        assert list(group.quarter) == list(expected), f"VA time gap: {sector_id}"
    return df


def weather_quarterly() -> pd.DataFrame:
    columns = {"temperature": "mean_temp", "humidity": "mean_rh", "rainfall": "total_rainfall"}
    frames = []
    for name, value in columns.items():
        frame = pd.read_csv(RAW / f"{name}_monthly.csv", na_values=["na", "NA", "-", ""])
        frame["month"] = pd.PeriodIndex(frame.month, freq="M")
        frame[value] = pd.to_numeric(frame[value], errors="raise")
        assert not frame.month.duplicated().any(), f"Duplicate {name} months"
        assert frame[value].notna().all(), f"Missing {name} observations"
        assert list(frame.sort_values("month").month) == list(pd.period_range(frame.month.min(), frame.month.max(), freq="M"))
        frames.append(frame.set_index("month")[[value]].rename(columns={value: name}))
    monthly = pd.concat(frames, axis=1).sort_index()
    assert monthly.temperature.dropna().between(20, 35).all()
    assert monthly.humidity.dropna().between(0, 100).all()
    assert monthly.rainfall.dropna().ge(0).all()
    monthly["days"] = monthly.index.days_in_month
    monthly["quarter"] = monthly.index.asfreq("Q")
    rows = []
    for quarter, group in monthly.groupby("quarter"):
        # Reject partial quarters and any month unavailable in one of the three sources.
        if len(group) != 3 or group[list(columns)].isna().any().any():
            continue
        days = group.days.sum()
        rows.append({
            "quarter": quarter,
            "temp_c": np.average(group.temperature, weights=group.days),
            "temp_c_unweighted": group.temperature.mean(),
            "rh_pct": np.average(group.humidity, weights=group.days),
            "rain_mm": group.rainfall.sum(),
            "days": days,
            "weather_months": len(group),
        })
    monthly.to_csv(PROCESSED / "weather_monthly.csv")
    quarterly = pd.DataFrame(rows).sort_values("quarter")
    assert len(quarterly) > 100 and quarterly.weather_months.eq(3).all()
    quarterly.to_csv(PROCESSED / "weather_quarterly.csv", index=False)
    return quarterly


def build() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sa, nsa = parse_va("va_sa"), parse_va("va_nsa")
    va = sa.merge(nsa, on=["sector_id", "sector", "quarter"], validate="one_to_one")
    for col in ("va_sa", "va_nsa"):
        assert va[col].dropna().gt(0).all(), "Non-positive VA prevents log growth"
        log_va = np.log(va[col])
        va[f"g_{col}_qoq"] = 100 * log_va.groupby(va.sector_id).diff()
        va[f"g_{col}_yoy"] = 100 * log_va.groupby(va.sector_id).diff(4)
    weather = weather_quarterly()
    panel = va.merge(weather, on="quarter", validate="many_to_one")
    panel["year"] = panel.quarter.dt.year
    panel["season"] = panel.quarter.dt.quarter
    panel["time"] = panel.quarter.astype("int64") - pd.Period("1982Q1").ordinal
    panel["trend"] = panel.time / 4
    panel["rain_100mm"] = panel.rain_mm / 100
    for lag in range(1, 5):
        weather[f"temp_l{lag}"] = weather.temp_c.shift(lag)
    panel = panel.merge(weather[["quarter"] + [f"temp_l{i}" for i in range(1, 5)]], on="quarter", validate="many_to_one")
    parent_ids = set(va.sector_id)
    detailed = [s for s in parent_ids if any(s == b or s.startswith(b + ".") for b in BROAD)
                and not any(other.startswith(s + ".") for other in parent_ids)]
    assert len(BROAD) == 11 and len(detailed) == 19
    catalog = va[["sector_id", "sector"]].drop_duplicates()
    catalog["broad_panel"] = catalog.sector_id.isin(BROAD)
    catalog["detailed_panel"] = catalog.sector_id.isin(detailed)
    catalog.to_csv(PROCESSED / "sector_catalog.csv", index=False)
    summary = {"weather_sources": {}}
    for name, column in {"temperature": "mean_temp", "rainfall": "total_rainfall", "humidity": "mean_rh"}.items():
        monthly_source = pd.read_csv(RAW / f"{name}_monthly.csv")
        info_path = RAW / f"{name}_metadata.json"
        meta = json.loads(info_path.read_text())["data"] if info_path.exists() else {}
        summary["weather_sources"][name] = {
            "observed_start": str(monthly_source.month.min()),
            "observed_end": str(monthly_source.month.max()), "months": len(monthly_source),
            "metadata_coverage_end": meta.get("coverageEnd"),
            "missing_values": int(monthly_source[column].isna().sum()),
        }
    for name, ids, start in [("broad", BROAD, "1982Q1"), ("detailed", detailed, "2005Q2")]:
        frame = panel[panel.sector_id.isin(ids) & panel.quarter.ge(pd.Period(start))].copy()
        assert frame.g_va_sa_qoq.notna().all()
        counts = frame.groupby("quarter").sector_id.nunique()
        assert counts.eq(len(ids)).all(), f"Unbalanced {name} panel"
        assert not frame.duplicated(["sector_id", "quarter"]).any()
        frame = frame.sort_values(["sector_id", "quarter"])
        frame.to_csv(PROCESSED / f"panel_{name}.csv", index=False)
        summary[name] = {"n_sectors": len(ids), "n_quarters": len(counts), "n_obs": len(frame),
                         "start": str(frame.quarter.min()), "end": str(frame.quarter.max()),
                         "missing_va_sa": int(frame.va_sa.isna().sum())}
        print(f"{name}: {summary[name]}")
    panel[panel.sector_id.eq("1.4")].to_csv(PROCESSED / "aggregate_gva.csv", index=False)
    (PROCESSED / "data_checks.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    build()
