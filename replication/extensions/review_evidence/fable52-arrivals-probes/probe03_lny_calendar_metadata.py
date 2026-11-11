"""Probe 03 - HKO Lunar New Year calendar, January-2008 PPML retention, raw metadata.

Re-extracts every "1st Lunar month" anchor directly from the 20 official HKO
byte tables (2007 supplement + 2008-2026), rebuilds the -7..+7 inclusive 15-day
monthly window with datetime arithmetic, compares all 228 cached months, checks
December 2007 and the actual January-2008 PPML rows, and re-hashes the raw
manifests.  Also records the M550001 adjustment-label quirk against the data.
Writes audit_probes/probe03_result.json.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe03_lny_calendar_metadata.py
"""
from _probe_common import BASE, ROOT, sha, write

import csv
import json
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

RAW = ROOT / "data/raw/visitor_arrivals"
CAL = RAW / "calendar"
SUP = ROOT / "replication/extensions/supplementary_inputs/calendar"
ROW = re.compile(rb"^\s*(\d{4})/(\d{1,2})/(\d{1,2})\s+(.*?)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.M)
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

result = {"probe": "03 LNY calendar / Jan-2008 retention / raw metadata"}
anchors, tables = {}, {}
for year in range(2007, 2027):
    path = (SUP if year == 2007 else CAL) / f"T{year}e.txt"
    data = path.read_bytes()
    rows = [(date(int(a), int(b), int(c)), d.decode("ascii", "replace").strip(), e.decode()) for a, b, c, d, e in ROW.findall(data)]
    days = [r[0] for r in rows]
    expected = [date(year, 1, 1) + timedelta(days=i) for i in range((date(year + 1, 1, 1) - date(year, 1, 1)).days)]
    firsts = [r for r in rows if re.fullmatch(r"1st Lunar month", r[1], flags=re.I)]
    weekday_ok = all(WEEK[r[0].weekday()] == r[2] for r in rows)
    previous = rows[days.index(firsts[0][0]) - 1][1] if len(firsts) == 1 else None
    tables[year] = {"sha256": sha(path), "bytes": len(data), "gregorian_rows": len(rows), "all_days_present_in_order": days == expected,
                    "printed_weekdays_match_gregorian": weekday_ok, "first_lunar_month_rows": len(firsts),
                    "lunar_day_before_anchor": previous, "anchor": str(firsts[0][0]) if len(firsts) == 1 else None}
    assert len(firsts) == 1 and days == expected and weekday_ok and previous in ("29", "30"), (year, tables[year])
    anchors[year] = firsts[0][0]
result["tables_parsed"] = len(tables)
result["anchor_2007"] = str(anchors[2007])
result["anchors_all_in_jan21_feb20"] = all((a.month, a.day) >= (1, 21) and (a.month, a.day) <= (2, 20) for a in anchors.values())

# hashes: supplement vs its sources.json and the hard-coded production constant; 2008-2026 vs calendar manifest
sources = json.loads((SUP / "sources.json").read_text())
manifest = json.loads((CAL / "manifest.json").read_text())
old_json = json.loads((CAL / "lunar_new_year_dates.json").read_text())
result["hashes"] = {
    "supplement_sha256": tables[2007]["sha256"], "supplement_matches_sources_json": tables[2007]["sha256"] == sources["sha256"] and tables[2007]["bytes"] == sources["bytes"],
    "supplement_matches_production_constant": tables[2007]["sha256"] == "2eeec2ef316267d92c2ec93e44d2ccb3a889e02c408aecdcc696a2c350416673",
    "supplement_declared_anchor_matches": sources["lunar_new_year"] == str(anchors[2007]),
    "old_tables_matching_calendar_manifest": int(sum(tables[y]["sha256"] == manifest[f"T{y}e.txt"]["sha256"] and tables[y]["bytes"] == manifest[f"T{y}e.txt"]["bytes"] for y in range(2008, 2027))),
    "old_anchor_json_matches_independent_extraction": int(sum(str(anchors[item["year"]]) == item["lunar_new_year_date"] for item in old_json)), "old_anchor_json_rows": len(old_json)}
raw_manifest = json.loads((RAW / "manifest.json").read_text())
result["hashes"]["raw_arrival_manifest_files_matching"] = {name: sha(RAW / name) == info["sha256"] and (RAW / name).stat().st_size == info["bytes"] for name, info in raw_manifest.items()}
monthly_manifest = json.loads((BASE / "monthly/manifest.json").read_text())
result["hashes"]["monthly_manifest_raw_inputs_match"] = {name: sha(ROOT / name) == value for name, value in monthly_manifest["raw_inputs_sha256"].items()}

# independent 15-day window calendar, 2007-01 .. 2026-12
window = {}
for anchor in anchors.values():
    for offset in range(-7, 8):
        day = anchor + timedelta(days=offset)
        window[(day.year, day.month)] = window.get((day.year, day.month), 0) + 1
months = [(y, m) for y in range(2007, 2027) for m in range(1, 13)]
mine = {f"{y:04d}-{m:02d}": window.get((y, m), 0) for y, m in months}
cached = pd.read_csv(ROOT / "data/processed/visitor_arrivals/lunar_new_year.csv")
cached_map = dict(zip(cached.month, cached.lny_window_days.astype(int)))
result["calendar"] = {
    "cached_months": len(cached), "cached_first": cached.month.iloc[0], "cached_last": cached.month.iloc[-1],
    "cached_months_equal_independent": int(sum(mine[m] == v for m, v in cached_map.items())),
    "cached_anchor_column_equals_independent": bool(all(str(anchors[int(y)]) == d for y, d in zip(cached.year, cached.lunar_new_year_date))),
    "december_2007_window_days": mine["2007-12"], "all_2007_months": {k: v for k, v in mine.items() if k.startswith("2007") and v},
    "yearly_sums_all_15": all(sum(mine[f"{y:04d}-{m:02d}"] for m in range(1, 13)) == 15 for y in range(2007, 2027)),
    "january_2008_window_days": mine["2008-01"], "january_2008_lag1": mine["2007-12"],
    "any_december_ever_nonzero": any(mine[f"{y:04d}-12"] for y in range(2007, 2027)),
    "window_definition": "anchor-7 .. anchor+7 inclusive = 15 days"}
extended = pd.read_csv(BASE / "ppml/lunar_calendar_extended.csv")
ext_ok = all(mine[m] == int(v) for m, v in zip(extended.month, extended.lny_window_days))
lag_ok = all((np.isnan(l) and m == "2007-12") or (not np.isnan(l) and int(l) == mine[f"{int(m[:4]) - (1 if m[5:] == '01' else 0):04d}-{(int(m[5:]) - 2) % 12 + 1:02d}"]) for m, l in zip(extended.month, extended.lny_lag1))
result["calendar"]["ppml_extended_calendar_rows"] = len(extended)
result["calendar"]["ppml_extended_calendar_equals_independent"] = bool(ext_ok and lag_ok)

# actual January-2008 retention in the count models (all five cuts, long controls, all eligible)
retention = {}
specs = pd.read_csv(BASE / "ppml/specifications.csv", low_memory=False)
for cut in ["aggregate", "region", "country", "complete10", "complete52"]:
    spec = specs[specs.spec_id.eq(f"ppml__{cut}__long__all_eligible")].iloc[0]
    sample = pd.read_csv(ROOT / spec.sample_csv, dtype={"unit_id": str})
    jan = sample[sample.month.eq("2008-01")]
    with np.load(ROOT / spec.design_npz) as z:
        npz_months = z["months"]
        terms = [str(t) for t in z["terms"]]
        x_jan = z["X"][npz_months == "2008-01"]
    retention[cut] = {"n_periods": int(sample.month.nunique()), "first_month": sample.month.min(), "january_2008_rows": len(jan), "units": int(sample.unit_id.nunique()),
                      "january_2008_lny_lag1_values": sorted(set(jan.lny_lag1.tolist())), "january_2008_lny_window_values": sorted(set(jan.lny_window_days.tolist())),
                      "design_january_rows": int(len(x_jan)), "design_january_lny_lag1": sorted(set(x_jan[:, terms.index("lny_lag1")].tolist())),
                      "lny_lag1_equals_independent_all_rows": bool(all(mine[f"{int(m[:4]) - (1 if m[5:] == '01' else 0):04d}-{(int(m[5:]) - 2) % 12 + 1:02d}"] == v for m, v in zip(sample.month, sample.lny_lag1))),
                      "lny_window_equals_independent_all_rows": bool(all(mine[m] == v for m, v in zip(sample.month, sample.lny_window_days)))}
result["january_2008_ppml_retention"] = retention
legacy = pd.read_csv(ROOT / "data/processed/visitor_arrivals/weather.csv")
result["legacy_cached_weather_january_2008_lny_lag1"] = None if pd.isna(legacy.loc[legacy.month.eq("2008-01"), "lny_lag1"].iloc[0]) else float(legacy.loc[legacy.month.eq("2008-01"), "lny_lag1"].iloc[0])

# M550001 label quirk: metadata field versus data
nsa_meta = json.loads((RAW / "nsa_singstat_metadata.json").read_text())["Data"]["records"]
sa_meta = json.loads((RAW / "sa_singstat_metadata.json").read_text())["Data"]["records"]


def series(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0][1:], {row[0].strip(): row[1:] for row in rows[1:]}


nsa_header, nsa = series(RAW / "nsa.csv")
sa_header, sa = series(RAW / "sa.csv")
total_name = nsa_meta["row"][0]["rowText"].strip()
nsa_total = dict(zip(nsa_header, nsa[total_name]))
sa_key = next(k for k in sa if k.lower().startswith("total"))
sa_total = dict(zip(sa_header, sa[sa_key]))
common = [m for m in nsa_header if m in sa_total and sa_total[m] not in ("", "na")]
a = np.array([float(nsa_total[m]) for m in common])
b = np.array([float(sa_total[m]) for m in common])
order = np.argsort([int(m[:4]) * 12 + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(m[4:]) for m in common])
a, b = a[order], b[order]
month_of = np.array([["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(common[i][4:]) for i in order])
keep = slice(0, 144)  # 2008-2019 only: before the COVID level break


def seasonal_share(x):
    d = np.diff(np.log(x[keep]))
    m = month_of[keep][1:]
    fitted = np.array([d[m == k].mean() for k in range(12)])[m]
    return float(1 - np.sum((d - fitted) ** 2) / np.sum((d - d.mean()) ** 2))


result["m550001_adjustment_label"] = {
    "M550001_metadata_adjustmentType": nsa_meta.get("adjustmentType"), "M550001_title": nsa_meta.get("title"), "M550001_dataSource": nsa_meta.get("dataSource"),
    "M550002_metadata_adjustmentType": sa_meta.get("adjustmentType"), "M550002_title": sa_meta.get("title"),
    "nsa_rows": len(nsa), "sa_rows": len(sa), "common_total_months": len(common),
    "total_series_identical_months": int(np.sum(a == b)), "max_relative_gap_total_nsa_vs_sa": float(np.max(np.abs(a / b - 1))),
    "all_M550001_cells_integer": bool(all(v.strip().lstrip("-").isdigit() for row in nsa.values() for v in row)),
    "calendar_month_R2_of_MoM_log_growth_2008_2019": {"M550001": seasonal_share(a), "M550002": seasonal_share(b)},
    "disclosed_at": "notes/visitor_arrivals_methods.md:21",
    "reading": "metadata field contradicts the NSA label, but the M550001 values differ from the explicit SA table and carry strong calendar-month seasonality that M550002 lacks"}
ok = (result["calendar"]["cached_months_equal_independent"] == 228 and result["calendar"]["december_2007_window_days"] == 0
      and result["hashes"]["old_tables_matching_calendar_manifest"] == 19 and result["hashes"]["supplement_matches_sources_json"]
      and all(r["january_2008_rows"] == r["units"] and r["january_2008_lny_lag1_values"] == [0.0] and r["n_periods"] == 225 for r in retention.values()))
result["verdict"] = "calendar, hashes and January-2008 retention reproduce" if ok else "DISCREPANCY - inspect"
result["tables"] = tables
write("probe03_result.json", result)
print(json.dumps({k: v for k, v in result.items() if k != "tables"}, indent=1))
