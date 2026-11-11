"""Audit published VA aggregates without imposing additivity on chain volumes.

Own-source cache and outputs are separate from the estimation pipeline.  The
independent check reads the raw JSON again and uses Decimal loop arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
CACHE = RAW / "va_reconciliation"
OUT = ROOT / "outputs/reconciliation"
NOMINAL_TABLE = "M015651"
BROAD = [f"1.1.{i}" for i in range(1, 5)] + [f"1.2.{i}" for i in range(1, 8)]
SPLITS = {
    "1.2.1": ["1.2.1.1", "1.2.1.2"],
    "1.2.3": ["1.2.3.1", "1.2.3.2"],
    "1.2.6": ["1.2.6.1", "1.2.6.2", "1.2.6.3"],
    "1.2.7": [f"1.2.7.{i}" for i in range(1, 6)],
}
DETAILED = [leaf for broad in BROAD for leaf in SPLITS.get(broad, [broad])]
IDENTITIES = {
    "broad11_without_dwellings": (BROAD, "1.4", "11 panel sectors only: omits dwellings"),
    "broad11_plus_dwellings": (BROAD + ["1.3"], "1.4", "11 broad sectors + ownership of dwellings = GVA"),
    "top_groups_plus_dwellings": (["1.1", "1.2", "1.3"], "1.4", "Goods + services + dwellings = GVA"),
    "goods4": ([f"1.1.{i}" for i in range(1, 5)], "1.1", "Four goods sectors = goods-producing industries"),
    "services7": ([f"1.2.{i}" for i in range(1, 8)], "1.2", "Seven broad service sectors = services-producing industries"),
    "gva_plus_taxes": (["1.4", "1.5"], "1", "GVA + taxes on products = GDP"),
    "detailed19_plus_dwellings": (DETAILED + ["1.3"], "1.4", "19 detailed sectors + ownership of dwellings = GVA"),
    **{f"split_{broad}": (leaves, broad, f"Detailed children = broad parent {broad}") for broad, leaves in SPLITS.items()},
}
DOCUMENTS = {
    "benchmark_2015": {
        "title": "Benchmarking of Singapore's National Accounts to Reference Year 2015 (May 2019), methodological appendix, printed page 18",
        "url": "https://www.singstat.gov.sg/files/6448c773-d7c9-4798-9ac7-68ba46e19efb.pdf",
        "landing_url": "https://www.singstat.gov.sg/publication-resources/benchmarking-of-singapore-national-accounts-to-reference-year-2015-information-paper",
    },
    "quarterly_gdp": {
        "title": "Quarterly Estimates of Output-Based GDP (July 2023)",
        "url": "https://www.singstat.gov.sg/files/3cb0efc7-28d4-43b1-ad97-3ff03907c585.pdf",
        "landing_url": "https://www.singstat.gov.sg/publication-resources/quarterly-estimates-of-output-based-gdp",
    },
    "seasonal_adjustment": {
        "title": "Seasonal Adjustment of Economic Time Series (September 2006), paragraphs 18–22; historical methods",
        "url": "https://www.singstat.gov.sg/files/0937df16-76d1-405e-ab24-d31aceea7055.pdf",
        "landing_url": "https://www.singstat.gov.sg/publication-resources/seasonal-adjustment-of-economic-time-series-information-paper",
    },
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _period(label: str) -> pd.Period:
    year, q = re.fullmatch(r"(\d{4}) ([1-4])Q", label).groups()
    return pd.Period(f"{year}Q{q}", freq="Q")


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (official Singapore national-accounts reconciliation)"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    return session


def _fetch_sources(refresh: bool) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    session = _session()

    def save(name: str, body: bytes, details: dict) -> None:
        path = CACHE / name
        path.write_bytes(body)
        manifest[name] = {
            **details, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": _hash(path), "bytes": len(body),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def get(name: str, url: str, *, params: dict | None = None, extra: dict | None = None) -> bytes:
        path = CACHE / name
        if path.exists() and not refresh:
            if name not in manifest or _hash(path) != manifest[name]["sha256"]:
                raise ValueError(f"Source cache/hash mismatch: {path}")
            return path.read_bytes()
        response = session.get(url, params=params, timeout=90)
        response.raise_for_status()
        body = response.content
        if name.endswith(".pdf"):
            assert body.startswith(b"%PDF"), f"Not a PDF: {url}"
        else:
            payload = response.json()
            assert payload.get("StatusCode") == 200 and payload.get("Data"), f"Bad SingStat response: {url}"
        save(name, body, {"url": response.url, **(extra or {})})
        return body

    base = "https://tablebuilder.singstat.gov.sg/api/table"
    meta = json.loads(get("current_price_metadata.json", f"{base}/metadata/{NOMINAL_TABLE}"))["Data"]["records"]
    assert meta["id"] == NOMINAL_TABLE
    provenance = {
        "table_id": NOMINAL_TABLE, "source_last_updated": meta["dataLastUpdated"],
        "title": meta["title"], "landing_url": f"https://tablebuilder.singstat.gov.sg/table/TS/{NOMINAL_TABLE}",
    }
    merged_name = "current_price_tabledata.json"
    if not (CACHE / merged_name).exists() or refresh:
        offset, merged, pages, template = 0, {}, [], None
        while offset < meta["total"]:
            name = f"current_price_tabledata_page_{offset:05d}.json"
            page = json.loads(get(name, f"{base}/tabledata/{NOMINAL_TABLE}",
                                  params={"limit": 5000, "offset": offset}, extra=provenance))
            assert page["Data"]["id"] == NOMINAL_TABLE
            if template is None:
                template = page
            count = 0
            for row in page["Data"]["row"]:
                count += len(row["columns"])
                number = row["seriesNo"]
                if number not in merged:
                    merged[number] = {**row, "columns": []}
                merged[number]["columns"].extend(row["columns"])
            assert count > 0, "Empty API page before declared total"
            pages.append(name)
            offset += count
        assert offset == meta["total"], "API pagination failed completeness check"
        template["Data"]["row"] = list(merged.values())
        save(merged_name, (json.dumps(template, ensure_ascii=False) + "\n").encode(),
             {**provenance, "assembled_from": pages, "url": f"{base}/tabledata/{NOMINAL_TABLE}"})
    else:
        assert _hash(CACHE / merged_name) == manifest[merged_name]["sha256"]
    for key, details in DOCUMENTS.items():
        get(f"{key}.pdf", details["url"], extra={k: v for k, v in details.items() if k != "url"})
    return manifest


def _read_source(data_path: Path, metadata_path: Path, source: str) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(data_path.read_text())["Data"]
    meta = json.loads(metadata_path.read_text())["Data"]["records"]
    assert payload["id"] == meta["id"]
    assert payload["dataLastUpdated"] == meta["dataLastUpdated"]
    rows = []
    for row in payload["row"]:
        for cell in row["columns"]:
            string = cell["value"].replace(",", "")
            value = np.nan if string.lower() in {"na", "-", ""} else float(string)
            rows.append({"series_id": row["seriesNo"], "series": row["rowText"],
                         "quarter": _period(cell["key"]), "value": value})
    frame = pd.DataFrame(rows).sort_values(["series_id", "quarter"])
    assert len(frame) == meta["total"], "Incomplete cell download"
    assert not frame.duplicated(["series_id", "quarter"]).any(), "Duplicate source cells"
    assert frame.quarter.min() == _period(meta["startPeriod"])
    assert frame.quarter.max() == _period(meta["endPeriod"])
    assert set(frame.series_id) == {r["seriesNo"] for r in meta["row"]}
    for _, group in frame.groupby("series_id"):
        assert list(group.quarter) == list(pd.period_range(group.quarter.min(), group.quarter.max(), freq="Q"))
    info = {
        "source": source, "table_id": meta["id"], "title": meta["title"],
        "adjustment": meta.get("adjustmentType"), "last_updated": meta["dataLastUpdated"],
        "start": str(frame.quarter.min()), "end": str(frame.quarter.max()),
        "rows": int(frame.series_id.nunique()), "cells": len(frame),
        "missing_values": int(frame.value.isna().sum()), "footnote": meta.get("footnote"),
        "data_path": str(data_path.relative_to(ROOT)), "metadata_path": str(metadata_path.relative_to(ROOT)),
        "data_sha256": _hash(data_path), "metadata_sha256": _hash(metadata_path),
        "data_url": f"https://tablebuilder.singstat.gov.sg/api/table/tabledata/{meta['id']}",
        "metadata_url": f"https://tablebuilder.singstat.gov.sg/api/table/metadata/{meta['id']}",
        "landing_url": f"https://tablebuilder.singstat.gov.sg/table/TS/{meta['id']}",
    }
    return frame, info


def _reconcile(frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = frame.pivot(index="quarter", columns="series_id", values="value")
    records, annual_records = [], []
    for key, (components, target, label) in IDENTITIES.items():
        available = wide[[target] + components].dropna()
        for quarter, row in available.iterrows():
            total = row[components].sum()
            gap = total - row[target]
            records.append({
                "source": source, "identity": key, "label": label, "quarter": str(quarter),
                "target_id": target, "component_ids": "|".join(components), "n_components": len(components),
                "component_sum": total, "official_total": row[target],
                "gap_million": gap, "gap_pct": 100 * gap / row[target],
                "rounding_bound_million": (len(components) + 1) * .05,
                "expected_additive": source == "current_price_nsa" and key != "broad11_without_dwellings",
            })
        # Annual entries require four complete quarters for every component and target.
        for year, group in available.groupby(available.index.year):
            if len(group) != 4:
                continue
            component_sum = group[components].sum().sum()
            official_total = group[target].sum()
            gap = component_sum - official_total
            annual_records.append({
                "source": source, "identity": key, "label": label, "year": int(year),
                "target_id": target, "component_ids": "|".join(components), "n_components": len(components),
                "component_sum": component_sum, "official_total": official_total,
                "gap_million": gap, "gap_pct": 100 * gap / official_total,
                "rounding_bound_million": 4 * (len(components) + 1) * .05,
                "expected_additive": source == "current_price_nsa" and key != "broad11_without_dwellings",
            })
    return pd.DataFrame(records), pd.DataFrame(annual_records)


def _taxonomy(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for source, frame in frames.items():
        for broad in BROAD:
            for leaf in SPLITS.get(broad, [broad]):
                values = frame[frame.series_id.eq(leaf) & frame.value.notna()]
                first, last = values.quarter.min(), values.quarter.max()
                rows.append({
                    "source": source, "broad_id": broad,
                    "broad_sector": frame.loc[frame.series_id.eq(broad), "series"].iloc[0],
                    "detailed_id": leaf, "detailed_sector": values.series.iloc[0],
                    "is_split": leaf != broad, "level_start": str(first), "level_end": str(last),
                    "first_qoq_from_full_raw": str(first + 1), "first_yoy_from_full_raw": str(first + 4),
                    "n_levels": len(values), "in_broad11": leaf in BROAD, "in_detailed19": True,
                    "dwellings_in_panel": False,
                })
    taxonomy = pd.DataFrame(rows)
    for source in frames:
        group = taxonomy[taxonomy.source.eq(source)]
        assert len(group) == 19 and group.detailed_id.nunique() == 19
        assert group.level_start.max() == "2005Q1"
    return taxonomy


def _shares_and_nominal_contributions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Nominal shares and nominal contributions only; these do not reconstruct CVM.
    wide = frame.pivot(index="quarter", columns="series_id", values="value")
    annual = wide.groupby(wide.index.year).sum(min_count=4)
    names = frame.drop_duplicates("series_id").set_index("series_id").series.to_dict()
    share_rows, contribution_rows = [], []
    for granularity, leaves in [("broad11_plus_dwellings", BROAD + ["1.3"]),
                                ("detailed19_plus_dwellings", DETAILED + ["1.3"])]:
        complete = annual[leaves + ["1.4"]].dropna()
        for year, row in complete.iterrows():
            for leaf in leaves:
                share_rows.append({"year": int(year), "granularity": granularity,
                                   "series_id": leaf, "series": names[leaf],
                                   "nominal_va_million": row[leaf], "nominal_gva_million": row["1.4"],
                                   "share_of_nominal_gva_pct": 100 * row[leaf] / row["1.4"],
                                   "weight_interpretation": "Current-price composition share; not a fixed real-VA aggregation weight"})
        for year in complete.index:
            if year - 1 not in complete.index:
                continue
            previous = complete.loc[year - 1]
            current = complete.loc[year]
            official_growth = 100 * (current["1.4"] / previous["1.4"] - 1)
            for leaf in leaves:
                contribution_rows.append({"year": int(year), "granularity": granularity,
                                          "series_id": leaf, "series": names[leaf],
                                          "nominal_growth_contribution_pp": 100 * (current[leaf] - previous[leaf]) / previous["1.4"],
                                          "official_nominal_gva_growth_pct": official_growth,
                                          "interpretation": "Exact nominal change contribution up to reported rounding; not a real-growth contribution"})
    return pd.DataFrame(share_rows), pd.DataFrame(contribution_rows)


def _growth_comparison(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for source in ["real_nsa", "real_sa"]:
        wide = frames[source].pivot(index="quarter", columns="series_id", values="value")
        official = wide["1.4"]
        reconstructed = wide[BROAD + ["1.3"]].sum(axis=1, min_count=12)
        sector_growth = 100 * np.log(wide[BROAD]).diff()
        for quarter in wide.index[1:]:
            rows.append({
                "source": source, "quarter": str(quarter),
                "official_aggregate_qoq_log_growth": 100 * np.log(official.loc[quarter] / official.loc[quarter - 1]),
                "sum_chain_levels_qoq_log_growth": 100 * np.log(reconstructed.loc[quarter] / reconstructed.loc[quarter - 1]),
                "equal_sector_mean_qoq_log_growth": sector_growth.loc[quarter].mean(),
                "interpretation": "Only official aggregate growth is national GVA growth; chain-level sum and equal-sector mean are diagnostic comparators",
            })
    return pd.DataFrame(rows)


def _reference_levels(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    real = frames["real_nsa"]
    nominal = frames["current_price_nsa"]
    real = real[real.quarter.dt.year.eq(2015)]
    nominal = nominal[nominal.quarter.dt.year.eq(2015)]
    matched = real.merge(nominal, on=["series_id", "series", "quarter"], validate="one_to_one", suffixes=("_real", "_nominal"))
    matched["period"] = matched.quarter.astype(str)
    annual = matched.groupby(["series_id", "series"], as_index=False)[["value_real", "value_nominal"]].sum()
    annual["period"] = "2015 annual"
    result = pd.concat([matched.drop(columns="quarter"), annual], ignore_index=True)
    result["real_minus_current_price_million"] = result.value_real - result.value_nominal
    return result.sort_values(["series_id", "period"])


def _independent_checks(sources: dict[str, tuple[Path, Path]], quarterly: pd.DataFrame,
                        annual: pd.DataFrame, shares: pd.DataFrame,
                        contributions: pd.DataFrame, growth: pd.DataFrame,
                        reference_levels: pd.DataFrame) -> dict:
    """Re-read raw JSON independently; no pivot/reconciliation helpers in arithmetic."""
    qlookup = quarterly.set_index(["source", "identity", "quarter"])
    alookup = annual.set_index(["source", "identity", "year"])
    n_q = n_a = 0
    max_q = max_a = 0.
    raw_values = {}
    for source, (path, _) in sources.items():
        values = {}
        for row in json.loads(path.read_text())["Data"]["row"]:
            sid = row["seriesNo"]
            for cell in row["columns"]:
                v = cell["value"].replace(",", "")
                if v.lower() in {"na", "-", ""}:
                    continue
                year, season = cell["key"].split()
                key = f"{year}Q{season[0]}"
                assert (sid, key) not in values
                values[(sid, key)] = Decimal(v)
        raw_values[source] = values
        for (s, identity, period), actual in qlookup.iterrows():
            if s != source:
                continue
            component_ids, target, _ = IDENTITIES[identity]
            amount = Decimal(0)
            for sid in component_ids:
                amount += values[(sid, period)]
            target_amount = values[(target, period)]
            error = max(abs(float(amount) - actual.component_sum),
                        abs(float(target_amount) - actual.official_total),
                        abs(float(amount - target_amount) - actual.gap_million),
                        abs(float(100 * (amount - target_amount) / target_amount) - actual.gap_pct))
            max_q = max(max_q, error)
            assert error < 2e-8, (s, identity, period, error)
            n_q += 1
        for (s, identity, year), actual in alookup.iterrows():
            if s != source:
                continue
            component_ids, target, _ = IDENTITIES[identity]
            amount = Decimal(0)
            target_amount = Decimal(0)
            for q in range(1, 5):
                period = f"{year}Q{q}"
                target_amount += values[(target, period)]
                for sid in component_ids:
                    amount += values[(sid, period)]
            error = max(abs(float(amount) - actual.component_sum),
                        abs(float(target_amount) - actual.official_total),
                        abs(float(amount - target_amount) - actual.gap_million))
            max_a = max(max_a, error)
            assert error < 2e-8, (s, identity, year, error)
            n_a += 1
    expected_nominal = quarterly[quarterly.expected_additive]
    assert (expected_nominal.gap_million.abs() <= expected_nominal.rounding_bound_million + 1e-8).all()
    expected_annual = annual[annual.expected_additive]
    assert (expected_annual.gap_million.abs() <= expected_annual.rounding_bound_million + 1e-8).all()
    # Verify the annual CVM reference-year property empirically, not at quarterly or SA frequency.
    reference = annual[annual.source.eq("real_nsa") & annual.year.isin([2015, 2016]) & ~annual.identity.eq("broad11_without_dwellings")]
    assert (reference.gap_million.abs() <= reference.rounding_bound_million + 1e-8).all()
    nominal_raw = raw_values["current_price_nsa"]
    max_share_error = max_contribution_error = 0.
    for row in shares.itertuples():
        numerator = sum((nominal_raw[(row.series_id, f"{row.year}Q{q}")] for q in range(1, 5)), Decimal(0))
        denominator = sum((nominal_raw[("1.4", f"{row.year}Q{q}")] for q in range(1, 5)), Decimal(0))
        error = abs(float(100 * numerator / denominator) - row.share_of_nominal_gva_pct)
        max_share_error = max(max_share_error, error)
        assert error < 1e-10
    for row in contributions.itertuples():
        delta, denominator = Decimal(0), Decimal(0)
        for q in range(1, 5):
            delta += nominal_raw[(row.series_id, f"{row.year}Q{q}")] - nominal_raw[(row.series_id, f"{row.year-1}Q{q}")]
            denominator += nominal_raw[("1.4", f"{row.year-1}Q{q}")]
        error = abs(float(100 * delta / denominator) - row.nominal_growth_contribution_pp)
        max_contribution_error = max(max_contribution_error, error)
        assert error < 1e-10
    for row in reference_levels.itertuples():
        periods = [f"2015Q{q}" for q in range(1, 5)] if row.period == "2015 annual" else [row.period]
        real_amount = sum((raw_values["real_nsa"][(row.series_id, period)] for period in periods), Decimal(0))
        nominal_amount = sum((nominal_raw[(row.series_id, period)] for period in periods), Decimal(0))
        assert abs(float(real_amount - nominal_amount) - row.real_minus_current_price_million) < 1e-8
        if row.period == "2015 annual":
            assert real_amount == nominal_amount, "Annual reference-year real/nominal levels differ"
    max_growth_error = 0.
    for row in growth.itertuples():
        values = raw_values[row.source]
        year, quarter = int(row.quarter[:4]), int(row.quarter[-1])
        prior = f"{year}Q{quarter-1}" if quarter > 1 else f"{year-1}Q4"
        official = 100 * math.log(float(values[("1.4", row.quarter)] / values[("1.4", prior)]))
        current_sum, previous_sum = Decimal(0), Decimal(0)
        sector_growth = []
        for sid in BROAD + ["1.3"]:
            current_sum += values[(sid, row.quarter)]
            previous_sum += values[(sid, prior)]
            if sid in BROAD:
                sector_growth.append(100 * math.log(float(values[(sid, row.quarter)] / values[(sid, prior)])))
        sum_growth = 100 * math.log(float(current_sum / previous_sum))
        equal_growth = math.fsum(sector_growth) / 11
        error = max(abs(official - row.official_aggregate_qoq_log_growth),
                    abs(sum_growth - row.sum_chain_levels_qoq_log_growth),
                    abs(equal_growth - row.equal_sector_mean_qoq_log_growth))
        max_growth_error = max(max_growth_error, error)
        assert error < 1e-10
    assert n_q == len(quarterly) and n_a == len(annual)
    return {
        "status": "passed", "independent_method": "Raw JSON re-read, Decimal loops, no DataFrame sums",
        "quarterly_identities_checked": n_q, "annual_identities_checked": n_a,
        "nominal_share_cells_checked": len(shares), "nominal_contribution_cells_checked": len(contributions),
        "max_quarterly_arithmetic_difference": max_q, "max_annual_arithmetic_difference": max_a,
        "max_nominal_share_difference_pp": max_share_error,
        "max_nominal_contribution_difference_pp": max_contribution_error,
        "reference_year_series_checked": int(reference_levels.series_id.nunique()),
        "reference_year_nsa_annual_real_equals_nominal_exactly": True,
        "growth_comparison_rows_checked": len(growth), "max_growth_comparison_difference": max_growth_error,
        "all_current_price_complete_identities_within_rounding": True,
        "real_nsa_2015_and_2016_annual_within_rounding": True,
        "no_rescaling_or_forced_reconciliation": True,
    }


def _chart(quarterly: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
        "axes.linewidth": .6, "axes.edgecolor": "0.2", "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True,
    })
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, constrained_layout=True)
    labels = {"real_nsa": "Real chained-2015, NSA", "real_sa": "Real chained-2015, SA", "current_price_nsa": "Current-price, NSA"}
    colors = {"real_nsa": "#1f77b4", "real_sa": "#d62728", "current_price_nsa": "0.45"}
    for ax, identity, title in zip(axes, ["broad11_plus_dwellings", "top_groups_plus_dwellings"],
                                  ["11 broad sectors + dwellings − official GVA", "Goods + services + dwellings − official GVA"]):
        selected = quarterly[quarterly.identity.eq(identity)]
        for source, group in selected.groupby("source"):
            x = pd.PeriodIndex(group.quarter, freq="Q").to_timestamp()
            ax.plot(x, group.gap_pct, color=colors[source], lw=1.15,
                    ls="--" if source == "current_price_nsa" else "-", label=labels[source])
        ax.axhline(0, color="0.35", lw=.6)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("% of official GVA")
        ax.grid(axis="y", color="#e3e6e8", lw=.6)
        ax.tick_params(length=2.5, color="0.4")
    axes[0].legend(frameon=False, ncols=3, fontsize=8)
    fig.savefig(OUT / "va_reconciliation.png", dpi=300)
    plt.close(fig)


def run_va_reconciliation(refresh: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # Confirm that this additional audit never changes the two regression inputs.
    original_paths = [RAW / f"{name}_{kind}.json" for name in ["va_nsa", "va_sa"] for kind in ["metadata", "tabledata"]]
    original_hashes = {str(path): _hash(path) for path in original_paths}
    original_manifest = json.loads((RAW / "manifest.json").read_text())
    for path in original_paths:
        assert original_manifest[path.name]["sha256"] == _hash(path), f"Existing source hash changed: {path}"
    cache_manifest = _fetch_sources(refresh)
    sources = {
        "real_nsa": (RAW / "va_nsa_tabledata.json", RAW / "va_nsa_metadata.json"),
        "real_sa": (RAW / "va_sa_tabledata.json", RAW / "va_sa_metadata.json"),
        "current_price_nsa": (CACHE / "current_price_tabledata.json", CACHE / "current_price_metadata.json"),
    }
    frames, source_info, quarters, years = {}, {}, [], []
    for source, (data_path, metadata_path) in sources.items():
        frame, info = _read_source(data_path, metadata_path, source)
        frames[source], source_info[source] = frame, info
        q, a = _reconcile(frame, source)
        quarters.append(q)
        years.append(a)
    vintages = {info["last_updated"] for info in source_info.values()}
    if len(vintages) != 1:
        raise ValueError(f"Source vintage mismatch: {source_info}")
    assert len(BROAD) == 11 and len(DETAILED) == 19
    assert all(not any(other.startswith(sid + ".") for other in DETAILED) for sid in DETAILED)
    quarterly = pd.concat(quarters, ignore_index=True)
    annual = pd.concat(years, ignore_index=True)
    taxonomy = _taxonomy(frames)
    shares, contributions = _shares_and_nominal_contributions(frames["current_price_nsa"])
    growth = _growth_comparison(frames)
    reference_levels = _reference_levels(frames)
    verification = _independent_checks(sources, quarterly, annual, shares, contributions, growth, reference_levels)
    for path in original_paths:
        assert original_hashes[str(path)] == _hash(path), f"Audit modified existing source: {path}"
    verification["original_va_source_hashes_unchanged"] = True
    quarterly.to_csv(OUT / "va_quarterly.csv", index=False)
    annual.to_csv(OUT / "va_annual.csv", index=False)
    taxonomy.to_csv(OUT / "va_taxonomy.csv", index=False)
    shares.to_csv(OUT / "va_annual_current_price_shares.csv", index=False)
    contributions.to_csv(OUT / "va_annual_nominal_growth_contributions.csv", index=False)
    growth.to_csv(OUT / "va_growth_comparison.csv", index=False)
    reference_levels.to_csv(OUT / "va_reference_year_levels.csv", index=False)
    statistics = []
    for (source, identity), group in quarterly.groupby(["source", "identity"], sort=False):
        group = group.sort_values("quarter")
        latest = group.iloc[-1]
        extreme = group.loc[group.gap_million.abs().idxmax()]
        pct_extreme = group.loc[group.gap_pct.abs().idxmax()]
        statistics.append({
            "source": source, "identity": identity, "label": latest.label,
            "start": group.quarter.min(), "end": group.quarter.max(), "n_quarters": len(group),
            "latest_component_sum": float(latest.component_sum), "latest_official_total": float(latest.official_total),
            "latest_gap_million": float(latest.gap_million), "latest_gap_pct": float(latest.gap_pct),
            "max_abs_gap_million": float(abs(extreme.gap_million)), "max_abs_gap_quarter": extreme.quarter,
            "max_abs_gap_pct": float(abs(pct_extreme.gap_pct)), "max_abs_gap_pct_quarter": pct_extreme.quarter,
            "rounding_bound_million": float(latest.rounding_bound_million),
            "expected_additive": bool(latest.expected_additive),
            "all_within_rounding": bool((group.gap_million.abs() <= group.rounding_bound_million + 1e-8).all()),
        })
    pd.DataFrame(statistics).to_csv(OUT / "va_identity_summary.csv", index=False)
    reference = annual[annual.year.isin([2015, 2016]) & annual.identity.isin([
        "broad11_plus_dwellings", "top_groups_plus_dwellings", "gva_plus_taxes", "detailed19_plus_dwellings"])]
    summary = {
        "title": "Official sector-to-aggregate value-added reconciliation",
        "units": "Singapore dollars, million; real sources are chained-2015 dollars",
        "gap_definition": "sum of listed components minus the published official total",
        "as_of_source_update": next(iter(vintages)), "sources": source_info,
        "findings": [
            "The 11-sector panel omits ownership of dwellings. Its equal-sector mean growth is not national GVA growth.",
            "Adding dwellings completes category coverage, but chain-linked levels remain non-additive and must not be forced to match.",
            "At current prices the complete NSA identities reconcile within reported rounding on every available quarter.",
            "NSA annual chain-volume identities reconcile within rounding in 2015 and 2016; quarterly and seasonally adjusted identities need not.",
            "The 2019 benchmarking paper specifies direct seasonal adjustment for all real national-account series, superseding its earlier indirect approach.",
            "Current-price shares and nominal growth contributions are composition measures; they are not exact real chain-volume weights or real growth contributions.",
        ],
        "method_sources": DOCUMENTS, "identity_statistics": statistics,
        "reference_year_annual_checks": reference.to_dict(orient="records"),
        "taxonomy": {
            "broad_count": len(BROAD), "detailed_count": len(DETAILED),
            "broad_ids": BROAD, "detailed_ids": DETAILED,
            "splits": SPLITS, "balanced_levels_start": "2005Q1", "balanced_qoq_start": "2005Q2",
            "balanced_yoy_start": "2006Q1", "end": source_info["real_sa"]["end"],
            "balanced_2005q1_levels": len(pd.period_range("2005Q1", source_info["real_sa"]["end"], freq="Q")),
            "balanced_qoq_from_2005q2_dates": len(pd.period_range("2005Q2", source_info["real_sa"]["end"], freq="Q")),
            "balanced_yoy_from_2006q1_dates": len(pd.period_range("2006Q1", source_info["real_sa"]["end"], freq="Q")),
            "dwellings_id": "1.3", "dwellings_in_panel": False,
            "note": "Seven unsplit sectors have pre-2005 histories; 12 split leaves begin in 2005Q1. Common short-growth/YoY/level comparisons should start in 2006Q1.",
        },
        "verification": verification,
        "files": {"quarterly": "outputs/reconciliation/va_quarterly.csv",
                  "annual": "outputs/reconciliation/va_annual.csv",
                  "identity_summary": "outputs/reconciliation/va_identity_summary.csv",
                  "taxonomy": "outputs/reconciliation/va_taxonomy.csv",
                  "nominal_shares": "outputs/reconciliation/va_annual_current_price_shares.csv",
                  "nominal_contributions": "outputs/reconciliation/va_annual_nominal_growth_contributions.csv",
                  "growth_comparison": "outputs/reconciliation/va_growth_comparison.csv",
                  "reference_year_levels": "outputs/reconciliation/va_reference_year_levels.csv",
                  "chart": "outputs/reconciliation/va_reconciliation.png",
                  "source_manifest": "data/raw/va_reconciliation/manifest.json",
                  "methods": "notes/va_reconciliation_methods.md"},
        "source_cache_manifest": cache_manifest,
    }
    (OUT / "va_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (OUT / "va_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    _chart(quarterly)
    print(f"VA reconciliation passed: {len(quarterly):,} quarterly and {len(annual):,} annual identities; "
          f"same source vintage {next(iter(vintages))}; no rescaling.")
    return summary


if __name__ == "__main__":
    run_va_reconciliation()
