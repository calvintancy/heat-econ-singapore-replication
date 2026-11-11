"""Audit published IIP hierarchy, release weights and annual-overlap candidates.

This never estimates weights or replaces the official total with a synthetic index.
All index comparisons use the cached October 2026 SingStat vintage.  Release
weights keep their own reference year and vintage; they are not extrapolated.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/reconciliation"
OUT = ROOT / "outputs/reconciliation"
CLUSTERS = [f"1.{i}" for i in range(1, 7)]
PARENTS = ["1", *CLUSTERS]
LANDING = "https://www.edb.gov.sg/en/manufacturing-statistics"
ASSETS = "https://assets.app.optical.gov.sg/edb/production/published/collections/pages/en%252Fmanufacturing-statistics/"
LEGACY = "https://www.edb.gov.sg/content/dam/edb-en/about-edb/media-releases/manufacturing-statistics/"
SOURCES = {
    "iip_manufacturing_landing.html": (LANDING, "EDB manufacturing statistics landing page"),
    "iip_memorandum_rebase_2025.pdf": (ASSETS+"db0f05be-8254-45d7-8ff3-2b9fde4f10a6.pdf", "EDB February 2026 rebase memorandum, 2025=100"),
    "iip_chain_link_information_2019.pdf": (LEGACY+"chain_linking_of_monthly_index_of_industrial_production/Info-paper-on-Chain-Link-IIP-(2019100).pdf", "EDB February 2021 chain-link information paper"),
    "iip_monthly_release_current.pdf": (ASSETS+"6766e558-d08e-4e23-a2eb-037c7aa367b4.pdf", "EDB September 2026 release, 26 October 2026"),
    "iip_cma2024.xlsx": (ASSETS+"4d6e4074-1d7d-44a6-adbd-0eff407f8a0d.xlsx", "EDB Census of Manufacturing Activities 2024, Table 15"),
    "iip_cma2024_technical_notes.docx": ("https://assets.app.optical.gov.sg/edb/production/published/collections/corporate_news/en%252Fmonthly-manufacturing-performance/697e5caa-8b2d-4e25-b1f2-af33370e8e99.docx", "EDB CMA2024 technical notes; census scope and VA definition"),
    "iip_cma_va_metadata.json": ("https://tablebuilder.singstat.gov.sg/api/table/metadata/M355181", "CMA value added by cluster metadata"),
    "iip_cma_va_tabledata.json": ("https://tablebuilder.singstat.gov.sg/api/table/tabledata/M355181?limit=100000", "CMA value added by cluster, 2000–2025; 2025 preliminary"),
    "iip_dos_seasonal_adjustment_2006.pdf": ("https://www.singstat.gov.sg/files/0937df16-76d1-405e-ab24-d31aceea7055.pdf", "DOS Seasonal Adjustment of Economic Time Series, 2006; still linked from current DOS explainer"),
}
RELEASES = [
    (2022, 12, "2019=100", "iip_sgpc_detail_2022_12.json", "/media_releases/edb/press_release/P-20230126-1"),
    (2023, 1, "2019=100", "iip_sgpc_detail_2023.json", "/media_releases/edb/press_release/P-20230223-1"),
    (2024, 1, "2019=100", "iip_sgpc_detail_2024.json", "/media_releases/edb/press_release/P-20240223-2"),
    (2025, 1, "2019=100", "iip_sgpc_detail_2025.json", "/media_releases/edb/press_release/P-20250225-1"),
    (2026, 9, "2025=100", None, None),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def release_path(year: int, month: int) -> Path:
    return RAW / ("iip_monthly_release_current.pdf" if year == 2026 else f"iip_monthly_release_{year}_{month:02d}.pdf")


def cache_sources(fetch: bool = False) -> dict:
    """Fetch only explicitly requested/missing sources; preserve the cached vintage."""
    RAW.mkdir(parents=True, exist_ok=True)
    sources = dict(SOURCES)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Singapore heat-economy research)"
    for year, month, _, detail_name, release_url in RELEASES:
        if detail_name is None:
            continue
        url = "https://www.sgpc.gov.sg/api/ApiMainListing/Detail?"+urlencode({"url": release_url})
        detail = RAW / detail_name
        if fetch or not detail.exists():
            response = session.get(url, timeout=45)
            response.raise_for_status()
            json_file(detail, response.json())
        sources[detail_name] = (url, f"Official SGPC EDB release metadata, {year}-{month:02d}")
        documents = json.loads(detail.read_text())["result"]["RelatedDocuments"]
        pdfs = [d for d in documents if d["FileName"].lower().endswith(".pdf") and "infograph" not in d["FileName"].lower()]
        assert len(pdfs) == 1, (year, pdfs)
        doc = pdfs[0]
        url = "https://www.sgpc.gov.sg/api/file/getfile/"+quote(doc["FileName"])+"?"+urlencode({"path": doc["Path"]})
        sources[release_path(year, month).name] = (url, f"EDB monthly release for {year}-{month:02d}, original 2019-based vintage")
    for listing in RAW.glob("iip_sgpc_listing_*.json"):
        year = int(listing.stem.rsplit("_", 1)[-1])
        params = {"KeySearch": "Monthly Manufacturing Performance", "Agency": "EDB", "StartDate": f"01-Feb-{year}", "EndDate": f"28-Feb-{year}", "page": 1}
        sources[listing.name] = ("https://www.sgpc.gov.sg/api/ApiMainListing/SubmitData?"+urlencode(params), f"SGPC February{year} EDB release search; an empty result is not proof that older weights do not exist")
    manifest_path = RAW / "iip_source_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for name, (url, title) in sources.items():
        path = RAW / name
        if fetch or not path.exists():
            response = session.get(url, timeout=45)
            response.raise_for_status()
            if path.suffix == ".pdf":
                assert response.content.startswith(b"%PDF"), url
            if path.suffix == ".xlsx":
                assert response.content.startswith(b"PK"), url
            if path.suffix == ".json":
                response.json()
            path.write_bytes(response.content)
        manifest[name] = {
            "url": url, "title": title, "bytes": path.stat().st_size,
            "sha256": sha256(path), "retrieved_utc": old.get(name, {}).get("retrieved_utc", datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()),
        }
    json_file(manifest_path, manifest)
    return manifest


def compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().replace(" cluster", ""))


def source_levels(adjustment: str) -> tuple[pd.DataFrame, dict]:
    name = f"iip_{adjustment}"
    path = ROOT / "data/raw" / f"{name}_tabledata.json"
    payload = json.loads(path.read_text())["Data"]
    metadata_path = path.with_name(f"{name}_metadata.json")
    meta = json.loads(metadata_path.read_text())["Data"]["records"]
    frame = pd.DataFrame([
        {"sector_id": row["seriesNo"], "sector": row["rowText"], "month": pd.Period(pd.to_datetime(cell["key"], format="%Y %b"), freq="M"), "index": float(cell["value"])}
        for row in payload["row"] for cell in row["columns"]
    ])
    assert len(frame) == meta["total"]
    assert not frame.duplicated(["sector_id", "month"]).any()
    assert frame["index"].gt(0).all()
    for _, group in frame.groupby("sector_id"):
        assert group.month.nunique() == len(pd.period_range(group.month.min(), group.month.max(), freq="M"))
    return frame, {"table_id": meta["id"], "last_updated": meta["dataLastUpdated"], "cells": len(frame),
                   "tabledata_sha256": sha256(path), "metadata_sha256": sha256(metadata_path)}


def parse_weights(catalog: dict) -> pd.DataFrame:
    aliases = {compact_name(name): sector_id for sector_id, name in catalog.items()}
    aliases.update({"totalmanufacturing": "1", "generalmanufacturingindustries": "1.6", "specialties": "1.2.3", "others": "1.2.4"})
    records = []
    for year, month, index_base, _, _ in RELEASES:
        path = release_path(year, month)
        text = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
        pages = text.split("\f")
        if year < 2026:
            section = text.split("TABLE 2 : PERFORMANCE BY CLUSTERS", 1)[1].split("TABLE 3", 1)[0]
        else:
            section = text.split("PERFORMANCE BY CLUSTERS", 1)[1].split("The next", 1)[0]
        found = {}
        for line in section.splitlines():
            match = re.match(r"^\s*([A-Za-z].*?)\s{2,}(-?\d+\.\d+)(?:\s|$)", line)
            if match is None:
                continue
            label = match.group(1).strip()
            sector_id = aliases.get(compact_name(label))
            if sector_id is None:
                continue
            weight = float(match.group(2))
            assert sector_id not in found
            found[sector_id] = weight
            page_number = next(i+1 for i, page in enumerate(pages) if line in page)
            records.append({
                "weight_set": f"release_{year}_{month:02d}", "effective_year": year,
                "release_month": f"{year}-{month:02d}", "index_base": index_base,
                "sector_id": sector_id, "sector": catalog[sector_id],
                "parent_id": sector_id.rsplit(".", 1)[0] if "." in sector_id else "",
                "weight_pct_total_manufacturing": weight, "rounding_halfwidth_pct": .05,
                "source_file": str(path.relative_to(ROOT)), "source_page": page_number,
                "known_rebase_vintage_mismatch": year == 2025,
                "interpretation": "Displayed release weight; rounded, not an exact unlinked compilation coefficient",
            })
        assert len(found) == 25, (year, len(found), found)
        assert found["1"] == 100
    return pd.DataFrame(records)


def hierarchy_and_weight_checks(weights: pd.DataFrame, catalog: dict) -> tuple[dict, pd.DataFrame]:
    hierarchy = {parent: sorted(s for s in catalog if s.startswith(parent+".") and s.count(".") == parent.count(".")+1) for parent in PARENTS}
    assert [len(hierarchy[p]) for p in CLUSTERS] == [4, 4, 2, 2, 3, 3]
    assert len(hierarchy["1"]) == 6
    assert sum(len(hierarchy[p]) for p in CLUSTERS) == 18
    checks = []
    weights["normalized_weight_within_parent"] = np.nan
    for weight_set, group in weights.groupby("weight_set", sort=False):
        indexed = group.set_index("sector_id")
        for parent, children in hierarchy.items():
            child_sum = float(indexed.loc[children, "weight_pct_total_manufacturing"].sum())
            parent_weight = float(indexed.loc[parent, "weight_pct_total_manufacturing"])
            discrepancy = child_sum-parent_weight
            rounding_bound = .05*(len(children)+(parent != "1"))
            assert abs(discrepancy) <= rounding_bound+1e-10
            mask = weights.weight_set.eq(weight_set)&weights.sector_id.isin(children)
            weights.loc[mask, "normalized_weight_within_parent"] = weights.loc[mask, "weight_pct_total_manufacturing"]/child_sum
            checks.append({"weight_set": weight_set, "effective_year": int(group.effective_year.iloc[0]),
                           "parent_id": parent, "parent": catalog[parent], "n_children": len(children),
                           "sum_displayed_child_weights_pct": child_sum, "displayed_parent_weight_pct": parent_weight,
                           "difference_pct": discrepancy, "rounding_bound_pct": rounding_bound,
                           "weight_sum_compatible_with_display_rounding": True})
    assert weights.normalized_weight_within_parent.dropna().gt(0).all()
    return hierarchy, pd.DataFrame(checks)


def possible_weights(displayed: np.ndarray, parent_displayed: float, parent: str) -> np.ndarray:
    """Vertices of rounded-weight bounds; these are bounds, never fitted weights."""
    lower = np.maximum(0, displayed-.05)
    upper = displayed+.05
    sum_lower, sum_upper = (100., 100.) if parent == "1" else (parent_displayed-.05, parent_displayed+.05)
    vertices = []
    for values in itertools.product(*zip(lower, upper)):
        v = np.asarray(values)
        if sum_lower-1e-10 <= v.sum() <= sum_upper+1e-10:
            vertices.append(v/v.sum())
    for free in range(len(displayed)):
        fixed = [i for i in range(len(displayed)) if i != free]
        for values in itertools.product(*[(lower[i], upper[i]) for i in fixed]):
            for target in [sum_lower, sum_upper]:
                v = np.zeros(len(displayed))
                v[fixed] = values
                v[free] = target-v.sum()
                if lower[free]-1e-10 <= v[free] <= upper[free]+1e-10:
                    vertices.append(v/v.sum())
    assert vertices
    return np.unique(np.round(vertices, 15), axis=0)


def parent_comparisons(levels: dict, weights: pd.DataFrame, hierarchy: dict, catalog: dict) -> pd.DataFrame:
    records = []
    current_weights = weights[weights.weight_set.eq("release_2026_09")].set_index("sector_id")
    for adjustment, frame in levels.items():
        pivot = frame.pivot(index="month", columns="sector_id", values="index").sort_index()
        for parent, children in hierarchy.items():
            common = pivot[[parent, *children]].dropna()
            displayed = current_weights.loc[children, "weight_pct_total_manufacturing"].to_numpy()
            fixed = displayed/displayed.sum()
            methods = {
                "naive_sum": common[children].sum(axis=1),
                "naive_equal_average": common[children].mean(axis=1),
                "fixed_2025base_release_weights_diagnostic": common[children].mul(fixed).sum(axis=1),
            }
            for method, reconstructed in methods.items():
                growth = 100*np.log(reconstructed).diff()
                official_growth = 100*np.log(common[parent]).diff()
                for period, value in reconstructed.items():
                    official = float(common.loc[period, parent])
                    records.append({"adjustment": adjustment.upper(), "parent_id": parent, "parent": catalog[parent],
                                    "month": str(period), "method": method, "official_parent": official,
                                    "synthetic_diagnostic": float(value), "residual_index_points": float(value-official),
                                    "relative_residual_pct": float(100*(value/official-1)),
                                    "official_mom_log_growth": official_growth.loc[period],
                                    "synthetic_mom_log_growth": growth.loc[period],
                                    "n_children": len(children), "official_reconstruction": False})
    return pd.DataFrame(records)


def annual_overlap_candidates(nsa: pd.DataFrame, weights: pd.DataFrame, hierarchy: dict, catalog: dict) -> pd.DataFrame:
    pivot = nsa.pivot(index="month", columns="sector_id", values="index").sort_index()
    annual = pivot.groupby(pivot.index.year).mean()
    annual_counts = pivot.groupby(pivot.index.year).count()
    records = []
    delta = .0005  # SingStat IIP cells are printed to three decimals.
    for weight_set, group in weights.groupby("weight_set", sort=False):
        year = int(group.effective_year.iloc[0])
        indexed = group.set_index("sector_id")
        current = pivot.loc[pivot.index.year == year]
        for parent, children in hierarchy.items():
            assert annual_counts.loc[year-1, [parent, *children]].eq(12).all()
            prior_parent = float(annual.loc[year-1, parent])
            prior_children = annual.loc[year-1, children].to_numpy(dtype=float)
            displayed = indexed.loc[children, "weight_pct_total_manufacturing"].to_numpy()
            weight = displayed/displayed.sum()
            vertices = possible_weights(displayed, float(indexed.loc[parent, "weight_pct_total_manufacturing"]), parent)
            for period, values in current[[parent, *children]].dropna().iterrows():
                child = values[children].to_numpy(dtype=float)
                relative = child/prior_children
                point = prior_parent*float(weight@relative)
                low_relative = (child-delta)/(prior_children+delta)
                high_relative = (child+delta)/(prior_children-delta)
                lower = (prior_parent-delta)*float((vertices@low_relative).min())
                upper = (prior_parent+delta)*float((vertices@high_relative).max())
                official = float(values[parent])
                compatible = lower <= official+delta and upper >= official-delta
                records.append({
                    "weight_set": weight_set, "effective_year": year, "index_base_of_weights": str(group.index_base.iloc[0]),
                    "current_index_base": "2025=100", "known_rebase_vintage_mismatch": year == 2025,
                    "parent_id": parent, "parent": catalog[parent], "month": str(period), "n_children": len(children),
                    "official_parent": official, "prior_annual_parent": prior_parent,
                    "annual_overlap_candidate": point, "residual_index_points": point-official,
                    "weight_and_index_rounding_lower": lower, "weight_and_index_rounding_upper": upper,
                    "rounding_interval_compatible": bool(compatible),
                    "outside_rounding_interval_index_points": max(lower-(official+delta), official-delta-upper, 0.),
                    "status": "Old 2019-base weights cannot reconstruct rebased 2025" if year == 2025 else
                              "Candidate compatible with displayed rounding; not an exact-replication claim" if compatible else
                              "Not reconciled by displayed-weight/index rounding; matching compilation inputs unresolved",
                })
    return pd.DataFrame(records)


def two_stage_candidates(candidates: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Use leaf-derived cluster relatives, then combine the six cluster relatives."""
    rows = []
    for weight_set, group in candidates.groupby("weight_set", sort=False):
        indexed = weights[weights.weight_set.eq(weight_set)].set_index("sector_id")
        displayed = indexed.loc[CLUSTERS, "weight_pct_total_manufacturing"].to_numpy()
        fixed = displayed/displayed.sum()
        vertices = possible_weights(displayed, 100., "1")
        for month, period in group.groupby("month", sort=False):
            by_parent = period.set_index("parent_id")
            assert set(by_parent.index) == set(PARENTS)
            parent = by_parent.loc["1"]
            clusters = by_parent.loc[CLUSTERS]
            relative = (clusters.annual_overlap_candidate/clusters.prior_annual_parent).to_numpy()
            # Independent per-cluster bounds and top-level bounds need not use
            # one common feasible vector, so these are conservative outer bounds.
            low_relative = (clusters.weight_and_index_rounding_lower/(clusters.prior_annual_parent-.0005)).to_numpy()
            high_relative = (clusters.weight_and_index_rounding_upper/(clusters.prior_annual_parent+.0005)).to_numpy()
            predicted = float(parent.prior_annual_parent*(fixed@relative))
            lower = float((parent.prior_annual_parent-.0005)*(vertices@low_relative).min())
            upper = float((parent.prior_annual_parent+.0005)*(vertices@high_relative).max())
            official = float(parent.official_parent)
            compatible = lower <= official+.0005 and upper >= official-.0005
            rows.append({"weight_set": weight_set, "effective_year": int(parent.effective_year), "month": month,
                         "known_rebase_vintage_mismatch": bool(parent.known_rebase_vintage_mismatch),
                         "official_total": official, "two_stage_18_leaf_candidate": predicted,
                         "six_published_cluster_candidate": float(parent.annual_overlap_candidate),
                         "two_stage_minus_official_index_points": predicted-official,
                         "two_stage_minus_six_published_clusters_index_points": predicted-float(parent.annual_overlap_candidate),
                         "rounding_outer_lower": lower, "rounding_outer_upper": upper,
                         "rounding_interval_compatible": bool(compatible),
                         "n_leaves": 18, "n_clusters": 6,
                         "official_reconstruction": False})
    return pd.DataFrame(rows)


def within_year_growth(nsa: pd.DataFrame, weights: pd.DataFrame, hierarchy: dict, catalog: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive moving volume shares before aggregating arithmetic growth factors.

    Log growth is the logarithm of that aggregate factor, not a weighted mean
    of child log growth.  January/Q1 cross-year changes are deliberately absent.
    """
    monthly = nsa.pivot(index="month", columns="sector_id", values="index").sort_index()
    annual = monthly.groupby(monthly.index.year).mean()
    quarter_key = monthly.index.asfreq("Q")
    quarterly = monthly.groupby(quarter_key).mean()
    quarterly = quarterly.loc[monthly.groupby(quarter_key).size().eq(3)]
    rows, shares = [], []
    for weight_set, group in weights.groupby("weight_set", sort=False):
        year = int(group.effective_year.iloc[0])
        indexed = group.set_index("sector_id")
        for parent, children in hierarchy.items():
            displayed = indexed.loc[children, "weight_pct_total_manufacturing"].to_numpy()
            weight = displayed/displayed.sum()
            prior_annual = annual.loc[year-1, children].to_numpy(dtype=float)
            for frequency, pivot in [("monthly", monthly), ("quarterly", quarterly)]:
                current = pivot.loc[pivot.index.year == year, [parent, *children]].dropna()
                for i in range(1, len(current)):
                    period, previous = current.index[i], current.index[i-1]
                    assert period.ordinal == previous.ordinal+1
                    previous_values = current.loc[previous, children].to_numpy(dtype=float)
                    values = current.loc[period, children].to_numpy(dtype=float)
                    volume = weight*previous_values/prior_annual
                    effective_shares = volume/volume.sum()
                    child_factors = values/previous_values
                    factor = float(effective_shares@child_factors)
                    alternative = float(np.sum(weight*values/prior_annual)/np.sum(weight*previous_values/prior_annual))
                    assert abs(factor-alternative) < 1e-12
                    assert abs(effective_shares.sum()-1) < 1e-12
                    official_factor = float(current.loc[period, parent]/current.loc[previous, parent])
                    rows.append({"weight_set": weight_set, "effective_year": year, "frequency": frequency,
                                 "parent_id": parent, "parent": catalog[parent], "period": str(period), "previous_period": str(previous),
                                 "candidate_growth_pct": 100*(factor-1), "official_growth_pct": 100*(official_factor-1),
                                 "candidate_log_growth": 100*math.log(factor), "official_log_growth": 100*math.log(official_factor),
                                 "log_growth_error_points": 100*math.log(factor/official_factor),
                                 "known_rebase_vintage_mismatch": year == 2025,
                                 "sum_effective_previous_period_shares": float(effective_shares.sum()),
                                 "cross_year_growth_included": False})
                    for child, share in zip(children, effective_shares):
                        shares.append({"weight_set": weight_set, "effective_year": year, "frequency": frequency,
                                       "parent_id": parent, "sector_id": child, "sector": catalog[child], "period": str(period),
                                       "previous_period": str(previous), "derived_previous_period_volume_share": float(share),
                                       "known_rebase_vintage_mismatch": year == 2025,
                                       "interpretation": "Derived from displayed effective-year weights and previous-period NSA volumes; not an official published weight"})
    return pd.DataFrame(rows), pd.DataFrame(shares)


def cma_shares(catalog: dict) -> pd.DataFrame:
    """Published census VA shares: context, never relabelled as IIP weights."""
    payload = json.loads((RAW / "iip_cma_va_tabledata.json").read_text())["Data"]
    aliases = {compact_name(name): sector_id for sector_id, name in catalog.items()}
    aliases.update({"specialtychemicals": "1.2.3", "others": "1.2.4", "generalmanufacturingindustries": "1.6", "totalmanufacturing": "1"})
    rows = []
    for row in payload["row"]:
        sector_id = aliases.get(compact_name(row["rowText"]))
        assert sector_id is not None, row["rowText"]
        rows.extend({"sector_id": sector_id, "sector": catalog[sector_id], "year": int(cell["key"]),
                     "cma_va_million_sgd": float(cell["value"])} for cell in row["columns"])
    result = pd.DataFrame(rows)
    assert len(result) == 650 and not result.duplicated(["sector_id", "year"]).any()
    total = result[result.sector_id.eq("1")].set_index("year").cma_va_million_sgd
    result["share_pct_total_manufacturing"] = 100*result.cma_va_million_sgd/result.year.map(total)
    result["is_official_iip_compilation_weight"] = False
    result["preliminary"] = result.year.eq(2025)
    result["source"] = "SingStat M355181; census VA shares, not IIP release/compilation weights"
    return result


def division_rebase_weights() -> pd.DataFrame:
    """Direct transcription of EDB's rebase memorandum p7, Table2 (per mille)."""
    table = [
        ("10-12", "Food, Beverages and Tobacco", 58.2, 39.5),
        ("13,14", "Textiles and Wearing Apparel", 1.1, 1.2),
        ("16", "Wood and Products of Wood and Cork except Furniture", 1.5, 1.2),
        ("17", "Paper and Paper Products", 1.8, 2.3),
        ("18", "Printing and Reproduction of Recorded Media", 2.9, 8.4),
        ("19", "Refined Petroleum Products", 61.8, 11.2),
        ("20", "Chemicals and Chemical Products", 86.7, 116.4),
        ("21", "Pharmaceuticals and Biological Products", 36.8, 131.3),
        ("22", "Rubber and Plastic Products", 8.0, 7.8),
        ("23", "Non-Metallic Mineral Products", 3.6, 3.1),
        ("24", "Basic Metals", 1.8, 1.8),
        ("25", "Fabricated Metal Products except Machinery and Equipment", 21.2, 23.8),
        ("26", "Computer, Electronic and Optical Products", 430.9, 428.8),
        ("27", "Electrical Equipment", 8.7, 10.3),
        ("28", "Machinery and Equipment", 127.5, 99.8),
        ("29", "Motor Vehicles, Trailers and Semi-trailers", 3.3, 4.5),
        ("30", "Other Transport Equipment", 71.0, 59.5),
        ("31", "Furniture", 2.0, 2.9),
        ("15,32", "Other Manufacturing Industries", 71.2, 46.2),
    ]
    result = pd.DataFrame(table, columns=["ssic_division", "industry", "weight_per_mille_2025_base", "weight_per_mille_2019_base"])
    for base in [2025, 2019]:
        assert abs(result[f"weight_per_mille_{base}_base"].sum()-1000) < .3
        result[f"weight_pct_{base}_base"] = result[f"weight_per_mille_{base}_base"]/10
    result["source_file"] = "data/raw/reconciliation/iip_memorandum_rebase_2025.pdf"
    result["source_page"] = 7
    result["interpretation"] = "SSIC division base-pattern weights; do not map directly to cross-cutting six IIP clusters or extrapolate across years"
    return result


def summary_metrics(comparisons: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in comparisons.groupby(["adjustment", "parent_id", "parent", "method"], sort=False):
        errors = group.residual_index_points
        growth_errors = group.synthetic_mom_log_growth-group.official_mom_log_growth
        rows.append(dict(zip(["adjustment", "parent_id", "parent", "method"], keys)) | {
            "months": len(group), "start": group.month.min(), "end": group.month.max(),
            "mean_absolute_error_index_points": float(errors.abs().mean()), "max_absolute_error_index_points": float(errors.abs().max()),
            "root_mean_squared_error_index_points": float(np.sqrt(np.mean(errors**2))),
            "mean_absolute_mom_error_log_points": float(growth_errors.abs().mean()),
        })
    overlap = []
    for keys, group in candidates.groupby(["weight_set", "effective_year", "parent_id", "parent"], sort=False):
        errors = group.residual_index_points
        overlap.append(dict(zip(["weight_set", "effective_year", "parent_id", "parent"], keys)) | {
            "months": len(group), "mean_absolute_error_index_points": float(errors.abs().mean()),
            "max_absolute_error_index_points": float(errors.abs().max()),
            "min_signed_error_index_points": float(errors.min()), "max_signed_error_index_points": float(errors.max()),
            "months_outside_conservative_rounding_interval": int((~group.rounding_interval_compatible).sum()),
            "max_distance_outside_rounding_interval": float(group.outside_rounding_interval_index_points.max()),
            "known_rebase_vintage_mismatch": bool(group.known_rebase_vintage_mismatch.iloc[0]),
        })
    return pd.DataFrame(rows), pd.DataFrame(overlap)


def run_iip_reconciliation(fetch: bool = False) -> dict:
    manifest = cache_sources(fetch)
    OUT.mkdir(parents=True, exist_ok=True)
    nsa, nsa_meta = source_levels("nsa")
    sa, sa_meta = source_levels("sa")
    catalog = nsa[["sector_id", "sector"]].drop_duplicates().set_index("sector_id").sector.to_dict()
    assert len(catalog) == 26 and set(catalog) == set(sa.sector_id)
    weights = parse_weights(catalog)
    hierarchy, weight_checks = hierarchy_and_weight_checks(weights, catalog)
    comparisons = parent_comparisons({"nsa": nsa, "sa": sa}, weights, hierarchy, catalog)
    candidates = annual_overlap_candidates(nsa, weights, hierarchy, catalog)
    two_stage = two_stage_candidates(candidates, weights)
    growth, volume_shares = within_year_growth(nsa, weights, hierarchy, catalog)
    shares = cma_shares(catalog)
    division_weights = division_rebase_weights()
    metrics, overlap_metrics = summary_metrics(comparisons, candidates)
    hierarchy_rows = [{"parent_id": p, "parent": catalog[p], "child_id": c, "child": catalog[c],
                       "child_is_leaf": c not in hierarchy} for p, children in hierarchy.items() for c in children]
    data = {"iip_weights.csv": weights, "iip_weight_checks.csv": weight_checks,
            "iip_hierarchy.csv": pd.DataFrame(hierarchy_rows), "iip_parent_comparisons.csv": comparisons,
            "iip_comparison_summary.csv": metrics, "iip_annual_overlap_candidates.csv": candidates,
            "iip_annual_overlap_summary.csv": overlap_metrics, "iip_cma_va_shares_context.csv": shares,
            "iip_division_rebase_weights.csv": division_weights,
            "iip_two_stage_annual_overlap.csv": two_stage, "iip_within_year_growth.csv": growth,
            "iip_derived_volume_shares.csv": volume_shares}
    for name, frame in data.items():
        frame.to_csv(OUT / name, index=False)
    availability = pd.DataFrame([{"year": year, "official_release_weights_recovered": year >= 2022,
                                  "rounded_release_pattern_compatible_with_tested_current_link_segment": year in [2022, 2023, 2024, 2026],
                                  "regime": "annual chain link" if year >= 2019 else "historical fixed-weight series, subsequently linked",
                                  "reason_not_usable_as_complete_weight_panel": "Old 2019-base 2025 release does not cover rebased 2025" if year == 2025 else
                                      "Rounded display weights only; exact compilation weights unavailable" if year >= 2022 else "No matching compilation weights recovered; no imputation or backward extension"}
                                 for year in range(1992, 2027)])
    availability.to_csv(OUT / "iip_weight_availability.csv", index=False)
    # An independent scalar calculation checks vectorized index arithmetic and
    # every weight-normalisation identity, without estimating any weights.
    fixed_weights = weights[weights.weight_set.eq("release_2026_09")].set_index("sector_id").weight_pct_total_manufacturing.to_dict()
    index_values = {adj: {(r.sector_id, str(r.month)): r.index for r in frame.itertuples()} for adj, frame in [("NSA", nsa), ("SA", sa)]}
    independent_max = 0.
    for row in comparisons[comparisons.method.eq("fixed_2025base_release_weights_diagnostic")].itertuples():
        children = hierarchy[row.parent_id]
        manual = math.fsum(fixed_weights[c]*index_values[row.adjustment][(c, row.month)] for c in children)/math.fsum(fixed_weights[c] for c in children)
        independent_max = max(independent_max, abs(manual-row.synthetic_diagnostic))
    assert independent_max < 1e-10
    raw_nsa = index_values["NSA"]
    annual_scalar = {(sector, year): math.fsum(raw_nsa[(sector, str(p))] for p in pd.period_range(f"{year}-01", f"{year}-12", freq="M"))/12
                     for sector in ["1", *CLUSTERS, *[c for p in CLUSTERS for c in hierarchy[p]]]
                     for year in range(2021, 2026)}
    scalar_overlap_max = 0.
    for row in candidates.itertuples():
        children = hierarchy[row.parent_id]
        displayed = weights[weights.weight_set.eq(row.weight_set)].set_index("sector_id").weight_pct_total_manufacturing.to_dict()
        manual = annual_scalar[(row.parent_id, row.effective_year-1)]*math.fsum(
            displayed[c]*raw_nsa[(c, row.month)]/annual_scalar[(c, row.effective_year-1)] for c in children)/math.fsum(displayed[c] for c in children)
        scalar_overlap_max = max(scalar_overlap_max, abs(manual-row.annual_overlap_candidate))
    assert scalar_overlap_max < 1e-10
    assert len(weights) == 125 and len(weight_checks) == 35
    assert weights.groupby(["weight_set", "parent_id"]).normalized_weight_within_parent.sum().loc[(slice(None), ["1", *CLUSTERS])].sub(1).abs().max() < 1e-12
    assert candidates.loc[candidates.effective_year.eq(2025), "known_rebase_vintage_mismatch"].all()
    assert candidates.loc[~candidates.known_rebase_vintage_mismatch, "rounding_interval_compatible"].all()
    assert two_stage.loc[~two_stage.known_rebase_vintage_mismatch, "rounding_interval_compatible"].all()
    total_overlap = overlap_metrics[overlap_metrics.parent_id.eq("1")].to_dict("records")
    two_stage_summary = [{"weight_set": key, "effective_year": int(group.effective_year.iloc[0]), "months": len(group),
                          "mean_absolute_error_index_points": float(group.two_stage_minus_official_index_points.abs().mean()),
                          "max_absolute_error_index_points": float(group.two_stage_minus_official_index_points.abs().max()),
                          "months_outside_conservative_rounding_interval": int((~group.rounding_interval_compatible).sum()),
                          "known_rebase_vintage_mismatch": bool(group.known_rebase_vintage_mismatch.iloc[0])}
                         for key, group in two_stage.groupby("weight_set", sort=False)]
    pd.DataFrame(two_stage_summary).to_csv(OUT / "iip_two_stage_summary.csv", index=False)
    quarterly_path = ROOT / "data/processed/iip/data_checks.json"
    quarterly = json.loads(quarterly_path.read_text())
    independent_quarterly_path = ROOT / "outputs/iip/verification.json"
    quarterly_audit = json.loads(independent_quarterly_path.read_text())
    assert quarterly_audit["raw_growth_and_quarterly_aggregation"] == "passed"
    assert quarterly["official_quarterly_reconciliation_cells"] == 3778
    summary = {
        "classification": "SSIC 2025", "current_index_base": "2025=100",
        "sources": {"NSA": nsa_meta, "SA": sa_meta},
        "hierarchy": {"clusters": 6, "leaves": 18, "leaves_per_cluster": {catalog[p]: len(hierarchy[p]) for p in CLUSTERS},
                      "total_excluding_biomedical_is_overlapping_aggregate_excluded_from_children": True},
        "official_method": "Laspeyres annual chain-link from 2019; old fixed-weight history is multi-period linked; 2025 rebase uses CMA2024 and preserves pre-2025 growth by linking",
        "conditional_annual_overlap_formula": "I_parent(y,m) = mean_previous_year(I_parent) * sum_c[w_c,y * I_c(y,m)/mean_previous_year(I_c)]",
        "formula_scope": "Requires matching actual unlinked aggregation weights, hierarchy and vintage; this is a conditional derivation, not a claim that displayed rounded cluster weights suffice",
        "actual_release_weight_years_recovered": [2022, 2023, 2024, 2025, 2026],
        "ssic_division_base_pattern_weights": "EDB 2026 memorandum Table 2, p7: 19 published SSIC division group rows for 2025 and 2019 bases. These are not a six-cluster correspondence table or annual historical panel.",
        "complete_historical_compilation_weight_panel_recovered": False,
        "weight_rounding": "0.1 percentage points; apparent child/parent sum differences fit display rounding; diagnostics normalize transparently",
        "known_weight_vintage_mismatch": "January 2025 weights describe the then-current 2019-base series, not the 2025-base re-estimation for 2025. A base-year basket change can also require the previous-year overlap on that new basket.",
        "annual_overlap_total_checks": total_overlap,
        "annual_overlap_all_recovered_non2025_parent_months_within_conservative_rounding_bounds": True,
        "two_stage_18_leaves_to_six_clusters_to_total_checks": two_stage_summary,
        "within_link_year_growth_formula": "s_c,m-1 = w_c,y*(I_c,m-1/mean_previous_year(I_c))/sum_j[w_j,y*(I_j,m-1/mean_previous_year(I_j))]; aggregate factor=sum_c[s_c,m-1*(I_c,m/I_c,m-1)]; log growth=100ln(factor). Same for quarter means within a year.",
        "annual_overlap_result_limit": "All recovered non2025 parent/month intervals are individually compatible with display rounding. This is a conservative necessary check, not proof of a unique common exact-weight vector across months.",
        "decision": "Use published official manufacturing total for the national weighted outcome; retain equal-sector/industry panels as separate estimands. Do not label fixed displayed weights or census VA shares an official weighted IIP reconstruction.",
        "seasonal_adjustment": "SA parent/child additivity is not imposed. DOS's still-linked 2006 paper explicitly describes direct adjustment of overall IIP; this is dated documentation, not proof that every current component uses an unchanged method.",
        "quarterly_monthly_mean_audit_reused": {"cells": quarterly["official_quarterly_reconciliation_cells"], "max_abs_index_gap": quarterly["official_quarterly_max_rounding_gap"],
                                                "independently_verified_status": quarterly_audit["raw_growth_and_quarterly_aggregation"],
                                                "data_checks_sha256": sha256(quarterly_path), "verification_sha256": sha256(independent_quarterly_path)},
        "validation": {"displayed_weight_rows": len(weights), "parent_weight_sum_checks": len(weight_checks),
                       "all_weight_sum_differences_within_display_rounding": True,
                       "independent_scalar_fixed_weight_max_abs_error": independent_max,
                       "independent_scalar_annual_overlap_max_abs_error": scalar_overlap_max,
                       "index_source_hashes_checked_against_original_manifest": True,
                       "weights_estimated_from_official_total": False},
        "output_rows": {name: len(frame) for name, frame in data.items()},
        "source_manifest_sha256": sha256(RAW / "iip_source_manifest.json"),
    }
    original_manifest = json.loads((ROOT / "data/raw/manifest.json").read_text())
    for name in ["iip_nsa_tabledata.json", "iip_sa_tabledata.json", "iip_nsa_metadata.json", "iip_sa_metadata.json"]:
        assert sha256(ROOT / "data/raw" / name) == original_manifest[name]["sha256"]
    evidence = [
        {"claim": "Current IIP scope is SSIC 2025; monthly NSA/SA total, six clusters and 18 leaves", "source": "https://tablebuilder.singstat.gov.sg/table/TS/M355381", "cached": "data/raw/iip_nsa_metadata.json"},
        {"claim": "2025 rebase uses CMA2024; unchained weighted aggregation then annual chaining", "source": SOURCES["iip_memorandum_rebase_2025.pdf"][0], "pages": [2, 3, 4, 5], "equations_visually_checked": True},
        {"claim": "Annual chain linking introduced from 2019; ordinary additivity generally lost; historical growth retained by multi-period linking", "source": SOURCES["iip_chain_link_information_2019.pdf"][0], "pages": [2, 3, 4, 5, 6]},
        {"claim": "Displayed 2026 weights for six clusters and 18 leaves", "source": SOURCES["iip_monthly_release_current.pdf"][0], "pages": [3]},
        {"claim": "Census VA shares available 2000–2025; 2025 preliminary; shares are not identified as IIP compilation coefficients", "source": "https://tablebuilder.singstat.gov.sg/table/TS/M355181", "cached": "data/raw/reconciliation/iip_cma_va_tabledata.json"},
        {"claim": "Direct adjustment documented for overall IIP, dated 2006; document remains linked from current DOS explainer", "source": SOURCES["iip_dos_seasonal_adjustment_2006.pdf"][0], "section": "Appendix II(E)", "current_landing": "https://www.singstat.gov.sg/find-data/understand-data-basics/what-is-seasonal-adjustment"},
    ]
    json_file(OUT / "iip_evidence.json", evidence)
    json_file(OUT / "iip_reconciliation_summary.json", summary)
    print(json.dumps({"clusters": 6, "leaves": 18, "release_weight_years": [2022, 2023, 2024, 2025, 2026],
                      "scalar_check_max_error": independent_max, "total_annual_overlap_checks": total_overlap,
                      "decision": summary["decision"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Refresh the explicitly listed official source caches")
    run_iip_reconciliation(parser.parse_args().fetch)
