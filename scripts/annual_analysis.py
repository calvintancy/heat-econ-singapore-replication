"""Annual NSA-primary growth, SA comparison, pooled FE and sector interactions.

All coefficient fits use the isolated native-PyFixest backend.  Annual outcomes
are computed from complete source years before weather/sample exclusions.
"""
from __future__ import annotations

import calendar
import hashlib
import itertools
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.stats import f as f_dist, t as student_t
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum
from statsmodels.tsa.stattools import adfuller, kpss

import pyfixest_backend as backend
from sector_cluster import sector_cluster_covariance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/annual"
DATA = ROOT / "data/processed/annual"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
ADDED = ["wind_mps", "pm25_10ug"]
ADJUSTMENTS = {"NSA": "y_nsa", "SA": "y_sa"}
CONTEXTS = ["aggregate_va", "va11", "va19", "iip_total", "iip6", "iip18"]
VARIANTS = ["all_eligible", "exclude_2020_2022"]
MIN_DIAGNOSTIC_YEARS = 20


def reconcile_annual_sources() -> dict:
    """Compare derived NSA years with cached official annual publications.

    This adds evidence, never replaces the quarterly/monthly model inputs.
    Its own source manifest leaves the original global manifest unchanged.
    """
    raw = ROOT / "data/raw/annual"
    raw.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Singapore official annual aggregation audit)"
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=[429,500,502,503,504])))
    def save(name, body, details):
        (raw/name).write_bytes(body)
        manifest[name] = {**details, "sha256": _sha(raw/name), "bytes": len(body),
                          "retrieved_utc": datetime.now(timezone.utc).isoformat()}
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    def get(name, url, params=None):
        if (raw/name).exists():
            assert name in manifest and _sha(raw/name) == manifest[name]["sha256"]
            return json.loads((raw/name).read_text())
        response = session.get(url, params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()
        assert payload.get("StatusCode") == 200 and payload.get("Data")
        save(name,response.content,{"url":response.url})
        return payload
    derived = pd.read_csv(DATA / "annual_levels.csv", dtype={"sector_id":str})
    derived = derived[derived.adjustment.eq("NSA")]
    comparisons, sources = [], {}
    for family, table, bound in [("va","M015721",.25),("iip","M355401",.001)]:
        base = "https://tablebuilder.singstat.gov.sg/api/table"
        metadata = get(f"{family}_annual_metadata.json",f"{base}/metadata/{table}")["Data"]["records"]
        name = f"{family}_annual_tabledata.json"
        if not (raw/name).exists():
            offset, merged, pages, template = 0, {}, [], None
            while offset < metadata["total"]:
                page_name = f"{family}_annual_tabledata_page_{offset:05d}.json"
                page = get(page_name,f"{base}/tabledata/{table}",{"limit":5000,"offset":offset})
                if template is None:template=page
                count = 0
                for row in page["Data"]["row"]:
                    count += len(row["columns"])
                    sid = row["seriesNo"]
                    if sid not in merged:merged[sid]={**row,"columns":[]}
                    merged[sid]["columns"].extend(row["columns"])
                assert count > 0
                pages.append(page_name);offset += count
            assert offset == metadata["total"]
            template["Data"]["row"] = list(merged.values())
            save(name,(json.dumps(template)+"\n").encode(),{"url":f"{base}/tabledata/{table}","assembled_from":pages})
        payload = json.loads((raw/name).read_text())["Data"]
        assert _sha(raw/name) == manifest[name]["sha256"]
        assert payload["id"] == table and payload["dataLastUpdated"] == metadata["dataLastUpdated"]
        assert sum(len(row["columns"]) for row in payload["row"]) == metadata["total"]
        available = derived[derived.family.eq(family)].set_index(["sector_id","year"])
        seen = set()
        for row in payload["row"]:
            for cell in row["columns"]:
                key=(row["seriesNo"],int(cell["key"]))
                if key not in available.index:continue
                assert key not in seen;seen.add(key)
                value=available.loc[key]
                assert value.sector == row["rowText"]
                published=float(cell["value"].replace(",",""))
                gap=float(value.annual_level-published)
                comparisons.append({"family":family,"adjustment":"NSA","sector_id":key[0],"sector":row["rowText"],"year":key[1],
                    "derived_annual_level":float(value.annual_level),"published_annual_level":published,"gap":gap,
                    "rounding_bound":bound,"within_rounding":abs(gap)<=bound+1e-9,"annual_table_id":table,
                    "annual_source_updated":metadata["dataLastUpdated"],
                    "higher_frequency_source_updated":"11/08/2026" if family=="va" else "26/10/2026",
                    "units":"chained-2015 S$ million" if family=="va" else "IIP2025=100 index points"})
        assert len(seen)==len(available),f"Official annual reconciliation coverage incomplete: {family}"
        group=[item for item in comparisons if item["family"]==family]
        sources[family]={"table_id":table,"title":metadata["title"],"annual_updated":metadata["dataLastUpdated"],
                         "landing_url":f"https://tablebuilder.singstat.gov.sg/table/TS/{table}",
                         "source_cells":metadata["total"],"matched_sector_years":len(group),
                         "max_absolute_gap":max(abs(item["gap"]) for item in group),
                         "rounding_bound":bound,"all_within_rounding":all(item["within_rounding"] for item in group)}
    pd.DataFrame(comparisons).to_csv(OUT/"reconciliation.csv",index=False)
    summary={"status":"compared","sources":sources,"all_within_rounding":all(item["all_within_rounding"] for item in sources.values()),
             "source_manifest":"data/raw/annual/manifest.json",
             "scope":"NSA aggregation evidence only; annual SA sums/means remain explicitly labelled comparison outcomes",
             "vintage_caution":"Annual publications have different update dates from cached quarterly/monthly sources; gaps are reported without force-fitting or substituting inputs"}
    (OUT/"reconciliation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _identifier(*values) -> str:
    return "a" + hashlib.sha256(json.dumps(values).encode()).hexdigest()[:16]


def _source_levels() -> tuple[pd.DataFrame, dict, dict]:
    rows, provenance, catalog = [], {}, {}
    old_manifest = json.loads((ROOT / "data/raw/manifest.json").read_text())
    for family in ["va", "iip"]:
        catalog[family] = {}
        for adjustment in ADJUSTMENTS:
            stem = f"{family}_{adjustment.lower()}"
            path = ROOT / f"data/raw/{stem}_tabledata.json"
            meta_path = ROOT / f"data/raw/{stem}_metadata.json"
            for source in [path, meta_path]:
                assert _sha(source) == old_manifest[source.name]["sha256"], source
                provenance[_rel(source)] = {"sha256": _sha(source), "source": old_manifest[source.name]}
            data = json.loads(path.read_text())["Data"]
            meta = json.loads(meta_path.read_text())["Data"]["records"]
            assert data["id"] == meta["id"] and data["dataLastUpdated"] == meta["dataLastUpdated"]
            assert sum(len(row["columns"]) for row in data["row"]) == meta["total"]
            for row in data["row"]:
                sid = row["seriesNo"]
                catalog[family][sid] = row["rowText"]
                periods = {}
                for cell in row["columns"]:
                    if family == "va":
                        year, q = cell["key"].split()
                        period = pd.Period(f"{year}Q{q[0]}", freq="Q")
                    else:
                        period = pd.Period(pd.to_datetime(cell["key"], format="%Y %b"), freq="M")
                    value = float(cell["value"].replace(",", ""))
                    assert period not in periods and value > 0
                    periods[period] = value
                observed = sorted(periods)
                assert observed == list(pd.period_range(observed[0], observed[-1], freq="Q" if family == "va" else "M"))
                annual = {}
                required = 4 if family == "va" else 12
                for year in sorted({p.year for p in periods}):
                    year_values = [value for p, value in periods.items() if p.year == year]
                    if len(year_values) == required:
                        annual[year] = math.fsum(year_values) / (1 if family == "va" else 12)
                for year, value in sorted(annual.items()):
                    rows.append({"family": family, "sector_id": sid, "sector": row["rowText"],
                                 "adjustment": adjustment, "year": year, "annual_level": value,
                                 "annual_log_growth": 100*math.log(value/annual[year-1]) if year-1 in annual else np.nan,
                                 "source_periods": required, "source_table_id": meta["id"],
                                 "aggregation": "sum of four quarterly levels" if family == "va" else "mean of twelve monthly levels"})
    result = pd.DataFrame(rows)
    assert not result.duplicated(["family", "sector_id", "adjustment", "year"]).any()
    return result, catalog, provenance


def prepare_annual_data() -> tuple[dict[str, pd.DataFrame], dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    annual, catalog, provenance = _source_levels()
    annual.to_csv(DATA / "annual_levels.csv", index=False)
    monthly = {}
    for stem, column, target in [("temperature", "mean_temp", "temp_c"),
                                 ("humidity", "mean_rh", "rh_pct"),
                                 ("rainfall", "total_rainfall", "rain_mm")]:
        path = ROOT / f"data/raw/{stem}_monthly.csv"
        source = pd.read_csv(path)
        provenance[_rel(path)] = {"sha256": _sha(path)}
        for row in source.to_dict("records"):
            period = pd.Period(row["month"], freq="M")
            entry = monthly.setdefault(period, {})
            assert target not in entry
            entry[target] = float(row[column])
    weather_rows = []
    for year in sorted({p.year for p in monthly}):
        months = list(pd.period_range(f"{year}-01", f"{year}-12", freq="M"))
        complete = all(m in monthly and len(monthly[m]) == 3 and np.isfinite(list(monthly[m].values())).all() for m in months)
        days = 366 if calendar.isleap(year) else 365
        row = {"year": year, "weather_complete": complete,
               "weather_months": sum(m in monthly and len(monthly[m]) == 3 for m in months), "days": days}
        row.update({name: math.fsum(monthly[m][name]*m.days_in_month for m in months)/days if complete else np.nan for name in ["temp_c", "rh_pct"]})
        row["rain_mm"] = math.fsum(monthly[m]["rain_mm"] for m in months) if complete else np.nan
        row["rain_100mm"] = row["rain_mm"]/100
        weather_rows.append(row)
    weather = pd.DataFrame(weather_rows)
    weather.to_csv(DATA / "annual_weather.csv", index=False)
    control_path = ROOT / "data/processed/extended_controls/monthly_controls.csv"
    control_monthly = pd.read_csv(control_path)
    provenance[_rel(control_path)] = {"sha256": _sha(control_path)}
    control_monthly["period"] = pd.PeriodIndex(control_monthly.month, freq="M")
    assert not control_monthly.period.duplicated().any()
    controls_rows = []
    for year in sorted(control_monthly.period.dt.year.unique()):
        group = control_monthly[control_monthly.period.dt.year.eq(year)]
        days = 366 if calendar.isleap(int(year)) else 365
        wind_count = float(group.wind_bins.fillna(0).sum())
        pm_count = float(group.pm_hours.fillna(0).sum())
        wind_months = int((group.wind_mps.notna() & group.wind_bins.fillna(0).gt(0)).sum())
        pm_months = int((group.pm25_10ug.notna() & group.pm_hours.fillna(0).gt(0)).sum())
        wind_coverage, pm_coverage = wind_count/(days*48), pm_count/(days*24)
        controls_rows.append({
            "year": int(year), "control_months": len(group), "wind_months": wind_months, "pm_months": pm_months,
            "wind_bins": wind_count, "pm_hours": pm_count, "wind_bins_expected": days*48, "pm_hours_expected": days*24,
            "wind_coverage": wind_coverage, "pm_coverage": pm_coverage,
            "wind_mps": float((group.wind_mps*group.wind_bins).sum()/wind_count) if wind_count else np.nan,
            "pm25_10ug": float((group.pm25_10ug*group.pm_hours).sum()/pm_count) if pm_count else np.nan,
            "minimum_monthly_wind_coverage": float(group.wind_coverage.min()),
            "minimum_monthly_pm_coverage": float(group.pm_coverage.min()),
            "all_months_90": bool(len(group) == 12 and group.eligible_90.all()),
            "eligible_90": bool(len(group) == 12 and wind_months == 12 and pm_months == 12
                                and wind_coverage >= .90 and pm_coverage >= .90),
        })
    controls = pd.DataFrame(controls_rows)
    controls.to_csv(DATA / "annual_controls.csv", index=False)
    extended_manifest = json.loads((ROOT / "data/raw/extended_controls/manifest.json").read_text())
    for name, info in extended_manifest.items():
        path = ROOT / "data/raw/extended_controls" / name
        assert _sha(path) == info["sha256"]
        provenance[_rel(path)] = {"sha256": _sha(path), "source": info}
    broad_va = [f"1.1.{i}" for i in range(1, 5)] + [f"1.2.{i}" for i in range(1, 8)]
    broad_iip = [f"1.{i}" for i in range(1, 7)]
    def leaves(family, parents):
        ids = catalog[family]
        return sorted(s for s in ids if any(s == p or s.startswith(p+".") for p in parents)
                      and not any(other.startswith(s+".") for other in ids))
    va19, iip18 = leaves("va", broad_va), leaves("iip", broad_iip)
    assert len(va19) == 19 and len(iip18) == 18
    definitions = {"aggregate_va": ("va", ["1.4"]), "va11": ("va", broad_va), "va19": ("va", va19),
                   "iip_total": ("iip", ["1"]), "iip6": ("iip", broad_iip), "iip18": ("iip", iip18)}
    panels, coverage = {}, {}
    for context, (family, ids) in definitions.items():
        source = annual[annual.family.eq(family) & annual.sector_id.isin(ids)]
        frame = source.pivot(index=["sector_id", "sector", "year"], columns="adjustment", values=["annual_level", "annual_log_growth"])
        frame.columns = [("annual_" if measure == "annual_level" else "y_")+adjustment.lower()
                         for measure, adjustment in frame.columns]
        frame = frame.reset_index().dropna(subset=["y_nsa", "y_sa"])
        common_years = set.intersection(*[set(frame.loc[frame.sector_id.eq(s), "year"]) for s in ids])
        frame = frame[frame.year.isin(common_years)].merge(weather[weather.weather_complete], on="year", validate="many_to_one")
        frame = frame.merge(controls, on="year", how="left", validate="many_to_one")
        frame["eligible_90"] = frame.eligible_90.astype("boolean").fillna(False).astype(bool)
        frame["time"] = frame.year-1982
        frame["trend"] = frame.year-2000
        frame = frame.sort_values(["sector_id", "year"]).reset_index(drop=True)
        assert frame.groupby("year").sector_id.nunique().eq(len(ids)).all()
        frame.to_csv(DATA / f"panel_{context}.csv", index=False)
        panels[context] = frame
        coverage[context] = {"family": family, "sector_ids": ids, "n_sectors": len(ids),
                             "start": int(frame.year.min()), "end": int(frame.year.max()),
                             "n_years": int(frame.year.nunique()), "observations": len(frame)}
    preparation = {"source_hashes": provenance, "coverage": coverage,
                   "annual_full_control_years": controls.loc[controls.eligible_90, "year"].tolist(),
                   "annual_control_rule": "All 12 calendar months observed with positive counts for both variables; annual wind-bin and PM-hour coverage each >=90%",
                   "annual_weather_rule": "Complete 12 months; day-weighted temperature/humidity; summed rainfall",
                   "annual_added_means": "Retained-bin/hour-weighted monthly means; no imputation",
                   "annual_output_rule": "VA sums four quarters; IIP averages twelve months; no sums across sectors; growth computed by raw prior-calendar-year lookup before filters"}
    (DATA / "preparation.json").write_text(json.dumps(preparation, indent=2)+"\n")
    return panels, preparation


def _rank(x: np.ndarray) -> int:
    scales = np.linalg.norm(x, axis=0)
    return int(np.linalg.matrix_rank(x[:, scales > 0]/scales[scales > 0]))


def _inference(beta: float, variance: float, dof: int) -> dict:
    se = float(np.sqrt(max(variance, 0)))
    critical = student_t.ppf(.975, dof)
    return {"coefficient": float(beta), "std_error": se, "ci_lower": float(beta-critical*se),
            "ci_upper": float(beta+critical*se), "p_value": float(2*student_t.sf(abs(beta/se), dof)) if se else np.nan}


def _covariances(model, sample, kind):
    periods = sample.year.nunique()
    time = (sample.time-sample.time.min()).to_numpy(dtype=int)
    # Form original-coordinate influences before their score products.  The
    # algebraically equivalent raw bread@meat@bread loses cross-covariance
    # precision in the very short annual designs with highly correlated Xs.
    u = np.asarray(model.native_model.within_data.design)
    influence = (u@model.coefficient_transform.T)*model.wresid[:,None]
    n, k = model.model.exog.shape
    if kind == "common" and sample.sector_id.nunique() > 1:
        groups, units = pd.factorize(sample.sector_id,sort=False)
        scores = np.zeros((len(units),k));np.add.at(scores,groups,influence)
        covariance = scores.T@scores*len(units)/(len(units)-1)*(n-1)/(n-k)
        yield "Sector cluster", 0, (covariance+covariance.T)/2, len(units)-1, True
    scores = np.zeros((time.max()+1,k));np.add.at(scores,time,influence)
    for lag in [1, 2]:
        covariance = scores.T@scores
        for distance in range(1,lag+1):
            cross=scores[distance:].T@scores[:-distance]
            covariance += (1-distance/(lag+1))*(cross+cross.T)
        covariance *= periods/(periods-1)*(n-1)/(n-k)
        main = lag == 1 and (kind != "common" or sample.sector_id.nunique() == 1)
        yield "Driscoll-Kraay", lag, (covariance+covariance.T)/2, periods-1, main


def _fit(sample, context, controls, variant, adjustment, kind, sector_id=None):
    outcome = ADJUSTMENTS[adjustment]
    weather = BASE + (ADDED if controls == "full" else [])
    rhs = "temp_c * C(sector_id)" if kind == "H1" else "temp_c"
    formula = f"{outcome} ~ {rhs} + " + " + ".join(weather[1:]+["trend", "C(sector_id)"])
    spec_id = _identifier(context, controls, variant, adjustment, kind, sector_id)
    y, design = patsy.dmatrices(formula, sample, return_type="dataframe", NA_action="raise")
    x = design.to_numpy()
    rank = _rank(x)
    unique = sample.drop_duplicates("year").sort_values("year")
    time_x = np.column_stack([np.ones(len(unique)), unique[weather+["trend"]]])
    time_rank = _rank(time_x)
    reason = ""
    if rank < x.shape[1]:
        reason = f"Underidentified: full design rank {rank}/{x.shape[1]}; {len(unique)} years for {time_x.shape[1]} nominal time parameters"
    elif len(sample) <= x.shape[1] or len(unique) <= time_rank:
        reason = "No positive residual annual time degrees of freedom"
    status = "unavailable" if reason else "estimated"
    family = "sector_FE_interactions" if kind == "H1" else "separate_time_series" if kind == "separate" else "pooled_FE" if sample.sector_id.nunique() > 1 else "national"
    sample_path = OUT / "samples" / f"{spec_id}.csv"
    sample.to_csv(sample_path, index=False)
    metadata = {"spec_id": spec_id, "context": context, "control_set": controls, "sample_variant": variant,
                "adjustment": adjustment, "analysis_role": "primary_NSA" if adjustment == "NSA" else "SA_comparison",
                "model": kind, "family": family, "sector_id": sector_id or "pooled",
                "outcome": outcome, "formula": formula, "status": status, "reason": reason,
                "n_obs": len(sample), "n_sectors": int(sample.sector_id.nunique()), "n_periods": len(unique),
                "start": int(sample.year.min()), "end": int(sample.year.max()), "n_parameters": x.shape[1],
                "design_rank": rank, "n_time_parameters": time_x.shape[1], "time_design_rank": time_rank,
                "time_residual_df": len(unique)-time_rank, "nominal_time_residual_df": len(unique)-time_x.shape[1],
                "residual_df": len(sample)-rank, "calendar_span_years": int(sample.year.max()-sample.year.min()+1),
                "calendar_gaps": int(sample.year.max()-sample.year.min()+1-len(unique)),
                "mean_y": float(sample[outcome].mean()), "sd_y": float(sample[outcome].std()),
                "fragile_short_sample": bool(len(unique)-time_rank <= 3), "sample_csv": _rel(sample_path),
                "controls": json.dumps(weather[1:]), "weather_terms": json.dumps(weather),
                "estimator": "PyFixest", "estimator_version": backend.ESTIMATOR_VERSION,
                "year_fixed_effects": False, "linear_time_trend": True}
    if reason:
        return None, metadata, list(design.columns)
    model = backend.fit(sample, formula, f"Annual {context}/{controls}/{variant}/{adjustment}/{kind}/{sector_id or 'joint'}")
    assert len(model.params) == x.shape[1]
    return model, metadata, list(design.columns)


def _wald(beta, covariance, periods):
    restrictions = np.eye(len(beta))[1:]-np.eye(len(beta))[[0]]
    difference = restrictions@beta
    v = restrictions@covariance@restrictions.T
    singular = np.linalg.svd(v, compute_uv=False)
    rank = int((singular > singular.max()*1e-10).sum()) if singular.max() > 0 else 0
    rank = min(rank, periods-2)
    available = rank == len(difference)
    statistic = float(difference@np.linalg.solve(v, difference)) if available else np.nan
    return {"test": "All sector temperature slopes equal", "restrictions": len(difference),
            "covariance_rank": rank, "structural_rank_bound": periods-2,
            "test_status": "available" if available else "unavailable: singular restriction covariance",
            "wald_chi2": statistic, "f_statistic": statistic/len(difference) if available else np.nan,
            "p_value": float(f_dist.sf(statistic/len(difference), len(difference), periods-1)) if available else np.nan,
            "numerator_df": len(difference), "denominator_df": periods-1,
            "covariance_condition": float(np.linalg.cond(v))}


def _diagnostics(sample, context, control_set, variant):
    records, summaries = [], []
    candidates = []
    for sector, frame in sample.groupby("sector_id", sort=True):
        for adjustment, column in ADJUSTMENTS.items():
            candidates.append((sector, adjustment, column, frame.sort_values("year"), "c"))
    unique = sample.drop_duplicates("year").sort_values("year")
    for variable in BASE+(ADDED if control_set == "full" else []):
        candidates.append(("common", "common", variable, unique, "c" if variable == "rain_100mm" else "ct"))
    for sector, adjustment, variable, frame, deterministic in candidates:
        years = frame.year.to_numpy()
        values = frame[variable].to_numpy()
        reason = "Gapped exact annual sample: dates are not compressed" if np.any(np.diff(years) != 1) else ""
        if len(years) < MIN_DIAGNOSTIC_YEARS:
            reason = (reason+"; " if reason else "")+f"Fewer than {MIN_DIAGNOSTIC_YEARS} annual observations"
        if np.ptp(values) == 0:
            reason = (reason+"; " if reason else "")+"Constant series"
        info = {"context": context, "control_set": control_set, "sample_variant": variant, "sector_id": sector,
                "adjustment": adjustment, "variable": variable, "deterministic": deterministic,
                "start": int(years.min()), "end": int(years.max()), "n_periods": len(years), "reason": reason}
        pair = {}
        for test in ["ADF", "KPSS"]:
            row = {**info, "test": test, "status": "not_run" if reason else "ok", "statistic": np.nan,
                   "p_value": np.nan, "p_value_relation": "=", "used_lags": np.nan, "warnings": ""}
            if not reason:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        if test == "ADF":
                            stat, p, lag, nobs, critical, aic = adfuller(values, maxlag=2, regression=deterministic, autolag="AIC")
                            row.update(regression_nobs=int(nobs), requested_lag_cap=2, lag_cap_binding=bool(lag == 2), aic=float(aic))
                        else:
                            stat, p, lag, critical = kpss(values, regression=deterministic, nlags="auto")
                            row.update(regression_nobs=len(values), requested_lag_cap="uncapped automatic")
                            row["p_value_relation"] = "<" if stat > critical["1%"] else ">" if stat < critical["10%"] else "="
                        row.update(statistic=float(stat), p_value=float(p), used_lags=int(lag),
                                   critical_1pct=float(critical["1%"]), critical_5pct=float(critical["5%"]), critical_10pct=float(critical["10%"]),
                                   warnings=" | ".join(str(w.message) for w in caught))
                except (ValueError, OverflowError, ZeroDivisionError) as exc:
                    row.update(status="error", reason=str(exc))
            pair[test] = row
            records.append(row)
        evidence = "Unavailable"
        if all(pair[test]["status"] == "ok" for test in pair):
            a, k = pair["ADF"]["p_value"] < .05, pair["KPSS"]["p_value"] < .05
            evidence = ("Compatible with trend stationarity" if deterministic == "ct" else "Compatible with level stationarity") if a and not k else "Unit-root concern" if not a and k else "Mixed evidence" if a else "Inconclusive"
        summaries.append({**info, "evidence_5pct": evidence,
                          "adf_status": pair["ADF"]["status"], "adf_p_value": pair["ADF"]["p_value"],
                          "kpss_status": pair["KPSS"]["status"], "kpss_p_value": pair["KPSS"]["p_value"],
                          "kpss_p_value_relation": pair["KPSS"]["p_value_relation"]})
    return records, summaries


def run_annual_analysis() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    (OUT / "diagnostics").mkdir(exist_ok=True)
    protected = {path: _sha(path) for path in [ROOT / "outputs/pyfixest/fit_log.csv", ROOT / "outputs/pyfixest/estimator_manifest.json"] if path.exists()}
    panels, preparation = prepare_annual_data()
    reconciliation = reconcile_annual_sources()
    books = {key: [] for key in ["estimates", "sector_estimates", "separate_estimates", "coefficients", "coefficient_covariances",
                                "sector_covariances", "interaction_components", "joint_tests", "pairwise_tests", "specs", "year_fe_collinearity"]}
    tests, diagnostic_summary = [], []
    old_state = (backend.LOG_DIR, list(backend._FIT_LOG), backend._API_COUNTER, backend._RUN_STARTED)
    try:
        backend.LOG_DIR = OUT / "pyfixest"
        backend.reset_fit_log()
        for context in CONTEXTS:
            panel = panels[context]
            for control_set in ["weather", "full"]:
                eligible = panel if control_set == "weather" else panel[panel.eligible_90].copy()
                for variant in VARIANTS:
                    sample = eligible if variant == "all_eligible" else eligible[~eligible.year.between(2020, 2022)].copy()
                    sample = sample.sort_values(["sector_id", "year"]).reset_index(drop=True)
                    assert not sample.empty
                    periods, sectors = sample.year.nunique(), sample.sector_id.nunique()
                    assert len(sample) == periods*sectors
                    weather = BASE+(ADDED if control_set == "full" else [])
                    rows, summary = _diagnostics(sample, context, control_set, variant)
                    tests += rows
                    diagnostic_summary += summary
                    unique = sample.drop_duplicates("year").sort_values("year")
                    year_fe = np.eye(periods)
                    augmented_rank = _rank(np.column_stack([year_fe, unique.temp_c]))
                    books["year_fe_collinearity"].append({"context": context, "control_set": control_set, "sample_variant": variant,
                        "n_years": periods, "year_FE_rank": periods, "year_FE_plus_temperature_rank": augmented_rank,
                        "temperature_residual_max": float(np.max(np.abs(unique.temp_c.to_numpy()-year_fe@unique.temp_c.to_numpy())))})
                    for adjustment in ADJUSTMENTS:
                        common_coefficient = np.nan
                        for kind in ["common"]+(["H1"] if sectors > 1 else []):
                            model, meta, coefficient_names = _fit(sample, context, control_set, variant, adjustment, kind)
                            books["specs"].append(meta)
                            for name in coefficient_names:
                                books["coefficients"].append({**meta, "term": name, "coefficient": float(model.params[name]) if model is not None else np.nan})
                            target = "estimates" if kind == "common" else "sector_estimates"
                            if model is None:
                                if kind == "common":
                                    for term in weather+["trend"]:
                                        books[target].append({**meta, "term": term, "main_inference": True, "coefficient": np.nan,
                                                              "std_error": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan})
                                else:
                                    for sector, frame in sample.groupby("sector_id"):
                                        books[target].append({**meta, "sector_id": sector, "sector": frame.sector.iloc[0], "term": "total_temperature_effect",
                                                              "coefficient": np.nan, "std_error": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan,
                                                              "mean_y": float(frame[ADJUSTMENTS[adjustment]].mean()), "sd_y": float(frame[ADJUSTMENTS[adjustment]].std())})
                                    books["joint_tests"].append({**meta, "test": "All sector temperature slopes equal", "test_status": "unavailable: model underidentified",
                                                                "restrictions": sectors-1, "p_value": np.nan})
                                continue
                            if kind == "common":
                                common_coefficient = float(model.params["temp_c"])
                            sector_ids = sorted(sample.sector_id.unique())
                            weights = np.zeros((sectors, len(model.params)))
                            if kind == "H1":
                                main_index = coefficient_names.index("temp_c")
                                weights[:, main_index] = 1
                                component_names = ["temp_c"]
                                for i, sector in enumerate(sector_ids[1:], 1):
                                    candidates = [f"temp_c:C(sector_id)[T.{sector}]", f"C(sector_id)[T.{sector}]:temp_c"]
                                    term, = [name for name in candidates if name in coefficient_names]
                                    weights[i, coefficient_names.index(term)] = 1
                                    component_names.append(term)
                                beta = weights@model.params.to_numpy()
                                assert abs(float(beta.mean())-common_coefficient) < 1e-7
                            for covariance, lag, matrix, dof, main in _covariances(model, sample, kind):
                                info = {**meta, "covariance": covariance, "hac_lags": lag, "inference_df": int(dof), "main_inference": main,
                                        "n_clusters": sectors if covariance == "Sector cluster" else np.nan,
                                        "r_squared": float(model.rsquared)}
                                for i, first in enumerate(coefficient_names):
                                    for j, second in enumerate(coefficient_names):
                                        books["coefficient_covariances"].append({"spec_id": meta["spec_id"], "covariance": covariance, "hac_lags": lag,
                                            "term_i": first, "term_j": second, "covariance_value": float(matrix[i,j])})
                                if kind == "common":
                                    for term in weather+["trend"]:
                                        j = coefficient_names.index(term)
                                        books[target].append({**info, "term": term, **_inference(model.params[term], matrix[j,j], dof)})
                                else:
                                    total_covariance = weights@matrix@weights.T
                                    for i, sector in enumerate(sector_ids):
                                        frame = sample[sample.sector_id.eq(sector)]
                                        books[target].append({**info, "sector_id": sector, "sector": frame.sector.iloc[0], "term": "total_temperature_effect",
                                            "reference_sector_id": sector_ids[0], "temperature_main": float(model.params["temp_c"]),
                                            "temperature_interaction": float(beta[i]-model.params["temp_c"]),
                                            "mean_sector_slope": float(beta.mean()), "common_slope_benchmark": common_coefficient,
                                            "mean_y": float(frame[ADJUSTMENTS[adjustment]].mean()), "sd_y": float(frame[ADJUSTMENTS[adjustment]].std()),
                                            **_inference(beta[i], total_covariance[i,i], dof)})
                                        for j, other in enumerate(sector_ids):
                                            books["sector_covariances"].append({"spec_id": meta["spec_id"], "hac_lags": lag, "sector_i": sector,
                                                "sector_j": other, "covariance_value": float(total_covariance[i,j])})
                                    for component in component_names:
                                        j = coefficient_names.index(component)
                                        books["interaction_components"].append({**info, "term": component, **_inference(model.params[component], matrix[j,j], dof)})
                                    books["joint_tests"].append({**info, **_wald(beta, total_covariance, periods)})
                                    pairs = []
                                    for i,j in itertools.combinations(range(sectors), 2):
                                        w = np.zeros(sectors);w[i]=1;w[j]=-1
                                        pairs.append({**info, "first_sector_id": sector_ids[i], "second_sector_id": sector_ids[j],
                                                      **_inference(beta[i]-beta[j], w@total_covariance@w, dof)})
                                    valid = [i for i,row in enumerate(pairs) if np.isfinite(row["p_value"])]
                                    adjusted = multipletests([pairs[i]["p_value"] for i in valid], method="holm")[1]
                                    for i,p in zip(valid,adjusted):pairs[i]["p_value_holm"]=float(p)
                                    books["pairwise_tests"] += pairs
                        if sectors > 1:
                            for sector, frame in sample.groupby("sector_id", sort=True):
                                model, meta, coefficient_names = _fit(frame.reset_index(drop=True), context, control_set, variant, adjustment, "separate", sector)
                                books["specs"].append(meta)
                                for name in coefficient_names:
                                    books["coefficients"].append({**meta, "term": name, "coefficient": float(model.params[name]) if model is not None else np.nan})
                                if model is None:
                                    for term in weather+["trend"]:
                                        books["separate_estimates"].append({**meta, "term": term, "coefficient": np.nan, "std_error": np.nan,
                                                                            "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan})
                                else:
                                    for covariance, lag, matrix, dof, main in _covariances(model, frame, "separate"):
                                        for term in weather+["trend"]:
                                            j = coefficient_names.index(term)
                                            books["separate_estimates"].append({**meta, "term": term, "covariance": covariance, "hac_lags": lag,
                                                "inference_df": dof, "main_inference": main, "r_squared": model.rsquared,
                                                **_inference(model.params[term],matrix[j,j],dof)})
                                        for i, first in enumerate(coefficient_names):
                                            for j, second in enumerate(coefficient_names):
                                                books["coefficient_covariances"].append({"spec_id": meta["spec_id"], "covariance": covariance, "hac_lags": lag,
                                                    "term_i": first, "term_j": second, "covariance_value": float(matrix[i,j])})
                    print(f"Annual {context}/{control_set}/{variant}: {sectors} units × {periods} years", flush=True)
        native = backend.write_fit_log()
    finally:
        backend.LOG_DIR = old_state[0]
        backend._FIT_LOG[:] = old_state[1]
        backend._API_COUNTER, backend._RUN_STARTED = old_state[2:]
    for name, rows in books.items():
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "specs.json").write_text(json.dumps(books["specs"], indent=2)+"\n")
    pd.DataFrame(tests).to_csv(OUT / "diagnostics/tests.csv", index=False)
    pd.DataFrame(diagnostic_summary).to_csv(OUT / "diagnostics/summary.csv", index=False)
    for path, digest in protected.items():
        assert _sha(path) == digest, f"Original receipt changed while running annual stage: {path}"
    specs = pd.DataFrame(books["specs"])
    inventory = {"status": "produced", "contexts": CONTEXTS, "preparation": preparation,
                 "annual_publication_reconciliation": reconciliation,
                 "requested_specifications": len(specs), "estimated_specifications": int(specs.status.eq("estimated").sum()),
                 "unavailable_specifications": int(specs.status.eq("unavailable").sum()), "native_fits": native["native_fits_recorded"],
                 "diagnostic_test_rows": len(tests), "diagnostic_minimum_annual_observations": MIN_DIAGNOSTIC_YEARS,
                 "diagnostic_rule": "ADF AIC cap2 and uncapped automatic KPSS; growth constant, temperature/RH/wind/PM constant+trend, rain constant; gapped exact samples not compressed",
                 "annual_identification": "Sector FE and common linear time trend; no year FE because annual national weather would be absorbed",
                 "inference": "Common pooled slopes sector CR1 primary; singleton and sector-specific coefficients calendar HAC1 primary/HAC2 sensitivity; t(S-1) or approximate t(T-1)",
                 "full_control_limitation": "All nine annual years fit with two residual time DF if full rank; six-year exclusion is underidentified without dropping requested controls",
                 "original_global_receipts_unchanged": True, "verification_scope": "Production/preparation checks only; independent raw-source and QR/covariance verifier is separate",
                 "output_row_counts": {name:len(rows) for name,rows in books.items()}}
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    print(f"Annual analysis produced {native['native_fits_recorded']} native fits; {inventory['unavailable_specifications']} explicitly unavailable requests", flush=True)
    return inventory


if __name__ == "__main__":
    run_annual_analysis()
