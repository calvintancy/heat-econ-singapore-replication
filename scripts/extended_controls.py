"""Add verified wind and hourly PM2.5 controls, with matched-sample comparisons."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html as lxml_html

from estimate import FE, WEATHER, estimates, fit, read_panel
from fetch_data import session
from iip_analysis import read_iip
from regression_tables import tables_with_controls
from inference_display import main_rows

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/extended_controls"
PROCESSED = ROOT / "data/processed/extended_controls"
OUT = ROOT / "outputs/extended_controls"
PM_ID = "d_6c0d5fc34838b12472475fdb73c0af29"
ADDED = ["wind_mps", "pm25_10ug"]
CONTEXTS = [("va", "quarterly", "quarter", "g_va_sa_qoq", "QoQ real VA log growth (SA)"),
            ("iip-monthly", "monthly", "month", "g_iip_sa", "MoM IIP log growth (SA)"),
            ("iip-quarterly", "quarterly", "quarter", "g_iip_sa", "QoQ IIP log growth (SA)")]


def cache_sources() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}

    def save(name, content, info):
        (RAW / name).write_bytes(content)
        manifest[name] = {**info, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                          "cached_utc": datetime.now(timezone.utc).isoformat()}
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    for name in ["wind_speed_halfhour_station_avg_2016_2025.csv",
                 "wind_speed_halfhour_active_station_count_2016_2025.csv", "wind_speed_2016_2025_quality_report.json"]:
        if not (RAW / name).exists():
            raise FileNotFoundError(
                f"Missing frozen NEA wind input: {RAW / name}. "
                "Restore the replication inputs with python scripts/restore_replication.py. "
                "The frozen cache replaces any dependency on the earlier electricity project's workstation.")
    s = session()
    if not (RAW / "pm25_metadata.json").exists():
        url = f"https://api-open.data.gov.sg/v2/public/api/datasets/{PM_ID}/metadata"
        r = s.get(url, timeout=90)
        r.raise_for_status()
        save("pm25_metadata.json", r.content, {"source": "National Environment Agency", "url": url})
        time.sleep(13)
    if not (RAW / "pm25_hourly.csv").exists():
        url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{PM_ID}/poll-download"
        for attempt in range(8):
            r = s.get(url, timeout=90)
            r.raise_for_status()
            signed = r.json().get("data", {}).get("url")
            if signed:
                break
            time.sleep(min(3*(attempt+1), 20))
        else:
            raise RuntimeError("PM2.5 bulk download not ready")
        response = s.get(signed, timeout=120)
        response.raise_for_status()
        save("pm25_hourly.csv", response.content, {
            "source": "NEA Historical 1-hr PM2.5", "dataset_id": PM_ID, "url": url,
            "source_url": f"https://data.gov.sg/datasets/{PM_ID}/view", "unit": "micrograms per cubic metre"})
    for name, info in manifest.items():
        assert hashlib.sha256((RAW / name).read_bytes()).hexdigest() == info["sha256"], name
    return manifest


def build_controls() -> dict:
    cache_sources()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    wind = pd.read_csv(RAW / "wind_speed_halfhour_station_avg_2016_2025.csv")
    wind["period_start"] = pd.to_datetime(wind.period_start, format="%Y-%m-%d %H:%M:%S")
    assert not wind.period_start.duplicated().any()
    assert wind.wind_speed.notna().all() and wind.wind_speed.ge(0).all()
    assert wind.period_start.dt.minute.isin([0, 30]).all()
    n_wind = len(wind)
    counts = pd.read_csv(RAW / "wind_speed_halfhour_active_station_count_2016_2025.csv")
    counts["period_start"] = pd.to_datetime(counts.period_start)
    count_column = next(c for c in counts if c != "period_start")
    wind = wind.merge(counts[["period_start", count_column]].rename(columns={count_column: "stations"}),
                      on="period_start", validate="one_to_one")
    assert len(wind) == n_wind and wind.stations.notna().all()
    wind["wind_mps"] = wind.wind_speed*(1852/3600)
    wind["month"] = wind.period_start.dt.to_period("M")
    wm = wind.groupby("month").agg(wind_mps=("wind_mps", "mean"), wind_bins=("wind_mps", "size"),
                                     wind_stations_mean=("stations", "mean"), wind_stations_min=("stations", "min"))
    pm = pd.read_csv(RAW / "pm25_hourly.csv")
    regions = ["north", "south", "east", "west", "central"]
    assert list(pm.columns) == ["1hr_pm25"] + regions
    pm["timestamp"] = pd.to_datetime(pm["1hr_pm25"], format="%d/%m/%Y %H:%M", errors="coerce")
    invalid_time = pm.timestamp.isna()
    invalid_values = pm[regions].apply(pd.to_numeric, errors="coerce").isna().any(axis=1) | pm[regions].lt(0).any(axis=1)
    rejected = pm[invalid_time | invalid_values].copy()
    clean = pm[~invalid_time & ~invalid_values].copy()
    exact_duplicate_count = int(clean.duplicated(["timestamp"]+regions).sum())
    clean = clean.drop_duplicates(["timestamp"]+regions)
    conflicting = clean.timestamp.duplicated(keep=False)
    conflict_rows = clean[conflicting].copy()
    clean = clean[~conflicting].copy()
    assert not clean.timestamp.duplicated().any()
    assert clean.timestamp.dt.minute.eq(0).all()
    # NEA one-hour readings describe the preceding hour; both inputs are in SGT.
    clean["hour_start"] = clean.timestamp-pd.Timedelta(hours=1)
    clean["pm25_ugm3"] = clean[regions].mean(axis=1)
    clean["month"] = clean.hour_start.dt.to_period("M")
    pm_monthly = clean.groupby("month").agg(pm25_ugm3=("pm25_ugm3", "mean"), pm_hours=("pm25_ugm3", "size"))
    rejected.to_csv(PROCESSED / "pm25_rejected_invalid_rows.csv", index=False)
    conflict_rows.to_csv(PROCESSED / "pm25_rejected_conflicting_rows.csv", index=False)
    monthly = wm.join(pm_monthly, how="outer").sort_index()
    monthly["days"] = monthly.index.days_in_month
    monthly["wind_bins_expected"] = monthly.days*48
    monthly["pm_hours_expected"] = monthly.days*24
    monthly["wind_coverage"] = monthly.wind_bins.fillna(0)/monthly.wind_bins_expected
    monthly["pm_coverage"] = monthly.pm_hours.fillna(0)/monthly.pm_hours_expected
    monthly["pm25_10ug"] = monthly.pm25_ugm3/10
    for threshold in [90, 80]:
        monthly[f"eligible_{threshold}"] = monthly.wind_coverage.ge(threshold/100) & monthly.pm_coverage.ge(threshold/100)
    monthly.index.name = "month"
    monthly.to_csv(PROCESSED / "monthly_controls.csv")
    quarterly_rows = []
    for quarter, group in monthly.groupby(monthly.index.asfreq("Q")):
        row = {"quarter": quarter, "wind_mps": np.nan, "pm25_10ug": np.nan,
               "wind_coverage": float(group.wind_coverage.min()), "pm_coverage": float(group.pm_coverage.min())}
        for threshold in [90, 80]:
            row[f"eligible_{threshold}"] = bool(len(group) == 3 and group[f"eligible_{threshold}"].all())
        if len(group) == 3 and group[["wind_mps", "pm25_10ug"]].notna().all().all():
            for name in ADDED:
                row[name] = float(np.average(group[name], weights=group.days))
        quarterly_rows.append(row)
    quarterly = pd.DataFrame(quarterly_rows)
    quarterly.to_csv(PROCESSED / "quarterly_controls.csv", index=False)
    summary = {"wind_halfhours": len(wind), "wind_unit_conversion": "knots × 1852/3600 = m/s",
               "pm_raw_rows": len(pm), "pm_invalid_timestamp_rows": int(invalid_time.sum()),
               "pm_invalid_value_rows": int(invalid_values.sum()), "pm_exact_duplicates_removed": exact_duplicate_count,
               "pm_conflicting_rows_removed": len(conflict_rows), "pm_valid_hours": len(clean),
               "timezone": "Asia/Singapore (timestamps interpreted as local SGT)",
               "pm_hour_assignment": "timestamp minus one hour, because readings are hour-ending",
               "primary_coverage_threshold": .90, "sensitivity_coverage_threshold": .80,
               "source_window_wind": [str(wind.period_start.min()), str(wind.period_start.max())],
               "source_window_pm_hour_start": [str(clean.hour_start.min()), str(clean.hour_start.max())]}
    overlap = monthly.loc["2017-01":"2025-12"]
    for threshold in [90, 80]:
        summary[f"eligible_months_{threshold}"] = int(overlap[f"eligible_{threshold}"].sum())
        summary[f"excluded_months_{threshold}"] = list(overlap.index[~overlap[f"eligible_{threshold}"]].astype(str))
        q = quarterly[quarterly.quarter.between(pd.Period("2017Q1"), pd.Period("2025Q4"))]
        summary[f"eligible_quarters_{threshold}"] = int(q[f"eligible_{threshold}"].sum())
        summary[f"excluded_quarters_{threshold}"] = list(q.loc[~q[f"eligible_{threshold}"], "quarter"].astype(str))
    (PROCESSED / "data_checks.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def fit_controls() -> None:
    for key, frequency, period, outcome, _ in CONTEXTS:
        folder = OUT / key
        folder.mkdir(parents=True, exist_ok=True)
        panel = read_panel("broad") if key == "va" else read_iip(frequency, "broad")
        controls = pd.read_csv(PROCESSED / f"{frequency}_controls.csv")
        controls[period] = pd.PeriodIndex(controls[period], freq="M" if frequency == "monthly" else "Q")
        merged = panel.merge(controls[[period]+ADDED+["eligible_90", "eligible_80", "wind_coverage", "pm_coverage"]],
                             on=period, validate="many_to_one")
        common = merged[merged.eligible_90].reset_index(drop=True)
        relaxed = merged[merged.eligible_80].reset_index(drop=True)
        assert common[ADDED].notna().all().all()
        assert common.groupby(period).sector_id.nunique().eq(panel.sector_id.nunique()).all()
        bandwidth = 12 if frequency == "monthly" else 4
        base_formula = f"{outcome} ~ temp_c + {WEATHER} + {FE}"
        full_formula = base_formula+" + "+" + ".join(ADDED)
        specs = [("C1", "Matched sample: rainfall and humidity", common, base_formula),
                 ("C2", "Matched sample: add wind", common, base_formula+" + wind_mps"),
                 ("C3", "Matched sample: add PM2.5", common, base_formula+" + pm25_10ug"),
                 ("C4", "Matched sample: wind and PM2.5", common, full_formula),
                 ("C5", "Full controls: exclude 2020-2022", common[~common.year.between(2020, 2022)].reset_index(drop=True), full_formula),
                 ("C6", "Full controls: 80% coverage", relaxed, full_formula)]
        original_path = ROOT / "outputs/regression_results.csv" if key == "va" else ROOT / f"outputs/iip/{frequency}/regression_results.csv"
        original = pd.read_csv(original_path)
        baseline_covariance = ((original.covariance.eq("Driscoll-Kraay") & original.hac_lags.eq(bandwidth))
                               | original.covariance.eq("Sector cluster"))
        old = original[original.model.eq("S3") & baseline_covariance].copy()
        old["model"], old["label"] = "C0", "Original long-sample baseline"
        rows = old.to_dict("records")
        all_coefficients = []
        for model_id, label, df, formula in specs:
            model = fit(df, formula, key+" "+label)
            terms = [term for term in ["temp_c", "rain_100mm", "rh_pct"]+ADDED if term in model.params.index]
            result = estimates(model, df, model_id, label, formula, terms, lags=bandwidth, period_col=period)
            rows.extend(result)
            all_coefficients.extend({"model": model_id, "term": term, "coefficient": value} for term, value in model.params.items())
            print(f"{key} {model_id}: temp={result[0]['coefficient']:.4f}, SE={result[0]['std_error']:.4f}, periods={result[0][f'n_{period}s']}", flush=True)
        pd.DataFrame(rows).to_csv(folder / "regression_results.csv", index=False)
        pd.DataFrame(all_coefficients).to_csv(folder / "all_coefficients.csv", index=False)
        common.to_csv(folder / "baseline_sample.csv", index=False)
        relaxed.to_csv(folder / "sensitivity_80_sample.csv", index=False)


def control_tables(key: str, frequency: str, outcome: str) -> tuple[str, str]:
    return tables_with_controls(OUT / key / "regression_results.csv", outcome, frequency, key)


def build_controls_report() -> Path:
    panels = []
    summaries = []
    for key, frequency, period, _, outcome in CONTEXTS:
        main, checks = control_tables(key, frequency, outcome)
        path = OUT / key / "regression_results.csv"
        results = pd.read_csv(path)
        selected = main_rows(results, frequency)
        temp = selected[selected.term.eq("temp_c")].set_index("model")
        base, full, no_covid = temp.loc["C1"], temp.loc["C4"], temp.loc["C5"]
        text = (f"Same retained dates: {full.start}–{full.end}, {int(full[f'n_{period}s'])} {period}s and {int(full.n_obs):,} observations. "
                f"The temperature coefficient changes from {base.coefficient:.3f} (SE {base.std_error:.3f}) to "
                f"{full.coefficient:.3f} (SE {full.std_error:.3f}) when both controls are added. "
                f"Full-control 95% interval [{full.ci_lower:.3f}, {full.ci_upper:.3f}]; p={full.p_value:.3f}. "
                f"Excluding 2020–2022 gives {no_covid.coefficient:.3f} (SE {no_covid.std_error:.3f}; p={no_covid.p_value:.3f}).")
        uri = "data:text/csv;base64,"+base64.b64encode(path.read_bytes()).decode()
        label = "Real VA · quarterly" if key == "va" else "IIP · "+frequency
        panels.append(f'<section id="{key}" class="reg-panel" aria-labelledby="tab-{key}" {"" if key == "va" else "hidden"}>'
                      f'<div class="section-title"><h2>{label}</h2><a href="{uri}" download="{key}_extended_controls.csv">Download coefficients ↓</a></div>'
                      f'<p class="notes">{html.escape(text)}</p><div class="etable-scroll">{main}</div>'
                      f'<h3>Pandemic exclusion and coverage sensitivity</h3><div class="etable-scroll">{checks}</div></section>')
        summaries.append({"analysis": key, "matched_baseline": float(base.coefficient), "full_controls": float(full.coefficient),
                          "full_se": float(full.std_error), "ci_lower": float(full.ci_lower), "ci_upper": float(full.ci_upper),
                          "p_value": float(full.p_value), "periods": int(full[f'n_{period}s']), "start": full.start, "end": full.end})
    template = (ROOT / "templates/regression_tables.html").read_text()
    template = template.replace("The regression tables.", "Adding wind and PM2.5.")
    template = template.replace("Models side by side, with coefficients, standard errors, fixed effects and fit statistics. Select an outcome and frequency to inspect the full specification.",
        "The baseline includes rainfall, humidity, wind and PM2.5. Columns 1–4 use identical dates to separate control choices from sample effects. The final column is long-history robustness without wind/PM2.5.")
    doc = lxml_html.fromstring(template.replace("__TABLE_PANELS__", "\n".join(panels)))
    note = doc.xpath('//div[@class="notes"]')[0]
    quality = json.loads((PROCESSED / "data_checks.json").read_text())
    coverage_uri = "data:text/csv;base64,"+base64.b64encode((PROCESSED / "monthly_controls.csv").read_bytes()).decode()
    note.clear()
    note.set("class", "notes")
    note.append(lxml_html.fragment_fromstring(f'''<div><p><strong>Coverage and construction.</strong> Wind is the earlier study's available-station half-hour mean, converted from knots to m/s using 1852/3600. PM2.5 is the actual hourly mean across five NEA regions, scaled per 10 µg/m³. Its hour-ending timestamps are shifted back one hour before assigning exposure months. The Changi temperature, humidity and rainfall variables are unchanged.</p>
<p>Primary models require at least 90% of expected half-hours for wind and 90% of expected hours for PM2.5 in each month. Quarterly means weight the three monthly means by calendar days, and all three months must pass. Retained in 2017–2025: {quality['eligible_months_90']} of 108 months and {quality['eligible_quarters_90']} of 36 quarters. The 80% sensitivity retains {quality['eligible_months_80']} months and {quality['eligible_quarters_80']} quarters. Missing observations are not imputed; these are observed-reading averages.</p>
<p>Monthly periods failing 90% coverage: {html.escape(', '.join(quality['excluded_months_90']))}. <a href="{coverage_uri}" download="extended_control_monthly_coverage.csv">Download monthly controls and coverage ↓</a>.</p>
<p>The PM2.5 source contains {quality['pm_invalid_timestamp_rows']} unparseable timestamp and {quality['pm_conflicting_rows_removed']} rows in conflicting duplicate timestamp groups; these are excluded, without fabricating dates or selecting a conflicting version. Station-count diagnostics are retained for wind. Source: <a href="https://data.gov.sg/datasets/{PM_ID}/view">NEA Historical 1-hr PM2.5</a> and <a href="https://data.gov.sg/datasets/d_7677738484067741bf3b56ab5d69c7e9/view">NEA wind speed</a>. Cached inputs, source hashes and transformation checks are retained in the project.</p>
<p>Fixed effects and uncertainty follow the earlier models: sector × month-of-year and year effects for monthly IIP; sector × quarter-of-year and year effects for quarterly IIP/VA. Pooled common-slope panels use sector-clustered CR1 errors and t(S−1) inference, with S=11 VA sectors or six IIP clusters. Calendar-aware Driscoll–Kraay results with 12 monthly or four quarterly lags are retained as comparisons. The added controls shorten the weather sample substantially; repeated sector rows do not add independent weather periods.</p>
<p><strong>Quarterly results are highly sample- and pandemic-sensitive.</strong> The large negative quarterly coefficients are already present in the matched baseline before wind or PM2.5 is added. Excluding 2020–2022 substantially weakens them. There are only 29 independent quarterly weather periods in the main controlled sample, or 19 after the pandemic exclusion. The equivalent time regression has only 12 or five residual degrees of freedom, respectively; nominal t(T−1) tests with HAC covariance are approximate in such a short sample.</p>
<p>PM2.5 can respond to weather and industrial activity. These specifications measure temperature conditional on pollution and weather, and do not identify total causal heat damage. Wind is averaged across a changing station network; temperature and humidity remain measured at Changi.</p></div>'''))
    path = ROOT / "outputs/extended_controls.html"
    path.write_text("<!doctype html>\n"+lxml_html.tostring(doc, encoding="unicode"))
    pd.DataFrame(summaries).to_csv(OUT / "temperature_summary.csv", index=False)
    print(f"Wrote {path.relative_to(ROOT)}")
    return path


def run_extended_controls() -> None:
    build_controls()
    fit_controls()
    from verify_extended_controls import verify_extended_controls
    verify_extended_controls()
    build_controls_report()


if __name__ == "__main__":
    run_extended_controls()
