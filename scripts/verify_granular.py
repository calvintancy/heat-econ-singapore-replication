"""Independent raw-source, balanced-panel QR and manual-HAC granular checks.

No production preparation or regression helper is imported.  The balanced-panel
common slope is recovered from date means; sector-season deviations recover
the full panel residuals and R².  Individual fits share one QR factorisation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/granular"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
ADDED = ["wind_mps", "pm25_10ug"]
OUTCOMES = {"G1": "y_short", "G2": "y_yoy", "L1": "y_log_level", "L3": "y_raw_level", "G2N": "y_yoy_nsa"}
CONTEXTS = [
    ("va-detailed", "va", "Q", "detailed"),
    ("va-broad-matched", "va", "Q", "broad"),
    ("iip-monthly-total", "iip", "M", "total"),
    ("iip-monthly-detailed", "iip", "M", "detailed"),
    ("iip-monthly-broad-matched", "iip", "M", "broad"),
    ("iip-quarterly-total", "iip", "Q", "total"),
    ("iip-quarterly-detailed", "iip", "Q", "detailed"),
    ("iip-quarterly-broad-matched", "iip", "Q", "broad"),
]


def _close(actual, expected, label: str, atol: float = 3e-7, rtol: float = 2e-7) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(f"{label}: saved={actual}, independent={expected}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_levels(family: str, adjustment: str = "sa") -> tuple[dict, dict, dict]:
    assert adjustment in {"sa", "nsa"}
    stem = ("va_" if family == "va" else "iip_")+adjustment
    raw = ROOT / "data/raw"
    metadata = json.loads((raw / f"{stem}_metadata.json").read_text())["Data"]["records"]
    data = json.loads((raw / f"{stem}_tabledata.json").read_text())["Data"]
    manifest = json.loads((raw / "manifest.json").read_text())
    checked_hashes = 0
    for filename, provenance in manifest.items():
        if filename.startswith(stem + "_"):
            assert _sha(raw / filename) == provenance["sha256"], filename
            checked_hashes += 1
    assert data["id"] == metadata["id"] and data["dataLastUpdated"] == metadata["dataLastUpdated"]
    values, names = {}, {}
    for row in data["row"]:
        sid = row["seriesNo"]
        names[sid] = row["rowText"]
        series = {}
        for cell in row["columns"]:
            if family == "va":
                year, quarter = cell["key"].split()
                period = pd.Period(f"{year}Q{quarter[0]}", freq="Q")
            else:
                period = pd.Period(datetime.strptime(cell["key"], "%Y %b"), freq="M")
            assert period not in series
            value = float(cell["value"].replace(",", ""))
            assert value > 0 and np.isfinite(value)
            series[period] = value
        periods = sorted(series)
        assert periods == list(pd.period_range(periods[0], periods[-1], freq="Q" if family == "va" else "M"))
        values[sid] = series
    assert sum(len(series) for series in values.values()) == metadata["total"]
    assert set(values) == {row["seriesNo"] for row in metadata["row"]}
    return values, names, {"table_id": data["id"], "source_cells": metadata["total"],
                           "source_hashes_checked": checked_hashes}


def _quarterly_levels(monthly: dict) -> dict:
    result = {}
    for sid, series in monthly.items():
        quarters = sorted({period.asfreq("Q") for period in series})
        result[sid] = {}
        for quarter in quarters:
            start = quarter.asfreq("M", how="start")
            months = [start + i for i in range(3)]
            if all(month in series for month in months):
                result[sid][quarter] = math.fsum(series[month] for month in months) / 3
    return result


def _raw_weather() -> dict:
    columns = [("temperature", "mean_temp", "temp_c", 1),
               ("rainfall", "total_rainfall", "rain_100mm", .01),
               ("humidity", "mean_rh", "rh_pct", 1)]
    monthly = {}
    for filename, source, target, scale in columns:
        with (ROOT / f"data/raw/{filename}_monthly.csv").open() as handle:
            for row in csv.DictReader(handle):
                period = pd.Period(row["month"], freq="M")
                month = monthly.setdefault(period, {})
                assert target not in month
                month[target] = float(row[source]) * scale
    monthly = {p: w for p, w in monthly.items() if len(w) == 3 and all(np.isfinite(list(w.values())))}
    quarterly = {}
    for quarter in sorted({p.asfreq("Q") for p in monthly}):
        start = quarter.asfreq("M", how="start")
        months = [start + i for i in range(3)]
        if not all(month in monthly for month in months):
            continue
        days = sum(month.days_in_month for month in months)
        quarterly[quarter] = {
            "temp_c": math.fsum(monthly[m]["temp_c"] * m.days_in_month for m in months) / days,
            "rh_pct": math.fsum(monthly[m]["rh_pct"] * m.days_in_month for m in months) / days,
            "rain_100mm": math.fsum(monthly[m]["rain_100mm"] for m in months),
        }
    return {"M": monthly, "Q": quarterly}


def _raw_extended() -> tuple[dict, dict]:
    """Reconstruct retained-hour means and coverage using CSV/dictionary loops."""
    folder = ROOT / "data/raw/extended_controls"
    manifest = json.loads((folder / "manifest.json").read_text())
    for filename, provenance in manifest.items():
        assert _sha(folder / filename) == provenance["sha256"], filename
    wind, timestamps = {}, set()
    with (folder / "wind_speed_halfhour_station_avg_2016_2025.csv").open() as handle:
        for row in csv.DictReader(handle):
            timestamp = datetime.fromisoformat(row["period_start"])
            assert timestamp not in timestamps and timestamp.minute in {0, 30}
            timestamps.add(timestamp)
            value = float(row["wind_speed"]) * 1852 / 3600
            assert value >= 0 and np.isfinite(value)
            wind.setdefault(timestamp.strftime("%Y-%m"), []).append(value)
    regions = ["north", "south", "east", "west", "central"]
    hours, invalid, exact_duplicates, conflicts = {}, 0, 0, set()
    with (folder / "pm25_hourly.csv").open() as handle:
        for row in csv.DictReader(handle):
            try:
                timestamp = datetime.strptime(row["1hr_pm25"], "%d/%m/%Y %H:%M")
                values = tuple(float(row[region]) for region in regions)
                if not all(math.isfinite(v) and v >= 0 for v in values):
                    raise ValueError("Invalid regional PM value")
            except ValueError:
                invalid += 1
                continue
            assert timestamp.minute == 0
            if timestamp in hours:
                if hours[timestamp] == values:
                    exact_duplicates += 1
                else:
                    conflicts.add(timestamp)
            else:
                hours[timestamp] = values
    pm = {}
    for timestamp, values in hours.items():
        if timestamp in conflicts:
            continue
        # NEA reports the previous-hour mean, with SGT hour-ending timestamps.
        start = timestamp - timedelta(hours=1)
        pm.setdefault(start.strftime("%Y-%m"), []).append(math.fsum(values) / 5)
    monthly = {}
    for label in sorted(set(wind) | set(pm)):
        period = pd.Period(label, freq="M")
        nwind, npm = len(wind.get(label, [])), len(pm.get(label, []))
        wc, pc = nwind / (period.days_in_month * 48), npm / (period.days_in_month * 24)
        monthly[period] = {
            "wind_mps": math.fsum(wind[label]) / nwind if nwind else np.nan,
            "pm25_10ug": math.fsum(pm[label]) / npm / 10 if npm else np.nan,
            "wind_coverage": wc, "pm_coverage": pc, "eligible_90": wc >= .9 and pc >= .9,
        }
    quarterly = {}
    for quarter in sorted({p.asfreq("Q") for p in monthly}):
        start = quarter.asfreq("M", how="start")
        months = [start + i for i in range(3)]
        if not all(month in monthly for month in months):
            quarterly[quarter] = {"wind_mps": np.nan, "pm25_10ug": np.nan, "eligible_90": False}
            continue
        days = sum(month.days_in_month for month in months)
        jointly_available = all(np.isfinite(monthly[m][term]) for m in months for term in ADDED)
        quarterly[quarter] = {
            **{term: math.fsum(monthly[m][term] * m.days_in_month for m in months) / days if jointly_available else np.nan for term in ADDED},
            "eligible_90": all(monthly[m]["eligible_90"] for m in months),
        }
    for frequency, mapping, period_col, freq in [("monthly", monthly, "month", "M"), ("quarterly", quarterly, "quarter", "Q")]:
        saved = pd.read_csv(ROOT / f"data/processed/extended_controls/{frequency}_controls.csv")
        assert set(pd.PeriodIndex(saved[period_col], freq=freq)) == set(mapping)
        for row in saved.to_dict("records"):
            period = pd.Period(row[period_col], freq=freq)
            expected = mapping[period]
            assert bool(row["eligible_90"]) == expected["eligible_90"]
            for field in ADDED + (["wind_coverage", "pm_coverage"] if freq == "M" else []):
                _close(row[field], expected[field], f"Raw wind/PM aggregation {period}/{field}", atol=1e-11, rtol=1e-11)
    return {"M": monthly, "Q": quarterly}, {
        "source_hashes_checked": len(manifest), "wind_halfhours": len(timestamps),
        "pm_invalid_rows": invalid, "pm_exact_duplicates": exact_duplicates,
        "pm_conflicting_timestamp_hours": len(conflicts), "pm_retained_hours": len(hours) - len(conflicts),
        "monthly_eligible_90": sum(row["eligible_90"] for row in monthly.values()),
        "quarterly_eligible_90": sum(row["eligible_90"] for row in quarterly.values()),
    }


def _ids(levels: dict, family: str) -> tuple[list, list]:
    broad = ([f"1.1.{i}" for i in range(1, 5)] + [f"1.2.{i}" for i in range(1, 8)]) if family == "va" else [f"1.{i}" for i in range(1, 7)]
    detailed = sorted(sid for sid in levels if any(sid == parent or sid.startswith(parent + ".") for parent in broad)
                      and not any(other.startswith(sid + ".") for other in levels))
    assert len(detailed) == (19 if family == "va" else 18)
    catalog = pd.read_csv(ROOT / ("data/processed/sector_catalog.csv" if family == "va" else "data/processed/iip/sector_catalog.csv"), dtype={"sector_id": str})
    assert set(catalog.loc[catalog.broad_panel, "sector_id"]) == set(broad)
    assert set(catalog.loc[catalog.detailed_panel, "sector_id"]) == set(detailed)
    return broad, detailed


def _valid_dates(levels: dict, ids: list, weather: dict, horizon: int) -> set:
    result = set(weather)
    for sid in ids:
        series = levels[sid]
        result &= {period for period in series if period - 1 in series and period - horizon in series}
    return result


def _inverse_design(dates: list, terms: list, weather: dict, extended: dict) -> tuple[np.ndarray, np.ndarray]:
    years, seasons = sorted({p.year for p in dates}), sorted({p.month if p.freqstr == "M" else p.quarter for p in dates})
    x = np.array([[1.] + [weather[p][term] if term in BASE else extended[p][term] for term in terms]
                  + [float(p.year == year) for year in years[1:]]
                  + [float((p.month if p.freqstr == "M" else p.quarter) == season) for season in seasons[1:]] for p in dates])
    scales = np.linalg.norm(x, axis=0)
    q, r = np.linalg.qr(x / scales, mode="reduced")
    assert np.linalg.matrix_rank(x / scales, tol=1e-11) == x.shape[1]
    inverse = np.linalg.solve(r, q.T) / scales[:, None]
    _close(inverse @ x, np.eye(x.shape[1]), "Independent QR inverse", atol=2e-10, rtol=1e-10)
    return x, inverse


def _hac_variances(influence: np.ndarray, dates: list, lags: int, correction: float) -> np.ndarray:
    """Diagonal Bartlett sandwich from zero-filled actual-calendar influences."""
    positions = [p.ordinal - dates[0].ordinal for p in dates]
    shape = influence.shape[1:]
    scores = np.zeros((positions[-1] + 1, int(np.prod(shape))))
    scores[positions] = influence.reshape(len(dates), -1)
    variance = np.sum(scores * scores, axis=0)
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * np.sum(scores[lag:] * scores[:-lag], axis=0)
    assert np.min(variance) >= -1e-10
    return np.maximum(variance, 0).reshape(shape) * correction


def verify_granular() -> dict:
    levels_va, names_va, audit_va = _raw_levels("va")
    levels_iip, names_iip, audit_iip = _raw_levels("iip")
    nsa_va, _, audit_nsa_va = _raw_levels("va", "nsa")
    nsa_iip, _, audit_nsa_iip = _raw_levels("iip", "nsa")
    levels = {("va", "Q"): levels_va, ("iip", "M"): levels_iip, ("iip", "Q"): _quarterly_levels(levels_iip)}
    nsa_levels = {("va", "Q"): nsa_va, ("iip", "M"): nsa_iip, ("iip", "Q"): _quarterly_levels(nsa_iip)}
    names = {"va": names_va, "iip": names_iip}
    selections = {"va": _ids(levels_va, "va"), "iip": _ids(levels_iip, "iip")}
    weather = _raw_weather()
    extended, audit_extended = _raw_extended()
    summary = pd.read_csv(OUT / "summary.csv")
    number_outcomes = len(OUTCOMES)
    assert len(summary) == len(CONTEXTS) * 3 * number_outcomes
    assert not summary.duplicated(["context", "window", "model"]).any()
    checked, point_error, se_error, definition_error, n_raw, n_individual = [], 0., 0., 0., 0, 0
    expected_contexts = {key for key, _, _, _ in CONTEXTS}
    assert set(summary.context) == expected_contexts
    for key, family, freq, granularity in CONTEXTS:
        period_col, horizon = ("month", 12) if freq == "M" else ("quarter", 4)
        source = levels[(family, freq)]
        nsa_source = nsa_levels[(family, freq)]
        broad, detailed = selections[family]
        ids = detailed if granularity == "detailed" else broad if granularity == "broad" else ["1"]
        ids = sorted(ids)
        dates = _valid_dates(source, ids, weather[freq], horizon)
        if granularity == "broad":
            dates &= _valid_dates(source, detailed, weather[freq], horizon)
        for window in ["long", "full", "exclude"]:
            retained = dates if window == "long" else {p for p in dates if extended[freq].get(p, {}).get("eligible_90", False)}
            if window == "exclude":
                retained = {p for p in retained if not 2020 <= p.year <= 2022}
            retained = sorted(retained)
            terms = BASE + ([] if window == "long" else ADDED)
            folder = OUT / key / window
            sample_path = folder / "samples/G1.csv"
            digest = _sha(sample_path)
            for model in OUTCOMES:
                assert _sha(folder / "samples" / f"{model}.csv") == digest, f"Core sample changed across outcomes {key}/{window}/{model}"
            sample = pd.read_csv(sample_path, dtype={"sector_id": str})
            sample[period_col] = pd.PeriodIndex(sample[period_col], freq=freq)
            assert set(sample.sector_id) == set(ids) and set(sample[period_col]) == set(retained)
            assert not sample.duplicated(["sector_id", period_col]).any()
            assert len(sample) == len(ids) * len(retained)
            assert sample.groupby(period_col).sector_id.nunique().eq(len(ids)).all()
            assert sample.groupby("sector_id")[period_col].nunique().eq(len(retained)).all()
            # Source outcomes are rebuilt using true prior calendar dates before screens.
            y = np.empty((len(retained), len(ids), number_outcomes))
            by_date, by_sector = {p: i for i, p in enumerate(retained)}, {sid: i for i, sid in enumerate(ids)}
            for row in sample.to_dict("records"):
                p, sid = row[period_col], row["sector_id"]
                v, prior, year_prior = source[sid][p], source[sid][p - 1], source[sid][p - horizon]
                nsa, nsa_prior = nsa_source[sid][p], nsa_source[sid][p-horizon]
                expected_by_outcome = {"y_short": 100*math.log(v/prior), "y_yoy": 100*math.log(v/year_prior),
                    "y_log_level": 100*math.log(v), "y_raw_level": v, "y_yoy_nsa": 100*math.log(nsa/nsa_prior)}
                expected = [expected_by_outcome[column] for column in OUTCOMES.values()]
                assert row["sector"] == names[family][sid]
                assert row["year"] == p.year and row["season"] == (p.month if freq == "M" else p.quarter)
                assert row["time"] == p.ordinal - pd.Period("1982-01" if freq == "M" else "1982Q1", freq=freq).ordinal
                _close(row["output_sa"], v, "Raw official level", atol=1e-10, rtol=1e-12)
                _close(row["output_nsa"], nsa, "Raw official NSA level", atol=1e-10, rtol=1e-12)
                for column, value in zip(OUTCOMES.values(), expected):
                    _close(row[column], value, f"Raw full-history outcome {key}/{sid}/{p}/{column}", atol=1e-10, rtol=1e-12)
                    definition_error = max(definition_error, abs(row[column] - value))
                y[by_date[p], by_sector[sid]] = expected
                for term in BASE:
                    _close(row[term], weather[freq][p][term], "Raw current weather", atol=1e-12, rtol=1e-12)
                for term in ADDED:
                    _close(row[term], extended[freq].get(p, {}).get(term, np.nan), "Raw wind/PM source mean", atol=1e-11, rtol=1e-11)
                assert bool(row["eligible_90"]) == extended[freq].get(p, {}).get("eligible_90", False)
            n_raw += len(sample)
            x, inverse = _inverse_design(retained, terms, weather[freq], extended[freq])
            ntime, nsectors, p = len(retained), len(ids), len(terms)
            nobs = ntime * nsectors
            seasons = np.array([period.month if freq == "M" else period.quarter for period in retained])
            n_years = len({period.year for period in retained})
            k_panel = nsectors * len(set(seasons)) + n_years - 1 + p
            mean_y = y.mean(axis=1)
            beta = inverse @ mean_y
            mean_residual = mean_y - x @ beta
            deviation = y - mean_y[:, None, :]
            for season in set(seasons):
                use = seasons == season
                deviation[use] -= deviation[use].mean(axis=0)
            panel_residual = mean_residual[:, None, :] + deviation
            r2 = 1 - np.sum(panel_residual ** 2, axis=(0, 1)) / np.sum((y - y.mean(axis=(0, 1))) ** 2, axis=(0, 1))
            pooled = pd.read_csv(folder / "regression_results.csv")
            assert not pooled.duplicated(["model", "term", "covariance", "hac_lags"]).any()
            expected_keys = {(model, term, "Driscoll-Kraay", horizon) for model in OUTCOMES for term in terms}
            expected_keys |= {(model, "temp_c", "Driscoll-Kraay", 2*horizon) for model in OUTCOMES}
            if nsectors > 1:
                expected_keys |= {(model, term, "Sector cluster", 0) for model in OUTCOMES for term in terms}
            assert set(zip(pooled.model, pooled.term, pooled.covariance, pooled.hac_lags)) == expected_keys
            # Panel common-slope finite correction differs from a singleton fit.
            correction = ntime / (ntime - 1) * (nobs - 1) / (nobs - k_panel)
            pooled_influence = inverse[1:1+p].T[:, :, None] * mean_residual[:, None, :]
            hac = {h: np.sqrt(_hac_variances(pooled_influence, retained, h, correction)) for h in [horizon, 2*horizon]}
            c = student_t.ppf(.975, ntime - 1)
            for i, (model, outcome) in enumerate(OUTCOMES.items()):
                rows = pooled[pooled.model.eq(model)]
                assert rows.term.isin(terms).all() and rows.outcome.eq(outcome).all()
                assert rows.n_obs.eq(nobs).all() and rows.n_sectors.eq(nsectors).all()
                assert rows[f"n_{period_col}s"].eq(ntime).all()
                assert rows.time_residual_df.eq(ntime - x.shape[1]).all()
                assert rows.start.eq(str(retained[0])).all() and rows.end.eq(str(retained[-1])).all()
                _close(rows.mean_y, y[:, :, i].mean(), "Exact pooled mean Y", atol=1e-9)
                _close(rows.sd_y, y[:, :, i].std(ddof=1), "Exact pooled SD Y", atol=1e-9)
                _close(rows.r_squared, r2[i], "Full panel R2 from season deviations", atol=1e-9)
                assert rows.djo_log_growth_form.eq(model in {"G1", "G2", "G2N"}).all()
                assert rows.output_adjustment.eq("NSA" if outcome.endswith("_nsa") else "SA").all()
                for row in rows.itertuples():
                    j = terms.index(row.term)
                    _close(row.coefficient, beta[1+j, i], "Pooled mean/FWL point coefficient")
                    point_error = max(point_error, abs(row.coefficient - beta[1+j, i]))
                    if row.term == "temp_c":
                        _close(row.effect_0p1c, .1 * beta[1+j, i], "+0.1°C pooled effect")
                    if row.covariance == "Driscoll-Kraay":
                        expected_se = hac[int(row.hac_lags)][j, i]
                        _close(row.std_error, expected_se, "Manual panel calendar HAC")
                        se_error = max(se_error, abs(row.std_error - expected_se))
                        assert int(row.inference_df) == ntime - 1
                        _close(row.ci_lower, beta[1+j, i] - c * expected_se, "Panel HAC lower CI")
                        _close(row.ci_upper, beta[1+j, i] + c * expected_se, "Panel HAC upper CI")
                        _close(row.p_value, 2 * student_t.sf(abs(beta[1+j, i] / expected_se), ntime - 1), "Panel HAC p value", atol=1e-9)
                stats = summary[summary.context.eq(key) & summary.window.eq(window) & summary.model.eq(model)]
                assert len(stats) == 1 and int(stats.pooled_parameters.iloc[0]) == k_panel
                assert int(stats.n_periods.iloc[0]) == ntime and int(stats.n_obs.iloc[0]) == nobs
                _close(stats.mean_y, y[:, :, i].mean(), "Summary mean Y")
                _close(stats.sd_y, y[:, :, i].std(ddof=1), "Summary SD Y")
            if granularity == "detailed":
                # All individual outcome/sector regressions in one matrix solve.
                separate = pd.read_csv(folder / "separate_sector_coefficients.csv", dtype={"sector_id": str})
                assert len(separate) == number_outcomes * nsectors and separate.covariance.eq("Driscoll-Kraay").all()
                assert not separate.duplicated(["sector_id", "model"]).any()
                individual_beta = (inverse @ y.reshape(ntime, -1)).reshape(x.shape[1], nsectors, number_outcomes)
                individual_residual = y - (x @ individual_beta.reshape(x.shape[1], -1)).reshape(ntime, nsectors, number_outcomes)
                individual_influence = inverse[1, :, None, None] * individual_residual
                individual_se = np.sqrt(_hac_variances(individual_influence, retained, horizon, ntime / (ntime - x.shape[1])))
                individual_r2 = 1 - np.sum(individual_residual ** 2, axis=0) / np.sum((y - y.mean(axis=0)) ** 2, axis=0)
                for row in separate.itertuples():
                    s, i = ids.index(row.sector_id), list(OUTCOMES).index(row.model)
                    b, se = individual_beta[1, s, i], individual_se[s, i]
                    _close(row.coefficient, b, "Individual QR temperature slope")
                    _close(row.std_error, se, "Individual manual calendar HAC")
                    _close(row.ci_lower, b-c*se, "Individual lower CI")
                    _close(row.ci_upper, b+c*se, "Individual upper CI")
                    _close(row.p_value, 2*student_t.sf(abs(b/se), ntime-1), "Individual p value", atol=1e-9)
                    _close(row.mean_y, y[:, s, i].mean(), "Individual mean Y", atol=1e-9)
                    _close(row.sd_y, y[:, s, i].std(ddof=1), "Individual SD Y", atol=1e-9)
                    _close(row.r_squared, individual_r2[s, i], "Individual R2", atol=1e-9)
                    assert row.n_obs == ntime and row.n_sectors == 1 and row.inference_df == ntime-1
                    assert row.time_residual_df == ntime-x.shape[1] and row.hac_lags == horizon
                    point_error = max(point_error, abs(row.coefficient-b))
                    se_error = max(se_error, abs(row.std_error-se))
                n_individual += len(separate)
            checked.append({"context": key, "window": window, "sectors": nsectors, "observed_periods": ntime,
                            "calendar_periods": retained[-1].ordinal-retained[0].ordinal+1,
                            "start": str(retained[0]), "end": str(retained[-1]), "time_rank": x.shape[1],
                            "time_residual_df": ntime-x.shape[1], "panel_rank": k_panel,
                            "pooled_outcomes_checked": number_outcomes, "individual_outcomes_checked": number_outcomes*nsectors if granularity == "detailed" else 0})
    audit = {
        "status": "passed", "method": "Independent raw calendar lookup, balanced date-mean QR, sector-season deviation R², zero-filled manual Bartlett HAC",
        "source_audits": {"va": audit_va, "iip": audit_iip, "va_nsa": audit_nsa_va, "iip_nsa": audit_nsa_iip, "extended_controls": audit_extended},
        "exact_sample_rows_checked": n_raw, "pooled_outcome_fits_checked": number_outcomes * len(checked),
        "individual_outcome_fits_checked": n_individual, "sample_windows_checked": len(checked),
        "maximum_absolute_raw_definition_error": definition_error,
        "maximum_absolute_coefficient_difference": point_error, "maximum_absolute_hac_se_difference": se_error,
        "sector_CR1_SE_scope": "Pooled points, samples and descriptive fields checked here; sector CR1 covariance is checked by the separate sector-cluster verifier.",
        "checks": checked,
    }
    (OUT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Granular verification passed: {len(checked)} sample windows, {number_outcomes*len(checked)} pooled and {n_individual} individual fits; raw SA/NSA source and calendar HAC checks.")
    return audit


def run_verify_granular() -> dict:
    return verify_granular()


if __name__ == "__main__":
    verify_granular()
