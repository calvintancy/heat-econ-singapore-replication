"""Univariate persistence diagnostics on contiguous cached outcome histories."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import acf, adfuller, kpss

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/outcomes"
OUT = ROOT / "outputs/heterogeneity/diagnostics"
CONTEXTS = [("va", "quarter", "Q", 4), ("iip-monthly", "month", "M", 12),
            ("iip-quarterly", "quarter", "Q", 4)]
OUTCOMES = {"y_log_level": ("100 × log seasonally adjusted output", "ct"),
            "y_short": ("MoM/QoQ log growth × 100 (SA), not annualised", "c"),
            "y_yoy": ("YoY log growth × 100 (SA)", "c"),
            "y_yoy_nsa": ("YoY log growth × 100 (NSA)", "c")}
WEATHER = {"temp_c": ("Air temperature, °C", "ct"),
           "rain_100mm": ("Period rainfall total, 100 mm", "c"),
           "rh_pct": ("Relative humidity, percentage points", "ct")}
CAUTIONS = [
    "ADF has a unit-root null; KPSS has a level/trend-stationarity null. Non-rejection proves neither null.",
    "Finite-sample power, lag choice, deterministic terms and structural breaks affect both tests.",
    "Long comparison histories include pandemic and other breaks; pre-2020 results are a contiguous sensitivity, not a break test.",
    "YoY growth overlaps across dates and can be persistent even when one-period growth is stationary.",
    "Weather tests follow deterministic seasonal adjustment; these are descriptive screens with approximate usual critical values.",
    "No panel unit-root inference, residual cointegration test or causal-validity claim is made.",
    "The gapped full-control sample is not compressed into a regular time series for ADF/KPSS.",
]


def longest_run(series: pd.Series) -> tuple[pd.Series, dict]:
    """Keep calendar gaps in the grid, then select the longest finite run."""
    if series.index.has_duplicates:
        raise ValueError("Duplicate calendar observations in diagnostic series")
    series = series.sort_index()
    grid = pd.period_range(series.index.min(), series.index.max(), freq=series.index.freq)
    regular = series.reindex(grid).astype(float)
    valid = np.isfinite(regular.to_numpy())
    padded = np.r_[False, valid, False].astype(int)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    if not len(starts):
        raise ValueError("No finite observations in diagnostic series")
    best = int(np.argmax(ends-starts))
    chosen = regular.iloc[starts[best]:ends[best]].copy()
    assert np.all(np.diff(chosen.index.asi8) == 1)
    return chosen, {"source_start": str(grid.min()), "source_end": str(grid.max()),
        "source_expected_periods": len(grid), "source_finite_periods": int(valid.sum()),
        "source_missing_periods": int((~valid).sum()), "finite_runs": len(starts),
        "start": str(chosen.index.min()), "end": str(chosen.index.max()), "n_periods": len(chosen),
        "finite_observations_outside_selected_run": int(valid.sum()-len(chosen))}


def seasonal_adjust(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Estimate seasonal offsets alongside a trend, but subtract only those offsets.
    season = series.index.month if series.index.freqstr.startswith("M") else series.index.quarter
    dummy = pd.get_dummies(season, drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(len(series)), np.arange(len(series)), dummy])
    beta = np.linalg.lstsq(design, series.to_numpy(), rcond=None)[0]
    offsets = dummy @ beta[2:]
    offsets -= offsets.mean()
    component = pd.Series(offsets, index=series.index)
    adjusted = series-component
    assert abs(adjusted.mean()-series.mean()) < 1e-10
    return adjusted, component


def autocorrelations(values: np.ndarray, horizon: int, deterministic: str) -> dict:
    raw_acf = acf(values, nlags=horizon, fft=False, missing="raise")
    x = np.ones((len(values), 1))
    if deterministic == "ct":
        x = np.column_stack([x, np.arange(len(values))])
    residual = values-x @ np.linalg.lstsq(x, values, rcond=None)[0]
    adjusted_acf = acf(residual, nlags=horizon, fft=False, missing="raise")
    return {"acf_1": float(raw_acf[1]), "acf_h": float(raw_acf[horizon]),
            "detrended_acf_1": float(adjusted_acf[1]), "detrended_acf_h": float(adjusted_acf[horizon])}


def run_tests(values: np.ndarray, deterministic: str, horizon: int) -> list[dict]:
    result = []
    for test in ["ADF", "KPSS"]:
        row = {"test": test, "deterministic": deterministic, "requested_lag_cap": horizon,
               "status": "ok", "statistic": np.nan, "p_value": np.nan, "p_value_relation": "=",
               "used_lags": np.nan, "automatic_lags_before_cap": np.nan,
               "regression_nobs": np.nan, "critical_1pct": np.nan, "critical_5pct": np.nan,
               "critical_10pct": np.nan, "critical_2_5pct": np.nan, "aic": np.nan,
               "lag_cap_binding": False, "warnings": "", "error": ""}
        if len(values) < 5*horizon or np.ptp(values) == 0:
            row.update(status="not_run", error="Fewer than five seasonal cycles or constant series")
            result.append(row)
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if test == "ADF":
                    feasible = len(values)//2-(2 if deterministic == "ct" else 1)-1
                    cap = min(horizon, feasible)
                    statistic, pvalue, lags, nobs, critical, ic = adfuller(
                        values, maxlag=cap, regression=deterministic, autolag="AIC")
                    row.update(regression_nobs=int(nobs), aic=float(ic),
                               automatic_lags_before_cap=int(lags), lag_cap_binding=bool(lags == cap))
                else:
                    statistic, pvalue, auto_lags, critical = kpss(values, regression=deterministic, nlags="auto")
                    row["automatic_lags_before_cap"] = int(auto_lags)
                    lags = min(auto_lags, horizon)
                    if auto_lags > horizon:
                        # The first call selects the bandwidth; only the second statistic is reported.
                        caught.clear()
                        statistic, pvalue, lags, critical = kpss(values, regression=deterministic, nlags=horizon)
                        row["lag_cap_binding"] = True
                    if statistic > critical["1%"]:
                        row["p_value_relation"] = "<"
                    elif statistic < critical["10%"]:
                        row["p_value_relation"] = ">"
                    row["regression_nobs"] = len(values)
                row.update(statistic=float(statistic), p_value=float(pvalue), used_lags=int(lags),
                    critical_1pct=float(critical["1%"]), critical_5pct=float(critical["5%"]),
                    critical_10pct=float(critical["10%"]), critical_2_5pct=critical.get("2.5%", np.nan),
                    warnings=" | ".join(f"{w.category.__name__}: {w.message}" for w in caught))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        result.append(row)
    return result


def paired_evidence(adf: dict, kpss_result: dict) -> str:
    if adf["status"] != "ok" or kpss_result["status"] != "ok":
        return "Tests unavailable"
    a, k = adf["p_value"] < .05, kpss_result["p_value"] < .05
    if a and not k:
        return "ADF rejects / KPSS does not reject"
    if not a and k:
        return "ADF does not reject / KPSS rejects"
    return "Both reject: mixed evidence" if a else "Neither rejects: inconclusive"


def diagnose_series(series: pd.Series, metadata: dict, deterministic: str, horizon: int,
                    is_weather: bool) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    tests, summaries, observations, audits = [], [], [], []
    for window in ["long", "pre_2020"]:
        candidate = series if window == "long" else series[series.index.year < 2020]
        selected, audit = longest_run(candidate)
        info = {**metadata, "window": window, **audit, "horizon": horizon,
                "includes_2020_2022": bool(np.isin(selected.index.year, [2020, 2021, 2022]).any()),
                "seasonality": "Seasonal offsets only; trend retained" if is_weather else "Official SA output"}
        audits.append(info)
        value, seasonal_component = seasonal_adjust(selected) if is_weather else (selected, selected*0)
        for date, raw, tested, seasonal in zip(selected.index, selected, value, seasonal_component):
            observations.append({**metadata, "window": window, "period": str(date),
                "raw_value": float(raw), "test_value": float(tested), "seasonal_component": float(seasonal)})
        specifications = [("primary", deterministic)]
        if metadata["variable"] == "temp_c":
            specifications.append(("constant_sensitivity", "c"))
        for specification, det in specifications:
            common = {**info, "specification": specification, "deterministic": det}
            raw_tests = run_tests(value.to_numpy(), det, horizon)
            tests.extend({**common, **r} for r in raw_tests)
            correlations = autocorrelations(value.to_numpy(), horizon, det)
            original_correlations = autocorrelations(selected.to_numpy(), horizon, det)
            adf, kpss_result = raw_tests
            summaries.append({**common, **correlations,
                "raw_acf_1": original_correlations["acf_1"], "raw_acf_h": original_correlations["acf_h"],
                "adf_statistic": adf["statistic"], "adf_p_value": adf["p_value"],
                "adf_lags": adf["used_lags"], "adf_status": adf["status"],
                "kpss_statistic": kpss_result["statistic"], "kpss_p_value": kpss_result["p_value"],
                "kpss_p_value_relation": kpss_result["p_value_relation"],
                "kpss_lags": kpss_result["used_lags"], "kpss_status": kpss_result["status"],
                "evidence_5pct": paired_evidence(adf, kpss_result),
                "warnings": " | ".join(r["warnings"] for r in raw_tests if r["warnings"]),
                "errors": " | ".join(r["error"] for r in raw_tests if r["error"])})
    return tests, summaries, observations, audits


def full_control_audit(frame: pd.DataFrame, context: str, period: str, freq: str) -> dict:
    unique = frame.groupby(period, as_index=False).first()
    eligible = unique[unique.eligible_90.astype(bool)]
    if eligible.empty:
        return {"context": context, "status": "No eligible full-control dates", "tests_run": False}
    grid = pd.period_range(eligible[period].min(), eligible[period].max(), freq=freq)
    indicator = pd.Series(np.nan, index=grid)
    indicator.loc[pd.PeriodIndex(eligible[period], freq=freq)] = 1.
    _, audit = longest_run(indicator)
    return {"context": context, "frequency": freq, "eligible_start": str(grid.min()),
        "eligible_end": str(grid.max()), "eligible_periods": len(eligible), "calendar_span_periods": len(grid),
        "missing_calendar_periods": len(grid)-len(eligible), "longest_contiguous_run": audit["n_periods"],
        "longest_run_start": audit["start"], "longest_run_end": audit["end"], "tests_run": False,
        "status": "Not tested: coverage-screened, gapped and short; retain long-series diagnostics"}


def verify_statistics(tests: pd.DataFrame, summary: pd.DataFrame, observations: list[dict]) -> dict:
    keys = ["context", "kind", "sector_id", "variable", "window"]
    values = {key: group.test_value.to_numpy() for key, group in pd.DataFrame(observations).groupby(keys, sort=False)}
    errors = {"adf_statistic_max_abs_error": 0., "kpss_statistic_max_abs_error": 0., "acf_max_abs_error": 0.}
    for row in tests.to_dict("records"):
        y = values[tuple(row[k] for k in keys)]
        lag = int(row["used_lags"])
        deterministic = np.ones((len(y), 1))
        if row["deterministic"] == "ct":
            deterministic = np.column_stack([deterministic, np.arange(len(y))])
        if row["test"] == "ADF":
            index = np.arange(lag+1, len(y))
            difference = np.diff(y)
            x = np.column_stack([y[index-1], *[difference[index-j-1] for j in range(1, lag+1)],
                                 deterministic[index]])
            inverse = np.linalg.pinv(x)
            beta = inverse @ difference[index-1]
            residual = difference[index-1]-x @ beta
            variance = float(residual @ residual)/(len(index)-np.linalg.matrix_rank(x))
            statistic = beta[0]/np.sqrt(variance*(inverse[0] @ inverse[0]))
            field = "adf_statistic_max_abs_error"
        else:
            residual = y-deterministic @ np.linalg.lstsq(deterministic, y, rcond=None)[0]
            long_variance = float(residual @ residual)
            for j in range(1, lag+1):
                long_variance += 2*(1-j/(lag+1))*float(residual[j:] @ residual[:-j])
            long_variance /= len(y)
            statistic = float(np.cumsum(residual) @ np.cumsum(residual))/(len(y)**2*long_variance)
            field = "kpss_statistic_max_abs_error"
        errors[field] = max(errors[field], abs(float(statistic)-row["statistic"]))
    for row in summary.to_dict("records"):
        y = values[tuple(row[k] for k in keys)]
        centered = y-y.mean()
        for lag, column in [(1, "acf_1"), (row["horizon"], "acf_h")]:
            manual = float(centered[lag:] @ centered[:-lag]) / float(centered @ centered)
            errors["acf_max_abs_error"] = max(errors["acf_max_abs_error"], abs(manual-row[column]))
    assert max(errors.values()) < 1e-8, errors
    return errors


def run_diagnostics() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tests, summaries, observations, audits, full_windows = [], [], [], [], []
    checks = {"input_hashes": {}, "comparison_samples": {}, "outcome_transform_max_abs_errors": {},
              "software": {"statsmodels": statsmodels.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    sentinel = pd.Series([1., 2., 4., 5., 6.], index=pd.PeriodIndex(
        ["2000-01", "2000-02", "2000-04", "2000-05", "2000-06"], freq="M"))
    run, _ = longest_run(sentinel)
    assert list(run.index.astype(str)) == ["2000-04", "2000-05", "2000-06"]
    checks["calendar_gap_sentinel_passed"] = True
    for context, period, freq, horizon in CONTEXTS:
        source = INPUT / f"{context}.csv"
        checks["input_hashes"][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        frame = pd.read_csv(source, dtype={"sector_id": str})
        frame[period] = pd.PeriodIndex(frame[period], freq=freq)
        frame = frame.sort_values(["sector_id", period])
        assert not frame.duplicated(["sector_id", period]).any()
        for variable in WEATHER:
            if not frame.groupby(period)[variable].nunique(dropna=False).eq(1).all():
                raise ValueError(f"{context}: weather differs across sectors: {variable}")
        full_windows.append(full_control_audit(frame, context, period, freq))
        comparison_path = ROOT / f"outputs/outcomes/{context}/weather/common_sample.csv"
        checks["input_hashes"][str(comparison_path.relative_to(ROOT))] = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
        comparison = pd.read_csv(comparison_path, dtype={"sector_id": str})
        comparison[period] = pd.PeriodIndex(comparison[period], freq=freq)
        assert not comparison.duplicated(["sector_id", period]).any()
        assert set(comparison.sector_id) == set(frame.sector_id)
        dates = pd.PeriodIndex(comparison[period].unique(), freq=freq).sort_values()
        assert np.all(np.diff(dates.asi8) == 1), f"Long comparison sample has calendar gaps: {context}"
        assert comparison.groupby("sector_id")[period].nunique().eq(len(dates)).all()
        original_columns = list(frame)
        compared_variables = list(OUTCOMES)+list(WEATHER)
        matched = frame.merge(comparison[["sector_id", period]+compared_variables],
            on=["sector_id", period], suffixes=("", "_comparison"), validate="one_to_one")
        assert len(matched) == len(comparison)
        comparison_error = max(float((matched[c]-matched[f"{c}_comparison"]).abs().max()) for c in compared_variables)
        assert comparison_error < 1e-9, (context, comparison_error)
        frame = matched[original_columns].sort_values(["sector_id", period])
        checks["comparison_samples"][context] = {"start": str(dates.min()), "end": str(dates.max()),
            "n_periods": len(dates), "n_sectors": int(frame.sector_id.nunique()),
            "max_abs_value_difference_from_comparison_sample": comparison_error}
        for sector_id, group in frame.groupby("sector_id", sort=False):
            group = group.set_index(period)
            assert np.all(np.diff(group.index.asi8) == 1), f"Gap in cached outcomes: {context}/{sector_id}"
            errors = {"log_level": float((group.y_log_level-100*np.log(group.y_raw_level)).abs().max()),
                      "one_period_growth": float((group.y_short-group.y_log_level.diff()).abs().max()),
                      "yoy_growth": float((group.y_yoy-group.y_log_level.diff(horizon)).abs().max()),
                      "yoy_growth_nsa": float((group.y_yoy_nsa-(100*np.log(group.output_nsa)).diff(horizon)).abs().max())}
            assert max(errors.values()) < 1e-9, (context, sector_id, errors)
            checks["outcome_transform_max_abs_errors"][f"{context}/{sector_id}"] = errors
            for variable, (label, deterministic) in OUTCOMES.items():
                metadata = {"context": context, "frequency": freq, "kind": "output", "sector_id": sector_id,
                            "sector": group.sector.iloc[0], "variable": variable, "description": label}
                rows = diagnose_series(group[variable], metadata, deterministic, horizon, False)
                for target, addition in zip([tests, summaries, observations, audits], rows):
                    target.extend(addition)
        unique = frame.groupby(period, as_index=True).first()
        for variable, (label, deterministic) in WEATHER.items():
            metadata = {"context": context, "frequency": freq, "kind": "weather", "sector_id": "common",
                        "sector": "Common national weather", "variable": variable, "description": label}
            rows = diagnose_series(unique[variable], metadata, deterministic, horizon, True)
            for target, addition in zip([tests, summaries, observations, audits], rows):
                target.extend(addition)
    test_frame, summary = pd.DataFrame(tests), pd.DataFrame(summaries)
    if not test_frame.status.eq("ok").all():
        raise RuntimeError("Unexpected unavailable tests in long cached histories; inspect diagnostics before publishing")
    assert test_frame.used_lags.between(0, test_frame.requested_lag_cap).all()
    assert test_frame.p_value.between(0, 1).all()
    assert np.isfinite(test_frame.statistic).all()
    checks["independent_statistic_checks"] = verify_statistics(test_frame, summary, observations)
    test_frame.to_csv(OUT / "tests.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    summary[summary.kind.eq("output")].to_csv(OUT / "outcome_summary.csv", index=False)
    summary[summary.kind.eq("weather")].to_csv(OUT / "weather_summary.csv", index=False)
    pd.DataFrame(observations).to_csv(OUT / "series.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "sample_audit.csv", index=False)
    pd.DataFrame(full_windows).to_csv(OUT / "full_control_windows.csv", index=False)
    primary = summary[summary.window.eq("long") & summary.specification.eq("primary")]
    counts = primary.groupby(["context", "kind", "variable", "evidence_5pct"]).size().rename("n_series").reset_index()
    counts.to_csv(OUT / "evidence_counts.csv", index=False)
    report = {"alpha": .05, "adf_lags": "AIC selects 0…12 monthly or 0…4 quarterly lags",
        "kpss_lags": "Hobijn automatic bandwidth, capped at 12 monthly / 4 quarterly; uncapped selection retained",
        "primary_sample": "Exact long weather-control outcome comparison dates for every variable; contiguous calendar verified, without added-control eligibility filtering",
        "sensitivity_sample": "Longest contiguous pre-2020 run; temperature also tested with a constant instead of trend",
        "primary_evidence_counts": counts.to_dict("records"), "full_control_windows": full_windows,
        "primary_weather": primary[primary.kind.eq("weather")].to_dict("records"), "cautions": CAUTIONS,
        "software": checks["software"], "n_test_rows": len(test_frame), "n_paired_rows": len(summary)}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    checks.update(n_test_rows=len(test_frame), n_paired_rows=len(summary),
                  all_tests_completed=True, all_test_dates_contiguous=True,
                  full_control_subset_not_tested=True,
                  kpss_censored_p_values=int(test_frame.p_value_relation.ne("=").sum()),
                  kpss_automatic_bandwidths_capped=int((test_frame.test.eq("KPSS") & test_frame.lag_cap_binding).sum()))
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2, allow_nan=False)+"\n")
    print(f"Diagnostics: {len(test_frame)} tests / {len(summary)} paired rows, all on contiguous histories", flush=True)
    print(counts.to_string(index=False), flush=True)
    return report


if __name__ == "__main__":
    run_diagnostics()
