"""Independently verify marginal stationarity evidence and exact-model coverage.

No ADF/KPSS test routine or stationarity-audit implementation is imported.
OLS, Bartlett long-run variances, bandwidth selection and evidence states are
rebuilt here. Statsmodels supplies only MacKinnon reference distributions.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stationarity"
YVARS = {"y_short", "y_yoy", 'y_yoy_nsa', "y_log_level", "y_raw_level", "y_yoy_pct", 'y_yoy_pct_nsa', "y_short_nsa"}
ALIASES = {"g_va_sa_qoq": "y_short", "g_va_sa_yoy": "y_yoy", "g_va_nsa_qoq": "y_short_nsa",
           "g_va_nsa_yoy":"y_yoy_nsa", "g_iip_sa": "y_short", "g_iip_nsa": "y_short_nsa"}
SUPPORT = {"stationarity-compatible", "trend-stationarity-compatible"}
BASE = ["temp_c", "rain_100mm", "rh_pct"]
LABELS = {"stationarity-compatible": "Stationarity-compatible", "trend-stationarity-compatible": "Trend-stationarity-compatible",
          "unit-root-concern": "Unit-root concern", "mixed": "Mixed evidence", "inconclusive": "Inconclusive",
          "sensitivity-dependent": "Lag-sensitive evidence", "unavailable-gaps": "Not established · calendar gaps",
          "unavailable-short": "Not established · short history", "unavailable": "Tests unavailable"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(actual, expected, label: str, atol=2e-7, rtol=2e-8) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), (label, a, b)
    finite = np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.


def variable_name(value: str) -> str:
    match = re.fullmatch(r"temp_l(\d+)", value)
    return "temperature_lag_"+match.group(1) if match else ALIASES.get(value, value)


def context_for(path: Path) -> str:
    name = str(path.relative_to(ROOT / "outputs"))
    return "iip-monthly" if "iip-monthly" in name or name.startswith("iip/monthly/") else "iip-quarterly" if "iip-quarterly" in name or name.startswith("iip/quarterly/") else "va"


def raw_levels(name: str, frequency: str) -> pd.DataFrame:
    payload = json.loads((ROOT / f"data/raw/{name}_tabledata.json").read_text())["Data"]
    rows = []
    for row in payload["row"]:
        for cell in row["columns"]:
            label = cell["key"]
            if frequency == "Q":
                year, quarter = re.fullmatch(r"(\d{4}) ([1-4])Q", label).groups()
                period = pd.Period(f"{year}Q{quarter}", freq="Q")
            else:
                period = pd.Period(pd.to_datetime(label, format="%Y %b"), freq="M")
            text = cell["value"].replace(",", "")
            rows.append({"sector_id": str(row["seriesNo"]), "period": period,
                         name: float(text) if text.lower() not in {"na", "-", ""} else np.nan})
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["sector_id", "period"]).any()
    return frame


def independent_weather() -> dict[str, pd.DataFrame]:
    frames = []
    for name, col, new in [("temperature", "mean_temp", "temp_c"), ("humidity", "mean_rh", "rh_pct"), ("rainfall", "total_rainfall", "rain_mm")]:
        frame = pd.read_csv(ROOT / f"data/raw/{name}_monthly.csv")
        frame["period"] = pd.PeriodIndex(frame.month, freq="M")
        frames.append(frame.set_index("period")[[col]].rename(columns={col: new}))
    monthly = pd.concat(frames, axis=1).sort_index()
    monthly["rain_100mm"] = monthly.rain_mm/100
    quarters = []
    for quarter, group in monthly.groupby(monthly.index.asfreq("Q")):
        if len(group) != 3 or group[["temp_c", "rh_pct", "rain_mm"]].isna().any().any():
            continue
        days = np.asarray(group.index.days_in_month, dtype=float)
        quarters.append({"period": quarter, "temp_c": math.fsum(group.temp_c*days)/days.sum(),
                         "rh_pct": math.fsum(group.rh_pct*days)/days.sum(), "rain_100mm": math.fsum(group.rain_mm)/100,
                         "temp_c_unweighted": math.fsum(group.temp_c)/3})
    return {"M": monthly.drop(columns="rain_mm"), "Q": pd.DataFrame(quarters).set_index("period")}


def canonical_sources() -> dict:
    weather_sources = independent_weather()
    result = {}
    for context in ["va", "iip-monthly", "iip-quarterly"]:
        frequency, horizon = ("M", 12) if context == "iip-monthly" else ("Q", 4)
        names = ("va_sa", "va_nsa") if context == "va" else ("iip_sa", "iip_nsa")
        original_frequency = "Q" if context == "va" else "M"
        frame = raw_levels(names[0], original_frequency).merge(raw_levels(names[1], original_frequency), on=["sector_id", "period"], validate="one_to_one")
        if context == "iip-quarterly":
            frame["period"] = frame.period.dt.asfreq("Q")
            frame = frame.groupby(["sector_id", "period"], as_index=False).agg(**{
                names[0]: (names[0], "mean"), names[1]: (names[1], "mean"), "months": (names[0], "size")})
            frame = frame[frame.months.eq(3)].copy()
        frame = frame.sort_values(["sector_id", "period"])
        frame["y_raw_level"] = frame[names[0]]
        frame["y_log_level"] = 100*np.log(frame[names[0]])
        grouped = frame.groupby("sector_id", sort=False)
        frame["y_short"] = frame.y_log_level-grouped.y_log_level.shift(1)
        frame["y_yoy"] = frame.y_log_level-grouped.y_log_level.shift(horizon)
        frame["y_yoy_pct"] = 100*(frame[names[0]]/grouped[names[0]].shift(horizon)-1)
        frame["nsa_log"] = 100*np.log(frame[names[1]])
        frame["y_short_nsa"] = frame.nsa_log-frame.groupby("sector_id").nsa_log.shift(1)
        frame['y_yoy_nsa']=frame.nsa_log-frame.groupby('sector_id').nsa_log.shift(horizon)
        frame['y_yoy_pct_nsa']=100*(frame[names[1]]/grouped[names[1]].shift(horizon)-1)
        weather = weather_sources[frequency].copy()
        for col in BASE:
            stem = "temp" if col == "temp_c" else col
            weather[stem+"_year_lag"] = weather[col].shift(horizon)
            weather[stem+"_year_change"] = weather[col]-weather[col].shift(horizon)
            weather[stem+"_year_mean"] = weather[col].rolling(horizon, min_periods=horizon).mean()
        for lag in range(1, horizon+1):
            weather[f"temperature_lag_{lag}"] = weather.temp_c.shift(lag)
        control_path = ROOT / f'data/processed/extended_controls/{"monthly" if frequency=="M" else "quarterly"}_controls.csv'
        controls = pd.read_csv(control_path)
        period = "month" if frequency == "M" else "quarter"
        controls["period"] = pd.PeriodIndex(controls[period], freq=frequency)
        weather = weather.merge(controls.set_index("period")[["wind_mps", "pm25_10ug", "wind_coverage", "pm_coverage"]], left_index=True, right_index=True, how="left", validate="one_to_one")
        frame = frame[frame.period.isin(weather.index)].copy()
        result[context] = {"outcomes": frame.set_index(["sector_id", "period"]), "weather": weather.loc[weather.index.isin(frame.period.unique())],
                           "frequency": frequency, "horizon": horizon, "period_column": period}
    return result


def resolve_sample(path: Path, model: str) -> tuple[pd.DataFrame, Path]:
    saved = path.parent / "samples" / f"{model}.csv"
    parts = path.relative_to(ROOT / "outputs").parts
    if saved.exists():
        source = saved
    elif parts[0] == "outcomes":
        source = path.parent / "common_sample.csv"
    elif parts[0] == "extended_controls":
        context = parts[1]
        source = (ROOT / "data/processed/panel_broad.csv" if context == "va" else ROOT / f'data/processed/iip/panel_{"monthly" if context=="iip-monthly" else "quarterly"}_broad.csv') if model == "C0" else path.parent / ("sensitivity_80_sample.csv" if model == "C6" else "baseline_sample.csv")
    else:
        frequency = "monthly" if len(parts)>1 and parts[1] == "monthly" else "quarterly"
        source = ROOT / "data/processed" / (f"iip/panel_{frequency}_{'detailed' if model=='S7' else 'broad'}.csv" if parts[0] == "iip" else f"panel_{'detailed' if model=='S7' else 'broad'}.csv")
    frame = pd.read_csv(source, dtype={"sector_id": str})
    period = "month" if context_for(path) == "iip-monthly" else "quarter"
    if not saved.exists():
        if parts[0] == "outcomes" and model.endswith("X") or parts[0] == "extended_controls" and model == "C5" or model == "S5":
            frame = frame[~frame.year.between(2020, 2022)].copy()
        if model == "S4":
            frame = frame[frame.year.lt(2020)].copy()
        if model == "S6":
            detailed_path = ROOT / (f"data/processed/iip/panel_{'monthly' if period=='month' else 'quarterly'}_detailed.csv" if parts[0] == "iip" else "data/processed/panel_detailed.csv")
            detailed = pd.read_csv(detailed_path)
            frame = frame[frame[period].between(detailed[period].min(), detailed[period].max())].copy()
        if path.name == "distributed_lag_results.csv":
            h = 12 if period == "month" else 4
            frame = frame.dropna(subset=["temp_c", *[f"temp_l{i}" for i in range(1, h+1)]])
    return frame, source


def formula_variables(formula: str, known: set[str]) -> tuple[str, list[str]]:
    lhs, rhs = formula.split("~", 1)
    candidates = set(re.findall(r"\b[a-zA-Z_]\w*\b", rhs))
    mapped = {variable_name(v) for v in candidates}
    x = sorted(mapped&known-YVARS)
    return variable_name(lhs.strip()), x


def expected_inventory(sources: dict) -> list[dict]:
    inventory = []
    for path in sorted([*(ROOT / "outputs").rglob("regression_results.csv"), *(ROOT / "outputs").rglob("distributed_lag_results.csv")]):
        if "visitor_arrivals" in path.parts:
            continue  # visitor source/schema is verified separately
        results = pd.read_csv(path, dtype={"sector_id": str})
        if "formula" not in results:
            continue
        context = context_for(path)
        known = set(sources[context]["weather"].columns)
        for model, rows in results.groupby("model", sort=False):
            assert rows.formula.nunique() == 1
            y, x = formula_variables(rows.formula.iloc[0], known)
            sample, source = resolve_sample(path, model)
            assert len(sample) == int(rows.n_obs.iloc[0]), (path, model, len(sample), rows.n_obs.iloc[0])
            inventory.append({"path": path, "model": model, "outcome": y, "x": x, "sample": sample, "source": source, "family": "pooled"})
    for frequency in ["monthly", "quarterly"]:
        path = ROOT / f"outputs/iip/{frequency}/aggregate_results.csv"
        sample_source = ROOT / f"data/processed/iip/aggregate_{frequency}.csv"
        sample = pd.read_csv(sample_source, dtype={"sector_id": str})
        baseline = pd.read_csv(path.parent / "baseline_sample.csv")
        period = "month" if frequency == "monthly" else "quarter"
        sample = sample[sample[period].between(baseline[period].min(), baseline[period].max())]
        inventory.append({"path": path, "model": "A1", "outcome": "y_short", "x": BASE, "sample": sample, "source": sample_source, "family": "national_IIP_original"})
    for path in sorted((ROOT / "outputs/heterogeneity").rglob("sector_slopes.csv")):
        results = pd.read_csv(path)
        sample_source = path.parent / "common_sample.csv"
        sample = pd.read_csv(sample_source, dtype={"sector_id": str})
        x = BASE+(["wind_mps", "pm25_10ug"] if path.parent.name == "full" else [])
        for (model, y), _ in results.groupby(["model", "outcome"], sort=False):
            selected = sample[~sample.year.between(2020, 2022)] if model == "H3X" else sample
            inventory.append({"path": path, "model": model, "outcome": y, "x": x, "sample": selected, "source": sample_source,
                              "family": "sector_interactions_and_equivalent_separate_series"})
    for path in sorted((ROOT / "outputs/granular_interactions").glob("*/*/sector_slopes.csv")):
        results = pd.read_csv(path, dtype={"sector_id": str})
        source = path.parent / "common_sample.csv"
        sample = pd.read_csv(source, dtype={"sector_id": str})
        assert set(results.model) == {"H1"}
        assert set(results.outcome) == {"y_short", "y_yoy", 'y_yoy_nsa', "y_log_level", "y_raw_level"}
        x = BASE+([] if path.parent.name == "long" else ["wind_mps", "pm25_10ug"])
        for (model, outcome), rows in results.groupby(["model", "outcome"], sort=False):
            assert rows.formula.nunique() == 1
            lhs, rhs = formula_variables(rows.formula.iloc[0], set(sources[context_for(path)]["weather"].columns))
            assert lhs == outcome and set(rhs) == set(x)
            inventory.append({"path": path, "model": model, "outcome": outcome, "x": x, "sample": sample,
                              "source": source, "family": "detailed_sector_FE_interactions_main"})
    for path in sorted((ROOT / "outputs/granular").rglob("separate_sector_coefficients.csv")):
        results = pd.read_csv(path, dtype={"sector_id": str})
        for model, rows in results.groupby("model", sort=False):
            source = path.parent / "samples" / f"{model}.csv"
            sample = pd.read_csv(source, dtype={"sector_id": str})
            assert set(sample.sector_id) == set(rows.sector_id)
            inventory.append({"path": path, "model": model, "outcome": rows.outcome.iloc[0],
                              "x": BASE+([] if path.parent.name == "long" else ["wind_mps", "pm25_10ug"]),
                              "sample": sample, "source": source, "family": "individual_sector_time_series"})
    return inventory


def verify_equivalent_files(coverage: pd.DataFrame, sources: dict) -> dict:
    """Map additional saved fits to already audited identical marginal inputs."""
    original_fits, separate_fits, checked_files = 0, 0, []
    for prefix in [ROOT / "outputs", ROOT / "outputs/iip/monthly", ROOT / "outputs/iip/quarterly"]:
        path = prefix / "sector_slopes.csv"
        rows = pd.read_csv(path, dtype={"sector_id": str})
        baseline = prefix / "regression_results.csv"
        context = context_for(path)
        known = set(sources[context]["weather"].columns)
        assert rows.formula.nunique() == 1
        outcome, variables = formula_variables(rows.formula.iloc[0], known)
        links = coverage[coverage.source_csv.eq(str(baseline.relative_to(ROOT)))&coverage.model.eq("S3")&coverage.outcome.eq(outcome)]
        assert set(rows.sector_id) == set(links.sector_id)
        period = sources[context]["period_column"]
        sample, _ = resolve_sample(baseline, "S3")
        for sector, group in rows.groupby("sector_id", sort=False):
            covered = links[links.sector_id.eq(sector)]
            assert set(covered.variable) == {outcome, *variables}
            dates = sample[sample.sector_id.eq(sector)][period]
            assert group.start.eq(dates.min()).all() and group.end.eq(dates.max()).all()
            assert group.n_obs.eq(len(sample)).all()
            original_fits += 1
        checked_files.append(str(path.relative_to(ROOT)))
    for path in sorted((ROOT / "outputs/heterogeneity").rglob("separate_sector_coefficients.csv")):
        coefficients = pd.read_csv(path, dtype={"sector_id": str})
        series_path = path.parent / "separate_series.csv"
        separate_series = pd.read_csv(series_path, dtype={"sector_id": str})
        pooled_path = path.parent / "sector_slopes.csv"
        pooled = pd.read_csv(pooled_path, dtype={"sector_id": str})
        sample = pd.read_csv(path.parent / "common_sample.csv", dtype={"sector_id": str})
        context = context_for(path)
        variables = set(BASE+(["wind_mps", "pm25_10ug"] if path.parent.name=="full" else []))
        links = coverage[coverage.source_csv.eq(str(pooled_path.relative_to(ROOT)))&coverage.model.eq("H3")]
        expected_pairs = set(links[["sector_id", "outcome"]].itertuples(index=False, name=None))
        assert set(coefficients[["sector_id", "outcome"]].itertuples(index=False, name=None)) == expected_pairs
        assert set(separate_series[["sector_id", "outcome"]].itertuples(index=False, name=None)) == expected_pairs
        for (sector, outcome), group in coefficients.groupby(["sector_id", "outcome"], sort=False):
            covered = links[links.sector_id.eq(sector)&links.outcome.eq(outcome)]
            assert set(covered.variable) == {outcome, *variables}
            assert set(group.term) == variables
            count = int(sample.sector_id.eq(sector).sum())
            assert group.n_obs.eq(count).all()
            matched = separate_series[separate_series.sector_id.eq(sector)&separate_series.outcome.eq(outcome)]
            assert matched.n_obs.eq(count).all()
            benchmark = pooled[pooled.model.eq("H3")&pooled.sector_id.eq(sector)&pooled.outcome.eq(outcome)]
            for row in matched.itertuples():
                coefficient = group[group.term.eq("temp_c")&group.hac_lags.eq(row.hac_lags)].coefficient
                assert len(coefficient) == 1
                assert_close(row.coefficient, coefficient.iloc[0], "Separate summary versus coefficient", atol=1e-10)
                other = benchmark[benchmark.hac_lags.eq(row.hac_lags)].coefficient
                assert len(other) == 1
                assert_close(row.coefficient, other.iloc[0], "Separate fit versus H3 point-estimate equivalence", atol=2e-7)
            separate_fits += 1
        checked_files += [str(path.relative_to(ROOT)), str(series_path.relative_to(ROOT))]
    assert original_fits == 23 and separate_fits == 2*4*23, (original_fits, separate_fits)
    return {"original_sector_slopes_fits_linked_to_identical_S3_series": original_fits,
            "heterogeneity_separate_fits_linked_to_identical_H3_series": separate_fits,
            "additional_result_files_checked": checked_files}


def ols(response: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    scale = np.sqrt(np.mean(design**2, axis=0))
    scaled = design/scale
    q, triangular = np.linalg.qr(scaled, mode="reduced")
    beta = np.linalg.solve(triangular, q.T@response)
    error = response-scaled@beta
    ssr = float(error@error)
    inverse = np.linalg.solve(triangular, np.eye(design.shape[1]))
    covariance = inverse@inverse.T*(ssr/(len(response)-design.shape[1]))
    return beta/scale, covariance/np.outer(scale, scale), ssr


def adf_design(values: np.ndarray, lags: int, deterministic: str, trim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    start = lags if trim is None else trim
    diff = np.diff(values)
    response = diff[start:]
    columns = [values[start:-1], *[diff[start-j:-j] for j in range(1, lags+1)], np.ones(len(response))]
    if deterministic == "ct":
        columns.append(np.arange(1., len(response)+1))
    return response, np.column_stack(columns)


def manual_adf(values: np.ndarray, lags: int, deterministic: str, cap: int) -> dict:
    y, x = adf_design(values, lags, deterministic)
    beta, covariance, _ = ols(y, x)
    statistic = float(beta[0]/np.sqrt(covariance[0, 0]))
    aics = []
    for k in range(cap+1):
        yy, xx = adf_design(values, k, deterministic, trim=cap)
        _, _, ssr = ols(yy, xx)
        n = len(yy)
        aics.append(n*(math.log(2*math.pi)+1+math.log(ssr/n))+2*xx.shape[1])
    return {"statistic": statistic, "p_value": float(mackinnonp(statistic, regression=deterministic, N=1)),
            "critical_5pct": float(mackinnoncrit(N=1, regression=deterministic, nobs=len(y))[1]),
            "nobs": len(y), "aic_lag": int(np.argmin(aics)), "aic": min(aics)}


def manual_kpss(values: np.ndarray, lags: int, deterministic: str) -> dict:
    n = len(values)
    x = np.ones((n, 1)) if deterministic == "c" else np.column_stack([np.ones(n), np.arange(1., n+1)])
    beta = np.linalg.lstsq(x, values, rcond=None)[0]
    residual = values-x@beta
    eta = float(np.cumsum(residual)@np.cumsum(residual)/n**2)
    long_variance = float((residual@residual+2*math.fsum((1-j/(lags+1))*float(residual[j:]@residual[:-j]) for j in range(1, lags+1)))/n)
    statistic = eta/long_variance
    critical = np.array([.347, .463, .574, .739]) if deterministic == "c" else np.array([.119, .146, .176, .216])
    p = float(np.interp(statistic, critical, [.10, .05, .025, .01]))
    # Hobijn automatic Bartlett bandwidth, computed without the test routine.
    covlags = int(n**(2/9))
    products = [2*float(residual[j:]@residual[:-j])/n for j in range(1, covlags+1)]
    s0 = float(residual@residual)/n+math.fsum(products)
    s1 = math.fsum(j*value for j, value in enumerate(products, 1))
    auto = min(int(1.1447*((s1/s0)**2)**(1/3)*n**(1/3)), n-1)
    return {"statistic": statistic, "p_value": p, "critical_5pct": float(critical[1]), "auto_lags": auto,
            "p_relation": "<" if statistic>critical[-1] else ">" if statistic<critical[0] else "="}


def evidence_state(adf_p: float, kpss_p: float, deterministic: str) -> str:
    reject_adf, reject_kpss = adf_p < .05, kpss_p < .05
    if reject_adf and not reject_kpss:
        return "trend-stationarity-compatible" if deterministic == "ct" else "stationarity-compatible"
    if not reject_adf and reject_kpss:
        return "unit-root-concern"
    return "mixed" if reject_adf else "inconclusive"


def independently_adjust(values: np.ndarray, periods: pd.PeriodIndex) -> tuple[np.ndarray, np.ndarray]:
    seasons = periods.month if periods.freqstr.startswith("M") else periods.quarter
    dummy = np.column_stack([(seasons==s).astype(float) for s in sorted(set(seasons))[1:]])
    design = np.column_stack([np.ones(len(values)), np.arange(len(values)), dummy])
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    seasonal = dummy@beta[2:]
    seasonal -= seasonal.mean()
    return values-seasonal, seasonal


def significance_boundaries() -> int:
    from significance import significance_class, significance_label, significance_stars
    cases = [(None, "Unavailable", "", "unavailable"), (np.nan, "Unavailable", "", "unavailable"),
             (-.1, "Unavailable", "", "unavailable"), (1.01, "Unavailable", "", "unavailable"),
             (0., "Significant at 1%", "***", "sig-1"),
             (np.nextafter(.01, 0), "Significant at 1%", "***", "sig-1"),
             (.01, "Significant at 5%", "**", "sig-5"),
             (np.nextafter(.05, 0), "Significant at 5%", "**", "sig-5"),
             (.05, "10% only", "*", "sig-10"), (np.nextafter(.10, 0), "10% only", "*", "sig-10"),
             (.10, "Not significant at 10%", "", "not-sig"), (1., "Not significant at 10%", "", "not-sig")]
    for p, label, stars, cls in cases:
        assert (significance_label(p), significance_stars(p), significance_class(p)) == (label, stars, cls)
    return len(cases)


def verify_stationarity_audit() -> dict:
    paths = [OUT / f"{n}.csv" for n in ["diagnostics", "tests", "series", "model_variable_coverage", "model_assessments"]]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    diagnostics, tests, series, coverage, assessments = [pd.read_csv(p, dtype={"sector_id": str}) for p in paths]
    summary = json.loads((OUT / "summary.json").read_text())
    assert len(diagnostics) == summary["diagnostics"] and len(tests) == summary["test_rows"]
    assert len(assessments) == summary["model_outcomes"] and len(coverage) == summary["model_variable_links"]
    assert not diagnostics.diagnostic_id.duplicated().any()
    assert not tests.duplicated(["diagnostic_id", "specification", "test"]).any()
    assert not series.duplicated(["diagnostic_id", "period"]).any()
    assert not coverage.duplicated(["source_csv", "model", "outcome", "sector_id", "variable", "role"]).any()
    diag = diagnostics.set_index("diagnostic_id")
    grouped_series = {key: group for key, group in series.groupby("diagnostic_id", sort=False)}
    sources = canonical_sources()
    inventory = expected_inventory(sources)
    expected_links, expected_models, exact_dates, source_hashes = set(), set(), {}, {}
    raw_checks, raw_max = 0, 0.
    granular_fits = 0
    for item in inventory:
        path, model, outcome, sample = item["path"], item["model"], item["outcome"], item["sample"].copy()
        relative = str(path.relative_to(ROOT))
        context = context_for(path)
        src = sources[context]
        period, freq = src["period_column"], src["frequency"]
        sample[period] = pd.PeriodIndex(sample[period], freq=freq)
        assert not sample.duplicated(["sector_id", period]).any()
        selected = coverage[coverage.source_csv.eq(relative)&coverage.model.eq(model)&coverage.outcome.eq(outcome)]
        assert not selected.empty, (relative, model, outcome)
        assert set(selected.family) == {item["family"]}
        expected_models.add((relative, model, outcome))
        source_hashes[str(item["source"].relative_to(ROOT))] = sha256(item["source"])
        for sector, group in sample.groupby("sector_id", sort=False):
            dates = pd.PeriodIndex(group[period], freq=freq).sort_values()
            variables = [outcome, *item["x"]]
            if item["family"] == "individual_sector_time_series":
                granular_fits += 1
            for variable in variables:
                role = "Y" if variable == outcome else "X"
                expected_links.add((relative, model, outcome, sector, variable, role))
                link = selected[selected.sector_id.eq(sector)&selected.variable.eq(variable)]
                assert len(link) == 1, (relative, model, sector, variable, len(link))
                diagnostic_id = link.diagnostic_id.iloc[0]
                d = diag.loc[diagnostic_id]
                assert d.scope == "exact_estimation_window", (diagnostic_id, d.scope)
                assert d.context == context and d.variable == variable
                assert str(d.sector_id) == ("common" if variable not in YVARS else sector)
                assert d.start == str(dates.min()) and d.end == str(dates.max()) and d.n_periods == len(dates)
                old_dates = exact_dates.setdefault(diagnostic_id, dates)
                assert np.array_equal(old_dates.asi8, dates.asi8), diagnostic_id
                assert link.exact_state.iloc[0] == d.state
                background = diag.loc[link.background_id.iloc[0]]
                assert background.scope == "available_history" and link.background_state.iloc[0] == background.state
                independent = (src["outcomes"].loc[sector][variable] if variable in YVARS else src["weather"][variable]).reindex(dates).to_numpy()
                sample_column = next((c for c in group.columns if variable_name(c) == variable), None)
                assert sample_column is not None, (relative, model, variable)
                observed = group.set_index(period).loc[dates, sample_column].to_numpy()
                raw_max = max(raw_max, assert_close(observed, independent, f"Exact sample source {relative}/{model}/{sector}/{variable}", atol=3e-9, rtol=3e-12))
                raw_checks += len(dates)
        model_row = assessments[assessments.source_csv.eq(relative)&assessments.model.eq(model)&assessments.outcome.eq(outcome)]
        assert len(model_row) == 1
        assert int(model_row.n_units.iloc[0]) == sample.sector_id.nunique()
        assert int(model_row.variable_checks.iloc[0]) == len(selected)
        all_support = selected.groupby("sector_id").exact_state.apply(lambda values: set(values)<=SUPPORT)
        y_support = selected[selected.role.eq("Y")].exact_state.isin(SUPPORT).sum()
        assert int(model_row.compatible_all_variables_units.iloc[0]) == int(all_support.sum())
        assert int(model_row.compatible_y_units.iloc[0]) == int(y_support)
        gaps = any(len(g) != g[period].max().ordinal-g[period].min().ordinal+1 for _, g in sample.groupby("sector_id"))
        assert bool(model_row.any_calendar_gaps.iloc[0]) == gaps
        expected_assessment = "Screen-compatible on this window" if all_support.all() else "Stationarity not established for every series"
        assert model_row.assessment.iloc[0] == expected_assessment
        expected_role = "Descriptive levels" if outcome in {"y_log_level", "y_raw_level"} else "Growth · stationarity screen compatible" if all_support.all() else "Exploratory growth · stationarity caution"
        assert model_row.analysis_role.iloc[0] == expected_role
    actual_links = set(coverage[["source_csv", "model", "outcome", "sector_id", "variable", "role"]].itertuples(index=False, name=None))
    assert actual_links == expected_links, ("Coverage mismatch", list(expected_links-actual_links)[:10], list(actual_links-expected_links)[:10])
    assert set(assessments[["source_csv", "model", "outcome"]].itertuples(index=False, name=None)) == expected_models
    assert granular_fits == 3*5*(19+18+18), granular_fits
    granular_contexts = {"va-detailed", "va-broad-matched", "iip-monthly-total", "iip-monthly-detailed", "iip-monthly-broad-matched",
                         "iip-quarterly-total", "iip-quarterly-detailed", "iip-quarterly-broad-matched"}
    actual_granular = {(item["path"].parent.parent.name, item["path"].parent.name, item["model"]) for item in inventory
                      if item["path"].name=="regression_results.csv" and "granular" in item["path"].parts}
    assert actual_granular == {(context, window, model) for context in granular_contexts for window in ["long", "full", "exclude"] for model in ["G1", "G2", 'G2N', "L1", "L3"]}
    interaction_groups = [item for item in inventory if item["family"] == "detailed_sector_FE_interactions_main"]
    assert len(interaction_groups) == 3*3*5
    assert {(item["path"].parent.parent.name, item["path"].parent.name, item["outcome"]) for item in interaction_groups} == {
        (context, window, outcome) for context in ["va-detailed", "iip-monthly-detailed", "iip-quarterly-detailed"]
        for window in ["long", "full", "exclude"] for outcome in ["y_short", "y_yoy", 'y_yoy_nsa', "y_log_level", "y_raw_level"]}
    for path in sorted((ROOT / "outputs/aggregate_va").rglob("specifications.json")):
        definitions = json.loads(path.read_text())
        expected = {(str((path.parent / "regression_results.csv").relative_to(ROOT)), row["model"], variable_name(row["outcome"])) for row in definitions}
        assert expected <= expected_models
    equivalences = verify_equivalent_files(coverage, sources)
    for relative, value in summary["coefficient_inputs_sha256"].items():
        assert sha256(ROOT / relative) == value
    # Verify each distinct series/window directly against raw source histories.
    adjustment_max = 0.
    eligible_ids = set()
    for d in diagnostics.itertuples():
        src = sources[d.context]
        frequency, h = src["frequency"], src["horizon"]
        original = src["outcomes"].loc[d.sector_id][d.variable] if d.kind == "output" else src["weather"][d.variable]
        if d.scope == "exact_estimation_window":
            dates = exact_dates[d.diagnostic_id]
        else:
            dates = original.dropna().index.sort_values()
            if d.scope == "pre_2020_history":
                dates = dates[dates.year<2020]
            else:
                assert d.scope == "available_history"
        values = original.reindex(dates).copy()
        if d.scope != "exact_estimation_window" and d.variable in {"wind_mps", "pm25_10ug"}:
            q = src["weather"]["wind_coverage" if d.variable == "wind_mps" else "pm_coverage"].reindex(dates)
            values = values.where(q.ge(.90))
        gaps = dates.max().ordinal-dates.min().ordinal+1-len(dates)
        missing = int(values.isna().sum())
        assert d.start == str(dates.min()) and d.end == str(dates.max()) and d.n_periods == len(dates)
        assert d.calendar_gaps == gaps and d.missing_values == missing
        deterministic = "ct" if d.variable in {"y_log_level", "y_raw_level"} or d.kind == "weather" and "rain_" not in d.variable and "year_change" not in d.variable else "c"
        assert d.deterministic == deterministic
        if gaps or missing:
            assert d.state == "unavailable-gaps" and d.test_status == "not_run"
            assert d.diagnostic_id not in grouped_series and not tests.diagnostic_id.eq(d.diagnostic_id).any()
            continue
        if len(values)<5*h or np.ptp(values.to_numpy()) == 0:
            assert d.state == "unavailable-short" and d.test_status == "not_run"
            assert d.diagnostic_id not in grouped_series and not tests.diagnostic_id.eq(d.diagnostic_id).any()
            continue
        eligible_ids.add(d.diagnostic_id)
        saved = grouped_series[d.diagnostic_id]
        periods = pd.PeriodIndex(saved.period, freq=frequency)
        assert np.array_equal(periods.asi8, dates.asi8) and np.all(np.diff(periods.asi8)==1)
        assert_close(saved.raw_value.to_numpy(), values.to_numpy(), d.diagnostic_id+" raw source", atol=3e-9, rtol=3e-12)
        adjusted, seasonal = independently_adjust(values.to_numpy(), dates) if d.kind == "weather" or d.variable.endswith('_nsa') else (values.to_numpy(), np.zeros(len(values)))
        adjustment_max = max(adjustment_max, assert_close(saved.test_value.to_numpy(), adjusted, d.diagnostic_id+" adjusted", atol=3e-9, rtol=3e-12))
        assert_close(saved.seasonal_component.to_numpy(), seasonal, d.diagnostic_id+" seasonal offsets", atol=3e-9, rtol=3e-12)
        assert np.isfinite(saved[["raw_value", "test_value", "seasonal_component"]].to_numpy()).all()
    assert set(tests.diagnostic_id) == eligible_ids == set(grouped_series)
    assert len(tests) == 4*len(eligible_ids)
    maxima = {"adf_statistic": 0., "adf_p": 0., "adf_aic": 0., "kpss_statistic": 0., "kpss_p": 0.}
    adf_cache, kpss_cache = {}, {}
    for row in tests.itertuples():
        assert row.status == "ok", (row.diagnostic_id, row.test, row.status)
        d = diag.loc[row.diagnostic_id]
        assert d.calendar_gaps == 0 and d.missing_values == 0 and d.n_periods >= 5*d.horizon
        values = grouped_series[row.diagnostic_id].test_value.to_numpy()
        signature = hashlib.sha256(values.tobytes()).hexdigest()
        h = int(d.horizon)
        legacy = row.specification == "legacy_lag_sensitivity"
        if row.test == "ADF":
            cap = min(h if legacy else 2*h, len(values)//2-(2 if d.deterministic=="ct" else 1)-1)
            assert row.lag_limit == cap and row.lags <= cap and bool(row.lag_binding) == (row.lags == cap)
            key = (signature, int(row.lags), d.deterministic, cap)
            result = adf_cache.get(key)
            if result is None:
                result = manual_adf(values, int(row.lags), d.deterministic, cap);adf_cache[key] = result
            maxima["adf_statistic"] = max(maxima["adf_statistic"], assert_close(row.statistic, result["statistic"], "ADF statistic", atol=4e-7))
            maxima["adf_p"] = max(maxima["adf_p"], assert_close(row.p_value, result["p_value"], "ADF MacKinnon p", atol=2e-8))
            maxima["adf_aic"] = max(maxima["adf_aic"], assert_close(row.aic, result["aic"], "ADF common-sample AIC", atol=2e-7))
            assert int(row.lags) == result["aic_lag"] and int(row.nobs) == result["nobs"]
            assert_close(row.critical_5pct, result["critical_5pct"], "ADF critical value", atol=2e-10)
            assert row.p_relation == "="
        else:
            assert row.test == "KPSS"
            key = (signature, int(row.lags), d.deterministic)
            result = kpss_cache.get(key)
            if result is None:
                result = manual_kpss(values, int(row.lags), d.deterministic);kpss_cache[key] = result
            maxima["kpss_statistic"] = max(maxima["kpss_statistic"], assert_close(row.statistic, result["statistic"], "KPSS statistic", atol=2e-7))
            maxima["kpss_p"] = max(maxima["kpss_p"], assert_close(row.p_value, result["p_value"], "KPSS interpolated p", atol=2e-8))
            assert int(row.auto_lags) == result["auto_lags"]
            assert int(row.lags) == (min(h, result["auto_lags"]) if legacy else result["auto_lags"])
            assert row.p_relation == result["p_relation"]
            assert_close(row.critical_5pct, result["critical_5pct"], "KPSS critical value", atol=1e-12)
    for diagnostic_id in eligible_ids:
        d = diag.loc[diagnostic_id]
        states = []
        for specification, prefix in [("primary", "primary"), ("legacy_lag_sensitivity", "legacy")]:
            pair = tests[tests.diagnostic_id.eq(diagnostic_id)&tests.specification.eq(specification)].set_index("test")
            assert set(pair.index) == {"ADF", "KPSS"}
            state = evidence_state(float(pair.loc["ADF", "p_value"]), float(pair.loc["KPSS", "p_value"]), d.deterministic)
            assert state == d[prefix+"_state"]
            assert_close(d[prefix+"_adf_p"], pair.loc["ADF", "p_value"], "Diagnostic ADF mapping", atol=1e-12)
            assert_close(d[prefix+"_kpss_p"], pair.loc["KPSS", "p_value"], "Diagnostic KPSS mapping", atol=1e-12)
            assert d[prefix+"_kpss_relation"] == pair.loc["KPSS", "p_relation"]
            assert int(d[prefix+"_adf_lags"]) == int(pair.loc["ADF", "lags"])
            assert int(d[prefix+"_kpss_lags"]) == int(pair.loc["KPSS", "lags"])
            states.append(state)
        assert d.state == (states[0] if states[0] == states[1] else "sensitivity-dependent")
        assert d.test_status == "completed"
    assert (diagnostics.label == diagnostics.state.map(LABELS)).all()
    assert evidence_state(.05, .05, "c") == "inconclusive"
    assert evidence_state(np.nextafter(.05, 0), .05, "c") == "stationarity-compatible"
    assert evidence_state(.05, np.nextafter(.05, 0), "ct") == "unit-root-concern"
    assert before == {str(p.relative_to(ROOT)): sha256(p) for p in paths}, "Audit changed during verification; rerun verifier"
    report = {"status": "passed", "interpretation": "Verified evidence and coverage, not a stationarity certificate or a causal-validity assessment",
              "diagnostic_windows_checked": len(diagnostics), "test_statistics_checked": len(tests), "contiguous_finite_tested_series": len(eligible_ids),
              "unavailable_gap_or_short_windows_checked": len(diagnostics)-len(eligible_ids), "model_outcome_groups_checked": len(inventory),
              "model_variable_links_checked": len(expected_links), "granular_individual_sector_fits_checked": granular_fits,
              "detailed_sector_FE_interaction_model_outcomes_checked": len(interaction_groups),
              "exact_sample_raw_values_checked": raw_checks, "max_exact_sample_raw_value_absolute_error": raw_max,
              "max_seasonal_adjustment_absolute_error": adjustment_max, "numerical_max_absolute_errors": maxima,
              "lag_selection": "ADF QR OLS with common maximum-lag AIC sample; KPSS Bartlett variance and Hobijn automatic bandwidth independently rebuilt",
              "significance_boundary_cases_checked": significance_boundaries(), "additional_equivalent_fit_coverage": equivalences,
              "audit_inputs_sha256": before, "sample_inputs_sha256": source_hashes,
              "canonical_source_inputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
                  *[ROOT / f"data/raw/{name}_tabledata.json" for name in ["va_sa", "va_nsa", "iip_sa", "iip_nsa"]],
                  *[ROOT / f"data/raw/{name}_monthly.csv" for name in ["temperature", "rainfall", "humidity"]],
                  *[ROOT / f"data/processed/extended_controls/{freq}_controls.csv" for freq in ["monthly", "quarterly"]]]}}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Verified stationarity audit: {len(tests)} manual test calculations, {len(expected_links)} exact-model variable links and {granular_fits} granular individual fits.", flush=True)
    return report


if __name__ == "__main__":
    verify_stationarity_audit()
