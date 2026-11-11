"""Check outcome definitions and estimates using raw calendar lookups and NumPy OLS.

This verifier deliberately does not import the outcome-estimation implementation.
It saves its audit record and leaves existing estimates unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/outcomes"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
ADDED = ["wind_mps", "pm25_10ug"]
CONTEXTS = [("va", "quarter", "Q", 4), ("iip-monthly", "month", "M", 12),
            ("iip-quarterly", "quarter", "Q", 4)]


def close(actual, expected, label: str, *, atol: float = 2e-7,
          rtol: float = 2e-7) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def raw_output(context: str, adjustment: str = "sa") -> dict[tuple[str, pd.Period], float]:
    assert adjustment in {"sa", "nsa"}
    filename = ("va_" if context == "va" else "iip_")+adjustment
    data = json.loads((ROOT / f"data/raw/{filename}_tabledata.json").read_text())["Data"]
    values: dict[tuple[str, pd.Period], float] = {}
    for sector in data["row"]:
        for item in sector["columns"]:
            if context == "va":
                year, season = item["key"].split()
                period = pd.Period(f"{year}Q{season[0]}", freq="Q")
            else:
                period = pd.Period(pd.to_datetime(item["key"], format="%Y %b"), freq="M")
            key = (sector["seriesNo"], period)
            assert key not in values
            values[key] = float(str(item["value"]).replace(",", ""))
    if context != "iip-quarterly":
        return values
    quarters: dict[tuple[str, pd.Period], list[float]] = {}
    for (sector, month), value in values.items():
        quarters.setdefault((sector, month.asfreq("Q")), []).append(value)
    return {key: sum(vals)/3 for key, vals in quarters.items() if len(vals) == 3}


def raw_weather(freq: str) -> dict[pd.Period, dict[str, float]]:
    monthly: dict[pd.Period, dict[str, float]] = {}
    for filename, source_col, target_col, scale in [
        ("temperature", "mean_temp", "temp_c", 1),
        ("rainfall", "total_rainfall", "rain_100mm", .01),
        ("humidity", "mean_rh", "rh_pct", 1),
    ]:
        frame = pd.read_csv(ROOT / f"data/raw/{filename}_monthly.csv")
        for row in frame.to_dict("records"):
            month = pd.Period(row["month"], freq="M")
            monthly.setdefault(month, {})[target_col] = float(row[source_col])*scale
    monthly = {month: row for month, row in monthly.items() if set(row) == set(BASE)}
    if freq == "M":
        return monthly
    quarters = sorted({month.asfreq("Q") for month in monthly})
    result = {}
    for quarter in quarters:
        months = list(pd.period_range(quarter.asfreq("M", how="start"),
                                     quarter.asfreq("M", how="end"), freq="M"))
        if not all(month in monthly for month in months):
            continue
        days = np.array([month.days_in_month for month in months])
        result[quarter] = {
            "temp_c": float(np.average([monthly[m]["temp_c"] for m in months], weights=days)),
            "rh_pct": float(np.average([monthly[m]["rh_pct"] for m in months], weights=days)),
            "rain_100mm": sum(monthly[m]["rain_100mm"] for m in months),
        }
    return result


def read_sample(path: Path, period: str, freq: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sector_id": str})
    frame[period] = pd.PeriodIndex(frame[period], freq=freq)
    assert not frame.duplicated(["sector_id", period]).any()
    return frame


def check_definitions(context: str, period: str, freq: str, horizon: int) -> tuple[dict, set]:
    output = raw_output(context)
    nsa = raw_output(context, "nsa")
    weather = raw_weather(freq)
    frame = read_sample(ROOT / f"data/processed/outcomes/{context}.csv", period, freq)
    assert frame.sector_id.nunique() == (11 if context == "va" else 6)
    max_error = 0.0
    for row in frame.to_dict("records"):
        p, sector = row[period], row["sector_id"]
        level = output[(sector, p)]
        previous = output.get((sector, p-1))
        annual_lag = output.get((sector, p-horizon))
        nsa_level, nsa_lag = nsa.get((sector, p)), nsa.get((sector, p-horizon))
        expected = {
            "y_short": 100*(math.log(level)-math.log(previous)) if previous else np.nan,
            "y_yoy": 100*(math.log(level)-math.log(annual_lag)) if annual_lag else np.nan,
            "y_yoy_pct": 100*(level/annual_lag-1) if annual_lag else np.nan,
            "y_log_level": 100*math.log(level), "y_raw_level": level,
            "y_yoy_nsa": 100*math.log(nsa_level/nsa_lag) if nsa_level and nsa_lag else np.nan,
            "y_yoy_pct_nsa": 100*(nsa_level/nsa_lag-1) if nsa_level and nsa_lag else np.nan,
        }
        for col, value in expected.items():
            close(row[col], value, f"{context} {sector} {p} {col}", atol=1e-10, rtol=1e-12)
            if np.isfinite(value):
                max_error = max(max_error, abs(row[col]-value))
        if annual_lag:
            intervening = [output.get((sector, p-j)) for j in range(horizon+1)]
            assert all(value is not None for value in intervening)
            summed_growth = sum(100*(math.log(intervening[j])-math.log(intervening[j+1]))
                                for j in range(horizon))
            close(row["y_yoy"], summed_growth, f"{context} annual growth telescoping", atol=1e-10)
        for col in BASE:
            close(row[col], weather[p][col], f"{context} current raw weather {col}", atol=1e-12)
            label = "temp" if col == "temp_c" else col
            lag = weather.get(p-horizon, {}).get(col, np.nan)
            window = [weather.get(p-j, {}).get(col, np.nan) for j in range(horizon)]
            close(row[f"{label}_year_lag"], lag, f"{context} annual weather lag {col}", atol=1e-12)
            close(row[f"{label}_year_change"], weather[p][col]-lag,
                  f"{context} annual weather difference {col}", atol=1e-12)
            close(row[f"{label}_year_mean"], np.mean(window),
                  f"{context} equal-period annual weather mean {col}", atol=1e-12)

    sectors = sorted(frame.sector_id.unique())
    available = []
    for p in sorted(weather):
        if all((sector, p) in output and (sector, p-1) in output
               and (sector, p-horizon) in output for sector in sectors):
            available.append(p)
    expected_core = set(available)
    actual_core = read_sample(OUT / context / "weather/common_sample.csv", period, freq)
    assert set(actual_core[period]) == expected_core, f"Unexpected core dates: {context}"
    assert actual_core[["y_yoy_nsa", "y_yoy_pct_nsa"]].notna().all().all(), "NSA coverage must not silently change SA common eligibility"
    expected_annual_weather = {p for p in available if p-horizon in weather
                               and all(p-j in weather for j in range(horizon))}
    actual_annual = read_sample(OUT / context / "yoy_weather/common_sample.csv", period, freq)
    assert set(actual_annual[period]) == expected_annual_weather
    controls = pd.read_csv(ROOT / f"data/processed/extended_controls/{'monthly' if freq == 'M' else 'quarterly'}_controls.csv")
    eligible = set(pd.PeriodIndex(controls.loc[controls.eligible_90, period], freq=freq))
    full = read_sample(OUT / context / "full/common_sample.csv", period, freq)
    assert set(full[period]) == expected_core & eligible
    return {"raw_definition_rows_checked": len(frame), "max_absolute_outcome_error": max_error,
            "core_periods": len(expected_core), "yoy_weather_periods": len(expected_annual_weather),
            "full_control_periods": int(full[period].nunique())}, expected_core


def model_description(group: str, model: str) -> tuple[str, list[str], bool]:
    if group == "yoy_weather":
        nsa = model.endswith("N")
        base_model = model[:-1] if nsa else model
        terms = {"Y1": BASE,
                 "Y2": ["temp_year_mean", "rain_100mm_year_mean", "rh_pct_year_mean"],
                 "Y3": ["temp_year_change", "rain_100mm_year_change", "rh_pct_year_change"],
                 "Y4": BASE+["temp_year_lag", "rain_100mm_year_lag", "rh_pct_year_lag"]}[base_model]
        return "y_yoy_nsa" if nsa else "y_yoy", terms, False
    base_model = model[:-1] if model.endswith("X") else model
    if base_model in {"G2N", "G3N"}:
        return "y_yoy_nsa" if base_model == "G2N" else "y_yoy_pct_nsa", BASE+(ADDED if group == "full" else []), False
    outcome = {"G1": "y_short", "G2": "y_yoy", "G3": "y_yoy_pct", "G2X": "y_yoy",
               "L1": "y_log_level", "L2": "y_log_level", "L1X": "y_log_level",
               "L3": "y_raw_level", "L4": "y_raw_level", "L3X": "y_raw_level"}[model]
    return outcome, BASE+(ADDED if group == "full" else []), model in {"L2", "L4"}


def numpy_estimate(frame: pd.DataFrame, period: str, outcome: str,
                   terms: list[str], changing_seasonality: bool, bandwidth: int) -> dict:
    counts = frame.groupby(period).sector_id.nunique()
    sectors = frame.sector_id.nunique()
    assert counts.eq(sectors).all()
    assert len(frame) == len(counts)*sectors
    columns = [outcome]+terms
    means = frame.groupby(period, sort=True)[columns].mean()
    years = np.array([p.year for p in means.index])
    seasons = np.array([p.month if p.freqstr.startswith("M") else p.quarter for p in means.index])
    year_values, season_values = sorted(set(years)), sorted(set(seasons))
    blocks = [np.ones((len(means), 1)), means[terms].to_numpy()]
    blocks.extend([(years[:, None] == np.array(year_values[1:])).astype(float),
                   (seasons[:, None] == np.array(season_values[1:])).astype(float)])
    trend_count = 0
    if changing_seasonality:
        trend_count = len(season_values)-1
        blocks.append((years[:, None]-2000)*(seasons[:, None] == np.array(season_values[1:])))
    x = np.column_stack(blocks)
    rank = np.linalg.matrix_rank(x)
    assert rank == x.shape[1], "Collapsed independent design is rank deficient"
    q, triangular = np.linalg.qr(x, mode="reduced")
    beta = np.linalg.solve(triangular, q.T @ means[outcome].to_numpy())
    fitted = x @ beta
    residual = means[outcome].to_numpy()-fitted
    r_inv = np.linalg.solve(triangular, np.eye(len(beta)))
    bread = r_inv @ r_inv.T
    calendar = np.array([p.ordinal-means.index[0].ordinal for p in means.index])
    scores = np.zeros((int(calendar.max())+1, x.shape[1]))
    scores[calendar] = x*residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, bandwidth+1):
        moment = scores[lag:].T @ scores[:-lag]
        meat += (1-lag/(bandwidth+1))*(moment+moment.T)
    n, periods = len(frame), len(means)
    panel_k = sectors*len(season_values)+len(year_values)-1+len(terms)+trend_count
    covariance = bread @ meat @ bread
    covariance *= periods/(periods-1)*(n-1)/(n-panel_k)
    # Recover the full panel residuals, including each sector's seasonal intercept.
    residual_panel = frame[outcome]-frame[period].map(dict(zip(means.index, fitted)))
    residual_panel = residual_panel-residual_panel.groupby([frame.sector_id, frame.season]).transform("mean")
    total_ss = np.square(frame[outcome]-frame[outcome].mean()).sum()
    r_squared = 1-np.square(residual_panel).sum()/total_ss
    return {"beta": beta, "cov": covariance, "periods": periods, "panel_k": panel_k,
            "time_df": int(periods-rank), "r_squared": r_squared,
            "calendar_gaps": len(scores)-periods, "terms": terms}


def check_estimates(context: str, period: str, freq: str, horizon: int) -> dict:
    records = []
    max_beta_error, max_se_error = 0.0, 0.0
    for group in ["weather", "full", "yoy_weather"]:
        folder = OUT / context / group
        common = read_sample(folder / "common_sample.csv", period, freq)
        result = pd.read_csv(folder / "regression_results.csv")
        # This verifier independently checks calendar HAC; sector CR1 has a
        # separate manual panel-score verifier and is not a HAC bandwidth.
        result = result.loc[result.covariance.eq("Driscoll-Kraay")]
        required = {"Y1", "Y2", "Y3", "Y4", "Y1N", "Y2N", "Y3N", "Y4N"} if group == "yoy_weather" else {"G2", "G3", "G2X", "G2N", "G3N", "G2NX"}
        assert required <= set(result.model), (context, group, "Missing retained SA or companion NSA YoY model", required-set(result.model))
        assert not result.duplicated(["model", "term", "hac_lags"]).any()
        for (model, bandwidth), stored in result.groupby(["model", "hac_lags"], sort=False):
            outcome, terms, trend = model_description(group, model)
            sample = common.loc[~common.year.between(2020, 2022)].copy() if model.endswith("X") else common
            independent = numpy_estimate(sample, period, outcome, terms, trend, int(bandwidth))
            assert set(stored.term) == set(terms)
            assert stored.outcome.eq(outcome).all()
            assert stored.output_adjustment.eq("NSA" if outcome.endswith("_nsa") else "SA").all()
            assert stored.n_obs.eq(len(sample)).all()
            assert stored.n_sectors.eq(sample.sector_id.nunique()).all()
            assert stored[f"n_{period}s"].eq(independent["periods"]).all()
            assert stored.time_residual_df.eq(independent["time_df"]).all()
            close(stored.r_squared, independent["r_squared"], f"{context}/{group}/{model} full-panel R2")
            dof = independent["periods"]-1
            for term in terms:
                row = stored.loc[stored.term.eq(term)].iloc[0]
                j = 1+terms.index(term)
                b = float(independent["beta"][j])
                se = float(np.sqrt(independent["cov"][j, j]))
                close(row.coefficient, b, f"{context}/{group}/{model}/{term} beta")
                close(row.std_error, se, f"{context}/{group}/{model}/{term} SE")
                p = float(2*student_t.sf(abs(b/se), dof))
                critical = student_t.ppf(.975, dof)
                close(row.p_value, p, f"{context}/{group}/{model}/{term} p")
                close([row.ci_lower, row.ci_upper], [b-critical*se, b+critical*se],
                      f"{context}/{group}/{model}/{term} interval")
                max_beta_error = max(max_beta_error, abs(row.coefficient-b))
                max_se_error = max(max_se_error, abs(row.std_error-se))
            if int(bandwidth) == horizon:
                records.append({"controls": group, "model": model,
                                "periods": independent["periods"], "time_residual_df": independent["time_df"],
                                "calendar_gaps": independent["calendar_gaps"],
                                "temperature_term": terms[0], "coefficient": float(independent["beta"][1]),
                                "std_error": float(np.sqrt(independent["cov"][1, 1]))})
            if group == "yoy_weather" and model in {"Y4", "Y4N"} and int(bandwidth) == horizon:
                filename = "temperature_difference_restriction_nsa.json" if model == "Y4N" else "temperature_difference_restriction.json"
                restriction = json.loads((folder / filename).read_text())
                indices = [1+terms.index("temp_c"), 1+terms.index("temp_year_lag")]
                b_sum = float(independent["beta"][indices].sum())
                se_sum = float(np.sqrt(independent["cov"][np.ix_(indices, indices)].sum()))
                close(restriction["sum"], b_sum, f"{context} temperature endpoint restriction")
                close(restriction["std_error"], se_sum, f"{context} temperature endpoint restriction SE")
                close(restriction["p_value"], 2*student_t.sf(abs(b_sum/se_sum), dof),
                      f"{context} temperature endpoint restriction p")
    return {"model_bandwidth_fits_checked": 2*len(records),
            "maximum_absolute_coefficient_difference": max_beta_error,
            "maximum_absolute_standard_error_difference": max_se_error,
            "representative_primary_bandwidth_results": records}


def verify_outcomes() -> dict:
    report = {}
    for context, period, freq, horizon in CONTEXTS:
        definitions, _ = check_definitions(context, period, freq, horizon)
        report[context] = {**definitions, **check_estimates(context, period, freq, horizon)}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    fits = sum(record["model_bandwidth_fits_checked"] for record in report.values())
    print(f"Independent outcome verification passed: {fits} model/bandwidth fits; "
          "raw definitions, calendar samples, estimates and R² checked.")
    return report


if __name__ == "__main__":
    verify_outcomes()
