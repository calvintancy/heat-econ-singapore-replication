"""Verify official aggregate GVA outcomes and all fits without importing its estimator."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from verify_outcomes import raw_weather

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/aggregate_va"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
ADDED = ["wind_mps", "pm25_10ug"]


def close(actual, expected, message: str, *, atol: float = 2e-7) -> None:
    if not np.allclose(actual, expected, rtol=2e-7, atol=atol, equal_nan=True):
        raise AssertionError(f"{message}: actual={actual}, independent={expected}")


def source_levels(stem: str) -> dict:
    payload = json.loads((ROOT / f"data/raw/{stem}_tabledata.json").read_text())["Data"]
    row, = [row for row in payload["row"] if row["seriesNo"] == "1.4"]
    assert row["rowText"] == "Gross Value Added At Basic Prices" and row["uoM"] == "Million Dollars"
    result = {}
    for item in row["columns"]:
        year, quarter = item["key"].split()
        period = pd.Period(f"{year}Q{quarter[0]}", freq="Q")
        assert period not in result
        result[period] = float(item["value"])
    return result


def read_sample(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sector_id": str})
    frame["quarter"] = pd.PeriodIndex(frame.quarter, freq="Q")
    frame = frame.sort_values("quarter").reset_index(drop=True)
    assert not frame.quarter.duplicated().any() and frame.sector_id.eq("1.4").all()
    return frame


def verify_definitions() -> pd.DataFrame:
    frame = read_sample(ROOT / "data/processed/aggregate_va/panel.csv")
    sa, nsa = source_levels("va_sa"), source_levels("va_nsa")
    weather = raw_weather("Q")
    monthly_temp = pd.read_csv(ROOT / "data/raw/temperature_monthly.csv")
    monthly_temp["quarter"] = pd.PeriodIndex(monthly_temp.month, freq="M").asfreq("Q")
    equal_month = monthly_temp.groupby("quarter").mean_temp.mean()
    for row in frame.to_dict("records"):
        period = row["quarter"]
        values = {
            "va_sa": sa[period], "va_nsa": nsa[period],
            "y_short": 100*math.log(sa[period]/sa[period-1]),
            "y_yoy": 100*math.log(sa[period]/sa[period-4]),
            "y_yoy_pct": 100*(sa[period]/sa[period-4]-1),
            "y_yoy_nsa": 100*math.log(nsa[period]/nsa[period-4]),
            "y_yoy_pct_nsa": 100*(nsa[period]/nsa[period-4]-1),
            "y_short_nsa": 100*math.log(nsa[period]/nsa[period-1]),
            "y_log_level": 100*math.log(sa[period]), "y_raw_level": sa[period],
            "temp_c_unweighted": equal_month.loc[period],
        }
        for column, value in values.items():
            close(row[column], value, f"Raw aggregate source {period}/{column}", atol=1e-10)
        close(row["y_yoy"], sum(100*math.log(sa[period-lag]/sa[period-lag-1]) for lag in range(4)),
              "Annual log growth telescopes", atol=1e-10)
        assert row["time"] == period.ordinal-pd.Period("1982Q1").ordinal
        for column in BASE:
            close(row[column], weather[period][column], "Raw calendar weather", atol=1e-12)
            prefix = "temp" if column == "temp_c" else column
            lagged = weather.get(period-4, {}).get(column, np.nan)
            annual_mean = np.mean([weather.get(period-lag, {}).get(column, np.nan) for lag in range(4)])
            close(row[f"{prefix}_year_lag"], lagged, "Annual weather lag", atol=1e-12)
            close(row[f"{prefix}_year_change"], weather[period][column]-lagged, "Annual weather change", atol=1e-12)
            close(row[f"{prefix}_year_mean"], annual_mean, "Annual weather equal-period mean", atol=1e-12)
        for lag in range(1, 5):
            close(row[f"temperature_lag_{lag}"], weather.get(period-lag, {}).get("temp_c", np.nan),
                  "Calendar distributed lag", atol=1e-12)
    expected_dates = {period for period in weather if period in sa and period in nsa
                      and period-1 in sa and period-4 in sa and period-1 in nsa}
    assert set(frame.quarter) == expected_dates, "Aggregate source/weather intersection is incomplete"
    assert all(period-4 in nsa for period in expected_dates), "NSA YoY coverage must not alter original SA eligibility"
    return frame


def specification(group: str, model: str) -> tuple[str, list[str]]:
    if group == "baseline":
        return {
            "S1": ("y_short", ["temp_c"]), "S2": ("y_short", ["temp_c", "rain_100mm"]),
            "S3": ("y_short", BASE), "S4": ("y_short", BASE), "S5": ("y_short", BASE),
            "S8": ("y_short_nsa", BASE), "S9": ("y_short", ["temp_c_unweighted", "rain_100mm", "rh_pct"]),
        }[model]
    if group in {"weather", "full"}:
        outcome = {"G1": "y_short", "G2": "y_yoy", "G3": "y_yoy_pct", "G2N": "y_yoy_nsa", "G3N": "y_yoy_pct_nsa", "L1": "y_log_level", "L3": "y_raw_level"}[model.rstrip("X")]
        return outcome, BASE+(ADDED if group == "full" else [])
    if group == "matched_controls":
        number = int(model[1])
        return "y_short", BASE+{1: [], 2: ["wind_mps"], 3: ["pm25_10ug"], 4: ADDED}[number]
    if group == "yoy_weather":
        nsa = model.endswith("N")
        return "y_yoy_nsa" if nsa else "y_yoy", {
            "Y1": BASE, "Y2": ["temp_year_mean", "rain_100mm_year_mean", "rh_pct_year_mean"],
            "Y3": ["temp_year_change", "rain_100mm_year_change", "rh_pct_year_change"],
            "Y4": BASE+["temp_year_lag", "rain_100mm_year_lag", "rh_pct_year_lag"],
        }[model[:-1] if nsa else model]
    return "y_short", BASE+([f"temperature_lag_{i}" for i in range(1, 5)] if model == "D1" else [])


def calendar_ols(frame: pd.DataFrame, outcome: str, terms: list[str], bandwidth: int) -> dict:
    years, seasons = sorted(frame.year.unique()), sorted(frame.season.unique())
    x = np.column_stack([np.ones(len(frame)), frame[terms],
                         frame.year.to_numpy()[:, None] == np.array(years[1:]),
                         frame.season.to_numpy()[:, None] == np.array(seasons[1:])]).astype(float)
    scale = np.linalg.norm(x, axis=0)
    standardized = x/scale
    assert np.linalg.matrix_rank(standardized) == x.shape[1]
    inverse = np.linalg.pinv(standardized)/scale[:, None]
    beta = inverse@frame[outcome].to_numpy()
    residual = frame[outcome].to_numpy()-x@beta
    influences = inverse.T*residual[:, None]
    positions = np.array([p.ordinal-frame.quarter.iloc[0].ordinal for p in frame.quarter])
    score = np.zeros((int(positions.max())+1, x.shape[1]))
    score[positions] = influences
    covariance = score.T@score
    for lag in range(1, bandwidth+1):
        cross = score[lag:].T@score[:-lag]
        covariance += (1-lag/(bandwidth+1))*(cross+cross.T)
    covariance *= len(frame)/(len(frame)-x.shape[1])
    return {"beta": beta, "cov": covariance, "k": x.shape[1], "time_df": len(frame)-x.shape[1],
            "r2": 1-residual@residual/np.square(frame[outcome]-frame[outcome].mean()).sum(),
            "gaps": len(score)-len(frame)}


def verify_aggregate_va() -> dict:
    prepared = verify_definitions()
    controls = pd.read_csv(ROOT / "data/processed/extended_controls/quarterly_controls.csv")
    controls["quarter"] = pd.PeriodIndex(controls.quarter, freq="Q")
    eligible = set(controls.loc[controls.eligible_90, "quarter"])
    assert set(prepared.loc[prepared.eligible_90, "quarter"]) == set(prepared.quarter)&eligible
    all_periods = set(prepared.quarter)
    fits, max_beta_error, max_se_error, checked = {}, 0.0, 0.0, []
    for group in ["baseline", "weather", "full", "matched_controls", "yoy_weather", "distributed_lag"]:
        result = pd.read_csv(OUT / group / "regression_results.csv")
        if group in {"weather", "full"}:
            assert {"G2", "G3", "G2X", "G3X", "G2N", "G3N", "G2NX", "G3NX"} <= set(result.model)
        if group == "yoy_weather":
            assert {"Y1", "Y2", "Y3", "Y4", "Y1N", "Y2N", "Y3N", "Y4N"} == set(result.model)
        assert not result.duplicated(["model", "term", "hac_lags"]).any()
        for (model, bandwidth), rows in result.groupby(["model", "hac_lags"], sort=False):
            sample = read_sample(OUT / group / "samples" / f"{model}.csv")
            outcome, terms = specification(group, model)
            expected_dates = all_periods & eligible if group in {"full", "matched_controls"} else all_periods
            if group == "baseline" and model == "S4":
                expected_dates = {p for p in expected_dates if p.year < 2020}
            if model.endswith("X") or group == "baseline" and model == "S5":
                expected_dates = {p for p in expected_dates if not 2020 <= p.year <= 2022}
            if group in {"yoy_weather", "distributed_lag"}:
                expected_dates = {p for p in expected_dates if p >= pd.Period("1983Q1")}
            assert set(sample.quarter) == expected_dates, f"Unexpected aggregate sample {group}/{model}"
            expected = prepared.set_index("quarter").loc[sample.quarter]
            close(sample[[outcome]+terms], expected[[outcome]+terms], "Unchanged aggregate estimation sample")
            estimate = calendar_ols(sample, outcome, terms, int(bandwidth))
            fits[(group, model, int(bandwidth))] = (estimate, terms, sample)
            assert set(rows.term) == set(terms)
            assert rows.outcome.eq(outcome).all() and rows.n_sectors.eq(1).all()
            assert rows.output_adjustment.eq("NSA" if outcome.endswith("_nsa") else "SA").all()
            assert rows.n_obs.eq(len(sample)).all() and rows.n_quarters.eq(len(sample)).all()
            assert rows.n_parameters.eq(estimate["k"]).all() and rows.time_residual_df.eq(estimate["time_df"]).all()
            assert not rows.formula.str.contains("sector_id", regex=False).any()
            close(rows.r_squared, estimate["r2"], "Aggregate R2")
            means = {"mean_y": sample[outcome].mean(), "sd_y": sample[outcome].std(),
                     "mean_qoq_log": sample.y_short.mean(), "sd_qoq_log": sample.y_short.std(),
                     "mean_yoy_log": sample.y_yoy.mean(), "sd_yoy_log": sample.y_yoy.std(),
                     "mean_yoy_pct": sample.y_yoy_pct.mean()}
            for field, value in means.items():
                close(rows[field], value, f"Exact-sample mean/SD {field}")
            for term in terms:
                row = rows.loc[rows.term.eq(term)].iloc[0]
                index = 1+terms.index(term)
                beta = float(estimate["beta"][index])
                se = float(np.sqrt(estimate["cov"][index, index]))
                p = float(2*student_t.sf(abs(beta/se), len(sample)-1))
                critical = student_t.ppf(.975, len(sample)-1)
                for field, value in {"coefficient": beta, "std_error": se, "p_value": p,
                                     "ci_lower": beta-critical*se, "ci_upper": beta+critical*se}.items():
                    close(row[field], value, f"Independent aggregate {group}/{model}/{term}/{field}")
                if term.startswith("temp"):
                    for target, value in {"effect_0p1c": beta*.1, "se_0p1c": se*.1,
                                          "ci_lower_0p1c": (beta-critical*se)*.1,
                                          "ci_upper_0p1c": (beta+critical*se)*.1}.items():
                        close(row[target], value, "0.1°C scaling")
                max_beta_error = max(max_beta_error, abs(row.coefficient-beta))
                max_se_error = max(max_se_error, abs(row.std_error-se))
            if int(bandwidth) == 4:
                checked.append({"group": group, "model": model, "quarters": len(sample),
                                "residual_time_df": estimate["time_df"], "calendar_gaps": estimate["gaps"],
                                "temperature_term": terms[0], "coefficient": float(estimate["beta"][1]),
                                "std_error": float(np.sqrt(estimate["cov"][1, 1])),
                                "mean_y": float(sample[outcome].mean())})
    for group, model, file, terms_to_sum, key in [
        ("yoy_weather", "Y4", "temperature_difference_restriction.json", ["temp_c", "temp_year_lag"], "sum"),
        ("yoy_weather", "Y4N", "temperature_difference_restriction_nsa.json", ["temp_c", "temp_year_lag"], "sum"),
        ("distributed_lag", "D1", "cumulative_response.json", ["temp_c"]+[f"temperature_lag_{i}" for i in range(1, 5)], "sum_lag_0_to_4"),
    ]:
        estimate, terms, sample = fits[(group, model, 4)]
        saved = json.loads((OUT / group / file).read_text())
        indices = [1+terms.index(term) for term in terms_to_sum]
        total = float(estimate["beta"][indices].sum())
        se = float(np.sqrt(estimate["cov"][np.ix_(indices, indices)].sum()))
        close(saved[key], total, "Independent aggregate sum of temperature effects")
        close(saved["std_error"], se, "Independent covariance for aggregate coefficient sum")
        close(saved["p_value"], 2*student_t.sf(abs(total/se), len(sample)-1), "Aggregate coefficient-sum inference")
    from stationarity_diagnostics import verify_statistics

    diagnostic = OUT / "diagnostics"
    tests = pd.read_csv(diagnostic / "tests.csv", dtype={"sector_id": str})
    diagnostic_summary = pd.read_csv(diagnostic / "summary.csv", dtype={"sector_id": str})
    observations = pd.read_csv(diagnostic / "observations.csv", dtype={"sector_id": str})
    assert tests.status.eq("ok").all() and tests.n_periods.eq(len(prepared)).all()
    assert set(tests.variable) == {"y_short", "y_yoy", "y_yoy_nsa", "y_log_level"}
    for variable, group in observations.groupby("variable"):
        group = group.sort_values("period")
        assert list(group.period) == list(prepared.quarter.astype(str))
        close(group.test_value, prepared[variable], "Stationarity input equals raw-verified aggregate outcome")
    diagnostic_checks = verify_statistics(tests, diagnostic_summary, observations.to_dict("records"))
    report = {"official_aggregate_series": "1.4 Gross Value Added At Basic Prices",
              "raw_source_outcome_rows_checked": len(prepared), "model_bandwidth_fits_checked": len(fits),
              "maximum_absolute_coefficient_difference": max_beta_error,
              "maximum_absolute_standard_error_difference": max_se_error,
              "stationarity_diagnostic_statistics": diagnostic_checks,
              "primary_bandwidth_checks": checked,
              "method": "Raw official GVA calendar ratios and separate NumPy OLS/manual calendar Bartlett time-series HAC"}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Independent aggregate GVA verification passed: {len(fits)} model/bandwidth fits, "
          "raw-source growth, exact samples, mean Y, 0.1°C scaling and manual HAC.")
    return report


if __name__ == "__main__":
    verify_aggregate_va()
