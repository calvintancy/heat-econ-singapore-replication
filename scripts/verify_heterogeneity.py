"""Independently verify sector heterogeneity through mean/deviation time regressions.

The estimator uses native PyFixest with explicit full-panel designs. This verifier fits small
national-mean, sector-deviation and separate-series regressions, combines their
per-date influence functions, and constructs calendar-aware Bartlett covariance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/heterogeneity"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
ADDED = ["wind_mps", "pm25_10ug"]
OUTCOMES = ["y_short", "y_yoy", "y_yoy_nsa", "y_log_level"]
CONTEXTS = [("va", "quarter", "Q", 4), ("iip-monthly", "month", "M", 12),
            ("iip-quarterly", "quarter", "Q", 4)]


def close(actual, expected, message: str, *, atol: float = 2e-7) -> None:
    if not np.allclose(actual, expected, rtol=2e-7, atol=atol, equal_nan=True):
        raise AssertionError(f"{message}: actual={actual}; independent={expected}")


def time_ols(x: np.ndarray, y: np.ndarray, temperature_index: int) -> tuple:
    scale = np.linalg.norm(x, axis=0)
    standardized = x/scale
    assert np.linalg.matrix_rank(standardized) == x.shape[1]
    inverse = np.linalg.pinv(standardized)/scale[:, None]
    coefficient = inverse@y
    residual = y-x@coefficient
    return coefficient[temperature_index], residual, inverse[temperature_index], coefficient, inverse


def analytic_fit(sample: pd.DataFrame, period: str, terms: list[str], model: str) -> dict:
    sectors = sorted(sample.sector_id.unique())
    dates = pd.PeriodIndex(sorted(sample[period].unique()))
    s, t = len(sectors), len(dates)
    assert len(sample) == s*t
    ordered = sample.set_index([period, "sector_id"]).reindex(
        pd.MultiIndex.from_product([dates, sectors], names=[period, "sector_id"]))
    assert ordered[OUTCOMES+terms].notna().all().all()
    y = ordered[OUTCOMES].to_numpy().reshape(t, s, len(OUTCOMES))
    weather = ordered[terms].to_numpy().reshape(t, s, len(terms))
    assert np.allclose(weather, weather[:, [0], :], atol=1e-12, rtol=0)
    weather = weather[:, 0, :]
    years = np.array([date.year for date in dates])
    seasons = np.array([date.month if date.freqstr.startswith("M") else date.quarter for date in dates])
    year_values, season_values = sorted(set(years)), sorted(set(seasons))
    b = (seasons[:, None] == np.array(season_values)).astype(float)
    a = (years[:, None] == np.array(year_values[1:])).astype(float)
    x = np.column_stack([b, a, weather])
    temperature_index = b.shape[1]+a.shape[1]
    time_k = x.shape[1]
    mean = y.mean(axis=1)
    deviations = y-mean[:, None, :]
    if model in {"H1", "H2"}:
        mean_beta, mean_error, mean_weight, _, _ = time_ols(x, mean, temperature_index)
        deviation_weather = weather[:, [0]] if model == "H1" else weather
        xd = np.column_stack([b, deviation_weather])
        db, de, dw, _, _ = time_ols(xd, deviations.reshape(t, -1), b.shape[1])
        beta = db.reshape(s, -1)+mean_beta[None, :]
        residual = de.reshape(t, s, -1)+mean_error[:, None, :]
        influence = (dw[:, None, None]*de.reshape(t, s, -1)
                     + mean_weight[:, None, None]*mean_error[:, None, :])
        p, q, ny = len(terms), b.shape[1], len(year_values)
        k = s*q+(ny-1)+s+(p-1) if model == "H1" else s*(q+p)+(ny-1)
        output_sectors = sectors
    else:
        direct_beta, error, weight, all_beta, inverse = time_ols(x, y.reshape(t, -1), temperature_index)
        direct_beta = direct_beta.reshape(s, -1)
        error = error.reshape(t, s, -1)
        if model == "R1":
            beta = direct_beta[1:]-direct_beta[[0]]
            influence = weight[:, None, None]*(error[:, 1:, :]-error[:, [0], :])
            residual = error-error.mean(axis=1, keepdims=True)
            output_sectors = sectors[1:]
            k = t+(s-1)*time_k
        else:
            beta = direct_beta
            residual = error
            influence = weight[:, None, None]*error
            output_sectors = sectors
            k = s*time_k
    ss_total = np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum(axis=(0, 1))
    r2 = 1-np.square(residual).sum(axis=(0, 1))/ss_total
    result = {"sectors": output_sectors, "all_sectors": sectors, "dates": dates, "beta": beta,
              "influence": influence, "residual": residual, "k": k, "time_k": time_k,
              "t": t, "n": s*t, "r2": r2}
    if model == "H3":
        result["weather_beta"] = all_beta[temperature_index:temperature_index+len(terms)].reshape(len(terms), s, -1)
        result["weather_weights"] = inverse[temperature_index:temperature_index+len(terms)]
        result["sector_r2"] = 1-np.square(residual).sum(axis=0)/np.square(y-y.mean(axis=0, keepdims=True)).sum(axis=0)
    return result


def bartlett(influence: np.ndarray, dates: pd.PeriodIndex, bandwidth: int) -> np.ndarray:
    positions = np.array([date.ordinal-dates[0].ordinal for date in dates])
    scores = np.zeros((int(positions.max())+1, influence.shape[1]))
    scores[positions] = influence
    covariance = scores.T@scores
    for lag in range(1, bandwidth+1):
        cross = scores[lag:].T@scores[:-lag]
        covariance += (1-lag/(bandwidth+1))*(cross+cross.T)
    return (covariance+covariance.T)/2


def correction(result: dict) -> float:
    return result["t"]/(result["t"]-1)*(result["n"]-1)/(result["n"]-result["k"])


def inference(beta: float, variance: float, periods: int) -> dict:
    assert variance >= -1e-10
    se = float(np.sqrt(max(variance, 0)))
    critical = student_t.ppf(.975, periods-1)
    return {"coefficient": beta, "std_error": se,
            "ci_lower": beta-critical*se, "ci_upper": beta+critical*se,
            "p_value": 2*student_t.sf(abs(beta/se), periods-1) if se else np.nan}


def holm(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.minimum(1, np.maximum.accumulate(pvalues[order]*(len(order)-np.arange(len(order)))))
    result = np.empty(len(order))
    result[order] = adjusted
    return result


def verify_context(context: str, control_set: str, period: str, freq: str, horizon: int) -> dict:
    folder = OUT / context / control_set
    sample = pd.read_csv(folder / "common_sample.csv", dtype={"sector_id": str})
    original = pd.read_csv(ROOT / f"outputs/outcomes/{context}/{control_set}/common_sample.csv", dtype={"sector_id": str})
    sort = [period, "sector_id"]
    sample = sample.sort_values(sort).reset_index(drop=True)
    original = original.sort_values(sort).reset_index(drop=True)
    assert sample[sort].equals(original[sort])
    close(sample[OUTCOMES+BASE+ADDED], original[OUTCOMES+BASE+ADDED], "Unchanged outcome source sample")
    sample[period] = pd.PeriodIndex(sample[period], freq=freq)
    slopes = pd.read_csv(folder / "sector_slopes.csv", dtype={"sector_id": str})
    covariances = pd.read_csv(folder / "slope_covariances.csv", dtype={"sector_i": str, "sector_j": str})
    pairs = pd.read_csv(folder / "pairwise_tests.csv", dtype={"first_sector_id": str, "second_sector_id": str})
    tests = pd.read_csv(folder / "joint_tests.csv")
    separate = pd.read_csv(folder / "separate_series.csv", dtype={"sector_id": str})
    separate_weather = pd.read_csv(folder / "separate_sector_coefficients.csv", dtype={"sector_id": str})
    assert not separate_weather.duplicated(["outcome", "sector_id", "term", "hac_lags"]).any()
    terms = BASE+(ADDED if control_set == "full" else [])
    max_beta_error, max_covariance_error = 0.0, 0.0
    unavailable = []
    fits = {}
    for model in ["H1", "H2", "H3", "R1", "H3X"]:
        data = sample.loc[~sample.year.between(2020, 2022)].copy() if model == "H3X" else sample
        result = analytic_fit(data, period, terms, model)
        fits[model] = result
        for j, outcome in enumerate(OUTCOMES):
            beta = result["beta"][:, j]
            for bandwidth in [horizon, 2*horizon]:
                uncorrected = bartlett(result["influence"][:, :, j], result["dates"], bandwidth)
                covariance = uncorrected*correction(result)
                selector = (slopes.model.eq(model)&slopes.outcome.eq(outcome)&slopes.hac_lags.eq(bandwidth))
                rows = slopes.loc[selector].set_index("sector_id").reindex(result["sectors"])
                assert len(rows) == len(beta) and rows.coefficient.notna().all()
                assert rows.n_parameters.eq(result["k"]).all()
                assert rows.n_periods.eq(result["t"]).all()
                close(rows.r_squared, result["r2"][j], f"{context}/{control_set}/{model} R2")
                expected_estimand = "Difference from reference sector" if model == "R1" else "Absolute sector slope"
                assert rows.estimand.eq(expected_estimand).all()
                for i, sector in enumerate(result["sectors"]):
                    expected = inference(beta[i], covariance[i, i], result["t"])
                    for field, value in expected.items():
                        close(rows.loc[sector, field], value, f"{context}/{control_set}/{model}/{outcome}/{sector}/{field}")
                max_beta_error = max(max_beta_error, float(np.max(abs(rows.coefficient.to_numpy()-beta))))
                cv = covariances.loc[covariances.model.eq(model)&covariances.outcome.eq(outcome)&covariances.hac_lags.eq(bandwidth)]
                stored_cov = cv.pivot(index="sector_i", columns="sector_j", values="covariance").reindex(
                    index=result["sectors"], columns=result["sectors"]).to_numpy()
                close(stored_cov, covariance, f"{context}/{control_set}/{model}/{outcome} joint covariance")
                max_covariance_error = max(max_covariance_error, float(np.max(abs(stored_cov-covariance))))
                identity = np.eye(len(beta))
                restrictions = identity if model == "R1" else identity[1:]-identity[[0]]
                difference = restrictions@beta
                variance = restrictions@covariance@restrictions.T
                singular_values = np.linalg.svd(variance, compute_uv=False)
                rank = int(np.count_nonzero(singular_values > singular_values.max()*1e-10))
                if model in {"H3", "H3X", "R1"}:
                    rank = min(rank, result["t"]-result["time_k"])
                test = tests.loc[tests.model.eq(model)&tests.outcome.eq(outcome)&tests.hac_lags.eq(bandwidth)].iloc[0]
                assert test.restrictions == len(difference)
                assert test.denominator_df == result["t"]-1
                if rank < len(difference):
                    assert str(test.status).startswith("unavailable"), f"Singular omnibus test was reported as available: {context}/{control_set}/{model}/{outcome}"
                    assert pd.isna(test.p_value) and pd.isna(test.wald_chi2)
                    unavailable.append(f"{model}/{outcome}/{bandwidth}")
                else:
                    assert test.status == "available"
                    statistic = float(difference@np.linalg.solve(variance, difference))
                    close(test.wald_chi2, statistic, "Independent omnibus Wald", atol=1e-5)
                    close(test.f_statistic, statistic/len(difference), "Independent omnibus F", atol=1e-5)
                    close(test.p_value, f_dist.sf(statistic/len(difference), len(difference), result["t"]-1), "Independent omnibus p")
                family = pairs.loc[pairs.model.eq(model)&pairs.outcome.eq(outcome)&pairs.hac_lags.eq(bandwidth)]
                pvalues = []
                for _, pair in family.iterrows():
                    weights = np.array([int(sector == pair.first_sector_id)-int(sector == pair.second_sector_id)
                                        for sector in result["sectors"]])
                    expected = inference(float(weights@beta), float(weights@covariance@weights), result["t"])
                    for field, value in expected.items():
                        close(pair[field], value, "Independent pairwise inference")
                    pvalues.append(expected["p_value"])
                close(family.p_value_holm, holm(np.array(pvalues)), "Independent Holm multiplicity adjustment")

                if model == "H3":
                    for i, sector in enumerate(result["sectors"]):
                        single = separate.loc[separate.sector_id.eq(sector)&separate.outcome.eq(outcome)&separate.hac_lags.eq(bandwidth)].iloc[0]
                        single_variance = uncorrected[i, i]*result["t"]/(result["t"]-result["time_k"])
                        for field, value in inference(beta[i], single_variance, result["t"]).items():
                            close(single[field], value, "Independent separate-series inference")
                        close(single.pooled_H3_coefficient, beta[i], "H3 equals separate-series point estimate")
                        assert abs(single.point_estimate_difference) < 1e-8
                        sector_error = result["residual"][:, i, j]
                        weather_influences = result["weather_weights"].T*sector_error[:, None]
                        weather_covariance = bartlett(weather_influences, result["dates"], bandwidth)
                        weather_covariance *= result["t"]/(result["t"]-result["time_k"])
                        weather_rows = separate_weather.loc[separate_weather.sector_id.eq(sector)
                            & separate_weather.outcome.eq(outcome)&separate_weather.hac_lags.eq(bandwidth)].set_index("term")
                        assert set(weather_rows.index) == set(terms)
                        assert weather_rows.n_obs.eq(result["t"]).all()
                        assert weather_rows.n_parameters.eq(result["time_k"]).all()
                        assert weather_rows.residual_df.eq(result["t"]-result["time_k"]).all()
                        close(weather_rows.r_squared, result["sector_r2"][i, j], "Separate-series total R2")
                        for term_index, term in enumerate(terms):
                            weather_beta = float(result["weather_beta"][term_index, i, j])
                            expected_weather = inference(weather_beta, weather_covariance[term_index, term_index], result["t"])
                            for field, value in expected_weather.items():
                                close(weather_rows.loc[term, field], value, f"Separate-series {term}/{field}")
    h3, relative = fits["H3"], fits["R1"]
    contrast = np.eye(len(h3["sectors"]))[1:]-np.eye(len(h3["sectors"]))[[0]]
    close(relative["beta"], contrast@h3["beta"], "R1 reference differences equal H3 contrasts")
    for j in range(len(OUTCOMES)):
        for bandwidth in [horizon, 2*horizon]:
            v_absolute = bartlett(h3["influence"][:, :, j], h3["dates"], bandwidth)
            v_relative = bartlett(relative["influence"][:, :, j], relative["dates"], bandwidth)
            close(v_relative, contrast@v_absolute@contrast.T, "Uncorrected R1 and H3 contrast covariance equivalence")
    if context == "va" and control_set == "full":
        assert all(f"H3X/{outcome}/{bandwidth}" in unavailable
                   for outcome in OUTCOMES for bandwidth in [horizon, 2*horizon])
    return {"model_outcome_bandwidth_groups_checked": 5*len(OUTCOMES)*2,
            "national_periods": int(sample[period].nunique()), "sectors": int(sample.sector_id.nunique()),
            "maximum_absolute_coefficient_difference": max_beta_error,
            "maximum_absolute_covariance_difference": max_covariance_error,
            "separate_series_weather_coefficients_checked": int(len(separate_weather)),
            "unavailable_singular_omnibus_tests": unavailable,
            "method": "Independent national-mean/deviation and separate-series OLS, joint calendar influence Bartlett covariance"}


def verify_heterogeneity() -> dict:
    report = {}
    for context, period, freq, horizon in CONTEXTS:
        report[context] = {controls: verify_context(context, controls, period, freq, horizon)
                           for controls in ["weather", "full"]}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Independent heterogeneity verification passed: {len(CONTEXTS)*2*5*len(OUTCOMES)*2} model/outcome/bandwidth groups, "
          "joint sector covariance, equality tests, pairwise tests and separate-series equivalence.")
    return report


if __name__ == "__main__":
    verify_heterogeneity()
