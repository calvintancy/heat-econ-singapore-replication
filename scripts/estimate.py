"""Estimate the first sector panel specification and sensitivity checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyfixest_backend import fit as pyfixest_fit, ESTIMATOR_VERSION
from scipy.stats import t as student_t
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_nw_groupsum

from sector_cluster import common_slope_terms, sector_cluster_covariance
from cumulative_inference import cluster_sum
from sector_temperature import sector_temperature_slopes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FE = "C(sector_id) * C(season) + C(year)"
WEATHER = "rain_100mm + rh_pct"


def read_panel(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"data/processed/panel_{name}.csv", dtype={"sector_id": str})
    df["quarter"] = pd.PeriodIndex(df.quarter, freq="Q")
    # Sort each sector's time series, retaining the original quarter indices across exclusions.
    return df.sort_values(["sector_id", "time"]).reset_index(drop=True)


def fit(df: pd.DataFrame, formula: str, label: str):
    return pyfixest_fit(df, formula, label)


def estimates(model, df: pd.DataFrame, model_id: str, label: str, formula: str,
              terms: list[str], lags: int = 4, covariance: str = "Driscoll-Kraay",
              period_col: str = "quarter") -> list[dict]:
    # cov_nw_groupsum sums scores across sectors for each calendar quarter, then
    # applies Bartlett HAC. It uses actual quarter indices, including empty periods.
    time = (df.time - df.time.min()).to_numpy(dtype=int)
    periods = df[period_col].nunique()
    cluster_metadata = {}
    if covariance == "Sector cluster":
        if common_slope_terms(terms) != terms:
            raise ValueError("Sector-specific slopes cannot use sector-clustered inference")
        cov, cluster_metadata = sector_cluster_covariance(model, df.sector_id)
    elif covariance == "Quarter cluster":
        cov = cov_cluster(model, time, use_correction=True)
    else:
        cov = cov_nw_groupsum(model, nlags=lags, time=time, use_correction="cluster")
    dof = cluster_metadata.get("inference_df", periods-1)
    critical = student_t.ppf(.975, dof)
    rows = []
    for term in terms:
        j = model.model.exog_names.index(term)
        beta, se = float(model.params.iloc[j]), float(np.sqrt(cov[j, j]))
        rows.append({
            "model": model_id, "label": label, "term": term,
            "estimator": "PyFixest", "estimator_version": ESTIMATOR_VERSION,
            "coefficient": beta, "std_error": se,
            "ci_lower": beta - critical * se, "ci_upper": beta + critical * se,
            "p_value": 2 * student_t.sf(abs(beta / se), dof),
            "n_obs": int(model.nobs), "n_sectors": df.sector_id.nunique(),
            f"n_{period_col}s": periods, "start": str(df[period_col].min()), "end": str(df[period_col].max()),
            "r_squared": model.rsquared, "covariance": covariance,
            "hac_lags": lags if covariance == "Driscoll-Kraay" else 0,
            "inference_df": dof, "formula": formula,
            **cluster_metadata,
        })
    # Keep every existing calendar-HAC row unchanged. Append the requested sector
    # CR1 alternative once per model, on the baseline-bandwidth call only, so
    # callers that also run 2h do not produce duplicate sector-cluster rows.
    baseline_lags = 12 if period_col == "month" else 4
    clustered_terms = common_slope_terms(terms)
    if (covariance == "Driscoll-Kraay" and lags == baseline_lags
            and df.sector_id.nunique() >= 2 and clustered_terms):
        rows.extend(estimates(model, df, model_id, label, formula, clustered_terms,
                              lags=0, covariance="Sector cluster", period_col=period_col))
    return rows


def estimate() -> None:
    OUT.mkdir(exist_ok=True)
    broad, detailed = read_panel("broad"), read_panel("detailed")
    primary = f"g_va_sa_qoq ~ temp_c + {WEATHER} + {FE}"
    specs = [
        ("S1", "No weather controls", broad, f"g_va_sa_qoq ~ temp_c + {FE}"),
        ("S2", "Rainfall control", broad, f"g_va_sa_qoq ~ temp_c + rain_100mm + {FE}"),
        ("S3", "Baseline: rainfall and humidity", broad, primary),
        ("S4", "Pre-COVID: 1982-2019", broad[broad.year.lt(2020)].copy(), primary),
        ("S5", "Exclude 2020-2022", broad[~broad.year.between(2020, 2022)].copy(), primary),
        ("S6", "2005Q2+ broad sectors", broad[broad.quarter.ge(pd.Period("2005Q2"))].copy(), primary),
        ("S7", "2005Q2+ detailed sectors", detailed, primary),
        ("S8", "NSA quarterly log growth", broad, f"g_va_nsa_qoq ~ temp_c + {WEATHER} + {FE}"),
        ("S9", "Equal-weight monthly temperature", broad,
         f"g_va_sa_qoq ~ temp_c_unweighted + {WEATHER} + {FE}"),
    ]
    rows, all_coefficients, fitted = [], [], {}
    for model_id, label, df, formula in specs:
        df = df.reset_index(drop=True)
        model = fit(df, formula, label)
        temp = "temp_c_unweighted" if model_id == "S9" else "temp_c"
        terms = [temp] + [term for term in ["rain_100mm", "rh_pct"] if term in model.params.index]
        result = estimates(model, df, model_id, label, formula, terms)
        rows.extend(result)
        fitted[model_id] = (model, df, formula)
        all_coefficients.extend({"model": model_id, "term": term, "coefficient": val}
                                for term, val in model.params.items())
        print(f"{model_id} {label}: {temp}={result[0]['coefficient']:.4f}, SE={result[0]['std_error']:.4f}")
    base, df, formula = fitted["S3"]
    for lags, covariance in [(0, "Quarter cluster"), (8, "Driscoll-Kraay")]:
        rows.extend(estimates(base, df, "S3", "Baseline inference check", formula,
                              ["temp_c"], lags=lags, covariance=covariance))
    pd.DataFrame(rows).to_csv(OUT / "regression_results.csv", index=False)
    pd.DataFrame(all_coefficients).to_csv(OUT / "all_coefficients.csv", index=False)

    # Distributed lag, kept on one common sample for current-only comparison.
    lag_terms = ["temp_c"] + [f"temp_l{i}" for i in range(1, 5)]
    lag_sample = broad.dropna(subset=lag_terms).reset_index(drop=True)
    lag_formula = "g_va_sa_qoq ~ " + " + ".join(lag_terms) + f" + {WEATHER} + {FE}"
    lag_model = fit(lag_sample, lag_formula, "Distributed lag")
    lag_rows = estimates(lag_model, lag_sample, "L1", "Current plus four lagged quarters", lag_formula, lag_terms)
    current = fit(lag_sample, primary, "Current-only matched lag sample")
    lag_rows.extend(estimates(current, lag_sample, "L0", "Current only on matched lag sample", primary, ["temp_c"]))
    time = (lag_sample.time - lag_sample.time.min()).to_numpy(dtype=int)
    cov = cov_nw_groupsum(lag_model, nlags=4, time=time, use_correction="cluster")
    indices = [lag_model.model.exog_names.index(term) for term in lag_terms]
    cumulative = float(lag_model.params[lag_terms].sum())
    cum_se = float(np.sqrt(cov[np.ix_(indices, indices)].sum()))
    dof = lag_sample.quarter.nunique() - 1
    critical = student_t.ppf(.975, dof)
    cumulative_result = {"sum_lag_0_to_4": cumulative, "std_error": cum_se,
                         "ci_lower": cumulative - critical * cum_se,
                         "ci_upper": cumulative + critical * cum_se,
                         "p_value": 2 * student_t.sf(abs(cumulative / cum_se), dof),
                         "interpretation": "Sum of growth responses over five quarters to a one-quarter +1C pulse; also the contemporaneous fitted change if all five temperature exposures rise by 1C. Not a long-run causal climate effect."}
    cumulative_result["sector_cluster"] = cluster_sum(lag_model, lag_sample, lag_terms, "sum_lag_0_to_4")
    pd.DataFrame(lag_rows).to_csv(OUT / "distributed_lag_results.csv", index=False)
    (OUT / "distributed_lag_sum.json").write_text(json.dumps(cumulative_result, indent=2) + "\n")

    # Sector slopes in the same pooled panel; shared weather controls and year FE.
    sector_rows = sector_temperature_slopes(broad, "g_va_sa_qoq", WEATHER, FE,
                                           label="Pooled sector FE: temperature + industry interactions")
    pd.DataFrame(sector_rows).to_csv(OUT / "sector_slopes.csv", index=False)

    # Save the exact baseline estimation sample and its partial regression data.
    temp_residual = fit(df, f"temp_c ~ {WEATHER} + {FE}", "VA temperature partial regression").resid
    growth_residual = fit(df, f"g_va_sa_qoq ~ {WEATHER} + {FE}", "VA growth partial regression").resid
    partial = df[["quarter", "sector_id", "sector", "temp_c", "g_va_sa_qoq"]].copy()
    partial["temp_residual"] = temp_residual
    partial["growth_residual"] = growth_residual
    partial.to_csv(OUT / "baseline_partial_regression.csv", index=False)
    df.to_csv(OUT / "baseline_sample.csv", index=False)
    checks = {
        "baseline_n": len(df), "quarters": df.quarter.nunique(),
        "baseline_design_rank": int(np.linalg.matrix_rank(base.model.exog)),
        "baseline_parameters": int(base.model.exog.shape[1]),
        "temperature_raw_sd_c": float(df.drop_duplicates("quarter").temp_c.std()),
        "temperature_residual_sd_c": float(temp_residual.std()),
        "temperature_min_c": float(df.temp_c.min()), "temperature_max_c": float(df.temp_c.max()),
        "fwl_beta_difference": float(np.dot(temp_residual, growth_residual) / np.dot(temp_residual, temp_residual) - base.params["temp_c"]),
        "fully_saturated_quarter_fe_temp_residual_max": float((df.temp_c - df.groupby("quarter").temp_c.transform("mean")).abs().max()),
    }
    assert abs(checks["fwl_beta_difference"]) < 1e-9
    assert checks["fully_saturated_quarter_fe_temp_residual_max"] < 1e-12
    (OUT / "estimation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    estimate()
