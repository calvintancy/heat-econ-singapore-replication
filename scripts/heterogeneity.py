"""Estimate sector temperature slopes and joint differences on common outcome samples."""
from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, t as student_t
from statsmodels.stats.multitest import multipletests

from pyfixest_backend import fit_many_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/heterogeneity"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
ADDED = ["wind_mps", "pm25_10ug"]
ESTIMATOR_META = {"estimator": "PyFixest feols", "estimator_version": version("pyfixest")}
OUTCOMES = ["y_short", "y_yoy", "y_yoy_nsa", "y_log_level"]
OUTCOME_LABELS = {"y_short": "One-period log growth (SA)", "y_yoy": "YoY log growth (SA)",
                  "y_yoy_nsa": "YoY log growth (NSA)", "y_log_level": "100 × log output level (SA)",
                  "y_raw_level": "Output level (SA)"}
CONTEXTS = [("va", "quarterly", "quarter", "Q", 4),
            ("iip-monthly", "monthly", "month", "M", 12),
            ("iip-quarterly", "quarterly", "quarter", "Q", 4)]
LABELS = {"H1": "Temperature interactions; shared controls and year effects",
          "H2": "Sector weather responses; shared year effects",
          "H3": "Sector weather responses and sector year effects",
          "R1": "Relative slopes with date and sector year effects",
          "H3X": "Fully flexible: exclude 2020–2022 endpoint dates"}
PAIRS = {"va": [("1.1.1", "1.2.5"), ("1.1.2", "1.2.5"),
                 ("1.2.3", "1.2.5"), ("1.1.1", "1.1.2")],
         "iip": [("1.1", "1.2"), ("1.1", "1.3"), ("1.4", "1.5")]}


def read_common(key: str, control_set: str, period: str, freq: str) -> pd.DataFrame:
    sample = pd.read_csv(ROOT / f"outputs/outcomes/{key}/{control_set}/common_sample.csv", dtype={"sector_id": str})
    sample[period] = pd.PeriodIndex(sample[period], freq=freq)
    sample = sample.sort_values(["time", "sector_id"]).reset_index(drop=True)
    assert not sample.duplicated([period, "sector_id"]).any()
    assert sample.groupby(period).sector_id.nunique().eq(sample.sector_id.nunique()).all()
    assert sample[OUTCOMES].notna().all().all()
    return sample


def design(sample: pd.DataFrame, terms: list[str], model: str):
    """Explicit full-rank nuisance and temperature columns for a balanced panel."""
    sectors = sorted(sample.sector_id.unique())
    sector = np.column_stack([sample.sector_id.eq(s) for s in sectors]).astype(float)
    season = pd.get_dummies(sample.season, dtype=float).to_numpy()
    year = pd.get_dummies(sample.year, drop_first=True, dtype=float).to_numpy()
    centered = sample[terms]-sample[terms].mean()
    weather = centered.to_numpy()
    relative = model == "R1"
    active = sector[:, 1:] if relative else sector
    sector_ids = sectors[1:] if relative else sectors
    nuisance = []
    names = []
    if relative:
        nuisance.append(pd.get_dummies(sample.time, dtype=float).to_numpy())
        names += [f"date[{time}]" for time in sorted(sample.time.unique())]
    nuisance.append((active[:, :, None]*season[:, None, :]).reshape(len(sample), -1))
    names += [f"sector_season[{s},{season_id}]" for s in sector_ids for season_id in sorted(sample.season.unique())]
    if model in {"H3", "H3X", "R1"}:
        nuisance.append((active[:, :, None]*year[:, None, :]).reshape(len(sample), -1))
        names += [f"sector_year[{s},{yr}]" for s in sector_ids for yr in sorted(sample.year.unique())[1:]]
    else:
        nuisance.append(year)
        names += [f"year[{yr}]" for yr in sorted(sample.year.unique())[1:]]
    if model == "H1":
        nuisance.append(weather[:, 1:])
        names += terms[1:]
    else:
        nuisance.append((active[:, :, None]*weather[:, None, 1:]).reshape(len(sample), -1))
        names += [f"{term}[{s}]" for s in sector_ids for term in terms[1:]]
    z = np.column_stack(nuisance)
    a = active*weather[:, [0]]
    assert z.shape[1] == len(names)
    return z, a, sector_ids, names, centered.mean().to_dict()


def project(sample: pd.DataFrame, terms: list[str], model: str,
            outcomes: list[str] | None = None, label: str | None = None) -> dict:
    """Fit all coefficients/residuals with PyFixest; retain FWL for covariance.

    The absolute-slope basis ``a`` is useful for the existing calendar covariance.
    Actual estimation instead uses a temperature main effect plus nonreference
    interactions, except R1 where date effects absorb the common main effect.
    """
    outcomes = OUTCOMES if outcomes is None else outcomes
    z, a, sectors, names, _ = design(sample, terms, model)
    q, r = np.linalg.qr(z, mode="reduced")
    assert np.linalg.matrix_rank(r) == z.shape[1], f"Nuisance rank failure: {model}"
    ar = a-q@(q.T@a)
    assert np.linalg.matrix_rank(ar) == a.shape[1], f"Temperature rank failure: {model}"
    bread = np.linalg.inv(ar.T@ar)
    reference = sorted(sample.sector_id.unique())[0]
    if model == "R1":
        temperature = a
        component_names = [f"temp_c:C(sector_id)[T.{sector}]" for sector in sectors]
        to_absolute = np.eye(len(sectors))
        to_component = np.eye(len(sectors))
    else:
        temperature = np.column_stack([a.sum(axis=1), a[:, 1:]])
        component_names = ["temp_c"]+[f"temp_c:C(sector_id)[T.{sector}]" for sector in sectors[1:]]
        to_absolute = np.eye(len(sectors))
        to_absolute[:, 0] = 1
        to_component = np.eye(len(sectors))
        to_component[1:, 0] = -1
    full_x = np.column_stack([z, temperature])
    fitted = fit_many_matrix(sample[outcomes].to_numpy(), full_x, names+component_names,
                             label or f"Sector panel {model}", outcome_names=outcomes)
    native_params = np.column_stack([fit.params.to_numpy() for fit in fitted])
    nuisance_beta = native_params[:z.shape[1]]
    component_beta = native_params[z.shape[1]:]
    beta = to_absolute@component_beta
    residual = np.column_stack([fit.resid for fit in fitted])
    k = z.shape[1]+a.shape[1]
    assert k < len(sample)
    t = sample.time.nunique()
    return {"beta": beta, "residual": residual, "bread": bread, "ar": ar,
            "nuisance_beta": nuisance_beta, "nuisance_names": names,
            "component_beta": component_beta, "component_names": component_names,
            "component_to_absolute": to_absolute, "absolute_to_component": to_component,
            "component_sectors": sectors, "reference_sector_id": reference,
            "temperature_center_c": float(sample.temp_c.mean()), "relative": model == "R1",
            "k": k, "t": t, "n": len(sample), "sectors": sectors,
            "r_squared": np.asarray([fit.rsquared for fit in fitted])}


def covariance(result: dict, sample: pd.DataFrame, outcome_index: int, bandwidth: int) -> np.ndarray:
    time = (sample.time-sample.time.min()).to_numpy(dtype=int)
    scores = np.zeros((time.max()+1, result["ar"].shape[1]))
    np.add.at(scores, time, result["ar"]*result["residual"][:, [outcome_index]])
    meat = scores.T@scores
    for lag in range(1, bandwidth+1):
        cross = scores[lag:].T@scores[:-lag]
        meat += (1-lag/(bandwidth+1))*(cross+cross.T)
    correction = result["t"]/(result["t"]-1)*(result["n"]-1)/(result["n"]-result["k"])
    absolute_cov = result["bread"]@meat@result["bread"]*correction
    # Express the joint temperature block in the fitted main/deviation basis,
    # then propagate it to βT+δi totals. This includes the covariance cross term;
    # adding the two component variances alone would give the wrong SE.
    component_cov = result["absolute_to_component"]@absolute_cov@result["absolute_to_component"].T
    v = result["component_to_absolute"]@component_cov@result["component_to_absolute"].T
    return (v+v.T)/2


def wald(beta: np.ndarray, cov: np.ndarray, restrictions: np.ndarray, periods: int,
         rank_bound: int | None = None) -> dict:
    diff = restrictions@beta
    v = restrictions@cov@restrictions.T
    singular = np.linalg.svd(v, compute_uv=False)
    rank = int(np.sum(singular > singular.max()*1e-10))
    if rank_bound is not None:
        rank = min(rank, rank_bound)
    count = len(diff)
    status = "available" if rank == count else "unavailable: singular restriction covariance"
    result = {"restrictions": count, "covariance_rank": int(rank), "status": status,
              "wald_chi2": None, "f_statistic": None, "p_value": None,
              "numerator_df": count, "denominator_df": periods-1, "structural_rank_bound": rank_bound,
              "covariance_condition": float(np.linalg.cond(v))}
    if rank == count:
        statistic = float(diff@np.linalg.solve(v, diff))
        assert statistic >= -1e-8
        result.update(wald_chi2=max(statistic, 0), f_statistic=max(statistic, 0)/count,
                      p_value=float(f_dist.sf(max(statistic, 0)/count, count, periods-1)))
    return result


def inference(beta: float, variance: float, periods: int) -> dict:
    assert variance >= -1e-10
    se = float(np.sqrt(max(variance, 0)))
    critical = student_t.ppf(.975, periods-1)
    return {"coefficient": float(beta), "std_error": se,
            "ci_lower": float(beta-critical*se), "ci_upper": float(beta+critical*se),
            "p_value": float(2*student_t.sf(abs(beta/se), periods-1)) if se > 0 else None}


def component_records(result: dict, cov: np.ndarray, outcome_index: int,
                      meta: dict, names: dict) -> tuple[list[dict], list[dict]]:
    """Expose native βT, estimated deviations and their joint HAC covariance."""
    transform = result["absolute_to_component"]
    component_cov = transform@cov@transform.T
    component_cov = (component_cov+component_cov.T)/2
    component_beta = result["component_beta"][:, outcome_index]
    components = ["interaction"]*len(component_beta)
    if not result["relative"]:
        components[0] = "base"
    rows, covariances = [], []
    for i, (sector, term, component) in enumerate(zip(result["component_sectors"], result["component_names"], components, strict=True)):
        rows.append({**meta, "reference_sector_id": result["reference_sector_id"],
                     "sector_id": sector, "sector": names[sector], "term": term, "component": component,
                     "temperature_center_c": result["temperature_center_c"],
                     **ESTIMATOR_META, "estimated": True, "normalization": False,
                     "estimand": "Reference-sector temperature slope" if component == "base" else "Difference from reference-sector temperature slope",
                     **inference(component_beta[i], component_cov[i, i], result["t"])})
        for j, other in enumerate(result["component_sectors"]):
            covariances.append({"model": meta["model"], "outcome": meta["outcome"], "hac_lags": meta["hac_lags"],
                                "reference_sector_id": result["reference_sector_id"],
                                "term_i": term, "term_j": result["component_names"][j],
                                "component_i": component, "component_j": components[j],
                                "sector_i": sector, "sector_j": other, "covariance": float(component_cov[i, j]),
                                **ESTIMATOR_META})
    if not result["relative"]:
        reference = result["reference_sector_id"]
        rows.append({**meta, **ESTIMATOR_META, "reference_sector_id": reference,
                     "sector_id": reference, "sector": names[reference],
                     "term": f"temp_c:C(sector_id)[T.{reference}]", "component": "interaction",
                     "temperature_center_c": result["temperature_center_c"],
                     "estimated": False, "normalization": True,
                     "estimand": "Reference interaction fixed to zero by normalization; not estimated",
                     "coefficient": 0.0, "std_error": None, "ci_lower": None, "ci_upper": None, "p_value": None})
    return rows, covariances


def fit_context(key: str, frequency: str, period: str, freq: str, horizon: int, control_set: str) -> dict:
    sample = read_common(key, control_set, period, freq)
    terms = BASE+(ADDED if control_set == "full" else [])
    folder = OUT / key / control_set
    folder.mkdir(parents=True, exist_ok=True)
    sample.to_csv(folder / "common_sample.csv", index=False)
    names = sample.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    sectors = sorted(names)
    reference = sectors[0]
    pairs = PAIRS["va" if key == "va" else "iip"]
    rows, joint, comparisons, covariances, coefficients, separate, separate_weather, checks, persistence = [], [], [], [], [], [], [], [], []
    components, component_covariances = [], []
    fitted = {}
    for model in ["H1", "H2", "H3", "R1", "H3X"]:
        df = sample[~sample.year.between(2020, 2022)].reset_index(drop=True) if model == "H3X" else sample
        result = project(df, terms, model, label=f"Heterogeneity {key}/{control_set}/{model}")
        time_df = result["t"]-(df.season.nunique()+df.year.nunique()-1+len(terms))
        fitted[model] = (result, df)
        print(f"Heterogeneity {key}/{control_set}/{model}: {result['n']} rows, {result['t']} dates, {result['k']} parameters", flush=True)
        for j, outcome in enumerate(OUTCOMES):
            for i, name in enumerate(result["nuisance_names"]):
                coefficients.append({"model": model, "outcome": outcome, "term": name,
                                     "coefficient": float(result["nuisance_beta"][i, j]), **ESTIMATOR_META})
            for bandwidth in [horizon, 2*horizon]:
                cov = covariance(result, df, j, bandwidth)
                beta = result["beta"][:, j]
                meta = {"model": model, "label": LABELS[model], "outcome": outcome, "outcome_label": OUTCOME_LABELS[outcome],
                        "control_set": control_set, "hac_lags": bandwidth,
                        "n_obs": result["n"], "n_sectors": len(sectors), "n_periods": result["t"],
                        "n_parameters": result["k"], "r_squared": float(result["r_squared"][j]),
                        "fully_flexible_time_df": int(time_df),
                        "start": str(df[period].min()), "end": str(df[period].max()),
                        "reference_sector_id": reference, "reference_sector": names[reference],
                        "estimand": "Difference from reference sector" if model == "R1" else "Absolute sector slope",
                        "covariance": "Driscoll-Kraay", "inference_df": result["t"]-1, **ESTIMATOR_META}
                component_rows, component_cov = component_records(result, cov, j, meta, names)
                components.extend(component_rows)
                component_covariances.extend(component_cov)
                for i, sector_id in enumerate(result["sectors"]):
                    rows.append({**meta, "sector_id": sector_id, "sector": names[sector_id],
                                 **inference(beta[i], cov[i, i], result["t"])})
                    for k, other in enumerate(result["sectors"]):
                        covariances.append({"model": model, "outcome": outcome, "hac_lags": bandwidth,
                                            "sector_i": sector_id, "sector_j": other, "covariance": float(cov[i, k]), **ESTIMATOR_META})
                if model == "R1":
                    equality = np.eye(len(beta))
                else:
                    equality = np.eye(len(beta))[1:]-np.eye(len(beta))[[0]]
                bound = int(time_df) if model in {"H3", "H3X", "R1"} else result["t"]-1
                joint.append({**meta, "test": "All sector temperature slopes equal",
                              **wald(beta, cov, equality, result["t"], rank_bound=bound)})
                for first, second in pairs:
                    weights = np.zeros(len(beta))
                    for sector_id, value in [(first, 1), (second, -1)]:
                        if sector_id in result["sectors"]:
                            weights[result["sectors"].index(sector_id)] = value
                    comparisons.append({**meta, "first_sector_id": first, "second_sector_id": second,
                                        "comparison": f"{names[first]} − {names[second]}",
                                        **inference(float(weights@beta), float(weights@cov@weights), result["t"])})
            residual_means = pd.DataFrame({"sector_id": df.sector_id, "time": df.time,
                                           "residual": result["residual"][:, j]})
            for sector_id, group in residual_means.groupby("sector_id"):
                series = group.set_index("time").residual
                for lag in [1, horizon]:
                    aligned = pd.concat([series, series.rename("lagged").set_axis(series.index+lag)], axis=1).dropna()
                    persistence.append({"model": model, "outcome": outcome, "sector_id": sector_id,
                                        "sector": names[sector_id], "lag": lag, "matched_pairs": len(aligned),
                                        "residual_correlation": float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))})

    # Direct independent-series fits: same point slopes as H3, without imposing
    # common sector responses. Individual-series HAC uses its own K correction.
    for sector_id, df in sample.groupby("sector_id", sort=True):
        weather = df[terms]-df[terms].mean()
        x = np.column_stack([np.ones(len(df)), weather,
                             pd.get_dummies(df.season, drop_first=True, dtype=float),
                             pd.get_dummies(df.year, drop_first=True, dtype=float)])
        assert np.linalg.matrix_rank(x) == x.shape[1]
        inverse = np.linalg.inv(x.T@x)
        coef_names = (["Intercept"]+terms+[f"season[{season}]" for season in sorted(df.season.unique())[1:]]
                      +[f"year[{year}]" for year in sorted(df.year.unique())[1:]])
        native = fit_many_matrix(df[OUTCOMES].to_numpy(), x, coef_names,
                                 f"Separate sector {key}/{control_set}/{sector_id}", outcome_names=OUTCOMES)
        beta = np.column_stack([fit.params.to_numpy() for fit in native])
        error = np.column_stack([fit.resid for fit in native])
        h3, _ = fitted["H3"]
        sector_index = h3["sectors"].index(sector_id)
        assert np.allclose(beta[1], h3["beta"][sector_index], atol=1e-8, rtol=0)
        time = (df.time-df.time.min()).to_numpy(dtype=int)
        for j, outcome in enumerate(OUTCOMES):
            for bandwidth in [horizon, 2*horizon]:
                scores = np.zeros((time.max()+1, x.shape[1]))
                scores[time] = x*error[:, [j]]
                meat = scores.T@scores
                for lag in range(1, bandwidth+1):
                    cross = scores[lag:].T@scores[:-lag]
                    meat += (1-lag/(bandwidth+1))*(cross+cross.T)
                v = inverse@meat@inverse*len(df)/(len(df)-x.shape[1])
                for term_index, term in enumerate(terms, start=1):
                    separate_weather.append({"outcome": outcome, "sector_id": sector_id, "sector": names[sector_id],
                                             "outcome_label": OUTCOME_LABELS[outcome],
                                             **ESTIMATOR_META,
                                             "term": term, "hac_lags": bandwidth, "n_obs": len(df),
                                             "n_parameters": x.shape[1], "residual_df": len(df)-x.shape[1],
                                             "r_squared": float(native[j].rsquared),
                                             **inference(beta[term_index, j], v[term_index, term_index], len(df))})
                separate.append({"outcome": outcome, "sector_id": sector_id, "sector": names[sector_id],
                                 "outcome_label": OUTCOME_LABELS[outcome],
                                 **ESTIMATOR_META,
                                 "hac_lags": bandwidth, "n_obs": len(df), "n_parameters": x.shape[1],
                                 "residual_df": len(df)-x.shape[1],
                                 "pooled_H3_coefficient": float(h3["beta"][sector_index, j]),
                                 "point_estimate_difference": float(beta[1, j]-h3["beta"][sector_index, j]),
                                 **inference(beta[1, j], v[1, 1], len(df))})

    # Full-date effects identify only relative weather slopes. With identical
    # balanced designs these equal H3 differences, rather than a new robustness fit.
    h3, _ = fitted["H3"]
    r1, _ = fitted["R1"]
    contrast = np.eye(len(sectors))[1:]-np.eye(len(sectors))[[0]]
    beta_error = float(np.max(np.abs(r1["beta"]-contrast@h3["beta"])))
    assert beta_error < 1e-8
    for j, outcome in enumerate(OUTCOMES):
        v_h3 = covariance(h3, sample, j, horizon)
        v_r1 = covariance(r1, sample, j, horizon)
        factor_h3 = (len(sample)-1)/(len(sample)-h3["k"])
        factor_r1 = (len(sample)-1)/(len(sample)-r1["k"])
        cov_error = float(np.max(np.abs(v_r1/factor_r1-contrast@v_h3@contrast.T/factor_h3)))
        assert cov_error < 1e-7
        checks.append({"outcome": outcome, "reference": names[reference], "R1_H3_max_beta_difference": beta_error,
                       "uncorrected_contrast_covariance_max_difference": cov_error,
                       "H3_parameters": h3["k"], "R1_parameters": r1["k"]})
    pair_frame = pd.DataFrame(comparisons)
    pair_frame["p_value_holm"] = pair_frame.groupby(["model", "outcome", "hac_lags"])["p_value"].transform(
        lambda p: multipletests(p, method="holm")[1])
    for filename, frame in [("sector_slopes.csv", pd.DataFrame(rows)), ("joint_tests.csv", pd.DataFrame(joint)),
                            ("pairwise_tests.csv", pair_frame), ("slope_covariances.csv", pd.DataFrame(covariances)),
                            ("interaction_components.csv", pd.DataFrame(components)),
                            ("component_covariances.csv", pd.DataFrame(component_covariances)),
                            ("nuisance_coefficients.csv", pd.DataFrame(coefficients)),
                            ("separate_series.csv", pd.DataFrame(separate)), ("relative_equivalence.csv", pd.DataFrame(checks)),
                            ("separate_sector_coefficients.csv", pd.DataFrame(separate_weather)),
                            ("residual_persistence.csv", pd.DataFrame(persistence))]:
        frame.to_csv(folder / filename, index=False)
    return {"start": str(sample[period].min()), "end": str(sample[period].max()),
            "periods": sample[period].nunique(), "sectors": len(sectors), "observations": len(sample),
            "reference_sector_id": reference, "reference_sector": names[reference],
            "relative_equivalence_max_beta_error": beta_error}


def run_heterogeneity() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, frequency, period, freq, horizon in CONTEXTS:
        summary[key] = {controls: fit_context(key, frequency, period, freq, horizon, controls)
                        for controls in ["weather", "full"]}
    (OUT / "samples.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    run_heterogeneity()
