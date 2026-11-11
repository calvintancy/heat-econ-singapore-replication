"""Requested H1 sector-cluster sensitivity beside the retained calendar DK model.

This is an additional diagnostic, not a replacement for sector-profile DK
inference. Each sector-specific temperature regressor belongs to one cluster.
Shared nuisance parameters can yield a nonzero absolute-slope CR1 covariance,
while sector-difference covariance collapses. Numerical zeros are exposed and
never turned into artificial p-values or zero-width confidence intervals.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution, t as student_t
from threadpoolctl import threadpool_limits

import pyfixest_backend as native
from heterogeneity import covariance, project
from significance import p_format, significance_label, significance_stars

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/clustered_sector_sensitivity"
REPORT = ROOT / "outputs/clustered_sector_sensitivity.html"
YS = ["y_short", "y_yoy_nsa", "y_yoy", "y_log_level", "y_raw_level"]
Y_LABELS = {"y_short": "One-period log growth (SA)", "y_yoy_nsa": "YoY log growth (NSA)",
            "y_yoy": "YoY log growth (SA)", "y_log_level": "100 × log output level (SA; descriptive)",
            "y_raw_level": "Raw output level (SA; descriptive)"}
CONTEXTS = {"va": "11 broad VA sectors", "iip-monthly": "6 manufacturing clusters · monthly",
            "iip-quarterly": "6 manufacturing clusters · quarterly", "va-detailed": "19 detailed VA sectors",
            "iip-monthly-detailed": "18 manufacturing industries · monthly",
            "iip-quarterly-detailed": "18 manufacturing industries · quarterly"}
WINDOWS = {"full": "Baseline · rainfall, humidity, wind + PM2.5",
           "exclude": "Same full controls · excluding 2020–2022 endpoints",
           "long": "Long-history robustness · rainfall + humidity"}
ZERO_RELATIVE_TOLERANCE = 1e-12
RANK_RELATIVE_TOLERANCE = 1e-10
META = {"estimator": "PyFixest feols", "estimator_version": native.ESTIMATOR_VERSION}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def source_sample(context: str, window: str) -> tuple[pd.DataFrame, Path, Path, int, str]:
    detailed = context.endswith("-detailed")
    monthly = "monthly" in context
    period, horizon = ("month", 12) if monthly else ("quarter", 4)
    if detailed:
        folder = ROOT / "outputs/granular_interactions" / context / window
        source = folder / "common_sample.csv"
    else:
        folder = ROOT / "outputs/heterogeneity" / context / ("weather" if window == "long" else "full")
        source = folder / "common_sample.csv"
    sample = pd.read_csv(source, dtype={"sector_id": str})
    if window == "exclude" and not detailed:
        sample = sample.loc[~sample.year.between(2020, 2022)].copy()
    sample = sample.sort_values(["time", "sector_id"]).reset_index(drop=True)
    assert not sample.duplicated(["time", "sector_id"]).any()
    assert sample.groupby("time").sector_id.nunique().eq(sample.sector_id.nunique()).all()
    assert len(sample) == sample.time.nunique()*sample.sector_id.nunique()
    if window == "exclude":
        assert not sample.year.between(2020, 2022).any()
    return sample, source, folder / "sector_slopes.csv", horizon, period


def uncertainty(beta: float, variance: float, reference_variance: float, degrees: int) -> dict:
    threshold = ZERO_RELATIVE_TOLERANCE*max(float(reference_variance), 0.)
    assert variance >= -max(threshold, np.finfo(float).tiny), (variance, threshold)
    degenerate = bool(abs(variance) <= threshold or reference_variance <= 0)
    se = float(np.sqrt(max(variance, 0.)))
    critical = float(student_t.ppf(.975, degrees))
    return {"coefficient": float(beta), "variance": float(variance), "std_error": se,
            "ci_lower": None if degenerate else float(beta-critical*se),
            "ci_upper": None if degenerate else float(beta+critical*se),
            "p_value": None if degenerate else float(2*student_t.sf(abs(beta/se), degrees)),
            "inference_status": "unavailable: numerically degenerate variance" if degenerate else "available",
            "degenerate": degenerate, "hc0_reference_variance": float(reference_variance),
            "variance_ratio_to_hc0": float(variance/reference_variance) if reference_variance > 0 else None,
            "zero_variance_threshold": threshold, "zero_relative_tolerance": ZERO_RELATIVE_TOLERANCE}


def joint_test(beta: np.ndarray, matrix: np.ndarray, hc0: np.ndarray, degrees: int, rank_bound: int) -> dict:
    restrictions = np.eye(len(beta))[1:]-np.eye(len(beta))[[0]]
    difference = restrictions@beta
    v = restrictions@matrix@restrictions.T
    v = (v+v.T)/2
    reference = restrictions@hc0@restrictions.T
    reference_scale = float(np.linalg.norm(reference, 2))
    singular = np.linalg.svd(v, compute_uv=False)
    zero_threshold = ZERO_RELATIVE_TOLERANCE*reference_scale
    cutoff = max(float(singular.max())*RANK_RELATIVE_TOLERANCE, zero_threshold)
    rank = min(int(np.sum(singular > cutoff)), rank_bound)
    degenerate = bool(float(singular.max()) <= zero_threshold)
    count = len(difference)
    status = "unavailable: numerically degenerate sector-difference covariance" if degenerate else (
        "unavailable: singular restriction covariance" if rank < count else "available")
    row = {"test": "All sector temperature slopes equal", "restrictions": count, "covariance_rank": rank,
           "structural_rank_bound": rank_bound, "reference_covariance_scale": reference_scale,
           "zero_covariance_threshold": zero_threshold, "rank_cutoff": cutoff, "degenerate": degenerate,
           "status": status, "numerator_df": count, "denominator_df": degrees,
           "wald_chi2": None, "f_statistic": None, "p_value": None,
           "restriction_covariance_max_abs": float(np.max(np.abs(v)))}
    if status == "available":
        statistic = float(difference@np.linalg.solve(v, difference))
        assert statistic >= -1e-8
        row.update(wald_chi2=max(statistic, 0.), f_statistic=max(statistic, 0.)/count,
                   p_value=float(f_distribution.sf(max(statistic, 0.)/count, count, degrees)))
    return row


def existing_dk(context: str, window: str, outcome: str, source: Path, paired: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if not context.endswith("-detailed") and window == "exclude":
        selected = paired[paired.context.eq(context)&paired.family.eq("sector_FE_H1")&
                          paired.source_kind.eq("broad_FE")&paired.control_set.eq("full")&
                          paired.sample_variant.eq("exclude_2020_2022")&paired.outcome.eq(outcome)]
        path = "outputs/covid_comparison/sector/estimates.csv"
    else:
        frame = pd.read_csv(source, dtype={"sector_id": str})
        selected = frame[frame.model.eq("H1")&frame.outcome.eq(outcome)]
        path = str(source.relative_to(ROOT))
    return selected, path if len(selected) else ""


def run_clustered_sector_sensitivity() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    (OUT / "designs").mkdir(exist_ok=True)
    rows, components, component_covariances, covariances, joints, coefficients, specifications = [], [], [], [], [], [], []
    summary, protected = [], {}
    paired_path = ROOT / "outputs/covid_comparison/sector/estimates.csv"
    paired = pd.read_csv(paired_path, dtype={"sector_id": str})
    protected[str(paired_path.relative_to(ROOT))] = digest(paired_path)
    for name in ["outputs/pyfixest/fit_log.csv", "outputs/pyfixest/estimator_manifest.json",
                 "outputs/pyfixest/migration_input_reference.json", "outputs/covid_comparison/original_inputs_reference.json"]:
        protected[name] = digest(ROOT / name)
    old_state = (native.LOG_DIR, list(native._FIT_LOG), native._API_COUNTER, native._RUN_STARTED)
    native.LOG_DIR = OUT / "pyfixest"
    native.reset_fit_log()
    try:
        with threadpool_limits(limits=1):
            for context in CONTEXTS:
                for window in WINDOWS:
                    sample_id = f"{context}__{window}"
                    sample, source, source_dk, horizon, period = source_sample(context, window)
                    protected[str(source.relative_to(ROOT))] = digest(source)
                    protected[str(source_dk.relative_to(ROOT))] = digest(source_dk)
                    terms = ["temp_c", "rain_100mm", "rh_pct"]+(["wind_mps", "pm25_10ug"] if window != "long" else [])
                    outcomes = [outcome for outcome in YS if outcome in sample]
                    assert outcomes and np.isfinite(sample[outcomes+terms]).all().all()
                    sample_path = OUT / "samples" / f"{sample_id}.csv"
                    sample.to_csv(sample_path, index=False)
                    native_label = f"Clustered sector sensitivity {sample_id}"
                    result = project(sample, terms, "H1", outcomes=outcomes, label=native_label)
                    sectors = result["sectors"]
                    names = sample.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
                    group = pd.Categorical(sample.sector_id, categories=sectors).codes
                    g, n, k, periods = len(sectors), result["n"], result["k"], result["t"]
                    correction = g/(g-1)*(n-1)/(n-k)
                    to_component = result["absolute_to_component"]
                    np.savez_compressed(OUT / "designs" / f"{sample_id}.npz",
                        beta=result["beta"], component_beta=result["component_beta"], nuisance_beta=result["nuisance_beta"],
                        residual=result["residual"], ar=result["ar"], bread=result["bread"],
                        sectors=np.asarray(sectors, str), outcomes=np.asarray(outcomes, str))
                    for j, outcome in enumerate(outcomes):
                        beta = result["beta"][:, j]
                        error = result["residual"][:, j]
                        influence = (result["ar"]@result["bread"])*error[:, None]
                        cluster_scores = np.zeros((g, g))
                        np.add.at(cluster_scores, group, influence)
                        cluster_cov = cluster_scores.T@cluster_scores*correction
                        cluster_cov = (cluster_cov+cluster_cov.T)/2
                        hc0 = influence.T@influence
                        hc0 = (hc0+hc0.T)/2
                        reference_scale = float(np.linalg.norm(hc0, 2))
                        singular = np.linalg.svd(cluster_cov, compute_uv=False)
                        absolute_rank = int(np.sum(singular > max(float(singular.max())*RANK_RELATIVE_TOLERANCE,
                                                                  ZERO_RELATIVE_TOLERANCE*reference_scale)))
                        existing, existing_path = existing_dk(context, window, outcome, source_dk, paired)
                        spec = {"sample_id": sample_id, "context": context, "context_label": CONTEXTS[context], "window": window,
                                "label": WINDOWS[window], "model": "H1", "outcome": outcome, "outcome_label": Y_LABELS[outcome],
                                "source_sample_csv": str(source.relative_to(ROOT)), "parent_sample_sha256": digest(source),
                                "sample_csv": str(sample_path.relative_to(ROOT)), "sample_sha256": digest(sample_path),
                                "source_dk_csv": existing_path, "terms": json.dumps(terms), "period_column": period,
                                "frequency": "monthly" if horizon == 12 else "quarterly", "n_obs": n, "n_periods": periods,
                                "n_parameters": k, "n_sectors": g, "reference_sector_id": sectors[0],
                                "native_label": native_label, "r_squared": float(result["r_squared"][j]),
                                "mean_y_all": float(sample[outcome].mean()), "sd_y_all": float(sample[outcome].std()),
                                "start": str(sample[period].min()), "end": str(sample[period].max()),
                                "formula": outcome+" ~ temp_c * sector + shared controls + sector×season FE + year FE",
                                "finite_sample_correction": correction, "cluster_absolute_covariance_rank": absolute_rank,
                                "cluster_absolute_reference_covariance_scale": reference_scale, **META}
                        specifications.append(spec)
                        for name, value in zip(result["nuisance_names"]+result["component_names"],
                                               np.r_[result["nuisance_beta"][:, j], result["component_beta"][:, j]], strict=True):
                            coefficients.append({"sample_id": sample_id, "outcome": outcome, "term": name,
                                                 "coefficient": float(value), **META})
                        bands = [("Sector cluster", 0, cluster_cov, g-1, g-1),
                                 *(('Driscoll-Kraay', band, covariance(result, sample, j, band), periods-1,
                                    periods-sample.season.nunique()-1) for band in [horizon, 2*horizon])]
                        for method, band, matrix, degrees, bound in bands:
                            meta = {**spec, "covariance": method, "hac_lags": band, "inference_df": degrees,
                                    "n_clusters": g if method == "Sector cluster" else 0,
                                    "shared_nuisance_limitation": method == "Sector cluster"}
                            joint = {**meta, **joint_test(beta, matrix, hc0, degrees, int(bound))}
                            joints.append(joint)
                            component_cov = to_component@matrix@to_component.T
                            component_hc0 = to_component@hc0@to_component.T
                            component_cov = (component_cov+component_cov.T)/2
                            for i, sector in enumerate(sectors):
                                unit = sample.loc[sample.sector_id.eq(sector)]
                                raw_scores = (unit.temp_c-sample.temp_c.mean()).to_numpy()*error[group == i]
                                score_sum, abs_sum = float(raw_scores.sum()), float(np.abs(raw_scores).sum())
                                inference = uncertainty(beta[i], matrix[i, i], hc0[i, i], degrees)
                                row = {**meta, "sector_id": sector, "sector": names[sector], "mean_y": float(unit[outcome].mean()),
                                       "sd_y": float(unit[outcome].std()), "temp_main": float(result["component_beta"][0, j]),
                                       "interaction_delta": 0. if i == 0 else float(result["component_beta"][i, j]),
                                       "reference_delta_normalization": i == 0, "var_temp_main": float(component_cov[0, 0]),
                                       "var_interaction_delta": 0. if i == 0 else float(component_cov[i, i]),
                                       "cov_temp_main_delta": 0. if i == 0 else float(component_cov[0, i]),
                                       "own_sector_temperature_score_sum": score_sum, "own_sector_temperature_score_abs_sum": abs_sum,
                                       "own_sector_temperature_score_relative": abs(score_sum)/abs_sum if abs_sum > 0 else 0., **inference}
                                if method == "Driscoll-Kraay" and len(existing):
                                    target = existing[existing.sector_id.eq(sector)&existing.hac_lags.eq(band)]
                                    assert len(target) == 1, (sample_id, outcome, sector, band)
                                    for field in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value"]:
                                        assert np.isclose(row[field], target.iloc[0][field], atol=2e-7, rtol=3e-7), (sample_id, outcome, field)
                                rows.append(row)
                                components.append({**meta, "sector_id": sector, "sector": names[sector],
                                    "term": result["component_names"][i], "component": "base" if i == 0 else "interaction",
                                    "estimated": True, "normalization": False,
                                    **uncertainty(result["component_beta"][i, j], component_cov[i, i], component_hc0[i, i], degrees)})
                                for h, other in enumerate(sectors):
                                    key = {"sample_id": sample_id, "outcome": outcome, "covariance": method, "hac_lags": band,
                                           "sector_i": sector, "sector_j": other}
                                    covariances.append({**key, "covariance_value": float(matrix[i, h]), "hc0_reference_value": float(hc0[i, h])})
                                    component_covariances.append({**key, "term_i": result["component_names"][i],
                                        "term_j": result["component_names"][h], "covariance_value": float(component_cov[i, h]),
                                        "hc0_reference_value": float(component_hc0[i, h])})
                            components.append({**meta, "sector_id": sectors[0], "sector": names[sectors[0]],
                                "term": f"temp_c:C(sector_id)[T.{sectors[0]}]", "component": "interaction",
                                "estimated": False, "normalization": True, "coefficient": 0., "variance": None,
                                "std_error": None, "ci_lower": None, "ci_upper": None, "p_value": None,
                                "inference_status": "Reference interaction fixed to zero by normalization; not estimated", "degenerate": False})
                        summary.append({"sample_id": sample_id, "outcome": outcome, "n_sectors": g, "n_periods": periods,
                                        "cluster_absolute_covariance_rank": absolute_rank,
                                        "cluster_equal_slopes_test_status": joints[-3]["status"], "existing_DK_parent": bool(len(existing))})
                    print(f"Sector-cluster H1 sensitivity: {sample_id}, {g} sectors × {periods} periods, {len(outcomes)} outcomes", flush=True)
        receipt = native.write_fit_log()
    finally:
        native.LOG_DIR, previous_records, native._API_COUNTER, native._RUN_STARTED = old_state
        native._FIT_LOG[:] = previous_records
    for name, values in [("specifications", specifications), ("sector_slopes", rows), ("slope_covariances", covariances),
                         ("interaction_components", components), ("component_covariances", component_covariances),
                         ("joint_tests", joints), ("all_coefficients", coefficients), ("summary", summary)]:
        pd.DataFrame(values).to_csv(OUT / f"{name}.csv", index=False)
    assert all(digest(ROOT / name) == expected for name, expected in protected.items()), "Existing source/output changed"
    manifest = {"status": "estimated", "model_outcomes": len(specifications), "native_fits": receipt["native_fits_recorded"],
        "samples": len({row["sample_id"] for row in specifications}), "sector_slope_rows": len(rows),
        "existing_DK_parent_model_outcomes": sum(row["existing_DK_parent"] for row in summary),
        "protected_sources_sha256": protected, "formula": "Temperature main effect plus sector interactions; shared other-weather slopes and year effects; sector×season FE",
        "cluster_inference": "CR1 G/(G−1)×(N−1)/(N−K), t(G−1); K includes every explicit FE/nuisance/temperature parameter",
        "DK_inference": "Existing calendar-aware Bartlett Driscoll–Kraay h/2h, t(T−1); same coefficients and exact calendar",
        "zero_criterion": "abs(variance) <= 1e−12 × unscaled independent-observation HC0 reference variance; retain raw SE but suppress p/CI",
        "joint_rank_criterion": "singular values > max(largest×1e−10, 1e−12×spectral norm of contrasted HC0); cap at G−1 (CR1) or T−Q−1 (DK)",
        "limitation": "Each sector-specific temperature coefficient is supported by one sector cluster. Shared nuisance estimates can leave a nonzero, often rank-one absolute CR1 covariance; that does not create independent cluster replication for each slope. Sector-difference CR1 covariance can collapse.",
        "stationarity": "No new stationarity clearance or sample filtering. Levels remain descriptive; unresolved screens remain exploratory.",
        "missing_outcomes": [{"sample_id": row["sample_id"], "outcome": outcome} for row in specifications[::5]
                             for outcome in YS if outcome not in {item["outcome"] for item in specifications if item["sample_id"] == row["sample_id"]}]}
    (OUT / "manifest.json").write_text(json.dumps(plain(manifest), indent=2, allow_nan=False)+"\n")
    build_report()
    return manifest


def build_report() -> None:
    import maketables as mt
    from heterogeneity_report import HeterogeneityResult
    from regression_tables import namespace_table_ids

    specs = pd.read_csv(OUT / "specifications.csv", dtype={"reference_sector_id": str})
    slopes = pd.read_csv(OUT / "sector_slopes.csv", dtype={"sector_id": str, "reference_sector_id": str})
    joints = pd.read_csv(OUT / "joint_tests.csv")
    components = pd.read_csv(OUT / "interaction_components.csv", dtype={"sector_id": str})
    for frame in [slopes, joints, components]:
        frame["p_display"] = frame.p_value.map(p_format)
        frame["significance"] = frame.p_value.map(significance_label)
        frame["stars"] = frame.p_value.map(significance_stars)
    native_tables = {}
    (OUT / "tables").mkdir(exist_ok=True)
    for spec in specs.to_dict("records"):
        part = slopes[slopes.sample_id.eq(spec["sample_id"])&slopes.outcome.eq(spec["outcome"])]
        cr = part[part.covariance.eq("Sector cluster")]
        names = cr.set_index("sector_id").sector.to_dict()
        horizon = 12 if spec["frequency"] == "monthly" else 4
        for band in [horizon, 2*horizon]:
            dk = part[part.covariance.eq("Driscoll-Kraay")&part.hac_lags.eq(band)]
            dependent = spec["outcome_label"].replace("One-period", "MoM" if spec["frequency"] == "monthly" else "QoQ")
            models = [HeterogeneityResult(frame.copy(), dependent, "H1") for frame in [dk, cr]]
            note = (f"Actual PyFixest {spec['estimator_version']} H1 point estimates. Same coefficients/sample in both columns; "
                    "standard errors in parentheses. DK uses calendar score sums; sector CR1 uses G/(G−1)×(N−1)/(N−K). "
                    "Sector-specific regressors belong to one cluster; shared nuisance parameters can leave nonzero absolute-slope CR1 SE without independent cluster replication for each slope. "
                    "Collapsed equal-slope covariance has no omnibus p-value. * p<.10, ** p<.05, *** p<.01; nominal pointwise tests. "
                    "Total effects are betaT+delta_i with the full component covariance cross term. No stationarity clearance follows; levels are descriptive.")
            controls = "Rainfall, humidity, wind, PM2.5" if spec["window"] != "long" else "Rainfall, humidity"
            table = mt.ETable(models, model_heads=[f"Calendar DK · {band} lags", "Sector CR1 · requested sensitivity"],
                head_order="dh", coef_fmt="b:.3f* \n (se:.3f)", signif_code=[.01, .05, .10],
                keep=list(names), exact_match=True, labels=names,
                felabels={"sector_season": "Sector × season FE", "year": "Common year FE"},
                feorder=["sector_season", "year"], model_stats=["N", "sectors", "periods", "r2"],
                model_stats_labels={"N": "Observations", "sectors": "Sector units", "periods": "Weather periods", "r2": "R² (including FE)"},
                custom_model_stats={"Estimator": ["PyFixest feols "+spec["estimator_version"]]*2,
                    "Shared controls": [controls]*2, "Inference df": [int(spec["n_periods"])-1, int(spec["n_sectors"])-1],
                    "Clusters": ["—", int(spec["n_sectors"])], "Estimated parameters": [int(spec["n_parameters"])]*2,
                    "Mean observed Y": [f"{spec['mean_y_all']:.3f}"]*2, "SD observed Y": [f"{spec['sd_y_all']:.3f}"]*2,
                    "First / last period": [spec["start"]+" / "+spec["end"]]*2}, stats_order="sc", notes=note)
            key = spec["sample_id"]+"|"+spec["outcome"]+"|"+str(band)
            fragment = namespace_table_ids(table.make(type="gt", gt_style={"table_width": "100%",
                "first_col_width": "300px", "table_font_size_all": "12px", "source_notes_font_size": "11px",
                "data_row_padding": "6px", "column_labels_padding": "8px"}).as_raw_html(), key.replace("|", "-"))
            native_tables[key] = fragment
            (OUT / "tables" / (key.replace("|", "_")+".html")).write_text(fragment)
    display_columns = ["sample_id", "context", "window", "outcome", "sector_id", "sector", "covariance", "hac_lags",
        "inference_df", "coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "p_display", "stars", "significance",
        "degenerate", "mean_y", "sd_y", "temp_main", "interaction_delta", "reference_delta_normalization",
        "var_temp_main", "var_interaction_delta", "cov_temp_main_delta", "own_sector_temperature_score_relative",
        "inference_status", "shared_nuisance_limitation"]
    joint_columns = ["sample_id", "outcome", "covariance", "hac_lags", "covariance_rank", "restrictions",
                     "numerator_df", "denominator_df", "f_statistic", "p_value", "p_display", "status"]
    payload = {"specs": specs.to_dict("records"), "rows": slopes[display_columns].to_dict("records"), "native_tables": native_tables,
               "joints": joints[joint_columns].to_dict("records"),
               "manifest": json.loads((OUT / "manifest.json").read_text()), "contexts": CONTEXTS,
               "windows": WINDOWS, "outcomes": Y_LABELS}
    document = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sector-cluster sensitivity · Singapore heat and economy</title><style>
:root{--navy:#142b47;--blue:#1765ad;--orange:#b8660d;--muted:#5c6d80;--line:#dbe3ed}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#18304b;font:15px/1.6 system-ui,-apple-system,sans-serif}header{background:var(--navy);color:#fff;padding:40px max(24px,calc((100vw - 1260px)/2))}header p{max-width:980px;color:#d9e5f2}h1{font-size:34px;line-height:1.2;margin:8px 0 16px}h2{font-size:22px;margin:0 0 16px}h3{font-size:17px}a{color:var(--blue)}header a{color:#d9eaff;margin-right:22px}main{max-width:1308px;margin:auto;padding:28px 24px 60px}.card{background:#fff;border:1px solid var(--line);border-radius:14px;margin-bottom:22px;padding:24px}.warning{background:#fff5df;border-left:5px solid var(--orange);border-radius:8px;padding:18px 22px;margin-bottom:22px}.warning strong{color:#7c4305}.eyebrow{letter-spacing:.12em;text-transform:uppercase;font-size:12px;font-weight:750;color:#a7cdf4}.controls{display:grid;grid-template-columns:1.2fr 1.6fr 1.2fr .6fr;gap:16px}label{font-size:12px;color:var(--muted);font-weight:700;display:block}select{display:block;width:100%;margin-top:6px;border:1px solid #bdcad9;background:#fff;color:var(--navy);border-radius:7px;padding:10px;font:inherit}.meta{display:flex;flex-wrap:wrap;gap:12px 28px;margin-top:18px;color:var(--muted);font-size:13px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{padding:11px 12px;border-bottom:1px solid #e6ecf3;text-align:right;vertical-align:top}th{background:#edf3f9;color:var(--navy);font-size:12px}td:first-child,th:first-child{text-align:left;min-width:210px;position:sticky;left:0;background:#fff}th:first-child{background:#edf3f9}tbody tr:hover td{background:#f5f9fd}small{display:block;color:var(--muted);font-size:11px;white-space:nowrap}.badge{display:inline-block;padding:2px 7px;border-radius:5px;background:#edf2f7;font-size:11px;white-space:nowrap}.yes{background:#e4f2e9;color:#21633b}.degenerate{background:#fff0d3;color:#8a4c00}.method-dk{color:var(--blue)}.method-cr{color:var(--orange)}.caption{font-size:12px;color:var(--muted);margin:12px 0 0}.button{display:inline-block;border:1px solid #b8cce0;color:var(--blue);background:#f1f7fd;padding:8px 12px;border-radius:7px;cursor:pointer;margin:8px 10px 0 0;font:inherit;font-size:13px}code{background:#edf3f9;border-radius:4px;padding:3px 6px;font-size:13px}.split{display:grid;grid-template-columns:1fr 1fr;gap:20px}.big{font-size:24px;font-weight:750}.formula{line-height:2}summary{cursor:pointer;font-weight:700}details .scroll{margin-top:18px}@media(max-width:850px){.controls,.split{grid-template-columns:1fr 1fr}h1{font-size:28px}}@media(max-width:520px){.controls,.split{grid-template-columns:1fr}.card{padding:16px}main{padding:16px}}
</style></head><body><header><div class="eyebrow">Additional inference sensitivity · H1 sector interactions</div><h1>What happens when we cluster by sector?</h1><p>The same PyFixest H1 coefficients and exact samples, with sector CR1 beside retained calendar DK uncertainty. This requested sensitivity makes the numerical consequences visible; it does not replace the existing sector-profile inference.</p><nav><a href="heat_economy.html">Complete analysis</a><a href="sector_heterogeneity.html">Broad-sector models</a><a href="granular_va.html">Detailed VA</a><a href="manufacturing_detail.html">Manufacturing</a></nav></header><main>
<div class="warning"><strong>Read the clustered standard errors with the model structure.</strong> Each sector-specific temperature regressor belongs to one sector cluster. Shared weather/year nuisance estimates can produce nonzero absolute-slope CR1 standard errors, often with rank-one joint covariance. Their presence does not provide multiple independent clusters for that sector's slope. Sector-difference covariance may collapse; those joint p-values are explicitly unavailable.</div>
<section class="card"><div class="controls"><label>Sector cut<select id="context"></select></label><label>Controls and sample<select id="window"></select></label><label>Dependent variable<select id="outcome"></select></label><label>DK bandwidth<select id="band"></select></label></div><div id="meta" class="meta"></div><p id="outcome-description" class="caption"></p><p class="formula"><code>Yᵢₜ = (βT + δᵢ)Tₜ + γ′Xₜ + sector × season FE + year FE + εᵢₜ</code><br>The reference has δ=0 by normalization. Every other total is βT+δᵢ; its variance includes the component covariance cross term. Temperature remains in current-period °C levels.</p></section>
<section class="card"><h2>Regression table · identical H1 coefficients</h2><div id="native" class="scroll">__NATIVE_DEFAULT__</div><p class="caption">MakeTables ETable, the table engine used by PyFixest etable. These are saved inference values; displaying the table does not refit or change the covariance.</p></section>
<section class="card"><h2>Sector readout · explicit p-values and significance</h2><div class="scroll"><table><thead><tr><th>Sector / industry</th><th>Total β per +1°C</th><th class="method-dk">DK SE / 95% interval</th><th class="method-dk">DK p / significance</th><th class="method-cr">Sector CR1 SE / 95% interval</th><th class="method-cr">Sector CR1 p / significance</th><th>Mean Y</th><th>β × 0.1°C</th></tr></thead><tbody id="slopes"></tbody></table></div><p class="caption">Stars use unrounded p-values: *** p&lt;.01, ** p&lt;.05, * p&lt;.10. Text distinguishes 1%, 5%, 10%-only and not significant. These are nominal pointwise tests, conditional on the model. Numerical-zero variances retain their raw SE but have no fabricated p-value or confidence interval.</p><button class="button" id="download-slopes">Download this comparison CSV</button></section>
<section class="card"><h2>Joint equality diagnostic</h2><div id="joint" class="scroll"></div><p class="caption">The null is equal sector slopes, not every sector slope equal to zero. No pseudoinverse or artificial omnibus p-value is used when the restriction covariance is singular or numerically zero.</p></section>
<section class="card"><details><summary>Inspect βT + δᵢ and the variance cross term</summary><div id="components" class="scroll"></div><p class="caption">Var(total)=Var(βT)+Var(δᵢ)+2Cov(βT,δᵢ). Reference δ=0 is a normalization with no estimated SE or p-value. Full component and total covariance matrices are saved in the accompanying output directory.</p></details></section>
<section class="card"><h2>How to interpret this sensitivity</h2><div class="split"><div><h3>Sector CR1</h3><p>Cluster scores sum over all dates within each sector, followed by <code>G/(G−1) × (N−1)/(N−K)</code> and t(G−1). K includes every nuisance, FE and temperature parameter. This permits arbitrary within-sector dependence and assumes independent sectors. Shared Singapore weather does not meet that assumption automatically.</p></div><div><h3>Retained calendar DK</h3><p>Time-aggregated scores use real calendar lags and Bartlett weights, with one- and two-year bandwidths saved. The reference distribution is t(T−1). It accommodates common sector shocks and temporal dependence, with finite-sample and short-window limitations.</p></div></div><p>H1 shares weather-control and year effects across sectors. These shared nuisance estimates are the source of the remaining cluster uncertainty when each sector's own temperature normal equation cancels within that sector. This explains why nonzero CR1 SE can coexist with a degenerate sector-difference covariance.</p><p>Numerical degeneracy is judged relative to the unscaled independent-observation HC0 sandwich as a scale reference: |variance| ≤ 10⁻¹² × reference variance. Joint rank uses the larger of 10⁻¹⁰ × largest singular value and 10⁻¹² × the contrasted HC0 spectral norm. HC0 is a numerical reference here, not an alternative claimed inference.</p><p>No observations are removed beyond the existing endpoint exclusion. Full controls include rainfall, humidity, wind and PM2.5; long controls include rainfall and humidity. Growth and lags were constructed before filtering. Levels remain descriptive; no new stationarity clearance or causal interpretation is supplied by changing standard errors.</p><div id="provenance" class="caption"></div></section>
</main><script id="cluster-data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('cluster-data').textContent),E=id=>document.getElementById(id),esc=v=>String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const f=(v,d=3)=>v===null||v===undefined?'Unavailable':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const table=(heads,rows)=>'<table><thead><tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
for(const [id,items,value] of [['context',D.contexts,'va-detailed'],['window',D.windows,'full'],['outcome',D.outcomes,'y_short']]){E(id).innerHTML=Object.entries(items).map(([key,label])=>'<option value="'+key+'">'+esc(label)+'</option>').join('');E(id).value=value;}
let chosen=[];
function draw(){const context=E('context').value,window=E('window').value,outcome=E('outcome').value,monthly=context.includes('monthly'),base=monthly?12:4;
 const previous=Number(E('band').value);E('band').innerHTML=[base,2*base].map(h=>'<option value="'+h+'">'+h+' '+(monthly?'months':'quarters')+'</option>').join('');E('band').value=String([base,2*base].includes(previous)?previous:base);const band=Number(E('band').value),sample=context+'__'+window;
 const spec=D.specs.find(s=>s.sample_id===sample&&s.outcome===outcome);if(!spec)throw Error('Unavailable specification');
 E('native').innerHTML=D.native_tables[sample+'|'+outcome+'|'+band];
 const cr=D.rows.filter(r=>r.sample_id===sample&&r.outcome===outcome&&r.covariance==='Sector cluster'),dk=D.rows.filter(r=>r.sample_id===sample&&r.outcome===outcome&&r.covariance==='Driscoll-Kraay'&&r.hac_lags===band);chosen=[...dk,...cr];
 E('meta').innerHTML=['<b>'+spec.n_sectors+'</b> sectors / clusters','<b>'+spec.n_periods+'</b> weather periods','N='+spec.n_obs+' · K='+spec.n_parameters,'CR1 df='+String(spec.n_sectors-1)+' · DK df='+String(spec.n_periods-1),esc(spec.start)+' – '+esc(spec.end),'Absolute CR1 covariance rank '+spec.cluster_absolute_covariance_rank+'/'+spec.n_sectors].map(v=>'<span>'+v+'</span>').join('');
 E('outcome-description').textContent=outcome==='y_short'?(monthly?'MoM':'QoQ')+' log growth (SA): 100 × [ln A(t) − ln A(t−1)], not annualised. A coefficient is log growth points per +1°C.':outcome==='y_yoy'||outcome==='y_yoy_nsa'?'YoY log growth ('+(outcome==='y_yoy_nsa'?'NSA':'SA')+'): 100 × [ln A(t) − ln A(t−'+(monthly?'12':'4')+')]. Rolling annual outcomes overlap; temperature is current-period level, not YoY temperature change.':outcome==='y_log_level'?'Descriptive log level (SA): 100 × ln A(t). A temperature slope approximates a percent output-level difference per +1°C.':'Descriptive raw level (SA): chained real VA in S$ million, or IIP index points. Sector slopes are not national GDP losses.';
 const interval=r=>f(r.std_error)+'<small>'+(r.degenerate?'Unavailable · numerical degeneracy':'['+f(r.ci_lower)+', '+f(r.ci_upper)+']')+'</small>';
 const sig=r=>esc(r.p_display)+esc(r.stars)+'<small><span class="badge '+(r.degenerate?'degenerate':r.p_value!==null&&r.p_value<.05?'yes':'')+'">'+esc(r.degenerate?'Unavailable · degenerate':r.significance)+'</span></small>';
 E('slopes').innerHTML=cr.map(r=>{const a=dk.find(x=>x.sector_id===r.sector_id);if(!a)throw Error('Missing DK match');return '<tr><td>'+esc(r.sector)+'</td><td>'+f(r.coefficient)+'</td><td>'+interval(a)+'</td><td>'+sig(a)+'</td><td>'+interval(r)+'</td><td>'+sig(r)+'</td><td>'+f(r.mean_y)+'</td><td>'+f(.1*r.coefficient)+'</td></tr>';}).join('');
 const joint=D.joints.filter(r=>r.sample_id===sample&&r.outcome===outcome&&(r.covariance==='Sector cluster'||r.hac_lags===band));E('joint').innerHTML=table(['Inference','Covariance rank / restrictions','Reference df','F','p-value','Status'],joint.map(r=>[esc(r.covariance)+(r.hac_lags?' · '+r.hac_lags+' lags':''),r.covariance_rank+' / '+r.restrictions,r.numerator_df+', '+r.denominator_df,f(r.f_statistic),esc(r.p_display),'<span class="badge '+(r.status==='available'?'':'degenerate')+'">'+esc(r.status)+'</span>']));
 E('components').innerHTML=table(['Sector','βT','δᵢ','Total','CR1 Var(βT)','CR1 Var(δᵢ)','CR1 Cov(βT,δᵢ)','Own-sector temperature score / absolute scores'],cr.map(r=>[esc(r.sector),f(r.temp_main),r.reference_delta_normalization?'0 · normalization':f(r.interaction_delta),f(r.coefficient),f(r.var_temp_main,6),f(r.var_interaction_delta,6),f(r.cov_temp_main_delta,6),Number(r.own_sector_temperature_score_relative).toExponential(2)]));
 E('provenance').textContent=D.manifest.native_fits+' actual native PyFixest '+spec.estimator_version+' fits across '+D.manifest.samples+' exact samples. '+D.manifest.existing_DK_parent_model_outcomes+'/'+D.manifest.model_outcomes+' model/outcomes match an existing saved DK H1 parent; new broad raw-level H1 sensitivity uses the already available raw-level column. Native ledger: outputs/clustered_sector_sensitivity/pyfixest. Source numerical results and their ledgers are preserved.';
}
for(const id of ['context','window','outcome','band'])E(id).addEventListener('change',draw);
E('download-slopes').addEventListener('click',()=>{const columns=['context','window','outcome','sector','covariance','hac_lags','coefficient','std_error','ci_lower','ci_upper','p_value','significance','mean_y','sd_y','inference_df','degenerate','inference_status','shared_nuisance_limitation'];const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"',csv=[columns,...chosen.map(r=>columns.map(c=>r[c]))].map(row=>row.map(quote).join(',')).join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download='sector_cluster_H1_comparison.csv';a.click();URL.revokeObjectURL(url);});
draw();
</script></body></html>'''
    document = document.replace("__NATIVE_DEFAULT__", native_tables["va-detailed__full|y_short|4"])
    REPORT.write_text(document.replace("__DATA__", json.dumps(plain(payload), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")))
    (OUT / "report_inventory.json").write_text(json.dumps({"html_sha256": digest(REPORT), "specifications": len(specs),
        "sector_rows": len(slopes), "joint_rows": len(joints), "native_tables": len(native_tables), "source_hashes": {name: digest(OUT / name)
        for name in ["specifications.csv", "sector_slopes.csv", "joint_tests.csv", "interaction_components.csv", "manifest.json"]}}, indent=2)+"\n")
    print(f"Built {REPORT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    build_report() if args.reports_only else run_clustered_sector_sensitivity()
