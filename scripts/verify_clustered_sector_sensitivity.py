"""Independent full-QR audit of the requested H1 sector-CR1 sensitivity.

Uses the independent explicit design/QR implementation, never the production
FWL estimator or its covariance helpers. The reference HC0 scale is uncorrected
individual-observation OLS influence covariance. Degeneracy is not significance.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution, t as student_t
from verify_covid_comparison import sector_design, full_qr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/clustered_sector_sensitivity"
CONTEXTS = ["va", "iip-monthly", "iip-quarterly", "va-detailed", "iip-monthly-detailed", "iip-quarterly-detailed"]
OUTCOMES = ["y_short", "y_yoy_nsa", "y_yoy", "y_log_level", "y_raw_level"]
WINDOWS = ["long", "full", "exclude"]
RELATIVE_ZERO = 1e-12


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={key: str for key in ["sector_id", "sector_i", "sector_j", "reference_sector_id"]})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label: str, *, atol=1e-7, rtol=1e-7) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)), (label, "Shape/unavailable masks")
    errors, allowance = np.abs(a-b), atol+rtol*np.abs(b)
    if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
        index = np.unravel_index(np.nanargmax(errors-allowance), errors.shape)
        raise AssertionError((label, "maximum error", float(np.nanmax(errors)), "worst", index,
                              "actual", float(a[index]), "expected", float(b[index]), "allowed", float(allowance[index])))
    finite = np.isfinite(a)&np.isfinite(b)
    return float(np.max(errors[finite])) if finite.any() else 0.


def select(frame: pd.DataFrame, sample_id: str, outcome: str, covariance=None, bandwidth=None) -> pd.DataFrame:
    subset = frame[frame.sample_id.eq(sample_id)&frame.outcome.eq(outcome)]
    if covariance is not None:
        subset = subset[subset.covariance.eq(covariance)]
    if bandwidth is not None:
        subset = subset[subset.hac_lags.eq(bandwidth)]
    return subset


def parent_sample(context: str, window: str) -> tuple[pd.DataFrame, Path]:
    if context.endswith("-detailed"):
        canonical = ROOT / "outputs/granular" / context / ("long" if window == "long" else "full") / "samples/G1.csv"
        path = ROOT / "outputs/granular_interactions" / context / window / "common_sample.csv"
    else:
        canonical = ROOT / "outputs/outcomes" / context / ("weather" if window == "long" else "full") / "common_sample.csv"
        path = ROOT / "outputs/heterogeneity" / context / ("weather" if window == "long" else "full") / "common_sample.csv"
    sample = read(canonical)
    if window == "exclude":
        sample = sample[~sample.year.between(2020, 2022)]
    direct = read(path)
    if window == "exclude" and not context.endswith("-detailed"):
        direct = direct[~direct.year.between(2020, 2022)]
    exact_sample(direct, sample, context+"/"+window+": parent versus original outcome preparation")
    return sample.reset_index(drop=True), path


def exact_sample(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    period = "month" if "month" in expected else "quarter"
    keys = [period, "sector_id"]
    assert not actual.duplicated(keys).any() and len(actual) == len(expected)
    assert set(expected) <= set(actual)
    a = actual.sort_values(keys).reset_index(drop=True)[list(expected)]
    b = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, atol=1e-11, rtol=1e-12)


def influence_covariances(fit: dict, sample: pd.DataFrame, contrasts: np.ndarray, outcome: int, horizon: int) -> tuple[dict, np.ndarray]:
    coefficient_influence = fit["normalized"]@fit["bread_scaled"]@(contrasts/fit["scale"]).T
    individual = coefficient_influence*fit["residual"][:, [outcome]]
    hc0 = individual.T@individual
    n, k = fit["n_obs"], fit["n_parameters"]
    sectors = sorted(sample.sector_id.unique())
    cluster_scores = np.vstack([individual[sample.sector_id.eq(sector)].sum(axis=0) for sector in sectors])
    cluster_correction = len(sectors)/(len(sectors)-1)*(n-1)/(n-k)
    results = {("Sector cluster", 0): (cluster_scores.T@cluster_scores*cluster_correction, len(sectors)-1, cluster_correction)}
    period = "month" if "month" in sample else "quarter"
    dates = pd.PeriodIndex(sample[period], freq="M" if period == "month" else "Q").asi8
    elapsed = dates-dates.min()
    score = np.zeros((elapsed.max()+1, len(contrasts)))
    for date in np.unique(elapsed):
        score[date] = individual[elapsed == date].sum(axis=0)
    periods = len(np.unique(dates))
    correction = periods/(periods-1)*(n-1)/(n-k)
    for bandwidth in [horizon, 2*horizon]:
        meat = score.T@score
        for lag in range(1, bandwidth+1):
            cross = score[lag:].T@score[:-lag]
            meat += (1-lag/(bandwidth+1))*(cross+cross.T)
        results["Driscoll-Kraay", bandwidth] = ((meat+meat.T)/2*correction, periods-1, correction)
    return results, (hc0+hc0.T)/2


def point(row: pd.Series, beta: float, variance: float, hc0: float, degrees: int, label: str) -> bool:
    close(row.coefficient, beta, label+": coefficient")
    close(row.hc0_reference_variance, hc0, label+": HC0 reference")
    threshold = RELATIVE_ZERO*hc0
    assert hc0 > 0
    close(row.zero_relative_tolerance, RELATIVE_ZERO, label+": relative degeneracy tolerance", atol=1e-24, rtol=1e-12)
    close(row.zero_variance_threshold, threshold, label+": degeneracy threshold", atol=1e-20, rtol=1e-7)
    close(row.variance_ratio_to_hc0, variance/hc0, label+": HC0 relative variance")
    degenerate = abs(variance) <= threshold
    assert bool(row.degenerate) == degenerate
    if degenerate:
        close(row.variance/hc0, variance/hc0, label+": degenerate variance on declared HC0 scale", atol=RELATIVE_ZERO, rtol=1e-7)
        close(row.std_error, math.sqrt(max(0., row.variance)), label+": retained raw numerical SE", atol=1e-15, rtol=1e-12)
        assert row.inference_status == "unavailable: numerically degenerate variance"
        assert row[["ci_lower", "ci_upper", "p_value"]].isna().all()
    else:
        close(row.variance, variance, label+": variance")
        close(row.std_error, math.sqrt(max(0., variance)), label+": raw SE")
        assert variance > 0 and row.inference_status == "available"
        standard_error = math.sqrt(variance)
        critical = student_t.ppf(.975, degrees)
        p = float(2*student_t.sf(abs(beta/standard_error), degrees))
        close([row.ci_lower, row.ci_upper, row.p_value], [beta-critical*standard_error, beta+critical*standard_error, p], label+": CI/p")
        close(p, f_distribution.sf(beta**2/variance, 1, degrees), label+": one-constraint total-effect t/F")
    return degenerate


def covariance_rank(matrix: np.ndarray, hc0: np.ndarray, bound: int) -> tuple[int, float, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    reference = float(np.linalg.norm(hc0, ord=2))
    threshold = max(float(singular.max())*1e-10, RELATIVE_ZERO*reference)
    return min(int(np.sum(singular > threshold)), bound), reference, threshold


def original_dk_reference(context: str, window: str, outcome: str, bandwidth: int) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    if context.endswith("-detailed"):
        folder = ROOT / "outputs/granular_interactions" / context / window
        rows = read(folder / "sector_slopes.csv")
        rows = rows[rows.model.eq("H1")&rows.outcome.eq(outcome)&rows.hac_lags.eq(bandwidth)]
        cov = read(folder / "slope_covariances.csv")
        cov = cov[cov.model.eq("H1")&cov.outcome.eq(outcome)&cov.hac_lags.eq(bandwidth)]
        return rows, cov, str(folder.relative_to(ROOT))
    if window != "exclude":
        folder = ROOT / "outputs/heterogeneity" / context / ("weather" if window == "long" else "full")
        rows, cov = read(folder / "sector_slopes.csv"), read(folder / "slope_covariances.csv")
        rows = rows[rows.model.eq("H1")&rows.outcome.eq(outcome)&rows.hac_lags.eq(bandwidth)]
        cov = cov[cov.model.eq("H1")&cov.outcome.eq(outcome)&cov.hac_lags.eq(bandwidth)]
        return rows, cov, str(folder.relative_to(ROOT))
    folder = ROOT / "outputs/covid_comparison/sector"
    metadata = read(folder / "metadata.csv")
    selected = metadata[metadata.context.eq(context)&metadata.source_kind.eq("broad_FE")&metadata.model.eq("H1")&metadata.control_set.eq("full")&metadata.outcome.eq(outcome)]
    if selected.empty:
        return pd.DataFrame(), None, "no pre-existing sector-H1 DK counterpart"
    assert len(selected) == 1
    pair = selected.pair_id.iloc[0]
    rows, cov = read(folder / "estimates.csv"), read(folder / "slope_covariances.csv")
    rows = rows[rows.pair_id.eq(pair)&rows.sample_variant.eq("exclude_2020_2022")&rows.hac_lags.eq(bandwidth)]
    cov = cov[cov.pair_id.eq(pair)&cov.sample_variant.eq("exclude_2020_2022")&cov.hac_lags.eq(bandwidth)]
    return rows, cov, str(folder.relative_to(ROOT))


def verify_clustered_sector_sensitivity() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "verification.json"
    target.write_text('{"status":"running","claim":"No completion claim before independent verification passes"}\n')
    specs = read(OUT / "specifications.csv")
    tables = {name: read(OUT / f"{name}.csv") for name in ["sector_slopes", "slope_covariances", "interaction_components", "component_covariances", "joint_tests", "all_coefficients"]}
    assert not specs.duplicated(["sample_id", "outcome"]).any()
    expected = {(context+"__"+window, outcome) for context in CONTEXTS for window in WINDOWS for outcome in OUTCOMES}
    assert set(specs[["sample_id", "outcome"]].itertuples(index=False, name=None)) == expected
    for name in ["sector_slopes", "interaction_components"]:
        keys = ["sample_id", "outcome", "covariance", "hac_lags", "sector_id" if name == "sector_slopes" else "term"]
        assert not tables[name].duplicated(keys).any()
    details, expected_native = [], []
    counts = {"model_outcomes": 0, "sample_families": 0, "slope_rows": 0, "component_rows": 0,
              "normalizations": 0, "slope_covariance_entries": 0, "component_covariance_entries": 0,
              "all_coefficients": 0, "degenerate_component_or_slope_rows": 0, "joint_full_rank": 0,
              "joint_degenerate_or_singular": 0, "original_DK_groups_matched": 0, "new_DK_groups_without_prior_sector_fit": 0}
    sources = {}
    for sample_id, specification in specs.groupby("sample_id", sort=False):
        context, window = sample_id.rsplit("__", 1)
        original, path = parent_sample(context, window)
        sample_path = ROOT / specification.sample_csv.iloc[0]
        sample = read(sample_path)
        exact_sample(sample, original, sample_id)
        assert specification.source_sample_csv.eq(str(path.relative_to(ROOT))).all()
        assert specification.parent_sample_sha256.eq(sha256(path)).all()
        assert specification.sample_sha256.eq(sha256(sample_path)).all()
        sources[str(path.relative_to(ROOT))] = sha256(path)
        period = "month" if "month" in sample else "quarter"
        horizon = 12 if period == "month" else 4
        sectors = sorted(sample.sector_id.unique())
        assert sample.groupby(period).sector_id.nunique().eq(len(sectors)).all()
        dates = pd.PeriodIndex(sample[period], freq="M" if period == "month" else "Q")
        assert np.ptp(sample.time.to_numpy()-dates.asi8) == 0
        assert not window == "exclude" or not sample.year.between(2020, 2022).any()
        terms = ["temp_c", "rain_100mm", "rh_pct"]+([] if window == "long" else ["wind_mps", "pm25_10ug"])
        assert all(json.loads(value) == terms for value in specification.terms)
        x, names, contrast, reported = sector_design(sample, "H1", terms)
        assert reported == sectors
        fit = full_qr(x, sample[OUTCOMES].to_numpy(), names)
        stored = np.load(OUT / "designs" / f"{sample_id}.npz", allow_pickle=False)
        assert stored["sectors"].tolist() == sectors and stored["outcomes"].tolist() == OUTCOMES
        close(stored["beta"], contrast@fit["beta"], sample_id+": stored absolute slope array")
        to_component_all = np.eye(len(sectors));to_component_all[1:, 0] = -1
        close(stored["component_beta"], to_component_all@contrast@fit["beta"], sample_id+": stored native component array")
        close(stored["nuisance_beta"], fit["beta"][:-len(sectors)], sample_id+": stored nuisance array")
        close(stored["residual"], fit["residual"], sample_id+": stored residual array")
        z = x[:, :-len(sectors)]
        qz, _ = np.linalg.qr(z/np.linalg.norm(z, axis=0), mode="reduced")
        a = np.column_stack([sample.sector_id.eq(sector) for sector in sectors])*(sample.temp_c-sample.temp_c.mean()).to_numpy()[:, None]
        residual_design = a-qz@(qz.T@a)
        close(stored["ar"], residual_design, sample_id+": partial exposure geometry")
        close(stored["bread"], np.linalg.solve(residual_design.T@residual_design, np.eye(len(sectors))), sample_id+": partial design bread")
        n, k, periods = len(sample), len(names), sample[period].nunique()
        assert specification.n_obs.eq(n).all() and specification.n_parameters.eq(k).all() and specification.n_periods.eq(periods).all() and specification.n_sectors.eq(len(sectors)).all()
        counts["sample_families"] += 1
        for outcome in OUTCOMES:
            j = OUTCOMES.index(outcome)
            spec = specification[specification.outcome.eq(outcome)].iloc[0]
            close(spec.r_squared, fit["r_squared"][j], sample_id+"/"+outcome+": full-model R squared")
            close([spec.mean_y_all, spec.sd_y_all], [sample[outcome].mean(), sample[outcome].std()], "Pooled exact-sample mean/SD")
            coefficient = fit["beta"][:, j]
            beta = contrast@coefficient
            calculations, hc0 = influence_covariances(fit, sample, contrast, j, horizon)
            coefficients = select(tables["all_coefficients"], sample_id, outcome).set_index("term")
            assert set(coefficients.index) == set(names)
            close(coefficients.loc[names, "coefficient"], coefficient, sample_id+"/"+outcome+": full-model coefficients")
            counts["all_coefficients"] += len(coefficients)
            expected_native.append((sample_id, outcome, n, k))
            to_component = np.eye(len(sectors));to_component[1:, 0] = -1
            component_names = ["temp_c"]+[f"temp_c:C(sector_id)[T.{sector}]" for sector in sectors[1:]]
            component_beta = to_component@beta
            component_hc0 = to_component@hc0@to_component.T
            for (covariance_name, bandwidth), (matrix, degrees, correction) in calculations.items():
                rows = select(tables["sector_slopes"], sample_id, outcome, covariance_name, bandwidth).set_index("sector_id").loc[sectors]
                saved = select(tables["slope_covariances"], sample_id, outcome, covariance_name, bandwidth)
                assert len(saved) == len(sectors)**2
                close(saved.pivot(index="sector_i", columns="sector_j", values="covariance_value").loc[sectors, sectors], matrix, sample_id+"/"+outcome+": joint slope covariance")
                counts["slope_covariance_entries"] += len(saved)
                assert rows.inference_df.eq(degrees).all() and rows.n_clusters.eq(len(sectors) if covariance_name == "Sector cluster" else 0).all()
                assert rows.n_obs.eq(n).all() and rows.n_sectors.eq(len(sectors)).all() and rows.n_parameters.eq(k).all() and rows.n_periods.eq(periods).all()
                close(rows.r_squared, np.repeat(fit["r_squared"][j], len(rows)), "Reported full-model R squared")
                close(rows.finite_sample_correction, np.repeat(len(sectors)/(len(sectors)-1)*(n-1)/(n-k), len(rows)), "Reported sector-CR1 finite correction")
                assert rows.shared_nuisance_limitation.eq(covariance_name == "Sector cluster").all()
                for i, sector in enumerate(sectors):
                    row = rows.loc[sector]
                    degenerate = point(row, float(beta[i]), float(matrix[i, i]), float(hc0[i, i]), degrees, sample_id+"/"+outcome+"/"+sector)
                    counts["degenerate_component_or_slope_rows"] += degenerate
                    close(row.temp_main+row.interaction_delta, beta[i], "Temperature main plus sector delta")
                    close(row.temp_main, beta[0], "Reference-sector temperature main")
                    assert bool(row.reference_delta_normalization) == (sector == sectors[0])
                    unit = sample[sample.sector_id.eq(sector)]
                    close([row.mean_y, row.sd_y], [unit[outcome].mean(), unit[outcome].std()], "Exact parent sector mean/SD")
                    products = (unit.temp_c-unit.temp_c.mean()).to_numpy()*fit["residual"][sample.sector_id.eq(sector), j]
                    absolute = float(np.abs(products).sum())
                    relative = abs(float(products.sum()))/absolute if absolute else 0.
                    assert relative < 1e-9 and row.own_sector_temperature_score_relative < 1e-9
                    close(row.own_sector_temperature_score_abs_sum, absolute, "Own-sector temperature score scale")
                    close(row.own_sector_temperature_score_sum, products.sum(), "Own-sector temperature score numerical zero", atol=1e-9*max(1., absolute), rtol=1e-7)
                    counts["slope_rows"] += 1
                components = select(tables["interaction_components"], sample_id, outcome, covariance_name, bandwidth)
                normalization = components[components.normalization.eq(True)]
                assert len(normalization) == 1 and normalization.coefficient.iloc[0] == 0 and not normalization.estimated.any()
                assert normalization[["std_error", "ci_lower", "ci_upper", "p_value"]].isna().all().all()
                estimated = components[components.estimated.eq(True)].set_index("term").loc[component_names]
                component_matrix = to_component@matrix@to_component.T
                component_rows = select(tables["component_covariances"], sample_id, outcome, covariance_name, bandwidth)
                assert len(component_rows) == len(sectors)**2
                recorded_component = component_rows.pivot(index="term_i", columns="term_j", values="covariance_value").loc[component_names, component_names].to_numpy()
                if covariance_name == "Sector cluster":
                    reference_scale = np.sqrt(np.outer(np.diag(component_hc0), np.diag(component_hc0)))
                    close(recorded_component/reference_scale, component_matrix/reference_scale, "Full main/deviation CR1 covariance on HC0 scale", atol=RELATIVE_ZERO, rtol=1e-7)
                else:
                    close(recorded_component, component_matrix, "Full main/deviation covariance")
                for i, term in enumerate(component_names):
                    counts["degenerate_component_or_slope_rows"] += point(estimated.loc[term], float(component_beta[i]), float(component_matrix[i, i]), float(component_hc0[i, i]), degrees, "Native temperature component "+term)
                counts["component_rows"] += len(components);counts["normalizations"] += 1;counts["component_covariance_entries"] += len(component_rows)
                joint = select(tables["joint_tests"], sample_id, outcome, covariance_name, bandwidth)
                assert len(joint) == 1
                restriction = np.eye(len(sectors))[1:]-np.eye(len(sectors))[[0]]
                delta, v, hc = restriction@beta, restriction@matrix@restriction.T, restriction@hc0@restriction.T
                bound = len(sectors)-1 if covariance_name == "Sector cluster" else periods-sample.season.nunique()-1
                rank, reference, zero = covariance_rank(v, hc, bound)
                record = joint.iloc[0]
                assert int(record.covariance_rank) == rank and int(record.restrictions) == len(delta) and int(record.structural_rank_bound) == bound
                close(record.reference_covariance_scale, reference, "Joint HC0 reference scale")
                close(record.zero_covariance_threshold, zero, "Joint HC0-scaled rank threshold", atol=1e-20, rtol=1e-7)
                if covariance_name == "Sector cluster":
                    assert rank == 0, "Balanced H1 sector differences must expose sector-CR1 score collapse"
                    assert covariance_rank(matrix, hc0, len(sectors)-1)[0] <= 1
                if rank < len(delta):
                    assert str(record.status).startswith("unavailable") and record[["wald_chi2", "f_statistic", "p_value"]].isna().all()
                    counts["joint_degenerate_or_singular"] += 1
                else:
                    statistic = float(delta@np.linalg.solve(v, delta))
                    close([record.wald_chi2, record.f_statistic, record.p_value], [statistic, statistic/len(delta), f_distribution.sf(statistic/len(delta), len(delta), degrees)], "Full-rank equal-slopes omnibus")
                    counts["joint_full_rank"] += 1
                if covariance_name == "Driscoll-Kraay":
                    old, old_cov, alias = original_dk_reference(context, window, outcome, bandwidth)
                    if old.empty:
                        assert not context.endswith("-detailed") and outcome == "y_raw_level"
                        counts["new_DK_groups_without_prior_sector_fit"] += 1
                    else:
                        old = old.set_index("sector_id").loc[sectors]
                        for field in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "r_squared"]:
                            close(rows[field], old[field], "Preserved original DK "+field)
                        if old_cov is not None:
                            close(matrix, old_cov.pivot(index="sector_i", columns="sector_j", values="covariance").loc[sectors, sectors], "Preserved original full DK covariance")
                        counts["original_DK_groups_matched"] += 1
                details.append({"sample_id": sample_id, "outcome": outcome, "covariance": covariance_name, "hac_lags": bandwidth,
                                "n_obs": n, "n_periods": periods, "n_sectors": len(sectors), "n_parameters": k,
                                "inference_df": degrees, "finite_sample_correction": correction, "joint_rank": rank,
                                "joint_restrictions": len(delta), "joint_status": record.status})
            counts["model_outcomes"] += 1
    assert counts["slope_rows"] == len(tables["sector_slopes"]) and counts["component_rows"] == len(tables["interaction_components"])
    assert counts["slope_covariance_entries"] == len(tables["slope_covariances"]) and counts["component_covariance_entries"] == len(tables["component_covariances"])
    assert counts["all_coefficients"] == len(tables["all_coefficients"]) and len(details) == len(tables["joint_tests"])
    log = read(OUT / "pyfixest/fit_log.csv")
    assert len(log) == len(expected_native) and log.native_class.eq("Feols").all()
    assert log.estimator_version.astype(str).eq(importlib.metadata.version("pyfixest")).all()
    assert log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    for field in ["native_mapped_coefficient_max_abs_error", "native_residual_max_abs_error", "native_response_max_abs_error", "native_design_max_abs_error"]:
        assert log[field].eq(0).all()
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all() and log.native_residuals_sha256.eq(log.residuals_sha256).all()
    actual = [(next(sample_id for sample_id in specs.sample_id.unique() if sample_id in row.label), row.outcome, row.n_obs, row.n_parameters) for row in log.itertuples()]
    assert Counter(actual) == Counter(expected_native)
    manifest = json.loads((OUT / "pyfixest/estimator_manifest.json").read_text())
    assert manifest["fit_log_sha256"] == sha256(OUT / "pyfixest/fit_log.csv") and manifest["native_fits_recorded"] == len(log)
    analysis_manifest = json.loads((OUT / "manifest.json").read_text())
    for path, expected in analysis_manifest["protected_sources_sha256"].items():
        assert sha256(ROOT / path) == expected, "Original source/output changed after sensitivity fit: "+path
    for path, digest in sources.items():
        assert sha256(ROOT / path) == digest, "Original parent sample changed during audit"
    pd.DataFrame(details).to_csv(OUT / "verification_details.csv", index=False)
    report = {"status": "passed", "method": "Independent full-design QR and unit/date influence score sums, CR1/DK corrections, HC0-relative degeneracy and direct full-rank Wald",
              "counts": counts, "native_models_verified": len(log), "native_api_calls": log.api_call_id.nunique(),
              "zero_relative_tolerance": RELATIVE_ZERO, "original_parent_sample_hashes": sources,
              "original_sources_and_receipts_unchanged": len(analysis_manifest["protected_sources_sha256"]),
              "scope": "Numerical/sample/provenance verification of a requested sensitivity; does not validate sector CR1 inference for a coefficient supported by one sector or establish causality/stationarity"}
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Verified sector-clustered H1 sensitivity: {counts['model_outcomes']} native models, {counts['slope_rows']} slope rows, {counts['original_DK_groups_matched']} original DK groups preserved.", flush=True)
    return report


if __name__ == "__main__":
    verify_clustered_sector_sensitivity()
