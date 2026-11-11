"""Check detailed-sector FE interactions with independent full-design QR OLS.

Patsy constructs the saved raw-temperature formula.  The estimator's FWL
projection, covariance, inference, Wald and multiplicity functions are not
imported. Calendar-HAC is computed from full-OLS coefficient influences.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import f as f_distribution, t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/granular_interactions"
CONTEXTS = {"va-detailed": 19, "iip-monthly-detailed": 18, "iip-quarterly-detailed": 18}
OUTCOMES = {"G1": "y_short", "G2": "y_yoy", "G2N": "y_yoy_nsa", "L1": "y_log_level", "L3": "y_raw_level"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label: str, atol=2e-7, rtol=3e-7) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), (label, a, b)
    finite = np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    adjusted = np.minimum(1., np.maximum.accumulate(ordered*np.arange(len(values), 0, -1)))
    result = np.empty(len(values))
    result[order] = adjusted
    return result


def qr_fit(sample: pd.DataFrame, formula: str, outcomes: list[str], sectors: list[str]) -> dict:
    rhs = formula.split("~", 1)[1]
    design = dmatrix(rhs, sample, return_type="dataframe", NA_action="raise")
    x = design.to_numpy()
    y = sample[outcomes].to_numpy()
    scale = np.sqrt(np.mean(x*x, axis=0))
    normalized = x/scale
    q, r = np.linalg.qr(normalized, mode="reduced")
    assert np.min(np.abs(np.diag(r))) > 1e-10
    scaled_beta = np.linalg.solve(r, q.T@y)
    beta = scaled_beta/scale[:, None]
    residual = y-normalized@scaled_beta
    inverse_r = np.linalg.solve(r, np.eye(x.shape[1]))
    bread_scaled = inverse_r@inverse_r.T
    names = list(design.columns)
    # The native formula estimates a main T and S-1 deviations. Reported
    # absolute effects are explicit one-constraint vectors in that full basis.
    assert "temp_c" in names
    contrast = np.zeros((len(sectors), len(names)))
    contrast[:, names.index("temp_c")] = 1
    for i, sector in enumerate(sectors[1:], start=1):
        contrast[i, names.index(f"temp_c:C(sector_id)[T.{sector}]")] = 1
    # Full-design OLS influence for each absolute sector slope. This avoids the
    # production FWL projection and preserves every nuisance coefficient's role.
    slope_influence = normalized@bread_scaled@(contrast/scale).T
    return {"temperature_beta": contrast@beta, "residual": residual,
            "slope_influence": slope_influence, "n_parameters": x.shape[1],
            "r_squared": 1-np.sum(residual**2, axis=0)/np.sum((y-y.mean(axis=0))**2, axis=0)}


def calendar_covariance(fit: dict, sample: pd.DataFrame, period: str, outcome: int, bandwidth: int) -> np.ndarray:
    dates = sample[period].astype("int64").to_numpy()
    elapsed = dates-dates.min()
    individual = fit["slope_influence"]*fit["residual"][:, [outcome]]
    score = np.zeros((elapsed.max()+1, individual.shape[1]))
    # Explicitly aggregate each observed date; missing calendar dates remain zero.
    for value in np.unique(elapsed):
        score[value] = individual[elapsed==value].sum(axis=0)
    covariance = score.T@score
    for lag in range(1, bandwidth+1):
        products = score[lag:].T@score[:-lag]
        covariance += (1-lag/(bandwidth+1))*(products+products.T)
    n, periods, parameters = len(sample), sample[period].nunique(), fit["n_parameters"]
    correction = periods/(periods-1)*(n-1)/(n-parameters)
    covariance *= correction
    return (covariance+covariance.T)/2


def verify_granular_interactions() -> dict:
    source_hashes, records = {}, []
    maxima = {"beta": 0., "covariance": 0., "standard_error": 0., "p_value": 0., "CI": 0., "R_squared": 0., "pairwise_Holm_p": 0., "mean_slope_benchmark": 0.}
    coefficient_rows, covariance_rows, pair_rows, joint_rows, available_joints, unavailable_joints = 0, 0, 0, 0, 0, 0
    for context, count in CONTEXTS.items():
        monthly = "monthly" in context
        period, frequency, h = ("month", "M", 12) if monthly else ("quarter", "Q", 4)
        for window in ["long", "full", "exclude"]:
            folder = OUT / context / window
            files = {name: folder / f"{name}.csv" for name in ["sector_slopes", "joint_tests", "pairwise_tests", "slope_covariances", "common_sample"]}
            source_hashes.update({str(path.relative_to(ROOT)): sha256(path) for path in files.values()})
            sample = pd.read_csv(files["common_sample"], dtype={"sector_id": str})
            source_path = ROOT / "outputs/granular" / context / window / "samples/G1.csv"
            original = pd.read_csv(source_path, dtype={"sector_id": str})
            source_hashes[str(source_path.relative_to(ROOT))] = sha256(source_path)
            key = [period, "sector_id"]
            pd.testing.assert_frame_equal(sample.sort_values(key).reset_index(drop=True), original.sort_values(key).reset_index(drop=True),
                                          check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-12)
            for model in OUTCOMES:
                other = pd.read_csv(source_path.with_name(model+".csv"), dtype={"sector_id": str})
                pd.testing.assert_frame_equal(original.sort_values(key).reset_index(drop=True), other.sort_values(key).reset_index(drop=True),
                                              check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-12)
            sample[period] = pd.PeriodIndex(sample[period], freq=frequency)
            sample = sample.sort_values([period, "sector_id"]).reset_index(drop=True)
            assert not sample.duplicated(key).any()
            sectors = sorted(sample.sector_id.unique())
            assert len(sectors) == count and sample.groupby(period).sector_id.nunique().eq(count).all()
            assert sample.groupby(period)["temp_c"].nunique().eq(1).all()
            assert np.ptp(sample.time.to_numpy()-sample[period].astype("int64").to_numpy()) == 0
            if window == "exclude":
                assert not sample.year.between(2020, 2022).any()
            slopes = pd.read_csv(files["sector_slopes"], dtype={"sector_id": str})
            pairs = pd.read_csv(files["pairwise_tests"], dtype={"first_sector_id": str, "second_sector_id": str})
            joints = pd.read_csv(files["joint_tests"])
            saved_covariances = pd.read_csv(files["slope_covariances"], dtype={"sector_i": str, "sector_j": str})
            assert set(slopes.model) == set(pairs.model) == set(joints.model) == {"H1"}
            assert set(slopes.outcome) == set(pairs.outcome) == set(joints.outcome) == set(OUTCOMES.values())
            assert set(slopes.hac_lags) == set(pairs.hac_lags) == set(joints.hac_lags) == {h, 2*h}
            assert not slopes.duplicated(["outcome", "hac_lags", "sector_id"]).any()
            assert not pairs.duplicated(["outcome", "hac_lags", "first_sector_id", "second_sector_id"]).any()
            assert not joints.duplicated(["outcome", "hac_lags"]).any()
            assert not saved_covariances.duplicated(["outcome", "hac_lags", "sector_i", "sector_j"]).any()
            outcome_bands = len(OUTCOMES)*2
            assert len(slopes) == count*outcome_bands and len(pairs) == count*(count-1)//2*outcome_bands and len(joints) == outcome_bands
            assert len(saved_covariances) == count**2*outcome_bands
            formulas = {row.outcome: row.formula for row in slopes.itertuples()}
            assert all(value.split("~", 1)[1] == next(iter(formulas.values())).split("~", 1)[1] for value in formulas.values())
            for formula in formulas.values():
                assert "temp_c * C(sector_id)" in formula and "C(sector_id)*C(season)" in formula and "C(year)" in formula
                assert "C(sector_id):rain" not in formula and "C(sector_id):rh" not in formula
            outcomes = list(OUTCOMES.values())
            fit = qr_fit(sample, slopes.formula.iloc[0], outcomes, sectors)
            periods, n, k = sample[period].nunique(), len(sample), fit["n_parameters"]
            critical = student_t.ppf(.975, periods-1)
            controls = ["rain_100mm", "rh_pct"]+([] if window=="long" else ["wind_mps", "pm25_10ug"])
            expected_k = count*sample.season.nunique()+sample.year.nunique()-1+len(controls)+count
            assert k == expected_k
            benchmark_path = ROOT / "outputs/granular" / context / window / "regression_results.csv"
            benchmark = pd.read_csv(benchmark_path)
            source_hashes[str(benchmark_path.relative_to(ROOT))] = sha256(benchmark_path)
            bounds, group_errors = [], []
            for model, outcome in OUTCOMES.items():
                j = outcomes.index(outcome)
                beta = fit["temperature_beta"][:, j]
                comparison = benchmark[benchmark.model.eq(model)&benchmark.term.eq("temp_c")].coefficient
                assert len(comparison) and comparison.max()-comparison.min() < 1e-9
                maxima["mean_slope_benchmark"] = max(maxima["mean_slope_benchmark"], close(beta.mean(), comparison.iloc[0], "Mean interaction slope versus common-slope benchmark", atol=2e-7))
                for bandwidth in [h, 2*h]:
                    covariance = calendar_covariance(fit, sample, period, j, bandwidth)
                    stored_covariance = saved_covariances[saved_covariances.outcome.eq(outcome)&saved_covariances.hac_lags.eq(bandwidth)].pivot(index="sector_i", columns="sector_j", values="covariance").loc[sectors, sectors].to_numpy()
                    maxima["covariance"] = max(maxima["covariance"], close(stored_covariance, covariance, f"Full-OLS calendar covariance {context}/{window}/{outcome}/{bandwidth}", atol=4e-6, rtol=5e-7))
                    observed = slopes[slopes.outcome.eq(outcome)&slopes.hac_lags.eq(bandwidth)].set_index("sector_id").loc[sectors]
                    se = np.sqrt(np.maximum(np.diag(covariance), 0))
                    p = 2*student_t.sf(np.abs(beta/se), periods-1)
                    maxima["beta"] = max(maxima["beta"], close(observed.coefficient.to_numpy(), beta, "Absolute sector slopes", atol=2e-7))
                    maxima["standard_error"] = max(maxima["standard_error"], close(observed.std_error.to_numpy(), se, "Sector HAC SE", atol=5e-7))
                    maxima["p_value"] = max(maxima["p_value"], close(observed.p_value.to_numpy(), p, "Sector t(T-1) p", atol=2e-8))
                    maxima["CI"] = max(maxima["CI"], close(observed[["ci_lower", "ci_upper"]].to_numpy(), np.column_stack([beta-critical*se, beta+critical*se]), "Sector confidence interval", atol=2e-6))
                    maxima["R_squared"] = max(maxima["R_squared"], close(observed.r_squared.to_numpy(), np.repeat(fit["r_squared"][j], count), "Full-model R squared", atol=2e-10))
                    assert observed.n_obs.eq(n).all() and observed.n_parameters.eq(k).all() and observed.n_sectors.eq(count).all()
                    assert observed.n_periods.eq(periods).all() and observed.inference_df.eq(periods-1).all()
                    assert observed.start.eq(str(sample[period].min())).all() and observed.end.eq(str(sample[period].max())).all()
                    means = sample.groupby("sector_id")[outcome].mean().loc[sectors].to_numpy()
                    standard_deviations = sample.groupby("sector_id")[outcome].std().loc[sectors].to_numpy()
                    close(observed.mean_y.to_numpy(), means, "Sector exact-sample means", atol=2e-10)
                    close(observed.sd_y.to_numpy(), standard_deviations, "Sector exact-sample SDs", atol=2e-10)
                    joint = joints[joints.outcome.eq(outcome)&joints.hac_lags.eq(bandwidth)]
                    assert len(joint) == 1
                    row = joint.iloc[0]
                    restrictions = np.eye(count)[1:]-np.eye(count)[[0]]
                    difference = restrictions@beta
                    contrast_covariance = restrictions@covariance@restrictions.T
                    singular_values = np.linalg.svd(contrast_covariance, compute_uv=False)
                    bound = periods-sample.season.nunique()-1
                    rank = min(int(np.sum(singular_values>singular_values.max()*1e-10)), bound)
                    assert int(row.structural_rank_bound) == bound and int(row.covariance_rank) == rank
                    assert int(row.restrictions) == count-1 and int(row.numerator_df) == count-1 and int(row.denominator_df) == periods-1
                    if rank < count-1:
                        assert row.status == "unavailable: singular restriction covariance"
                        assert pd.isna(row.p_value) and pd.isna(row.wald_chi2) and pd.isna(row.f_statistic)
                        unavailable_joints += 1
                    else:
                        assert row.status == "available"
                        statistic = float(difference@np.linalg.solve(contrast_covariance, difference))
                        assert statistic >= 0
                        close(row.wald_chi2, statistic, "Full-rank omnibus Wald", atol=1e-5, rtol=3e-6)
                        close(row.f_statistic, statistic/(count-1), "Omnibus F", atol=1e-6, rtol=3e-6)
                        close(row.p_value, f_distribution.sf(statistic/(count-1), count-1, periods-1), "Omnibus F p", atol=2e-8, rtol=3e-6)
                        available_joints += 1
                    family = pairs[pairs.outcome.eq(outcome)&pairs.hac_lags.eq(bandwidth)]
                    expected_pairs = set(itertools.combinations(sectors, 2))
                    assert set(family[["first_sector_id", "second_sector_id"]].itertuples(index=False, name=None)) == expected_pairs
                    p_values = []
                    for pair in family.itertuples():
                        first, second = sectors.index(pair.first_sector_id), sectors.index(pair.second_sector_id)
                        effect = beta[first]-beta[second]
                        variance = covariance[first, first]+covariance[second, second]-2*covariance[first, second]
                        assert variance > 0
                        standard_error = np.sqrt(variance)
                        probability = float(2*student_t.sf(abs(effect/standard_error), periods-1))
                        close(pair.coefficient, effect, "Pairwise sector difference", atol=3e-7)
                        close(pair.std_error, standard_error, "Pairwise covariance-based HAC SE", atol=8e-7)
                        close(pair.p_value, probability, "Pairwise t p", atol=3e-8)
                        close([pair.ci_lower, pair.ci_upper], [effect-critical*standard_error, effect+critical*standard_error], "Pairwise CI", atol=3e-6)
                        assert pair.inference_df == periods-1
                        assert pair.multiplicity_family == f"All {len(expected_pairs)} sector pairs in this context/window/outcome/bandwidth"
                        p_values.append(probability)
                    maxima["pairwise_Holm_p"] = max(maxima["pairwise_Holm_p"], close(family.p_value_holm.to_numpy(), holm_adjust(np.asarray(p_values)), "All-pairs Holm family", atol=4e-8))
                    coefficient_rows += len(observed);covariance_rows += len(stored_covariance)**2;pair_rows += len(family);joint_rows += 1
                    bounds.append(int(bound));group_errors.append(float(np.max(np.abs(observed.coefficient.to_numpy()-beta))))
            records.append({"context": context, "window": window, "n_obs": n, "n_periods": int(periods), "n_sectors": count,
                            "n_parameters": k, "calendar_periods": sample[period].max().ordinal-sample[period].min().ordinal+1,
                            "structural_contrast_rank_bound": bounds[0], "max_beta_absolute_error": max(group_errors)})
    outcome_windows_bands = len(OUTCOMES)*3*2
    expected_joints = len(CONTEXTS)*outcome_windows_bands
    assert coefficient_rows == sum(CONTEXTS.values())*outcome_windows_bands
    assert covariance_rows == sum(count**2 for count in CONTEXTS.values())*outcome_windows_bands
    assert pair_rows == sum(count*(count-1)//2 for count in CONTEXTS.values())*outcome_windows_bands
    assert joint_rows == expected_joints
    assert len(records) == len(CONTEXTS)*3 and available_joints+unavailable_joints == expected_joints
    for relative, digest in source_hashes.items():
        assert sha256(ROOT / relative) == digest, "Input changed during verification: "+relative
    report = {"status": "passed", "method": "Independent raw-formula full-design QR OLS and calendar-HAC slope influences; t(T-1); direct full-rank Wald; manually sorted Holm families",
              "model_outcomes_checked": len(CONTEXTS)*3*len(OUTCOMES), "sector_slope_rows_checked": coefficient_rows, "covariance_entries_checked": covariance_rows,
              "pairwise_rows_checked": pair_rows, "joint_rows_checked": joint_rows,
              "available_full_rank_joint_tests": available_joints, "singular_joint_tests_correctly_unavailable": unavailable_joints,
              "maximum_absolute_errors": maxima, "contexts_and_windows": records, "inputs_sha256": source_hashes,
              "scope": "Numerical and exact-sample verification; uncertainty remains conditional and common weather does not create independent sector exposures"}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Verified detailed FE interactions: {coefficient_rows} sector slopes, {pair_rows} pairwise comparisons and {joint_rows} omnibus checks.", flush=True)
    return report


if __name__ == "__main__":
    verify_granular_interactions()
