"""Verify native-Pyfixest provenance, pre-migration equality and slope sums.

The tolerance is abs(new-reference) <= 1e-7 + 1e-7*abs(reference).
Raw-level coefficients/covariances have much larger units; actual absolute,
relative-to-max(1,abs(reference)) and tolerance-scaled errors are all reported.
Unstable covariance-condition numbers are descriptive diagnostics, not targets
of numerical equality. Singular-test availability and every covariance entry
remain targets of equality and independent regression verification.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix, dmatrices
from scipy.stats import t as student_t, f as f_distribution
from provenance_omissions import omitted_artifact

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/reference_before_pyfixest/outputs"
OUT = ROOT / "outputs/pyfixest"
ATOL, RTOL = 1e-7, 1e-7
IDENTITIES = {"model", "outcome", "term", "sector_id", "sector_i", "sector_j", "first_sector_id", "second_sector_id", "reference_sector_id"}
EXACT_NUMERIC = {"n_obs", "n_sectors", "n_periods", "n_months", "n_quarters", "n_parameters", "n_clusters", "hac_lags", "inference_df", "restrictions", "covariance_rank", "numerator_df", "denominator_df", "structural_rank_bound", "residual_df", "time_residual_df", "max_lag"}
TEXT_CHANGES = {"label", "formula", "estimand"}
PRODUCTION = ["estimate.py", "iip_analysis.py", "extended_controls.py", "outcome_analysis.py", "aggregate_va.py", "granular_analysis.py", "heterogeneity.py", "granular_interactions.py", "sector_temperature.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={name: str for name in IDENTITIES})


def numeric_errors(current, reference, label: str, *, exact=False) -> dict:
    a, b = np.asarray(current, dtype=float), np.asarray(reference, dtype=float)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b)), (label, "NaN availability changed")
    assert np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(np.isneginf(a), np.isneginf(b)), (label, "Infinity changed")
    finite = np.isfinite(a)&np.isfinite(b)
    difference = np.abs(a[finite]-b[finite])
    tolerance = ATOL+RTOL*np.abs(b[finite])
    if exact:
        assert np.all(difference == 0), (label, "Count/degree metadata changed")
    else:
        assert np.all(difference <= tolerance), (label, float(difference.max()), float(np.max(difference/tolerance)))
    return {"finite_values": int(finite.sum()), "max_absolute_error": float(difference.max()) if len(difference) else 0.,
            "max_scaled_relative_error": float(np.max(difference/np.maximum(1., np.abs(b[finite])))) if len(difference) else 0.,
            "max_fraction_of_tolerance": float(np.max(difference/tolerance)) if len(difference) else 0.}


def row_keys(frame: pd.DataFrame, name: str) -> list[str]:
    keys = [col for col in ["model", "outcome", "term", "sector_id", "first_sector_id", "second_sector_id", "sector_i", "sector_j", "hac_lags", "test"] if col in frame]
    if "covariance" in frame and not pd.api.types.is_numeric_dtype(frame.covariance):
        keys.append("covariance")
    assert keys and not frame.duplicated(keys).any(), (name, keys, "Nonunique reference row keys")
    return keys


def aligned(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(frame[keys].fillna("__MISSING__").astype(str))
    result = frame.copy()
    result.index = index
    return result.sort_index()


def sample_for_formula(path: Path, model: str) -> pd.DataFrame:
    common = path.parent / "common_sample.csv"
    if common.exists():
        sample = read_csv(common)
        return sample[~sample.year.between(2020, 2022)].copy() if model == "H3X" else sample
    from verify_sector_cluster import exact_sample
    return exact_sample(path, model)


def equivalent_formula(old: str, new: str, sample: pd.DataFrame) -> dict:
    lhs_old, rhs_old = old.split("~", 1)
    lhs_new, rhs_new = new.split("~", 1)
    assert lhs_old.strip() == lhs_new.strip()
    matrices = [dmatrix(rhs, sample, return_type="dataframe", NA_action="raise").to_numpy() for rhs in [rhs_old, rhs_new]]
    qs = []
    for matrix in matrices:
        normalized = matrix/np.linalg.norm(matrix, axis=0)
        q, r = np.linalg.qr(normalized, mode="reduced")
        assert np.min(np.abs(np.diag(r))) > 1e-10
        qs.append(q)
    assert matrices[0].shape == matrices[1].shape
    error = max(float(np.max(np.abs(qs[0]-qs[1]@(qs[1].T@qs[0])))), float(np.max(np.abs(qs[1]-qs[0]@(qs[0].T@qs[1])))))
    assert error < 1e-9, (old, new, error)
    return {"n_obs": len(sample), "n_parameters": matrices[0].shape[1], "max_orthogonal_space_residual": error}


def compare_csv(reference: Path) -> tuple[dict, list[dict]]:
    relative = reference.relative_to(REFERENCE)
    current_path = ROOT / "outputs" / relative
    assert current_path.exists(), current_path
    before, after = read_csv(reference), read_csv(current_path)
    assert set(before.columns) <= set(after.columns), (relative, "Reference fields missing", set(before)-set(after))
    keys = row_keys(before, str(relative))
    old, new = aligned(before, keys), aligned(after, keys)
    assert not new.index.duplicated().any()
    assert old.index.isin(new.index).all(), (relative, "Original SA coefficient/model row missing")
    extra = new.loc[~new.index.isin(old.index)]
    if len(extra):
        identified_nsa = extra.outcome.astype(str).str.endswith("_nsa") if "outcome" in extra else extra.model.astype(str).str.fullmatch(r"(?:G[23]N|Y[1-4]N)X?")
        assert identified_nsa.all(), (relative, "Unexpected added non-NSA source rows")
    new = new.loc[old.index]
    results, formula_checks, text_changes, skipped = [], [], {}, []
    for column in before.columns:
        if column in keys:
            continue
        if column == "covariance_condition":
            skipped.append(column)
            continue
        if pd.api.types.is_numeric_dtype(before[column]) or pd.api.types.is_bool_dtype(before[column]):
            errors = numeric_errors(new[column], old[column], f"{relative}:{column}", exact=column in EXACT_NUMERIC)
            results.append({"source_file": str(relative), "field": column, **errors})
        else:
            a, b = new[column].fillna("__MISSING__").astype(str), old[column].fillna("__MISSING__").astype(str)
            changed = a.ne(b)
            if changed.any():
                assert column in TEXT_CHANGES, (relative, column, "Semantic metadata changed", a[changed].head().tolist(), b[changed].head().tolist())
                text_changes[column] = int(changed.sum())
                if column == "formula":
                    pairs = pd.DataFrame({"old": old.loc[changed, column], "new": new.loc[changed, column], "model": new.loc[changed, "model"]}).drop_duplicates()
                    seen = set()
                    for row in pairs.itertuples(index=False):
                        # Outcome changes do not alter the RHS space; avoid repeating
                        # identical large QR decompositions for the four Y definitions.
                        space_key = (row.old.split("~", 1)[1], row.new.split("~", 1)[1], row.model)
                        if space_key in seen:
                            continue
                        seen.add(space_key)
                        check = equivalent_formula(row.old, row.new, sample_for_formula(current_path, row.model))
                        formula_checks.append({"model": row.model, "old_formula": row.old, "new_formula": row.new, **check})
    return {"file": str(relative), "rows": len(before), "additional_explicit_NSA_rows": len(extra), "reference_sha256": sha256(reference), "current_sha256": sha256(current_path),
            "row_keys": keys, "numeric_fields_checked": len(results), "allowed_text_changes": text_changes,
            "equivalent_formula_design_checks": formula_checks, "unstable_diagnostics_excluded": skipped}, results


def compare_json(reference: Path) -> tuple[dict, list[dict]]:
    relative = reference.relative_to(REFERENCE)
    current = ROOT / "outputs" / relative
    old, new = json.loads(reference.read_text()), json.loads(current.read_text())
    records = []
    def compare(a, b, key=""):
        if isinstance(b, dict):
            assert isinstance(a, dict) and set(b) <= set(a), (relative, key, "JSON keys removed")
            for name, value in b.items():
                compare(a[name], value, key+"."+name)
        elif b is None:
            assert a is None, (relative, key, "Unavailable JSON result changed")
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            records.append({"source_file": str(relative), "field": key, **numeric_errors(a, b, str(relative)+key, exact=key.rsplit(".", 1)[-1] in EXACT_NUMERIC)})
        else:
            assert a == b, (relative, key, "JSON semantic metadata changed")
    compare(new, old)
    return {"file": str(relative), "reference_sha256": sha256(reference), "current_sha256": sha256(current), "numeric_fields_checked": len(records)}, records


def verify_input_preservation() -> dict:
    path = OUT / "migration_input_reference.json"
    reference = json.loads(path.read_text())["files"]
    preserved_hashes, extended_NSA_columns, extended_manifests = 0, [], []
    omitted = []
    for relative, row in reference.items():
        omission = omitted_artifact(ROOT, relative, row)
        if omission is not None:
            omitted.append(omission)
            continue
        current = ROOT / relative
        assert current.exists(), (relative, "Missing analytical input or required provenance manifest")
        if current.stat().st_size == row["bytes"] and sha256(current) == row["sha256"]:
            preserved_hashes += 1
            continue
        frozen = ROOT / "data/reference_before_nsa_inputs" / relative
        assert frozen.exists() and sha256(frozen) == row["sha256"], (relative, "No authenticated pre-NSA source checkpoint")
        if relative == "data/raw/manifest.json":
            before, after = json.loads(frozen.read_text()), json.loads(current.read_text())
            assert set(before) <= set(after) and all(after[key] == value for key, value in before.items())
            extended_manifests.append(relative)
            continue
        assert current.suffix == ".csv" and (relative.startswith("data/processed/") or relative.startswith("outputs/")), (relative, "Original raw/source changed")
        before, after = read_csv(frozen), read_csv(current)
        assert len(before) == len(after) and set(before) <= set(after)
        added = set(after)-set(before)
        assert added and added <= {"output_nsa", "y_yoy_nsa", "y_yoy_pct_nsa"}, (relative, "Unexpected input-column change", added)
        keys = [name for name in ["sector_id", "month", "quarter"] if name in before]
        if keys:
            before, after = before.sort_values(keys).reset_index(drop=True), after.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(after[list(before)], before, check_dtype=False, check_exact=False, atol=1e-11, rtol=1e-12)
        extended_NSA_columns.append({"file": relative, "additional_columns": sorted(added), "original_rows_and_columns_unchanged": True})
    full = []
    for context in ["va", "iip-monthly", "iip-quarterly"]:
        sample = read_csv(ROOT / "outputs/extended_controls" / context / "baseline_sample.csv")
        period = "month" if context == "iip-monthly" else "quarter"
        assert sample[period].nunique() == (100 if period == "month" else 29)
        assert sample[["wind_mps", "pm25_10ug", "rain_100mm", "rh_pct", "temp_c"]].notna().all().all()
        no_covid = sample[~sample.year.between(2020, 2022)]
        assert no_covid[period].nunique() == (66 if period == "month" else 19)
        full.append({"context": context, "periods": sample[period].nunique(), "n_obs": len(sample), "periods_excluding_2020_22": no_covid[period].nunique()})
    return {"source_and_exact_sample_files_referenced": len(reference),
            "source_and_exact_sample_files_checked": len(reference)-len(omitted), "files_with_original_hash": preserved_hashes,
            "reference_manifest_sha256": sha256(path), "provenance_only_omitted_unverified_count": len(omitted),
            "provenance_only_omitted_unverified": omitted,
            "verification_scope": "All analytical inputs and required citation manifests remain checked; only the exact absent third-party provenance artifacts listed here are unverified.",
            "files_with_only_explicit_NSA_columns_added": extended_NSA_columns, "manifests_with_only_new_source_entries": extended_manifests,
            "all_original_SA_rows_and_columns_unchanged": True, "all_controls_PM_wind_baselines_retained": full}


def linear_effect_receipt(*, source: str, model: str, outcome: str, bandwidth: int, sector: str,
                          reference: str, estimate: float, variance: float, degrees: int,
                          reported_p: float, relative=False) -> dict:
    """One linear restriction, distinct from jointly setting two coefficients to zero."""
    assert variance > 0 and degrees > 0
    statistic = estimate/math.sqrt(variance)
    wald = estimate**2/variance
    probability_t = float(2*student_t.sf(abs(statistic), degrees))
    probability_f = float(f_distribution.sf(wald, 1, degrees))
    numeric_errors(wald, statistic**2, "One-constraint Wald equals squared t")
    numeric_errors(probability_t, probability_f, "Two-sided t equals F(1,T-1) p")
    numeric_errors(reported_p, probability_t, "Reported sector significance tests total linear effect")
    null = "delta_sector = 0 (relative to reference; no identified common main effect)" if relative else "beta_temperature = 0 (reference interaction normalized to zero)" if sector == reference else "beta_temperature + delta_sector = 0"
    return {"source_csv": source, "model": model, "outcome": outcome, "hac_lags": bandwidth,
            "sector_id": sector, "reference_sector_id": reference, "null": null,
            "effect_type": "relative_to_reference" if relative else "absolute_sector_total",
            "estimate": estimate, "variance_of_linear_combination": variance, "t_statistic": statistic,
            "wald_chi2": wald, "f_statistic": wald, "numerator_df": 1, "denominator_df": degrees,
            "p_from_t": probability_t, "p_from_one_restriction_F": probability_f, "reported_sector_p": reported_p,
            "t_F_p_absolute_error": abs(probability_t-probability_f), "reported_p_absolute_error": abs(reported_p-probability_t),
            "tests_beta_and_delta_jointly_zero": False}


def verify_components() -> dict:
    files = sorted([*(ROOT / "outputs/heterogeneity").glob("*/*/interaction_components.csv"), *(ROOT / "outputs/granular_interactions").glob("*/*/interaction_components.csv")])
    assert len(files) == 15, len(files)
    groups, totals_checked, relative_groups, normalization_rows = 0, 0, 0, 0
    sector_null_checks, omnibus_checks, sample_cache = [], [], {}
    maxima = {"coefficient_sum": 0., "variance_with_cross_term": 0., "all_total_covariance_entries": 0., "component_inference": 0.}
    for path in files:
        components = read_csv(path)
        covariance_path = path.with_name("component_covariances.csv")
        covariance = read_csv(covariance_path)
        slopes = read_csv(path.with_name("sector_slopes.csv"))
        joint_tests = read_csv(path.with_name("joint_tests.csv"))
        total_covariance = read_csv(path.with_name("slope_covariances.csv"))
        assert not components.duplicated(["model", "outcome", "hac_lags", "term"]).any()
        assert not covariance.duplicated(["model", "outcome", "hac_lags", "term_i", "term_j"]).any()
        for (model, outcome, bandwidth), all_group in components.groupby(["model", "outcome", "hac_lags"], sort=False):
            assert {"estimated", "normalization"} <= set(all_group.columns)
            normalization = all_group[all_group.normalization.eq(True)]
            group = all_group[all_group.estimated.eq(True)]
            assert not group.normalization.any() and len(group)+len(normalization) == len(all_group)
            selector = covariance.model.eq(model)&covariance.outcome.eq(outcome)&covariance.hac_lags.eq(bandwidth)
            matrix_rows = covariance[selector]
            terms = group.term.tolist()
            matrix = matrix_rows.pivot(index="term_i", columns="term_j", values="covariance").loc[terms, terms].to_numpy()
            assert len(matrix_rows) == len(terms)**2 and np.isfinite(matrix).all()
            numeric_errors(matrix, matrix.T, "Native component covariance symmetry")
            reference = group.reference_sector_id.unique()
            assert len(reference) == 1
            reference = reference[0]
            assert group.term.nunique() == len(group)
            selected_slopes = slopes[slopes.model.eq(model)&slopes.outcome.eq(outcome)&slopes.hac_lags.eq(bandwidth)].set_index("sector_id")
            selected_covariance = total_covariance[total_covariance.model.eq(model)&total_covariance.outcome.eq(outcome)&total_covariance.hac_lags.eq(bandwidth)]
            degrees = int(selected_slopes.inference_df.iloc[0])
            critical = student_t.ppf(.975, degrees)
            for i, component in enumerate(group.itertuples()):
                standard_error = math.sqrt(max(0., float(matrix[i, i])))
                for field, expected in [("std_error", standard_error), ("ci_lower", component.coefficient-critical*standard_error), ("ci_upper", component.coefficient+critical*standard_error),
                                        ("p_value", float(2*student_t.sf(abs(component.coefficient/standard_error), degrees)))]:
                    error = numeric_errors(getattr(component, field), expected, "Component inference "+field)
                    maxima["component_inference"] = max(maxima["component_inference"], error["max_scaled_relative_error"])
            if model == "R1":
                assert normalization.empty
                assert not group.component.eq("base").any() and set(group.component) == {"interaction"}
                assert reference not in set(group.sector_id)
                assert set(group.sector_id) == set(selected_slopes.index)
                relative_groups += 1
                mapping = {row.sector_id: np.eye(len(group))[i] for i, row in enumerate(group.itertuples())}
            else:
                assert len(normalization) == 1 and normalization.sector_id.iloc[0] == reference
                assert normalization.component.iloc[0] == "interaction" and normalization.coefficient.iloc[0] == 0
                assert normalization[["std_error", "ci_lower", "ci_upper", "p_value"]].isna().all().all()
                assert not normalization.estimated.any()
                normalization_rows += 1
                base = group[group.component.eq("base")]
                assert len(base) == 1 and base.term.iloc[0] == "temp_c" and base.sector_id.iloc[0] == reference
                interactions = group[group.component.eq("interaction")]
                assert len(interactions) == len(group)-1 and reference not in set(interactions.sector_id)
                assert set(group.sector_id) == set(selected_slopes.index)
                base_position = terms.index("temp_c")
                mapping = {}
                for sector in selected_slopes.index:
                    weights = np.eye(len(group))[base_position].copy()
                    if sector != reference:
                        delta = interactions[interactions.sector_id.eq(sector)]
                        assert len(delta) == 1 and delta.term.iloc[0] == f"temp_c:C(sector_id)[T.{sector}]"
                        weights[terms.index(delta.term.iloc[0])] = 1
                    mapping[sector] = weights
            coefficients = group.coefficient.to_numpy()
            for sector, weights in mapping.items():
                absolute = selected_slopes.loc[sector]
                summed = float(weights@coefficients)
                variance = float(weights@matrix@weights)
                error = numeric_errors(absolute.coefficient, summed, "Main temperature plus interaction total")
                maxima["coefficient_sum"] = max(maxima["coefficient_sum"], error["max_absolute_error"])
                error = numeric_errors(absolute.std_error**2, variance, "Total variance includes both covariance cross terms")
                maxima["variance_with_cross_term"] = max(maxima["variance_with_cross_term"], error["max_scaled_relative_error"])
                assert variance > 0
                sector_null_checks.append(linear_effect_receipt(source=str(path.with_name("sector_slopes.csv").relative_to(ROOT)),
                    model=model, outcome=outcome, bandwidth=int(bandwidth), sector=sector, reference=reference,
                    estimate=summed, variance=variance, degrees=degrees, reported_p=float(absolute.p_value), relative=model=="R1"))
                totals_checked += 1
            sectors = list(selected_slopes.index)
            transform = np.vstack([mapping[sector] for sector in sectors])
            transformed = transform@matrix@transform.T
            actual = selected_covariance.pivot(index="sector_i", columns="sector_j", values="covariance").loc[sectors, sectors].to_numpy()
            error = numeric_errors(actual, transformed, "Full covariance of summed absolute sector slopes")
            maxima["all_total_covariance_entries"] = max(maxima["all_total_covariance_entries"], error["max_scaled_relative_error"])
            # Equal sector slopes means all nonreference deltas are zero. The
            # common main temperature coefficient is unrestricted in this null.
            if model == "R1":
                component_restriction = np.eye(len(group))
                absolute_restriction = np.eye(len(sectors))
            else:
                component_restriction = np.eye(len(group))[[i for i, term in enumerate(terms) if term != "temp_c"]]
                reference_position = sectors.index(reference)
                absolute_restriction = np.vstack([np.eye(len(sectors))[i]-np.eye(len(sectors))[reference_position]
                                                     for i, sector in enumerate(sectors) if sector != reference])
            delta = component_restriction@coefficients
            delta_covariance = component_restriction@matrix@component_restriction.T
            absolute_difference = absolute_restriction@selected_slopes.loc[sectors].coefficient.to_numpy()
            absolute_contrast_covariance = absolute_restriction@actual@absolute_restriction.T
            numeric_errors(delta, absolute_difference, "Omnibus equal-slopes point contrast basis invariance")
            numeric_errors(delta_covariance, absolute_contrast_covariance, "Omnibus equal-slopes covariance basis invariance")
            joint = joint_tests[joint_tests.model.eq(model)&joint_tests.outcome.eq(outcome)&joint_tests.hac_lags.eq(bandwidth)]
            assert len(joint) == 1
            joint = joint.iloc[0]
            assert np.array_equal(absolute_restriction@transform, component_restriction)
            sample_key = (str(path.parent), model == "H3X")
            if sample_key not in sample_cache:
                sample_cache[sample_key] = sample_for_formula(path.with_name("sector_slopes.csv"), model)
            estimation_sample = sample_cache[sample_key]
            periods = estimation_sample.time.nunique()
            season_count, year_count = estimation_sample.season.nunique(), estimation_sample.year.nunique()
            weather_count = 3+(2 if path.parent.name == "full" else 0)
            if "granular_interactions" in path.parts or model == "H1":
                independent_bound = periods-season_count-1
            elif model == "H2":
                independent_bound = periods-season_count-weather_count
            else:
                independent_bound = periods-(season_count+year_count-1+weather_count)
            assert int(joint.n_obs) == len(estimation_sample) and int(joint.n_periods) == periods and degrees == periods-1
            singular = np.linalg.svd(delta_covariance, compute_uv=False)
            numerical_rank = int(np.sum(singular>singular.max()*1e-10))
            rank_bound = int(joint.structural_rank_bound)
            assert rank_bound >= independent_bound
            rank = min(numerical_rank, independent_bound)
            restrictions = len(delta)
            assert int(joint.covariance_rank) == rank and int(joint.restrictions) == restrictions
            assert int(joint.numerator_df) == restrictions and int(joint.denominator_df) == degrees
            if rank < restrictions:
                assert str(joint.status).startswith("unavailable")
                assert pd.isna(joint.p_value) and pd.isna(joint.wald_chi2) and pd.isna(joint.f_statistic)
                wald_component = wald_absolute = probability = None
            else:
                assert joint.status == "available"
                wald_component = float(delta@np.linalg.solve(delta_covariance, delta))
                wald_absolute = float(absolute_difference@np.linalg.solve(absolute_contrast_covariance, absolute_difference))
                numeric_errors(wald_component, wald_absolute, "Full-rank omnibus statistic basis invariance")
                numeric_errors(joint.wald_chi2, wald_component, "Omnibus tests all nonreference deltas zero")
                numeric_errors(joint.f_statistic, wald_component/restrictions, "Omnibus F normalization")
                probability = float(f_distribution.sf(wald_component/restrictions, restrictions, degrees))
                numeric_errors(joint.p_value, probability, "Omnibus F(q,T-1) p")
            omnibus_checks.append({"source_csv": str(path.with_name("joint_tests.csv").relative_to(ROOT)), "model": model,
                "outcome": outcome, "hac_lags": int(bandwidth), "null": "All nonreference delta_sector = 0; all sector slopes equal",
                "main_temperature_constrained_to_zero": False, "restrictions": restrictions, "numerator_df": restrictions,
                "denominator_df": degrees, "structural_rank_bound": rank_bound, "covariance_rank": rank,
                "independent_residual_score_rank_bound": int(independent_bound),
                "status": joint.status, "wald_in_component_basis": wald_component, "wald_in_absolute_contrast_basis": wald_absolute,
                "p_from_component_basis": probability, "reported_omnibus_p": None if pd.isna(joint.p_value) else float(joint.p_value),
                "singular_covariance_inverted_or_pseudoinverted": False})
            groups += 1
    expected_groups = sum(read_csv(path).groupby(["model", "outcome", "hac_lags"]).ngroups for path in files)
    assert groups == expected_groups and normalization_rows == groups-relative_groups
    return {"files_checked": len(files), "model_outcome_bandwidth_groups_checked": groups,
            "relative_R1_groups_without_identified_base": relative_groups, "reported_total_or_relative_slopes_checked": totals_checked,
            "marked_reference_zero_normalizations_without_SE_or_p": normalization_rows,
            "reference_sector_fake_estimated_zero_interaction_rows": 0, "maximum_errors": maxima,
            "one_restriction_sector_p_values_checked": len(sector_null_checks), "equal_slopes_omnibus_nulls_checked": len(omnibus_checks),
            "_sector_null_checks": sector_null_checks, "_omnibus_checks": omnibus_checks}


def static_production_audit() -> dict:
    checks = []
    for name in PRODUCTION:
        path = ROOT / "scripts" / name
        tree = ast.parse(path.read_text())
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if call in {"smf.ols", "sm.OLS", "statsmodels.api.OLS", "np.linalg.lstsq", "numpy.linalg.lstsq"}:
                    forbidden.append({"line": node.lineno, "call": call})
        assert not forbidden, (name, "Remaining non-Pyfixest production FE estimation", forbidden)
        checks.append({"file": name, "sha256": sha256(path), "remaining_statsmodels_OLS_or_numpy_lstsq_calls": 0})
    # Numpy operations used for covariance geometry, linear contrasts, Wald
    # solves and independent diagnostics/verification remain permitted.
    return {"production_modules_checked": checks, "allowed_nonproduction_OLS": "Stationarity diagnostic auxiliary regressions and standalone independent verification"}


def verify_original_reference_slopes() -> dict:
    """Check the three original interaction fits in their new literal T+delta basis."""
    fits, slope_rows, maximum = 0, 0, 0.
    null_checks = []
    for prefix in [ROOT / "outputs", ROOT / "outputs/iip/monthly", ROOT / "outputs/iip/quarterly"]:
        path = prefix / "sector_slopes.csv"
        rows = read_csv(path)
        assert rows.formula.nunique() == 1
        assert {"temperature_main", "temperature_interaction", "reference_sector_id"} <= set(rows)
        sample = sample_for_formula(path, "H1")
        period, frequency = ("month", "M") if "monthly" in prefix.parts else ("quarter", "Q")
        sample[period] = pd.PeriodIndex(sample[period], freq=frequency)
        y, design = dmatrices(rows.formula.iloc[0], sample, return_type="dataframe", NA_action="raise")
        x, response = design.to_numpy(), y.iloc[:, 0].to_numpy()
        scale = np.linalg.norm(x, axis=0)
        scaled = x/scale
        q, r = np.linalg.qr(scaled, mode="reduced")
        beta_scaled = np.linalg.solve(r, q.T@response)
        beta = beta_scaled/scale
        residual = response-scaled@beta_scaled
        inverse_r = np.linalg.solve(r, np.eye(x.shape[1]))
        bread = inverse_r@inverse_r.T
        names, sectors = list(design.columns), sorted(sample.sector_id.unique())
        assert "temp_c" in names
        transform = np.zeros((len(sectors), len(names)))
        transform[:, names.index("temp_c")] = 1
        for i, sector in enumerate(sectors[1:], start=1):
            transform[i, names.index(f"C(sector_id)[T.{sector}]:temp_c")] = 1
        influence = (scaled@bread@(transform/scale).T)*residual[:, None]
        elapsed = sample[period].astype("int64").to_numpy()-sample[period].astype("int64").min()
        score = np.zeros((elapsed.max()+1, len(sectors)))
        for date in np.unique(elapsed):
            score[date] = influence[elapsed==date].sum(axis=0)
        bandwidth = int(rows.hac_lags.iloc[0])
        covariance = score.T@score
        for lag in range(1, bandwidth+1):
            cross = score[lag:].T@score[:-lag]
            covariance += (1-lag/(bandwidth+1))*(cross+cross.T)
        n, periods, k = len(sample), sample[period].nunique(), x.shape[1]
        covariance *= periods/(periods-1)*(n-1)/(n-k)
        total = transform@beta
        standard_errors = np.sqrt(np.diag(covariance))
        selected = rows.set_index("sector_id").loc[sectors]
        for column, expected in [("coefficient", total), ("std_error", standard_errors),
                                 ("temperature_main", np.repeat(beta[names.index("temp_c")], len(sectors))),
                                 ("temperature_interaction", total-beta[names.index("temp_c")])]:
            error = numeric_errors(selected[column], expected, "Original explicit-main sector model "+column)
            maximum = max(maximum, error["max_scaled_relative_error"])
        assert selected.reference_sector_id.eq(sectors[0]).all()
        assert selected.loc[sectors[0], "temperature_interaction"] == 0
        numeric_errors(selected.coefficient, selected.temperature_main+selected.temperature_interaction, "Legacy main+interaction reported total")
        for i, sector in enumerate(sectors):
            null_checks.append(linear_effect_receipt(source=str(path.relative_to(ROOT)), model="H1", outcome=y.columns[0],
                bandwidth=bandwidth, sector=sector, reference=sectors[0], estimate=float(total[i]), variance=float(covariance[i,i]),
                degrees=periods-1, reported_p=float(selected.loc[sector, "p_value"])))
        fits += 1;slope_rows += len(rows)
    assert fits == 3 and slope_rows == 23
    return {"fits_checked": fits, "absolute_sector_slope_rows_checked": slope_rows, "max_scaled_relative_error": maximum,
            "method": "Independent full-design QR OLS with calendar HAC and explicit main/nonreference contrast vectors", "_sector_null_checks": null_checks}


def verify_native_fit_log() -> dict:
    path = OUT / "fit_log.csv"
    log = pd.read_csv(path)
    expected = expected_original_native_inventory()
    assert len(log) == expected["all_models"], ("Actual native production fit count", len(log), expected)
    installed = importlib.metadata.version("pyfixest")
    assert log.estimator_version.astype(str).eq(installed).all()
    assert log.native_class.eq("Feols").all() and log.native_module.str.startswith("pyfixest.estimation").all()
    assert log.solver.eq("np.linalg.lstsq").all()
    assert log.fit_id.tolist() == list(range(1, len(log)+1))
    assert not log.duplicated(["api_call_id", "outcome"]).any()
    assert log.native_dropped_coefficients.eq(0).all() and log.native_dropped_rows.eq(0).all()
    assert log.native_mapped_coefficient_max_abs_error.eq(0).all() and log.native_residual_max_abs_error.eq(0).all()
    assert log.df_resid.eq(log.n_obs-log.n_parameters).all() and log.df_resid.gt(0).all()
    assert log.residuals_sha256.eq(log.native_residuals_sha256).all()
    for name in ["design_sha256", "scaled_design_sha256", "native_design_sha256", "coefficient_transform_sha256", "outcome_sha256", "coefficient_names_sha256", "column_scales_sha256", "coefficients_sha256", "native_coefficients_sha256", "residuals_sha256"]:
        assert log[name].str.fullmatch(r"[0-9a-f]{64}").all(), name
    assert {"native_outcome_sha256", "native_response_max_abs_error"} <= set(log.columns)
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all() and log.native_response_max_abs_error.eq(0).all()
    basis = "orthonormal SVD preconditioned full LSDV"
    assert log.native_design_basis.eq(basis).all()
    assert log.native_design_max_abs_error.eq(0).all()
    assert log.native_design_orthogonality_max_abs_error.between(0, 1e-10, inclusive="both").all()
    for dimension in ["coefficient_transform_rows", "coefficient_transform_columns", "coefficient_transform_rank"]:
        assert log[dimension].eq(log.n_parameters).all(), dimension
    interactions, relative, legacy_absolute = 0, 0, []
    for row in log.itertuples():
        names, scales = json.loads(row.coefficient_names), np.asarray(json.loads(row.column_scales), dtype=float)
        assert len(names) == len(scales) == row.n_parameters and len(set(names)) == len(names)
        assert np.isfinite(scales).all() and np.all(scales>0)
        names_hash = hashlib.sha256(json.dumps(names, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        assert names_hash == row.coefficient_names_sha256
        scale_hash = hashlib.sha256(str(scales.shape).encode()+np.ascontiguousarray(scales, dtype=np.float64).tobytes()).hexdigest()
        assert scale_hash == row.column_scales_sha256
        safe = re.findall(r"\bx\d{4}\b", row.native_formula)
        assert set(safe) == {f"x{i:04d}" for i in range(row.n_parameters)}
        deviations = [name for name in names if name.startswith("temp_c:C(sector_id)[T.") or name.startswith("C(sector_id)[T.") and name.endswith(":temp_c")]
        if deviations:
            if "temp_c" in names:
                interactions += 1
            else:
                relative += 1
        absolute = [name for name in names if name.startswith("C(sector_id)[") and "[T." not in name and name.endswith(":temp_c")]
        if len(absolute)>1:
            legacy_absolute.append({"fit_id": int(row.fit_id), "label": row.label, "outcome": row.outcome, "sector_slopes": len(absolute)})
    # An API call may estimate several outcomes; each has one genuine Feols receipt.
    calls = log.groupby("api_call_id")
    assert set(log.api_call_id) == set(range(1, int(log.api_call_id.max())+1))
    for _, group in calls:
        assert all(group[column].nunique() == 1 for column in ["design_sha256", "n_obs", "n_parameters", "coefficient_names_sha256", "column_scales_sha256", "native_call_formula", "native_design_basis", "native_design_sha256", "coefficient_transform_sha256"])
    manifest = json.loads((OUT / "estimator_manifest.json").read_text())
    assert manifest["status"] == "model_inventory_written" and manifest["native_fits_recorded"] == len(log)
    assert manifest["native_api_calls"] == log.api_call_id.nunique()
    assert manifest["fit_log_sha256"] == sha256(path) and manifest["estimator_version"] == installed
    assert manifest["all_native_samples_and_coefficients_retained"] is True and manifest["native_models_retained_on_wrappers"] is True
    assert manifest["native_design_basis"] == basis
    assert interactions == expected["absolute_interaction_models"] and relative == expected["relative_interaction_models"] and not legacy_absolute, (interactions, relative, legacy_absolute, expected)
    backend = (ROOT / "scripts/pyfixest_backend.py").read_text()
    tree = ast.parse(backend)
    assert any(isinstance(node, ast.Call) and ast.unparse(node.func) == "pf.feols" for node in ast.walk(tree))
    assert "native_model=fitted" in backend and "return self.native_model" in backend
    assert "fitted.coef()" in backend and "fitted.resid()" in backend
    geometry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_geometry")
    assert [argument.arg for argument in geometry.args.args] == ["x", "label"]
    assert "transform = (vt.T / singular) / scales[:, None]" in backend
    assert "native_data = pd.DataFrame(orthonormal_x, columns=safe_x)" in backend
    assert "coefficients = transform @ theta" in backend
    assert "residual = np.asarray(fitted.resid()" in backend
    return {"actual_native_production_models": len(log), "native_api_calls": log.api_call_id.nunique(),
            "independent_current_source_inventory": expected,
            "installed_pyfixest_version": installed, "fit_log_sha256": sha256(path),
            "native_model_class": "pyfixest.estimation Feols", "all_rows_and_columns_preserved": True,
            "main_plus_nonreference_interaction_native_models": interactions, "relative_only_R1_native_models": relative,
            "legacy_equivalent_absolute_sector_slope_basis_models": legacy_absolute,
            "native_design_basis": basis, "native_preconditioner_uses_outcome": False,
            "original_coefficient_mapping": "beta = (D^-1 V S^-1) @ native_theta; X/D = U S V'",
            "native_design_and_coefficient_transform_hashes_valid": True,
            "max_native_design_orthogonality_error": float(log.native_design_orthogonality_max_abs_error.max()),
            "max_native_original_design_residual_identity_error": float(log.original_design_residual_identity_max_abs_error.max()),
            "native_coefficient_residual_and_response_mapping_errors": 0.}


def expected_original_native_inventory() -> dict:
    """Count current original-analysis fits from source spec/outcome/unit keys.

    Annual and paired analyses have isolated receipts and use estimates.csv,
    so they are deliberately outside this original-analysis inventory.
    """
    output = ROOT / "outputs"
    main = 0
    for path in [*output.rglob("regression_results.csv"), *output.rglob("distributed_lag_results.csv"), *output.glob("iip/*/aggregate_results.csv")]:
        if "visitor_arrivals" in path.parts:
            continue  # isolated visitor schema and estimator receipts have their own audit
        assert "covid_comparison" not in path.parts and "annual" not in path.parts
        frame = read_csv(path)
        models = set(frame.model)
        if "extended_controls" in path.parts:
            models.discard("C0")  # copied original baseline, no new native fit
        main += len(models)
    absolute, relative = 3, 0  # one literal H1 fit in each original VA/IIP context
    for path in [*output.glob("heterogeneity/*/*/sector_slopes.csv"), *output.glob("granular_interactions/*/*/sector_slopes.csv")]:
        groups = read_csv(path)[["model", "outcome"]].drop_duplicates()
        absolute += int(groups.model.ne("R1").sum())
        relative += int(groups.model.eq("R1").sum())
    separate = 0
    for path in [*output.glob("heterogeneity/*/*/separate_sector_coefficients.csv"), *output.glob("granular/*/*/separate_sector_coefficients.csv")]:
        frame = read_csv(path)
        keys = [name for name in ["model", "outcome", "sector_id"] if name in frame]
        separate += len(frame[keys].drop_duplicates())
    auxiliary = 2*3  # S3 outcome/temperature residualization in three contexts
    return {"main_common_and_national_models": main, "absolute_interaction_models": absolute,
        "relative_interaction_models": relative, "separate_series_models": separate,
        "partial_regression_auxiliary_models": auxiliary, "all_models": main+absolute+relative+separate+auxiliary}


def verify_pyfixest_migration() -> dict:
    OUT.mkdir(exist_ok=True)
    (OUT / "migration_verification.json").write_text(json.dumps({"status": "running", "claim": "No completion claim until this independent verification finishes"}, indent=2)+"\n")
    references = sorted(path for path in REFERENCE.rglob("*") if path.is_file())
    assert len(references) == 177
    snapshot_manifest = json.loads((REFERENCE.parent / "manifest.json").read_text())
    for reference in references:
        key = "outputs/"+str(reference.relative_to(REFERENCE))
        assert sha256(reference) == snapshot_manifest[key], (key, "Reference snapshot hash mismatch")
    files, comparisons = [], []
    for path in references:
        record, numeric = compare_csv(path) if path.suffix == ".csv" else compare_json(path)
        files.append(record);comparisons.extend(numeric)
    compare_frame = pd.DataFrame(comparisons)
    compare_frame.to_csv(OUT / "migration_comparison.csv", index=False)
    component_checks = verify_components()
    legacy_checks = verify_original_reference_slopes()
    sector_nulls = component_checks.pop("_sector_null_checks")+legacy_checks.pop("_sector_null_checks")
    omnibus_nulls = component_checks.pop("_omnibus_checks")
    sector_null_frame, omnibus_frame = pd.DataFrame(sector_nulls), pd.DataFrame(omnibus_nulls)
    sector_null_frame.to_csv(OUT / "sector_total_effect_null_checks.csv", index=False)
    omnibus_frame.to_csv(OUT / "omnibus_null_checks.csv", index=False)
    significance_summary = {"absolute_sector_total_tests": int(sector_null_frame.effect_type.eq("absolute_sector_total").sum()),
        "relative_R1_tests": int(sector_null_frame.effect_type.eq("relative_to_reference").sum()),
        "maximum_t_vs_one_restriction_F_p_error": float(sector_null_frame.t_F_p_absolute_error.max()),
        "maximum_reported_sector_p_error": float(sector_null_frame.reported_p_absolute_error.max()),
        "available_equal_slopes_omnibus_tests": int(omnibus_frame.status.eq("available").sum()),
        "singular_equal_slopes_omnibus_tests_without_p": int(omnibus_frame.status.str.startswith("unavailable").sum()),
        "maximum_omnibus_basis_scaled_relative_error": float(((omnibus_frame.wald_in_component_basis-omnibus_frame.wald_in_absolute_contrast_basis).abs()/np.maximum(1., omnibus_frame.wald_in_component_basis.abs())).max())}
    report = {"status": "passed", "tolerance": {"absolute": ATOL, "relative": RTOL, "rule": "abs(new-reference) <= absolute + relative*abs(reference)",
              "raw_level_units": "Raw-level coefficients and covariance entries are compared in original units with the same relative tolerance; actual errors are retained"},
              "reference_result_files_checked": len(files), "numeric_field_groups_checked": len(compare_frame),
              "numeric_values_checked": int(compare_frame.finite_values.sum()),
              "max_fraction_of_allowed_tolerance": float(compare_frame.max_fraction_of_tolerance.max()),
              "input_preservation": verify_input_preservation(), "native_fit_provenance": verify_native_fit_log(),
              "temperature_component_transforms": component_checks, "original_reference_sector_slopes": legacy_checks,
              "sector_significance_null": "One linear restriction beta_temperature+delta_sector=0; reference delta fixed0. Relative R1 tests only delta_sector=0.",
              "omnibus_heterogeneity_null": "All nonreference deltas=0; equal sector slopes with the common main effect unrestricted. This does not test all temperature slopes zero.",
              "sector_effect_null_checks": len(sector_nulls), "omnibus_null_checks": len(omnibus_nulls),
              "significance_null_verification": significance_summary,
              "static_production_audit": static_production_audit(), "files": files,
              "condition_number_exclusion": "Covariance condition numbers near singularity depend on floating-point SVD details; all covariance entries, restriction ranks and unavailable test statuses are still checked",
              "interpretation": "Estimator migration validation; unchanged inferential limitations and stationarity cautions remain"}
    (OUT / "migration_verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    note_path = ROOT / "notes/pyfixest_migration_validation.md"
    note = note_path.read_text().replace("The production FE regressions are being migrated to native", "The production FE regressions use native")
    note = re.sub(r"\n<!-- validation-run:begin -->.*?<!-- validation-run:end -->\n?", "\n", note, flags=re.S)
    provenance = report["native_fit_provenance"]
    completed = ("\n<!-- validation-run:begin -->\n## Completed verification\n\n"
                 f"The completed audit passed for all {len(files)} reference result files and {report['numeric_values_checked']:,} numerical values. "
                 f"All {provenance['actual_native_production_models']:,} native models were recorded by Pyfixest {provenance['installed_pyfixest_version']} "
                 f"across {provenance['native_api_calls']:,} actual native API calls. Native response, coefficient mapping and residual checks agree exactly, with no dropped rows or coefficients.\n\n"
                 f"The maximum numerical error used {report['max_fraction_of_allowed_tolerance']:.6g} of the stated tolerance. "
                 "All checked original source/processed/exact-sample rows and columns remain unchanged; any added NSA columns and source-manifest entries are separately enumerated. "
                 f"The input audit lists {report['input_preservation']['provenance_only_omitted_unverified_count']} absent third-party literature artifacts as provenance-only omitted/unverified, not as checked inputs. "
                 f"All {component_checks['model_outcome_bandwidth_groups_checked']} model/outcome/bandwidth component groups passed the coefficient-sum and full covariance transformation checks, "
                 f"including {component_checks['marked_reference_zero_normalizations_without_SE_or_p']} explicitly marked reference zero normalizations without standard errors or p-values. "
                 "The three original interaction fits also passed independent full-design QR OLS and calendar-HAC checks for all 23 reported absolute sector slopes.\n\n"
                 f"The significance audit checked {significance_summary['absolute_sector_total_tests']:,} absolute-sector total-effect tests "
                 f"and {significance_summary['relative_R1_tests']:,} identified relative R1 tests. "
                 f"The largest reported-p discrepancy was {significance_summary['maximum_reported_sector_p_error']:.3g}; "
                 f"two-sided t and one-restriction F probabilities differed by at most {significance_summary['maximum_t_vs_one_restriction_F_p_error']:.3g}. "
                 f"All {significance_summary['available_equal_slopes_omnibus_tests']} available equal-slopes omnibus tests agreed across the two coefficient bases; "
                 f"the other {significance_summary['singular_equal_slopes_omnibus_tests_without_p']} had singular restriction covariance and correctly reported no p-value.\n"
                 "<!-- validation-run:end -->\n")
    note_path.write_text(note.rstrip()+"\n"+completed)
    print(f"Verified native Pyfixest migration: {len(files)} original reference files, {report['numeric_values_checked']} original-SA numerical values and {provenance['actual_native_production_models']} current logged production fits.", flush=True)
    return report


if __name__ == "__main__":
    verify_pyfixest_migration()
