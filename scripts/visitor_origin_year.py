"""Exactly matched origin-by-year FE visitor-arrival robustness.

Native PyFixest estimates the 24 predeclared H1 equations. Frozen parent samples,
outcomes, stochastic bases and current controls are reused without alteration.
Calendar HAC conventions are working approximations; joint tests are nominal.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import visitor_rigour_monthly as monthly
import crisis_sensitivity as math_shared
import literature_sensitivity as calendar_math
import pyfixest_backend as native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/visitor_rigour/origin_year"
PROTOCOL = ROOT / "notes/visitor_origin_year_protocol.md"
relative, sha, save_json = monthly.relative, monthly.sha, monthly.save_json
inference, wald, restrictions = monthly.inference, monthly.wald, monthly.restrictions
temporal_ranks, add_holm = monthly.temporal_ranks, monthly.add_holm


def design(sample, model_spec, model, horizon, control_set):
    assert model == "H1" and model_spec == "own_current"
    units, seasons, years = sorted(sample.unit_id.unique()), sorted(sample.season.unique()), sorted(sample.year.unique())
    indicator = np.column_stack([sample.unit_id.eq(unit) for unit in units]).astype(float)
    seasonal = np.column_stack([sample.season.eq(season) for season in seasons]).astype(float)
    fixed = (indicator[:, :, None] * seasonal[:, None, :]).reshape(len(sample), -1)
    names = [f"origin_season[{unit},{season}]" for unit in units for season in seasons]
    if len(years) > 1:
        annual = np.column_stack([sample.year.eq(year) for year in years[1:]]).astype(float)
        fixed = np.column_stack([fixed, (indicator[:, :, None] * annual[:, None, :]).reshape(len(sample), -1)])
        names += [f"origin_year[{unit},{year}]" for unit in units for year in years[1:]]
    temperatures, controls, ar, lags = monthly.feature_inventory(model_spec, horizon, control_set)
    first = len(names)
    blocks = [fixed, (sample[controls] - sample[controls].mean()).to_numpy()]
    names += controls
    selected = list(range(first, len(names)))
    components = []
    for term in temperatures:
        values = (sample[term] - sample[term].mean()).to_numpy()
        start = len(names)
        blocks.append(np.column_stack([values, indicator[:, 1:] * values[:, None]]))
        names += [term] + [f"{term}:C(unit_id)[T.{unit}]" for unit in units[1:]]
        components.append(list(range(start, len(names))))
        selected.extend(range(start, len(names)))
    return np.column_stack(blocks), names, np.asarray(selected), np.asarray(components), units, temperatures, controls, ar, lags


def fit_spec(sample, base, model_spec, model, output, records, preflight=False):
    matrix, names, selected, components, units, features, controls, ar, lags = design(sample, model_spec, model, base["horizon"], base["control_set"])
    geometry = math_shared.geometry(matrix)
    n, k, t, g = len(sample), matrix.shape[1], sample.time.nunique(), len(units)
    temporal = temporal_ranks(sample, matrix)
    spec_id = base["source_spec_id"] + "__origin_year"
    label = "Visitor matched origin-year " + spec_id
    meta = {**base, **temporal, "spec_id": spec_id, "native_label": label, "model": model, "model_spec": model_spec,
        "temp_features": json.dumps(features), "control_features": json.dumps([term for term in controls if term not in ar]), "ar_features": json.dumps(ar),
        "controls": json.dumps(controls), "n_obs": n, "n_periods": t, "n_sectors": g, "n_origins": g, "n_parameters": k, "residual_df": n - k,
        "design_rank": geometry["design_rank"], "design_rank_threshold": geometry["design_rank_threshold"],
        "design_condition_number": geometry["design_condition_number"], "scaled_design_condition_number": geometry["scaled_design_condition_number"],
        "time_residual_df": t - temporal["time_design_rank"], "status": geometry["status"], "fit_id": None,
        "start": sample.month.min(), "end": sample.month.max(), "calendar_gaps": int(sample.time.max() - sample.time.min() + 1 - t),
        "mean_y": float(sample[base["outcome"]].mean()), "sd_y": float(sample[base["outcome"]].std()),
        "reference_sector_id": units[0] if model == "H1" else "", "reference_unit_id": units[0] if model == "H1" else "",
        "formula": base["outcome"] + " ~ " + " + ".join(features) + (" main + origin interactions" if model == "H1" else "") + " + " + " + ".join(controls) + " | origin×calendar-month FE + origin×year FE",
        "design_normalization": "All origin×observed-month cells; origin×year indicators omitting first observed year in each origin; center continuous variables on exact pooled sample; main temperature plus nonreference deviations",
        "AR_slope_scope": "Shared across origins" if ar else "None", "estimator": "PyFixest feols", "estimator_version": native.ESTIMATOR_VERSION,
        "inference_reference_scope": "Working finite-sample convention; t(T−1)/F(q,T−1) is approximate, not exact or empirically calibrated",
        "design_npz": relative(output / "designs" / f"{spec_id}.npz"), "native_point_guard": "N>K and full scaled-column design rank; temporal covariance guard is separate",
        "time_correction_available": t > temporal["time_design_rank"],
        "primary_calendar_inference_available": geometry["status"] == "estimated" and t > temporal["time_design_rank"],
        "full_covariance_inventory": "covariance_0/1 = primary calendar HAC12/24; covariance_2 = pooled-origin CR1 when requested. raw_calendar_covariance_0/1 retain unadjusted full covariance for diagnostics; missing primary factor yields NaN primary matrices and no p/CI"}
    records["specifications"].append(meta)
    if preflight: return meta
    beta = residual = influence = hc0 = None
    if meta["status"] == "estimated":
        # A reversible, design-only QR bridge avoids a LAPACK full-vector SVD
        # failure in the enlarged origin-year basis. Native PyFixest still supplies
        # every point coefficient and residual; no outcome enters the bridge.
        norms = np.linalg.norm(matrix, axis=0)
        native_design, triangular = np.linalg.qr(matrix / norms, mode="reduced")
        coefficient_transform = np.linalg.solve(triangular, np.eye(k)) / norms[:, None]
        fitted = native.fit_many_matrix(sample[[base["outcome"]]].to_numpy(), native_design,
            [f"orthogonal_{j:04d}" for j in range(k)], label, outcome_names=[base["outcome"]],
            source_formula=meta["formula"] + "; invertible design-only QR bridge; beta_original=B@beta_native")[0]
        beta, residual = coefficient_transform @ fitted.params.to_numpy(), fitted.resid
        meta.update(native_input_basis="Design-only scaled-QR full LSDV; native backend orthonormal preconditioning",
            native_input_design_sha256=native._array_hash(native_design), original_design_sha256=native._array_hash(matrix),
            coefficient_transform_sha256=native._array_hash(coefficient_transform),
            coefficient_mapping_max_abs_error=float(np.max(np.abs(matrix @ coefficient_transform-native_design))),
            native_residual_identity_max_abs_error=float(np.max(np.abs(sample[base["outcome"]].to_numpy()-matrix@beta-residual))))
        influence = geometry["mapping"].T * residual[:, None]
        hc0 = influence.T @ influence
        meta.update(fit_id=native._FIT_LOG[-1]["fit_id"], r_squared=float(fitted.rsquared))
        records["coefficients"].extend({"spec_id": spec_id, "fit_id": meta["fit_id"], "term": name, "coefficient": float(value)} for name, value in zip(names, beta, strict=True))
    archive = {"X": matrix, "Y": sample[base["outcome"]].to_numpy(), "beta": np.full(k, np.nan) if beta is None else beta,
        "resid": np.full(n, np.nan) if residual is None else residual, "names": np.asarray(names, str), "time": sample.time.to_numpy(),
        "periods": sample.month.to_numpy(dtype=str), "unit_ids": sample.unit_id.to_numpy(dtype=str), "selected_positions": selected,
        "component_positions": components, "hc0_full": np.full((k, k), np.nan) if hc0 is None else hc0,
        "temperature_features": np.asarray(features, str), "control_features": np.asarray(controls, str), "lags": np.asarray(lags),
        "point_fit_available": np.asarray(beta is not None), "primary_inference_available": np.asarray(meta["primary_calendar_inference_available"] and beta is not None)}
    if beta is not None:
        archive.update(native_input_design=native_design, coefficient_transform=coefficient_transform,
            native_input_coefficients=fitted.params.to_numpy())
    methods = []
    for index, bandwidth in enumerate([12, 24]):
        raw = np.full((k, k), np.nan) if influence is None else calendar_math.calendar_covariance(influence, sample.time.to_numpy(), bandwidth, 1.)
        archive[f"raw_calendar_covariance_{index}"] = raw
        factor = t / meta["time_residual_df"] if meta["time_residual_df"] > 0 else np.nan
        methods.append(("Calendar HAC", bandwidth, raw * factor, factor, t - 1, "T/(T−K_time), K_time=max per-origin local full-design rank", bandwidth == 12,
            "working calendar correction undefined because T≤K_time" if not np.isfinite(factor) else ""))
    if model == "common" and g > 1:
        factor = g / (g - 1) * (n - 1) / (n - k) if n > k else np.nan
        scores = np.zeros((g, k))
        if influence is not None: np.add.at(scores, pd.Categorical(sample.unit_id, categories=units).codes, influence)
        covariance = scores.T @ scores * factor if influence is not None else np.full((k, k), np.nan)
        methods.append(("Origin cluster", 0, covariance, factor, g - 1, "G/(G−1)×(N−1)/(N−K)", False, ""))
    archive.update(covariance_labels=np.asarray([row[0] for row in methods], str), bandwidths=np.asarray([row[1] for row in methods]),
        corrections=np.asarray([row[3] for row in methods]), inference_df=np.asarray([row[4] for row in methods]))
    names_map = sample.drop_duplicates("unit_id").set_index("unit_id").unit.to_dict()
    selected_hc0 = np.full((len(selected), len(selected)), np.nan) if hc0 is None else hc0[np.ix_(selected, selected)]
    beta_selected = None if beta is None else beta[selected]
    positions = np.asarray([[list(selected).index(position) for position in row] for row in components])
    compact = {key: meta[key] for key in ["spec_id", "panel", "context", "outcome", "outcome_label", "control_set", "history", "sample_variant", "model_spec", "model", "status", "fit_id", "n_obs", "n_periods", "n_sectors", "n_parameters", "time_residual_df", "start", "end", "mean_y", "sd_y", "reference_unit_id", "estimator", "estimator_version"]}
    for index, (method, bandwidth, full_cov, factor, degrees, factor_formula, main, reason) in enumerate(methods):
        archive[f"covariance_{index}"] = full_cov
        covariance = full_cov[np.ix_(selected, selected)]
        band = {**compact, "covariance": method, "hac_lags": bandwidth, "inference_df": degrees, "finite_sample_correction": factor,
            "finite_sample_correction_formula": factor_formula, "main_inference": main, "n_clusters": g if method == "Origin cluster" else 0,
            "calibration_status": "Nominal working convention; no exact or demonstrated general small-sample calibration"}
        for position, term in enumerate(controls):
            records["control_estimates"].append(inference(beta_selected, covariance, selected_hc0, np.eye(len(selected))[position], {**band, "term": term, "role": "AR" if term in ar else "control"}, reason))
        profiles, basis_profiles = [], []
        profile_units = units if model == "H1" or g == 1 else ["pooled"]
        for number, unit in enumerate(profile_units):
            basis = np.zeros((len(features), len(selected)))
            for feature_number, indexes in enumerate(positions):
                basis[feature_number, indexes[0]] = 1.
                if model == "H1" and number: basis[feature_number, indexes[number]] = 1.
            basis_profiles.append(basis)
            contrast = basis
            if model_spec == "two_channel":
                expansion = np.zeros((base["horizon"] + 1, 2));expansion[0, 0], expansion[-1, 0] = 1., -1.;expansion[:-1, 1] = 1.
                contrast = expansion @ basis
            profiles.append(contrast)
            local = sample if unit == "pooled" else sample.loc[sample.unit_id.eq(unit)]
            unit_meta = {**band, "unit_id": unit, "sector_id": unit, "unit": names_map.get(unit, "Common equal-origin slope"), "sector": names_map.get(unit, "Common equal-origin slope"),
                "mean_y": float(local[base["outcome"]].mean()), "sd_y": float(local[base["outcome"]].std())}
            for lag_number, lag in enumerate(lags):
                detail = {**unit_meta, "effect": "lag", "lag": lag, "term": f"temp_l{lag}"}
                if model == "H1" and model_spec != "two_channel":
                    loc = positions[lag_number];main_pos, delta_pos = loc[0], loc[number]
                    detail.update(temp_main=None if beta_selected is None else float(beta_selected[main_pos]), interaction_delta=None if beta_selected is None else 0. if number == 0 else float(beta_selected[delta_pos]),
                        delta_is_normalization=number == 0, var_temp_main=None if reason or beta is None else float(covariance[main_pos, main_pos]),
                        var_interaction_delta=None if reason or beta is None else 0. if number == 0 else float(covariance[delta_pos, delta_pos]),
                        cov_temp_main_delta=None if reason or beta is None else 0. if number == 0 else float(covariance[main_pos, delta_pos]))
                records["temperature_effects"].append(inference(beta_selected, covariance, selected_hc0, contrast[lag_number], detail, reason))
            records["temperature_effects"].append(inference(beta_selected, covariance, selected_hc0, contrast.sum(axis=0),
                {**unit_meta, "effect": "cumulative_direct", "lag": -1, "term": "sum_temperature_lags", "interpretation": "Direct conditional lag sum; not an unconditional long-run multiplier with lagged growth"}, reason))
            if model_spec == "two_channel":
                for feature, vector in zip(features, basis, strict=True):
                    records["channel_estimates"].append(inference(beta_selected, covariance, selected_hc0, vector,
                        {**unit_meta, "channel": feature, "term": feature, "effect": "channel", "effect_units": "Per +1°C endpoint difference" if "difference" in feature else "Per +1°C-month exposure sum; coefficient on a mean would be H times larger"}, reason))
            profile_beta = None if beta_selected is None else contrast @ beta_selected
            profile_cov, profile_hc0 = contrast @ covariance @ contrast.T, contrast @ selected_hc0 @ contrast.T
            if model_spec == "two_channel":
                requested = [("All temperature channels zero", np.eye(2)), ("Cumulative direct temperature response zero; same null as growth-sum channel zero", np.array([[0., base["horizon"]]])),
                    ("Static coefficient shape; same null as growth-sum channel zero", np.array([[0., 1.]])), ("Equal-growth accumulation channel shape", np.array([[1., 0.]]))]
                for name, restriction in requested:
                    records["tests"].append(wald(None if beta_selected is None else basis @ beta_selected, basis @ covariance @ basis.T, basis @ selected_hc0 @ basis.T, restriction, unit_meta, name, reason))
            else:
                for name, restriction in restrictions(len(lags), base["outcome"], model_spec == "dynamic"):
                    records["tests"].append(wald(profile_beta, profile_cov, profile_hc0, restriction, unit_meta, name, reason))
        if model == "H1":
            for feature_number, feature in enumerate(features):
                for unit_number, unit in enumerate(units):
                    vector = np.eye(len(selected))[positions[feature_number, unit_number]]
                    records["interaction_components"].append(inference(beta_selected, covariance, selected_hc0, vector,
                        {**band, "unit_id": unit, "sector_id": unit, "feature": feature, "component": "base" if unit_number == 0 else "interaction",
                            "term": names[components[feature_number, unit_number]], "estimated": beta is not None, "normalization": False}, reason))
                records["interaction_components"].append({**band, "unit_id": units[0], "sector_id": units[0], "feature": feature, "component": "interaction",
                    "term": f"{feature}:C(unit_id)[T.{units[0]}]", "estimated": False, "normalization": True, "coefficient": 0., "std_error": None,
                    "ci_lower": None, "ci_upper": None, "p_value": None, "inference_status": "Reference interaction normalized to zero; not estimated"})
            cumulative = np.vstack([profile.sum(axis=0) for profile in profiles])
            difference = np.eye(g)[1:] - np.eye(g)[[0]]
            records["tests"].append(wald(None if beta_selected is None else cumulative @ beta_selected, cumulative @ covariance @ cumulative.T,
                cumulative @ selected_hc0 @ cumulative.T, difference, {**band, "unit_id": "all", "sector_id": "all"}, "All origin cumulative direct effects equal", reason))
            flattened = np.vstack(basis_profiles if model_spec == "two_channel" else profiles)
            records["tests"].append(wald(None if beta_selected is None else flattened @ beta_selected, flattened @ covariance @ flattened.T,
                flattened @ selected_hc0 @ flattened.T, np.eye(len(flattened)), {**band, "unit_id": "all", "sector_id": "all"}, "All origin temperature response coefficients jointly zero", reason))
    np.savez_compressed(output / "designs" / f"{spec_id}.npz", **archive)
    return meta


def protected_inventory():
    paths = [ROOT / "scripts/visitor_origin_year.py", PROTOCOL,
             ROOT / "scripts/visitor_rigour_monthly.py", ROOT / "scripts/visitor_rigour_companions.py",
             ROOT / "scripts/pyfixest_backend.py", ROOT / "scripts/crisis_sensitivity.py",
             ROOT / "scripts/literature_sensitivity.py", ROOT / "scripts/literature_full_controls.py",
             ROOT / "notes/visitor_rigour_protocol.md", ROOT / "notes/crisis_sensitivity_policy.json"]
    # Protect all parent analytical numeric/sample/design/native files. Independent
    # verification receipts and unfinished presentation files are not model inputs.
    for family in ["monthly", "annual", "ppml"]:
        base = ROOT / "outputs/visitor_rigour" / family
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in [".csv", ".json", ".npz"] and "verification" not in path.name and "report" not in path.name and "native_tables" not in path.parts:
                paths.append(path)
    for path in [ROOT / "outputs/pyfixest/fit_log.csv", ROOT / "outputs/pyfixest/estimator_manifest.json"]:
        if path.exists(): paths.append(path)
    parent = json.loads((ROOT / "outputs/visitor_rigour/monthly/manifest.json").read_text())
    for mapping in [parent["raw_inputs_sha256"], parent["source_sha256"]]:
        for name, expected in mapping.items():
            assert sha(ROOT / name) == expected, f"Frozen parent binding changed: {name}"
            paths.append(ROOT / name)
    return {relative(path): sha(path) for path in sorted(set(paths))}


def run_visitor_origin_year(*, preflight=False):
    output = OUT / "preflight" if preflight else OUT
    (output / "designs").mkdir(parents=True, exist_ok=True)
    protected = protected_inventory()
    source_path = ROOT / "outputs/visitor_rigour/monthly/specifications.csv"
    parent = pd.read_csv(source_path, low_memory=False)
    selected = parent.loc[parent.panel.isin(["region", "country"]) & parent.model.eq("H1") & parent.model_spec.eq("own_current")].copy()
    assert len(selected) == 24 and selected.spec_id.is_unique
    records = {name: [] for name in ["specifications", "coefficients", "temperature_effects", "channel_estimates", "interaction_components", "control_estimates", "tests"]}
    samples = []
    previous = native.LOG_DIR, list(native._FIT_LOG), native._API_COUNTER, native._RUN_STARTED
    receipt = {}
    if not preflight: native.LOG_DIR = output / "pyfixest"; native.reset_fit_log()
    save_json(output / "manifest.json", {"status": "running_preflight" if preflight else "running", "claim": "No completed-result or verification claim"})
    try:
        with threadpool_limits(limits=1):
            for source in selected.to_dict("records"):
                sample_path = ROOT / source["sample_csv"]
                assert sha(sample_path) == source["sample_sha256"]
                sample = pd.read_csv(sample_path, dtype={"unit_id": str, "sector_id": str}, float_precision="round_trip")
                source_design = ROOT / source["design_npz"]
                with np.load(source_design) as archive:
                    assert np.array_equal(sample[source["outcome"]].to_numpy(), archive["Y"])
                    assert np.array_equal(sample.time.to_numpy(), archive["time"])
                    assert np.array_equal(sample.unit_id.to_numpy(str), archive["unit_ids"])
                base = {**source, "source_spec_id": source["spec_id"], "source_design_npz": source["design_npz"],
                        "source_sample_csv": source["sample_csv"], "source_sample_sha256": sha(sample_path),
                        "source_design_sha256": sha(source_design), "source_specifications_csv": relative(source_path),
                        "nuisance_spec": "origin_year", "protocol_source_sha256": protected[relative(PROTOCOL)],
                        "sample_reuse": "Exact frozen own-current H1 parent observations; only deterministic annual nuisance restrictions change"}
                # Native archive Y uses the source CSV's archived float decoding.
                meta = fit_spec(sample, base, "own_current", "H1", output, records, preflight=preflight)
                assert meta["n_obs"] == source["n_obs"] and meta["n_periods"] == source["n_periods"]
                assert meta["time_design_rank"] == source["time_design_rank"], "Local temporal span must equal matched current parent"
                samples.append({key: meta[key] for key in ["spec_id", "source_spec_id", "sample_csv", "sample_sha256", "outcome", "panel", "control_set", "sample_variant", "n_obs", "n_periods", "n_sectors", "start", "end", "mean_y", "sd_y"]})
                print(f"Origin-year {'preflight' if preflight else 'fit'} {meta['spec_id']}: N{meta['n_obs']} T{meta['n_periods']} K{meta['n_parameters']}, {meta['status']}, timeDF{meta['time_residual_df']}", flush=True)
            if not preflight: receipt = native.write_fit_log()
    finally:
        if not preflight:
            native.LOG_DIR, old, native._API_COUNTER, native._RUN_STARTED = previous
            native._FIT_LOG[:] = old
    assert all(sha(ROOT / name) == value for name, value in protected.items()), "Protected source/input/analytical output changed"
    add_holm(records["temperature_effects"])
    for name, rows in records.items(): pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(samples).to_csv(output / "sample_inventory.csv", index=False)
    specs = pd.DataFrame(records["specifications"])
    manifest = {"status": "preflight_only" if preflight else "computed_not_yet_independently_verified",
        "requested_specs": len(specs), "native_fits": receipt.get("native_fits_recorded", 0),
        "point_unavailable_requests": int(specs.status.ne("estimated").sum()),
        "primary_time_correction_unavailable_requests": int((~specs.time_correction_available).sum()),
        "samples": len(samples), "row_counts": {key: len(rows) for key, rows in records.items()},
        "protected_sha256": protected, "source_sha256": sha(Path(__file__)), "protocol_sha256": sha(PROTOCOL),
        "native_ledger": relative(output / "pyfixest"),
        "nuisance_change": "Common year FE replaced by origin×year FE; all other raw basis values, exact rows, controls and centering retained",
        "inference": "Actual-calendar Bartlett HAC12/24, T/(T−max local full-X rank), approximate t(T−1); no origin-cluster H1 inference",
        "joint_tests": "Nominal asymptotic diagnostics; no demonstrated general small-sample calibration or causal certification",
        "full_covariance_storage": "designs/{spec_id}.npz: nuisance-inclusive X/Y/native beta/resid, HC0 and full raw/corrected calendar matrices",
        "zero_handling": "Inherited exact frozen parent positivity/history eligibility; no additional count, outlier or significance filtering"}
    save_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["status", "requested_specs", "native_fits", "point_unavailable_requests", "primary_time_correction_unavailable_requests"]}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    run_visitor_origin_year(preflight=parser.parse_args().preflight)
