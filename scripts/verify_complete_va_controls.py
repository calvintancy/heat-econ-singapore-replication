"""Independent raw-data/full-QR audit of complete VA full-control models.

Reuses only independent audit math and raw parsers. Neither the new producer
nor the production estimator/covariance routines is imported.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix
from threadpoolctl import threadpool_limits

import verify_crisis_sensitivity as audit
from verify_granular import _raw_extended, _raw_levels, _raw_weather

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/complete_va_controls"
WEATHER = ["temp_c", "rain_100mm", "rh_pct", "wind_mps", "pm25_10ug"]
SOURCES = {"va12": "outputs/outcomes/va/full/common_sample.csv",
           "va12-matched": "outputs/granular/va-broad-matched/full/samples/G1.csv",
           "va20": "outputs/granular/va-detailed/full/samples/G1.csv"}
COUNTS = {"va12": 12, "va12-matched": 12, "va20": 20}


def formula(model: str, sectors: int) -> str:
    weather = "temp_c * C(sector_id)" if model == "H1" else "temp_c"
    nuisance = "C(sector_id)*C(season) + C(year)" if sectors > 1 else "C(season) + C(year)"
    return weather+" + rain_100mm + rh_pct + wind_mps + pm25_10ug + "+nuisance


def parents_from_raw() -> tuple[dict, dict]:
    sa, labels, source_meta = _raw_levels("va", "sa")
    nsa, _, _ = _raw_levels("va", "nsa")
    climate = _raw_weather()["Q"]
    extended, extended_meta = _raw_extended()
    eligible_dates = {date for date, values in extended["Q"].items() if values["eligible_90"] and date in climate}
    assert len(eligible_dates) == 29
    parents = {}
    for context, source in SOURCES.items():
        base = audit.read(ROOT / source)
        base.quarter = pd.PeriodIndex(base.quarter, freq="Q")
        assert set(base.quarter) == eligible_dates and base.sector_id.nunique() == COUNTS[context]-1
        calendar = base.drop_duplicates("time").sort_values("time")
        ids = sorted(set(base.sector_id)|{"1.3"})
        rows = []
        for sector in ids:
            for date in calendar.itertuples():
                quarter = date.quarter
                current, unadjusted = sa[sector][quarter], nsa[sector][quarter]
                raw_weather = {**climate[quarter], **extended["Q"][quarter]}
                rows.append({"sector_id": sector, "sector": labels[sector], "quarter": quarter,
                    "time": date.time, "year": quarter.year, "season": quarter.quarter,
                    **{term: getattr(date, term) for term in WEATHER},
                    "y_short": 100*math.log(current/sa[sector][quarter-1]),
                    "y_yoy": 100*math.log(current/sa[sector][quarter-4]),
                    "y_yoy_nsa": 100*math.log(unadjusted/nsa[sector][quarter-4]),
                    "y_log_level": 100*math.log(current), "y_raw_level": current,
                    "output_sa": current, "output_nsa": unadjusted})
                for term in WEATHER:
                    audit.check(getattr(date, term), raw_weather[term], "Raw NEA/full coverage weather "+term, atol=2e-11, rtol=1e-11)
        parent = pd.DataFrame(rows)
        assert np.isfinite(parent[audit.OUTCOMES+WEATHER]).all().all()
        audit.compare_sample(audit.read(OUT / "parents" / f"{context}.csv"), parent, "quarter")
        parents[context] = parent
    return parents, {"official_VA": source_meta, "extended_weather": extended_meta,
        "coverage_90_quarters": len(eligible_dates), "dwellings_sector": "1.3", "raw_growth_bases": "Full raw calendar before every match/exclusion"}


def verify_complete_va_controls() -> dict:
    (OUT / "verification.json").write_text('{"status":"running"}\n')
    old = {name: getattr(audit, name) for name in ["OUT", "WEATHER", "CONTEXTS", "design_formula", "ERRORS"]}
    audit.OUT, audit.WEATHER, audit.CONTEXTS, audit.design_formula, audit.ERRORS = OUT, WEATHER, COUNTS, formula, {}
    try:
        manifest = json.loads((OUT / "manifest.json").read_text())
        policy = json.loads((OUT / "policy.json").read_text())
        original_policy = json.loads((ROOT / policy["source_crisis_policy"]).read_text())
        assert policy["events"] == original_policy["events"] and policy["controls"] == WEATHER
        assert audit.sha256(ROOT / policy["source_crisis_policy"]) == policy["source_crisis_policy_sha256"]
        assert manifest["controls"] == WEATHER and manifest["native_fits"] == 750
        protected = {**manifest["protected_sources_sha256"], **manifest["protected_core_sources_sha256"]}
        assert all(audit.sha256(ROOT / name) == expected for name, expected in protected.items())
        parents, raw_summary = parents_from_raw()
        samples, sample_rows = audit.audit_samples(parents, policy)
        expected_periods = {"all_eligible": 29, "event_dates_excluded": 19, "strict_growth_window_clean": 17}
        assert all(row["n_periods"] == expected_periods[row["sample_variant"]] for row in sample_rows)
        specs = audit.read(OUT / "specifications.csv")
        assert len(specs) == 750 and specs.fit_group_id.nunique() == 150 and specs.status.eq("estimated").all()
        assert specs.control_set.eq("full").all() and specs.includes_dwellings.map(audit._truth).all()
        tables = {name: audit.read(OUT / f"{name}.csv") for name in ["estimates", "sector_estimates", "separate_estimates", "shared_control_estimates",
            "all_coefficients", "coefficient_covariances", "sector_covariances", "interaction_components", "component_covariances", "joint_tests", "pairwise_tests"]}
        groups = {name: {str(key): frame for key, frame in data.groupby("spec_id", sort=False)} for name, data in tables.items()}
        expected_native, fits, joint_rows, separate = [], [], [], {}
        checked_cells = 0
        with threadpool_limits(limits=1):
            for group_id, group_specs in specs.groupby("fit_group_id", sort=True):
                first = group_specs.iloc[0]
                context, variant, model = first.context, first.sample_variant, first.model
                full_sample = samples[context+"__"+variant]
                with np.load(OUT / "designs" / f"{group_id}.npz", allow_pickle=False) as archive:
                    sample = audit._sample_for_design(full_sample, archive, str(first.unit_id) if model == "separate" else None)
                    names = list(archive["coef_names"].astype(str))
                    x = audit.rebuild_design(sample, model, names)
                    y = sample[audit.OUTCOMES].to_numpy(float)
                    audit.check(archive["X"], x, "Every full-control design cell", atol=2e-11, rtol=1e-12)
                    audit.check(archive["Y"], y, "Every full-control outcome cell", atol=0, rtol=0)
                    fit = audit.independent_qr(x, y, names)
                    audit.check(archive["coefficients"], fit["beta"], "All native/nuisance coefficients")
                    audit.check(archive["residuals"], fit["residual"], "Every native full-control residual", atol=5e-7, rtol=1e-8)
                    for j, outcome in enumerate(audit.OUTCOMES):
                        expected_native.append({"label": "Crisis sensitivity "+group_id, "outcome": outcome,
                            "n_obs": len(sample), "n_parameters": len(names), "design_sha256": audit.array_sha256(archive["X"]),
                            "outcome_sha256": audit.array_sha256(archive["Y"][:, j]), "coefficients_sha256": audit.array_sha256(archive["coefficients"][:, j]),
                            "residuals_sha256": audit.array_sha256(archive["residuals"][:, j]),
                            "coefficient_names_sha256": hashlib.sha256(json.dumps(names, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
                sectors = sorted(sample.sector_id.unique());t = sample.time.nunique()
                selected_names = [name for name in names if name in WEATHER or name.startswith("temp_c:C(sector_id)")]
                positions = [names.index(name) for name in selected_names]
                contrast = np.eye(len(names))[positions]
                if model == "separate":
                    separate.setdefault((context, variant), {})[str(first.unit_id)] = {
                        "beta": fit["beta"][names.index("temp_c")], "error": fit["residual"],
                        "influence": audit.influences(fit, np.eye(len(names))[[names.index("temp_c")]])[:, 0],
                        "time": sample.time.to_numpy(), "k": fit["k"]}
                for j, outcome in enumerate(audit.OUTCOMES):
                    spec = group_specs[group_specs.outcome.eq(outcome)].iloc[0]
                    audit.metadata(spec, sample, fit, outcome)
                    assert json.loads(spec.controls) == WEATHER and all(term in spec.formula for term in WEATHER)
                    assert spec.source_sample_csv == SOURCES[context] and spec.source_sample_sha256 == audit.sha256(ROOT / SOURCES[context])
                    assert spec.sample_sha256 == audit.sha256(ROOT / spec.sample_csv)
                    temporal = dmatrix(formula("separate", 1), sample.drop_duplicates("time"), return_type="dataframe").to_numpy()
                    assert spec.time_design_rank == temporal.shape[1] and spec.time_residual_df == t-temporal.shape[1]
                    assert spec.residual_df == len(sample)-len(names)
                    if spec.calendar_gaps: assert "Not cleared" in spec.stationarity_status
                    full_coef = groups["all_coefficients"][spec.spec_id].set_index("term").loc[names]
                    audit.check(full_coef.coefficient, fit["beta"][:, j], "Every exported native FE/nuisance point")
                    for method, band in ([("Sector cluster", 0)] if model == "common" else [])+[("Driscoll-Kraay" if model != "separate" else "Calendar HAC", band) for band in [4, 8]]:
                        cov, degrees, factor = audit.score_covariance(fit, sample, contrast, j, method, band)
                        score = audit.influences(fit, contrast)*fit["residual"][:, [j]];hc0 = score.T@score
                        saved = groups["coefficient_covariances"][spec.spec_id]
                        saved = saved[saved.covariance.eq(method)&saved.hac_lags.eq(band)]
                        audit.check(audit.covariance_matrix(saved, selected_names, "term_i", "term_j"), cov, "Full five-weather/component covariance", atol=5e-6, rtol=3e-7)
                        audit.check(saved.pivot(index="term_i", columns="term_j", values="hc0_reference_value").loc[selected_names, selected_names], hc0,
                                    "Every selected HC0 numerical reference", atol=2e-6, rtol=3e-7)
                        audit.check(saved.finite_sample_correction, np.full(len(saved), factor), "Method-specific full-control finite correction", atol=1e-12, rtol=1e-12)
                        checked_cells += len(saved)
                        if model != "H1":
                            scalar = groups["separate_estimates" if model == "separate" else "estimates"][spec.spec_id]
                            scalar = scalar[scalar.covariance.eq(method)&scalar.hac_lags.eq(band)].set_index("term")
                            assert set(scalar.index) == set(WEATHER)
                            for term in WEATHER:
                                index = selected_names.index(term)
                                audit.inference(scalar.loc[term], fit["beta"][names.index(term), j], cov[index, index], degrees, hc0[index, index])
                                audit.metadata(scalar.loc[term], sample, fit, outcome)
                        else:
                            total = audit.total_temperature_contrast(names, sectors)
                            transform = total[:, positions]
                            beta, v, reference = total@fit["beta"][:, j], transform@cov@transform.T, transform@hc0@transform.T
                            rows = groups["sector_estimates"][spec.spec_id];rows = rows[rows.hac_lags.eq(band)].set_index("sector_id").loc[sectors]
                            for i, sector in enumerate(sectors):
                                audit.inference(rows.loc[sector], beta[i], v[i, i], degrees, reference[i, i])
                                audit.metadata(rows.loc[sector], sample, fit, outcome, sector_id=sector)
                                audit.check(rows.loc[sector].variance, rows.loc[sector].var_temp_main+rows.loc[sector].var_interaction_delta+2*rows.loc[sector].cov_temp_main_delta, "Temperature total variance cross term", atol=5e-6)
                            saved_total = groups["sector_covariances"][spec.spec_id];saved_total = saved_total[saved_total.hac_lags.eq(band)]
                            audit.check(audit.covariance_matrix(saved_total, sectors, "sector_i", "sector_j"), v, "H1 total joint covariance", atol=5e-6, rtol=3e-7)
                            audit.check(saved_total.pivot(index="sector_i", columns="sector_j", values="hc0_reference_value").loc[sectors, sectors], reference,
                                        "H1 total joint HC0 reference", atol=2e-6, rtol=3e-7)
                            comps = groups["interaction_components"][spec.spec_id];comps = comps[comps.hac_lags.eq(band)]
                            estimated = comps[comps.estimated.map(audit._truth)].set_index("term")
                            for term, row in estimated.iterrows():
                                index = selected_names.index(term)
                                audit.inference(row, fit["beta"][names.index(term), j], cov[index, index], degrees, hc0[index, index])
                            normalization = comps[~comps.estimated.map(audit._truth)]
                            assert len(normalization) == 1 and normalization.coefficient.eq(0).all() and normalization.normalization.map(audit._truth).all()
                            assert normalization[["std_error", "p_value", "ci_lower", "ci_upper"]].isna().all().all()
                            component_names = list(estimated.index)
                            component_positions = [selected_names.index(term) for term in component_names]
                            serialized_components = groups["component_covariances"][spec.spec_id]
                            serialized_components = serialized_components[serialized_components.hac_lags.eq(band)]
                            audit.check(audit.covariance_matrix(serialized_components, component_names, "term_i", "term_j"), cov[np.ix_(component_positions, component_positions)],
                                        "Every serialized component covariance", atol=5e-6, rtol=3e-7)
                            audit.check(serialized_components.pivot(index="term_i", columns="term_j", values="hc0_reference_value").loc[component_names, component_names],
                                        hc0[np.ix_(component_positions, component_positions)], "Every serialized component HC0 reference", atol=2e-6, rtol=3e-7)
                            control = groups["shared_control_estimates"][spec.spec_id];control = control[control.hac_lags.eq(band)].set_index("term")
                            assert set(control.index) == set(WEATHER[1:])
                            for term, row in control.iterrows():
                                index = selected_names.index(term)
                                audit.inference(row, fit["beta"][names.index(term), j], cov[index, index], degrees, hc0[index, index])
                                audit.metadata(row, sample, fit, outcome)
                            joint = groups["joint_tests"][spec.spec_id];joint = joint[joint.hac_lags.eq(band)].iloc[0]
                            pairs = groups["pairwise_tests"][spec.spec_id];pairs = pairs[pairs.hac_lags.eq(band)]
                            detail = audit.verify_joint_pairs(beta, v, sectors, joint, pairs, degrees, t-sample.season.nunique()-1, reference)
                            joint_rows.append({"spec_id": spec.spec_id, "hac_lags": band, **detail})
                fits.append({"fit_group_id": group_id, "context": context, "sample_variant": variant, "model": model,
                             "n_obs": len(sample), "n_periods": t, "n_parameters": len(names), "outcomes": 5})
            for (context, variant), units in separate.items():
                sectors = sorted(units);first = units[sectors[0]];t, k = len(first["time"]), first["k"]
                for j, outcome in enumerate(audit.OUTCOMES):
                    beta = np.asarray([units[sector]["beta"][j] for sector in sectors])
                    score = np.column_stack([units[sector]["influence"]*units[sector]["error"][:, j] for sector in sectors]);hc0 = score.T@score
                    spec_id = context+"__"+variant+"__separate_joint__"+outcome
                    for band in [4, 8]:
                        cov = audit.bartlett_scores(score, first["time"], band)*t/(t-k)
                        saved = groups["sector_covariances"][spec_id];saved = saved[saved.hac_lags.eq(band)]
                        audit.check(audit.covariance_matrix(saved, sectors, "sector_i", "sector_j"), cov, "Full-control joint separate covariance", atol=5e-6, rtol=3e-7)
                        audit.check(saved.pivot(index="sector_i", columns="sector_j", values="hc0_reference_value").loc[sectors, sectors], hc0,
                                    "Full-control joint separate HC0 reference", atol=2e-6, rtol=3e-7)
                        audit.check(saved.finite_sample_correction, np.full(len(saved), t/(t-k)), "Separate joint finite correction", atol=1e-12, rtol=1e-12)
                        joint = groups["joint_tests"][spec_id];joint = joint[joint.hac_lags.eq(band)].iloc[0]
                        audit.check(joint.finite_sample_correction, t/(t-k), "Separate omnibus finite correction", atol=1e-12, rtol=1e-12)
                        pairs = groups["pairwise_tests"][spec_id];pairs = pairs[pairs.hac_lags.eq(band)]
                        detail = audit.verify_joint_pairs(beta, cov, sectors, joint, pairs, t-1, t-k, hc0)
                        joint_rows.append({"spec_id": spec_id, "hac_lags": band, **detail})
        native = audit.audit_native_receipt(expected_native)
        assert native["native_fits"] == 750 and native["native_api_calls"] == 150
        assert all(audit.sha256(ROOT / name) == expected for name, expected in protected.items())
        report = {"status": "passed", "scope": "Independent raw VA/dwellings, raw NEA wind/PM coverage and climate, calendar masks, named full-X/Patsy span, QR, CR1/DK/HAC, beta+delta, covariance cross terms, joint ranks/Holm and native receipts; no producer imports or causal/stationarity claim",
            "contexts": 3, "samples": 9, "native_fits": 750, "independent_designs": 150,
            "selected_covariance_cells": checked_cells, "joint_tests": len(joint_rows),
            "unavailable_joint_tests": sum(not row["available"] for row in joint_rows),
            "native_receipt": native, "raw_source_audit": raw_summary, "protected_files_unchanged": len(protected),
            "numerical_checks": audit.ERRORS, "policy_sha256": audit.sha256(OUT / "policy.json")}
        pd.DataFrame(fits).to_csv(OUT / "verification_fits.csv", index=False)
        pd.DataFrame(sample_rows).to_csv(OUT / "verification_samples.csv", index=False)
        pd.DataFrame(joint_rows).to_csv(OUT / "verification_joint_tests.csv", index=False)
        (OUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print("Complete full-control VA independent audit passed: 750 native fits, 150 full designs, all 9 samples.", flush=True)
        return report
    finally:
        for name, value in old.items(): setattr(audit, name, value)


if __name__ == "__main__":
    verify_complete_va_controls()
