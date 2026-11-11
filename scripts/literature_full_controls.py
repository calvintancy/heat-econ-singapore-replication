"""Isolated subannual distributed-weather sensitivity with wind and PM2.5.

All models retain the original current-date 90% wind/PM coverage rule. Weather
and outcome lags use their complete cached calendars before matching or masks.
This is a Singapore sensitivity analysis, not a replication of a published paper.
"""
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from threadpoolctl import threadpool_limits

import crisis_sensitivity as shared
import literature_sensitivity as history
import pyfixest_backend as native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/literature_sensitivity/full_controls"
OUTCOMES = list(history.OUTCOMES)
VARIANTS = list(shared.VARIANTS)
SPECIFICATIONS = ["benchmark", "distributed", "dynamic"]
HISTORY = "full_controls_max_AR_history"
ADDED = ["wind_mps", "pm25_10ug"]
CURRENT_WEATHER = ["temp_c", "rain_100mm", "rh_pct", *ADDED]
ZERO, RANK = 1e-12, 1e-10


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


@lru_cache(maxsize=None)
def controls(frequency):
    path = ROOT / f"data/processed/extended_controls/{frequency}_controls.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    period, freq = ("month", "M") if frequency == "monthly" else ("quarter", "Q")
    frame.index = pd.PeriodIndex(frame[period], freq=freq)
    assert frame.index.is_unique
    expected = frame.wind_coverage.ge(.9) & frame.pm_coverage.ge(.9)
    if frequency == "monthly":
        assert np.array_equal(expected, frame.eligible_90)
    else:
        monthly, _ = controls("monthly")
        for date, row in frame.iterrows():
            months = monthly.loc[monthly.index.asfreq("Q") == date]
            passes = len(months) == 3 and months.eligible_90.all()
            assert bool(row.eligible_90) == passes
            if len(months) == 3 and months[ADDED].notna().all().all():
                for term in ADDED:
                    assert np.isclose(row[term], np.average(months[term], weights=months.days), atol=1e-12, rtol=1e-12)
    assert frame.loc[frame.eligible_90, ADDED].notna().all().all()
    return frame, path


@lru_cache(maxsize=None)
def raw_full_control_history(context, outcome):
    maximum, base = history.maximum_raw_history(context, outcome)
    frame, metadata = maximum.copy(), dict(base)
    added, path = controls(context.frequency)
    periods = pd.PeriodIndex(frame[context.period], freq=context.freq)
    for field in ADDED + ["wind_coverage", "pm_coverage", "eligible_90"]:
        frame[field] = added[field].reindex(periods).to_numpy()
    finite = np.isfinite(frame[ADDED]).all(axis=1)
    covered = frame.eligible_90.fillna(False).astype(bool)
    valid = finite & covered
    by_date = pd.DataFrame({"time": frame.time, "finite": finite, "covered": covered, "valid": valid}).groupby("time").all()
    retained = frame.loc[frame.time.isin(by_date.index[by_date.valid])].copy().reset_index(drop=True)
    metadata.update(history=HISTORY, control_set="full90", control_history_csv=relative(path), control_history_sha256=shared.sha(path),
        full_control_rule="Current period original eligible_90; monthly wind and PM each >=90%; quarterly all three constituent months pass; no PM/wind lag requirement",
        current_weather_variables=json.dumps(CURRENT_WEATHER), added_control_units=json.dumps({"wind_mps": "m/s", "pm25_10ug": "10 micrograms/m3"}),
        raw_history_n_periods=maximum.time.nunique(), raw_history_n_obs=len(maximum),
        control_raw_missing_loss_periods=int((~by_date.finite).sum()),
        control_coverage_loss_periods=int((by_date.finite & ~by_date.covered).sum()),
        full_control_eligible_n_periods=retained.time.nunique(), full_control_eligible_n_obs=len(retained),
        control_total_loss_periods=maximum.time.nunique()-retained.time.nunique(),
        weather_history_rule="Full temperature/rain lags0..h and raw official outcome-growth lags1..h before current-date wind/PM eligibility; historical wind/PM not required")
    assert metadata["control_total_loss_periods"] == metadata["control_raw_missing_loss_periods"] + metadata["control_coverage_loss_periods"]
    assert len(retained) == retained.time.nunique()*frame.sector_id.nunique()
    return retained, metadata


def prepare(context, outcome, policy):
    frame, base = raw_full_control_history(context, outcome)
    frame, metadata = frame.copy(), dict(base)
    target_count = None
    if context.matched_to:
        target = next(item for item in shared.contexts() if item.key == context.matched_to)
        target_frame, _ = raw_full_control_history(target, outcome)
        target_count = target_frame.time.nunique()
        dates = set(pd.PeriodIndex(target_frame[target.period], freq=target.freq))
        frame = frame.loc[pd.PeriodIndex(frame[context.period], freq=context.freq).isin(dates)].copy().reset_index(drop=True)
    metadata.update(matched_template_n_periods=target_count,
        target_alignment_loss_periods=base["full_control_eligible_n_periods"]-frame.time.nunique(),
        maximum_history_n_periods=frame.time.nunique(), maximum_history_n_obs=len(frame),
        history_loss_periods=base["parent_n_periods"]-frame.time.nunique(),
        matching_rule="Named target full raw maximum-history and current90% eligibility intersected before crisis masks" if context.matched_to else "Original raw-supported full-control history", policy_id=policy["policy_id"])
    intervals = [(event["event_id"], pd.Period(event["start"], freq="M").asfreq(context.freq),
                  pd.Period(event["end"], freq="M").asfreq(context.freq)) for event in policy["events"]]
    masks = []
    for row in frame.drop_duplicates("time").itertuples():
        date = pd.Period(getattr(row, context.period), freq=context.freq)
        support = metadata["maximum_output_support_horizon"]
        current = [key for key, start, end in intervals if start <= date <= end]
        overlapping = [key for key, start, end in intervals if date-support <= end and date >= start]
        masks.append({"context": context.key, "history": HISTORY, "outcome": outcome, "time": int(row.time), "period": str(date),
            "output_support_start": str(date-support), "output_support_horizon": support,
            "event_ids_at_t": json.dumps(current), "event_ids_in_output_support": json.dumps(overlapping),
            "retain_event_dates": not current, "retain_strict": not overlapping})
    return frame, pd.DataFrame(masks), metadata


def design(sample, specification, structure, horizon):
    matrix, names, selected, components, sectors, lags = history.design(sample, specification, structure, horizon)
    first = matrix.shape[1]
    matrix = np.column_stack([matrix, (sample[ADDED]-sample[ADDED].mean()).to_numpy()])
    return matrix, names+ADDED, np.r_[selected, np.arange(first, first+len(ADDED))], components, sectors, lags


def matrix_rank(matrix):
    norms = np.linalg.norm(matrix, axis=0)
    singular = np.linalg.svd(matrix/np.where(norms > 0, norms, 1), compute_uv=False)
    return int(np.sum(singular > singular.max()*max(matrix.shape)*np.finfo(float).eps))


def wald(beta, covariance, reference, restriction, metadata, name):
    r = np.atleast_2d(restriction)
    estimate = r@beta
    matrix = r@covariance@r.T; matrix = (matrix+matrix.T)/2
    singular = np.linalg.svd(matrix, compute_uv=False)
    scale = float(np.linalg.norm(r@reference@r.T, 2))
    cutoff = max(float(singular[0])*RANK, scale*ZERO)
    uncapped = int(np.sum(singular > cutoff))
    bound = metadata["n_sectors"]-1 if metadata["covariance"] == "Sector cluster" else metadata["n_periods"]-1
    rank = min(uncapped, max(bound, 0)); q = len(r); degrees = metadata["inference_df"]
    available = rank == q and singular[0] > scale*ZERO and degrees > 0
    statistic = float(estimate@np.linalg.solve(matrix, estimate)) if available else np.nan
    assert not available or statistic >= -1e-8
    condition = float(np.linalg.cond(matrix)); flags = []
    if q >= .8*bound: flags.append("Restriction count approaches the universal score-rank bound")
    if condition > 1e8: flags.append("Ill-conditioned restriction covariance")
    if metadata["time_residual_df"] < 10: flags.append("Fewer than ten residual time degrees of freedom")
    return {**metadata, "test": name, "restrictions": q, "uncapped_covariance_rank": uncapped, "covariance_rank": rank,
        "structural_rank_bound": bound, "rank_bound_rule": "G−1 unit-cluster / T−1 universal calendar-score bound",
        "covariance_condition": condition, "hc0_covariance_scale": scale, "zero_covariance_threshold": scale*ZERO, "rank_cutoff": cutoff,
        "status": "available" if available else "unavailable: rank-deficient or numerically degenerate restriction covariance",
        "wald_chi2": max(statistic, 0.) if available else None, "f_statistic": max(statistic, 0.)/q if available else None,
        "numerator_df": q, "denominator_df": degrees,
        "p_value": float(f_distribution.sf(max(statistic, 0.)/q, q, degrees)) if available else None,
        "joint_test_fragile": bool(flags), "fragility_reason": "; ".join(flags)}


def unavailable_test(metadata, name, restriction):
    q = len(np.atleast_2d(restriction))
    bound = metadata["n_sectors"]-1 if metadata["covariance"] == "Sector cluster" else metadata["n_periods"]-1
    return {**metadata, "test": name, "restrictions": q, "covariance_rank": None, "uncapped_covariance_rank": None,
        "structural_rank_bound": bound, "rank_bound_rule": "G−1 unit-cluster / T−1 universal calendar-score bound; complete point design unavailable",
        "status": "unavailable: complete design/time support", "wald_chi2": None, "f_statistic": None, "p_value": None,
        "numerator_df": q, "denominator_df": metadata["inference_df"], "joint_test_fragile": True,
        "fragility_reason": "Point/time design unavailable"}


def effect(beta, covariance, hc0, vector, metadata):
    if beta is None:
        return {**metadata, **{name: None for name in ["coefficient", "variance", "std_error", "ci_lower", "ci_upper", "p_value", "hc0_reference_variance", "zero_variance_threshold"]},
                "degenerate": False, "inference_status": "unavailable: complete design/time support"}
    return {**metadata, **shared.inference(float(vector@beta), float(vector@covariance@vector), metadata["inference_df"], float(vector@hc0@vector))}


def fit_spec(context, sample, base, specification, structure, books, *, preflight):
    matrix, names, selected, components, sectors, lags = design(sample, specification, structure, context.horizon)
    geo = shared.geometry(matrix)
    n, k, t = len(sample), len(names), sample.time.nunique()
    local_ranks = [matrix_rank(matrix[sample.sector_id.eq(unit)]) for unit in sectors]
    time_rank = max(local_ranks)
    status = geo["status"] if geo["status"] != "estimated" else "estimated" if t > time_rank else "unavailable: nonpositive residual time degrees of freedom"
    temperatures = ["temp_endpoint_difference", "temp_growth_sum"] if specification == "two_channel" else [f"temp_l{lag}" for lag in lags]
    controls = [f"rain_l{lag}" for lag in lags]+["rh_pct"]+([f"ar_l{lag}" for lag in range(1, context.horizon+1)] if specification == "dynamic" else [])+ADDED
    spec_id = "__".join([context.key, base["outcome"], base["sample_variant"], structure, specification])
    dates = pd.PeriodIndex(sample[context.period], freq=context.freq)
    meta = {**base, "spec_id": spec_id, "native_label": "Literature full controls "+spec_id,
        "model_spec": specification, "model": structure, "specification": specification, "structure": structure,
        "formula": base["outcome"]+" ~ "+" + ".join(temperatures)+(" ×sector (main + nonreference deviations)" if structure == "H1" else "")+" + "+" + ".join(controls)+" + sector×season FE + common year FE",
        "controls": json.dumps(controls), "temperature_terms": json.dumps(temperatures),
        "current_temperature_representation": "T0−Th and sum(T0,...,T[h−1]); both contain current T0" if specification == "two_channel" else "temp_l0",
        "AR_slope_scope": "Shared across sectors" if specification == "dynamic" else "None",
        "n_obs": n, "n_parameters": k, "n_periods": t, "n_sectors": len(sectors), "residual_df": n-geo["design_rank"],
        "time_design_rank": time_rank, "time_design_rank_min": min(local_ranks), "time_residual_df": t-time_rank,
        "design_rank": geo["design_rank"], "design_condition_number": geo["design_condition_number"],
        "scaled_design_condition_number": geo["scaled_design_condition_number"], "design_rank_threshold": geo["design_rank_threshold"], "status": status,
        "start": str(dates.min()), "end": str(dates.max()), "calendar_gaps": dates.max().ordinal-dates.min().ordinal+1-t,
        "mean_y": float(sample[base["outcome"]].mean()), "sd_y": float(sample[base["outcome"]].std()),
        "reference_sector_id": sectors[0] if structure == "H1" else "", "estimator": "PyFixest feols", "estimator_version": native.ESTIMATOR_VERSION,
        "coefficient_sum_interpretation": "Sum of direct conditional temperature coefficients; no unconditional long-run multiplier in dynamic models",
        "two_channel_construction": "T0−Th and sum(T0,...,T[h−1]); a coefficient on the h-period mean is h times the sum-exposure coefficient" if specification == "two_channel" else ""}
    books["specifications"].append(meta)
    if preflight: return
    methods = ([('Sector cluster', 0, len(sectors)-1)] if structure == "common" and len(sectors)>1 else [])
    methods += [("Calendar HAC" if len(sectors)==1 else "Driscoll-Kraay", lag, t-1) for lag in [context.horizon, 2*context.horizon]]
    archive = {"X": matrix, "Y": sample[base["outcome"]].to_numpy(), "names": np.asarray(names, str), "coef_names": np.asarray(names, str),
        "time": sample.time.to_numpy(), "periods": dates.astype(str).to_numpy(dtype=str), "units": sample.sector_id.to_numpy(dtype=str),
        "selected_positions": selected, "component_positions": components, "lags": np.asarray(lags), "temperature_features": np.asarray(temperatures, str),
        "covariance_labels": np.asarray([m[0] for m in methods], str), "bandwidths": np.asarray([m[1] for m in methods]), "inference_df": np.asarray([m[2] for m in methods])}
    meta["design_npz"] = relative(OUT/"designs"/f"{spec_id}.npz")
    core_names = ["spec_id", "context", "context_label", "history", "outcome", "output_adjustment", "analysis_role", "sample_variant", "frequency", "horizon",
        "model_spec", "model", "specification", "structure", "n_obs", "n_parameters", "n_periods", "n_sectors", "time_residual_df", "status", "mean_y", "sd_y", "reference_sector_id", "estimator", "estimator_version"]
    core = {key: meta[key] for key in core_names}
    beta = residual = influence = hc0 = None
    if status == "estimated":
        fitted = native.fit_many_matrix(sample[[base["outcome"]]].to_numpy(), matrix, names, meta["native_label"], outcome_names=[base["outcome"]])[0]
        beta, residual = fitted.params.to_numpy(), fitted.resid
        influence = (np.asarray(fitted.native_model.within_data.design)@fitted.coefficient_transform.T)*residual[:, None]
        hc0 = influence[:, selected].T@influence[:, selected]
        archive.update(coefficients=beta, beta=beta, residuals=residual, residual=residual, hc0_selected=hc0)
        books["coefficients"].extend({**core, "term": name, "coefficient": float(value)} for name, value in zip(names, beta, strict=True))
    component_indexes = np.asarray([[list(selected).index(index) for index in part] for part in components])
    unit_names = sample.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    corrections = []
    for number, (method, bandwidth, degrees) in enumerate(methods):
        factor = None; covariance = None
        if beta is not None:
            if method == "Sector cluster":
                factor = len(sectors)/(len(sectors)-1)*(n-1)/(n-k)
                scores = np.zeros((len(sectors), k)); np.add.at(scores, pd.Categorical(sample.sector_id, categories=sectors).codes, influence)
                full = scores.T@scores*factor
            else:
                factor = t/(t-1)*(n-1)/(n-k)
                full = history.calendar_covariance(influence, sample.time.to_numpy(), bandwidth, factor)
            full = (full+full.T)/2; archive[f"covariance_{number}"] = full
            covariance = full[np.ix_(selected, selected)]
        corrections.append(np.nan if factor is None else factor)
        band = {**core, "covariance": method, "hac_lags": bandwidth, "inference_df": degrees,
            "finite_sample_correction": factor, "n_clusters": len(sectors) if method=="Sector cluster" else 0,
            "main_inference": method=="Sector cluster" if structure=="common" and len(sectors)>1 else bandwidth==context.horizon,
            "r_squared": float(fitted.rsquared) if beta is not None else None}
        selected_beta = beta[selected] if beta is not None else None
        profiles = []
        profile_units = sectors if structure=="H1" else [sectors[0] if len(sectors)==1 else "pooled"]
        identified_bases = []
        for i, unit in enumerate(profile_units):
            basis = np.zeros((len(components), len(selected)))
            for feature, indexes in enumerate(component_indexes):
                basis[feature, indexes[0]] = 1
                if structure=="H1" and i: basis[feature, indexes[i]] = 1
            identified_bases.append(basis)
            contrasts = basis
            if specification=="two_channel":
                expansion = np.zeros((len(lags),2)); expansion[0,0], expansion[-1,0] = 1,-1; expansion[:-1,1] = 1
                contrasts = expansion@basis
            profiles.append(contrasts)
            unit_sample = sample if unit=="pooled" else sample.loc[sample.sector_id.eq(unit)]
            unit_meta = {**band, "sector_id": unit, "sector": unit_names.get(unit,"Common equal-unit slope"),
                "mean_y": float(unit_sample[base["outcome"]].mean()), "sd_y": float(unit_sample[base["outcome"]].std())}
            for i_lag, lag in enumerate(lags):
                detail = {**unit_meta, "effect": "lag", "lag": lag, "term": f"temp_l{lag}"}
                if beta is not None and structure=="H1" and specification!="two_channel":
                    indexes = component_indexes[i_lag]; ref, delta = indexes[0], indexes[i]
                    detail.update(temp_main=float(selected_beta[ref]), interaction_delta=0. if i==0 else float(selected_beta[delta]), delta_is_normalization=i==0,
                        var_temp_main=float(covariance[ref,ref]), var_interaction_delta=0. if i==0 else float(covariance[delta,delta]), cov_temp_main_delta=0. if i==0 else float(covariance[ref,delta]))
                books["temperature_effects"].append(effect(selected_beta,covariance,hc0,contrasts[i_lag],detail))
            books["temperature_effects"].append(effect(selected_beta,covariance,hc0,contrasts.sum(axis=0),
                {**unit_meta,"effect":"cumulative_direct","lag":-1,"term":"sum_temperature_lags"}))
            if specification=="two_channel":
                for j, channel in enumerate(temperatures):
                    books["channel_estimates"].append(effect(selected_beta,covariance,hc0,basis[j],
                        {**unit_meta,"channel":channel,"term":channel,"effect_units":"Per +1°C endpoint difference" if j==0 else "Per +1°C-period exposure sum"}))
                nulls = [("All temperature-lag coefficients zero",np.eye(2)),("Cumulative direct temperature response zero",np.array([[0.,context.horizon]])),
                    ("Exact contemporaneous static-output-level restriction",np.array([[0.,1.]])),("Equal one-period growth responses over YoY accumulation window",np.array([[1.,0.]]))]
                tests = [(name,r@basis) for name,r in nulls]
            else:
                tests = [(name,r@contrasts) for name,r in history.restrictions(len(lags),base["outcome"])]
            for name, restriction in tests:
                books["tests"].append(wald(selected_beta,covariance,hc0,restriction,unit_meta,name) if beta is not None else unavailable_test(unit_meta,name,restriction))
        for control in controls:
            vector = np.eye(len(selected))[list(selected).index(names.index(control))]
            books["control_estimates"].append(effect(selected_beta,covariance,hc0,vector,{**band,"term":control,"sector_id":"shared","sector":"Shared control slope"}))
        if structure=="H1":
            for j, feature in enumerate(temperatures):
                for i, unit in enumerate(sectors):
                    vector = np.eye(len(selected))[component_indexes[j,i]]
                    books["interaction_components"].append(effect(selected_beta,covariance,hc0,vector,
                        {**band,"sector_id":unit,"lag":lags[j] if specification!="two_channel" else -2-j,"component":"base" if i==0 else "interaction",
                         "estimated":beta is not None,"normalization":False,"term":names[components[j,i]]}))
                books["interaction_components"].append({**band,"sector_id":sectors[0],"lag":lags[j] if specification!="two_channel" else -2-j,
                    "component":"interaction","estimated":False,"normalization":True,"term":f"{feature}:C(sector_id)[T.{sectors[0]}]","coefficient":0.,
                    "variance":None,"std_error":None,"ci_lower":None,"ci_upper":None,"p_value":None,"inference_status":"Reference interaction normalized to zero; not estimated"})
            global_tests = [("All sector cumulative direct temperature responses equal",np.vstack([(part-profiles[0]).sum(axis=0) for part in profiles[1:]])),
                ("All sector temperature-lag response vectors equal",np.vstack([part-identified_bases[0] for part in identified_bases[1:]])),
                ("All sector temperature-lag responses jointly zero",np.vstack(identified_bases))]
            for name, restriction in global_tests:
                books["tests"].append(wald(selected_beta,covariance,hc0,restriction,{**band,"sector_id":"all"},name) if beta is not None else unavailable_test({**band,"sector_id":"all"},name,restriction))
    archive["corrections"] = np.asarray(corrections)
    np.savez_compressed(OUT/"designs"/f"{spec_id}.npz",**archive)


def protected_inventory():
    paths = {ROOT/name for name in ["scripts/crisis_sensitivity.py","scripts/literature_sensitivity.py","scripts/annual_literature_sensitivity.py","scripts/annual_seasonal_literature.py",
        "scripts/literature_rank_correction.py","scripts/pyfixest_backend.py","scripts/extended_controls.py",
        "outputs/pyfixest/fit_log.csv","outputs/pyfixest/estimator_manifest.json","outputs/pyfixest/migration_input_reference.json",
        "outputs/covid_comparison/original_inputs_reference.json","outputs/crisis_sensitivity/manifest.json",
        "outputs/complete_va_controls/manifest.json"]}
    for folder in [ROOT/"outputs/literature_sensitivity",ROOT/"outputs/crisis_sensitivity/parents"]:
        for path in folder.rglob("*"):
            # Verification/presentation evidence is refreshed after native fits.
            # Its hashes are not analytical inputs; native/design/test lineage
            # remains protected, including rank/span and native-stage archives.
            derived = (path.name.startswith("verification") or "verification_history" in path.parts
                or path.name in {"report_inventory.json","report_verification.json","payload_packaging.json"})
            if path.is_file() and path.suffix in {".json",".csv",".npz"} and not derived and OUT not in path.parents:
                paths.add(path)
    return {relative(path):shared.sha(path) for path in sorted(paths)}


def run_literature_full_controls(policy_path=shared.DEFAULT_POLICY, *, preflight=False):
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ["samples","designs"]:(OUT/name).mkdir(exist_ok=True)
    own_sha = shared.sha(Path(__file__)); protected = protected_inventory()
    raw_full_control_history.cache_clear(); controls.cache_clear(); history.maximum_raw_history.cache_clear(); history.raw_growth_registry.cache_clear()
    policy_path = Path(policy_path).resolve(); policy = shared.load_policy(policy_path)
    input_paths = [policy_path,ROOT/"data/processed/weather_monthly.csv",ROOT/"data/processed/weather_quarterly.csv",
        ROOT/"data/processed/extended_controls/monthly_controls.csv",ROOT/"data/processed/extended_controls/quarterly_controls.csv",
        ROOT/"data/processed/extended_controls/data_checks.json",ROOT/"data/raw/manifest.json",ROOT/"data/raw/extended_controls/manifest.json"]
    input_paths += [ROOT/f"data/raw/{prefix}_{kind}.json" for prefix in ["va_sa","va_nsa","iip_sa","iip_nsa"] for kind in ["tabledata","metadata"]]
    input_paths += [path for path in (ROOT/"data/raw/extended_controls").iterdir() if path.is_file()]
    inputs = {relative(path):shared.sha(path) for path in input_paths}
    books = {name:[] for name in ["specifications","coefficients","temperature_effects","control_estimates","interaction_components","channel_estimates","tests"]}
    masks,inventory = [],[]
    old = native.LOG_DIR,list(native._FIT_LOG),native._API_COUNTER,native._RUN_STARTED
    receipt = {}
    if not preflight:native.LOG_DIR=OUT/"pyfixest";native.reset_fit_log()
    try:
        with threadpool_limits(limits=1):
            for context in shared.contexts():
                for outcome in OUTCOMES:
                    maximum, mask, base = prepare(context,outcome,policy); masks.extend(mask.to_dict("records"))
                    for variant in VARIANTS:
                        keep = np.ones(len(mask),bool) if variant==VARIANTS[0] else mask["retain_event_dates" if variant==VARIANTS[1] else "retain_strict"].to_numpy()
                        sample = maximum.loc[maximum.time.isin(mask.loc[keep,"time"])].copy().reset_index(drop=True)
                        assert not sample.empty
                        path=OUT/"samples"/f"{context.key}__{outcome}__{variant}.csv";sample.to_csv(path,index=False)
                        meta={**base,"sample_variant":variant,"sample_csv":relative(path),"sample_sha256":shared.sha(path),
                            "policy_source":relative(policy_path),"policy_source_sha256":inputs[relative(policy_path)],
                            "crisis_loss_periods":maximum.time.nunique()-sample.time.nunique(),"retained_n_periods":sample.time.nunique(),
                            "sample_comparability":"Exact same dates/Y/current90% eligibility across all four requested weather models within context/outcome/variant"}
                        inventory.append({**meta,"n_periods":sample.time.nunique(),"n_sectors":sample.sector_id.nunique()})
                        for specification in SPECIFICATIONS+(["two_channel"] if outcome!="y_short" else []):
                            for structure in ["common"]+(["H1"] if sample.sector_id.nunique()>1 else []):
                                fit_spec(context,sample,meta,specification,structure,books,preflight=preflight)
                        print(f"Full-control literature {'preflight' if preflight else 'fit'} {context.key}/{outcome}/{variant}: {sample.sector_id.nunique()} units × {sample.time.nunique()} dates",flush=True)
            if not preflight:receipt=native.write_fit_log()
    finally:
        if not preflight:
            native.LOG_DIR,log,native._API_COUNTER,native._RUN_STARTED=old;native._FIT_LOG[:]=log
    assert all(shared.sha(ROOT/path)==value for path,value in inputs.items()),"Cached input changed"
    assert all(shared.sha(ROOT/path)==value for path,value in protected.items()),"Prior protected analytical file changed"
    assert shared.sha(Path(__file__))==own_sha,"Producer source changed during run"
    specs=pd.DataFrame(books["specifications"])
    specs.to_csv(OUT/("preflight_specifications.csv" if preflight else "specifications.csv"),index=False)
    if not preflight:
        effects=pd.DataFrame(books["temperature_effects"])
        effects["p_value_holm_within_sector_family"]=np.nan
        family=["context","outcome","sample_variant","model_spec","covariance","hac_lags","effect","lag","term"]
        for _,part in effects.loc[effects.model.eq("H1")].groupby(family,sort=False):
            p=part.p_value.fillna(1).to_numpy();order=np.argsort(p);adjusted=np.empty(len(p));adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
            effects.loc[part.index,"p_value_holm_within_sector_family"]=np.where(part.p_value.notna(),adjusted,np.nan)
        effects.to_csv(OUT/"temperature_effects.csv",index=False)
        for name,rows in books.items():
            if name not in {"specifications","temperature_effects"}:pd.DataFrame(rows).to_csv(OUT/f"{name}.csv",index=False)
    pd.DataFrame(inventory).to_csv(OUT/"sample_inventory.csv",index=False);pd.DataFrame(masks).to_csv(OUT/"masks.csv",index=False)
    (OUT/"policy.json").write_text(json.dumps(policy,indent=2,ensure_ascii=False)+"\n")
    manifest={"status":"preflight_only" if preflight else "estimated_not_yet_independently_verified","contexts":len(shared.contexts()),"history":HISTORY,
        "growth_outcomes":OUTCOMES,"sample_variants":VARIANTS,"samples":len(inventory),"requested_specs":len(specs),
        "identified_specs":int(specs.status.eq("estimated").sum()),"unavailable_specs":int(specs.status.ne("estimated").sum()),
        "native_fits":receipt.get("native_fits_recorded",0),"row_counts":{name:len(rows) for name,rows in books.items()},
        "inputs_sha256":inputs,"protected_sources_sha256":protected,"producer_source":relative(Path(__file__)),"producer_source_sha256":own_sha,
        "scope":"Current,DL0..h,dynamic DL0..h+Ylags1..h; two-channel YoY; common/H1 only; five current weather variables included in every actual design",
        "control_set":"full90","current_weather_variables":CURRENT_WEATHER,
        "selection":"Full raw maximum weather/output-growth history, current-date finite wind/PM plus original90% coverage, named-target eligibility alignment, then fixed crisis masks; losses separately reported",
        "historical_PM_rule":"No historical wind/PM lags or historical PM coverage requirement; only current values enter X",
        "strict_output_support":"t−h−1..t for short growth; t−2h..t for YoY, identical maximum support across all model specifications",
        "rank_rule":"Actual numerical SVD rank with universal T−1 calendar /G−1 unit-score upper bounds; no single-temperature seasonal cap",
        "point_guard":"Full column rank, N>K and positive residual time df T−max_sector_rank(X_i); unavailable requests preserve complete X/Y/controls and placeholder effects/tests",
        "inference":"Common multiunit: sectorCR1 main, calendar DK h/2h sensitivity. H1: joint DK h/2h. National: calendar HAC h/2h; actual gaps retained",
        "numerical_thresholds":{"zero_relative_to_HC0":ZERO,"covariance_rank_relative":RANK},
        "not_replication":"Singapore adaptation; no published-paper replication, stationarity clearance, dynamic-panel bias correction or causal validation claim",
        "native_ledger":relative(OUT/"pyfixest"),"full_covariance_storage":"designs/{spec_id}.npz full nuisance-inclusive covariance_0,... with method labels/bandwidths/df/corrections"}
    (OUT/("preflight.json" if preflight else "manifest.json")).write_text(json.dumps(shared.clean(manifest),indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({key:manifest[key] for key in ["status","requested_specs","identified_specs","unavailable_specs","native_fits"]},indent=2),flush=True)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--policy",type=Path,default=shared.DEFAULT_POLICY);parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args();run_literature_full_controls(args.policy,preflight=args.preflight)
