"""Annual growth and current/prior calendar-quarter weather exposures.

Colacito, Hoffmann and Phan motivate season-specific annual-output regressions.
This Singapore adaptation uses real annual output, Q1--Q4 (not climatic season
names), quarterly temperature/rain, current humidity, sector FE and a common
trend. It preserves unavailable requests and never edits a previous analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import annual_crisis_sensitivity as crisis
import crisis_sensitivity as shared
import literature_sensitivity as subannual
import pyfixest_backend as native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/literature_sensitivity/annual_seasonal"
OUTCOMES = ["y_nsa", "y_sa"]
HISTORIES = ["model_own_history", "matched_max_AR1_history"]
MODELS = ["seasonal_current", "seasonal_current_prior", "seasonal_current_AR1", "seasonal_current_prior_AR1"]
VARIANTS = shared.VARIANTS
CONTEXT_LABELS = {"aggregate_va": "Official aggregate GVA", "va11": "11 broad VA sectors",
    "va19": "19 detailed VA sectors", "va12": "12 complete broad VA components", "va20": "20 complete detailed VA components",
    "iip_total": "Published weighted IIP total", "iip6": "6 manufacturing clusters", "iip18": "18 manufacturing industries",
    "aggregate_va_matched_va20": "Official aggregate GVA · complete-detailed matched dates",
    "va11_matched_va19": "11 broad VA sectors · detailed-matched dates", "va12_matched_va20": "12 complete broad VA components · detailed-matched dates",
    "iip_total_matched_iip6": "Published weighted IIP total · cluster-matched dates",
    "iip_total_matched_iip18": "Published weighted IIP total · industry-matched dates",
    "iip6_matched_iip18": "6 manufacturing clusters · industry-matched dates"}
PROTOCOLS = ["notes/annual_seasonal_specification_protocol.md", "notes/literature_history_alignment_v2.md"]


def holm_probabilities(probabilities):
    """Holm adjusted p-values; unavailable family slots remain at p=1.

    This is the same sorted cumulative-maximum formula as statsmodels, without
    forcing a process-wide garbage collection for every small sector family.
    """
    order = np.argsort(probabilities)
    adjusted = np.minimum(np.maximum.accumulate(probabilities[order] * np.arange(len(order), 0, -1)), 1.)
    result = np.empty(len(order));result[order] = adjusted
    return result


def rel(path): return str(Path(path).resolve().relative_to(ROOT))


def feature_names(model):
    lags = [0, 1] if "prior" in model else [0]
    temperature = [f"temp_Q{quarter}_l{lag}" for lag in lags for quarter in range(1, 5)]
    controls = [f"rain_Q{quarter}_l{lag}" for lag in lags for quarter in range(1, 5)] + ["rh_pct"]
    if model.endswith("AR1"): controls += ["ar_l1"]
    return temperature, controls


def parent_histories():
    panels, provenance = crisis.contexts()
    raw_path = ROOT / "data/processed/annual/annual_levels.csv"
    raw = pd.read_csv(raw_path, dtype={"sector_id": str})
    registry = raw.set_index(["family", "sector_id", "adjustment", "year"]).annual_log_growth
    assert registry.index.is_unique
    path = ROOT / "data/processed/weather_quarterly.csv"
    weather = pd.read_csv(path)
    periods = pd.PeriodIndex(weather.quarter, freq="Q")
    weather["year"], weather["season"] = periods.year, periods.quarter
    weather["rain_100mm"] = weather.rain_mm / 100
    complete = weather.groupby("year").season.nunique().eq(4)
    weather = weather.loc[weather.year.isin(complete.index[complete])]
    seasonal = weather.pivot(index="year", columns="season", values=["temp_c", "rain_100mm"])
    seasonal.columns = [f"{'temp' if term == 'temp_c' else 'rain'}_Q{quarter}" for term, quarter in seasonal.columns]
    for context, parent in panels.items():
        frame = parent.sort_values(["year", "sector_id"]).reset_index(drop=True).copy()
        for prefix in ["temp", "rain"]:
            for quarter in range(1, 5):
                for lag in [0, 1]:
                    frame[f"{prefix}_Q{quarter}_l{lag}"] = seasonal[f"{prefix}_Q{quarter}"].reindex(frame.year - lag).to_numpy()
        for outcome in OUTCOMES:
            family = "iip" if context.startswith("iip") else "va"
            adjustment = "NSA" if outcome == "y_nsa" else "SA"
            actual = registry.reindex([(family, sector, adjustment, int(year)) for sector, year in zip(frame.sector_id, frame.year, strict=True)]).to_numpy()
            assert np.allclose(frame[outcome], actual, atol=2e-10, rtol=2e-11), (context, outcome, "raw annual-growth registry")
            frame[f"ar_l1_{outcome}"] = registry.reindex([(family, sector, adjustment, int(year) - 1) for sector, year in zip(frame.sector_id, frame.year, strict=True)]).to_numpy()
        panels[context] = frame
    return panels, provenance, path


def history_eligible(parent, model, history, outcome):
    frame = parent.copy()
    frame["ar_l1"] = frame[f"ar_l1_{outcome}"]
    temp, controls = feature_names("seasonal_current_prior_AR1" if history == "matched_max_AR1_history" else model)
    eligible = np.isfinite(frame[[outcome] + temp + controls]).all(axis=1)
    complete = pd.Series(eligible, index=frame.index).groupby(frame.year).all()
    return frame.loc[frame.year.isin(complete.index[complete])].copy()


def sample_frame(parent, model, history, outcome, variant, policy, match_template=None):
    frame = history_eligible(parent, model, history, outcome)
    raw_periods = int(frame.year.nunique())
    target_periods = None
    if match_template is not None:
        target = history_eligible(match_template, model, history, outcome)
        target_periods = int(target.year.nunique())
        frame = frame.loc[frame.year.isin(target.year)].copy()
    event_years = {year for event in policy["events"] for year in range(int(event["start"][:4]), int(event["end"][:4]) + 1)}
    masks = []
    for year in sorted(frame.year.unique()):
        current = int(year) in event_years
        support = sorted(event_years.intersection(range(int(year) - 2, int(year) + 1)))
        retain = variant == "all_eligible" or (not current if variant == "event_dates_excluded" else not support)
        masks.append({"year": int(year), "retained": retain, "current_year_event": current,
            "event_years_in_output_support": json.dumps(support), "output_support_start": int(year) - 2})
    years = [row["year"] for row in masks if row["retained"]]
    result = frame.loc[frame.year.isin(years)].copy().reset_index(drop=True)
    return result, masks, int(frame.year.nunique()), {"raw_model_history_eligible_n_periods": raw_periods,
        "matched_template_eligible_n_periods": target_periods, "matched_history_loss_periods": raw_periods - frame.year.nunique()}


def design(frame, model, structure):
    sectors = sorted(frame.sector_id.unique())
    heat, controls = feature_names(model)
    indicators = np.column_stack([frame.sector_id.eq(sector) for sector in sectors]).astype(float)
    names = [f"sector[{sector}]" for sector in sectors] + ["trend"] + controls
    trend = frame.year.to_numpy() - frame.year.mean()
    centered_controls = (frame[controls] - frame[controls].mean()).to_numpy()
    blocks = [indicators, trend[:, None], centered_controls]
    selected = list(range(len(sectors), len(names)))
    components = []
    for term in heat:
        centered = frame[term].to_numpy() - frame[term].mean()
        position = len(names)
        if structure == "H1":
            blocks.append(np.column_stack([centered, indicators[:, 1:] * centered[:, None]]))
            names += [term] + [f"{term}:C(sector_id)[T.{sector}]" for sector in sectors[1:]]
            components.append(list(range(position, position + len(sectors))))
        else:
            blocks.append(centered[:, None]);names.append(term);components.append([position])
        selected += list(range(position, len(names)))
    return np.column_stack(blocks), names, np.asarray(selected), np.asarray(components), sectors


def effect_contrasts(basis, model):
    heat, _ = feature_names(model)
    result = [("season_lag", quarter, lag, basis[i], heat[i]) for i, (lag, quarter) in
              enumerate((lag, quarter) for lag in ([0, 1] if "prior" in model else [0]) for quarter in range(1, 5))]
    result.append(("current_year_sum", 0, 0, basis[:4].sum(axis=0), "sum_current_Q1_Q4"))
    if "prior" in model:
        result.append(("prior_year_sum", 0, 1, basis[4:].sum(axis=0), "sum_prior_Q1_Q4"))
        for quarter in range(1, 5):
            result.append(("same_season_pair_sum", quarter, -1, basis[quarter - 1] + basis[quarter + 3], f"Q{quarter}_current_plus_prior"))
    result.append(("all_season_sum", 0, -1, basis.sum(axis=0), "sum_all_temperature_features"))
    return result


def nulls(basis, model):
    tests = [("All seasonal temperature coefficients zero", basis),
             ("Current Q1–Q4 temperature coefficients equal", basis[1:4] - basis[[0]]),
             ("Sum of all direct seasonal temperature coefficients zero", basis.sum(axis=0, keepdims=True))]
    if "prior" in model:
        tests += [("Prior Q1–Q4 temperature coefficients equal", basis[5:] - basis[[4]]),
                  ("Current and prior same-quarter temperature coefficients equal", basis[:4] - basis[4:]),
                  ("All same-quarter current-plus-prior coefficient sums zero", basis[:4] + basis[4:])]
    return tests


def fit_spec(sample, metadata, model, structure, books, *, preflight):
    spec_id = "__".join([metadata[key] for key in ["context", "history", "model_spec", "outcome", "sample_variant"]] + [structure])
    heat, controls = feature_names(model)
    x, names, selected, components, sectors = design(sample, model, structure)
    geo = shared.geometry(x)
    n, k, t = len(sample), x.shape[1], sample.year.nunique()
    temporal_ranks = [shared.geometry(x[sample.sector_id.eq(sector).to_numpy()])["design_rank"] for sector in sectors]
    temporal_rank = max(temporal_ranks)
    status = geo["status"]
    reason = "" if status == "estimated" else "Requested complete design is rank deficient or has no observation residual df"
    if t <= temporal_rank:
        status = "unavailable";reason = "No residual time degrees of freedom in the complete reference-unit temporal design"
    meta = {**metadata, "spec_id": spec_id, "model": structure, "native_label": "Annual seasonal literature " + spec_id,
        "temperature_terms": json.dumps(heat), "controls": json.dumps(controls), "status": status, "reason": reason,
        "n_obs": n, "n_periods": t, "n_sectors": len(sectors), "n_parameters": k, "design_rank": geo["design_rank"],
        "residual_df": n - geo["design_rank"], "time_design_rank": temporal_rank, "time_design_rank_min": min(temporal_ranks),
        "time_residual_df": t - temporal_rank, "design_condition_number": geo["design_condition_number"],
        "scaled_design_condition_number": geo["scaled_design_condition_number"], "design_rank_threshold": geo["design_rank_threshold"],
        "mean_y": float(sample[metadata["outcome"]].mean()), "sd_y": float(sample[metadata["outcome"]].std()),
        "start": int(sample.year.min()), "end": int(sample.year.max()), "calendar_gaps": int(sample.year.max() - sample.year.min() + 1 - t),
        "reference_sector_id": sectors[0] if structure == "H1" else "", "linear_trend": True, "year_fixed_effects": False,
        "sector_fixed_effects": len(sectors) > 1, "season_fixed_effects": False, "frequency": "annual",
        "AR_slope_scope": "Shared across sectors" if model.endswith("AR1") else "None",
        "formula": metadata["outcome"] + " ~ " + " + ".join(heat) + (" interacted with sector" if structure == "H1" else "") + " + " + " + ".join(controls) + " + common linear trend + sector intercepts",
        "design_npz": rel(OUT / "designs" / f"{spec_id}.npz"),
        "estimator": "PyFixest feols", "estimator_version": native.ESTIMATOR_VERSION,
        "stationarity_status": "Not cleared: annual short/gapped histories are exploratory; no stationarity-driven sample selection",
        "interpretation": "Direct conditional seasonal coefficients; sums in AR1 models are not unconditional long-run multipliers"}
    books["specifications"].append(meta)
    if preflight: return
    archive = {"X": x, "Y": sample[metadata["outcome"]].to_numpy(), "names": np.asarray(names, str),
        "time": sample.year.to_numpy(), "sector_id": sample.sector_id.to_numpy(dtype=str), "selected_positions": selected,
        "component_positions": components, "temperature_features": np.asarray(heat, str)}
    methods = [("Sector cluster", 0, len(sectors) - 1)] if structure == "common" and len(sectors) > 1 else []
    methods += [("Calendar HAC" if len(sectors) == 1 else "Driscoll-Kraay", lag, t - 1) for lag in [1, 2]]
    covariance_items = []
    if status == "estimated":
        fit = native.fit_many_matrix(sample[[metadata["outcome"]]].to_numpy(), x, names, meta["native_label"], outcome_names=[metadata["outcome"]])[0]
        beta, residual = fit.params.to_numpy(), fit.resid
        influence = geo["mapping"].T * residual[:, None]
        hc0 = influence[:, selected].T @ influence[:, selected]
        archive.update(beta=beta, residual=residual, hc0_selected=hc0)
        books["coefficients"].extend({"spec_id": spec_id, "context": meta["context"], "context_label": meta["context_label"], "outcome": meta["outcome"],
            "term": term, "coefficient": float(value), "estimator": meta["estimator"], "estimator_version": meta["estimator_version"]}
            for term, value in zip(names, beta, strict=True))
        for method, lag, degrees in methods:
            if method == "Sector cluster":
                factor = len(sectors) / (len(sectors) - 1) * (n - 1) / (n - k)
                scores = np.zeros((len(sectors), k))
                np.add.at(scores, pd.Categorical(sample.sector_id, categories=sectors).codes, influence)
                covariance = scores.T @ scores * factor
                formula = "G/(G−1)×(N−1)/(N−K)"
            else:
                factor = t / (t - 1) * (n - 1) / (n - k)
                covariance = subannual.calendar_covariance(influence, sample.year.to_numpy(), lag, factor)
                formula = "T/(T−K)" if len(sectors) == 1 else "T/(T−1)×(N−1)/(N−K)"
            covariance_items.append((method, lag, degrees, factor, formula, covariance))
    else:
        covariance_items = [(method, lag, degrees, None, "Unavailable: complete point/time design not estimable", None) for method, lag, degrees in methods]
    positions = np.asarray([[list(selected).index(index) for index in row] for row in components])
    sector_names = sample.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    for number, (method, lag, degrees, factor, formula, full_covariance) in enumerate(covariance_items):
        band_meta = {**meta, "covariance": method, "hac_lags": lag, "inference_df": degrees,
            "finite_sample_correction": factor, "finite_sample_correction_formula": formula,
            "n_clusters": len(sectors) if method == "Sector cluster" else 0,
            "main_inference": method == "Sector cluster" if structure == "common" and len(sectors) > 1 else lag == 1}
        if full_covariance is not None:
            archive[f"covariance_{number}"] = full_covariance
            covariance = full_covariance[np.ix_(selected, selected)]
            band_meta["r_squared"] = float(fit.rsquared)
        profiles = []
        for unit_number, sector in enumerate(sectors if structure == "H1" else ["pooled"]):
            basis = np.eye(len(selected))[positions[:, 0]]
            if structure == "H1" and unit_number: basis = basis + np.eye(len(selected))[positions[:, unit_number]]
            profiles.append(basis)
            unit = sample if sector == "pooled" else sample.loc[sample.sector_id.eq(sector)]
            detail = {**band_meta, "sector_id": sector, "sector": sector_names.get(sector, "Published national total" if len(sectors) == 1 else "Common equal-component slope"),
                "mean_y": float(unit[metadata["outcome"]].mean()), "sd_y": float(unit[metadata["outcome"]].std())}
            for effect, quarter, exposure_lag, vector, term in effect_contrasts(basis, model):
                row = {**detail, "effect": effect, "calendar_quarter": quarter, "weather_lag_years": exposure_lag, "term": term}
                if full_covariance is None:
                    row.update(coefficient=None, variance=None, std_error=None, ci_lower=None, ci_upper=None, p_value=None,
                               inference_status="Unavailable: complete design/time support", degenerate=None)
                else:
                    row.update(shared.inference(float(vector @ beta[selected]), float(vector @ covariance @ vector), degrees, float(vector @ hc0 @ vector)))
                books["temperature_effects"].append(row)
            for name, restriction in nulls(basis, model):
                bound = len(sectors) - 1 if method == "Sector cluster" else t - 2
                if full_covariance is None:
                    row = {**detail, "test": name, "status": "unavailable: complete design/time support", "restrictions": len(restriction),
                           "covariance_rank": None, "structural_rank_bound": bound, "p_value": None, "f_statistic": None}
                else:
                    row = subannual.test_record(beta[selected], covariance, hc0, restriction, detail, name, bound)
                books["tests"].append(row)
        if structure == "H1" and full_covariance is not None:
            for feature_number, feature in enumerate(heat):
                for unit_number, sector in enumerate(sectors):
                    vector = np.eye(len(selected))[positions[feature_number, unit_number]]
                    books["interaction_components"].append({**band_meta, "sector_id": sector, "term": names[components[feature_number, unit_number]],
                        "temperature_feature": feature, "component": "base" if unit_number == 0 else "interaction", "estimated": True, "normalization": False,
                        **shared.inference(float(vector @ beta[selected]), float(vector @ covariance @ vector), degrees, float(vector @ hc0 @ vector))})
                books["interaction_components"].append({**band_meta, "sector_id": sectors[0], "term": f"{feature}:C(sector_id)[T.{sectors[0]}]",
                    "temperature_feature": feature, "component": "interaction", "estimated": False, "normalization": True,
                    "coefficient": 0., "std_error": None, "p_value": None, "variance": None, "ci_lower": None, "ci_upper": None,
                    "inference_status": "Reference deviation normalized to zero, not estimated"})
            for name, restriction in [("All sector seasonal temperature response vectors equal", np.vstack([basis - profiles[0] for basis in profiles[1:]])),
                ("All sector sums of direct seasonal temperature coefficients equal", np.vstack([(basis - profiles[0]).sum(axis=0) for basis in profiles[1:]]))]:
                books["tests"].append(subannual.test_record(beta[selected], covariance, hc0, restriction, {**band_meta, "sector_id": "all"}, name, t - 2))
    archive.update(covariance_labels=np.asarray([item[0] for item in covariance_items], str),
        bandwidths=np.asarray([item[1] for item in covariance_items]), inference_df=np.asarray([item[2] for item in covariance_items]),
        corrections=np.asarray([item[3] if item[3] is not None else np.nan for item in covariance_items]))
    np.savez_compressed(OUT / "designs" / f"{spec_id}.npz", **archive)


def run_annual_seasonal_literature(*, preflight=False):
    OUT.mkdir(parents=True, exist_ok=True)
    producer_sha256 = shared.sha(Path(__file__))
    for folder in ["samples", "designs"]: (OUT / folder).mkdir(exist_ok=True)
    panels, provenance, weather_path = parent_histories()
    policy = json.loads(crisis.POLICY.read_text())
    source_files = sorted({name for record in provenance.values() for name in record["source_files"]} | {rel(weather_path), rel(crisis.POLICY), "data/processed/annual/annual_levels.csv"})
    input_hashes = {name: shared.sha(ROOT / name) for name in source_files}
    protected_names = ["scripts/crisis_sensitivity.py", "scripts/annual_crisis_sensitivity.py", "scripts/literature_sensitivity.py",
        "scripts/annual_literature_sensitivity.py",
        "outputs/literature_sensitivity/manifest.json", "outputs/literature_sensitivity/pyfixest/fit_log.csv",
        "outputs/literature_sensitivity/annual/manifest.json", "outputs/literature_sensitivity/annual/pyfixest/fit_log.csv",
        "outputs/pyfixest/fit_log.csv", "outputs/annual/pyfixest/fit_log.csv"]
    protected = {name: shared.sha(ROOT / name) for name in protected_names}
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    books = {name: [] for name in ["specifications", "coefficients", "temperature_effects", "interaction_components", "tests", "sample_inventory", "masks"]}
    previous = native.LOG_DIR, list(native._FIT_LOG), native._API_COUNTER, native._RUN_STARTED
    if not preflight: native.LOG_DIR = OUT / "pyfixest";native.reset_fit_log()
    receipt = {}
    try:
        with threadpool_limits(limits=1):
            for context, parent in panels.items():
                for history in HISTORIES:
                    for model in MODELS:
                        for outcome in OUTCOMES:
                            for variant in VARIANTS:
                                target_name = provenance[context]["matched_to"]
                                sample, masks, maximum_periods, eligibility = sample_frame(parent, model, history, outcome, variant, policy,
                                    panels[target_name] if target_name else None)
                                assert len(sample), (context, model, history, outcome, variant)
                                key = "__".join([context, history, model, outcome, variant])
                                path = OUT / "samples" / f"{key}.csv";sample.to_csv(path, index=False)
                                meta = {"context": context, "context_label": CONTEXT_LABELS[context], "history": history, "model_spec": model, "outcome": outcome,
                                    "outcome_label": "Annual log growth (" + ("NSA" if outcome == "y_nsa" else "SA") + " real output)",
                                    "output_adjustment": "NSA" if outcome == "y_nsa" else "SA", "sample_variant": variant,
                                    "sample_csv": rel(path), "sample_sha256": shared.sha(path), "parent_n_periods": parent.year.nunique(),
                                    "maximum_history_n_periods": maximum_periods, "n_periods": sample.year.nunique(),
                                    **eligibility, "matched_history_template": target_name or "",
                                    "matching_rule": "Intersect raw model/history-eligible dates with the named target eligibility before event masks; distinct from missing raw AR history" if target_name else "Unmatched original history retains every raw model/history-eligible year",
                                    "history_loss_periods": parent.year.nunique() - maximum_periods,
                                    "matched_to": provenance[context]["matched_to"], "includes_dwellings": provenance[context]["includes_dwellings"],
                                    "weather_history_csv": rel(weather_path), "policy_id": policy["policy_id"],
                                    "strict_rule": "All output years t−2,t−1,t avoid fixed event blocks; same support rule for all four seasonal specifications",
                                    "growth_and_lags": "Original complete annual outcome transformations; AR1 from full original annual-output growth registry before weather/context cropping and actual year predecessor",
                                    "adaptation": "Singapore real output and calendar Q1–Q4; quarterly mean air temperature rather than paper station/season construction; quarter rainfall totals and current RH added; sector intercepts+common trend, no year FE"}
                                books["sample_inventory"].append(meta)
                                books["masks"].extend({**{key: meta[key] for key in ["context", "history", "model_spec", "outcome", "sample_variant"]}, **row} for row in masks)
                                for structure in ["common"] + (["H1"] if sample.sector_id.nunique() > 1 else []):
                                    fit_spec(sample, meta, model, structure, books, preflight=preflight)
                    print(f"Annual seasonal {'preflight' if preflight else 'fit'} {context}/{history}", flush=True)
            if not preflight: receipt = native.write_fit_log()
    finally:
        if not preflight:
            native.LOG_DIR, old, native._API_COUNTER, native._RUN_STARTED = previous;native._FIT_LOG[:] = old
    assert all(shared.sha(ROOT / name) == value for name, value in input_hashes.items()), "Frozen analytical input changed"
    assert all(shared.sha(ROOT / name) == value for name, value in protected.items()), "Prior analysis/reference/receipt changed"
    assert shared.sha(Path(__file__)) == producer_sha256, "Producer source changed during the run"
    specs = pd.DataFrame(books["specifications"])
    if not preflight:
        effects = pd.DataFrame(books["temperature_effects"])
        adjusted_all = np.full(len(effects), np.nan)
        probability_array = effects.p_value.to_numpy(dtype=float)
        for _, indexes in effects.loc[effects.model.eq("H1")].groupby(["context", "history", "model_spec", "outcome", "sample_variant", "covariance", "hac_lags", "effect", "calendar_quarter", "weather_lag_years"]).groups.items():
            indexes = np.asarray(indexes, dtype=int)
            probabilities = probability_array[indexes]
            valid = np.isfinite(probabilities)
            adjusted = holm_probabilities(np.where(valid, probabilities, 1.))
            adjusted_all[indexes] = np.where(valid, adjusted, np.nan)
        effects["p_value_holm_within_sector_family"] = adjusted_all
        books["temperature_effects"] = effects.to_dict("records")
    for name, rows in books.items():
        filename = "preflight_specifications" if preflight and name == "specifications" else name
        if not preflight or name in ["specifications", "sample_inventory", "masks"]:
            pd.DataFrame(rows).to_csv(OUT / f"{filename}.csv", index=False)
    manifest = {"status": "preflight_only" if preflight else "estimated_not_yet_independently_verified",
        "contexts": len(panels), "requested_specs": len(specs), "identified_specs": int(specs.status.eq("estimated").sum()),
        "unavailable_specs": int(specs.status.ne("estimated").sum()), "native_fits": receipt.get("native_fits_recorded", 0),
        "row_counts": {name: len(rows) for name, rows in books.items()}, "inputs_sha256": input_hashes,
        "producer_source": rel(Path(__file__)), "producer_source_sha256": producer_sha256,
        "protected_sources_sha256": protected, "models": MODELS, "histories": HISTORIES, "outcomes": OUTCOMES,
        "strict_output_support": "t−2 through t inclusive for every model and history set; broader than non-AR annual crisis companion",
        "weather_history": "Current/prior Q1–Q4 features from the complete broadest national quarterly calendar, formed before masks",
        "maximum_history": "matched_max_AR1_history requires all current/prior weather and calendar Ylag1, shared across all four specifications within outcome/context",
        "own_history": "Model-own finite regressors; separately labelled, not a same-sample comparison",
        "time_df_guard": "Maximum rank of the full design restricted to any one sector; no fit when T−rank≤0 even if cross-sector N is larger",
        "covariance": "Full nuisance-inclusive VCV saved per design: common panel CR1 + calendar DK1/2; national/H1 calendar HAC/DK1/2, actual missing years preserved",
        "interpretation": "Seasonal conditional associations, not a replication. Current/prior sums include covariance; AR sums are not unconditional long-run multipliers. No stationarity or causal clearance.",
        "Holm_family": "Within displayed H1 sector family for the same outcome/model/history/sample/covariance/effect; unavailable requested sectors retain family slots; not a grid-wide adjustment",
        "native_ledger": rel(OUT / "pyfixest"), "source_methods": "notes/temperature_horizon_review.md and the frozen annual-seasonal/v2 history-alignment protocols",
        "protocols_sha256": {name: shared.sha(ROOT / name) for name in PROTOCOLS}}
    target = OUT / ("preflight.json" if preflight else "manifest.json")
    target.write_text(json.dumps(shared.clean(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: manifest[key] for key in ["status", "requested_specs", "identified_specs", "unavailable_specs", "native_fits"]}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    run_annual_seasonal_literature(preflight=parser.parse_args().preflight)
