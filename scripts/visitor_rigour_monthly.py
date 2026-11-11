"""Monthly visitor-arrival timing and origin-heterogeneity sensitivities.

Frozen inputs only. Native PyFixest supplies every fitted coefficient/residual.
This companion does not write the preserved visitor or economy analyses.
Calendar t/F references are working approximations, not calibrated exact laws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import crisis_sensitivity as math_shared
import literature_sensitivity as calendar_math
import literature_full_controls as restriction_math
import pyfixest_backend as native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/visitor_rigour/monthly"
PROTOCOL = ROOT / "notes/visitor_rigour_protocol.md"
POLICY = ROOT / "notes/crisis_sensitivity_policy.json"
PANELS = {"aggregate": "Official total arrivals", "region": "Nine selected regions", "country": "42 named markets"}
OUTCOMES = {"y_mom_nsa": (1, 3, "MoM log arrivals growth (NSA; not annualised)"),
            "y_yoy_nsa": (12, 12, "YoY log arrivals growth (NSA)")}
VARIANTS = math_shared.VARIANTS
OTHER = ["rh_pct", "lny_window_days", "lny_lag1"]
ADDED = ["wind_mps", "pm25_10ug"]
ZERO, RANK = 1e-12, 1e-10


def relative(path): return str(Path(path).resolve().relative_to(ROOT))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(math_shared.clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def rank_scaled(matrix):
    matrix = np.asarray(matrix, float)
    scales = np.linalg.norm(matrix, axis=0)
    singular = np.linalg.svd(matrix / np.where(scales > 0, scales, 1.), compute_uv=False)
    threshold = float(singular[0] * max(matrix.shape) * np.finfo(float).eps)
    return int(np.sum(singular > threshold)), threshold


def load_inputs():
    """Official ID/count registry and full weather histories, with no fetch/writes."""
    raw = ROOT / "data/raw/visitor_arrivals"
    data = ROOT / "data/processed/visitor_arrivals"
    metadata = json.loads((raw / "nsa_singstat_metadata.json").read_text())["Data"]["records"]
    catalog = pd.DataFrame(metadata["row"])[["seriesNo", "rowText"]].rename(columns={"seriesNo": "unit_id", "rowText": "unit"})
    catalog.unit_id = catalog.unit_id.astype(str)
    catalog["panel"] = np.select([catalog.unit_id.eq("1"), catalog.unit_id.str.count(r"\.").eq(1) & ~catalog.unit_id.eq("1.999"),
        catalog.unit_id.str.count(r"\.").eq(2) & ~catalog.unit_id.str.endswith(".999")], list(PANELS), default="residual")
    source = pd.read_csv(raw / "nsa.csv")
    source.DataSeries = source.DataSeries.str.strip()
    assert source.DataSeries.is_unique and set(source.DataSeries) == set(catalog.unit)
    counts = source.set_index("DataSeries").T
    counts.index = pd.PeriodIndex(pd.to_datetime(counts.index, format="%Y%b"), freq="M", name="month")
    counts = counts.sort_index().apply(pd.to_numeric, errors="raise")
    assert counts.index.equals(pd.period_range(counts.index.min(), counts.index.max(), freq="M", name="month"))
    assert counts.notna().all().all() and counts.ge(0).all().all()
    frozen = pd.read_csv(data / "all_source_counts.csv", index_col="month")
    frozen.index = pd.PeriodIndex(frozen.index, freq="M", name="month")
    pd.testing.assert_frame_equal(counts, frozen, check_dtype=False, check_names=False)
    registry, levels = {}, {}
    for panel in PANELS:
        chosen = catalog.loc[catalog.panel.eq(panel)].sort_values("unit_id")
        values = counts[chosen.unit].rename(columns=chosen.set_index("unit").unit_id.to_dict())
        levels[panel] = values
        logs = np.log(values.where(values.gt(0)))
        registry[panel] = {}
        for outcome, (horizon, _, _) in OUTCOMES.items():
            registry[panel][outcome] = 100 * (logs - logs.reindex(logs.index - horizon).set_axis(logs.index))
    assert [len(levels[panel].columns) for panel in PANELS] == [1, 9, 42]
    weather_parts, inputs = [], [raw / "nsa.csv", raw / "nsa_singstat_metadata.json", data / "all_source_counts.csv"]
    for stem, old, new in [("temperature", "mean_temp", "temp_c"), ("humidity", "mean_rh", "rh_pct"),
                           ("rainfall", "total_rainfall", "rain_mm")]:
        path = ROOT / f"data/raw/{stem}_monthly.csv"
        part = pd.read_csv(path, index_col="month").rename(columns={old: new})
        part.index = pd.PeriodIndex(part.index, freq="M")
        assert part.index.is_unique
        weather_parts.append(part[[new]]);inputs.append(path)
    weather = pd.concat(weather_parts, axis=1).sort_index()
    weather["rain_100mm"] = weather.rain_mm / 100
    lunar_path = data / "lunar_new_year.csv"
    lunar = pd.read_csv(lunar_path, index_col="month")
    lunar.index = pd.PeriodIndex(lunar.index, freq="M")
    supplementary = ROOT / "replication/extensions/supplementary_inputs/calendar"
    anchor_path, anchor_sources = supplementary / "T2007e.txt", supplementary / "sources.json"
    assert sha(anchor_path) == "2eeec2ef316267d92c2ec93e44d2ccb3a889e02c408aecdcc696a2c350416673"
    matches = [value.decode("ascii") for value in re.findall(rb"(2007/\d{2}/\d{2})\s+1st Lunar month", anchor_path.read_bytes())]
    assert len(matches) == 1 and matches[0] == "2007/02/18"
    anchors = sorted(set(pd.to_datetime(lunar.lunar_new_year_date).tolist() + [pd.Timestamp(matches[0])]))
    calendar_index = pd.period_range("2007-01", max(lunar.index.max(), counts.index.max()), freq="M")
    lunar_days = pd.Series(0, index=calendar_index, dtype=int)
    for anchor in anchors:
        for date in pd.date_range(anchor - pd.Timedelta(days=7), anchor + pd.Timedelta(days=7), freq="D"):
            month = date.to_period("M")
            if month in lunar_days.index: lunar_days.loc[month] += 1
    assert np.array_equal(lunar_days.reindex(lunar.index), lunar.lny_window_days)
    assert lunar_days.loc["2007-12"] == 0
    weather["lny_window_days"] = lunar_days.reindex(weather.index)
    weather["lny_lag1"] = lunar_days.reindex(weather.index - 1).set_axis(weather.index)
    controls_path = ROOT / "data/processed/extended_controls/monthly_controls.csv"
    controls = pd.read_csv(controls_path, index_col="month")
    controls.index = pd.PeriodIndex(controls.index, freq="M")
    assert controls.index.is_unique
    assert np.array_equal(controls.eligible_90, controls.wind_coverage.ge(.9) & controls.pm_coverage.ge(.9))
    for term in ADDED + ["eligible_90", "wind_coverage", "pm_coverage"]:
        weather[term] = controls[term].reindex(weather.index)
    inputs += [lunar_path, controls_path, anchor_path, anchor_sources]
    checks = {"status": "source_reconstruction_passed", "calendar": "15-day Lunar New Year window, −7..+7 days inclusive",
        "unchanged_old_monthly_LNY_values": len(lunar), "maximum_old_monthly_LNY_difference": 0,
        "supplemental_2007_anchor": str(pd.Timestamp(matches[0]).date()), "supplemental_December_2007_window_days": 0,
        "January_2008_LNY_lag1": float(weather.loc[pd.Period("2008-01", freq="M"), "lny_lag1"]),
        "new_input_scope": "One newly packaged official HKO 2007 calendar; no prior raw/source file refreshed or rewritten"}
    return counts.index, levels, registry, weather, catalog, {relative(path): sha(path) for path in inputs}, checks


def construct_samples(dates, levels, registry, weather, catalog, panel, outcome, control_set, history, policy):
    growth_horizon, horizon, outcome_label = OUTCOMES[outcome]
    maximum = history == "matched_maximum_growth_history"
    lag_window = horizon if maximum else 0
    weather_basis = weather.reindex(dates).copy()
    for lag in range(horizon + 1):
        for term, stem in [("temp_c", "temp"), ("rain_100mm", "rain")]:
            weather_basis[f"{stem}_l{lag}"] = weather[term].reindex(dates - lag).to_numpy()
    own = registry[panel][outcome]
    current_output = own.notna().all(axis=1)
    own_history = current_output.copy()
    history_valid = {}
    for cut in PANELS:
        valid = registry[cut][outcome].notna().all(axis=1)
        if maximum:
            for lag in range(1, horizon + 1):
                valid &= registry[cut][outcome].reindex(dates - lag).notna().all(axis=1).set_axis(dates)
        history_valid[cut] = valid
    own_history &= history_valid[panel]
    weather_terms = [f"{stem}_l{lag}" for stem in ["temp", "rain"] for lag in range(lag_window + 1)] + OTHER
    weather_finite = pd.Series(np.isfinite(weather_basis[weather_terms]).all(axis=1), index=dates)
    stage_weather = own_history & weather_finite
    shared_valid = pd.concat(history_valid, axis=1).all(axis=1)
    stage_alignment = stage_weather & shared_valid
    finite_added = pd.Series(np.isfinite(weather_basis[ADDED]).all(axis=1), index=dates) if control_set == "full" else pd.Series(True, index=dates)
    stage_control = stage_alignment & finite_added
    coverage = weather_basis.eligible_90.fillna(False).astype(bool) if control_set == "full" else pd.Series(True, index=dates)
    eligible = stage_control & coverage
    support = (horizon + growth_horizon) if maximum else 12
    intervals = [(row["event_id"], pd.Period(row["start"], freq="M"), pd.Period(row["end"], freq="M")) for row in policy["events"]]
    mask_rows = []
    for date in dates:
        current = [name for name, start, end in intervals if start <= date <= end]
        overlapping = [name for name, start, end in intervals if date - support <= end and date >= start]
        mask_rows.append({"panel": panel, "outcome": outcome, "control_set": control_set, "history": history,
            "month": str(date), "time": date.ordinal, "current_output_finite": bool(current_output.loc[date]),
            "own_output_history_finite": bool(own_history.loc[date]), "weather_history_finite": bool(weather_finite.loc[date]),
            "all_three_cuts_output_history_finite": bool(shared_valid.loc[date]), "current_added_controls_finite": bool(finite_added.loc[date]),
            "current_control_coverage_pass": bool(coverage.loc[date]), "eligible_before_event_masks": bool(eligible.loc[date]),
            "event_ids_at_t": json.dumps(current), "event_ids_in_output_support": json.dumps(overlapping),
            "output_support_start": str(date - support), "output_support_horizon": support,
            "retain_event_dates": not current, "retain_strict": not overlapping})
    masks = pd.DataFrame(mask_rows)
    names = catalog.set_index("unit_id").unit.to_dict()
    rows = []
    for unit in sorted(levels[panel].columns):
        frame = weather_basis.copy()
        frame["month"], frame["time"], frame["season"], frame["year"] = dates.astype(str), dates.asi8, dates.month, dates.year
        frame["unit_id"], frame["unit"], frame["sector_id"], frame["sector"] = unit, names[unit], unit, names[unit]
        frame["arrivals"] = levels[panel][unit]
        for y in OUTCOMES: frame[y] = registry[panel][y][unit]
        for lag in range(1, horizon + 1): frame[f"ar_l{lag}"] = registry[panel][outcome][unit].reindex(dates - lag).to_numpy()
        frame["temp_endpoint_difference"] = frame.temp_l0 - frame[f"temp_l{horizon}"]
        frame["temp_growth_sum"] = frame[[f"temp_l{lag}" for lag in range(horizon)]].sum(axis=1, min_count=horizon)
        frame["rain_endpoint_difference"] = frame.rain_l0 - frame[f"rain_l{horizon}"]
        frame["rain_growth_sum"] = frame[[f"rain_l{lag}" for lag in range(horizon)]].sum(axis=1, min_count=horizon)
        rows.append(frame.reset_index(drop=True))
    parent = pd.concat(rows, ignore_index=True).sort_values(["time", "unit_id"]).reset_index(drop=True)
    base = {"panel": panel, "context": panel, "context_label": PANELS[panel], "frequency": "monthly", "outcome": outcome,
        "outcome_label": outcome_label, "output_adjustment": "NSA", "growth_horizon": growth_horizon, "horizon": horizon,
        "control_set": control_set, "history": history, "parent_n_periods": len(dates), "parent_n_obs": len(parent),
        "current_own_output_n_periods": int(current_output.sum()), "own_output_history_n_periods": int(own_history.sum()),
        "output_history_loss_periods": int((~own_history).sum()), "weather_history_loss_periods": int(own_history.sum() - stage_weather.sum()),
        "shared_panel_alignment_loss_periods": int(stage_weather.sum() - stage_alignment.sum()),
        "control_raw_missing_loss_periods": int(stage_alignment.sum() - stage_control.sum()),
        "control_coverage_loss_periods": int(stage_control.sum() - eligible.sum()),
        "maximum_history_n_periods": int(eligible.sum()), "output_support_horizon": support,
        "positive_calendar_scope": "Common finite output/history calendar across official total, nine selected regions and 42 named markets",
        "history_scope": "Full raw NSA arrival growth and actual month lookup before panel alignment, current-control eligibility or exclusions",
        "control_history_scope": "Full temperature/rain history; current RH/wind/PM/LNY plus prior-month LNY. Historical PM is not required",
        "coverage_rule": "Current wind and PM coverage each at least 90%" if control_set == "full" else "No wind or PM controls",
        "partition_scope": "Official total" if panel == "aggregate" else "Selected nonexhaustive origin partition; equal-origin coefficients are not national count effects",
        "strict_rule": "Common YoY-safe output support t−12..t" if not maximum else f"Full dynamic output support t−{support}..t",
        "stationarity_status": "Not cleared: marginal diagnostics separate; gaps and short histories unresolved; no pretest selection"}
    assert len(dates) - sum(base[key] for key in ["output_history_loss_periods", "weather_history_loss_periods", "shared_panel_alignment_loss_periods", "control_raw_missing_loss_periods", "control_coverage_loss_periods"]) == int(eligible.sum())
    samples = []
    for variant in VARIANTS:
        keep = eligible.to_numpy().copy()
        if variant != "all_eligible": keep &= masks["retain_event_dates" if variant == "event_dates_excluded" else "retain_strict"].to_numpy()
        sample = parent.loc[parent.time.isin(dates.asi8[keep])].copy().reset_index(drop=True)
        assert not sample.empty and len(sample) == sample.time.nunique() * len(levels[panel].columns)
        samples.append((variant, sample, {**base, "sample_variant": variant, "crisis_loss_periods": int(eligible.sum() - keep.sum())}))
    return samples, masks


def feature_inventory(model_spec, horizon, control_set):
    lags = [0] if model_spec in {"own_current", "benchmark"} else list(range(horizon + 1))
    temperatures = [f"temp_l{lag}" for lag in lags]
    controls = [f"rain_l{lag}" for lag in lags]
    if model_spec == "two_channel":
        temperatures = ["temp_endpoint_difference", "temp_growth_sum"]
    controls += OTHER + (ADDED if control_set == "full" else [])
    ar = [f"ar_l{lag}" for lag in range(1, horizon + 1)] if model_spec == "dynamic" else []
    return temperatures, controls + ar, ar, lags


def design(sample, model_spec, model, horizon, control_set):
    units, seasons, years = sorted(sample.unit_id.unique()), sorted(sample.season.unique()), sorted(sample.year.unique())
    indicator = np.column_stack([sample.unit_id.eq(unit) for unit in units]).astype(float)
    seasonal = np.column_stack([sample.season.eq(season) for season in seasons]).astype(float)
    fixed = (indicator[:, :, None] * seasonal[:, None, :]).reshape(len(sample), -1)
    names = [f"origin_season[{unit},{season}]" for unit in units for season in seasons]
    if len(years) > 1:
        fixed = np.column_stack([fixed, np.column_stack([sample.year.eq(year) for year in years[1:]])])
        names += [f"year[{year}]" for year in years[1:]]
    temperatures, controls, ar, lags = feature_inventory(model_spec, horizon, control_set)
    first = len(names);blocks = [fixed, (sample[controls] - sample[controls].mean()).to_numpy()]
    names += controls;selected = list(range(first, len(names)));components = []
    for term in temperatures:
        values = (sample[term] - sample[term].mean()).to_numpy();start = len(names)
        if model == "H1":
            blocks.append(np.column_stack([values, indicator[:, 1:] * values[:, None]]))
            names += [term] + [f"{term}:C(unit_id)[T.{unit}]" for unit in units[1:]]
        else: blocks.append(values[:, None]);names.append(term)
        components.append(list(range(start, len(names))));selected.extend(range(start, len(names)))
    matrix = np.column_stack(blocks)
    return matrix, names, np.asarray(selected), np.asarray(components), units, temperatures, controls, ar, lags


def temporal_ranks(sample, matrix):
    """Exact local X spans; max local rank is a declared working correction."""
    units = sorted(sample.unit_id.unique())
    blocks, ranks, cutoffs = [], {}, {}
    for unit in units:
        local = matrix[sample.unit_id.eq(unit).to_numpy()]
        # Removing exactly-zero columns preserves every nonzero singular value.
        # Retain the original full local shape in the declared cutoff.
        block = local[:, np.any(local != 0, axis=0)];blocks.append(block)
        scales = np.linalg.norm(block, axis=0)
        singular = np.linalg.svd(block / scales, compute_uv=False)
        cutoff = float(singular[0] * max(local.shape) * np.finfo(float).eps)
        ranks[unit], cutoffs[unit] = int(np.sum(singular > cutoff)), cutoff
    fullspan, fullcutoff = rank_scaled(np.column_stack(blocks))
    return {"time_design_rank": max(ranks.values()), "local_time_design_ranks": json.dumps(ranks, sort_keys=True),
        "local_time_rank_thresholds": json.dumps(cutoffs, sort_keys=True), "full_cross_origin_temporal_span_rank": fullspan,
        "full_cross_origin_temporal_span_rank_threshold": fullcutoff,
        "time_rank_rule": "K_time=max exact per-origin local X rank; reference invariant. Full cross-origin temporal-span rank is diagnostic, not a native-fit guard"}


def inference(beta, covariance, hc0, vector, metadata, unavailable_reason=""):
    estimate = None if beta is None else float(vector @ beta)
    if beta is None or unavailable_reason:
        return {**metadata, "coefficient": estimate, "variance": None, "std_error": None, "ci_lower": None, "ci_upper": None,
            "p_value": None, "degenerate": False, "inference_status": "unavailable: " + (unavailable_reason or "native point design")}
    return {**metadata, **math_shared.inference(estimate, float(vector @ covariance @ vector), metadata["inference_df"], float(vector @ hc0 @ vector))}


def wald(beta, covariance, hc0, restriction, metadata, name, unavailable_reason=""):
    working = {**metadata, "covariance": "Sector cluster" if metadata["covariance"] == "Origin cluster" else metadata["covariance"]}
    if beta is None or unavailable_reason:
        result = restriction_math.unavailable_test(working, name, restriction)
        result["status"] = "unavailable: " + (unavailable_reason or "native point design")
    else: result = restriction_math.wald(beta, covariance, hc0, restriction, working, name)
    result["covariance"] = metadata["covariance"]
    result["calibration_status"] = "Not established; nominal asymptotic diagnostic" if len(np.atleast_2d(restriction)) > 1 else "Nominal approximate conditional inference"
    return result


def restrictions(length, outcome, dynamic):
    eye = np.eye(length)
    result = [("All temperature-lag coefficients zero", eye), ("Cumulative direct temperature response zero", np.ones((1, length)))]
    if length > 1:
        endpoint = length - 1 if outcome == "y_yoy_nsa" else 1
        shape = np.vstack([eye[j] for j in range(1, length) if j != endpoint] + [eye[0] + eye[endpoint]])
        result.append(("Conditional direct-weather level-channel shape; not exact structural static level with AR" if dynamic else "Exact contemporaneous static-output-level coefficient restriction", shape))
        if outcome == "y_yoy_nsa":
            result.append(("Conditional direct-weather equal-growth coefficient shape; growth lags held fixed" if dynamic else "Equal one-period growth response over YoY accumulation window", np.vstack([eye[j] - eye[0] for j in range(1, length - 1)] + [eye[-1]])))
            result.append(("Conditional direct-weather response lies in endpoint-difference plus growth-sum span; growth lags held fixed" if dynamic else "Unrestricted YoY response lies in endpoint-difference plus growth-sum span",
                np.vstack([eye[j] - eye[1] for j in range(2, length - 1)] + [eye[0] + eye[-1] - eye[1]])))
    return result


def add_holm(rows):
    """All requested origin total effects for one feature/sum, unavailable slots kept."""
    families = {}
    for row in rows: families.setdefault((row["spec_id"], row["covariance"], row["hac_lags"], row["term"], row["effect"]), []).append(row)
    for family in families.values():
        p = np.asarray([row["p_value"] if row["p_value"] is not None else 1. for row in family]);order = np.argsort(p, kind="stable")
        adjusted = np.empty(len(p));adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
        for row, value in zip(family, adjusted, strict=True):
            row.update(p_value_holm_profile=float(value) if row["p_value"] is not None else None,
                multiplicity_family_size=len(p), multiplicity_family="All requested origin total effects for this single feature/sum, spec and covariance; unavailable slots retained; does not adjust across the research grid")


def fit_spec(sample, base, model_spec, model, output, records, preflight=False):
    matrix, names, selected, components, units, features, controls, ar, lags = design(sample, model_spec, model, base["horizon"], base["control_set"])
    geometry = math_shared.geometry(matrix)
    n, k, t, g = len(sample), matrix.shape[1], sample.time.nunique(), len(units)
    temporal = temporal_ranks(sample, matrix)
    spec_id = "__".join([base[key] for key in ["panel", "outcome", "control_set", "sample_variant"]] + [model, model_spec])
    label = "Visitor monthly rigour " + spec_id
    meta = {**base, **temporal, "spec_id": spec_id, "native_label": label, "model": model, "model_spec": model_spec,
        "temp_features": json.dumps(features), "control_features": json.dumps([term for term in controls if term not in ar]), "ar_features": json.dumps(ar),
        "controls": json.dumps(controls), "n_obs": n, "n_periods": t, "n_sectors": g, "n_origins": g, "n_parameters": k, "residual_df": n - k,
        "design_rank": geometry["design_rank"], "design_rank_threshold": geometry["design_rank_threshold"],
        "design_condition_number": geometry["design_condition_number"], "scaled_design_condition_number": geometry["scaled_design_condition_number"],
        "time_residual_df": t - temporal["time_design_rank"], "status": geometry["status"], "fit_id": None,
        "start": sample.month.min(), "end": sample.month.max(), "calendar_gaps": int(sample.time.max() - sample.time.min() + 1 - t),
        "mean_y": float(sample[base["outcome"]].mean()), "sd_y": float(sample[base["outcome"]].std()),
        "reference_sector_id": units[0] if model == "H1" else "", "reference_unit_id": units[0] if model == "H1" else "",
        "formula": base["outcome"] + " ~ " + " + ".join(features) + (" main + origin interactions" if model == "H1" else "") + " + " + " + ".join(controls) + " | origin×calendar-month FE + common year FE",
        "design_normalization": "All origin×observed-month cells; omit first observed year; center continuous variables on exact pooled sample; main temperature plus nonreference deviations",
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
        fitted = native.fit_many_matrix(sample[[base["outcome"]]].to_numpy(), matrix, names, label, outcome_names=[base["outcome"]])[0]
        beta, residual = fitted.params.to_numpy(), fitted.resid
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


def run_visitor_rigour_monthly(*, preflight=False):
    output = OUT / "preflight" if preflight else OUT
    for child in [output, output / "samples", output / "designs"]: child.mkdir(parents=True, exist_ok=True)
    policy = math_shared.load_policy(POLICY)
    dates, levels, registry, weather, catalog, raw_hashes, source_checks = load_inputs()
    save_json(output / "source_checks.json", source_checks)
    protected_names = ["scripts/pyfixest_backend.py", "scripts/crisis_sensitivity.py", "scripts/literature_sensitivity.py", "scripts/literature_full_controls.py",
        "outputs/visitor_arrivals/regression_results.csv", "outputs/visitor_arrivals/pyfixest/fit_log.csv", "outputs/visitor_arrivals/pyfixest/estimator_manifest.json",
        "outputs/visitor_arrivals/crisis/regression_results.csv", "outputs/visitor_arrivals/crisis/pyfixest/fit_log.csv", "outputs/pyfixest/fit_log.csv",
        relative(POLICY), relative(PROTOCOL), relative(Path(__file__))]
    protected = {name: sha(ROOT / name) for name in protected_names}
    save_json(output / "policy.json", policy)
    records = {name: [] for name in ["specifications", "coefficients", "temperature_effects", "channel_estimates", "interaction_components", "control_estimates", "tests"]}
    inventories, masks = [], []
    previous = native.LOG_DIR, list(native._FIT_LOG), native._API_COUNTER, native._RUN_STARTED
    receipt = {}
    if not preflight: native.LOG_DIR = output / "pyfixest";native.reset_fit_log()
    save_json(output / "manifest.json", {"status": "running_preflight" if preflight else "running", "claim": "No completed-result or verification claim"})
    try:
        with threadpool_limits(limits=1):
            for panel in PANELS:
                for outcome in OUTCOMES:
                    for control_set in ["long", "full"]:
                        for history in ["own_current_history", "matched_maximum_growth_history"]:
                            samples, mask = construct_samples(dates, levels, registry, weather, catalog, panel, outcome, control_set, history, policy)
                            masks.extend(mask.to_dict("records"))
                            for variant, sample, base in samples:
                                sample_id = "__".join([panel, outcome, control_set, history, variant])
                                sample_path = output / "samples" / f"{sample_id}.csv";sample.to_csv(sample_path, index=False)
                                base.update(sample_id=sample_id, sample_csv=relative(sample_path), sample_sha256=sha(sample_path),
                                    policy_id=policy["policy_id"], policy_source_sha256=protected[relative(POLICY)], protocol_source_sha256=protected[relative(PROTOCOL)])
                                inventories.append({**base, "n_obs": len(sample), "n_periods": sample.time.nunique(), "n_sectors": sample.unit_id.nunique(),
                                    "start": sample.month.min(), "end": sample.month.max()})
                                specifications = ["own_current"] if history == "own_current_history" else ["benchmark", "distributed", "dynamic"] + (["two_channel"] if outcome == "y_yoy_nsa" else [])
                                for specification in specifications:
                                    for model in ["common"] + (["H1"] if panel != "aggregate" else []):
                                        meta = fit_spec(sample, base, specification, model, output, records, preflight=preflight)
                                        print(f"Visitor monthly {'preflight' if preflight else 'fit'} {meta['spec_id']}: G{meta['n_sectors']} T{meta['n_periods']} K{meta['n_parameters']}, {meta['status']}, timeDF{meta['time_residual_df']}", flush=True)
            if not preflight: receipt = native.write_fit_log()
    finally:
        if not preflight:
            native.LOG_DIR, old, native._API_COUNTER, native._RUN_STARTED = previous;native._FIT_LOG[:] = old
    assert all(sha(ROOT / path) == value for path, value in protected.items()), "Protected analytical source changed during companion run"
    assert all(sha(ROOT / path) == value for path, value in raw_hashes.items()), "Frozen visitor/weather input changed"
    add_holm(records["temperature_effects"]);add_holm(records["channel_estimates"])
    for name, rows in records.items(): pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(inventories).to_csv(output / "sample_inventory.csv", index=False)
    pd.DataFrame(masks).to_csv(output / "masks.csv", index=False)
    specs = pd.DataFrame(records["specifications"])
    assert len(specs) == 270 and len(inventories) == 72 and specs.spec_id.is_unique
    manifest = {"status": "preflight_only" if preflight else "computed_not_yet_independently_verified", "requested_specs": len(specs),
        "native_fits": receipt.get("native_fits_recorded", 0), "native_point_estimable_requests": int(specs.status.eq("estimated").sum()),
        "point_unavailable_requests": int(specs.status.ne("estimated").sum()), "primary_time_correction_unavailable_requests": int((~specs.time_correction_available).sum()),
        "samples": len(inventories), "row_counts": {name: len(rows) for name, rows in records.items()}, "source_sha256": protected, "raw_inputs_sha256": raw_hashes,
        "source_months": len(dates), "source_units_per_panel": {panel: len(value.columns) for panel, value in levels.items()},
        "source_zero_cells_per_panel": {panel: int(value.eq(0).sum().sum()) for panel, value in levels.items()},
        "zero_handling": "Actual reported zeros remain in the source. Log growth/AR requires positive source levels; common finite calendar losses are explicit; no +1, interpolation, unit deletion or imputed growth",
        "models": "Own-current plus max-growth-history matched benchmark, DL0..H, DL+shared AR1..H, and YoY joint endpoint/growth-sum temperature channels conditional on unrestricted rain0..H",
        "inference": "Primary actual-calendar Bartlett HAC12/24, correction T/(T−max local X rank), approximate t(T−1); pooled-origin CR1 sensitivity. Native estimability and temporal correction availability are separate",
        "not_claimed": "Causal identification, exact finite-sample HAC/Wald calibration, stationarity, exhaustive selected partitions or dynamic FE bias correction",
        "full_covariance_storage": "designs/{spec_id}.npz full nuisance-inclusive matrices, unadjusted calendar matrices, full native X/Y/beta/resid and exact row calendars",
        "native_ledger": relative(output / "pyfixest"), "method_helpers": "Frozen shared design geometry/inference, calendar score Bartlett covariance and rank-safe restriction math; no prior producer invoked",
        "strict_masks": "Own-current: t−12..t for both outcomes. Matched maximum-history: MoM t−4..t; YoY t−24..t. Growth and raw AR formed before all masks",
        "holms": "Within a spec/covariance and one feature or cumulative effect across requested origins; unavailable slots retained. No grid-wide error control",
        "numerical_thresholds": {"covariance_zero_relative_HC0": ZERO, "covariance_rank_relative": RANK}}
    save_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["status", "requested_specs", "native_fits", "point_unavailable_requests", "primary_time_correction_unavailable_requests", "samples"]}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Write samples and full-rank/time-rank inventory; no native fits or old-output writes")
    run_visitor_rigour_monthly(preflight=parser.parse_args().preflight)
