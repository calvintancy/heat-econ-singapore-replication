"""Additional subannual distributed-weather and dynamic FE sensitivities.

This is a post-exploration Singapore adaptation, not a paper replication. Frozen
parents fix current-outcome eligibility; full raw official output supplies AR
histories before weather/context cropping, matching and fixed event masks.
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
import pyfixest_backend as native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/literature_sensitivity"
OUTCOMES = ["y_short", "y_yoy_nsa", "y_yoy"]
SPECIFICATIONS = ["benchmark", "distributed", "dynamic"]
VARIANTS = shared.VARIANTS
ZERO, RANK = 1e-12, 1e-10


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def weather_history(context: shared.Context) -> tuple[pd.DataFrame, Path]:
    path = ROOT / f"data/processed/weather_{context.frequency}.csv"
    frame = pd.read_csv(path).rename(columns={"temperature": "temp_c", "humidity": "rh_pct", "rainfall": "rain_mm"})
    frame["rain_100mm"] = frame.rain_mm / 100
    frame.index = pd.PeriodIndex(frame[context.period], freq=context.freq)
    assert frame.index.is_unique
    assert list(frame.index) == list(pd.period_range(frame.index.min(), frame.index.max(), freq=context.freq))
    return frame.sort_index(), path


@lru_cache(maxsize=None)
def raw_growth_registry(family: str, freq: str) -> pd.DataFrame:
    """Full official output history, before weather/taxonomy/sample cropping."""
    if family == "va":
        from build_panel import parse_va
        period = "quarter"
        sa = parse_va("va_sa").rename(columns={"va_sa": "output_sa"})
        nsa = parse_va("va_nsa").rename(columns={"va_nsa": "output_nsa"})
    else:
        from iip_analysis import source_frame
        period = "month"
        sa = source_frame("iip_sa", "M").rename(columns={"iip_sa": "output_sa"})
        nsa = source_frame("iip_nsa", "M").rename(columns={"iip_nsa": "output_nsa"})
    frame = sa.merge(nsa[["sector_id", period, "output_nsa"]], on=["sector_id", period], how="left", validate="one_to_one")
    if family == "iip" and freq == "Q":
        frame["quarter"] = frame.month.dt.asfreq("Q")
        frame = frame.groupby(["sector_id", "quarter"], as_index=False).agg(output_sa=("output_sa", "mean"),
            output_nsa=("output_nsa", "mean"), months=("month", "size"), nsa_months=("output_nsa", "count"))
        frame = frame.loc[frame.months.eq(3)].copy()
        frame.loc[frame.nsa_months.ne(3), "output_nsa"] = np.nan
        period = "quarter"
    for field in ["output_sa", "output_nsa"]:
        assert frame.loc[frame[field].notna(), field].gt(0).all(), "Nonpositive raw output cannot be logged or silently shifted"
    frame = frame.sort_values(["sector_id", period]).reset_index(drop=True)
    index = pd.MultiIndex.from_arrays([frame.sector_id, frame[period]])
    assert index.is_unique
    horizon = 12 if freq == "M" else 4
    for outcome, field, growth_horizon in [("y_short", "output_sa", 1), ("y_yoy", "output_sa", horizon), ("y_yoy_nsa", "output_nsa", horizon)]:
        levels = pd.Series(frame[field].to_numpy(), index=index)
        previous = levels.reindex(pd.MultiIndex.from_arrays([frame.sector_id, frame[period] - growth_horizon])).to_numpy()
        frame[outcome] = 100 * (np.log(frame[field]) - np.log(previous))
    return frame.set_index(["sector_id", period])[OUTCOMES]


@lru_cache(maxsize=None)
def maximum_raw_history(context: shared.Context, outcome: str) -> tuple[pd.DataFrame, dict]:
    path = ROOT / "outputs/crisis_sensitivity/parents" / f"{context.key}.csv"
    parent = pd.read_csv(path, dtype={"sector_id": str}, float_precision="round_trip")
    parent = parent.sort_values(["time", "sector_id"]).reset_index(drop=True)
    weather, weather_path = weather_history(context)
    periods = pd.PeriodIndex(parent[context.period], freq=context.freq)
    assert np.unique(parent.time.to_numpy() - periods.asi8).size == 1
    for term in ["temp_c", "rain_100mm", "rh_pct"]:
        assert np.allclose(parent[term], weather[term].reindex(periods), atol=1e-10, rtol=1e-12)
    frame = parent.copy()
    family = "iip" if context.key.startswith("iip") else "va"
    registry = raw_growth_registry(family, context.freq)
    current_index = pd.MultiIndex.from_arrays([frame.sector_id, periods])
    assert np.allclose(frame[outcome], registry[outcome].reindex(current_index), atol=2e-10, rtol=2e-11), "Saved current Y differs from raw full-calendar growth"
    for lag in range(context.horizon + 1):
        for term, prefix in [("temp_c", "temp"), ("rain_100mm", "rain")]:
            frame[f"{prefix}_l{lag}"] = weather[term].reindex(periods - lag).to_numpy()
    for lag in range(1, context.horizon + 1):
        frame[f"ar_l{lag}"] = registry[outcome].reindex(pd.MultiIndex.from_arrays([frame.sector_id, periods - lag])).to_numpy()
    required = [outcome, "rh_pct"] + [f"{prefix}_l{lag}" for prefix in ["temp", "rain"] for lag in range(context.horizon + 1)] + [f"ar_l{lag}" for lag in range(1, context.horizon + 1)]
    eligible = np.isfinite(frame[required]).all(axis=1)
    balanced = pd.Series(eligible, index=frame.index).groupby(frame.time).all()
    eligible = frame.time.isin(balanced.index[balanced])
    maximum = frame.loc[eligible].copy().reset_index(drop=True)
    assert len(maximum) == maximum.time.nunique() * parent.sector_id.nunique()
    support = context.horizon + 1 if outcome == "y_short" else 2 * context.horizon
    metadata = {"context": context.key, "context_label": context.label, "frequency": context.frequency,
        "outcome": outcome, "outcome_label": shared.OUTCOME_LABELS[outcome], "output_adjustment": "NSA" if outcome == "y_yoy_nsa" else "SA",
        "analysis_role": "Primary YoY" if outcome == "y_yoy_nsa" else "SA growth comparison",
        "implementation_version": "raw_history_v2",
        "horizon": context.horizon, "parent_csv": relative(path), "parent_sha256": shared.sha(path),
        "weather_history_csv": relative(weather_path), "weather_history_sha256": shared.sha(weather_path),
        "parent_n_obs": len(parent), "parent_n_periods": parent.time.nunique(),
        "maximum_history_n_obs": len(maximum), "maximum_history_n_periods": maximum.time.nunique(),
        "raw_model_history_n_periods": maximum.time.nunique(), "raw_history_loss_periods": parent.time.nunique() - maximum.time.nunique(),
        "raw_AR_history_loss_periods": int(frame.loc[~np.isfinite(frame[[f"ar_l{lag}" for lag in range(1, context.horizon + 1)]]).all(axis=1), "time"].nunique()),
        "history_loss_periods": parent.time.nunique() - maximum.time.nunique(),
        "weather_history_loss_periods": int(frame.loc[~np.isfinite(frame[[f"{prefix}_l{lag}" for prefix in ["temp", "rain"] for lag in range(context.horizon + 1)]]).all(axis=1), "time"].nunique()),
        "AR_history_scope": "Complete raw official output-growth calendar before weather/context cropping; actual period lookup; no imputation or compressed row shifts",
        "weather_history_scope": "Broadest national processed weather calendar, not truncated to the output parent",
        "maximum_output_support_horizon": support, "includes_dwellings": context.add_dwellings if context.key.startswith("va") else context.key.startswith("aggregate_va"),
        "matched_to": context.matched_to,
        "stationarity_status": "Not cleared: dynamic/gapped histories are exploratory; no stationarity pretest or sample trimming",
        "selection_rule": "Common finite maximum-history dates within context/outcome, then fixed output-support calendar masks; no selection by realised values or inference"}
    return maximum, metadata


def prepare_history(context: shared.Context, outcome: str, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Raw history first, explicit target support matching second, masks last."""
    maximum, base = maximum_raw_history(context, outcome)
    maximum, metadata = maximum.copy(), dict(base)
    target_count = None
    if context.matched_to:
        target = next(item for item in shared.contexts() if item.key == context.matched_to)
        template, _ = maximum_raw_history(target, outcome)
        target_count = template.time.nunique()
        target_dates = set(pd.PeriodIndex(template[target.period], freq=target.freq))
        maximum = maximum.loc[pd.PeriodIndex(maximum[context.period], freq=context.freq).isin(target_dates)].copy().reset_index(drop=True)
        metadata.update(matched_history_template_csv=f"outputs/crisis_sensitivity/parents/{target.key}.csv",
            matched_history_template_sha256=shared.sha(ROOT / f"outputs/crisis_sensitivity/parents/{target.key}.csv"))
    metadata.update(maximum_history_n_obs=len(maximum), maximum_history_n_periods=maximum.time.nunique(),
        target_alignment_loss_periods=base["raw_model_history_n_periods"] - maximum.time.nunique(),
        matched_template_n_periods=target_count, history_loss_periods=base["parent_n_periods"] - maximum.time.nunique(),
        matched_history_rule="Intersect full raw maximum-history support with named target support before event masks" if context.matched_to else "Unmatched original history retains all full raw supported parent dates",
        policy_id=policy["policy_id"])
    intervals = [(event["event_id"], pd.Period(event["start"], freq="M").asfreq(context.freq),
                  pd.Period(event["end"], freq="M").asfreq(context.freq)) for event in policy["events"]]
    support = metadata["maximum_output_support_horizon"]
    masks = []
    for _, date in maximum.drop_duplicates("time").iterrows():
        period = pd.Period(date[context.period], freq=context.freq)
        at_t = [name for name, start, end in intervals if start <= period <= end]
        overlap = [name for name, start, end in intervals if period - support <= end and period >= start]
        masks.append({"context": context.key, "outcome": outcome, "time": int(date.time), "period": str(period),
            "output_support_start": str(period - support), "output_support_horizon": support,
            "event_ids_at_t": json.dumps(at_t), "event_ids_in_output_support": json.dumps(overlap),
            "retain_event_dates": not at_t, "retain_strict": not overlap})
    return maximum, pd.DataFrame(masks), metadata


def design(sample: pd.DataFrame, specification: str, structure: str, horizon: int):
    sectors, seasons, years = sorted(sample.sector_id.unique()), sorted(sample.season.unique()), sorted(sample.year.unique())
    indicators = np.column_stack([sample.sector_id.eq(unit) for unit in sectors]).astype(float)
    seasonal = np.column_stack([sample.season.eq(season) for season in seasons]).astype(float)
    fixed = (indicators[:, :, None] * seasonal[:, None, :]).reshape(len(sample), -1)
    names = [f"sector_season[{unit},{season}]" for unit in sectors for season in seasons]
    if len(years) > 1:
        fixed = np.column_stack([fixed, np.column_stack([sample.year.eq(year) for year in years[1:]])])
        names += [f"year[{year}]" for year in years[1:]]
    lags = [0] if specification == "benchmark" else list(range(horizon + 1))
    controls = [f"rain_l{lag}" for lag in lags] + ["rh_pct"]
    if specification == "dynamic": controls += [f"ar_l{lag}" for lag in range(1, horizon + 1)]
    blocks = [fixed, (sample[controls] - sample[controls].mean()).to_numpy()]
    selected = list(range(len(names), len(names) + len(controls)))
    names += controls
    components = []
    features = {f"temp_l{lag}": sample[f"temp_l{lag}"].to_numpy() for lag in lags}
    if specification == "two_channel":
        features = {"temp_endpoint_difference": sample.temp_l0.to_numpy() - sample[f"temp_l{horizon}"].to_numpy(),
                    "temp_growth_sum": sample[[f"temp_l{lag}" for lag in range(horizon)]].sum(axis=1).to_numpy()}
    for feature, values in features.items():
        centered = values - values.mean()
        position = len(names)
        if structure == "H1":
            blocks.append(np.column_stack([centered, indicators[:, 1:] * centered[:, None]]))
            names += [feature] + [f"{feature}:C(sector_id)[T.{unit}]" for unit in sectors[1:]]
            components.append(list(range(position, position + len(sectors))))
        else:
            blocks.append(centered[:, None]);names.append(feature);components.append([position])
        selected += list(range(position, len(names)))
    matrix = np.column_stack(blocks)
    return matrix, names, np.asarray(selected), np.asarray(components), sectors, lags


def calendar_covariance(scores: np.ndarray, dates: np.ndarray, bandwidth: int, correction: float):
    """Bartlett covariance preserving missing calendar positions, with one meat product."""
    indexes = np.asarray(dates, int) - int(np.min(dates))
    aggregate = np.zeros((int(indexes.max()) + 1, scores.shape[1]))
    np.add.at(aggregate, indexes, scores)
    preceding = np.zeros_like(aggregate)
    for lag in range(1, bandwidth + 1):
        if lag >= len(aggregate): break
        preceding[lag:] += (1 - lag / (bandwidth + 1)) * aggregate[:-lag]
    cross = aggregate.T @ preceding
    matrix = aggregate.T @ aggregate + cross + cross.T
    return (matrix + matrix.T) / 2 * correction


def test_record(beta, covariance, reference, restriction, metadata, name, rank_bound):
    restriction = np.atleast_2d(restriction)
    effect = restriction @ beta
    matrix = restriction @ covariance @ restriction.T;matrix = (matrix + matrix.T) / 2
    singular = np.linalg.svd(matrix, compute_uv=False)
    scale = float(np.linalg.norm(restriction @ reference @ restriction.T, 2))
    zero_cutoff = ZERO * scale
    cutoff = max(float(singular[0]) * RANK, zero_cutoff)
    rank = min(int(np.sum(singular > cutoff)), max(rank_bound, 0))
    q = len(effect)
    available = rank == q and singular[0] > zero_cutoff and metadata["inference_df"] > 0
    statistic = float(effect @ np.linalg.solve(matrix, effect)) if available else np.nan
    assert not available or statistic >= -1e-8
    return {**metadata, "test": name, "restrictions": q, "covariance_rank": rank, "structural_rank_bound": rank_bound,
        "covariance_condition": float(np.linalg.cond(matrix)), "hc0_covariance_scale": scale,
        "zero_covariance_threshold": zero_cutoff, "rank_cutoff": cutoff,
        "status": "available" if available else "unavailable: rank-deficient or numerically degenerate restriction covariance",
        "wald_chi2": max(statistic, 0.) if available else None,
        "f_statistic": max(statistic, 0.) / q if available else None,
        "numerator_df": q, "denominator_df": metadata["inference_df"],
        "p_value": float(f_distribution.sf(max(statistic, 0.) / q, q, metadata["inference_df"])) if available else None}


def restrictions(length: int, outcome: str):
    eye = np.eye(length)
    result = [("All temperature-lag coefficients zero", eye), ("Cumulative direct temperature response zero", np.ones((1, length)))]
    if length > 1:
        endpoint = 1 if outcome == "y_short" else length - 1
        intermediate = [eye[j] for j in range(1, length) if j != endpoint]
        result.append(("Exact contemporaneous static-output-level restriction", np.vstack(intermediate + [eye[0] + eye[endpoint]])))
        if outcome != "y_short":
            result.append(("Equal one-period growth responses over YoY accumulation window", np.vstack([eye[j] - eye[0] for j in range(1, length - 1)] + [eye[-1]])))
    return result


def inference_rows(beta, covariance, reference, contrast, metadata):
    estimate = contrast @ beta
    variance = float(contrast @ covariance @ contrast)
    numerical = float(contrast @ reference @ contrast)
    return {**metadata, **shared.inference(float(estimate), variance, metadata["inference_df"], numerical)}


def fit_spec(context, sample, metadata, specification, structure, records, *, preflight=False, unit_id="pooled"):
    matrix, names, selected, components, sectors, lags = design(sample, specification, structure, context.horizon)
    geo = shared.geometry(matrix)
    n, k, t = len(sample), matrix.shape[1], sample.time.nunique()
    local = matrix[sample.sector_id.eq(sectors[0]).to_numpy()]
    local_rank = int(np.linalg.matrix_rank(local))
    spec_id = "__".join([context.key, metadata["outcome"], metadata["sample_variant"], structure, specification] + ([unit_id] if structure == "separate" else []))
    temperatures = ["temp_endpoint_difference", "temp_growth_sum"] if specification == "two_channel" else [f"temp_l{lag}" for lag in lags]
    controls = [f"rain_l{lag}" for lag in lags] + ["rh_pct"] + ([f"ar_l{lag}" for lag in range(1, context.horizon + 1)] if specification == "dynamic" else [])
    rhs = " + ".join(temperatures) + (" interacted with sector (main + nonreference deviations)" if structure == "H1" else "") + " + " + " + ".join(controls)
    periods = pd.PeriodIndex(sample[context.period], freq=context.freq)
    meta = {**metadata, "spec_id": spec_id, "native_label": "Literature sensitivity " + spec_id,
        "specification": specification, "structure": structure, "unit_id": unit_id, "sector_id": unit_id,
        "formula": metadata["outcome"] + " ~ " + rhs + " + sector×season FE + common year FE",
        "controls": json.dumps(controls), "temperature_terms": json.dumps(temperatures), "control_set": "long_weather",
        "AR_slope_scope": "Shared across sectors" if specification == "dynamic" and structure == "H1" else "Common" if specification == "dynamic" else "None",
        "n_obs": n, "n_periods": t, "n_sectors": len(sectors), "n_parameters": k, "residual_df": n - k,
        "time_design_rank": local_rank, "time_residual_df": t - local_rank,
        "design_rank": geo["design_rank"], "design_condition_number": geo["design_condition_number"],
        "scaled_design_condition_number": geo["scaled_design_condition_number"], "design_rank_threshold": geo["design_rank_threshold"],
        "status": geo["status"], "start": str(periods.min()), "end": str(periods.max()),
        "calendar_gaps": periods.max().ordinal - periods.min().ordinal + 1 - t,
        "mean_y": float(sample[metadata["outcome"]].mean()), "sd_y": float(sample[metadata["outcome"]].std()),
        "reference_sector_id": sectors[0] if structure == "H1" else "",
        "estimator": "PyFixest feols", "estimator_version": native.ESTIMATOR_VERSION,
        "coefficient_sum_interpretation": "Sum of direct conditional weather coefficients; dynamic specifications do not report an unconditional long-run multiplier",
        "two_channel_construction": "T0−Th and sum(T0,...,T[h−1]); sum coefficient is not a coefficient on the h-period mean" if specification == "two_channel" else ""}
    records["specifications"].append(meta)
    if preflight or geo["status"] != "estimated": return meta
    fitted = native.fit_many_matrix(sample[[metadata["outcome"]]].to_numpy(), matrix, names, meta["native_label"], outcome_names=[metadata["outcome"]])[0]
    beta, residual = fitted.params.to_numpy(), fitted.resid
    influence = geo["mapping"].T * residual[:, None]
    selected_influence = influence[:, selected]
    hc0 = selected_influence.T @ selected_influence
    base_record = {key: meta[key] for key in ["spec_id", "context", "outcome", "sample_variant", "specification", "structure", "unit_id", "n_obs", "n_periods", "n_parameters", "estimator", "estimator_version"]}
    records["coefficients"].extend({**base_record, "term": name, "coefficient": float(coefficient)} for name, coefficient in zip(names, beta, strict=True))
    calendar_factor = t / (t - 1) * (n - 1) / (n - k)
    methods = []
    if structure == "common" and len(sectors) > 1:
        scores = np.zeros((len(sectors), k))
        np.add.at(scores, pd.Categorical(sample.sector_id, categories=sectors).codes, influence)
        factor = len(sectors) / (len(sectors) - 1) * (n - 1) / (n - k)
        methods.append(("Sector cluster", 0, scores.T @ scores * factor, factor, len(sectors) - 1, "G/(G−1)×(N−1)/(N−K)"))
    for bandwidth in [context.horizon, 2 * context.horizon]:
        methods.append(("Calendar HAC" if len(sectors) == 1 else "Driscoll-Kraay", bandwidth,
            calendar_covariance(influence, sample.time.to_numpy(), bandwidth, calendar_factor), calendar_factor, t - 1,
            "T/(T−K_unit)" if len(sectors) == 1 else "T/(T−1)×(N−1)/(N−K)"))
    component_indexes = np.asarray([[list(selected).index(index) for index in row] for row in components])
    archive = {"X": matrix, "Y": sample[metadata["outcome"]].to_numpy(), "coefficients": beta, "residuals": residual,
        "coef_names": np.asarray(names, str), "time": sample.time.to_numpy(), "periods": periods.astype(str).to_numpy(dtype=str),
        "units": sample.sector_id.to_numpy(dtype=str), "selected_positions": selected,
        "hc0_selected": hc0, "component_positions": components, "lags": np.asarray(lags),
        "covariance_labels": np.asarray([item[0] for item in methods], str), "bandwidths": np.asarray([item[1] for item in methods]),
        "corrections": np.asarray([item[3] for item in methods]), "inference_df": np.asarray([item[4] for item in methods])}
    unit_names = sample.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    for number, (method, bandwidth, full_covariance, factor, degrees, correction_formula) in enumerate(methods):
        archive[f"covariance_{number}"] = full_covariance
        covariance = full_covariance[np.ix_(selected, selected)]
        band_meta = {**meta, "covariance": method, "hac_lags": bandwidth, "inference_df": degrees,
            "finite_sample_correction": factor, "finite_sample_correction_formula": correction_formula,
            "n_clusters": len(sectors) if method == "Sector cluster" else 0,
            "main_inference": method == "Sector cluster" if structure == "common" and len(sectors) > 1 else bandwidth == context.horizon,
            "r_squared": float(fitted.rsquared)}
        temperature_contrasts = []
        profile_sectors = sectors if structure == "H1" else [sectors[0] if len(sectors) == 1 else "pooled"]
        for unit_number, sector in enumerate(profile_sectors):
            basis_contrasts = np.zeros((len(components), len(selected)))
            for lag_number, indexes in enumerate(component_indexes):
                basis_contrasts[lag_number, indexes[0]] = 1.
                if structure == "H1" and unit_number: basis_contrasts[lag_number, indexes[unit_number]] = 1.
            contrasts = basis_contrasts
            if specification == "two_channel":
                expansion = np.zeros((len(lags), 2))
                expansion[0, 0], expansion[-1, 0] = 1., -1.
                expansion[:-1, 1] = 1.
                contrasts = expansion @ basis_contrasts
            temperature_contrasts.append(contrasts)
            unit = sample.loc[sample.sector_id.eq(sector)] if sector != "pooled" else sample
            unit_meta = {**band_meta, "sector_id": sector, "sector": unit_names.get(sector, "Common equal-unit slope"),
                "mean_y": float(unit[metadata["outcome"]].mean()), "sd_y": float(unit[metadata["outcome"]].std())}
            for lag_number, lag in enumerate(lags):
                detail = {**unit_meta, "effect": "lag", "lag": lag, "term": f"temp_l{lag}"}
                if structure == "H1" and specification != "two_channel":
                    positions = component_indexes[lag_number]
                    detail.update(temp_main=float(beta[selected[positions[0]]]),
                        interaction_delta=0. if unit_number == 0 else float(beta[selected[positions[unit_number]]]),
                        delta_is_normalization=unit_number == 0,
                        var_temp_main=float(covariance[positions[0], positions[0]]),
                        var_interaction_delta=0. if unit_number == 0 else float(covariance[positions[unit_number], positions[unit_number]]),
                        cov_temp_main_delta=0. if unit_number == 0 else float(covariance[positions[0], positions[unit_number]]))
                records["temperature_effects"].append(inference_rows(beta[selected], covariance, hc0, contrasts[lag_number], detail))
            records["temperature_effects"].append(inference_rows(beta[selected], covariance, hc0, contrasts.sum(axis=0),
                {**unit_meta, "effect": "cumulative_direct", "lag": -1, "term": "sum_temperature_lags"}))
            lag_beta, lag_covariance, lag_reference = contrasts @ beta[selected], contrasts @ covariance @ contrasts.T, contrasts @ hc0 @ contrasts.T
            rank_bound = len(sectors) - 1 if method == "Sector cluster" else t - sample.season.nunique() - 1
            if structure == "separate" or len(sectors) == 1: rank_bound = t - k
            if specification == "two_channel":
                for number, channel in enumerate(temperatures):
                    records["channel_estimates"].append(inference_rows(beta[selected], covariance, hc0, basis_contrasts[number],
                        {**unit_meta, "channel": channel, "term": channel, "effect_units": "Per +1°C endpoint difference" if number == 0 else "Per +1°C-period exposure sum; coefficient on a mean would be h times larger"}))
                basis_beta = basis_contrasts @ beta[selected]
                basis_cov, basis_hc0 = basis_contrasts @ covariance @ basis_contrasts.T, basis_contrasts @ hc0 @ basis_contrasts.T
                for test_name, restriction in [("All temperature-lag coefficients zero", np.eye(2)),
                    ("Cumulative direct temperature response zero", np.array([[0., context.horizon]])),
                    ("Exact contemporaneous static-output-level restriction", np.array([[0., 1.]])),
                    ("Equal one-period growth responses over YoY accumulation window", np.array([[1., 0.]]))]:
                    records["tests"].append(test_record(basis_beta, basis_cov, basis_hc0, restriction, unit_meta, test_name, rank_bound))
            else:
                for test_name, restriction in restrictions(len(lags), metadata["outcome"]):
                    records["tests"].append(test_record(lag_beta, lag_covariance, lag_reference, restriction, unit_meta, test_name, rank_bound))
        if structure == "H1":
            for lag_number, feature in enumerate(temperatures):
                lag = lags[lag_number] if specification != "two_channel" else -2-lag_number
                for unit_number, sector in enumerate(sectors):
                    contrast = np.eye(len(selected))[component_indexes[lag_number, unit_number]]
                    records["interaction_components"].append(inference_rows(beta[selected], covariance, hc0, contrast,
                        {**band_meta, "sector_id": sector, "lag": lag, "component": "base" if unit_number == 0 else "interaction",
                         "estimated": True, "normalization": False, "term": names[components[lag_number, unit_number]]}))
                records["interaction_components"].append({**band_meta, "sector_id": sectors[0], "lag": lag,
                    "component": "interaction", "estimated": False, "normalization": True,
                    "term": f"{feature}:C(sector_id)[T.{sectors[0]}]", "coefficient": 0., "variance": None,
                    "std_error": None, "p_value": None, "ci_lower": None, "ci_upper": None,
                    "inference_status": "Reference interaction normalized to zero; not estimated"})
            cumulative = np.vstack([item.sum(axis=0) for item in temperature_contrasts])
            restriction = np.eye(len(sectors))[1:] - np.eye(len(sectors))[[0]]
            records["tests"].append(test_record(cumulative @ beta[selected], cumulative @ covariance @ cumulative.T,
                cumulative @ hc0 @ cumulative.T, restriction, {**band_meta, "sector_id": "all"},
                "All sector cumulative direct temperature responses equal", t - sample.season.nunique() - 1))
            flattened = np.vstack(temperature_contrasts)
            if specification == "two_channel":
                # The implied h+1 lag coefficients obey exact restrictions by
                # construction; test the identified two-channel basis instead.
                flattened = np.vstack([np.eye(len(selected))[component_indexes[:, 0]] +
                    (np.eye(len(selected))[component_indexes[:, unit]] if unit else 0.) for unit in range(len(sectors))])
            records["tests"].append(test_record(flattened @ beta[selected], flattened @ covariance @ flattened.T,
                flattened @ hc0 @ flattened.T, np.eye(len(flattened)), {**band_meta, "sector_id": "all"},
                "All sector temperature-lag responses jointly zero", t - sample.season.nunique() - 1))
    np.savez_compressed(OUT / "designs" / f"{spec_id}.npz", **archive)
    return meta


def run_literature_sensitivity(policy_path: Path = shared.DEFAULT_POLICY, *, preflight: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    producer_sha256 = shared.sha(Path(__file__))
    maximum_raw_history.cache_clear();raw_growth_registry.cache_clear()
    for folder in ["samples", "designs"]: (OUT / folder).mkdir(exist_ok=True)
    policy = shared.load_policy(Path(policy_path))
    policy_sha = shared.sha(Path(policy_path))
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n")
    protected_paths = ["scripts/crisis_sensitivity.py", "scripts/complete_va_controls.py",
        "outputs/crisis_sensitivity/manifest.json",
        "outputs/complete_va_controls/manifest.json",
        "outputs/pyfixest/fit_log.csv", "outputs/pyfixest/estimator_manifest.json",
        "outputs/pyfixest/migration_input_reference.json", "outputs/covid_comparison/original_inputs_reference.json",
        "notes/aea_temperature_literature.md", "notes/temperature_horizon_review.md",
        "data/processed/weather_monthly.csv", "data/processed/weather_quarterly.csv"]
    protected_paths += [f"outputs/crisis_sensitivity/parents/{context.key}.csv" for context in shared.contexts()]
    raw_inputs = [f"data/raw/{prefix}_{kind}.json" for prefix in ["va_sa", "va_nsa", "iip_sa", "iip_nsa"] for kind in ["tabledata", "metadata"]]
    raw_hashes = {name: shared.sha(ROOT / name) for name in raw_inputs}
    protocol_name = "notes/literature_history_alignment_v2.md"
    protocol_sha256 = shared.sha(ROOT / protocol_name)
    protected = {name: shared.sha(ROOT / name) for name in protected_paths}
    records = {name: [] for name in ["specifications", "coefficients", "temperature_effects", "interaction_components", "channel_estimates", "tests"]}
    masks, calendar_summaries = [], []
    previous = (native.LOG_DIR, list(native._FIT_LOG), native._API_COUNTER, native._RUN_STARTED)
    if not preflight:
        native.LOG_DIR = OUT / "pyfixest";native.reset_fit_log()
    receipt = {}
    try:
        with threadpool_limits(limits=1):
            for context in shared.contexts():
                for outcome in OUTCOMES:
                    maximum, mask, base = prepare_history(context, outcome, policy)
                    masks.extend(mask.to_dict("records"))
                    for variant in VARIANTS:
                        keep = np.ones(len(mask), bool) if variant == "all_eligible" else mask["retain_event_dates" if variant == "event_dates_excluded" else "retain_strict"].to_numpy()
                        sample = maximum.loc[maximum.time.isin(mask.loc[keep, "time"])].copy().reset_index(drop=True)
                        assert not sample.empty
                        sample_path = OUT / "samples" / f"{context.key}__{outcome}__{variant}.csv"
                        sample.to_csv(sample_path, index=False)
                        meta = {**base, "sample_variant": variant, "sample_csv": relative(sample_path), "sample_sha256": shared.sha(sample_path),
                            "policy_source": relative(Path(policy_path)), "policy_source_sha256": policy_sha,
                            "sample_comparability": "Exact same retained dates and Y across benchmark, distributed and dynamic within context/outcome/variant"}
                        calendar_summaries.append({**meta, "n_periods": sample.time.nunique(), "n_sectors": sample.sector_id.nunique(),
                            "strict_support": "t−h−1 through t" if outcome == "y_short" else "t−2h through t"})
                        requested = SPECIFICATIONS + (["two_channel"] if outcome != "y_short" else [])
                        for specification in requested:
                            fit_spec(context, sample, meta, specification, "common", records, preflight=preflight)
                            if sample.sector_id.nunique() > 1:
                                fit_spec(context, sample, meta, specification, "H1", records, preflight=preflight)
                        if sample.sector_id.nunique() > 1:
                            for sector, unit in sample.groupby("sector_id", sort=True):
                                fit_spec(context, unit.reset_index(drop=True), meta, "distributed", "separate", records, preflight=preflight, unit_id=sector)
                        print(f"Literature {'preflight' if preflight else 'fit'} {context.key}/{outcome}/{variant}: {sample.sector_id.nunique()} units × {sample.time.nunique()} dates", flush=True)
            if not preflight: receipt = native.write_fit_log()
    finally:
        if not preflight:
            native.LOG_DIR, old, native._API_COUNTER, native._RUN_STARTED = previous
            native._FIT_LOG[:] = old
    assert all(shared.sha(ROOT / name) == value for name, value in protected.items()), "Prior protected file changed"
    assert all(shared.sha(ROOT / name) == value for name, value in raw_hashes.items()), "Raw official output input changed"
    assert shared.sha(Path(policy_path)) == policy_sha and shared.sha(ROOT / protocol_name) == protocol_sha256
    assert shared.sha(Path(__file__)) == producer_sha256, "Producer source changed during the run"
    specifications = pd.DataFrame(records["specifications"])
    specs_name = "preflight_specifications.csv" if preflight else "specifications.csv"
    specifications.to_csv(OUT / specs_name, index=False)
    if not preflight:
        for name, rows in records.items():
            if name == "specifications": continue
            pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame(masks).to_csv(OUT / "masks.csv", index=False)
    pd.DataFrame(calendar_summaries).to_csv(OUT / "sample_inventory.csv", index=False)
    manifest = {"status": "preflight_only" if preflight else "estimated_not_yet_independently_verified",
        "contexts": 20, "growth_outcomes": OUTCOMES, "sample_variants": VARIANTS,
        "samples": len(calendar_summaries), "requested_specs": len(specifications),
        "identified_specs": int(specifications.status.eq("estimated").sum()),
        "unavailable_specs": int(specifications.status.ne("estimated").sum()),
        "native_fits": receipt.get("native_fits_recorded", 0), "row_counts": {name: len(rows) for name, rows in records.items()},
        "policy_source": relative(Path(policy_path)), "policy_source_sha256": policy_sha,
        "protected_sources_sha256": protected,
        "raw_output_sources_sha256": raw_hashes, "producer_source": relative(Path(__file__)), "producer_source_sha256": producer_sha256,
        "implementation_version": "raw_history_v2", "history_alignment_protocol": protocol_name, "history_alignment_protocol_sha256": protocol_sha256,
        "scope": "Current matched-date benchmark, unrestricted temperature/rain DL0..h, DL plus Y lags1..h; joint endpoint-difference/growth-sum model for YoY; common/H1 throughout, DL-only own-series robustness",
        "not_replication": "Singapore national-weather adaptation; no country/date FE replication, daily-maximum-temperature replication, GMM or finite-T bias correction",
        "history": "Weather uses broadest national history. AR uses complete raw official output calendar before weather/context cropping. Named matched contexts intersect maximum-history eligibility with target support; missing source history and explicit alignment losses remain separate. All models share dates within each outcome.",
        "strict_mask": "Output support t−h−1..t for short growth, t−2h..t for YoY; weather history alone does not trigger an economic-crisis exclusion",
        "inference": "Common panels: sector CR1 plus calendar DK h/2h. H1: joint calendar DK. National/own-series: calendar HAC. Actual time gaps retained.",
        "numerical_thresholds": {"zero_relative_to_HC0": ZERO, "covariance_rank_relative": RANK},
        "native_ledger": "outputs/literature_sensitivity/pyfixest/", "full_covariance_storage": "designs/{spec_id}.npz covariance_0,... with labels/bandwidths/factors/df; full nuisance-inclusive matrices",
        "dynamic_interpretation": "Conditional direct weather response; shared AR slopes in pooled/H1; no unconditional long-run multiplier or causal-certification claim"}
    target = "preflight.json" if preflight else "manifest.json"
    (OUT / target).write_text(json.dumps(shared.clean(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: manifest[key] for key in ["status", "requested_specs", "identified_specs", "unavailable_specs", "native_fits"]}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=shared.DEFAULT_POLICY)
    parser.add_argument("--preflight", action="store_true", help="Save exact eligibility/design-rank inventory without native fits")
    args = parser.parse_args()
    run_literature_sensitivity(args.policy.resolve(), preflight=args.preflight)
