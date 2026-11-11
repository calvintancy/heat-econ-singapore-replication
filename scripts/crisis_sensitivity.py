"""Long-history weather-only crisis/event-window sensitivity on frozen samples.

This additional analysis never changes a baseline sample or reference. The
calendar policy is an analytical sensitivity, not official recession dating.
Outcomes are loaded after their original full-history transformations; one
YoY-safe calendar mask is applied to all five outcomes in each context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution, t as student_t
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

import pyfixest_backend as native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/crisis_sensitivity"
DEFAULT_POLICY = ROOT / "notes/crisis_sensitivity_policy.json"
TERMS = ["temp_c", "rain_100mm", "rh_pct"]
OUTCOMES = ["y_short", "y_yoy_nsa", "y_yoy", "y_log_level", "y_raw_level"]
VARIANTS = ["all_eligible", "event_dates_excluded", "strict_growth_window_clean"]
OUTCOME_LABELS = {"y_short": "MoM/QoQ log growth (SA; not annualised)", "y_yoy_nsa": "YoY log growth (NSA)",
                  "y_yoy": "YoY log growth (SA)", "y_log_level": "100 × log output level (SA; descriptive)",
                  "y_raw_level": "Raw output level (SA; descriptive)"}
META = {"estimator": "PyFixest feols", "estimator_version": native.ESTIMATOR_VERSION}
ZERO_RELATIVE_TOLERANCE, RANK_RELATIVE_TOLERANCE = 1e-12, 1e-10


@dataclass(frozen=True)
class Context:
    key: str
    label: str
    source: str
    frequency: str
    matched_to: str = ""
    match_template: str = ""
    add_dwellings: bool = False

    @property
    def period(self): return "month" if self.frequency == "monthly" else "quarter"

    @property
    def freq(self): return "M" if self.frequency == "monthly" else "Q"

    @property
    def horizon(self): return 12 if self.frequency == "monthly" else 4


def contexts() -> list[Context]:
    result = [Context("aggregate_va", "Official aggregate GVA", "outputs/aggregate_va/weather/samples/G1.csv", "quarterly"),
              Context("va", "11 broad VA sectors · original history", "outputs/outcomes/va/weather/common_sample.csv", "quarterly"),
              Context("va-broad-matched", "11 broad VA sectors · detailed-matched history", "outputs/granular/va-broad-matched/long/samples/G1.csv", "quarterly", "va-detailed"),
              Context("va-detailed", "19 detailed VA sectors", "outputs/granular/va-detailed/long/samples/G1.csv", "quarterly"),
              Context("aggregate_va-matched-va19", "Official aggregate GVA · VA19 matched dates",
                      "outputs/aggregate_va/weather/samples/G1.csv", "quarterly", "va-detailed",
                      "outputs/granular/va-detailed/long/samples/G1.csv"),
              Context("va12", "12 complete broad VA components · includes ownership of dwellings",
                      "outputs/outcomes/va/weather/common_sample.csv", "quarterly", add_dwellings=True),
              Context("va12-matched", "12 complete broad VA components · complete-detailed matched dates",
                      "outputs/granular/va-broad-matched/long/samples/G1.csv", "quarterly", "va20", add_dwellings=True),
              Context("va20", "20 complete detailed VA components · includes ownership of dwellings",
                      "outputs/granular/va-detailed/long/samples/G1.csv", "quarterly", add_dwellings=True)]
    for frequency in ["monthly", "quarterly"]:
        for suffix, label in [("-total", "Published weighted IIP total"), ("", "6 manufacturing clusters · original history"),
                              ("-broad-matched", "6 manufacturing clusters · detailed-matched history"),
                              ("-detailed", "18 manufacturing industries")]:
            key = "iip-"+frequency+suffix
            source = f"outputs/outcomes/{key}/weather/common_sample.csv" if not suffix else f"outputs/granular/{key}/long/samples/G1.csv"
            result.append(Context(key, label+" · "+frequency, source, frequency,
                                  "iip-"+frequency+"-detailed" if suffix == "-broad-matched" else ""))
        for target, template in [("broad", f"outputs/outcomes/iip-{frequency}/weather/common_sample.csv"),
                                 ("detailed", f"outputs/granular/iip-{frequency}-detailed/long/samples/G1.csv")]:
            result.append(Context(f"iip-{frequency}-total-matched-{target}",
                "Published weighted IIP total · "+target+" matched dates · "+frequency,
                f"outputs/granular/iip-{frequency}-total/long/samples/G1.csv", frequency,
                f"iip-{frequency}"+("-detailed" if target == "detailed" else ""), template))
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def clean(value):
    if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    assert policy["variants"] == VARIANTS and policy["controls"] == TERMS and policy["outcomes"] == OUTCOMES
    assert policy["events"] and len({event["event_id"] for event in policy["events"]}) == len(policy["events"])
    for event in policy["events"]:
        assert pd.Period(event["start"], freq="M") <= pd.Period(event["end"], freq="M")
    return policy


def append_dwellings(parent: pd.DataFrame) -> pd.DataFrame:
    """Read row1.3 and transform its complete history before the parent join."""
    from build_panel import parse_va

    sa = parse_va("va_sa")
    nsa = parse_va("va_nsa")
    sa = sa.loc[sa.sector_id.eq("1.3"), ["quarter", "sector", "va_sa"]]
    nsa = nsa.loc[nsa.sector_id.eq("1.3"), ["quarter", "va_nsa"]]
    levels = sa.merge(nsa, on="quarter", validate="one_to_one").sort_values("quarter").set_index("quarter")
    assert levels[["va_sa", "va_nsa"]].gt(0).all().all()
    assert list(levels.index) == list(pd.period_range(levels.index.min(), levels.index.max(), freq="Q"))
    for name, column, horizon in [("y_short", "va_sa", 1), ("y_yoy", "va_sa", 4), ("y_yoy_nsa", "va_nsa", 4)]:
        base = levels[column].reindex(levels.index-horizon).to_numpy()
        levels[name] = 100*(np.log(levels[column])-np.log(base))
    levels["y_raw_level"] = levels.va_sa
    levels["y_log_level"] = 100*np.log(levels.va_sa)
    levels["output_sa"], levels["output_nsa"] = levels.va_sa, levels.va_nsa
    levels["sector_id"] = "1.3"
    nsa_name = next(name for name in ["output_nsa", "va_nsa"] if name in parent)
    canonical = parent[["sector_id", "sector", "quarter", "time", "year", "season"]+TERMS+OUTCOMES].copy()
    canonical["output_sa"], canonical["output_nsa"] = parent.y_raw_level.to_numpy(), parent[nsa_name].to_numpy()
    calendar = canonical.drop_duplicates("time")[["quarter", "time", "year", "season"]+TERMS].copy()
    calendar["quarter"] = pd.PeriodIndex(calendar.quarter, freq="Q")
    dwellings = calendar.merge(levels.reset_index()[["quarter", "sector_id", "sector", "output_sa", "output_nsa"]+OUTCOMES],
                               on="quarter", how="left", validate="one_to_one")
    assert dwellings[OUTCOMES+TERMS].notna().all().all(), "Dwellings cannot alter the original parent eligibility calendar"
    canonical["quarter"] = pd.PeriodIndex(canonical.quarter, freq="Q")
    result = pd.concat([canonical, dwellings], ignore_index=True).sort_values(["time", "sector_id"]).reset_index(drop=True)
    result["quarter"] = result.quarter.astype(str)
    assert result.sector_id.nunique() == parent.sector_id.nunique()+1
    return result


def read_parent(context: Context) -> tuple[pd.DataFrame, list[dict]]:
    sample = pd.read_csv(ROOT / context.source, dtype={"sector_id": str})
    original_source_counts = (len(sample), sample.time.nunique())
    if context.add_dwellings:
        sample = append_dwellings(sample)
    sample = sample.sort_values(["time", "sector_id"]).reset_index(drop=True)
    periods = pd.PeriodIndex(sample[context.period], freq=context.freq)
    assert not sample.duplicated(["sector_id", context.period]).any()
    assert sample.groupby("time").sector_id.nunique().eq(sample.sector_id.nunique()).all()
    assert len(sample) == sample.time.nunique()*sample.sector_id.nunique()
    assert np.unique(sample.time.to_numpy()-periods.asi8).size == 1, "Calendar index must remain faithful"
    assert np.array_equal(sample.year, periods.year)
    assert np.isfinite(sample[OUTCOMES+TERMS]).all().all(), "No additional eligibility or missing-value deletion is allowed"
    transform_checks = []
    assert np.allclose(sample.y_log_level, 100*np.log(sample.y_raw_level), atol=1e-10, rtol=1e-12)
    nsa_name = next((name for name in ["output_nsa", "va_nsa", "iip_nsa"] if name in sample), None)
    for sector_id, unit in sample.groupby("sector_id", sort=True):
        unit = unit.copy()
        unit.index = pd.PeriodIndex(unit[context.period], freq=context.freq)
        for outcome, source, horizon in [("y_short", "y_raw_level", 1), ("y_yoy", "y_raw_level", context.horizon),
                                         ("y_yoy_nsa", nsa_name, context.horizon)]:
            assert source is not None
            bases = unit[source].reindex(unit.index-horizon).to_numpy()
            available = np.isfinite(bases)
            expected = 100*(np.log(unit[source].to_numpy()[available])-np.log(bases[available]))
            actual = unit[outcome].to_numpy()[available]
            assert np.allclose(actual, expected, atol=2e-10, rtol=2e-11), (context.key, sector_id, outcome)
            transform_checks.append({"context": context.key, "sector_id": sector_id, "outcome": outcome,
                "calendar_bases_inside_parent": int(available.sum()), "maximum_absolute_error": float(np.max(np.abs(actual-expected))) if len(actual) else 0.,
                "scope": "Within-parent calendar bases only; initial growth bases remain the saved pre-parent full-history transforms"})
    source_counts = {"source_parent_n_obs": original_source_counts[0], "source_parent_n_periods": original_source_counts[1],
                     "calendar_match_applied": bool(context.match_template),
                     "match_template_source_csv": context.match_template,
                     "match_template_sha256": sha(ROOT / context.match_template) if context.match_template else "",
                     "dwellings_added_from_raw": context.add_dwellings,
                     "dwellings_source_sector_id": "1.3" if context.add_dwellings else "",
                     "includes_dwellings": context.add_dwellings or context.key.startswith("aggregate_va") if context.key.startswith(("va", "aggregate_va")) else None,
                     "weighting": "Equal-component mean; distinct from the published aggregate and its official weighting" if sample.sector_id.nunique() > 1 else "Official separately published national aggregate",
                     "real_level_accounting": "Chain-linked real component levels need not add to official total; no sum identity imposed" if context.key.startswith(("va", "aggregate_va")) else "Official weighted IIP total remains separate from equal-cluster/industry panels",
                     "additional_raw_source_hashes": json.dumps({name: sha(ROOT / name) for name in
                         ["data/raw/va_sa_tabledata.json", "data/raw/va_sa_metadata.json", "data/raw/va_nsa_tabledata.json", "data/raw/va_nsa_metadata.json"]}) if context.add_dwellings else "{}"}
    if context.match_template:
        template = pd.read_csv(ROOT / context.match_template, dtype={"sector_id": str})
        target = pd.PeriodIndex(template[context.period], freq=context.freq).unique()
        sample = sample.loc[pd.PeriodIndex(sample[context.period], freq=context.freq).isin(target)].copy().reset_index(drop=True)
        assert not sample.empty and sample.time.nunique() == len(set(periods).intersection(target))
    sample.attrs.update(source_counts)
    return sample, transform_checks


def mask_table(context: Context, parent: pd.DataFrame, policy: dict) -> pd.DataFrame:
    dates = parent.drop_duplicates("time")[[context.period, "time"]].sort_values("time")
    intervals = [(event["event_id"], pd.Period(event["start"], freq="M").asfreq(context.freq),
                  pd.Period(event["end"], freq="M").asfreq(context.freq)) for event in policy["events"]]
    rows = []
    for _, date in dates.iterrows():
        period = pd.Period(date[context.period], freq=context.freq)
        base = period-context.horizon
        at_t = [name for name, start, end in intervals if start <= period <= end]
        in_window = [name for name, start, end in intervals if base <= end and period >= start]
        rows.append({"context": context.key, "period": str(period), "time": int(date.time), "base_period": str(base),
            "frequency": context.frequency, "horizon": context.horizon, "event_ids_at_t": json.dumps(at_t),
            "intersecting_event_ids_in_growth_window": json.dumps(in_window), "retain_event_dates": not at_t,
            "retain_strict": not in_window, "policy_id": policy["policy_id"]})
    return pd.DataFrame(rows)


def design(sample: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str], list[int], list[int]]:
    sectors, seasons, years = sorted(sample.sector_id.unique()), sorted(sample.season.unique()), sorted(sample.year.unique())
    units = np.column_stack([sample.sector_id.eq(unit) for unit in sectors]).astype(float)
    seasonal = np.column_stack([sample.season.eq(season) for season in seasons]).astype(float)
    annual = np.column_stack([sample.year.eq(year) for year in years[1:]]).astype(float) if len(years) > 1 else np.empty((len(sample), 0))
    base = (units[:, :, None]*seasonal[:, None, :]).reshape(len(sample), -1)
    names = [f"sector_season[{unit},{season}]" for unit in sectors for season in seasons]+[f"year[{year}]" for year in years[1:]]
    weather = (sample[TERMS]-sample[TERMS].mean()).to_numpy()
    if model == "H1":
        temperature = np.column_stack([weather[:, 0], units[:, 1:]*weather[:, [0]]])
        selected = list(range(base.shape[1]+annual.shape[1], base.shape[1]+annual.shape[1]+2+len(sectors)))
        component_positions = selected[2:]
        names += TERMS[1:]+["temp_c"]+[f"temp_c:C(sector_id)[T.{unit}]" for unit in sectors[1:]]
        matrix = np.column_stack([base, annual, weather[:, 1:], temperature])
    else:
        selected = list(range(base.shape[1]+annual.shape[1], base.shape[1]+annual.shape[1]+3))
        component_positions = []
        names += TERMS
        matrix = np.column_stack([base, annual, weather])
    return matrix, names, selected, component_positions


def geometry(matrix: np.ndarray) -> dict:
    n, k = matrix.shape
    norms = np.linalg.norm(matrix, axis=0)
    scaled = matrix/np.where(norms > 0, norms, 1.)
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = float(singular.max()*max(n, k)*np.finfo(float).eps)
    rank = int(np.sum(singular > tolerance))
    result = {"n_parameters": k, "design_rank": rank, "design_rank_threshold": tolerance,
              "design_condition_number": float(np.linalg.cond(matrix)),
              "scaled_design_condition_number": float(singular.max()/singular.min()) if singular.min() else np.inf,
              "status": "estimated" if n > k and rank == k else "unavailable: underidentified or rank-deficient design"}
    if result["status"] == "estimated":
        q, r = np.linalg.qr(scaled, mode="reduced")
        result["mapping"] = np.linalg.solve(r, q.T)/norms[:, None]
    return result


def covariance_from_scores(influence: np.ndarray, time: np.ndarray, bandwidth: int, correction: float) -> np.ndarray:
    positions = np.asarray(time, dtype=int)-int(np.min(time))
    scores = np.zeros((int(positions.max())+1, influence.shape[1]))
    np.add.at(scores, positions, influence)
    meat = scores.T@scores
    for lag in range(1, bandwidth+1):
        if lag >= len(scores): break
        cross = scores[lag:].T@scores[:-lag]
        meat += (1-lag/(bandwidth+1))*(cross+cross.T)
    return (meat+meat.T)/2*correction


def inference(coefficient: float, variance: float, degrees: int, hc0: float) -> dict:
    threshold = ZERO_RELATIVE_TOLERANCE*max(float(hc0), 0.)
    assert variance >= -max(threshold, np.finfo(float).tiny), (variance, hc0)
    degenerate = abs(variance) <= threshold or degrees <= 0 or hc0 <= 0
    se = float(np.sqrt(max(variance, 0.)))
    critical = float(student_t.ppf(.975, degrees)) if degrees > 0 else np.nan
    return {"coefficient": float(coefficient), "variance": float(variance), "std_error": se,
            "ci_lower": None if degenerate else float(coefficient-critical*se),
            "ci_upper": None if degenerate else float(coefficient+critical*se),
            "p_value": None if degenerate else float(2*student_t.sf(abs(coefficient/se), degrees)),
            "degenerate": bool(degenerate), "inference_status": "unavailable: numerical variance degeneracy" if degenerate else "available",
            "hc0_reference_variance": float(hc0), "zero_variance_threshold": threshold}


def profile_tests(beta: np.ndarray, covariance: np.ndarray, hc0: np.ndarray, sectors: list[str], names: dict,
                  meta: dict, rank_bound: int, records: dict) -> None:
    restrictions = np.eye(len(beta))[1:]-np.eye(len(beta))[[0]]
    difference, v = restrictions@beta, restrictions@covariance@restrictions.T
    v = (v+v.T)/2
    singular = np.linalg.svd(v, compute_uv=False)
    reference_scale = float(np.linalg.norm(restrictions@hc0@restrictions.T, 2))
    zero_threshold = ZERO_RELATIVE_TOLERANCE*reference_scale
    rank_cutoff = max(float(singular.max())*RANK_RELATIVE_TOLERANCE, zero_threshold)
    rank = min(int(np.sum(singular > rank_cutoff)), rank_bound)
    count = len(difference)
    status = "available" if rank == count else "unavailable: singular restriction covariance"
    if float(singular.max()) <= zero_threshold: status = "unavailable: numerically degenerate restriction covariance"
    row = {**meta, "test": "All sector temperature slopes equal", "restrictions": count, "covariance_rank": rank,
           "structural_rank_bound": rank_bound, "covariance_condition": float(np.linalg.cond(v)),
           "reference_covariance_scale": reference_scale, "zero_covariance_threshold": zero_threshold, "rank_cutoff": rank_cutoff,
           "numerator_df": count, "denominator_df": meta["inference_df"], "status": status,
           "wald_chi2": None, "f_statistic": None, "p_value": None}
    if status == "available":
        statistic = float(difference@np.linalg.solve(v, difference))
        assert statistic >= -1e-8
        row.update(wald_chi2=max(statistic, 0.), f_statistic=max(statistic, 0.)/count,
                   p_value=float(f_distribution.sf(max(statistic, 0.)/count, count, meta["inference_df"])))
    records["joint_tests"].append(row)
    pair_rows = []
    for i, j in combinations(range(len(sectors)), 2):
        weights = np.zeros(len(beta));weights[i], weights[j] = 1., -1.
        pair_rows.append({**meta, "first_sector_id": sectors[i], "second_sector_id": sectors[j],
            "comparison": names[sectors[i]]+" − "+names[sectors[j]],
            **inference(float(weights@beta), float(weights@covariance@weights), meta["inference_df"], float(weights@hc0@weights))})
    adjusted = multipletests([row["p_value"] if row["p_value"] is not None else 1. for row in pair_rows], method="holm")[1]
    for row, probability in zip(pair_rows, adjusted, strict=True):
        row.update(p_value_holm=float(probability) if row["p_value"] is not None else None,
                   multiplicity_family_size=len(pair_rows),
                   multiplicity_family="All requested sector pairs in context/variant/model/outcome/bandwidth; unavailable tests retain family slots")
    records["pairwise_tests"].extend(pair_rows)


def base_metadata(context: Context, parent: pd.DataFrame, sample: pd.DataFrame, variant: str,
                  policy: dict, sample_path: Path) -> dict:
    periods = pd.PeriodIndex(sample[context.period], freq=context.freq)
    t = sample.time.nunique()
    return {"context": context.key, "context_label": context.label, "frequency": context.frequency,
        "matched_to": context.matched_to, "sample_variant": variant, "sample_id": context.key+"__"+variant,
        **parent.attrs,
        "sample_csv": relative(sample_path), "sample_sha256": sha(sample_path), "source_sample_csv": context.source,
        "source_sample_sha256": sha(ROOT / context.source), "policy_id": policy["policy_id"], "policy_sha256": sha(OUT / "policy.json"),
        "control_set": "weather", "controls": json.dumps(TERMS), "controls_label": "Temperature, rainfall and relative humidity only; no wind or PM2.5",
        "n_obs": len(sample), "n_sectors": sample.sector_id.nunique(), "n_periods": t,
        "parent_n_obs": len(parent), "parent_n_periods": parent.time.nunique(), "parent_n_sectors": parent.sector_id.nunique(),
        "n_removed_obs": len(parent)-len(sample), "n_removed_periods": parent.time.nunique()-t,
        "start": str(periods.min()), "end": str(periods.max()), "calendar_span_periods": periods.max().ordinal-periods.min().ordinal+1,
        "calendar_gaps": periods.max().ordinal-periods.min().ordinal+1-t,
        "stationarity_status": "Not cleared: crisis exclusion creates calendar gaps; no new stationarity pretest" if variant != "all_eligible" else "Parent marginal screens only; no new stationarity clearance",
        "selection_rule": "Fixed event-date or complete YoY growth-window mask; no outcome/residual/significance/pretest trimming",
        "growth_construction": "Saved full-history output transforms loaded before all masks; identical retained dates for every outcome", **META}


def store_matrix(records: dict, filename: str, meta: dict, matrix: np.ndarray, names: list[str],
                 hc0: np.ndarray, *, sectors: bool = False) -> None:
    compact = {key: meta[key] for key in ["context", "sample_variant", "model", "spec_id", "fit_group_id", "outcome",
        "covariance", "hac_lags", "inference_df", "n_obs", "n_sectors", "n_periods", "n_parameters", "finite_sample_correction"]}
    compact["unit_id"] = meta.get("unit_id", "joint_profile")
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            records[filename].append({**compact, "sector_i" if sectors else "term_i": first,
                "sector_j" if sectors else "term_j": second, "covariance_value": float(matrix[i, j]),
                "hc0_reference_value": float(hc0[i, j])})


def fit_model(context: Context, parent: pd.DataFrame, sample: pd.DataFrame, variant: str, policy: dict,
              sample_path: Path, model: str, records: dict, unit_id: str = "pooled") -> dict | None:
    matrix, coef_names, selected, components = design(sample, model)
    geo = geometry(matrix)
    sectors = sorted(sample.sector_id.unique())
    t, n, k = sample.time.nunique(), len(sample), matrix.shape[1]
    local = matrix[sample.sector_id.eq(sectors[0]).to_numpy()]
    local_rank = int(np.linalg.matrix_rank(local))
    group_id = context.key+"__"+variant+"__"+model+("__"+unit_id if model == "separate" else "")
    effective_parent = parent.loc[parent.sector_id.eq(unit_id)].copy() if model == "separate" else parent
    meta = {**base_metadata(context, effective_parent, sample, variant, policy, sample_path), "model": model,
        "family": "separate_time_series" if model == "separate" else "sector_FE_interactions" if model == "H1" else "national" if len(sectors) == 1 else "pooled_FE",
        "fit_group_id": group_id, "native_label": "Crisis sensitivity "+group_id, "unit_id": unit_id,
        "sector_id": unit_id, "n_parameters": k, "design_rank": geo["design_rank"], "residual_df": n-k,
        "time_design_rank": local_rank, "time_residual_df": t-local_rank,
        "design_condition_number": geo["design_condition_number"], "scaled_design_condition_number": geo["scaled_design_condition_number"],
        "design_rank_threshold": geo["design_rank_threshold"], "status": geo["status"],
        "reference_sector_id": sectors[0] if model == "H1" else "",
        "formula": "Y ~ temp_c * C(sector_id) + rain_100mm + rh_pct + C(sector_id)*C(season) + C(year)" if model == "H1" else
                   "Y ~ temp_c + rain_100mm + rh_pct + C(season) + C(year)" if len(sectors) == 1 else
                   "Y ~ temp_c + rain_100mm + rh_pct + C(sector_id)*C(season) + C(year)",
        "design_basis": "Complete sector×season indicators; year indicators excluding first observed year; centered weather; H1 main temperature plus nonreference-sector deviations"}
    for outcome in OUTCOMES:
        records["specifications"].append({**meta, "spec_id": group_id+"__"+outcome, "outcome": outcome,
            "formula": meta["formula"].replace("Y ~", outcome+" ~"),
            "outcome_label": OUTCOME_LABELS[outcome], "mean_y": float(sample[outcome].mean()), "sd_y": float(sample[outcome].std()),
            "output_adjustment": "NSA" if outcome.endswith("_nsa") else "SA"})
    if geo["status"] != "estimated":
        records["unavailable_specs"].extend({**meta, "spec_id": group_id+"__"+outcome, "outcome": outcome,
            "reason": "All requested controls and fixed effects retained; no rank-based regressor dropping or pseudo-inverse inference"} for outcome in OUTCOMES)
        return None
    fitted = native.fit_many_matrix(sample[OUTCOMES].to_numpy(), matrix, coef_names, meta["native_label"], outcome_names=OUTCOMES)
    beta = np.column_stack([fit.params.to_numpy() for fit in fitted])
    residual = np.column_stack([fit.resid for fit in fitted])
    np.savez_compressed(OUT / "designs" / f"{group_id}.npz", X=matrix, Y=sample[OUTCOMES].to_numpy(),
        coefficients=beta, residuals=residual, coef_names=np.asarray(coef_names, str), outcomes=np.asarray(OUTCOMES, str),
        time=sample.time.to_numpy(), periods=sample[context.period].astype(str).to_numpy(dtype=str), units=sample.sector_id.to_numpy(dtype=str))
    names = sample.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    selected_names = [coef_names[i] for i in selected]
    profiles = {}
    for j, outcome in enumerate(OUTCOMES):
        outcome_meta = {**meta, "spec_id": group_id+"__"+outcome, "outcome": outcome, "outcome_label": OUTCOME_LABELS[outcome],
            "formula": meta["formula"].replace("Y ~", outcome+" ~"),
            "mean_y": float(sample[outcome].mean()), "sd_y": float(sample[outcome].std()), "r_squared": float(fitted[j].rsquared)}
        for name, coefficient in zip(coef_names, beta[:, j], strict=True):
            records["all_coefficients"].append({key: outcome_meta[key] for key in ["context", "sample_variant", "model", "spec_id",
                "fit_group_id", "unit_id", "outcome", "n_obs", "n_periods", "n_parameters", "estimator", "estimator_version"]} |
                {"term": name, "coefficient": float(coefficient)})
        influence = geo["mapping"][selected].T*residual[:, [j]]
        hc0 = influence.T@influence
        calendar_factor = t/(t-1)*(n-1)/(n-k)
        bands = [("Calendar HAC" if len(sectors) == 1 else "Driscoll-Kraay", band,
                  covariance_from_scores(influence, sample.time.to_numpy(), band, calendar_factor), calendar_factor, t-1)
                 for band in [context.horizon, 2*context.horizon]]
        if model == "common" and len(sectors) > 1:
            scores = np.zeros((len(sectors), len(selected)))
            np.add.at(scores, pd.Categorical(sample.sector_id, categories=sectors).codes, influence)
            factor = len(sectors)/(len(sectors)-1)*(n-1)/(n-k)
            bands.insert(0, ("Sector cluster", 0, scores.T@scores*factor, factor, len(sectors)-1))
        for method, band, covariance, factor, degrees in bands:
            band_meta = {**outcome_meta, "covariance": method, "hac_lags": band, "inference_df": degrees,
                "n_clusters": len(sectors) if method == "Sector cluster" else 0, "finite_sample_correction": factor,
                "finite_sample_correction_formula": "G/(G−1)×(N−1)/(N−K)" if method == "Sector cluster" else "T/(T−K)" if len(sectors) == 1 else "T/(T−1)×(N−1)/(N−K)",
                "main_inference": method == "Sector cluster" if model == "common" and len(sectors) > 1 else band == context.horizon}
            store_matrix(records, "coefficient_covariances", band_meta, covariance, selected_names, hc0)
            if model != "H1":
                target = "separate_estimates" if model == "separate" else "estimates"
                for index, term in enumerate(selected_names):
                    records[target].append({**band_meta, "term": term, "sector_id": sectors[0] if len(sectors) == 1 else "pooled",
                        "sector": names[sectors[0]] if len(sectors) == 1 else "Equal-sector common slope",
                        **inference(beta[selected[index], j], covariance[index, index], degrees, hc0[index, index])})
            else:
                component_cov, component_hc0 = covariance[2:, 2:], hc0[2:, 2:]
                transform = np.eye(len(sectors));transform[:, 0] = 1.
                total_beta = transform@beta[components, j]
                total_cov, total_hc0 = transform@component_cov@transform.T, transform@component_hc0@transform.T
                for i, sector in enumerate(sectors):
                    unit = sample.loc[sample.sector_id.eq(sector)]
                    records["sector_estimates"].append({**band_meta, "sector_id": sector, "sector": names[sector],
                        "term": "total_temperature_effect", "temp_main": float(beta[components[0], j]),
                        "interaction_delta": 0. if i == 0 else float(beta[components[i], j]), "delta_is_normalization": i == 0,
                        "var_temp_main": float(component_cov[0, 0]), "var_interaction_delta": 0. if i == 0 else float(component_cov[i, i]),
                        "cov_temp_main_delta": 0. if i == 0 else float(component_cov[0, i]),
                        "mean_y": float(unit[outcome].mean()), "sd_y": float(unit[outcome].std()),
                        **inference(total_beta[i], total_cov[i, i], degrees, total_hc0[i, i])})
                    records["interaction_components"].append({**band_meta, "sector_id": sector, "sector": names[sector],
                        "term": selected_names[i+2], "component": "base" if i == 0 else "interaction",
                        "estimated": True, "normalization": False,
                        **inference(beta[components[i], j], component_cov[i, i], degrees, component_hc0[i, i])})
                records["interaction_components"].append({**band_meta, "sector_id": sectors[0], "sector": names[sectors[0]],
                    "term": f"temp_c:C(sector_id)[T.{sectors[0]}]", "component": "interaction", "estimated": False,
                    "normalization": True, "coefficient": 0., "variance": None, "std_error": None, "ci_lower": None,
                    "ci_upper": None, "p_value": None, "inference_status": "Reference interaction normalized to zero; not estimated"})
                store_matrix(records, "component_covariances", band_meta, component_cov, selected_names[2:], component_hc0)
                store_matrix(records, "sector_covariances", band_meta, total_cov, sectors, total_hc0, sectors=True)
                profile_tests(total_beta, total_cov, total_hc0, sectors, names, band_meta,
                              t-sample.season.nunique()-1, records)
                profiles[(outcome, band)] = total_beta
    return {"meta": meta, "beta": beta, "residuals": residual, "mapping": geo["mapping"],
            "selected": selected, "coef_names": coef_names, "profiles": profiles}


def separate_joint(context: Context, parent: pd.DataFrame, sample: pd.DataFrame, variant: str, policy: dict,
                   sample_path: Path, separate: dict[str, dict], records: dict) -> None:
    sectors = sorted(separate)
    if len(sectors) != sample.sector_id.nunique(): return
    names = sample.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    t = sample.time.nunique()
    first = separate[sectors[0]]
    k = first["meta"]["n_parameters"]
    assert all(item["meta"]["n_parameters"] == k for item in separate.values())
    time = sample.drop_duplicates("time").sort_values("time").time.to_numpy()
    for j, outcome in enumerate(OUTCOMES):
        beta = np.asarray([item["beta"][item["selected"][0], j] for item in separate.values()])
        influence = np.column_stack([item["mapping"][item["selected"][0]]*item["residuals"][:, j] for item in separate.values()])
        hc0 = influence.T@influence
        for band in [context.horizon, 2*context.horizon]:
            factor = t/(t-k)
            covariance = covariance_from_scores(influence, time, band, factor)
            meta = {**base_metadata(context, parent, sample, variant, policy, sample_path), "model": "separate", "family": "joint_separate_time_series_profile",
                "fit_group_id": context.key+"__"+variant+"__separate_joint", "spec_id": context.key+"__"+variant+"__separate_joint__"+outcome,
                "outcome": outcome, "n_parameters": k*len(sectors), "time_design_rank": k, "time_residual_df": t-k,
                "residual_df": (t-k)*len(sectors), "reference_sector_id": sectors[0], "covariance": "Calendar HAC",
                "hac_lags": band, "inference_df": t-1, "n_clusters": 0, "finite_sample_correction": factor,
                "finite_sample_correction_formula": "T/(T−K_unit)", "main_inference": band == context.horizon}
            store_matrix(records, "sector_covariances", meta, covariance, sectors, hc0, sectors=True)
            profile_tests(beta, covariance, hc0, sectors, names, meta, t-k, records)


def run_crisis_sensitivity(policy_path: Path = DEFAULT_POLICY) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ["parents", "samples", "designs"]: (OUT / name).mkdir(exist_ok=True)
    policy_source_sha256 = sha(Path(policy_path))
    policy = load_policy(Path(policy_path))
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False)+"\n")
    (OUT / "manifest.json").write_text(json.dumps({"status": "running", "policy_id": policy["policy_id"],
        "claim": "New additional analysis started; no completed estimation or independent-verification claim"}, indent=2)+"\n")
    filenames = ["specifications", "estimates", "sector_estimates", "separate_estimates", "all_coefficients",
                 "coefficient_covariances", "sector_covariances", "interaction_components", "component_covariances",
                 "joint_tests", "pairwise_tests", "unavailable_specs"]
    records = {name: [] for name in filenames}
    protected = {context.source: sha(ROOT / context.source) for context in contexts()}
    protected.update({context.match_template: sha(ROOT / context.match_template) for context in contexts() if context.match_template})
    protected.update({name: sha(ROOT / name) for name in ["data/raw/va_sa_tabledata.json", "data/raw/va_sa_metadata.json",
        "data/raw/va_nsa_tabledata.json", "data/raw/va_nsa_metadata.json"]})
    protected.update({name: sha(ROOT / name) for name in ["outputs/pyfixest/fit_log.csv", "outputs/pyfixest/estimator_manifest.json",
        "outputs/pyfixest/migration_input_reference.json", "outputs/covid_comparison/original_inputs_reference.json"]})
    all_masks, transform_checks, calendar_sets, comparisons = [], [], {}, []
    previous = (native.LOG_DIR, list(native._FIT_LOG), native._API_COUNTER, native._RUN_STARTED)
    native.LOG_DIR = OUT / "pyfixest";native.reset_fit_log()
    try:
        with threadpool_limits(limits=1):
            for context in contexts():
                parent, checks = read_parent(context);transform_checks.extend(checks)
                parent_path = OUT / "parents" / (context.key+".csv")
                parent.to_csv(parent_path, index=False)
                parent.attrs.update(eligible_parent_csv=relative(parent_path), eligible_parent_sha256=sha(parent_path))
                masks = mask_table(context, parent, policy);all_masks.extend(masks.to_dict("records"))
                for variant in VARIANTS:
                    keep = np.ones(len(masks), dtype=bool) if variant == "all_eligible" else masks[
                        "retain_event_dates" if variant == "event_dates_excluded" else "retain_strict"].to_numpy()
                    sample = parent[parent.time.isin(masks.loc[keep, "time"])].copy().reset_index(drop=True)
                    assert not sample.empty and sample.groupby("time").sector_id.nunique().eq(parent.sector_id.nunique()).all()
                    calendar_sets[(context.key, variant)] = set(sample.time)
                    sample_path = OUT / "samples" / (context.key+"__"+variant+".csv")
                    sample.to_csv(sample_path, index=False)
                    common = fit_model(context, parent, sample, variant, policy, sample_path, "common", records)
                    if sample.sector_id.nunique() > 1:
                        interactions = fit_model(context, parent, sample, variant, policy, sample_path, "H1", records)
                        separate = {}
                        for sector, unit in sample.groupby("sector_id", sort=True):
                            result = fit_model(context, parent, unit.reset_index(drop=True), variant, policy, sample_path, "separate", records, sector)
                            if result is not None: separate[sector] = result
                        separate_joint(context, parent, sample, variant, policy, sample_path, separate, records)
                        if common is not None and interactions is not None:
                            for j, outcome in enumerate(OUTCOMES):
                                mean_slope = float(interactions["profiles"][(outcome, context.horizon)].mean())
                                common_slope = float(common["beta"][common["selected"][0], j])
                                assert np.isclose(mean_slope, common_slope, atol=1e-7, rtol=1e-9)
                                comparisons.append({"context": context.key, "sample_variant": variant, "outcome": outcome,
                                    "mean_H1_sector_slope": mean_slope, "pooled_common_slope": common_slope,
                                    "difference": mean_slope-common_slope, "interpretation": "Balanced equal-sector regression identity; not an official aggregate identity"})
                    print(f"Crisis sensitivity {context.key}/{variant}: {sample.sector_id.nunique()} units × {sample.time.nunique()} periods", flush=True)
            native_receipt = native.write_fit_log()
    finally:
        native.LOG_DIR, old_records, native._API_COUNTER, native._RUN_STARTED = previous
        native._FIT_LOG[:] = old_records
    for context in contexts():
        if context.matched_to:
            for variant in VARIANTS:
                assert calendar_sets[(context.key, variant)] == calendar_sets[(context.matched_to, variant)], "Matched taxonomy calendars differ"
    for filename, rows in records.items():
        frame = pd.DataFrame(rows)
        if not len(frame): frame = pd.DataFrame(columns=["spec_id", "status", "reason"])
        frame.to_csv(OUT / f"{filename}.csv", index=False)
    pd.DataFrame(all_masks).to_csv(OUT / "masks.csv", index=False)
    pd.DataFrame(transform_checks).to_csv(OUT / "parent_transform_checks.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "mean_slope_identities.csv", index=False)
    assert all(sha(ROOT / path) == expected for path, expected in protected.items()), "Existing baseline source/receipt changed"
    policy_source = Path(policy_path).resolve()
    manifest = {"status": "estimated_not_yet_independently_verified", "policy_id": policy["policy_id"],
        "policy_source": relative(policy_source) if policy_source.is_relative_to(ROOT) else str(policy_source),
        "policy_source_sha256": policy_source_sha256, "policy_sha256": sha(OUT / "policy.json"), "contexts": len(contexts()),
        "sample_variants": VARIANTS, "samples": len(calendar_sets), "requested_specs": len(records["specifications"]),
        "native_fits": native_receipt["native_fits_recorded"], "unavailable_specs": len(records["unavailable_specs"]),
        "row_counts": {name: len(rows) for name, rows in records.items()}, "protected_sources_sha256": protected,
        "strict_mask": "Every calendar period from t−h through t inclusive must avoid all configured event/rebound blocks. SAME YoY-safe mask used for all five outcomes.",
        "calendar": "Original period/time indices preserved; omitted dates contribute zero HAC scores, never compressed adjacency",
        "controls": TERMS, "outcomes": OUTCOMES, "levels_role": "Descriptive, not a long-run growth or causal interpretation",
        "stationarity": "No ADF/KPSS sample trimming and no new clearance; exclusions create gapped estimation histories",
        "inference": "Pooled common slopes: sector CR1 plus calendar DK h/2h. National: calendar HAC. H1 profiles: joint DK; separate profiles: joint own-series calendar HAC.",
        "degeneracy": {"zero_relative_tolerance": ZERO_RELATIVE_TOLERANCE, "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
                       "reference": "Unscaled HC0 coefficient/contrast covariance is only a numerical scale reference, not reported alternative inference"},
        "method_sources_read": ["notes/heterogeneity_review.md", "notes/granular_interactions_methods.md", "notes/aea_temperature_literature.md"],
        "interpretation": policy["interpretation"]}
    (OUT / "manifest.json").write_text(json.dumps(clean(manifest), indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({key: manifest[key] for key in ["status", "contexts", "samples", "requested_specs", "native_fits", "unavailable_specs"]}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="Calendar policy JSON; defaults to the explicitly labelled Singapore event/rebound preset")
    args = parser.parse_args()
    run_crisis_sensitivity(args.policy.resolve())
