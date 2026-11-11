"""Pair sector FE and own-series estimates before/after COVID-date exclusion.

This module writes a separate analysis family. It never changes the eligibility
rules, input estimates or original reports. Growth and lags already exist in the
saved source samples; exclusion removes dated observations after construction.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

for _thread_name in ["OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_thread_name, "1")

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import pyfixest_backend as backend
from heterogeneity import (ADDED, BASE, ESTIMATOR_META, PAIRS, component_records,
                           covariance, inference, project, wald)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/covid_comparison/sector"
VARIANTS = ["all_eligible", "exclude_2020_2022"]
BROAD_OUTCOMES = ["y_short", "y_yoy_nsa", "y_yoy", "y_log_level"]
DETAILED_OUTCOMES = ["y_short", "y_yoy_nsa", "y_yoy", "y_log_level", "y_raw_level"]
Y_MODELS = {"y_short": "G1", "y_yoy": "G2", 'y_yoy_nsa':'G2N', "y_log_level": "L1", "y_raw_level": "L3"}


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def identifier(prefix: str, *parts: str) -> str:
    content = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix+hashlib.sha256(content.encode()).hexdigest()[:20]


@dataclass(frozen=True)
class Group:
    context: str
    frequency: str
    control_set: str
    model: str
    family: str
    source_sample: Path
    source_csv: Path
    outcomes: tuple[str, ...]
    source_kind: str
    legacy_outcome: str = ""

    @property
    def period(self) -> str:
        return "month" if self.frequency == "monthly" else "quarter"

    @property
    def horizon(self) -> int:
        return 12 if self.frequency == "monthly" else 4

    @property
    def terms(self) -> list[str]:
        return BASE+(ADDED if self.control_set == "full" else [])

    def source_model(self, outcome: str) -> str:
        if self.source_kind == "detailed_TS":
            return Y_MODELS[outcome]
        return "separate" if self.family == "separate_series" else self.model


def inventory() -> list[Group]:
    groups = []
    for context, frequency in [("va", "quarterly"), ("iip-monthly", "monthly"), ("iip-quarterly", "quarterly")]:
        for controls in ["full", "weather"]:
            folder = ROOT / "outputs/heterogeneity" / context / controls
            for model in ["H1", "H2", "H3", "R1"]:
                groups.append(Group(context, frequency, controls, model, "sector_FE_"+model,
                                    folder/"common_sample.csv", folder/"sector_slopes.csv", tuple(BROAD_OUTCOMES), "broad_FE"))
            groups.append(Group(context, frequency, controls, "separate", "separate_series",
                                folder/"common_sample.csv", folder/"separate_sector_coefficients.csv", tuple(BROAD_OUTCOMES), "broad_TS"))
    for context, frequency in [("va-detailed", "quarterly"), ("iip-monthly-detailed", "monthly"), ("iip-quarterly-detailed", "quarterly")]:
        for controls in ["full", "long"]:
            sample = ROOT / "outputs/granular" / context / controls / "samples/G1.csv"
            source = ROOT / "outputs/granular_interactions" / context / controls / "sector_slopes.csv"
            groups.append(Group(context, frequency, controls, "H1", "sector_FE_H1", sample, source, tuple(DETAILED_OUTCOMES), "detailed_FE"))
            source = ROOT / "outputs/granular" / context / controls / "separate_sector_coefficients.csv"
            groups.append(Group(context, frequency, controls, "separate", "separate_series", sample, source, tuple(DETAILED_OUTCOMES), "detailed_TS"))
    for context, frequency, folder, outcome in [
        ("va", "quarterly", ROOT/"outputs", "g_va_sa_qoq"),
        ("iip-monthly", "monthly", ROOT/"outputs/iip/monthly", "g_iip_sa"),
        ("iip-quarterly", "quarterly", ROOT/"outputs/iip/quarterly", "g_iip_sa"),
    ]:
        groups.append(Group(context, frequency, "weather", "H1", "legacy_sector_FE_H1",
                            folder/"baseline_sample.csv", folder/"sector_slopes.csv", ("y_short",), "legacy_FE", outcome))
    return groups


def read_eligible(group: Group) -> pd.DataFrame:
    sample = pd.read_csv(group.source_sample, dtype={"sector_id": str})
    sample[group.period] = pd.PeriodIndex(sample[group.period], freq="M" if group.frequency == "monthly" else "Q")
    if group.legacy_outcome:
        sample["y_short"] = sample[group.legacy_outcome]
    sample = sample.sort_values(["time", "sector_id"]).reset_index(drop=True)
    assert not sample.duplicated(["sector_id", group.period]).any(), group
    assert sample.groupby(group.period).sector_id.nunique().eq(sample.sector_id.nunique()).all(), group
    assert sample[list(group.outcomes)+group.terms].notna().all().all(), group
    return sample


def pair_metadata(group: Group, outcome: str, sample_key: str, eligible: pd.DataFrame) -> dict:
    source_model = group.source_model(outcome)
    source_csv = relative(group.source_csv)
    pair_id = identifier("sector_", source_csv, source_model, group.context, group.control_set, outcome, group.family)
    controls = "rainfall, relative humidity"+(", wind speed, PM2.5" if group.control_set == "full" else "")
    scope = "own-series time-series robustness" if group.family == "separate_series" else "legacy common-controls temperature interactions" if group.source_kind == "legacy_FE" else group.model+" sector FE temperature interactions"
    return {"pair_id": pair_id, "context": group.context, "frequency": group.frequency,
            "family": group.family, "model": group.model, "source_kind": group.source_kind,
            "source_csv": source_csv, "source_model": source_model, "source_outcome": group.legacy_outcome or outcome,
            "source_sample_csv": relative(group.source_sample), "sample_key": sample_key,
            "reference_sector_id": "" if group.family == "separate_series" else sorted(eligible.sector_id.unique())[0],
            "weather_inference_csv": relative(OUT/"weather_coefficients.csv") if group.family == "separate_series" else "",
            "outcome": outcome, "description": group.context+" · "+scope+" · "+outcome,
            "controls": controls, "control_set": group.control_set,
            "control_role": "Baseline · includes wind and PM2.5" if group.control_set == "full" else "Long-history robustness · no wind/PM2.5",
            "all_sample_csv": relative(OUT/"samples"/(sample_key+"__all_eligible.csv")),
            "exclude_sample_csv": relative(OUT/"samples"/(sample_key+"__exclude_2020_2022.csv")),
            "eligible_n_obs": len(eligible), "eligible_n_periods": int(eligible[group.period].nunique()),
            "eligible_n_sectors": int(eligible.sector_id.nunique()),
            "covid_periods_in_eligible": int(eligible.loc[eligible.year.between(2020, 2022), group.period].nunique()),
            "exclusion_rule": "Drop observations dated 2020–2022 after constructing Y and weather; retain calendar indices and eligibility mask",
            "stationarity_scope": "No new stationarity clearance; exclusion can leave calendar gaps", **ESTIMATOR_META}


def aliases(group: Group, meta: dict) -> list[dict]:
    base = {"pair_id": meta["pair_id"], "outcome": meta["outcome"], "source_csv": meta["source_csv"],
            "source_model": meta["source_model"], "source_outcome": "" if group.source_kind == "legacy_FE" else meta["outcome"],
            "sample_variant": "all_eligible", "term": "temp_c",
            "source_term_filter": "temp_c" if group.family == "separate_series" else ""}
    result = [base]
    if group.source_kind == "broad_TS":
        result.append({**base, "source_csv": relative(group.source_csv.parent/"separate_series.csv"), "term": "temperature_only_alias", "source_term_filter": ""})
    if group.source_kind == "broad_FE" and group.model == "H3":
        result.append({**base, "source_model": "H3X", "sample_variant": "exclude_2020_2022"})
    if group.control_set == "full" and group.source_kind in {"detailed_FE", "detailed_TS"}:
        result.append({**base, "source_csv": relative(group.source_csv.parent.parent/"exclude"/group.source_csv.name), "sample_variant": "exclude_2020_2022"})
    return result


def observation_meta(group: Group, pair: dict, sample: pd.DataFrame, eligible: pd.DataFrame,
                     variant: str, bandwidth: int, *, n_parameters: int, n_sectors: int,
                     r_squared: float) -> dict:
    periods = int(sample[group.period].nunique())
    return {**pair, "sample_variant": variant,
            "sample_csv": relative(OUT/"samples"/(pair["sample_key"]+"__"+variant+".csv")),
            "covariance": "Driscoll-Kraay" if group.family != "separate_series" else "Calendar HAC",
            "hac_lags": bandwidth, "inference_df": periods-1, "n_clusters": 0,
            "n_obs": len(sample), "n_sectors": n_sectors, "n_periods": periods,
            "n_parameters": n_parameters, "r_squared": r_squared,
            "start": str(sample[group.period].min()), "end": str(sample[group.period].max()),
            "n_removed_periods": int(eligible[group.period].nunique()-periods),
            "n_removed_obs": len(eligible)-len(sample), "main_inference": bandwidth == group.horizon}


def panel_results(group: Group, sample: pd.DataFrame, eligible: pd.DataFrame,
                  variant: str, pairs: dict[str, dict], records: dict[str, list]) -> None:
    result = project(sample, group.terms, group.model, outcomes=list(group.outcomes),
                     label=f"COVID sector {group.context}/{group.control_set}/{group.model}/{variant}")
    names = sample.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    sectors = sorted(names)
    reference = result["reference_sector_id"]
    for j, outcome in enumerate(group.outcomes):
        pair = pairs[outcome]
        for term, coefficient in zip(result["nuisance_names"]+result["component_names"],
                                     np.r_[result["nuisance_beta"][:, j], result["component_beta"][:, j]], strict=True):
            records["all_coefficients"].append({**pair, "sample_variant": variant, "unit_id": "panel",
                                                 "term": term, "coefficient": float(coefficient), "estimated": True})
        for bandwidth in [group.horizon, 2*group.horizon]:
            cov = covariance(result, sample, j, bandwidth)
            beta = result["beta"][:, j]
            meta = observation_meta(group, pair, sample, eligible, variant, bandwidth,
                                    n_parameters=result["k"], n_sectors=len(sectors), r_squared=float(result["r_squared"][j]))
            component_rows, component_cov = component_records(result, cov, j, meta, names)
            records["interaction_components"].extend(component_rows)
            records["component_covariances"].extend({**pair, "sample_variant": variant, **row} for row in component_cov)
            for i, sector in enumerate(result["sectors"]):
                unit = sample.loc[sample.sector_id.eq(sector)]
                if group.model == "R1":
                    main, delta = np.nan, float(result["component_beta"][i, j])
                else:
                    main = float(result["component_beta"][0, j])
                    delta = 0.0 if i == 0 else float(result["component_beta"][i, j])
                records["estimates"].append({**meta, "sector_id": sector, "sector": names[sector],
                    "reference_sector_id": reference, "term": "relative_temperature_effect" if group.model == "R1" else "total_temperature_effect",
                    "temp_main": main, "interaction_delta": delta, "delta_is_normalization": group.model != "R1" and i == 0,
                    "mean_y": float(unit[outcome].mean()), "sd_y": float(unit[outcome].std()),
                    **inference(beta[i], cov[i, i], result["t"])})
                for k, other in enumerate(result["sectors"]):
                    records["slope_covariances"].append({**pair, "sample_variant": variant, "hac_lags": bandwidth,
                                                           "sector_i": sector, "sector_j": other, "covariance": float(cov[i, k])})
            equality = np.eye(len(beta)) if group.model == "R1" else np.eye(len(beta))[1:]-np.eye(len(beta))[[0]]
            flexible_df = result["t"]-(sample.season.nunique()+sample.year.nunique()-1+len(group.terms))
            if group.model in {"H3", "R1"}:
                bound = int(flexible_df)
            elif group.model == "H1":
                bound = int(result["t"]-sample.season.nunique()-1)
            else:
                bound = result["t"]-1
            records["joint_tests"].append({**meta, "reference_sector_id": reference,
                "test": "All sector temperature slopes equal", **wald(beta, cov, equality, result["t"], rank_bound=bound)})
            chosen = list(combinations(sectors, 2)) if group.source_kind == "detailed_FE" else PAIRS["va" if group.context == "va" else "iip"]
            family = []
            for first, second in chosen:
                weights = np.zeros(len(beta))
                for sector, weight in [(first, 1), (second, -1)]:
                    if sector in result["sectors"]:
                        weights[result["sectors"].index(sector)] = weight
                family.append({**meta, "first_sector_id": first, "second_sector_id": second,
                    "comparison": names[first]+" − "+names[second],
                    **inference(float(weights@beta), float(weights@cov@weights), result["t"])})
            adjusted = multipletests([row["p_value"] for row in family], method="holm")[1]
            for row, p_value in zip(family, adjusted, strict=True):
                row.update(p_value_holm=float(p_value), multiplicity_family=f"{len(family)} displayed sector pairs within pair_id/sample_variant/bandwidth")
            records["pairwise_tests"].extend(family)


def series_results(group: Group, sample: pd.DataFrame, eligible: pd.DataFrame,
                   variant: str, pairs: dict[str, dict], records: dict[str, list]) -> None:
    for sector, unit in sample.groupby("sector_id", sort=True):
        unit = unit.reset_index(drop=True)
        original_unit = eligible.loc[eligible.sector_id.eq(sector)]
        weather = unit[group.terms]-unit[group.terms].mean()
        x = np.column_stack([np.ones(len(unit)), weather,
                             pd.get_dummies(unit.season, drop_first=True, dtype=float),
                             pd.get_dummies(unit.year, drop_first=True, dtype=float)])
        names = (["Intercept"]+group.terms+[f"season[{season}]" for season in sorted(unit.season.unique())[1:]]
                 +[f"year[{year}]" for year in sorted(unit.year.unique())[1:]])
        native = backend.fit_many_matrix(unit[list(group.outcomes)].to_numpy(), x, names,
                    f"COVID own series {group.context}/{group.control_set}/{sector}/{variant}", outcome_names=group.outcomes)
        inverse = native[0].normalized_cov_params
        time = (unit.time-unit.time.min()).to_numpy(dtype=int)
        for j, outcome in enumerate(group.outcomes):
            fit, pair = native[j], pairs[outcome]
            for term, coefficient in fit.params.items():
                records["all_coefficients"].append({**pair, "sample_variant": variant, "unit_id": sector,
                                                     "term": term, "coefficient": float(coefficient), "estimated": True})
            for bandwidth in [group.horizon, 2*group.horizon]:
                scores = np.zeros((time.max()+1, x.shape[1]))
                scores[time] = x*np.asarray(fit.resid)[:, None]
                meat = scores.T@scores
                for lag in range(1, bandwidth+1):
                    moment = scores[lag:].T@scores[:-lag]
                    meat += (1-lag/(bandwidth+1))*(moment+moment.T)
                cov = inverse@meat@inverse*len(unit)/(len(unit)-x.shape[1])
                meta = observation_meta(group, pair, unit, original_unit, variant, bandwidth,
                                        n_parameters=x.shape[1], n_sectors=1, r_squared=float(fit.rsquared))
                temperature = float(fit.params["temp_c"])
                result = {**meta, "sector_id": sector, "sector": unit.sector.iloc[0], "reference_sector_id": "",
                    "term": "total_temperature_effect", "temp_main": temperature, "interaction_delta": np.nan,
                    "delta_is_normalization": False, "mean_y": float(unit[outcome].mean()), "sd_y": float(unit[outcome].std()),
                    "residual_df": len(unit)-x.shape[1], **inference(temperature, cov[1, 1], len(unit))}
                records["estimates"].append(result)
                for index, term in enumerate(group.terms, start=1):
                    records["weather_coefficients"].append({**meta, "sector_id": sector, "sector": unit.sector.iloc[0],
                        "reference_sector_id": "", "term": term, "residual_df": len(unit)-x.shape[1],
                        "mean_y": float(unit[outcome].mean()), "sd_y": float(unit[outcome].std()),
                        **inference(float(fit.params[term]), cov[index, index], len(unit))})
                records["interaction_components"].append({**result, "term": "temp_c", "component": "base", "estimated": True,
                    "normalization": False, "temperature_center_c": float(unit.temp_c.mean()),
                    "estimand": "Own-series temperature coefficient; no cross-sector interaction"})
                records["component_covariances"].append({**pair, "sample_variant": variant, "hac_lags": bandwidth,
                    "reference_sector_id": "", "term_i": "temp_c", "term_j": "temp_c", "component_i": "base", "component_j": "base",
                    "sector_i": sector, "sector_j": sector, "covariance": float(cov[1, 1])})


def run_covid_sector() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"samples").mkdir(exist_ok=True)
    (OUT/"run_status.json").write_text(json.dumps({"status": "running"})+"\n")
    old_state = (backend.LOG_DIR, list(backend._FIT_LOG), backend._API_COUNTER, backend._RUN_STARTED)
    backend.LOG_DIR = OUT/"pyfixest"
    records = {name: [] for name in ["estimates", "joint_tests", "pairwise_tests", "interaction_components",
                                    "component_covariances", "slope_covariances", "all_coefficients", "weather_coefficients"]}
    metadata, source_aliases, saved_samples, source_hashes = [], [], {}, {}
    try:
        backend.reset_fit_log()
        groups = inventory()
        for group in groups:
            eligible = read_eligible(group)
            sample_key = identifier("sample_", relative(group.source_sample), group.context, group.control_set)
            if sample_key not in saved_samples:
                samples = {"all_eligible": eligible,
                           "exclude_2020_2022": eligible.loc[~eligible.year.between(2020, 2022)].reset_index(drop=True)}
                for variant, frame in samples.items():
                    frame.to_csv(OUT/"samples"/(sample_key+"__"+variant+".csv"), index=False)
                saved_samples[sample_key] = samples
            else:
                samples = saved_samples[sample_key]
                assert samples["all_eligible"].equals(eligible), "Shared sample key has different observations"
            pairs = {outcome: pair_metadata(group, outcome, sample_key, eligible) for outcome in group.outcomes}
            metadata.extend(pairs.values())
            for pair in pairs.values():
                source_aliases.extend(aliases(group, pair))
            for source in [group.source_sample, group.source_csv]:
                source_hashes[relative(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            for variant, sample in samples.items():
                if group.family == "separate_series":
                    series_results(group, sample, eligible, variant, pairs, records)
                else:
                    panel_results(group, sample, eligible, variant, pairs, records)
            print(f"COVID sector {group.context}/{group.control_set}/{group.model}: {eligible[group.period].nunique()} → {samples['exclude_2020_2022'][group.period].nunique()} dates; {len(group.outcomes)} Y definitions", flush=True)
        tables = {name: pd.DataFrame(rows) for name, rows in records.items()}
        assert not tables["estimates"].duplicated(["pair_id", "sample_variant", "sector_id", "hac_lags"]).any()
        assert len({row["pair_id"] for row in metadata}) == len(metadata)
        for name, frame in tables.items():
            frame.to_csv(OUT/(name+".csv"), index=False)
        pd.DataFrame(metadata).to_csv(OUT/"metadata.csv", index=False)
        pd.DataFrame(source_aliases).to_csv(OUT/"source_aliases.csv", index=False)
        receipt = backend.write_fit_log()
        summary = {"status": "estimation_complete_pending_independent_verification", "canonical_pairs": len(metadata),
                   "specification_groups": len(groups), "sample_families": len(saved_samples),
                   "estimate_rows": len(tables["estimates"]), "joint_test_rows": len(tables["joint_tests"]),
                   "weather_inference_rows": len(tables["weather_coefficients"]),
                   "pairwise_test_rows": len(tables["pairwise_tests"]), "native_fits": receipt["native_fits_recorded"],
                   "native_api_calls": receipt["native_api_calls"], "source_hashes": source_hashes,
                   "eligibility": "Saved source eligibility masks; only observation dates in 2020–2022 are dropped",
                   "growth": "Constructed in full source histories before exclusion; a 2023 YoY observation can retain a 2022 base",
                   "inference": "Sector-specific FE slopes use joint calendar DK; own series use their own-K calendar HAC; no individual-sector CR1",
                   "rank": "Singular joint restriction covariances retain status/rank and unavailable omnibus p-values",
                   "multiplicity": "Broad economic pair families or all detailed pairs; Holm separately within pair/sample variant/bandwidth",
                   "baseline": "Full controls include rainfall, humidity, wind and PM2.5; long history and COVID exclusion are robustness", **ESTIMATOR_META}
        (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        (OUT/"run_status.json").write_text(json.dumps({"status": "complete"})+"\n")
        return summary
    except Exception as error:
        (OUT/"run_status.json").write_text(json.dumps({"status": "failed", "error": repr(error)})+"\n")
        pd.DataFrame(backend._FIT_LOG).to_csv(OUT/"pyfixest/fit_log.csv", index=False)
        raise
    finally:
        backend.LOG_DIR, previous_log, backend._API_COUNTER, backend._RUN_STARTED = old_state
        backend._FIT_LOG[:] = previous_log


if __name__ == "__main__":
    run_covid_sector()
