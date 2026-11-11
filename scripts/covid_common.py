"""Pair every ordinary saved FE specification with its endpoint-year exclusion.

Uses the original model eligibility before masking 2020–2022.  Source outcomes,
annual bases, lagged exposures and original calendar indices are never rebuilt.
All coefficient fits are native PyFixest through the shared backend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum

import pyfixest_backend as backend
from estimate import estimates
from sector_cluster import sector_cluster_covariance
from verify_sector_cluster import exact_sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/covid_comparison/common"
VARIANTS = ["all_eligible", "exclude_2020_2022"]


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[Path, pd.DataFrame]:
    paths = sorted(path for path in (ROOT / "outputs").rglob("*.csv")
                   if "covid_comparison" not in path.parts and "visitor_arrivals" not in path.parts
                   and (path.name in {"regression_results.csv", "distributed_lag_results.csv"}
                        or path.name == "aggregate_results.csv" and path.parent.parent.name == "iip"))
    sources = {path: pd.read_csv(path) for path in paths}
    for path, frame in sources.items():
        assert {"model", "formula", "term", "covariance", "hac_lags"}.issubset(frame), path
        assert not frame.empty
        assert frame.groupby("model").formula.nunique().eq(1).all(), path
    return sources


def _canonical(path: Path, model: str, sources: dict) -> tuple[Path, str, list[str]]:
    reasons = []
    if "granular" in path.parts and path.parent.name == "exclude":
        path = path.parent.parent / "full" / path.name
        assert path in sources
        reasons.append("granular exclude window maps to its full-eligibility sibling")
    models = set(sources[path].model)
    if model.endswith("X") and model[:-1] in models:
        model = model[:-1]
        reasons.append("endpoint-excluded X alias maps to base model")
    if model == "C5" and "C4" in models:
        model = "C4"
        reasons.append("C5 endpoint exclusion maps to C4")
    if model == "S5" and "S3" in models:
        model = "S3"
        reasons.append("S5 endpoint exclusion maps to S3")
    return path, model, reasons


def _sample(path: Path, model: str) -> pd.DataFrame:
    if path.name == "aggregate_results.csv":
        frequency = path.parent.name
        frame = pd.read_csv(ROOT / f"data/processed/iip/aggregate_{frequency}.csv", dtype={"sector_id": str})
        broad = pd.read_csv(ROOT / f"data/processed/iip/panel_{frequency}_broad.csv")
        period = "month" if frequency == "monthly" else "quarter"
        frame = frame.loc[frame[period].between(broad[period].min(), broad[period].max())].copy()
        assert frame.sector_id.eq("1").all()
        return frame
    return exact_sample(path, model)


def _inventory(sources: dict) -> tuple[list[dict], pd.DataFrame]:
    specs, by_key, aliases = [], {}, []
    for path, frame in sources.items():
        for model, group in frame.groupby("model", sort=False):
            canonical_path, canonical_model, reasons = _canonical(path, str(model), sources)
            key = (canonical_path, canonical_model)
            source = sources[canonical_path].loc[sources[canonical_path].model.eq(canonical_model)]
            formula = source.formula.iloc[0]
            if key not in by_key:
                identity = json.dumps([_relative(canonical_path), canonical_model, formula], ensure_ascii=False, separators=(",", ":"))
                spec = {"pair_id": "p"+hashlib.sha256(identity.encode()).hexdigest()[:16],
                        "source_csv": _relative(canonical_path), "source_model": canonical_model,
                        "formula": formula, "outcome": formula.split("~", 1)[0].strip(),
                        "source_label": str(source.label.iloc[0]),
                        "source_covariances": sorted(source.covariance.unique().tolist()),
                        "source_rows": int(len(source))}
                by_key[key] = spec
                specs.append(spec)
            spec = by_key[key]
            for index, row in group.iterrows():
                aliases.append({"source_csv": _relative(path), "source_model": model,
                                "source_row": int(index), "source_term": row.term,
                                "source_covariance": row.covariance, "source_hac_lags": int(row.hac_lags),
                                "source_formula": row.formula, "pair_id": spec["pair_id"],
                                "canonical_source_csv": spec["source_csv"], "canonical_source_model": canonical_model,
                                "alias_reason": "; ".join(reasons) or "original specification retained",
                                "source_is_endpoint_excluded": bool(reasons),
                                "source_is_pre_2020": model == "S4"})
    assert len({spec["pair_id"] for spec in specs}) == len(specs), "Pair-id collision"
    assert len(aliases) == sum(len(frame) for frame in sources.values())
    return specs, pd.DataFrame(aliases)


def _terms(model, sample: pd.DataFrame) -> list[str]:
    # Bare numeric columns are the common weather/exposure variables and any
    # explicitly specified changing-seasonality slopes.  FE dummy rows stay out.
    terms = [name for name in model.params.index if name in sample.columns
             and pd.api.types.is_numeric_dtype(sample[name])]
    assert terms and all("sector_id" not in name for name in terms)
    return terms


def _temperature(name: str) -> bool:
    return name.startswith("temp_") or name.startswith("temperature_")


def _contrasts(spec: dict, terms: list[str]) -> dict[str, list[str]]:
    temperature = [name for name in terms if _temperature(name)]
    if (spec["source_model"] == "L1" and spec["source_csv"].endswith("distributed_lag_results.csv")
            or spec["source_model"] == "D1" and "/distributed_lag/" in spec["source_csv"]):
        assert len(temperature) > 1
        return {"sum_temperature_lags": temperature}
    if spec["source_model"] in {"Y4","Y4N"} and "temp_c" in terms and "temp_year_lag" in terms:
        return {"sum_temperature_current_and_annual_lag": ["temp_c", "temp_year_lag"]}
    return {}


def _estimate(sample: pd.DataFrame, spec: dict, period: str, horizon: int) -> tuple[list[dict], list[str], int, int]:
    model = backend.fit(sample, spec["formula"], f"COVID pair {spec['pair_id']} {sample[period].min()}–{sample[period].max()}")
    terms = _terms(model, sample)
    rows = []
    for bandwidth in [horizon, 2*horizon]:
        rows += estimates(model, sample, spec["source_model"], spec["source_label"], spec["formula"],
                          terms, lags=bandwidth, period_col=period)
    # Keep any original time-cluster sensitivity, in addition to the new main
    # unit-CR1 and h/2h calendar-HAC rows.
    for covariance in spec["source_covariances"]:
        if covariance in {"Quarter cluster", "Month cluster"}:
            extra = estimates(model, sample, spec["source_model"], spec["source_label"], spec["formula"],
                              terms, lags=0, covariance="Quarter cluster", period_col=period)
            for row in extra:
                row["covariance"] = covariance
            rows += extra
    for name, contrast_terms in _contrasts(spec, terms).items():
        positions = [model.model.exog_names.index(term) for term in contrast_terms]
        beta = float(model.params[contrast_terms].sum())
        for covariance, bandwidth in [("Driscoll-Kraay", horizon), ("Driscoll-Kraay", 2*horizon)] + ([("Sector cluster", 0)] if sample.sector_id.nunique() >= 2 else []):
            if covariance == "Sector cluster":
                matrix, _ = sector_cluster_covariance(model, sample.sector_id)
            else:
                time = (sample.time - sample.time.min()).to_numpy(dtype=int)
                matrix = cov_nw_groupsum(model, bandwidth, time, use_correction="cluster")
            template = next(row for row in rows if row["covariance"] == covariance and row["hac_lags"] == bandwidth)
            se = float(np.sqrt(matrix[np.ix_(positions, positions)].sum()))
            dof = template["inference_df"]
            critical = student_t.ppf(.975, dof)
            rows.append({**template, "term": name, "coefficient": beta, "std_error": se,
                         "ci_lower": beta-critical*se, "ci_upper": beta+critical*se,
                         "p_value": float(2*student_t.sf(abs(beta/se), dof)),
                         "term_kind": "linear_contrast", "contrast_terms": json.dumps(contrast_terms)})
    # Restrict the complete explicit design to one unit. Its rank is the
    # effective national-time design size, including changing seasonality.
    first_sector = sample.sector_id.iloc[0]
    time_rank = int(np.linalg.matrix_rank(model.model.exog[sample.sector_id.eq(first_sector).to_numpy()]))
    return rows, terms, model.model.exog.shape[1], time_rank


def run_covid_common() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    sources = _sources()
    source_hashes = {_relative(path): _sha(path) for path in sources}
    original_receipts = {path: _sha(path) for path in [ROOT / "outputs/pyfixest/fit_log.csv", ROOT / "outputs/pyfixest/estimator_manifest.json"] if path.exists()}
    specs, aliases = _inventory(sources)
    aliases.to_csv(OUT / "aliases.csv", index=False)
    # Preserve both the original disk receipt and the original in-memory ledger.
    old_state = (backend.LOG_DIR, list(backend._FIT_LOG), backend._API_COUNTER, backend._RUN_STARTED)
    all_rows, completed = [], []
    try:
        backend.LOG_DIR = OUT / "pyfixest"
        backend.reset_fit_log()
        for number, spec in enumerate(specs, 1):
            path = ROOT / spec["source_csv"]
            frame = _sample(path, spec["source_model"]).copy()
            source_rows = sources[path].loc[sources[path].model.eq(spec["source_model"])]
            monthly = "n_months" in source_rows and source_rows.n_months.notna().any()
            frequency, period, freq, horizon = ("monthly", "month", "M", 12) if monthly else ("quarterly", "quarter", "Q", 4)
            frame[period] = pd.PeriodIndex(frame[period], freq=freq)
            frame = frame.sort_values(["sector_id", "time"]).reset_index(drop=True)
            assert not frame.duplicated(["sector_id", period]).any()
            assert (frame[period].dt.year.to_numpy() == frame.year.to_numpy()).all()
            offset = frame.time.to_numpy() - frame[period].astype("int64").to_numpy()
            assert np.unique(offset).size == 1, "Source time index is not calendar faithful"
            sectors, periods = frame.sector_id.nunique(), frame[period].nunique()
            assert frame.groupby(period).sector_id.nunique().eq(sectors).all()
            assert len(frame) == sectors*periods
            if path.name == "aggregate_results.csv":
                assert sectors == 1
            # This matches the unexcluded source entry, not its historical X/S5/C5 alias.
            assert source_rows.n_obs.dropna().eq(len(frame)).all(), (spec["source_csv"], spec["source_model"])
            outcome = spec["outcome"]
            assert outcome in frame and frame[outcome].notna().all()
            excluded = frame.loc[~frame.year.between(2020, 2022)].copy().reset_index(drop=True)
            assert not excluded.empty
            removed_periods = periods-excluded[period].nunique()
            context = ("iip-monthly" if monthly else "iip-quarterly") if "iip" in spec["source_csv"] else "va"
            family = "pooled_FE" if sectors >= 2 else "national"
            eligible_rows, terms, k, eligible_time_rank = _estimate(frame, spec, period, horizon)
            controls = [name for name in terms if not _temperature(name)]
            sample_stats = []
            for variant, sample in [(VARIANTS[0], frame), (VARIANTS[1], excluded)]:
                reused = variant == VARIANTS[1] and removed_periods == 0
                if variant == VARIANTS[0] or reused:
                    fitted, variant_k, time_rank = eligible_rows, k, eligible_time_rank
                else:
                    fitted, variant_terms, variant_k, time_rank = _estimate(sample, spec, period, horizon)
                    assert variant_terms == terms
                sample_path = OUT / "samples" / f"{spec['pair_id']}__{variant}.csv"
                sample.to_csv(sample_path, index=False)
                stat = {"sample_variant": variant, "sample_csv": _relative(sample_path),
                        "n_obs": len(sample), "n_sectors": sectors, "n_periods": sample[period].nunique(),
                        "start": str(sample[period].min()), "end": str(sample[period].max()),
                        "mean_y": float(sample[outcome].mean()), "sd_y": float(sample[outcome].std()),
                        "fit_reused": reused, "n_parameters": int(variant_k), "time_design_rank": time_rank,
                        "time_residual_df": int(sample[period].nunique()-time_rank),
                        "calendar_span_periods": int(sample[period].max().ordinal-sample[period].min().ordinal+1),
                        "calendar_gaps": int(sample[period].max().ordinal-sample[period].min().ordinal+1-sample[period].nunique())}
                sample_stats.append(stat)
                for row in fitted:
                    main = row["covariance"] == "Sector cluster" if sectors >= 2 else row["covariance"] == "Driscoll-Kraay" and row["hac_lags"] == horizon
                    all_rows.append({**row, **stat, "pair_id": spec["pair_id"], "context": context,
                                     "frequency": frequency, "family": family,
                                     "source_csv": spec["source_csv"], "source_model": spec["source_model"],
                                     "outcome": outcome, "description": spec["source_label"],
                                     "controls": json.dumps(controls), "control_set": " + ".join(controls) or "none",
                                     "temperature_terms": json.dumps([name for name in terms if _temperature(name)]),
                                     "season_trend": any(name.startswith("season_trend_") for name in controls),
                                     "n_removed_periods": int(removed_periods), "n_removed_obs": len(frame)-len(excluded),
                                     "identical_pair": removed_periods == 0, "main_inference": bool(main),
                                     "term_kind": row.get("term_kind", "coefficient"),
                                     "contrast_terms": row.get("contrast_terms", "[]"),
                                     "n_clusters": row.get("n_clusters", np.nan),
                                     "se": row["std_error"]})
            completed.append({**spec, "context": context, "frequency": frequency, "period_column": period,
                              "horizon": horizon, "family": family, "terms": terms, "controls": controls,
                              "control_set": " + ".join(controls) or "none", "n_removed_periods": int(removed_periods),
                              "identical_pair": removed_periods == 0, "source_n_parameters": int(k),
                              "source_eligibility": "Exact saved source-model eligibility, before the added endpoint-year mask",
                              "samples": sample_stats})
            if number % 20 == 0 or number == len(specs):
                print(f"COVID ordinary FE pairs: {number}/{len(specs)}", flush=True)
        native_inventory = backend.write_fit_log()
    finally:
        backend.LOG_DIR = old_state[0]
        backend._FIT_LOG[:] = old_state[1]
        backend._API_COUNTER, backend._RUN_STARTED = old_state[2:]
    result = pd.DataFrame(all_rows)
    assert not result.duplicated(["pair_id", "sample_variant", "term", "covariance", "hac_lags"]).any()
    assert result.groupby(["pair_id", "sample_variant"]).main_inference.any().all()
    assert set(result.pair_id) == set(aliases.pair_id)
    for path, digest in original_receipts.items():
        assert _sha(path) == digest, f"Original PyFixest receipt changed: {path}"
    for path in sources:
        assert _sha(path) == source_hashes[_relative(path)], f"Original source estimates changed: {path}"
    result.to_csv(OUT / "estimates.csv", index=False)
    (OUT / "specs.json").write_text(json.dumps(completed, indent=2)+"\n")
    inventory = {"status": "produced", "source_files": len(sources), "source_model_rows": sum(frame.model.nunique() for frame in sources.values()),
                 "source_covariance_coefficient_rows": len(aliases), "canonical_pairs": len(specs),
                 "identical_pairs_no_removed_periods": sum(item["identical_pair"] for item in completed),
                 "estimate_rows": len(result), "native_fits": native_inventory["native_fits_recorded"],
                 "source_hashes": source_hashes, "original_estimates_and_receipts_unchanged": True,
                 "variants": VARIANTS, "main_inference": "Sector CR1 for common-slope panels; h-calendar HAC for national series",
                 "comparators": "h and 2h calendar HAC, plus original time-cluster sensitivity where present",
                 "contrasts": "All distributed temperature lags summed with full covariance; current+annual-lag sum for unrestricted Y4",
                 "interpretation": "Exclude observations dated 2020–2022 after outcomes/exposures were precomputed; 2023 YoY can retain a 2022 base",
                 "verification_scope": "Producer checks source coverage/calendar/sample identities; independent numerical verification is separate"}
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    print(f"COVID ordinary FE comparison ready: {len(specs)} pairs, {len(result):,} estimate rows, {len(aliases):,} original-row aliases", flush=True)
    return inventory


if __name__ == "__main__":
    run_covid_common()
