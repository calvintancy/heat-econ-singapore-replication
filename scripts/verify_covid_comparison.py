"""Independent exact-sample and full-QR audit of every paired COVID regression.

This module does not import paired producers, the production estimator, or its
covariance helpers. It checks saved native receipts but estimates its numerical
benchmarks independently with QR and explicit unit/calendar score sums.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for _variable in ["OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.stats import f as f_distribution, t as student_t
from threadpoolctl import threadpool_limits
from provenance_omissions import omitted_artifact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/covid_comparison"
VARIANTS = ("all_eligible", "exclude_2020_2022")
ATOL, RTOL = 1e-7, 1e-7
ID_COLUMNS = {"sector_id", "sector_i", "sector_j", "first_sector_id", "second_sector_id", "reference_sector_id"}
CHECKS: list[dict] = []
NATIVE_EXPECTED: dict[str, list[dict]] = {"common": [], "sector": []}
SOURCE_COVERAGE: list[dict] = []


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_original_inputs_reference(refresh=False, path: Path | None = None) -> dict:
    """Explicit pre-producer checkpoint; verification never calls this itself."""
    path = Path(path) if path is not None else OUT / "original_inputs_reference.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    baseline = json.loads((ROOT / "outputs/pyfixest/migration_input_reference.json").read_text())["files"]
    files = {ROOT / name for name in baseline}
    for family in ["iip", "extended_controls", "outcomes", "heterogeneity", "aggregate_va", "granular", "granular_interactions"]:
        files.update(p for p in (ROOT / "outputs" / family).rglob("*") if p.is_file() and p.suffix in {".csv", ".json"})
    files.update(p for p in (ROOT / "outputs").iterdir() if p.is_file() and p.suffix == ".csv")
    files.update(ROOT / "outputs/pyfixest" / name for name in ["fit_log.csv", "estimator_manifest.json"])
    records, omitted = {}, []
    for source in sorted(files):
        relative = str(source.relative_to(ROOT))
        omission = omitted_artifact(ROOT, relative, baseline.get(relative, {}))
        if omission is not None:
            # Preserve the historical metadata without pretending a missing
            # third-party citation artifact was read or verified in this run.
            records[relative] = {"bytes": baseline[relative]["bytes"], "sha256": baseline[relative]["sha256"]}
            omitted.append(omission)
        else:
            assert source.is_file(), (relative, "Missing analytical input or required provenance manifest")
            records[relative] = {"bytes": source.stat().st_size, "sha256": digest(source)}
    payload = {"captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All available original raw/processed/exact input files captured for the Pyfixest migration, all original analytical-family CSV/JSON files, original root regression CSVs, and original native fit receipt; excludes HTML/report-only rebuilds. Exact absent third-party provenance artifacts are separately listed as omitted/unverified, retaining only their historical recorded metadata.", "files": records}
    payload["provenance_only_omitted_unverified_count"] = len(omitted)
    payload["provenance_only_omitted_unverified"] = omitted
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2)+"\n")
    return payload


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={name: str for name in ID_COLUMNS})


def close(actual, expected, label: str, *, exact=False, atol=ATOL, rtol=RTOL) -> None:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (label, "shape", a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), (label, "unavailable masks")
    assert np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(np.isneginf(a), np.isneginf(b)), (label, "infinite values")
    finite = np.isfinite(a)&np.isfinite(b)
    errors = np.abs(a[finite]-b[finite])
    tolerance = atol+rtol*np.abs(b[finite])
    assert np.all(errors == 0) if exact else np.all(errors <= tolerance), (label, float(errors.max()), float(np.max(errors/tolerance)))
    CHECKS.append({"check": label, "values": int(finite.sum()), "exact": exact,
        "maximum_absolute_error": float(errors.max()) if len(errors) else 0.,
        "maximum_scaled_relative_error": float(np.max(errors/np.maximum(1., np.abs(b[finite])))) if len(errors) else 0.,
        "maximum_fraction_of_tolerance": float(np.max(errors/tolerance)) if len(errors) else 0.})


def frame_equal(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    """Check every canonical source value, permitting explicitly added aliases."""
    period = "month" if "month" in expected else "quarter"
    keys = [period]+(["sector_id"] if "sector_id" in expected else [])
    assert set(expected) <= set(actual), (label, "missing source columns", set(expected)-set(actual))
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    assert len(a) == len(b), (label, len(a), len(b))
    for column in b:
        if pd.api.types.is_numeric_dtype(b[column]) and column not in ID_COLUMNS:
            close(a[column], b[column], label+":"+column, atol=1e-11, rtol=1e-12)
        else:
            assert a[column].fillna("__NA__").astype(str).equals(b[column].fillna("__NA__").astype(str)), (label, column)


def original_input_integrity() -> dict:
    path = OUT / "original_inputs_reference.json"
    baseline = json.loads(path.read_text())
    omitted = []
    for relative, recorded in baseline["files"].items():
        omission = omitted_artifact(ROOT, relative, recorded)
        if omission is not None:
            omitted.append(omission)
            continue
        current = ROOT / relative
        assert current.exists() and current.stat().st_size == recorded["bytes"] and digest(current) == recorded["sha256"], (relative, "original input/result changed")
    return {"referenced_files": len(baseline["files"]), "unchanged_files": len(baseline["files"])-len(omitted),
            "provenance_only_omitted_unverified_count": len(omitted), "provenance_only_omitted_unverified": omitted,
            "baseline_sha256": digest(path), "captured_utc": baseline["captured_utc"], "scope": baseline["scope"],
            "verification_scope": "All analytical inputs/results/receipts and required citation manifests remain hash-checked; only the exact absent third-party provenance artifacts listed here are unverified."}


def original_inventory() -> pd.DataFrame:
    """Enumerate original visible regression rows without producer inventories."""
    rows = []
    baseline = json.loads((OUT / "original_inputs_reference.json").read_text())["files"]
    names = {"regression_results.csv", "distributed_lag_results.csv", "aggregate_results.csv", "sector_slopes.csv", "separate_sector_coefficients.csv", "separate_series.csv"}
    for relative in sorted(baseline):
        path = ROOT / relative
        if path.name not in names or not relative.startswith("outputs/"):
            continue
        frame = read(path)
        for index, row in frame.iterrows():
            model = str(row.get("model", "separate"))
            outcome = str(row.get("outcome", str(row.get("formula", "")).split("~")[0].strip()))
            rows.append({"source_csv": relative, "source_row": int(index), "source_model": model,
                "source_outcome": outcome, "sector_id": str(row.get("sector_id", "")), "term": str(row.get("term", "temp_c")),
                "covariance": str(row.get("covariance", "Driscoll-Kraay")), "hac_lags": int(row.hac_lags),
                "source_family": "sector" if path.name in {"sector_slopes.csv", "separate_sector_coefficients.csv", "separate_series.csv"} else "common"})
    return pd.DataFrame(rows)


def canonical_source(path: Path, model: str) -> tuple[Path, str, str]:
    variant = "all_eligible"
    if "exclude" in path.parts:
        parts = list(path.parts);parts[parts.index("exclude")] = "full"
        path = Path(*parts);variant = "exclude_2020_2022"
    if model in {"S5", "C5", "H3X"}:
        model = {"S5": "S3", "C5": "C4", "H3X": "H3"}[model]
        variant = "exclude_2020_2022"
    elif model.endswith("X"):
        frame = read(path)
        if "model" in frame and model[:-1] in set(frame.model):
            model = model[:-1];variant = "exclude_2020_2022"
    return path, model, variant


def source_sample(path: Path, model: str) -> pd.DataFrame:
    """Independent resolver for the original canonical exact model rows."""
    saved = path.parent / "samples" / f"{model}.csv"
    if saved.exists():
        return read(saved)
    relative = path.relative_to(ROOT / "outputs")
    parts = relative.parts
    if parts[0] in {"outcomes", "heterogeneity", "granular_interactions"}:
        return read(path.parent / "common_sample.csv")
    if parts[0] == "extended_controls":
        if model == "C0":
            context = parts[1]
            source = ROOT / "data/processed/panel_broad.csv" if context == "va" else ROOT / f"data/processed/iip/panel_{'monthly' if context == 'iip-monthly' else 'quarterly'}_broad.csv"
        else:
            source = path.parent / ("sensitivity_80_sample.csv" if model == "C6" else "baseline_sample.csv")
        return read(source)
    if path.name == "aggregate_results.csv":
        period = "month" if parts[1] == "monthly" else "quarter"
        frame = read(ROOT / f"data/processed/iip/aggregate_{parts[1]}.csv")
        baseline = read(path.parent / "baseline_sample.csv")
        return frame[frame[period].between(baseline[period].min(), baseline[period].max())].reset_index(drop=True)
    if len(parts) == 1 or parts[0] == "iip":
        prefix = ROOT / "data/processed"
        source = prefix / "panel_broad.csv" if len(parts) == 1 else prefix / f"iip/panel_{parts[1]}_broad.csv"
        frame = read(source)
        if path.name == "sector_slopes.csv":
            return read(path.parent / "baseline_sample.csv")
        if path.name == "distributed_lag_results.csv":
            h = 12 if "monthly" in parts else 4
            return frame.dropna(subset=["temp_c"]+[f"temp_l{i}" for i in range(1, h+1)]).reset_index(drop=True)
        if model == "S4":
            return frame[frame.year.lt(2020)].reset_index(drop=True)
        if model in {"S6", "S7"}:
            detailed = read(prefix / "panel_detailed.csv" if len(parts) == 1 else prefix / f"iip/panel_{parts[1]}_detailed.csv")
            if model == "S7":
                return detailed
            period = "month" if "monthly" in parts else "quarter"
            return frame[frame[period].between(detailed[period].min(), detailed[period].max())].reset_index(drop=True)
        return frame
    raise AssertionError((str(path), model, "No independent exact source-sample resolver"))


def full_qr(x: np.ndarray, outcomes: np.ndarray, names: list[str]) -> dict:
    """Full-model QR, independent of native Feols/FWL production geometry."""
    y = np.asarray(outcomes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    assert n > k and np.isfinite(x).all() and np.isfinite(y).all()
    scale = np.linalg.norm(x, axis=0)
    assert np.all(scale > 0)
    normalized = x/scale
    with threadpool_limits(limits=1):
        q, r = np.linalg.qr(normalized, mode="reduced")
        assert np.min(np.abs(np.diag(r))) > 1e-10, ("Independent full design rank", n, k)
        scaled_beta = np.linalg.solve(r, q.T@y)
        beta = scaled_beta/scale[:, None]
        residual = y-normalized@scaled_beta
        inverse = np.linalg.solve(r, np.eye(k))
        bread = inverse@inverse.T
    centered = np.max(np.abs(q@(q.T@np.ones(n))-1)) < 1e-8
    denominator = ((y-y.mean(axis=0))**2).sum(axis=0) if centered else (y*y).sum(axis=0)
    return {"beta": beta, "residual": residual, "normalized": normalized, "scale": scale, "bread_scaled": bread,
        "terms": names, "n_obs": n, "n_parameters": k, "r_squared": 1-(residual**2).sum(axis=0)/denominator}


def formula_qr(sample: pd.DataFrame, formula: str) -> dict:
    y, design = dmatrices(formula, sample, return_type="dataframe", NA_action="raise")
    assert len(y) == len(sample) and y.shape[1] == 1
    result = full_qr(design.to_numpy(), y.to_numpy(), list(design.columns))
    result["outcomes"] = list(y.columns)
    return result


def scores_covariance(fit: dict, sample: pd.DataFrame, contrasts: np.ndarray, *, outcome=0,
                      covariance="Driscoll-Kraay", bandwidth=4, separate=False) -> tuple[np.ndarray, int, float]:
    """Direct full-OLS linear-effect influences, CR1 or actual-calendar HAC."""
    contrasts = np.atleast_2d(contrasts)
    influence = fit["normalized"]@fit["bread_scaled"]@(contrasts/fit["scale"]).T
    individual = influence*fit["residual"][:, [outcome]]
    n, k = fit["n_obs"], fit["n_parameters"]
    period = "month" if "month" in sample else "quarter"
    dates = pd.PeriodIndex(sample[period], freq="M" if period == "month" else "Q").asi8
    periods = len(np.unique(dates))
    if covariance == "Sector cluster":
        units = sample.sector_id.astype(str).to_numpy()
        unique = sorted(set(units));count = len(unique)
        assert count >= 2 and not separate
        score = np.vstack([individual[units == unit].sum(axis=0) for unit in unique])
        correction = count/(count-1)*(n-1)/(n-k)
        meat = score.T@score
        degrees = count-1
    else:
        elapsed = dates-dates.min()
        score = np.zeros((elapsed.max()+1, individual.shape[1]))
        for date in np.unique(elapsed):
            score[date] = individual[elapsed == date].sum(axis=0)
        meat = score.T@score
        if covariance != "Quarter cluster" and covariance != "Month cluster":
            for lag in range(1, bandwidth+1):
                cross = score[lag:].T@score[:-lag]
                meat += (1-lag/(bandwidth+1))*(cross+cross.T)
        correction = periods/(periods-1)*(n-1)/(n-k)
        if separate:
            assert n == periods
            close(correction, n/(n-k), "Singleton TS own-K versus panel-cluster correction", atol=1e-12, rtol=1e-12)
        degrees = periods-1
    value = (meat+meat.T)/2*correction
    return value, degrees, correction


def check_point_row(row: pd.Series, estimate: float, variance: float, degrees: int, label: str) -> None:
    assert np.isfinite(variance) and variance > 0, (label, "Nonpositive variance", variance)
    se = math.sqrt(variance)
    probability = float(2*student_t.sf(abs(estimate/se), degrees))
    critical = student_t.ppf(.975, degrees)
    for field, expected in [("coefficient", estimate), ("std_error", se), ("p_value", probability),
                             ("ci_lower", estimate-critical*se), ("ci_upper", estimate+critical*se)]:
        close(row[field], expected, label+":"+field)
    if "inference_df" in row and pd.notna(row.inference_df):
        close(row.inference_df, degrees, label+":degrees", exact=True)
    wald = estimate**2/variance
    close(probability, f_distribution.sf(wald, 1, degrees), label+":one-linear-restriction F/t p", atol=1e-13, rtol=1e-13)


def holm(values) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1., np.maximum.accumulate(values[order]*np.arange(len(values), 0, -1)))
    result = np.empty(len(values));result[order] = adjusted
    return result


def paired_samples(metadata: dict, folder: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    pair_id = metadata["pair_id"]
    source_path, source_model, variant = canonical_source(ROOT / metadata["source_csv"], str(metadata["source_model"]))
    assert variant == "all_eligible", (pair_id, "Metadata must name canonical full source")
    canonical = source_sample(source_path, source_model)
    period = "month" if "month" in canonical else "quarter"
    frequency = "M" if period == "month" else "Q"
    paths = {row["sample_variant"]: ROOT / row["sample_csv"] for row in metadata["samples"]}
    assert set(paths) == set(VARIANTS), (pair_id, "Not exactly two saved sample variants")
    samples = {variant: read(path) for variant, path in paths.items()}
    # A producer may add a documented common outcome alias in legacy fits.
    frame_equal(samples[VARIANTS[0]], canonical, pair_id+":canonical full sample")
    expected = samples[VARIANTS[0]].loc[~samples[VARIANTS[0]].year.between(2020, 2022)].copy()
    frame_equal(samples[VARIANTS[1]], expected, pair_id+":only endpoint-year mask")
    dates = pd.PeriodIndex(samples[VARIANTS[0]][period], freq=frequency)
    removed_periods = len(dates.unique())-samples[VARIANTS[1]][period].nunique()
    assert bool(metadata["identical_pair"]) == (removed_periods == 0)
    assert int(metadata["n_removed_periods"]) == removed_periods
    for variant, sample in samples.items():
        assert not sample.duplicated([period, "sector_id"]).any()
        dates = pd.PeriodIndex(sample[period], freq=frequency)
        assert np.array_equal(dates.year, sample.year)
        offset = sample.time.to_numpy()-dates.asi8
        assert np.ptp(offset) == 0, (pair_id, variant, "Compressed calendar time")
        if variant == VARIANTS[1]:
            assert not sample.year.between(2020, 2022).any()
        else:
            assert sample[period].nunique() == canonical[period].nunique()
        sectors = sample.sector_id.nunique()
        assert sample.groupby(period).sector_id.nunique().eq(sectors).all()
    return samples, {"pair_id": pair_id, "period": period, "full_n_obs": len(samples[VARIANTS[0]]),
        "excluded_n_obs": len(samples[VARIANTS[1]]), "removed_periods": removed_periods,
        "identical_pair": removed_periods == 0, "sample_files": {variant: str(path.relative_to(ROOT)) for variant, path in paths.items()}}


def check_sample_metadata(rows: pd.DataFrame, sample: pd.DataFrame, fit: dict, outcome: str, label: str) -> None:
    period = "month" if "month" in sample else "quarter"
    expected = {"n_obs": len(sample), "n_parameters": fit["n_parameters"], "n_sectors": sample.sector_id.nunique(),
        "n_periods": sample[period].nunique()}
    for field, value in expected.items():
        if field in rows:
            close(rows[field], np.repeat(value, len(rows)), label+":"+field, exact=True)
    for field, value in [("start", sample[period].min()), ("end", sample[period].max())]:
        assert rows[field].astype(str).eq(str(value)).all(), (label, field)
    if "mean_y" in rows:
        close(rows.mean_y, np.repeat(sample[outcome].mean(), len(rows)), label+":exact-sample signed mean")
        close(rows.sd_y, np.repeat(sample[outcome].std(), len(rows)), label+":exact-sample SD")
    if "r_squared" in rows:
        close(rows.r_squared, np.repeat(fit["r_squared"][0], len(rows)), label+":R squared")
    if "calendar_gaps" in rows:
        dates = pd.PeriodIndex(sample[period], freq="M" if period == "month" else "Q")
        span = int(dates.max().ordinal-dates.min().ordinal+1)
        close(rows.calendar_span_periods, np.repeat(span, len(rows)), label+":calendar span", exact=True)
        close(rows.calendar_gaps, np.repeat(span-len(dates.unique()), len(rows)), label+":calendar gaps", exact=True)
    if "time_design_rank" in rows:
        first = sample.sector_id.iloc[0]
        restricted = fit["normalized"][sample.sector_id.eq(first).to_numpy()]
        rank = np.linalg.matrix_rank(restricted)
        close(rows.time_design_rank, np.repeat(rank, len(rows)), label+":one-unit time-design rank", exact=True)
        close(rows.time_residual_df, np.repeat(sample[period].nunique()-rank, len(rows)), label+":effective time residual df", exact=True)


def verify_common() -> tuple[dict, pd.DataFrame, list[dict]]:
    folder = OUT / "common"
    metadata = json.loads((folder / "specs.json").read_text())
    estimates, aliases = read(folder / "estimates.csv"), read(folder / "aliases.csv")
    assert not estimates.duplicated(["pair_id", "sample_variant", "term", "covariance", "hac_lags"]).any()
    assert len(metadata) == estimates.pair_id.nunique() == aliases.pair_id.nunique()
    assert set(estimates.sample_variant) == set(VARIANTS)
    counts = {"pairs": len(metadata), "native_models_expected": 0, "estimate_rows": len(estimates), "coefficient_or_contrast_tests": 0,
        "identical_preexcluded_pairs": 0, "source_alias_rows": len(aliases), "retained_2023_YoY_rows_with_2022_base": 0}
    sample_records, spec_keys = [], set()
    for spec in metadata:
        pair_id = spec["pair_id"]
        assert (spec["source_csv"], spec["source_model"]) not in spec_keys
        spec_keys.add((spec["source_csv"], spec["source_model"]))
        samples, sample_record = paired_samples(spec, folder)
        sample_records.append(sample_record)
        counts["native_models_expected"] += 1+(not sample_record["identical_pair"])
        counts["identical_preexcluded_pairs"] += sample_record["identical_pair"]
        for variant, sample in samples.items():
            selected = estimates[estimates.pair_id.eq(pair_id)&estimates.sample_variant.eq(variant)]
            assert not selected.empty and selected.formula.eq(spec["formula"]).all()
            assert set(selected.source_csv) == {spec["source_csv"]} and set(selected.source_model) == {spec["source_model"]}
            fit = formula_qr(sample, spec["formula"])
            if variant == VARIANTS[0] or not sample_record["identical_pair"]:
                period = spec["period_column"]
                NATIVE_EXPECTED["common"].append({"label": f"COVID pair {pair_id} {sample[period].min()}–{sample[period].max()}",
                    "outcome": spec["outcome"], "n_obs": len(sample), "n_parameters": fit["n_parameters"], "source_formula": spec["formula"]})
            check_sample_metadata(selected, sample, fit, spec["outcome"], f"common/{pair_id}/{variant}")
            n_sectors = sample.sector_id.nunique()
            primary = selected.covariance.eq("Sector cluster") if n_sectors >= 2 else selected.covariance.eq("Driscoll-Kraay")&selected.hac_lags.eq(spec["horizon"])
            assert np.array_equal(primary.to_numpy(), selected.main_inference.to_numpy()), (pair_id, variant, "main inference mapping")
            assert set(selected.loc[selected.covariance.eq("Driscoll-Kraay"), "hac_lags"]) == {spec["horizon"], 2*spec["horizon"]}
            if n_sectors >= 2:
                assert selected.covariance.eq("Sector cluster").any()
                cluster_rows = selected[selected.covariance.eq("Sector cluster")]
                assert cluster_rows.n_clusters.eq(n_sectors).all()
            else:
                assert not selected.covariance.eq("Sector cluster").any()
            # All requested coefficient and linear-combination rows share the
            # same full fitted model. Infer their joint covariance together.
            for (covariance, bandwidth), group in selected.groupby(["covariance", "hac_lags"], sort=False):
                contrasts = np.zeros((len(group), len(fit["terms"])))
                for i, row in enumerate(group.itertuples()):
                    terms = json.loads(row.contrast_terms) if row.term_kind == "linear_contrast" else [row.term]
                    assert terms
                    for term in terms:
                        contrasts[i, fit["terms"].index(term)] += 1
                beta = contrasts@fit["beta"][:, 0]
                matrix, degrees, correction = scores_covariance(fit, sample, contrasts, covariance=covariance, bandwidth=int(bandwidth), separate=n_sectors == 1)
                for i, (_, row) in enumerate(group.iterrows()):
                    label = f"common/{pair_id}/{variant}/{covariance}/{bandwidth}/{row.term}"
                    check_point_row(row, float(beta[i]), float(matrix[i, i]), degrees, label)
                    if covariance == "Sector cluster":
                        close(row.finite_sample_correction, correction, label+":CR1 multiplier", atol=1e-12, rtol=1e-12)
                    counts["coefficient_or_contrast_tests"] += 1
            if "yoy" in spec["outcome"]:
                counts["retained_2023_YoY_rows_with_2022_base"] += int(sample.year.eq(2023).sum()) if variant == VARIANTS[1] else 0
    # Every original coefficient/covariance source row is checked, including
    # existing X/C5/S5/exclude aliases against the excluded counterpart.
    for alias in aliases.itertuples():
        source = read(ROOT / alias.source_csv).iloc[int(alias.source_row)]
        path, model, variant = canonical_source(ROOT / alias.source_csv, str(alias.source_model))
        assert str(path.relative_to(ROOT)) == alias.canonical_source_csv and model == alias.canonical_source_model
        assert source.model == alias.source_model and source.term == alias.source_term and source.formula == alias.source_formula
        row = estimates[estimates.pair_id.eq(alias.pair_id)&estimates.sample_variant.eq(variant)&estimates.term.eq(source.term)
            &estimates.covariance.eq(source.covariance)&estimates.hac_lags.eq(source.hac_lags)]
        assert len(row) == 1, (alias.source_csv, alias.source_row, "Source row not exactly represented")
        for field in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "r_squared"]:
            close(row[field].iloc[0], source[field], f"common original source/{alias.source_csv}/{alias.source_row}/{field}")
        for field in ["n_obs", "n_sectors", "inference_df"]:
            if field in source and pd.notna(source[field]):
                close(row[field].iloc[0], source[field], f"common original source/{alias.source_csv}/{alias.source_row}/{field}", exact=True)
        assert row.start.iloc[0] == source.start and row.end.iloc[0] == source.end
        SOURCE_COVERAGE.append({"source_csv": alias.source_csv, "source_row": int(alias.source_row), "pair_id": alias.pair_id,
            "sample_variant": variant, "artifact": "common/estimates.csv", "scope": "primary regression coefficient/covariance row"})
    return counts, aliases, sample_records


def sector_design(sample: pd.DataFrame, model: str, terms: list[str], *, separate=False) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Build the stated full indicator model directly, without FWL projection."""
    sectors, seasons, years = sorted(sample.sector_id.unique()), sorted(sample.season.unique()), sorted(sample.year.unique())
    centered = {term: sample[term].to_numpy()-sample[term].mean() for term in terms}
    if separate:
        columns, names = [np.ones(len(sample))], ["Intercept"]
        for term in terms:
            columns.append(centered[term]);names.append(term)
        for variable, values in [("season", seasons[1:]), ("year", years[1:])]:
            for value in values:
                columns.append(sample[variable].eq(value).to_numpy(dtype=float));names.append(f"{variable}[{value}]")
        x = np.column_stack(columns)
        contrast = np.zeros((1, x.shape[1]));contrast[0, names.index("temp_c")] = 1
        return x, names, contrast, sectors
    relative = model == "R1"
    active = sectors[1:] if relative else sectors
    columns, names = [], []
    if relative:
        for date in sorted(sample.time.unique()):
            columns.append(sample.time.eq(date).to_numpy(dtype=float));names.append(f"date[{date}]")
    for sector in active:
        for season in seasons:
            columns.append((sample.sector_id.eq(sector)&sample.season.eq(season)).to_numpy(dtype=float))
            names.append(f"sector_season[{sector},{season}]")
    if model in {"H3", "R1"}:
        for sector in active:
            for year in years[1:]:
                columns.append((sample.sector_id.eq(sector)&sample.year.eq(year)).to_numpy(dtype=float))
                names.append(f"sector_year[{sector},{year}]")
    else:
        for year in years[1:]:
            columns.append(sample.year.eq(year).to_numpy(dtype=float));names.append(f"year[{year}]")
    if model == "H1":
        for term in terms[1:]:
            columns.append(centered[term]);names.append(term)
    else:
        for sector in active:
            for term in terms[1:]:
                columns.append(sample.sector_id.eq(sector).to_numpy(dtype=float)*centered[term]);names.append(f"{term}[{sector}]")
    if not relative:
        columns.append(centered["temp_c"]);names.append("temp_c")
    for sector in sectors[1:]:
        columns.append(sample.sector_id.eq(sector).to_numpy(dtype=float)*centered["temp_c"])
        names.append(f"temp_c:C(sector_id)[T.{sector}]")
    x = np.column_stack(columns)
    reported_sectors = sectors[1:] if relative else sectors
    contrasts = np.zeros((len(reported_sectors), len(names)))
    for i, sector in enumerate(reported_sectors):
        if not relative:
            contrasts[i, names.index("temp_c")] = 1
        if sector != sectors[0]:
            contrasts[i, names.index(f"temp_c:C(sector_id)[T.{sector}]")] = 1
    return x, names, contrasts, reported_sectors


def sector_sample_metadata(meta: pd.Series) -> dict:
    return {"pair_id": meta.pair_id, "source_csv": meta.source_csv, "source_model": meta.source_model,
        "identical_pair": meta.covid_periods_in_eligible == 0, "n_removed_periods": int(meta.covid_periods_in_eligible),
        "samples": [{"sample_variant": VARIANTS[0], "sample_csv": meta.all_sample_csv},
                    {"sample_variant": VARIANTS[1], "sample_csv": meta.exclude_sample_csv}]}


def verify_sector() -> tuple[dict, pd.DataFrame, list[dict]]:
    folder = OUT / "sector"
    metadata = read(folder / "metadata.csv")
    table_names = ["estimates", "joint_tests", "pairwise_tests", "interaction_components", "component_covariances", "slope_covariances", "all_coefficients", "source_aliases", "weather_coefficients"]
    tables = {name: read(folder / f"{name}.csv") for name in table_names}
    estimates, aliases = tables["estimates"], tables["source_aliases"]
    assert metadata.pair_id.is_unique and set(estimates.pair_id) == set(metadata.pair_id)
    assert not estimates.duplicated(["pair_id", "sample_variant", "sector_id", "hac_lags"]).any()
    counts = {"pairs": len(metadata), "native_models_expected": 0, "estimate_rows": len(estimates), "all_coefficients_checked": 0,
        "joint_tests_checked": 0, "joint_available": 0, "joint_singular_without_p": 0, "pairwise_tests_checked": 0,
        "component_rows_checked": 0, "component_covariance_entries_checked": 0, "slope_covariance_entries_checked": 0,
        "weather_coefficient_rows_checked": 0, "source_temperature_and_control_rows_checked": 0,
        "reference_zero_normalizations_without_SE_or_p": 0, "families": metadata.groupby("family").size().to_dict()}
    sample_records, sample_cache, fit_cache = [], {}, {}
    for _, meta in metadata.iterrows():
        if meta.sample_key not in sample_cache:
            samples, record = paired_samples(sector_sample_metadata(meta), folder)
            sample_cache[meta.sample_key] = samples;sample_records.append(record)
        samples = sample_cache[meta.sample_key]
        terms = ["temp_c", "rain_100mm", "rh_pct"]+(["wind_mps", "pm25_10ug"] if meta.control_set == "full" else [])
        separate = meta.family == "separate_series"
        for variant, sample in samples.items():
            selected = estimates[estimates.pair_id.eq(meta.pair_id)&estimates.sample_variant.eq(variant)]
            assert set(selected.hac_lags) == {12, 24} if meta.frequency == "monthly" else set(selected.hac_lags) == {4, 8}
            horizon = 12 if meta.frequency == "monthly" else 4
            assert np.array_equal(selected.main_inference.to_numpy(), selected.hac_lags.eq(horizon).to_numpy())
            assert not selected.covariance.eq("Sector cluster").any(), "Single-sector effects must not use sector CR1"
            units = list(sample.groupby("sector_id", sort=True)) if separate else [("panel", sample)]
            for unit_id, unit in units:
                unit = unit.reset_index(drop=True)
                cache_key = (meta.sample_key, variant, meta.model, unit_id)
                if cache_key not in fit_cache:
                    x, names, contrasts, sectors = sector_design(unit, meta.model, terms, separate=separate)
                    outcomes = metadata.loc[metadata.sample_key.eq(meta.sample_key)&metadata.model.eq(meta.model), "outcome"].drop_duplicates().tolist()
                    fit = full_qr(x, unit[outcomes].to_numpy(), names)
                    fit_cache[cache_key] = (fit, contrasts, sectors, outcomes)
                fit, contrasts, sectors, outcomes = fit_cache[cache_key]
                outcome_position = outcomes.index(meta.outcome)
                fit_one = {**fit, "r_squared": fit["r_squared"][[outcome_position]]}
                rows = selected[selected.sector_id.eq(unit_id)] if separate else selected
                check_sample_metadata(rows.drop(columns=["mean_y", "sd_y"]), unit, fit_one, meta.outcome, f"sector/{meta.pair_id}/{variant}/{unit_id}")
                expected_label = f"COVID own series {meta.context}/{meta.control_set}/{unit_id}/{variant}" if separate else f"COVID sector {meta.context}/{meta.control_set}/{meta.model}/{variant}"
                NATIVE_EXPECTED["sector"].append({"label": expected_label, "outcome": meta.outcome, "n_obs": len(unit), "n_parameters": fit["n_parameters"], "source_formula": "Explicit full-rank design matrix"})
                counts["native_models_expected"] += 1
                all_coefs = tables["all_coefficients"]
                all_coefs = all_coefs[all_coefs.pair_id.eq(meta.pair_id)&all_coefs.sample_variant.eq(variant)&all_coefs.unit_id.eq(unit_id)].set_index("term")
                assert len(all_coefs) == len(fit["terms"]) and set(all_coefs.index) == set(fit["terms"])
                close(all_coefs.loc[fit["terms"], "coefficient"], fit["beta"][:, outcome_position], f"sector/{meta.pair_id}/{variant}/{unit_id}:all full-model coefficients")
                counts["all_coefficients_checked"] += len(all_coefs)
                for bandwidth, group in rows.groupby("hac_lags", sort=False):
                    ordered = group.set_index("sector_id").loc[sectors]
                    beta = contrasts@fit["beta"][:, outcome_position]
                    matrix, degrees, _ = scores_covariance(fit, unit, contrasts, outcome=outcome_position, bandwidth=int(bandwidth), separate=separate)
                    for i, sector in enumerate(sectors):
                        label = f"sector/{meta.pair_id}/{variant}/{sector}/{bandwidth}"
                        row = ordered.loc[sector]
                        check_point_row(row, float(beta[i]), float(matrix[i, i]), degrees, label)
                        sector_unit = unit[unit.sector_id.eq(sector)]
                        close(row.mean_y, sector_unit[meta.outcome].mean(), label+":sector signed mean")
                        close(row.sd_y, sector_unit[meta.outcome].std(), label+":sector SD")
                        if separate:
                            close(row.temp_main, beta[i], label+":own temperature coefficient")
                            assert pd.isna(row.interaction_delta) and not row.delta_is_normalization
                        elif meta.model == "R1":
                            assert pd.isna(row.temp_main) and not row.delta_is_normalization
                            close(row.interaction_delta, beta[i], label+":relative delta only")
                        else:
                            base = fit["beta"][fit["terms"].index("temp_c"), outcome_position]
                            close(row.temp_main, base, label+":temperature main")
                            close(row.coefficient, row.temp_main+row.interaction_delta, label+":main plus interaction")
                            assert bool(row.delta_is_normalization) == (sector == sorted(unit.sector_id.unique())[0])
                    verify_sector_components(tables, meta, variant, int(bandwidth), unit_id, fit, contrasts, sectors, outcome_position, matrix, counts)
                    if not separate:
                        verify_sector_joint_and_pairs(tables, meta, variant, int(bandwidth), unit, beta, matrix, sectors, degrees, counts)
                if separate:
                    verify_TS_weather(tables["weather_coefficients"], meta, variant, unit_id, unit, fit, outcome_position, terms, counts)
    verify_sector_original_aliases(metadata, tables, counts)
    return counts, aliases, sample_records


def subset(frame: pd.DataFrame, pair: str, variant: str, bandwidth: int | None = None) -> pd.DataFrame:
    selected = frame[frame.pair_id.eq(pair)&frame.sample_variant.eq(variant)]
    return selected[selected.hac_lags.eq(bandwidth)] if bandwidth is not None else selected


def verify_sector_components(tables: dict, meta, variant: str, bandwidth: int, unit_id: str, fit: dict,
                             contrasts: np.ndarray, sectors: list[str], outcome: int, matrix: np.ndarray, counts: dict) -> None:
    components = subset(tables["interaction_components"], meta.pair_id, variant, bandwidth)
    covariance = subset(tables["component_covariances"], meta.pair_id, variant, bandwidth)
    separate = meta.family == "separate_series"
    if separate:
        components = components[components.sector_id.eq(unit_id)]
        covariance = covariance[covariance.sector_i.eq(unit_id)&covariance.sector_j.eq(unit_id)]
    normalizations = components[components.normalization.eq(True)]
    estimated = components[components.estimated.eq(True)]
    assert len(estimated)+len(normalizations) == len(components) and not estimated.normalization.any()
    if separate or meta.model == "R1":
        assert normalizations.empty
    else:
        assert len(normalizations) == 1 and normalizations.coefficient.iloc[0] == 0
        assert normalizations.sector_id.iloc[0] == sectors[0]
        assert normalizations[["std_error", "ci_lower", "ci_upper", "p_value"]].isna().all().all()
        assert not normalizations.estimated.any()
        counts["reference_zero_normalizations_without_SE_or_p"] += 1
    term_names = estimated.term.tolist()
    assert len(term_names) == len(sectors) and len(set(term_names)) == len(term_names)
    component_to_total = np.eye(len(sectors))
    if not separate and meta.model != "R1":
        # Arrange the saved native component order as main, then deviations.
        term_names = ["temp_c"]+[f"temp_c:C(sector_id)[T.{sector}]" for sector in sectors[1:]]
        component_to_total[:, 0] = 1
    total_to_component = np.linalg.inv(component_to_total)
    component_matrix = total_to_component@matrix@total_to_component.T
    component_beta = fit["beta"][[fit["terms"].index(term) for term in term_names], outcome]
    recorded = covariance.pivot(index="term_i", columns="term_j", values="covariance").loc[term_names, term_names].to_numpy()
    assert len(covariance) == len(term_names)**2
    label = f"sector/{meta.pair_id}/{variant}/{unit_id}/{bandwidth}"
    close(recorded, component_matrix, label+":full component covariance")
    close(component_to_total@recorded@component_to_total.T, matrix, label+":main/delta covariance propagation including cross terms")
    ordered = estimated.set_index("term").loc[term_names]
    periods = fit["n_obs"] if separate else int(ordered.n_periods.iloc[0])
    for i, (_, row) in enumerate(ordered.iterrows()):
        check_point_row(row, float(component_beta[i]), float(component_matrix[i, i]), periods-1, label+":component "+row.name)
    if not separate:
        saved = subset(tables["slope_covariances"], meta.pair_id, variant, bandwidth)
        actual = saved.pivot(index="sector_i", columns="sector_j", values="covariance").loc[sectors, sectors].to_numpy()
        assert len(saved) == len(sectors)**2
        close(actual, matrix, label+":all joint slope covariance entries")
        counts["slope_covariance_entries_checked"] += len(saved)
    counts["component_rows_checked"] += len(components)
    counts["component_covariance_entries_checked"] += len(covariance)


def verify_sector_joint_and_pairs(tables: dict, meta, variant: str, bandwidth: int, sample: pd.DataFrame,
                                 beta: np.ndarray, matrix: np.ndarray, sectors: list[str], degrees: int, counts: dict) -> None:
    selected = subset(tables["joint_tests"], meta.pair_id, variant, bandwidth)
    assert len(selected) == 1
    row = selected.iloc[0]
    restriction = np.eye(len(beta)) if meta.model == "R1" else np.eye(len(beta))[1:]-np.eye(len(beta))[[0]]
    delta, covariance = restriction@beta, restriction@matrix@restriction.T
    periods = sample.time.nunique()
    weather_count = 5 if meta.control_set == "full" else 3
    bound = periods-sample.season.nunique()-1 if meta.model == "H1" else periods-sample.season.nunique()-weather_count if meta.model == "H2" else periods-(sample.season.nunique()+sample.year.nunique()-1+weather_count)
    singular = np.linalg.svd(covariance, compute_uv=False)
    rank = min(int(np.sum(singular > singular.max()*1e-10)), int(bound))
    q = len(delta)
    close(row.restrictions, q, "Omnibus equal-slopes restrictions", exact=True)
    close(row.covariance_rank, rank, "Omnibus residual-score/numerical rank", exact=True)
    close(row.numerator_df, q, "Omnibus numerator df", exact=True)
    close(row.denominator_df, degrees, "Omnibus denominator df", exact=True)
    assert row.structural_rank_bound >= bound
    if rank < q:
        assert str(row.status).startswith("unavailable") and row[["wald_chi2", "f_statistic", "p_value"]].isna().all()
        counts["joint_singular_without_p"] += 1
    else:
        assert row.status == "available"
        statistic = float(delta@np.linalg.solve(covariance, delta))
        close(row.wald_chi2, statistic, "Equal-slopes omnibus Wald, common temperature unrestricted")
        close(row.f_statistic, statistic/q, "Equal-slopes omnibus F")
        close(row.p_value, f_distribution.sf(statistic/q, q, degrees), "Equal-slopes omnibus F p")
        counts["joint_available"] += 1
    counts["joint_tests_checked"] += 1
    family = subset(tables["pairwise_tests"], meta.pair_id, variant, bandwidth)
    if meta.source_kind == "detailed_FE":
        expected = {(first, second) for i, first in enumerate(sectors) for second in sectors[i+1:]}
    else:
        source = ROOT / meta.source_csv
        source = source.parent / "pairwise_tests.csv"
        if not source.exists():
            context = "va" if meta.context == "va" else "iip-monthly" if meta.frequency == "monthly" else "iip-quarterly"
            source = ROOT / "outputs/heterogeneity" / context / "weather/pairwise_tests.csv"
        original = read(source)
        original = original[original.model.eq(meta.model)&original.outcome.eq(meta.outcome)&original.hac_lags.eq(bandwidth)]
        expected = set(original[["first_sector_id", "second_sector_id"]].itertuples(index=False, name=None))
    assert set(family[["first_sector_id", "second_sector_id"]].itertuples(index=False, name=None)) == expected
    probabilities = []
    for _, pair in family.iterrows():
        weights = np.zeros(len(sectors))
        for sector, sign in [(pair.first_sector_id, 1), (pair.second_sector_id, -1)]:
            if sector in sectors:
                weights[sectors.index(sector)] += sign
        effect, variance = float(weights@beta), float(weights@matrix@weights)
        check_point_row(pair, effect, variance, degrees, "Pairwise difference with joint covariance")
        probabilities.append(float(2*student_t.sf(abs(effect/math.sqrt(variance)), degrees)))
        assert pair.multiplicity_family == f"{len(expected)} displayed sector pairs within pair_id/sample_variant/bandwidth"
    close(family.p_value_holm, holm(probabilities), "Per-spec/sample/bandwidth Holm family")
    counts["pairwise_tests_checked"] += len(family)


def verify_TS_weather(weather: pd.DataFrame, meta, variant: str, unit_id: str, unit: pd.DataFrame, fit: dict,
                      outcome: int, terms: list[str], counts: dict) -> None:
    rows = subset(weather, meta.pair_id, variant)
    rows = rows[rows.sector_id.eq(unit_id)]
    assert len(rows) == 2*len(terms)
    one_fit = {**fit, "r_squared": fit["r_squared"][[outcome]]}
    check_sample_metadata(rows, unit, one_fit, meta.outcome, "Own-series weather metadata/"+meta.pair_id+"/"+variant+"/"+unit_id)
    contrasts = np.eye(len(fit["terms"]))[[fit["terms"].index(term) for term in terms]]
    coefficients = contrasts@fit["beta"][:, outcome]
    for bandwidth, group in rows.groupby("hac_lags", sort=False):
        assert set(group.term) == set(terms)
        matrix, degrees, _ = scores_covariance(fit, unit, contrasts, outcome=outcome, bandwidth=int(bandwidth), separate=True)
        group = group.set_index("term")
        for i, term in enumerate(terms):
            check_point_row(group.loc[term], float(coefficients[i]), float(matrix[i, i]), degrees, "Own-series weather coefficient "+term)
            counts["weather_coefficient_rows_checked"] += 1


def verify_sector_original_aliases(metadata: pd.DataFrame, tables: dict, counts: dict) -> None:
    for _, alias in tables["source_aliases"].iterrows():
        meta = metadata.loc[metadata.pair_id.eq(alias.pair_id)].iloc[0]
        source_path = ROOT / alias.source_csv
        source = read(source_path)
        if "model" in source:
            source = source[source.model.eq(alias.source_model)]
        if "outcome" in source:
            source = source[source.outcome.eq(meta.outcome)]
        assert not source.empty
        for index, original in source.iterrows():
            name = "weather_coefficients" if source_path.name == "separate_sector_coefficients.csv" else "estimates"
            target = subset(tables[name], alias.pair_id, alias.sample_variant, int(original.hac_lags))
            target = target[target.sector_id.eq(original.sector_id)]
            if name == "weather_coefficients":
                target = target[target.term.eq(original.term)]
            assert len(target) == 1, (alias.source_csv, index, "Sector source row not represented")
            for field in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "r_squared"]:
                if field in original:
                    close(target[field].iloc[0], original[field], f"Original sector row/{alias.source_csv}/{index}/{field}")
            for field in ["n_obs", "n_parameters"]:
                if field in original:
                    close(target[field].iloc[0], original[field], "Original sector sample/K", exact=True)
            SOURCE_COVERAGE.append({"source_csv": alias.source_csv, "source_row": int(index), "pair_id": alias.pair_id,
                "sample_variant": alias.sample_variant, "artifact": "sector/"+name+".csv", "scope": "primary regression coefficient/covariance row"})
            counts["source_temperature_and_control_rows_checked"] += 1
        if source_path.name == "sector_slopes.csv":
            verify_original_covariance_companions(source_path.parent, alias, tables)


def verify_original_covariance_companions(folder: Path, alias, tables: dict) -> None:
    definitions = {"slope_covariances": ["sector_i", "sector_j"], "component_covariances": ["term_i", "term_j"],
        "interaction_components": ["term"], "joint_tests": [], "pairwise_tests": ["first_sector_id", "second_sector_id"]}
    for name, keys in definitions.items():
        source_path = folder / f"{name}.csv"
        if not source_path.exists():
            continue
        source = read(source_path)
        source = source[source.model.eq(alias.source_model)&source.outcome.eq(alias.outcome)]
        target = subset(tables[name], alias.pair_id, alias.sample_variant)
        for index, original in source.iterrows():
            selected = target[target.hac_lags.eq(original.hac_lags)]
            for key in keys:
                selected = selected[selected[key].eq(original[key])]
            assert len(selected) == 1, (str(source_path), int(index), name)
            row = selected.iloc[0]
            fields = ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "p_value_holm", "covariance", "wald_chi2", "f_statistic"]
            for field in fields:
                if field in original and not isinstance(original[field], str):
                    close(row[field], original[field], f"Original {name}/{index}/{field}")
            for field in ["covariance_rank", "restrictions", "numerator_df", "denominator_df", "estimated", "normalization"]:
                if field in original:
                    close(row[field], original[field], f"Original {name}/{index}/{field}", exact=True)
            if "status" in original:
                assert row.status == original.status
            SOURCE_COVERAGE.append({"source_csv": str(source_path.relative_to(ROOT)), "source_row": int(index), "pair_id": alias.pair_id,
                "sample_variant": alias.sample_variant, "artifact": "sector/"+name+".csv", "scope": "joint covariance/component/pairwise companion"})


def verify_structured_contrasts() -> int:
    estimates = read(OUT / "common/estimates.csv")
    metadata = json.loads((OUT / "common/specs.json").read_text())
    pair_lookup = {(spec["source_csv"], spec["source_model"]): spec["pair_id"] for spec in metadata}
    horizons = {spec["pair_id"]: int(spec["horizon"]) for spec in metadata}
    paths = sorted((ROOT / "outputs").glob("**/distributed_lag_sum.json"))
    paths += [ROOT / "outputs/aggregate_va/distributed_lag/cumulative_response.json"]
    paths += sorted((ROOT / "outputs").glob("**/temperature_difference_restriction*.json"))
    checked = 0
    for path in paths:
        if "covid_comparison" in path.parts:
            continue
        document = json.loads(path.read_text())
        annual = path.name.startswith("temperature_difference_restriction")
        nsa = path.name == "temperature_difference_restriction_nsa.json"
        model = ("Y4N" if nsa else "Y4") if annual else "D1" if "aggregate_va" in path.parts else "L1"
        source_name = "regression_results.csv" if annual or "aggregate_va" in path.parts else "distributed_lag_results.csv"
        source = str(path.with_name(source_name).relative_to(ROOT))
        pair = pair_lookup[source, model]
        term = "sum_temperature_current_and_annual_lag" if annual else "sum_temperature_lags"
        for branch, old in [("root", document)]+([("sector_cluster", document["sector_cluster"])] if "sector_cluster" in document else []):
            covariance, bandwidth = ("Sector cluster", 0) if branch == "sector_cluster" else ("Driscoll-Kraay", horizons[pair])
            row = estimates[estimates.pair_id.eq(pair)&estimates.sample_variant.eq(VARIANTS[0])&estimates.term.eq(term)
                &estimates.covariance.eq(covariance)&estimates.hac_lags.eq(bandwidth)]
            assert len(row) == 1
            coef_key = "sum" if "sum" in old else next(key for key in old if key.startswith("sum_lag_"))
            close(row.coefficient.iloc[0], old[coef_key], f"Original structured contrast/{path}/{branch}/coefficient")
            for field in ["std_error", "ci_lower", "ci_upper", "p_value"]:
                if field in old:
                    close(row[field].iloc[0], old[field], f"Original structured contrast/{path}/{branch}/{field}")
            SOURCE_COVERAGE.append({"source_csv": str(path.relative_to(ROOT)), "source_row": branch, "pair_id": pair,
                "sample_variant": VARIANTS[0], "artifact": "common/estimates.csv", "scope": "structured full-covariance linear contrast"})
            checked += 1
    return checked


def verify_source_coverage() -> dict:
    inventory = original_inventory()
    expected = {(row.source_csv, str(row.source_row)) for row in inventory.itertuples()}
    primary = pd.DataFrame(SOURCE_COVERAGE)
    relevant = primary[primary.scope.eq("primary regression coefficient/covariance row")]
    actual = {(row.source_csv, str(row.source_row)) for row in relevant.itertuples()}
    assert len(relevant) == len(actual), "Original primary regression row mapped more than once"
    assert actual == expected, ("Primary source-row coverage", sorted(expected-actual)[:10], sorted(actual-expected)[:10])
    companion_names = {name+".csv" for name in ["slope_covariances", "component_covariances", "interaction_components", "joint_tests", "pairwise_tests"]}
    baseline = json.loads((OUT / "original_inputs_reference.json").read_text())["files"]
    companion_expected = set()
    for name in baseline:
        path = ROOT / name
        if path.name in companion_names:
            companion_expected.update((name, str(i)) for i in range(len(read(path))))
    companion = primary[primary.scope.eq("joint covariance/component/pairwise companion")]
    companion_actual = {(row.source_csv, str(row.source_row)) for row in companion.itertuples()}
    assert len(companion) == len(companion_actual), "Companion covariance/test row mapped more than once"
    assert companion_actual == companion_expected, ("Original joint covariance/test coverage", len(companion_expected-companion_actual), sorted(companion_expected-companion_actual)[:5])
    inventory.to_csv(OUT / "original_regression_inventory.csv", index=False)
    primary.to_csv(OUT / "source_coverage.csv", index=False)
    return {"original_regression_source_files": inventory.source_csv.nunique(), "original_regression_rows": len(inventory),
        "primary_covariance_rows_by_family": inventory.source_family.value_counts().to_dict(),
        "all_primary_rows_accounted_exactly_once": True,
        "original_component_covariance_joint_pair_source_files": companion.source_csv.nunique(),
        "original_component_covariance_joint_pair_rows": len(companion), "all_covariance_companion_rows_accounted_exactly_once": True,
        "structured_contrast_source_rows": int(primary.scope.eq("structured full-covariance linear contrast").sum())}


def verify_native_receipt(family: str) -> dict:
    folder = OUT / family / "pyfixest"
    path = folder / "fit_log.csv"
    log = read(path)
    expected = pd.DataFrame(NATIVE_EXPECTED[family])
    assert len(log) == len(expected) and len(log) > 0
    columns = ["label", "outcome", "n_obs", "n_parameters", "source_formula"]
    assert Counter(log[columns].itertuples(index=False, name=None)) == Counter(expected[columns].itertuples(index=False, name=None)), (family, "Native inventory differs from independently reconstructed paired model/outcome/unit fits")
    assert log.fit_id.tolist() == list(range(1, len(log)+1))
    assert set(log.api_call_id) == set(range(1, int(log.api_call_id.max())+1))
    assert not log.duplicated(["api_call_id", "outcome"]).any()
    installed = importlib.metadata.version("pyfixest")
    assert log.estimator_version.astype(str).eq(installed).all()
    assert log.native_class.eq("Feols").all() and log.native_module.str.startswith("pyfixest.estimation").all()
    assert log.native_design_basis.eq("orthonormal SVD preconditioned full LSDV").all()
    assert log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    for field in ["native_mapped_coefficient_max_abs_error", "native_residual_max_abs_error", "native_response_max_abs_error", "native_design_max_abs_error"]:
        assert log[field].eq(0).all()
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all() and log.native_residuals_sha256.eq(log.residuals_sha256).all()
    assert log.native_design_orthogonality_max_abs_error.between(0, 1e-10).all()
    for field in ["coefficient_transform_rows", "coefficient_transform_columns", "coefficient_transform_rank"]:
        assert log[field].eq(log.n_parameters).all()
    assert log.df_resid.eq(log.n_obs-log.n_parameters).all() and log.df_resid.gt(0).all()
    for field in ["design_sha256", "scaled_design_sha256", "native_design_sha256", "coefficient_transform_sha256", "outcome_sha256", "coefficient_names_sha256", "column_scales_sha256", "coefficients_sha256", "native_coefficients_sha256", "residuals_sha256"]:
        assert log[field].str.fullmatch("[a-f0-9]{64}").all()
    for row in log.itertuples():
        names, scales = json.loads(row.coefficient_names), np.asarray(json.loads(row.column_scales), dtype=float)
        assert len(names) == len(scales) == row.n_parameters and len(set(names)) == len(names)
        assert np.isfinite(scales).all() and np.all(scales > 0)
        assert hashlib.sha256(json.dumps(names, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == row.coefficient_names_sha256
        assert hashlib.sha256(str(scales.shape).encode()+np.ascontiguousarray(scales, dtype=np.float64).tobytes()).hexdigest() == row.column_scales_sha256
    manifest = json.loads((folder / "estimator_manifest.json").read_text())
    assert manifest["status"] == "model_inventory_written" and manifest["native_fits_recorded"] == len(log)
    assert manifest["native_api_calls"] == log.api_call_id.nunique() and manifest["fit_log_sha256"] == digest(path)
    assert manifest["all_native_samples_and_coefficients_retained"] and manifest["native_models_retained_on_wrappers"]
    return {"actual_native_models": len(log), "independently_expected_models": len(expected), "actual_native_API_calls": log.api_call_id.nunique(),
        "version": installed, "all_expected_fits_accounted": True, "no_dropped_rows_or_columns": True,
        "fit_log_sha256": digest(path), "maximum_native_design_orthogonality_error": float(log.native_design_orthogonality_max_abs_error.max())}


def verify_covid_comparison() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKS.clear();SOURCE_COVERAGE.clear()
    for values in NATIVE_EXPECTED.values():
        values.clear()
    target = OUT / "verification.json"
    target.write_text(json.dumps({"status": "running", "claim": "No completed-verification claim until all independent checks finish"})+"\n")
    original_input_integrity()
    common, _, common_samples = verify_common()
    print(f"Independent paired common-slope audit: {common['pairs']} pairs and {common['estimate_rows']} rows passed.", flush=True)
    sector, _, sector_samples = verify_sector()
    print(f"Independent paired sector audit: {sector['pairs']} pairs, {sector['estimate_rows']} slope rows and {sector['joint_tests_checked']} joint tests passed.", flush=True)
    contrasts = verify_structured_contrasts()
    coverage = verify_source_coverage()
    provenance = {family: verify_native_receipt(family) for family in ["common", "sector"]}
    preservation = original_input_integrity()
    checks = pd.DataFrame(CHECKS)
    checks.to_csv(OUT / "numerical_checks.csv", index=False)
    pd.DataFrame(common_samples+sector_samples).to_csv(OUT / "sample_checks.csv", index=False)
    report = {"status": "passed", "method": "Independent full-design QR OLS, explicit sector CR1/calendar-HAC coefficient influences, direct full-rank Wald and manually sorted Holm",
        "tolerance": {"absolute": ATOL, "relative": RTOL, "rule": "abs(new-reference) <= 1e-7 + 1e-7*abs(reference)"},
        "common": common, "sector": sector, "coverage": coverage, "structured_contrasts": contrasts,
        "native_provenance": provenance, "original_preservation": preservation,
        "canonical_pairs": common["pairs"]+sector["pairs"], "paired_estimate_rows": common["estimate_rows"]+sector["estimate_rows"],
        "native_models": sum(row["actual_native_models"] for row in provenance.values()),
        "numeric_check_groups": len(checks), "numeric_values_checked": int(checks["values"].sum()),
        "maximum_fraction_of_tolerance": float(checks.maximum_fraction_of_tolerance.max()),
        "sample_rule": "Only observations dated 2020–2022 removed, after full-history outcomes/lags/annual differences and original eligibility were constructed",
        "covid_yoy_base_caveat": "Retained 2023 year-on-year outcomes/annual weather changes can use 2022 as their comparison base",
        "main_inference": "Common pooled slopes: sector CR1; sector-specific slopes and singleton series: calendar DK/HAC with h main and 2h sensitivity",
        "significance_nulls": "Sector p tests one total linear effect; heterogeneity omnibus tests equal slopes with common temperature unrestricted; singular covariance has no p",
        "stationarity_limit": "No new stationarity clearance: standard marginal ADF/KPSS must not silently compress excluded calendar gaps",
        "interpretation": "Numerical/sample/source-coverage consistency only; not a causal or small-sample-inference certificate"}
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    note_path = ROOT / "notes/covid_comparison_methods.md"
    note = note_path.read_text().split("\n<!-- completed-audit:begin -->")[0].rstrip()
    completed = (f"\n\n<!-- completed-audit:begin -->\n## Completed independent audit\n\n"
        f"All {report['canonical_pairs']} model pairs passed, covering {report['paired_estimate_rows']:,} temperature/common-coefficient or contrast rows "
        f"and {sector['weather_coefficient_rows_checked']:,} paired own-series weather-coefficient rows. "
        f"The two isolated native receipts contain {report['native_models']:,} actual models and match the independent model/outcome/unit inventory. "
        f"All {preservation['unchanged_files']} checked original input/result/receipt hashes remained unchanged; "
        f"{preservation['provenance_only_omitted_unverified_count']} absent third-party literature artifacts are explicitly omitted/unverified, not checked inputs.\n\n"
        f"The coverage inventory accounts exactly once for {coverage['original_regression_rows']:,} original coefficient/covariance rows across "
        f"{coverage['original_regression_source_files']} source CSVs, every row of {coverage['original_component_covariance_joint_pair_source_files']} original "
        f"component/covariance/joint/pairwise companion CSVs, and {contrasts} structured linear-contrast results. "
        f"The common branch has {common['identical_preexcluded_pairs']} unchanged pre-2020 pairs. "
        f"Of {sector['joint_tests_checked']} sector heterogeneity tests, {sector['joint_available']} are full rank and "
        f"{sector['joint_singular_without_p']} correctly omit p-values because restriction covariance is singular. "
        f"All {sector['pairwise_tests_checked']:,} pairwise comparisons and Holm families passed.\n"
        "<!-- completed-audit:end -->\n")
    note_path.write_text(note+completed)
    print(f"Verified all COVID comparisons: {report['canonical_pairs']} pairs, {report['native_models']} native fits, {coverage['original_regression_rows']} original source rows.", flush=True)
    return report


if __name__ == "__main__":
    verify_covid_comparison()
