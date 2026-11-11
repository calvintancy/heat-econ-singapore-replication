"""Independently check pooled sector-CR1 rows using QR OLS and sector scores.

Patsy builds the saved formula's design, but neither the estimation module nor
statsmodels' covariance routine is imported. Exact samples are reconstructed
from saved inputs; new analyses can supply ``samples/{model}.csv`` beside their
result CSV. Existing calendar-HAC rows are also checked for identical betas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def close(actual, expected, label: str, *, atol: float = 2e-7, rtol: float = 2e-7) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def read_sample(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sector_id": str})
    if "sector_id" not in frame:
        raise AssertionError(f"Missing clustering unit in {path}")
    return frame


def exact_sample(path: Path, model: str) -> pd.DataFrame:
    """Resolve the original estimator's exact observations without importing it."""
    saved_sample = path.parent / "samples" / f"{model}.csv"
    if saved_sample.exists():
        return read_sample(saved_sample)
    relative = path.relative_to(OUT)
    parts = relative.parts
    if parts[0] == "outcomes":
        frame = read_sample(path.parent / "common_sample.csv")
        return frame.loc[~frame.year.between(2020, 2022)].reset_index(drop=True) if model.endswith("X") else frame
    if parts[0] == "extended_controls":
        if model == "C0":
            context = parts[1]
            source = ROOT / "data/processed/panel_broad.csv" if context == "va" else ROOT / f"data/processed/iip/panel_{'monthly' if context == 'iip-monthly' else 'quarterly'}_broad.csv"
            return read_sample(source)
        source = path.parent / ("sensitivity_80_sample.csv" if model == "C6" else "baseline_sample.csv")
        frame = read_sample(source)
        return frame.loc[~frame.year.between(2020, 2022)].reset_index(drop=True) if model == "C5" else frame
    if len(parts) == 1 or parts[0] == "iip":
        period = "month" if len(parts) > 1 and parts[1] == "monthly" else "quarter"
        prefix = ROOT / "data/processed"
        broad_source = prefix / "panel_broad.csv" if len(parts) == 1 else prefix / f"iip/panel_{parts[1]}_broad.csv"
        detailed_source = prefix / "panel_detailed.csv" if len(parts) == 1 else prefix / f"iip/panel_{parts[1]}_detailed.csv"
        broad = read_sample(broad_source)
        if path.name == "distributed_lag_results.csv":
            horizon = 12 if period == "month" else 4
            return broad.dropna(subset=["temp_c"]+[f"temp_l{i}" for i in range(1, horizon+1)]).reset_index(drop=True)
        if model == "S4":
            return broad.loc[broad.year.lt(2020)].reset_index(drop=True)
        if model == "S5":
            return broad.loc[~broad.year.between(2020, 2022)].reset_index(drop=True)
        if model in {"S6", "S7"}:
            detailed = read_sample(detailed_source)
            if model == "S7":
                return detailed
            return broad.loc[broad[period].between(detailed[period].min(), detailed[period].max())].reset_index(drop=True)
        return broad
    raise ValueError(f"No exact-sample rule for {relative}/{model}; save samples/{model}.csv")


def manual_fit(sample: pd.DataFrame, formula: str) -> dict:
    y_frame, x_frame = dmatrices(formula, sample, return_type="dataframe", NA_action="raise")
    assert len(x_frame) == len(sample), "Formula changed the saved estimation sample"
    x, y = x_frame.to_numpy(), y_frame.iloc[:, 0].to_numpy()
    n, k = x.shape
    q, triangular = np.linalg.qr(x, mode="reduced")
    assert np.min(np.abs(np.diag(triangular))) > 1e-10, "Rank-deficient independent design"
    beta = np.linalg.solve(triangular, q.T @ y)
    residual = y-x @ beta
    inverse_r = np.linalg.solve(triangular, np.eye(k))
    bread = inverse_r @ inverse_r.T
    sectors = sample.sector_id.unique()
    assert len(sectors) >= 2
    meat = np.zeros((k, k))
    # Explicit unit-by-unit score sums are independent of statsmodels' grouping.
    for sector in sectors:
        mask = sample.sector_id.eq(sector).to_numpy()
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    correction = len(sectors)/(len(sectors)-1)*(n-1)/(n-k)
    covariance = correction*(bread @ meat @ bread)
    return {"beta": beta, "covariance": covariance, "terms": list(x_frame.columns),
            "n_obs": n, "n_parameters": k, "n_clusters": len(sectors),
            "correction": correction, "inference_df": len(sectors)-1}


def verify_file(path: Path) -> list[dict]:
    saved = pd.read_csv(path)
    clustered = saved.loc[saved.covariance.eq("Sector cluster")]
    common = saved.loc[saved.covariance.eq("Driscoll-Kraay") & saved.hac_lags.isin([4, 12])]
    common = common.loc[common.n_sectors.ge(2) & ~common.term.str.contains("sector_id", regex=False)]
    if common.empty and clustered.empty:
        return []
    assert not clustered.duplicated(["model", "term"]).any(), f"Duplicate sector-cluster rows: {path}"
    expected = set(zip(common.model, common.term))
    actual = set(zip(clustered.model, clustered.term))
    assert actual == expected, f"Sector-cluster coverage differs in {path}: missing={expected-actual}, extra={actual-expected}"
    records = []
    for model, rows in clustered.groupby("model", sort=False):
        assert rows.formula.nunique() == 1
        formula = rows.formula.iloc[0]
        sample = exact_sample(path, model)
        independent = manual_fit(sample, formula)
        dof = independent["inference_df"]
        critical = student_t.ppf(.975, dof)
        max_beta_error, max_se_error = 0.0, 0.0
        for row in rows.itertuples():
            label = f"{path.relative_to(ROOT)}/{model}/{row.term}"
            j = independent["terms"].index(row.term)
            beta = float(independent["beta"][j])
            se = float(np.sqrt(independent["covariance"][j, j]))
            assert np.isfinite(se) and se > 0, f"Degenerate common-slope clustered SE: {label}"
            close(row.coefficient, beta, f"{label} beta")
            close(row.std_error, se, f"{label} sector CR1 SE")
            close(row.p_value, 2*student_t.sf(abs(beta/se), dof), f"{label} t(S-1) p")
            close([row.ci_lower, row.ci_upper], [beta-critical*se, beta+critical*se], f"{label} CI")
            assert int(row.inference_df) == dof and int(row.hac_lags) == 0, label
            assert int(row.n_obs) == independent["n_obs"], label
            assert int(row.n_sectors) == independent["n_clusters"], label
            assert int(row.n_clusters) == independent["n_clusters"], label
            assert int(row.n_parameters) == independent["n_parameters"], label
            close(row.finite_sample_correction, independent["correction"], f"{label} CR1 multiplier", atol=1e-12)
            benchmark = saved.loc[saved.model.eq(model) & saved.term.eq(row.term)
                                  & saved.covariance.eq("Driscoll-Kraay")]
            assert not benchmark.empty, f"Missing unchanged HAC benchmark: {label}"
            close(benchmark.coefficient, row.coefficient, f"{label} covariance-invariant coefficient", atol=1e-12, rtol=1e-12)
            max_beta_error = max(max_beta_error, abs(row.coefficient-beta))
            max_se_error = max(max_se_error, abs(row.std_error-se))
        records.append({"coefficient_csv": str(path.relative_to(ROOT)), "model": model,
                        "terms_checked": len(rows), "n_obs": independent["n_obs"],
                        "n_clusters": independent["n_clusters"], "n_parameters": independent["n_parameters"],
                        "inference_df": dof, "max_beta_absolute_error": max_beta_error,
                        "max_se_absolute_error": max_se_error})
    return records


def verify_sector_cluster(paths: list[Path] | None = None) -> dict:
    # Visitor-arrival estimates have their own schema, inference and independent audit.
    selected = paths or sorted(path for path in [*OUT.rglob("regression_results.csv"), *OUT.rglob("distributed_lag_results.csv")]
                               if "visitor_arrivals" not in path.parts)
    records = []
    for path in selected:
        records.extend(verify_file(path))
    assert records, "No sector-cluster estimates found; rerun pooled estimation first"
    # Baseline sector profiles retain calendar HAC. The separately labelled,
    # explicitly requested cluster sensitivity has its own independent audit and
    # degeneracy diagnostics; it must not change this baseline coverage check.
    excluded_files = [path for path in [*OUT.rglob("sector_slopes.csv"), *OUT.rglob("aggregate_results.csv")]
                      if "clustered_sector_sensitivity" not in path.parts]
    for path in excluded_files:
        saved = pd.read_csv(path)
        if "covariance" in saved:
            assert not saved.covariance.eq("Sector cluster").any(), f"Invalid sector clustering in {path}"
    audit = {"method": "Independent full-design QR OLS, explicit sector score sums, CR1, t(S-1)",
             "model_fits_checked": len(records), "coefficient_rows_checked": sum(row["terms_checked"] for row in records),
             "files_checked": len({row["coefficient_csv"] for row in records}),
             "excluded_single_unit_or_sector_specific_files_checked": len(excluded_files), "models": records}
    (OUT / "sector_cluster_verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(f"Verified sector CR1: {audit['coefficient_rows_checked']} coefficients across {len(records)} model fits and {audit['files_checked']} result files", flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", type=Path, help="Check only this result CSV; can repeat")
    arguments = parser.parse_args()
    paths = [path.resolve() for path in arguments.csv] if arguments.csv else None
    verify_sector_cluster(paths)
