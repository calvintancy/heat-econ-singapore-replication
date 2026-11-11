"""Verify source hashes, transformations and panel/HAC estimates independently."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify() -> None:
    manifest = json.loads((ROOT / "data/raw/manifest.json").read_text())
    for name, item in manifest.items():
        content = (ROOT / "data/raw" / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"], f"Raw data changed: {name}"
    panel = pd.read_csv(ROOT / "data/processed/panel_broad.csv", dtype={"sector_id": str})
    results = pd.read_csv(ROOT / "outputs/regression_results.csv")
    results = results.loc[results.covariance.eq("Driscoll-Kraay")]
    checks = {"raw_source_hashes_checked": len(manifest), "manual_quarter_mean_checks": {}}
    for model_id, subset in [("S3", panel), ("S5", panel[~panel.year.between(2020, 2022)])]:
        q = subset.groupby("time").agg({"g_va_sa_qoq": "mean", "temp_c": "first", "rain_100mm": "first",
                                       "rh_pct": "first", "year": "first", "season": "first"})
        x = np.column_stack([np.ones(len(q)), q[["temp_c", "rain_100mm", "rh_pct"]],
                             pd.get_dummies(q.year, drop_first=True, dtype=float),
                             pd.get_dummies(q.season, drop_first=True, dtype=float)])
        y = q.g_va_sa_qoq.to_numpy()
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        scores = np.zeros((int(q.index.max() - q.index.min() + 1), x.shape[1]))
        scores[(q.index - q.index.min()).to_numpy(dtype=int)] = x * (y - x @ beta)[:, None]
        meat = scores.T @ scores
        for lag in range(1, 5):
            cross = scores[lag:].T @ scores[:-lag]
            meat += (1 - lag/5) * (cross + cross.T)
        bread = np.linalg.inv(x.T @ x)
        # Recover the panel's finite-sample multiplier; the uncorrected common-
        # regressor covariance equals the quarterly mean regression covariance.
        n = len(subset)
        k = subset[["sector_id", "season"]].drop_duplicates().shape[0] + subset.year.nunique() - 1 + 3
        correction = len(q)/(len(q)-1) * (n-1)/(n-k)
        se = np.sqrt((bread @ meat @ bread)[1, 1] * correction)
        target = results[(results.model.eq(model_id)) & (results.term.eq("temp_c")) &
                         (results.hac_lags.eq(4))].iloc[0]
        assert abs(beta[1] - target.coefficient) < 1e-9
        assert abs(se - target.std_error) < 1e-9
        checks["manual_quarter_mean_checks"][model_id] = {
            "coefficient": beta[1], "std_error": se,
            "beta_abs_error": abs(beta[1]-target.coefficient),
            "se_abs_error": abs(se-target.std_error),
            "observed_quarters": len(q), "calendar_quarters": len(scores),
        }
    # An independent leap-year quarter aggregation check using days from timestamps.
    monthly = pd.read_csv(ROOT / "data/raw/temperature_monthly.csv")
    monthly["date"] = pd.to_datetime(monthly.month)
    leap = monthly[monthly.month.isin(["2024-01", "2024-02", "2024-03"])]
    expected_temp = (leap.mean_temp * leap.date.dt.days_in_month).sum() / 91
    observed = panel[panel.quarter.eq("2024Q1")].temp_c.unique()
    assert len(observed) == 1 and abs(observed[0]-expected_temp) < 1e-12
    # Verify growth against raw levels, including the pre-weather VA lag.
    source = json.loads((ROOT / "data/raw/va_sa_tabledata.json").read_text())["Data"]["row"]
    manufacturing = next(r for r in source if r["seriesNo"] == "1.1.1")
    values = {c["key"]: float(c["value"]) for c in manufacturing["columns"]}
    expected_growth = 100 * np.log(values["1982 1Q"] / values["1981 4Q"])
    observed_growth = panel[panel.sector_id.eq("1.1.1") & panel.quarter.eq("1982Q1")].g_va_sa_qoq.iloc[0]
    assert abs(observed_growth-expected_growth) < 1e-10
    for name in ["temperature_coefficients", "partial_regression", "sector_slopes"]:
        assert (ROOT / f"outputs/{name}.png").stat().st_size > 10000
    checks.update({"leap_quarter_temperature_check": "passed", "raw_growth_check": "passed",
                   "versions": {name: importlib.metadata.version(name) for name in
                                ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "requests"]}})
    (ROOT / "outputs/verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("Verified source hashes, leap-year aggregation, raw growth, panel/quarter-mean OLS and independent Bartlett HAC")


if __name__ == "__main__":
    verify()
