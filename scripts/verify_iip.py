"""Independently reproduce IIP panel coefficients and calendar-aware HAC errors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify_iip() -> dict:
    checks = {}
    for frequency, period, bandwidth in [("monthly", "month", 12), ("quarterly", "quarter", 4)]:
        folder = ROOT / f"outputs/iip/{frequency}"
        panel = pd.read_csv(folder / "baseline_sample.csv", dtype={"sector_id": str})
        results = pd.read_csv(folder / "regression_results.csv")
        results = results.loc[results.covariance.eq("Driscoll-Kraay")]
        checks[frequency] = {}
        for model_id, subset in [("S3", panel), ("S5", panel[~panel.year.between(2020, 2022)])]:
            q = subset.groupby("time").agg({"g_iip_sa": "mean", "temp_c": "first", "rain_100mm": "first",
                                            "rh_pct": "first", "year": "first", "season": "first"})
            x = np.column_stack([np.ones(len(q)), q[["temp_c", "rain_100mm", "rh_pct"]],
                                 pd.get_dummies(q.year, drop_first=True, dtype=float),
                                 pd.get_dummies(q.season, drop_first=True, dtype=float)])
            y = q.g_iip_sa.to_numpy()
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            scores = np.zeros((int(q.index.max()-q.index.min()+1), x.shape[1]))
            scores[(q.index-q.index.min()).to_numpy(dtype=int)] = x*(y-x@beta)[:, None]
            meat = scores.T@scores
            for lag in range(1, bandwidth+1):
                cross = scores[lag:].T@scores[:-lag]
                meat += (1-lag/(bandwidth+1))*(cross+cross.T)
            bread = np.linalg.inv(x.T@x)
            n, t = len(subset), len(q)
            k = len(subset[["sector_id", "season"]].drop_duplicates()) + subset.year.nunique()-1+3
            correction = t/(t-1)*(n-1)/(n-k)
            se = np.sqrt((bread@meat@bread)[1, 1]*correction)
            target = results[results.model.eq(model_id) & results.term.eq("temp_c") & results.hac_lags.eq(bandwidth)].iloc[0]
            assert abs(beta[1]-target.coefficient) < 1e-9
            assert abs(se-target.std_error) < 1e-8
            checks[frequency][model_id] = {"coefficient": beta[1], "std_error": se,
                "beta_absolute_error": abs(beta[1]-target.coefficient), "se_absolute_error": abs(se-target.std_error),
                "observed_periods": t, "calendar_periods": len(scores), "hac_lags": bandwidth,
                "n_obs": n, "n_sectors": subset.sector_id.nunique()}
        fwl = json.loads((folder / "estimation_checks.json").read_text())
        assert abs(fwl["fwl_beta_difference"]) < 1e-9
    source_check = json.loads((ROOT / "data/processed/iip/data_checks.json").read_text())
    meta = json.loads((ROOT / "data/raw/iip_quarterly_nsa_metadata.json").read_text())["Data"]["records"]
    assert source_check["official_quarterly_reconciliation_cells"] == meta["total"]
    assert source_check["official_quarterly_max_rounding_gap"] < .002
    raw = json.loads((ROOT / "data/raw/iip_sa_tabledata.json").read_text())["Data"]["row"]
    electronics = {c["key"]: float(c["value"]) for r in raw if r["seriesNo"] == "1.1" for c in r["columns"]}
    expected_m = 100*np.log(electronics["1992 Feb"]/electronics["1992 Jan"])
    q1 = np.mean([electronics[f"1992 {m}"] for m in ["Jan", "Feb", "Mar"]])
    q2 = np.mean([electronics[f"1992 {m}"] for m in ["Apr", "May", "Jun"]])
    expected_q = 100*np.log(q2/q1)
    for frequency, period, label, expected in [("monthly", "month", "1992-02", expected_m), ("quarterly", "quarter", "1992Q2", expected_q)]:
        panel = pd.read_csv(ROOT / f"data/processed/iip/panel_{frequency}_broad.csv", dtype={"sector_id": str})
        observed = panel[panel.sector_id.eq("1.1") & panel[period].eq(label)].g_iip_sa.iloc[0]
        assert abs(observed-expected) < 1e-10
    checks["raw_growth_and_quarterly_aggregation"] = "passed"
    (ROOT / "outputs/iip/verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("Verified IIP raw growth, official quarter averages, panel/period-mean coefficients and independent calendar HAC")
    return checks


if __name__ == "__main__":
    verify_iip()
