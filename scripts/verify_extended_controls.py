"""Verify extended-control source hashes and collapsed-time coefficients/HAC errors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify_extended_controls() -> dict:
    raw = ROOT / "data/raw/extended_controls"
    manifest = json.loads((raw / "manifest.json").read_text())
    for name, info in manifest.items():
        assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == info["sha256"], name
    audit = {"source_hashes_checked": len(manifest), "analyses": {}}
    for key, bandwidth in [("va", 4), ("iip-monthly", 12), ("iip-quarterly", 4)]:
        folder = ROOT / "outputs/extended_controls" / key
        panel = pd.read_csv(folder / "baseline_sample.csv", dtype={"sector_id": str})
        relaxed = pd.read_csv(folder / "sensitivity_80_sample.csv", dtype={"sector_id": str})
        outcome = "g_va_sa_qoq" if key == "va" else "g_iip_sa"
        saved = pd.read_csv(folder / "regression_results.csv")
        saved = saved.loc[saved.covariance.eq("Driscoll-Kraay")]
        audit["analyses"][key] = {}
        for model_id in ["C1", "C4", "C5", "C6"]:
            sample = relaxed if model_id == "C6" else panel
            if model_id == "C5":
                sample = sample[~sample.year.between(2020, 2022)]
            terms = ["temp_c", "rain_100mm", "rh_pct"]+([] if model_id == "C1" else ["wind_mps", "pm25_10ug"])
            grouped = sample.groupby("time").agg({outcome: "mean", "year": "first", "season": "first", **{c: "first" for c in terms}})
            covariates = grouped[terms]-grouped[terms].mean()
            x = np.column_stack([np.ones(len(grouped)), covariates,
                                 pd.get_dummies(grouped.year, drop_first=True, dtype=float),
                                 pd.get_dummies(grouped.season, drop_first=True, dtype=float)])
            y = grouped[outcome].to_numpy()
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            scores = np.zeros((int(grouped.index.max()-grouped.index.min()+1), x.shape[1]))
            scores[(grouped.index-grouped.index.min()).to_numpy(dtype=int)] = x*(y-x@beta)[:, None]
            meat = scores.T@scores
            for lag in range(1, bandwidth+1):
                cross = scores[lag:].T@scores[:-lag]
                meat += (1-lag/(bandwidth+1))*(cross+cross.T)
            bread = np.linalg.inv(x.T@x)
            t, n = len(grouped), len(sample)
            k = len(sample[["sector_id", "season"]].drop_duplicates())+sample.year.nunique()-1+len(terms)
            se = np.sqrt((bread@meat@bread)[1, 1]*t/(t-1)*(n-1)/(n-k))
            target = saved[saved.model.eq(model_id) & saved.term.eq("temp_c")].iloc[0]
            assert abs(beta[1]-target.coefficient) < 1e-8
            assert abs(se-target.std_error) < 1e-7
            audit["analyses"][key][model_id] = {"coefficient": beta[1], "std_error": se,
                "beta_error": abs(beta[1]-target.coefficient), "se_error": abs(se-target.std_error),
                "observed_periods": t, "calendar_periods": len(scores), "time_design_rank": int(np.linalg.matrix_rank(x)),
                "time_residual_df": t-int(np.linalg.matrix_rank(x)), "n_obs": n, "n_sectors": sample.sector_id.nunique()}
        for model_id in ["C1", "C2", "C3", "C4"]:
            row = saved[saved.model.eq(model_id)].iloc[0]
            assert int(row.n_obs) == len(panel)
        assert panel.wind_coverage.ge(.9).all() and panel.pm_coverage.ge(.9).all()
    (ROOT / "outputs/extended_controls/verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    print("Verified extended controls: source hashes, identical sample columns, period-mean OLS and direct calendar Bartlett HAC")
    return audit


if __name__ == "__main__":
    verify_extended_controls()
