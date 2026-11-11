"""Zero-retaining native PPML sensitivity with common-shock calendar inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from threadpoolctl import threadpool_limits

from visitor_analysis import BASE, FULL, OUT, DATA, influence_hac, inference, save_json


def weighted_residualize(x, weights, groups):
    """Alternating weighted projections on additive FE spaces."""
    result = x.copy()
    for iteration in range(10000):
        previous = result.copy()
        for codes in groups:
            denominator = np.bincount(codes, weights=weights)
            means = np.column_stack([np.bincount(codes, weights=weights*result[:, j]) / denominator
                                     for j in range(result.shape[1])])
            result -= means[codes]
        if np.max(np.abs(result-previous)) < 1e-11:
            return result, iteration+1
    raise RuntimeError("Weighted FE projection failed to converge")


def run():
    counts = pd.read_csv(DATA / "all_source_counts.csv", index_col="month")
    catalog = pd.read_csv(DATA / "market_catalog.csv")
    weather = pd.read_csv(DATA / "weather.csv")
    rows, receipts = [], []
    folder = OUT / "ppml"
    folder.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for panel in ["aggregate", "region", "country"]:
            markets = catalog.loc[catalog.panel.eq(panel), "unit"]
            long = counts[markets].stack().rename("arrivals").reset_index().rename(columns={"level_1": "unit"})
            long = long.merge(weather, on="month", validate="many_to_one")
            long["unit_season"] = (long.unit + "_" + long.season.astype(str)).astype("string")
            for sample in ["long_all", "long_pre2020", "full_all", "full_clean"]:
                terms = FULL if sample.startswith("full") else BASE
                mask = long[terms].notna().all(axis=1)
                if sample.startswith("full"):
                    mask &= long.eligible_90.eq(True) & long.year.between(2017, 2025)
                if sample.endswith("pre2020"):
                    mask &= long.year.lt(2020)
                if sample.endswith("clean"):
                    # Count levels have no lagged base period.
                    mask &= ~long.year.between(2020, 2022)
                frame = long.loc[mask].copy().reset_index(drop=True)
                formula = "arrivals ~ " + " + ".join(terms) + " | unit_season + year"
                fit = pf.fepois(formula, frame, vcov="iid", iwls_tol=1e-11, iwls_maxiter=200,
                               collin_tol=1e-12, fixef_rm="none")
                assert fit.sample_info.n_obs == len(frame) and not fit.sample_info.dropped_row_index
                beta = fit.coef().reindex(terms).to_numpy()
                residual = np.asarray(fit.resid())
                mu = frame.arrivals.to_numpy()-residual
                assert np.isfinite(mu).all() and (mu>0).all()
                assert np.allclose(np.exp(fit.predict()), mu, rtol=1e-6, atol=1e-4)
                groups = [pd.factorize(frame.unit_season)[0], pd.factorize(frame.year)[0]]
                xr, iterations = weighted_residualize(frame[terms].to_numpy(), mu, groups)
                bread = np.linalg.inv(xr.T @ (xr * mu[:, None]))
                scores = xr * residual[:, None]
                score_months = pd.DataFrame(scores).groupby(frame.time).sum()
                influence = score_months.to_numpy() @ bread
                t = len(score_months)
                k_time = 12 + frame.year.nunique()-1+len(terms)
                correction = t/(t-k_time)
                model_id = f"{panel}__{sample}"
                receipt = {"model_id": model_id, "panel": panel, "sample": sample,
                    "estimator": f"PyFixest {pf.__version__} fepois", "formula": formula,
                    "n_obs": len(frame), "n_units": len(markets), "n_months": t,
                    "zero_cells_retained": int(frame.arrivals.eq(0).sum()),
                    "start": frame.month.min(), "end": frame.month.max(), "df_time_resid": t-k_time,
                    "weighted_projection_iterations": iterations,
                    "relative_score_error": float(np.max(np.abs(frame[terms].to_numpy().T @ residual)) / frame.arrivals.sum())}
                assert receipt["relative_score_error"] < 1e-6
                receipts.append(receipt)
                for bandwidth in [12, 24]:
                    covariance = influence_hac(influence, score_months.index.to_numpy(), bandwidth, correction)
                    rows.append({**receipt, "hac_lags": bandwidth,
                                 **inference(100*beta[0], 10000*covariance[0, 0], t-1)})
                    np.savez_compressed(folder / f"{model_id}__{bandwidth}.npz",
                        X=frame[terms].to_numpy(), y=frame.arrivals.to_numpy(), mu=mu,
                        groups=np.column_stack(groups), xr=xr, beta=beta, covariance=covariance,
                        time=frame.time.to_numpy(), correction=correction, terms=np.array(terms, str))
                frame[["month", "unit", "arrivals", "unit_season", "year", "time"]+terms].to_csv(folder / f"{model_id}_sample.csv", index=False)
                print(f"PPML {model_id}: {t} months, {receipt['zero_cells_retained']} zeros retained", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ppml_results.csv", index=False)
    save_json(folder / "estimator_receipts.json", receipts)


if __name__ == "__main__":
    run()
