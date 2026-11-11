"""Origin-cluster CR1 comparison for balanced common-weather FE regressions."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from visitor_analysis import DATA, OUT, inference, save_json


def run():
    results, coefficients = [], []
    with threadpool_limits(limits=1):
        for folder, source in [(OUT, "core"), (OUT / "crisis", "crisis")]:
            specs = json.loads((folder / "specifications.json").read_text())
            for spec in specs:
                if spec["panel"] not in ["region", "country"]:
                    continue
                model_id = spec["model_id"]
                z = np.load(folder / "designs" / f"{model_id}.npz")
                x, y, units = z["X"], z["y"], z["Y_units"]
                t, s = units.shape
                months = pd.PeriodIndex(z["months"], freq="M")
                deviations = pd.DataFrame(units-y[:, None], index=months)
                residual_units = (deviations-deviations.groupby(deviations.index.month).transform("mean")).to_numpy()+z["residual"][:, None]
                weights = np.full(s, 1/s)
                if spec["weighted"]:
                    w = pd.read_csv(DATA / f"{spec['panel']}_weights.csv", index_col=0).iloc[:, 0]
                    # Saved unit order is the original published catalog order.
                    catalog = pd.read_csv(DATA / "market_catalog.csv")
                    names = catalog.loc[catalog.panel.eq(spec["panel"]), "unit"]
                    weights = w.reindex(names).to_numpy()
                assert np.allclose(residual_units @ weights, z["residual"], atol=1e-7)
                q, r = np.linalg.qr(x, mode="reduced")
                influence_design = np.linalg.solve(r, q.T).T
                positions = [list(z["columns"]).index(term) for term in spec["terms"]]
                cluster_scores = residual_units.T @ influence_design[:, positions] * weights[:, None]
                n = t*s
                k = x.shape[1]+(s-1)*len(np.unique(months.month))
                correction = s/(s-1)*(n-1)/(n-k)
                covariance = cluster_scores.T @ cluster_scores * correction
                common = {key: spec[key] for key in ["model_id", "panel", "outcome", "sample", "variant", "n_months", "n_units", "n_panel_obs", "df_time_resid", "mean_y", "sd_y", "start", "end"]}
                common.update(source=source, inference_method="origin_CR1", n_clusters=s,
                    inference_df=s-1, full_fe_parameters=k, finite_sample_correction=correction)
                for j, term in enumerate(spec["terms"]):
                    value = inference(z["beta"][positions[j]], covariance[j, j], s-1)
                    coefficients.append({**common, "term": term,
                        **{key:value[key] for key in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value"]}})
                    if term == spec["temperature_term"]:
                        results.append({**common, **value})
                target = OUT / "origin_cluster/designs"
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / f"{model_id}.npz", covariance=covariance,
                    cluster_scores=cluster_scores, residual_units=residual_units, weights=weights,
                    terms=np.array(spec["terms"], str), correction=correction, full_fe_parameters=k)
    folder = OUT / "origin_cluster"
    pd.DataFrame(results).to_csv(folder / "regression_results.csv", index=False)
    pd.DataFrame(coefficients).to_csv(folder / "all_weather_coefficients.csv", index=False)
    save_json(folder / "manifest.json", {"status": "computed", "models": len(results),
        "primary_inference": "Calendar HAC retained; origin CR1 is a separate comparison.",
        "cluster_definition": "Nine regions or 42 named origin markets; arbitrary within-origin time dependence, independent origins assumed.",
        "finite_sample": "G/(G-1) * (N-1)/(N-K_full), t(G-1); full explicit origin×month and common-year FE parameter count.",
        "limits": "Origin clustering omits cross-origin common-shock dependence; nine regional clusters is small. It cannot infer a separate-origin slope from one origin cluster."})
    print(f"Computed origin CR1 comparison for {len(results)} models", flush=True)


if __name__ == "__main__":
    run()
