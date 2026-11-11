"""Paper-motivated annual distributed-lag and quadratic growth associations.

Cached inputs only; isolated native PyFixest ledger; no published references are
rewritten. Full designs, native residuals, covariance and linear contrasts ship
with every estimate. See notes/literature_specification_protocol.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy.stats import f as f_dist
from threadpoolctl import threadpool_limits

import annual_analysis as annual
import annual_crisis_sensitivity as crisis
import pyfixest_backend as backend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/literature_sensitivity/annual"
MODELS = {"DJO0": 0, "DJO1": 1, "DJO5": 5, "DJO10": 10,
          "BHM_common_trend": 0, "BHM_unit_quadratic_trends": 0}
VARIANTS = ["all_eligible", "event_dates_excluded", "strict_growth_window_clean"]
OUTCOMES = ["y_nsa", "y_sa"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(x):
    scale = np.linalg.norm(x, axis=0)
    if (scale == 0).any():
        return int(np.linalg.matrix_rank(x))
    return int(np.linalg.matrix_rank(x / scale))


def parents():
    panels, provenance = crisis.contexts()
    weather = panels["aggregate_va"].drop_duplicates("year").set_index("year")
    for context, frame in panels.items():
        frame = frame.copy()
        for term in ["temp_c", "rain_100mm"]:
            lookup = weather[term].to_dict()
            for lag in range(11):
                frame[f"{term}_l{lag}"] = (frame.year-lag).map(lookup)
        panels[context] = frame
    return panels, provenance


def terms(model):
    if model.startswith("DJO"):
        lags = MODELS[model]
        return [f"temp_c_l{i}" for i in range(lags+1)], [f"rain_100mm_l{i}" for i in range(lags+1)]+["rh_pct"]
    return ["temp_center", "temp_center_sq"], ["rain_center", "rain_center_sq", "rh_pct"]


def sample_frame(parent, model, history, variant, policy):
    needed = 10 if history == "matched_10_year_weather_history" else MODELS[model]
    required = [f"{term}_l{i}" for term in ["temp_c", "rain_100mm"] for i in range(needed+1)]
    eligible = parent.loc[parent[required+OUTCOMES+["rh_pct"]].notna().all(axis=1)].copy()
    variants, _, _ = crisis.masks(eligible, policy)
    frame = variants[variant].reset_index(drop=True)
    if len(frame):
        unique = frame.drop_duplicates("year")
        center_t, center_p = unique.temp_c.mean(), unique.rain_100mm.mean()
        frame["temp_center"] = frame.temp_c-center_t
        frame["temp_center_sq"] = frame.temp_center**2
        frame["rain_center"] = frame.rain_100mm-center_p
        frame["rain_center_sq"] = frame.rain_center**2
        frame["trend"] = frame.year-frame.year.mean()
        frame["trend_sq"] = frame.trend**2
    return frame


def formula(model, kind):
    heat, controls = terms(model)
    exposed = " + ".join(heat)
    if kind == "H1":
        exposed = f"({exposed}) * C(sector_id)"
    trend = "(trend + trend_sq) * C(sector_id)" if model == "BHM_unit_quadratic_trends" else "trend + C(sector_id)"
    return "1 + "+exposed+" + "+" + ".join(controls)+" + "+trend


def contrast(names, term, sector, reference):
    vector = np.zeros(len(names)); vector[names.index(term)] = 1
    if sector != reference:
        column = f"{term}:C(sector_id)[T.{sector}]"
        if column in names:
            vector[names.index(column)] = 1
    return vector


def wald(beta, covariance, restriction, degrees):
    q = len(restriction)
    delta = restriction@beta; v = restriction@covariance@restriction.T
    singular = np.linalg.svd(v, compute_uv=False)
    rank_v = min(degrees,int(np.sum(singular > singular.max()*1e-10))) if singular.size and singular.max()>0 else 0
    condition = float(np.linalg.cond(v))
    result = {"restrictions": q, "covariance_rank": rank_v, "covariance_condition": condition,
              "status": "available", "f_statistic": None, "p_value": None}
    if rank_v < q or not np.isfinite(condition):
        result["status"] = "unavailable: singular restriction covariance"
    else:
        value = float(delta@np.linalg.solve(v, delta))
        if value < -1e-7:
            result["status"] = "unavailable: nonpositive covariance"
        else:
            result.update(f_statistic=max(value,0)/q, p_value=float(f_dist.sf(max(value,0)/q,q,degrees)))
    return result


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ["samples", "designs"]:
        (OUT/name).mkdir(exist_ok=True)
    policy = json.loads(crisis.POLICY.read_text())
    panels, provenance = parents()
    inputs = {p:sha(ROOT/p) for v in provenance.values() for p in v["source_files"]}
    protected_paths = [ROOT/"outputs/pyfixest/fit_log.csv", ROOT/"outputs/annual/pyfixest/fit_log.csv"]
    protected = {str(p.relative_to(ROOT)):sha(p) for p in protected_paths}
    books = {name:[] for name in ["specifications", "estimates", "tests", "sample_inventory"]}
    old = backend.LOG_DIR, list(backend._FIT_LOG), backend._API_COUNTER, backend._RUN_STARTED
    backend.LOG_DIR = OUT/"pyfixest"; backend.reset_fit_log()
    try:
        with threadpool_limits(limits=1):
            for context, parent in panels.items():
                for history in ["model_own_weather_history", "matched_10_year_weather_history"]:
                    for model in MODELS:
                        for variant in VARIANTS:
                            sample = sample_frame(parent,model,history,variant,policy)
                            sample_id = hashlib.sha256(json.dumps([context,history,model,variant]).encode()).hexdigest()[:16]
                            sample_path = OUT/"samples"/f"{sample_id}.csv"
                            sample.to_csv(sample_path,index=False)
                            t = sample.year.nunique()
                            sectors = sorted(sample.sector_id.unique())
                            inv = {"context":context, "history":history,"model_spec":model,"sample_variant":variant,
                                   "sample_csv":str(sample_path.relative_to(ROOT)),"n_periods":t,
                                   "parent_periods":parent.year.nunique(),"n_sectors":len(sectors),
                                   "sample_sha256":sha(sample_path)}
                            books["sample_inventory"].append(inv)
                            kinds = ["common"]+(["H1"] if len(sectors)>1 else [])
                            if history == "model_own_weather_history" and model in {"DJO5","BHM_unit_quadratic_trends"} and len(sectors)>1:
                                kinds.append("separate")
                            for kind in kinds:
                                units = [(s,g.reset_index(drop=True)) for s,g in sample.groupby("sector_id",sort=True)] if kind == "separate" else [("pooled",sample)]
                                for sid, data in units:
                                    rhs = formula(model,kind)
                                    spec_id = hashlib.sha256(json.dumps([sample_id,kind,sid]).encode()).hexdigest()[:20]
                                    meta = {**inv,"fit_group_id":spec_id,"model":kind,"sector_id":sid,
                                            "includes_dwellings":provenance[context]["includes_dwellings"],
                                            "matched_to":provenance[context]["matched_to"],"frequency":"annual",
                                            "estimator":"PyFixest","estimator_version":backend.ESTIMATOR_VERSION,
                                            "formula_rhs":rhs,"status":"unavailable","reason":"No eligible years"}
                                    if not len(data):
                                        books["specifications"].append(meta); continue
                                    xframe = patsy.dmatrix(rhs,data,NA_action="raise",return_type="dataframe")
                                    x = xframe.to_numpy(); names = list(xframe)
                                    unique = data.drop_duplicates("year")
                                    heat, controls = terms(model)
                                    time_terms = heat+controls+["trend"]+(["trend_sq"] if model == "BHM_unit_quadratic_trends" else [])
                                    tx = np.column_stack([np.ones(len(unique)), unique[time_terms]])
                                    k = x.shape[1]; r = rank(x); tr = rank(tx)
                                    meta.update(n_obs=len(data),n_sectors=data.sector_id.nunique(),n_parameters=k,
                                                design_rank=r,time_design_rank=tr,time_residual_df=t-tr,
                                                residual_df=len(data)-r,start=int(data.year.min()),end=int(data.year.max()),
                                                calendar_gaps=int(data.year.max()-data.year.min()+1-t),
                                                controls=json.dumps(controls),temperature_terms=json.dumps(heat),
                                                mean_temperature=float(unique.temp_c.mean()),
                                                temperature_min=float(unique.temp_c.min()),temperature_max=float(unique.temp_c.max()),
                                                condition_number=float(np.linalg.cond(x/np.linalg.norm(x,axis=0))),
                                                linear_trend=True,quadratic_unit_trends=model=="BHM_unit_quadratic_trends",
                                                stationarity_status="Not cleared by these regressions; gapped samples are not contiguous series")
                                    if r < k or len(data)<=k or t<=tr:
                                        meta["reason"] = f"Underidentified or no residual time df: rank {r}/{k}; temporal df {t-tr}"
                                        books["specifications"].append(meta); continue
                                    meta.update(status="estimated",reason="")
                                    label = f"Annual literature {context}/{history}/{model}/{variant}/{kind}/{sid}"
                                    fits = backend.fit_many_matrix(data[OUTCOMES],x,names,label,outcome_names=OUTCOMES,
                                        source_formula=rhs,row_index=data.index)
                                    design_path = OUT/"designs"/f"{spec_id}.npz"
                                    arrays = {"X":x,"Y":data[OUTCOMES].to_numpy(),"names":np.array(names),
                                              "time":data.year.to_numpy(),"sector_id":data.sector_id.to_numpy(dtype=str),
                                              "outcomes":np.array(OUTCOMES)}
                                    for oi, (outcome, fit) in enumerate(zip(OUTCOMES,fits)):
                                        current = {**meta,"spec_id":f"{spec_id}_{outcome}","outcome":outcome,
                                                   "output_adjustment":"NSA" if outcome=="y_nsa" else "SA",
                                                   "mean_y":float(data[outcome].mean()),"sd_y":float(data[outcome].std()),
                                                   "r_squared":float(fit.rsquared),"design_npz":str(design_path.relative_to(ROOT))}
                                        books["specifications"].append(current)
                                        arrays[f"beta_{oi}"] = fit.params.to_numpy(); arrays[f"residual_{oi}"] = fit.wresid
                                        for method, lag, covariance, degrees, primary in annual._covariances(fit,data,kind):
                                            code = "SC" if method=="Sector cluster" else f"DK{lag}"
                                            arrays[f"covariance_{oi}_{code}"] = covariance
                                            rec = {**current,"covariance":method,"hac_lags":lag,"inference_df":degrees,
                                                   "main_inference":primary,"n_clusters":len(sectors) if method=="Sector cluster" else 0}
                                            unit_ids = sorted(data.sector_id.unique()) if kind in {"H1","separate"} else ["pooled"]
                                            reference = sorted(data.sector_id.unique())[0]
                                            transforms = []
                                            for sector in unit_ids:
                                                c = np.vstack([contrast(names,h,sector,reference) for h in heat])
                                                transforms.append(c)
                                                unit = data if sector=="pooled" else data[data.sector_id.eq(sector)]
                                                detail = {**rec,"sector_id":sector,"sector":"National total" if len(sectors)==1 else "Equal-sector mean" if sector=="pooled" else unit.sector.iloc[0],
                                                          "mean_y":float(unit[outcome].mean()),"sd_y":float(unit[outcome].std())}
                                                if model.startswith("DJO"):
                                                    contrasts = [(f"T_lag_{j}",row,None) for j,row in enumerate(c)]
                                                    contrasts += [("temperature_lag_sum",c.sum(axis=0),None)]
                                                else:
                                                    contrasts = [("temperature_linear_centered",c[0],None),("temperature_quadratic",c[1],None)]
                                                    center = unique.temp_c.mean()
                                                    for percentile in [5,50,95]:
                                                        exposure = float(np.percentile(unique.temp_c,percentile))
                                                        contrasts.append((f"marginal_effect_p{percentile}",c[0]+2*(exposure-center)*c[1],exposure))
                                                for term, vector, exposure in contrasts:
                                                    value = float(vector@fit.params); variance=float(vector@covariance@vector)
                                                    books["estimates"].append({**detail,"term":term,"evaluation_temperature":exposure,
                                                        **annual._inference(value,variance,degrees)})
                                                tests = [("All temperature coefficients zero",c)]
                                                if model.startswith("DJO"):
                                                    tests.append(("Temperature coefficient sum zero",c.sum(axis=0,keepdims=True)))
                                                    if len(c)>1:
                                                        tests.append(("Static contemporaneous annual level channel",np.vstack([c[0]+c[1],c[2:]])))
                                                else:
                                                    tests.append(("Temperature curvature zero",c[[1]]))
                                                for null, restriction in tests:
                                                    books["tests"].append({**detail,"null":null,**wald(fit.params.to_numpy(),covariance,restriction,degrees)})
                                            if kind=="H1":
                                                rtest=np.vstack([c-transforms[0] for c in transforms[1:]])
                                                books["tests"].append({**rec,"sector_id":"all","null":"All sector temperature response functions equal",
                                                    **wald(fit.params.to_numpy(),covariance,rtest,degrees)})
                                    np.savez_compressed(design_path,**arrays)
                            print(f"Annual literature {context}/{history}/{model}/{variant}: {t} years",flush=True)
        ledger=backend.write_fit_log()
    finally:
        backend.LOG_DIR,previous,backend._API_COUNTER,backend._RUN_STARTED=old
        backend._FIT_LOG[:]=previous
    for name,rows in books.items():
        pd.DataFrame(rows).to_csv(OUT/f"{name}.csv",index=False)
    assert all(sha(ROOT/p)==h for p,h in {**inputs,**protected}.items())
    receipt={"status":"estimated_not_independently_verified","native_fits":ledger["native_fits_recorded"],
             "contexts":len(panels),"source_hashes":inputs,"protected_outputs":protected,
             "models":MODELS,"protocol_sha256":sha(ROOT/"notes/literature_specification_protocol.md"),
             "policy_sha256":sha(crisis.POLICY),"limits":"Single-country adaptations; no year FE annually. Exploratory associations, no causal/stationarity certificate. Direct lag sums are not unconditional dynamic multipliers."}
    (OUT/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
