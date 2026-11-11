"""Annual companion to the fixed-calendar, long-weather crisis sensitivity.

Read cached complete-year inputs only. Native PyFixest estimation follows the
existing annual sector-FE/trend design, with its own output directory/ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

import annual_analysis as annual
import pyfixest_backend as backend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/crisis_sensitivity/annual"
POLICY = ROOT / "notes/crisis_sensitivity_policy.json"
YS = {"NSA": ("y_nsa", "Annual log growth (NSA real output)"),
      "SA": ("y_sa", "Annual log growth (SA real output)"),
      "LOGLEVEL": ("y_log_level", "100 × log annual real output (SA; descriptive)"),
      "RAWLEVEL": ("y_raw_level", "Annual real output level (SA; descriptive)")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def contexts():
    panels, provenance = {}, {}
    for name in annual.CONTEXTS:
        path = ROOT / f"data/processed/annual/panel_{name}.csv"
        frame = pd.read_csv(path, dtype={"sector_id": str})
        panels[name] = frame
        provenance[name] = {"source_files": [rel(path)], "matched_to": None,
                            "includes_dwellings": name == "aggregate_va"}
    source = ROOT / "data/processed/annual/annual_levels.csv"
    levels = pd.read_csv(source, dtype={"sector_id": str})
    dwell = levels.loc[levels.family.eq("va") & levels.sector_id.eq("1.3")]
    dwell = dwell.pivot(index=["sector_id", "sector", "year"], columns="adjustment",
                        values=["annual_level", "annual_log_growth"])
    dwell.columns = [("annual_" if measure == "annual_level" else "y_")+adj.lower()
                     for measure, adj in dwell.columns]
    dwell = dwell.reset_index()
    fields = ["year", "temp_c", "rain_100mm", "rh_pct", "time", "trend"]
    for name, parent in [("va12", "va11"), ("va20", "va19")]:
        base = panels[parent]
        housing = dwell.merge(base[fields].drop_duplicates("year"), on="year", validate="one_to_one")
        required = ["sector_id", "sector", "annual_nsa", "annual_sa", "y_nsa", "y_sa"]+fields
        panels[name] = pd.concat([base[required], housing[required]], ignore_index=True)
        provenance[name] = {"source_files": provenance[parent]["source_files"]+[rel(source)],
                            "matched_to": None, "includes_dwellings": True}
    for name, parent, template in [
        ("aggregate_va_matched_va20", "aggregate_va", "va20"),
        ("va11_matched_va19", "va11", "va19"),
        ("va12_matched_va20", "va12", "va20"),
        ("iip_total_matched_iip6", "iip_total", "iip6"),
        ("iip_total_matched_iip18", "iip_total", "iip18"),
        ("iip6_matched_iip18", "iip6", "iip18")]:
        panels[name] = panels[parent].loc[panels[parent].year.isin(panels[template].year)].copy()
        provenance[name] = {"source_files": sorted(set(provenance[parent]["source_files"]+
                                                      provenance[template]["source_files"])),
                            "matched_to": template,
                            "includes_dwellings": provenance[parent]["includes_dwellings"]}
    for name, frame in panels.items():
        frame["y_log_level"] = 100*np.log(frame.annual_sa)
        frame["y_raw_level"] = frame.annual_sa
        frame = frame.sort_values(["sector_id", "year"]).reset_index(drop=True)
        assert frame[[v[0] for v in YS.values()]+annual.BASE].notna().all().all(), name
        assert frame.groupby("year").sector_id.nunique().eq(frame.sector_id.nunique()).all(), name
        panels[name] = frame
    return panels, provenance


def masks(frame, policy):
    years = sorted(frame.year.unique())
    event_years = {int(str(e["start"])[:4])+offset for e in policy["events"]
                   for offset in range(int(str(e["end"])[:4])-int(str(e["start"])[:4])+1)}
    return {"all_eligible": frame.copy(),
            "event_dates_excluded": frame.loc[~frame.year.isin(event_years)].copy(),
            "strict_growth_window_clean": frame.loc[~frame.year.isin(event_years) &
                                                      ~(frame.year-1).isin(event_years)].copy()}, years, event_years


def joint(beta, v, periods):
    r = np.eye(len(beta))[1:]-np.eye(len(beta))[[0]]
    d, c = r@beta, r@v@r.T
    singular = np.linalg.svd(c, compute_uv=False)
    rank = min(int(np.sum(singular > singular.max()*1e-10)), periods-2)
    q = len(d)
    result = {"null": "All sector temperature slopes equal", "restrictions": q,
              "covariance_rank": max(rank, 0), "structural_rank_bound": periods-2,
              "denominator_df": periods-1, "covariance_condition": float(np.linalg.cond(c)),
              "status": "available" if rank == q else "unavailable: singular contrast covariance",
              "f_statistic": None, "p_value": None}
    if rank == q:
        w = float(d@np.linalg.solve(c, d))
        assert w >= -1e-8
        result.update(f_statistic=max(w, 0)/q, p_value=float(f_dist.sf(max(w, 0)/q, q, periods-1)))
    return result


def run(policy_path=POLICY):
    policy = json.loads(Path(policy_path).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    panels, provenance = contexts()
    source_hashes = {path: sha(ROOT/path) for p in provenance.values() for path in p["source_files"]}
    protected = {rel(p): sha(p) for p in [ROOT/"outputs/pyfixest/fit_log.csv",
                                         ROOT/"outputs/annual/pyfixest/fit_log.csv"]}
    specs, common, profiles, series, joints, covrows, components, rows_removed = [], [], [], [], [], [], [], []
    old = annual.OUT, annual.ADJUSTMENTS, backend.LOG_DIR, list(backend._FIT_LOG), backend._API_COUNTER, backend._RUN_STARTED
    annual.OUT, annual.ADJUSTMENTS = OUT, {k:v[0] for k,v in YS.items()}
    backend.LOG_DIR = OUT / "pyfixest"
    backend.reset_fit_log()
    try:
        with threadpool_limits(limits=1):
            for context, parent in panels.items():
                samples, years, event_years = masks(parent, policy)
                for variant, sample in samples.items():
                    sample = sample.reset_index(drop=True)
                    t, s = sample.year.nunique(), sample.sector_id.nunique()
                    if not t:
                        raise ValueError(f"No remaining annual dates: {context}/{variant}")
                    for yr in years:
                        rows_removed.append({"context": context, "sample_variant": variant, "year": yr,
                            "retained": yr in set(sample.year), "current_year_event": yr in event_years,
                            "prior_year_event": yr-1 in event_years})
                    for adjustment, (outcome, label) in YS.items():
                        for kind in (["common", "H1", "separate"] if s > 1 else ["common"]):
                            units = [(sid, x.reset_index(drop=True)) for sid,x in sample.groupby("sector_id",sort=True)] if kind == "separate" else [(None, sample)]
                            for sid, data in units:
                                model, meta, terms = annual._fit(data, context, "weather", variant, adjustment, kind, sid)
                                meta.update(outcome_label=label, output_adjustment="NSA" if adjustment == "NSA" else "SA",
                                    analysis_role="descriptive_level" if "LEVEL" in adjustment else "annual_growth_sensitivity",
                                    includes_dwellings=provenance[context]["includes_dwellings"],
                                    matched_to=provenance[context]["matched_to"], policy_id=policy["policy_id"],
                                    frequency="annual", year_fixed_effects=False, sector_fixed_effects=data.sector_id.nunique() > 1,
                                    season_fixed_effects=False, trend=True, controls_label="Temperature, rainfall, humidity",
                                    calendar_mask="Current/prior full-calendar years event-free; same dates across four annual outcomes" if variant == "strict_growth_window_clean" else variant,
                                    stationarity_status="Not cleared: gapped exact sample" if meta["calendar_gaps"] else "No new stationarity clearance; levels descriptive")
                                if model is not None:
                                    meta["r_squared"] = float(model.rsquared)
                                specs.append(meta)
                                if model is None:
                                    continue
                                n, k = model.model.exog.shape
                                for method, band, matrix, df, primary in annual._covariances(model, data, kind):
                                    factor = s/(s-1)*(n-1)/(n-k) if method == "Sector cluster" else t/(t-1)*(n-1)/(n-k)
                                    record = {**meta, "covariance": method, "hac_lags": band, "inference_df": df,
                                        "finite_sample_correction": factor, "main_inference": primary,
                                        "n_clusters": s if method == "Sector cluster" else 0}
                                    if kind != "H1":
                                        j = terms.index("temp_c")
                                        target = series if kind == "separate" else common
                                        target.append({**record, "sector": data.sector.iloc[0] if sid else "Published national total" if s == 1 else "Equal-component mean",
                                                       **annual._inference(float(model.params.iloc[j]), matrix[j,j], df)})
                                    else:
                                        sectors = sorted(data.sector_id.unique())
                                        transform = np.zeros((s,k)); transform[:,terms.index("temp_c")] = 1
                                        for j, sector in enumerate(sectors[1:],1):
                                            transform[j,terms.index(f"temp_c:C(sector_id)[T.{sector}]")] = 1
                                        beta = transform@model.params.to_numpy(); v = transform@matrix@transform.T
                                        component = np.eye(s); component[1:,0] = -1
                                        cv, cb = component@v@component.T, component@beta
                                        for j, sector in enumerate(sectors):
                                            unit = data.loc[data.sector_id.eq(sector)]
                                            profiles.append({**record, "sector_id": sector, "sector": unit.sector.iloc[0],
                                                "mean_y": float(unit[outcome].mean()), "sd_y": float(unit[outcome].std()),
                                                "temp_main": beta[0], "interaction_delta": beta[j]-beta[0],
                                                "var_main": cv[0,0], "var_delta": 0. if j == 0 else cv[j,j],
                                                "cov_main_delta": 0. if j == 0 else cv[0,j],
                                                **annual._inference(beta[j], v[j,j], df)})
                                            components.append({**record, "sector_id": sector, "component": "base" if j == 0 else "interaction",
                                                               **annual._inference(cb[j], cv[j,j], df)})
                                            for h, other in enumerate(sectors):
                                                covrows.append({"spec_id": meta["spec_id"], "covariance": method, "hac_lags": band,
                                                                "sector_i": sector, "sector_j": other, "value": v[j,h]})
                                        joints.append({**record, **joint(beta,v,t)})
                    print(f"Annual crisis {context}/{variant}: {s} units × {t} years",flush=True)
        native = backend.write_fit_log()
    finally:
        annual.OUT, annual.ADJUSTMENTS, backend.LOG_DIR, previous, backend._API_COUNTER, backend._RUN_STARTED = old
        backend._FIT_LOG[:] = previous
    for name, records in [("specs", specs), ("common_estimates", common), ("sector_estimates", profiles),
                          ("separate_estimates", series), ("joint_tests", joints), ("sector_covariances", covrows),
                          ("interaction_components", components), ("calendar_masks", rows_removed)]:
        frame = pd.DataFrame(records)
        if name in {"sector_estimates", "separate_estimates"}:
            frame["p_value_holm_within_table"] = frame.groupby(["context","sample_variant","adjustment","covariance","hac_lags"]).p_value.transform(lambda x:multipletests(x,method="holm")[1])
        frame.to_csv(OUT/f"{name}.csv",index=False)
    assert all(sha(ROOT/p)==h for p,h in {**source_hashes,**protected}.items())
    receipt = {"status":"estimated_not_independently_verified", "policy_id":policy["policy_id"],
        "policy_sha256":sha(policy_path), "contexts":len(panels), "native_fits":native["native_fits_recorded"],
        "requested_models":len(specs), "unavailable_models":sum(x["status"]!="estimated" for x in specs),
        "source_hashes":source_hashes,"protected_outputs":protected,"context_provenance":provenance,
        "controls":annual.BASE,"annual_output":"VA sums four quarters; IIP averages twelve months. Growth computed before exclusions.",
        "FE":"Sector intercepts and common linear trend; no year FE at annual national-weather frequency.",
        "inference":"Requested sector CR1 for common pooled slopes with annual calendar DK1/DK2 retained; profiles/separate/national calendar HAC. No invented p-values for rank-deficient omnibus tests.",
        "limits":"Calendar masks are sensitivity choices. No new stationarity or causal certificate. Complete VA12/20 includes dwellings; chain-linked levels need not add to published GVA."}
    (OUT/"manifest.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",type=Path,default=POLICY)
    print(json.dumps(run(parser.parse_args().policy),indent=2,ensure_ascii=False))
