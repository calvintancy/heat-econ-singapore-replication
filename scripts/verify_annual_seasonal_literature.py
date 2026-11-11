"""Independent annual seasonal/AR calendar, full QR, covariance and null audit."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, t as student_t
from threadpoolctl import threadpool_limits
from verify_annual_literature_sensitivity import parents as raw_parents, qr
from verify_outcomes import raw_weather

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/literature_sensitivity/annual_seasonal"
MODELS=["seasonal_current","seasonal_current_prior","seasonal_current_AR1","seasonal_current_prior_AR1"]
HISTORIES=["model_own_history","matched_max_AR1_history"]
YS=["y_nsa","y_sa"]
VARIANTS=["all_eligible","event_dates_excluded","strict_growth_window_clean"]
COUNTS=Counter();ERRORS={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return pd.read_csv(path,dtype={"sector_id":str,"reference_sector_id":str},low_memory=False)


def close(actual,expected,label,atol=1e-7,rtol=1e-7):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)),(label,"shape/missing")
    keep=np.isfinite(a)&np.isfinite(b)
    assert np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b))
    error=np.abs(a[keep]-b[keep]);allow=atol+rtol*np.abs(b[keep])
    if np.any(error>allow):
        i=np.argmax(error-allow);raise AssertionError((label,"actual",a[keep][i],"expected",b[keep][i],"allowed",allow[i]))
    COUNTS["numeric_values"]+=int(keep.sum())
    if keep.any():ERRORS[label]=max(ERRORS.get(label,0.),float(np.max(error/np.maximum(1.,np.abs(b[keep])))))


def features(model):
    lags=[0,1] if "prior" in model else [0]
    heat=[f"temp_Q{q}_l{lag}" for lag in lags for q in range(1,5)]
    controls=[f"rain_Q{q}_l{lag}" for lag in lags for q in range(1,5)]+["rh_pct"]
    if model.endswith("AR1"):controls.append("ar_l1")
    return heat,controls


def parents():
    panels,raw_counts,matches=raw_parents()
    monthly_manifest=json.loads((ROOT/"data/raw/manifest.json").read_text())
    for filename in ["temperature_monthly.csv","rainfall_monthly.csv","humidity_monthly.csv"]:
        assert sha(ROOT/"data/raw"/filename)==monthly_manifest[filename]["sha256"]
    quarterly=raw_weather("Q")
    saved=read(ROOT/"data/processed/weather_quarterly.csv")
    assert set(saved.quarter)=={str(q) for q in quarterly}
    for row in saved.itertuples():
        source=quarterly[pd.Period(row.quarter,freq="Q")]
        close([row.temp_c,row.rain_mm/100,row.rh_pct],[source[t] for t in ["temp_c","rain_100mm","rh_pct"]],"Raw quarterly weather")
    years={q.year for q in quarterly if all(pd.Period(f"{q.year}Q{i}",freq="Q") in quarterly for i in range(1,5))}
    levels=read(ROOT/"data/processed/annual/annual_levels.csv")
    registry={(r.family,r.sector_id,r.adjustment,int(r.year)):r.annual_log_growth for r in levels.itertuples()}
    for context,parent in panels.items():
        frame=parent.drop(columns=[c for c in parent if c.startswith(("temp_c_l","rain_100mm_l"))]).sort_values(["year","sector_id"]).reset_index(drop=True)
        for prefix,source_term in [("temp","temp_c"),("rain","rain_100mm")]:
            for q in range(1,5):
                for lag in [0,1]:
                    frame[f"{prefix}_Q{q}_l{lag}"]=[quarterly[pd.Period(f"{year-lag}Q{q}",freq="Q")][source_term] if year-lag in years else np.nan for year in frame.year]
        family="iip" if context.startswith("iip") else "va"
        for outcome,adjustment in [("y_nsa","NSA"),("y_sa","SA")]:
            close(frame[outcome],[registry[family,sid,adjustment,int(year)] for sid,year in zip(frame.sector_id,frame.year)],"Current raw annual growth")
            frame[f"ar_l1_{outcome}"]=[registry.get((family,sid,adjustment,int(year)-1),np.nan) for sid,year in zip(frame.sector_id,frame.year)]
        panels[context]=frame
    return panels,matches,raw_counts


def eligible(parent,model,history,outcome):
    frame=parent.copy();frame["ar_l1"]=frame[f"ar_l1_{outcome}"]
    heat,controls=features(MODELS[-1] if history==HISTORIES[-1] else model)
    available=np.isfinite(frame[[outcome]+heat+controls]).all(axis=1)
    years=available.groupby(frame.year).all()
    return frame[frame.year.isin(years.index[years])].copy()


def sample(parent,model,history,outcome,variant,event_years,target):
    frame=eligible(parent,model,history,outcome);raw_n=frame.year.nunique();target_n=None
    if target is not None:
        target=eligible(target,model,history,outcome);target_n=target.year.nunique()
        frame=frame[frame.year.isin(target.year)]
    masks=[]
    for year in sorted(frame.year.unique()):
        support=sorted(set(range(int(year)-2,int(year)+1))&event_years)
        keep=variant==VARIANTS[0] or (year not in event_years if variant==VARIANTS[1] else not support)
        masks.append({"year":int(year),"retained":bool(keep),"current_year_event":year in event_years,
                      "event_years_in_output_support":json.dumps(support),"output_support_start":int(year)-2})
    data=frame[frame.year.isin([r["year"] for r in masks if r["retained"]])].reset_index(drop=True)
    return data,masks,raw_n,target_n,frame.year.nunique()


def design(frame,model,kind):
    sectors=sorted(frame.sector_id.unique());heat,controls=features(model)
    columns={f"sector[{sid}]":frame.sector_id.eq(sid).to_numpy(float) for sid in sectors}
    columns["trend"]=(frame.year-frame.year.mean()).to_numpy()
    for term in controls:columns[term]=(frame[term]-frame[term].mean()).to_numpy()
    for term in heat:
        centered=(frame[term]-frame[term].mean()).to_numpy();columns[term]=centered
        if kind=="H1":
            for sid in sectors[1:]:columns[f"{term}:C(sector_id)[T.{sid}]"]=centered*frame.sector_id.eq(sid).to_numpy()
    names=list(columns);x=np.column_stack(list(columns.values()))
    selected=np.arange(len(sectors),len(names));components=[]
    for term in heat:
        components.append([names.index(term)]+([names.index(f"{term}:C(sector_id)[T.{sid}]") for sid in sectors[1:]] if kind=="H1" else []))
    return x,names,selected,np.asarray(components),sectors


def rank(x):
    scales=np.linalg.norm(x,axis=0);z=x/np.where(scales>0,scales,1.)
    singular=np.linalg.svd(z,compute_uv=False);cutoff=singular.max()*max(x.shape)*np.finfo(float).eps
    return int(np.sum(singular>cutoff)),cutoff


def covariance(inverse,residual,sample,method,band):
    n,k=inverse.shape[1],inverse.shape[0];influence=inverse.T*residual[:,None];t=sample.year.nunique()
    if method=="Sector cluster":
        units=sorted(sample.sector_id.unique());scores=np.vstack([influence[sample.sector_id.eq(sid)].sum(axis=0) for sid in units])
        factor=len(units)/(len(units)-1)*(n-1)/(n-k);degrees=len(units)-1
    else:
        years=sample.year.to_numpy();scores=np.zeros((years.max()-years.min()+1,k));np.add.at(scores,years-years.min(),influence)
        factor=t/(t-1)*(n-1)/(n-k);degrees=t-1
    matrix=scores.T@scores
    if method!="Sector cluster":
        for lag in range(1,band+1):
            cross=scores[lag:].T@scores[:-lag];matrix+=(1-lag/(band+1))*(cross+cross.T)
    return (matrix+matrix.T)/2*factor,influence.T@influence,degrees,factor


def effects(basis,model):
    lags=[0,1] if "prior" in model else [0]
    result=[("season_lag",q,lag,f"temp_Q{q}_l{lag}",basis[i]) for i,(lag,q) in enumerate((lag,q) for lag in lags for q in range(1,5))]
    result.append(("current_year_sum",0,0,"sum_current_Q1_Q4",basis[:4].sum(axis=0)))
    if "prior" in model:
        result.append(("prior_year_sum",0,1,"sum_prior_Q1_Q4",basis[4:].sum(axis=0)))
        result += [("same_season_pair_sum",q,-1,f"Q{q}_current_plus_prior",basis[q-1]+basis[q+3]) for q in range(1,5)]
    return result+[("all_season_sum",0,-1,"sum_all_temperature_features",basis.sum(axis=0))]


def nulls(basis,model):
    rows=[("All seasonal temperature coefficients zero",basis),("Current Q1–Q4 temperature coefficients equal",basis[1:4]-basis[[0]]),
          ("Sum of all direct seasonal temperature coefficients zero",basis.sum(axis=0,keepdims=True))]
    if "prior" in model:
        rows += [("Prior Q1–Q4 temperature coefficients equal",basis[5:]-basis[[4]]),
                 ("Current and prior same-quarter temperature coefficients equal",basis[:4]-basis[4:]),
                 ("All same-quarter current-plus-prior coefficient sums zero",basis[:4]+basis[4:])]
    return rows


def point(row,beta,var,hc0,df):
    close([row.coefficient,row.variance,row.hc0_reference_variance,row.zero_variance_threshold],[beta,var,hc0,1e-12*max(hc0,0)],"Point/HC0 variance")
    degenerate=abs(var)<=1e-12*max(hc0,0) or df<=0 or hc0<=0
    assert bool(row.degenerate)==degenerate
    if degenerate:
        assert pd.isna(row.p_value) and pd.isna(row.ci_lower) and pd.isna(row.ci_upper)
        assert row.inference_status=="unavailable: numerical variance degeneracy"
        close(row.std_error,math.sqrt(max(row.variance,0)),"Retained numerical SE",atol=1e-15,rtol=1e-12)
        COUNTS["degenerate_point_rows"]+=1
    else:
        se=math.sqrt(var);critical=student_t.ppf(.975,df);p=2*student_t.sf(abs(beta/se),df)
        close([row.std_error,row.ci_lower,row.ci_upper,row.p_value],[se,beta-critical*se,beta+critical*se,p],"SE/CI/p")
        close(p,f_dist.sf(beta*beta/var,1,df),"Total-effect t/F identity")
        assert row.inference_status=="available"


def wald(row,beta,v,hc0,r,df,bound):
    matrix=r@v@r.T;matrix=(matrix+matrix.T)/2;delta=r@beta
    scale=float(np.linalg.norm(r@hc0@r.T,2));sv=np.linalg.svd(matrix,compute_uv=False)
    cutoff=max(sv[0]*1e-10,scale*1e-12);uncapped=int(np.sum(sv>cutoff));rank_v=min(uncapped,max(bound,0))
    assert row.restrictions==len(r) and row.covariance_rank==rank_v and row.structural_rank_bound==bound
    assert row.uncapped_covariance_rank==uncapped and row.rank_reassessment_version=="universal_score_rank_v1"
    close([row.hc0_covariance_scale,row.zero_covariance_threshold],[scale,scale*1e-12],"Wald HC0 rank scale")
    close(row.rank_cutoff,cutoff,"Wald numerical rank cutoff")
    condition=float(np.linalg.cond(matrix));flags=[]
    if len(r)>=.8*bound:flags.append("Restriction count approaches the universal score-rank bound")
    if condition>1e8:flags.append("Ill-conditioned restriction covariance")
    if row.time_residual_df<10:flags.append("Fewer than ten residual time degrees of freedom")
    assert bool(row.joint_test_fragile)==bool(flags)
    assert ("" if pd.isna(row.fragility_reason) else row.fragility_reason)=="; ".join(flags)
    available=rank_v==len(r) and sv[0]>scale*1e-12 and df>0
    if available:
        statistic=float(delta@np.linalg.solve(matrix,delta));assert statistic>=-1e-8
        close([row.wald_chi2,row.f_statistic,row.p_value],[max(statistic,0),max(statistic,0)/len(r),f_dist.sf(max(statistic,0)/len(r),len(r),df)],"Direct full-rank Wald")
        assert row.status=="available";COUNTS["available_Wald_tests"]+=1
    else:
        assert str(row.status).startswith("unavailable") and pd.isna(row.p_value) and pd.isna(row.f_statistic)
        COUNTS["unavailable_Wald_tests"]+=1


def rank_correction_lineage():
    """Check the retained native-stage bytes and every protected covariance input."""
    receipt_path=OUT/"rank_correction_manifest.json"
    receipt=json.loads(receipt_path.read_text())
    assert receipt["version"]=="universal_score_rank_v1" and receipt["branch"]=="annual_seasonal"
    assert receipt["thresholds"]=={"relative_singular_value":1e-10,"relative_to_HC0_zero":1e-12}
    checks={receipt["source"]:receipt["source_sha256"],
            receipt["native_stage_tests"]:receipt["native_stage_tests_sha256"],
            receipt["corrected_tests"]:receipt["corrected_tests_sha256"],
            **receipt["protected_native_inputs_sha256"],**receipt["design_archives_sha256"]}
    assert all(sha(ROOT/path)==value for path,value in checks.items())
    assert set(receipt["design_archives_sha256"])=={str(path.relative_to(ROOT)) for path in (OUT/"designs").glob("*.npz")}
    archive=ROOT/receipt["archive"]/"tests.csv"
    assert sha(archive)==receipt["native_stage_tests_sha256"]
    load=lambda path:pd.read_csv(path,dtype={"sector_id":str},float_precision="round_trip",low_memory=False)
    native,current=load(ROOT/receipt["native_stage_tests"]),load(ROOT/receipt["corrected_tests"])
    keys=["spec_id","covariance","hac_lags","sector_id","test"]
    assert not native.duplicated(keys).any() and not current.duplicated(keys).any()
    assert native[keys].equals(current[keys]) and len(current)==receipt["test_rows"]
    numeric_changes={name:int(native[name].fillna(-1).ne(current[name].fillna(-1)).sum())
                     for name in ["covariance_rank","structural_rank_bound","p_value","wald_chi2","f_statistic"]}
    status_changes=int(native.status.ne(current.status).sum())
    assert numeric_changes["covariance_rank"]==receipt["rank_fields_changed"]
    assert status_changes==receipt["statuses_changed"]==receipt["newly_available"]==receipt["newly_unavailable"]==0
    assert numeric_changes["p_value"]==numeric_changes["wald_chi2"]==numeric_changes["f_statistic"]==0
    fragile=int((current.status.eq("available")&current.joint_test_fragile).sum())
    assert fragile==receipt["fragile_available_tests"]
    stable=[name for name in native if name not in {"covariance_rank","structural_rank_bound"}]
    pd.testing.assert_frame_equal(native[stable],current[stable],check_exact=True)
    return {"status":"independently_verified","manifest_sha256":sha(receipt_path),
        "source_sha256":receipt["source_sha256"],"native_stage_tests_sha256":receipt["native_stage_tests_sha256"],
        "corrected_tests_sha256":receipt["corrected_tests_sha256"],"protected_native_files":len(receipt["protected_native_inputs_sha256"]),
        "protected_design_archives":len(receipt["design_archives_sha256"]),"changed_fields":numeric_changes,
        "status_changes":status_changes,"fragile_available_tests":fragile,
        "rank_bound":"Universal T−1 calendar-score / G−1 sector-score upper bound, combined with independently recomputed actual numerical rank",
        "checked_hashes":checks}


def verify_annual_seasonal_literature():
    OUT.mkdir(exist_ok=True);target=OUT/"verification.json";target.write_text('{"status":"running"}\n');COUNTS.clear();ERRORS.clear()
    manifest=json.loads((OUT/"manifest.json").read_text())
    before={**manifest["inputs_sha256"],**manifest["protected_sources_sha256"]}
    assert all(sha(ROOT/path)==value for path,value in before.items())
    correction=rank_correction_lineage()
    panels,matches,raw_counts=parents()
    policy=json.loads((ROOT/"notes/crisis_sensitivity_policy.json").read_text())
    events={year for event in policy["events"] for year in range(int(event["start"][:4]),int(event["end"][:4])+1)}
    inv=read(OUT/"sample_inventory.csv");specs=read(OUT/"specifications.csv");masks=read(OUT/"masks.csv")
    effects_csv=read(OUT/"temperature_effects.csv");components_csv=read(OUT/"interaction_components.csv");tests_csv=read(OUT/"tests.csv");coefficients=read(OUT/"coefficients.csv")
    keys=["context","history","model_spec","outcome","sample_variant"]
    expected={(c,h,m,y,v) for c in panels for h in HISTORIES for m in MODELS for y in YS for v in VARIANTS}
    assert len(inv)==len(expected) and set(inv[keys].itertuples(index=False,name=None))==expected
    mask_groups={key:part for key,part in masks.groupby(keys,sort=False)}
    spec_groups={key:part for key,part in specs.groupby(keys,sort=False)}
    assert set(mask_groups)==set(spec_groups)==expected
    effect_groups={key:part for key,part in effects_csv.groupby(["spec_id","covariance","hac_lags"],sort=False)}
    component_groups={key:part for key,part in components_csv.groupby(["spec_id","covariance","hac_lags"],sort=False)}
    test_groups={key:part for key,part in tests_csv.groupby(["spec_id","covariance","hac_lags"],sort=False)}
    coefficient_groups={key:part for key,part in coefficients.groupby("spec_id",sort=False)}
    native=[];seen_specs=set();details=[]
    with threadpool_limits(limits=1):
        for meta in inv.itertuples():
            key=tuple(getattr(meta,k) for k in keys);template=matches.get(meta.context,(None,None))[1]
            data,expected_masks,raw_n,target_n,maximum=sample(panels[meta.context],meta.model_spec,meta.history,meta.outcome,meta.sample_variant,events,panels[template] if template else None)
            saved=read(ROOT/meta.sample_csv);pd.testing.assert_frame_equal(saved,data,check_dtype=False,check_exact=False,atol=1e-9,rtol=1e-11)
            assert sha(ROOT/meta.sample_csv)==meta.sample_sha256 and meta.n_periods==data.year.nunique()
            assert meta.raw_model_history_eligible_n_periods==raw_n and meta.maximum_history_n_periods==maximum
            assert meta.matched_history_loss_periods==raw_n-maximum and meta.history_loss_periods==panels[meta.context].year.nunique()-maximum
            assert (pd.isna(meta.matched_template_eligible_n_periods) if target_n is None else meta.matched_template_eligible_n_periods==target_n)
            assert (pd.isna(meta.matched_history_template) if template is None else meta.matched_history_template==template)
            actual_masks=mask_groups[key][list(expected_masks[0])].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual_masks,pd.DataFrame(expected_masks),check_dtype=False)
            wanted=["common"]+(["H1"] if data.sector_id.nunique()>1 else [])
            records=spec_groups[key];assert set(records.model)==set(wanted) and len(records)==len(wanted)
            for record in records.itertuples():
                seen_specs.add(record.spec_id);kind=record.model
                x,names,selected,positions,sectors=design(data,meta.model_spec,kind);n,k=x.shape;t=data.year.nunique()
                r,threshold=rank(x);temporal=[rank(x[data.sector_id.eq(sid)])[0] for sid in sectors]
                available=n>k and r==k and t>max(temporal)
                assert (record.status=="estimated")==available
                for field,value in [("n_obs",n),("n_parameters",k),("n_sectors",len(sectors)),("design_rank",r),("residual_df",n-r),("time_design_rank",max(temporal)),("time_design_rank_min",min(temporal)),("time_residual_df",t-max(temporal))]:assert getattr(record,field)==value,(record.spec_id,field)
                assert not record.year_fixed_effects and not record.season_fixed_effects and record.linear_trend
                assert record.output_adjustment==("NSA" if meta.outcome=="y_nsa" else "SA") and record.frequency=="annual"
                assert record.sector_fixed_effects==(len(sectors)>1)
                assert record.AR_slope_scope=="Shared across sectors" if meta.model_spec.endswith("AR1") else pd.isna(record.AR_slope_scope) or record.AR_slope_scope=="None"
                assert json.loads(record.temperature_terms)==features(meta.model_spec)[0] and json.loads(record.controls)==features(meta.model_spec)[1]
                packed=np.load(ROOT/record.design_npz,allow_pickle=False)
                assert packed["names"].tolist()==names and packed["sector_id"].tolist()==data.sector_id.tolist()
                assert packed["selected_positions"].tolist()==selected.tolist() and packed["component_positions"].tolist()==positions.tolist()
                assert packed["temperature_features"].tolist()==features(meta.model_spec)[0]
                close(packed["X"],x,"Full explicit X",atol=1e-10,rtol=1e-12);close(packed["Y"],data[meta.outcome],"Raw/calendar Y",atol=1e-10,rtol=1e-12)
                close(packed["time"],data.year,"Actual calendar positions",atol=0,rtol=0)
                methods=([("Sector cluster",0,len(sectors)-1)] if kind=="common" and len(sectors)>1 else [])+[("Calendar HAC" if len(sectors)==1 else "Driscoll-Kraay",band,t-1) for band in [1,2]]
                assert packed["covariance_labels"].tolist()==[method for method,band,df in methods] and packed["bandwidths"].tolist()==[band for method,band,df in methods]
                assert packed["inference_df"].tolist()==[df for method,band,df in methods]
                if available:
                    b,u,inverse,rsq=qr(x,data[[meta.outcome]].to_numpy());b,u,rsq=b[:,0],u[:,0],rsq[0]
                    close(packed["beta"],b,"Native coefficients");close(packed["residual"],u,"Native residuals")
                    coeff=coefficient_groups[record.spec_id].set_index("term");assert set(coeff.index)==set(names)
                    close(coeff.loc[names,"coefficient"],b,"Exported full coefficients");COUNTS["full_coefficient_rows"]+=len(coeff)
                    native.append((record.native_label,meta.outcome,n,k))
                    COUNTS["native_fits"]+=1
                else:
                    assert "beta" not in packed.files and "residual" not in packed.files and not any(name.startswith("covariance_") and name!="covariance_labels" for name in packed.files)
                    assert record.spec_id not in coefficient_groups;COUNTS["unavailable_specs"]+=1
                for ci,(method,band,df) in enumerate(methods):
                    lookup=(record.spec_id,method,band);rows=effect_groups[lookup];checks=test_groups[lookup]
                    assert rows.inference_df.eq(df).all() and rows.n_clusters.eq(len(sectors) if method=="Sector cluster" else 0).all()
                    if available:
                        v,hc,actual_df,factor=covariance(inverse,u,data,method,band);assert actual_df==df
                        close(packed[f"covariance_{ci}"],v,"Full coefficient VCV");COUNTS["coefficient_covariance_entries"]+=v.size
                        close(packed["hc0_selected"],hc[np.ix_(selected,selected)],"HC0 selected reference")
                        close(rows.finite_sample_correction,np.repeat(factor,len(rows)),"Finite-sample multiplier")
                        close(rows.r_squared,np.repeat(rsq,len(rows)),"Full R squared")
                        v=v[np.ix_(selected,selected)];hc=hc[np.ix_(selected,selected)]
                    else:
                        assert rows[["coefficient","variance","std_error","p_value","ci_lower","ci_upper"]].isna().all().all()
                        assert checks.p_value.isna().all() and checks.f_statistic.isna().all()
                    found={(str(r.sector_id),r.effect,int(r.calendar_quarter),int(r.weather_lag_years),r.term):r for r in rows.itertuples()}
                    tests={(str(r.sector_id),r.test):r for r in checks.itertuples()}
                    assert len(found)==len(rows) and len(tests)==len(checks)
                    expected_effects=set();expected_tests=set();profiles=[]
                    for i,sid in enumerate(sectors if kind=="H1" else ["pooled"]):
                        basis=np.zeros((len(positions),len(selected)))
                        for f,part in enumerate(positions):
                            basis[f,list(selected).index(part[0])]=1
                            if kind=="H1" and i:basis[f,list(selected).index(part[i])]=1
                        profiles.append(basis)
                        unit=data if sid=="pooled" else data[data.sector_id.eq(sid)]
                        for effect,q,lag,term,vector in effects(basis,meta.model_spec):
                            ek=(sid,effect,q,lag,term);row=found[ek];expected_effects.add(ek)
                            close([row.mean_y,row.sd_y],[unit[meta.outcome].mean(),unit[meta.outcome].std()],"Exact-sector mean/SD")
                            if available:point(row,float(vector@b[selected]),float(vector@v@vector),float(vector@hc@vector),df)
                            COUNTS["effect_rows"]+=1
                        for null,restriction in nulls(basis,meta.model_spec):
                            expected_tests.add((sid,null));row=tests[sid,null]
                            if available:wald(row,b[selected],v,hc,restriction,df,len(sectors)-1 if method=="Sector cluster" else t-1)
                            else:
                                assert row.status=="unavailable: complete design/time support"
                                assert row.structural_rank_bound==(len(sectors)-1 if method=="Sector cluster" else t-1)
                                assert pd.isna(row.uncapped_covariance_rank) and row.joint_test_fragile
                                assert row.fragility_reason=="Point/time design unavailable"
                            COUNTS["test_rows"]+=1
                    if available and kind=="H1":
                        for null,restriction in [("All sector seasonal temperature response vectors equal",np.concatenate([basis-profiles[0] for basis in profiles[1:]],axis=0)),
                            ("All sector sums of direct seasonal temperature coefficients equal",np.vstack([(basis-profiles[0]).sum(axis=0) for basis in profiles[1:]]))]:
                            expected_tests.add(("all",null));wald(tests["all",null],b[selected],v,hc,restriction,df,t-1);COUNTS["test_rows"]+=1
                        comps=component_groups[lookup]
                        estimated=comps[comps.estimated.eq(True)].set_index("term")
                        norm=comps[comps.normalization.eq(True)];assert len(norm)==len(positions) and norm.coefficient.eq(0).all() and norm.std_error.isna().all() and norm.p_value.isna().all()
                        for part in positions:
                            for position in part:
                                j=list(selected).index(position);row=estimated.loc[names[position]]
                                point(row,float(b[position]),float(v[j,j]),float(hc[j,j]),df)
                        assert len(estimated)==positions.size;COUNTS["component_rows"]+=len(comps)
                    else:assert lookup not in component_groups
                    assert set(found)==expected_effects and set(tests)==expected_tests
                COUNTS["requested_specs"]+=1
                details.append({"spec_id":record.spec_id,"status":record.status,"N":n,"T":t,"K":k,"time_residual_df":t-max(temporal)})
            COUNTS["exact_samples"]+=1
    assert seen_specs==set(specs.spec_id) and len(seen_specs)==len(specs)
    assert {path.stem for path in (OUT/"designs").glob("*.npz")}==seen_specs
    assert {path.stem for path in (OUT/"samples").glob("*.csv")}=={Path(path).stem for path in inv.sample_csv}
    assert COUNTS["effect_rows"]==len(effects_csv) and COUNTS["test_rows"]==len(tests_csv) and COUNTS["component_rows"]==len(components_csv) and COUNTS["full_coefficient_rows"]==len(coefficients)
    assert COUNTS["native_fits"]==manifest["identified_specs"] and COUNTS["unavailable_specs"]==manifest["unavailable_specs"] and COUNTS["requested_specs"]==manifest["requested_specs"]
    part=effects_csv[effects_csv.model.eq("H1")]
    family=["context","history","model_spec","outcome","sample_variant","covariance","hac_lags","effect","calendar_quarter","weather_lag_years"]
    for _,group in part.groupby(family,sort=False):
        p=group.p_value.fillna(1).to_numpy();order=np.argsort(p);holm=np.empty(len(p));holm[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
        holm=np.where(group.p_value.notna(),holm,np.nan);close(group.p_value_holm_within_sector_family,holm,"Within-family Holm")
        COUNTS["Holm_families"]+=1
    assert effects_csv.loc[~effects_csv.model.eq("H1"),"p_value_holm_within_sector_family"].isna().all()
    log=read(OUT/"pyfixest/fit_log.csv")
    assert Counter(log[["label","outcome","n_obs","n_parameters"]].itertuples(index=False,name=None))==Counter(native)
    assert log.native_class.eq("Feols").all() and log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    assert log.fit_id.tolist()==list(range(1,len(log)+1)) and log.api_call_id.nunique()==len(native)
    assert log.estimator_version.eq(importlib.metadata.version("pyfixest")).all()
    for field in ["native_mapped_coefficient_max_abs_error","native_residual_max_abs_error","native_response_max_abs_error","native_design_max_abs_error"]:assert log[field].eq(0).all()
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all() and log.native_residuals_sha256.eq(log.residuals_sha256).all()
    ledger=json.loads((OUT/"pyfixest/estimator_manifest.json").read_text());assert ledger["native_fits_recorded"]==len(native)==manifest["native_fits"] and ledger["fit_log_sha256"]==sha(OUT/"pyfixest/fit_log.csv")
    assert all(sha(ROOT/path)==value for path,value in before.items())
    assert all(sha(ROOT/path)==value for path,value in correction["checked_hashes"].items())
    pd.DataFrame(details).to_csv(OUT/"verification_details.csv",index=False)
    result={"status":"passed","method":"Independent official-raw annual/quarterly reconstruction, full raw AR lookups, explicit FE design, QR, full unit/calendar covariance, HC0 degeneracy, joint contrasts and Holm",
        "counts":dict(COUNTS),"maximum_scaled_errors":ERRORS,"raw_preparation":raw_counts,"protected_and_source_hashes":before,
        "rank_correction":correction,
        "boundary_rule_verified":"Raw annual-growth AR1 is constructed before weather/cross-cut cropping; target eligibility is then explicitly intersected before event/strict t-2..t masks",
        "scope":"Numerical, source, calendar and declared-null verification; no causal, stationarity, small-sample coverage or exact published-paper replication certificate."}
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Verified annual seasonal sensitivity: {COUNTS['native_fits']} native fits, {COUNTS['unavailable_specs']} unavailable requests, {COUNTS['effect_rows']} effects and {COUNTS['test_rows']} tests.",flush=True)
    return result


if __name__=="__main__":verify_annual_seasonal_literature()
