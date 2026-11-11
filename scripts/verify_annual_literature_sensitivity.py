"""Independent raw-calendar, explicit-design QR and score-covariance audit.

Imports only existing independent raw-data checks, never the annual literature
producer, crisis producer, native backend or production covariance helpers.
"""
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
from verify_annual import raw_annual

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/literature_sensitivity/annual"
MODELS = {"DJO0":0,"DJO1":1,"DJO5":5,"DJO10":10,"BHM_common_trend":0,"BHM_unit_quadratic_trends":0}
HISTORIES = ["model_own_weather_history","matched_10_year_weather_history"]
VARIANTS = ["all_eligible","event_dates_excluded","strict_growth_window_clean"]
YS = ["y_nsa","y_sa"]
COUNTS = Counter()
MAXIMA = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path,dtype={"sector_id":str},low_memory=False)


def close(actual, expected, label, *, atol=1e-7, rtol=1e-7):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)), (label,"shape/missing mask")
    assert np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b)),(label,"infinite mask")
    keep=np.isfinite(a)&np.isfinite(b)
    if not np.allclose(a[keep],b[keep],atol=atol,rtol=rtol):
        errors=np.abs(a[keep]-b[keep]);allow=atol+rtol*np.abs(b[keep]);index=int(np.argmax(errors-allow))
        raise AssertionError((label,"actual",a[keep][index],"expected",b[keep][index],"allowed",allow[index]))
    COUNTS["numeric_values"]+=int(keep.sum())
    if keep.any():MAXIMA[label]=max(MAXIMA.get(label,0.),float(np.max(np.abs(a[keep]-b[keep])/np.maximum(1.,np.abs(b[keep])))))


def rank(x):
    scale=np.linalg.norm(x,axis=0);keep=scale>0
    return int(np.linalg.matrix_rank(x[:,keep]/scale[keep]))


def parents():
    panels,raw_counts=raw_annual()
    levels=read(ROOT/"data/processed/annual/annual_levels.csv")
    dwell=levels[levels.family.eq("va")&levels.sector_id.eq("1.3")]
    nsa=dwell[dwell.adjustment.eq("NSA")].set_index("year")
    sa=dwell[dwell.adjustment.eq("SA")].set_index("year")
    for name,parent in [("va12","va11"),("va20","va19")]:
        base=panels[parent];fields=["year","temp_c","rain_100mm","rh_pct","time","trend"]
        housing=base[fields].drop_duplicates("year").copy()
        housing["sector_id"]="1.3";housing["sector"]=nsa.sector.iloc[0]
        for adjustment,source in [("nsa",nsa),("sa",sa)]:
            housing["annual_"+adjustment]=housing.year.map(source.annual_level)
            housing["y_"+adjustment]=housing.year.map(source.annual_log_growth)
        columns=["sector_id","sector","annual_nsa","annual_sa","y_nsa","y_sa"]+fields
        panels[name]=pd.concat([base[columns],housing[columns]],ignore_index=True)
    matches={"aggregate_va_matched_va20":("aggregate_va","va20"),"va11_matched_va19":("va11","va19"),
        "va12_matched_va20":("va12","va20"),"iip_total_matched_iip6":("iip_total","iip6"),
        "iip_total_matched_iip18":("iip_total","iip18"),"iip6_matched_iip18":("iip6","iip18")}
    for name,(parent,template) in matches.items():
        panels[name]=panels[parent][panels[parent].year.isin(panels[template].year)].copy()
    national=panels["aggregate_va"].set_index("year")
    for context,frame in panels.items():
        frame=frame.copy();frame["y_log_level"]=100*np.log(frame.annual_sa);frame["y_raw_level"]=frame.annual_sa
        for term in ["temp_c","rain_100mm"]:
            for lag in range(11):frame[f"{term}_l{lag}"]=(frame.year-lag).map(national[term])
        panels[context]=frame.sort_values(["sector_id","year"]).reset_index(drop=True)
    return panels,raw_counts,matches


def sample(parent, model, history, variant, event_years):
    max_lag=10 if history==HISTORIES[1] else MODELS[model]
    required=[f"{term}_l{lag}" for term in ["temp_c","rain_100mm"] for lag in range(max_lag+1)]+YS+["rh_pct"]
    frame=parent[parent[required].notna().all(axis=1)].copy()
    if variant!=VARIANTS[0]:
        frame=frame[~frame.year.isin(event_years)]
    if variant==VARIANTS[2]:frame=frame[~(frame.year-1).isin(event_years)]
    frame=frame.reset_index(drop=True)
    if len(frame):
        unique=frame.drop_duplicates("year")
        frame["temp_center"]=frame.temp_c-unique.temp_c.mean();frame["temp_center_sq"]=frame.temp_center**2
        frame["rain_center"]=frame.rain_100mm-unique.rain_100mm.mean();frame["rain_center_sq"]=frame.rain_center**2
        frame["trend"]=frame.year-frame.year.mean();frame["trend_sq"]=frame.trend**2
    return frame


def exposures(model):
    if model.startswith("DJO"):
        lag=MODELS[model]
        return [f"temp_c_l{i}" for i in range(lag+1)],[f"rain_100mm_l{i}" for i in range(lag+1)]+["rh_pct"]
    return ["temp_center","temp_center_sq"],["rain_center","rain_center_sq","rh_pct"]


def design(data, model, kind):
    sectors=sorted(data.sector_id.unique());heat,controls=exposures(model)
    columns={"Intercept":np.ones(len(data))}
    for sector in sectors[1:]:columns[f"C(sector_id)[T.{sector}]"]=data.sector_id.eq(sector).to_numpy(float)
    for term in heat+controls+["trend"]+(["trend_sq"] if model=="BHM_unit_quadratic_trends" else []):
        columns[term]=data[term].to_numpy()
        if (kind=="H1" and term in heat) or (model=="BHM_unit_quadratic_trends" and term in {"trend","trend_sq"}):
            for sector in sectors[1:]:columns[f"{term}:C(sector_id)[T.{sector}]"]=data[term].to_numpy()*data.sector_id.eq(sector).to_numpy(float)
    return columns


def expected_rhs(model,kind):
    heat,controls=exposures(model);exposed=" + ".join(heat)
    if kind=="H1":exposed="("+exposed+") * C(sector_id)"
    trend="(trend + trend_sq) * C(sector_id)" if model=="BHM_unit_quadratic_trends" else "trend + C(sector_id)"
    return "1 + "+exposed+" + "+" + ".join(controls)+" + "+trend


def qr(x,y):
    scale=np.linalg.norm(x,axis=0);q,r=np.linalg.qr(x/scale,mode="reduced")
    inverse=np.linalg.solve(r,q.T)/scale[:,None]
    beta=inverse@y;residual=y-x@beta
    return beta,residual,inverse,1-(residual**2).sum(axis=0)/((y-y.mean(axis=0))**2).sum(axis=0)


def covariances(inverse,residual,data,kind):
    n,k=inverse.shape[1],inverse.shape[0];t=data.year.nunique()
    influence=inverse.T*residual[:,None]
    if kind=="common" and data.sector_id.nunique()>1:
        units=sorted(data.sector_id.unique());scores=np.vstack([influence[data.sector_id.eq(unit)].sum(axis=0) for unit in units])
        factor=len(units)/(len(units)-1)*(n-1)/(n-k)
        yield "Sector cluster",0,(scores.T@scores)*factor,len(units)-1,True
    years=data.year.to_numpy();scores=np.zeros((years.max()-years.min()+1,k));np.add.at(scores,years-years.min(),influence)
    for band in [1,2]:
        matrix=scores.T@scores
        for lag in range(1,band+1):
            cross=scores[lag:].T@scores[:-lag];matrix+=(1-lag/(band+1))*(cross+cross.T)
        matrix*=t/(t-1)*(n-1)/(n-k)
        yield "Driscoll-Kraay",band,(matrix+matrix.T)/2,t-1,band==1 and (kind!="common" or data.sector_id.nunique()==1)


def heat_vectors(names,heat,sector,reference):
    matrix=np.zeros((len(heat),len(names)))
    for i,term in enumerate(heat):
        matrix[i,names.index(term)]=1
        interaction=f"{term}:C(sector_id)[T.{sector}]"
        if sector!=reference and interaction in names:matrix[i,names.index(interaction)]=1
    return matrix


def point_rows(c,model,unique):
    if model.startswith("DJO"):
        return [(f"T_lag_{i}",row,np.nan) for i,row in enumerate(c)]+[("temperature_lag_sum",c.sum(axis=0),np.nan)]
    rows=[("temperature_linear_centered",c[0],np.nan),("temperature_quadratic",c[1],np.nan)]
    for percentile in [5,50,95]:
        exposure=float(np.percentile(unique.temp_c,percentile))
        rows.append((f"marginal_effect_p{percentile}",c[0]+2*(exposure-unique.temp_c.mean())*c[1],exposure))
    return rows


def restrictions(c,model):
    rows=[("All temperature coefficients zero",c)]
    if model.startswith("DJO"):
        rows.append(("Temperature coefficient sum zero",c.sum(axis=0,keepdims=True)))
        if len(c)>1:rows.append(("Static contemporaneous annual level channel",np.concatenate([(c[0]+c[1])[None,:],c[2:]],axis=0)))
    else:rows.append(("Temperature curvature zero",c[[1]]))
    return rows


def test(row,beta,v,r,degrees):
    delta=r@beta;matrix=r@v@r.T;singular=np.linalg.svd(matrix,compute_uv=False)
    rank_v=min(degrees,int(np.sum(singular>singular.max()*1e-10))) if singular.max()>0 else 0
    condition=float(np.linalg.cond(matrix));q=len(r)
    assert row.restrictions==q and row.covariance_rank==rank_v
    if rank_v<q or not np.isfinite(condition):
        assert row.status=="unavailable: singular restriction covariance" and pd.isna(row.p_value) and pd.isna(row.f_statistic)
        COUNTS["unavailable_Wald_tests"]+=1
    else:
        statistic=float(delta@np.linalg.solve(matrix,delta))
        if statistic < -1e-7:
            assert row.status=="unavailable: nonpositive covariance" and pd.isna(row.p_value)
        else:
            assert row.status=="available"
            close([row.f_statistic,row.p_value],[max(statistic,0)/q,f_dist.sf(max(statistic,0)/q,q,degrees)],"Wald F/p")
            COUNTS["available_Wald_tests"]+=1
    COUNTS["Wald_test_rows"]+=1


def verify_annual_literature_sensitivity():
    OUT.mkdir(exist_ok=True);target=OUT/"verification.json";target.write_text('{"status":"running"}\n')
    COUNTS.clear();MAXIMA.clear()
    manifest=json.loads((OUT/"manifest.json").read_text())
    assert manifest["models"]==MODELS and manifest["contexts"]==14
    initial={path:sha(ROOT/path) for path in {**manifest["source_hashes"],**manifest["protected_outputs"]}}
    references={path:sha(ROOT/path) for path in ["outputs/covid_comparison/original_inputs_reference.json",
        "outputs/pyfixest/migration_input_reference.json","data/reference_before_nsa_inputs/manifest.json"]}
    assert initial=={**manifest["source_hashes"],**manifest["protected_outputs"]}
    policy=json.loads((ROOT/"notes/crisis_sensitivity_policy.json").read_text())
    assert sha(ROOT/"notes/crisis_sensitivity_policy.json")==manifest["policy_sha256"]
    assert sha(ROOT/"notes/literature_specification_protocol.md")==manifest["protocol_sha256"]
    event_years={year for event in policy["events"] for year in range(int(event["start"][:4]),int(event["end"][:4])+1)}
    panels,raw_counts,matches=parents()
    inventory=read(OUT/"sample_inventory.csv");specs=read(OUT/"specifications.csv")
    estimates=read(OUT/"estimates.csv");tests=read(OUT/"tests.csv")
    expected={(context,history,model,variant) for context in panels for history in HISTORIES for model in MODELS for variant in VARIANTS}
    assert set(inventory[["context","history","model_spec","sample_variant"]].itertuples(index=False,name=None))==expected and len(inventory)==len(expected)
    estimate_groups={key:part for key,part in estimates.groupby(["fit_group_id","outcome","covariance","hac_lags"],sort=False)}
    test_groups={key:part for key,part in tests.groupby(["fit_group_id","outcome","covariance","hac_lags"],sort=False)}
    spec_groups={key:part for key,part in specs.groupby("fit_group_id",sort=False)}
    used_specs=set();native=[];details=[]
    with threadpool_limits(limits=1):
        for inv in inventory.itertuples():
            key=(inv.context,inv.history,inv.model_spec,inv.sample_variant)
            frame=sample(panels[inv.context],inv.model_spec,inv.history,inv.sample_variant,event_years)
            saved=read(ROOT/inv.sample_csv)
            assert sha(ROOT/inv.sample_csv)==inv.sample_sha256
            pd.testing.assert_frame_equal(saved,frame,check_dtype=False,check_exact=False,atol=1e-9,rtol=1e-11)
            assert inv.n_periods==frame.year.nunique() and inv.n_sectors==frame.sector_id.nunique() and inv.parent_periods==panels[inv.context].year.nunique()
            sample_id=hashlib.sha256(json.dumps(list(key)).encode()).hexdigest()[:16]
            assert Path(inv.sample_csv).stem==sample_id
            sectors=sorted(frame.sector_id.unique());kinds=["common"]+(["H1"] if len(sectors)>1 else [])
            if inv.history==HISTORIES[0] and inv.model_spec in {"DJO5","BHM_unit_quadratic_trends"} and len(sectors)>1:kinds.append("separate")
            for kind in kinds:
                units=[(sid,group.reset_index(drop=True)) for sid,group in frame.groupby("sector_id",sort=True)] if kind=="separate" else [("pooled",frame)]
                for sid,data in units:
                    group_id=hashlib.sha256(json.dumps([sample_id,kind,sid]).encode()).hexdigest()[:20]
                    assert group_id in spec_groups and group_id not in used_specs;used_specs.add(group_id)
                    records=spec_groups[group_id]
                    columns=design(data,inv.model_spec,kind);names=list(columns);x=np.column_stack(list(columns.values()))
                    heat,controls=exposures(inv.model_spec);unique=data.drop_duplicates("year")
                    temporal=np.column_stack([np.ones(len(unique)),unique[heat+controls+["trend"]+(["trend_sq"] if inv.model_spec=="BHM_unit_quadratic_trends" else [])]])
                    r,tr=rank(x),rank(temporal);n,k=x.shape;t=len(unique)
                    assert records.formula_rhs.eq(expected_rhs(inv.model_spec,kind)).all()
                    includes_dwellings=inv.context.startswith("aggregate_va") or inv.context.startswith("va12") or inv.context=="va20"
                    assert records.includes_dwellings.eq(includes_dwellings).all()
                    expected_match=matches.get(inv.context,(None,None))[1]
                    assert (records.matched_to.isna().all() if expected_match is None else records.matched_to.eq(expected_match).all())
                    assert records.frequency.eq("annual").all() and records.linear_trend.eq(True).all()
                    assert records.quadratic_unit_trends.eq(inv.model_spec=="BHM_unit_quadratic_trends").all()
                    assert all(json.loads(value)==heat for value in records.temperature_terms)
                    assert all(json.loads(value)==controls for value in records.controls)
                    close(records.mean_temperature,np.repeat(unique.temp_c.mean(),len(records)),"Mean temperature metadata")
                    close(records.temperature_min,np.repeat(unique.temp_c.min(),len(records)),"Temperature lower support")
                    close(records.temperature_max,np.repeat(unique.temp_c.max(),len(records)),"Temperature upper support")
                    for field,value in [("n_obs",n),("n_sectors",data.sector_id.nunique()),("n_parameters",k),("design_rank",r),("time_design_rank",tr),("time_residual_df",t-tr),("residual_df",n-r)]:
                        assert records[field].eq(value).all(),(group_id,field)
                    unavailable=r<k or n<=k or t<=tr
                    if unavailable:
                        assert len(records)==1 and records.status.eq("unavailable").all() and records.outcome.isna().all()
                        assert group_id not in set(estimates.fit_group_id) and group_id not in set(tests.fit_group_id)
                        COUNTS["unavailable_design_requests"]+=1;continue
                    assert len(records)==2 and set(records.outcome)==set(YS) and records.status.eq("estimated").all()
                    packed=np.load(ROOT/records.design_npz.iloc[0],allow_pickle=False)
                    saved_names=packed["names"].tolist();assert set(saved_names)==set(names) and len(saved_names)==len(names)
                    x=np.column_stack([columns[name] for name in saved_names]);names=saved_names
                    close(packed["X"],x,"Full explicit X",atol=1e-10,rtol=1e-12)
                    close(packed["Y"],data[YS].to_numpy(),"Raw-calendar Y",atol=1e-10,rtol=1e-12)
                    assert packed["sector_id"].tolist()==data.sector_id.tolist() and packed["outcomes"].tolist()==YS
                    close(packed["time"],data.year.to_numpy(),"Uncompressed annual calendar",atol=0,rtol=0)
                    beta,residual,inverse,r_squared=qr(x,data[YS].to_numpy())
                    for oi,outcome in enumerate(YS):
                        record=records[records.outcome.eq(outcome)].iloc[0]
                        assert record.spec_id==group_id+"_"+outcome
                        assert record.output_adjustment==("NSA" if outcome=="y_nsa" else "SA")
                        close(packed[f"beta_{oi}"],beta[:,oi],"Native coefficients")
                        close(packed[f"residual_{oi}"],residual[:,oi],"Native residuals")
                        close(record.r_squared,r_squared[oi],"Full-model R squared")
                        close([record.mean_y,record.sd_y],[data[outcome].mean(),data[outcome].std()],"Exact fit mean/SD")
                        native.append((f"Annual literature {inv.context}/{inv.history}/{inv.model_spec}/{inv.sample_variant}/{kind}/{sid}",outcome,n,k,expected_rhs(inv.model_spec,kind)))
                        for method,band,v,degrees,primary in covariances(inverse,residual[:,oi],data,kind):
                            code="SC" if method=="Sector cluster" else f"DK{band}"
                            close(packed[f"covariance_{oi}_{code}"],v,"Full coefficient covariance")
                            COUNTS["coefficient_covariance_entries"]+=v.size
                            lookup=(group_id,outcome,method,band)
                            rows=estimate_groups[lookup];checks=test_groups[lookup]
                            assert rows.inference_df.eq(degrees).all() and rows.main_inference.eq(primary).all()
                            assert checks.inference_df.eq(degrees).all() and checks.main_inference.eq(primary).all()
                            assert rows.n_clusters.eq(data.sector_id.nunique() if method=="Sector cluster" else 0).all()
                            results={(str(row.sector_id),row.term):row for row in rows.itertuples()}
                            test_results={(str(row.sector_id),row.null):row for row in checks.itertuples()}
                            assert len(results)==len(rows) and len(test_results)==len(checks)
                            actual_sectors=sorted(data.sector_id.unique()) if kind in {"H1","separate"} else ["pooled"]
                            transforms=[];expected_point=set();expected_tests=set()
                            for sector in actual_sectors:
                                c=heat_vectors(names,heat,sector,sorted(data.sector_id.unique())[0]);transforms.append(c)
                                unit=data if sector=="pooled" else data[data.sector_id.eq(sector)]
                                for term,vector,exposure in point_rows(c,inv.model_spec,unique):
                                    row=results[sector,term];value=float(vector@beta[:,oi]);variance=float(vector@v@vector)
                                    se=math.sqrt(max(variance,0));critical=student_t.ppf(.975,degrees)
                                    p=2*student_t.sf(abs(value/se),degrees) if se else np.nan
                                    close([row.coefficient,row.std_error,row.ci_lower,row.ci_upper,row.p_value,row.evaluation_temperature],
                                          [value,se,value-critical*se,value+critical*se,p,exposure],"Linear/lag-sum/marginal effect inference")
                                    close([row.mean_y,row.sd_y],[unit[outcome].mean(),unit[outcome].std()],"Reported sector mean/SD")
                                    if variance>0:close(p,f_dist.sf(value**2/variance,1,degrees),"Total-effect one-constraint t/F")
                                    expected_point.add((sector,term));COUNTS["estimate_rows"]+=1
                                for null,restriction in restrictions(c,inv.model_spec):
                                    test(test_results[sector,null],beta[:,oi],v,restriction,degrees)
                                    expected_tests.add((sector,null))
                            if kind=="H1":
                                restriction=np.concatenate([c-transforms[0] for c in transforms[1:]],axis=0)
                                null="All sector temperature response functions equal"
                                test(test_results["all",null],beta[:,oi],v,restriction,degrees);expected_tests.add(("all",null))
                            assert set(results)==expected_point and set(test_results)==expected_tests
                    COUNTS["estimated_fit_groups"]+=1
                    details.append({"fit_group_id":group_id,"context":inv.context,"model_spec":inv.model_spec,"kind":kind,"history":inv.history,"variant":inv.sample_variant,"N":n,"T":t,"K":k,"time_residual_df":t-tr})
            COUNTS["exact_sample_windows"]+=1
    assert used_specs==set(spec_groups) and COUNTS["estimate_rows"]==len(estimates) and COUNTS["Wald_test_rows"]==len(tests)
    assert {path.stem for path in (OUT/"samples").glob("*.csv")}=={Path(path).stem for path in inventory.sample_csv}
    assert {path.stem for path in (OUT/"designs").glob("*.npz")}==set(specs.fit_group_id[specs.status.eq("estimated")])
    log=read(OUT/"pyfixest/fit_log.csv")
    keys=["label","outcome","n_obs","n_parameters","source_formula"]
    assert Counter(log[keys].itertuples(index=False,name=None))==Counter(native)
    assert log.native_class.eq("Feols").all() and log.estimator_version.eq(importlib.metadata.version("pyfixest")).all()
    assert log.fit_id.tolist()==list(range(1,len(log)+1)) and log.api_call_id.nunique()==COUNTS["estimated_fit_groups"]
    assert log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    for field in ["native_mapped_coefficient_max_abs_error","native_residual_max_abs_error","native_response_max_abs_error","native_design_max_abs_error"]:assert log[field].eq(0).all()
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all() and log.native_residuals_sha256.eq(log.residuals_sha256).all()
    ledger=json.loads((OUT/"pyfixest/estimator_manifest.json").read_text())
    assert ledger["fit_log_sha256"]==sha(OUT/"pyfixest/fit_log.csv") and ledger["native_fits_recorded"]==len(native)==manifest["native_fits"]
    assert initial=={path:sha(ROOT/path) for path in initial}
    assert references=={path:sha(ROOT/path) for path in references}
    pd.DataFrame(details).to_csv(OUT/"verification_details.csv",index=False)
    report={"status":"passed","method":"Independent official-raw annual reconstruction, manual full FE/interaction design, QR OLS, explicit unit/calendar influence covariance and direct full-rank Wald",
        "native_fits_verified":len(native),"native_API_calls":log.api_call_id.nunique(),"counts":dict(COUNTS),"raw_preparation":raw_counts,
        "maximum_scaled_errors":MAXIMA,"source_and_protected_hashes":initial,
        "old_reference_hashes_unchanged":references,
        "scope":"All saved annual literature sensitivity models and unavailable requests; numerical/source/calendar verification, not causal identification, stationarity, valid small-sample size or a replication of the papers' multicountry estimands."}
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Verified annual literature sensitivity: {len(native)} native fits, {COUNTS['unavailable_design_requests']} unavailable designs, {COUNTS['estimate_rows']} estimates and {COUNTS['Wald_test_rows']} Wald tests.",flush=True)
    return report


if __name__=="__main__":verify_annual_literature_sensitivity()
