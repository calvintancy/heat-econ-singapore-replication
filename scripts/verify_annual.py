"""Independent raw-calendar annual aggregation, QR, covariance and test audit."""
from __future__ import annotations
import calendar
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.stats import t as student_t, f as f_distribution
from threadpoolctl import threadpool_limits
from verify_outcomes import raw_output, raw_weather
from verify_granular import _raw_extended
from verify_stationarity_audit import manual_adf, manual_kpss

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA = ROOT / "outputs/annual", ROOT / "data/processed/annual"
BASE, ADDED = ["temp_c", "rain_100mm", "rh_pct"], ["wind_mps", "pm25_10ug"]
IDS = {"sector_id": str, "sector_i": str, "sector_j": str, "first_sector_id": str, "second_sector_id": str}
ERRORS = {"comparisons": 0, "values": 0, "max_scaled_error": 0.}


def read(path):
    return pd.read_csv(path, dtype=IDS)


def close(a, b, label, exact=False):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape == b.shape and np.array_equal(np.isnan(a),np.isnan(b)), (label,"shape/unavailability")
    difference=np.abs(a-b);tolerance=np.zeros_like(b)if exact else 1e-7+1e-7*np.abs(b)
    if not np.allclose(a,b,atol=0 if exact else 1e-7,rtol=0 if exact else 1e-7,equal_nan=True):
        index=np.unravel_index(np.nanargmax(difference-tolerance),a.shape)
        raise AssertionError((label,"max_absolute",float(np.nanmax(difference)),"worst_index",index,"actual",float(a[index]),"expected",float(b[index]),"allowed",float(tolerance[index])))
    keep=np.isfinite(a)&np.isfinite(b)
    ERRORS["comparisons"] += 1;ERRORS["values"] += int(keep.sum())
    if keep.any():ERRORS["max_scaled_error"]=max(ERRORS["max_scaled_error"],float(np.max(np.abs(a[keep]-b[keep])/np.maximum(1,np.abs(b[keep])))))


def raw_annual():
    levels={}
    manifest=json.loads((ROOT/"data/raw/manifest.json").read_text())
    for family,context,h in [("va","va",4),("iip","iip-monthly",12)]:
        for adjustment in ["NSA","SA"]:
            stem=family+"_"+adjustment.lower()
            for suffix in["metadata","tabledata"]:
                source=ROOT/f"data/raw/{stem}_{suffix}.json"
                assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest[source.name]["sha256"]
            grouped={}
            for (sid,p),value in raw_output(context,adjustment.lower()).items():grouped.setdefault((sid,p.year),[]).append(value)
            for (sid,year),values in grouped.items():
                if len(values)==h:levels[family,sid,adjustment,year]=math.fsum(values)/(1 if family=="va" else 12)
    saved=read(DATA/"annual_levels.csv")
    assert len(saved)==len(levels) and not saved.duplicated(["family","sector_id","adjustment","year"]).any()
    for r in saved.itertuples():
        value=levels[r.family,r.sector_id,r.adjustment,r.year];prior=levels.get((r.family,r.sector_id,r.adjustment,r.year-1))
        close(r.annual_level,value,"Raw complete-year output")
        close(r.annual_log_growth,100*math.log(value/prior) if prior else np.nan,"Raw prior-calendar-year log growth")
        assert r.source_periods==(4 if r.family=="va" else 12)
        assert r.source_table_id=={"va":{"NSA":"M015661","SA":"M015662"},"iip":{"NSA":"M355381","SA":"M355382"}}[r.family][r.adjustment]
    monthly=raw_weather("M");weather={}
    annual_weather=read(DATA/"annual_weather.csv")
    for r in annual_weather.itertuples():
        months=list(pd.period_range(f"{r.year}-01",f"{r.year}-12",freq="M"));complete=all(m in monthly for m in months)
        assert r.weather_complete==complete and r.weather_months==sum(m in monthly for m in months)
        assert r.days==(366 if calendar.isleap(r.year) else 365)
        values={term:(math.fsum(monthly[m][term]*(1 if term=="rain_100mm"else m.days_in_month)for m in months)/(1 if term=="rain_100mm"else r.days))if complete else np.nan for term in BASE}
        for term,value in values.items():close(getattr(r,term),value,"Annual baseline-weather raw aggregation")
        if complete:weather[r.year]=values
    raw_extended,_=_raw_extended();monthly_ext=raw_extended["M"];controls={}
    for r in read(DATA/"annual_controls.csv").itertuples():
        months=[m for m in monthly_ext if m.year==r.year];days=366 if calendar.isleap(r.year) else 365
        counts={term:{m:int(round(monthly_ext[m][coverage]*m.days_in_month*bins))for m in months}for term,coverage,bins in [("wind_mps","wind_coverage",48),("pm25_10ug","pm_coverage",24)]}
        wc,pc=sum(counts["wind_mps"].values())/(days*48),sum(counts["pm25_10ug"].values())/(days*24)
        eligible=len(months)==12 and all(counts[t][m]>0 for t in ADDED for m in months)and wc>=.9 and pc>=.9
        assert r.eligible_90==eligible and r.control_months==len(months)
        close([r.wind_bins,r.pm_hours,r.wind_coverage,r.pm_coverage],[sum(counts["wind_mps"].values()),sum(counts["pm25_10ug"].values()),wc,pc],"Annual retained counts/coverage")
        values={t:math.fsum(monthly_ext[m][t]*n for m,n in counts[t].items()if n)/sum(counts[t].values())if sum(counts[t].values())else np.nan for t in ADDED}
        for term,value in values.items():close(getattr(r,term),value,"Raw observed-count-weighted annual control")
        controls[r.year]={**values,"eligible_90":eligible}
    va=[f"1.1.{i}"for i in range(1,5)]+[f"1.2.{i}"for i in range(1,8)];iip=[f"1.{i}"for i in range(1,7)]
    def leaves(family,parents):
        all_ids={key[1]for key in levels if key[0]==family}
        return sorted(s for s in all_ids if any(s==p or s.startswith(p+".")for p in parents)and not any(o.startswith(s+".")for o in all_ids))
    definitions={"aggregate_va":("va",["1.4"]),"va11":("va",va),"va19":("va",leaves("va",va)),"iip_total":("iip",["1"]),"iip6":("iip",iip),"iip18":("iip",leaves("iip",iip))}
    assert len(definitions["va19"][1])==19 and len(definitions["iip18"][1])==18
    panels={}
    for context,(family,ids)in definitions.items():
        panel=read(DATA/f"panel_{context}.csv")
        years={y for y in weather if all((family,s,a,y)in levels and(family,s,a,y-1)in levels for s in ids for a in["NSA","SA"])}
        assert set(panel.year)==years and set(panel.sector_id)==set(ids)and len(panel)==len(ids)*len(years)
        for r in panel.itertuples():
            for adjustment,outcome in[("NSA","y_nsa"),("SA","y_sa")]:
                value=levels[family,r.sector_id,adjustment,r.year];prior=levels[family,r.sector_id,adjustment,r.year-1]
                close(getattr(r,outcome),100*math.log(value/prior),"Raw annual panel growth")
                close(getattr(r,"annual_"+adjustment.lower()),value,"Raw annual panel level")
            for term in BASE:close(getattr(r,term),weather[r.year][term],"Raw annual panel weather")
            for term in ADDED:close(getattr(r,term),controls.get(r.year,{}).get(term,np.nan),"Raw annual panel wind/PM")
            assert r.eligible_90==controls.get(r.year,{}).get("eligible_90",False)and r.trend==r.year-2000 and r.time==r.year-1982
        panels[context]=panel
    assert {y for y,c in controls.items()if c["eligible_90"]}==set(range(2017,2026))
    return panels,{"raw_annual_levels_checked":len(saved),"annual_weather_years":len(weather),"annual_control_years":len(controls),"contexts":{c:{"periods":p.year.nunique(),"sectors":p.sector_id.nunique()}for c,p in panels.items()}}


def rank(x):
    scale=np.linalg.norm(x,axis=0);return int(np.linalg.matrix_rank(x[:,scale>0]/scale[scale>0]))


def qr(x,y):
    scale=np.linalg.norm(x,axis=0);z=x/scale
    with threadpool_limits(limits=1):
        q,r=np.linalg.qr(z,mode="reduced");inverse=np.linalg.solve(r,q.T)/scale[:,None]
    beta=inverse@y;residual=y-x@beta
    return beta,residual,inverse,1-residual@residual/((y-y.mean())@(y-y.mean()))


def covariance(inverse,residual,sample,kind,band):
    n,k=inverse.shape[1],inverse.shape[0];individual=inverse.T*residual[:,None]
    years=sorted(sample.year.unique());t=len(years)
    if kind=="Sector cluster":
        units=sorted(sample.sector_id.unique());score=np.vstack([individual[sample.sector_id.eq(s)].sum(axis=0)for s in units]);factor=len(units)/(len(units)-1)*(n-1)/(n-k);df=len(units)-1
    else:
        score=np.zeros((max(years)-min(years)+1,k))
        for year in years:score[year-min(years)]=individual[sample.year.eq(year)].sum(axis=0)
        factor=t/(t-1)*(n-1)/(n-k);df=t-1
    meat=score.T@score
    if kind!="Sector cluster":
        for lag in range(1,band+1):
            cross=score[lag:].T@score[:-lag];meat+=(1-lag/(band+1))*(cross+cross.T)
    return (meat+meat.T)/2*factor,df


def point(row,beta,var,df):
    assert var>0
    se=math.sqrt(var);critical=student_t.ppf(.975,df);p=2*student_t.sf(abs(beta/se),df)
    close([row.coefficient,row.std_error,row.ci_lower,row.ci_upper,row.p_value],[beta,se,beta-critical*se,beta+critical*se,p],"Annual independent point/SE/CI/p")
    close(row.inference_df,df,"Annual inferential DF",exact=True)
    close(p,f_distribution.sf(beta*beta/var,1,df),"Annual one-restriction total effect t/F")


def verify_annual():
    OUT.mkdir(exist_ok=True);(OUT/"verification.json").write_text('{"status":"running"}\n')
    panels,preparation=raw_annual()
    publication_reconciliation=verify_publications()
    tables={n:read(OUT/f"{n}.csv")for n in["specs","estimates","sector_estimates","separate_estimates","coefficients","coefficient_covariances","sector_covariances","interaction_components","joint_tests","pairwise_tests","year_fe_collinearity"]}
    specs=tables["specs"];assert specs.spec_id.is_unique
    expected=set()
    for context,panel in panels.items():
        units=sorted(panel.sector_id.unique())
        for controls in["weather","full"]:
            for variant in["all_eligible","exclude_2020_2022"]:
                for adjustment in["NSA","SA"]:
                    for model,sid in[("common","pooled")]+([("H1","pooled")]+[("separate",s)for s in units]if len(units)>1 else[]):expected.add((context,controls,variant,adjustment,model,sid))
    assert set(specs[["context","control_set","sample_variant","adjustment","model","sector_id"]].itertuples(index=False,name=None))==expected
    counters={"requested_specs":len(specs),"estimated_specs":0,"unavailable_specs":0,"coefficient_rows":0,"covariance_entries":0,"sector_covariance_entries":0,"point_rows":0,"joint_available":0,"joint_singular_no_p":0,"joint_model_unavailable":0,"pairwise_rows":0}
    native=[]
    for m in specs.itertuples():
        sample=read(ROOT/m.sample_csv);base=panels[m.context]
        if m.control_set=="full":base=base[base.eligible_90]
        if m.sample_variant=="exclude_2020_2022":base=base[~base.year.between(2020,2022)]
        if m.model=="separate":base=base[base.sector_id.eq(m.sector_id)]
        keys=["sector_id","year"];pd.testing.assert_frame_equal(sample.sort_values(keys).reset_index(drop=True),base.sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=False,atol=1e-11,rtol=1e-12)
        weather=BASE+(ADDED if m.control_set=="full"else[])
        outcome="y_nsa"if m.adjustment=="NSA"else"y_sa"
        expected_formula=f"{outcome} ~ "+("temp_c * C(sector_id)"if m.model=="H1"else"temp_c")+" + "+" + ".join(weather[1:]+["trend","C(sector_id)"])
        assert m.formula==expected_formula and not m.year_fixed_effects and m.linear_time_trend and m.analysis_role==("primary_NSA"if m.adjustment=="NSA"else"SA_comparison")
        yy,xx=dmatrices(m.formula,sample,return_type="dataframe",NA_action="raise");x,y=xx.to_numpy(),yy.iloc[:,0].to_numpy();names=list(xx.columns)
        unique=sample.drop_duplicates("year");tx=np.column_stack([np.ones(len(unique)),unique[weather+["trend"]]])
        r,tr=rank(x),rank(tx);t=len(unique);n,k=x.shape
        for value,want in[(m.design_rank,r),(m.time_design_rank,tr),(m.n_time_parameters,tx.shape[1]),(m.time_residual_df,t-tr),(m.nominal_time_residual_df,t-tx.shape[1]),(m.n_obs,n),(m.n_parameters,k),(m.n_periods,t),(m.residual_df,n-r),(m.calendar_gaps,int(sample.year.max()-sample.year.min()+1-t))]:close(value,want,"Annual exact sample/design metadata",exact=True)
        assert m.n_sectors==sample.sector_id.nunique()and m.start==sample.year.min()and m.end==sample.year.max()and m.calendar_span_years==sample.year.max()-sample.year.min()+1
        close([m.mean_y,m.sd_y],[y.mean(),y.std(ddof=1)],"Annual exact-sample mean/SD")
        assert bool(m.fragile_short_sample)==(t-tr<=3)
        available=r==k and n>k and t>tr
        assert (m.status=="estimated")==available
        coefs=tables["coefficients"];coefs=coefs[coefs.spec_id.eq(m.spec_id)].set_index("term");assert set(coefs.index)==set(names)
        target="sector_estimates"if m.model=="H1"else"separate_estimates"if m.model=="separate"else"estimates"
        rows=tables[target];rows=rows[rows.spec_id.eq(m.spec_id)]
        if not available:
            assert rows[["coefficient","std_error","ci_lower","ci_upper","p_value"]].isna().all().all()and coefs.coefficient.isna().all()
            assert not tables["coefficient_covariances"].spec_id.eq(m.spec_id).any()
            if m.model=="H1":
                joint=tables["joint_tests"];joint=joint[joint.spec_id.eq(m.spec_id)];assert len(joint)==1 and joint.test_status.iloc[0]=="unavailable: model underidentified"and joint.p_value.isna().all();counters["joint_model_unavailable"]+=1
            assert m.control_set=="full"and m.sample_variant=="exclude_2020_2022"and t==6
            counters["unavailable_specs"]+=1;continue
        beta,residual,inverse,r2=qr(x,y);close(coefs.loc[names,"coefficient"],beta,"Annual full QR coefficients");counters["coefficient_rows"]+=len(names)
        counters["estimated_specs"]+=1;native.append((f"Annual {m.context}/{m.control_set}/{m.sample_variant}/{m.adjustment}/{m.model}/{m.sector_id if m.model=='separate'else'joint'}",m.outcome,n,k,m.formula))
        assert set(rows.hac_lags.dropna())==({0,1,2}if m.model=="common"and m.n_sectors>1 else{1,2})
        units=sorted(sample.sector_id.unique());transform=np.zeros((len(units),k))
        if m.model=="H1":
            transform[:,names.index("temp_c")]=1
            for i,s in enumerate(units[1:],1):transform[i,names.index(f"temp_c:C(sector_id)[T.{s}]")]=1
        for(kind,band),group in rows.groupby(["covariance","hac_lags"]):
            matrix,df=covariance(inverse,residual,sample,kind,int(band));main=kind=="Sector cluster"if m.model=="common"and m.n_sectors>1 else int(band)==1
            assert group.main_inference.eq(main).all();close(group.r_squared,np.repeat(r2,len(group)),"Annual R-squared")
            saved=tables["coefficient_covariances"];saved=saved[saved.spec_id.eq(m.spec_id)&saved.covariance.eq(kind)&saved.hac_lags.eq(band)]
            assert len(saved)==k*k;close(saved.pivot(index="term_i",columns="term_j",values="covariance_value").loc[names,names],matrix,f"Annual all coefficient covariance entries/{m.spec_id}/{m.context}/{m.control_set}/{m.model}/{kind}/{band}");counters["covariance_entries"]+=len(saved)
            if m.model!="H1":
                assert set(group.term)==set(weather+["trend"])
                for row in group.itertuples():j=names.index(row.term);point(row,beta[j],matrix[j,j],df);counters["point_rows"]+=1
            else:verify_interactions(m,sample,group,beta,matrix,transform,names,df,int(band),tables,counters)
    diagnostics=verify_diagnostics(panels)
    assert counters["covariance_entries"]==len(tables["coefficient_covariances"])and counters["sector_covariance_entries"]==len(tables["sector_covariances"])and counters["pairwise_rows"]==len(tables["pairwise_tests"])
    log=read(OUT/"pyfixest/fit_log.csv");cols=["label","outcome","n_obs","n_parameters","source_formula"]
    assert Counter(log[cols].itertuples(index=False,name=None))==Counter(native)
    assert len(log)==counters["estimated_specs"]and log.native_class.eq("Feols").all()and log.native_dropped_rows.eq(0).all()and log.native_dropped_coefficients.eq(0).all()
    for field in["native_mapped_coefficient_max_abs_error","native_residual_max_abs_error","native_response_max_abs_error","native_design_max_abs_error"]:assert log[field].eq(0).all()
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all()and log.native_residuals_sha256.eq(log.residuals_sha256).all()
    manifest=json.loads((OUT/"pyfixest/estimator_manifest.json").read_text());assert manifest["fit_log_sha256"]==hashlib.sha256((OUT/"pyfixest/fit_log.csv").read_bytes()).hexdigest()and manifest["native_fits_recorded"]==len(log)
    proof=tables["year_fe_collinearity"];assert len(proof)==len(panels)*4 and not proof.duplicated(["context","control_set","sample_variant"]).any()
    for r in proof.itertuples():assert r.n_years==r.year_FE_rank==r.year_FE_plus_temperature_rank and r.temperature_residual_max==0
    report={"status":"passed","preparation":preparation,"official_annual_publication_reconciliation":publication_reconciliation,"fits":counters,"native_models_verified":len(log),"native_API_calls":log.api_call_id.nunique(),"diagnostics":diagnostics,"numerical_checks":ERRORS,"scope":"Independent raw annual SA/NSA calendar construction, full QR, explicit CR1/calendar covariance, total-effect contrasts, direct full-rank Wald, Holm, and auxiliary unit-root calculations; not causal/stationarity certification"}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n");print(f"Annual independent verification passed: {len(specs)} requested specs, {len(log)} native models, {counters['unavailable_specs']} explicitly unavailable.",flush=True)
    return report


def verify_publications():
    folder=ROOT/"data/raw/annual";manifest=json.loads((folder/"manifest.json").read_text())
    for name,info in manifest.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==info["sha256"]
    published={}
    for family in["va","iip"]:
        data=json.loads((folder/f"{family}_annual_tabledata.json").read_text())["Data"]
        for row in data["row"]:
            for cell in row["columns"]:published[family,row["seriesNo"],int(cell["key"])]=float(str(cell["value"]).replace(",",""))
    levels=read(DATA/"annual_levels.csv");levels=levels[levels.adjustment.eq("NSA")]
    expected={(r.family,r.sector_id,r.year):r.annual_level for r in levels.itertuples()}
    table=read(OUT/"reconciliation.csv")
    assert not table.duplicated(["family","sector_id","year"]).any()and set(table[["family","sector_id","year"]].itertuples(index=False,name=None))==set(expected)
    for r in table.itertuples():
        key=r.family,r.sector_id,r.year;bound=.25 if r.family=="va"else .001
        close([r.derived_annual_level,r.published_annual_level,r.gap,r.rounding_bound],[expected[key],published[key],expected[key]-published[key],bound],"Independent official annual publication reconciliation")
        assert r.within_rounding==(abs(expected[key]-published[key])<=bound+1e-9)
    return {"official_annual_NSA_cells":len(table),"sources":{family:int(table.family.eq(family).sum())for family in["va","iip"]},"all_within_published_precision_envelope":bool(table.within_rounding.all()),"source_hashes_checked":len(manifest)}


def verify_interactions(m,sample,group,beta,matrix,transform,names,df,band,tables,counters):
    units=sorted(sample.sector_id.unique());totals=transform@beta;cov=transform@matrix@transform.T
    actual=group.set_index("sector_id").loc[units]
    for i,s in enumerate(units):
        row=actual.loc[s];point(row,totals[i],cov[i,i],df);counters["point_rows"]+=1
        close(row.coefficient,row.temperature_main+row.temperature_interaction,"Annual beta main plus sector deviation")
        close([row.mean_y,row.sd_y],[sample.loc[sample.sector_id.eq(s),m.outcome].mean(),sample.loc[sample.sector_id.eq(s),m.outcome].std()],"Annual sector-specific signed mean/SD")
    close(actual.mean_sector_slope,np.repeat(totals.mean(),len(units)),"Mean sector slopes")
    close(actual.common_slope_benchmark,np.repeat(totals.mean(),len(units)),"Balanced common-slope benchmark")
    saved=tables["sector_covariances"];saved=saved[saved.spec_id.eq(m.spec_id)&saved.hac_lags.eq(band)]
    assert len(saved)==len(units)**2;close(saved.pivot(index="sector_i",columns="sector_j",values="covariance_value").loc[units,units],cov,"Annual all joint total-slope covariance entries");counters["sector_covariance_entries"]+=len(saved)
    components=tables["interaction_components"];components=components[components.spec_id.eq(m.spec_id)&components.hac_lags.eq(band)]
    assert len(components)==len(units)
    for r in components.itertuples():j=names.index(r.term);point(r,beta[j],matrix[j,j],df)
    restriction=np.eye(len(units))[1:]-np.eye(len(units))[[0]];delta=restriction@totals;v=restriction@cov@restriction.T
    singular=np.linalg.svd(v,compute_uv=False);r=min(int((singular>singular.max()*1e-10).sum()),sample.year.nunique()-2);q=len(delta)
    joint=tables["joint_tests"];joint=joint[joint.spec_id.eq(m.spec_id)&joint.hac_lags.eq(band)];assert len(joint)==1;j=joint.iloc[0]
    assert j.covariance_rank==r and j.restrictions==q and j.numerator_df==q and j.denominator_df==df
    if r<q:assert j.test_status.startswith("unavailable")and pd.isna(j.p_value)and pd.isna(j.wald_chi2);counters["joint_singular_no_p"]+=1
    else:
        statistic=float(delta@np.linalg.solve(v,delta));close([j.wald_chi2,j.f_statistic,j.p_value],[statistic,statistic/q,f_distribution.sf(statistic/q,q,df)],"Annual equal-slopes Wald/F/p");counters["joint_available"]+=1
    pairs=tables["pairwise_tests"];pairs=pairs[pairs.spec_id.eq(m.spec_id)&pairs.hac_lags.eq(band)];assert len(pairs)==len(units)*(len(units)-1)//2
    probabilities=[]
    for row in pairs.itertuples():
        a,b=units.index(row.first_sector_id),units.index(row.second_sector_id);effect=totals[a]-totals[b];var=cov[a,a]+cov[b,b]-2*cov[a,b];point(row,effect,var,df);probabilities.append(2*student_t.sf(abs(effect/math.sqrt(var)),df))
    order=np.argsort(probabilities);adjusted=np.empty(len(order));adjusted[order]=np.minimum(1,np.maximum.accumulate(np.asarray(probabilities)[order]*np.arange(len(order),0,-1)))
    close(pairs.p_value_holm,adjusted,"Annual all-pair Holm family");counters["pairwise_rows"]+=len(pairs)


def verify_diagnostics(panels):
    tests=read(OUT/"diagnostics/tests.csv");summary=read(OUT/"diagnostics/summary.csv");count=skipped=0
    expected=set()
    for context,panel in panels.items():
        for controls in["weather","full"]:
            for variant in["all_eligible","exclude_2020_2022"]:
                for sid in panel.sector_id.unique():
                    for adjustment,var in[("NSA","y_nsa"),("SA","y_sa")]:expected.add((context,controls,variant,sid,adjustment,var))
                for var in BASE+(ADDED if controls=="full"else[]):expected.add((context,controls,variant,"common","common",var))
    keys_list=["context","control_set","sample_variant","sector_id","adjustment","variable"]
    assert set(tests[keys_list].itertuples(index=False,name=None))==expected and set(summary[keys_list].itertuples(index=False,name=None))==expected and len(summary)==len(expected)
    for keys,group in tests.groupby(["context","control_set","sample_variant","sector_id","adjustment","variable"]):
        context,controls,variant,sid,adjustment,var=keys;sample=panels[context]
        if controls=="full":sample=sample[sample.eligible_90]
        if variant=="exclude_2020_2022":sample=sample[~sample.year.between(2020,2022)]
        sample=sample.drop_duplicates("year")if sid=="common"else sample[sample.sector_id.eq(sid)]
        sample=sample.sort_values("year");values=sample[var].to_numpy();years=sample.year.to_numpy();det="c"if var in["y_nsa","y_sa","rain_100mm"]else"ct"
        invalid=len(years)<20 or np.any(np.diff(years)!=1)or np.ptp(values)==0
        assert len(group)==2 and set(group.test)=={"ADF","KPSS"}and group.deterministic.eq(det).all()and group.n_periods.eq(len(values)).all()
        if invalid:
            assert group.status.eq("not_run").all()and group[["statistic","p_value","used_lags"]].isna().all().all()
            selected=summary
            for column,value in zip(keys_list,keys):selected=selected[selected[column].eq(value)]
            assert selected.evidence_5pct.iloc[0]=="Unavailable"
            skipped+=2;continue
        probabilities={}
        for row in group.itertuples():
            assert row.status=="ok"
            result=manual_adf(values,int(row.used_lags),det,2)if row.test=="ADF"else manual_kpss(values,int(row.used_lags),det)
            close([row.statistic,row.p_value,row.critical_5pct],[result["statistic"],result["p_value"],result["critical_5pct"]],"Annual independent ADF/KPSS")
            assert int(row.used_lags)==result["aic_lag"if row.test=="ADF"else"auto_lags"]
            if row.test=="KPSS":assert row.p_value_relation==result["p_relation"]
            probabilities[row.test]=row.p_value;count+=1
        selected=summary
        for column,value in zip(["context","control_set","sample_variant","sector_id","adjustment","variable"],keys):selected=selected[selected[column].eq(value)]
        assert len(selected)==1
        a,k=probabilities["ADF"]<.05,probabilities["KPSS"]<.05
        expected=("Compatible with trend stationarity"if det=="ct"else"Compatible with level stationarity")if a and not k else"Unit-root concern"if not a and k else"Mixed evidence"if a else"Inconclusive"
        assert selected.evidence_5pct.iloc[0]==expected
    assert count+skipped==len(tests)
    return {"executed_tests_independently_verified":count,"short_gapped_or_constant_tests_correctly_not_run":skipped,"minimum_contiguous_years":20}


if __name__=="__main__":verify_annual()
