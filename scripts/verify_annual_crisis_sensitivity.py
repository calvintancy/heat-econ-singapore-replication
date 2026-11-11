"""Independent raw-annual and QR/score-covariance crisis companion audit.

This checker imports only independent source parsers and pure verification
linear algebra. It never imports the annual producer or its covariance helper.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import t as student_t, f as f_distribution
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_crisis_sensitivity as audit
from verify_granular import _raw_levels, _raw_weather

OUT = ROOT / "outputs/crisis_sensitivity/annual"
DATA = ROOT / "data/processed/annual"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
YS = {"NSA": "y_nsa", "SA": "y_sa", "LOGLEVEL": "y_log_level", "RAWLEVEL": "y_raw_level"}
VARIANTS = ["all_eligible", "event_dates_excluded", "strict_growth_window_clean"]
MATCHES = {"aggregate_va_matched_va20": ("aggregate_va", "va20"),
           "va11_matched_va19": ("va11", "va19"), "va12_matched_va20": ("va12", "va20"),
           "iip_total_matched_iip6": ("iip_total", "iip6"), "iip_total_matched_iip18": ("iip_total", "iip18"),
           "iip6_matched_iip18": ("iip6", "iip18")}
CORE = ["sector_id", "sector", "year", "annual_nsa", "annual_sa", "y_nsa", "y_sa"]+BASE+["time", "trend", "y_log_level", "y_raw_level"]


def read(path):
    return pd.read_csv(path, dtype={key: str for key in audit.ID_COLUMNS}, float_precision="round_trip", low_memory=False)


def raw_annual_parents():
    """Aggregate full source calendars and derive growth before any masking."""
    levels, names, source_meta = {}, {}, {}
    for family in ("va", "iip"):
        for adjustment in ("SA", "NSA"):
            source, labels, meta = _raw_levels(family, adjustment.lower())
            names[family] = labels
            source_meta[family+"_"+adjustment] = meta
            required = 4 if family=="va" else 12
            for sid, series in source.items():
                for year in sorted({date.year for date in series}):
                    dates = pd.period_range(f"{year}Q1", f"{year}Q4", freq="Q") if family=="va" else pd.period_range(f"{year}-01", f"{year}-12", freq="M")
                    if len(dates)==required and all(date in series for date in dates):
                        levels[family,sid,adjustment,year] = math.fsum(series[date] for date in dates)/(1 if family=="va" else 12)
    monthly = _raw_weather()["M"]
    weather = {}
    for year in sorted({date.year for date in monthly}):
        dates = list(pd.period_range(f"{year}-01",f"{year}-12",freq="M"))
        if not all(date in monthly for date in dates):
            continue
        days = sum(date.days_in_month for date in dates)
        weather[year] = {term: math.fsum(monthly[date][term]*(1 if term=="rain_100mm" else date.days_in_month) for date in dates)/(1 if term=="rain_100mm" else days) for term in BASE}
    va_catalog = read(ROOT/"data/processed/sector_catalog.csv")
    iip_catalog = read(ROOT/"data/processed/iip/sector_catalog.csv")
    ids = {"aggregate_va": ("va", ["1.4"]),
           "va11": ("va", va_catalog.loc[va_catalog.broad_panel,"sector_id"].tolist()),
           "va19": ("va", va_catalog.loc[va_catalog.detailed_panel,"sector_id"].tolist()),
           "iip_total": ("iip", ["1"]),
           "iip6": ("iip", iip_catalog.loc[iip_catalog.broad_panel,"sector_id"].tolist()),
           "iip18": ("iip", iip_catalog.loc[iip_catalog.detailed_panel,"sector_id"].tolist())}
    parents, counts = {}, []
    for context, (family, sectors) in ids.items():
        panel = read(DATA/f"panel_{context}.csv")
        years = {year for year in weather if all((family,sid,adj,year) in levels and (family,sid,adj,year-1) in levels for sid in sectors for adj in ("SA","NSA"))}
        assert set(panel.year)==years and set(panel.sector_id)==set(sectors)
        assert len(panel)==len(years)*len(sectors) and not panel.duplicated(["sector_id","year"]).any()
        for row in panel.itertuples():
            for adjustment in ("SA","NSA"):
                current, prior = levels[family,row.sector_id,adjustment,row.year], levels[family,row.sector_id,adjustment,row.year-1]
                audit.check(getattr(row,"annual_"+adjustment.lower()), current, "Annual raw complete-calendar level", atol=2e-10, rtol=2e-12)
                audit.check(getattr(row,"y_"+adjustment.lower()), 100*math.log(current/prior), "Annual raw prior-calendar growth", atol=3e-10, rtol=2e-12)
            for term in BASE:
                audit.check(getattr(row,term), weather[row.year][term], "Annual raw day-weighted / summed weather", atol=5e-11, rtol=2e-12)
            assert row.time==row.year-1982 and row.trend==row.year-2000 and row.sector==names[family][row.sector_id]
        panel["y_log_level"] = 100*np.log(panel.annual_sa)
        panel["y_raw_level"] = panel.annual_sa
        parents[context] = panel[CORE].copy()
    # Independently derive ownership of dwellings from full raw annual levels.
    for context, base in (("va12","va11"),("va20","va19")):
        rows=[]
        for year in sorted(parents[base].year.unique()):
            calendar = parents[base].loc[parents[base].year.eq(year)].iloc[0]
            sa,nsa = levels["va","1.3","SA",year], levels["va","1.3","NSA",year]
            rows.append({"sector_id":"1.3","sector":names["va"]["1.3"],"year":year,
                "annual_sa":sa,"annual_nsa":nsa,"y_sa":100*math.log(sa/levels["va","1.3","SA",year-1]),
                "y_nsa":100*math.log(nsa/levels["va","1.3","NSA",year-1]),
                "y_log_level":100*math.log(sa),"y_raw_level":sa,
                **{term:calendar[term] for term in BASE},"time":year-1982,"trend":year-2000})
        parents[context] = pd.concat([parents[base],pd.DataFrame(rows)],ignore_index=True)[CORE]
    for context,(base,template) in MATCHES.items():
        parents[context] = parents[base][parents[base].year.isin(parents[template].year)].copy()
        assert set(parents[context].year)==set(parents[template].year)
    for context,parent in parents.items():
        parent=parent.sort_values(["sector_id","year"]).reset_index(drop=True)
        parents[context]=parent
        counts.append({"context":context,"n_sectors":parent.sector_id.nunique(),"n_periods":parent.year.nunique(),"start":int(parent.year.min()),"end":int(parent.year.max()),"includes_dwellings":context.startswith("aggregate_va") or context in {"va12","va20","va12_matched_va20"}})
    assert len(parents)==14
    return parents,{"source_tables":source_meta,"raw_complete_annual_cells":len(levels),"complete_weather_years":len(weather),"contexts":counts}


def annual_covariance(fit, sample, outcome, method, bandwidth):
    n,k=fit["n"],fit["k"]
    influence=audit.influences(fit,np.eye(k))*fit["residual"][:,[outcome]]
    years=sorted(sample.year.unique());t=len(years)
    if method=="Sector cluster":
        sectors=sorted(sample.sector_id.unique());g=len(sectors)
        score=np.vstack([influence[sample.sector_id.eq(sid)].sum(axis=0) for sid in sectors])
        correction=g/(g-1)*(n-1)/(n-k)
        return score.T@score*correction,g-1,correction
    score=np.vstack([influence[sample.year.eq(year)].sum(axis=0) for year in years])
    correction=t/(t-1)*(n-1)/(n-k)
    return audit.bartlett_scores(score,np.asarray(years),bandwidth)*correction,t-1,correction


def point(row,beta,variance,df):
    assert variance>0 and np.isfinite(variance)
    se=math.sqrt(variance);critical=student_t.ppf(.975,df)
    probability=float(2*student_t.sf(abs(beta/se),df))
    audit.check(row[["coefficient","std_error","p_value","ci_lower","ci_upper"]].to_numpy(float),
        [beta,se,probability,beta-critical*se,beta+critical*se],"Annual crisis point / SE / CI / p",atol=5e-7)
    assert row.inference_df==df


def verify_annual_crisis_sensitivity():
    audit.ERRORS.clear()
    manifest=json.loads((OUT/"manifest.json").read_text())
    policy=json.loads((ROOT/"notes/crisis_sensitivity_policy.json").read_text())
    assert manifest["policy_sha256"]==audit.sha256(ROOT/"notes/crisis_sensitivity_policy.json")
    assert manifest["controls"]==BASE and manifest["contexts"]==14
    for relative,digest in {**manifest["source_hashes"],**manifest["protected_outputs"]}.items():
        assert audit.sha256(ROOT/relative)==digest,("Annual crisis source or protected receipt changed",relative)
    parents,source_audit=raw_annual_parents()
    expected_sources={name:[f"data/processed/annual/panel_{name}.csv"] for name in ("aggregate_va","va11","va19","iip_total","iip6","iip18")}
    expected_sources["va12"]=["data/processed/annual/panel_va11.csv","data/processed/annual/annual_levels.csv"]
    expected_sources["va20"]=["data/processed/annual/panel_va19.csv","data/processed/annual/annual_levels.csv"]
    for context,(base,template) in MATCHES.items():
        expected_sources[context]=sorted(set(expected_sources[base]+expected_sources[template]))
    assert set(manifest["context_provenance"])==set(parents)
    for context,expected_paths in expected_sources.items():
        provenance=manifest["context_provenance"][context]
        assert set(provenance["source_files"])==set(expected_paths)
        assert provenance["matched_to"]==(MATCHES[context][1] if context in MATCHES else None)
        assert provenance["includes_dwellings"]==(context.startswith("aggregate_va") or context in {"va12","va20","va12_matched_va20"})
    assert set(manifest["source_hashes"])==set(path for paths in expected_sources.values() for path in paths)
    event_years={month.year for event in policy["events"] for month in pd.period_range(event["start"],event["end"],freq="M")}
    masks=read(OUT/"calendar_masks.csv")
    assert not masks.duplicated(["context","sample_variant","year"]).any()
    selected={};sample_records=[]
    for context,parent in parents.items():
        for variant in VARIANTS:
            years=set(parent.year)
            retained=years if variant=="all_eligible" else {year for year in years if year not in event_years and (variant!="strict_growth_window_clean" or year-1 not in event_years)}
            saved=masks[masks.context.eq(context)&masks.sample_variant.eq(variant)].sort_values("year")
            assert set(saved.year)==years and set(saved.loc[saved.retained,"year"])==retained
            assert np.array_equal(saved.current_year_event.to_numpy(),np.array([year in event_years for year in saved.year]))
            assert np.array_equal(saved.prior_year_event.to_numpy(),np.array([year-1 in event_years for year in saved.year]))
            selected[context,variant]=parent[parent.year.isin(retained)].copy()
            sample_records.append({"context":context,"sample_variant":variant,"n_periods":len(retained),"n_sectors":parent.sector_id.nunique(),"n_obs":len(selected[context,variant]),"start":min(retained),"end":max(retained),"calendar_gaps":max(retained)-min(retained)+1-len(retained)})
    specs=read(OUT/"specs.csv");assert specs.spec_id.is_unique and len(specs)==1656 and specs.status.eq("estimated").all()
    expected=set()
    for (context,variant),sample in selected.items():
        sectors=sorted(sample.sector_id.unique())
        for adjustment in YS:
            for model,unit in [("common","pooled")]+([("H1","pooled")]+[("separate",sid) for sid in sectors] if len(sectors)>1 else []):
                expected.add((context,variant,adjustment,model,unit))
    assert set(specs[["context","sample_variant","adjustment","model","sector_id"]].itertuples(index=False,name=None))==expected
    tables={name:read(OUT/f"{name}.csv") for name in ("common_estimates","sector_estimates","separate_estimates","joint_tests","sector_covariances","interaction_components")}
    grouped={name:{str(key):group for key,group in frame.groupby("spec_id",sort=False)} for name,frame in tables.items()}
    native=[];fit_records=[];joint_records=[]
    with threadpool_limits(limits=1):
        for (context,variant,model,unit),models in specs.groupby(["context","sample_variant","model","sector_id"],sort=True):
            assert len(models)==4 and set(models.adjustment)==set(YS)
            first=models.iloc[0];sample=read(ROOT/first.sample_csv)
            base=selected[context,variant]
            if model=="separate":base=base[base.sector_id.eq(unit)]
            keys=["sector_id","year"]
            actual=sample.sort_values(keys).reset_index(drop=True);base=base.sort_values(keys).reset_index(drop=True)
            assert len(actual)==len(base) and not actual.duplicated(keys).any()
            for name in CORE:
                if name in {"sector_id","sector"}:assert actual[name].tolist()==base[name].tolist()
                else:audit.check(actual[name].to_numpy(float),base[name].to_numpy(float),"Annual crisis exact raw-derived sample",atol=3e-10,rtol=2e-12)
            rhs=("temp_c * C(sector_id)" if model=="H1" else "temp_c")+" + rain_100mm + rh_pct + trend + C(sector_id)"
            design=dmatrix(rhs,sample,return_type="dataframe",NA_action="raise")
            names=list(design.columns);x=design.to_numpy(float)
            outcomes=list(YS.values());y=sample[outcomes].to_numpy(float)
            fit=audit.independent_qr(x,y,names)
            n,k,t=len(sample),len(names),sample.year.nunique();sectors=sorted(sample.sector_id.unique())
            for j,(adjustment,outcome) in enumerate(YS.items()):
                spec=models[models.adjustment.eq(adjustment)].iloc[0]
                assert spec.outcome==outcome and spec.formula==outcome+" ~ "+rhs
                other=read(ROOT/spec.sample_csv)
                pd.testing.assert_frame_equal(other,sample,check_dtype=False,check_exact=True)
                for field,value in [("n_obs",n),("n_periods",t),("n_parameters",k),("design_rank",k),("n_time_parameters",5),("time_design_rank",5),("time_residual_df",t-5),("nominal_time_residual_df",t-5),("residual_df",n-k),("calendar_span_years",int(sample.year.max()-sample.year.min()+1)),("calendar_gaps",int(sample.year.max()-sample.year.min()+1-t))]:
                    assert int(spec[field])==value,(field,spec[field],value)
                assert spec.n_sectors==len(sectors) and spec.start==sample.year.min() and spec.end==sample.year.max()
                assert not audit._truth(spec.year_fixed_effects) and audit._truth(spec.linear_time_trend) and not audit._truth(spec.season_fixed_effects)
                assert json.loads(spec.weather_terms)==BASE and json.loads(spec.controls)==BASE[1:]
                audit.check([spec.mean_y,spec.sd_y],[y[:,j].mean(),y[:,j].std(ddof=1)],"Annual crisis exact mean and SD",atol=3e-10,rtol=2e-12)
                assert "r_squared" in spec
                audit.check(spec.r_squared,fit["r_squared"][j],"Annual specification centered R squared",atol=5e-10)
                assert audit._truth(spec.fragile_short_sample)==(t-5<=3)
                if spec.calendar_gaps:assert "Not cleared" in spec.stationarity_status and "gapped" in spec.stationarity_status
                includes=context.startswith("aggregate_va") or context in {"va12","va20","va12_matched_va20"}
                assert audit._truth(spec.includes_dwellings)==includes
                native.append({"label":f"Annual {context}/weather/{variant}/{adjustment}/{model}/{unit if model=='separate' else 'joint'}",
                    "outcome":outcome,"n_obs":n,"n_parameters":k,"source_formula":spec.formula,
                    "design_sha256":audit.array_sha256(x),"outcome_sha256":audit.array_sha256(y[:,j]),
                    "coefficient_names_sha256":hashlib.sha256(json.dumps(names,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
                filename="sector_estimates" if model=="H1" else "separate_estimates" if model=="separate" else "common_estimates"
                rows=grouped[filename][spec.spec_id]
                expected_bands={("Sector cluster",0),("Driscoll-Kraay",1),("Driscoll-Kraay",2)} if model=="common" and len(sectors)>1 else {("Driscoll-Kraay",1),("Driscoll-Kraay",2)}
                assert set(zip(rows.covariance,rows.hac_lags))==expected_bands
                for (method,bandwidth),points in rows.groupby(["covariance","hac_lags"],sort=False):
                    bandwidth=int(bandwidth);v,df,correction=annual_covariance(fit,sample,j,method,bandwidth)
                    audit.check(points.finite_sample_correction,np.full(len(points),correction),"Annual method-specific finite correction",atol=1e-12,rtol=1e-12)
                    assert "r_squared" in points
                    audit.check(points.r_squared,np.full(len(points),fit["r_squared"][j]),"Annual centered R squared",atol=5e-10)
                    main=method=="Sector cluster" if model=="common" and len(sectors)>1 else bandwidth==1
                    assert points.main_inference.eq(main).all()
                    if model!="H1":
                        assert len(points)==1
                        index=names.index("temp_c");point(points.iloc[0],fit["beta"][index,j],v[index,index],df)
                    else:
                        contrast=audit.total_temperature_contrast(names,sectors)
                        beta=contrast@fit["beta"][:,j];cov=contrast@v@contrast.T
                        p=points.set_index("sector_id").loc[sectors]
                        assert len(p)==len(sectors)
                        saved=grouped["sector_covariances"][spec.spec_id];saved=saved[saved.covariance.eq(method)&saved.hac_lags.eq(bandwidth)]
                        audit.check(audit.covariance_matrix(saved,sectors,"sector_i","sector_j"),cov,"Annual H1 full joint covariance",atol=5e-6,rtol=3e-7)
                        components=grouped["interaction_components"][spec.spec_id];components=components[components.covariance.eq(method)&components.hac_lags.eq(bandwidth)].set_index("sector_id").loc[sectors]
                        for index,sid in enumerate(sectors):
                            point(p.loc[sid],beta[index],cov[index,index],df)
                            unit_y=sample.loc[sample.sector_id.eq(sid),outcome]
                            audit.check([p.loc[sid].mean_y,p.loc[sid].sd_y],[unit_y.mean(),unit_y.std(ddof=1)],"Annual sector exact mean/SD",atol=3e-10,rtol=2e-12)
                            delta=beta[index]-beta[0]
                            var_delta=0. if index==0 else cov[index,index]+cov[0,0]-2*cov[0,index]
                            covariance_delta=0. if index==0 else cov[0,index]-cov[0,0]
                            audit.check([p.loc[sid].temp_main,p.loc[sid].interaction_delta,p.loc[sid].var_main,p.loc[sid].var_delta,p.loc[sid].cov_main_delta],
                                [beta[0],delta,cov[0,0],var_delta,covariance_delta],"Annual baseline plus increment components",atol=5e-6)
                            point(components.loc[sid],beta[0] if index==0 else delta,cov[0,0] if index==0 else var_delta,df)
                        joint=grouped["joint_tests"][spec.spec_id];joint=joint[joint.covariance.eq(method)&joint.hac_lags.eq(bandwidth)]
                        assert len(joint)==1;joint=joint.iloc[0]
                        r=np.eye(len(sectors))[1:]-np.eye(len(sectors))[[0]];d=r@beta;c=r@cov@r.T
                        singular=np.linalg.svd(c,compute_uv=False);rank=min(int(np.sum(singular>singular.max()*1e-10)),t-2)
                        assert joint.covariance_rank==rank and joint.structural_rank_bound==t-2 and joint.restrictions==len(sectors)-1 and joint.denominator_df==t-1
                        if rank<len(sectors)-1:
                            assert "unavailable" in joint.status and pd.isna(joint.p_value) and pd.isna(joint.f_statistic)
                        else:
                            assert joint.status=="available";stat=float(d@np.linalg.solve(c,d))/(len(sectors)-1)
                            audit.check(joint.f_statistic,stat,"Annual joint F statistic",atol=2e-6,rtol=3e-6)
                            audit.check(joint.p_value,f_distribution.sf(stat,len(sectors)-1,t-1),"Annual joint F p",atol=2e-8)
                        joint_records.append({"spec_id":spec.spec_id,"context":context,"sample_variant":variant,"adjustment":adjustment,"hac_lags":bandwidth,"rank":rank,"bound":t-2,"restrictions":len(sectors)-1,"available":rank==len(sectors)-1})
            fit_records.append({"context":context,"sample_variant":variant,"model":model,"sector_id":unit,"n_obs":n,"n_periods":t,"n_parameters":k,"outcomes":4})
            if len(fit_records)%60==0:print(f"Annual crisis independent QR / covariance audit: {len(fit_records)}/414 designs",flush=True)
    for filename in ("sector_estimates","separate_estimates"):
        for _,family in tables[filename].groupby(["context","sample_variant","adjustment","covariance","hac_lags"],sort=False):
            audit.check(family.p_value_holm_within_table.to_numpy(),audit.holm(family.p_value.to_numpy()),"Annual within-table Holm zero-effect tests",atol=3e-8)
    log_path=OUT/"pyfixest/fit_log.csv";log=read(log_path);receipt=json.loads((OUT/"pyfixest/estimator_manifest.json").read_text())
    expected_native=pd.DataFrame(native)
    assert Counter(log[["label","outcome","n_obs","n_parameters","source_formula"]].itertuples(index=False,name=None))==Counter(expected_native[["label","outcome","n_obs","n_parameters","source_formula"]].itertuples(index=False,name=None))
    lookup=log.set_index(["label","outcome"])
    for expected_row in native:
        actual=lookup.loc[(expected_row["label"],expected_row["outcome"])]
        for field in ("design_sha256","outcome_sha256","coefficient_names_sha256"):
            assert actual[field]==expected_row[field],("Annual native exact logical-input identity",field,expected_row["label"])
    assert len(log)==1656==receipt["native_fits_recorded"] and log.api_call_id.nunique()==1656
    assert receipt["fit_log_sha256"]==audit.sha256(log_path)
    assert log.native_class.eq("Feols").all() and log.native_module.str.startswith("pyfixest").all()
    assert log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all() and log.native_residuals_sha256.eq(log.residuals_sha256).all()
    for field in ("native_mapped_coefficient_max_abs_error","native_residual_max_abs_error","native_response_max_abs_error","native_outcome_max_abs_error","native_design_max_abs_error"):
        assert log[field].eq(0).all()
    for relative,digest in {**manifest["source_hashes"],**manifest["protected_outputs"]}.items():assert audit.sha256(ROOT/relative)==digest
    report={"status":"passed","contexts":len(parents),"samples":len(selected),"independent_designs":len(fit_records),"native_fits":len(log),
        "joint_tests":len(joint_records),"unavailable_joint_tests":sum(not row["available"] for row in joint_records),
        "source_audit":source_audit,"policy_sha256":manifest["policy_sha256"],"native_fit_log_sha256":audit.sha256(log_path),
        "protected_sources_and_receipts_unchanged":True,"numerical_checks":audit.ERRORS,
        "scope":"Independent full raw-calendar annual VA sums / IIP means and pre-mask SA/NSA growth; raw annual weather; complete dwellings component; declared calendar matching; current/prior-year exclusions; full QR; unit CR1 / actual-year Bartlett1/2; H1 total baseline-plus-increment covariance and direct full-rank joint tests; within-table Holm; native logical-input receipts. Annual sector intercepts plus common trend, no year FE. Not stationarity/causal certification or a Fable execution claim."}
    pd.DataFrame(fit_records).to_csv(OUT/"verification_fits.csv",index=False)
    pd.DataFrame(sample_records).to_csv(OUT/"verification_samples.csv",index=False)
    pd.DataFrame(joint_records).to_csv(OUT/"verification_joint_tests.csv",index=False)
    (OUT/"verification.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Annual crisis independent verification passed: {len(parents)} contexts, {len(selected)} samples, {len(log)} native fits.",flush=True)
    return report


if __name__=="__main__":verify_annual_crisis_sensitivity()
