"""Independent audit of the subannual literature-motivated sensitivity.

Raw official output calendars and raw NEA weather are reconstructed before
eligibility, matching, or crisis masks. Numerical benchmarks use independent QR
and calendar-keyed Bartlett scores, never a production regression/covariance
helper. Passing is not paper replication, causal certification, or a claim that
dynamic FE bias and many-restriction Wald fragility have been resolved.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from collections import OrderedDict, Counter

for _key in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular
from scipy.stats import f as f_distribution
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_crisis_sensitivity as independent
from verify_granular import _raw_levels, _quarterly_levels, _raw_weather

OUT = ROOT / "outputs/literature_sensitivity"
POLICY = ROOT / "notes/crisis_sensitivity_policy.json"
YS = ["y_short", "y_yoy_nsa", "y_yoy"]
VARIANTS = ["all_eligible", "event_dates_excluded", "strict_growth_window_clean"]
MATCHES = {name: target for name, (_, target) in independent.MATCHED_TOTALS.items()}
MATCHES.update({"va-broad-matched": "va-detailed", "va12-matched": "va20",
                "iip-monthly-broad-matched": "iip-monthly-detailed",
                "iip-quarterly-broad-matched": "iip-quarterly-detailed"})
GEOMETRY: OrderedDict = OrderedDict()


def read(path: Path, columns: set | None = None):
    return pd.read_csv(path, dtype={name: str for name in independent.ID_COLUMNS},
        float_precision="round_trip", low_memory=False,
        usecols=None if columns is None else lambda column: column in columns)


def raw_histories():
    """Return full raw-growth dictionaries, including pre-weather histories."""
    levels, source_files = {}, {}
    for family in ("va", "iip"):
        for adjustment in ("sa", "nsa"):
            values, _, _ = _raw_levels(family, adjustment)
            levels[family, adjustment, "Q" if family == "va" else "M"] = values
            if family == "iip":
                levels[family, adjustment, "Q"] = _quarterly_levels(values)
            for suffix in ("metadata", "tabledata"):
                path = ROOT / f"data/raw/{family}_{adjustment}_{suffix}.json"
                source_files[str(path.relative_to(ROOT))] = independent.sha256(path)
    growth = {}
    for (family, adjustment, frequency), units in levels.items():
        h = 12 if frequency == "M" else 4
        for outcome, distance in [("y_short", 1), ("y_yoy", h)] if adjustment == "sa" else [("y_yoy_nsa", h)]:
            for sid, series in units.items():
                growth[family, frequency, outcome, sid] = {date: 100*math.log(value/series[date-distance])
                    for date, value in series.items() if date-distance in series}
    return growth, _raw_weather(), source_files


def build_history_samples(policy: dict):
    """Independently separate raw support, target alignment and event masks."""
    parents, parent_audit = independent.audit_raw_parents()
    growth, weather, source_hashes = raw_histories()
    own, features, support_metadata = {}, {}, {}
    for context in independent.CONTEXTS:
        frequency = "M" if "monthly" in context else "Q"
        period, h = ("month", 12) if frequency == "M" else ("quarter", 4)
        family = "iip" if context.startswith("iip") else "va"
        path = ROOT / f"outputs/crisis_sensitivity/parents/{context}.csv"
        parent = read(path).sort_values(["time", "sector_id"]).reset_index(drop=True)
        independent.compare_sample(parent, parents[context], period)
        dates = pd.PeriodIndex(parent[period], freq=frequency)
        for outcome in YS:
            frame = parent.copy()
            for lag in range(h+1):
                for raw_name, name in [("temp_c", "temp"), ("rain_100mm", "rain")]:
                    frame[f"{name}_l{lag}"] = [weather[frequency].get(date-lag, {}).get(raw_name, np.nan) for date in dates]
            for lag in range(1, h+1):
                frame[f"ar_l{lag}"] = [growth[family, frequency, outcome, str(sid)].get(date-lag, np.nan)
                    for sid, date in zip(frame.sector_id, dates)]
            current = [growth[family, frequency, outcome, str(sid)].get(date, np.nan) for sid, date in zip(frame.sector_id, dates)]
            independent.check(parent[outcome].to_numpy(), current, "Current Y from full raw official calendar", atol=3e-10, rtol=2e-12)
            weather_columns = [f"{name}_l{lag}" for name in ("temp", "rain") for lag in range(h+1)]
            ar_columns = [f"ar_l{lag}" for lag in range(1, h+1)]
            raw_weather_valid = np.isfinite(frame[weather_columns]).all(axis=1)
            raw_ar_valid = np.isfinite(frame[ar_columns]).all(axis=1)
            eligible = raw_weather_valid & raw_ar_valid & np.isfinite(frame[[outcome, "rh_pct"]]).all(axis=1)
            valid_dates = frame.loc[eligible].groupby("time").sector_id.nunique()
            valid_dates = set(valid_dates.index[valid_dates.eq(parent.sector_id.nunique())])
            own[context, outcome] = valid_dates
            features[context, outcome] = frame
            support_metadata[context, outcome] = {"parent_n_periods": parent.time.nunique(),
                "raw_model_history_n_periods": len(valid_dates),
                "raw_history_loss_periods": parent.time.nunique()-len(valid_dates),
                "raw_AR_history_loss_periods": frame.loc[~raw_ar_valid, "time"].nunique(),
                "weather_history_loss_periods": frame.loc[~raw_weather_valid, "time"].nunique()}
    masks = read(OUT / "masks.csv")
    inventory = read(OUT / "sample_inventory.csv")
    assert len(inventory) == 180 and not inventory.duplicated(["context", "outcome", "sample_variant"]).any()
    assert not masks.duplicated(["context", "outcome", "time"]).any()
    result, history_records = {}, []
    for context in independent.CONTEXTS:
        frequency = "M" if "monthly" in context else "Q"
        period, h = ("month", 12) if frequency == "M" else ("quarter", 4)
        for outcome in YS:
            frame, raw_dates = features[context, outcome], own[context, outcome]
            target = MATCHES.get(context)
            aligned = raw_dates if target is None else raw_dates & own[target, outcome]
            dates = pd.PeriodIndex(frame.loc[frame.time.isin(aligned), period].drop_duplicates(), freq=frequency)
            support = h+1 if outcome == "y_short" else 2*h
            event_at_t, event_in_support = [], []
            event_sets = [(event["event_id"], {month.ordinal if frequency=="M" else month.asfreq("Q").ordinal
                for month in pd.period_range(event["start"], event["end"], freq="M")}) for event in policy["events"]]
            for date in dates:
                event_at_t.append([name for name, values in event_sets if date.ordinal in values])
                needed = set(range(date.ordinal-support, date.ordinal+1))
                event_in_support.append([name for name, values in event_sets if not needed.isdisjoint(values)])
            saved = masks[masks.context.eq(context) & masks.outcome.eq(outcome)].set_index("period").loc[dates.astype(str)]
            assert len(saved)==len(dates) and saved.output_support_horizon.eq(support).all()
            assert saved.output_support_start.tolist()==[str(date-support) for date in dates]
            assert saved.event_ids_at_t.map(independent._event_list).tolist()==event_at_t
            assert saved.event_ids_in_output_support.map(independent._event_list).tolist()==event_in_support
            assert saved.retain_event_dates.map(independent._truth).tolist()==[not event for event in event_at_t]
            assert saved.retain_strict.map(independent._truth).tolist()==[not event for event in event_in_support]
            details = {**support_metadata[context, outcome], "target_alignment_loss_periods": len(raw_dates)-len(aligned),
                       "maximum_history_n_periods": len(aligned), "history_loss_periods": frame.time.nunique()-len(aligned)}
            for variant in VARIANTS:
                retain = np.ones(len(dates), dtype=bool) if variant=="all_eligible" else np.array([not events for events in (event_at_t if variant=="event_dates_excluded" else event_in_support)])
                retained = set(dates[retain].astype(str))
                expected = frame[frame[period].isin(retained)].copy()
                path = OUT / "samples" / f"{context}__{outcome}__{variant}.csv"
                sample = read(path)
                independent.compare_sample(sample, expected, period)
                info = inventory[inventory.context.eq(context)&inventory.outcome.eq(outcome)&inventory.sample_variant.eq(variant)].iloc[0]
                assert info.implementation_version=="raw_history_v2"
                assert info.parent_sha256==independent.sha256(ROOT/info.parent_csv)
                assert info.sample_sha256==independent.sha256(path)
                assert info.weather_history_sha256==independent.sha256(ROOT/info.weather_history_csv)
                for field, value in details.items():
                    assert int(info[field])==value, (context,outcome,variant,field,info[field],value)
                if target:
                    assert info.matched_to==target and info.matched_template_n_periods==len(own[target,outcome])
                    assert info.matched_history_template_csv==f"outputs/crisis_sensitivity/parents/{target}.csv"
                    assert info.matched_history_template_sha256==independent.sha256(ROOT/info.matched_history_template_csv)
                else:
                    assert pd.isna(info.matched_to) and pd.isna(info.matched_template_n_periods)
                assert info.n_periods==len(retained) and info.maximum_history_n_obs==len(aligned)*independent.CONTEXTS[context]
                result[context,outcome,variant] = sample
                history_records.append({"context":context,"outcome":outcome,"sample_variant":variant,
                    **details,"matched_to":target or "","retained_periods":len(retained),"n_obs":len(sample),
                    "first":str(min(retained)),"last":str(max(retained)),"output_support_horizon":support})
    for (context,outcome,variant),sample in result.items():
        if context in MATCHES:
            assert set(sample.time)==set(result[MATCHES[context],outcome,variant].time), "Matched comparisons lost their common post-history calendar"
    return result, history_records, source_hashes, parent_audit


def expected_design(sample, specification, structure, h, names):
    features = ["temp_endpoint_difference","temp_growth_sum"] if specification=="two_channel" else [f"temp_l{lag}" for lag in ([0] if specification=="benchmark" else range(h+1))]
    controls = [f"rain_l{lag}" for lag in ([0] if specification=="benchmark" else range(h+1))]+["rh_pct"]
    if specification=="dynamic":controls += [f"ar_l{lag}" for lag in range(1,h+1)]
    values = {name:sample[name].to_numpy(float) for name in controls+features if name in sample}
    if specification=="two_channel":
        values["temp_endpoint_difference"] = sample.temp_l0.to_numpy()-sample[f"temp_l{h}"].to_numpy()
        # An explicit sum, not a mean or endpoint difference.
        values["temp_growth_sum"] = np.add.reduce(sample[[f"temp_l{lag}" for lag in range(h)]].to_numpy(float),axis=1)
    centered = {name:value-value.mean() for name,value in values.items()}
    columns=[]
    for name in names:
        if match:=re.fullmatch(r"sector_season\[([^,]+),(\d+)\]",name):
            columns.append((sample.sector_id.eq(match[1])&sample.season.eq(int(match[2]))).to_numpy(float))
        elif match:=re.fullmatch(r"year\[(\d+)\]",name):
            columns.append(sample.year.eq(int(match[1])).to_numpy(float))
        elif name in centered:
            columns.append(centered[name])
        elif match:=re.fullmatch(r"(.+):C\(sector_id\)\[T\.(.+)\]",name):
            assert structure=="H1" and match[1] in features
            columns.append(centered[match[1]]*sample.sector_id.eq(match[2]).to_numpy(float))
        else:raise AssertionError(("Unexpected design column",name))
    sectors=sorted(sample.sector_id.unique());seasons=sorted(sample.season.unique());years=sorted(sample.year.unique())
    required=[f"sector_season[{sid},{season}]" for sid in sectors for season in seasons]+[f"year[{year}]" for year in years[1:]]+controls
    for feature in features:
        required += [feature]+([f"{feature}:C(sector_id)[T.{sid}]" for sid in sectors[1:]] if structure=="H1" else [])
    assert names==required and len(set(names))==len(names)
    return np.column_stack(columns),controls,features


def qr_fit(x,y,names):
    key=independent.array_sha256(x)
    cached=GEOMETRY.get(key)
    if cached is None:
        scale=np.linalg.norm(x,axis=0);assert (scale>0).all()
        q,r=qr(x/scale,mode="economic")
        singular=np.linalg.svd(r,compute_uv=False)
        assert np.sum(singular>singular[0]*1e-10)==len(names)
        inverse=solve_triangular(r,q.T)/scale[:,None]
        cached=(q,r,scale,inverse)
        if x.shape[0]<1500 and x.shape[1]<150:
            GEOMETRY[key]=cached
            if len(GEOMETRY)>64:GEOMETRY.popitem(last=False)
    else:GEOMETRY.move_to_end(key)
    q,r,scale,inverse=cached
    beta=solve_triangular(r,q.T@y)/scale
    residual=y-q@(q.T@y)
    return {"beta":beta,"residual":residual,"inverse":inverse,"names":names,"n":len(y),"k":len(names),
            "r_squared":1-residual@residual/((y-y.mean())@(y-y.mean()))}


def calendar_meat(score,dates,bandwidth):
    """Join actual date keys; accumulate weighted past scores before products."""
    dates=np.asarray(dates,dtype=int);assert len(np.unique(dates))==len(dates)
    lookup={int(date):i for i,date in enumerate(dates)}
    preceding=np.zeros_like(score)
    for lag in range(1,bandwidth+1):
        pairs=[(i,lookup[int(date)-lag]) for i,date in enumerate(dates) if int(date)-lag in lookup]
        if pairs:
            current,previous=np.array(pairs).T
            preceding[current]+=(1-lag/(bandwidth+1))*score[previous]
    cross=score.T@preceding
    return score.T@score+cross+cross.T


def full_covariances(fit,sample,h,structure):
    individual=fit["inverse"].T*fit["residual"][:,None]
    n,k,t=fit["n"],fit["k"],sample.time.nunique()
    sectors=sorted(sample.sector_id.unique());dates=sorted(sample.time.unique())
    result=[]
    if structure=="common" and len(sectors)>1:
        score=np.vstack([individual[sample.sector_id.eq(sid)].sum(axis=0) for sid in sectors])
        factor=len(sectors)/(len(sectors)-1)*(n-1)/(n-k)
        result.append(("Sector cluster",0,score.T@score*factor,factor,len(sectors)-1))
    score=np.vstack([individual[sample.time.eq(date)].sum(axis=0) for date in dates])
    factor=t/(t-1)*(n-1)/(n-k)
    for band in [h,2*h]:
        result.append(("Calendar HAC" if len(sectors)==1 else "Driscoll-Kraay",band,calendar_meat(score,dates,band)*factor,factor,t-1))
    return individual,result


def profile_contrasts(names,features,sectors,structure,h,specification):
    profiles=sectors if structure=="H1" else [sectors[0] if len(sectors)==1 else "pooled"]
    result={}
    for sid in profiles:
        basis=np.zeros((len(features),len(names)))
        for j,feature in enumerate(features):
            basis[j,names.index(feature)]=1
            if structure=="H1" and sid!=sectors[0]:basis[j,names.index(f"{feature}:C(sector_id)[T.{sid}]")]=1
        if specification=="two_channel":
            expansion=np.zeros((h+1,2));expansion[0,0]=1;expansion[-1,0]=-1;expansion[:h,1]=1
            result[sid]=(expansion@basis,basis)
        else:result[sid]=(basis,basis)
    return result


def restriction_operators(length,outcome):
    identity=np.eye(length)
    operators={"All temperature-lag coefficients zero":identity,
               "Cumulative direct temperature response zero":np.ones((1,length))}
    if length>1:
        endpoint=1 if outcome=="y_short" else length-1
        zeros=np.delete(identity,[0,endpoint],axis=0)
        operators["Exact contemporaneous static-output-level restriction"]=np.vstack([zeros,identity[0]+identity[endpoint]])
        if outcome!="y_short":
            operators["Equal one-period growth responses over YoY accumulation window"]=np.vstack([identity[1:-1]-identity[0],identity[-1]])
    return operators


def verify_test(row,beta,covariance,hc0,restriction,df,bound):
    effect=restriction@beta;matrix=restriction@covariance@restriction.T;matrix=(matrix+matrix.T)/2
    singular=np.linalg.svd(matrix,compute_uv=False)
    scale=float(np.linalg.norm(restriction@hc0@restriction.T,2));zero=1e-12*scale
    cutoff=max(float(singular[0])*1e-10,zero)
    rank=min(int(np.sum(singular>cutoff)),bound)
    q=len(effect);available=rank==q and singular[0]>zero and df>0
    assert int(row.restrictions)==q and int(row.numerator_df)==q and int(row.denominator_df)==df
    assert int(row.structural_rank_bound)==bound and int(row.covariance_rank)==rank
    independent.check([row.hc0_covariance_scale,row.zero_covariance_threshold,row.rank_cutoff],
        [scale,zero,cutoff],"Restriction HC0 / numerical rank cutoffs",atol=2e-10,rtol=2e-6)
    assert (row.status=="available")==available
    if available:
        statistic=max(float(effect@np.linalg.solve(matrix,effect)),0.)
        independent.check([row.wald_chi2,row.f_statistic],[statistic,statistic/q],"Direct full-rank Wald/F",atol=1e-5,rtol=1e-5)
        independent.check(row.p_value,f_distribution.sf(statistic/q,q,df),"Restriction F p",atol=2e-8,rtol=2e-6)
    else:assert pd.isna(row.p_value) and pd.isna(row.wald_chi2) and pd.isna(row.f_statistic)
    return available


def nesting_embedding(full_names,reduced_names,h):
    """Map reduced two-channel coordinates into the unrestricted DL basis."""
    transform=np.zeros((len(full_names),len(reduced_names)))
    for j,name in enumerate(reduced_names):
        if name in full_names:transform[full_names.index(name),j]=1;continue
        match=re.fullmatch(r"(temp_endpoint_difference|temp_growth_sum)(:C\(sector_id\)\[T\..+\])?",name)
        assert match,name
        suffix=match[2] or ""
        if match[1]=="temp_endpoint_difference":
            transform[full_names.index("temp_l0"+suffix),j]=1
            transform[full_names.index(f"temp_l{h}"+suffix),j]=-1
        else:
            for lag in range(h):transform[full_names.index(f"temp_l{lag}"+suffix),j]=1
    return transform


def verify_lineage(manifest):
    paths={**manifest["protected_sources_sha256"],**manifest["raw_output_sources_sha256"],
        manifest["producer_source"]:manifest["producer_source_sha256"],
        manifest["history_alignment_protocol"]:manifest["history_alignment_protocol_sha256"],
        manifest["policy_source"]:manifest["policy_source_sha256"]}
    assert manifest["implementation_version"]=="raw_history_v2" and len(manifest["raw_output_sources_sha256"])==8
    for relative,digest in paths.items():assert independent.sha256(ROOT/relative)==digest,("Protected/source file changed",relative)
    receipts={}
    for filename in ["rank_correction_manifest.json","two_channel_span_manifest.json"]:
        value=json.loads((OUT/filename).read_text());receipts[filename]=value
        assert independent.sha256(ROOT/value["source"])==value["source_sha256"]
        for field in ("protected_native_inputs_sha256","frozen_inputs_sha256"):
            for relative,digest in value.get(field,{}).items():assert independent.sha256(ROOT/relative)==digest,(filename,relative)
        for relative,digest in value["design_archives_sha256"].items():assert independent.sha256(ROOT/relative)==digest,(filename,relative)
    rank=receipts["rank_correction_manifest.json"]
    assert independent.sha256(ROOT/rank["native_stage_tests"])==rank["native_stage_tests_sha256"]
    assert independent.sha256(ROOT/rank["corrected_tests"])==rank["corrected_tests_sha256"]
    span=receipts["two_channel_span_manifest.json"]
    assert independent.sha256(ROOT/span["output"])==span["output_sha256"]
    return paths,receipts


def verify_native(expected):
    path=OUT/"pyfixest/fit_log.csv";log=read(path)
    receipt=json.loads((OUT/"pyfixest/estimator_manifest.json").read_text())
    assert len(log)==len(expected)==2361 and log.api_call_id.nunique()==2361
    assert receipt["native_fits_recorded"]==2361 and receipt["fit_log_sha256"]==independent.sha256(path)
    lookup=log.set_index(["label","outcome"]);assert lookup.index.is_unique
    assert set(lookup.index)=={(row["label"],row["outcome"]) for row in expected}
    for row in expected:
        native=lookup.loc[(row["label"],row["outcome"])]
        for field,value in row.items():
            if field not in {"label","outcome"}:assert native[field]==value,("Native exact identity",row["label"],field)
    assert log.native_class.eq("Feols").all() and log.native_module.str.startswith("pyfixest").all()
    assert log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all() and log.native_residuals_sha256.eq(log.residuals_sha256).all()
    for field in ["native_mapped_coefficient_max_abs_error","native_residual_max_abs_error","native_response_max_abs_error","native_outcome_max_abs_error","native_design_max_abs_error"]:
        assert log[field].eq(0).all()
    assert log.native_design_basis.eq("orthonormal SVD preconditioned full LSDV").all()
    assert (log.native_design_orthogonality_max_abs_error<1e-10).all()
    return {"fits":len(log),"api_calls":log.api_call_id.nunique(),"fit_log_sha256":independent.sha256(path)}


def verify_nesting(specs):
    records=[]
    for row in specs[specs.specification.eq("two_channel")].itertuples():
        full_id=row.spec_id.replace("__two_channel","__distributed")
        with np.load(OUT/"designs"/f"{full_id}.npz",allow_pickle=False) as full,np.load(OUT/"designs"/f"{row.spec_id}.npz",allow_pickle=False) as reduced:
            assert np.array_equal(full["time"],reduced["time"]) and np.array_equal(full["units"],reduced["units"]) and np.array_equal(full["Y"],reduced["Y"])
            mapping=nesting_embedding(list(full["coef_names"]),list(reduced["coef_names"]),int(row.horizon))
            nested_x=full["X"]@mapping
            independent.check(nested_x,reduced["X"],"Exact two-channel design embedding in unrestricted DL",atol=2e-10,rtol=2e-12)
            nested=qr_fit(nested_x,full["Y"],list(reduced["coef_names"]))
            independent.check(nested["beta"],reduced["coefficients"],"Constrained DL / two-channel native coefficient equivalence",atol=2e-7)
            independent.check(nested["residual"],reduced["residuals"],"Constrained DL / two-channel residual equivalence",atol=2e-8)
            records.append({"spec_id":row.spec_id,"distributed_spec_id":full_id,"unrestricted_parameters":full["X"].shape[1],
                "constrained_parameters":reduced["X"].shape[1],"independent_model_restrictions":full["X"].shape[1]-reduced["X"].shape[1],
                "interpretation":"No-AR two-channel model is a constrained reparameterization of distributed weather; covariance is recomputed using its own residuals and K"})
    assert len(records)==192
    return records


def _verify_literature_sensitivity():
    independent.ERRORS.clear();GEOMETRY.clear()
    manifest=json.loads((OUT/"manifest.json").read_text())
    assert manifest["implementation_version"]=="raw_history_v2" and manifest["native_fits"]==2361
    protected,postprocessing=verify_lineage(manifest)
    policy=json.loads((OUT/"policy.json").read_text());assert policy==json.loads(POLICY.read_text())
    samples,history,raw_hashes,parent_audit=build_history_samples(policy)
    assert raw_hashes==manifest["raw_output_sources_sha256"]
    specs=read(OUT/"specifications.csv")
    assert len(specs)==2361 and specs.spec_id.is_unique and specs.status.eq("estimated").all()
    expected=set()
    for (context,outcome,variant),sample in samples.items():
        sectors=sorted(sample.sector_id.unique())
        for specification in ["benchmark","distributed","dynamic"]+(["two_channel"] if outcome!="y_short" else []):
            expected.add((context,outcome,variant,"common",specification,"pooled"))
            if len(sectors)>1:expected.add((context,outcome,variant,"H1",specification,"pooled"))
        if len(sectors)>1:
            expected.update((context,outcome,variant,"separate","distributed",sid) for sid in sectors)
    columns=["context","outcome","sample_variant","structure","specification","unit_id"]
    assert set(specs[columns].itertuples(index=False,name=None))==expected
    keep=set(columns+["spec_id","sector_id","term","coefficient","std_error","p_value","ci_lower","ci_upper","variance",
        "covariance","hac_lags","inference_df","n_clusters","main_inference","r_squared","finite_sample_correction",
        "n_obs","n_periods","n_parameters","mean_y","sd_y","effect","lag","channel","effect_units","temp_main",
        "interaction_delta","delta_is_normalization","var_temp_main","var_interaction_delta","cov_temp_main_delta",
        "component","estimated","normalization","inference_status","degenerate","hc0_reference_variance","zero_variance_threshold",
        "test","restrictions","covariance_rank","uncapped_covariance_rank","structural_rank_bound","covariance_condition",
        "hc0_covariance_scale","zero_covariance_threshold","rank_cutoff","status","wald_chi2","f_statistic","numerator_df",
        "denominator_df","joint_test_fragile","fragility_reason","time_residual_df","rank_reassessment_version","interpretation"])
    tables={name:read(OUT/f"{name}.csv",keep) for name in ["coefficients","temperature_effects","interaction_components","channel_estimates","tests","two_channel_span_tests"]}
    groups={name:{str(key):value for key,value in frame.groupby("spec_id",sort=False)} for name,frame in tables.items()}
    native_expected=[];fit_records=[];test_records=[]
    counts={"coefficients":0,"residual_values":0,"full_covariance_cells":0,"temperature_effects":0,"interaction_components":0,"channel_estimates":0,"tests":0,"span_tests":0}
    with threadpool_limits(limits=1):
        for number,spec in enumerate(specs.itertuples(),1):
            context,outcome,variant=spec.context,spec.outcome,spec.sample_variant
            sample=samples[context,outcome,variant]
            if spec.structure=="separate":sample=sample[sample.sector_id.eq(str(spec.unit_id))].reset_index(drop=True)
            with np.load(OUT/"designs"/f"{spec.spec_id}.npz",allow_pickle=False) as archive:
                units=archive["units"].astype(str);times=archive["time"].astype(int)
                assert set(zip(sample.sector_id,sample.time))==set(zip(units,times)) and len(sample)==len(units)
                sample=sample.set_index(["sector_id","time"]).loc[list(zip(units,times))].reset_index()
                period,h=("month",12) if spec.frequency=="monthly" else ("quarter",4)
                assert sample[period].astype(str).tolist()==archive["periods"].astype(str).tolist()
                names=list(archive["coef_names"].astype(str));x,controls,features=expected_design(sample,spec.specification,spec.structure,h,names)
                y=sample[outcome].to_numpy(float)
                independent.check(archive["X"],x,"Every reconstructed logical design cell",atol=2e-10,rtol=2e-12)
                independent.check(archive["Y"],y,"Every saved native response cell",atol=0.,rtol=0.)
                fit=qr_fit(x,y,names);n,k,t=fit["n"],fit["k"],sample.time.nunique();sectors=sorted(sample.sector_id.unique())
                independent.check(archive["coefficients"],fit["beta"],"Every native coefficient including nuisance",atol=2e-7,rtol=2e-7)
                independent.check(archive["residuals"],fit["residual"],"Every native residual",atol=2e-8,rtol=1e-8)
                for field,value in [("n_obs",n),("n_periods",t),("n_parameters",k),("design_rank",k),("residual_df",n-k),("n_sectors",len(sectors))]:
                    assert int(getattr(spec,field))==value,(spec.spec_id,field,getattr(spec,field),value)
                first_unit=x[sample.sector_id.eq(sectors[0])]
                assert spec.time_design_rank==np.linalg.matrix_rank(first_unit) and spec.time_residual_df==t-spec.time_design_rank
                assert json.loads(spec.controls)==controls and json.loads(spec.temperature_terms)==features
                assert spec.start==min(sample[period]) and spec.end==max(sample[period])
                dates=pd.PeriodIndex(sample[period],freq="M" if period=="month" else "Q")
                assert spec.calendar_gaps==dates.max().ordinal-dates.min().ordinal+1-t
                independent.check([spec.mean_y,spec.sd_y],[y.mean(),y.std(ddof=1)],"Exact-sample growth mean and SD",atol=3e-10,rtol=2e-12)
                assert "Not cleared" in spec.stationarity_status
                if spec.specification=="dynamic":assert "AR" in spec.coefficient_sum_interpretation or "dynamic" in spec.coefficient_sum_interpretation
                native_expected.append({"label":"Literature sensitivity "+spec.spec_id,"outcome":outcome,"n_obs":n,"n_parameters":k,
                    "design_sha256":independent.array_sha256(archive["X"]),"outcome_sha256":independent.array_sha256(archive["Y"]),
                    "coefficients_sha256":independent.array_sha256(archive["coefficients"]),"residuals_sha256":independent.array_sha256(archive["residuals"]),
                    "coefficient_names_sha256":hashlib.sha256(json.dumps(names,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
                coefficient_rows=groups["coefficients"][spec.spec_id].set_index("term")
                assert len(coefficient_rows)==k and set(coefficient_rows.index)==set(names)
                independent.check(coefficient_rows.loc[names,"coefficient"].to_numpy(),fit["beta"],"All exported native coefficients",atol=2e-7)
                counts["coefficients"]+=k;counts["residual_values"]+=n
                selected=archive["selected_positions"].astype(int)
                assert selected.tolist()==list(range(k-len(controls)-len(features)*(len(sectors) if spec.structure=="H1" else 1),k))
                individual,methods=full_covariances(fit,sample,h,spec.structure)
                hc0=individual[:,selected].T@individual[:,selected]
                independent.check(archive["hc0_selected"],hc0,"Every selected unscaled HC0 reference",atol=1e-7,rtol=2e-7)
                profiles=profile_contrasts(names,features,sectors,spec.structure,h,spec.specification)
                assert len(methods)==len(archive["covariance_labels"])
                for band_number,(method,bandwidth,full_v,correction,degrees) in enumerate(methods):
                    assert str(archive["covariance_labels"][band_number])==method and int(archive["bandwidths"][band_number])==bandwidth
                    independent.check(archive[f"covariance_{band_number}"],full_v,"Every full nuisance-inclusive covariance cell",atol=2e-7,rtol=2e-7)
                    independent.check(archive["corrections"][band_number],correction,"Finite sample covariance multiplier",atol=1e-12,rtol=1e-12)
                    assert archive["inference_df"][band_number]==degrees
                    counts["full_covariance_cells"]+=k*k
                    # Work in selected coordinates for hypotheses: the full VCV
                    # was already checked, while this avoids cancellation from
                    # multiplying thousands of explicit zero nuisance weights.
                    v=full_v[np.ix_(selected,selected)];beta=fit["beta"][selected]
                    bound=len(sectors)-1 if method=="Sector cluster" else t-1
                    point_rows=groups["temperature_effects"][spec.spec_id]
                    point_rows=point_rows[point_rows.covariance.eq(method)&point_rows.hac_lags.eq(bandwidth)]
                    test_rows=groups["tests"][spec.spec_id]
                    test_rows=test_rows[test_rows.covariance.eq(method)&test_rows.hac_lags.eq(bandwidth)]
                    main=method=="Sector cluster" if spec.structure=="common" and len(sectors)>1 else bandwidth==h
                    assert point_rows.main_inference.eq(main).all()
                    independent.check(point_rows.r_squared,np.full(len(point_rows),fit["r_squared"]),"Centered full-model R squared",atol=5e-10)
                    cumulative_rows=[];basis_rows=[]
                    for sid,(lag_full,basis_full) in profiles.items():
                        lag=lag_full[:,selected];basis=basis_full[:,selected]
                        lag_beta=lag@beta;lag_v=lag@v@lag.T;lag_hc0=lag@hc0@lag.T
                        unit=sample if sid=="pooled" else sample[sample.sector_id.eq(sid)]
                        observed=point_rows[point_rows.sector_id.eq(sid)]
                        expected_lags=[0] if spec.specification=="benchmark" else list(range(h+1))
                        assert len(observed)==len(expected_lags)+1
                        for position,lag_number in enumerate(expected_lags+[-1]):
                            r=lag[position] if lag_number!=-1 else lag.sum(axis=0)
                            row=observed[observed.lag.eq(lag_number)]
                            assert len(row)==1;row=row.iloc[0]
                            independent.inference(row,float(r@beta),float(r@v@r),degrees,float(r@hc0@r))
                            independent.check([row.mean_y,row.sd_y],[unit[outcome].mean(),unit[outcome].std(ddof=1)],"Profile exact outcome mean/SD",atol=3e-10,rtol=2e-12)
                            if spec.structure=="H1" and spec.specification!="two_channel" and lag_number>=0:
                                main_pos=names.index(f"temp_l{lag_number}");delta_pos=None if sid==sectors[0] else names.index(f"temp_l{lag_number}:C(sector_id)[T.{sid}]")
                                independent.check([row.temp_main,row.interaction_delta,row.var_temp_main,row.var_interaction_delta,row.cov_temp_main_delta],
                                    [fit["beta"][main_pos],0. if delta_pos is None else fit["beta"][delta_pos],full_v[main_pos,main_pos],0. if delta_pos is None else full_v[delta_pos,delta_pos],0. if delta_pos is None else full_v[main_pos,delta_pos]],
                                    "Lag baseline-plus-interaction point and covariance",atol=2e-7)
                            counts["temperature_effects"]+=1
                        if spec.specification=="two_channel":
                            channel=groups["channel_estimates"][spec.spec_id]
                            channel=channel[channel.covariance.eq(method)&channel.hac_lags.eq(bandwidth)&channel.sector_id.eq(sid)].set_index("channel")
                            assert set(channel.index)==set(features) and len(channel)==2
                            for index,feature in enumerate(features):
                                r=basis[index];independent.inference(channel.loc[feature],float(r@beta),float(r@v@r),degrees,float(r@hc0@r));counts["channel_estimates"]+=1
                            operators={"All temperature-lag coefficients zero":np.eye(2),"Cumulative direct temperature response zero":np.array([[0.,h]]),
                                "Exact contemporaneous static-output-level restriction":np.array([[0.,1.]]),
                                "Equal one-period growth responses over YoY accumulation window":np.array([[1.,0.]])}
                            base_beta=basis@beta;base_v=basis@v@basis.T;base_hc0=basis@hc0@basis.T
                        else:
                            operators=restriction_operators(len(expected_lags),outcome)
                            base_beta,base_v,base_hc0=lag_beta,lag_v,lag_hc0
                        for name,restriction in operators.items():
                            row=test_rows[test_rows.sector_id.eq(sid)&test_rows.test.eq(name)]
                            assert len(row)==1
                            available=verify_test(row.iloc[0],base_beta,base_v,base_hc0,restriction,degrees,bound)
                            test_records.append({"spec_id":spec.spec_id,"sector_id":sid,"test":name,"covariance":method,"hac_lags":bandwidth,"available":available})
                            counts["tests"]+=1
                        if spec.specification in {"distributed","dynamic"} and outcome!="y_short" and spec.structure!="separate":
                            span_rows=groups["two_channel_span_tests"][spec.spec_id]
                            row=span_rows[span_rows.covariance.eq(method)&span_rows.hac_lags.eq(bandwidth)&span_rows.sector_id.eq(sid)]
                            assert len(row)==1;eye=np.eye(h+1)
                            restriction=np.vstack([eye[2:h]-eye[1],eye[0]+eye[h]-eye[1]])
                            assert restriction.shape==(h-1,h+1)
                            verify_test(row.iloc[0],lag_beta,lag_v,lag_hc0,restriction,degrees,bound)
                            counts["span_tests"]+=1
                        cumulative_rows.append(lag.sum(axis=0));basis_rows.append(basis if spec.specification=="two_channel" else lag)
                    if spec.structure=="H1":
                        components=groups["interaction_components"][spec.spec_id]
                        components=components[components.covariance.eq(method)&components.hac_lags.eq(bandwidth)]
                        assert len(components)==len(features)*(len(sectors)+1)
                        for feature_index,feature in enumerate(features):
                            lag_label=feature_index if spec.specification!="two_channel" else -2-feature_index
                            for index,sid in enumerate(sectors):
                                term=feature if index==0 else f"{feature}:C(sector_id)[T.{sid}]"
                                row=components[components.term.eq(term)&components.estimated.map(independent._truth)]
                                assert len(row)==1 and row.lag.iloc[0]==lag_label
                                position=names.index(term);j=list(selected).index(position)
                                independent.inference(row.iloc[0],fit["beta"][position],v[j,j],degrees,hc0[j,j]);counts["interaction_components"]+=1
                            zero=components[components.lag.eq(lag_label)&~components.estimated.map(independent._truth)]
                            assert len(zero)==1 and zero.coefficient.eq(0).all() and zero[["std_error","p_value","ci_lower","ci_upper"]].isna().all().all()
                            counts["interaction_components"]+=1
                        cumulative=np.vstack(cumulative_rows);difference=np.eye(len(sectors))[1:]-np.eye(len(sectors))[[0]]
                        global_tests={"All sector cumulative direct temperature responses equal":difference@cumulative,
                                      "All sector temperature-lag responses jointly zero":np.vstack(basis_rows)}
                        for name,r in global_tests.items():
                            row=test_rows[test_rows.sector_id.eq("all")&test_rows.test.eq(name)];assert len(row)==1
                            available=verify_test(row.iloc[0],beta,v,hc0,r,degrees,bound)
                            test_records.append({"spec_id":spec.spec_id,"sector_id":"all","test":name,"covariance":method,"hac_lags":bandwidth,"available":available})
                            counts["tests"]+=1
                fit_records.append({"spec_id":spec.spec_id,"context":context,"outcome":outcome,"sample_variant":variant,
                    "structure":spec.structure,"specification":spec.specification,"n_obs":n,"n_periods":t,"n_parameters":k})
            if number%75==0:print(f"Literature independent QR / full-VCV audit: {number}/2361 native fits",flush=True)
    for filename in ["coefficients","temperature_effects","interaction_components","channel_estimates","tests"]:
        assert counts[filename]==len(tables[filename]),("Exported coverage",filename,counts[filename],len(tables[filename]))
    assert counts["span_tests"]==len(tables["two_channel_span_tests"])==4104
    nesting=verify_nesting(specs)
    native=verify_native(native_expected)
    lineage_after,postprocessing_after=verify_lineage(manifest);assert protected==lineage_after and postprocessing==postprocessing_after
    rank=postprocessing["rank_correction_manifest.json"]
    original=read(ROOT/rank["native_stage_tests"])
    corrected=read(OUT/"tests.csv")
    changed=original.status.ne(corrected.status)
    assert int(changed.sum())==rank["statuses_changed"]==32
    assert int((changed&corrected.status.eq("available")).sum())==rank["newly_available"]==32
    assert int(original.covariance_rank.ne(corrected.covariance_rank).sum())==rank["rank_fields_changed"]==172
    report={"status":"passed","implementation_version":"raw_history_v2","contexts":20,"samples":len(samples),"native_fits":len(specs),
        "counts":counts,"raw_output_source_hashes":raw_hashes,"parent_raw_source_audit":parent_audit,"native_receipt":native,
        "rank_reassessment":{"corrected_rank_rows":172,"newly_available_tests":32,"native_fits_unchanged":True,
            "universal_calendar_score_bound":"T−1; repeated sectors do not add independent weather dates","unit_cluster_score_bound":"G−1"},
        "postprocessing_receipts":{name:independent.sha256(OUT/name) for name in postprocessing},
        "producer_source_sha256":manifest["producer_source_sha256"],"verifier_source_sha256":independent.sha256(Path(__file__)),
        "numerical_checks":independent.ERRORS,"scope":"Independent full raw SA/NSA calendar histories, raw NEA weather, target-support alignment, strict output-support event masks, all named design cells, full QR coefficients/residuals, every nuisance-inclusive CR1/calendar-Bartlett covariance, baseline+interaction lag/channel/cumulative effects, direct rank-safe restrictions, supplemental two-channel span tests, constrained-design nesting and native provenance. Not paper replication, causal/physical mechanism identification, stationarity certification, dynamic-FE bias correction or unconditional long-run multipliers.",
        "limits":"Many-restriction and ill-conditioned available Wald tests remain exploratory; numerical equality does not validate their finite-sample reference distribution. Shared-AR slopes are a restriction, and separate-series robustness covers distributed weather only."}
    pd.DataFrame(fit_records).to_csv(OUT/"verification_fits.csv",index=False)
    pd.DataFrame(history).to_csv(OUT/"verification_history.csv",index=False)
    pd.DataFrame(test_records).to_csv(OUT/"verification_tests.csv",index=False)
    pd.DataFrame(nesting).to_csv(OUT/"verification_nesting.csv",index=False)
    return report


def verify_literature_sensitivity():
    """Fail-closed receipt: a stale green result cannot survive a failed replay."""
    path=OUT/"verification.json"
    initial={"status":"running","verifier_source_sha256":independent.sha256(Path(__file__))}
    path.write_text(json.dumps(initial,indent=2)+"\n")
    try:
        report=_verify_literature_sensitivity()
    except BaseException as error:
        path.write_text(json.dumps({**initial,"status":"failed","error_type":type(error).__name__,"error":str(error)},indent=2)+"\n")
        raise
    temporary=path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");os.replace(temporary,path)
    print(f"Literature independent verification passed: 2361 native fits, 180 calendars, {report['counts']['full_covariance_cells']:,} covariance cells and all restrictions.",flush=True)
    return report


if __name__=="__main__":verify_literature_sensitivity()
