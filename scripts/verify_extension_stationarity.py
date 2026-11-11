"""Independent exact-calendar and ADF/KPSS audit for the new extensions.

No extension producer or production diagnostic routine is imported. The saved
regression bases are reconstructed from their frozen samples; test statistics,
AIC lag search, Bartlett variances and automatic KPSS bandwidths use the prior
independent mathematical audit, not statsmodels' ADF/KPSS functions.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for _key in ("OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_key,"1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import verify_crisis_sensitivity as numerical
import verify_stationarity_audit as math_audit

OUT=ROOT/"outputs/extension_stationarity"
SOURCES={
    "core":"outputs/crisis_sensitivity/specifications.csv",
    "full":"outputs/complete_va_controls/specifications.csv",
    "annual":"outputs/crisis_sensitivity/annual/specs.csv",
    "literature":"outputs/literature_sensitivity/specifications.csv",
    "annual_literature":"outputs/literature_sensitivity/annual/specifications.csv",
    "seasonal":"outputs/literature_sensitivity/annual_seasonal/specifications.csv",
    "full_controls":"outputs/literature_sensitivity/full_controls/specifications.csv"}
NPZ_CACHE: OrderedDict=OrderedDict()
SAMPLE_CACHE: OrderedDict=OrderedDict()


def read(path):
    return pd.read_csv(path,dtype={"sector_id":str,"unit_id":str,"unit":str,"reference_sector_id":str},
                       float_precision="round_trip",low_memory=False)


def sample_at(path):
    if path not in SAMPLE_CACHE:
        SAMPLE_CACHE[path]=read(ROOT/path)
        if len(SAMPLE_CACHE)>32:SAMPLE_CACHE.popitem(last=False)
    else:SAMPLE_CACHE.move_to_end(path)
    return SAMPLE_CACHE[path].copy()


def archive_at(path):
    if path not in NPZ_CACHE:
        with np.load(ROOT/path,allow_pickle=False) as archive:
            NPZ_CACHE[path]={key:archive[key] for key in archive.files if key in {
                "X","Y","names","coef_names","outcomes","time","periods","units","sector_id"}}
        if len(NPZ_CACHE)>8:NPZ_CACHE.popitem(last=False)
    else:NPZ_CACHE.move_to_end(path)
    return NPZ_CACHE[path]


def variables(row,branch):
    if branch=="annual":return json.loads(row.weather_terms)
    if branch in {"core","full"}:return json.loads(row.controls)
    return list(dict.fromkeys(json.loads(row.temperature_terms)+json.loads(row.controls)))


def design_path(row,branch):
    value=getattr(row,"design_npz",np.nan)
    if isinstance(value,str) and value:return value
    if branch in {"core","full"}:return str(Path(SOURCES[branch]).parent/"designs"/f"{row.fit_group_id}.npz")
    if branch=="literature":return f"outputs/literature_sensitivity/designs/{row.spec_id}.npz"
    return None


def actual_bases(row,branch,outcome):
    """Reconstruct actual main stochastic bases without testing FE interactions."""
    sample=sample_at(row.sample_csv)
    model=getattr(row,"structure",getattr(row,"model","common"))
    if model=="separate":
        sid=str(getattr(row,"unit_id",getattr(row,"sector_id","")))
        sample=sample[sample.sector_id.eq(sid)].reset_index(drop=True)
    freq="Y" if row.frequency=="annual" else "M" if row.frequency=="monthly" else "Q"
    period="year" if freq=="Y" else "month" if freq=="M" else "quarter"
    design=design_path(row,branch)
    archive=archive_at(design) if design is not None else None
    if archive is not None:
        units=archive.get("units",archive.get("sector_id")).astype(str)
        if freq=="Y":
            keys=list(zip(units,archive["time"].astype(int)))
            sample=sample.set_index(["sector_id","year"]).loc[keys].reset_index()
        else:
            keys=list(zip(units,archive["time"].astype(int)))
            sample=sample.set_index(["sector_id","time"]).loc[keys].reset_index()
        names=list(archive.get("coef_names",archive.get("names")).astype(str))
        y=archive["Y"]
        if y.ndim==2:y=y[:,list(archive["outcomes"].astype(str)).index(outcome)]
        numerical.check(y,sample[outcome].to_numpy(float),"Exact saved diagnostic Y basis",atol=3e-10,rtol=2e-12)
    else:
        units=sample.sector_id.to_numpy(str);names=[];y=sample[outcome].to_numpy(float)
    values={outcome:y}
    for term in variables(row,branch):
        if term=="temp_endpoint_difference":
            horizon=12 if freq=="M" else 4
            expected=sample.temp_l0.to_numpy()-sample[f"temp_l{horizon}"].to_numpy()
        elif term=="temp_growth_sum":
            horizon=12 if freq=="M" else 4
            expected=np.add.reduce(sample[[f"temp_l{lag}" for lag in range(horizon)]].to_numpy(float),axis=1)
        else:expected=sample[term].to_numpy(float)
        if branch in {"core","full","literature","seasonal","full_controls"}:expected=expected-expected.mean()
        if archive is not None:
            actual=archive["X"][:,names.index(term)]
            numerical.check(actual,expected,"Exact transformed diagnostic X basis",atol=2e-10,rtol=2e-12)
            values[term]=actual
        else:values[term]=expected
    dates=pd.PeriodIndex(sample[period].astype(str),freq=freq)
    assert len(sample)==len(units) and not pd.MultiIndex.from_arrays([units,dates]).duplicated().any()
    return sample,dates,units,values,design


def expected_conventions(variable,role,outcome,frequency):
    is_growth=role=="Y" and "level" not in outcome
    deterministic="c" if is_growth or variable.startswith(("rain","ar_")) or variable=="temp_endpoint_difference" else "ct"
    seasonal=not frequency.startswith("Y") and ((role=="Y" and outcome.endswith("_nsa")) or
        (role=="X" and (not variable.startswith("ar_") or outcome.endswith("_nsa"))))
    return deterministic,seasonal


def manual_test(row,values,det,h,legacy):
    if row.test=="ADF":
        cap=min(h if legacy else 2*h,len(values)//2-(2 if det=="ct" else 1)-1)
        result=math_audit.manual_adf(values,int(row.lags),det,cap)
        assert row.lags==result["aic_lag"] and row.lag_limit==cap and bool(row.lag_binding)==(row.lags==cap)
        assert int(row.nobs)==result["nobs"]
        numerical.check(row.aic,result["aic"],"Independent ADF lag-search AIC",atol=2e-6,rtol=2e-9)
    else:
        result=math_audit.manual_kpss(values,int(row.lags),det)
        auto=result["auto_lags"]
        assert int(row.auto_lags)==auto and int(row.lags)==(min(auto,h) if legacy else auto)
        assert row.p_relation==result["p_relation"]
    numerical.check([row.statistic,row.p_value,row.critical_5pct],
        [result["statistic"],result["p_value"],result["critical_5pct"]],"Independent ADF/KPSS statistic, reference p and critical value",atol=2e-7,rtol=2e-8)
    return result


def signature(dates,values,frequency,det,seasonal):
    policy={"frequency":frequency,"deterministic":det,
            "seasonal_rule":"season_offsets_with_trend_retained" if seasonal else "none"}
    prefix=json.dumps(policy,sort_keys=True,separators=(',',':')).encode()
    calendar=dates.asi8.astype('<i8').tobytes()
    raw=np.asarray(values,dtype='<f8').copy();raw[np.isnan(raw)]=np.nan
    digest=hashlib.sha256(prefix+b'\0'+calendar+raw.tobytes()).hexdigest()
    return 'es_'+digest[:24],digest,hashlib.sha256(calendar).hexdigest(),hashlib.sha256(raw.tobytes()).hexdigest()


def verify_diagnostic(diagnostic,series,tests,dates,values,frequency,det,seasonal):
    order=np.argsort(dates.asi8);dates=dates[order];values=np.asarray(values,float)[order]
    ident,digest,calendar_hash,raw_hash=signature(dates,values,frequency,det,seasonal)
    assert diagnostic.diagnostic_id==ident and diagnostic.input_sha256==digest
    assert diagnostic.calendar_sha256==calendar_hash and diagnostic.raw_values_sha256==raw_hash
    n=len(values);gaps=dates[-1].ordinal-dates[0].ordinal+1-n if n else 0
    h=12 if frequency=='M' else 4 if frequency=='Q' else 1
    minimum=60 if frequency=='M' else 20
    assert diagnostic.frequency==frequency and diagnostic.deterministic==det
    assert diagnostic.seasonal_rule==('season_offsets_with_trend_retained' if seasonal else 'none')
    assert diagnostic.n_periods==n and diagnostic.minimum_periods==minimum and diagnostic.frequency_horizon==h
    assert diagnostic.calendar_gaps==gaps and diagnostic.missing_values==int((~np.isfinite(values)).sum())
    if n:assert diagnostic.start==str(dates[0]) and diagnostic.end==str(dates[-1])
    assert len(series)==n and series.ordinal.tolist()==dates.asi8.tolist() and series.period.tolist()==dates.astype(str).tolist()
    numerical.check(series.raw_value.to_numpy(),values,"All exact diagnostic raw values",atol=0.,rtol=0.)
    missing=not np.isfinite(values).all()
    state='unavailable-gaps' if gaps else 'unavailable-missing' if missing else 'unavailable-empty' if not n else 'unavailable-short' if n<minimum else 'unavailable-constant' if np.ptp(values)==0 else None
    adjusted,component=(None,None)
    if state is None:
        adjusted,component=math_audit.independently_adjust(values,dates) if seasonal else (values,values*0)
        if np.ptp(adjusted)==0:state='unavailable-constant'
    if state is not None:
        assert diagnostic.state==state and diagnostic.test_status=='not_run'
        assert tests.empty and pd.isna(diagnostic.test_values_sha256)
        assert series[['test_value','seasonal_component']].isna().all().all()
        assert pd.isna(diagnostic.primary_state) and pd.isna(diagnostic.legacy_state)
        return False
    numerical.check(series.test_value.to_numpy(),adjusted,"Independently season-adjusted diagnostic values",atol=3e-9,rtol=3e-11)
    numerical.check(series.seasonal_component.to_numpy(),component,"Independently estimated seasonal offsets",atol=3e-9,rtol=3e-11)
    # The saved exact tested vector is independently checked numerically; its
    # hash binds the precise production arithmetic, including roundoff.
    saved_values=series.test_value.to_numpy(dtype='<f8')
    assert diagnostic.test_values_sha256==hashlib.sha256(saved_values.tobytes()).hexdigest()
    assert len(tests)==4 and set(tests.specification)=={'primary','legacy_lag_sensitivity'}
    states={}
    for label,legacy in [('primary',False),('legacy_lag_sensitivity',True)]:
        pair=tests[tests.specification.eq(label)]
        assert set(pair.test)=={'ADF','KPSS'} and len(pair)==2
        assert pair.status.eq('ok').all() and pair.deterministic.eq(det).all() and pair.frequency.eq(frequency).all()
        results={row.test:manual_test(row,adjusted,det,h,legacy) for row in pair.itertuples()}
        states[label]=math_audit.evidence_state(results['ADF']['p_value'],results['KPSS']['p_value'],det)
        stem='primary_' if not legacy else 'legacy_'
        numerical.check([getattr(diagnostic,stem+'adf_p'),getattr(diagnostic,stem+'kpss_p')],
            [results['ADF']['p_value'],results['KPSS']['p_value']],"Diagnostic evidence p summaries",atol=2e-7,rtol=2e-8)
        observed=pair.set_index('test')
        assert getattr(diagnostic,stem+'adf_lags')==observed.loc['ADF','lags']
        assert getattr(diagnostic,stem+'kpss_lags')==observed.loc['KPSS','lags']
        assert getattr(diagnostic,stem+'kpss_relation')==results['KPSS']['p_relation']
    final=states['primary'] if states['primary']==states['legacy_lag_sensitivity'] else 'sensitivity-dependent'
    assert diagnostic.primary_state==states['primary'] and diagnostic.legacy_state==states['legacy_lag_sensitivity']
    assert diagnostic.state==final and diagnostic.test_status=='completed'
    return True


def verify_source_hashes(summary):
    path=OUT/'source_hashes.json'
    assert numerical.sha256(path)==summary['source_hashes_sha256']
    hashes=json.loads(path.read_text())
    for name,digest in hashes.items():assert numerical.sha256(ROOT/name)==digest,('Frozen stationarity source changed',name)
    for name,digest in summary['outputs_sha256'].items():assert numerical.sha256(OUT/name)==digest,('Stationarity output changed',name)
    for source in SOURCES.values():assert source in hashes
    return hashes


def _verify_extension_stationarity():
    numerical.ERRORS.clear();SAMPLE_CACHE.clear();NPZ_CACHE.clear()
    summary=json.loads((OUT/'summary.json').read_text());assert summary['families']==SOURCES
    protected=verify_source_hashes(summary)
    models=read(OUT/'model_inventory.csv');links=read(OUT/'model_variable_coverage.csv')
    diagnostics=read(OUT/'diagnostics.csv');series=read(OUT/'series.csv');tests=read(OUT/'tests.csv')
    assert models.coverage_id.is_unique and diagnostics.diagnostic_id.is_unique
    assert not links.duplicated(['coverage_id','variable','role','unit_id']).any()
    assert not series.duplicated(['diagnostic_id','period']).any() and not tests.duplicated(['diagnostic_id','specification','test']).any()
    model_by=models.set_index('coverage_id');link_by={key:part for key,part in links.groupby('coverage_id',sort=False)}
    diag_by=diagnostics.set_index('diagnostic_id',drop=False)
    series_by={key:part for key,part in series.groupby('diagnostic_id',sort=False)}
    tests_by={key:part for key,part in tests.groupby('diagnostic_id',sort=False)}
    expected_models=set();expected_links=set();seen={};variables_by={};roles_by={};source_rows=0;records=[]
    completed=0;source_receipts={}
    for branch,path in SOURCES.items():
        source=read(ROOT/path);source_rows+=len(source)
        receipt=Path(path).parent/'verification.json'
        assert json.loads((ROOT/receipt).read_text())['status']=='passed'
        source_receipts[str(receipt)]=numerical.sha256(ROOT/receipt)
        for index,row in enumerate(source.itertuples(index=False)):
            outcomes=[row.outcome] if isinstance(row.outcome,str) else ['y_nsa','y_sa']
            for outcome in outcomes:
                identity=hashlib.sha256(json.dumps([branch,path,index,outcome],separators=(',',':')).encode()).hexdigest()[:24]
                coverage_id='em_'+identity;expected_models.add(coverage_id)
                observed=model_by.loc[coverage_id]
                assert observed.branch==branch and observed.source_spec_csv==path and observed.source_spec_row==index
                expected_id=row.spec_id if isinstance(row.spec_id,str) else str(row.fit_group_id)+'_'+outcome
                assert observed.spec_id==expected_id and observed.outcome==outcome and observed.source_status==row.status
                sample,dates,units,bases,design=actual_bases(row,branch,outcome)
                frequency='Y-DEC' if row.frequency=='annual' else 'M' if row.frequency=='monthly' else 'Q'
                assert observed.frequency==frequency and observed.context==row.context and observed.sample_variant==row.sample_variant
                assert observed.sample_csv==row.sample_csv and observed.sample_sha256==numerical.sha256(ROOT/row.sample_csv)
                assert protected[row.sample_csv]==observed.sample_sha256
                if design:
                    assert observed.design_npz==design and observed.design_sha256==numerical.sha256(ROOT/design) and protected[design]==observed.design_sha256
                else:assert pd.isna(observed.design_npz) and pd.isna(observed.design_sha256)
                unit_list=sorted(set(units))
                assert observed.n_obs==len(sample) and observed.n_units==len(unit_list) and observed.n_periods==len(set(dates))
                terms=variables(row,branch);reported_terms=json.loads(observed.requested_basis_terms)
                assert len(reported_terms)==len(terms) and set(reported_terms)==set(terms)
                assert bool(observed.growth_outcome)==(outcome not in ['y_log_level','y_raw_level'])
                actual_links=link_by[coverage_id]
                expected=[]
                for unit in unit_list:
                    keep=units==unit;expected.append((outcome,'Y',unit,[unit],dates[keep],bases[outcome][keep]))
                for term in terms:
                    array=bases[term]
                    separated=[(unit,dates[units==unit],array[units==unit]) for unit in unit_list]
                    shared=len(separated)>1 and all(np.array_equal(date.asi8,separated[0][1].asi8) and np.array_equal(value,separated[0][2],equal_nan=True) for _,date,value in separated[1:])
                    for unit,date,value in separated[:1] if shared else separated:
                        expected.append((term,'X','common' if shared else unit,unit_list if shared else [unit],date,value))
                assert observed.n_variable_links==len(expected)==len(actual_links)
                for variable,role,unit,applies,date,value in expected:
                    selected=actual_links.loc[actual_links.variable.eq(variable)&actual_links.role.eq(role)&actual_links.unit_id.eq(unit)]
                    assert len(selected)==1,(branch,index,outcome,variable,role,unit)
                    link=selected.iloc[0]
                    assert json.loads(link.applies_to_units)==applies and link.source_status==row.status
                    assert link.source_spec_csv==path and link.source_spec_row==index and link.spec_id==expected_id
                    expected_links.add((coverage_id,variable,role,unit))
                    order=np.argsort(date.asi8);date=date[order];value=value[order]
                    det,seasonal=expected_conventions(variable,role,outcome,frequency)
                    ident,digest,_,_=signature(date,value,frequency,det,seasonal)
                    assert link.diagnostic_id==ident
                    if ident in seen:
                        old=seen[ident]
                        assert old==(frequency,det,seasonal,digest)
                    else:
                        seen[ident]=(frequency,det,seasonal,digest)
                        d=diag_by.loc[ident]
                        sg=series_by.get(ident,pd.DataFrame(columns=series.columns)).sort_values('ordinal')
                        tg=tests_by.get(ident,pd.DataFrame(columns=tests.columns))
                        completed+=verify_diagnostic(d,sg,tg,date,value,frequency,det,seasonal)
                        records.append({'diagnostic_id':ident,'frequency':frequency,'state':d.state,'test_status':d.test_status,
                            'n_periods':d.n_periods,'calendar_gaps':d.calendar_gaps,'first_branch':branch,'first_variable':variable})
                    variables_by.setdefault(ident,set()).add(variable);roles_by.setdefault(ident,set()).add(role)
        print(f'Extension stationarity independent inventory/test audit: {branch}; {len(seen):,} exact arrays checked',flush=True)
    assert set(model_by.index)==expected_models
    assert set(links[['coverage_id','variable','role','unit_id']].itertuples(index=False,name=None))==expected_links
    assert set(diag_by.index)==set(seen)==set(series_by)
    assert set(tests_by)==set(diagnostics.loc[diagnostics.test_status.eq('completed'),'diagnostic_id'])
    for ident in seen:
        d=diag_by.loc[ident]
        assert json.loads(d.variables)==sorted(variables_by[ident]) and json.loads(d.roles)==sorted(roles_by[ident])
        expected_label=math_audit.LABELS.get(d.state,{'unavailable-missing':'Not established · missing values','unavailable-constant':'Not established · constant series','unavailable-empty':'Not established · empty calendar'}.get(d.state))
        assert d.label==expected_label
    counts={'source_spec_rows':source_rows,'logical_model_outcomes':len(expected_models),'model_variable_links':len(expected_links),
        'distinct_diagnostics':len(seen),'completed_diagnostics':completed,'unavailable_diagnostics':int(diagnostics.test_status.eq('not_run').sum()),
        'error_diagnostics':int(diagnostics.test_status.eq('error').sum()),'test_rows':len(tests),'series_rows':len(series)}
    assert summary['counts']==counts and summary['diagnostic_states']==diagnostics.state.value_counts().to_dict()
    assert counts['error_diagnostics']==0 and len(tests)==4*completed
    assert verify_source_hashes(summary)==protected
    assert all(numerical.sha256(ROOT/path)==digest for path,digest in source_receipts.items())
    pd.DataFrame(records).to_csv(OUT/'verification_diagnostics.csv',index=False)
    return {'status':'passed','counts':counts,'diagnostic_states':summary['diagnostic_states'],'numerical_checks':numerical.ERRORS,
        'source_files_and_native_receipts_unchanged':len(protected),'independent_raw_preparation_receipts':source_receipts,
        'producer_source_sha256':protected['scripts/extension_stationarity.py'],'verifier_source_sha256':numerical.sha256(Path(__file__)),
        'scope':'All seven frozen extension inventories, exact actual Y and non-FE X bases reconstructed from samples and checked against archived matrices, full per-unit/shared coverage and dedup policies, uncompressed calendars, short/missing/constant guards, independent seasonal offsets, ADF QR/AIC/MacKinnon arithmetic and KPSS Bartlett/Hobijn bandwidth and reference interpolation, paired evidence states and provenance.',
        'limits':'Marginal ordinary zero-frequency diagnostics only. Passing this audit does not certify stationarity, causality, absence of structural or seasonal unit roots, cointegration, dynamic FE unbiasedness or small-sample validity. Gapped or short windows remain unresolved; no sample trimming or result-based model selection.'}


def verify_extension_stationarity():
    path=OUT/'verification.json';initial={'status':'running','verifier_source_sha256':numerical.sha256(Path(__file__))}
    path.write_text(json.dumps(initial,indent=2)+'\n')
    try:
        with threadpool_limits(limits=1):report=_verify_extension_stationarity()
    except BaseException as error:
        path.write_text(json.dumps({**initial,'status':'failed','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n');raise
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)
    print(f"Extension stationarity independent verification passed: {report['counts']['logical_model_outcomes']:,} model requests, {report['counts']['distinct_diagnostics']:,} arrays and {report['counts']['test_rows']:,} test rows.",flush=True)
    return report


if __name__=='__main__':verify_extension_stationarity()
