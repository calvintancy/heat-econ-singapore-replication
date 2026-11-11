"""Isolated annual-growth and zero-retaining PPML arrivals companions.

Frozen official counts, actual calendar histories, native PyFixest point fits,
declared calendar inference and explicit unavailable/alias/sample audit records.
No original arrivals/GDP outputs or monthly-companion files are rewritten.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
import re
import warnings
from datetime import timedelta
from pathlib import Path

for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(_key,'1')

import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.demeaners import LsmrDemeaner
from scipy.stats import f as f_dist,t as student_t
from threadpoolctl import threadpool_limits

import pyfixest_backend as native
from crisis_sensitivity import geometry
from literature_sensitivity import calendar_covariance
from stationarity_audit import test_pair,state
from stationarity_diagnostics import seasonal_adjust
from visitor_counts import weighted_residualize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/visitor_rigour'
DATA=ROOT/'data/processed/visitor_arrivals'
RAW=ROOT/'data/raw/visitor_arrivals'
PROTOCOL=ROOT/'notes/visitor_rigour_protocol.md'
POLICY=ROOT/'notes/crisis_sensitivity_policy.json'
BASE=['temp_c','rain_100mm','rh_pct']
FULL=BASE+['wind_mps','pm25_10ug']
CALENDAR=['lny_window_days','lny_lag1']
VARIANTS=['all_eligible','event_dates_excluded','strict_growth_window_clean']
SUPPLEMENT=ROOT/'replication/extensions/supplementary_inputs/calendar/T2007e.txt'
SUPPLEMENT_SHA='2eeec2ef316267d92c2ec93e44d2ccb3a889e02c408aecdcc696a2c350416673'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):v=v.item()
    return None if isinstance(v,float) and not np.isfinite(v) else v
def write_json(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_inputs():
    metadata=json.loads((RAW/'nsa_singstat_metadata.json').read_text())['Data']['records']
    catalog=pd.DataFrame(metadata['row'])[['seriesNo','rowText']].rename(columns={'seriesNo':'unit_id','rowText':'unit'})
    catalog.unit_id=catalog.unit_id.astype(str);assert catalog.unit_id.is_unique and catalog.unit.is_unique
    source=pd.read_csv(RAW/'nsa.csv');source.DataSeries=source.DataSeries.str.strip()
    wide=source.set_index('DataSeries').T;wide.index=pd.PeriodIndex(pd.to_datetime(wide.index,format='%Y%b'),freq='M')
    wide=wide.sort_index().apply(pd.to_numeric,errors='raise');wide.index.name='month'
    assert set(wide)==set(catalog.unit) and wide.notna().all().all() and wide.ge(0).all().all()
    frozen=pd.read_csv(DATA/'all_source_counts.csv',index_col='month');frozen.index=pd.PeriodIndex(frozen.index,freq='M')
    pd.testing.assert_frame_equal(wide,frozen,check_dtype=False,check_names=False)
    wide=wide.rename(columns=catalog.set_index('unit').unit_id.to_dict())
    assert len(wide)==225 and list(wide.index)==list(pd.period_range('2008-01','2026-09',freq='M'))
    ids=catalog.unit_id
    cuts={'aggregate':['1'],'region':ids[ids.str.count(r'\.').eq(1)&ids.ne('1.999')].tolist(),
          'country':ids[ids.str.count(r'\.').eq(2)&~ids.str.endswith('.999')].tolist()}
    residual=ids[ids.eq('1.999')|ids.str.endswith('.999')].tolist()
    cuts['complete10']=cuts['region']+['1.999'];cuts['complete52']=cuts['country']+residual
    assert [len(cuts[k]) for k in cuts]==[1,9,42,10,52] and len(set(cuts['complete52']))==52
    reconciliations=[]
    for cut in ['complete10','complete52']:
        delta=wide[cuts[cut]].sum(axis=1)-wide['1'];assert delta.eq(0).all()
        reconciliations.extend({'month':str(date),'identity':cut+'_sum_minus_total','difference':float(value)} for date,value in delta.items())
    for parent in cuts['region']:
        children=ids[ids.str.startswith(parent+'.')].tolist()
        delta=wide[children].sum(axis=1)-wide[parent];assert delta.eq(0).all()
        reconciliations.extend({'month':str(date),'identity':parent+'_children_minus_parent','difference':float(value)} for date,value in delta.items())
    weather=[]
    for file,old,new in [('temperature','mean_temp','temp_c'),('humidity','mean_rh','rh_pct'),('rainfall','total_rainfall','rain_mm')]:
        frame=pd.read_csv(ROOT/f'data/raw/{file}_monthly.csv',index_col='month').rename(columns={old:new})
        frame.index=pd.PeriodIndex(frame.index,freq='M');weather.append(frame[[new]])
    weather=pd.concat(weather,axis=1).reindex(wide.index);weather['rain_100mm']=weather.rain_mm/100
    controls=pd.read_csv(ROOT/'data/processed/extended_controls/monthly_controls.csv',index_col='month');controls.index=pd.PeriodIndex(controls.index,freq='M')
    weather=weather.join(controls[['wind_mps','pm25_10ug','eligible_90','wind_coverage','pm_coverage']])
    holidays=pd.read_csv(DATA/'lunar_new_year.csv');holidays['month']=pd.PeriodIndex(holidays.month,freq='M')
    events=pd.to_datetime(holidays.lunar_new_year_date.drop_duplicates()).dt.date.tolist()
    assert sha(SUPPLEMENT)==SUPPLEMENT_SHA
    text=SUPPLEMENT.read_bytes().decode('ascii',errors='replace')
    matches=re.findall(r'^(2007/\d\d/\d\d)\s+1st Lunar Month',text,flags=re.M|re.I)
    assert matches==['2007/02/18'];events=[pd.Timestamp(matches[0]).date()]+events
    event_days={day+timedelta(days=offset) for day in events for offset in range(-7,8)}
    calendar=pd.DataFrame({'month':pd.period_range('2007-12',max(wide.index.max(),holidays.month.max()),freq='M')})
    calendar['lny_window_days']=[sum(day.year==date.year and day.month==date.month for day in event_days) for date in calendar.month]
    calendar['lny_lag1']=calendar.lny_window_days.shift(1)
    saved=holidays.set_index('month').lny_window_days
    rebuilt=calendar.set_index('month').lny_window_days
    assert np.array_equal(rebuilt.reindex(saved.index).to_numpy(),saved.to_numpy()) and rebuilt.loc[pd.Period('2007-12')]==0
    weather=weather.join(calendar.set_index('month')[CALENDAR]);weather['time']=weather.index.asi8;weather['year']=weather.index.year;weather['season']=weather.index.month
    cached=pd.read_csv(DATA/'weather.csv',index_col='month');cached.index=pd.PeriodIndex(cached.index,freq='M')
    for term in FULL+['lny_window_days']:
        assert np.allclose(weather[term],cached[term],atol=1e-12,rtol=1e-12,equal_nan=True),term
    assert weather.lny_lag1.iloc[0]==0 and np.allclose(weather.lny_lag1.iloc[1:],cached.lny_lag1.iloc[1:])
    checks={'source_months':len(wide),'source_units':len(catalog),'cuts':{k:len(v) for k,v in cuts.items()},'partition_max_abs_residual':0,
        'LNY_old_monthly_days_verified':len(saved),'LNY_restored_previous_month':'2007-12','LNY_restored_model_month':'2008-01','LNY_restored_lag_days':0,
        'LNY_supplement_sha256':sha(SUPPLEMENT),'LNY_supplement_url':json.loads((SUPPLEMENT.parent/'sources.json').read_text())['url'],
        'LNY_rule':'15-day window -7..+7 inclusive from official2007 event and frozen2008–2026 event dates; lookup before arrival calendar cropping'}
    return wide,weather,catalog,cuts,pd.DataFrame(reconciliations),calendar,checks


def event_mask(dates,annual=False):
    policy=json.loads(POLICY.read_text());mask=np.zeros(len(dates),bool)
    for event in policy['events']:
        if annual:
            mask|=(np.asarray(dates,int)>=int(event['start'][:4]))&(np.asarray(dates,int)<=int(event['end'][:4]))
        else:mask|=(dates>=pd.Period(event['start'],freq='M'))&(dates<=pd.Period(event['end'],freq='M'))
    return mask


def rank(x):
    scales=np.linalg.norm(x,axis=0);s=np.linalg.svd(x/np.where(scales>0,scales,1),compute_uv=False)
    return int(np.sum(s>s.max()*max(x.shape)*np.finfo(float).eps)) if len(s) else 0


def point(beta,variance,reference,df,available=True):
    if not available:return {'coefficient':float(beta),'std_error':None,'variance':None,'p_value':None,'ci_lower':None,'ci_upper':None,'inference_status':'unavailable: nonpositive working temporal degrees of freedom'}
    threshold=1e-12*max(float(reference),0);assert variance>=-max(threshold,np.finfo(float).tiny)
    se=float(np.sqrt(max(variance,0)));degenerate=abs(variance)<=threshold or reference<=0 or df<=0
    critical=student_t.ppf(.975,df) if df>0 else np.nan
    return {'coefficient':float(beta),'variance':float(variance),'std_error':se,'ci_lower':None if degenerate else float(beta-critical*se),'ci_upper':None if degenerate else float(beta+critical*se),
        'p_value':None if degenerate else float(2*student_t.sf(abs(beta/se),df)),'inference_status':'unavailable: numerical variance degeneracy' if degenerate else 'available',
        'hc0_reference_variance':float(reference),'zero_variance_threshold':threshold}


def joint(beta,covariance,hc0,restriction,meta,name):
    q=len(restriction);bound=meta['n_units']-1 if meta['covariance']=='Origin cluster' else meta['n_periods']-1
    base={**meta,'test':name,'restrictions':q,'structural_rank_bound':bound,'calibration_status':'Not established; nominal/asymptotic diagnostic, not calibrated headline inference'}
    if covariance is None:return {**base,'status':'unavailable: working temporal degrees of freedom','p_value':None}
    effect=restriction@beta;v=restriction@covariance@restriction.T;v=(v+v.T)/2;s=np.linalg.svd(v,compute_uv=False)
    scale=float(np.linalg.norm(restriction@hc0@restriction.T,2));cutoff=max(s[0]*1e-10,scale*1e-12);r=min(int(np.sum(s>cutoff)),max(bound,0))
    available=r==q and s[0]>scale*1e-12
    value=float(effect@np.linalg.solve(v,effect))/q if available else None
    return {**base,'status':'computable: calibration not established' if available else 'unavailable: singular restriction covariance','covariance_rank':r,'covariance_condition':float(np.linalg.cond(v)),
        'hc0_covariance_scale':scale,'rank_cutoff':cutoff,'f_statistic':value,'numerator_df':q,'denominator_df':meta['inference_df'],
        'p_value':float(f_dist.sf(max(value,0),q,meta['inference_df'])) if available else None,'fragile':q>=.8*max(bound,1) or meta['time_residual_df']<10 or np.linalg.cond(v)>1e8}


def annual_data(wide,weather):
    count_rows=[];weather_rows=[]
    for year,group in wide.groupby(wide.index.year):
        if len(group)!=12:continue
        for unit,value in group.sum(axis=0).items():count_rows.append({'year':int(year),'unit_id':unit,'arrivals':float(value)})
        w=weather.loc[group.index];days=w.index.days_in_month.to_numpy();row={'year':int(year),'months':len(w),'eligible_90':bool(w.eligible_90.eq(True).all())}
        for term in ['temp_c','rh_pct','wind_mps','pm25_10ug']:row[term]=float(np.average(w[term],weights=days)) if w[term].notna().all() else np.nan
        row['rain_100mm']=float(w.rain_100mm.sum()) if w.rain_100mm.notna().all() else np.nan;weather_rows.append(row)
    levels=pd.DataFrame(count_rows).sort_values(['year','unit_id']).reset_index(drop=True)
    lookup=levels.set_index(['unit_id','year']).arrivals
    levels['previous_arrivals']=lookup.reindex(pd.MultiIndex.from_arrays([levels.unit_id,levels.year-1])).to_numpy()
    valid=levels.arrivals.gt(0)&levels.previous_arrivals.gt(0)
    levels['y_annual_nsa']=np.nan;levels.loc[valid,'y_annual_nsa']=100*np.log(levels.loc[valid,'arrivals']/levels.loc[valid,'previous_arrivals'])
    return levels,pd.DataFrame(weather_rows)


def annual_design(frame,terms,model):
    units=sorted(frame.unit_id.unique());columns={f'origin[{unit}]':frame.unit_id.eq(unit).to_numpy(float) for unit in units}
    columns['trend']=(frame.year-frame.year.mean()).to_numpy()
    for term in terms[1:]:columns[term]=(frame[term]-frame[term].mean()).to_numpy()
    temperature=(frame.temp_c-frame.temp_c.mean()).to_numpy();columns['temp_c']=temperature
    components=[len(columns)-1]
    if model=='H1':
        for unit in units[1:]:columns[f'temp_c:C(unit_id)[T.{unit}]']=temperature*frame.unit_id.eq(unit).to_numpy();components.append(len(columns)-1)
    return np.column_stack(list(columns.values())),list(columns),units,np.asarray(components)


def annual_models(wide,weather,catalog,cuts,folder,preflight):
    folder.mkdir(parents=True,exist_ok=True)
    for name in ['samples','designs']:(folder/name).mkdir(exist_ok=True)
    levels,w=annual_data(wide,weather);levels.to_csv(folder/'annual_counts.csv',index=False);w.to_csv(folder/'annual_weather.csv',index=False)
    books={name:[] for name in ['specifications','coefficients','estimates','control_estimates','interaction_components','slope_covariances','joint_tests','time_design_ranks','calendar_masks']}
    names=catalog.set_index('unit_id').unit.to_dict();ledger={}
    selected=set().union(*(set(cuts[key]) for key in ['aggregate','region','country']))
    shared=levels[levels.unit_id.isin(selected)].groupby('year').y_annual_nsa.apply(lambda values:np.isfinite(values).all())
    shared_years=set(shared.index[shared])
    previous=native.LOG_DIR,list(native._FIT_LOG),native._API_COUNTER,native._RUN_STARTED
    if not preflight:native.LOG_DIR=folder/'pyfixest';native.reset_fit_log()
    try:
        for context in ['aggregate','region','country']:
            parent=levels[levels.unit_id.isin(cuts[context])].merge(w,on='year',validate='many_to_one')
            finite_y=parent.groupby('year').y_annual_nsa.apply(lambda s:np.isfinite(s).all());raw_growth_periods=int(finite_y.sum())
            parent=parent[parent.year.isin(set(finite_y.index[finite_y])&shared_years)].copy()
            for control_set,terms in [('long',BASE),('full',FULL)]:
                finite=parent.groupby('year')[terms].apply(lambda d:np.isfinite(d.to_numpy()).all());eligible=parent.year.isin(finite.index[finite])
                if control_set=='full':eligible&=parent.eligible_90.eq(True)
                maximum=parent[eligible].copy()
                for variant in VARIANTS:
                    mask=event_mask(maximum.year,True)
                    if variant=='strict_growth_window_clean':mask|=event_mask(maximum.year-1,True)
                    sample=maximum if variant=='all_eligible' else maximum.loc[~mask]
                    sample=sample.sort_values(['year','unit_id']).reset_index(drop=True);sample['time']=sample.year
                    books['calendar_masks'].extend({'panel':context,'control_set':control_set,'sample_variant':variant,'year':int(year),'current_event':bool(event_mask([year],True)[0]),
                        'prior_year_event':bool(event_mask([year-1],True)[0]),'retained':year in set(sample.year),'output_support_start':int(year)-1} for year in sorted(maximum.year.unique()))
                    for model in ['common']+(['H1'] if len(cuts[context])>1 else []):
                        spec_id='__'.join(['annual',context,control_set,variant,model]);sample_path=folder/'samples'/f'{spec_id}.csv';sample.to_csv(sample_path,index=False)
                        x,xnames,units,positions=annual_design(sample,terms,model);geo=geometry(x);n,k=x.shape;t=sample.year.nunique()
                        local=[rank(x[sample.unit_id.eq(unit)]) for unit in units];span=rank(np.column_stack([x[sample.unit_id.eq(unit)] for unit in units]))
                        point_status=geo['status'];inference_available=t>max(local)
                        meta={'spec_id':spec_id,'family':'annual','context':context,'panel':context,'control_set':control_set,'sample_variant':variant,'model':model,'model_spec':'current','outcome':'y_annual_nsa','frequency':'annual',
                            'status':point_status,'point_status':point_status,'time_correction_available':inference_available,'primary_calendar_inference_available':point_status=='estimated' and inference_available,
                            'primary_inference_status':'available (approximate)' if point_status=='estimated' and inference_available else 'unavailable: point design or nonpositive working temporal degrees of freedom',
                            'n_obs':n,'n_units':len(units),'n_sectors':len(units),'n_origins':len(units),'n_periods':t,'n_parameters':k,'design_rank':geo['design_rank'],'residual_df':n-geo['design_rank'],
                            'time_design_rank':max(local),'time_residual_df':t-max(local),'full_temporal_span_rank':span,'start':int(sample.year.min()),'end':int(sample.year.max()),
                            'calendar_gaps':int(sample.year.max()-sample.year.min()+1-t),'sample_csv':rel(sample_path),'sample_sha256':sha(sample_path),'design_npz':rel(folder/'designs'/f'{spec_id}.npz'),
                            'controls':json.dumps(terms[1:]),'terms':json.dumps(terms),'temperature_terms':'["temp_c"]','mean_y':float(sample.y_annual_nsa.mean()),'sd_y':float(sample.y_annual_nsa.std()),
                            'source_complete_level_years':levels.year.nunique(),'source_growth_missing_periods':levels.year.nunique()-raw_growth_periods,'raw_growth_eligible_periods':raw_growth_periods,
                            'shared_cut_alignment_loss_periods':raw_growth_periods-parent.year.nunique(),'parent_growth_periods':parent.year.nunique(),
                            'control_raw_missing_periods':parent.year.nunique()-int(finite.sum()),'control_coverage_excluded_periods':int(finite.sum())-maximum.year.nunique(),
                            'control_eligible_periods':maximum.year.nunique(),'crisis_removed_periods':maximum.year.nunique()-t,
                            'year_fixed_effects':False,'origin_fixed_effects':True,'linear_trend':True,'annual_LNY_included':False,'reference_unit_id':units[0] if model=='H1' else '',
                            'estimator':'PyFixest feols','estimator_version':native.ESTIMATOR_VERSION,'native_label':'Visitor rigour '+spec_id,
                            'formula':'y_annual_nsa ~ temp_c'+(' * C(unit_id)' if model=='H1' else '')+' + '+' + '.join(terms[1:])+' + trend + C(unit_id)',
                            'units':'100 log points annual NSA count growth per1°C annual mean temperature','finite_sample_rule':'Calendar T/(T−max local rank); approximate t(T−1); no stationarity or finite-sample calibration certificate'}
                        books['specifications'].append(meta)
                        for unit,r in zip(units,local):books['time_design_ranks'].append({'spec_id':spec_id,'unit_id':unit,'local_design_rank':r,'n_periods':t,'full_temporal_span_rank':span})
                        if preflight:continue
                        archive={'X':x,'Y':sample.y_annual_nsa.to_numpy(),'names':np.asarray(xnames,str),'units':sample.unit_id.to_numpy(str),'time':sample.year.to_numpy(),'component_positions':positions,'terms':np.asarray(terms,str),'local_ranks':np.asarray(local),'full_temporal_span_rank':np.asarray(span)}
                        beta=residual=influence=hc0=None
                        if point_status=='estimated':
                            fitted=native.fit_many_matrix(sample[['y_annual_nsa']].to_numpy(),x,xnames,meta['native_label'],outcome_names=['y_annual_nsa'],source_formula=meta['formula'])[0]
                            beta,residual=fitted.params.to_numpy(),fitted.resid;influence=(np.asarray(fitted.native_model.within_data.design)@fitted.coefficient_transform.T)*residual[:,None];hc0=influence.T@influence
                            meta.update(r_squared=float(fitted.rsquared),fit_id=native._FIT_LOG[-1]['fit_id'])
                            archive.update(beta=beta,residual=residual,hc0=hc0,influence=influence)
                            books['coefficients'].extend({'spec_id':spec_id,'term':term,'coefficient':float(value)} for term,value in zip(xnames,beta))
                        methods=[('Calendar HAC',band,t-1) for band in [1,2]]+([('Origin cluster',0,len(units)-1)] if model=='common' and len(units)>1 else [])
                        factors=[];profiles=[]
                        for i,(method,band,df) in enumerate(methods):
                            v=None;factor=None
                            if beta is not None and (method=='Origin cluster' or inference_available):
                                factor=len(units)/(len(units)-1)*(n-1)/(n-k) if method=='Origin cluster' else t/(t-max(local))
                                if method=='Origin cluster':
                                    scores=np.vstack([influence[sample.unit_id.eq(unit)].sum(axis=0) for unit in units]);v=scores.T@scores*factor
                                else:v=calendar_covariance(influence,sample.year.to_numpy(),band,factor)
                                archive[f'covariance_{i}']=v
                            factors.append(np.nan if factor is None else factor)
                            bandmeta={**meta,'covariance':method,'hac_lags':band,'inference_df':df,'main_inference':method=='Calendar HAC' and band==1,'finite_sample_correction':factor}
                            profiles=[]
                            for number,unit in enumerate(units if model=='H1' else ['pooled']):
                                r=np.zeros(k);r[positions[0]]=1
                                if model=='H1' and number:r[positions[number]]=1
                                profiles.append(r);data=sample if unit=='pooled' else sample[sample.unit_id.eq(unit)]
                                info={**bandmeta,'unit_id':unit,'unit':names.get(unit,'Common slope'),'sector_id':unit,'sector':names.get(unit,'Common slope'),'term':'temp_c','effect':'current','lag':0,'mean_y':float(data.y_annual_nsa.mean()),'sd_y':float(data.y_annual_nsa.std())}
                                values=point(float(r@beta),float(r@v@r) if v is not None else 0,float(r@hc0@r) if hc0 is not None else 0,df,v is not None) if beta is not None else {'coefficient':None,'std_error':None,'ci_lower':None,'ci_upper':None,'p_value':None,'inference_status':'unavailable: point design'}
                                if beta is not None:info.update(temperature_main=float(beta[positions[0]]),temperature_interaction=0. if model!='H1' or number==0 else float(beta[positions[number]]),reference_interaction_normalized=model=='H1' and number==0)
                                books['estimates'].append({**info,**values})
                            for term in terms[1:]:
                                j=xnames.index(term);value=point(beta[j],v[j,j] if v is not None else 0,hc0[j,j] if hc0 is not None else 0,df,v is not None) if beta is not None else {'coefficient':None,'std_error':None,'p_value':None,'ci_lower':None,'ci_upper':None,'inference_status':'unavailable: point design'}
                                books['control_estimates'].append({**bandmeta,'term':term,**value})
                            if model=='H1':
                                for number,unit in enumerate(units):
                                    j=positions[number];value=point(beta[j],v[j,j] if v is not None else 0,hc0[j,j] if hc0 is not None else 0,df,v is not None) if beta is not None else {'coefficient':None,'std_error':None,'p_value':None,'inference_status':'unavailable: point design'}
                                    books['interaction_components'].append({**bandmeta,'unit_id':unit,'term':xnames[j],'component':'base' if number==0 else 'interaction','estimated':beta is not None,'normalization':False,**value})
                                books['interaction_components'].append({**bandmeta,'unit_id':units[0],'term':f'temp_c:C(unit_id)[T.{units[0]}]','component':'interaction','estimated':False,'normalization':True,'coefficient':0.,'std_error':None,'p_value':None})
                                profile=np.vstack(profiles);cv=profile@v@profile.T if v is not None else None
                                if cv is not None:books['slope_covariances'].extend({'spec_id':spec_id,'covariance':method,'hac_lags':band,'unit_i':a,'unit_j':b,'covariance_value':float(cv[ii,jj])} for ii,a in enumerate(units) for jj,b in enumerate(units))
                                for name,r in [('All origin temperature slopes equal',profile[1:]-profile[[0]]),('All origin temperature slopes zero',profile)]:books['joint_tests'].append(joint(beta,v,hc0,r,bandmeta,name))
                        archive.update(covariance_labels=np.asarray([m[0] for m in methods],str),bandwidths=np.asarray([m[1] for m in methods]),inference_df=np.asarray([m[2] for m in methods]),corrections=np.asarray(factors))
                        np.savez_compressed(folder/'designs'/f'{spec_id}.npz',**archive)
        if not preflight:ledger=native.write_fit_log()
    finally:
        if not preflight:native.LOG_DIR,old,native._API_COUNTER,native._RUN_STARTED=previous;native._FIT_LOG[:]=old
    for name,rows in books.items():pd.DataFrame(rows).to_csv(folder/('preflight_specifications.csv' if preflight and name=='specifications' else name+'.csv'),index=False)
    if not preflight:
        effects=pd.DataFrame(books['estimates']);effects['p_value_holm_within_profile']=np.nan
        for _,group in effects[effects.model.eq('H1')].groupby(['spec_id','covariance','hac_lags'],sort=False):
            p=group.p_value.fillna(1).to_numpy();order=np.argsort(p);adjusted=np.empty(len(p));adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
            effects.loc[group.index,'p_value_holm_within_profile']=np.where(group.p_value.notna(),adjusted,np.nan)
        effects.to_csv(folder/'temperature_effects.csv',index=False);pd.DataFrame(books['joint_tests']).to_csv(folder/'tests.csv',index=False)
    return books,ledger


def time_geometry(frame,terms,annual=False):
    dates=np.sort(frame.time.unique());local=[];spans=[]
    for unit,part in frame.groupby('unit_id',sort=True):
        blocks=[pd.get_dummies(part.year,dtype=float).to_numpy()]
        if not annual:blocks.append(pd.get_dummies(part.season,dtype=float).to_numpy())
        else:blocks.append((part.year-part.year.mean()).to_numpy()[:,None])
        blocks.append(part[terms].to_numpy());x=np.column_stack(blocks);r=rank(x)
        local.append({'unit_id':unit,'n_periods':part.time.nunique(),'local_design_rank':r})
        embedded=np.zeros((len(dates),x.shape[1]));embedded[np.searchsorted(dates,part.time.to_numpy())]=x;spans.append(embedded)
    return local,rank(np.column_stack(spans))


def fixed_effect_rank(groups):
    a,b=groups.T;na,nb=int(a.max()+1),int(b.max()+1);parent=np.arange(na+nb)
    def find(node):
        while parent[node]!=node:parent[node]=parent[parent[node]];node=parent[node]
        return node
    for first,second in zip(a,b):
        ra,rb=find(int(first)),find(na+int(second));parent[ra]=rb
    components=len({find(i) for i in range(na+nb)})
    return na+nb-components


def ppml_models(wide,weather,catalog,cuts,folder,preflight):
    folder.mkdir(parents=True,exist_ok=True)
    for name in ['samples','input_samples','designs','pyfixest']:(folder/name).mkdir(exist_ok=True)
    books={name:[] for name in ['specifications','coefficients','temperature_effects','control_estimates','tests','time_design_ranks','dropped_observations','aliases','calendar_masks']}
    receipts=[];names=catalog.set_index('unit_id').unit.to_dict();fit_number=0
    for context,units in cuts.items():
        data=wide[units].rename_axis(index='month',columns='unit_id').stack().rename('arrivals').reset_index()
        data=data.merge(weather.reset_index(),on='month',validate='many_to_one');data['unit']=data.unit_id.map(names)
        data['unit_season']=(data.unit_id+'_'+data.season.astype(str)).astype('string')
        for control_set,weather_terms in [('long',BASE),('full',FULL)]:
            terms=weather_terms+CALENDAR;finite=np.isfinite(data[terms]).all(axis=1);raw_finite_periods=data.loc[finite,'time'].nunique()
            if control_set=='full':finite&=data.eligible_90.eq(True)
            maximum=data[finite].copy()
            for row in data.drop_duplicates('time').itertuples():
                valid=bool(np.isfinite([getattr(row,term) for term in terms]).all());quality=control_set!='full' or row.eligible_90==True
                at_event=bool(event_mask(pd.PeriodIndex([row.month],freq='M'))[0])
                books['calendar_masks'].append({'panel':context,'control_set':control_set,'month':str(row.month),'finite_current_controls':valid,'coverage_pass':quality,'current_event':at_event,
                    'retain_all':valid and quality,'retain_event':valid and quality and not at_event,'retain_strict':valid and quality and not at_event,'strict_alias_of_event':True})
            for variant in VARIANTS[:2]:
                frame=maximum if variant=='all_eligible' else maximum[~event_mask(pd.PeriodIndex(maximum.month,freq='M'))]
                frame=frame.sort_values(['time','unit_id']).reset_index(drop=True);frame['input_row']=np.arange(len(frame))
                spec_id='__'.join(['ppml',context,control_set,variant]);sample_path=folder/'samples'/f'{spec_id}.csv';input_path=folder/'input_samples'/f'{spec_id}.csv';frame.to_csv(input_path,index=False)
                local,span=time_geometry(frame,terms);t=frame.time.nunique();working_rank=max(r['local_design_rank'] for r in local)
                formula='arrivals ~ '+' + '.join(terms)+' | unit_season + year'
                meta={'spec_id':spec_id,'native_spec_id':spec_id,'family':'ppml','context':context,'panel':context,'control_set':control_set,'sample_variant':variant,'model':'common','model_spec':'current','outcome':'arrival_count','frequency':'monthly',
                    'status':'preflight' if preflight else 'pending','point_status':'preflight' if preflight else 'pending','n_obs':len(frame),'n_periods':t,'n_units':len(units),'n_sectors':len(units),
                    'time_design_rank':working_rank,'time_residual_df':t-working_rank,'full_temporal_span_rank':span,'start':str(frame.month.min()),'end':str(frame.month.max()),
                    'mean_y':float(frame.arrivals.mean()),'sd_y':float(frame.arrivals.std()),'input_sample_csv':rel(input_path),'input_sample_sha256':sha(input_path),'sample_csv':rel(sample_path),
                    'design_npz':rel(folder/'designs'/f'{spec_id}.npz'),'terms':json.dumps(terms),'controls':json.dumps(terms[1:]),'temperature_terms':'["temp_c"]','formula':formula,
                    'input_zero_cells':int(frame.arrivals.eq(0).sum()),'input_n_obs':len(frame),'input_n_units':len(units),'source_count_months':len(wide),'weather_eligible_periods':maximum.time.nunique(),
                    'control_raw_missing_periods':len(wide)-raw_finite_periods,'control_coverage_excluded_periods':raw_finite_periods-maximum.time.nunique(),'shared_cut_alignment_loss_periods':0,
                    'removed_for_control_or_coverage':len(wide)-maximum.time.nunique(),'removed_for_event_dates':maximum.time.nunique()-t,'growth_history_required':False,
                    'estimator':'PyFixest fepois','estimator_version':pf.__version__,'offset':None,'primary_inference':'Calendar HAC12/24, T/(T−max local rank), approximate t(T−1)',
                    'coefficient_units':'100 log expected-count points per1°C; raw_native_coefficient is log-mean semi-elasticity; exact0.1°C percent effect separately saved','alias_of':'','native_fit_executed':not preflight}
                input_groups=np.column_stack([pd.factorize(frame.unit_season)[0],pd.factorize(frame.year)[0]]);meta['n_parameters']=fixed_effect_rank(input_groups)+len(terms)
                meta.update(n_origins=len(units),time_correction_available=t>working_rank,primary_calendar_inference_available=t>working_rank)
                if preflight:
                    books['specifications'].append(meta);continue
                fit_number+=1;caught=[];fit=None
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        fit=pf.fepois(formula,frame,vcov='iid',fixef_rm='none',separation_check=['fe','ir'],iwls_tol=1e-12,iwls_maxiter=500,collin_tol=1e-12,
                            demeaner=LsmrDemeaner(fixef_atol=1e-12,fixef_btol=1e-12,fixef_maxiter=10000))
                    dropped=sorted(fit.sample_info.dropped_row_index);native_frame=frame.drop(index=dropped).copy().reset_index(drop=True)
                    for row in frame.loc[dropped].itertuples():books['dropped_observations'].append({'spec_id':spec_id,'input_row':row.input_row,'month':str(row.month),'unit_id':row.unit_id,'arrivals':row.arrivals,'zero_count':row.arrivals==0})
                    assert fit.sample_info.n_rows==len(native_frame)
                    beta=fit.coef().reindex(terms).to_numpy();residual=np.asarray(fit.resid(type='response')).reshape(-1);mu=np.asarray(fit.fitted_values.response).reshape(-1);eta=np.asarray(fit.fitted_values.link).reshape(-1)
                    assert np.array_equal(np.asarray(fit.model_matrix.dependent).reshape(-1),native_frame.arrivals.to_numpy())
                    assert np.allclose(mu+residual,native_frame.arrivals.to_numpy(),atol=1e-8,rtol=1e-12) and np.allclose(np.exp(eta),mu,atol=1e-6,rtol=1e-10)
                    assert np.isfinite(mu).all() and (mu>0).all()
                    dropped_terms=[term for term in terms if term not in fit.coef().index]
                    groups=np.column_stack([pd.factorize(native_frame.unit_season)[0],pd.factorize(native_frame.year)[0]])
                    local,span=time_geometry(native_frame,terms);t=native_frame.time.nunique();working_rank=max(r['local_design_rank'] for r in local)
                    status='estimated' if bool(fit.convergence) and not dropped_terms and np.isfinite(beta).all() else 'unavailable: nonconverged or requested native regressor dropped'
                    meta.update(status=status,point_status=status,n_obs=len(native_frame),n_units=native_frame.unit_id.nunique(),n_sectors=native_frame.unit_id.nunique(),n_periods=t,
                        time_design_rank=working_rank,time_residual_df=t-working_rank,full_temporal_span_rank=span,
                        n_parameters=fixed_effect_rank(groups)+len(fit.coef()),native_converged=bool(fit.convergence),native_class=type(fit).__name__,native_module=type(fit).__module__,
                        dropped_rows=len(dropped),dropped_rows_by_stage=json.dumps(dataclasses.asdict(fit.sample_info.dropped_by_stage)),native_dropped_requested_terms=json.dumps(dropped_terms),
                        native_dropped_coef_names=json.dumps(fit.collinearity.dropped_coef_names),retained_zero_cells=int(native_frame.arrivals.eq(0).sum()),
                        dropped_zero_cells=int(frame.loc[dropped].arrivals.eq(0).sum()),dropped_units=json.dumps(sorted(set(frame.unit_id)-set(native_frame.unit_id))),
                        native_warnings=' | '.join(str(item.message) for item in caught),mean_y=float(native_frame.arrivals.mean()),sd_y=float(native_frame.arrivals.std()),
                        zero_retention_status='All input zeros retained' if not dropped else 'Native sample changed; see dropped-observation audit')
                    meta.update(n_origins=meta['n_units'],design_rank=meta['n_parameters'],time_correction_available=t>working_rank,primary_calendar_inference_available=status=='estimated' and t>working_rank)
                    native_frame.to_csv(sample_path,index=False);meta['sample_sha256']=sha(sample_path)
                    for item in local:books['time_design_ranks'].append({'spec_id':spec_id,**item,'full_temporal_span_rank':span})
                    archive={'X':native_frame[terms].to_numpy(),'Y':native_frame.arrivals.to_numpy(),'mu':mu,'eta':eta,'residual':residual,'beta':beta,'terms':np.asarray(terms,str),'names':np.asarray(terms,str),
                        'units':native_frame.unit_id.to_numpy(str),'months':native_frame.month.astype(str).to_numpy(str),'time':native_frame.time.to_numpy(),'groups':groups,'input_row':native_frame.input_row.to_numpy(),
                        'local_ranks':np.asarray([r['local_design_rank'] for r in local]),'full_temporal_span_rank':np.asarray(span)}
                    native_receipt={'fit_id':fit_number,'spec_id':spec_id,'api':'pyfixest.fepois','native_class':type(fit).__name__,'native_module':type(fit).__module__,'estimator_version':pf.__version__,'formula':formula,'n_obs':len(native_frame),'native_converged':bool(fit.convergence),
                        'response_hash_dtype':'float64','native_response_sha256':array_hash(np.asarray(fit.model_matrix.dependent,dtype=float).reshape(-1)),'response_sha256':array_hash(native_frame.arrivals.to_numpy(dtype=float)),
                        'native_coefficients_sha256':array_hash(fit.coef().reindex(terms).to_numpy()),'native_mu_sha256':array_hash(mu),'native_residual_sha256':array_hash(residual),
                        'dropped_row_positions':json.dumps(dropped),'dropped_by_stage':json.dumps(dataclasses.asdict(fit.sample_info.dropped_by_stage)),'postprocessing_status':'pending'}
                    receipts.append(native_receipt);pd.DataFrame(receipts).to_csv(folder/'pyfixest/fit_log.csv',index=False)
                    np.savez_compressed(folder/'designs'/f'{spec_id}.npz',**archive)
                    if status=='estimated':
                        xr,iterations=weighted_residualize(native_frame[terms].to_numpy(),mu,groups.T)
                        information=xr.T@(mu[:,None]*xr);assert rank(information)==len(terms)
                        bread=np.linalg.inv(information);scores=xr*residual[:,None];influence=scores@bread
                        hc0=influence.T@influence;archive.update(xr=xr,bread=bread,information=information,scores=scores,influence=influence,hc0=hc0,weights=mu)
                        relative_score=float(np.max(np.abs(native_frame[terms].to_numpy().T@residual))/(native_frame.arrivals.sum()+1))
                        assert relative_score<1e-6;meta['relative_score_error']=relative_score;meta['weighted_projection_iterations']=iterations
                        books['coefficients'].extend({'spec_id':spec_id,'term':term,'coefficient':float(value),'estimator':'PyFixest fepois'} for term,value in zip(terms,beta))
                        methods=[('Calendar HAC',h,t-1) for h in [12,24]]+([('Origin cluster',0,meta['n_units']-1)] if meta['n_units']>1 else [])
                        factors=[]
                        for i,(method,band,df) in enumerate(methods):
                            v=None;factor=None
                            if method=='Origin cluster':
                                factor=meta['n_units']/(meta['n_units']-1)*(len(native_frame)-1)/(len(native_frame)-meta['n_parameters'])
                                origin_scores=np.vstack([influence[native_frame.unit_id.eq(unit)].sum(axis=0) for unit in sorted(native_frame.unit_id.unique())]);v=origin_scores.T@origin_scores*factor
                            else:
                                raw_v=calendar_covariance(influence,native_frame.time.to_numpy(),band,1.);archive[f'raw_calendar_covariance_{i}']=raw_v
                                if t>working_rank:factor=t/(t-working_rank);v=raw_v*factor
                            if v is not None:archive[f'covariance_{i}']=v
                            factors.append(np.nan if factor is None else factor)
                            bandmeta={**meta,'covariance':method,'hac_lags':band,'inference_df':df,'main_inference':method=='Calendar HAC' and band==12,'finite_sample_correction':factor,'unit_id':'pooled','unit':'Common expected-count slope','sector_id':'pooled','sector':'Common expected-count slope'}
                            for j,term in enumerate(terms):
                                value=point(100*beta[j],10000*v[j,j] if v is not None else 0,10000*hc0[j,j],df,v is not None)
                                result={**bandmeta,'term':term,'effect':'current','lag':0,'raw_native_coefficient':float(beta[j]),**value}
                                if term=='temp_c':
                                    result.update(effect_0p1c=10*float(beta[j]),exact_percent_0p1c=100*float(np.expm1(.1*beta[j])))
                                    books['temperature_effects'].append(result)
                                    r=np.eye(len(terms))[[j]];books['tests'].append({**joint(beta,v,hc0,r,bandmeta,'Temperature expected-count semi-elasticity zero'),'duplicate_of_pointwise_temperature_test':True})
                                else:books['control_estimates'].append(result)
                        archive.update(covariance_labels=np.asarray([m[0] for m in methods],str),bandwidths=np.asarray([m[1] for m in methods]),inference_df=np.asarray([m[2] for m in methods]),corrections=np.asarray(factors))
                    np.savez_compressed(folder/'designs'/f'{spec_id}.npz',**archive)
                    native_receipt['postprocessing_status']='complete';pd.DataFrame(receipts).to_csv(folder/'pyfixest/fit_log.csv',index=False)
                except Exception as error:
                    if fit is not None:raise
                    meta.update(status='unavailable: native PPML error',point_status='unavailable: native PPML error',reason=type(error).__name__+': '+str(error),native_warnings=' | '.join(str(item.message) for item in caught))
                    receipts.append({'fit_id':fit_number,'spec_id':spec_id,'api':'pyfixest.fepois','native_class':type(fit).__name__ if fit is not None else '', 'native_returned':fit is not None,'error':meta['reason'],'formula':formula,'estimator_version':pf.__version__})
                if meta['status']!='estimated':
                    for method,band,df in [('Calendar HAC',12,meta['n_periods']-1),('Calendar HAC',24,meta['n_periods']-1)]+([('Origin cluster',0,meta['n_units']-1)] if meta['n_units']>1 else []):
                        for term in terms:
                            result={**meta,'unit_id':'pooled','unit':'Common expected-count slope','sector_id':'pooled','sector':'Common expected-count slope','covariance':method,'hac_lags':band,'inference_df':df,
                                'main_inference':method=='Calendar HAC' and band==12,'term':term,'effect':'current','lag':0,'coefficient':None,'std_error':None,'p_value':None,'ci_lower':None,'ci_upper':None,
                                'inference_status':'unavailable: native point fit'}
                            books['temperature_effects' if term=='temp_c' else 'control_estimates'].append(result)
                books['specifications'].append(meta)
                print(f"Visitor PPML {context}/{control_set}/{variant}: {meta['status']}",flush=True)
    for meta in list(books['specifications']):
        if meta['sample_variant']!='event_dates_excluded':continue
        alias={**meta,'spec_id':meta['spec_id'].replace('event_dates_excluded','strict_growth_window_clean'),'sample_variant':'strict_growth_window_clean','alias_of':meta['spec_id'],'native_fit_executed':False}
        books['specifications'].append(alias);books['aliases'].append({'spec_id':alias['spec_id'],'native_spec_id':meta['spec_id'],'reason':'Count-level outcome has no growth base: strict output support is current date only, identical to event-date mask'})
        for table in ['temperature_effects','control_estimates','tests']:
            books[table].extend({**row,'spec_id':alias['spec_id'],'sample_variant':'strict_growth_window_clean','alias_of':meta['spec_id'],'native_fit_executed':False} for row in list(books[table]) if row['spec_id']==meta['spec_id'])
    for name,rows in books.items():
        frame=pd.DataFrame(rows)
        if not len(frame):frame=pd.DataFrame(columns=['spec_id','input_row','month','unit_id','arrivals','zero_count'] if name=='dropped_observations' else ['spec_id'])
        frame.to_csv(folder/('preflight_specifications.csv' if preflight and name=='specifications' else name+'.csv'),index=False)
    if not preflight:
        pd.DataFrame(receipts).to_csv(folder/'pyfixest/fit_log.csv',index=False)
        write_json(folder/'pyfixest/estimator_manifest.json',{'native_calls':fit_number,'native_models_returned':sum(bool(row.get('native_class')) for row in receipts),'fit_log_sha256':sha(folder/'pyfixest/fit_log.csv'),'estimator':'PyFixest fepois','version':pf.__version__,'aliases_are_not_fits':True})
    return books,receipts


def aggregation_checks(folder):
    specs=pd.read_csv(folder/'specifications.csv');rows=[]
    for cut in ['complete10','complete52']:
        for control in ['long','full']:
            for variant in VARIANTS[:2]:
                ids=['__'.join(['ppml',name,control,variant]) for name in ['aggregate',cut]]
                meta=[specs.loc[specs.spec_id.eq(ident)].iloc[0] for ident in ids]
                row={'panel':cut,'control_set':control,'sample_variant':variant,'aggregate_spec_id':ids[0],'component_spec_id':ids[1]}
                if any(m.status!='estimated' or int(m.dropped_rows)>0 for m in meta):
                    rows.append({**row,'status':'not comparable: native availability or retained sample differs'});continue
                with np.load(ROOT/meta[0].design_npz) as total,np.load(ROOT/meta[1].design_npz) as pieces:
                    periods=np.unique(pieces['time']);assert np.array_equal(periods,total['time']) and np.array_equal(total['terms'],pieces['terms'])
                    fitted=np.array([pieces['mu'][pieces['time']==date].sum() for date in periods])
                    point_error=float(np.max(np.abs(total['beta']-pieces['beta'])));mu_error=float(np.max(np.abs(total['mu']-fitted)))
                    assert np.allclose(total['beta'],pieces['beta'],atol=1e-6,rtol=1e-6),(row,'beta',point_error)
                    assert np.allclose(total['mu'],fitted,atol=1e-3,rtol=1e-7),(row,'fitted count',mu_error)
                    v_error=0.
                    for i in [0,1]:
                        assert np.isclose(total['corrections'][i],pieces['corrections'][i],atol=1e-12)
                        assert np.allclose(total[f'covariance_{i}'],pieces[f'covariance_{i}'],atol=1e-7,rtol=1e-6),(row,'HAC covariance',i)
                        v_error=max(v_error,float(np.max(np.abs(total[f'covariance_{i}']-pieces[f'covariance_{i}']))))
                    rows.append({**row,'status':'verified within native numerical tolerance','max_abs_shared_coefficient_difference':point_error,'max_abs_aggregated_fitted_count_difference':mu_error,
                        'max_abs_calendar_covariance_difference':v_error,'same_time_correction':True,'n_months':len(periods),
                        'interpretation':'Count-specific shared-X PPML likelihood factorization; not additivity of component log growth or an identity for selected nonexhaustive cuts'})
    pd.DataFrame(rows).to_csv(folder/'aggregation_checks.csv',index=False)
    return rows


def source_bindings():
    inputs=[RAW/'nsa.csv',RAW/'nsa_singstat_metadata.json',DATA/'all_source_counts.csv',DATA/'market_catalog.csv',DATA/'weather.csv',DATA/'lunar_new_year.csv',
        ROOT/'data/processed/extended_controls/monthly_controls.csv',SUPPLEMENT,SUPPLEMENT.parent/'sources.json',PROTOCOL,POLICY]
    inputs += [ROOT/f'data/raw/{name}_monthly.csv' for name in ['temperature','rainfall','humidity']]
    protected=[Path(__file__),ROOT/'scripts/pyfixest_backend.py',ROOT/'scripts/visitor_analysis.py',ROOT/'scripts/visitor_counts.py',ROOT/'scripts/crisis_sensitivity.py',ROOT/'scripts/literature_sensitivity.py',
        ROOT/'outputs/pyfixest/fit_log.csv',ROOT/'outputs/pyfixest/estimator_manifest.json']
    for path in (ROOT/'outputs/visitor_arrivals').rglob('*'):
        if path.is_file() and 'R-library' not in path.parts and path.suffix in ['.csv','.json','.npz']:protected.append(path)
    return {rel(p):sha(p) for p in inputs},{rel(p):sha(p) for p in protected}


def run_visitor_rigour_companions(*,preflight=False):
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/('companions_preflight.json' if preflight else 'companions_manifest.json')
    write_json(target,{'status':'running'})
    inputs,protected=source_bindings();wide,weather,catalog,cuts,reconciliation,calendar,checks=load_inputs()
    with threadpool_limits(limits=1):
        annual,annual_ledger=annual_models(wide,weather,catalog,cuts,OUT/'annual',preflight)
        ppml,ppml_ledger=ppml_models(wide,weather,catalog,cuts,OUT/'ppml',preflight)
    reconciliation.to_csv(OUT/'ppml/count_partition_reconciliation.csv',index=False);calendar.to_csv(OUT/'ppml/lunar_calendar_extended.csv',index=False)
    write_json(OUT/'annual/source_checks.json',checks);write_json(OUT/'ppml/source_checks.json',checks)
    aggregation=[] if preflight else aggregation_checks(OUT/'ppml')
    assert all(sha(ROOT/name)==value for name,value in inputs.items()),'Frozen analytical input changed'
    assert all(sha(ROOT/name)==value for name,value in protected.items()),'Preserved source, native ledger or original arrival result changed'
    manifest={'status':'preflight_only' if preflight else 'estimated_not_yet_independently_verified','source':rel(Path(__file__)),'source_sha256':sha(Path(__file__)),
        'annual_requested':len(annual['specifications']),'annual_native_fits':annual_ledger.get('native_fits_recorded',0),'annual_point_statuses':pd.Series([r['status'] for r in annual['specifications']]).value_counts().to_dict(),
        'ppml_requested':len(ppml['specifications']),'ppml_native_calls':len(ppml_ledger),'ppml_aliases':len(ppml['aliases']),'ppml_point_statuses':pd.Series([r['status'] for r in ppml['specifications']]).value_counts().to_dict(),
        'inputs_sha256':inputs,'protected_source_results_sha256':protected,'source_checks':checks,'aggregation_checks':aggregation,
        'annual_scope':'Current annual NSA count growth,complete12month counts/dayweighted meanweather/rain sum;unitFE+trend;fullall12monthly90%;all/event/current-prior-year clean masks;short annual history not stationarity cleared',
        'PPML_scope':'Zero-retaining count levels,nativefepois origin×month+year FE,current weather+LNY0/1,all/event andexplicit strict alias; native separation/drops/convergence examined, no pseudocount or growth-history mask',
        'inference':'Calendar HAC primary T/(T−max local rank),approx t(T−1),with full temporal span retained; originCR1 pooledcomparison only; no finite-sample calibration or causal-validity certificate',
        'stationarity':'Separate visitor_rigour_stationarity diagnostic appendix covers actual calendars; no stationarity-based model or sample selection'}
    write_json(target,manifest)
    if not preflight:
        for name in ['annual','ppml']:write_json(OUT/name/'manifest.json',{**manifest,'branch':name})
    print(json.dumps({key:manifest[key] for key in ['status','annual_requested','annual_native_fits','annual_point_statuses','ppml_requested','ppml_native_calls','ppml_aliases','ppml_point_statuses']},indent=2),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();run_visitor_rigour_companions(preflight=args.preflight)
