"""Independent raw-calendar, explicit-design and score audit of visitor rigour.

No production visitor preparation, fitting or covariance helper is imported.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from datetime import date,timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for _key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_key,'1')
import numpy as np
import pandas as pd
from scipy.linalg import qr,solve_triangular
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import verify_crisis_sensitivity as proof
import verification_artifacts as artifact_bindings
from verify_granular import _raw_extended
from verify_literature_sensitivity import calendar_meat,verify_test,profile_contrasts,nesting_embedding

OUT=ROOT/'outputs/visitor_rigour/monthly'
PANELS={'aggregate':1,'region':9,'country':42}
YS={'y_mom_nsa':(1,3),'y_yoy_nsa':(12,12)}
VARIANTS=['all_eligible','event_dates_excluded','strict_growth_window_clean']
HISTORIES=['own_current_history','matched_maximum_growth_history']
OTHER=['rh_pct','lny_window_days','lny_lag1'];ADDED=['wind_mps','pm25_10ug']
RANKS:OrderedDict=OrderedDict()


def read(path):return pd.read_csv(path,dtype={name:str for name in ['unit_id','sector_id','reference_unit_id','reference_sector_id']},float_precision='round_trip',low_memory=False)


def independent_inputs():
    """Rebuild all62 count rows, disjoint partitions, raw growth and lunar days."""
    folder=ROOT/'data/raw/visitor_arrivals';manifest=json.loads((folder/'manifest.json').read_text())
    hashes={}
    for filename,record in manifest.items():
        path=folder/filename;assert proof.sha256(path)==record['sha256'];hashes[str(path.relative_to(ROOT))]=record['sha256']
    metadata=json.loads((folder/'nsa_singstat_metadata.json').read_text())['Data']['records']
    catalog={str(row['seriesNo']):row['rowText'] for row in metadata['row']}
    original=pd.read_csv(folder/'nsa.csv');original.DataSeries=original.DataSeries.str.strip()
    assert original.DataSeries.is_unique and set(original.DataSeries)==set(catalog.values())
    named=original.set_index('DataSeries').T
    named.index=pd.PeriodIndex(pd.to_datetime(named.index,format='%Y%b'),freq='M');named=named.sort_index().astype(float)
    assert named.shape==(225,62) and named.index.equals(pd.period_range('2008-01','2026-09',freq='M'))
    assert named.size==metadata['total'] and np.isfinite(named).all().all() and (named>=0).all().all()
    assert np.equal(named,np.floor(named)).all().all()
    counts=named.rename(columns={name:sid for sid,name in catalog.items()})
    cuts={'aggregate':['1'],'region':sorted(sid for sid in catalog if re.fullmatch(r'1\.[1-9]',sid)),
          'country':sorted(sid for sid in catalog if re.fullmatch(r'1\.[1-9]\.\d+',sid) and not sid.endswith('.999'))}
    cuts['complete10']=sorted(sid for sid in catalog if sid.count('.')==1)
    cuts['complete52']=sorted(sid for sid in catalog if sid.count('.')==2 or sid=='1.999')
    assert {cut:len(ids) for cut,ids in cuts.items()}=={'aggregate':1,'region':9,'country':42,'complete10':10,'complete52':52}
    for cut in ['complete10','complete52']:
        # Integer accumulation independent of the source's DataFrame sum route.
        values=np.array([sum(int(counts.loc[period,sid]) for sid in cuts[cut]) for period in counts.index])
        proof.check(values,counts['1'].to_numpy(),'Complete arrival count partition '+cut,atol=0,rtol=0)
    for parent in cuts['region']:
        children=[sid for sid in catalog if sid.startswith(parent+'.')]
        values=np.array([sum(int(counts.loc[period,sid]) for sid in children) for period in counts.index])
        proof.check(values,counts[parent].to_numpy(),'Every raw arrival region-child identity',atol=0,rtol=0)
    frozen=read(ROOT/'data/processed/visitor_arrivals/all_source_counts.csv').set_index('month')
    assert frozen.index.tolist()==named.index.astype(str).tolist() and set(frozen.columns)==set(named.columns)
    proof.check(frozen[named.columns].to_numpy(),named.to_numpy(),'Frozen complete source counts',atol=0,rtol=0)
    growth={}
    for outcome,(distance,_) in YS.items():
        growth[outcome]={}
        for sid in catalog:
            series=counts[sid]
            growth[outcome][sid]={period:100*np.log(series.loc[period]/series.loc[period-distance])
                for period in series.index if period-distance in series.index and series.loc[period]>0 and series.loc[period-distance]>0}
    # Verify all20 HKO tables, including the separately frozen 2007 supplement.
    calfolder=folder/'calendar';cm=json.loads((calfolder/'manifest.json').read_text());anchors={};calendar_hashes={}
    paths=[(calfolder/f'T{year}e.txt',cm[f'T{year}e.txt']['sha256']) for year in range(2008,2027)]
    supplement=ROOT/'replication/extensions/supplementary_inputs/calendar/sources.json'
    supplement_info=json.loads(supplement.read_text());paths.append((ROOT/supplement_info['filename'],supplement_info['sha256']))
    for path,digest in paths:
        assert proof.sha256(path)==digest;calendar_hashes[str(path.relative_to(ROOT))]=digest
        content=path.read_bytes();all_dates=re.findall(rb'(?m)^(\d{4}/\d{1,2}/\d{1,2})\s+',content)
        year=int(path.name[1:5]);parsed=[date(*map(int,value.decode().split('/'))) for value in all_dates]
        assert len(parsed)==(date(year+1,1,1)-date(year,1,1)).days and len(set(parsed))==len(parsed)
        assert parsed==[date(year,1,1)+timedelta(days=i) for i in range(len(parsed))]
        first=re.findall(rb'(?mi)^(\d{4}/\d{1,2}/\d{1,2})\s+1st\s+Lunar\s+month\b',content)
        assert len(first)==1;anchors[year]=date(*map(int,first[0].decode().split('/')))
    assert anchors[2007]==date(2007,2,18)
    lunar={period:0 for period in pd.period_range('2007-01','2026-12',freq='M')}
    for anchor in anchors.values():
        for offset in range(-7,8):lunar[pd.Period(anchor+timedelta(days=offset),freq='M')]+=1
    old=read(ROOT/'data/processed/visitor_arrivals/lunar_new_year.csv')
    old_dates=pd.PeriodIndex(old.month,freq='M')
    assert old_dates.equals(pd.period_range('2008-01','2026-12',freq='M'))
    proof.check(old.lny_window_days.to_numpy(),[lunar[p] for p in old_dates],'All228 old LNY monthly window values unchanged',atol=0,rtol=0)
    assert all(str(anchors[int(row.year)])==row.lunar_new_year_date for row in old.itertuples())
    assert lunar[pd.Period('2007-12',freq='M')]==0
    weather={}
    for stem,column,new in [('temperature','mean_temp','temp_c'),('humidity','mean_rh','rh_pct'),('rainfall','total_rainfall','rain_100mm')]:
        path=ROOT/f'data/raw/{stem}_monthly.csv';table=pd.read_csv(path)
        hashes[str(path.relative_to(ROOT))]=proof.sha256(path)
        for row in table.to_dict('records'):
            period=pd.Period(row['month'],freq='M');weather.setdefault(period,{})[new]=float(row[column])/(100 if stem=='rainfall' else 1)
    extended,extended_audit=_raw_extended()
    for period,record in weather.items():
        record['lny_window_days']=lunar.get(period,np.nan);record['lny_lag1']=lunar.get(period-1,np.nan)
        record.update(extended['M'].get(period,{}))
    hashes.update(calendar_hashes);hashes[str(supplement.relative_to(ROOT))]=proof.sha256(supplement)
    audit={'source_months':225,'source_rows':62,'units_per_cut':{cut:len(ids) for cut,ids in cuts.items()},
        'zero_cells_per_cut':{cut:int(counts[ids].eq(0).sum().sum()) for cut,ids in cuts.items()},'lunar_raw_tables':20,
        'old_lunar_calendar_rows':len(old),'supplement_2007_new_year':str(anchors[2007]),'december_2007_lunar_window_days':0,'raw_hashes':hashes,'wind_pm':extended_audit}
    return counts,cuts,catalog,growth,weather,audit


def reconstruct_samples(folder,inputs):
    counts,cuts,catalog,growth,weather,audit=inputs;dates=counts.index
    policy=json.loads((folder/'policy.json').read_text());assert policy==json.loads((ROOT/'notes/crisis_sensitivity_policy.json').read_text())
    events=[(e['event_id'],set(pd.period_range(e['start'],e['end'],freq='M').asi8)) for e in policy['events']]
    inventories=read(folder/'sample_inventory.csv');masks=read(folder/'masks.csv');assert len(inventories)==72
    samples={};summary=[];all_masks=0
    finite=lambda values:all(np.isfinite(value) for value in values)
    for panel in PANELS:
        for outcome,(base_h,h) in YS.items():
            for controls in ['long','full']:
                for history in HISTORIES:
                    maximum=history==HISTORIES[1];support=h+base_h if maximum else 12
                    required_lags=range(h+1) if maximum else [0]
                    rows=[];stage=[]
                    for current in dates:
                        current_ok=finite(growth[outcome][sid].get(current,np.nan) for sid in cuts[panel])
                        valid_by={cut:finite(growth[outcome][sid].get(current-lag,np.nan) for sid in cuts[cut]
                            for lag in (range(h+1) if maximum else [0])) for cut in PANELS}
                        own=valid_by[panel]
                        weather_ok=finite(weather.get(current-lag,{}).get(term,np.nan) for term in ['temp_c','rain_100mm'] for lag in required_lags)
                        weather_ok &= finite(weather.get(current,{}).get(term,np.nan) for term in OTHER)
                        shared=all(valid_by.values())
                        added_ok=finite(weather.get(current,{}).get(term,np.nan) for term in ADDED) if controls=='full' else True
                        covered=bool(weather.get(current,{}).get('eligible_90',False)) if controls=='full' else True
                        eligible=own and weather_ok and shared and added_ok and covered
                        at=[key for key,values in events if current.ordinal in values]
                        required=set(range(current.ordinal-support,current.ordinal+1));over=[key for key,values in events if not required.isdisjoint(values)]
                        stage.append(dict(month=str(current),current_output_finite=current_ok,own_output_history_finite=own,weather_history_finite=weather_ok,
                            all_three_cuts_output_history_finite=shared,current_added_controls_finite=added_ok,current_control_coverage_pass=covered,
                            eligible_before_event_masks=eligible,event_ids_at_t=at,event_ids_in_output_support=over,retain_event_dates=not at,retain_strict=not over))
                    saved=masks.loc[masks.panel.eq(panel)&masks.outcome.eq(outcome)&masks.control_set.eq(controls)&masks.history.eq(history)].set_index('month')
                    assert set(saved.index)==set(dates.astype(str));saved=saved.loc[dates.astype(str)];all_masks+=len(saved)
                    for record in stage:
                        actual=saved.loc[record['month']]
                        for field,value in record.items():
                            if field=='month':continue
                            if isinstance(value,list):assert proof._event_list(actual[field])==value
                            else:assert proof._truth(actual[field])==value,(panel,outcome,controls,history,record['month'],field)
                        assert actual.time==pd.Period(record['month'],freq='M').ordinal and actual.output_support_horizon==support
                        assert actual.output_support_start==str(pd.Period(record['month'],freq='M')-support)
                    records=pd.DataFrame(stage)
                    own=records.own_output_history_finite;w=own&records.weather_history_finite;aligned=w&records.all_three_cuts_output_history_finite
                    ctrl=aligned&records.current_added_controls_finite;eligible=ctrl&records.current_control_coverage_pass
                    losses={'parent_n_periods':len(dates),'current_own_output_n_periods':int(records.current_output_finite.sum()),
                        'own_output_history_n_periods':int(own.sum()),'output_history_loss_periods':int((~own).sum()),
                        'weather_history_loss_periods':int(own.sum()-w.sum()),'shared_panel_alignment_loss_periods':int(w.sum()-aligned.sum()),
                        'control_raw_missing_loss_periods':int(aligned.sum()-ctrl.sum()),'control_coverage_loss_periods':int(ctrl.sum()-eligible.sum()),
                        'maximum_history_n_periods':int(eligible.sum()),'output_support_horizon':support}
                    for variant in VARIANTS:
                        keep=eligible if variant==VARIANTS[0] else eligible&records['retain_event_dates' if variant==VARIANTS[1] else 'retain_strict']
                        retained=dates[keep];sample_id='__'.join([panel,outcome,controls,history,variant]);path=folder/'samples'/f'{sample_id}.csv'
                        sample=read(path);assert len(sample)==len(retained)*PANELS[panel]
                        assert not sample.duplicated(['unit_id','month']).any()
                        assert set(sample.unit_id)==set(cuts[panel]) and set(sample.month)==set(retained.astype(str))
                        for unit,local in sample.groupby('unit_id',sort=False):
                            local=local.set_index('month').loc[retained.astype(str)]
                            proof.check(local.arrivals.to_numpy(),counts.loc[retained,unit].to_numpy(),'Monthly raw arrivals before transformations',atol=0,rtol=0)
                            for y in YS:proof.check(local[y].to_numpy(),[growth[y][unit].get(d,np.nan) for d in retained],'Complete raw calendar growth before masks',atol=3e-10,rtol=2e-12)
                            for lag in range(1,h+1):proof.check(local[f'ar_l{lag}'].to_numpy(),[growth[outcome][unit].get(d-lag,np.nan) for d in retained],'Raw growth-lag calendars before cropping',atol=3e-10,rtol=2e-12)
                            for lag in range(h+1):
                                for term,stem in [('temp_c','temp'),('rain_100mm','rain')]:
                                    proof.check(local[f'{stem}_l{lag}'].to_numpy(),[weather.get(d-lag,{}).get(term,np.nan) for d in retained],'Full uncropped raw weather lags',atol=3e-11,rtol=2e-12)
                            for term in OTHER+ADDED+['temp_c','rain_100mm']:
                                proof.check(local[term].to_numpy(),[weather.get(d,{}).get(term,np.nan) for d in retained],'Current weather/LNY/coverage controls from independent raw inputs',atol=3e-11,rtol=2e-12)
                            proof.check(local.time.to_numpy(),retained.asi8,'Actual monthly time ordinals',atol=0,rtol=0)
                            assert local.season.tolist()==retained.month.tolist() and local.year.tolist()==retained.year.tolist()
                        info=inventories.loc[inventories.sample_id.eq(sample_id)];assert len(info)==1;info=info.iloc[0]
                        assert info.sample_sha256==proof.sha256(path)
                        for field,value in losses.items():assert int(info[field])==value,(sample_id,field,info[field],value)
                        assert info.n_periods==len(retained) and info.crisis_loss_periods==int(eligible.sum()-len(retained))
                        samples[panel,outcome,controls,history,variant]=sample
                        summary.append({'sample_id':sample_id,**losses,'retained_n_periods':len(retained),'first':str(retained.min()),'last':str(retained.max())})
    assert all_masks==len(masks)
    for key,sample in samples.items():
        for cut in PANELS:assert set(sample.time)==set(samples[(cut,)+key[1:]].time),'Shared three-cut history support differs'
    return samples,summary


def features(spec):
    h=int(spec.horizon);lags=[0] if spec.model_spec in ['own_current','benchmark'] else list(range(h+1))
    heat=['temp_endpoint_difference','temp_growth_sum'] if spec.model_spec=='two_channel' else [f'temp_l{lag}' for lag in lags]
    controls=[f'rain_l{lag}' for lag in lags]+OTHER+(ADDED if spec.control_set=='full' else [])
    if spec.model_spec=='dynamic':controls += [f'ar_l{lag}' for lag in range(1,h+1)]
    return heat,controls,lags


def independent_design(sample,spec):
    units=sorted(sample.unit_id.unique());months=sorted(sample.season.unique());years=sorted(sample.year.unique())
    heat,controls,lags=features(spec);names=[];cols=[]
    for unit in units:
        for month in months:names.append(f'origin_season[{unit},{month}]');cols.append((sample.unit_id.eq(unit)&sample.season.eq(month)).to_numpy(float))
    for year in years[1:]:names.append(f'year[{year}]');cols.append(sample.year.eq(year).to_numpy(float))
    for term in controls:names.append(term);cols.append((sample[term]-sample[term].mean()).to_numpy(float))
    for term in heat:
        if term=='temp_endpoint_difference':value=sample.temp_l0-sample[f'temp_l{spec.horizon}']
        elif term=='temp_growth_sum':value=sample[[f'temp_l{lag}' for lag in range(spec.horizon)]].sum(axis=1,min_count=spec.horizon)
        else:value=sample[term]
        value=(value-value.mean()).to_numpy(float);names.append(term);cols.append(value)
        if spec.model=='H1':
            for unit in units[1:]:names.append(f'{term}:C(unit_id)[T.{unit}]');cols.append(value*sample.unit_id.eq(unit).to_numpy())
    return np.column_stack(cols),names,heat,controls,lags


def numerical_rank(x):
    nonzero=np.any(x!=0,axis=0);z=x[:,nonzero];scale=np.linalg.norm(z,axis=0)
    sv=np.linalg.svd(z/scale,compute_uv=False);cutoff=sv[0]*max(x.shape)*np.finfo(float).eps
    return int(np.sum(sv>cutoff)),float(cutoff)


def time_ranks(sample,x):
    ranks={};thresholds={};blocks=[]
    for unit in sorted(sample.unit_id.unique()):
        local=x[sample.unit_id.eq(unit)];ranks[unit],thresholds[unit]=numerical_rank(local)
        blocks.append(local[:,np.any(local!=0,axis=0)])
    span,span_cutoff=numerical_rank(np.column_stack(blocks))
    return ranks,thresholds,span,span_cutoff


def verify_preflight():
    """Bounded raw-calendar/rank audit; explicitly not a native-fit receipt."""
    folder=OUT/'preflight';manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['requested_specs']==270 and manifest['samples']==72
    inputs=independent_inputs();samples,sample_records=reconstruct_samples(folder,inputs)
    specs=read(folder/'specifications.csv');assert len(specs)==270 and specs.spec_id.is_unique
    records=[]
    for number,spec in enumerate(specs.itertuples(),1):
        sample=samples[spec.panel,spec.outcome,spec.control_set,spec.history,spec.sample_variant]
        x,names,heat,controls,lags=independent_design(sample,spec)
        rank,threshold=numerical_rank(x);ranks,cutoffs,span,span_cutoff=time_ranks(sample,x)
        estimated=len(sample)>len(names) and rank==len(names);t=sample.time.nunique()
        assert (spec.status=='estimated')==estimated
        assert spec.design_rank==rank and spec.n_parameters==len(names) and spec.n_obs==len(sample) and spec.n_periods==t
        assert spec.time_design_rank==max(ranks.values()) and spec.time_residual_df==t-max(ranks.values())
        assert json.loads(spec.local_time_design_ranks)==ranks and spec.full_cross_origin_temporal_span_rank==span
        proof.check(spec.design_rank_threshold,threshold,'Declared full design SVD rank cutoff',atol=1e-17,rtol=2e-10)
        proof.check(list(json.loads(spec.local_time_rank_thresholds).values()),[cutoffs[key] for key in sorted(cutoffs)],'Declared reference-invariant local rank cutoffs',atol=1e-17,rtol=2e-10)
        assert proof._truth(spec.time_correction_available)==(t>max(ranks.values()))
        assert proof._truth(spec.primary_calendar_inference_available)==(estimated and t>max(ranks.values()))
        assert json.loads(spec.temp_features)==heat and json.loads(spec.controls)==controls
        records.append({'spec_id':spec.spec_id,'point_estimable':estimated,'n_obs':len(sample),'n_periods':t,'full_rank':rank,'K':len(names),
            'maximum_local_rank':max(ranks.values()),'full_cross_origin_span_rank':span,'primary_calendar_available':estimated and t>max(ranks.values())})
        if number%30==0:print(f'Visitor independent preflight: {number}/270 complete designs',flush=True)
    assert sum(row['point_estimable'] for row in records)==260
    for path,digest in manifest['source_sha256'].items():assert proof.sha256(ROOT/path)==digest
    for path,digest in manifest['raw_inputs_sha256'].items():assert proof.sha256(ROOT/path)==digest
    report={'status':'passed','scope':'Raw sources, disjoint count identities, broader lunar/NEA calendars, all72 sample stages/masks and270 full/local design ranks only; no native-fit or covariance verification claim',
        'samples':72,'requests':270,'point_estimable':260,'point_unavailable':10,'source_audit':inputs[-1],'numerical_checks':proof.ERRORS,
        'verifier_source_sha256':proof.sha256(Path(__file__))}
    pd.DataFrame(records).to_csv(OUT/'verification_preflight_designs.csv',index=False)
    pd.DataFrame(sample_records).to_csv(OUT/'verification_preflight_samples.csv',index=False)
    (OUT/'verification_preflight.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Visitor independent raw/calendar/design preflight passed:72samples,270requests,260point-estimable.',flush=True)
    return report


def independent_fit(x,y,*,sample=None,spec=None,names=None):
    if sample is not None:
        # Exact block FWL avoids a large dense triangular inverse. Origin-local
        # nuisance spaces share their calendar and heat bases; only the small
        # shared year/weather/AR block needs a pooled QR decomposition.
        units=sorted(sample.unit_id.unique());months=sorted(sample.season.unique())
        heat,controls,_=features(spec)
        indexes={unit:np.flatnonzero(sample.unit_id.eq(unit).to_numpy()) for unit in units}
        first=indexes[units[0]]
        local_terms=[f'origin_season[{units[0]},{month}]' for month in months]+(heat if spec.model=='H1' else [])
        Z=x[np.ix_(first,[names.index(term) for term in local_terms])]
        for unit,positions in indexes.items():assert np.array_equal(sample.time.iloc[positions].to_numpy(),sample.time.iloc[first].to_numpy())
        scale=np.linalg.norm(Z,axis=0);qz,rz=qr(Z/scale,mode='economic');iz=solve_triangular(rz,qz.T)/scale[:,None]
        proof.check(iz@Z,np.eye(Z.shape[1]),'Local FWL QR inverse identity',atol=2e-10,rtol=2e-10)
        shared=[i for i,name in enumerate(names) if name.startswith('year[')]+[names.index(term) for term in controls]+([names.index(term) for term in heat] if spec.model=='common' else [])
        C=x[:,shared];Ctilde=np.empty_like(C)
        for positions in indexes.values():Ctilde[positions]=C[positions]-Z@(iz@C[positions])
        scale=np.linalg.norm(Ctilde,axis=0);qc,rc=qr(Ctilde/scale,mode='economic');ic=solve_triangular(rc,qc.T)/scale[:,None]
        # Explicitly enforce the known within-origin local projection on the
        # inverse's rows, avoiding small FWL leakage from large outcome values.
        for positions in indexes.values():ic[:,positions]-=(ic[:,positions]@Z)@iz
        proof.check(ic@Ctilde,np.eye(len(shared)),'Shared FWL QR inverse identity',atol=2e-10,rtol=2e-10)
        inverse=np.zeros((len(names),len(y)));inverse[shared]=ic
        local_maps={}
        for unit,positions in indexes.items():
            mapping=-(iz@C[positions])@ic;mapping[:,positions]+=iz;local_maps[unit]=mapping
            for j,month in enumerate(months):inverse[names.index(f'origin_season[{unit},{month}]')]=mapping[j]
        if spec.model=='H1':
            reference=local_maps[units[0]]
            for j,term in enumerate(heat):
                inverse[names.index(term)]=reference[len(months)+j]
                for unit in units[1:]:inverse[names.index(f'{term}:C(unit_id)[T.{unit}]')]=local_maps[unit][len(months)+j]-reference[len(months)+j]
        beta=inverse@y;resid=y-x@beta
        return beta,resid,inverse
    scales=np.linalg.norm(x,axis=0);q,r=qr(x/scales,mode='economic')
    inverse=solve_triangular(r,q.T)/scales[:,None]
    beta=solve_triangular(r,q.T@y)/scales;resid=y-q@(q.T@y)
    return beta,resid,inverse


def restrictions(length,outcome,dynamic):
    eye=np.eye(length)
    result={'All temperature-lag coefficients zero':eye,'Cumulative direct temperature response zero':np.ones((1,length))}
    if length>1:
        end=length-1 if outcome=='y_yoy_nsa' else 1
        name='Conditional direct-weather level-channel shape; not exact structural static level with AR' if dynamic else 'Exact contemporaneous static-output-level coefficient restriction'
        result[name]=np.vstack([np.delete(eye,[0,end],axis=0),eye[0]+eye[end]])
        if outcome=='y_yoy_nsa':
            name='Conditional direct-weather equal-growth coefficient shape; growth lags held fixed' if dynamic else 'Equal one-period growth response over YoY accumulation window'
            result[name]=np.vstack([eye[1:-1]-eye[[0]],eye[-1]])
            name='Conditional direct-weather response lies in endpoint-difference plus growth-sum span; growth lags held fixed' if dynamic else 'Unrestricted YoY response lies in endpoint-difference plus growth-sum span'
            result[name]=np.vstack([eye[2:-1]-eye[[1]],eye[0]+eye[-1]-eye[1]])
    return result


def wald_check(row,beta,v,hc0,R,df,bound):
    """Symmetric eigenvalues independently reproduce PSD contrast singular ranks."""
    value=R@beta;C=R@v@R.T;C=(C+C.T)/2
    eig=np.abs(np.linalg.eigvalsh(C));svmax=float(eig.max())
    ref=R@hc0@R.T;ref=(ref+ref.T)/2;scale=float(np.max(np.abs(np.linalg.eigvalsh(ref))))
    cutoff=max(svmax*1e-10,scale*1e-12);rank=min(int(np.sum(eig>cutoff)),bound);q=len(R)
    assert row.restrictions==q and row.numerator_df==q and row.denominator_df==df and row.structural_rank_bound==bound
    assert row.covariance_rank==rank,(row.spec_id,row.test,row.covariance_rank,rank)
    proof.check([row.hc0_covariance_scale,row.zero_covariance_threshold,row.rank_cutoff],[scale,scale*1e-12,cutoff],'Wald HC0 and numerical-rank thresholds',atol=1e-7,rtol=2e-6)
    available=rank==q and svmax>scale*1e-12 and df>0
    assert (row.status=='available')==available
    if available:
        statistic=max(float(value@np.linalg.solve(C,value)),0.)
        from scipy.stats import f as f_dist
        proof.check([row.wald_chi2,row.f_statistic],[statistic,statistic/q],'Independent joint Wald/F statistics',atol=1e-5,rtol=1e-5)
        proof.check(row.p_value,float(f_dist.sf(statistic/q,q,df)),'Independent joint Wald F probability',atol=2e-8,rtol=2e-6)
    else:assert pd.isna(row.wald_chi2) and pd.isna(row.f_statistic) and pd.isna(row.p_value)
    assert row.calibration_status==('Not established; nominal asymptotic diagnostic' if q>1 else 'Nominal approximate conditional inference')
    return available


def _verify_visitor_rigour():
    proof.ERRORS.clear();folder=OUT;manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['requested_specs']==270 and manifest['native_fits']==260 and manifest['point_unavailable_requests']==10
    bound_hashes={**manifest['source_sha256'],**manifest['raw_inputs_sha256']}
    for path,digest in bound_hashes.items():assert proof.sha256(ROOT/path)==digest
    inputs=independent_inputs();samples,sample_records=reconstruct_samples(folder,inputs)
    specs=read(folder/'specifications.csv');assert len(specs)==270 and specs.spec_id.is_unique
    tables={name:read(folder/f'{name}.csv') for name in ['coefficients','temperature_effects','interaction_components','channel_estimates','control_estimates','tests']}
    groups={name:{key:part for key,part in table.groupby('spec_id',sort=False)} for name,table in tables.items()}
    counts={name:0 for name in tables};counts.update(full_covariance_cells=0,native_residual_values=0,native_fits=0,point_unavailable=0,static_difference_regressions=0)
    native_expected=[];fit_records=[];test_records=[];static_records=[]
    for number,spec in enumerate(specs.itertuples(),1):
        sample=samples[spec.panel,spec.outcome,spec.control_set,spec.history,spec.sample_variant]
        x,names,heat,controls,lags=independent_design(sample,spec);y=sample[spec.outcome].to_numpy()
        n,k=x.shape;t=sample.time.nunique();units=sorted(sample.unit_id.unique());g=len(units)
        with np.load(ROOT/spec.design_npz,allow_pickle=False) as archive:
            assert all(archive[key].dtype.kind!='O' for key in archive.files)
            assert archive['names'].tolist()==names and archive['unit_ids'].tolist()==sample.unit_id.tolist()
            assert archive['periods'].tolist()==sample.month.tolist() and np.array_equal(archive['time'],sample.time.to_numpy())
            proof.check(archive['X'],x,'Every native visitor logical design cell',atol=0,rtol=0)
            proof.check(archive['Y'],y,'Every native visitor response cell',atol=0,rtol=0)
            rank,rankcut=numerical_rank(x);local,localcut,span,spancut=time_ranks(sample,x);kt=max(local.values());available=n>k and rank==k
            assert spec.design_rank==rank and spec.n_obs==n and spec.n_parameters==k and spec.n_periods==t and spec.n_origins==g
            assert (spec.status=='estimated')==available and bool(archive['point_fit_available'])==available
            assert spec.time_design_rank==kt and spec.time_residual_df==t-kt and json.loads(spec.local_time_design_ranks)==local
            assert spec.full_cross_origin_temporal_span_rank==span
            assert proof._truth(spec.time_correction_available)==(t>kt)
            assert proof._truth(spec.primary_calendar_inference_available)==bool(archive['primary_inference_available'])==(available and t>kt)
            assert json.loads(spec.temp_features)==heat and json.loads(spec.controls)==controls
            proof.check([spec.mean_y,spec.sd_y],[y.mean(),y.std(ddof=1)],'Every visitor exact-sample outcome mean/SD',atol=3e-10,rtol=2e-12)
            selected=archive['selected_positions'].astype(int);first=next(i for i,name in enumerate(names) if not name.startswith(('origin_season[','year[')))
            assert selected.tolist()==list(range(first,k)) and archive['temperature_features'].tolist()==heat and archive['control_features'].tolist()==controls
            components=[[names.index(term)]+([names.index(f'{term}:C(unit_id)[T.{unit}]') for unit in units[1:]] if spec.model=='H1' else []) for term in heat]
            assert archive['component_positions'].tolist()==components
            if available:
                beta,residual,inverse=independent_fit(x,y,sample=sample,spec=spec,names=names)
                proof.check(archive['beta'],beta,'Every native visitor coefficient including nuisance FE',atol=2e-7,rtol=2e-7)
                proof.check(archive['resid'],residual,'Every native visitor residual',atol=2e-8,rtol=1e-8)
                coeff=groups['coefficients'][spec.spec_id].set_index('term');assert len(coeff)==k and set(coeff.index)==set(names)
                proof.check(coeff.loc[names,'coefficient'].to_numpy(),beta,'Every exported native visitor coefficient',atol=2e-7)
                counts['coefficients']+=k;counts['native_residual_values']+=n;counts['native_fits']+=1
                influence=inverse.T*residual[:,None];hc0_full=influence.T@influence
                proof.check(archive['hc0_full'],hc0_full,'Every full visitor HC0 covariance cell',atol=2e-7,rtol=2e-7)
                dates=sorted(sample.time.unique());score=np.vstack([influence[sample.time.eq(d)].sum(axis=0) for d in dates])
                native_expected.append({'label':spec.native_label,'outcome':spec.outcome,'n_obs':n,'n_parameters':k,'fit_id':spec.fit_id,
                    'design_sha256':proof.array_sha256(archive['X']),'outcome_sha256':proof.array_sha256(archive['Y']),
                    'coefficients_sha256':proof.array_sha256(archive['beta']),'residuals_sha256':proof.array_sha256(archive['resid']),
                    'coefficient_names_sha256':hashlib.sha256(json.dumps(names,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
            else:
                assert spec.spec_id not in groups['coefficients'] and np.isnan(archive['beta']).all() and np.isnan(archive['resid']).all() and np.isnan(archive['hc0_full']).all()
                beta=residual=influence=hc0_full=score=None;counts['point_unavailable']+=1
            profiles={}
            for index,unit in enumerate(units if spec.model=='H1' or g==1 else ['pooled']):
                basis=np.zeros((len(heat),len(selected)))
                for j,positions in enumerate(components):
                    basis[j,list(selected).index(positions[0])]=1
                    if spec.model=='H1' and index:basis[j,list(selected).index(positions[index])]=1
                if spec.model_spec=='two_channel':
                    expansion=np.zeros((spec.horizon+1,2));expansion[0,0]=1;expansion[-1,0]=-1;expansion[:-1,1]=1;lag=expansion@basis
                else:lag=basis
                profiles[unit]=(lag,basis)
            methods=[('Calendar HAC',12,t-1),('Calendar HAC',24,t-1)]+([('Origin cluster',0,g-1)] if spec.model=='common' and g>1 else [])
            assert archive['covariance_labels'].tolist()==[m[0] for m in methods] and archive['bandwidths'].tolist()==[m[1] for m in methods]
            assert archive['inference_df'].tolist()==[m[2] for m in methods]
            for mi,(method,band,df) in enumerate(methods):
                inference_available=available and (t>kt if method=='Calendar HAC' else n>k)
                factor=t/(t-kt) if method=='Calendar HAC' and t>kt else g/(g-1)*(n-1)/(n-k) if method=='Origin cluster' and n>k else np.nan
                if available:
                    if method=='Calendar HAC':
                        raw_cov=calendar_meat(score,np.array(dates),band)
                        proof.check(archive[f'raw_calendar_covariance_{mi}'],raw_cov,'Every unadjusted full visitor calendar covariance',atol=2e-7,rtol=2e-7)
                    else:
                        cs=np.vstack([influence[sample.unit_id.eq(unit)].sum(axis=0) for unit in units]);raw_cov=cs.T@cs
                    full_v=raw_cov*factor if np.isfinite(factor) else np.full((k,k),np.nan)
                else:full_v=np.full((k,k),np.nan)
                proof.check(archive[f'covariance_{mi}'],full_v,'Every full visitor method covariance cell',atol=2e-7,rtol=2e-7)
                proof.check(archive['corrections'][mi],factor,'Declared max-local-rank / origin-CR1 finite correction',atol=1e-12,rtol=1e-12)
                counts['full_covariance_cells']+=k*k
                b=beta[selected] if available else None;v=full_v[np.ix_(selected,selected)];hc0=hc0_full[np.ix_(selected,selected)] if available else np.full_like(v,np.nan)
                bound=t-1 if method=='Calendar HAC' else g-1
                current={name:part.loc[part.covariance.eq(method)&part.hac_lags.eq(band)] for name in tables if name!='coefficients' and (part:=groups[name].get(spec.spec_id)) is not None}
                points=current['temperature_effects'];tests=current['tests']
                assert points.main_inference.eq(band==12).all() and points.inference_df.eq(df).all()
                def effect(row,R):
                    if available:proof.check(row.coefficient,float(R@b),'Every visitor contrast point',atol=2e-7)
                    else:assert pd.isna(row.coefficient)
                    if inference_available:proof.inference(row,float(R@b),float(R@v@R),df,float(R@hc0@R))
                    else:
                        assert row[['std_error','ci_lower','ci_upper','p_value','variance']].isna().all() and row.inference_status.startswith('unavailable:')
                def test(row,R,bb=b,vv=v,hh=hc0):
                    assert row.restrictions==len(R) and row.numerator_df==len(R) and row.denominator_df==df and row.structural_rank_bound==bound
                    if inference_available:ok=wald_check(row,bb,vv,hh,R,df,bound)
                    else:
                        assert row.status.startswith('unavailable:') and pd.isna(row.p_value) and pd.isna(row.wald_chi2) and pd.isna(row.f_statistic) and pd.isna(row.covariance_rank);ok=False
                    test_records.append({'spec_id':spec.spec_id,'unit_id':row.unit_id,'covariance':method,'hac_lags':band,'test':row.test,'available':ok});counts['tests']+=1
                profiles_list=[];basis_list=[]
                for unit,(lag,basis) in profiles.items():
                    observed=points[points.unit_id.eq(unit)];assert len(observed)==len(lags)+1
                    local_sample=sample if unit=='pooled' else sample[sample.unit_id.eq(unit)]
                    for i,lag_number in enumerate(lags+[-1]):
                        row=observed.loc[observed.lag.eq(lag_number)];assert len(row)==1;row=row.iloc[0]
                        R=lag[i] if lag_number>=0 else lag.sum(axis=0);effect(row,R)
                        proof.check([row.mean_y,row.sd_y],[local_sample[spec.outcome].mean(),local_sample[spec.outcome].std(ddof=1)],'Every origin exact outcome mean and SD',atol=3e-10,rtol=2e-12)
                        if spec.model=='H1' and spec.model_spec!='two_channel' and lag_number>=0 and available:
                            index=units.index(unit);positions=components[i];m=list(selected).index(positions[0]);d=list(selected).index(positions[index])
                            proof.check([row.temp_main,row.interaction_delta],[b[m],0. if index==0 else b[d]],'Native baseline-plus-origin point decomposition',atol=2e-7)
                            if inference_available:proof.check([row.var_temp_main,row.var_interaction_delta,row.cov_temp_main_delta],[v[m,m],0. if index==0 else v[d,d],0. if index==0 else v[m,d]],'Native baseline/delta covariance cross terms',atol=2e-7)
                        counts['temperature_effects']+=1
                    if spec.model_spec=='two_channel':
                        channel=current['channel_estimates'];channel=channel[channel.unit_id.eq(unit)].set_index('channel');assert len(channel)==2 and set(channel.index)==set(heat)
                        for j,feature in enumerate(heat):effect(channel.loc[feature],basis[j]);counts['channel_estimates']+=1
                        ops={'All temperature channels zero':np.eye(2),'Cumulative direct temperature response zero; same null as growth-sum channel zero':np.array([[0.,spec.horizon]]),
                            'Static coefficient shape; same null as growth-sum channel zero':np.array([[0.,1.]]),'Equal-growth accumulation channel shape':np.array([[1.,0.]])}
                        requested={name:R@basis for name,R in ops.items()}
                    else:requested={name:R@lag for name,R in restrictions(len(lags),spec.outcome,spec.model_spec=='dynamic').items()}
                    for name,R in requested.items():
                        row=tests.loc[tests.unit_id.eq(unit)&tests.test.eq(name)];assert len(row)==1;test(row.iloc[0],R)
                    profiles_list.append(lag);basis_list.append(basis)
                controls_rows=current['control_estimates'].set_index('term');assert len(controls_rows)==len(controls) and set(controls_rows.index)==set(controls)
                for j,term in enumerate(controls):effect(controls_rows.loc[term],np.eye(len(selected))[j]);counts['control_estimates']+=1
                if spec.model=='H1':
                    components_rows=current['interaction_components'];assert len(components_rows)==len(heat)*(g+1)
                    for j,feature in enumerate(heat):
                        for i,unit in enumerate(units):
                            term=feature if not i else f'{feature}:C(unit_id)[T.{unit}]';row=components_rows[components_rows.term.eq(term)&~components_rows.normalization.map(proof._truth)]
                            assert len(row)==1 and row.unit_id.iloc[0]==unit and proof._truth(row.estimated.iloc[0])==available
                            R=np.eye(len(selected))[list(selected).index(names.index(term))];effect(row.iloc[0],R);counts['interaction_components']+=1
                        zero=components_rows.loc[components_rows.feature.eq(feature)&components_rows.normalization.map(proof._truth)]
                        assert len(zero)==1 and zero.coefficient.eq(0).all() and zero[['std_error','ci_lower','ci_upper','p_value']].isna().all().all();counts['interaction_components']+=1
                    cumulative=np.vstack([p.sum(axis=0) for p in profiles_list]);D=np.eye(g)[1:]-np.eye(g)[[0]]
                    row=tests[tests.unit_id.eq('all')&tests.test.eq('All origin cumulative direct effects equal')];assert len(row)==1;test(row.iloc[0],D@cumulative)
                    flattened=np.vstack(basis_list if spec.model_spec=='two_channel' else profiles_list)
                    row=tests[tests.unit_id.eq('all')&tests.test.eq('All origin temperature response coefficients jointly zero')];assert len(row)==1;test(row.iloc[0],flattened)
            if available and spec.model=='H1' and spec.model_spec=='own_current':
                reference=units[0];source=sample.pivot(index='time',columns='unit_id',values=spec.outcome).sort_index()
                period=sample.drop_duplicates('time').set_index('time').loc[source.index];z=np.column_stack([period.season.eq(month).to_numpy(float) for month in sorted(period.season.unique())]+[period.temp_l0.to_numpy(float)])
                totals=[]
                for i,unit in enumerate(units):
                    R=profiles[unit][1][0];totals.append(float(R@beta[selected]))
                    if not i:continue
                    diff=source[unit].to_numpy()-source[reference].to_numpy();delta=independent_fit(z,diff)[0][-1]
                    position=names.index(f'temp_l0:C(unit_id)[T.{unit}]')
                    proof.check(beta[position],delta,'Static H1 delta equals origin-minus-reference regression with calendar-month FE',atol=2e-7,rtol=2e-7)
                    static_records.append({'spec_id':spec.spec_id,'unit_id':unit,'reference_unit_id':reference,'native_interaction':float(archive['beta'][position]),'independent_difference_slope':float(delta)})
                    counts['static_difference_regressions']+=1
                other_id=spec.spec_id.replace('__H1__','__common__')
                with np.load(folder/'designs'/f'{other_id}.npz',allow_pickle=False) as common:
                    common_temp=common['beta'][common['names'].tolist().index('temp_l0')]
                proof.check(np.mean(totals),common_temp,'Static balanced H1 mean slope equals matched common slope',atol=2e-7)
            fit_records.append({'spec_id':spec.spec_id,'point_available':available,'primary_available':available and t>kt,'N':n,'T':t,'K':k,'rank':rank,'K_time_max':kt,'full_cross_origin_span_rank':span})
        if number%20==0:print(f'Visitor independent full QR/covariance audit: {number}/270 requests',flush=True)
    for name in tables:assert counts[name]==len(tables[name]),('Every exported visitor row audited',name,counts[name],len(tables[name]))
    for name in ['temperature_effects','channel_estimates']:
        family=['spec_id','covariance','hac_lags','term','effect']
        for _,part in tables[name].groupby(family,sort=False):
            adjusted=proof.holm(part.p_value.fillna(1).to_numpy());expected=np.where(part.p_value.notna(),adjusted,np.nan)
            proof.check(part.p_value_holm_profile.to_numpy(),expected,'Every defined within-profile origin Holm family',atol=2e-12,rtol=2e-12)
            assert part.multiplicity_family_size.eq(len(part)).all()
    assert len(static_records)==588 and len({r['spec_id'] for r in static_records})==24
    native=read(folder/'pyfixest/fit_log.csv');native_manifest=json.loads((folder/'pyfixest/estimator_manifest.json').read_text())
    assert len(native)==len(native_expected)==260 and native_manifest['native_fits_recorded']==260 and native.api_call_id.nunique()==260
    assert native_manifest['fit_log_sha256']==proof.sha256(folder/'pyfixest/fit_log.csv')
    lookup=native.set_index(['label','outcome']);assert lookup.index.is_unique and set(lookup.index)=={(row['label'],row['outcome']) for row in native_expected}
    for expected in native_expected:
        row=lookup.loc[(expected['label'],expected['outcome']),:]
        for field,value in expected.items():
            if field not in ['label','outcome']:assert row[field]==value,('Native visitor identity',expected['label'],field)
    assert native.native_class.eq('Feols').all() and native.native_module.str.startswith('pyfixest').all()
    assert native.native_dropped_rows.eq(0).all() and native.native_dropped_coefficients.eq(0).all()
    assert native.native_residuals_sha256.eq(native.residuals_sha256).all() and native.native_outcome_sha256.eq(native.outcome_sha256).all()
    for field in ['native_mapped_coefficient_max_abs_error','native_residual_max_abs_error','native_response_max_abs_error','native_design_max_abs_error']:assert native[field].eq(0).all()
    for path,digest in bound_hashes.items():assert proof.sha256(ROOT/path)==digest
    pd.DataFrame(fit_records).to_csv(folder/'verification_fits.csv',index=False)
    pd.DataFrame(sample_records).to_csv(folder/'verification_samples.csv',index=False)
    pd.DataFrame(test_records).to_csv(folder/'verification_tests.csv',index=False)
    pd.DataFrame(static_records).to_csv(folder/'verification_static_relative_identity.csv',index=False)
    return {'status':'passed','requests':270,'native_fits':260,'point_unavailable':10,'samples':72,'counts':counts,'raw_source_audit':inputs[-1],
        'native_fit_log_sha256':proof.sha256(folder/'pyfixest/fit_log.csv'),'source_and_old_native_bindings_unchanged':len(bound_hashes),
        'static_relative_identity':{'static_H1_cases':24,'origin_minus_reference_regressions':588,'scope':'Point identities with shared current weather/LNY/year controls and balanced month samples; not asserted for AR/dynamic models. Full actual-month FE has the same relative identification; no absolute shared-weather coefficient is identified with full date FE.'},
        'numerical_checks':proof.ERRORS,'verifier_source_sha256':proof.sha256(Path(__file__)),
        'limits':'Native/computational and raw-source verification does not establish causality, stationarity, dynamic-FE unbiasedness or exact finite-sample HAC/Wald calibration. Shared national exposure and dependent multiplicity remain. Calendar corrections use the declared reference-invariant max-local-rank working convention.'}


def verify_visitor_rigour():
    path=OUT/'verification.json';artifact_bindings.archive_passed(path)
    initial={'status':'running','verifier_source_sha256':proof.sha256(Path(__file__))};path.write_text(json.dumps(initial,indent=2)+'\n')
    try:
        checked=artifact_bindings.snapshot(OUT)
        with threadpool_limits(limits=1):report=_verify_visitor_rigour()
        artifact_bindings.assert_unchanged(checked);assert artifact_bindings.snapshot(OUT)==checked
        report['verified_artifacts_sha256']=checked
        report['artifact_binding_helper_sha256']=proof.sha256(Path(artifact_bindings.__file__))
    except BaseException as error:
        path.write_text(json.dumps({**initial,'status':'failed','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n');raise
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Visitor monthly independent verification passed:270 requests,260 native fits,72 samples.',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.preflight:verify_preflight()
        else:verify_visitor_rigour()
