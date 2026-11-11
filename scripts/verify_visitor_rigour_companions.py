"""Independent annual/count companions audit; imports no production visitor code."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for _key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_key,'1')
import numpy as np
import pandas as pd
from scipy.linalg import qr,solve_triangular
from scipy.special import xlogy
from scipy.stats import f as f_dist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from verify_visitor_rigour import independent_inputs,read,numerical_rank,independent_fit
from verify_literature_sensitivity import calendar_meat
import verify_crisis_sensitivity as proof
import verification_artifacts as artifact_bindings

OUT=ROOT/'outputs/visitor_rigour'
VARIANTS=['all_eligible','event_dates_excluded','strict_growth_window_clean']
BASE=['temp_c','rain_100mm','rh_pct'];FULL=BASE+['wind_mps','pm25_10ug'];LNY=['lny_window_days','lny_lag1']


def bindings(branch):
    manifest=json.loads((OUT/branch/'manifest.json').read_text())
    bound={**manifest['inputs_sha256'],**manifest['protected_source_results_sha256']}
    assert manifest['source_sha256']==proof.sha256(ROOT/manifest['source'])
    for path,digest in bound.items():assert proof.sha256(ROOT/path)==digest,(branch,path)
    return manifest,bound


def yearly_inputs(inputs):
    counts,cuts,catalog,growth,weather,audit=inputs
    years=sorted(year for year in counts.index.year.unique() if sum(counts.index.year==year)==12)
    assert years==list(range(2008,2026))
    levels={(sid,year):sum(int(counts.loc[month,sid]) for month in counts.index[counts.index.year==year]) for sid in counts for year in years}
    outcomes={(sid,year):100*math.log(levels[sid,year]/levels[sid,year-1]) for sid in counts for year in years if (sid,year-1) in levels and levels[sid,year]>0 and levels[sid,year-1]>0}
    annual={}
    for year in years:
        dates=counts.index[counts.index.year==year];days=sum(month.days_in_month for month in dates)
        record={}
        for term in ['temp_c','rh_pct','wind_mps','pm25_10ug']:
            values=[weather.get(month,{}).get(term,np.nan) for month in dates]
            record[term]=math.fsum(value*month.days_in_month for month,value in zip(dates,values))/days if np.isfinite(values).all() else np.nan
        rain=[weather.get(month,{}).get('rain_100mm',np.nan) for month in dates]
        record['rain_100mm']=math.fsum(rain) if np.isfinite(rain).all() else np.nan
        record['eligible_90']=all(weather.get(month,{}).get('eligible_90',False) for month in dates);annual[year]=record
    return levels,outcomes,annual


def check_joint(row,beta,v,hc0,R,df,bound):
    assert row.restrictions==len(R) and row.structural_rank_bound==bound
    if v is None:
        assert row.status.startswith('unavailable:') and pd.isna(row.p_value);return False
    value=R@beta;C=R@v@R.T;C=(C+C.T)/2;sing=np.linalg.svd(C,compute_uv=False)
    scale=float(np.linalg.norm(R@hc0@R.T,2));cutoff=max(sing[0]*1e-10,scale*1e-12)
    rank=min(int(np.sum(sing>cutoff)),bound);q=len(R);ok=rank==q and sing[0]>scale*1e-12
    assert row.covariance_rank==rank and row.numerator_df==q and row.denominator_df==df
    proof.check([row.hc0_covariance_scale,row.rank_cutoff],[scale,cutoff],'Companion joint-test scale/rank thresholds',atol=2e-7,rtol=2e-6)
    assert row.status==('computable: calibration not established' if ok else 'unavailable: singular restriction covariance')
    if ok:
        statistic=float(value@np.linalg.solve(C,value))/q
        proof.check(row.f_statistic,statistic,'Companion independently computed Wald F',atol=1e-5,rtol=1e-5)
        proof.check(row.p_value,float(f_dist.sf(max(statistic,0),q,df)),'Companion independent nominal F probability',atol=2e-8,rtol=2e-6)
    else:assert pd.isna(row.p_value) and pd.isna(row.f_statistic)
    return ok


def _verify_annual(inputs):
    folder=OUT/'annual';manifest,bound=bindings('annual');counts,cuts,catalog,growth,weather,raw_audit=inputs
    levels,outcomes,annual=yearly_inputs(inputs)
    source=read(folder/'annual_counts.csv');assert len(source)==62*18
    for row in source.itertuples():
        proof.check(row.arrivals,levels[row.unit_id,row.year],'Every complete annual arrival sum',atol=0,rtol=0)
        proof.check(row.previous_arrivals,levels.get((row.unit_id,row.year-1),np.nan),'Raw previous-calendar-year annual count',atol=0,rtol=0)
        proof.check(row.y_annual_nsa,outcomes.get((row.unit_id,row.year),np.nan),'Annual log count growth before all eligibility/exclusions',atol=3e-10,rtol=2e-12)
    source_weather=read(folder/'annual_weather.csv');assert len(source_weather)==18
    for row in source_weather.itertuples():
        assert row.months==12 and proof._truth(row.eligible_90)==annual[row.year]['eligible_90']
        for term in FULL:proof.check(getattr(row,term),annual[row.year][term],'Day-weighted annual weather and summed rainfall',atol=3e-11,rtol=2e-12)
    policy=json.loads((ROOT/'notes/crisis_sensitivity_policy.json').read_text())
    event_years={year for event in policy['events'] for year in range(int(event['start'][:4]),int(event['end'][:4])+1)}
    specs=read(folder/'specifications.csv');assert len(specs)==30 and specs.spec_id.is_unique
    tables={name:read(folder/f'{name}.csv') for name in ['coefficients','temperature_effects','control_estimates','interaction_components','slope_covariances','tests','time_design_ranks']}
    groups={name:{key:part for key,part in table.groupby('spec_id',sort=False)} for name,table in tables.items()}
    masks=read(folder/'calendar_masks.csv');native_expected=[];records=[];tally={name:0 for name in tables};tally.update(native_fits=0,unavailable_point=0,full_covariance_cells=0)
    shared={year for year in annual if all((sid,year) in outcomes for cut in ['aggregate','region','country'] for sid in cuts[cut])}
    expected=set()
    for context in ['aggregate','region','country']:
        for control in ['long','full']:
            for variant in VARIANTS:
                for model in ['common']+(['H1'] if context!='aggregate' else []):expected.add((context,control,variant,model))
    assert set(specs[['context','control_set','sample_variant','model']].itertuples(index=False,name=None))==expected
    for spec in specs.itertuples():
        ids=cuts[spec.context];terms=BASE if spec.control_set=='long' else FULL
        own={year for year in annual if all((sid,year) in outcomes for sid in ids)};parent=own&shared
        finite={year for year in parent if all(np.isfinite(annual[year][term]) for term in terms)}
        eligible={year for year in finite if spec.control_set!='full' or annual[year]['eligible_90']}
        retained=eligible if spec.sample_variant==VARIANTS[0] else eligible-event_years
        if spec.sample_variant==VARIANTS[2]:retained={year for year in retained if year-1 not in event_years}
        sample=read(ROOT/spec.sample_csv);assert proof.sha256(ROOT/spec.sample_csv)==spec.sample_sha256
        assert set(sample.unit_id)==set(ids) and set(sample.year)==retained and len(sample)==len(ids)*len(retained)
        assert not sample.duplicated(['year','unit_id']).any()
        for row in sample.itertuples():
            proof.check([row.arrivals,row.y_annual_nsa],[levels[row.unit_id,row.year],outcomes[row.unit_id,row.year]],'Annual exact raw Y/sample support',atol=3e-10,rtol=2e-12)
            for term in FULL:proof.check(getattr(row,term),annual[row.year][term],'Annual retained weather bases',atol=3e-11,rtol=2e-12)
        losses={'source_complete_level_years':18,'source_growth_missing_periods':18-len(own),'raw_growth_eligible_periods':len(own),
            'shared_cut_alignment_loss_periods':len(own)-len(parent),'parent_growth_periods':len(parent),'control_raw_missing_periods':len(parent)-len(finite),
            'control_coverage_excluded_periods':len(finite)-len(eligible),'control_eligible_periods':len(eligible),'crisis_removed_periods':len(eligible)-len(retained)}
        for field,value in losses.items():assert getattr(spec,field)==value,(spec.spec_id,field)
        assert not proof._truth(spec.year_fixed_effects) and proof._truth(spec.linear_trend) and not proof._truth(spec.annual_LNY_included)
        observed_masks=masks[masks.panel.eq(spec.context)&masks.control_set.eq(spec.control_set)&masks.sample_variant.eq(spec.sample_variant)]
        assert set(observed_masks.year)==eligible
        for row in observed_masks.itertuples():
            assert proof._truth(row.current_event)==(row.year in event_years) and proof._truth(row.prior_year_event)==(row.year-1 in event_years)
            assert proof._truth(row.retained)==(row.year in retained) and row.output_support_start==row.year-1
        units=sorted(ids);columns={f'origin[{unit}]':sample.unit_id.eq(unit).to_numpy(float) for unit in units}
        columns['trend']=(sample.year-sample.year.mean()).to_numpy()
        for term in terms[1:]:columns[term]=(sample[term]-sample[term].mean()).to_numpy()
        heat=(sample.temp_c-sample.temp_c.mean()).to_numpy();columns['temp_c']=heat
        if spec.model=='H1':
            for unit in units[1:]:columns[f'temp_c:C(unit_id)[T.{unit}]']=heat*sample.unit_id.eq(unit).to_numpy()
        names=list(columns);x=np.column_stack(list(columns.values()));y=sample.y_annual_nsa.to_numpy();n,k=x.shape;t=len(retained)
        r,_=numerical_rank(x);local={unit:numerical_rank(x[sample.unit_id.eq(unit)])[0] for unit in units}
        span,_=numerical_rank(np.column_stack([x[sample.unit_id.eq(unit)] for unit in units]));kt=max(local.values());available=n>k and r==k
        assert spec.n_obs==n and spec.n_parameters==k and spec.design_rank==r and spec.time_design_rank==kt and spec.time_residual_df==t-kt and spec.full_temporal_span_rank==span
        assert (spec.status=='estimated')==available and proof._truth(spec.primary_calendar_inference_available)==(available and t>kt)
        proof.check([spec.mean_y,spec.sd_y],[y.mean(),y.std(ddof=1)],'Annual exact sample mean/SD',atol=3e-10,rtol=2e-12)
        rank_rows=groups['time_design_ranks'][spec.spec_id].set_index('unit_id');assert rank_rows.local_design_rank.to_dict()==local;tally['time_design_ranks']+=len(units)
        with np.load(ROOT/spec.design_npz,allow_pickle=False) as archive:
            assert archive['names'].tolist()==names and archive['units'].tolist()==sample.unit_id.tolist() and np.array_equal(archive['time'],sample.year)
            proof.check(archive['X'],x,'Every annual full FE design cell',atol=0,rtol=0);proof.check(archive['Y'],y,'Every annual native response',atol=0,rtol=0)
            if available:
                beta,resid,inverse=independent_fit(x,y);influence=inverse.T*resid[:,None];hc0=influence.T@influence
                proof.check(archive['beta'],beta,'All annual native FE coefficients',atol=2e-7,rtol=2e-7);proof.check(archive['residual'],resid,'All annual native residuals',atol=2e-8,rtol=1e-8)
                proof.check(archive['hc0'],hc0,'Full annual HC0',atol=2e-7,rtol=2e-7)
                coefs=groups['coefficients'][spec.spec_id].set_index('term');assert set(coefs.index)==set(names);proof.check(coefs.loc[names,'coefficient'],beta,'All annual exported native coefficients',atol=2e-7)
                tally['coefficients']+=k;tally['native_fits']+=1
                native_expected.append({'label':spec.native_label,'outcome':'y_annual_nsa','n_obs':n,'n_parameters':k,'fit_id':spec.fit_id,'design_sha256':proof.array_sha256(archive['X']),
                    'outcome_sha256':proof.array_sha256(archive['Y']),'coefficients_sha256':proof.array_sha256(archive['beta']),'residuals_sha256':proof.array_sha256(archive['residual'])})
            else:
                assert 'beta' not in archive.files and 'residual' not in archive.files and spec.spec_id not in groups['coefficients'];beta=hc0=influence=None;tally['unavailable_point']+=1
            methods=[('Calendar HAC',h,t-1) for h in [1,2]]+([('Origin cluster',0,len(units)-1)] if spec.model=='common' and len(units)>1 else [])
            assert archive['covariance_labels'].tolist()==[m[0] for m in methods] and archive['bandwidths'].tolist()==[m[1] for m in methods]
            L=np.zeros((len(units) if spec.model=='H1' else 1,k));L[:,names.index('temp_c')]=1
            if spec.model=='H1':
                for i,unit in enumerate(units[1:],1):L[i,names.index(f'temp_c:C(unit_id)[T.{unit}]')]=1
            for mi,(method,band,df) in enumerate(methods):
                can=available and (method=='Origin cluster' or t>kt);factor=None;v=None
                if can:
                    if method=='Origin cluster':
                        score=np.vstack([influence[sample.unit_id.eq(unit)].sum(axis=0) for unit in units]);factor=len(units)/(len(units)-1)*(n-1)/(n-k);v=score.T@score*factor
                    else:
                        score=np.vstack([influence[sample.year.eq(year)].sum(axis=0) for year in sorted(retained)]);factor=t/(t-kt);v=calendar_meat(score,np.array(sorted(retained)),band)*factor
                    proof.check(archive[f'covariance_{mi}'],v,'Every full annual calendar/cluster covariance',atol=2e-7,rtol=2e-7);tally['full_covariance_cells']+=k*k
                else:assert f'covariance_{mi}' not in archive.files
                proof.check(archive['corrections'][mi],factor if factor is not None else np.nan,'Annual max-local-rank finite correction',atol=1e-12,rtol=1e-12)
                assert archive['inference_df'][mi]==df
                def check_point(row,R):
                    if available:proof.check(row.coefficient,float(R@beta),'Every annual contrast point',atol=2e-7)
                    else:assert pd.isna(row.coefficient)
                    if can:proof.inference(row,float(R@beta),float(R@v@R),df,float(R@hc0@R))
                    else:assert row[['std_error','p_value']].isna().all() and row.inference_status.startswith('unavailable:')
                observed={name:part.loc[part.covariance.eq(method)&part.hac_lags.eq(band)] for name in tables if name not in ['coefficients','time_design_ranks'] and (part:=groups[name].get(spec.spec_id)) is not None}
                effects=observed['temperature_effects'];assert len(effects)==len(L) and effects.main_inference.eq(method=='Calendar HAC' and band==1).all()
                for j,unit in enumerate(units if spec.model=='H1' else ['pooled']):
                    row=effects.loc[effects.unit_id.eq(unit)].iloc[0];check_point(row,L[j]);local_sample=sample if unit=='pooled' else sample[sample.unit_id.eq(unit)]
                    proof.check([row.mean_y,row.sd_y],[local_sample.y_annual_nsa.mean(),local_sample.y_annual_nsa.std(ddof=1)],'Annual origin mean/SD',atol=3e-10,rtol=2e-12)
                    tally['temperature_effects']+=1
                cr=observed['control_estimates'].set_index('term');assert set(cr.index)==set(terms[1:])
                for term in terms[1:]:check_point(cr.loc[term],np.eye(k)[names.index(term)]);tally['control_estimates']+=1
                if spec.model=='H1':
                    parts=observed['interaction_components'];assert len(parts)==len(units)+1
                    for i,unit in enumerate(units):
                        term='temp_c' if not i else f'temp_c:C(unit_id)[T.{unit}]';row=parts.loc[parts.term.eq(term)&~parts.normalization.map(proof._truth)];assert len(row)==1
                        check_point(row.iloc[0],np.eye(k)[names.index(term)]);tally['interaction_components']+=1
                    zero=parts.loc[parts.normalization.map(proof._truth)];assert len(zero)==1 and zero.coefficient.eq(0).all() and zero[['std_error','p_value']].isna().all().all();tally['interaction_components']+=1
                    if v is not None:
                        rows=observed['slope_covariances'];sv=rows.pivot(index='unit_i',columns='unit_j',values='covariance_value').loc[units,units]
                        proof.check(sv.to_numpy(),L@v@L.T,'Every annual total-slope covariance cross term',atol=2e-7,rtol=2e-7);tally['slope_covariances']+=len(rows)
                    jt=observed['tests'];assert len(jt)==2
                    for name,R in [('All origin temperature slopes equal',L[1:]-L[[0]]),('All origin temperature slopes zero',L)]:
                        row=jt[jt.test.eq(name)];assert len(row)==1;check_joint(row.iloc[0],beta,v,hc0,R,df,t-1);tally['tests']+=1
        records.append({'spec_id':spec.spec_id,'point_estimated':available,'n_periods':t,'max_local_rank':kt,'full_rank':r,'K':k})
    for name in tables:assert tally[name]==len(tables[name]),('Annual export coverage',name,tally[name],len(tables[name]))
    assert tally['native_fits']==15 and tally['unavailable_point']==15
    for _,part in tables['temperature_effects'].loc[tables['temperature_effects'].model.eq('H1')].groupby(['spec_id','covariance','hac_lags'],sort=False):
        expected=np.where(part.p_value.notna(),proof.holm(part.p_value.fillna(1)),np.nan);proof.check(part.p_value_holm_within_profile,expected,'Annual profile Holm adjustment',atol=2e-12,rtol=2e-12)
    log=read(folder/'pyfixest/fit_log.csv');receipt=json.loads((folder/'pyfixest/estimator_manifest.json').read_text())
    assert len(log)==len(native_expected)==15 and receipt['fit_log_sha256']==proof.sha256(folder/'pyfixest/fit_log.csv')
    lookup=log.set_index(['label','outcome'])
    for expected in native_expected:
        actual=lookup.loc[(expected['label'],expected['outcome']),:]
        for field,value in expected.items():
            if field not in ['label','outcome']:assert actual[field]==value
    assert log.native_class.eq('Feols').all() and log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    for path,digest in bound.items():assert proof.sha256(ROOT/path)==digest
    pd.DataFrame(records).to_csv(folder/'verification_fits.csv',index=False)
    return {'status':'passed','requests':30,'native_fits':15,'point_unavailable':15,'counts':tally,'raw_source_audit':raw_audit,
        'maximum_annual_growth_years':17,'annual_stationarity':'Not established: all annual samples shorter than20 declared years',
        'native_fit_log_sha256':proof.sha256(folder/'pyfixest/fit_log.csv'),'protected_bindings_unchanged':len(bound),'numerical_checks':dict(proof.ERRORS),
        'limits':'Computational/source agreement does not establish causality or calibrated small-sample inference. Full-control requests retain all controls and are explicitly unavailable.'}


def poisson_fit(x,y):
    """Independent small aggregate Poisson IRLS with likelihood step halving."""
    def least_squares(a,b):
        scale=np.linalg.norm(a,axis=0);q,r=qr(a/scale,mode='economic')
        return solve_triangular(r,q.T@b)/scale
    beta=least_squares(x,np.log(np.maximum(y,.1)))
    objective=lambda eta:float(np.sum(xlogy(y,y/np.exp(eta))-(y-np.exp(eta))))
    for iteration in range(200):
        eta=x@beta;mu=np.exp(eta);weights=np.sqrt(mu)
        candidate=least_squares(x*weights[:,None],weights*(eta+(y-mu)/mu))
        step=candidate-beta;old=objective(eta);fraction=1.
        while objective(x@(beta+fraction*step))>old+max(1e-8,abs(old)*1e-13):
            fraction/=2
            assert fraction>1e-10,'Independent Poisson line search failed'
        update=fraction*step;beta+=update
        score=np.max(np.abs(x.T@(y-np.exp(x@beta))))/(y.sum()+1)
        if np.max(np.abs(update)/(1+np.abs(beta)))<1e-11 and score<1e-10:break
    else:raise AssertionError('Independent Poisson IRLS did not converge')
    return beta,np.exp(x@beta),iteration+1,float(score)


def array_hash(values):return hashlib.sha256(np.ascontiguousarray(values,dtype=np.float64).tobytes()).hexdigest()


def _verify_ppml(inputs):
    folder=OUT/'ppml';manifest,bound=bindings('ppml');counts,cuts,catalog,growth,weather,raw_audit=inputs
    specs=read(folder/'specifications.csv');assert len(specs)==30 and specs.spec_id.is_unique
    canonical=specs.loc[specs.native_fit_executed.map(proof._truth)];assert len(canonical)==20
    aliases=read(folder/'aliases.csv');assert len(aliases)==10
    ledger=read(folder/'pyfixest/fit_log.csv');receipt=json.loads((folder/'pyfixest/estimator_manifest.json').read_text())
    assert len(ledger)==receipt['native_calls']==receipt['native_models_returned']==20 and receipt['fit_log_sha256']==proof.sha256(folder/'pyfixest/fit_log.csv')
    assert ledger.native_class.eq('Fepois').all() and ledger.native_module.str.startswith('pyfixest').all() and ledger.native_converged.map(proof._truth).all()
    assert ledger.native_response_sha256.eq(ledger.response_sha256).all()
    assert all(json.loads(value)==[] for value in ledger.dropped_row_positions)
    dropped=read(folder/'dropped_observations.csv');assert dropped.empty
    tables={name:read(folder/f'{name}.csv') for name in ['coefficients','temperature_effects','control_estimates','tests']}
    groups={name:{key:part for key,part in table.groupby('spec_id',sort=False)} for name,table in tables.items()}
    masks=read(folder/'calendar_masks.csv');seen_masks=set();policy=json.loads((ROOT/'notes/crisis_sensitivity_policy.json').read_text())
    events={month for event in policy['events'] for month in pd.period_range(event['start'],event['end'],freq='M')}
    local_rows=read(folder/'time_design_ranks.csv');native_lookup=ledger.set_index('spec_id')
    tally={name:0 for name in tables};tally.update(mu_values=0,covariance_cells=0,retained_zero_cells=0,canonical_models=0)
    independent_models={};native_data={};records=[]
    for spec in canonical.itertuples():
        ids=cuts[spec.context];terms=(BASE if spec.control_set=='long' else FULL)+LNY
        assert json.loads(spec.terms)==terms and json.loads(spec.controls)==terms[1:] and json.loads(spec.temperature_terms)==['temp_c']
        finite={d for d in counts.index if all(np.isfinite(weather.get(d,{}).get(term,np.nan)) for term in terms)}
        eligible={d for d in finite if spec.control_set!='full' or weather[d].get('eligible_90',False)}
        retained=eligible if spec.sample_variant=='all_eligible' else eligible-events
        actual=read(ROOT/spec.sample_csv);original=read(ROOT/spec.input_sample_csv)
        assert proof.sha256(ROOT/spec.sample_csv)==spec.sample_sha256 and proof.sha256(ROOT/spec.input_sample_csv)==spec.input_sample_sha256
        assert len(actual)==len(original)==len(retained)*len(ids) and set(actual.month)=={str(d) for d in retained} and set(actual.unit_id)==set(ids)
        assert actual.input_row.tolist()==original.input_row.tolist()==list(range(len(actual))) and actual[['month','unit_id']].equals(original[['month','unit_id']])
        for unit,group in actual.groupby('unit_id',sort=False):
            periods=pd.PeriodIndex(group.month,freq='M')
            proof.check(group.arrivals.to_numpy(),counts.loc[periods,unit].to_numpy(),'Every retained PPML raw count including zeros',atol=0,rtol=0)
            for term in terms:proof.check(group[term].to_numpy(),[weather[d].get(term,np.nan) for d in periods],'Every PPML current raw weather and broader LNY value',atol=3e-11,rtol=2e-12)
            proof.check(group.time.to_numpy(),periods.asi8,'PPML actual time ordinals',atol=0,rtol=0)
        zero=int(actual.arrivals.eq(0).sum());assert spec.input_zero_cells==spec.retained_zero_cells==zero and spec.dropped_zero_cells==spec.dropped_rows==0
        assert json.loads(spec.dropped_units)==[] and json.loads(spec.native_dropped_requested_terms)==[] and json.loads(spec.native_dropped_coef_names)==[]
        assert spec.status=='estimated' and proof._truth(spec.native_converged)
        assert spec.control_raw_missing_periods==225-len(finite) and spec.control_coverage_excluded_periods==len(finite)-len(eligible)
        assert spec.weather_eligible_periods==len(eligible) and spec.removed_for_event_dates==len(eligible)-len(retained)
        assert spec.n_obs==len(actual) and spec.n_periods==len(retained) and spec.n_units==len(ids)
        key=(spec.context,spec.control_set)
        if key not in seen_masks:
            seen_masks.add(key);m=masks.loc[masks.panel.eq(spec.context)&masks.control_set.eq(spec.control_set)].set_index('month');assert set(m.index)==set(counts.index.astype(str))
            for d in counts.index:
                row=m.loc[str(d)];ok=d in eligible
                assert proof._truth(row.finite_current_controls)==(d in finite)
                assert proof._truth(row.current_event)==(d in events)
                assert proof._truth(row.retain_all)==ok and proof._truth(row.retain_event)==(ok and d not in events) and proof._truth(row.retain_strict)==(ok and d not in events)
        dates=sorted(retained);calendar=actual.drop_duplicates('time').set_index('time').loc[[d.ordinal for d in dates]]
        W=calendar[terms].to_numpy(float);years=sorted(calendar.year.unique());seasons=sorted(calendar.season.unique())
        F=np.column_stack([calendar.season.eq(month).to_numpy(float) for month in seasons]+[calendar.year.eq(year).to_numpy(float) for year in years[1:]])
        Xaggregate=np.column_stack([F,W]);Yaggregate=actual.groupby('time').arrivals.sum().reindex(calendar.index).to_numpy(float)
        b_full,mu_total,iterations,relative_score=poisson_fit(Xaggregate,Yaggregate);b=b_full[-len(terms):]
        # The independently profiled origin-season composition has an analytic
        # multinomial MLE. Its product with the fitted total mean reconstructs
        # every native count prediction without a second high-dimensional fit.
        seasonal_totals=actual.groupby('season').arrivals.sum()
        composition=actual.groupby(['unit_id','season']).arrivals.sum()/actual.groupby(['unit_id','season']).season.first().map(seasonal_totals)
        assert (composition>0).all(), 'A structural zero FE group needs an explicit separated-sample audit'
        fitted_map=dict(zip(calendar.index,mu_total))
        mu=np.array([composition.loc[row.unit_id,row.season]*fitted_map[row.time] for row in actual.itertuples()])
        with np.load(ROOT/spec.design_npz,allow_pickle=False) as archive:
            assert archive['names'].tolist()==archive['terms'].tolist()==terms and archive['units'].tolist()==actual.unit_id.tolist()
            assert archive['months'].tolist()==actual.month.tolist() and np.array_equal(archive['time'],actual.time) and np.array_equal(archive['input_row'],actual.input_row)
            proof.check(archive['X'],actual[terms].to_numpy(),'Every PPML raw requested design cell',atol=0,rtol=0)
            proof.check(archive['Y'],actual.arrivals.to_numpy(),'Every PPML native response cell',atol=0,rtol=0)
            proof.check(archive['beta'],b,'Native PPML / independent profiled Poisson IRLS coefficients',atol=1e-7,rtol=1e-7)
            proof.check(archive['mu'],mu,'Native PPML / independent Poisson-plus-composition fitted counts',atol=2e-5,rtol=2e-8)
            proof.check(archive['eta'],np.log(mu),'Native PPML linear predictor including absorbed FE',atol=2e-8,rtol=2e-8)
            native_mu=archive['mu'];residual=actual.arrivals.to_numpy()-native_mu
            proof.check(archive['residual'],residual,'Native PPML raw response residuals',atol=1e-8,rtol=1e-12)
            proof.check(archive['weights'],native_mu,'Native Poisson curvature weights',atol=0,rtol=0)
            native_totals=np.array([native_mu[actual.time.eq(time)].sum() for time in calendar.index])
            weights=np.sqrt(native_totals);scale=np.linalg.norm(F*weights[:,None],axis=0);qf,rf=qr(F*weights[:,None]/scale,mode='economic')
            nuisance=solve_triangular(rf,qf.T@(W*weights[:,None]))/scale[:,None]
            xr_time=W-F@nuisance;loc={time:i for i,time in enumerate(calendar.index)}
            xr=np.array([xr_time[loc[time]] for time in actual.time])
            proof.check(archive['xr'],xr,'Independent weighted FE residualization of every PPML weather column',atol=2e-7,rtol=2e-7)
            weighted=xr*np.sqrt(native_mu)[:,None];scales=np.linalg.norm(weighted,axis=0);q,r=qr(weighted/scales,mode='economic');rinv=solve_triangular(r,np.eye(len(terms)))
            bread=(rinv@rinv.T)/np.outer(scales,scales);information=weighted.T@weighted
            proof.check(archive['information'],information,'Independent Poisson efficient information',atol=2e-4,rtol=2e-7)
            proof.check(archive['bread'],bread,'Independent weighted-QR PPML covariance bread',atol=1e-10,rtol=2e-7)
            scores=xr*residual[:,None];influence=scores@bread;hc0=influence.T@influence
            proof.check(archive['scores'],scores,'Independent raw PPML efficient scores',atol=2e-4,rtol=2e-7)
            proof.check(archive['influence'],influence,'Independent PPML full influence vectors',atol=2e-9,rtol=2e-7)
            proof.check(archive['hc0'],hc0,'Independent PPML unscaled HC0 reference',atol=1e-9,rtol=2e-7)
            fe_codes=archive['groups'];assert fe_codes.shape==(len(actual),2)
            assert all(len(set(fe_codes[actual.unit_id.eq(unit)&actual.season.eq(month),0]))==1 for unit in ids for month in seasons)
            assert len(set(fe_codes[:,0]))==len(ids)*len(seasons) and len(set(fe_codes[:,1]))==len(years)
            assert all(len(set(fe_codes[actual.year.eq(year),1]))==1 for year in years)
            K=len(ids)*len(seasons)+len(years)-1+len(terms)
            local_X=np.column_stack([pd.get_dummies(calendar.year,dtype=float),pd.get_dummies(calendar.season,dtype=float),W])
            kt,_=numerical_rank(local_X);t=len(dates);assert spec.n_parameters==K and spec.time_design_rank==kt and spec.time_residual_df==t-kt and spec.full_temporal_span_rank==kt
            ranks=local_rows[local_rows.spec_id.eq(spec.spec_id)];assert len(ranks)==len(ids) and ranks.local_design_rank.eq(kt).all() and archive['local_ranks'].tolist()==[kt]*len(ids)
            methods=[('Calendar HAC',h,t-1) for h in [12,24]]+([('Origin cluster',0,len(ids)-1)] if len(ids)>1 else [])
            assert archive['covariance_labels'].tolist()==[m[0] for m in methods] and archive['bandwidths'].tolist()==[m[1] for m in methods]
            for mi,(method,band,df) in enumerate(methods):
                if method=='Calendar HAC':
                    calendar_score=np.vstack([influence[actual.time.eq(time)].sum(axis=0) for time in calendar.index]);raw_v=calendar_meat(calendar_score,calendar.index.to_numpy(),band)
                    proof.check(archive[f'raw_calendar_covariance_{mi}'],raw_v,'Every PPML raw calendar score covariance',atol=1e-9,rtol=2e-7);factor=t/(t-kt);v=raw_v*factor
                else:
                    cs=np.vstack([influence[actual.unit_id.eq(unit)].sum(axis=0) for unit in sorted(ids)]);factor=len(ids)/(len(ids)-1)*(len(actual)-1)/(len(actual)-K);v=cs.T@cs*factor
                proof.check(archive[f'covariance_{mi}'],v,'Every full PPML calendar/cluster covariance',atol=1e-9,rtol=2e-7)
                proof.check(archive['corrections'][mi],factor,'PPML declared finite correction',atol=1e-12,rtol=1e-12);assert archive['inference_df'][mi]==df
                tally['covariance_cells']+=len(terms)**2
                for name,chosen in [('temperature_effects',['temp_c']),('control_estimates',terms[1:])]:
                    frame=tables[name];part=frame.loc[frame.spec_id.eq(spec.spec_id)&frame.covariance.eq(method)&frame.hac_lags.eq(band)].set_index('term')
                    assert set(part.index)==set(chosen)
                    for term in chosen:
                        j=terms.index(term);row=part.loc[term]
                        proof.inference(row,100*archive['beta'][j],10000*v[j,j],df,10000*hc0[j,j])
                        proof.check(row.raw_native_coefficient,archive['beta'][j],'Raw PPML semi-elasticity /100-log-point export scale',atol=0,rtol=0)
                        if term=='temp_c':proof.check([row.effect_0p1c,row.exact_percent_0p1c],[10*archive['beta'][j],100*np.expm1(.1*archive['beta'][j])],'Exact PPML0.1C expected-count percent and log-point effects',atol=2e-12,rtol=2e-12)
                        tally[name]+=1
                test_row=tables['tests'].loc[tables['tests'].spec_id.eq(spec.spec_id)&tables['tests'].covariance.eq(method)&tables['tests'].hac_lags.eq(band)]
                assert len(test_row)==1;check_joint(test_row.iloc[0],archive['beta'],v,hc0,np.eye(len(terms))[[0]],df,t-1 if method=='Calendar HAC' else len(ids)-1);tally['tests']+=1
            coefficient=tables['coefficients'].loc[tables['coefficients'].spec_id.eq(spec.spec_id)].set_index('term');assert set(coefficient.index)==set(terms)
            proof.check(coefficient.loc[terms,'coefficient'].to_numpy(),archive['beta'],'Every exported native PPML coefficient',atol=0,rtol=0);tally['coefficients']+=len(terms)
            actual_ledger=native_lookup.loc[spec.spec_id]
            for field,array in [('native_response_sha256',archive['Y']),('response_sha256',archive['Y']),('native_coefficients_sha256',archive['beta']),('native_mu_sha256',native_mu),('native_residual_sha256',archive['residual'])]:assert actual_ledger[field]==array_hash(array),(spec.spec_id,field)
            tally['mu_values']+=len(actual);tally['retained_zero_cells']+=zero;tally['canonical_models']+=1
            native_data[spec.spec_id]={key:archive[key] for key in ['beta','mu','time','terms','raw_calendar_covariance_0','raw_calendar_covariance_1','covariance_0','covariance_1','corrections']}
            records.append({'spec_id':spec.spec_id,'n_obs':len(actual),'n_periods':t,'zeros':zero,'native_dropped':0,'independent_IRLS_iterations':iterations,'independent_relative_score':relative_score,'K_time':kt})
    # Strict count-level requests are explicit aliases, not ten extra fits.
    for alias in specs.loc[~specs.native_fit_executed.map(proof._truth)].itertuples():
        source=specs.loc[specs.spec_id.eq(alias.alias_of)].iloc[0];assert source.sample_variant=='event_dates_excluded' and alias.sample_variant==VARIANTS[2]
        assert alias.native_spec_id==source.spec_id and alias.sample_csv==source.sample_csv and alias.design_npz==source.design_npz
        for name in ['temperature_effects','control_estimates','tests']:
            original=tables[name].loc[tables[name].spec_id.eq(source.spec_id)].reset_index(drop=True)
            copy=tables[name].loc[tables[name].spec_id.eq(alias.spec_id)].reset_index(drop=True)
            assert len(copy)==len(original)
            for field in ['coefficient','std_error','p_value','ci_lower','ci_upper','finite_sample_correction','f_statistic','raw_native_coefficient']:
                if field in original:proof.check(copy[field].to_numpy(),original[field].to_numpy(),'Every strict PPML alias equals its sole canonical native model',atol=0,rtol=0)
            tally[name]+=len(copy)
    for name in tables:assert tally[name]==len(tables[name]),('PPML export coverage',name,tally[name],len(tables[name]))
    assert tally['canonical_models']==20 and len(seen_masks)==10 and len(masks)==2250
    aggregation=[]
    for cut in ['complete10','complete52']:
        for control in ['long','full']:
            for variant in VARIANTS[:2]:
                a_id=f'ppml__aggregate__{control}__{variant}';p_id=f'ppml__{cut}__{control}__{variant}';a,p=native_data[a_id],native_data[p_id]
                dates=np.unique(p['time']);assert np.array_equal(a['time'],dates)
                mu=np.array([p['mu'][p['time']==time].sum() for time in dates])
                proof.check(a['beta'],p['beta'],'Complete-partition native PPML shared coefficient aggregation identity',atol=1e-7,rtol=1e-7)
                proof.check(a['mu'],mu,'Complete-partition native PPML expected-count aggregation identity',atol=2e-5,rtol=2e-8)
                for index in [0,1]:
                    proof.check(a[f'raw_calendar_covariance_{index}'],p[f'raw_calendar_covariance_{index}'],'Complete-partition raw calendar covariance aggregation identity',atol=1e-9,rtol=2e-7)
                    proof.check(a[f'covariance_{index}'],p[f'covariance_{index}'],'Complete-partition corrected calendar covariance aggregation identity',atol=1e-9,rtol=2e-7)
                    proof.check(a['corrections'][index],p['corrections'][index],'Complete-partition common working time correction',atol=0,rtol=0)
                aggregation.append({'aggregate_spec_id':a_id,'component_spec_id':p_id,'beta_max_abs_error':float(np.max(np.abs(a['beta']-p['beta']))),'mu_max_abs_error':float(np.max(np.abs(a['mu']-mu))),'scope':'Exact balanced shared-X common Poisson aggregation; no corresponding growth-slope identity'})
    assert len(aggregation)==8
    for path,digest in bound.items():assert proof.sha256(ROOT/path)==digest
    pd.DataFrame(records).to_csv(folder/'verification_fits.csv',index=False);pd.DataFrame(aggregation).to_csv(folder/'verification_aggregation.csv',index=False)
    return {'status':'passed','logical_requests':30,'native_Fepois_models':20,'strict_aliases':10,'counts':tally,'raw_source_audit':raw_audit,
        'zero_retention':'Every intended zero/unit/row retained in all20 observed native models; checked actual sample indices, raw counts and native receipts',
        'count_aggregation_identities':aggregation,'native_fit_log_sha256':proof.sha256(folder/'pyfixest/fit_log.csv'),'protected_bindings_unchanged':len(bound),
        'numerical_checks':dict(proof.ERRORS),'limits':'Count-model common-weather likelihood identity does not apply to sums of log growth or selected nonexhaustive cuts. Marginal count stationarity is not an imposed Poisson admission criterion. No causal/inference-calibration certificate.'}


def _run(branch,function,inputs=None):
    folder=OUT/branch;path=folder/'verification.json';artifact_bindings.archive_passed(path)
    initial={'status':'running','verifier_source_sha256':proof.sha256(Path(__file__))};path.write_text(json.dumps(initial,indent=2)+'\n')
    try:
        checked=artifact_bindings.snapshot(folder)
        proof.ERRORS.clear()
        with threadpool_limits(limits=1):report=function(inputs or independent_inputs())
        artifact_bindings.assert_unchanged(checked);assert artifact_bindings.snapshot(folder)==checked
        report['verified_artifacts_sha256']=checked
        report['artifact_binding_helper_sha256']=proof.sha256(Path(artifact_bindings.__file__))
    except BaseException as error:
        path.write_text(json.dumps({**initial,'status':'failed','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n');raise
    report['verifier_source_sha256']=proof.sha256(Path(__file__));path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Visitor {branch} independent verification passed.',flush=True)
    return report


def verify_visitor_annual():return _run('annual',_verify_annual)
def verify_visitor_ppml():return _run('ppml',_verify_ppml)
def verify_visitor_rigour_companions():
    # Raw-source preparation belongs inside each active proof lifecycle too.
    return {'annual':verify_visitor_annual(),'ppml':verify_visitor_ppml()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['annual','ppml','both'],default='both');args=parser.parse_args()
    if args.branch=='annual':verify_visitor_annual()
    elif args.branch=='ppml':verify_visitor_ppml()
    else:verify_visitor_rigour_companions()
