"""Independent enlarged-nuisance FWL/score audit of24 matched origin-year fits."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from scipy.linalg import qr,solve_triangular
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import verify_crisis_sensitivity as proof
import verification_artifacts as artifact_bindings
import verify_visitor_rigour as independent
from verify_literature_sensitivity import calendar_meat
OUT=ROOT/'outputs/visitor_rigour/origin_year'


def build_design(sample,spec):
    units=sorted(sample.unit_id.unique());months=sorted(sample.season.unique());years=sorted(sample.year.unique());names=[];cols=[]
    for unit in units:
        for month in months:names.append(f'origin_season[{unit},{month}]');cols.append((sample.unit_id.eq(unit)&sample.season.eq(month)).to_numpy(float))
    for unit in units:
        for year in years[1:]:names.append(f'origin_year[{unit},{year}]');cols.append((sample.unit_id.eq(unit)&sample.year.eq(year)).to_numpy(float))
    controls=['rain_l0','rh_pct','lny_window_days','lny_lag1']+(['wind_mps','pm25_10ug'] if spec.control_set=='full' else [])
    for term in controls:names.append(term);cols.append((sample[term]-sample[term].mean()).to_numpy())
    heat=(sample.temp_l0-sample.temp_l0.mean()).to_numpy();names.append('temp_l0');cols.append(heat)
    for unit in units[1:]:names.append(f'temp_l0:C(unit_id)[T.{unit}]');cols.append(heat*sample.unit_id.eq(unit))
    return np.column_stack(cols),names,units,months,years,controls


def fwl(sample,x,y,names,units,months,years,controls):
    """Origin-specific month/year/heat projection; small pooled shared controls."""
    ix={unit:np.flatnonzero(sample.unit_id.eq(unit).to_numpy()) for unit in units}
    terms=[f'origin_season[{units[0]},{month}]' for month in months]+[f'origin_year[{units[0]},{year}]' for year in years[1:]]+['temp_l0']
    z=x[np.ix_(ix[units[0]],[names.index(term) for term in terms])];norm=np.linalg.norm(z,axis=0);q,r=qr(z/norm,mode='economic');iz=solve_triangular(r,q.T)/norm[:,None]
    proof.check(iz@z,np.eye(len(terms)),'Origin-year local QR inverse identity',atol=2e-10,rtol=2e-10)
    shared=[names.index(term) for term in controls];c=x[:,shared];cr=np.empty_like(c)
    for pos in ix.values():cr[pos]=c[pos]-z@(iz@c[pos])
    norm=np.linalg.norm(cr,axis=0);q,r=qr(cr/norm,mode='economic');ic=solve_triangular(r,q.T)/norm[:,None]
    for pos in ix.values():ic[:,pos]-=(ic[:,pos]@z)@iz
    proof.check(ic@cr,np.eye(len(shared)),'Origin-year shared-control QR inverse identity',atol=2e-10,rtol=2e-10)
    inverse=np.zeros((len(names),len(y)));inverse[shared]=ic;local={}
    for unit,pos in ix.items():
        local[unit]=-(iz@c[pos])@ic;local[unit][:,pos]+=iz
        for j,month in enumerate(months):inverse[names.index(f'origin_season[{unit},{month}]')]=local[unit][j]
        for j,year in enumerate(years[1:]):inverse[names.index(f'origin_year[{unit},{year}]')]=local[unit][len(months)+j]
    inverse[names.index('temp_l0')]=local[units[0]][-1]
    for unit in units[1:]:inverse[names.index(f'temp_l0:C(unit_id)[T.{unit}]')]=local[unit][-1]-local[units[0]][-1]
    beta=inverse@y;return beta,y-x@beta,inverse


def _verify():
    proof.ERRORS.clear();manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['requested_specs']==manifest['native_fits']==24 and manifest['point_unavailable_requests']==0
    bindings=manifest['protected_sha256']
    for path,digest in bindings.items():assert proof.sha256(ROOT/path)==digest,('Protected binding changed',path)
    assert proof.sha256(ROOT/'scripts/visitor_origin_year.py')==manifest['source_sha256']
    inputs=independent.independent_inputs();parents,losses=independent.reconstruct_samples(ROOT/'outputs/visitor_rigour/monthly',inputs)
    specs=independent.read(OUT/'specifications.csv');assert len(specs)==24 and specs.spec_id.is_unique
    assert set(specs.panel)=={'region','country'} and specs.model.eq('H1').all() and specs.model_spec.eq('own_current').all() and specs.nuisance_spec.eq('origin_year').all()
    tables={name:independent.read(OUT/f'{name}.csv') for name in ['coefficients','temperature_effects','control_estimates','interaction_components','tests']}
    groups={name:{key:part for key,part in frame.groupby('spec_id',sort=False)} for name,frame in tables.items()};counts={name:0 for name in tables};counts.update(native_residual_values=0,full_covariance_cells=0,raw_calendar_covariance_cells=0)
    ledger=independent.read(OUT/'pyfixest/fit_log.csv');native_manifest=json.loads((OUT/'pyfixest/estimator_manifest.json').read_text())
    assert len(ledger)==native_manifest['native_fits_recorded']==24 and native_manifest['fit_log_sha256']==proof.sha256(OUT/'pyfixest/fit_log.csv')
    assert ledger.label.is_unique and ledger.native_class.eq('Feols').all() and ledger.native_module.str.startswith('pyfixest').all()
    assert ledger.native_dropped_rows.eq(0).all() and ledger.native_dropped_coefficients.eq(0).all()
    assert ledger.native_residuals_sha256.eq(ledger.residuals_sha256).all() and ledger.native_outcome_sha256.eq(ledger.outcome_sha256).all()
    records=[]
    for number,spec in enumerate(specs.itertuples(),1):
        source=ROOT/spec.source_sample_csv;assert spec.sample_csv==spec.source_sample_csv
        assert proof.sha256(source)==spec.sample_sha256==spec.source_sample_sha256
        assert proof.sha256(ROOT/spec.source_design_npz)==spec.source_design_sha256
        sample=parents[spec.panel,spec.outcome,spec.control_set,'own_current_history',spec.sample_variant]
        actual=independent.read(source);assert actual.unit_id.tolist()==sample.unit_id.tolist() and actual.month.tolist()==sample.month.tolist()
        x,names,units,months,years,controls=build_design(sample,spec);y=sample[spec.outcome].to_numpy();n,k=x.shape;t=sample.time.nunique();g=len(units)
        # Equal calendars mean the local nuisance+heat geometry is identical.
        beta,resid,inverse=fwl(sample,x,y,names,units,months,years,controls)
        local,cutoffs,span,span_cutoff=independent.time_ranks(sample,x);kt=max(local.values());assert t>kt
        assert spec.n_parameters==k and spec.design_rank==k and spec.n_obs==n and spec.n_periods==t and spec.n_origins==g and spec.status=='estimated'
        assert spec.time_design_rank==kt and spec.time_residual_df==t-kt and json.loads(spec.local_time_design_ranks)==local and spec.full_cross_origin_temporal_span_rank==span
        proof.check([spec.mean_y,spec.sd_y],[y.mean(),y.std(ddof=1)],'Origin-year exact parent Y mean/SD',atol=3e-10,rtol=2e-12)
        proof.check(spec.r_squared,1-resid@resid/np.sum((y-y.mean())**2),'Origin-year native centered R squared',atol=2e-10)
        coeff=groups['coefficients'][spec.spec_id].set_index('term');assert set(coeff.index)==set(names)
        proof.check(coeff.loc[names,'coefficient'],beta,'All origin-year native coefficients including FE',atol=2e-7,rtol=2e-7);counts['coefficients']+=k
        with np.load(ROOT/spec.design_npz,allow_pickle=False) as archive:
            assert all(archive[key].dtype.kind!='O' for key in archive.files)
            assert archive['names'].tolist()==names and archive['unit_ids'].tolist()==sample.unit_id.tolist() and archive['periods'].tolist()==sample.month.tolist()
            proof.check(archive['X'],x,'Every origin-year full design cell',atol=0,rtol=0);proof.check(archive['Y'],y,'Origin-year unchanged parent raw-growth response',atol=0,rtol=0)
            proof.check(archive['beta'],beta,'Origin-year native beta / independent local-block FWL QR',atol=2e-7,rtol=2e-7);proof.check(archive['resid'],resid,'Origin-year native residual / independent FWL',atol=2e-8,rtol=1e-8)
            counts['native_residual_values']+=n
            # The extra preconditioner is invertible design-only geometry. The
            # backend and its response/coefficient receipts remain actual feols.
            q=archive['native_input_design'];transform=archive['coefficient_transform'];theta=archive['native_input_coefficients']
            proof.check(x@transform,q,'Declared original-X to native-Q invertible mapping',atol=3e-10,rtol=3e-10)
            proof.check(q.T@q,np.eye(k),'Native outer QR orthonormal geometry',atol=3e-10,rtol=3e-10)
            proof.check(transform@theta,archive['beta'],'Actual native coefficient back-transform',atol=0,rtol=0)
            proof.check(y-q@theta,archive['resid'],'Actual native Q coefficient/residual receipt',atol=2e-9,rtol=2e-9)
            assert proof.array_sha256(q)==spec.native_input_design_sha256 and proof.array_sha256(x)==spec.original_design_sha256 and proof.array_sha256(transform)==spec.coefficient_transform_sha256
            native=ledger.loc[ledger.label.eq(spec.native_label)];assert len(native)==1;native=native.iloc[0]
            assert native.fit_id==spec.fit_id and native.n_obs==n and native.n_parameters==k and native.outcome==spec.outcome
            assert native.design_sha256==proof.array_sha256(q) and native.outcome_sha256==proof.array_sha256(y)
            assert native.coefficients_sha256==proof.array_sha256(theta) and native.residuals_sha256==proof.array_sha256(archive['resid'])
            selected=archive['selected_positions'].astype(int);assert [names[j] for j in selected]==controls+['temp_l0']+[f'temp_l0:C(unit_id)[T.{unit}]' for unit in units[1:]]
            components=archive['component_positions'][0].astype(int);assert components.tolist()==[names.index('temp_l0')]+[names.index(f'temp_l0:C(unit_id)[T.{unit}]') for unit in units[1:]]
            influence=inverse.T*resid[:,None];hc0=influence.T@influence
            proof.check(archive['hc0_full'],hc0,'Every origin-year nuisance-inclusive HC0 covariance cell',atol=2e-7,rtol=2e-7)
            dates=sorted(sample.time.unique());scores=np.vstack([influence[sample.time.eq(date)].sum(axis=0) for date in dates])
            profiles=np.zeros((g,k));profiles[:,names.index('temp_l0')]=1
            for i,unit in enumerate(units[1:],1):profiles[i,names.index(f'temp_l0:C(unit_id)[T.{unit}]')]=1
            assert archive['covariance_labels'].tolist()==['Calendar HAC','Calendar HAC'] and archive['bandwidths'].tolist()==[12,24]
            assert archive['inference_df'].tolist()==[t-1,t-1]
            for mi,band in enumerate([12,24]):
                raw=calendar_meat(scores,np.array(dates),band);v=raw*t/(t-kt)
                proof.check(archive[f'raw_calendar_covariance_{mi}'],raw,'Every origin-year full raw calendar covariance',atol=2e-7,rtol=2e-7)
                proof.check(archive[f'covariance_{mi}'],v,'Every origin-year full corrected calendar covariance',atol=2e-7,rtol=2e-7)
                proof.check(archive['corrections'][mi],t/(t-kt),'Origin-year reference-invariant local-rank finite factor',atol=1e-12,rtol=1e-12)
                counts['full_covariance_cells']+=k*k;counts['raw_calendar_covariance_cells']+=k*k
                def effect(row,R):proof.inference(row,float(R@beta),float(R@v@R),t-1,float(R@hc0@R))
                points=groups['temperature_effects'][spec.spec_id];points=points[points.hac_lags.eq(band)]
                assert len(points)==2*g and points.main_inference.eq(band==12).all()
                for i,unit in enumerate(units):
                    local_sample=sample[sample.unit_id.eq(unit)]
                    for lag in [0,-1]:
                        row=points[points.unit_id.eq(unit)&points.lag.eq(lag)];assert len(row)==1;row=row.iloc[0];effect(row,profiles[i]);counts['temperature_effects']+=1
                        proof.check([row.mean_y,row.sd_y],[local_sample[spec.outcome].mean(),local_sample[spec.outcome].std()], 'Origin-year per-unit Y context',atol=3e-10,rtol=2e-12)
                        if lag==0:
                            main=components[0];delta=components[i]
                            proof.check([row.temp_main,row.interaction_delta],[beta[main],0 if not i else beta[delta]],'Origin-year baseline plus deviation points',atol=2e-7)
                            proof.check([row.var_temp_main,row.var_interaction_delta,row.cov_temp_main_delta],[v[main,main],0 if not i else v[delta,delta],0 if not i else v[main,delta]],'Origin-year baseline/deviation covariance cross term',atol=2e-7)
                control=groups['control_estimates'][spec.spec_id];control=control[control.hac_lags.eq(band)].set_index('term');assert set(control.index)==set(controls)
                for term in controls:effect(control.loc[term],np.eye(k)[names.index(term)]);counts['control_estimates']+=1
                comp=groups['interaction_components'][spec.spec_id];comp=comp[comp.hac_lags.eq(band)];assert len(comp)==g+1
                for i,unit in enumerate(units):
                    row=comp[comp.unit_id.eq(unit)&~comp.normalization.map(proof._truth)];assert len(row)==1
                    effect(row.iloc[0],np.eye(k)[components[i]]);counts['interaction_components']+=1
                zero=comp[comp.normalization.map(proof._truth)];assert len(zero)==1 and zero.coefficient.iloc[0]==0 and zero[['std_error','ci_lower','ci_upper','p_value']].isna().all().all();counts['interaction_components']+=1
                tests=groups['tests'][spec.spec_id];tests=tests[tests.hac_lags.eq(band)];assert len(tests)==2*g+2
                for i,unit in enumerate(units):
                    rows=tests[tests.unit_id.eq(unit)];assert len(rows)==2
                    for row in rows.itertuples():independent.wald_check(row,beta,v,hc0,profiles[[i]],t-1,t-1);counts['tests']+=1
                difference=np.eye(g)[1:]-np.eye(g)[[0]]
                for name,R in [('All origin cumulative direct effects equal',difference@profiles),('All origin temperature response coefficients jointly zero',profiles)]:
                    row=tests[tests.test.eq(name)&tests.unit_id.eq('all')];assert len(row)==1;independent.wald_check(row.iloc[0],beta,v,hc0,R,t-1,t-1);counts['tests']+=1
        with np.load(ROOT/spec.source_design_npz,allow_pickle=False) as parent:
            # The enlarged nuisance set nests the original shared-year model.
            assert parent['Y'].shape==y.shape and np.array_equal(parent['Y'],y) and np.array_equal(parent['time'],sample.time.to_numpy())
            assert np.sum(resid**2)<=np.sum(parent['resid']**2)+1e-7
            assert spec.time_design_rank==max(independent.time_ranks(sample,parent['X'])[0].values())
        records.append({'spec_id':spec.spec_id,'source_spec_id':spec.source_spec_id,'N':n,'T':t,'K':k,'time_rank':kt,'time_residual_df':t-kt,'exact_parent_sample':True,'nuisance_space_enlarged':True})
        print(f'Independent origin-year FWL audit: {number}/24',flush=True)
    for name in tables:assert counts[name]==len(tables[name]),(name,counts[name],len(tables[name]))
    for _,part in tables['temperature_effects'].groupby(['spec_id','covariance','hac_lags','term','effect'],sort=False):
        adjusted=proof.holm(part.p_value.fillna(1).to_numpy());expected=np.where(part.p_value.notna(),adjusted,np.nan)
        proof.check(part.p_value_holm_profile.to_numpy(),expected,'Every origin-year origin-profile Holm family',atol=2e-12,rtol=2e-12);assert part.multiplicity_family_size.eq(len(part)).all()
    for path,digest in bindings.items():assert proof.sha256(ROOT/path)==digest,('End-of-audit protected binding changed',path)
    pd.DataFrame(records).to_csv(OUT/'verification_fits.csv',index=False)
    return {'status':'passed','requests':24,'native_fits':24,'exact_parent_samples':24,'point_unavailable':0,'counts':counts,'raw_source_audit':inputs[-1],'protected_bindings_unchanged':len(bindings),'native_fit_log_sha256':proof.sha256(OUT/'pyfixest/fit_log.csv'),'numerical_checks':proof.ERRORS,'verifier_source_sha256':proof.sha256(Path(__file__)),'limits':'Matched deterministic-nuisance sensitivity only. No sample or stochastic-control changes; no causal or unit-root certification. Calendar t/F and joint rank tests remain nominal working approximations, without demonstrated general finite-sample calibration.'}


def verify_visitor_origin_year():
    path=OUT/'verification.json';artifact_bindings.archive_passed(path)
    initial={'status':'running','verifier_source_sha256':proof.sha256(Path(__file__))};path.write_text(json.dumps(initial,indent=2)+'\n')
    try:
        checked=artifact_bindings.snapshot(OUT)
        with threadpool_limits(limits=1):report=_verify()
        artifact_bindings.assert_unchanged(checked);assert artifact_bindings.snapshot(OUT)==checked
        report['verified_artifacts_sha256']=checked
        report['artifact_binding_helper_sha256']=proof.sha256(Path(artifact_bindings.__file__))
    except BaseException as error:
        path.write_text(json.dumps({**initial,'status':'failed','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n');raise
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Independent origin-year matched audit passed:24native models.',flush=True);return report


if __name__=='__main__':verify_visitor_origin_year()
