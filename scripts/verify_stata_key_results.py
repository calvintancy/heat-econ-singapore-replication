"""Separately executed Python compatibility port for native Stata key results.

Official/raw source parsers, QR coefficients and calendar-keyed score products
are independent of the Stata preparation, reghdfe and Mata implementations.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from scipy.linalg import qr,solve_triangular
from scipy.stats import t as student_t, f as f_distribution
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import verify_crisis_sensitivity as raw
from verify_granular import _raw_extended
from stata_capture import verify_package_capture,portable_native_artifacts,optional_transport_artifacts,assert_artifacts_unchanged
from verification_artifacts import archive_passed
OUT=ROOT/'outputs/stata_key_results'


def matrix(path):return np.loadtxt(path,delimiter=',',ndmin=2)
def close(a,b,label,atol=1e-7,rtol=1e-7):raw.check(a,b,label,atol=atol,rtol=rtol)


def source_samples(specs,manifest):
    parents,source_audit=raw.audit_raw_parents();extended,extended_audit=_raw_extended()
    policy=json.loads((OUT/'policy.json').read_text());assert policy==json.loads((ROOT/'notes/crisis_sensitivity_policy.json').read_text())
    checked=set()
    for row in specs:
        path=ROOT/row['sample_csv']
        assert raw.sha256(path)==row['sample_sha256']==manifest['prepared_sample_sha256'][row['sample_csv']]
        if row['sample_csv'] in checked:continue
        checked.add(row['sample_csv']);parent=parents[row['context']].copy()
        frequency='M' if row['frequency']=='monthly' else 'Q';period='month' if frequency=='M' else 'quarter';h=12 if frequency=='M' else 4
        if row['control_set']=='full90':
            for term in ['wind_mps','pm25_10ug']:
                parent[term]=[extended[frequency].get(date,{}).get(term,np.nan) for date in parent[period]]
            valid=[extended[frequency].get(date,{}).get('eligible_90',False) for date in parent[period]]
            parent=parent.loc[valid].copy()
        excluded=set()
        for event in policy['events']:
            excluded.update(month if frequency=='M' else month.asfreq('Q') for month in pd.period_range(event['start'],event['end'],freq='M'))
        dates=set(parent[period])
        if row['sample_variant']=='event_dates_excluded':dates-=excluded
        elif row['sample_variant']=='strict_growth_window_clean':dates={date for date in dates if all(date-lag not in excluded for lag in range(h+1))}
        parent=parent[parent[period].isin(dates)].sort_values([period,'sector_id']).reset_index(drop=True)
        actual=pd.read_csv(path,dtype={'sector_id':str},float_precision='round_trip')
        assert len(actual)==len(parent) and set(actual.period)=={str(date) for date in dates}
        assert actual.row_id.tolist()==list(range(1,len(actual)+1))
        assert actual.sector_id.tolist()==parent.sector_id.tolist() and actual.period.tolist()==parent[period].astype(str).tolist()
        fields=['time','year','season','temp_c','rain_100mm','rh_pct','y_short','y_yoy_nsa','y_yoy']
        if row['control_set']=='full90':fields+=['wind_mps','pm25_10ug']
        close(actual[fields].to_numpy(),parent[fields].to_numpy(),'Frozen Stata input from raw official output/NEA calendars',atol=3e-10,rtol=2e-12)
    return {'distinct_samples':len(checked),'raw_sources':source_audit,'wind_pm':extended_audit}


def singleton_sample(data):
    retained=np.ones(len(data),bool)
    while True:
        active=data.loc[retained]
        a=active.groupby(['industry_id','season']).time.transform('size').eq(1)
        b=active.groupby('year').time.transform('size').eq(1)
        removed=active.index[a|b]
        if not len(removed):break
        retained[removed]=False
    return retained


def explicit_design(data,rhs):
    units=sorted(data.industry_id.unique());seasons=sorted(data.season.unique());years=sorted(data.year.unique())
    columns=[(data.industry_id.eq(unit)&data.season.eq(season)).to_numpy(float) for unit in units for season in seasons]
    columns += [data.year.eq(year).to_numpy(float) for year in years[1:]]
    columns += [data[name].to_numpy(float) for name in rhs]
    x=np.column_stack(columns);norm=np.linalg.norm(x,axis=0);q,r=qr(x/norm,mode='economic')
    assert np.linalg.matrix_rank(r)==x.shape[1]
    inv=solve_triangular(r,q.T)/norm[:,None]
    y=data.y.to_numpy();beta=solve_triangular(r,q.T@y)/norm;residual=y-q@(q.T@y)
    return x,beta,residual,inv


def manual_calendar(z,times,band):
    dates=sorted(set(times));score={int(date):z[times==date].sum(axis=0) for date in dates}
    result=sum((np.outer(value,value) for value in score.values()),start=np.zeros((z.shape[1],z.shape[1])))
    for lag in range(1,band+1):
        cross=sum((np.outer(value,score[date-lag]) for date,value in score.items() if date-lag in score),start=np.zeros_like(result))
        result+=(1-lag/(band+1))*(cross+cross.T)
    return result


def _verify_stata_key_results():
    native_inputs=portable_native_artifacts(OUT,root=ROOT,include_results=False)
    raw.ERRORS.clear();manifest=json.loads((OUT/'preparation_manifest.json').read_text())
    specs=json.loads((OUT/'specifications.json').read_text());assert len(specs)==99
    assert len({row['target_id'] for row in specs})==99 and set(row['structure'] for row in specs)=={'common','H1'}
    for path,digest in manifest['inputs_sha256'].items():assert raw.sha256(ROOT/path)==digest
    assert raw.sha256(OUT/'do/key_results.do')==manifest['do_sha256'] and raw.sha256(OUT/'do/postprocess.mata.do')==manifest['mata_sha256']
    environment_path=OUT/'mcp/environment.json'
    package_capture=verify_package_capture(manifest['packages'],json.loads(environment_path.read_text()))
    assert (OUT/'native/completed.txt').read_text().strip()=='99 target analyses completed'
    log=(OUT/'native/master_stata_output.txt').read_text()
    assert not re.search(r'^r\(\d+\);\s*$',log,re.M), 'Native Stata error code in master log'
    assert len(re.findall(r'^TARGET \d+/99 ',log,re.M))==99
    protocol=json.loads((OUT/'mcp/heat_key_results.json').read_text())
    assert protocol['status']=='completed' and protocol['transport']=='MCP stdio'
    call=protocol['calls'][-1]['response'];body=call.get('structuredContent',{})
    body=json.loads(body['result']) if isinstance(body.get('result'),str) else body
    assert body['success'] and body['data']['rc']==0
    sources=source_samples(specs,manifest)
    runs=pd.read_csv(OUT/'native/runs.csv',skipinitialspace=True)
    assert not runs.duplicated(['target_id','run_variant']).any() and set(runs.target_id)=={row['target_id'] for row in specs}
    assert runs.loc[runs.run_variant.eq('native_default')].target_id.nunique()==99
    effects=[];coefficients=[];contrasts=[];comparisons=[];drop_records=[];full_cov_cells=0
    for spec in specs:
        original=pd.read_csv(ROOT/spec['sample_csv'],dtype={'sector_id':str},float_precision='round_trip')
        sectors=sorted(original.sector_id.unique());mapping={sid:i+1 for i,sid in enumerate(sectors)}
        original['industry_id']=original.sector_id.map(mapping);original['y']=original[spec['outcome']]
        rhs=list(spec['regressor_aliases'])
        for i,term in enumerate(spec['weather_terms']):original[f'x{i:03}']=original[term]-original[term].mean()
        if spec['structure']=='H1':
            for i,sid in enumerate(sectors[1:],1):original[f'd{i:03}']=original.x000*original.sector_id.eq(sid)
        default_mask=singleton_sample(original)
        observed=runs[runs.target_id.eq(spec['target_id'])]
        expected_variants={'native_default'}|({'exact_eligible'} if not default_mask.all() else set())
        assert set(observed.run_variant)==expected_variants
        drop_records.append({'target_id':spec['target_id'],'n_eligible':len(original),'default_dropped':int((~default_mask).sum()),'exact_eligible_refit':not default_mask.all()})
        for native in observed.itertuples():
            folder=OUT/'native'/spec['target_id']/native.run_variant
            saved=pd.read_csv(folder/'sample_residuals.csv',float_precision='round_trip')
            assert saved.row_id.tolist()==original.row_id.tolist() and saved.time.tolist()==original.time.tolist()
            mask=np.ones(len(original),bool) if native.run_variant=='exact_eligible' else default_mask
            assert np.array_equal(saved.__sample.astype(bool),mask)
            data=original.loc[mask].copy();n=len(data);t=data.time.nunique();s=len(sectors);p=len(rhs)
            assert native.n_obs==n and native.n_eligible==len(original) and native.converged==1
            assert native.n_parameters_reported==p and native.df_m==p
            x,beta,residual,inverse=explicit_design(data,rhs);k=x.shape[1]
            time_rank=max(np.linalg.matrix_rank(x[data.industry_id.eq(unit)][:,np.any(x[data.industry_id.eq(unit)]!=0,axis=0)]) for unit in sorted(data.industry_id.unique()))
            time_df=t-time_rank
            b=matrix(folder/'native_beta.csv').ravel()
            close(b,beta[-p:],'Native reghdfe / independent QR coefficients')
            close(saved.loc[mask,'__resid'].to_numpy(),residual,'Native reghdfe / independent QR residuals',atol=2e-8,rtol=1e-8)
            assert saved.loc[~mask,'__resid'].isna().all()
            close(native.r_squared,1-residual@residual/np.sum((data.y-data.y.mean())**2),'Native reghdfe centered R squared',atol=2e-10)
            geometry=matrix(folder/'geometry.csv').ravel();close(geometry[:6],np.array([n,k,p,t,s,k]),'Stata full design rank and dimensions',atol=0,rtol=0)
            assert geometry[6]<1e-10
            z=inverse[-p:,:].T*residual[:,None];hc0=matrix(folder/'hc0.csv')
            close(hc0,z.T@z,'Native Mata / independent QR HC0')
            factor=t/(t-1)*(n-1)/(n-k)
            matrices={}
            if s>1:
                cluster=np.vstack([z[data.industry_id.eq(unit)].sum(axis=0) for unit in sorted(data.industry_id.unique())]);unscaled=cluster.T@cluster
                qnative=s/(s-1)*(n-1)/(n-native.df_m-native.df_a-int(native.df_a_nested>0))
                qproject=s/(s-1)*(n-1)/(n-k)
                vnative=unscaled*qnative
                close(matrix(folder/'cluster_unscaled.csv'),unscaled,'Native Mata / independent sector scores')
                project_cluster=matrix(folder/'project_cluster.csv')
                close(project_cluster,unscaled*qproject,'Native Mata / project-full-K CR1')
                matrices['project_cluster']=(project_cluster,s-1,qproject)
            else:
                qnative=n/(n-native.df_m-native.df_a);qproject=np.nan;vnative=hc0*qnative
            posted_native=matrix(folder/'native_vcov.csv')
            native_unavailable=spec['structure']=='H1' and np.max(np.abs(posted_native))==0 and np.max(np.abs(vnative))>1e-10
            if native_unavailable:
                assert 'variance matrix is nonsymmetric or highly singular' in log
            else:close(posted_native,vnative,'Default reghdfe VCE with native nested-FE df')
            matrices['native']=(posted_native,int(native.df_r),qnative)
            for multiple in [1,2]:
                band=multiple*spec['horizon'];v=manual_calendar(z,data.time.to_numpy(),band)*factor
                native_calendar=matrix(folder/f'calendar_h{multiple}.csv')
                close(native_calendar,v,'Native Mata / actual-calendar Bartlett covariance')
                matrices[f'calendar_h{multiple}']=(native_calendar,t-1,factor)
            L=np.zeros((s if spec['structure']=='H1' else 1,p));L[:,0]=1
            codes=np.arange(1,s+1) if spec['structure']=='H1' else np.zeros(1)
            if spec['structure']=='H1':
                for i in range(1,s):L[i,rhs.index(f'd{i:03}')]=1
            for method,(v,degrees,correction) in matrices.items():
                full_cov_cells+=p*p
                delta=L.copy();delta[:,0]-=1
                expected=np.column_stack([codes,L@b,np.diag(L@v@L.T),np.full(len(L),b[0]),delta@b,
                    np.full(len(L),v[0,0]),np.diag(delta@v@delta.T),delta@v[:,0]])
                close(matrix(folder/f'profiles_{method}.csv'),expected,'Native Stata baseline-plus-industry totals and full covariance')
                close(expected[:,2],expected[:,5]+expected[:,6]+2*expected[:,7],'Total variance includes baseline/delta cross covariance')
                main=method=='project_cluster' if spec['structure']=='common' and s>1 else method=='calendar_h1'
                for j,code in enumerate(codes):
                    sid=sectors[int(code)-1] if code else sectors[0] if s==1 else 'pooled'
                    unit=data if sid=='pooled' else data[data.sector_id.eq(sid)]
                    coef=float(expected[j,1]);variance=float(expected[j,2]);ref=float(L[j]@hc0@L[j]);degenerate=variance<=max(ref,0)*1e-12
                    unsupported=spec['structure']=='H1' and method=='project_cluster'
                    se=np.nan if method=='native' and native_unavailable else math.sqrt(max(variance,0));prob=np.nan if degenerate or unsupported else float(2*student_t.sf(abs(coef/se),degrees))
                    critical=float(student_t.ppf(.975,degrees));meta={k:spec[k] for k in ['target_id','context','outcome','control_set','sample_variant','structure']}
                    effects.append({**meta,'run_variant':native.run_variant,'sector_id':sid,'method':method,'main_inference':main,'coefficient':coef,'std_error':se,
                        'p_value':prob,'ci_lower':np.nan if degenerate or unsupported else coef-critical*se,'ci_upper':np.nan if degenerate or unsupported else coef+critical*se,'effect_0p1c':.1*coef,
                        'inference_df':degrees,'finite_sample_correction':correction,'n_obs':n,'n_periods':t,'full_design_k':k,'time_residual_df':time_df,'mean_y':float(unit.y.mean()),'sd_y':float(unit.y.std()),
                        'baseline':float(expected[j,3]),'industry_delta':float(expected[j,4]),'cov_baseline_delta':float(expected[j,7]),'degenerate':degenerate,
                        'inference_status':'Native Stata rejected highly singular unit-cluster VCE' if method=='native' and native_unavailable else 'Historical unit-cluster covariance diagnostic; single-industry slope inference unsupported' if unsupported else 'Degenerate variance; no test' if degenerate else 'available'})
                for j,name in enumerate(rhs):coefficients.append({'target_id':spec['target_id'],'run_variant':native.run_variant,'method':method,'term':spec['regressor_aliases'][name],'coefficient':float(b[j]),'std_error':np.nan if method=='native' and native_unavailable else float(np.sqrt(max(v[j,j],0))),'inference_df':degrees})
                if spec['structure']=='H1':
                    R=L[1:]-L[[0]];C=R@v@R.T;C=(C+C.T)/2;sv=np.linalg.svd(C,compute_uv=False);reference=np.linalg.norm(R@hc0@R.T,2)
                    cutoff=max(sv[0]*1e-10,reference*1e-12);bound=s-1 if method in ['project_cluster','native'] else t-1
                    rank=min(int(np.sum(sv>cutoff)),bound);q=len(R);available=rank==q and sv[0]>reference*1e-12
                    stat=float((R@b)@np.linalg.solve(C,R@b)) if available else np.nan
                    contrasts.append({'target_id':spec['target_id'],'run_variant':native.run_variant,'method':method,'test':'All industry total heat coefficients equal','restrictions':q,'covariance_rank':rank,'structural_bound':bound,
                        'status':'available' if available else 'unavailable covariance rank/scale','wald_chi2':stat,'p_value':float(f_distribution.sf(stat/q,q,degrees)) if available else np.nan,
                        'condition_number':np.linalg.cond(C),'time_residual_df':time_df,'exploratory_many_restrictions':q>=.8*bound or q>=time_df or time_df<10})
            comparisons.append({'target_id':spec['target_id'],'run_variant':native.run_variant,'n_obs':n,'full_design_k':k,'native_df_m':native.df_m,'native_df_a':native.df_a,'native_df_a_nested':native.df_a_nested,
                'native_factor':qnative,'project_cluster_factor':qproject,'native_over_project_se_ratio':math.sqrt(qnative/qproject) if s>1 and not native_unavailable else np.nan,
                'native_posted_vce_status':'unavailable: Stata rejected highly singular unit-cluster VCE' if native_unavailable else 'available',
                'default_dropped':int((~default_mask).sum())})
        if spec['order']%15==0:print(f'Stata/Python independent parity: {spec["order"]}/99 target analyses',flush=True)
    pd.DataFrame(effects).to_csv(OUT/'estimates.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'control_coefficients.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'joint_tests.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'vce_convention_comparison.csv',index=False)
    pd.DataFrame(drop_records).to_csv(OUT/'singleton_sample_comparison.csv',index=False)
    coverage=[('setup/log/package version','Protocol receipts, native package header/hash and native completion/log checks'),
        ('frozen outcomes, current control coverage and event masks','Independent raw official output/NEA parsers and dictionary calendar support masks'),
        ('centering, temp×industry and absorbed FE','Independent complete LSDV matrix construction and QR'),
        ('reghdfe coefficients/residuals/R²/e(sample)','Separately executed NumPy/SciPy QR, exact singleton recursion and row-set comparison'),
        ('native robust/cluster VCE and nested-FE df','Independent influence HC0/sector score products and native documented finite multiplier'),
        ('Mata design-only inverse and CR1/calendar h/2h','Independent QR inverse plus calendar-keyed Bartlett products preserving holes'),
        ('Mata baseline+industry profile totals','Independent linear contrasts of every full joint covariance'),
        ('esttab native side-by-side tables','Native esttab artifacts retained; equivalent Python flat estimates/inference with labels, means and N'),
        ('probe.do/probe_h1.do','Execution subsets of identical master preparation/reghdfe/Mata sections; durable real MCP receipts')]
    pd.DataFrame([{'stata_file':'do/key_results.do + do/postprocess.mata.do','section':a,'python_port':'scripts/verify_stata_key_results.py','equivalent_behavior':b} for a,b in coverage]).to_csv(OUT/'stata_python_coverage.csv',index=False)
    for path,digest in manifest['inputs_sha256'].items():assert raw.sha256(ROOT/path)==digest
    assert portable_native_artifacts(OUT,root=ROOT,include_results=False)==native_inputs
    verified_native=portable_native_artifacts(OUT,root=ROOT)
    report={'status':'passed','targets':99,'native_runs':len(runs),'default_dropped_rows':int(sum(row['default_dropped'] for row in drop_records)),
        'targets_with_default_singleton_drops':sum(row['default_dropped']>0 for row in drop_records),'distinct_eligible_datasets':sources['distinct_samples'],
        'covariance_cells_checked':full_cov_cells,'profile_effect_rows':len(effects),'source_audit':sources,'numerical_checks':raw.ERRORS,
        'stata_evidence':'Historical native Stata17/IC reghdfe6.13.1 run, real MCP stdio protocol with stored schemas, requests, response and durable native log; no new native estimation by this checker',
        'package_capture':package_capture,'historical_environment_sha256':raw.sha256(environment_path),
        'verified_native_artifacts_sha256':verified_native,'optional_transport_artifacts_sha256':optional_transport_artifacts(OUT,root=ROOT),
        'native_artifact_binding_scope':'Required portable native/MCPJSON/do/Mata/input/native-table/result/environment/privacy-provenance files. Inputs unchanged across the port; regenerated checked result exports bound after creation and rechecked before proof activation. Excludes proof/history/self cycles, local-environment probes and optional unshipped transport logs.',
        'python_evidence':'Separately executed independent QR/score compatibility port; not a replacement for native Stata. Reported coefficient and covariance exports retain the actual native reghdfe/Mata arrays after parity validation.',
        'sample_note':'Native singleton deletion retained; exact-eligible keepsingletons companion exists only where necessary. Project finite-K comparisons use actual native coefficients/residuals.',
        'limits':'Native nested-FE cluster correction differs by design from full-LSDV-K project CR1. H1 sector totals rely on joint calendar covariance, not unit-cluster inference for each single-unit slope. No causal, stationarity or published-paper replication certification.',
        'protected_inputs_unchanged':len(manifest['inputs_sha256']),'verifier_sha256':raw.sha256(Path(__file__))}
    return report


def verify_stata_key_results():
    path=OUT/'verification.json';archive_passed(path)
    initial={'status':'running','verifier_sha256':raw.sha256(Path(__file__))};path.write_text(json.dumps(initial,indent=2)+'\n')
    try:
        with threadpool_limits(limits=1):report=_verify_stata_key_results()
        assert_artifacts_unchanged(report['verified_native_artifacts_sha256'],root=ROOT)
    except BaseException as error:
        path.write_text(json.dumps({**initial,'status':'failed','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n');raise
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Stata/Python parity passed:99 targets,{report['native_runs']} native runs and {report['covariance_cells_checked']:,} covariance cells",flush=True)
    return report


if __name__=='__main__':verify_stata_key_results()
