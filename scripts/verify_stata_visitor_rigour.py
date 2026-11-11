"""Separately executed raw-source/QR port of the 38 native arrival Stata checks.

Uses independent visitor source/calendar parsers and block FWL, never production
arrival preparation, estimation or covariance functions. Exported result points
and covariance matrices are the actual native reghdfe/Mata arrays after checks.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, f as f_distribution
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import verify_crisis_sensitivity as proof
import verify_visitor_rigour as independent
from verify_stata_key_results import matrix,manual_calendar
from stata_capture import verify_environment_sources,portable_native_artifacts,optional_transport_artifacts,assert_artifacts_unchanged
from verification_artifacts import archive_passed
OUT=ROOT/'outputs/stata_key_results/arrivals'


def close(a,b,label,atol=1e-7,rtol=1e-7):proof.check(a,b,label,atol=atol,rtol=rtol)


def protected_bindings(manifest):
    lineage=[]
    for path,digest in manifest['inputs_sha256'].items():
        actual=proof.sha256(ROOT/path)
        if actual==digest:continue
        # Verified-receipt strengthening/coverage extensions may replace an
        # observational proof, while its exact original bytes stay portable.
        assert path.endswith('/verification.json') and path.startswith('outputs/visitor_rigour/')
        pointer=json.loads((ROOT/path).read_text())
        archive=(ROOT/path).parent/'verification_history'/f'{digest}.json'
        if not archive.exists() and path=='outputs/visitor_rigour/diagnostics/verification.json':archive=ROOT/'replication/extensions/review_evidence/visitor_diagnostics_330_receipt.json'
        assert proof.sha256(archive)==digest and json.loads(archive.read_text())['status']=='passed'
        assert pointer['status'] in ['superseded_coverage','passed']
        for name,expected in pointer.get('verified_artifacts_sha256',{}).items():assert proof.sha256(ROOT/name)==expected
        lineage.append({'path':path,'preparation_sha256':digest,'current_sha256':actual,'identical_original_green_receipt':str(archive.relative_to(ROOT))})
    return lineage


def capture_runtime_and_artifacts(protocol,*,resolve_local=None):
    """Bind historical native capture; separately label optional local sources."""
    executable=Path(protocol['server_command']);environment_path=OUT/'mcp/environment.json'
    upstream=json.loads((ROOT/'outputs/stata_key_results/mcp/environment.json').read_text())
    assert environment_path.is_file() and (OUT/'mcp/requirements.txt').is_file(),'Stored-native verification requires packaged historical runtime capture'
    environment=json.loads(environment_path.read_text())
    assert environment['package_sources']==upstream['package_sources'],'Historical arrival/heat package captures disagree'
    local=verify_environment_sources(environment,resolve_local=resolve_local)
    local.update(mcp_server_python_available=executable.is_file(),historical_mcp_server_command=protocol['server_command'],
        historical_environment_sha256=proof.sha256(environment_path),fresh_native_estimation=False,
        scope='Stored-native Python verification only. Packaged historical environment is unchanged. Current local package availability/hash confirmation is optional and separately labelled; an available Python executable does not establish a fresh Stata/ado environment.')
    (OUT/'mcp/local_environment_check.json').write_text(json.dumps(local,indent=2)+'\n')
    files=portable_native_artifacts(OUT,root=ROOT)
    receipt={'status':'captured','native_transport':protocol['transport'],'native_evidence_scope':'Historical MCP/native outputs; no new Stata estimation by this verifier','local_environment_check':local,'source_files':{str(path.relative_to(ROOT)):proof.sha256(path) for path in [ROOT/'scripts/verify_stata_visitor_rigour.py',ROOT/'scripts/stata_visitor_rigour.py',ROOT/'scripts/stata_mcp_client.py',ROOT/'scripts/stata_capture.py']},
        'portable_artifacts_sha256':files,'artifact_sha256':files,'optional_transport_artifacts_sha256':optional_transport_artifacts(OUT,root=ROOT),
        'artifact_binding_scope':'Repository-relative required packed members only. Optional transport stderr is historical/local evidence and not required after restore. Local runtime probes and verification/runtime proof/history/self cycles are excluded.',
        'native_tables':len(list((OUT/'native_tables').glob('*.html')))}
    (OUT/'runtime_proof.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return proof.sha256(OUT/'runtime_proof.json')


def singleton_sample(data,relative_only=False):
    mask=np.ones(len(data),bool)
    while True:
        active=data.loc[mask]
        a=active.groupby(['industry_id','season']).time.transform('size').eq(1)
        b=active.groupby('time' if relative_only else 'year').time.transform('size').eq(1)
        removed=active.index[a|b]
        if not len(removed):return mask
        mask[removed]=False


def source_samples(specs,manifest):
    inputs=independent.independent_inputs()
    samples,losses=independent.reconstruct_samples(ROOT/'outputs/visitor_rigour/monthly',inputs)
    done=set()
    for spec in specs:
        path=ROOT/spec['sample_csv'];assert proof.sha256(path)==spec['sample_sha256']==manifest['prepared_samples_sha256'][spec['sample_csv']]
        assert proof.sha256(ROOT/spec['source_sample_csv'])==spec['source_sample_sha256']
        assert proof.sha256(ROOT/spec['source_design_npz'])==spec['source_design_sha256']
        if spec['sample_csv'] in done:continue
        done.add(spec['sample_csv'])
        parent=samples[spec['panel'],spec['outcome'],spec['control_set'],'own_current_history',spec['sample_variant']]
        parent=parent.sort_values(['time','unit_id']).reset_index(drop=True)
        actual=pd.read_csv(path,dtype={'sector_id':str},float_precision='round_trip')
        assert actual.sector_id.tolist()==parent.unit_id.tolist() and actual.period.tolist()==parent.month.tolist()
        assert actual.row_id.tolist()==list(range(1,len(actual)+1))
        for field in ['time','year','season','arrivals',spec['outcome']]+spec['weather_terms']:
            close(actual[field].to_numpy(),parent[field].to_numpy(),'Native Stata input rebuilt from raw counts/weather/LNY: '+field,atol=3e-10,rtol=2e-12)
    return inputs[-1],len(done),samples


def _verify():
    native_inputs=portable_native_artifacts(OUT,root=ROOT,include_results=False)
    proof.ERRORS.clear();manifest=json.loads((OUT/'preparation_manifest.json').read_text());specs=json.loads((OUT/'specifications.json').read_text())
    assert len(specs)==38 and sum(row['relative_only'] for row in specs)==2
    receipt_lineage=protected_bindings(manifest)
    assert proof.sha256(OUT/'do/key_results.do')==manifest['do_sha256'] and proof.sha256(OUT/'do/postprocess.mata.do')==manifest['mata_sha256']
    assert proof.sha256(ROOT/'scripts/stata_key_results.py')==manifest['shared_mata_source_sha256']
    heat_proof=ROOT/'outputs/stata_key_results/verification.json';heat_digest=manifest['preserved_heat_verification_sha256'];heat_current=proof.sha256(heat_proof)
    if heat_current!=heat_digest:
        historical=heat_proof.parent/'verification_history'/f'{heat_digest}.json'
        assert historical.is_file() and proof.sha256(historical)==heat_digest and json.loads(historical.read_text())['status']=='passed'
        assert json.loads(heat_proof.read_text())['status']=='passed'
    assert (OUT/'native/completed.txt').read_text().strip()=='38 arrival target analyses completed'
    log=(OUT/'native/master_stata_output.txt').read_text();assert not re.search(r'^r\(\d+\);\s*$',log,re.M)
    assert len(re.findall(r'^ARRIVAL TARGET \d+/38 ',log,re.M))==38
    protocol=json.loads((OUT/'mcp/arrival_key_results.json').read_text());assert protocol['status']=='completed' and protocol['transport']=='MCP stdio'
    response=protocol['calls'][-1]['response']['structuredContent'];response=json.loads(response['result']) if isinstance(response.get('result'),str) else response
    assert response['success'] and response['data']['rc']==0
    audit,distinct,samples=source_samples(specs,manifest)
    runs=pd.read_csv(OUT/'native/runs.csv',skipinitialspace=True);assert len(runs)==38 and runs.run_variant.eq('native_default').all()
    assert not runs.duplicated(['target_id','run_variant']).any() and set(runs.target_id)=={row['target_id'] for row in specs}
    effects=[];controls=[];joints=[];comparisons=[];dropped=[];bridges=[];relative_records=[];cells=0
    for spec in specs:
        original=pd.read_csv(ROOT/spec['sample_csv'],dtype={'sector_id':str},float_precision='round_trip');units=sorted(original.sector_id.unique());g=len(units)
        original['industry_id']=original.sector_id.map({sid:i+1 for i,sid in enumerate(units)});original['y']=original[spec['outcome']]
        row=runs.loc[runs.target_id.eq(spec['target_id'])].iloc[0]
        mask=singleton_sample(original,spec['relative_only']);assert mask.all(),'Unexpected native singleton deletion; explicit companion audit needed'
        folder=OUT/'native'/spec['target_id']/'native_default';posted=pd.read_csv(folder/'sample_residuals.csv',float_precision='round_trip')
        assert posted.row_id.tolist()==original.row_id.tolist() and posted.time.tolist()==original.time.tolist() and np.array_equal(posted.__sample.astype(bool),mask)
        assert row.n_obs==row.n_eligible==len(original) and row.n_singletons==0 and row.converged==1
        data=original.copy();n=len(data);t=data.time.nunique();rhs=list(spec['regressor_aliases']);p=len(rhs)
        for i,term in enumerate(spec['weather_terms']):data[f'x{i:03}']=data[term]-data[term].mean()
        if spec['structure']=='H1':
            for i,sid in enumerate(units[1:],1):data[f'd{i:03}']=data.x000*data.sector_id.eq(sid)
        parent=samples[spec['panel'],spec['outcome'],spec['control_set'],'own_current_history',spec['sample_variant']].sort_values(['time','unit_id']).reset_index(drop=True).copy()
        parent['temp_l0']=parent.temp_c;parent['rain_l0']=parent.rain_100mm
        design_spec=SimpleNamespace(horizon=3 if spec['outcome']=='y_mom_nsa' else 12,model_spec='own_current',model=spec['structure'],control_set=spec['control_set'])
        x,names,heat,other,lags=independent.independent_design(parent,design_spec)
        beta,residual,inverse=independent.independent_fit(x,parent[spec['outcome']].to_numpy(),sample=parent,spec=design_spec,names=names)
        name_map={'x000':'temp_l0','x001':'rain_l0','x002':'rh_pct','x003':'lny_window_days','x004':'lny_lag1','x005':'wind_mps','x006':'pm25_10ug'}
        name_map.update({f'd{i:03}':f'temp_l0:C(unit_id)[T.{sid}]' for i,sid in enumerate(units[1:],1)})
        selected=[names.index(name_map[term]) for term in rhs];b=matrix(folder/'native_beta.csv').ravel()
        dropped.append({'target_id':spec['target_id'],'n_eligible':n,'default_dropped':0,'exact_eligible_refit':False})
        if spec['relative_only']:
            assert row.df_m==g-1
            close(b[:len(spec['weather_terms'])],np.zeros(len(spec['weather_terms'])),'Shared weather omitted under actual-date FE',atol=0,rtol=0)
            delta=beta[selected][len(spec['weather_terms']):]
            close(b[len(spec['weather_terms']):],delta,'Native actual-date FE deviations equal original H1 deviations')
            pivot=parent.pivot(index='time',columns='unit_id',values=spec['outcome']).loc[sorted(parent.time.unique()),units]
            local=parent[parent.unit_id.eq(units[0])];z=np.column_stack([(local.season==season).to_numpy(float) for season in sorted(local.season.unique())]+[(local.temp_c-local.temp_c.mean()).to_numpy()])
            for j,unit in enumerate(units[1:]):
                db,dr,di=independent.independent_fit(z,(pivot[unit]-pivot[units[0]]).to_numpy())
                close(db[-1],b[len(spec['weather_terms'])+j],'Relative native date-FE / independent origin-minus-reference regression')
                relative_records.append({'target_id':spec['target_id'],'unit_id':unit,'reference_unit_id':units[0],'native_relative_temperature':b[len(spec['weather_terms'])+j],'difference_regression_temperature':db[-1]})
            slopes=np.r_[0,b[len(spec['weather_terms']):]]
            adjusted=pivot.to_numpy()-local.temp_c.to_numpy()[:,None]*slopes[None,:]
            rr=adjusted-adjusted.mean(axis=1,keepdims=True)
            for season in sorted(local.season.unique()):
                ix=local.season.to_numpy()==season;rr[ix]-=rr[ix].mean(axis=0,keepdims=True)
            expected=np.array([rr[pivot.index.get_loc(date),units.index(unit)] for date,unit in zip(parent.time,parent.unit_id)])
            close(posted.__resid.to_numpy(),expected,'Actual-date FE residual projection',atol=2e-8,rtol=1e-8)
            close(row.r_squared,1-expected@expected/np.sum((data.y-data.y.mean())**2),'Relative native centered R squared',atol=2e-10)
            for j,unit in enumerate(units):effects.append({**spec,'run_variant':'native_default','sector_id':unit,'method':'relative_date_FE_point','main_inference':False,'coefficient':float(slopes[j]),'std_error':np.nan,'p_value':np.nan,'ci_lower':np.nan,'ci_upper':np.nan,'effect_0p1c':float(.1*slopes[j]),'inference_status':'Reference normalization zero; not an estimated absolute response' if not j else 'Relative point identified; absolute national temperature absorbed; no SE/p claim'})
            continue
        k=x.shape[1];rank,cutoff=independent.numerical_rank(x);assert rank==k and row.df_m==p
        close(b,beta[selected],'Native reghdfe / independent block-FWL QR coefficients')
        close(posted.__resid.to_numpy(),residual,'Native reghdfe / independent QR residuals',atol=2e-8,rtol=1e-8)
        close(row.r_squared,1-residual@residual/np.sum((data.y-data.y.mean())**2),'Native centered R squared',atol=2e-10)
        ranks,thresholds,span,span_cutoff=independent.time_ranks(parent,x);kt=max(ranks.values());factor=t/(t-kt)
        geometry=matrix(folder/'geometry.csv').ravel();close(geometry[[0,1,2,3,4,5,7]],[n,k,p,t,g,k,kt],'Native Mata full/local rank and dimensions',atol=0,rtol=0);assert geometry[6]<1e-10
        local_posted=matrix(folder/'time_ranks.csv');close(local_posted[:,1],list(ranks.values()),'Native Mata actual local design ranks',atol=0,rtol=0)
        influence=inverse[selected].T*residual[:,None];hc0=influence.T@influence
        close(matrix(folder/'hc0.csv'),hc0,'Mata / independent HC0 full selected covariance')
        matrices={};qproject=np.nan
        if g>1:
            scores=np.vstack([influence[data.industry_id.eq(i)].sum(axis=0) for i in range(1,g+1)]);unscaled=scores.T@scores
            qnative=g/(g-1)*(n-1)/(n-row.df_m-row.df_a-int(row.df_a_nested>0));qproject=g/(g-1)*(n-1)/(n-k)
            close(matrix(folder/'cluster_unscaled.csv'),unscaled,'Native Mata origin score covariance')
            vproject=matrix(folder/'project_cluster.csv');close(vproject,unscaled*qproject,'Native Mata full-K origin CR1')
            matrices['project_cluster']=(vproject,g-1,qproject)
            expected_native=unscaled*qnative
        else:qnative=n/(n-row.df_m-row.df_a);expected_native=hc0*qnative
        native_v=matrix(folder/'native_vcov.csv');native_unavailable=spec['structure']=='H1' and np.max(abs(native_v))==0 and np.max(abs(expected_native))>1e-10
        if native_unavailable:assert 'variance matrix is nonsymmetric or highly singular' in log
        else:close(native_v,expected_native,'Native reghdfe robust/cluster covariance and nested-FE df')
        matrices['native']=(native_v,int(row.df_r),qnative)
        for multiple in [1,2]:
            v=matrix(folder/f'calendar_h{multiple}.csv');close(v,manual_calendar(influence,data.time.to_numpy(),12*multiple)*factor,'Native Mata actual-calendar Bartlett covariance max-local-rank correction')
            matrices[f'calendar_h{multiple}']=(v,t-1,factor)
        L=np.zeros((g if spec['structure']=='H1' else 1,p));L[:,0]=1
        codes=np.arange(1,g+1) if spec['structure']=='H1' else np.zeros(1)
        if spec['structure']=='H1':
            for j in range(1,g):L[j,rhs.index(f'd{j:03}')]=1
        for method,(v,df,correction) in matrices.items():
            cells+=p*p;delta=L.copy();delta[:,0]-=1
            profile=np.column_stack([codes,L@b,np.diag(L@v@L.T),np.full(len(L),b[0]),delta@b,np.full(len(L),v[0,0]),np.diag(delta@v@delta.T),delta@v[:,0]])
            close(matrix(folder/f'profiles_{method}.csv'),profile,'Native main+interaction totals / all covariance cross terms')
            close(profile[:,2],profile[:,5]+profile[:,6]+2*profile[:,7],'Total variance baseline+delta+twice cross covariance')
            for j,code in enumerate(codes):
                unit=units[int(code)-1] if code else units[0] if g==1 else 'pooled';sub=data if unit=='pooled' else data[data.sector_id.eq(unit)]
                coefficient=float(profile[j,1]);variance=float(profile[j,2]);reference=float(L[j]@hc0@L[j]);degenerate=variance<=max(reference,0)*1e-12
                unsupported=spec['structure']=='H1' and method in ['native','project_cluster'];available=not (degenerate or unsupported or native_unavailable and method=='native')
                se=math.sqrt(max(variance,0)) if not(native_unavailable and method=='native') else np.nan;critical=float(student_t.ppf(.975,df))
                meta={key:spec[key] for key in ['target_id','context','outcome','control_set','sample_variant','structure']}
                effects.append({**meta,'run_variant':'native_default','sector_id':unit,'method':method,'main_inference':method=='calendar_h1','relative_only':False,'coefficient':coefficient,'std_error':se,'p_value':float(2*student_t.sf(abs(coefficient/se),df)) if available else np.nan,'ci_lower':coefficient-critical*se if available else np.nan,'ci_upper':coefficient+critical*se if available else np.nan,'effect_0p1c':.1*coefficient,'inference_df':df,'finite_sample_correction':correction,'n_obs':n,'n_periods':t,'n_origins':g,'full_design_k':k,'maximum_local_rank':kt,'time_residual_df':t-kt,'mean_y':float(sub.y.mean()),'sd_y':float(sub.y.std()),'baseline':float(profile[j,3]),'industry_delta':float(profile[j,4]),'cov_baseline_delta':float(profile[j,7]),'inference_status':'available; approximate calendar t inference' if available else 'Native Stata rejected highly singular origin-cluster VCE' if native_unavailable and method=='native' else 'Individual-origin slope unit-cluster inference unsupported' if unsupported else 'Degenerate variance; no test'})
            for j,name in enumerate(rhs):
                if name.startswith('d'):continue
                variance=float(v[j,j]);degenerate=variance<=max(hc0[j,j],0)*1e-12;available=not(degenerate or native_unavailable and method=='native');se=np.sqrt(max(variance,0)) if available else np.nan;critical=student_t.ppf(.975,df)
                controls.append({**meta,'run_variant':'native_default','method':method,'main_inference':method=='calendar_h1','term':spec['regressor_aliases'][name],'coefficient':b[j],'std_error':se,'p_value':float(2*student_t.sf(abs(b[j]/se),df)) if available else np.nan,'ci_lower':b[j]-critical*se if available else np.nan,'ci_upper':b[j]+critical*se if available else np.nan,'inference_df':df,'time_residual_df':t-kt,'mean_y':float(data.y.mean()),'sd_y':float(data.y.std())})
            if spec['structure']=='H1':
                R=L[1:]-L[[0]];C=R@v@R.T;C=(C+C.T)/2;sv=np.linalg.svd(C,compute_uv=False);reference=np.linalg.norm(R@hc0@R.T,2);cut=max(sv[0]*1e-10,reference*1e-12);bound=g-1 if method in ['native','project_cluster'] else t-1;q=len(R);r=min(int(np.sum(sv>cut)),bound);available=r==q and sv[0]>reference*1e-12
                statistic=max(float((R@b)@np.linalg.solve(C,R@b)),0.) if available else np.nan
                joints.append({**meta,'method':method,'test':'All origin temperature total coefficients equal','restrictions':q,'covariance_rank':r,'structural_rank_bound':bound,'status':'available; nominal asymptotic diagnostic, calibration not established' if available else 'unavailable covariance rank/scale','wald_chi2':statistic,'f_statistic':statistic/q,'p_value':float(f_distribution.sf(statistic/q,q,df)) if available else np.nan,'inference_df':df,'time_residual_df':t-kt,'many_restrictions_warning':q>=.8*bound or q>=t-kt or t-kt<10})
        with np.load(ROOT/spec['source_design_npz'],allow_pickle=False) as archive:
            indexes=[archive['names'].tolist().index(name_map[term]) for term in rhs]
            close(b,archive['beta'][indexes],'Native Stata / source PyFixest fixed-sample coefficients')
            for i in range(2):close(matrices[f'calendar_h{i+1}'][0],archive[f'covariance_{i}'][np.ix_(indexes,indexes)],'Native Mata / source calendar joint covariance')
        comparisons.append({'target_id':spec['target_id'],'native_factor':qnative,'project_origin_factor':qproject,'native_over_fullK_origin_se_ratio':np.sqrt(qnative/qproject) if g>1 and not native_unavailable else np.nan,'native_vce_status':'unavailable highly singular' if native_unavailable else 'available','native_df_m':row.df_m,'native_df_a':row.df_a,'native_df_a_nested':row.df_a_nested,'calendar_factor':factor,'maximum_local_rank':kt,'time_residual_df':t-kt})
        bridges.append({'target_id':spec['target_id'],'source_spec_id':spec['source_spec_id'],'exact_row_set':True,'all_native_coefficients_matched':True,'both_full_selected_calendar_covariances_matched':True})
        if spec['order']%8==0:print(f'Arrival Stata/Python parity: {spec["order"]}/38',flush=True)
    pd.DataFrame(effects).to_csv(OUT/'temperature_effects.csv',index=False);pd.DataFrame(controls).to_csv(OUT/'control_estimates.csv',index=False);pd.DataFrame(joints).to_csv(OUT/'joint_tests.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'vce_convention_comparison.csv',index=False);pd.DataFrame(dropped).to_csv(OUT/'singleton_sample_comparison.csv',index=False);pd.DataFrame(bridges).to_csv(OUT/'source_bridge.csv',index=False);pd.DataFrame(relative_records).to_csv(OUT/'relative_date_FE_identity.csv',index=False)
    coverage=[('raw data and samples','Raw SingStat62series, full-calendar NSA loggrowth, NEA/PM/wind90%, all228HKO lunar rows, static full12month crisis masks'),('reghdfe estimation and FE spaces','Independent block FWL QR with explicit origin×month and sharedyear FE, full joint inverse and actual native row sets'),('native covariance convention','Independent HC0/origin score covariance and actual nested-FE finite multiplier; raw all-zero native H1 preserved'),('Mata primary calendar inference','Calendar-distance12/24 Bartlett score sums and actual max per-origin full-design rank correction'),('industry interaction totals','Full covariance baseline+delta+twice cross term; nominal rank-safe equality tests'),('actual-date FE relative identification','Native deviations and residuals; independent origin-minus-reference regressions; absolute weather absorbed and no SE/p claim'),('side-by-side tables','Native esttab files retained; validated actual native arrays exported separately for primary calendar inference')]
    pd.DataFrame([{'stata_file':'do/key_results.do + do/postprocess.mata.do','section':a,'python_port':'scripts/verify_stata_visitor_rigour.py','equivalent_behavior':b} for a,b in coverage]).to_csv(OUT/'stata_python_coverage.csv',index=False)
    runtime_proof_sha=capture_runtime_and_artifacts(protocol)
    assert portable_native_artifacts(OUT,root=ROOT,include_results=False)==native_inputs
    verified_native=portable_native_artifacts(OUT,root=ROOT)
    assert protected_bindings(manifest)==receipt_lineage
    assert proof.sha256(heat_proof)==heat_current
    return {'status':'passed','targets':38,'regular_targets':36,'native_runs':len(runs),'relative_actual_date_FE_point_checks':2,'relative_difference_regressions':len(relative_records),'native_default_singleton_deletions':0,'distinct_input_datasets':distinct,'full_selected_covariance_cells':cells,'source_bridges':len(bridges),'profile_effect_rows':len(effects),'source_audit':audit,'numerical_checks':proof.ERRORS,'protected_analytical_bindings_unchanged':len(manifest['inputs_sha256'])-len(receipt_lineage),'diagnostic_receipt_lineage':receipt_lineage,'runtime_proof_sha256':runtime_proof_sha,
        'verified_native_artifacts_sha256':verified_native,'optional_transport_artifacts_sha256':optional_transport_artifacts(OUT,root=ROOT),
        'native_artifact_binding_scope':'Required portable native/MCPJSON/do/Mata/input/native-table/result/environment/privacy-provenance files. Inputs unchanged across the port; checked regenerated results bound before proof activation. Transport logs and local runtime probes are optional; proof/history/self cycles excluded.',
        'native_evidence':'Real local MCP stdio Stata17/IC reghdfe and native Mata design-only geometry/calendar scores; actual native results retained after independent validation','inference_limits':'Calendar correction T/(T−max local rank), approximate t(T−1). Native default origin-cluster and full-K CR1 convention separately labelled. H1 individual-origin cluster inference unavailable/unsupported. Date-FE checks identify relative points only. No causal, unit-root or published-paper replication certification.','verifier_sha256':proof.sha256(Path(__file__))}


def verify_stata_visitor_rigour():
    path=OUT/'verification.json';archive_passed(path)
    initial={'status':'running','verifier_sha256':proof.sha256(Path(__file__))};path.write_text(json.dumps(initial,indent=2)+'\n')
    try:
        with threadpool_limits(limits=1):report=_verify()
        assert_artifacts_unchanged(report['verified_native_artifacts_sha256'],root=ROOT)
    except BaseException as error:
        path.write_text(json.dumps({**initial,'status':'failed','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n');raise
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Arrival native Stata / independent Python parity passed:38targets,36 fixed-sample source bridges and2 actual-date-FE point checks.',flush=True);return report


if __name__=='__main__':verify_stata_visitor_rigour()
