"""Selected fixed-X Gaussian diagnostics under the native chi2 Wald reference.

Heat uses an explicitly labelled reference-only recalculation of archived W
statistics. The two arrivals designs replay the archived fixed-X Gaussian
experiment and invoke the actual PyFixest kernel for every available draw.
Neither operation refits an economic model or changes historical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import chi2, f
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import native_wald

ARRIVAL_DESIGNS=[
    ('region__y_mom_nsa__long__all_eligible__H1__own_current',2026111102),
    ('country__y_mom_nsa__long__all_eligible__H1__own_current',2026111103),
]
REPETITIONS=4000
HEAT_DRAWS='outputs/aea_audit_calibration/draw_statistics.csv'
HEAT_SUMMARY='outputs/aea_audit_calibration/summary.json'
ARRIVAL_BASELINE='replication/extensions/review_evidence/arrivals_review_baseline.json'
ARRIVAL_BENCHMARK='replication/extensions/review_evidence/arrivals_selected_benchmarks.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_record(value):
    value=np.asarray(value)
    header={'shape':list(value.shape),'dtype':value.dtype.str}
    return {**header,'sha256':hashlib.sha256(json.dumps(header,sort_keys=True).encode()+np.ascontiguousarray(value).tobytes()).hexdigest()}


def _heat_reference_only(destination):
    frame=pd.read_csv(ROOT/HEAT_DRAWS)
    historical=json.loads((ROOT/HEAT_SUMMARY).read_text())
    assert len(frame)==8000 and frame['available'].eq(True).all()
    assert frame['seed'].eq(20261119).all() and set(frame.restrictions)=={11,60}
    assert frame['denominator_df'].eq(173).all() and set(frame.hac_lags)=={4,8}
    assert frame['design'].eq(historical['design']).all()
    frame=frame.copy()
    frame['archived_W']=frame['f_statistic']*frame['restrictions']
    frame['native_chi2_reference_p']=chi2.sf(frame['archived_W'],frame['restrictions'])
    frame['native_chi2_reference_reject_05']=frame['native_chi2_reference_p']<.05
    frame['recalculation_scope']='Reference-only from archived W; no native kernel call or Gaussian replay'
    rows=[]
    for (band,test,q),group in frame.groupby(['hac_lags','test','restrictions'],sort=True):
        old=next(row for row in historical['rejections'] if row['hac_lags']==band and row['test']==test)
        assert len(group)==2000 and group.draw.nunique()==2000
        assert int(group.reject_05.sum())==old['rejections']
        critical=float(chi2.ppf(.95,int(q)))
        old_critical=float(int(q)*f.ppf(.95,int(q),173))
        assert critical<old_critical
        new_count=int(group['native_chi2_reference_reject_05'].sum())
        assert new_count>=old['rejections']
        rows.append({'hac_lags':int(band),'test':test,'restrictions':int(q),
            'requested_draws':len(group),'available_draws':len(group),
            'legacy_F_rejections_at_05':int(old['rejections']),
            'native_chi2_reference_rejections_at_05':new_count,
            'legacy_F_rejection_fraction':float(old['rejections']/len(group)),
            'native_chi2_reference_rejection_fraction':float(new_count/len(group)),
            'chi2_95pct_critical_W':critical,'legacy_F_95pct_critical_W':old_critical,
            'native_kernel_calls':0,'scope':'Reference-only recalculation of archived W; not a new simulation or native call'})
    path=destination/'calibration_heat_reference_draws.csv'
    frame.to_csv(path,index=False)
    return {'design':historical['design'],'seed':20261119,'repetitions_per_test_band':2000,
        'n_obs':historical['n_obs'],'n_parameters':historical['n_parameters'],'n_periods':historical['n_periods'],
        'dgp':'Historical fixed-X iid standard Gaussian outcome errors; all conditional coefficients zero',
        'results':rows,'native_kernel_calls':0,'native_model_refits':0,
        'draws_csv':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        'scope':'Only the reference distribution changes on archived W; original Gaussian draws, covariance and rank evidence are not regenerated'}


def _arrival_exact(spec_id,seed,writer):
    relative=f'outputs/visitor_rigour/monthly/designs/{spec_id}.npz'
    baseline=json.loads((ROOT/ARRIVAL_BASELINE).read_text())['files'][relative]
    archived=next(row for row in json.loads((ROOT/ARRIVAL_BENCHMARK).read_text())['designs'] if row['spec_id']==spec_id)
    assert archived['seed']==seed and archived['reps']==REPETITIONS
    with np.load(ROOT/relative,allow_pickle=False) as data:
        for name in data.files:
            assert array_record(data[name])==baseline['arrays'][name],(spec_id,name,'historical array binding')
        x=data['X'];beta=data['beta'];time=data['time'].astype(int)
        positions=data['component_positions'];names=data['names'].tolist()
        assert positions.shape[0]==1 and not any(name.startswith('ar_l') for name in names)
        selected=positions[0,1:].astype(int);q=len(selected);n,k=x.shape
        dates=np.unique(time);t=len(dates)
        assert (n,k,t,q)==(archived['n'],archived['k'],archived['t'],archived['q'])
        # These are simulation OLS maps for a new Gaussian outcome, not a model
        # coefficient estimator substituted for native production estimates.
        scale=np.linalg.norm(x,axis=0)
        orthogonal,triangular=np.linalg.qr(x/scale,mode='reduced')
        influence_map=(orthogonal@solve_triangular(triangular,np.eye(k)[:,selected],trans='T'))/scale[selected]
        order=np.argsort(time,kind='stable');sorted_time=time[order]
        starts=np.flatnonzero(np.r_[True,np.diff(sorted_time)>0])
        distance=np.abs(np.subtract.outer(dates,dates))
        kernels=[np.clip(1-distance/(band+1),0,None) for band in (12,24)]
        factors=[float(value) for value in data['corrections'][:2]]
        assert list(data['bandwidths'][:2])==[12,24]
        observed=[]
        for index in range(2):
            covariance=data[f'covariance_{index}'][np.ix_(selected,selected)]
            observed.append(float(beta[selected]@np.linalg.solve(covariance,beta[selected])/q))
            np.testing.assert_allclose(observed[-1],archived['observed'][index]['f'],atol=1e-8,rtol=1e-8)
        bindings={name:array_record(data[name]) for name in ['X','Y','time','unit_ids','names','component_positions','beta','resid']}
    identity=np.eye(q);rng=np.random.default_rng(seed)
    rows=[{'hac_lags':band,'requested_draws':REPETITIONS,'available_draws':0,'unavailable_draws':0,
        'native_chi2_rejections_at_05':0,'legacy_F_rejections_at_05':0,'F_ge_observed':0,
        'minimum_uncapped_rank':q,'maximum_condition':0.,'maximum_scaled_native_W_vs_direct_error':0.} for band in (12,24)]
    native_calls=0
    for start in range(0,REPETITIONS,200):
        size=min(200,REPETITIONS-start)
        noise=rng.standard_normal((size,n))
        simulated_beta=noise@influence_map
        residual=(noise-(noise@orthogonal)@orthogonal.T)[:,order]
        for index in range(size):
            influence=influence_map[order]*residual[index,:,None]
            scores=np.add.reduceat(influence,starts,axis=0)
            hc0=influence.T@influence
            for method,band in enumerate((12,24)):
                covariance=factors[method]*(scores.T@kernels[method]@scores)
                covariance=(covariance+covariance.T)/2
                result=native_wald.evaluate(simulated_beta[index],covariance,identity,
                    df2=t-1,reference_covariance=hc0,structural_rank_bound=t-1,
                    vcov_type='Fixed-X Gaussian replay supplied actual-calendar covariance')
                out=rows[method]
                native_calls+=int(result['native_kernel_called'])
                out['minimum_uncapped_rank']=min(out['minimum_uncapped_rank'],result['uncapped_covariance_rank'])
                out['maximum_condition']=max(out['maximum_condition'],result['covariance_condition'])
                available=result['status']=='available'
                if available:
                    direct=float(simulated_beta[index]@np.linalg.solve(covariance,simulated_beta[index]))
                    error=abs(result['wald_chi2']-direct)/(1+abs(direct))
                    assert error<1e-8,(spec_id,band,start+index,error)
                    out['maximum_scaled_native_W_vs_direct_error']=max(out['maximum_scaled_native_W_vs_direct_error'],error)
                    out['available_draws']+=1
                    out['native_chi2_rejections_at_05']+=int(result['p_value']<.05)
                    out['legacy_F_rejections_at_05']+=int(result['legacy_f_p_value']<.05)
                    out['F_ge_observed']+=int(result['legacy_f_statistic']>=observed[method])
                else:
                    out['unavailable_draws']+=1
                writer.writerow({'spec_id':spec_id,'seed':seed,'draw':start+index+1,'hac_lags':band,
                    'restrictions':q,'denominator_df_legacy':t-1,'status':result['status'],
                    'native_kernel_called':result['native_kernel_called'],'native_W':result['wald_chi2'],
                    'native_chi2_p':result['p_value'],'legacy_F':result['legacy_f_statistic'],
                    'legacy_F_p':result['legacy_f_p_value'],'covariance_rank':result['covariance_rank'],
                    'covariance_condition':result['covariance_condition']})
        print(f'Native chi2 calibration {spec_id}: {start+size}/{REPETITIONS}',flush=True)
    for result,old in zip(rows,archived['results'],strict=True):
        assert result['available_draws']==old['available'] and result['unavailable_draws']==old['unavailable']==0
        assert result['legacy_F_rejections_at_05']==old['reject_at_nominal_5pct']
        assert result['F_ge_observed']==old['F_ge_observed']
        assert result['native_chi2_rejections_at_05']>=result['legacy_F_rejections_at_05']
        result['native_chi2_rejection_fraction']=result['native_chi2_rejections_at_05']/REPETITIONS
        result['legacy_F_rejection_fraction']=result['legacy_F_rejections_at_05']/REPETITIONS
        result['chi2_95pct_critical_W']=float(chi2.ppf(.95,q))
        result['legacy_F_95pct_critical_W']=float(q*f.ppf(.95,q,t-1))
    assert native_calls==REPETITIONS*2
    return {'spec_id':spec_id,'seed':seed,'repetitions':REPETITIONS,'n_obs':n,'n_parameters':k,
        'n_periods':t,'restrictions':q,'null':'All origin temperature slopes equal: nonreference deviations are zero',
        'dgp':'Independent N(0,1) outcome errors on each saved observation, fixed full X, all true conditional coefficients zero; no AR regressors',
        'finite_sample_corrections':factors,'results':rows,'native_kernel_calls':native_calls,
        'native_model_refits':0,'array_bindings':bindings,
        'scope':'Exact fixed-seed replay with original observation order/batches, QR OLS simulation map, actual-calendar Bartlett covariance and actual native Wald kernel on every available draw; old F counts independently recovered'}


def run_native_wald_calibration(output_dir=None):
    import csv
    destination=Path(output_dir) if output_dir is not None else ROOT/'outputs/native_wald'
    destination.mkdir(parents=True,exist_ok=True)
    receipt=destination/'calibration.json'
    if receipt.exists():
        previous=receipt.read_bytes()
        history=destination/'calibration_history';history.mkdir(exist_ok=True)
        (history/(hashlib.sha256(previous).hexdigest()+'.json')).write_bytes(previous)
    receipt.write_text(json.dumps({'status':'running','native_model_refits':0},indent=2)+'\n')
    try:
        sources=[HEAT_DRAWS,HEAT_SUMMARY,ARRIVAL_BASELINE,ARRIVAL_BENCHMARK,
            'replication/extensions/review_evidence/arrivals_selected_benchmarks.py',
            'scripts/selected_null_calibration.py','scripts/native_wald.py','scripts/native_wald_calibration.py']
        sources += [f'outputs/visitor_rigour/monthly/designs/{spec}.npz' for spec,_ in ARRIVAL_DESIGNS]
        # Capture failures also invalidate the current proof; a historical pass
        # must never remain current when a declared input cannot be opened.
        heat_manifest=json.loads((ROOT/HEAT_SUMMARY).read_text())
        sources += list(heat_manifest['frozen_sources_sha256'])
        before={path:sha(ROOT/path) for path in dict.fromkeys(sources)}
        for path,digest in heat_manifest['frozen_sources_sha256'].items():
            assert before[path]==digest,('Historical heat source changed',path)
        with threadpool_limits(limits=1):
            heat=_heat_reference_only(destination)
            draw_path=destination/'calibration_arrival_draws.csv'
            fields=['spec_id','seed','draw','hac_lags','restrictions','denominator_df_legacy','status',
                'native_kernel_called','native_W','native_chi2_p','legacy_F','legacy_F_p',
                'covariance_rank','covariance_condition']
            with draw_path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                arrivals=[_arrival_exact(spec,seed,writer) for spec,seed in ARRIVAL_DESIGNS]
        assert all(sha(ROOT/path)==digest for path,digest in before.items()),'Calibration input/source mutated during replay'
        result={'status':'passed','native_backend':native_wald.provenance(),'heat_reference_only':heat,
            'arrivals_exact_native_kernel_replay':arrivals,
            'native_kernel_calls':sum(row['native_kernel_calls'] for row in arrivals),
            'native_model_refits':0,'economic_model_results_changed':False,
            'historical_calibration_evidence_changed':False,'source_artifacts_sha256':before,
            'verified_artifacts_sha256':{str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):sha(path)
                for path in [destination/'calibration_heat_reference_draws.csv',draw_path]},
            'claim':'The selected joint tests remain badly miscalibrated in these fixed-design iid Gaussian counterexamples under the nominal native chi2 reference. Switching the reference distribution does not repair this problem.',
            'limits':'Selected designs and nulls only. This Gaussian experiment is not an empirical error law, empirical p-value, causal test, universal restriction-count cutoff, validation of every scalar test, or calibration remedy. Heat is reference-only rebasing, whereas arrivals rerun 4000 draws per design and actually call the native kernel.'}
        receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    except BaseException as error:
        receipt.write_text(json.dumps({'status':'failed','error':repr(error),'native_model_refits':0},indent=2)+'\n')
        raise
    print('Selected native chi2 calibration passed: heat reference-only rebasing; 16000 actual arrivals kernel calls, 0 model refits.',flush=True)
    return result


def verify_calibration_failure_lifecycle():
    """A missing declared input invalidates a prior pass without any replay."""
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix='native-wald-calibration-boundary-') as directory:
        destination=Path(directory)
        prior=b'{"status":"passed","fixture":"historical"}\n'
        (destination/'calibration.json').write_bytes(prior)
        missing=str(destination/'missing-declared-heat-summary.json')
        with patch.dict(globals(),{'HEAT_SUMMARY':missing}):
            try:
                run_native_wald_calibration(destination)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Missing declared calibration source was accepted')
        assert json.loads((destination/'calibration.json').read_text())['status']=='failed'
        assert (destination/'calibration_history'/(hashlib.sha256(prior).hexdigest()+'.json')).read_bytes()==prior
    return {'status':'passed','missing_declared_input_invalidates_current_pass':True,
        'prior_pass_preserved_as_historical_bytes':True,'native_kernel_calls':0,'native_model_refits':0,
        'source_sha256':sha(Path(__file__))}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-lifecycle',action='store_true')
    arguments=parser.parse_args()
    if arguments.check_lifecycle:
        print(json.dumps(verify_calibration_failure_lifecycle(),indent=2))
    else:
        run_native_wald_calibration(arguments.output_dir)
