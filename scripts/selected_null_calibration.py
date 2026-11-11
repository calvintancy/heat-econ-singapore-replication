"""Portable fixed-X IID-null diagnostic for one frozen current H1 design.

No Fable sandbox, native estimator, source fetch, analytical output mutation or
model grid is required. This reproduces a selected finite-sample counterexample;
it is not a size estimate for every design or for the observed GDP process.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_name]='1'

import numpy as np
import pandas as pd
import scipy
from scipy.stats import f as f_distribution
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/aea_audit_calibration'
DESIGN='va12__y_yoy_nsa__all_eligible__H1__distributed'
INPUT=ROOT/'outputs/literature_sensitivity/designs'/f'{DESIGN}.npz'
SEED,REPETITIONS,CHUNK=20261119,2000,250
EXPECTED={
 'X':'a7c3dfc68c7ee3c4366924d80695d2ba49fa7e86a8a8f43dd6ad6d38860c1738',
 'Y':'672f04ab69b9578ba3e0b184d1f9ddf037556f1b2ef283a336e26a185f8433a6',
 'time':'fd0013603cb98a1e088b2a7cfbc6b8e046df8043ffb891ca03affeecd7715d4f',
 'periods':'0d17ca6cfe3fd98f8d64a642c41abce3488286df330c481854461338f52c2ef7',
 'units':'be84c41f74e72f92854835c3fee27f84c0ea30dc4c913990acec276deea28419'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_selected_null_calibration():
    OUT.mkdir(parents=True,exist_ok=True)
    summary_path=OUT/'summary.json';summary_path.write_text('{"status":"running"}\n')
    protected=[INPUT,Path(__file__),ROOT/'outputs/literature_sensitivity/specifications.csv',
        ROOT/'outputs/literature_sensitivity/pyfixest/fit_log.csv',ROOT/'outputs/literature_sensitivity/pyfixest/estimator_manifest.json',
        ROOT/'outputs/pyfixest/fit_log.csv',ROOT/'outputs/pyfixest/estimator_manifest.json']
    hashes={str(path.relative_to(ROOT)):sha(path) for path in protected}
    with np.load(INPUT,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in ['X','Y','time','periods','units','component_positions']}
    bindings={}
    for key,expected in EXPECTED.items():
        value=np.ascontiguousarray(arrays[key]);digest=hashlib.sha256(value.tobytes()).hexdigest()
        assert digest==expected,(key,'Frozen diagnostic design changed; reassess the published claim')
        bindings[key]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':digest}
    x=arrays['X'];units=arrays['units'].astype(str)
    dates_all=pd.PeriodIndex(arrays['periods'],freq='Q').asi8
    sectors,dates=np.unique(units),np.unique(dates_all)
    ui,ti=np.searchsorted(sectors,units),np.searchsorted(dates,dates_all)
    n,k=x.shape;g,t=len(sectors),len(dates)
    assert (n,k,g,t)==(2088,157,12,174)
    components=arrays['component_positions'];assert components.shape==(5,12)
    used=np.unique(components.ravel());assert len(used)==60
    zero=np.eye(len(used));equality=np.zeros((g-1,len(used)))
    for unit in range(1,g):
        for feature in range(len(components)):
            equality[unit-1,list(used).index(components[feature,unit])]=1
    restrictions=[('All sector temperature-lag responses jointly zero',zero),('All sector cumulative direct temperature responses equal',equality)]
    factor=t/(t-1)*(n-1)/(n-k)
    kernels={band:np.clip(1-np.abs(dates[:,None]-dates[None,:])/(band+1),0.,None) for band in [4,8]}
    rows=[];rng=np.random.default_rng(SEED)
    with threadpool_limits(limits=1):
        scales=np.linalg.norm(x,axis=0);q,r=np.linalg.qr(x/scales,mode='reduced')
        mapping=np.linalg.solve(r,q.T)/scales[:,None];selected=mapping[used]
        local=x[ui==0];norms=np.linalg.norm(local,axis=0);keep=norms>0
        time_rank=int(np.linalg.matrix_rank(local[:,keep]/norms[keep]));assert t-time_rank==116
        date_rows=[np.flatnonzero(ti==index) for index in range(t)]
        trace_weights={name:np.sum((restriction@selected)**2,axis=0) for name,restriction in restrictions}
        for first in range(0,REPETITIONS,CHUNK):
            size=min(CHUNK,REPETITIONS-first)
            errors=rng.standard_normal((t,g,size))[ti,ui,:]
            beta=mapping@errors;residual=errors-x@beta;estimate=beta[used]
            scores=np.empty((size,t,len(used)))
            for number,index in enumerate(date_rows):scores[:,number,:]=(selected[:,index]@residual[index]).T
            for band,kernel in kernels.items():
                covariance=scores.transpose(0,2,1)@(kernel@scores)*factor
                for name,restriction in restrictions:
                    count=len(restriction);delta=(restriction@estimate).T
                    middle=restriction@covariance@restriction.T;middle=(middle+middle.transpose(0,2,1))/2
                    eigen=np.linalg.eigvalsh(middle)
                    relative=eigen[:,-1]*1e-10
                    trace=trace_weights[name]@(residual**2)
                    # For PSD HC0, its spectral norm <= trace. This proves the
                    # production 1e-12*||R HC0 R'|| gate cannot bind here.
                    assert (eigen[:,0]>relative).all() and (eigen[:,0]>trace*1e-12).all()
                    assert count<=t-1
                    statistic=np.einsum('rq,rq->r',delta,np.linalg.solve(middle,delta[...,None])[...,0])/count
                    p=f_distribution.sf(statistic,count,t-1)
                    for i in range(size):
                        rows.append({'draw':first+i+1,'design':DESIGN,'seed':SEED,'dgp':'Fixed X; iid standard Gaussian outcome errors; all conditional coefficients zero',
                            'covariance':'Driscoll-Kraay','hac_lags':band,'test':name,'restrictions':count,'n_periods':t,'time_residual_df':t-time_rank,
                            'numerator_df':count,'denominator_df':t-1,'f_statistic':float(statistic[i]),'p_value':float(p[i]),'reject_05':bool(p[i]<.05),
                            'available':True,'hc0_zero_gate_inactive_proved':True})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'draw_statistics.csv',index=False)
    counts=frame.groupby(['covariance','hac_lags','test','restrictions'],sort=False).agg(draws=('draw','size'),available=('available','sum'),rejections=('reject_05','sum')).reset_index()
    counts['rejection_rate']=counts.rejections/counts.draws
    expected={(4,60):2000,(8,60):2000,(4,11):712,(8,11):1091}
    for row in counts.itertuples():assert row.draws==REPETITIONS and row.available==REPETITIONS and row.rejections==expected[row.hac_lags,row.restrictions]
    counts.to_csv(OUT/'rejections.csv',index=False)
    assert all(sha(ROOT/name)==value for name,value in hashes.items()),'Frozen source, analytical input or ledger changed'
    result={'status':'passed_selected_fixed_X_null_replay','design':DESIGN,'seed':SEED,'repetitions':REPETITIONS,'chunk':CHUNK,
        'n_obs':n,'n_parameters':k,'n_sectors':g,'n_periods':t,'time_residual_df':t-time_rank,'reference_df':t-1,
        'calendar_score_rank_bound':t-1,'finite_sample_correction':factor,'native_fits_run':0,'array_bindings':bindings,'frozen_sources_sha256':hashes,
        'rejections':counts.to_dict('records'),'matches_independently_replayed_reference_counts':True,
        'numpy_version':np.__version__,'scipy_version':scipy.__version__,
        'outputs_sha256':{name:sha(OUT/name) for name in ['draw_statistics.csv','rejections.csv']},
        'claim':'Selected current fixed-X multi-df Wald tests fail nominal5% calibration under this specified iid Gaussian null despite full numerical rank. Do not use their nominal p-values as calibrated headline inference for this design.',
        'limits':'One frozen design/noise process, not every model or the observed GDP DGP; no new rank guard, universal covariance correction, model selection or causal-validity claim.'}
    summary_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(counts.to_string(index=False),flush=True)
    return result


if __name__=='__main__':run_selected_null_calibration()
