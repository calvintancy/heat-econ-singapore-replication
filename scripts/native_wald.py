"""Guarded native PyFixest0.70 chi-squared Wald inference.

This uses the actual post-estimation kernel called by Feols.wald_test. It does
not estimate covariance or coefficients and does not construct a fitted-model
facade. Callers retain their original design/time/unavailable guards. The
previous approximate F reference is returned only under explicit legacy names.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pyfixest
from scipy.stats import f as f_distribution
from pyfixest.estimation.post_estimation.wald import wald_test as _native_kernel

EXPECTED_VERSION='0.70.0'
BACKEND='pyfixest.estimation.post_estimation.wald.wald_test'


def provenance():
    path=Path(inspect.getsourcefile(_native_kernel))
    return {'backend':BACKEND,'pyfixest_version':pyfixest.__version__,
        'kernel_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope':'Actual shared native Wald kernel; supplied coefficient/covariance/restriction arrays, not native covariance estimation or a fresh model fit'}


def evaluate(beta,covariance,restriction,*,q=None,df2=None,reference_covariance=None,
             structural_rank_bound=None,rank_rtol=1e-10,zero_rtol=1e-12,vcov_type='Supplied project covariance'):
    """Evaluate R beta=q with chi2 reference after direct validity guards.

    ``q`` is the null RHS, not the number of restrictions. ``df2`` is metadata
    for the native chi2 result and the denominator for the legacy F comparison;
    it does not change a chi2 p-value. Nonfinite/singular/degenerate requests
    return unavailable without invoking the native pseudoinverse. Malformed
    dimensions are programmer errors. No caller array is changed.
    """
    if pyfixest.__version__!=EXPECTED_VERSION:raise RuntimeError('native_wald requires the audited PyFixest '+EXPECTED_VERSION)
    b=np.asarray(beta,dtype=float).reshape(-1);v=np.asarray(covariance,dtype=float);r=np.atleast_2d(np.asarray(restriction,dtype=float))
    if v.shape!=(len(b),len(b)) or r.ndim!=2 or r.shape[1]!=len(b):raise ValueError('beta, covariance and restriction dimensions disagree')
    if not 0<=rank_rtol<1 or not 0<=zero_rtol<1:raise ValueError('Guard tolerances must lie in [0,1)')
    count=r.shape[0];rhs=np.zeros(count) if q is None else np.asarray(q,dtype=float)
    if rhs.ndim==0:rhs=np.full(count,float(rhs))
    if rhs.shape!=(count,):raise ValueError('Null RHS must be scalar or one value per restriction')
    reference=None if reference_covariance is None else np.asarray(reference_covariance,dtype=float)
    if reference is not None and reference.shape!=v.shape:raise ValueError('Reference covariance dimensions disagree')
    bound=count if structural_rank_bound is None else int(structural_rank_bound)
    if structural_rank_bound is not None and bound!=structural_rank_bound:raise ValueError('Structural rank bound must be integral')
    degrees=None if df2 is None or not np.isfinite(float(df2)) else float(df2)
    result={'status':'unavailable: not evaluated','distribution':'chi2','reference_distribution':'chi2','chi2_df':count,'numerator_df':count,
        'p_value':None,'chi2_statistic':None,'wald_chi2':None,'native_wald_statistic':None,'native_kernel_called':False,
        'legacy_f_statistic':None,'legacy_f_p_value':None,'legacy_f_denominator_df':degrees,'native_df2_metadata':degrees,
        'legacy_f_scope':'Previous approximate F(number of restrictions, declared denominator df) comparison; not the main chi2 reference',
        'wald_backend':BACKEND,'pyfixest_version':pyfixest.__version__,'structural_rank_bound':bound,
        'covariance_rank':None,'uncapped_covariance_rank':None,'restriction_rank':None,'covariance_condition':None,
        'hc0_covariance_scale':None,'zero_covariance_threshold':None,'rank_cutoff':None,
        'calibration_status':'Nominal asymptotic chi2 inference; software arithmetic does not establish finite-sample calibration'}
    def unavailable(reason):result['status']='unavailable: '+reason;return result
    if count==0:return unavailable('empty restriction set')
    if not all(np.isfinite(value).all() for value in [b,v,r,rhs]+([] if reference is None else [reference])):return unavailable('nonfinite coefficient/covariance/restriction input')
    result['restriction_rank']=int(np.linalg.matrix_rank(r))
    if result['restriction_rank']!=count:return unavailable('restriction matrix is not full row rank')
    contrast=r@v@r.T;contrast=(contrast+contrast.T)/2
    singular=np.linalg.svd(contrast,compute_uv=False);scale=0. if reference is None else float(np.linalg.norm(r@reference@r.T,2))
    zero=float(zero_rtol*scale);cutoff=max(float(singular[0])*rank_rtol,zero)
    rank=int(np.sum(singular>cutoff));capped=min(rank,max(bound,0))
    result.update(uncapped_covariance_rank=rank,covariance_rank=capped,covariance_condition=float(np.linalg.cond(contrast)),
        hc0_covariance_scale=scale,zero_covariance_threshold=zero,rank_cutoff=cutoff)
    if singular[0]<=zero:return unavailable('numerically degenerate restriction covariance')
    if capped!=count:return unavailable('rank-deficient restriction covariance or structural score-rank bound')
    if np.linalg.eigvalsh(contrast)[0]<=0:return unavailable('restriction covariance is not positive definite')
    native=_native_kernel(beta_hat=b,vcov=v,R=r,q=rhs,df2=np.nan if degrees is None else degrees,distribution='chi2',vcov_type=vcov_type)
    result['native_kernel_called']=True
    if not np.isfinite(native.wald_statistic) or not np.isfinite(native.pvalue) or native.wald_statistic<0:return unavailable('native kernel returned invalid statistic/probability')
    if native.distribution!='chi2' or native.df1!=count:raise RuntimeError('Native Wald kernel changed its audited result contract')
    result.update(status='available',p_value=float(native.pvalue),chi2_statistic=float(native.stat),
        wald_chi2=float(native.wald_statistic),native_wald_statistic=float(native.wald_statistic),
        legacy_f_statistic=float(native.f_statistic),
        legacy_f_p_value=float(f_distribution.sf(native.f_statistic,count,degrees)) if degrees is not None and degrees>0 else None)
    return result
