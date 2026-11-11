"""Portable adapter for selected archived independent arrivals probes.

Original probe source bytes are preserved; only filesystem bindings are redirected.
No production estimator or ledger is invoked or modified.
"""
from pathlib import Path
import os,sys,json,hashlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=next(path for path in Path(__file__).resolve().parents if (path/'pyproject.toml').exists())
LANE=ROOT;PROBES=ROOT/'replication/extensions/review_evidence/fable52-arrivals-probes'
sys.path.insert(0,str(PROBES))
import _probe_common
_probe_common.ROOT=ROOT;_probe_common.BASE=ROOT/'outputs/visitor_rigour'
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import probe04_h1_origin_year_design as p04
import probe06_calendar_hac_gaps as p06
import probe12_origin_cluster_common_shock_blindness as p12

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(o):
 if isinstance(o,dict):return {k:clean(v) for k,v in o.items()}
 if isinstance(o,(list,tuple)):return [clean(v) for v in o]
 if isinstance(o,np.ndarray):return clean(o.tolist())
 if isinstance(o,np.generic):return clean(o.item())
 if isinstance(o,float) and not np.isfinite(o):return None
 return o
with threadpool_limits(limits=1):
 specs=pd.read_csv(p06.MONTHLY/'specifications.csv',low_memory=False)
 hac=p06.run('region__y_yoy_nsa__full__event_dates_excluded__common__two_channel','Replay of alternative finite-sample conventions, not a replacement law',specs)
 assert all(v['raw_full_matrix_max_rel_diff_autocovariance_sum']<1e-9 for v in hac['bandwidths'].values())
 specid='country__y_yoy_nsa__long__event_dates_excluded__common__own_current'
 panel=p12.Panel(p12.MONTHLY,specid,False);panel.factorize()
 errors=panel.grid(panel.resid);weights=panel.weights;exposure=weights.sum(axis=0)
 shock=(np.sin(np.arange(panel.t)*.37)+np.cos(np.arange(panel.t)*.11))*errors.mean(axis=0).std()
 index=np.searchsorted(panel.dates,panel.time);newY=panel.y+shock[index]
 newbeta=newY@panel.p_sel;newres=panel.grid(newY-panel.q@(panel.q.T@newY))
 oldscore=np.einsum('gtc,gt->gc',weights,errors);newscore=np.einsum('gtc,gt->gc',weights,newres)
 factor=float(panel.archive['corrections'][panel.labels.index('Origin cluster')])
 oldv=oldscore.T@oldscore*factor;newv=newscore.T@newscore*factor
 basebeta=panel.archive['beta'][panel.temperature]
 added=shock[index];added_residual=added-panel.q@(panel.q.T@added)
 dates=panel.dates-panel.dates.min();kernel=p12.bartlett_matrix(int(dates.max())+1,12)[np.ix_(dates,dates)]
 oldtime=np.einsum('gtc,gt->tc',weights,errors);newtime=np.einsum('gtc,gt->tc',weights,newres)
 cf=float(panel.archive['corrections'][0]);oldcal=cf*oldtime.T@kernel@oldtime;newcal=cf*newtime.T@kernel@newtime
 common={'spec_id':specid,'g':panel.g,'t':panel.t,'raw_X_sha256':sha(p12.MONTHLY/'designs'/f'{specid}.npz'),
  'weight_spread_absolute':float(np.max(np.abs(weights-weights.mean(axis=0)))),
  'beta_before':basebeta,'beta_after':newbeta,'beta_shift':newbeta-basebeta,
  'added_shock_norm':float(np.linalg.norm(added)),
  'shock_component_outside_full_X_norm':float(np.linalg.norm(added_residual)),
  'shock_outside_X_fraction_of_norm':float(np.linalg.norm(added_residual)/np.linalg.norm(added)),
  'calendar_covariance_before':oldcal,'calendar_covariance_after':newcal,
  'calendar_covariance_change_relative':float(np.max(np.abs(newcal-oldcal))/np.max(np.abs(oldcal))),
  'beta_shift_formula_error':float(np.max(np.abs(newbeta-basebeta-shock@exposure))),
  'cluster_covariance_change_absolute':float(np.max(np.abs(newv-oldv))),
  'cluster_covariance_change_relative':float(np.max(np.abs(newv-oldv))/np.max(np.abs(oldv))),
  'dgp':'Deterministic additive common-date perturbation of observed Y; a linear-algebra sensitivity identity, not a null experiment'}
 assert common['beta_shift_formula_error']<1e-8 and common['cluster_covariance_change_relative']<1e-8
 assert common['shock_outside_X_fraction_of_norm']>.1 and common['calendar_covariance_change_relative']>.01
 own=p04.ledger(p04.OY/'specifications.csv');parent=p04.ledger(p04.MONTHLY/'specifications.csv')
 rows=[];mean_error=control_error=0.
 for item in own.to_dict('records'):
  old=parent.loc[parent.spec_id.eq(item['source_spec_id'])].iloc[0]
  with np.load(LANE/old.design_npz) as a,np.load(LANE/item['design_npz']) as b:
   ai=a['component_positions'][0];bi=b['component_positions'][0]
   A=np.r_[a['beta'][ai[0]],a['beta'][ai[0]]+a['beta'][ai[1:]]]
   B=np.r_[b['beta'][bi[0]],b['beta'][bi[0]]+b['beta'][bi[1:]]]
   mean_error=max(mean_error,abs(float(A.mean()-B.mean())))
   controls=json.loads(item['controls']);an=list(a['names']);bn=list(b['names'])
   errors_controls=[abs(a['beta'][an.index(n)]-b['beta'][bn.index(n)]) for n in controls]
   control_error=max(control_error,max(errors_controls))
   rows.append({'spec_id':item['spec_id'],'origins':len(A),'parent_mean_beta':float(A.mean()),'origin_year_mean_beta':float(B.mean()),
    'mean_difference':float(B.mean()-A.mean()),'sum_of_profile_changes':float((B-A).sum()),'profile_sign_changes':int(np.sum(np.sign(A)!=np.sign(B)))})
 assert mean_error<1e-8 and control_error<1e-8
 chosen=own.loc[own.panel.eq('region')&own.outcome.eq('y_yoy_nsa')&own.control_set.eq('full')&own.sample_variant.eq('all_eligible')].iloc[0]
 records=p04.read_rows(LANE/chosen.sample_csv)
 D=p04.decomposition(records,chosen.outcome,chosen.control_set,'year');E=p04.decomposition(records,chosen.outcome,chosen.control_set,'origin_year')
 assert abs(D['mean_slope']-E['mean_slope'])<1e-12
 result={'status':'passed','native_fits_executed':0,'HAC_scaling_conventions':hac,'common_shock_identity':common,
  'origin_year_mean_invariance':{'requests':len(rows),'profiles':sum(row['origins'] for row in rows),'mean_max_abs_error':mean_error,
    'shared_controls_max_abs_error':control_error,'sign_changes':sum(row['profile_sign_changes'] for row in rows),
    'independent_small_mean_equation_example':{'spec_id':chosen.spec_id,'parent_mean':D['mean_slope'],'origin_year_mean':E['mean_slope']},'requests_data':rows},
  'source_bindings':{str(p.relative_to(LANE)):sha(p) for p in [PROBES/'probe04_h1_origin_year_design.py',PROBES/'probe06_calendar_hac_gaps.py',PROBES/'probe12_origin_cluster_common_shock_blindness.py']},
  'limits':'Selected independent arithmetic and exact balanced-static identities; no universal covariance law or empirical-null claim.'}
 out=ROOT/'outputs/visitor_rigour/review_semantics_replay.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':result['status'],'mean_error':mean_error,'control_error':control_error,'sign_changes':result['origin_year_mean_invariance']['sign_changes'],'common_shock':clean(common)},indent=2))
