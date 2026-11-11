"""Replay two selected Gaussian-null diagnostics from the arrivals review.

No native estimation. This is a conditional fixed-X counterexample to a universal
finite-sample F reference, not a null law for the observed arrivals. It applies
production's declared numerical availability checks independently on every draw.
"""
from pathlib import Path
import argparse,hashlib,json,os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
 os.environ[key]='1'
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import f
from threadpoolctl import threadpool_limits
ROOT=next(path for path in Path(__file__).resolve().parents if (path/'pyproject.toml').exists())
BASELINE=ROOT/'replication/extensions/review_evidence/arrivals_review_baseline.json'
DESIGNS=[('region__y_mom_nsa__long__all_eligible__H1__own_current',2026111102),
 ('country__y_mom_nsa__long__all_eligible__H1__own_current',2026111103)]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_record(a):
 a=np.asarray(a);header={'shape':list(a.shape),'dtype':a.dtype.str}
 return {**header,'sha256':hashlib.sha256(json.dumps(header,sort_keys=True).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()}

def run_one(spec_id,seed,reps=4000):
 relative=f'outputs/visitor_rigour/monthly/designs/{spec_id}.npz'
 baseline=json.loads(BASELINE.read_text())['files'][relative]
 with np.load(ROOT/relative,allow_pickle=False) as z:
  for name in z.files:assert array_record(z[name])==baseline['arrays'][name],(spec_id,name)
  x,y,beta,resid,time=z['X'],z['Y'],z['beta'],z['resid'],z['time'].astype(int)
  names=z['names'].tolist();positions=z['component_positions']
  assert positions.shape[0]==1 and not any(name.startswith('ar_l') for name in names)
  selected=positions[0,1:].astype(int);q=len(selected);n,k=x.shape;dates=np.unique(time);t=len(dates)
  scale=np.linalg.norm(x,axis=0);Q,R=np.linalg.qr(x/scale,mode='reduced')
  P=(Q@solve_triangular(R,np.eye(k)[:,selected],trans='T'))/scale[selected]
  order=np.argsort(time,kind='stable');sorted_time=time[order]
  starts=np.flatnonzero(np.r_[True,np.diff(sorted_time)>0]);calendar=dates-int(dates.min())
  span=int(calendar.max())+1;distance=np.abs(np.subtract.outer(calendar,calendar))
  kernels=[np.clip(1-distance/(h+1),0,None) for h in [12,24]]
  factors=[float(value) for value in z['corrections'][:2]]
  assert list(z['bandwidths'][:2])==[12,24]
  observed=[]
  for j,h in enumerate([12,24]):
   v=z[f'covariance_{j}'][np.ix_(selected,selected)]
   F=float(beta[selected]@np.linalg.solve(v,beta[selected])/q)
   observed.append({'hac_lags':h,'f':F,'nominal_p':float(f.sf(F,q,t-1))})
  bindings={name:array_record(z[name]) for name in ['X','Y','time','unit_ids','names','component_positions','beta','resid']}
  original_factor=factors[0]
 rng=np.random.default_rng(seed);critical=float(f.ppf(.95,q,t-1))
 counts=[{'hac_lags':h,'available':0,'unavailable':0,'reject_at_nominal_5pct':0,
  'F_ge_observed':0,'maximum_condition':0.,'minimum_uncapped_rank':q} for h in [12,24]]
 for start in range(0,reps,200):
  size=min(200,reps-start);noise=rng.standard_normal((size,n));b=noise@P
  e=(noise-(noise@Q)@Q.T)[:,order]
  for row in range(size):
   influence=P[order]*e[row,:,None]
   scores=np.add.reduceat(influence,starts,axis=0)
   hc0=influence.T@influence;reference=float(np.linalg.norm(hc0,2))
   for j,h in enumerate([12,24]):
    v=factors[j]*(scores.T@kernels[j]@scores);v=(v+v.T)/2
    singular=np.linalg.svd(v,compute_uv=False);cutoff=max(singular[0]*1e-10,reference*1e-12)
    rank=int(np.sum(singular>cutoff));available=min(rank,t-1)==q and singular[0]>reference*1e-12
    out=counts[j];out['minimum_uncapped_rank']=min(out['minimum_uncapped_rank'],rank)
    out['maximum_condition']=max(out['maximum_condition'],float(singular[0]/singular[-1]))
    if not available:out['unavailable']+=1;continue
    F=float(b[row]@np.linalg.solve(v,b[row])/q);out['available']+=1
    out['reject_at_nominal_5pct']+=int(F>critical)
    out['F_ge_observed']+=int(F>=observed[j]['f'])
 for row in counts:
  row['rejection_fraction_of_requested']=row['reject_at_nominal_5pct']/reps
  row['rejection_fraction_of_available']=row['reject_at_nominal_5pct']/row['available'] if row['available'] else None
 return {'spec_id':spec_id,'n':n,'k':k,'t':t,'q':q,'reps':reps,'seed':seed,
  'dgp':'Y*=independent N(0,1) errors on every saved observation, fixed X; all true slope coefficients zero; no AR regressors',
  'null':'All origin temperature slopes equal (all nonreference deviations zero)',
  'nominal_reference':f'F({q},{t-1})','nominal_5pct_critical_F':critical,'finite_sample_correction':original_factor,
  'archive_sha256':sha(ROOT/relative),'array_bindings':bindings,'observed':observed,'results':counts}

def run_selected_arrivals_benchmarks(output=None):
 with threadpool_limits(limits=1):rows=[run_one(spec_id,seed) for spec_id,seed in DESIGNS]
 expected=[[(1045,1880),(1948,2018)],[(4000,3998),(4000,3992)]]
 for design,target in zip(rows,expected,strict=True):
  assert [(r['reject_at_nominal_5pct'],r['F_ge_observed']) for r in design['results']]==target
  assert all(r['unavailable']==0 for r in design['results'])
 result={'status':'passed','source_sha256':sha(Path(__file__)),'baseline_sha256':sha(BASELINE),
  'native_fits_executed':0,'designs':rows,
  'claim':'The selected nominal F tests are badly miscalibrated under these explicit fixed-design Gaussian nulls, even after numerical availability checks.',
  'limits':'Two selected MoM designs only. Not the error law of observed arrivals; not a calibrated empirical p-value, a universal q cutoff, a verdict on all scalar tests, or a replacement covariance/reference rule.'}
 output=Path(output) if output else ROOT/'outputs/visitor_rigour/selected_review_benchmarks.json'
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
 result=run_selected_arrivals_benchmarks(parser.parse_args().output)
 print(json.dumps({'status':result['status'],'designs':[{'spec_id':r['spec_id'],'results':r['results']} for r in result['designs']]},indent=2))
