"""Independent direct-solve/chi2 audit of the native-Wald layer and provenance.

Does not invoke the production Wald adapter or the PyFixest Wald kernel.
"""
from __future__ import annotations
import csv,hashlib,io,json,sqlite3,zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.stats import chi2,f
from threadpoolctl import threadpool_limits
from presentation_verification_state import presentation_run,persist

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/native_wald'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def decode(blob):return None if blob is None else np.load(io.BytesIO(zlib.decompress(blob)),allow_pickle=False)

def binding(value):
    if value is None:return None
    a=np.ascontiguousarray(value,dtype=float)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}

@lru_cache(maxsize=4)
def load_model(row):
    metadata=json.loads(row[1]);pointer=metadata.get('source_arrays')
    if pointer:
        with np.load(ROOT/pointer['path'],allow_pickle=False) as z:
            b=z[pointer['beta_key']];v=z[pointer['covariance_key']]
            reference=z[pointer['reference_key']] if pointer.get('reference_key') else None
            if reference is not None and pointer.get('reference_positions') is not None:
                full=np.zeros(v.shape,dtype=float);ix=np.asarray(pointer['reference_positions'],int)
                full[np.ix_(ix,ix)]=reference;reference=full
    else:b,v,reference=decode(row[2]),decode(row[3]),decode(row[4])
    for name,value in [('beta',b),('covariance',v),('reference',reference)]:assert binding(value)==metadata['bindings'][name],('Source coefficient/covariance binding',row[0],name)
    return b,v,reference

def check_number(text,value,label):
    if value is None:assert text=='',(label,text,value)
    elif isinstance(value,bool):assert text==str(value),(label,text,value)
    elif isinstance(value,(int,float)):assert float(text)==float(value),(label,text,value)
    else:assert text==str(value),(label,text,value)

def run():
    with presentation_run(OUT/'verification.json','Independent current native-Wald contrast arithmetic, full source-row coverage and provenance'):
        produced=json.loads((OUT/'production.json').read_text());assert produced['status']=='passed'
        for name,digest in produced['outputs_sha256'].items():assert sha(OUT/name)==digest,('Production artifact changed',name)
        checked_outputs={name:sha(OUT/name) for name in ['tests.csv','contrasts.sqlite','inventory.json','production.json']}
        checked_verifier_sha=sha(Path(__file__))
        assert sha(ROOT/'scripts/native_wald_analysis.py')==produced['producer_sha256']
        assert sha(ROOT/'scripts/native_wald.py')==produced['adapter_sha256']
        for name,digest in produced['algorithm_sources_sha256'].items():assert sha(ROOT/name)==digest,('Algorithm source changed',name)
        inventory=json.loads((OUT/'inventory.json').read_text())
        for name,digest in inventory['input_sources_sha256'].items():assert sha(ROOT/name)==digest,('Fitted/source input changed',name)
        connection=sqlite3.connect('file:'+str(OUT/'contrasts.sqlite')+'?mode=ro',uri=True)
        assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        count=0;called=0;available=0;unavailable=0;models=0;counts=Counter();families=Counter();max_W=0.;max_p=0.;max_legacy=0.
        with threadpool_limits(limits=1), (OUT/'tests.csv').open(newline='') as handle:
            reader=csv.DictReader(handle)
            for text in reader:
                count+=1;test_id=text['test_id'];saved=connection.execute('SELECT model_id,restriction,metadata FROM tests WHERE test_id=?',(test_id,)).fetchone();assert saved
                model_id,Rblob,raw=saved;record=json.loads(raw);R=decode(Rblob)
                for key in ['test_id','source_csv','source_row','status','p_value','p_value_native_chi2','wald_chi2','chi2_df','native_kernel_called','legacy_f_statistic','legacy_df1','legacy_df2','legacy_p_value_F']:
                    check_number(text.get(key,''),record.get(key),key)
                counts[record['source_csv']]+=1;families[record['family']]+=1
                assert record['distribution']=='chi2' and record['p_value']==record['p_value_native_chi2']
                called+=int(record['native_kernel_called'])
                b=v=reference=None
                if model_id:
                    model=connection.execute('SELECT * FROM models WHERE model_id=?',(model_id,)).fetchone();assert model
                    b,v,reference=load_model(model);models+=1
                    assert R is not None and R.shape[1]==len(b) and R.shape[0]==record['chi2_df']
                elif R is not None:
                    assert R.shape[0]==record['chi2_df'] and np.isfinite(R).all()
                if record['status']!='available':
                    unavailable+=1;assert record['p_value'] is None and record['wald_chi2'] is None
                    if not record['old_available']:assert not record['native_kernel_called']
                    continue
                available+=1;assert record['old_available'] and record['native_kernel_called'] and model_id
                assert b is not None and v is not None
                assert R.shape[1]==len(b) and R.shape[0]==record['chi2_df'] and np.linalg.matrix_rank(R)==len(R)
                delta=R@b;C=R@v@R.T;C=(C+C.T)/2
                singular=np.linalg.svd(C,compute_uv=False);scale=0. if reference is None else float(np.linalg.norm(R@reference@R.T,2))
                cutoff=max(float(singular[0])*1e-10,scale*1e-12)
                rank=int(np.sum(singular>cutoff));bound=record.get('structural_rank_bound');bound=len(R) if bound is None else int(bound)
                assert min(rank,max(bound,0))==len(R) and singular[0]>scale*1e-12 and np.linalg.eigvalsh(C)[0]>0
                W=float(delta@np.linalg.solve(C,delta));prob=float(chi2.sf(W,len(R)))
                error=abs(W-record['wald_chi2'])/(1+abs(W));max_W=max(max_W,error)
                assert error<2e-5,('Independent direct-solve W',test_id,error)
                # Ill-conditioned high-dimensional cases are already cautioned.
                # Compare probability with the native reported W separately to
                # avoid masking a wrong reference with a broad W tolerance.
                exact_prob=float(chi2.sf(record['wald_chi2'],len(R)))
                assert abs(record['p_value']-exact_prob)<1e-14
                max_p=max(max_p,abs(record['p_value']-prob))
                old_W=record.get('legacy_wald_chi2')
                if old_W is not None:
                    legacy_error=abs(W-old_W)/(1+abs(old_W));max_legacy=max(max_legacy,legacy_error);assert legacy_error<2e-5
                df=record.get('legacy_df2');oldp=record.get('legacy_p_value_F')
                if df is not None and df>0 and oldp is not None:
                    expected=float(f.sf(W/len(R),len(R),df));assert abs(expected-oldp)<2e-6,('Historical reference mismatch',test_id,expected,oldp)
                assert record['nominal_significant_5pct']==(record['p_value']<.05)
                if count%10000==0:print(f'Independent native-Wald contrast checks: {count:,}',flush=True)
        assert connection.execute('SELECT count(*) FROM tests').fetchone()[0]==count==inventory['test_rows']
        assert called==inventory['native_kernel_calls'] and available+unavailable==count
        assert dict(counts)==inventory['coverage']['covered_sources'] and dict(families)==inventory['family_counts']
        connection.close()
        for name,digest in inventory['input_sources_sha256'].items():assert sha(ROOT/name)==digest,('Input changed during verifier',name)
        for name,digest in checked_outputs.items():assert sha(OUT/name)==digest,('Output changed during verifier',name)
        assert sha(Path(__file__))==checked_verifier_sha, 'Verifier source changed during execution'
        for name,digest in produced['algorithm_sources_sha256'].items():assert sha(ROOT/name)==digest,('Algorithm changed during verifier',name)
        result={'status':'passed','test_rows':count,'available_native_tests':available,'unavailable_tests_retained':unavailable,'native_kernel_calls':called,
                'source_files_covered':len(counts),'family_counts':dict(families),'maximum_scaled_direct_solve_W_difference':max_W,
                'maximum_absolute_p_difference_direct_solve':max_p,'maximum_scaled_W_difference_from_historical_results':max_legacy,
                'files_sha256':checked_outputs,
                'verifier_source_sha256':checked_verifier_sha,'scope':'All requests: available contrast arithmetic and chi2 probabilities, unavailable-row support/status preservation and any saved array bindings, complete source-row counts and CSV-SQLite identity. No economic refits or empirical calibration guarantee.'}
        persist(OUT/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
