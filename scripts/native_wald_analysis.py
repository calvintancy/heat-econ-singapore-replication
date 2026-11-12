"""Build the current native-PyFixest Wald layer from frozen fitted results.

This performs no economic regression. Every available test supplies real saved
coefficients, joint covariance and an explicit restriction matrix to the native
PyFixest Wald kernel. Old inference/fit/source files are read only.
"""
from __future__ import annotations
import csv,hashlib,io,json,os,sqlite3,zlib
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import native_wald
from native_wald_legacy import collect_legacy_tests,finite
from native_wald_extensions import collect_extension_tests,CSV_ALIASES,HISTORICAL_ONLY
from presentation_verification_state import presentation_run,persist

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/native_wald'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def encoded(value):return json.dumps(clean(value),sort_keys=True,ensure_ascii=False,allow_nan=False)

def pack(array):
    if array is None:return None
    stream=io.BytesIO();np.save(stream,np.asarray(array,dtype=float),allow_pickle=False)
    return zlib.compress(stream.getvalue(),6)

def unpack(blob):return None if blob is None else np.load(io.BytesIO(zlib.decompress(blob)),allow_pickle=False)

def binding(array):
    if array is None:return None
    value=np.ascontiguousarray(array,dtype=float)
    return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}

def load_source_arrays(root,pointer):
    with np.load(root/pointer['path'],allow_pickle=False) as z:
        b=z[pointer['beta_key']];v=z[pointer['covariance_key']]
        reference=z[pointer['reference_key']] if pointer.get('reference_key') else None
        if reference is not None and pointer.get('reference_positions') is not None:
            positions=np.asarray(pointer['reference_positions'],int)
            full=np.zeros_like(v);full[np.ix_(positions,positions)]=reference;reference=full
    return b,v,reference

def source_coverage(counts):
    classified=set(counts); excluded={name:'Historical pre-rank output; final tests.csv is current' for name in HISTORICAL_ONLY}
    excluded['outputs/aea_audit_calibration/draw_statistics.csv']='Historical fixed-Gaussian benchmark draws; new calibration report is separate'
    excluded.update({alias:'Exact byte alias of '+canonical for alias,canonical in CSV_ALIASES.items()})
    for alias,canonical in CSV_ALIASES.items():assert sha(ROOT/alias)==sha(ROOT/canonical),(alias,'Alias bytes differ')
    detected=set()
    for path in (ROOT/'outputs').rglob('*.csv'):
        relative=path.relative_to(ROOT).as_posix()
        if relative.startswith('outputs/native_wald/'):continue
        try:
            with path.open(newline='') as f:header=next(csv.reader(f))
        except (StopIteration,UnicodeError):continue
        if 'restrictions' in header and ('wald_chi2' in header or 'f_statistic' in header):detected.add(relative)
    assert not detected-classified-set(excluded),('Unmapped reported Wald source',sorted(detected-classified-set(excluded)))
    for source,observed in counts.items():
        path=ROOT/source
        if path.suffix=='.csv':
            with path.open(newline='') as f:expected=sum(1 for _ in csv.DictReader(f))
        else:
            value=json.loads(path.read_text());expected=len(value) if isinstance(value,list) else 1+int('sector_cluster' in value)
        assert observed==expected,('Wald source row loss',source,observed,expected)
    return {'detected_joint_csv_sources':len(detected),'covered_sources':dict(counts),'explicit_exclusions':excluded,
            'aliases':CSV_ALIASES,'empty_source_files':['outputs/visitor_arrivals/crisis/joint_tests.json']}

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with presentation_run(OUT/'production.json','Current native PyFixest Wald recalculation from unchanged fitted results'):
        algorithm_sources={name:sha(ROOT/name) for name in ['scripts/native_wald_analysis.py','scripts/native_wald.py','scripts/native_wald_legacy.py','scripts/native_wald_extensions.py']}
        backend_provenance=native_wald.provenance()
        destination=OUT/'contrasts.sqlite';temporary=OUT/f'contrasts.{os.getpid()}.sqlite.tmp'
        if temporary.exists():raise RuntimeError('Interrupted temporary evidence exists: '+str(temporary))
        connection=sqlite3.connect(temporary)
        connection.executescript('CREATE TABLE models(model_id TEXT PRIMARY KEY, metadata TEXT NOT NULL, beta BLOB, covariance BLOB, reference BLOB); CREATE TABLE tests(test_id TEXT PRIMARY KEY, model_id TEXT, restriction BLOB, metadata TEXT NOT NULL);')
        models={};fieldnames={};row_count=0;inputs={};counts=Counter();families=Counter();native_calls=0;new_unavailable=0;max_error=0.
        try:
            with threadpool_limits(limits=1):
                for generator in [collect_legacy_tests(ROOT),collect_extension_tests(ROOT)]:
                    for request in generator:
                        original=clean(request['row']);source=request['source_csv'];index=int(request['source_row'])
                        test_id=hashlib.sha256(f'{source}#{index}'.encode()).hexdigest()[:24]
                        counts[source]+=1;families[request['family']]+=1
                        for name,digest in request['sources'].items():
                            if name in inputs:assert inputs[name]==digest,(name,'Source changed while collecting')
                            inputs[name]=digest
                        old_status=str(original.get('test_status',original.get('status','available')))
                        old_p=original.get('p_value');old_available=finite(old_p) and 'unavailable' not in old_status.lower()
                        b=request['beta'];v=request['covariance'];R=request['restriction'];reference=request.get('reference_covariance')
                        df=request.get('df2');bound=request.get('structural_rank_bound')
                        bound=None if bound is None or not finite(bound) else int(bound)
                        q=int(original.get('restrictions',original.get('numerator_df',1)) or 0)
                        old_W=original.get('wald_chi2');old_F=original.get('f_statistic')
                        if not finite(old_W) and finite(old_F):old_W=float(old_F)*q
                        if not finite(old_F) and finite(old_W) and q:old_F=float(old_W)/q
                        if old_available:
                            assert b is not None and v is not None and R is not None,(source,index,'Available test lacks arrays')
                            assert len(np.atleast_2d(R))==q,(source,index,'Restriction count differs')
                            result=native_wald.evaluate(b,v,R,df2=df,reference_covariance=reference,structural_rank_bound=bound,vcov_type=str(original.get('covariance','Saved project joint covariance')))
                            native_calls+=int(result['native_kernel_called'])
                            if result['status']!='available':new_unavailable+=1
                            if result['status']=='available' and finite(old_W):
                                error=abs(result['wald_chi2']-float(old_W))/(1+abs(float(old_W)));max_error=max(max_error,error)
                                assert error<2e-5,(source,index,'Wald arithmetic discrepancy',error)
                        else:
                            result={'status':old_status if 'unavailable' in old_status.lower() else 'unavailable: prior model/test support or covariance guard',
                                    'p_value':None,'wald_chi2':None,'chi2_statistic':None,'chi2_df':q,'native_kernel_called':False,'distribution':'chi2',
                                    'covariance_rank':original.get('covariance_rank'),'calibration_status':'Unavailable; no native pseudoinverse bypass'}
                        model_id=None
                        if b is not None and v is not None and R is not None:
                            pointer=request.get('source_arrays')
                            binds={name:binding(value) for name,value in [('beta',b),('covariance',v),('reference',reference)]}
                            model_id=hashlib.sha256(encoded({'bindings':binds,'sources':request['sources'],'pointer':pointer}).encode()).hexdigest()[:32]
                            if model_id not in models:
                                if pointer:
                                    loaded=load_source_arrays(ROOT,pointer)
                                    assert all(binding(a)==binding(c) for a,c in zip([b,v,reference],loaded)),(source,index,'Source-array pointer differs from supplied arrays')
                                metadata={'sources':request['sources'],'source_arrays':pointer,'bindings':binds,'basis_names':request.get('basis_names')}
                                connection.execute('INSERT INTO models VALUES(?,?,?,?,?)',(model_id,encoded(metadata),None if pointer else pack(b),None if pointer else pack(v),None if pointer else pack(reference)))
                                models[model_id]=True
                        record={**original,**clean(result),'test_id':test_id,'source_csv':source,'source_row':index,'family':request['family'],
                                'old_status':old_status,'old_available':old_available,'p_value_native_chi2':result.get('p_value'),
                                'legacy_f_statistic':float(old_F) if finite(old_F) else None,'legacy_df1':q,'legacy_df2':df,
                                'legacy_p_value_F':float(old_p) if finite(old_p) else None,'legacy_wald_chi2':float(old_W) if finite(old_W) else None,
                                'covariance_source':'Preserved source covariance; native Wald kernel does not estimate the VCE',
                                'native_kernel':native_wald.BACKEND,'native_pyfixest_version':native_wald.EXPECTED_VERSION,
                                'unavailable_reason':None if result['status']=='available' else result['status'],
                                'nominal_significant_5pct':bool(result.get('p_value') is not None and result['p_value']<.05),
                                'native_model_id':model_id,'original_source_csv':original.get('source_csv')}
                        record['test']=original.get('test',original.get('null','Linear hypothesis'))
                        for ambiguous in ['f_statistic','denominator_df','statistic']:
                            record.pop(ambiguous,None)
                        if 'inference_df' in record:record['original_inference_df']=record.pop('inference_df')
                        connection.execute('INSERT INTO tests VALUES(?,?,?,?)',(test_id,model_id,pack(R),encoded(record)))
                        fieldnames.update(dict.fromkeys(record));row_count+=1
                        if row_count%5000==0:print(f'Native Wald inputs and calls: {row_count:,} rows; {native_calls:,} kernel calls',flush=True)
            empty='outputs/visitor_arrivals/crisis/joint_tests.json'
            assert json.loads((ROOT/empty).read_text())==[], 'Previously empty crisis joint-test source requires coverage'
            inputs[empty]=sha(ROOT/empty)
            coverage=source_coverage(counts)
            for name,digest in inputs.items():assert sha(ROOT/name)==digest,(name,'Fitted result/input changed during native recalculation')
            connection.commit()
            temporary_csv=OUT/f'tests.{os.getpid()}.csv.tmp'
            with temporary_csv.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(fieldnames));writer.writeheader()
                for (metadata,) in connection.execute('SELECT metadata FROM tests ORDER BY rowid'):writer.writerow(json.loads(metadata))
            connection.close();temporary.replace(destination);temporary_csv.replace(OUT/'tests.csv')
            for name,digest in inputs.items():assert sha(ROOT/name)==digest,(name,'Input changed during export')
            for name,digest in algorithm_sources.items():assert sha(ROOT/name)==digest,(name,'Algorithm source changed during execution')
            assert native_wald.provenance()==backend_provenance, 'Installed native Wald kernel changed during execution'
            inventory={'schema_version':1,'status':'produced_pending_independent_verification','test_rows':row_count,'native_kernel_calls':native_calls,
                       'family_counts':dict(families),'coverage':coverage,'input_sources_sha256':inputs,'native_backend':backend_provenance,'algorithm_sources_sha256':algorithm_sources,
                       'models_in_contrast_archive':len(models),'new_unavailable_after_native_guards':new_unavailable,'maximum_scaled_W_difference_from_legacy':max_error,
                       'fields':{'primary':'wald_chi2 / chi2_df / p_value_native_chi2; actual native kernel, supplied covariance',
                                 'legacy':'legacy_f_statistic / legacy_df1 / legacy_df2 / legacy_p_value_F; historical approximate F or equivalent one-constraint t comparison',
                                 'coefficient_tests':'Existing coefficient t-tests and intervals unchanged; explicit one-constraint native Wald p-values are separately labelled'},
                       'limits':'Nominal asymptotic chi2 inference. Native computation does not repair calibration; selected fixed-Gaussian counterexamples remain. No new economic fits or covariance/sample/control changes.'}
            persist(OUT/'inventory.json',inventory)
            proof={'status':'passed','scope':'Produced native-Wald layer and complete source-row coverage; independent verification is separate','rows':row_count,'native_kernel_calls':native_calls,
                   'input_sources_sha256':inputs,'outputs_sha256':{name:sha(OUT/name) for name in ['tests.csv','contrasts.sqlite','inventory.json']},
                   'algorithm_sources_sha256':algorithm_sources,'producer_sha256':algorithm_sources['scripts/native_wald_analysis.py'],'adapter_sha256':algorithm_sources['scripts/native_wald.py']}
            persist(OUT/'production.json',proof)
            print(json.dumps({k:v for k,v in proof.items() if k not in ['input_sources_sha256','outputs_sha256']},indent=2))
        finally:
            try:connection.close()
            except Exception:pass

if __name__=='__main__':run()
