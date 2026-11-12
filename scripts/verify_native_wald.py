"""Bounded real-kernel checks for the new chi2 overlay; no native model refits."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from scipy.stats import chi2,f,norm
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import native_wald


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b):np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-8)


def fixtures():
    b=np.array([1.,2.]);v=np.array([[1.,.2],[.2,2.]]);r=np.eye(2);real=native_wald._native_kernel
    with patch.object(native_wald,'_native_kernel',wraps=real) as called:
        result=native_wald.evaluate(b,v,r,q=[.5,.5],df2=20,reference_covariance=v)
        assert result['status']=='available' and called.call_count==1 and result['native_kernel_called']
        d=b-.5;w=float(d@np.linalg.solve(v,d));close(result['wald_chi2'],w);close(result['p_value'],chi2.sf(w,2));close(result['legacy_f_p_value'],f.sf(w/2,2,20))
        assert called.call_args.kwargs['distribution']=='chi2' and 'f_statistic' not in result
        transform=np.array([[1.,.3],[0.,2.]]);inverse=np.linalg.inv(transform)
        alternative=native_wald.evaluate(inverse@b,inverse@v@inverse.T,r@transform,q=[.5,.5],df2=20)
        close(alternative['wald_chi2'],w)
        scaled=native_wald.evaluate(b,4*v,r,q=[.5,.5],df2=20);close(scaled['wald_chi2'],w/4)
        one=native_wald.evaluate(b,v,r[[0]],df2=5);close(one['p_value'],2*norm.sf(abs(b[0]/np.sqrt(v[0,0]))))
        otherdf=native_wald.evaluate(b,v,r[[0]],df2=1000);close(one['p_value'],otherdf['p_value']);assert one['legacy_f_p_value']!=otherdf['legacy_f_p_value']
        invalid=[(b,np.ones((2,2)),r,{}),(b,v,np.array([[1.,0.],[1.,0.]]),{}),
            (b,v,r,{'structural_rank_bound':1}),(b,np.eye(2)*1e-14,r,{'reference_covariance':np.eye(2)}),
            (np.array([np.nan,1.]),v,r,{}),(b,np.full((2,2),np.nan),r,{}),(b,v,np.full((1,2),np.nan),{}),
            (b,v,r,{'q':[0.,np.nan]}),(b,np.diag([-1.,1.]),r,{}),(b,v,np.zeros((0,2)),{})]
        for bb,vv,rr,kwargs in invalid:
            before=called.call_count;bad=native_wald.evaluate(bb,vv,rr,df2=20,**kwargs)
            assert bad['status'].startswith('unavailable:') and bad['p_value'] is None and not bad['native_kernel_called'] and called.call_count==before
        try:native_wald.evaluate(b,np.ones((3,3)),r,df2=20)
        except ValueError:pass
        else:raise AssertionError('Malformed covariance shape accepted')
    return {'actual_native_kernel_calls':called.call_count,'unavailable_guards_before_pinv':len(invalid),'nonzero_null_rhs':True,'original_native_basis_invariance':True,'covariance_scaling':True,'one_constraint_normal_chi2_identity':True,'legacy_df_changes_only_legacy_F':True,'malformed_dimensions_rejected':True}


def verify_native_wald(output_dir=None):
    destination=Path(output_dir) if output_dir is not None else ROOT/'dist/native_wald_adapter_20261112';destination.mkdir(parents=True,exist_ok=True)
    specs=[('heat','outputs/literature_sensitivity/designs/va12__y_yoy_nsa__all_eligible__H1__distributed.npz'),
        ('region','outputs/visitor_rigour/monthly/designs/region__y_mom_nsa__long__all_eligible__H1__own_current.npz'),
        ('country','outputs/visitor_rigour/monthly/designs/country__y_mom_nsa__long__all_eligible__H1__own_current.npz')]
    before={path:sha(ROOT/path) for _,path in specs};rows=[]
    canonical=ROOT/'replication/native_wald/public_api_audit/comparisons.csv'
    archive=canonical if canonical.is_file() else ROOT/'dist/wald_api_audit_20261112/comparisons.csv'
    source_artifacts={}
    if archive.is_file():
        source_artifacts[archive.relative_to(ROOT).as_posix()]=sha(archive)
        if archive==canonical:
            inventory_path=canonical.parent/'inventory.json'
            source_artifacts[inventory_path.relative_to(ROOT).as_posix()]=sha(inventory_path)
            inventory=json.loads(inventory_path.read_text())
            entry=next(row for row in inventory['files'] if row['path']=='comparisons.csv')
            assert entry['sha256']==sha(archive) and entry['bytes']==archive.stat().st_size,'Packaged public-API comparison differs from inventory'
    historical=pd.read_csv(archive).set_index(['design','bandwidth','test']) if archive.is_file() else None
    with threadpool_limits(limits=1):
        fixture=fixtures()
        for label,path in specs:
            with np.load(ROOT/path,allow_pickle=False) as z:
                beta=z['coefficients' if label=='heat' else 'beta'];k=len(beta);components=z['component_positions'];features,units=components.shape
                reference=z['hc0_full'] if 'hc0_full' in z else np.zeros((k,k))
                if 'hc0_full' not in z:reference[np.ix_(z['selected_positions'],z['selected_positions'])]=z['hc0_selected']
                equality=np.zeros((units-1,k))
                for i in range(1,units):equality[i-1,components[:,i]]=1
                allzero=np.eye(k)[components.ravel()];total=np.zeros((1,k));total[0,components[:,0]]=1;total[0,components[:,1]]=1
                for mi,band in enumerate(z['bandwidths'][:2]):
                    covariance=z[f'covariance_{mi}'];df2=int(z['inference_df'][mi])
                    for name,r in [('equality',equality),('all_temperature_zero',allzero),('one_unit_total',total)]:
                        result=native_wald.evaluate(beta,covariance,r,df2=df2,reference_covariance=reference,structural_rank_bound=df2)
                        assert result['status']=='available' and result['native_kernel_called']
                        d=r@beta;w=float(d@np.linalg.solve(r@covariance@r.T,d));close(result['wald_chi2'],w)
                        close(result['p_value'],chi2.sf(w,len(r)));close(result['legacy_f_p_value'],f.sf(w/len(r),len(r),df2))
                        if historical is not None:
                            old=historical.loc[(label,int(band),name)];close(result['wald_chi2'],old.project_W);close(result['p_value'],old.native_public_p);close(result['legacy_f_p_value'],old.project_F_p)
                        rows.append({'design':label,'source_npz':path,'bandwidth':int(band),'test':name,'restrictions':len(r),'direct_W':w,**result})
    assert len(rows)==18 and all(sha(ROOT/path)==value for path,value in before.items())
    assert all(sha(ROOT/path)==digest for path,digest in source_artifacts.items()),'Historical public-API evidence changed during verification'
    pd.DataFrame(rows).to_csv(destination/'comparisons.csv',index=False)
    result={'status':'passed','saved_designs':3,'native_kernel_comparisons':18,'historical_18_row_audit_compared':historical is not None,
        'historical_audit_dependency':'Optional historical three-refit/18-comparison public-API evidence; prefer packaged canonical archive, with local historical fallback only if absent. All18 source restrictions are independently reconstructed without it; no new model refits.',
        'historical_audit_source':archive.relative_to(ROOT).as_posix() if historical is not None else None,
        'source_artifacts_sha256':source_artifacts,
        'fixtures':fixture,'native_backend':native_wald.provenance(),'source_npz_sha256':before,
        'source_sha256':{str(Path(path).relative_to(ROOT)):sha(path) for path in [Path(native_wald.__file__),Path(__file__)]},
        'native_model_refits':0,'old_sources_results_references_changed':False,
        'limits':'Native-kernel arithmetic and guarded input behavior only; supplied saved covariance is not claimed as covariance estimated by PyFixest. No calibration, causality or stationarity certificate.'}
    (destination/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Native Wald adapter passed:18 source-derived kernel comparisons,10 pre-pinv unavailable guards and0model refits.',flush=True);return result


def verify_extension_reconstruction(output_dir=None):
    """All row/shape/source coverage plus bounded original-statistic checks."""
    import native_wald_extensions as extensions
    destination=Path(output_dir) if output_dir is not None else ROOT/'dist/native_wald_adapter_20261112';destination.mkdir(parents=True,exist_ok=True)
    seen=set();sources={};counts={};checked=set();comparisons=[]
    with threadpool_limits(limits=1):
        for item in extensions.collect_extension_tests():
            row=item['row'];key=(item['source_csv'],item['source_row']);assert key not in seen;seen.add(key)
            counts[item['source_csv']]=counts.get(item['source_csv'],0)+1
            for path,digest in item['sources'].items():
                if path in sources:assert sources[path]==digest
                else:sources[path]=digest
            r=item['restriction'];assert r.shape[0]==int(row['restrictions']) and np.isfinite(r).all()
            if pd.notna(row.get('p_value')):
                b,v=item['beta'],item['covariance'];assert b is not None and v is not None and np.isfinite(b).all() and np.isfinite(v).all()
                signature=(item['family'],row.get('test',row.get('null')),row.get('model_spec',row.get('specification')),row.get('outcome'),row.get('covariance'),row.get('hac_lags'))
                if signature not in checked:
                    checked.add(signature);d=r@b;c=r@v@r.T;w=float(d@np.linalg.solve(c,d));legacy=float(row['f_statistic'])*len(r)
                    relative=abs(w-legacy)/max(1,abs(w),abs(legacy));condition=float(np.linalg.cond(c))
                    # Ill-conditioned historical contrasts are recorded rather
                    # than given a false byte-level arithmetic-equivalence claim.
                    assert relative<1e-5 if condition<1e8 else relative<1e-2,(signature,w,legacy,condition)
                    comparisons.append({'source_csv':item['source_csv'],'source_row':item['source_row'],'source_R_W':w,'old_F_times_restrictions':legacy,'scaled_difference':relative,'contrast_condition':condition})
            if sum(counts.values())%25000==0:print('Extension restriction coverage:',sum(counts.values()),flush=True)
    for path,count in counts.items():assert len(pd.read_csv(ROOT/path,usecols=[0]))==count
    assert all(sha(ROOT/path)==digest for path,digest in sources.items())
    pd.DataFrame(comparisons).to_csv(destination/'extension_restriction_checks.csv',index=False)
    result={'status':'passed','current_test_rows':len(seen),'source_csv_counts':counts,'bounded_original_statistic_checks':len(comparisons),
        'maximum_scaled_W_difference':max(row['scaled_difference'] for row in comparisons),'source_sha256':sources,
        'csv_aliases':extensions.CSV_ALIASES,'historical_only':extensions.HISTORICAL_ONLY,'model_refits':0,
        'scope':'Every current source row maps to a real saved-parameter R or an explicit unavailable design; full coverage/shape/source invariance plus one available original-W comparison per family/null/model/outcome/covariance/band signature. Not a full-grid new-Wald computation or calibration certificate.'}
    (destination/'extension_reconstruction.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Extension restriction reconstruction passed:',len(seen),'rows;',len(comparisons),'bounded W checks.',flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path);parser.add_argument('--extension-coverage',action='store_true');arguments=parser.parse_args()
    verify_extension_reconstruction(arguments.output_dir) if arguments.extension_coverage else verify_native_wald(arguments.output_dir)
