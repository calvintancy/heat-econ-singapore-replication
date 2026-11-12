"""Real contrast inputs for native-Wald overlays of preserved legacy results.

No model fitting and no alteration of the source estimates, covariance or tests.
Sector profiles are a linear transformation of the saved main/deviation basis.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
KEYS=['spec_id','pair_id','sample_id','model','outcome','sample_variant','covariance','hac_lags']

@lru_cache(maxsize=None)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

@lru_cache(maxsize=64)
def read(path):return pd.read_csv(path,dtype={'sector_id':str,'sector_i':str,'sector_j':str},float_precision='round_trip',low_memory=False)

def select(frame,row,*,skip=()):
    selected=frame
    for key in KEYS:
        if key in skip or key not in frame or key not in row or pd.isna(row[key]):continue
        value=row[key]
        if pd.api.types.is_numeric_dtype(frame[key]):selected=selected[selected[key].eq(float(value))]
        else:selected=selected[selected[key].astype(str).eq(str(value))]
    return selected

def finite(value):
    try:return np.isfinite(float(value))
    except (ValueError,TypeError):return False

def degrees(row):
    for key in ['denominator_df','inference_df','legacy_df2']:
        if finite(row.get(key)):return float(row[key])
    return None

def item(path,index,row,**arrays):
    return {'source_csv':str(path.relative_to(ROOT)),'source_row':int(index),'family':arrays.pop('family','legacy_sector'),
            'row':dict(row),'df2':degrees(row),'structural_rank_bound':row.get('structural_rank_bound',row.get('structural_bound')),
            'sources':{str(path.relative_to(ROOT)):sha(path)},**arrays}

def collect_sector_groups(root=ROOT):
    global ROOT
    ROOT=Path(root)
    folders=[]
    folders += [(p.parent,'sector_slopes.csv','slope_covariances.csv','legacy_heterogeneity') for p in sorted((ROOT/'outputs/heterogeneity').glob('*/*/joint_tests.csv'))]
    folders += [(p.parent,'sector_slopes.csv','slope_covariances.csv','legacy_granular') for p in sorted((ROOT/'outputs/granular_interactions').glob('*/*/joint_tests.csv'))]
    folders += [(ROOT/'outputs/covid_comparison/sector','estimates.csv','slope_covariances.csv','paired_covid'),
                (ROOT/'outputs/annual','sector_estimates.csv','sector_covariances.csv','legacy_annual'),
                (ROOT/'outputs/crisis_sensitivity','sector_estimates.csv','sector_covariances.csv','crisis'),
                (ROOT/'outputs/crisis_sensitivity/annual','sector_estimates.csv','sector_covariances.csv','annual_crisis'),
                (ROOT/'outputs/complete_va_controls','sector_estimates.csv','sector_covariances.csv','complete_va'),
                (ROOT/'outputs/clustered_sector_sensitivity','sector_slopes.csv','slope_covariances.csv','cluster_sensitivity')]
    for folder,slopes_name,cov_name,family in folders:
        path=folder/'joint_tests.csv';slopes_path=folder/slopes_name;cov_path=folder/cov_name
        table=read(path);slopes=read(slopes_path);cov=read(cov_path)
        value_column=next((name for name in ['covariance_value','value','covariance'] if name in cov),None)
        assert value_column
        sources={str(p.relative_to(ROOT)):sha(p) for p in [path,slopes_path,cov_path]}
        for index,row in table.iterrows():
            record=row.to_dict();record.setdefault('context',folder.parent.name);record.setdefault('window',folder.name)
            q=int(row.restrictions) if finite(row.restrictions) else 0
            block=select(slopes,row);vblock=select(cov,row,skip=[value_column])
            local_sources=dict(sources)
            if block.empty and str(row.get('model',''))=='separate' and (folder/'separate_estimates.csv').exists():
                alternate=folder/'separate_estimates.csv';block=select(read(alternate),row,skip=['spec_id'])
                if 'term' in block:block=block[block.term.eq('temp_c')]
                local_sources[str(alternate.relative_to(ROOT))]=sha(alternate)
            if block.empty or vblock.empty:
                assert not finite(row.get('p_value')),(path,index,'Available test missing source profile')
                yield item(path,index,record,family=family,beta=None,covariance=None,restriction=None,reference_covariance=None,model_id=f'{path}:{index}',sources=local_sources)
                continue
            assert not block.sector_id.duplicated().any(),(path,index,'Ambiguous profile')
            units=sorted(block.sector_id.astype(str).tolist());block=block.set_index('sector_id').loc[units]
            assert not vblock.duplicated(['sector_i','sector_j']).any(),(path,index,'Ambiguous covariance')
            v=vblock.pivot(index='sector_i',columns='sector_j',values=value_column).reindex(index=units,columns=units).to_numpy(float)
            b=block.coefficient.to_numpy(float)
            reference=None
            if 'hc0_reference_value' in vblock:
                reference=vblock.pivot(index='sector_i',columns='sector_j',values='hc0_reference_value').reindex(index=units,columns=units).to_numpy(float)
            if len(units)==q+1:R=np.eye(len(units))[1:]-np.eye(len(units))[[0]]
            elif len(units)==q:
                assert str(row.get('model','')).startswith('R'),(path,index,'Unexpected identity-profile null')
                R=np.eye(q)
            else:raise AssertionError((path,index,'Restriction count differs from saved sector profile',len(units),q))
            model_id=hashlib.sha256(json.dumps([str(folder.relative_to(ROOT)),[(key,str(row.get(key,''))) for key in KEYS]],sort_keys=True).encode()).hexdigest()[:24]
            yield item(path,index,record,family=family,beta=b,covariance=v,restriction=R,reference_covariance=reference,
                       basis_names=units,model_id=model_id,sources=local_sources)
        read.cache_clear()

def collect_stata(root=ROOT):
    for family,folder in [('stata_heat',Path(root)/'outputs/stata_key_results'),('stata_arrivals',Path(root)/'outputs/stata_key_results/arrivals')]:
        path=folder/'joint_tests.csv';table=read(path)
        specifications=json.loads((folder/'specifications.json').read_text());specifications=specifications if isinstance(specifications,list) else specifications['specifications']
        specs={row['target_id']:row for row in specifications}
        runs=read(folder/'native/runs.csv')
        for index,row in table.iterrows():
            spec=specs[row.target_id];variant=row.get('run_variant','native_default');native_folder=folder/'native'/row.target_id/variant
            rhs=list(spec['regressor_aliases']);q=int(row.restrictions)
            R=np.zeros((q,len(rhs)))
            for i in range(q):R[i,rhs.index(f'd{i+1:03}')]=1.
            covariance_file='native_vcov.csv' if row.method=='native' else str(row.method)+'.csv'
            beta_path=native_folder/'native_beta.csv';v_path=native_folder/covariance_file;ref_path=native_folder/'hc0.csv'
            b=np.loadtxt(beta_path,delimiter=',',ndmin=1).reshape(-1);v=np.loadtxt(v_path,delimiter=',',ndmin=2);hc=np.loadtxt(ref_path,delimiter=',',ndmin=2)
            match=runs[runs.target_id.eq(row.target_id)]
            if 'run_variant' in match:match=match[match.run_variant.eq(variant)]
            assert len(match)==1
            df=float(match.iloc[0].df_r) if row.method=='native' else int(spec['n_sectors'])-1 if row.method=='project_cluster' else int(spec['n_periods'])-1
            record={**spec,**row.to_dict(),'denominator_df':df,'covariance':str(row.method),'hac_lags':0 if row.method in ['native','project_cluster'] else int(spec.get('horizon',12))*int(str(row.method)[-1])}
            sources={str(p.relative_to(root)):sha(p) for p in [path,folder/'specifications.json',folder/'native/runs.csv',beta_path,v_path,ref_path]}
            yield item(path,index,record,family=family,beta=b,covariance=v,restriction=R,reference_covariance=hc,basis_names=rhs,
                       model_id=f'{family}:{row.target_id}:{variant}:{row.method}',sources=sources,df2=df)

def collect_original_arrivals(root=ROOT):
    for path in [Path(root)/'outputs/visitor_arrivals/joint_tests.json',Path(root)/'outputs/visitor_arrivals/crisis/joint_tests.json']:
        rows=json.loads(path.read_text())
        for index,original in enumerate(rows):
            archive=path.parent/'designs'/f'profiles__{original["group"]}__{original["hac_lags"]}.npz'
            with np.load(archive,allow_pickle=False) as z:
                b=z['beta'][int(z['temperature_index']),:];v=z['covariance'];units=z['units'].tolist();t=len(z['time'])
            record={**original,'test':'All origin temperature slopes equal','model':'fully_interacted_profiles','spec_id':original['group'],
                    'panel':original['group'].split('__')[0],'outcome':original['group'].split('__')[1],'sample_variant':original['group'].split('__',2)[2],
                    'covariance':'Calendar HAC','n_periods':t,'n_sectors':len(units),'denominator_df':t-1,
                    'f_statistic':original.get('statistic'),'wald_chi2':None if original.get('statistic') is None else original['statistic']*original['restrictions']}
            yield item(path,index,record,family='original_arrivals',beta=b,covariance=v,restriction=np.eye(len(units))[1:]-np.eye(len(units))[[0]],reference_covariance=None,
                       basis_names=units,model_id=f'original_arrivals:{original["group"]}:{original["hac_lags"]}',sources={str(p.relative_to(root)):sha(p) for p in [path,archive]})

def collect_scalar_audits(root=ROOT):
    path=Path(root)/'outputs/pyfixest/sector_total_effect_null_checks.csv'
    for index,row in read(path).iterrows():
        record={**row.to_dict(),'test':row['null'],'p_value':row.p_from_one_restriction_F,'restrictions':1,'status':'available',
                'covariance':'Saved full-covariance total-effect contrast','audit_role':'One-restriction audit; coefficient t-test remains separately reported'}
        yield item(path,index,record,family='sector_total_audit',beta=np.asarray([row.estimate]),covariance=np.asarray([[row.variance_of_linear_combination]]),
                   restriction=np.ones((1,1)),reference_covariance=None,basis_names=['sector total beta_T+delta_i'],model_id=f'sector_total_audit:{index}')
    paths=sorted((Path(root)/'outputs/outcomes').glob('*/yoy_weather/temperature_difference_restriction*.json'))+sorted((Path(root)/'outputs/aggregate_va/yoy_weather').glob('temperature_difference_restriction*.json'))
    for path in paths:
        saved=json.loads(path.read_text());sample_path=path.parent/'common_sample.csv'
        if not sample_path.exists():sample_path=path.parent/'samples'/('Y4N.csv' if saved['outcome']=='y_yoy_nsa' else 'Y4.csv')
        if sample_path.exists():sample=pd.read_csv(sample_path,usecols=['time']);t=sample.time.nunique()
        else:
            result=pd.read_csv(path.parent/'regression_results.csv');match=result[result.model.eq('Y4N' if saved['outcome']=='y_yoy_nsa' else 'Y4')]
            assert len(match);t=int(match['n_periods' if 'n_periods' in match else 'n_quarters'].iloc[0])
        cases=[('calendar',saved,t-1)]+([('sector_cluster',saved['sector_cluster'],saved['sector_cluster']['inference_df'])] if 'sector_cluster' in saved else [])
        for index,(method,values,df) in enumerate(cases):
            record={**saved,**values,'test':saved['null'],'restrictions':1,'status':'available','denominator_df':df,'covariance':method,
                    'n_periods':t,'context':'aggregate_va' if 'aggregate_va' in path.parts else path.parts[-3],
                    'audit_role':'Temperature-only one-restriction Wald; coefficient t-test retained separately'}
            record.pop('sector_cluster',None)
            yield item(path,index,record,family='temperature_difference',beta=np.asarray([values['sum']]),covariance=np.asarray([[values['std_error']**2]]),
                       restriction=np.ones((1,1)),reference_covariance=None,basis_names=['sum of current and year-lagged temperature coefficients'],
                       model_id=f'{path.relative_to(root)}:{method}',sources={str(p.relative_to(root)):sha(p) for p in [path,sample_path if sample_path.exists() else path.parent/'regression_results.csv']})

def collect_legacy_tests(root=ROOT):
    yield from collect_sector_groups(root)
    yield from collect_stata(root)
    yield from collect_original_arrivals(root)
    yield from collect_scalar_audits(root)
