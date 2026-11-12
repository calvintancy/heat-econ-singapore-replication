"""Reconstruct real coefficient-space restrictions for the native-Wald overlay.

Old CSVs, NPZs, fit ledgers and guards are read only. Every yielded row keeps its
positional source row and complete original metadata. Coefficients/covariance
come from the saved native model arrays, never from an old Wald statistic.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FAMILIES={
    'literature':'outputs/literature_sensitivity',
    'literature_full':'outputs/literature_sensitivity/full_controls',
    'literature_annual':'outputs/literature_sensitivity/annual',
    'literature_seasonal':'outputs/literature_sensitivity/annual_seasonal',
    'visitor_monthly':'outputs/visitor_rigour/monthly',
    'visitor_origin_year':'outputs/visitor_rigour/origin_year',
    'visitor_annual':'outputs/visitor_rigour/annual',
    'visitor_ppml':'outputs/visitor_rigour/ppml'}
CSV_ALIASES={'outputs/visitor_rigour/annual/joint_tests.csv':'outputs/visitor_rigour/annual/tests.csv'}
HISTORICAL_ONLY=['outputs/literature_sensitivity/tests_native_stage.csv','outputs/literature_sensitivity/annual_seasonal/tests_native_stage.csv']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return pd.read_csv(path,dtype={'sector_id':str,'unit_id':str,'reference_sector_id':str,'reference_unit_id':str},float_precision='round_trip',low_memory=False)


def _present(value):return value is not None and not (isinstance(value,(float,np.floating)) and np.isnan(value))


def _unit(row):
    for field in ['sector_id','unit_id']:
        if _present(row.get(field)):return str(row[field])
    return 'pooled'


def _bases(components,k,units,structure):
    components=np.asarray(components,dtype=int)
    if components.ndim==1:components=components[None,:]
    result=[]
    for number in range(len(units) if structure=='H1' else 1):
        basis=np.zeros((len(components),k));basis[np.arange(len(components)),components[:,0]]=1
        if structure=='H1' and number:basis[np.arange(len(components)),components[:,number]]+=1
        result.append(basis)
    return result


def _expansion(h):
    expansion=np.zeros((h+1,2));expansion[0,0]=1;expansion[-1,0]=-1;expansion[:-1,1]=1
    return expansion


def _lag_null(name,basis,lagbasis,outcome,two_channel,horizon):
    if name in ['All temperature-lag coefficients zero','All temperature channels zero']:return basis if two_channel else lagbasis
    if name.startswith('Cumulative direct temperature response zero'):return lagbasis.sum(axis=0,keepdims=True)
    if 'span' in name:
        eye=np.eye(len(lagbasis));return np.vstack([eye[j]-eye[1] for j in range(2,len(lagbasis)-1)]+[eye[0]+eye[-1]-eye[1]])@lagbasis
    if name.startswith(('Exact contemporaneous static-output-level','Conditional direct-weather level-channel','Static coefficient shape')):
        if two_channel:return basis[[1]]
        eye=np.eye(len(lagbasis));endpoint=1 if outcome in ['y_short','y_mom_nsa'] else len(lagbasis)-1
        return np.vstack([eye[j] for j in range(1,len(eye)) if j!=endpoint]+[eye[0]+eye[endpoint]])@lagbasis
    if name.startswith(('Equal one-period growth','Conditional direct-weather equal-growth','Equal-growth accumulation')):
        if two_channel:return basis[[0]]
        eye=np.eye(len(lagbasis));return np.vstack([eye[j]-eye[0] for j in range(1,len(eye)-1)]+[eye[-1]])@lagbasis
    raise KeyError('Unmapped lag restriction: '+name)


def _lag_restriction(row,spec,z,names,cache):
    k=len(names);structure=spec.get('structure',spec.get('model','common'));specification=spec.get('specification',spec.get('model_spec'))
    unit_key='unit_ids' if 'unit_ids' in z else 'units';units=sorted(set(z[unit_key].astype(str)))
    two=specification=='two_channel';h=int(spec.get('horizon',row.get('horizon',0)))
    if 'bases' not in cache:
        bases=_bases(z['component_positions'],k,units,structure);profiles=[_expansion(h)@basis if two else basis for basis in bases];cache.update(bases=bases,profiles=profiles)
    bases,profiles=cache['bases'],cache['profiles']
    name=row.get('test',row.get('null'));unit=_unit(row)
    if unit=='all':
        if 'cumulative' in name:return np.vstack([(basis-profiles[0]).sum(axis=0) for basis in profiles[1:]])
        if 'vectors equal' in name:return np.vstack([basis-bases[0] for basis in bases[1:]])
        if 'jointly zero' in name:return np.vstack(bases)
        raise KeyError('Unmapped global lag restriction: '+str(name))
    number=units.index(unit) if structure=='H1' else 0
    return _lag_null(name,bases[number],profiles[number],row.get('outcome',spec.get('outcome')),two,h)


def _seasonal_restriction(row,spec,z,names,cache):
    k=len(names);units=sorted(set(z['sector_id'].astype(str)));structure=spec['model']
    if 'bases' not in cache:cache['bases']=_bases(z['component_positions'],k,units,structure)
    bases=cache['bases']
    name=row['test'];unit=_unit(row)
    if unit=='all':
        if name=='All sector seasonal temperature response vectors equal':return np.vstack([basis-bases[0] for basis in bases[1:]])
        if name=='All sector sums of direct seasonal temperature coefficients equal':return np.vstack([(basis-bases[0]).sum(axis=0) for basis in bases[1:]])
        raise KeyError(name)
    basis=bases[units.index(unit) if structure=='H1' else 0]
    options={'All seasonal temperature coefficients zero':basis,'Current Q1–Q4 temperature coefficients equal':basis[1:4]-basis[[0]],'Sum of all direct seasonal temperature coefficients zero':basis.sum(axis=0,keepdims=True)}
    if len(basis)==8:options.update({'Prior Q1–Q4 temperature coefficients equal':basis[5:]-basis[[4]],'Current and prior same-quarter temperature coefficients equal':basis[:4]-basis[4:],'All same-quarter current-plus-prior coefficient sums zero':basis[:4]+basis[4:]})
    return options[name]


def _annual_literature_restriction(row,spec,z,names):
    terms=json.loads(spec['temperature_terms']);units=sorted(set(z['sector_id'].astype(str)));kind=spec['model'];reference=units[0]
    def basis(unit):
        value=np.zeros((len(terms),len(names)));value[np.arange(len(terms)),[names.index(term) for term in terms]]=1
        if kind=='H1' and unit!=reference:
            for j,term in enumerate(terms):value[j,names.index(f'{term}:C(sector_id)[T.{unit}]')]+=1
        return value
    unit=_unit(row);name=row['null']
    if unit=='all':
        assert name=='All sector temperature response functions equal'
        return np.vstack([basis(unit)-basis(reference) for unit in units[1:]])
    current=basis(unit)
    if name=='All temperature coefficients zero':return current
    if name=='Temperature coefficient sum zero':return current.sum(axis=0,keepdims=True)
    if name=='Static contemporaneous annual level channel':return np.vstack([current[0]+current[1],current[2:]])
    if name=='Temperature curvature zero':return current[[1]]
    raise KeyError(name)


def _arrival_annual_restriction(row,spec,z,names):
    units=sorted(set(z['units'].astype(str)));positions=np.asarray(z['component_positions'],int);profiles=np.zeros((len(units),len(names)));profiles[:,positions[0]]=1
    for i in range(1,len(units)):profiles[i,positions[i]]=1
    return profiles[1:]-profiles[[0]] if row['test']=='All origin temperature slopes equal' else profiles


def _reference(z,k):
    for key in ['hc0_full','hc0']:
        if key in z:return z[key],key,None
    if 'hc0_selected' in z:
        positions=z['selected_positions'].astype(int);reference=np.zeros((k,k));reference[np.ix_(positions,positions)]=z['hc0_selected']
        return reference,'hc0_selected',positions.tolist()
    return None,None,None


def _archive_path(root,folder,spec):
    value=spec.get('design_npz')
    return root/value if _present(value) else root/folder/'designs'/f"{spec['spec_id']}.npz"


def collect_extension_tests(root=ROOT,*,families=None):
    """Yield plain row/array records for eight NPZ-backed families and spans.

    The annual visitor ``joint_tests.csv`` duplicate is an explicit row alias;
    native-stage literature files are historical inventory, not current tests.
    This generator never evaluates tests, changes guards or writes artifacts.
    """
    root=Path(root).resolve();chosen=list(FAMILIES) if families is None else list(families)
    for family in chosen:
        folder=FAMILIES[family];spec_path=root/folder/'specifications.csv';specs=read(spec_path);lookup={str(row['spec_id']):row for row in specs.to_dict('records')}
        sources_base={str(spec_path.relative_to(root)):sha(spec_path),str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))}
        files=['tests.csv']+(['two_channel_span_tests.csv'] if family in ['literature','literature_full'] else [])
        if family=='visitor_annual':
            alias=root/folder/'joint_tests.csv';assert alias.read_bytes()==(root/folder/'tests.csv').read_bytes(),'Annual visitor joint/tests aliases disagree'
        for filename in files:
            csv_path=root/folder/filename;frame=read(csv_path);source_csv=str(csv_path.relative_to(root));csv_sha=sha(csv_path)
            for spec_id,part in frame.groupby('spec_id',sort=False):
                spec=lookup[str(spec_id)];path=_archive_path(root,folder,spec);archive_sha=sha(path);relative=str(path.relative_to(root))
                with np.load(path,allow_pickle=False) as z:
                    names=z['names' if 'names' in z else 'coef_names'].astype(str).tolist();k=len(names)
                    beta_key='coefficients' if 'coefficients' in z else 'beta' if 'beta' in z else None
                    if family=='literature_annual':beta_key='beta_'+str(z['outcomes'].astype(str).tolist().index(str(spec['outcome'])))
                    beta=z[beta_key] if beta_key is not None and beta_key in z else None
                    reference,reference_key,reference_positions=_reference(z,k)
                    matrices={};restriction_cache={};basis_cache={}
                    for source_row,original in part.iterrows():
                        row=original.to_dict();df2=row.get('inference_df',row.get('denominator_df'));bound=row.get('structural_rank_bound',df2)
                        if not _present(bound):bound=df2
                        cache_key=(row.get('test',row.get('null')),_unit(row))
                        if family=='literature_annual':
                            code='SC' if row['covariance']=='Sector cluster' else 'DK'+str(int(row['hac_lags']));covariance_key='covariance_'+beta_key.split('_')[1]+'_'+code
                            if cache_key not in restriction_cache:restriction_cache[cache_key]=_annual_literature_restriction(row,spec,z,names)
                        else:
                            pairs=list(zip(z['covariance_labels'].astype(str),z['bandwidths'].astype(int)))
                            position=pairs.index((str(row['covariance']),int(row['hac_lags'])));covariance_key=f'covariance_{position}'
                            if cache_key not in restriction_cache:
                                if family=='literature_seasonal':restriction_cache[cache_key]=_seasonal_restriction(row,spec,z,names,basis_cache)
                                elif family=='visitor_annual':restriction_cache[cache_key]=_arrival_annual_restriction(row,spec,z,names)
                                elif family=='visitor_ppml':restriction_cache[cache_key]=np.eye(k)[[names.index('temp_c')]]
                                else:restriction_cache[cache_key]=_lag_restriction(row,spec,z,names,basis_cache)
                        restriction=restriction_cache[cache_key]
                        assert restriction.shape==(int(row['restrictions']),k),(family,spec_id,source_row,row.get('test',row.get('null')),restriction.shape,row['restrictions'])
                        if covariance_key not in matrices:matrices[covariance_key]=z[covariance_key] if covariance_key in z else None
                        covariance=matrices[covariance_key]
                        if pd.notna(row.get('p_value')):assert beta is not None and covariance is not None,(family,spec_id,'Available row missing coefficient/covariance archive')
                        pointer={'path':relative,'sha256':archive_sha,'beta_key':beta_key if beta_key in z else None,'covariance_key':covariance_key if covariance_key in z else None,
                            'reference_key':reference_key,'reference_positions':reference_positions,'names_key':'names' if 'names' in z else 'coef_names'}
                        record={'source_csv':source_csv,'source_row':int(source_row),'family':family,'row':row,'beta':beta,'covariance':covariance,
                            'restriction':restriction,'reference_covariance':reference,'df2':None if not _present(df2) else float(df2),
                            'structural_rank_bound':None if not _present(bound) else int(bound),
                            'sources':{**sources_base,source_csv:csv_sha,relative:archive_sha},'basis_names':names,
                            'model_id':str(row.get('native_spec_id')) if _present(row.get('native_spec_id')) else str(spec_id),'source_arrays':pointer}
                        if family=='visitor_annual':record['row_aliases']=[{'source_csv':str(Path(folder)/'joint_tests.csv'),'source_row':int(source_row)}]
                        yield record
