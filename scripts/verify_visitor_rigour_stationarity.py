"""Independent exact raw-basis stationarity audit for354 visitor requests."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_key,'1')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import verify_extension_stationarity as math_check
import verify_crisis_sensitivity as proof
import verification_artifacts as artifact_bindings

OUT=ROOT/'outputs/visitor_rigour/diagnostics'
SOURCES={'monthly':'outputs/visitor_rigour/monthly/specifications.csv','annual':'outputs/visitor_rigour/annual/specifications.csv','ppml':'outputs/visitor_rigour/ppml/specifications.csv','origin_year':'outputs/visitor_rigour/origin_year/specifications.csv'}
EXCLUDE={'lny_window_days','lny_lag1','trend','trend_sq','time'}


def read(path):return pd.read_csv(path,dtype={'unit_id':str,'sector_id':str},float_precision='round_trip',low_memory=False)


def diagnostic_artifacts():
    checked=artifact_bindings.snapshot(OUT)
    models=read(OUT/'model_inventory.csv')
    for field in ['sample_csv','design_npz']:
        for name in models[field].dropna().unique():checked[name]=proof.sha256(ROOT/name)
    for source in SOURCES.values():
        checked[source]=proof.sha256(ROOT/source)
        parent=str(Path(source).parent/'verification.json');checked[parent]=proof.sha256(ROOT/parent)
    return checked


def _verify_visitor_rigour_stationarity():
    proof.ERRORS.clear();summary=json.loads((OUT/'summary.json').read_text());assert summary['families']==SOURCES
    hashes=json.loads((OUT/'source_hashes.json').read_text());assert proof.sha256(OUT/'source_hashes.json')==summary['source_hashes_sha256']
    def guard():
        for path,digest in hashes.items():assert proof.sha256(ROOT/path)==digest,('Frozen visitor diagnostic input',path)
        for name,digest in summary['outputs_sha256'].items():assert proof.sha256(OUT/name)==digest
    guard()
    models=read(OUT/'model_inventory.csv');links=read(OUT/'model_variable_coverage.csv');d=read(OUT/'diagnostics.csv');series=read(OUT/'series.csv');tests=read(OUT/'tests.csv')
    assert len(models)==354 and models.coverage_id.is_unique and d.diagnostic_id.is_unique
    assert not links.duplicated(['coverage_id','variable','role','unit_id']).any() and not series.duplicated(['diagnostic_id','period']).any()
    assert not tests.duplicated(['diagnostic_id','specification','test']).any() and not links.variable.isin(EXCLUDE).any()
    models_by=models.set_index('coverage_id');links_by={key:part for key,part in links.groupby('coverage_id',sort=False)}
    diag_by=d.set_index('diagnostic_id',drop=False);series_by={key:part for key,part in series.groupby('diagnostic_id',sort=False)};tests_by={key:part for key,part in tests.groupby('diagnostic_id',sort=False)}
    seen={};expected_models=set();expected_links=set();variable_sets={};role_sets={};branch_sets={};count_flags={};completed=0;records=[];parent_proofs={}
    samples={}
    for branch,path in SOURCES.items():
        source=read(ROOT/path);assert len(source)=={'monthly':270,'annual':30,'ppml':30,'origin_year':24}[branch]
        parent_path=Path(path).parent/'verification.json';assert json.loads((ROOT/parent_path).read_text())['status']=='passed';parent_proofs[str(parent_path)]=proof.sha256(ROOT/parent_path)
        for index,row in enumerate(source.itertuples(index=False)):
            key='vm_'+hashlib.sha256(json.dumps([branch,path,index,row.outcome],separators=(',',':')).encode()).hexdigest()[:24]
            expected_models.add(key);model=models_by.loc[key]
            assert model.branch==branch and model.source_spec_csv==path and model.source_spec_row==index and model.spec_id==row.spec_id
            assert model.source_status==row.status and model.sample_variant==row.sample_variant and model.outcome==row.outcome
            sample_name=row.sample_csv
            assert (ROOT/sample_name).is_file(), 'Every observed PPML model returned an actual retained sample'
            if sample_name not in samples:samples[sample_name]=read(ROOT/sample_name)
            sample=samples[sample_name];units=sample.unit_id.to_numpy(str);freq='Y-DEC' if branch=='annual' else 'M';period='year' if branch=='annual' else 'month'
            dates=pd.PeriodIndex(sample[period].astype(str),freq=freq);yfield='arrivals' if branch=='ppml' else row.outcome
            if branch in ['monthly','origin_year']:requested=list(dict.fromkeys(json.loads(row.temp_features)+json.loads(row.control_features)+json.loads(row.ar_features)))
            else:requested=list(dict.fromkeys(json.loads(row.temperature_terms)+json.loads(row.controls)))
            terms=[term for term in requested if term not in EXCLUDE];excluded=[term for term in requested if term in EXCLUDE]
            assert json.loads(model.requested_basis_terms)==terms and json.loads(model.excluded_deterministic_terms)==excluded
            assert model.n_obs==len(sample) and model.n_units==len(set(units)) and model.n_periods==len(set(dates))
            assert model.frequency==freq and model.sample_csv==sample_name and model.sample_sha256==proof.sha256(ROOT/sample_name)==hashes[sample_name]
            assert model.design_npz==row.design_npz and model.design_sha256==proof.sha256(ROOT/row.design_npz)==hashes[row.design_npz]
            assert bool(model.count_level_screen)==(branch=='ppml') and bool(model.growth_outcome)==(branch!='ppml')
            if branch=='ppml':assert model.sample_role=='Native retained sample'
            with np.load(ROOT/row.design_npz,allow_pickle=False) as archive:
                uid=archive['unit_ids' if 'unit_ids' in archive else 'units'].astype(str)
                assert np.array_equal(uid,units) and np.array_equal(archive['time'],sample.year if branch=='annual' else dates.asi8)
                proof.check(archive['Y'],sample[yfield].to_numpy(),'Every visitor diagnostic actual native Y',atol=0,rtol=0)
                names=archive['names'].tolist()
                for term in terms:
                    raw=sample[term].to_numpy(float);basis=raw if branch=='ppml' else raw-raw.mean()
                    proof.check(archive['X'][:,names.index(term)],basis,'Every visitor stochastic diagnostic basis before FE projection',atol=2e-10,rtol=2e-12)
            unit_list=sorted(set(units));expected=[]
            for unit in unit_list:
                keep=units==unit;expected.append((row.outcome,'Y',unit,[unit],dates[keep],sample.loc[keep,yfield].to_numpy(float)))
            for term in terms:
                separated=[(unit,dates[units==unit],sample.loc[units==unit,term].to_numpy(float)) for unit in unit_list]
                shared=len(separated)>1 and all(np.array_equal(date.asi8,separated[0][1].asi8) and np.array_equal(value,separated[0][2],equal_nan=True) for _,date,value in separated[1:])
                for unit,date,value in separated[:1] if shared else separated:expected.append((term,'X','common' if shared else unit,unit_list if shared else [unit],date,value))
            actual=links_by[key];assert len(actual)==len(expected)==model.n_variable_links
            for variable,role,unit,applies,date,value in expected:
                link=actual.loc[actual.variable.eq(variable)&actual.role.eq(role)&actual.unit_id.eq(unit)];assert len(link)==1;link=link.iloc[0]
                assert json.loads(link.applies_to_units)==applies and link.source_status==row.status
                expected_links.add((key,variable,role,unit))
                order=np.argsort(date.asi8);date=date[order];value=value[order]
                growth=role=='Y' and branch!='ppml';det='c' if growth or variable.startswith(('ar_l','rain')) or variable=='temp_endpoint_difference' else 'ct';seasonal=branch!='annual'
                es_id,digest,_,_=math_check.signature(date,value,freq,det,seasonal);ident='vs_'+es_id[3:]
                assert link.diagnostic_id==ident
                if ident not in seen:
                    seen[ident]=digest;diagnostic=diag_by.loc[ident].copy();assert diagnostic.input_sha256==digest
                    diagnostic['diagnostic_id']=es_id
                    completed+=math_check.verify_diagnostic(diagnostic,series_by[ident].sort_values('ordinal'),tests_by.get(ident,pd.DataFrame(columns=tests.columns)),date,value,freq,det,seasonal)
                    records.append({'diagnostic_id':ident,'frequency':freq,'state':diagnostic.state,'n_periods':len(date),'calendar_gaps':diagnostic.calendar_gaps,'first_branch':branch,'first_variable':variable})
                else:assert seen[ident]==digest
                variable_sets.setdefault(ident,set()).add(variable);role_sets.setdefault(ident,set()).add(role);branch_sets.setdefault(ident,set()).add(branch)
                count_flags[ident]=count_flags.get(ident,False) or (branch=='ppml' and role=='Y')
        print(f'Visitor diagnostic independent coverage/test audit: {branch}, {len(seen)} arrays',flush=True)
    assert set(expected_models)==set(models.coverage_id)
    assert expected_links==set(links[['coverage_id','variable','role','unit_id']].itertuples(index=False,name=None))
    assert set(seen)==set(d.diagnostic_id)==set(series_by)
    assert set(tests_by)==set(d.loc[d.test_status.eq('completed'),'diagnostic_id'])
    for ident in seen:
        row=diag_by.loc[ident]
        assert json.loads(row.variables)==sorted(variable_sets[ident]) and json.loads(row.roles)==sorted(role_sets[ident]) and json.loads(row.branches)==sorted(branch_sets[ident])
        assert bool(row.count_level_descriptive_screen)==count_flags[ident]
    calculated={'source_spec_rows':354,'logical_model_outcomes':354,'model_variable_links':len(links),'distinct_diagnostics':len(d),'completed_diagnostics':completed,
        'unavailable_diagnostics':int(d.test_status.eq('not_run').sum()),'error_diagnostics':int(d.test_status.eq('error').sum()),'test_rows':len(tests),'series_rows':len(series)}
    assert summary['counts']==calculated and summary['diagnostic_states']==d.state.value_counts().to_dict()
    assert completed==65 and len(tests)==260 and calculated['error_diagnostics']==0
    assert all(row['first_branch']=='ppml' for row in records if row['state'] not in ['unavailable-gaps','unavailable-short'])
    assert not d.loc[d.frequency.eq('Y-DEC'),'test_status'].eq('completed').any()
    # The original330 baseline is independently source-validated above. Derive
    # it from the portable current inventory, not an ignored historical folder.
    baseline_models=models.loc[models.branch.ne('origin_year')]
    baseline_links=links.loc[links.branch.ne('origin_year')]
    assert len(baseline_models)==330 and len(baseline_links)==19644
    assert set(baseline_links.diagnostic_id)==set(d.diagnostic_id)
    assert len(links)==20352 and models.branch.eq('origin_year').sum()==24 and links.branch.eq('origin_year').sum()==708
    # New nuisance indicators are deterministic and excluded from this marginal
    # series inventory. Every new link must map to the identical parent array.
    for row in models.loc[models.branch.eq('origin_year')].itertuples():
        spec=read(ROOT/SOURCES['origin_year']);spec=spec.loc[spec.spec_id.eq(row.spec_id)].iloc[0]
        parent=models.loc[models.branch.eq('monthly')&models.spec_id.eq(spec.source_spec_id)];assert len(parent)==1
        new=links.loc[links.coverage_id.eq(row.coverage_id)].sort_values(['variable','role','unit_id'])
        old=links.loc[links.coverage_id.eq(parent.coverage_id.iloc[0])].sort_values(['variable','role','unit_id'])
        assert new[['variable','role','unit_id','diagnostic_id','applies_to_units']].reset_index(drop=True).equals(old[['variable','role','unit_id','diagnostic_id','applies_to_units']].reset_index(drop=True))
    guard();assert all(proof.sha256(ROOT/path)==digest for path,digest in parent_proofs.items())
    pd.DataFrame(records).to_csv(OUT/'verification_diagnostics.csv',index=False)
    return {'status':'passed','counts':calculated,'states':summary['diagnostic_states'],'numerical_checks':proof.ERRORS,'source_bindings_unchanged':len(hashes),
        'numerical_parent_receipts':parent_proofs,'verifier_source_sha256':proof.sha256(Path(__file__)),
        'origin_year_extension':{'baseline_requests':330,'baseline_links':19644,'added_requests':24,'added_links':708,'new_unique_diagnostics':0,'baseline_derivation':'Current non-origin_year inventory, independently validated against all330 actual source specifications/sample/designs; every new link maps the same parent diagnostic ID. All current raw arrays/tests independently checked; no historical output archive required.'},
        'scope':'All354 model/alias requests, actual retained PPML samples, raw stochastic Y/X and declared OLS centering, shared-unit coverage/dedup, complete uncompressed calendars, short/constant/missing guards, independent seasonal offsets and ADF/KPSS arithmetic/states.',
        'limits':'No new growth-window stationarity clearance is available: exact monthly growth windows have gaps and annual samples have fewer than20years. Completed screens concern contiguous PPML count/weather inputs, descriptively; I(0) screening is not a Poisson admission criterion or causal/inference-validity certificate.'}


def verify_visitor_rigour_stationarity():
    path=OUT/'verification.json';artifact_bindings.archive_passed(path)
    initial={'status':'running','verifier_source_sha256':proof.sha256(Path(__file__))};path.write_text(json.dumps(initial,indent=2)+'\n')
    try:
        checked=diagnostic_artifacts()
        with threadpool_limits(limits=1):report=_verify_visitor_rigour_stationarity()
        artifact_bindings.assert_unchanged(checked);assert diagnostic_artifacts()==checked
        report['verified_artifacts_sha256']=checked
        report['artifact_binding_helper_sha256']=proof.sha256(Path(artifact_bindings.__file__))
    except BaseException as error:
        path.write_text(json.dumps({**initial,'status':'failed','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n');raise
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Visitor exact-window stationarity independent verification passed.',flush=True);return report


if __name__=='__main__':verify_visitor_rigour_stationarity()
