"""Exact-calendar marginal stationarity appendix for seven frozen extensions.

No FE residuals, fitted parameters, native fits, sample deletion or reselection.
Identical input arrays and test policies share one diagnostic, not one model.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from stationarity_audit import test_pair, state, LABELS
from stationarity_diagnostics import seasonal_adjust

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/extension_stationarity'
FAMILIES={
 'core':'outputs/crisis_sensitivity/specifications.csv',
 'full':'outputs/complete_va_controls/specifications.csv',
 'annual':'outputs/crisis_sensitivity/annual/specs.csv',
 'literature':'outputs/literature_sensitivity/specifications.csv',
 'annual_literature':'outputs/literature_sensitivity/annual/specifications.csv',
 'seasonal':'outputs/literature_sensitivity/annual_seasonal/specifications.csv',
 'full_controls':'outputs/literature_sensitivity/full_controls/specifications.csv'}
EXTRA_LABELS={'unavailable-missing':'Not established · missing values','unavailable-constant':'Not established · constant series','unavailable-empty':'Not established · empty calendar'}
ALL_LABELS={**LABELS,**EXTRA_LABELS}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def present(value):return isinstance(value,str) and bool(value) and value!='nan'
def items(value):return json.loads(value) if present(value) else []
def binary(values):
    result=np.asarray(values,dtype='<f8').copy();result[np.isnan(result)]=np.nan
    return result.tobytes()


class Appendix:
    def __init__(self):
        self.hashes={};self.frames={};self.archives={};self.diagnostics={};self.tests=[];self.series=[];self.links=[];self.models=[]

    def bind(self,path):
        path=Path(path);name=rel(path)
        if name not in self.hashes:self.hashes[name]=sha(path)
        return name

    def frame(self,path):
        name=self.bind(path)
        if name not in self.frames:self.frames[name]=pd.read_csv(path,dtype={'sector_id':str},float_precision='round_trip',low_memory=False)
        return self.frames[name]

    def archive(self,path):
        name=self.bind(path)
        if name not in self.archives:
            with np.load(path,allow_pickle=False) as source:
                self.archives[name]={k:source[k] for k in ['X','Y','time','units','sector_id','periods','names','coef_names','outcomes'] if k in source}
        return self.archives[name]

    def diagnostic(self,dates,values,frequency,variable,role,adjustment):
        dates=pd.PeriodIndex(dates,freq=frequency)
        order=np.argsort(dates.asi8);dates=dates[order];values=np.asarray(values,float)[order]
        assert dates.is_unique and len(dates)==len(values)
        annual=frequency=='Y-DEC';h=1 if annual else 12 if frequency=='M' else 4
        growth=role=='Y' and variable not in ['y_log_level','y_raw_level']
        is_ar=variable.startswith('ar_l')
        rain=variable.startswith('rain')
        difference=variable=='temp_endpoint_difference' or 'year_change' in variable
        det='c' if growth or is_ar or rain or difference else 'ct'
        seasonal=not annual and (role=='X' and not is_ar or adjustment=='NSA')
        rule='season_offsets_with_trend_retained' if seasonal else 'none'
        policy={'frequency':frequency,'deterministic':det,'seasonal_rule':rule}
        prefix=json.dumps(policy,sort_keys=True,separators=(',',':')).encode()
        calendar=dates.asi8.astype('<i8').tobytes();raw=binary(values)
        digest=hashlib.sha256(prefix+b'\0'+calendar+raw).hexdigest();ident='es_'+digest[:24]
        if ident in self.diagnostics:
            row=self.diagnostics[ident];assert row['input_sha256']==digest
            row['_variables'].add(variable);row['_roles'].add(role);return ident
        n=len(values);gaps=int(dates[-1].ordinal-dates[0].ordinal+1-n) if n else 0
        missing=int((~np.isfinite(values)).sum());minimum=20 if annual else 5*h
        row={'diagnostic_id':ident,'input_sha256':digest,'calendar_sha256':hashlib.sha256(calendar).hexdigest(),'raw_values_sha256':hashlib.sha256(raw).hexdigest(),
             **policy,'frequency_horizon':h,'minimum_periods':minimum,'n_periods':n,'start':str(dates[0]) if n else '', 'end':str(dates[-1]) if n else '',
             'calendar_gaps':gaps,'missing_values':missing,'primary_state':None,'legacy_state':None,'test_status':'not_run',
             'test_values_sha256':None,'_variables':{variable},'_roles':{role}}
        if gaps:result='unavailable-gaps'
        elif missing:result='unavailable-missing'
        elif not n:result='unavailable-empty'
        elif n<minimum:result='unavailable-short'
        elif np.ptp(values)==0:result='unavailable-constant'
        else:result=None
        adjusted=component=None
        if result is None:
            source=pd.Series(values,index=dates)
            adjusted,component=seasonal_adjust(source) if seasonal else (source,source*0)
            if np.ptp(adjusted.to_numpy())==0:result='unavailable-constant'
        if result is None:
            tested=adjusted.to_numpy();row['test_values_sha256']=hashlib.sha256(binary(tested)).hexdigest()
            for label,legacy in [('primary',False),('legacy_lag_sensitivity',True)]:
                pair=test_pair(tested,det,h,legacy)
                for item in pair:self.tests.append({'diagnostic_id':ident,'specification':label,'deterministic':det,'frequency':frequency,**item})
                classification=state(*pair,det)
                row['primary_state' if not legacy else 'legacy_state']=classification
                stem='primary_' if not legacy else 'legacy_'
                for suffix,value in [('adf_p',pair[0]['p_value']),('kpss_p',pair[1]['p_value']),('kpss_relation',pair[1]['p_relation']),('adf_lags',pair[0]['lags']),('kpss_lags',pair[1]['lags'])]:row[stem+suffix]=value
            result=row['primary_state'] if row['primary_state']==row['legacy_state'] else 'sensitivity-dependent'
            row['test_status']='completed' if row['primary_state']!='unavailable' and row['legacy_state']!='unavailable' else 'error'
        row['state']=result;row['label']=ALL_LABELS[result]
        for i,(date,value) in enumerate(zip(dates,values)):
            self.series.append({'diagnostic_id':ident,'period':str(date),'ordinal':int(date.ordinal),'raw_value':float(value),
                'test_value':float(adjusted.iloc[i]) if row['test_values_sha256'] else None,
                'seasonal_component':float(component.iloc[i]) if row['test_values_sha256'] else None})
        self.diagnostics[ident]=row
        return ident

    def register(self,branch,path,index,record,outcome):
        source=self.frame(ROOT/record['sample_csv']);sample=source.copy()
        model=record.get('model',record.get('structure',''))
        if model=='separate':
            unit=str(record.get('unit_id',record.get('sector_id')))
            sample=sample.loc[sample.sector_id.eq(unit)].copy()
        freq={'monthly':'M','quarterly':'Q','annual':'Y-DEC'}[record['frequency']]
        period='month' if freq=='M' else 'quarter' if freq=='Q' else 'year'
        design_path=record.get('design_npz')
        if not present(design_path) and branch in ['core','full']:
            design_path=str(Path(path).parent/'designs'/f"{record['fit_group_id']}.npz")
        if not present(design_path) and branch=='literature':
            design_path=str(Path(path).parent/'designs'/f"{record['spec_id']}.npz")
        heat=items(record.get('temperature_terms'))
        terms=list(dict.fromkeys(heat+items(record.get('controls'))+items(record.get('weather_terms'))))
        assert all(term not in ['trend','trend_sq','time'] for term in terms)
        if branch in ['core','full']:terms=items(record['controls'])
        if present(design_path):
            packed=self.archive(ROOT/design_path)
            names=packed['names' if 'names' in packed else 'coef_names'].tolist()
            units=packed['units' if 'units' in packed else 'sector_id'].astype(str)
            dates=pd.PeriodIndex(packed['periods'],freq=freq) if 'periods' in packed else pd.PeriodIndex(packed['time'].astype(int).astype(str),freq=freq)
            x=packed['X'];ys=packed['Y'];y=ys[:,packed['outcomes'].tolist().index(outcome)] if ys.ndim==2 else ys
            assert set(units)==set(sample.sector_id) and len(units)==len(sample)
            sample_index=pd.MultiIndex.from_arrays([sample.sector_id,sample[period].astype(str)])
            actual_index=pd.MultiIndex.from_arrays([units,dates.astype(str)])
            raw_y=pd.Series(sample[outcome].to_numpy(),index=sample_index).reindex(actual_index).to_numpy()
            assert np.allclose(raw_y,y,atol=1e-10,rtol=1e-12,equal_nan=True)
            values={term:x[:,names.index(term)] for term in terms}
            basis_rule='Actual archived non-FE basis column before FE projection; native centering retained; H1 dummy interactions covered by their within-unit basis'
        else:
            units=sample.sector_id.astype(str).to_numpy();dates=pd.PeriodIndex(sample[period].astype(str),freq=freq);y=sample[outcome].to_numpy()
            values={term:sample[term].to_numpy() for term in terms}
            basis_rule='Frozen sample/formula basis before FE projection; no archived matrix for this design request'
        identity=hashlib.sha256(json.dumps([branch,path,int(index),outcome],separators=(',',':')).encode()).hexdigest()[:24]
        group={'coverage_id':'em_'+identity,'branch':branch,'source_spec_csv':path,'source_spec_row':int(index),'spec_id':record.get('spec_id') if present(record.get('spec_id')) else str(record.get('fit_group_id'))+'_'+outcome,
            'outcome':outcome,'model':model,'model_spec':record.get('model_spec',record.get('specification','static')),'context':record['context'],
            'history':record.get('history','exact_sample'),'sample_variant':record['sample_variant'],'source_status':record['status'],
            'sample_csv':record['sample_csv'],'sample_sha256':self.hashes[record['sample_csv']],
            'design_npz':design_path if present(design_path) else '', 'design_sha256':self.hashes.get(design_path,''),
            'frequency':freq,'n_obs':len(y),'n_units':len(set(units)),'n_periods':len(set(dates)),'basis_rule':basis_rule}
        adjustment=record.get('output_adjustment',record.get('adjustment','NSA' if outcome in ['y_nsa','y_yoy_nsa'] else 'SA'))
        unit_list=sorted(set(units));count=0
        for unit in unit_list:
            keep=units==unit;ident=self.diagnostic(dates[keep],y[keep],freq,outcome,'Y',adjustment)
            self.links.append({**group,'variable':outcome,'role':'Y','unit_id':unit,'applies_to_units':json.dumps([unit]),'diagnostic_id':ident});count+=1
        for term,array in values.items():
            separated=[(unit,dates[units==unit],array[units==unit]) for unit in unit_list]
            common=len(separated)>1 and all(np.array_equal(part[1].asi8,separated[0][1].asi8) and np.array_equal(part[2],separated[0][2],equal_nan=True) for part in separated[1:])
            for unit,date,value in separated[:1] if common else separated:
                ident=self.diagnostic(date,value,freq,term,'X',adjustment)
                self.links.append({**group,'variable':term,'role':'X','unit_id':'common' if common else unit,
                    'applies_to_units':json.dumps(unit_list if common else [unit]),'diagnostic_id':ident});count+=1
        self.models.append({**group,'n_variable_links':count,'requested_basis_terms':json.dumps(terms),'growth_outcome':outcome not in ['y_log_level','y_raw_level']})


def render_extension_stationarity():
    summary=json.loads((OUT/'summary.json').read_text());rows=pd.read_csv(OUT/'diagnostics.csv').fillna('')
    for stem in ['primary','legacy']:
        rows[stem+'_kpss_bound']=[str(relation)+f'{float(value):.4g}' if value!='' else '' for relation,value in zip(rows[stem+'_kpss_relation'],rows[stem+'_kpss_p'])]
    show=['diagnostic_id','variables','frequency','n_periods','calendar_gaps','state','primary_adf_p','primary_kpss_bound','legacy_adf_p','legacy_kpss_bound']
    downloads={name:base64.b64encode(gzip.compress((OUT/name).read_bytes(),mtime=0)).decode() for name in ['diagnostics.csv','tests.csv','model_inventory.csv','source_hashes.json']}
    payload={'rows':rows[show].to_dict('records'),'downloads':downloads}
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0)).decode()
    counts=summary['counts'];states=summary['diagnostic_states']
    source_table=''.join(f'<tr><td>{html.escape(key)}</td><td>{value}</td></tr>' for key,value in states.items())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exact-calendar stationarity appendix</title><style>body{font:15px system-ui;margin:32px auto;max-width:1180px;padding:0 20px;color:#172d32;background:#f7faf8}h1{font-size:29px}p{max-width:1000px;line-height:1.55}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #dce7e1;text-align:left;font-size:12px}th{background:#e5efea}input,select,button{font:inherit;padding:8px;margin:5px 8px 10px 0}a{color:#176b60}.stats{padding:14px;background:#e5efea}#wrap{overflow:auto}#downloads{margin:20px 0}.small{font-size:12px;color:#425f63}</style><h1>Exact-calendar stationarity appendix</h1><p>This diagnostic appendix covers the seven extension families and their exact saved model calendars. It does not change regression samples, coefficients, inference or the requested sector-CR1 main results. Levels remain descriptive. ADF/KPSS screens do not certify stationarity, causality or valid small-sample regression inference.</p><p class="stats">COUNTS</p><p>Gapped calendars are never compressed. Missing, short and constant series are explicitly untestable. Primary: ADF with AIC and lag cap 24 monthly / 8 quarterly / 2 annual; automatic KPSS bandwidth. Sensitivity: ADF caps 12 / 4 / 1 and KPSS automatic bandwidth capped there. Minimum histories: 60 months, 20 quarters or 20 years. Rejection decisions use a strict 5% threshold; mixed and lag-sensitive results are retained.</p><p>Tests use actual non-FE regressor bases before fixed-effect projection, including AR growth, endpoint-difference/growth-sum channels and centered quadratic weather terms. FE indicators, constructed trends and zero-padded sector interactions are not unit-root tested. Monthly/quarterly weather and NSA output have deterministic season offsets removed alongside a retained trend; annual series have no seasonal adjustment. Constant/trend screens are marginal zero-frequency diagnostics, not seasonal-unit-root, break or cointegration tests. A linear trend does not capture every deterministic pattern of a squared basis.</p><table aria-label="Diagnostic state counts"><thead><tr><th>Evidence state</th><th>Distinct diagnostics</th></tr></thead><tbody>STATES</tbody></table><div id="downloads">Loading embedded downloads…</div><label>State <select id="state"><option value="">All states</option></select></label><label>Frequency <select id="frequency"><option value="">All</option><option>M</option><option>Q</option><option>Y-DEC</option></select></label><label>Find <input id="search" placeholder="Variable or diagnostic ID"></label><button id="back">Previous</button><button id="next">Next</button><p id="status" class="small"></p><div id="wrap"><table><thead><tr>HEADERS</tr></thead><tbody id="body"></tbody></table></div><p class="small">The full downloads include every source-specification row, intended unavailable design, source/sample/design hashes, per-unit outcome link and shared-basis unit coverage. Identical arrays and policies are deduplicated; model counts are not independent test counts. Untestable raw values are retained without fabricated test arrays. This file contains its own data and uses no network requests.</p><script>const blob="PACKED";const columns=COLUMNS;let data,offset=0;const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));async function inflate(text){const bytes=Uint8Array.from(atob(text),c=>c.charCodeAt(0));return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text()}function draw(){const state=$('state').value,freq=$('frequency').value,q=$('search').value.toLowerCase();const rows=data.rows.filter(r=>(!state||r.state===state)&&(!freq||r.frequency===freq)&&(!q||(r.variables+' '+r.diagnostic_id).toLowerCase().includes(q)));offset=Math.max(0,Math.min(offset,Math.floor(Math.max(rows.length-1,0)/100)*100));$('body').innerHTML=rows.slice(offset,offset+100).map(r=>'<tr>'+columns.map(c=>'<td>'+esc(typeof r[c]==='number'&&!Number.isInteger(r[c])?r[c].toPrecision(4):r[c])+'</td>').join('')+'</tr>').join('');$('status').textContent=`${rows.length} diagnostics; showing ${rows.length?offset+1:0}–${Math.min(offset+100,rows.length)}. KPSS p-values may be censored; inspect tests.csv for bound symbols and warnings.`}async function start(){data=JSON.parse(await inflate(blob));for(const value of [...new Set(data.rows.map(r=>r.state))].sort())$('state').add(new Option(value,value));$('downloads').innerHTML=Object.keys(data.downloads).map(name=>'<button data-download="'+esc(name)+'">'+esc(name)+'</button>').join('');$('downloads').onclick=async e=>{const name=e.target.dataset.download;if(!name)return;const text=await inflate(data.downloads[name]);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};for(const id of ['state','frequency','search'])$(id).oninput=()=>{offset=0;draw()};$('back').onclick=()=>{offset-=100;draw()};$('next').onclick=()=>{offset+=100;draw()};draw()}start().catch(e=>{$('status').textContent='Appendix data could not load: '+e.message;throw e});</script></html>'''
    text=f"{counts['logical_model_outcomes']:,} model/outcome requests · {counts['model_variable_links']:,} variable links · {counts['distinct_diagnostics']:,} distinct diagnostics · {counts['completed_diagnostics']:,} completed screens · {counts['unavailable_diagnostics']:,} unavailable"
    page=page.replace('COUNTS',text).replace('STATES',source_table).replace('HEADERS',''.join('<th>'+html.escape(c)+'</th>' for c in show)).replace('PACKED',packed).replace('COLUMNS',json.dumps(show))
    inventory=pd.read_csv(OUT/'model_inventory.csv')
    family_table='<h2>Model coverage</h2><table><thead><tr><th>Extension family</th><th>Model/outcome requests</th></tr></thead><tbody>'+''.join('<tr><td>'+html.escape(branch)+'</td><td>'+str(count)+'</td></tr>' for branch,count in inventory.groupby('branch',sort=False).size().items())+'</tbody></table>'
    page=page.replace('<div id="downloads">',family_table+'<p class="small">Complete raw arrays are saved separately in outputs/extension_stationarity/series.csv; every per-model basis/unit/source binding is in model_variable_coverage.csv. They remain part of the replication outputs. This compact standalone appendix embeds its full diagnostic table, tests, model inventory and hash manifest; it does not embed those two large audit ledgers.</p><div id="downloads">')
    page=page.replace('The full downloads include every source-specification row, intended unavailable design, source/sample/design hashes, per-unit outcome link and shared-basis unit coverage.', 'The model inventory includes every source-specification row and intended unavailable design with its sample/design hashes; the separate coverage ledger supplies every per-unit outcome and shared-basis link.')
    path=ROOT/'outputs/extension_stationarity.html';path.write_text(page)
    return path


def run_extension_stationarity():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.json').write_text('{"status":"running"}\n')
    audit=Appendix()
    for path in [Path(__file__),ROOT/'scripts/stationarity_audit.py',ROOT/'scripts/stationarity_diagnostics.py',ROOT/'outputs/pyfixest/fit_log.csv',ROOT/'outputs/pyfixest/estimator_manifest.json']:audit.bind(path)
    with threadpool_limits(limits=1):
        for branch,path in FAMILIES.items():
            source=audit.frame(ROOT/path)
            ledger=Path(path).parent/'pyfixest'
            for name in ['fit_log.csv','estimator_manifest.json']:audit.bind(ROOT/ledger/name)
            for index,row in source.iterrows():
                record=row.to_dict();outcomes=[record['outcome']] if present(record.get('outcome')) else ['y_nsa','y_sa']
                for outcome in outcomes:audit.register(branch,path,index,record,outcome)
            print(f"Stationarity inventory {branch}: {len(source)} saved rows, {len(audit.diagnostics)} distinct arrays so far",flush=True)
    diagnostics=[]
    for row in audit.diagnostics.values():
        result={key:value for key,value in row.items() if not key.startswith('_')}
        result['variables']=json.dumps(sorted(row['_variables']));result['roles']=json.dumps(sorted(row['_roles']));diagnostics.append(result)
    frames={'diagnostics':pd.DataFrame(diagnostics),'tests':pd.DataFrame(audit.tests),'series':pd.DataFrame(audit.series),
            'model_variable_coverage':pd.DataFrame(audit.links),'model_inventory':pd.DataFrame(audit.models)}
    for name,frame in frames.items():frame.to_csv(OUT/f'{name}.csv',index=False)
    assert all(sha(ROOT/path)==value for path,value in audit.hashes.items()),'Frozen analytical or test-helper input changed'
    (OUT/'source_hashes.json').write_text(json.dumps(audit.hashes,indent=2)+'\n')
    d=frames['diagnostics'];completed=int(d.test_status.eq('completed').sum())
    summary={'status':'computed_not_yet_independently_verified','families':FAMILIES,
        'counts':{'source_spec_rows':sum(len(audit.frames[path]) for path in FAMILIES.values()),'logical_model_outcomes':len(audit.models),'model_variable_links':len(audit.links),
            'distinct_diagnostics':len(d),'completed_diagnostics':completed,'unavailable_diagnostics':int(d.test_status.eq('not_run').sum()),'error_diagnostics':int(d.test_status.eq('error').sum()),'test_rows':len(audit.tests),'series_rows':len(audit.series)},
        'diagnostic_states':d.state.value_counts().to_dict(),'source_hashes_sha256':sha(OUT/'source_hashes.json'),
        'policy':{'primary':'ADF AIC cap2h; uncapped automatic KPSS','sensitivity':'ADF AIC caph; automatic KPSS cappedh','minimum_periods':{'M':60,'Q':20,'Y-DEC':20},
            'deterministic':'c growth/AR/rain/difference; ct levels/temperature/RH/wind/PM/temperature quadratic bases','seasonality':'M/Q weather and NSA output: subtract season offsets estimated with retained trend; annual:none',
            'dedup_serialization':'SHA256(sorted compact JSON frequency/deterministic/seasonal_rule + null byte + little-endian int64 period ordinals + little-endian float64 actual raw basis values; canonical NaN bits)',
            'calendar':'Any gaps/missingness unavailable before minimum-length or constant checks; no compression or longest-run substitution',
            'basis':'Actual archived non-FE columns before projection where available; frozen sample/formula bases for annual/no-fit fallback. H1 main raw basis covers within-sector interactions; time/FE/dummy-zero arrays excluded'},
        'scope':'Marginal ordinary zero-frequency screens only. No stationarity certificate, seasonal-unit-root, break, cointegration, causal-validity or inference calibration claim; no sample deletion or model selection',
        'outputs_sha256':{name+'.csv':sha(OUT/(name+'.csv')) for name in frames}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    render_extension_stationarity()
    print(json.dumps(summary['counts'],indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reports-only',action='store_true');args=parser.parse_args()
    render_extension_stationarity() if args.reports_only else run_extension_stationarity()
