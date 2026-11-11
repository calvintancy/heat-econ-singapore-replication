"""Independent source-cell and real-JavaScript audit of the arrival report.

Presentation only: consumes byte-bound numerical receipts and saved result CSVs;
does not import production preparation, estimation, covariance or report code.
"""
from __future__ import annotations
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd
from lxml import html
from significance import significance_stars
from presentation_verification_state import persist,presentation_run

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/visitor_rigour'
BRANCHES=['monthly','origin_year','annual','ppml'];VARIANTS=['all_eligible','event_dates_excluded','strict_growth_window_clean']
DIMENSIONS=['panel','outcome','control_set','model_spec','model']
ALIASES={'parent_growth_periods':'parent_n_periods','shared_cut_alignment_loss_periods':'shared_panel_alignment_loss_periods','control_raw_missing_periods':'control_raw_missing_loss_periods','control_coverage_excluded_periods':'control_coverage_loss_periods','crisis_removed_periods':'crisis_loss_periods','source_count_months':'parent_n_periods','removed_for_event_dates':'crisis_loss_periods'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    try:return pd.read_csv(path,dtype={'unit_id':str,'sector_id':str},float_precision='round_trip',low_memory=False)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def equal(a,b,label):
    if b is None or isinstance(b,(float,np.floating)) and not np.isfinite(b):assert a is None,(label,a,b)
    elif isinstance(b,(int,float,np.number)) and not isinstance(b,(bool,np.bool_)):assert isinstance(a,(int,float)) and math.isclose(a,float(b),abs_tol=2e-14,rel_tol=2e-13),(label,a,b)
    else:assert a==b,(label,a,b)


def normalized_frames(branch):
    frames={name:read(OUT/branch/(name+'.csv')) for name in ['specifications','temperature_effects','control_estimates','channel_estimates','interaction_components','tests'] if (OUT/branch/(name+'.csv')).exists()}
    specs=frames['specifications'];rows={row['spec_id']:row for row in specs.to_dict('records')}
    for row in rows.values():
        for field,value in [('model_spec','current'),('model','common'),('outcome','count_level'),('outcome_label','Expected arrival counts')]:row.setdefault(field,value)
        if branch=='annual':row['outcome_label']='Annual log arrivals growth (NSA; 100 log points)'
        if branch=='ppml':row['outcome_label']='Expected arrival count levels (PPML; 100 log-mean points)'
    for name,frame in list(frames.items()):
        normalized=[]
        for row in frame.to_dict('records'):
            for key,value in rows[row['spec_id']].items():row.setdefault(key,value)
            if 'unit_id' not in row and 'sector_id' in row:row['unit_id']=row['sector_id']
            if 'unit' not in row and 'sector' in row:row['unit']=row['sector']
            if 'p_value_holm_within_profile' in row:row['p_value_holm_profile']=row['p_value_holm_within_profile']
            if branch=='annual' and name=='interaction_components':row.setdefault('feature',row['term'])
            if 'fragile' in row:row.setdefault('joint_test_fragile',row['fragile'])
            for old,new in ALIASES.items():
                if old in row:row.setdefault(new,row[old])
            if name=='specifications':
                if branch=='annual':row['outcome_label']='Annual log arrivals growth (NSA; 100 log points)'
                if branch=='ppml':row['outcome_label']='Expected arrival count levels (PPML; 100 log-mean points)'
            normalized.append(row)
        frames[name]=pd.DataFrame(normalized)
    return frames


def check_rows(saved,source,label):
    assert len(saved)==len(source),(label,len(saved),len(source))
    for i,(actual,original) in enumerate(zip(saved,source.to_dict('records'),strict=True)):
        for field,value in actual.items():assert field in original,(label,i,field);equal(value,original[field],(label,i,field))
    return len(saved)


def term_label(term):
    if term.startswith('temp_l'):return 'Temperature · lag '+term[6:]
    if term.startswith('rain_l'):return 'Rainfall / 100 mm · lag '+term[6:]
    if term.startswith('ar_l'):return 'Lagged growth · '+term[4:]
    return {'temp_c':'Temperature (°C)','rain_100mm':'Rainfall (100 mm)','rh_pct':'Relative humidity (pp)','wind_mps':'Wind (m/s)','pm25_10ug':'PM2.5 (10 μg/m³)','lny_window_days':'Lunar New Year window days','lny_lag1':'Prior-month holiday window days','sum_temperature_lags':'Sum of direct temperature coefficients','temp_endpoint_difference':'Endpoint temperature difference','temp_growth_sum':'Trailing-year temperature SUM'}.get(term,term)


NODE=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert'),crypto=require('crypto');const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),data=input.data,nodes=new Map(),blobs=new Map(),links=[];let next=0,historyCalls=0;
const decode=s=>String(s).replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');
const strip=s=>decode(String(s).replace(/<[^>]*>/g,'')).replace(/\s+/g,' ').trim();
function node(id){if(nodes.has(id))return nodes.get(id);let value='',markup='';const isSelect=input.select_ids.includes(id);const item={id,textContent:'',hidden:false,dataset:{},options:[],attributes:{},setAttribute(k,v){this.attributes[k]=v;},add(o){this.options.push(o);},get value(){return value;},set value(v){value=String(v);},get innerHTML(){return markup;},set innerHTML(s){markup=String(s);if(isSelect){this.options=[...markup.matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)].map(m=>({value:decode(m[1].match(/\bvalue="([^"]*)"/)?.[1]??strip(m[2]))}));value=this.options[0]?.value??'';}}};nodes.set(id,item);return item;}
for(const id of input.ids)node(id);node('visitor-rigour-data').textContent=input.packed;const tabs=['results','sources','diagnostics','stata','methods','downloads'].map(id=>{const n=node('tab-'+id);n.dataset.tab=id;return n;});
const sandbox={document:{getElementById(id){assert(nodes.has(id),'Unknown DOM id '+id);return node(id);},querySelectorAll(query){assert.equal(query,'[data-tab]');return tabs;},createElement(tag){assert.equal(tag,'a');return {click(){links.push(this);}};}},Blob,Response,DecompressionStream,Uint8Array,atob,Option:function(text,value){return {text,value};},URL:{createObjectURL(blob){const key='blob:'+next++;blobs.set(key,blob);return key;},revokeObjectURL(){}},setTimeout(fn){fn();},history:{replaceState(){historyCalls++;throw Error('Opaque-file history mutation');}}};sandbox.window=sandbox;vm.createContext(sandbox);vm.runInContext(input.script,sandbox);const actualData=await sandbox.visitorRigourReady;assert.equal(JSON.stringify(actualData),JSON.stringify(data));
const keys=['branch','panel','outcome','control_set','model_spec','model','method'];for(const k of keys)assert.equal(node(k).value,data.default[k]);
assert(/299.*25.*20.*10.*not independent hypotheses/.test(node('model-counts').textContent));
const number=x=>x==null||!Number.isFinite(+x)?'—':Number(x).toFixed(3).replace('-','−');const probability=x=>x==null?'Unavailable':x<.001?'<0.001':number(x);
function rows(id){return [...(node(id).innerHTML.match(/<tbody[^>]*>([\s\S]*?)<\/tbody>/)?.[1]??'').matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g)].map(m=>[...m[1].matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g)].map(x=>strip(x[1])));}
function csvRows(text){let result=[],row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell='';}else if(c==='\n'&&!quoted){row.push(cell);result.push(row);row=[];cell='';}else cell+=c;}return result;}
let views=0,readouts=0,effectCells=0,controlCells=0,componentCells=0,testCells=0,lossCells=0,exports=0,diagnosticPages=0;const form=x=>typeof x==='number'?number(x):String(x??'—');
function checkGeneric(id,source,fields){const actual=rows(id);assert.equal(actual.length,source.length,id+' count');for(let i=0;i<source.length;i++)assert.deepEqual(actual[i],fields.map(k=>form(source[i][k])),id+' '+i);return actual.length*fields.length;}
for(const view of data.views){
 for(const key of keys){node(key).value=view[key];node(key).onchange();}assert.equal(node('native').innerHTML,view.table||'<p>No native point estimate is available for this request. See the sample reasons below.</p>');
 assert(node('equation').textContent.includes(view.outcome_label));assert(node('equation').textContent.includes(view.branch==='annual'?'common linear trend':view.branch==='origin_year'?'origin × year FE':'common year FE'));
 const cards=[...node('cards').innerHTML.matchAll(/<div class="card">([\s\S]*?)<\/div>/g)];assert.equal(cards.length,view.requests.length);for(let i=0;i<cards.length;i++){const r=view.requests[i],text=strip(cards[i][1]);for(const value of [r.n_periods,r.n_obs,'time df '+r.time_residual_df,r.start+'–'+r.end])assert(text.includes(String(value)));}
 const bad=view.requests.filter(r=>r.status!=='estimated');for(const r of bad){assert(strip(node('unavailable').innerHTML).includes(r.sample_variant));assert(strip(node('unavailable').innerHTML).includes(r.reason||r.status));}assert.equal(!node('unavailable').innerHTML,bad.length===0);
 const thin=view.requests.some(r=>r.status==='estimated'&&r.time_residual_df<10);assert.equal(Boolean(node('caution').innerHTML),thin);
 lossCells+=checkGeneric('losses',view.requests,['sample_variant','parent_n_periods','output_history_loss_periods','weather_history_loss_periods','shared_panel_alignment_loss_periods','control_raw_missing_loss_periods','control_coverage_loss_periods','crisis_loss_periods','n_periods','calendar_gaps']);
 const expectedUnits=[...new Map((view.temperature_effects||[]).map(r=>[r.unit_id,r.unit||r.unit_id])).keys()];assert.deepEqual(node('unit').options.map(o=>o.value),['__all__',...expectedUnits]);
 for(const variant of view.requests.map(r=>r.sample_variant)){
  node('sample').value=variant;node('sample').onchange();
  for(const unit of ['__all__',...new Set([expectedUnits[0],expectedUnits.at(-1)].filter(Boolean))]){
   node('unit').value=unit;node('unit').onchange();const filter=r=>r.sample_variant===variant&&(unit==='__all__'||r.unit_id===unit);const source=[...(view.temperature_effects||[]),...(view.channel_estimates||[])].filter(filter),actual=rows('effects');assert.equal(actual.length,source.length);
   for(let i=0;i<source.length;i++){const r=source[i];assert.deepEqual(actual[i],[String(r.unit||r.unit_id)+' · '+r.term,number(r.coefficient),number(r.std_error),'['+number(r.ci_lower)+', '+number(r.ci_upper)+']',probability(r.p_value),r.p_value==null?'Unavailable':r.p_value<.05?'Yes':'No',probability(r.p_value_holm_profile),number(r.mean_y)+' / '+number(r.sd_y)]);effectCells+=8;}
   controlCells+=checkGeneric('controls',(view.control_estimates||[]).filter(r=>r.sample_variant===variant),['term','coefficient','std_error','p_value','finite_sample_correction','inference_df']);
   componentCells+=checkGeneric('components',(view.interaction_components||[]).filter(filter),['unit_id','feature','component','coefficient','std_error','p_value','normalization','estimated']);
   const tests=(view.tests||[]).filter(r=>r.sample_variant===variant&&(unit==='__all__'||r.unit_id===unit||r.unit_id==='all'));const saved=rows('tests');assert.equal(saved.length,tests.length);for(let i=0;i<tests.length;i++){const t=tests[i],status=t.status==='available'&&t.restrictions>1?'Computable; calibration not established':t.status;assert.deepEqual(saved[i],[form(t.unit_id),form(t.test),form(t.restrictions),form(t.covariance_rank),form(t.f_statistic),probability(t.p_value),status,form(t.fragility_reason)]);if(String(t.status).startsWith('unavailable:'))assert.equal(t.p_value,null);testCells+=8;}
   node('download-selected').onclick();const link=links.at(-1),text=await blobs.get(link.href).text(),parsed=csvRows(text),columns=[...new Set(source.flatMap(Object.keys))];assert.deepEqual(parsed[0],columns);assert.equal(parsed.length,source.length+1);for(let i=0;i<source.length;i++)assert.deepEqual(parsed[i+1],columns.map(k=>String(source[i][k]??'')));assert.equal(link.download,view.id+'__'+variant+'__'+unit+'.csv');exports++;readouts++;
  }
 }
 views++;
}
for(const tab of tabs){tab.onclick();for(const id of ['results','sources','diagnostics','stata','methods','downloads'])assert.equal(node(id).hidden,id!==tab.dataset.tab);}
for(const[name,markup]of Object.entries(data.stata.tables)){node('stata-table').value=name;node('stata-table').onchange();assert.equal(node('stata-native').innerHTML,markup);}
const d=data.diagnostics;function checkDiagnostic(expected,offset){const begin=Math.max(0,Math.min(offset,Math.floor(Math.max(expected.length-1,0)/100)*100));const actual=rows('diagnostic-table');assert.deepEqual(actual,expected.slice(begin,begin+100).map(r=>d.display_columns.map(c=>form(r[c]))));assert(node('diagnostic-status').textContent.startsWith(expected.length+' diagnostics; '+(expected.length?begin+1:0)+'–'+Math.min(begin+100,expected.length)+'.'));diagnosticPages++;}
for(const state of ['',...new Set(d.rows.map(r=>r.state))]){node('diagnostic-state').value=state;node('diagnostic-search').value='';node('diagnostic-state').oninput();const expected=d.rows.filter(r=>!state||r.state===state);checkDiagnostic(expected,0);for(let offset=100;offset<expected.length;offset+=100){node('diagnostic-next').onclick();checkDiagnostic(expected,offset);}node('diagnostic-next').onclick();checkDiagnostic(expected,expected.length);node('diagnostic-back').onclick();}
for(const query of ['y_mom_nsa','arrival_count','NO_SUCH_VARIABLE',d.rows[0].diagnostic_id]){node('diagnostic-state').value='';node('diagnostic-search').value=query;node('diagnostic-search').oninput();checkDiagnostic(d.rows.filter(r=>JSON.stringify(r).toLowerCase().includes(query.toLowerCase())),0);}
let exactDownloads=0;for(const [kind,items]of [['download-list',input.downloads],['diagnostic-downloads',input.diagnostic_downloads]])for(const[name,digest]of Object.entries(items)){await node(kind).onclick({target:{dataset:kind==='download-list'?{export:name}:{diag:name}}});const link=links.at(-1),bytes=Buffer.from(await blobs.get(link.href).arrayBuffer());assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),digest);exactDownloads++;}
assert.equal(historyCalls,0);process.stdout.write(JSON.stringify({status:'passed',async_gzip_bootstrap:true,selector_views:views,readouts,effect_cells:effectCells,control_cells:controlCells,component_cells:componentCells,joint_cells:testCells,sample_loss_cells:lossCells,selected_csv_exports:exports,byte_exact_downloads:exactDownloads,diagnostic_pages:diagnosticPages,stata_tables:Object.keys(data.stata.tables).length,tabs:tabs.length,history_mutations:historyCalls,scope:'Actual embedded JavaScript executed in NodeVM DOM adapter; separate actual-browser receipt required for visual/browser claims'}));
"""


def _verify():
    path=ROOT/'outputs/visitor_arrivals_rigour.html';raw=path.read_text();doc=html.fromstring(raw,parser=html.HTMLParser(huge_tree=True));inventory=json.loads((OUT/'report_inventory.json').read_text())
    assert not doc.xpath('//script[@src]|//link[@rel="stylesheet"]|//img[@src]') and not doc.xpath('//table')
    ids=doc.xpath('//@id');assert len(ids)==len(set(ids))
    packed=doc.get_element_by_id('visitor-rigour-data').text_content().strip();decoded=gzip.decompress(base64.b64decode(packed));data=json.loads(decoded)
    assert hashlib.sha256(decoded).hexdigest()==inventory['payload_sha256'] and sha(path)==inventory['html_sha256']
    assert data['sources']==inventory['source_sha256']
    checked={str(path.relative_to(ROOT)):sha(path),'outputs/visitor_rigour/report_inventory.json':sha(OUT/'report_inventory.json')}
    for name,digest in data['sources'].items():assert sha(ROOT/name)==digest;checked[name]=digest
    receipts={}
    for branch in BRANCHES+['diagnostics']:
        receipt_path=OUT/branch/'verification.json';receipt=json.loads(receipt_path.read_text());assert receipt['status']=='passed' and receipt.get('verified_artifacts_sha256')
        for name,digest in receipt['verified_artifacts_sha256'].items():assert sha(ROOT/name)==digest;checked[name]=digest
        assert sha(ROOT/'scripts/verification_artifacts.py')==receipt['artifact_binding_helper_sha256']
        receipts[branch]=sha(receipt_path);checked[str(receipt_path.relative_to(ROOT))]=receipts[branch]
    views=data['views'];assert len(views)==inventory['views']==284 and len({v['id'] for v in views})==284
    assert sum(bool(v['table']) for v in views)==inventory['native_tables']==272
    caches={branch:normalized_frames(branch) for branch in BRANCHES};expected=set();source_rows=request_rows=coefficient_cells=stat_cells=0;native_ids=set();unavailable=0
    for branch,frames in caches.items():
        specs=frames['specifications'];counts=data['counts'][branch];assert len(specs)==counts['requests']
        assert counts['estimated_requests']==specs.status.eq('estimated').sum()
        native=int(specs.native_fit_executed.sum()) if branch=='ppml' else int(specs.fit_id.dropna().nunique())
        assert counts['native_fits']==native=={'monthly':260,'origin_year':24,'annual':15,'ppml':20}[branch]
        assert counts['aliases']==(10 if branch=='ppml' else 0)
        for group,requests in specs.groupby(DIMENSIONS,sort=False,dropna=False):
            effects=frames['temperature_effects'];selected=effects.loc[effects.spec_id.isin(requests.spec_id)]
            for method,band in selected[['covariance','hac_lags']].drop_duplicates().itertuples(index=False,name=None):expected.add((branch,*group,method,int(band)))
    assert expected=={(v['branch'],*(v[k] for k in DIMENSIONS),v['covariance'],v['hac_lags']) for v in views}
    for view in views:
        frames=caches[view['branch']];specs=frames['specifications'];mask=np.ones(len(specs),bool)
        for key in DIMENSIONS:mask &= specs[key].eq(view[key])
        requests=specs.loc[mask];request_rows+=check_rows(view['requests'],requests,(view['id'],'requests'));ids_here=set(requests.spec_id)
        sources={}
        for name in ['temperature_effects','control_estimates','channel_estimates','interaction_components','tests']:
            frame=frames.get(name,pd.DataFrame());selected=frame.loc[frame.spec_id.isin(ids_here)] if len(frame) else frame
            if 'covariance' in selected:selected=selected.loc[selected.covariance.eq(view['covariance'])&selected.hac_lags.eq(view['hac_lags'])]
            source_rows+=check_rows(view[name],selected,(view['id'],name));sources[name]=selected
        if view['branch']=='annual':assert 'Annual log' in view['outcome_label']
        if view['branch']=='ppml':assert 'PPML' in view['outcome_label'] and 'count' in view['outcome_label']
        effects=sources['temperature_effects'];channels=sources['channel_estimates'];controls=sources['control_estimates']
        if view['model']=='H1':selected=channels.copy() if view['model_spec']=='two_channel' else effects.loc[effects.effect.eq('current' if view['branch']=='annual' else 'cumulative_direct')].copy()
        elif view['branch']=='ppml':selected=effects.copy()
        else:
            selected=channels.copy() if view['model_spec']=='two_channel' else effects.loc[effects.effect.eq('current' if view['branch']=='annual' else 'lag')].copy()
            selected=pd.concat([selected,controls],ignore_index=True)
        selected=selected.loc[selected.coefficient.notna()]
        if selected.empty:assert not view['table'] and requests.status.ne('estimated').all();unavailable+=1;continue
        fragment=html.fragment_fromstring(view['table'],create_parent='div',parser=html.HTMLParser(huge_tree=True));assert len(fragment.xpath('.//table[contains(@class,"gt_table")]'))==1
        current_ids=fragment.xpath('.//@id');assert len(current_ids)==len(set(current_ids)) and not native_ids.intersection(current_ids);native_ids.update(current_ids)
        for header in fragment.xpath('.//@headers'):assert set(header.split())<=set(current_ids)
        selected=selected.copy();selected['row_key']=selected.unit_id.astype(str)+'|'+selected.term if view['model']=='H1' else selected.term
        row_keys=list(selected.drop_duplicates('row_key').row_key);variants=[v for v in VARIANTS if selected.sample_variant.eq(v).any()]
        coef=[];inside=False;stats={}
        for row in fragment.xpath('.//tbody/tr'):
            cells=row.xpath('./th|./td')
            if row.xpath('./th[contains(@class,"gt_group_heading")]'):inside=row.text_content().strip()=='coef'
            elif inside:coef.append(row)
            if len(cells)==len(variants)+1:stats[cells[0].text_content().strip()]=cells[1:]
        assert len(coef)==len(row_keys)
        indexed=selected.set_index(['sample_variant','row_key']);assert indexed.index.is_unique
        for key,row in zip(row_keys,coef,strict=True):
            first=selected.loc[selected.row_key.eq(key)].iloc[0];label=str(first.unit)+(' · '+term_label(first.term) if view['model_spec']=='two_channel' else '') if view['model']=='H1' else term_label(first.term)
            assert row.xpath('./th')[0].text_content().strip()==label
            cells=row.xpath('./td');assert len(cells)==len(variants)
            for variant,cell in zip(variants,cells,strict=True):
                original=indexed.loc[(variant,key)];text=cell.text_content().replace('−','-');numbers=re.findall(r'[-+]?\d[\d,]*\.\d+',text)
                assert [n.replace(',','') for n in numbers]==[f'{original.coefficient:.3f}',f'{original.std_error:.3f}'],(view['id'],key,variant,text)
                assert text.count('*')==len(significance_stars(original.p_value));coefficient_cells+=1
        for j,variant in enumerate(variants):
            source=requests.loc[requests.sample_variant.eq(variant)].iloc[0]
            for label,value in [('Observations',source.n_obs),('Distinct weather dates',source.n_periods),('Parameters',source.n_parameters),('Residual time df',source.time_residual_df)]:assert float(stats[label][j].text_content().replace(',','').strip())==value;stat_cells+=1
            for label,value in [('First period',source.start),('Last period',source.end)]:assert stats[label][j].text_content().strip()==str(value);stat_cells+=1
            assert stats['Time controls'][j].text_content().strip()==('Common linear trend' if view['branch']=='annual' else 'Origin × year FE' if view['branch']=='origin_year' else 'Common year FE')
            assert stats['Inference'][j].text_content().strip()==view['method']
            assert stats['Weather controls'][j].text_content().strip()==('Full controls, including wind / PM2.5' if view['control_set']=='full' else 'Long weather controls')
            if view['model']=='common':
                for label,value in [('Mean Y',source.mean_y),('SD Y',source.sd_y)]:assert stats[label][j].text_content().strip().replace(',','')==f'{value:.3f}';stat_cells+=1
            assert ('Origin × year FE' in stats)==(view['branch']=='origin_year')
            assert ('Common year FE' in stats)==(view['branch'] not in ['annual','origin_year'])
            assert ('Origin FE' in stats)==(view['branch']=='annual' and source.n_sectors>1)
            assert ('Origin × calendar-month FE' in stats)==(view['branch']=='origin_year' or view['branch']!='annual' and source.n_sectors>1)
            assert ('Calendar-month FE' in stats)==(view['branch'] not in ['annual','origin_year'] and source.n_sectors==1)
    assert unavailable==12
    assert data['diagnostics']==json.loads((OUT/'diagnostics/report_payload.json').read_text())
    assert data['diagnostics']['counts']['logical_model_outcomes']==354
    assert data['stata']['proof']==json.loads((ROOT/'outputs/stata_key_results/arrivals/verification.json').read_text()) and data['stata']['proof']['status']=='passed'
    expected_stata={p.stem:p.read_text() for p in sorted((ROOT/'outputs/stata_key_results/arrivals/native_tables').glob('*.html'))};assert data['stata']['tables']==expected_stata and len(expected_stata)==22
    for p in (ROOT/'outputs/stata_key_results/arrivals/native_tables').glob('*.html'):checked[str(p.relative_to(ROOT))]=sha(p)
    downloads={};diagnostic_downloads={}
    for name,packed_item in data['downloads'].items():
        target=ROOT/'outputs/stata_key_results/arrivals'/name.removeprefix('stata/') if name.startswith('stata/') else OUT/name
        assert gzip.decompress(base64.b64decode(packed_item))==target.read_bytes();downloads[name]=sha(target);checked[str(target.relative_to(ROOT))]=sha(target)
    for name,packed_item in data['diagnostics']['downloads'].items():assert gzip.decompress(base64.b64decode(packed_item))==(OUT/'diagnostics'/name).read_bytes();diagnostic_downloads[name]=sha(OUT/'diagnostics'/name)
    text=doc.text_content()
    for qualifier in ['Gaps are not compressed','not an unconditional long-run multiplier','calendar correction','calibration','zero','not independent hypotheses']:
        # Counts phrasing is filled by real JS, checked in the runtime below.
        if qualifier!='not independent hypotheses':assert qualifier.lower() in text.lower(),qualifier
    scripts=[s.text_content() for s in doc.xpath('//script') if s.get('type')!='application/octet-stream'];assert len(scripts)==1
    with tempfile.TemporaryDirectory(prefix='visitor-rigour-ui-') as temporary:
        input_path=Path(temporary)/'input.json';input_path.write_text(json.dumps({'script':scripts[0],'packed':packed,'data':data,'ids':ids+['branch','panel','outcome','control_set','model_spec','model','method'],'select_ids':['branch','panel','outcome','control_set','model_spec','model','method','sample','unit','diagnostic-state','stata-table'],'downloads':downloads,'diagnostic_downloads':diagnostic_downloads},allow_nan=False))
        run=subprocess.run(['node','-e','(async()=>{\n'+NODE+'\n})().catch(e=>{console.error(e.stack);process.exitCode=1;});',str(input_path)],capture_output=True,text=True)
        assert run.returncode==0,run.stderr[-7000:];runtime=json.loads(run.stdout)
    for name,digest in checked.items():assert sha(ROOT/name)==digest,('Presentation input changed during verification',name)
    return {'status':'passed','views':284,'native_tables':272,'all_unavailable_views':12,'native_coefficient_cells':coefficient_cells,'native_stat_cells':stat_cells,'source_result_rows':source_rows,'source_request_rows':request_rows,'numerical_receipt_sha256':receipts,'verified_artifacts_sha256':checked,'html_sha256':sha(path),'payload_sha256':inventory['payload_sha256'],'runtime':runtime,'verifier_source_sha256':sha(Path(__file__)),'scope':'Source cells, exact numerical receipt bindings, native MakeTables gt tables, actual async embedded JS selectors/readouts/exports, diagnostics pagination and22native Stata tables. No model fits; browser/visual proof remains separate; no causal, stationarity or inference-calibration certificate.'}


def verify_visitor_rigour_report():
    with presentation_run(OUT/'report_verification.json','Source-bound new arrival four-branch/native-table/JavaScript audit'):
        result=_verify();persist(OUT/'report_verification.json',result);print('Visitor arrival report presentation passed:284views/272native tables/22Stata tables.',flush=True);return result


if __name__=='__main__':verify_visitor_rigour_report()
