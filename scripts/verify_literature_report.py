"""Verify all literature-report source rows, native tables and selector runtime.

This is a presentation audit, separate from each branch's numerical QR and raw
source audit. It never fits models, changes results, or claims visual-browser
coverage. The embedded JavaScript is actually executed in a Node VM DOM adapter.
"""
from __future__ import annotations

import hashlib
import base64
import gzip
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html

from significance import significance_stars
from presentation_verification_state import persist, presentation_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/literature_sensitivity"
VARIANTS = ["all_eligible", "event_dates_excluded", "strict_growth_window_clean"]
DIMENSIONS = ["context", "history", "outcome", "model_spec", "model"]
SOURCE_FIELDS = set("context context_label frequency outcome outcome_label structure specification model model_spec history sample_variant sector_id sector effect lag term coefficient std_error ci_lower ci_upper p_value mean_y sd_y n_obs n_sectors n_periods n_parameters time_residual_df calendar_gaps start end r_squared covariance hac_lags inference_df main_inference controls formula formula_rhs includes_dwellings estimator_version history_loss_periods weather_history_loss_periods maximum_output_support_horizon parent_n_periods parent_periods status reason test null restrictions covariance_rank covariance_condition f_statistic evaluation_temperature effect_units mean_temperature joint_test_fragile fragility_reason".split())
SOURCE_FIELDS.update("maximum_history_n_periods raw_history_loss_periods raw_AR_history_loss_periods target_alignment_loss_periods raw_model_history_eligible_n_periods matched_history_loss_periods control_raw_missing_loss_periods control_coverage_loss_periods control_total_loss_periods crisis_loss_periods".split())
SOURCE_FIELDS.add("spec_id")
SUBANNUAL_BRANCHES = {"subannual", "full_controls"}


NODE_HARNESS = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const decode=s=>String(s).replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');
const strip=s=>decode(String(s).replace(/<br\s*\/?\s*>/g,' ').replace(/<[^>]*>/g,'')).replace(/\s+/g,' ').trim();
const nodes=new Map(),downloads=[],blobs=new Map();let blobCounter=0,historyCalls=0;
function element(id,select=false){if(nodes.has(id))return nodes.get(id);let value='',markup='';const e={id,hidden:false,textContent:'',listeners:{},dataset:{},options:[],attributes:{},
 addEventListener(k,fn){this.listeners[k]=fn;},setAttribute(k,v){this.attributes[k]=String(v);},getAttribute(k){return this.attributes[k]??null;},
 appendChild(){},remove(){},click(){if(this.listeners.click)this.listeners.click();if(this.download)downloads.push({filename:this.download,href:this.href});},
 set innerHTML(s){markup=String(s);if(select){this.options=[...markup.matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)].map(m=>({value:decode(m[1].match(/\bvalue="([^"]*)"/)?.[1]??strip(m[2])),disabled:/\bdisabled\b/.test(m[1]),selected:/\bselected\b/.test(m[1])}));value=(this.options.find(o=>o.selected)||this.options[0])?.value??'';}},get innerHTML(){return markup;},
 set value(v){value=String(v);},get value(){return value;}};nodes.set(id,e);return e;}
for(const id of input.ids)element(id,input.select_ids.includes(id));
element('literature-data').textContent=input.payload_text;
Object.assign(element('literature-data').attributes,input.payload_attributes);
for(const [key,value] of Object.entries(input.payload_attributes))if(key.startsWith('data-'))element('literature-data').dataset[key.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=value;
const tabs=['regressions','yoy-guide','papers','methods'].map(tab=>{const e=element('tab-'+tab);e.dataset.tab=tab;return e;});
const NativeBlob=globalThis.Blob;
class Blob extends NativeBlob {constructor(parts,options){super(parts,options);this.parts=parts;}}
const URL={createObjectURL(blob){const key='blob:test-'+(++blobCounter);blobs.set(key,blob);return key;},revokeObjectURL(){}};
const document={getElementById(id){assert(nodes.has(id),'Unknown static element '+id);return nodes.get(id);},querySelectorAll(q){assert.equal(q,'[data-tab]');return tabs;},
 createElement(tag){return element('created-'+tag+'-'+Math.random());},body:element('__body'),documentElement:element('__html')};
const sandbox={document,Blob,URL,Response:globalThis.Response,DecompressionStream:globalThis.DecompressionStream,atob:globalThis.atob,
 TextDecoder:globalThis.TextDecoder,TextEncoder:globalThis.TextEncoder,setTimeout,clearTimeout,
 console,location:{hash:''},history:{replaceState(){historyCalls++;throw Error('SecurityError: opaque-origin history mutation');}}};sandbox.window=sandbox;sandbox.window.parent={};
vm.createContext(sandbox);const bootstrap=vm.runInContext(input.script,sandbox,{timeout:20000});
if(bootstrap&&typeof bootstrap.then==='function')await bootstrap;
if(sandbox.literatureReportReady)await sandbox.literatureReportReady;
for(let i=0;i<1000&&!element('inference').value;i++)await new Promise(resolve=>setTimeout(resolve,20));
if(input.payload_attributes['data-encoding']==='gzip-base64'){
 assert(sandbox.literatureReportReady&&typeof sandbox.literatureReportReady.then==='function','Async bootstrap readiness promise missing');
 assert.equal(element('literature-data').textContent,'','Decoded compressed payload not released');assert.equal(element('report-loading').hidden,true);
}
function change(id){assert(element(id).listeners.change,'Missing listener '+id);element(id).listeners.change();}
function number(x,d=3){return x==null||!Number.isFinite(+x)?'—':Number(x).toLocaleString('en',{minimumFractionDigits:d,maximumFractionDigits:d});}
function probability(x){return x==null||!Number.isFinite(+x)?'Unavailable':x<.001?'<0.001':number(x);}
function expectedTestLabel(t,v){
 if((v.model_spec==='dynamic'||v.model_spec.endsWith('AR1'))&&/static/i.test(t.test))return 'Conditional direct-weather level-channel shape (growth lags held fixed; not an exact structural static-level test)';
 if(v.model_spec==='two_channel'&&/static/i.test(t.test))return t.test+' · same null as cumulative direct response zero';
 if(v.model_spec==='DJO1'&&/static/i.test(t.test))return t.test+' · same null as temperature coefficient sum zero';
 if(['benchmark','DJO0'].includes(v.model_spec)&&/sum zero|Cumulative direct/i.test(t.test))return t.test+' · same null as the current temperature coefficient';
 return t.test;
}
function rows(markup){return [...(markup.match(/<tbody[^>]*>([\s\S]*?)<\/tbody>/)?.[1]??'').matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g)].map(m=>[...m[1].matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g)].map(c=>strip(c[1])));}
function csvRows(csv){let result=[],row=[],cell='',quoted=false;for(let i=0;i<csv.length;i++){const c=csv[i];if(c==='"'){if(quoted&&csv[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell='';}else if(c==='\n'&&!quoted){row.push(cell);result.push(row);row=[];cell='';}else cell+=c;}row.push(cell);result.push(row);return result;}
let viewChecks=0,sampleChecks=0,unitChecks=0,rowChecks=0,testChecks=0,downloadChecks=0,fragilityChecks=0,unavailableViews=0;
const initial=input.data.views.find(v=>v.id===element('inference').value);assert(initial,'No default view');
for(const [key,value] of Object.entries({branch:'subannual',context:'va20',outcome:'y_yoy_nsa',model_spec:'distributed',model:'H1'}))assert.equal(initial[key],value,'Default '+key);
function readout(view,variant,unit){
 const expected=view.rows.filter(r=>r.sample_variant===variant&&(unit==='__all__'||r.sector_id===unit)),markup=element('readout').innerHTML;
 const actual=rows(markup);assert.equal(actual.length,expected.length,view.id+' rows');
 for(let i=0;i<expected.length;i++){
  const r=expected[i],c=actual[i];assert.equal(c.length,9);
  assert.equal(c[0],strip(r.sector)+' '+strip(r.term_label||r.term));assert.equal(c[1],number(r.coefficient));assert.equal(c[2],number(r.std_error));
  assert.equal(c[3],'['+number(r.ci_lower)+', '+number(r.ci_upper)+']');assert.equal(c[4],probability(r.p_value));
  assert.equal(c[5],r.p_value==null?'Unavailable':r.p_value<.05?'Yes':'No');
  assert.equal(c[6],number(r.mean_y));assert.equal(c[7],number(r.sd_y));assert.equal(c[8],number(r.evaluation_temperature));rowChecks++;
 }
 const testSource=view.tests.filter(t=>t.sample_variant===variant&&(unit==='__all__'||t.sector_id===unit||t.sector_id==='all'||t.sector_id==='all_sectors'));
 const rendered=rows(element('tests').innerHTML);assert.equal(rendered.length,testSource.length);
 for(let i=0;i<testSource.length;i++){
  const t=testSource[i],c=rendered[i];assert.equal(c.length,6);assert(c[0].includes(expectedTestLabel(t,view)));assert.equal(c[1],String(t.restrictions));assert.equal(c[2],number(t.covariance_rank,0));
  assert.equal(c[3],number(t.f_statistic));assert.equal(c[4],probability(t.p_value));
  assert(c[5].includes(t.status==='available'&&t.restrictions>1?'Computable (calibration not established)':t.status));
  assert(element('tests').innerHTML.includes('Nominal Wald p'));
  if(t.status!=='available')assert.equal(t.p_value,null);
  if(t.joint_test_fragile){assert(c[5].includes(t.fragility_reason));fragilityChecks++;}
  testChecks++;
 }
 for(const id of ['readout','tests','cards','unavailable'])assert(!/undefined|NaN/.test(element(id).innerHTML),'Malformed '+id);
 const before=downloads.length;element('download').click();
 if(expected.length){assert.equal(downloads.length,before+1);const download=downloads.at(-1),blob=blobs.get(download.href);assert(blob);
  const csv=csvRows(blob.parts.join('')),fields=Object.keys(expected[0]);assert.deepEqual(csv[0],fields);assert.equal(csv.length,expected.length+1);
   for(let i=0;i<expected.length;i++)assert.deepEqual(csv[i+1],fields.map(field=>String(expected[i][field]??'')));blobs.delete(download.href);downloadChecks++;
 }else assert.equal(downloads.length,before,'Empty readout exported phantom results');
}
for(const view of input.data.views){
 for(const [id,value] of [['branch',view.branch],['context',view.context],['outcome',view.outcome],['model-spec',view.model_spec],['model',view.model],['history',view.history],['inference',view.id]]){element(id).value=value;change(id);}
 assert.equal(element('native').innerHTML,view.table||'<p>No identified equation for this comparison.</p>');
 assert(element('title').textContent.includes(view.outcome_label));
 const restrictionNote=element('restriction-scope').textContent;
 if(view.model_spec==='dynamic'||view.model_spec.endsWith('AR1'))assert(restrictionNote.includes('not test the exact structural static-level null'));
 if(view.model_spec==='two_channel')assert(restrictionNote.includes('both test the growth-sum coefficient equal to zero'));
 if(view.model_spec==='DJO1')assert(restrictionNote.includes('same one restriction'));
 if(['benchmark','DJO0'].includes(view.model_spec))assert(restrictionNote.includes('identical estimates/tests'));
 if(['full_controls','annual_seasonal'].includes(view.branch))assert(!element('model').options.some(o=>o.value==='separate'));
 const unavailable=view.requests.filter(r=>r.status!=='estimated');
 assert.equal(Boolean(element('unavailable').innerHTML),Boolean(unavailable.length));
 for(const r of unavailable)assert(strip(element('unavailable').innerHTML).includes(r.reason||r.status));
 if(!view.table){assert(unavailable.length,'An available native table silently vanished');unavailableViews++;}
 if(view.model_spec==='dynamic'||view.model_spec.endsWith('AR1'))assert(element('interpretation').textContent.includes('not an unconditional long-run multiplier'));
 if(view.model_spec==='two_channel')assert(element('interpretation').textContent.includes('SUM'));
 const cards=[...element('cards').innerHTML.matchAll(/<div class="card">([\s\S]*?)<\/div>/g)];assert.equal(cards.length,3);
 const losses=rows(element('sample-losses').innerHTML);assert.equal(losses.length,3);
 for(let i=0;i<3;i++){const requested=view.requests.filter(r=>r.sample_variant===input.variants[i]);if(requested.length){const r=requested[0];assert(strip(cards[i][1]).includes(String(r.n_periods)));assert(strip(cards[i][1]).includes('residual time df '+(r.time_residual_df??'—')));
  const parent=r.parent_n_periods??r.parent_periods,max=r.maximum_history_n_periods??(parent!=null&&r.history_loss_periods!=null?parent-r.history_loss_periods:null),cells=losses[i];
  assert.equal(cells.length,8);assert.equal(cells[0],input.heads[i]);assert.equal(cells[1],number(parent,0));assert.equal(cells[2],number(r.raw_history_loss_periods,0));
  assert.equal(cells[3],view.branch==='full_controls'?number(r.control_total_loss_periods,0)+' ('+number(r.control_raw_missing_loss_periods,0)+' missing; '+number(r.control_coverage_loss_periods,0)+' coverage)':'Not applied');
  assert.equal(cells[4],number(r.target_alignment_loss_periods,0));assert.equal(cells[5],number(max,0));assert.equal(cells[6],number(max!=null?max-r.n_periods:null,0));assert.equal(cells[7],number(r.n_periods,0));
 }}
 for(const variant of input.variants){element('sample').value=variant;change('sample');element('unit').value='__all__';change('unit');readout(view,variant,'__all__');sampleChecks++;
  const units=[...new Set(view.rows.filter(r=>r.sample_variant===variant).map(r=>r.sector_id))];
  for(const unit of units){element('unit').value=unit;change('unit');readout(view,variant,unit);unitChecks++;}
 }
 viewChecks++;
}
for(const tab of tabs){tab.click();for(const id of ['regressions','yoy-guide','papers','methods'])assert.equal(element(id).hidden,id!==tab.dataset.tab);}
assert.equal(historyCalls,0);
process.stdout.write(JSON.stringify({status:'passed',views:viewChecks,three_sample_selections:sampleChecks,unit_selections:unitChecks,
 source_rows_rendered:rowChecks,test_rows_rendered:testChecks,fragility_cautions:fragilityChecks,selected_csv_exports:downloadChecks,
 fully_unavailable_views:unavailableViews,default_view:initial.id,history_replaceState_calls:historyCalls,
 async_compressed_bootstrap:input.payload_attributes['data-encoding']==='gzip-base64',
 scope:'Actual embedded JavaScript in Node VM DOM adapter; no visual-browser or cross-browser claim'}));
"""


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, usecols=lambda column: column in SOURCE_FIELDS, dtype={"sector_id": str}, float_precision="round_trip", low_memory=False)


def normalize(frame, branch):
    frame = frame.copy()
    if "sector_id" not in frame: frame["sector_id"] = "pooled"
    if branch in SUBANNUAL_BRANCHES:
        for old, new in {"specification": "model_spec", "structure": "model"}.items():
            if old in frame and new in frame: assert frame[old].equals(frame[new]), (branch, "Conflicting canonical aliases", old, new)
        frame = frame.rename(columns={old: new for old, new in {"specification": "model_spec", "structure": "model"}.items() if new not in frame})
        frame["history"] = "full_controls_max_AR_history" if branch == "full_controls" else "matched_maximum_AR_history"
    if "null" in frame: frame = frame.rename(columns={"null": "test"})
    if branch == "annual" and "parent_periods" in frame:
        frame["maximum_history_n_periods"] = frame.groupby(["context", "history", "model_spec", "model", "sector_id"], dropna=False).n_periods.transform("max")
        frame["raw_history_loss_periods"] = frame.parent_periods - frame.maximum_history_n_periods
        frame["target_alignment_loss_periods"] = 0
    if "target_alignment_loss_periods" not in frame and "matched_history_loss_periods" in frame: frame["target_alignment_loss_periods"] = frame.matched_history_loss_periods
    if "raw_history_loss_periods" not in frame and "raw_model_history_eligible_n_periods" in frame and "parent_n_periods" in frame:
        frame["raw_history_loss_periods"] = frame.parent_n_periods - frame.raw_model_history_eligible_n_periods
    return frame


def equal(actual, expected, label):
    if pd.isna(expected): assert actual is None or pd.isna(actual), label
    elif isinstance(expected, (float, int, np.number)) and not isinstance(expected, (bool, np.bool_)):
        assert np.isclose(actual, expected, atol=1e-12, rtol=1e-12), (label, actual, expected)
    else: assert actual == expected, (label, actual, expected)


def key_rows(frame, fields):
    result = {}
    for row in frame.to_dict("records"):
        key = tuple(row[field] for field in fields)
        assert key not in result, key
        result[key] = row
    return result


def term_label(term):
    if term.startswith("temp_Q"):
        quarter, lag = term.removeprefix("temp_").split("_l")
        return quarter + " mean temperature · " + ("current year" if lag == "0" else "prior year")
    if term.endswith("_current_plus_prior"): return term.split("_")[0] + " · current + prior coefficient sum"
    if term.startswith("temp_l"): return "Temperature · lag " + term.removeprefix("temp_l")
    if term.startswith("T_lag_"): return "Temperature · lag " + term.removeprefix("T_lag_")
    return {"sum_temperature_lags": "Sum of direct temperature coefficients", "temperature_lag_sum": "Sum of temperature coefficients",
        "temp_endpoint_difference": "Endpoint temperature difference coefficient", "temp_growth_sum": "Accumulated temperature SUM coefficient",
        "temperature_linear_centered": "Temperature · linear, centered", "temperature_quadratic": "Temperature · squared, centered",
        "marginal_effect_p5": "Marginal effect at observed T p5", "marginal_effect_p50": "Marginal effect at observed T median",
        "marginal_effect_p95": "Marginal effect at observed T p95", "sum_all_temperature_features": "Sum of all direct seasonal coefficients",
        "sum_current_Q1_Q4": "Sum of current Q1–Q4 coefficients", "sum_prior_Q1_Q4": "Sum of prior Q1–Q4 coefficients"}.get(term, term)


def native_stats(fragment):
    return {row.xpath('./th|./td')[0].text_content().strip(): row.xpath('./th|./td')[1:]
            for row in fragment.xpath('.//tbody/tr') if len(row.xpath('./th|./td')) > 1}


def _verify_literature_report():
    path = ROOT / "outputs/literature_sensitivity.html"
    initial_html_sha256 = sha(path)
    initial_producer_sha256 = sha(ROOT / "scripts/literature_report.py")
    initial_template_sha256 = sha(ROOT / "templates/literature_sensitivity.html")
    doc = html.fromstring(path.read_text(), parser=html.HTMLParser(huge_tree=True))
    payload_node = doc.get_element_by_id("literature-data")
    payload_text = payload_node.text_content().strip()
    payload_encoding = payload_node.get("data-encoding", "json")
    decoded_payload = gzip.decompress(base64.b64decode(payload_text)) if "gzip" in payload_encoding else payload_text.encode()
    data = json.loads(decoded_payload)
    inventory = json.loads((OUT / "report_inventory.json").read_text())
    views = data["views"]
    assert len(views) == inventory["views"] and len({view["id"] for view in views}) == len(views)
    assert {view["branch"] for view in views} == {"subannual", "full_controls", "annual", "annual_seasonal"}
    ids = doc.xpath('//@id');assert len(ids) == len(set(ids))
    assert not doc.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[@src]')
    assert data["sources"] == inventory["sources"]
    for source, digest in data["sources"].items(): assert sha(ROOT / source) == digest, source
    for branch, receipt in data["verification"].items():
        assert sha(ROOT / receipt["source"]) == receipt["sha256"]
        assert json.loads((ROOT / receipt["source"]).read_text())["status"] == "passed"
    cache = {}
    expected_dimensions = set()
    for branch in ["subannual", "full_controls", "annual", "annual_seasonal"]:
        folder = OUT if branch == "subannual" else OUT / branch
        specs = normalize(read(folder / "specifications.csv"), branch)
        estimates = normalize(read(folder / ("estimates.csv" if branch == "annual" else "temperature_effects.csv")), branch)
        tests = normalize(read(folder / "tests.csv"), branch)
        def enriched(frame):
            if branch != "full_controls": return frame
            assert specs.spec_id.is_unique
            missing = [column for column in specs if column not in frame]
            return frame.merge(specs[["spec_id"] + missing], on="spec_id", how="left", validate="many_to_one")
        estimates, tests = enriched(estimates), enriched(tests)
        if branch in SUBANNUAL_BRANCHES:
            estimates = pd.concat([estimates, enriched(normalize(read(folder / "channel_estimates.csv"), branch))], ignore_index=True)
            tests = pd.concat([tests, enriched(normalize(read(folder / "two_channel_span_tests.csv"), branch))], ignore_index=True)
        if branch == "annual":
            missing = specs.loc[specs.outcome.isna()].copy();specs = specs.loc[specs.outcome.notna()].copy()
            for outcome in ["y_nsa", "y_sa"]:
                added = missing.copy();added["outcome"] = outcome;specs = pd.concat([specs, added], ignore_index=True)
        expected_dimensions.update((branch, *key) for key in specs.groupby(DIMENSIONS, sort=False).groups)
        cache[branch] = {"estimates": {key: frame for key, frame in estimates.groupby(DIMENSIONS, sort=False)},
                         "tests": {key: frame for key, frame in tests.groupby(DIMENSIONS, sort=False)},
                         "specs": {key: frame for key, frame in specs.groupby(DIMENSIONS, sort=False)}}
    assert expected_dimensions == {(view["branch"], *(view[key] for key in DIMENSIONS)) for view in views}
    all_native_ids = set();native_tables = native_cells = source_rows = test_rows = request_rows = stat_cells = unavailable_views = 0
    for view in views:
        dimensions = tuple(view[key] for key in DIMENSIONS)
        if view["model"] == "separate":
            assert (view["branch"] == "subannual" and view["model_spec"] == "distributed") or (view["branch"] == "annual" and view["history"] == "model_own_weather_history" and view["model_spec"] in {"DJO5", "BHM_unit_quadratic_trends"}), "Unrequested own-series family exposed"
        frames = cache[view["branch"]]
        estimates = frames["estimates"].get(dimensions, pd.DataFrame())
        if len(estimates): estimates = estimates.loc[estimates.covariance.eq(view["covariance"]) & estimates.hac_lags.eq(view["hac_lags"])]
        stored = key_rows(pd.DataFrame(view["rows"]), ["sample_variant", "sector_id", "term"]) if view["rows"] else {}
        assert len(stored) == len(estimates), view["id"]
        for row in estimates.to_dict("records"):
            actual = stored[(row["sample_variant"], row["sector_id"], row["term"])]
            for field, value in actual.items():
                if field == "term_label": assert value == term_label(row["term"])
                else: equal(value, row[field], (view["id"], row["term"], field))
            source_rows += 1
        requested = frames["specs"][dimensions]
        request_fields = ["sample_variant", "sector_id"]
        request_map = key_rows(requested, request_fields)
        assert len(request_map) == len(view["requests"])
        for row in view["requests"]:
            source = request_map[(row["sample_variant"], row["sector_id"])]
            for field, value in row.items(): equal(value, source[field], (view["id"], "request", field))
            if view["branch"] in SUBANNUAL_BRANCHES:
                coverage = source.get("control_total_loss_periods", 0) if view["branch"] == "full_controls" else 0
                assert source["parent_n_periods"] - source["raw_history_loss_periods"] - coverage - source["target_alignment_loss_periods"] == source["maximum_history_n_periods"]
                if view["branch"] == "full_controls":
                    assert coverage == source["control_raw_missing_loss_periods"] + source["control_coverage_loss_periods"]
                    controls = set(json.loads(source["controls"]));assert {"rh_pct", "wind_mps", "pm25_10ug"} <= controls
                assert source["maximum_history_n_periods"] >= source["n_periods"]
            request_rows += 1
        source_tests = frames["tests"].get(dimensions, pd.DataFrame())
        if len(source_tests): source_tests = source_tests.loc[source_tests.covariance.eq(view["covariance"]) & source_tests.hac_lags.eq(view["hac_lags"])]
        test_map = key_rows(source_tests, ["sample_variant", "sector_id", "test"]) if len(source_tests) else {}
        assert len(test_map) == len(view["tests"])
        for row in view["tests"]:
            source = test_map[(row["sample_variant"], row["sector_id"], row["test"])]
            for field, value in row.items(): equal(value, source[field], (view["id"], "test", field))
            if row["status"] != "available": assert row["p_value"] is None
            test_rows += 1
        if view["branch"] in SUBANNUAL_BRANCHES:
            assert view["frequency"] in {"monthly", "quarterly"}
            if view["outcome"] == "y_short": assert view["outcome_label"].startswith("MoM" if view["frequency"] == "monthly" else "QoQ")
        else: assert view["frequency"] == "annual" and "Annual log growth" in view["outcome_label"]
        if view["model_spec"].startswith("seasonal"):
            assert ("prior" in view["controls_label"]) == ("prior" in view["model_spec"])
        if view["branch"] == "full_controls": assert "PM2.5" in view["controls_label"] and "wind" in view["controls_label"].lower()
        else: assert "PM2.5" not in view["controls_label"] and "wind" not in view["controls_label"].lower()
        selected = estimates.loc[estimates.coefficient.notna()].copy() if len(estimates) else pd.DataFrame()
        if len(selected) and view["model"] != "common":
            terms = ["temp_endpoint_difference", "temp_growth_sum"] if view["model_spec"] == "two_channel" else ["marginal_effect_p50"] if view["model_spec"].startswith("BHM") else ["sum_temperature_lags", "temperature_lag_sum", "sum_all_temperature_features"]
            selected = selected.loc[selected.term.isin(terms)]
        if not len(selected):
            assert not view["table"] and any(row["status"] != "estimated" for row in view["requests"])
            unavailable_views += 1;continue
        assert view["table"]
        fragment = html.fragment_fromstring(view["table"], create_parent="div", parser=html.HTMLParser(huge_tree=True))
        assert len(fragment.xpath('.//table[contains(concat(" ",normalize-space(@class)," ")," gt_table ")]')) == 1
        fragment_ids = fragment.xpath('.//@id');assert len(fragment_ids) == len(set(fragment_ids)) and not all_native_ids.intersection(fragment_ids)
        all_native_ids.update(fragment_ids)
        for headers in fragment.xpath('.//@headers'): assert set(headers.split()) <= set(fragment_ids)
        variants = [variant for variant in VARIANTS if selected.sample_variant.eq(variant).any()]
        profile = view["model"] != "common"
        selected["row_key"] = selected.sector_id + "|" + selected.term if profile else selected.term
        names = list(selected.drop_duplicates("row_key").row_key)
        coefficient_rows = [];inside = False
        for row in fragment.xpath('.//tbody/tr'):
            if row.xpath('./th[contains(@class,"gt_group_heading")]'): inside = row.text_content().strip() == "coef"
            elif inside: coefficient_rows.append(row)
        assert len(coefficient_rows) == len(names)
        index = key_rows(selected, ["sample_variant", "row_key"])
        for key, row in zip(names, coefficient_rows, strict=True):
            first = selected.loc[selected.row_key.eq(key)].iloc[0]
            expected_label = str(first.sector) + " · " + term_label(first.term) if profile and view["model_spec"] == "two_channel" else str(first.sector) if profile else term_label(first.term)
            assert row.xpath('./th')[0].text_content().strip() == expected_label
            cells = row.xpath('./td');assert len(cells) == len(variants)
            for variant, cell in zip(variants, cells, strict=True):
                source = index[(variant, key)]
                numbers = re.findall(r'[-+]?\d[\d,]*\.\d+', cell.text_content())
                assert [number.replace(',', '') for number in numbers] == [f'{source["coefficient"]:.3f}', f'{source["std_error"]:.3f}']
                assert cell.text_content().count('*') == len(significance_stars(source["p_value"]))
                native_cells += 1
        table_rows = native_stats(fragment)
        for col, variant in enumerate(variants):
            row = selected.loc[selected.sample_variant.eq(variant)].iloc[0]
            for label, value in {"Observations per equation" if view["model"] == "separate" else "Observations": row.n_obs,
                "Distinct weather dates": row.n_periods, "Parameters per equation": row.n_parameters,
                "Residual time df": row.time_residual_df, "Inference df": row.inference_df,
                "Missing calendar dates": row.calendar_gaps}.items():
                assert float(table_rows[label][col].text_content().strip().replace(',', '')) == value, (view["id"], label)
                stat_cells += 1
            if view["branch"] in SUBANNUAL_BRANCHES:
                assert "Common year FE" in table_rows
                assert ("Sector × calendar-season FE" in table_rows) == (row.n_sectors > 1)
                assert ("Calendar-season FE" in table_rows) == (row.n_sectors == 1)
            else:
                assert "Common year FE" not in table_rows and "Calendar-season FE" not in table_rows
                assert table_rows["Time trends"][col].text_content().strip() == ("Sector linear + quadratic" if view["model_spec"] == "BHM_unit_quadratic_trends" else "Common linear")
            if view["model"] == "common":
                for label, value in [("Mean Y", row.mean_y), ("SD Y", row.sd_y)]: assert table_rows[label][col].text_content().strip().replace(',', '') == f'{value:.3f}'
            if view["model"] == "separate": assert not any(label.startswith(("R2", "R²")) for label in table_rows)
            else: assert table_rows.get("R² including FE", table_rows.get("R2 including FE"))[col].text_content().strip() == f'{row.r_squared:.3f}'
            if view["model_spec"].startswith("BHM"):
                assert table_rows["Temperature basis center (°C)"][col].text_content().strip() == f'{row.mean_temperature:.3f}'
                if profile: assert table_rows["Derivative evaluation T (°C)"][col].text_content().strip() == f'{row.evaluation_temperature:.3f}'
        native_tables += 1
    script = doc.xpath('//script[not(@id="literature-data") and not(@src)]')[0].text_content()
    assert "history.replaceState" not in script
    methods = doc.get_element_by_id("methods").text_content()
    assert "Separate-sector equations are requested for the long-history distributed model only" in methods
    assert "Seasonal and full wind/PM literature companions request common/H1 equations" in methods
    assert "unrequested" in methods and "unavailable request remains visible" in methods
    assert "not independent confirmation" in doc.text_content()
    assert "B(L)=γ[1−Φ(L)](1−L^h)" in doc.get_element_by_id("yoy-guide").text_content()
    for qualifier in ["Multi-restriction Wald inference is not calibrated for headline claims", "selected-design diagnostic", "Full covariance rank establishes computability only", "do not use them as headline evidence"]:
        assert qualifier in doc.text_content(), qualifier
    input_payload = {"data": data, "script": script, "payload_text": payload_text, "payload_attributes": dict(payload_node.attrib),
        "ids": ids, "select_ids": [node.get('id') for node in doc.xpath('//select')], "variants": VARIANTS,
        "heads": ["All eligible dates", "Crisis dates excluded", "Strict growth window clean"]}
    with tempfile.TemporaryDirectory(prefix="heat-literature-presentation-") as temporary:
        payload = Path(temporary) / "payload.json";payload.write_text(json.dumps(input_payload, ensure_ascii=False, allow_nan=False))
        runtime_code = "(async()=>{\n" + NODE_HARNESS + "\n})().catch(error=>{console.error(error.stack||error);process.exitCode=1;});"
        run = subprocess.run(["node", "--max-old-space-size=8192", "-e", runtime_code, str(payload)], cwd=ROOT, text=True, capture_output=True, check=False)
        assert run.returncode == 0, "Literature selector runtime failed: " + run.stderr[-7000:]
        runtime = json.loads(run.stdout)
    assert sha(path) == initial_html_sha256 and sha(ROOT / "scripts/literature_report.py") == initial_producer_sha256 and sha(ROOT / "templates/literature_sensitivity.html") == initial_template_sha256, "Report/source changed during presentation audit"
    report = {"status": "passed", "scope": "Source-backed native tables and actual Node selector/readout/export runtime; numerical branch audits remain separate; no visual-browser claim",
        "views": len(views), "native_tables": native_tables, "fully_unavailable_views": unavailable_views,
        "native_coefficient_cells": native_cells, "native_stat_cells": stat_cells, "saved_estimate_rows": source_rows,
        "saved_request_rows": request_rows, "saved_test_rows": test_rows, "selector_runtime": runtime,
        "payload_encoding": payload_encoding, "decompressed_payload_sha256": hashlib.sha256(decoded_payload).hexdigest(),
        "html_sha256": initial_html_sha256, "producer_sha256": initial_producer_sha256,
        "template_sha256": initial_template_sha256, "source_hashes": data["sources"], "numerical_receipts": data["verification"]}
    persist(OUT / "report_verification.json", report)
    print(f"Literature presentation passed: {len(views)} views, {native_tables} native tables, {native_cells} coefficient cells", flush=True)
    return report


def verify_literature_report():
    with presentation_run(OUT / "report_verification.json", "New source-backed presentation verification"):
        return _verify_literature_report()

if __name__ == "__main__": verify_literature_report()
