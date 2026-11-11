"""Audit complete-VA overview, extension evidence and offline master embedding.

Legacy collection/reference verifiers remain separate and unchanged. This adds
source-backed checks for the 27 complete-coverage rows and the new chapters.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html

from significance import p_format, significance_class, significance_label
from presentation_verification_state import persist, presentation_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EVIDENCE = ROOT / "replication/extensions/review_evidence"
FAMILIES = {"outputs/crisis_sensitivity": 2655, "outputs/complete_va_controls": 750,
    "outputs/crisis_sensitivity/annual": 1656, "outputs/literature_sensitivity/annual": 2428,
    "outputs/literature_sensitivity": 2361, "outputs/literature_sensitivity/annual_seasonal": 684,
    "outputs/literature_sensitivity/full_controls": 574}
ORIGINAL_LEDGERS = ["outputs/pyfixest/fit_log.csv", "outputs/annual/pyfixest/fit_log.csv",
    "outputs/covid_comparison/common/pyfixest/fit_log.csv", "outputs/covid_comparison/sector/pyfixest/fit_log.csv",
    "outputs/clustered_sector_sensitivity/pyfixest/fit_log.csv", "outputs/visitor_arrivals/pyfixest/fit_log.csv",
    "outputs/visitor_arrivals/crisis/pyfixest/fit_log.csv"]

NODE_HARNESS = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),nodes=new Map(),docListeners={};
function element(id){if(!nodes.has(id))nodes.set(id,{id,innerHTML:'',textContent:'',hidden:false,dataset:{},listeners:{},attributes:{},offsetTop:0,tabIndex:-1,
 addEventListener(k,fn){(this.listeners[k]??=[]).push(fn);},setAttribute(k,v){this.attributes[k]=String(v);},scrollIntoView(){},
 click(){for(const fn of this.listeners.click??[])fn();},contentWindow:{messages:[],postMessage(value){this.messages.push(value);}}});return nodes.get(id);}
element('collection-data').textContent=JSON.stringify(input.data);
const chapters=['overview',...Object.keys(input.data.reports),'methods'].map(key=>element('panel-'+key));
const tabs=['overview',...Object.keys(input.data.reports),'methods'].map(key=>{let e=element('tab-'+key);e.dataset.tab=key;return e;});
const windows=['full','exclude','long'].map(window=>{let e=element('window-'+window);e.dataset.window=window;return e;});
const growths=['short','yoy_nsa','yoy'].map(growth=>{let e=element('growth-'+growth);e.dataset.growth=growth;return e;});
for(const key of Object.keys(input.data.reports)){element('frame-'+key).dataset.report=key;}
let hash='';const location={};Object.defineProperty(location,'hash',{get(){return hash;},set(value){hash=String(value).startsWith('#')?value:'#'+value;}});
const document={getElementById(id){if(id.startsWith('panel-')&&!chapters.some(p=>p.id===id))return null;if(id.startsWith('frame-')&&!input.data.reports[id.slice(6)])return null;return element(id);},
 querySelectorAll(selector){if(selector==='.chapter')return chapters;if(selector==='[data-tab]')return tabs;if(selector==='[data-window]')return windows;if(selector==='[data-growth]')return growths;if(selector==='[data-window],[data-growth]')return [...windows,...growths];throw Error(selector);},
 querySelector(selector){if(selector==='.navigation')return element('navigation');const match=selector.match(/^\[data-tab="([^"]+)"\]$/);if(match)return tabs.find(b=>b.dataset.tab===match[1]);throw Error(selector);},
 addEventListener(k,fn){(docListeners[k]??=[]).push(fn);}};
const sandbox={document,location,window:{scrollTo(){}},history:{replaceState(){throw Error('Unsafe history.replaceState');}}};vm.createContext(sandbox);
vm.runInContext(input.script,sandbox,{timeout:10000});
const decode=s=>String(s).replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');
const text=s=>decode(s.replace(/<[^>]*>/g,' ')).replace(/\s+/g,' ').trim();
let combinations=0,rowChecks=0,navigationChecks=0;
for(const growth of ['short','yoy_nsa','yoy'])for(const window of ['full','exclude','long']){
 growths.find(b=>b.dataset.growth===growth).click();windows.find(b=>b.dataset.window===window).click();
 const expected=input.data.complete_overview.filter(r=>r.growth===growth&&r.window===window),markup=element('complete-overview-rows').innerHTML;
 const rows=[...markup.matchAll(/<tr>([\s\S]*?)<\/tr>/g)];assert.equal(rows.length,3);assert(!/undefined|NaN/.test(markup));
 for(let i=0;i<rows.length;i++){
  const r=expected[i],cells=[...rows[i][1].matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g)].map(m=>text(m[1]));assert.equal(cells.length,6);
  const number=x=>Number(x).toFixed(3).replace('-','−');
  assert(cells[0].includes(r.label)&&cells[0].includes(r.outcome_label));assert.equal(cells[1],number(r.coefficient)+' ('+number(r.std_error)+')');
  assert(cells[2].includes('p='+r.p_display)&&cells[2].includes(r.significance));assert.equal(cells[3],number(r.mean_y));assert.equal(cells[4],number(r.sd_y));
  assert(cells[5].includes(String(r.periods))&&cells[5].includes(r.start+'–'+r.end));
  const targets=[...rows[i][1].matchAll(/data-go="([^"]+)"/g)].map(m=>m[1]);assert.deepEqual(targets,['complete-crisis']);
  for(const target of targets){assert(input.data.reports[target]);const button={dataset:{go:target}};for(const listener of docListeners.click??[])listener({target:{closest(selector){assert.equal(selector,'[data-go]');return button;}}});
   assert.equal(element('panel-'+target).hidden,false);assert.equal(element('tab-'+target).attributes['aria-selected'],'true');assert.equal(element('frame-'+target).srcdoc,input.data.reports[target]);navigationChecks++;}
  rowChecks++;
 }
 combinations++;
}
for(const target of ['complete-crisis','literature-models','new-stationarity','referee-audit']){assert(input.data.reports[target]);vm.runInContext(`showTab(${JSON.stringify(target)});`,sandbox);assert.equal(element('frame-'+target).srcdoc,input.data.reports[target]);navigationChecks++;}
process.stdout.write(JSON.stringify({status:'passed',window_growth_combinations:combinations,complete_rows_rendered:rowChecks,navigation_targets_checked:navigationChecks,
 scope:'Actual master render/control/navigation JavaScript in Node VM DOM adapter; no visual-browser claim'}));
"""


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def offline(document, label):
    identifiers = document.xpath('//@id');assert len(identifiers) == len(set(identifiers)), (label, "Duplicate IDs")
    assert not document.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[not(starts-with(@src,"data:"))]'), (label, "External assets")
    for headers in document.xpath('//@headers'): assert set(headers.split()) <= set(identifiers)


STATIONARITY_NODE = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert'),crypto=require('crypto');
const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),nodes=new Map(),blobs=new Map(),anchors=[];
function element(id){if(!nodes.has(id))nodes.set(id,{id,innerHTML:'',textContent:'',value:'',options:[],dataset:{},add(option){this.options.push(option);}});return nodes.get(id);}
const document={getElementById:element,createElement(tag){assert.equal(tag,'a');const a={click(){anchors.push(this);}};return a;}};
let objectId=0;class TestBlob extends Blob{constructor(parts,options){super(parts,options);this.parts=parts;}}
const sandbox={document,Blob:TestBlob,Response,DecompressionStream,Uint8Array,atob,Option:function(text,value){return {text,value};},
 URL:{createObjectURL(blob){const id='blob:'+objectId++;blobs.set(id,blob);return id;},revokeObjectURL(){}},setTimeout(fn){fn();},
 history:{replaceState(){throw Error('Unsafe history.replaceState');}}};
vm.createContext(sandbox);
(async()=>{
 await vm.runInContext(input.script,sandbox,{timeout:10000});
 assert(!element('status').textContent.includes('could not load'));
 const states=['',...new Set(input.rows.map(r=>r.state))],freqs=['','M','Q','Y-DEC'];
 assert.deepEqual(element('state').options.map(x=>x.value),states.slice(1).sort());
 const decode=s=>s.replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');
 const fmt=v=>typeof v==='number'&&!Number.isInteger(v)?v.toPrecision(4):String(v);
 let selections=0,pages=0,cells=0;
 function check(expected,offset){
  const start=Math.max(0,Math.min(offset,Math.floor(Math.max(expected.length-1,0)/100)*100));
  const actual=[...element('body').innerHTML.matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map(m=>[...m[1].matchAll(/<td>([\s\S]*?)<\/td>/g)].map(x=>decode(x[1])));
  const want=expected.slice(start,start+100).map(r=>input.columns.map(c=>fmt(r[c])));assert.deepEqual(actual,want);
  const status=`${expected.length} diagnostics; showing ${expected.length?start+1:0}–${Math.min(start+100,expected.length)}.`;
  assert(element('status').textContent.startsWith(status));cells+=actual.length*input.columns.length;pages++;
 }
 for(const state of states)for(const freq of freqs){
  element('state').value=state;element('frequency').value=freq;element('search').value='';element('state').oninput();
  const expected=input.rows.filter(r=>(!state||r.state===state)&&(!freq||r.frequency===freq));check(expected,0);
  for(let offset=100;offset<expected.length;offset+=100){element('next').onclick();check(expected,offset);}
  element('next').onclick();check(expected,Math.max(0,expected.length-1));element('back').onclick();check(expected,Math.max(0,Math.floor(Math.max(expected.length-1,0)/100)*100-100));
  selections++;
 }
 for(const search of ['y_short','temp','NO_SUCH_DIAGNOSTIC',input.rows[0].diagnostic_id]){
  element('state').value='';element('frequency').value='';element('search').value=search;element('search').oninput();
  check(input.rows.filter(r=>(r.variables+' '+r.diagnostic_id).toLowerCase().includes(search.toLowerCase())),0);selections++;
 }
 for(const [name,expected] of Object.entries(input.downloads)){
  await element('downloads').onclick({target:{dataset:{download:name}}});const anchor=anchors.at(-1);assert.equal(anchor.download,name);
  const bytes=Buffer.from(await blobs.get(anchor.href).arrayBuffer());assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),expected);
 }
 process.stdout.write(JSON.stringify({status:'passed',async_gzip_bootstrap:true,filter_search_combinations:selections,pages_checked:pages,
  displayed_cells_checked:cells,byte_exact_downloads:Object.keys(input.downloads).length,scope:'Actual async report script in Node VM DOM adapter; no visual-browser claim'}));
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
"""


def verify_stationarity_appendix():
    folder = OUT / "extension_stationarity"
    receipt_path = folder / "report_verification.json"
    with presentation_run(receipt_path, "Source-bound extension stationarity appendix presentation"):
        page = OUT / "extension_stationarity.html"
        guarded = [page, ROOT / "scripts/extension_stationarity.py", ROOT / "scripts/extension_stationarity_report.py",
                   folder / "verification.json", folder / "summary.json", folder / "diagnostics.csv", folder / "tests.csv",
                   folder / "model_inventory.csv", folder / "source_hashes.json", Path(__file__)]
        before = {str(path.relative_to(ROOT)): sha(path) for path in guarded}
        numerical = json.loads((folder / "verification.json").read_text())
        assert numerical["status"] == "passed"
        assert numerical["producer_source_sha256"] == sha(ROOT / "scripts/extension_stationarity.py")
        text = page.read_text();document = html.fromstring(text, parser=html.HTMLParser(huge_tree=True));offline(document, "Stationarity appendix")
        script = document.xpath('//script')[0].text_content()
        packed = re.search(r'const blob=("[^"]+");const columns=(\[[^;]+\]);', script)
        assert packed, "Actual packed appendix bootstrap not found"
        payload = json.loads(gzip.decompress(base64.b64decode(json.loads(packed[1]))))
        columns = json.loads(packed[2])
        source = pd.read_csv(folder / "diagnostics.csv").fillna("")
        for stem in ["primary", "legacy"]:
            source[stem + "_kpss_bound"] = [str(relation) + f"{float(value):.4g}" if value != "" else ""
                for relation, value in zip(source[stem + "_kpss_relation"], source[stem + "_kpss_p"])]
        expected = json.loads(source[columns].to_json(orient="records", double_precision=15))
        assert len(payload["rows"]) == len(expected) == numerical["counts"]["distinct_diagnostics"]
        for index, (actual, original) in enumerate(zip(payload["rows"], expected)):
            for field in columns:
                a, b = actual[field], original[field]
                if isinstance(a, (float, int)) and isinstance(b, (float, int)):
                    assert math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-15), (index, field, a, b)
                else: assert a == b, (index, field)
        expected_downloads = {name: sha(folder / name) for name in ["diagnostics.csv", "tests.csv", "model_inventory.csv", "source_hashes.json"]}
        assert set(payload["downloads"]) == set(expected_downloads)
        for name, value in payload["downloads"].items():
            assert gzip.decompress(base64.b64decode(value)) == (folder / name).read_bytes(), name
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["counts"] == numerical["counts"] and summary["diagnostic_states"] == numerical["diagnostic_states"]
        assert source.state.value_counts().to_dict() == summary["diagnostic_states"]
        inventory = pd.read_csv(folder / "model_inventory.csv")
        assert set(inventory.branch) == {"core", "full", "annual", "literature", "annual_literature", "seasonal", "full_controls"}
        tables = document.xpath('//table')
        state_counts = {row.xpath('./td')[0].text_content(): int(row.xpath('./td')[1].text_content()) for row in tables[0].xpath('./tbody/tr')}
        family_counts = {row.xpath('./td')[0].text_content(): int(row.xpath('./td')[1].text_content()) for row in tables[1].xpath('./tbody/tr')}
        assert state_counts == summary["diagnostic_states"] and family_counts == inventory.groupby('branch', sort=False).size().to_dict()
        prose = document.text_content()
        for qualifier in ["do not certify stationarity", "zero-frequency diagnostics", "not seasonal-unit-root", "does not embed those two large audit ledgers", "Gapped calendars are never compressed"]:
            assert qualifier in prose, qualifier
        for key in ["logical_model_outcomes", "model_variable_links", "distinct_diagnostics", "completed_diagnostics", "unavailable_diagnostics"]:
            assert f"{summary['counts'][key]:,}" in document.xpath('//p[@class="stats"]')[0].text_content()
        with tempfile.TemporaryDirectory(prefix="heat-stationarity-presentation-") as temporary:
            path = Path(temporary) / "payload.json"
            path.write_text(json.dumps({"script": script, "columns": columns, "rows": payload["rows"], "downloads": expected_downloads}, allow_nan=False))
            run = subprocess.run(["node", "-e", STATIONARITY_NODE, str(path)], cwd=ROOT, text=True, capture_output=True)
            assert run.returncode == 0, "Stationarity runtime failed: " + run.stderr[-6000:]
            runtime = json.loads(run.stdout)
        assert all(sha(ROOT / relative) == value for relative, value in before.items()), "Appendix source changed during verification"
        report = {"status": "passed", "scope": "Source-backed appendix diagnostics, state/family counts, qualifiers and byte-exact embedded downloads; actual async bootstrap/filter/page/download JavaScript",
                  "html_sha256": sha(page), "numerical_receipt_sha256": sha(folder / "verification.json"), "diagnostic_rows": len(expected),
                  "native_fit_family": False, "source_sha256": before, "runtime": runtime}
        persist(receipt_path, report)
        print(f"Stationarity appendix presentation passed: {len(expected)} diagnostics, four exact downloads", flush=True)
        return report


def verify_selected_null_claim(audit_inventory):
    """Bind the displayed bounded calibration claim to current fixed X/row arrays.

    Whole historical NPZ/native-ledger hashes remain snapshot provenance: native
    beta roundoff and receipt times may legitimately change on reproduction.
    The calibration experiment conditions on X and its exact row calendar.
    """
    manifest_path = EVIDENCE / "new_methods_evidence_manifest.json"
    evidence = json.loads(manifest_path.read_text())
    assert evidence["status"] == "verified_canonical_evidence_copies" and evidence["native_fits_run"] == 0
    for name, digest in evidence["files_sha256"].items(): assert sha(EVIDENCE / name) == digest, name
    portable = evidence["portable_source"]
    assert sha(ROOT / portable["source"]) == portable["source_sha256"]
    archived_path = EVIDENCE / "new_methods_selected_fixed_design.json"
    live_path = OUT / "aea_audit_calibration/summary.json"
    archived, live = json.loads(archived_path.read_text()), json.loads(live_path.read_text())
    for relative, digest in live["frozen_sources_sha256"].items(): assert sha(ROOT / relative) == digest, relative
    expected_counts = {(4, 60): 2000, (8, 60): 2000, (4, 11): 712, (8, 11): 1091}
    for proof in [archived, live]:
        assert proof["status"] == "passed_selected_fixed_X_null_replay" and proof["native_fits_run"] == 0
        assert proof["design"] == "va12__y_yoy_nsa__all_eligible__H1__distributed"
        assert proof["n_periods"] == 174 and proof["time_residual_df"] == 116 and proof["seed"] == 20261119
        assert proof["repetitions"] == 2000 and proof["matches_independently_replayed_reference_counts"] is True
        assert len(proof["rejections"]) == 4
        for row in proof["rejections"]:
            assert row["draws"] == row["available"] == 2000
            assert row["rejections"] == expected_counts[(row["hac_lags"], row["restrictions"])]
            assert row["rejection_rate"] == row["rejections"] / row["draws"]
            assert row["test"] == ("All sector temperature-lag responses jointly zero" if row["restrictions"] == 60 else "All sector cumulative direct temperature responses equal")
    path = OUT / "literature_sensitivity/designs" / (archived["design"] + ".npz")
    with np.load(path, allow_pickle=False) as arrays:
        for name, binding in archived["array_bindings"].items():
            value = arrays[name]
            assert list(value.shape) == binding["shape"] and str(value.dtype) == binding["dtype"]
            assert hashlib.sha256(value.tobytes()).hexdigest() == binding["sha256"], name
            assert live["array_bindings"][name] == binding
    rejection_path = OUT / "aea_audit_calibration/rejections.csv"
    draws_path = OUT / "aea_audit_calibration/draw_statistics.csv"
    assert live["outputs_sha256"]["rejections.csv"] == sha(rejection_path)
    assert live["outputs_sha256"]["draw_statistics.csv"] == sha(draws_path)
    draws = pd.read_csv(draws_path)
    assert len(draws) == 8000
    assert draws.available.all() and np.array_equal(draws.reject_05, draws.p_value.lt(.05))
    for (bandwidth, count), group in draws.groupby(["hac_lags", "restrictions"]):
        assert len(group) == 2000 and group.draw.nunique() == 2000
        assert int(group.reject_05.sum()) == expected_counts[(bandwidth, count)]
    record = audit_inventory["new_methods_review_adjudication"]
    assert record["status"] == "current_design_known_null_calibration_failure_reproduced"
    assert record["note_sha256"] == sha(EVIDENCE / "new_methods_adjudication.md")
    assert record["proof_sha256"] in {sha(archived_path), sha(live_path)}
    assert record["comparison_sha256"] == sha(EVIDENCE / "new_methods_inference_comparison.csv")
    assert record["design"] == archived["design"] and record["draws"] == 2000 and record["seed"] == 20261119
    execution_path = OUT / "aea_audit/selected_null_current_execution.json"
    assert execution_path.read_bytes() == live_path.read_bytes()
    assert record["current_execution_sha256"] == sha(execution_path)
    return {"status": "passed_bounded_claim_binding", "archived_proof_sha256": sha(archived_path),
        "current_replay_sha256": sha(live_path), "canonical_evidence_manifest_sha256": sha(manifest_path),
        "fixed_array_fields": sorted(archived["array_bindings"]), "null_draw_rows": len(draws), "rejection_counts": {str(key): value for key, value in expected_counts.items()},
        "scope": "One exact current design under the declared Gaussian-null process; all-zero and equality nulls distinguished; no claim of every-table calibration or a new native fit family"}


def _verify_extension_collection():
    path = OUT / "heat_economy.html"
    doc = html.fromstring(path.read_text(), parser=html.HTMLParser(huge_tree=True))
    data = json.loads(doc.get_element_by_id("collection-data").text)
    rows = data["complete_overview"]
    assert len(rows) == 27 and len({(row["context"], row["growth"], row["window"]) for row in rows}) == 27
    assert {row["context"] for row in rows} == {"va12", "va12-matched", "va20"}
    assert {row["growth"] for row in rows} == {"short", "yoy_nsa", "yoy"}
    assert {row["window"] for row in rows} == {"full", "exclude", "long"}
    source_frames, sample_hashes = {}, {}
    for row in rows:
        source = ROOT / row["source_csv"]
        if source not in source_frames: source_frames[source] = pd.read_csv(source, float_precision="round_trip", low_memory=False)
        frame = source_frames[source]
        match = frame.loc[frame.spec_id.eq(row["spec_id"]) & frame.term.eq("temp_c") & frame.covariance.eq("Sector cluster") & frame.main_inference.eq(True)]
        assert len(match) == 1
        original = match.iloc[0]
        for field in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "mean_y", "sd_y"]:
            assert math.isclose(row[field], original[field], abs_tol=1e-12, rel_tol=1e-12), (row["context"], row["window"], field)
        assert row["covariance"] == "Sector cluster" and row["n_clusters"] == original.n_sectors == (20 if row["context"] == "va20" else 12)
        assert original.n_clusters == original.n_sectors and original.inference_df == original.n_sectors - 1
        assert bool(original.includes_dwellings) and row["includes_dwellings"] is True
        assert row["p_display"] == p_format(original.p_value) and row["significance"] == significance_label(original.p_value)
        assert row["significance_class"] == significance_class(original.p_value)
        assert row["target"] == "complete-crisis" and row["target"] in data["reports"]
        expected_outcome = {"short": "y_short", "yoy_nsa": "y_yoy_nsa", "yoy": "y_yoy"}[row["growth"]]
        assert row["outcome"] == original.outcome == expected_outcome
        assert row["sample_variant"] == original.sample_variant == ("event_dates_excluded" if row["window"] == "exclude" else "all_eligible")
        controls = set(json.loads(original.controls))
        assert controls == {"temp_c", "rain_100mm", "rh_pct"} | (set() if row["window"] == "long" else {"wind_mps", "pm25_10ug"})
        sample_path = ROOT / row["sample_csv"]
        assert row["sample_csv"] == original.sample_csv
        sample = pd.read_csv(sample_path, dtype={"sector_id": str}, float_precision="round_trip")
        values = sample[expected_outcome].to_numpy()
        mean = math.fsum(values) / len(values)
        sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
        assert math.isclose(mean, row["mean_y"], abs_tol=1e-12, rel_tol=1e-12)
        assert math.isclose(sd, row["sd_y"], abs_tol=1e-12, rel_tol=1e-12)
        assert row["n_obs"] == original.n_obs == len(sample)
        assert row["periods"] == original.n_periods == sample.quarter.nunique()
        assert row["start"] == original.start == sample.quarter.min() and row["end"] == original.end == sample.quarter.max()
        assert "1.3" in set(sample.sector_id) and sample.sector_id.nunique() == row["n_clusters"]
        if row["window"] != "long": assert row["periods"] == (19 if row["window"] == "exclude" else 29)
        sample_hashes[row["sample_csv"]] = sha(sample_path)
    manifest = json.loads((OUT / "collection/manifest.json").read_text())
    assert manifest["complete_overview"]["rows"] == 27
    chapters = {}
    for key in ["complete-crisis", "literature-models", "new-stationarity", "referee-audit"]:
        record = manifest[key];source = OUT / record["file"]
        assert sha(source) == record["source_sha256"]
        assert hashlib.sha256(data["reports"][key].encode()).hexdigest() == record["embedded_sha256"]
        standalone = html.fromstring(source.read_text(), parser=html.HTMLParser(huge_tree=True));offline(standalone, key)
        embedded = html.fromstring(data["reports"][key], parser=html.HTMLParser(huge_tree=True));offline(embedded, key + " embedded")
        for target in embedded.xpath('//@data-report-target'): assert target in data["reports"] or target in {"overview", "methods"}
        chapters[key] = record
    for name, field, source in [("crisis_sensitivity", "report_sha256", OUT / "crisis_sensitivity.html"),
                                ("literature_sensitivity", "html_sha256", OUT / "literature_sensitivity.html"),
                                ("extension_stationarity", "html_sha256", OUT / "extension_stationarity.html")]:
        receipt = json.loads((OUT / name / "report_verification.json").read_text())
        assert receipt["status"] == "passed" and receipt[field] == sha(source)
        if name == "extension_stationarity":
            assert receipt["numerical_receipt_sha256"] == sha(OUT / name / "verification.json")
            assert receipt["native_fit_family"] is False
            assert all(sha(ROOT / relative) == value for relative, value in receipt["source_sha256"].items())
    audit_inventory_path = OUT / "aea_audit/evidence_inventory.json"
    audit_inventory = json.loads(audit_inventory_path.read_text())
    selected_null_claim = verify_selected_null_claim(audit_inventory)
    extension_total = 0;native_receipts = {}
    assert set(FAMILIES) == set(audit_inventory["numerical_receipts"]), "Seven native families; stationarity is a separate appendix"
    for relative, recorded in audit_inventory["numerical_receipts"].items():
        folder = ROOT / relative
        receipt_path = folder / "verification.json";receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "passed"
        ledger = pd.read_csv(folder / "pyfixest/fit_log.csv", usecols=["fit_id", "design_sha256", "outcome_sha256", "native_class", "native_module"])
        native_manifest = json.loads((folder / "pyfixest/estimator_manifest.json").read_text())
        expected = FAMILIES.get(relative, native_manifest["native_fits_recorded"])
        assert len(ledger) == expected == native_manifest["native_fits_recorded"]
        assert ledger.fit_id.tolist() == list(range(1, expected + 1))
        assert ledger.native_class.eq("Feols").all() and ledger.native_module.str.startswith("pyfixest").all()
        assert recorded["status"] == "passed" and recorded["sha256"] == sha(receipt_path)
        assert recorded["native_fits"] == expected and recorded["native_ledger_sha256"] == sha(folder / "pyfixest/fit_log.csv")
        extension_total += expected;native_receipts[relative] = {"native_fits": expected, "receipt_sha256": sha(receipt_path)}
    originals = pd.concat([pd.read_csv(ROOT / name, usecols=["design_sha256", "outcome_sha256"]) for name in ORIGINAL_LEDGERS], ignore_index=True)
    ppml = json.loads((OUT / "visitor_arrivals/ppml/estimator_receipts.json").read_text())
    assert len(originals) == 4781 and len(originals.drop_duplicates()) == 3984
    assert len(ppml) == 12 and len({row["model_id"] for row in ppml}) == 12
    assert all("fepois" in row["estimator"] for row in ppml)
    assert audit_inventory["extension_native_fits"] == extension_total
    assert audit_inventory["original_native_OLS_fits"] == len(originals) and audit_inventory["original_native_PPML_fits"] == len(ppml)
    assert audit_inventory["cumulative_recorded_OLS_fits"] == len(originals) + extension_total
    review_receipts = {}
    for lane in ["design", "inference", "new-methods"]:
        receipt_path = EVIDENCE / f"fable52-{lane}-receipt.json";review_path = EVIDENCE / f"fable52-{lane}-review.md"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "complete" and receipt["exitCode"] == 0 and receipt["error"] is None
        assert receipt["modelVerified"] is True and receipt["modelEvidence"] == "provider-report"
        assert receipt["reportedModel"] == receipt["model"] == "claude-fable-5-2" and receipt["effort"] == "max"
        assert receipt["mode"] == "isolated-write" and receipt["writerSandbox"] is not None
        assert receipt["preflight"]["status"] == "passed"
        argv = receipt["argv"]
        assert argv[argv.index("--model") + 1] == "claude-fable-5-2" and argv[argv.index("--effort") + 1] == "max"
        assert receipt["completedAt"] >= receipt["startedAt"] and receipt["elapsedMs"] > 0 and review_path.stat().st_size > 1000
        record = next(row for row in audit_inventory["reviews"] if row["lane"] == lane)
        assert record["receipt_sha256"] == sha(receipt_path) and record["report_sha256"] == sha(review_path)
        assert record["reported_model"] == "claude-fable-5-2" and record["status"] == "complete"
        review_receipts[lane] = {"receipt_sha256": sha(receipt_path), "review_sha256": sha(review_path), "provider_model": receipt["reportedModel"]}
    acceptance = json.loads((EVIDENCE / "implementation_acceptance.json").read_text())
    assert acceptance["status"] == "passed" and audit_inventory["baseline_reverification"] == acceptance
    assert (EVIDENCE / "native_findings.json").is_file()
    audit_html = html.fromstring((OUT / "aea_audit.html").read_text(), parser=html.HTMLParser(huge_tree=True))
    offline_downloads = 0
    for link in audit_html.xpath('//a[@download]'):
        href = link.get('href');assert href.startswith('data:text/plain;base64,')
        content = base64.b64decode(href.split(',', 1)[1]);filename = link.get('download')
        candidates = [EVIDENCE / filename, OUT / "aea_audit" / filename]
        assert any(candidate.is_file() and candidate.read_bytes() == content for candidate in candidates), filename
        offline_downloads += 1
    script = doc.xpath('//script[not(@type="application/json")]')[0].text_content()
    prefixes = ('const DATA=', 'const number=', 'function renderOverview()', 'function renderPairedOverview()',
                'function ensureFrame(', 'function showTab(', 'function renderCompleteOverview()',
                "document.addEventListener('click',", "document.querySelectorAll('[data-window]').forEach", "document.querySelectorAll('[data-window],[data-growth]').forEach")
    selected_code = '\n'.join(line for line in script.splitlines() if line.startswith(prefixes))
    assert 'function renderCompleteOverview()' in selected_code and 'function showTab(' in selected_code
    node_payload = {"script": selected_code, "data": {"overview": data["overview"], "complete_overview": rows,
        "reports": {key: "<html><body>checked embedded chapter " + key + "</body></html>" for key in data["reports"]}}}
    with tempfile.TemporaryDirectory(prefix="heat-extension-master-") as temporary:
        payload = Path(temporary) / "payload.json";payload.write_text(json.dumps(node_payload, ensure_ascii=False, allow_nan=False))
        run = subprocess.run(["node", "-e", NODE_HARNESS, str(payload)], cwd=ROOT, text=True, capture_output=True, check=False)
        assert run.returncode == 0, "Complete overview runtime failed: " + run.stderr[-6000:]
        runtime = json.loads(run.stdout)
    report = {"status": "passed", "scope": "New complete-VA source/sample overview, offline embedded chapters, extension/native/reviewer evidence inventory and actual master JavaScript; legacy verifier scope remains separate",
        "complete_overview_rows": len(rows), "sample_hashes": sample_hashes, "master_sha256": sha(path),
        "new_embedded_reports": chapters, "native_extension_fits": extension_total, "original_OLS_fits": len(originals),
        "original_unique_design_outcome_pairs": len(originals.drop_duplicates()), "original_PPML_fits": len(ppml),
        "native_receipts": native_receipts, "exact_Fable52_review_receipts": review_receipts,
        "native_extension_families": len(native_receipts), "stationarity_appendix_native_family": False,
        "stationarity_numerical_receipt_sha256": sha(OUT / "extension_stationarity/verification.json"),
        "stationarity_presentation_receipt_sha256": sha(OUT / "extension_stationarity/report_verification.json"),
        "audit_evidence_inventory_sha256": sha(audit_inventory_path), "audit_offline_downloads": offline_downloads,
        "selected_fixed_design_claim": selected_null_claim,
        "complete_overview_runtime": runtime}
    persist(OUT / "collection/extension_verification.json", report)
    print(f"Extension collection passed: 27 complete-VA rows, {extension_total} native extension fits, three exact-model review receipts", flush=True)
    return report


def verify_extension_collection():
    with presentation_run(OUT / "collection/extension_verification.json", "New source-backed presentation verification"):
        return _verify_extension_collection()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stationarity-only", action="store_true", help="Check the standalone appendix before the final master exists")
    args = parser.parse_args()
    if args.stationarity_only: verify_stationarity_appendix()
    else: verify_extension_collection()
