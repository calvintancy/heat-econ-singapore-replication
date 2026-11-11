"""Source-backed and actual-JavaScript audit of the qualified CR1 display."""
from __future__ import annotations
import base64
import gzip
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from lxml import html
from significance import p_format, significance_stars
from presentation_verification_state import persist, presentation_run

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/clustered_sector_sensitivity"
OUT = ROOT / "outputs/clustered_sector_sensitivity_qualified"
REPORT = ROOT / "outputs/clustered_sector_sensitivity_qualified.html"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


NODE = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert'),crypto=require('crypto');
const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),nodes=new Map(),blobs=new Map(),downloads=[];let next=0;
function element(id,select=false){if(nodes.has(id))return nodes.get(id);let value='',markup='';const e={id,textContent:'',listeners:{},dataset:{},options:[],addEventListener(k,fn){this.listeners[k]=fn;},
 set value(v){value=String(v);},get value(){return value;},set innerHTML(s){markup=String(s);if(select){this.options=[...markup.matchAll(/<option\b[^>]*value="([^"]+)"[^>]*>/g)].map(m=>({value:m[1]}));value=this.options[0]?.value??'';}},get innerHTML(){return markup;}};nodes.set(id,e);return e;}
for(const id of input.ids)element(id,input.selects.includes(id));element('qcs-data').textContent=input.packed;
const document={getElementById(id){assert(nodes.has(id),id);return nodes.get(id);},createElement(tag){assert.equal(tag,'a');return {click(){downloads.push(this);}};}};
const sandbox={document,Blob,Response,DecompressionStream,atob,Uint8Array,setTimeout(fn){fn();},URL:{createObjectURL(blob){const id='blob:'+next++;blobs.set(id,blob);return id;},revokeObjectURL(){}},history:{replaceState(){throw Error('Unsafe history replacement');}}};sandbox.window=sandbox;
vm.createContext(sandbox);await vm.runInContext(input.script,sandbox);await sandbox.qualifiedSensitivityReady;
const data=sandbox.qualifiedSensitivityData;assert.equal(element('qcs-context').value,'va-detailed');assert.equal(element('qcs-window').value,'full');assert.equal(element('qcs-outcome').value,'y_short');assert.equal(element('qcs-band').value,'4');assert.equal(element('qcs-data').textContent,'');
const decode=s=>s.replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');const strip=s=>decode(s.replace(/<[^>]*>/g,'')).replace(/\s+/g,' ').trim();
const fmt=(v,d=3)=>v==null?'—':Number(v).toLocaleString('en',{minimumFractionDigits:d,maximumFractionDigits:d});let views=0,rows=0;
for(const spec of data.specs)for(const band of spec.frequency==='monthly'?[12,24]:[4,8]){
 for(const[id,value]of[['context',spec.context],['window',spec.window],['outcome',spec.outcome],['band',band]])element('qcs-'+id).value=value;
 sandbox.showQualifiedSensitivity();const key=spec.sample_id+'|'+spec.outcome;
 assert.equal(element('qcs-native').innerHTML,data.tables[key+'|'+band]);assert.equal(element('qcs-shared').textContent,fmt(data.shared[key].algebraic_shared_se,6));
 assert(element('qcs-ranks').textContent.includes('rank 1 / '+spec.n_sectors));assert(element('qcs-ranks').textContent.includes('rank 0 / '+(spec.n_sectors-1)));
 const source=data.rows.filter(r=>r.sample_id===spec.sample_id&&r.outcome===spec.outcome&&r.hac_lags===band);
 const actual=[...element('qcs-readout').innerHTML.matchAll(/<tr>([\s\S]*?)<\/tr>/g)].slice(1).map(m=>[...m[1].matchAll(/<td>([\s\S]*?)<\/td>/g)].map(x=>strip(x[1])));assert.equal(actual.length,source.length);
 for(let i=0;i<source.length;i++){const r=source[i],cells=actual[i];assert.equal(cells.length,10);assert.equal(cells[0],r.sector);assert.equal(cells[1],fmt(r.temp_main));assert.equal(cells[2],r.reference_delta_normalization?'0 · normalization':fmt(r.interaction_delta));assert.equal(cells[3],fmt(r.coefficient)+r.stars);assert.equal(cells[4],fmt(r.std_error));assert.equal(cells[5],'['+fmt(r.ci_lower)+', '+fmt(r.ci_upper)+']');assert.equal(cells[6],r.p_display);assert.equal(cells[7],r.p_value==null?'Unavailable':r.p_value<.05?'Yes':'No');assert.equal(cells[8],fmt(r.mean_y));assert.equal(cells[9],fmt(r.sd_y));rows++;}
 const joint=data.joints.find(r=>r.sample_id===spec.sample_id&&r.outcome===spec.outcome&&r.hac_lags===band);assert(element('qcs-joint').textContent.includes(joint.p_display));if(joint.status==='available')assert(element('qcs-joint').textContent.includes('Computable (calibration not established)'));
 assert(!/undefined|NaN|Infinity/.test(element('qcs-readout').innerHTML));views++;
}
for(const[name,digest]of Object.entries(input.downloads)){await element('qcs-downloads').onclick({target:{dataset:{raw:name}}});const a=downloads.at(-1);assert.equal(a.download,name);const buffer=Buffer.from(await blobs.get(a.href).arrayBuffer());assert.equal(crypto.createHash('sha256').update(buffer).digest('hex'),digest);}
process.stdout.write(JSON.stringify({status:'passed',async_gzip_bootstrap:true,selector_views:views,calendar_rows_rendered:rows,byte_exact_raw_downloads:downloads.length,scope:'Actual script in Node VM DOM adapter; no visual-browser claim'}));
"""


def verify_qualified_clustered_sector_sensitivity():
    with presentation_run(OUT / "verification.json", "Qualified CR1 presentation and raw-evidence preservation"):
        inventory = json.loads((OUT / "report_inventory.json").read_text())
        initial = sha(REPORT)
        assert initial == inventory["html_sha256"] and inventory["native_fits_run"] == 0
        assert sha(ROOT / "scripts/qualified_clustered_sector_sensitivity.py") == inventory["builder_sha256"]
        for path, digest in inventory["source_hashes"].items(): assert sha(ROOT / path) == digest
        doc = html.fromstring(REPORT.read_text(), parser=html.HTMLParser(huge_tree=True));ids = doc.xpath('//@id')
        assert doc.find("head") is not None and doc.find("body") is not None, "Master embedding requires explicit head/body"
        assert len(ids) == len(set(ids)) and not doc.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[@src]')
        node = doc.get_element_by_id("qcs-data");data = json.loads(gzip.decompress(base64.b64decode(node.text)))
        assert len(data["specs"]) == 90 and len(data["tables"]) == 180 and len(data["shared"]) == 90
        assert all(row["covariance"] == "Driscoll-Kraay" for row in data["rows"])
        assert all(not {"p_value", "ci_lower", "ci_upper", "stars"}.intersection(row) for row in data["shared"].values())
        raw = pd.read_csv(RAW / "sector_slopes.csv", dtype={"sector_id": str, "reference_sector_id": str}, float_precision="round_trip")
        indexed = raw.set_index(["sample_id", "outcome", "covariance", "hac_lags", "sector_id"])
        for row in data["rows"]:
            source = indexed.loc[(row["sample_id"], row["outcome"], "Driscoll-Kraay", row["hac_lags"], row["sector_id"])]
            for field in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "mean_y", "sd_y", "temp_main", "interaction_delta"]:
                assert row[field] is None if pd.isna(source[field]) else math.isclose(row[field], source[field], abs_tol=1e-13, rel_tol=1e-13)
            assert row["p_display"] == p_format(source.p_value) and row["stars"] == significance_stars(source.p_value)
        native_cells = 0
        for spec in data["specs"]:
            key = spec["sample_id"] + "|" + spec["outcome"]
            reference = indexed.loc[(spec["sample_id"], spec["outcome"], "Sector cluster", 0, spec["reference_sector_id"])]
            assert data["shared"][key]["algebraic_shared_se"] == reference.std_error
            assert data["shared"][key]["absolute_rank"] == 1 and data["shared"][key]["difference_rank"] == 0
            horizon = 12 if spec["frequency"] == "monthly" else 4
            for band in [horizon, 2*horizon]:
                table = html.fromstring(data["tables"][key + "|" + str(band)])
                table_ids = table.xpath('.//@id');assert len(table_ids) == len(set(table_ids))
                for references in table.xpath('.//@headers'): assert set(references.split()) <= set(table_ids)
                cells = {row[0].text_content().strip(): row for row in table.xpath('.//tr') if len(row) == 2}
                selected = raw.loc[raw.sample_id.eq(spec["sample_id"]) & raw.outcome.eq(spec["outcome"]) & raw.covariance.eq("Driscoll-Kraay") & raw.hac_lags.eq(band)]
                for row in selected.itertuples():
                    text = cells[row.sector][1].text_content()
                    assert f"{row.coefficient:.3f}" + significance_stars(row.p_value) in text and f"({row.std_error:.3f})" in text
                    native_cells += 1
                for label, value in [("Observations", spec["n_obs"]), ("Sector units", spec["n_sectors"]), ("Weather periods", spec["n_periods"]),
                    ("Nominal calendar df", spec["n_periods"] - 1), ("Estimated parameters", spec["n_parameters"])]:
                    assert float(cells[label][1].text_content().strip().replace(',', '')) == value
                for label, value in [("Mean observed Y", spec["mean_y_all"]), ("SD observed Y", spec["sd_y_all"])]:
                    assert cells[label][1].text_content().strip() == f"{value:.3f}"
        for phrase in ["Individual CR1 p-values, confidence intervals and stars are suppressed", "shared-nuisance component only", "historical CR1 p/CI fields are unsupported", "nominal asymptotic diagnostics"]:
            assert phrase in doc.text_content(), phrase
        downloads = {}
        for name, value in data["downloads"].items():
            path = ROOT / "outputs/clustered_sector_sensitivity.html" if name == "original_raw_display.html" else RAW / name
            assert gzip.decompress(base64.b64decode(value)) == path.read_bytes(), name
            downloads[name] = sha(path)
        script = doc.xpath('//script[not(@id="qcs-data")]')[0].text_content()
        with tempfile.TemporaryDirectory(prefix="heat-qualified-cluster-") as temporary:
            payload = Path(temporary) / "input.json";payload.write_text(json.dumps({"script": script, "packed": node.text, "ids": ids,
                "selects": doc.xpath('//select/@id'), "downloads": downloads}))
            code = "(async()=>{\n" + NODE + "\n})().catch(e=>{console.error(e.stack);process.exitCode=1;});"
            run = subprocess.run(["node", "-e", code, str(payload)], cwd=ROOT, text=True, capture_output=True)
            assert run.returncode == 0, "Qualified display runtime failed: " + run.stderr[-6000:]
            runtime = json.loads(run.stdout)
        assert sha(REPORT) == initial
        for path, digest in inventory["source_hashes"].items(): assert sha(ROOT / path) == digest
        result = {"status": "passed", "html_sha256": initial, "native_fits_run": 0, "native_DK_tables": 180,
            "native_DK_coefficient_cells": native_cells, "source_backed_calendar_rows": len(data["rows"]), "shared_algebraic_CR1_numbers": 90,
            "individual_CR1_p_CI_stars_suppressed": True, "original_evidence_unchanged": inventory["source_hashes"], "selector_runtime": runtime,
            "builder_sha256": inventory["builder_sha256"], "verifier_sha256": sha(Path(__file__)),
            "scope": "Qualified presentation only: raw numbers remain reproducible algebra, not supported individual CR1 inference. Calendar-DK inference remains nominal; no calibration or causal certificate"}
        persist(OUT / "verification.json", result)
        print(f"Qualified sector-cluster display passed: 180 native DK views, {native_cells} cells; raw evidence unchanged", flush=True)
        return result


if __name__ == "__main__": verify_qualified_clustered_sector_sensitivity()
