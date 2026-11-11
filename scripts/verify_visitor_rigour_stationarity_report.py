"""Check the compact visitor-diagnostic payload and its real standalone UI."""
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

import pandas as pd
from lxml import html
from presentation_verification_state import persist, presentation_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/visitor_rigour/diagnostics"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


NODE = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert'),crypto=require('crypto');const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),nodes=new Map(),blobs=new Map(),links=[];let next=0;
function node(id){if(!nodes.has(id))nodes.set(id,{id,value:'',innerHTML:'',textContent:'',options:[],add(option){this.options.push(option);}});return nodes.get(id);}
const sandbox={document:{getElementById(id){assert(input.ids.includes(id),id);return node(id);},createElement(tag){assert.equal(tag,'a');return {click(){links.push(this);}};}},Blob,Response,DecompressionStream,Uint8Array,atob,Option:function(text,value){return {text,value};},setTimeout(fn){fn();},URL:{createObjectURL(blob){const key='blob:'+next++;blobs.set(key,blob);return key;},revokeObjectURL(){}},history:{replaceState(){throw Error('Unsafe history replacement');}}};sandbox.window=sandbox;vm.createContext(sandbox);await vm.runInContext(input.script,sandbox);await sandbox.visitorDiagnosticsReady;
const decode=s=>s.replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');const fmt=x=>typeof x==='number'&&!Number.isInteger(x)?x.toPrecision(4):String(x);
let filters=0,pages=0,cells=0;function check(expected,offset){const begin=Math.max(0,Math.min(offset,Math.floor(Math.max(expected.length-1,0)/100)*100));const actual=[...node('vd-body').innerHTML.matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map(m=>[...m[1].matchAll(/<td>([\s\S]*?)<\/td>/g)].map(x=>decode(x[1])));assert.deepEqual(actual,expected.slice(begin,begin+100).map(r=>input.columns.map(c=>fmt(r[c]))));assert(node('vd-status').textContent.startsWith(`${expected.length} diagnostics; showing ${expected.length?begin+1:0}–${Math.min(begin+100,expected.length)}.`));pages++;cells+=actual.length*input.columns.length;}
const states=['',...new Set(input.rows.map(r=>r.state))],families=['','monthly','annual','ppml','origin_year'];assert.deepEqual(node('vd-state').options.map(x=>x.value),states.slice(1).sort());
for(const state of states)for(const freq of ['','M','Y-DEC'])for(const family of families){node('vd-state').value=state;node('vd-frequency').value=freq;node('vd-family').value=family;node('vd-search').value='';node('vd-state').oninput();const expected=input.rows.filter(r=>(!state||r.state===state)&&(!freq||r.frequency===freq)&&(!family||JSON.parse(r.branches).includes(family)));check(expected,0);for(let offset=100;offset<expected.length;offset+=100){node('vd-next').onclick();check(expected,offset);}node('vd-next').onclick();check(expected,expected.length);node('vd-back').onclick();check(expected,Math.max(0,Math.floor(Math.max(expected.length-1,0)/100)*100-100));filters++;}
for(const search of ['y_mom_nsa','arrival_count','temp','NO_MATCHING_ID',input.rows[0].diagnostic_id]){node('vd-state').value='';node('vd-frequency').value='';node('vd-family').value='';node('vd-search').value=search;node('vd-search').oninput();check(input.rows.filter(r=>(r.variables+' '+r.diagnostic_id).toLowerCase().includes(search.toLowerCase())),0);filters++;}
for(const[name,digest]of Object.entries(input.downloads)){await node('vd-downloads').onclick({target:{dataset:{download:name}}});const a=links.at(-1);assert.equal(a.download,name);const bytes=Buffer.from(await blobs.get(a.href).arrayBuffer());assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),digest);}
process.stdout.write(JSON.stringify({status:'passed',async_gzip_bootstrap:true,filter_search_combinations:filters,pages_checked:pages,displayed_cells_checked:cells,byte_exact_downloads:links.length,scope:'Actual standalone JavaScript in Node VM DOM adapter; no visual-browser claim'}));
"""


def verify_visitor_rigour_stationarity_report():
    with presentation_run(OUT / "report_verification.json", "Source-bound visitor exact-window diagnostic presentation"):
        files = [OUT / name for name in ["index.html", "report_payload.json", "diagnostics.csv", "tests.csv", "model_inventory.csv", "source_hashes.json", "summary.json", "verification.json"]]
        before = {str(path.relative_to(ROOT)): sha(path) for path in files}
        proof = json.loads((OUT / "verification.json").read_text());summary = json.loads((OUT / "summary.json").read_text())
        assert proof["status"] == "passed" and summary["counts"]["logical_model_outcomes"] == 354
        for path, digest in json.loads((OUT / "source_hashes.json").read_text()).items(): assert sha(ROOT / path) == digest
        doc = html.fromstring((OUT / "index.html").read_text(), parser=html.HTMLParser(huge_tree=True));ids = doc.xpath('//@id')
        assert len(ids) == len(set(ids)) and not doc.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[@src]')
        script = doc.xpath('//script')[0].text_content();match = re.search(r'const visitorPacked=("[^"]+");', script)
        assert match
        data = json.loads(gzip.decompress(base64.b64decode(json.loads(match[1]))))
        assert data == json.loads((OUT / "report_payload.json").read_text())
        assert data["counts"] == summary["counts"] and data["states"] == summary["diagnostic_states"]
        source = pd.read_csv(OUT / "diagnostics.csv", float_precision="round_trip").fillna("")
        for stem in ["primary", "legacy"]:
            source[stem + "_kpss_bound"] = [str(symbol) + f"{float(value):.4g}" if value != "" else "" for symbol, value in zip(source[stem + "_kpss_relation"], source[stem + "_kpss_p"])]
        columns = data["display_columns"];expected = source[columns].to_dict("records")
        assert len(expected) == len(data["rows"]) == summary["counts"]["distinct_diagnostics"]
        for actual, original in zip(data["rows"], expected, strict=True):
            for field in columns:
                a, b = actual[field], original[field]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)): assert math.isclose(a, b, abs_tol=1e-15, rel_tol=1e-13)
                else: assert a == b, field
        for qualifier in ["Poisson validity", "not a precondition imposed on the count model", "Gaps are never compressed", "annual minimum:20", "not embedded", "LNY calendar controls"]:
            assert qualifier in doc.text_content(), qualifier
        downloads = {}
        for name, packed in data["downloads"].items():
            assert gzip.decompress(base64.b64decode(packed)) == (OUT / name).read_bytes()
            downloads[name] = sha(OUT / name)
        with tempfile.TemporaryDirectory(prefix="visitor-unit-roots-ui-") as temporary:
            path = Path(temporary) / "input.json";path.write_text(json.dumps({"script": script, "rows": data["rows"], "columns": columns, "downloads": downloads, "ids": ids}, allow_nan=False))
            run = subprocess.run(["node", "-e", "(async()=>{\n" + NODE + "\n})().catch(e=>{console.error(e.stack);process.exitCode=1;});", str(path)], text=True, capture_output=True)
            assert run.returncode == 0, run.stderr[-6000:]
            runtime = json.loads(run.stdout)
        assert all(sha(ROOT / path) == digest for path, digest in before.items())
        result = {"status": "passed", "logical_model_outcomes": 354, "source_backed_diagnostics": len(expected), "native_fits_run": 0,
            "html_sha256": before["outputs/visitor_rigour/diagnostics/index.html"], "payload_sha256": before["outputs/visitor_rigour/diagnostics/report_payload.json"],
            "numerical_receipt_sha256": sha(OUT / "verification.json"), "source_sha256": before, "runtime": runtime,
            "scope": "Presentation/data-binding only; no stationarity, causal, Poisson-validity or inference-calibration certificate"}
        persist(OUT / "report_verification.json", result)
        print(f"Visitor diagnostics presentation passed:354requests/{len(expected)}arrays/four exact downloads", flush=True)
        return result


if __name__ == "__main__": verify_visitor_rigour_stationarity_report()
