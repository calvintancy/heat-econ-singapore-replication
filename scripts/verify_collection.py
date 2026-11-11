"""Check the portable report, embedded chapters and overview sample means."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from lxml import html
from inference_display import main_rows
from combined_report import REPORTS
from significance import significance_label,significance_class,p_format
from stationarity_context import model_assessment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def validate_document(document, label: str) -> dict:
    ids = document.xpath('//*[@id]/@id')
    assert len(ids) == len(set(ids)), (label, "Duplicate IDs")
    identifiers = set(ids)
    for node in document.xpath('//*[@headers]'):
        assert set(node.get("headers").split()) <= identifiers, (label, node.get("headers"))
    assert not document.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[not(starts-with(@src,"data:"))]'), label
    for node in document.xpath('//a[@href]'):
        url = urlsplit(node.get("href"))
        assert url.scheme in {"", "data", "http", "https"}, (label, node.get("href"))
        assert url.scheme or not url.path, (label, "Unembedded local link", node.get("href"))
    return {"ids": len(ids), "tables": len(document.xpath('//table')), "figures": len(document.xpath('//svg'))}


def independent_sample(row: dict) -> tuple[pd.DataFrame, str]:
    context, growth, window = row["context"], row["growth"], row["window"]
    outcome_name='y_short' if growth=='short' else 'y_yoy_nsa' if growth=='yoy_nsa' else 'y_yoy'
    if row["source_csv"].startswith("outputs/granular/"):
        return pd.read_csv(ROOT / Path(row["source_csv"]).parent / "samples" / f'{row["model"]}.csv'), outcome_name
    if context == "aggregate":
        group = "weather" if window == "long" else "full"
        path = OUT / "aggregate_va" / group / "samples" / f'{row["model"]}.csv'
        return pd.read_csv(path), outcome_name
    if growth.startswith('yoy'):
        path = OUT / "outcomes" / context / ("weather" if window == "long" else "full") / "common_sample.csv"
        sample = pd.read_csv(path)
        outcome = outcome_name
    elif window == "long":
        path = OUT / "baseline_sample.csv" if context == "va" else OUT / "iip" / row["frequency"] / "baseline_sample.csv"
        sample = pd.read_csv(path)
        outcome = "g_va_sa_qoq" if context == "va" else "g_iip_sa"
    else:
        sample = pd.read_csv(OUT / "extended_controls" / context / "baseline_sample.csv")
        outcome = "g_va_sa_qoq" if context == "va" else "g_iip_sa"
    if window == "exclude":
        sample = sample[~sample.year.between(2020, 2022)].copy()
    return sample, outcome


def verify_collection() -> dict:
    path = OUT / "heat_economy.html"
    document = html.fromstring(path.read_text(), parser=html.HTMLParser(huge_tree=True))
    report = {"outer": validate_document(document, "Collection"), "embedded": {}, "overview_rows_checked": 0}
    payload = json.loads(document.xpath('//script[@id="collection-data"]')[0].text)
    manifest = json.loads((OUT / "collection/manifest.json").read_text())
    assert len(payload["reports"]) == len(REPORTS)
    assert len(document.xpath('//button[@data-tab]')) == len(REPORTS)+2
    assert len(document.xpath('//iframe')) == len(REPORTS)
    assert len(document.xpath('//*[@role="tabpanel" and not(@hidden)]')) == 1
    for key, source in payload["reports"].items():
        chapter = html.fromstring(source, parser=html.HTMLParser(huge_tree=True))
        report["embedded"][key] = validate_document(chapter, key)
        assert hashlib.sha256(source.encode()).hexdigest() == manifest[key]["embedded_sha256"]
        assert hashlib.sha256((OUT / manifest[key]["file"]).read_bytes()).hexdigest() == manifest[key]["source_sha256"]
        assert chapter.xpath('//script[contains(text(),"heat-economy-collection")]'), key
        for target in chapter.xpath('//@data-report-target'):
            assert target in payload["reports"] or target in {"overview", "methods"}
    overview = payload["overview"]
    assert len(overview) == 9*3*3
    assert len({(r["context"], r["growth"], r["window"]) for r in overview}) == 9*3*3
    assert {'short','yoy','yoy_nsa'}=={r['growth'] for r in overview}
    for row in overview:
        stored = pd.read_csv(ROOT / row["source_csv"])
        lag = 12 if row["frequency"] == "monthly" else 4
        stored = main_rows(stored, row["frequency"])
        coefficient = stored[stored.model.eq(row["model"])&stored.term.eq("temp_c")].iloc[0]
        assert coefficient.covariance == row["covariance"]
        assert row["n_clusters"] == (int(coefficient.n_sectors) if row["covariance"] == "Sector cluster" else 0)
        for column in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value"]:
            assert abs(row[column]-float(coefficient[column])) < 1e-10
        assert row["significance"] == significance_label(row["p_value"])
        assert row["significance_class"] == significance_class(row["p_value"])
        assert row["p_display"] == p_format(row["p_value"])
        assessment = model_assessment(ROOT/row["source_csv"], row["model"])
        assert row["stationarity_y"] == assessment["y_label"] and row["stationarity_all"] == assessment["all_label"]
        assert row["analysis_role"] == assessment["analysis_role"]
        sample, outcome = independent_sample(row)
        values = sample[outcome].tolist()
        mean = math.fsum(values)/len(values)
        sd = math.sqrt(math.fsum((value-mean)**2 for value in values)/(len(values)-1))
        assert abs(mean-row["mean_y"]) < 1e-10
        assert abs(sd-row["sd_y"]) < 1e-10
        assert row["n_obs"] == len(sample)
        period = "month" if row["frequency"] == "monthly" else "quarter"
        assert row["periods"] == sample[period].nunique()
        assert row["start"] == sample[period].min() and row["end"] == sample[period].max()
        assert np.allclose([row["effect_0p1c"],row["ci_0p1c_lower"],row["ci_0p1c_upper"]],
                           .1*np.array([row["coefficient"],row["ci_lower"],row["ci_upper"]]), atol=1e-12)
        if row["window"] == "exclude":
            assert not sample.year.between(2020, 2022).any()
            assert row["periods"] == (66 if row["frequency"] == "monthly" else 19)
        report["overview_rows_checked"] += 1
    script='\n'.join(document.xpath('//script[not(@type)]/text()'))
    functions='\n'.join(line for line in script.splitlines() if line.startswith(('const number=','function renderOverview()','function renderPairedOverview()')))
    assert functions.count('function render')==2
    node=shutil.which('node');assert node
    harness=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const DATA=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),elements=new Map();
const controls={'[data-window]':['full','exclude','long'].map(window=>({dataset:{window},setAttribute(){}})),'[data-growth]':['short','yoy_nsa','yoy'].map(growth=>({dataset:{growth},setAttribute(){}}))};
const document={getElementById(id){if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:''});return elements.get(id);},querySelectorAll(selector){return controls[selector]||[];}};
const context=vm.createContext({DATA,document});
vm.runInContext("const scale={window:'full',growth:'short'};"+fs.readFileSync(process.argv[2],'utf8'),context);
let checked=0;
for(const growth of ['short','yoy_nsa','yoy'])for(const window of ['full','exclude','long']){
 context.choice={growth,window};vm.runInContext('Object.assign(scale,choice);renderOverview();',context);
 for(const id of ['overview-rows','paired-overview-rows']){
  const rendered=elements.get(id).innerHTML;assert.equal((rendered.match(/<tr>/g)||[]).length,9);assert(!/undefined|NaN/.test(rendered));
  assert(rendered.includes(growth==='yoy_nsa'?'log growth (NSA)':'log growth (SA)'));
 }
 checked++;
}
console.log(JSON.stringify({combinations:checked,status:'passed',scope:'Overview runtime/data wiring, not visual-browser inspection'}));
'''
    with tempfile.TemporaryDirectory(prefix='heat-overview-runtime-') as folder:
        data_file=Path(folder)/'overview.json';data_file.write_text(json.dumps({'overview':overview}))
        code_file=Path(folder)/'overview.js';code_file.write_text(functions)
        result=subprocess.run([node,'-e',harness,str(data_file),str(code_file)],capture_output=True,text=True,check=True)
        report['overview_selector_runtime']=json.loads(result.stdout)
    report["html_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "collection/verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Verified single-file collection: {len(REPORTS)} embedded analyses, {len(REPORTS)+2} tabs and all {len(overview)} exact-sample mean/coefficient comparisons.")
    return report


if __name__ == "__main__":
    verify_collection()
