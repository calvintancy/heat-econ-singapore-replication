"""Source-backed native-table and actual-JavaScript crisis report checks.

This validates presentation only. Independent numerical audits of each branch
remain separate, and this command never fits or rewrites analytical results.
"""
from __future__ import annotations

import hashlib
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
OUT = ROOT / "outputs/crisis_sensitivity"
VARIANTS = ["all_eligible", "event_dates_excluded", "strict_growth_window_clean"]
BRANCHES = {"long": OUT, "full": ROOT / "outputs/complete_va_controls", "annual": OUT / "annual"}
LABELS = {"temp_c": "Mean air temperature (°C)", "rain_100mm": "Rainfall (100 mm)",
          "rh_pct": "Relative humidity (pp)", "wind_mps": "Wind (m/s)", "pm25_10ug": "PM2.5 (10 µg/m³)"}

NODE_HARNESS = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const decode=s=>String(s).replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');
const strip=s=>decode(String(s).replace(/<[^>]*>/g,'')).replace(/\s+/g,' ').trim();
const nodes=new Map(),downloads=[];let historyCalls=0;
function element(id,select=false){
 if(nodes.has(id))return nodes.get(id);
 let value='',markup='';
 const e={id,hidden:false,dataset:{},listeners:{},options:[],textContent:'',attributes:{},
   addEventListener(kind,fn){this.listeners[kind]=fn;},setAttribute(k,v){this.attributes[k]=String(v);},
   getAttribute(k){return this.attributes[k]??null;},appendChild(){},remove(){},
   click(){if(this.listeners.click)this.listeners.click();if(this.download)downloads.push({filename:this.download,href:this.href});},
   set innerHTML(s){markup=String(s);if(select){this.options=[...markup.matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)].map(m=>({value:decode(m[1].match(/\bvalue="([^"]*)"/)?.[1]??strip(m[2])),disabled:/\bdisabled\b/.test(m[1]),selected:/\bselected\b/.test(m[1])}));value=(this.options.find(o=>o.selected)||this.options[0])?.value??'';}},
   get innerHTML(){return markup;},set value(v){value=String(v);},get value(){return value;}};
 nodes.set(id,e);return e;
}
for(const id of input.ids)element(id,input.select_ids.includes(id));
element('report-data').textContent=JSON.stringify(input.data);
const tabs=['results','definitions','coverage'].map(tab=>{const b=element('tab-'+tab);b.dataset.tab=tab;return b;});
const blobs=new Map();let nextBlob=0;
class Blob {constructor(parts,options){this.parts=parts;this.type=options?.type;}}
const URL={createObjectURL(blob){const key='blob:test-'+(++nextBlob);blobs.set(key,blob);return key;},revokeObjectURL(){}};
const document={getElementById(id){assert(nodes.has(id),'Unknown element '+id);return nodes.get(id);},
 querySelectorAll(query){assert.equal(query,'[data-tab]');return tabs;},createElement(tag){return element('created-'+tag+'-'+Math.random());},body:{appendChild(){}}};
const sandbox={document,Blob,URL,console,location:{hash:''},history:{replaceState(){historyCalls++;throw Error('SecurityError: opaque-origin history mutation');}}};
sandbox.window=sandbox;sandbox.window.parent={};
vm.createContext(sandbox);vm.runInContext(input.script,sandbox,{timeout:15000});
function dispatch(id){assert(element(id).listeners.change,'No selector listener '+id);element(id).listeners.change();}
function number(x,d=3){return x==null||!Number.isFinite(+x)?'—':Number(x).toLocaleString('en',{minimumFractionDigits:d,maximumFractionDigits:d});}
function probability(x){return x==null||!Number.isFinite(+x)?'Unavailable':x<.001?'<0.001':number(x,3);}
function tableRows(markup){return [...(markup.match(/<tbody[^>]*>([\s\S]*?)<\/tbody>/)?.[1]??'').matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g)].map(m=>[...m[1].matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g)].map(c=>strip(c[1])));}
function csvRows(csv){let rows=[],row=[],cell='',quoted=false;for(let i=0;i<csv.length;i++){const c=csv[i];if(c==='"'){if(quoted&&csv[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell='';}else if(c==='\n'&&!quoted){row.push(cell);rows.push(row);row=[];cell='';}else cell+=c;}row.push(cell);rows.push(row);return rows;}
let views=0,readouts=0,rowsChecked=0,downloadChecks=0,jointChecks=0,holmChecks=0;
const defaultView=input.data.views.find(v=>v.id===element('inference').value);
assert(defaultView,'No initial selection');
assert.equal(defaultView.branch,'full');assert.equal(defaultView.context,'va20');assert.equal(defaultView.outcome,'y_yoy_nsa');assert.equal(defaultView.model,'H1');
const defaultRecord={branch:defaultView.branch,context:defaultView.context,outcome:defaultView.outcome,model:defaultView.model,covariance:defaultView.covariance,hac_lags:defaultView.hac_lags};
for(const view of input.data.views){
 for(const [id,value] of [['branch',view.branch],['context',view.context],['outcome',view.outcome],['model',view.model],['inference',view.id]]){element(id).value=value;dispatch(id);}
 assert.equal(element('native-table').innerHTML,view.table,'Native table selection '+view.id);
 assert(element('table-title').textContent.includes(view.outcome_label));
 assert(!/undefined|NaN/.test(element('sample-cards').innerHTML));
 const cards=[...element('sample-cards').innerHTML.matchAll(/<div class="card">([\s\S]*?)<\/div>/g)];
 assert.equal(cards.length,3);
 for(let i=0;i<3;i++){
  const row=view.rows.find(r=>r.sample_variant===input.variants[i]);
  assert(strip(cards[i][1]).includes(String(row.n_periods)));
  assert(strip(cards[i][1]).includes(String(row.n_obs)));
  assert(strip(cards[i][1]).includes('time residual df '+row.time_residual_df));
 }
 for(const variant of input.variants){
  element('readout-sample').value=variant;dispatch('readout-sample');
  const source=view.rows.filter(r=>r.sample_variant===variant),markup=element('readout').innerHTML;
  const heads=[...(markup.match(/<thead[^>]*>([\s\S]*?)<\/thead>/)?.[1]??'').matchAll(/<th\b[^>]*>([\s\S]*?)<\/th>/g)].map(m=>strip(m[1]));
  const actual=tableRows(markup);assert.equal(actual.length,source.length,view.id+' readout count');
  for(let i=0;i<source.length;i++){
   const row=source[i],cells=actual[i];assert.equal(cells.length,heads.length);
   const cell=label=>cells[heads.indexOf(label)];
   assert.equal(cells[0],view.model==='common'?row.term:row.sector);
   assert.equal(cell('Estimate'),number(row.coefficient));assert.equal(cell('SE'),number(row.std_error));
   assert.equal(cell('95% CI'),'['+number(row.ci_lower)+', '+number(row.ci_upper)+']');
   assert.equal(cell('p-value'),probability(row.p_value));
   assert.equal(cell('Significant at 5%?'),row.p_value==null?'Unavailable':row.p_value<.05?'Yes':'No');
   assert.equal(cell('Mean Y'),number(row.mean_y));assert.equal(cell('SD Y'),number(row.sd_y));
   const holmIndex=heads.findIndex(h=>/Holm/i.test(h));
   if(row.p_value_holm_within_table!=null){assert(holmIndex>=0,'Missing actual saved Holm readout');assert.equal(cells[holmIndex],probability(row.p_value_holm_within_table));holmChecks++;}
   rowsChecked++;
  }
  assert(!/undefined|NaN/.test(markup));
  const joints=view.joint.filter(j=>j.sample_variant===variant);
  const jointText=strip(element('joint').innerHTML);
  if(joints.length){assert(jointText.includes('Computability is not calibration'));assert(jointText.includes('nominal Wald p'));}
  for(const joint of joints){assert(jointText.includes(joint.status));assert(jointText.includes(probability(joint.p_value)));if(joint.status!=='available')assert.equal(joint.p_value,null);jointChecks++;}
  if(!joints.length)assert.equal(element('joint').innerHTML,'');
  if(input.export_id){
   const prior=downloads.length;assert(element(input.export_id).listeners.click,'Missing selected CSV click handler');element(input.export_id).click();
   assert.equal(downloads.length,prior+1);const download=downloads.at(-1),blob=blobs.get(download.href);
   assert(blob,'Selected CSV export is not a generated blob');const csv=blob.parts.join('');
   assert(csv.includes('coefficient')&&csv.includes('p_value')&&csv.includes('mean_y'));
   assert(csv.includes(variant));assert(!/undefined|NaN/.test(csv));
   const parsed=csvRows(csv),fields=Object.keys(source[0]);assert.deepEqual(parsed[0],fields);assert.equal(parsed.length,source.length+1);
   for(let i=0;i<source.length;i++)assert.deepEqual(parsed[i+1],fields.map(field=>String(source[i][field]??'')));
   downloadChecks++;
  }
  readouts++;
 }
 views++;
}
for(const tab of tabs){tab.click();for(const id of ['results','definitions','coverage'])assert.equal(element(id).hidden,id!==tab.dataset.tab);}
assert.equal(historyCalls,0);
process.stdout.write(JSON.stringify({status:'passed',selector_views:views,three_sample_readouts:readouts,source_row_readouts:rowsChecked,
 Holm_readouts:holmChecks,joint_test_readouts:jointChecks,selected_csv_exports:downloadChecks,default_selection:defaultRecord,
 history_replaceState_calls:historyCalls,scope:'Actual embedded JavaScript in Node VM with DOM adapter; no visual browser or cross-browser claim'}));
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sector_id": str}, float_precision="round_trip", low_memory=False)


def equal(actual, expected, name):
    if pd.isna(expected): assert actual is None or pd.isna(actual), name
    elif isinstance(expected, (float, int, np.number)) and not isinstance(expected, (bool, np.bool_)):
        assert np.isclose(actual, expected, atol=1e-12, rtol=1e-12), (name, actual, expected)
    else: assert actual == expected, (name, actual, expected)


def rows_by_label(fragment):
    return {row.xpath('./th|./td')[0].text_content().strip(): row.xpath('./th|./td')[1:]
            for row in fragment.xpath('.//tbody/tr') if len(row.xpath('./th|./td')) == 4}


def _verify_crisis_report() -> dict:
    destination = ROOT / "outputs/crisis_sensitivity.html"
    doc = html.fromstring(destination.read_text(), parser=html.HTMLParser(huge_tree=True))
    data = json.loads(doc.get_element_by_id("report-data").text)
    inventory = json.loads((OUT / "report_inventory.json").read_text())
    views = data["views"]
    assert len(views) == inventory["views"] == 897
    assert not doc.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[@src]'), "External assets"
    static_ids = doc.xpath('//@id');assert len(static_ids) == len(set(static_ids))
    assert len({view["id"] for view in views}) == len(views)
    assert not doc.xpath('//table'), "Native tables should be selected into one runtime DOM slot"
    assert data["sources"] == inventory["sources"]
    for source, digest in data["sources"].items(): assert sha(ROOT / source) == digest, source
    for branch, receipt in data["verification"].items():
        path = ROOT / receipt["source"]
        assert sha(path) == receipt["sha256"] and json.loads(path.read_text())["status"] == "passed"
    cache = {}
    for branch, folder in BRANCHES.items():
        for model, name in [("common", "common_estimates" if branch == "annual" else "estimates"), ("H1", "sector_estimates"), ("separate", "separate_estimates")]:
            frame = read(folder / f"{name}.csv")
            if "term" not in frame: frame["term"] = "temp_c" if model == "common" else "total_temperature_effect"
            if model == "separate": frame = frame.loc[frame.term.isin(["temp_c", "total_temperature_effect"])]
            cache[(branch, model)] = frame
        cache[(branch, "joint")] = read(folder / "joint_tests.csv")
    saved_rows = coefficient_cells = mean_cells = native_stat_cells = joint_rows = holm_rows = 0
    all_native_ids = set()
    for view in views:
        source = cache[(view["branch"], view["model"])]
        source = source.loc[source.context.eq(view["context"]) & source.outcome.eq(view["outcome"]) & source.covariance.eq(view["covariance"]) & source.hac_lags.eq(view["hac_lags"])]
        key = "term" if view["model"] == "common" else "sector_id"
        saved = {(row["sample_variant"], row[key]): row for row in view["rows"]}
        assert len(saved) == len(source) == len(view["rows"]), view["id"]
        for row in source.to_dict("records"):
            actual = saved[(row["sample_variant"], row[key])]
            for field in actual:
                assert field in row, (view["id"], field)
                equal(actual[field], row[field], (view["id"], row[key], field))
            assert (actual["p_value"] < .05) == (row["p_value"] < .05)
            if pd.notna(row.get("p_value_holm_within_table", np.nan)): holm_rows += 1
            saved_rows += 1
        if view["outcome"] == "y_short":
            assert ("MoM" if view["frequency"] == "monthly" else "QoQ") in view["outcome_label"]
            assert ("QoQ" if view["frequency"] == "monthly" else "MoM") not in view["outcome_label"]
        expected_controls = ["temp_c", "rain_100mm", "rh_pct"] + (["wind_mps", "pm25_10ug"] if view["branch"] == "full" else [])
        assert ("PM2.5" in view["controls"]) == (view["branch"] == "full")
        if "controls" in source:
            for value in source.controls.unique():
                listed = json.loads(value)
                # Annual producer lists other-weather controls separately from T.
                assert set(listed) | {"temp_c"} == set(expected_controls)
        fragment = html.fragment_fromstring(view["table"], create_parent="div", parser=html.HTMLParser(huge_tree=True))
        tables = fragment.xpath('.//table[contains(concat(" ",normalize-space(@class)," ")," gt_table ")]')
        assert len(tables) == 1
        ids = fragment.xpath('.//@id');assert len(ids) == len(set(ids)) and not all_native_ids.intersection(ids)
        all_native_ids.update(ids)
        for headers in fragment.xpath('.//@headers'): assert set(headers.split()) <= set(ids)
        coefficient_rows = [];inside = False
        for row in fragment.xpath('.//tbody/tr'):
            headings = row.xpath('./th[contains(@class,"gt_group_heading")]')
            if headings: inside = row.text_content().strip() == "coef"
            elif inside: coefficient_rows.append(row)
        first = source.loc[source.sample_variant.eq(VARIANTS[0])]
        identifiers = list(first[key]);assert len(coefficient_rows) == len(identifiers)
        labels = first.set_index(key).sector.to_dict() if key == "sector_id" else LABELS
        for identifier, row in zip(identifiers, coefficient_rows, strict=True):
            assert row.xpath('./th')[0].text_content().strip() == labels[identifier]
            cells = row.xpath('./td');assert len(cells) == 3
            for variant, cell in zip(VARIANTS, cells, strict=True):
                actual = saved[(variant, identifier)]
                text = re.sub(r'\s+', ' ', cell.text_content()).strip()
                numbers = re.findall(r'[-+]?\d[\d,]*\.\d+', text)
                assert [value.replace(',', '') for value in numbers] == [f'{actual["coefficient"]:.3f}', f'{actual["std_error"]:.3f}'], (view["id"], identifier, text)
                assert text.count('*') == len(significance_stars(actual["p_value"])), (view["id"], identifier, "stars")
                coefficient_cells += 1
        table_rows = rows_by_label(fragment)
        for col, variant in enumerate(VARIANTS):
            row = source.loc[source.sample_variant.eq(variant)].iloc[0]
            counts = {"Observations per equation" if view["model"] == "separate" else "Observations": row.n_obs,
                "Distinct weather dates": row.n_periods, "Inference df": row.inference_df,
                "Estimated parameters per fit": row.n_parameters, "Time residual df": row.time_residual_df,
                "Missing calendar dates": row.calendar_gaps}
            for label, value in counts.items():
                assert float(table_rows[label][col].text_content().replace(',', '').strip()) == value, (view["id"], label)
                native_stat_cells += 1
            for label, value in [("First period", row.start), ("Last period", row.end)]:
                assert table_rows[label][col].text_content().strip() == str(value)
            if view["model"] == "common":
                for label, value in [("Mean observed Y", row.mean_y), ("SD observed Y", row.sd_y)]:
                    assert table_rows[label][col].text_content().strip().replace(',', '') == f'{value:.3f}'
                    mean_cells += 1
            if view["model"] == "separate":
                assert not any(label.startswith(("R²", "R2")) or label.lower() == "r-squared" for label in table_rows)
            else: assert table_rows.get("R² including FE", table_rows.get("R2 including FE"))[col].text_content().strip() == f'{row.r_squared:.3f}'
            if view["annual"]:
                assert "Year FE" not in table_rows and "Calendar-season FE" not in table_rows and "Sector × calendar-season FE" not in table_rows
                assert table_rows["Linear time trend"][col].text_content().strip() == "Yes"
            else:
                assert "Year FE" in table_rows
                assert ("Sector × calendar-season FE" in table_rows) == (row.n_sectors > 1)
                assert ("Calendar-season FE" in table_rows) == (row.n_sectors == 1)
            if view["branch"] == "full": assert row.n_periods == dict(zip(VARIANTS, [29, 19, 17]))[variant]
        joint_source = cache[(view["branch"], "joint")]
        joint_source = joint_source.loc[joint_source.context.eq(view["context"]) & joint_source.outcome.eq(view["outcome"]) & joint_source.model.eq(view["model"]) & joint_source.covariance.eq(view["covariance"]) & joint_source.hac_lags.eq(view["hac_lags"])]
        assert len(joint_source) == len(view["joint"])
        for saved_joint in view["joint"]:
            row = joint_source.loc[joint_source.sample_variant.eq(saved_joint["sample_variant"])].iloc[0]
            for field, value in saved_joint.items(): equal(value, row[field], (view["id"], "joint", field))
            if row.status != "available": assert pd.isna(row.p_value)
            joint_rows += 1
    script = doc.xpath('//script[not(@type="application/json")]')[0].text_content()
    assert "history.replaceState" not in script
    export_buttons = [node.get('id') for node in doc.xpath('//button[@id]') if 'csv' in node.text_content().lower()]
    assert len(export_buttons) == 1, "Selected readout CSV export is missing or ambiguous"
    input_payload = {"data": data, "script": script, "ids": static_ids, "select_ids": [node.get('id') for node in doc.xpath('//select')],
                     "variants": VARIANTS, "export_id": export_buttons[0]}
    with tempfile.TemporaryDirectory(prefix="heat-crisis-presentation-") as temporary:
        path = Path(temporary) / "payload.json";path.write_text(json.dumps(input_payload, ensure_ascii=False, allow_nan=False))
        run = subprocess.run(["node", "-e", NODE_HARNESS, str(path)], cwd=ROOT, text=True, capture_output=True, check=False)
        assert run.returncode == 0, "Crisis selector runtime failed: " + run.stderr[-6000:]
        runtime = json.loads(run.stdout)
    report = {"status": "passed", "scope": "Presentation/source mapping and actual Node selector runtime; numerical verification is separately supplied for each branch. No visual-browser claim.",
        "native_three_column_tables": len(views), "saved_rows_checked": saved_rows, "native_coefficient_cells": coefficient_cells,
        "native_stat_cells": native_stat_cells, "native_mean_sd_cells": mean_cells, "joint_rows": joint_rows, "saved_Holm_rows": holm_rows,
        "selector_runtime": runtime, "report_sha256": sha(destination), "report_source_sha256": sha(ROOT / "scripts/crisis_report.py"),
        "template_sha256": sha(ROOT / "templates/crisis_sensitivity.html"), "source_hashes": {**data["sources"], **{str(folder.relative_to(ROOT) / "joint_tests.csv"): sha(folder / "joint_tests.csv") for folder in BRANCHES.values()}},
        "independent_numerical_receipts": data["verification"]}
    persist(OUT / "report_verification.json", report)
    print(f"Crisis presentation passed: {len(views)} native tables, {coefficient_cells} coefficient cells, {runtime['three_sample_readouts']} readouts", flush=True)
    return report


def verify_crisis_report():
    with presentation_run(OUT / "report_verification.json", "New source-backed presentation verification"):
        return _verify_crisis_report()

if __name__ == "__main__": verify_crisis_report()
