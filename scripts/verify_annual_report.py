"""Verify annual HTML presentation against saved estimates/statuses, without fits."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html

from significance import p_format, significance_label, significance_stars

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/annual"
VARIANTS = ["all_eligible","exclude_2020_2022"]
TERMS = {"temp_c":"Annual mean air temperature (°C)","rain_100mm":"Annual rainfall (100 mm)",
         "rh_pct":"Annual mean relative humidity (pp)","wind_mps":"Annual mean wind speed (m/s)",
         "pm25_10ug":"Annual mean PM2.5 (10 µg/m³)","trend":"Linear calendar-year trend"}

NODE_SELECTOR_CHECK = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
function scenario(embedded,denyHash){
 let historyCalls=0,hashAttempts=0,hashValue='';
 const nodes={};
 function select(value,values){return {value,options:values.map(value=>({value,disabled:false})),listeners:{},addEventListener(kind,fn){this.listeners[kind]=fn;}};}
 nodes.context=select('va19',['aggregate_va','va11','va19','iip_total','iip6','iip18']);
 nodes.controls=select('full',['full','weather']);
 nodes.adjustment=select('NSA',['NSA','SA']);
 nodes.model=select('H1',['H1','common','separate']);
 nodes['annual-data']={textContent:JSON.stringify({views:input.views})};
 for(const id of ['selected-title','selected-role','selected-outcome','formula','status-grid','native-table','readout','joint'])nodes[id]={textContent:'',innerHTML:''};
 nodes['annual-selection']={dataset:{}};
 const location={};
 Object.defineProperty(location,'hash',{get(){return hashValue;},set(value){hashAttempts++;if(denyHash)throw new Error('SecurityError: hash mutation denied');hashValue=value;}});
 const window={};window.parent=embedded?{}:window;window.location=location;
 const sandbox={document:{getElementById(id){assert(nodes[id],id);return nodes[id];}},location,window,
   history:{replaceState(){historyCalls++;throw new Error('SecurityError: replaceState denied in opaque srcdoc/file');}}};
 vm.createContext(sandbox);
 vm.runInContext(input.script,sandbox,{timeout:10000});
 function render(context,controls,adjustment,model){
  nodes.context.value=context;nodes.controls.value=controls;nodes.adjustment.value=adjustment;nodes.model.value=model;
  nodes.context.listeners.change();
  const effectiveModel=['aggregate_va','iip_total'].includes(context)?'common':model;
  const expected=input.views.find(v=>v.context===context&&v.control_set===controls&&v.adjustment===adjustment&&v.model===effectiveModel);
  assert(expected,'selection is missing');
  assert.equal(nodes['annual-selection'].dataset.viewId,expected.id);
  assert.equal(nodes['native-table'].innerHTML,expected.native_html);
  assert.equal(nodes.readout.innerHTML,expected.readout_html);
  assert.equal(nodes.joint.innerHTML,expected.joint_html+expected.pairwise_html);
  assert.equal((nodes['native-table'].innerHTML.match(/class="[^"]*\bgt_table\b/g)||[]).length,1);
  assert.equal(nodes.model.value,effectiveModel);
  if(expected.statuses.some(s=>s.status!=='estimated'))assert(nodes['status-grid'].innerHTML.includes('Unavailable · underidentified'));
 }
 let viewChecks=0,fallbackChecks=0;
 for(const view of input.views){render(view.context,view.control_set,view.adjustment,view.model);viewChecks++;}
 for(const context of ['aggregate_va','iip_total'])for(const controls of ['weather','full'])for(const adjustment of ['NSA','SA']){
  render(context,controls,adjustment,'H1');
  assert(nodes.model.options.find(o=>o.value==='H1').disabled);
  assert(nodes.model.options.find(o=>o.value==='separate').disabled);
  fallbackChecks++;
 }
 render('va19','full','NSA','H1');
 assert(!nodes.model.options.find(o=>o.value==='H1').disabled);
 assert.equal(historyCalls,0,'unsafe history API was called');
 if(embedded)assert.equal(hashAttempts,0,'embedded srcdoc URL should not be mutated');
 else assert(hashAttempts>0,'standalone guarded hash path was not exercised');
 return {environment:embedded?'opaque_srcdoc':denyHash?'standalone_hash_denied':'standalone_file',view_checks:viewChecks,national_H1_fallback_checks:fallbackChecks,
         history_replaceState_calls:historyCalls,hash_attempts:hashAttempts};
}
process.stdout.write(JSON.stringify({status:'passed',scenarios:[scenario(true,false),scenario(false,false),scenario(false,true)]}));
"""


def selector_runtime(script: str, views: list[dict]) -> dict:
    result = subprocess.run(["node","-e",NODE_SELECTOR_CHECK],input=json.dumps({"script":script,"views":views}),
                            text=True,capture_output=True,cwd=ROOT,check=False)
    if result.returncode:
        raise AssertionError("Annual selector runtime failed: "+result.stderr[-4000:])
    return json.loads(result.stdout)


def equal(actual,expected,label):
    if pd.isna(expected):
        assert actual is None or pd.isna(actual),label
    elif isinstance(expected,(float,int,np.number)) and not isinstance(expected,(bool,np.bool_)):
        assert np.isclose(float(actual),float(expected),atol=1e-12,rtol=1e-12),label
    else:
        assert actual==expected,label


def verify_annual_report() -> dict:
    path = ROOT/"outputs/annual.html"
    root = html.fromstring(path.read_text())
    inventory = json.loads((OUT/"report_inventory.json").read_text())
    payload = json.loads(root.get_element_by_id("annual-data").text)
    views = payload["views"]
    assert views==inventory["views"] and len(views)==56
    assert len(root.xpath('//table[contains(@class,"gt_table")]'))==1,"More than one native table in initial DOM"
    assert not root.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[@src]'),"External report asset"
    ids = root.xpath('//@id');assert len(ids)==len(set(ids))
    assert {option.get("value") for option in root.get_element_by_id("context").xpath('./option')}=={"aggregate_va","va11","va19","iip_total","iip6","iip18"}
    for field,value in [("context","va19"),("controls","full"),("adjustment","NSA"),("model","H1")]:
        options = root.get_element_by_id(field).xpath('./option')
        selected = [option for option in options if "selected" in option.attrib]
        assert (selected[0] if selected else options[0]).get("value")==value
    specs = pd.read_csv(OUT/"specs.csv",dtype={"sector_id":str},low_memory=False)
    requested = set(specs[["context","control_set","adjustment","model"]].itertuples(index=False,name=None))
    assert requested=={(v["context"],v["control_set"],v["adjustment"],v["model"]) for v in views}
    for relative,digest in inventory["source_hashes"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,("Report source changed",relative)
    tables = {name:pd.read_csv(OUT/(name+".csv"),dtype={"sector_id":str,"reference_sector_id":str},low_memory=False)
              for name in ["estimates","sector_estimates","separate_estimates"]}
    names_by_context = {context:group.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
                        for context,group in tables["sector_estimates"].groupby("context")}
    joint_source=pd.read_csv(OUT/'joint_tests.csv',low_memory=False)
    pair_source=pd.read_csv(OUT/'pairwise_tests.csv',dtype={'first_sector_id':str,'second_sector_id':str},low_memory=False)
    cell_count=0;unavailable_columns=0;record_count=0;readout_p_count=0;mean_cells=0;joint_rows=0;pair_rows=0
    for view in views:
        profile = view["model"]!="common"
        source = tables["estimates" if not profile else "sector_estimates" if view["model"]=="H1" else "separate_estimates"]
        source = source[source.spec_id.isin(view["source_spec_ids"])]
        source = source[source.status.ne("estimated") | source.main_inference.eq(True)]
        if view["model"]=="separate":source=source[source.term.eq("temp_c")]
        key = "sector_id" if profile else "term"
        assert len(source)==len(view["source_rows"])
        stored = {(r["sample_variant"],r[key]):r for r in view["source_rows"]}
        assert len(stored)==len(source)
        for row in source.to_dict("records"):
            saved=stored[(row["sample_variant"],row[key])]
            for field in ["coefficient","std_error","ci_lower","ci_upper","p_value","mean_y","sd_y","n_obs","n_periods","n_parameters","design_rank","status"]:
                equal(saved[field],row[field],(view["id"],row[key],field))
            record_count+=1
        fragment=html.fragment_fromstring(view["native_html"],create_parent="div")
        assert len(fragment.xpath('.//table[contains(@class,"gt_table")]'))==1
        fragment_ids=fragment.xpath('.//@id');assert len(fragment_ids)==len(set(fragment_ids))
        for node in fragment.xpath('.//*[@headers]'):
            assert set(node.get('headers').split())<=set(fragment_ids)
        native_rows=fragment.xpath('.//tbody/tr')
        coefficient_rows=[];in_coefficients=False
        for row in native_rows:
            if row.xpath('./th[contains(@class,"gt_group_heading")]'):
                in_coefficients=row.text_content().strip()=="coef"
            elif in_coefficients:coefficient_rows.append(row)
        keys=list(source[source.sample_variant.eq(VARIANTS[0])].sort_values("sector_id").sector_id) if profile else list(source[source.sample_variant.eq(VARIANTS[0])].term)
        assert len(coefficient_rows)==len(keys)
        names=names_by_context.get(view["context"],{})
        for identifier,row in zip(keys,coefficient_rows,strict=True):
            assert row.xpath('./th')[0].text_content().strip()==(names[identifier] if profile else TERMS[identifier])
            cells=row.xpath('./td');assert len(cells)==2
            for variant,cell in zip(VARIANTS,cells,strict=True):
                data=stored[(variant,identifier)]
                text=re.sub(r'\s+',' ',cell.text_content()).strip()
                if data["status"]!="estimated":
                    assert not re.search(r'\d',text),"Unavailable coefficient shown numerically"
                else:
                    numbers=re.findall(r'[-+]?\d[\d,]*\.\d+',text)
                    assert numbers==[f'{data["coefficient"]:.3f}',f'{data["std_error"]:.3f}'],(view["id"],identifier,text)
                    assert text.count('*')==len(significance_stars(data["p_value"]))
                cell_count+=1
        native_text=' '.join(fragment.itertext())
        assert "Year fixed effects" in native_text and "No · would absorb temperature" in native_text
        assert "Annual time residual df" in native_text
        for info in view["statuses"]:
            requested_spec=specs[specs.spec_id.isin(view["source_spec_ids"])&specs.sample_variant.eq(info["sample_variant"])]
            assert requested_spec.status.eq(info["status"]).all()
            if view["control_set"]=="full":
                assert info["n_periods"]==(9 if info["sample_variant"]==VARIANTS[0] else 6)
                if info["sample_variant"]==VARIANTS[1]:
                    assert info["status"]=="unavailable" and "Underidentified" in info["reason"]
                    assert "Unavailable · underidentified" in native_text
                    unavailable_columns+=1
            assert "Unavailable" in info["y_screen"] or "screens compatible" in info["y_screen"] or info["y_screen"] in {"Unit-root concern","Mixed evidence","Inconclusive","Compatible with level stationarity","Compatible with trend stationarity"}
        readout=html.fragment_fromstring(view["readout_html"],create_parent="div")
        heat_source=source if profile else source[source.term.eq("temp_c")]
        heat_keys=keys if profile else ['temp_c']
        rendered_rows=readout.xpath('.//tbody/tr');assert len(rendered_rows)==len(heat_keys)
        for identifier,row in zip(heat_keys,rendered_rows,strict=True):
            cells=row.xpath('./td');assert len(cells)==5
            for i,variant in enumerate(VARIANTS):
                data=stored[(variant,identifier)]
                assert cells[1+2*i].text_content()==f'{data["mean_y"]:,.3f}',(view['id'],identifier,'rendered mean')
                mean_cells+=1
                impact=cells[2+2*i]
                if data['status']=='estimated':
                    assert f'p={p_format(data["p_value"])}' in impact.text_content()
                    assert f'+0.1°C: {.1*data["coefficient"]:+.3f} log points' in impact.text_content()
                else:
                    assert 'Unavailable · not estimated' in impact.text_content()
        expected_p=list(heat_source.loc[heat_source.status.eq("estimated"),"p_value"])
        badges=readout.xpath('.//span[contains(concat(" ",normalize-space(@class)," ")," sig ")]')
        actual_p=[float(node.get("data-p")) for node in badges]
        assert np.allclose(sorted(actual_p),sorted(expected_p),atol=1e-15,rtol=1e-12)
        for node in badges:
            p=float(node.get("data-p"));assert node.getnext().text_content()==significance_label(p)
            assert node.text_content()=="5%: "+("Yes" if p<.05 else "No")
            assert p_format(p) in readout.text_content()
        readout_p_count+=len(badges)
        if view["model"]=="H1":
            assert "Joint test" in view["joint_html"] and "cross-sector covariance" in view["joint_html"]
            fragment=html.fragment_fromstring(view['joint_html'],create_parent='div')
            observed=fragment.xpath('.//tbody/tr');assert len(observed)==2
            chosen=joint_source[joint_source.spec_id.isin(view['source_spec_ids'])]
            chosen=chosen[chosen.status.ne('estimated')|chosen.main_inference.eq(True)]
            for variant,row in zip(VARIANTS,observed,strict=True):
                target=chosen[chosen.sample_variant.eq(variant)].iloc[0];cells=row.xpath('./td')
                assert cells[2].text_content()==p_format(target.p_value)
                assert cells[5].text_content()==target.test_status
                if target.test_status!='available':
                    assert pd.isna(target.p_value) and cells[1].text_content()=='Unavailable'
                joint_rows+=1
            fragment=html.fragment_fromstring(view['pairwise_html'],create_parent='div')
            chosen=pair_source[pair_source.spec_id.isin(view['source_spec_ids'])&pair_source.main_inference.eq(True)]
            observed=fragment.xpath('.//tbody/tr');assert len(observed)==len(chosen)
            for target,row in zip(chosen.itertuples(),observed,strict=True):
                cells=row.xpath('./td')
                assert cells[1].text_content()==names[target.first_sector_id]+' − '+names[target.second_sector_id]
                assert cells[2].text_content()==f'{target.coefficient:+.3f}'
                assert cells[3].text_content()==p_format(target.p_value)
                assert cells[4].text_content()==p_format(target.p_value_holm)
                assert cells[5].xpath('./span')[1].text_content()==significance_label(target.p_value_holm)
                pair_rows+=1
    assert unavailable_columns==28
    script=root.xpath('//script[not(@type="application/json")]')[0].text_content()
    assert "history.replaceState" not in script and "if(window.parent===window)location.hash=" in script
    assert "document.getElementById('native-table').innerHTML=view.native_html" in script
    assert "Object.values(controls).forEach(input=>input.addEventListener('change',show))" in script
    for context in {view['context'] for view in views}:
        for controls in ['weather','full']:
            for adjustment in ['NSA','SA']:
                available={view['model'] for view in views if view['context']==context}
                for model in available:
                    assert sum((view['context'],view['control_set'],view['adjustment'],view['model'])==(context,controls,adjustment,model) for view in views)==1
    runtime = selector_runtime(script,views)
    downloads=0
    for link in root.xpath('//a[@download]'):
        value=link.get('href')
        assert value.startswith('data:application/gzip;base64,')
        content=gzip.decompress(base64.b64decode(value.split(',',1)[1]))
        filename=link.get('download').removeprefix('annual_').removesuffix('.gz')
        candidates=[OUT/filename,OUT/'diagnostics'/filename]
        assert any(file.exists() and file.read_bytes()==content for file in candidates),filename
        downloads+=1
    report={"status":"passed","scope":"Presentation/source mapping only; independent numerical annual verification is separate",
            "paired_views_checked":len(views),"saved_primary_rows_checked":record_count,"native_coefficient_cells_checked":cell_count,
            "explicit_unavailable_exclusion_columns":unavailable_columns,"pointwise_p_readouts_checked":readout_p_count,
            "rendered_mean_cells_checked":mean_cells,"joint_test_rows_checked":joint_rows,"Holm_pairwise_rows_checked":pair_rows,
            "offline_downloads_checked":downloads,"initial_native_tables_in_dom":1,
            "selector_runtime":runtime,
            "report_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_hashes":inventory["source_hashes"]}
    (OUT/'presentation_verification.json').write_text(json.dumps(report,indent=2)+"\n")
    print(f'Annual presentation passed: {len(views)} paired views, {cell_count} native cells, {unavailable_columns} explicit unavailable columns',flush=True)
    return report


if __name__=="__main__":verify_annual_report()
