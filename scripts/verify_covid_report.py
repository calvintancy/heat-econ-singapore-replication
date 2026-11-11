"""Check paired native tables and exercise every selector against the saved estimates."""
from __future__ import annotations
import hashlib,json,math,shutil,subprocess,tempfile
from pathlib import Path
import pandas as pd
from lxml import html
from significance import p_format,significance_class,significance_label,significance_stars
from covid_report import analysis_label,label_term

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/covid_comparison'


def verify_covid_report():
    path=ROOT/'outputs/covid_comparison.html'
    doc=html.fromstring(path.read_text(),parser=html.HTMLParser(huge_tree=True))
    data=json.loads(doc.xpath('//script[@id="covid-data"]/text()')[0])
    originals={engine:pd.read_csv(OUT/engine/'estimates.csv',dtype={'sector_id':str}) for engine in ['common','sector']}
    expected_ids=set().union(*(set(f.pair_id) for f in originals.values()))
    assert set(data['tables'])==set(data['rows'])==expected_ids
    assert len(data['specs'])==len(expected_ids)
    assert {'y_yoy','y_yoy_nsa'}<=set(s['outcome'] for s in data['specs'])
    total_rows=0;native_cells=0
    for spec in data['specs']:
        pid=spec['pair_id'];source=originals[spec['engine']]
        source=source[source.pair_id.eq(pid)&source.main_inference]
        assert spec['analysis']==analysis_label(source.iloc[0])
        key='sector_id' if spec['engine']=='sector' else 'term'
        index={(r['sample_variant'],r[key]):r for r in data['rows'][pid]}
        assert len(index)==len(source)
        table=html.fromstring(data['tables'][pid])
        ids=table.xpath('.//@id');assert len(ids)==len(set(ids))
        for headers in table.xpath('.//@headers'):assert set(headers.split())<=set(ids)
        text=' '.join(table.text_content().split())
        assert 'All eligible dates' in text and 'Exclude 2020–2022' in text
        assert 'PyFixest feols' in text
        if spec['engine']=='common' and spec['n_units']>1:
            assert source.covariance.eq('Sector cluster').all() and 'CR1' in spec['inference']
            assert source.inference_df.eq(spec['n_units']-1).all()
        else:assert 'Calendar-HAC' in spec['inference']
        table_rows={r[0].text_content().strip():r for r in table.xpath('.//tr') if len(r)==3}
        for _,row in source.iterrows():
            saved=index[(row.sample_variant,row[key])]
            for field in ['coefficient','std_error','ci_lower','ci_upper','p_value','mean_y','sd_y']:
                if pd.isna(row[field]):assert saved[field] is None
                else:assert math.isclose(saved[field],row[field],abs_tol=1e-13,rel_tol=1e-13),(pid,field)
            assert saved['p_value_display']==p_format(row.p_value)
            assert saved['p_value_class']==significance_class(row.p_value)
            assert saved['p_value_significance']==significance_label(row.p_value)
            label=row.sector if spec['engine']=='sector' else label_term(row.term)
            target=table_rows[label][1 if row.sample_variant=='all_eligible' else 2].text_content()
            assert f'{row.coefficient:.3f}'+significance_stars(row.p_value) in target,(pid,label,target)
            assert f'({row.std_error:.3f})' in target,(pid,label,target)
            native_cells+=1;total_rows+=1
        for variant,field in [('all_eligible','all_periods'),('exclude_2020_2022','excluded_periods')]:
            r=source[source.sample_variant.eq(variant)].iloc[0]
            assert spec[field]==r.n_periods
            own=spec['family']=='separate_series';col=1 if variant=='all_eligible' else 2
            counts={'Observed periods per series' if own else 'Observations':r.n_obs,
                'Unit count per fit' if own else 'Sector / industry units':r.n_sectors,
                ('Months per series' if spec['frequency']=='monthly' else 'Quarters per series') if own else 'Weather periods':r.n_periods,
                'Inference df':r.inference_df,'Estimated parameters':r.n_parameters,'Residual df':r.n_obs-r.n_parameters}
            for label,value in counts.items():assert float(table_rows[label][col].text_content().replace(',','').strip())==value,(pid,label)
            if spec['engine']=='common':
                for label,value in [('Mean Y (exact sample)',r.mean_y),('SD Y (exact sample)',r.sd_y)]:
                    assert table_rows[label][col].text_content().strip()==f'{value:,.3f}'
        assert spec['all_periods']-spec['excluded_periods']==spec['n_removed_periods']
    for name in ['joint_tests','pairwise_tests','weather_coefficients']:
        frame=pd.read_csv(OUT/'sector'/f'{name}.csv',dtype={'sector_id':str,'first_sector_id':str,'second_sector_id':str})
        frame=frame[frame.main_inference]
        assert sum(map(len,data[name].values()))==len(frame)
        for pid,rows in frame.groupby('pair_id',sort=False):
            saved=pd.DataFrame(data[name][pid])
            assert len(saved)==len(rows)
            for col in ['coefficient','std_error','ci_lower','ci_upper','p_value','p_value_holm','f_statistic','wald_chi2','restrictions','covariance_rank','numerator_df','denominator_df','structural_rank_bound']:
                if col in rows:
                    for new,old in zip(saved[col],rows[col]):
                        if pd.isna(old):assert pd.isna(new)
                        else:assert math.isclose(new,old,abs_tol=1e-13,rel_tol=1e-13),(pid,name,col)
            for col in ['p_value','p_value_holm']:
                if col in rows:
                    assert saved[col+'_display'].tolist()==rows[col].map(p_format).tolist()
                    assert saved[col+'_significance'].tolist()==rows[col].map(significance_label).tolist()
    default=next(s for s in data['specs'] if s['pair_id']==data['default'])
    assert default['engine']=='common' and default['n_units']==19 and 'CR1' in default['inference']
    assert all(t in default['controls'] for t in ['Rainfall','humidity','Wind','PM2.5'])
    # A small DOM adapter runs the actual embedded selector code for all specs.
    # This checks runtime/data wiring; it is not a visual browser inspection.
    node=shutil.which('node');assert node,'Node is required to verify embedded JavaScript'
    harness=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const payload=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const nodes=new Map();
function element(id){if(!nodes.has(id))nodes.set(id,{value:'',hidden:false,textContent:'',_html:'',addEventListener(){},set innerHTML(x){this._html=x;if(['analysis','spec'].includes(id))this.value=String(x).match(/<option value="([^"]*)"/)?.[1]||'';},get innerHTML(){return this._html;}});return nodes.get(id);}
element('covid-data').textContent=JSON.stringify(payload);
const context=vm.createContext({document:{getElementById:element}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
for(const spec of payload.specs){
 context.selectedSpec=spec;
 vm.runInContext("E('engine').value=selectedSpec.engine;choices(selectedSpec);",context);
 assert.equal(element('spec').value,spec.pair_id);
 const base=payload.rows[spec.pair_id].filter(r=>r.sample_variant==='all_eligible');
 assert.equal((element('readout').innerHTML.match(/<tr>/g)||[]).length,base.length);
 assert(element('native-table').innerHTML.includes('All eligible dates'));
 assert(element('native-table').innerHTML.includes('Exclude 2020–2022'));
 assert(element('outcome-definition').textContent.length>50);
 for(const [name,id] of [['joint_tests','joint'],['pairwise_tests','pairwise'],['weather_coefficients','weather']]){
  let rows=payload[name][spec.pair_id]||[];
  if(name==='weather_coefficients')rows=rows.filter(r=>r.term!=='temp_c');
  assert.equal(element(id+'-panel').hidden,!rows.length);
  assert.equal((element(id+'-readout').innerHTML.match(/<tr>/g)||[]).length, name==='joint_tests'?rows.length:rows.filter(r=>r.sample_variant==='all_eligible').length);
 }
 for(const id of ['readout','joint-readout','pairwise-readout','weather-readout'])assert(!/undefined|NaN/.test(element(id).innerHTML));
}
console.log(JSON.stringify({specifications:payload.specs.length,status:'passed'}));
'''
    with tempfile.TemporaryDirectory(prefix='heat-covid-display-') as folder:
        payload=Path(folder)/'payload.json';payload.write_text(json.dumps(data))
        run=subprocess.run([node,'-e',harness,str(payload),str(ROOT/'templates/covid_comparison.js')],capture_output=True,text=True,check=True)
        runtime=json.loads(run.stdout)
    report={'status':'passed','paired_native_tables':len(expected_ids),'native_coefficient_cells':native_cells,
        'saved_row_values_checked':total_rows,'selector_runtime':runtime,
        'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope':'Native HTML structure, unrounded inference labels, all selector combinations and exact saved-data mapping; no visual-browser claim.'}
    (OUT/'report_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Verified {len(expected_ids)} paired native tables, {native_cells} estimate cells and every selector combination.')
    return report


if __name__=='__main__':verify_covid_report()
