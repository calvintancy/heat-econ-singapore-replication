"""Native two-column full/no-COVID regression tables for the complete analysis inventory."""
from __future__ import annotations
import base64,gzip,html,json,re
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import maketables as mt
from lxml import html as lh
from significance import significance_label,significance_class,significance_stars,p_format
from regression_tables import namespace_table_ids
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/covid_comparison'
VARIANTS=['all_eligible','exclude_2020_2022']
TERMS={'temp_c':'Mean air temperature (°C)','temp_c_unweighted':'Temperature, equal-month mean (°C)',
       'rain_100mm':'Rainfall (100 mm)','rh_pct':'Relative humidity (pp)','wind_mps':'Wind speed (m/s)',
       'pm25_10ug':'PM2.5 (10 µg/m³)','temp_year_mean':'Trailing-year mean temperature (°C)',
       'temp_year_change':'YoY temperature change (°C)','temp_year_lag':'Same period last year temperature (°C)'}


def label_term(term):
    if term in TERMS:return TERMS[term]
    if term.startswith('temperature_lag_'):return term.replace('temperature_lag_','Temperature lag ')
    if term.startswith('temp_l'):return term.replace('temp_l','Temperature lag ')
    return term.replace('_',' ')


def outcome_label(variable,frequency):
    short='MoM' if frequency=='monthly' else 'QoQ'
    return {'y_short':short+' log growth','g_iip_sa':short+' IIP log growth','g_iip_nsa':short+' IIP log growth (NSA)',
      'g_va_sa_qoq':'QoQ VA log growth','g_va_nsa_qoq':'QoQ VA log growth (NSA)','y_short_nsa':short+' log growth (NSA)',
      'y_yoy':'YoY log growth (SA)','y_yoy_nsa':'YoY log growth (NSA)','y_yoy_pct':'YoY exact percentage growth (SA)','y_yoy_pct_nsa':'YoY exact percentage growth (NSA)','y_log_level':'100 × log output (SA)',
      'y_raw_level':'Output level'}.get(variable,variable)


def analysis_label(row):
    p=str(row.source_csv);frequency=str(row.frequency)
    if 'aggregate_va' in p:return 'Aggregate VA'
    if 'iip-monthly-total' in p or ('aggregate_results' in p and frequency=='monthly'):return 'Monthly IIP · published total'
    if 'iip-quarterly-total' in p or ('aggregate_results' in p and frequency=='quarterly'):return 'Quarterly IIP · published total'
    iip='iip' in p or str(row.context).startswith('iip')
    if iip:
        detail='18 industries' if 'detailed' in p else '6 clusters'
        return ('Monthly' if frequency=='monthly' else 'Quarterly')+' IIP · '+detail
    return 'VA · '+('19 detailed sectors' if 'va-detailed' in p else '11 broad sectors')


def clean_records(frame):
    frame=frame.copy()
    for col in ['p_value','p_value_holm']:
        if col in frame:
            frame[col+'_display']=frame[col].map(p_format)
            frame[col+'_significance']=frame[col].map(significance_label)
            frame[col+'_class']=frame[col].map(significance_class)
    return json.loads(frame.to_json(orient='records',double_precision=15))


def controls_label(value):
    try:
        terms=json.loads(value)
    except (TypeError,ValueError):return str(value)
    labels=[label_term(t) for t in terms if not t.startswith('season_trend_')]
    if any(t.startswith('season_trend_') for t in terms):labels.append('season-specific linear trends')
    return ', '.join(labels) or 'None'


def download(path,label):
    target=path.with_suffix(path.suffix+'.gz');target.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    return f'<a href="data:application/gzip;base64,{base64.b64encode(target.read_bytes()).decode()}" download="{path.parent.name}_{target.name}">{html.escape(label)} · CSV.gz ↓</a>'


@dataclass
class PairResult:
    rows:pd.DataFrame
    dependent:str
    sector:bool
    own_series:bool
    @property
    def __maketables_coef_table__(self):
        key='sector_id' if self.sector else 'term'
        f=self.rows.set_index(key)[['coefficient','std_error','p_value','ci_lower','ci_upper']].copy()
        assert f.index.is_unique
        return f.rename(columns={'coefficient':'b','std_error':'se','p_value':'p','ci_lower':'ci95l','ci_upper':'ci95u'})
    @property
    def __maketables_depvar__(self):return self.dependent
    @property
    def __maketables_fixef_string__(self):
        r=self.rows.iloc[0];f=str(r.formula) if 'formula' in self.rows else ''
        if self.own_series:return 'season+year'
        if str(r.source_model)=='R1':return 'sector_season+sector_year+date'
        if self.sector and str(r.source_model) in ['H3','H3X']:return 'sector_season+sector_year'
        return ('sector_season' if int(r.n_sectors)>1 else 'season')+'+year'
    def __maketables_stat__(self,key):
        r=self.rows.iloc[0]
        return {'N':int(r.n_obs),'periods':int(r.n_periods),'units':int(r.n_sectors)}.get(key)


def native_table(rows,info):
    sector=info['engine']=='sector';own='separate' in str(info['family']) or 'series' in str(info['family']).lower()
    models=[PairResult(rows[rows.sample_variant.eq(v)].copy(),info['outcome_label'],sector,own) for v in VARIANTS]
    assert all(len(m.rows) for m in models)
    observed=models[0].rows
    labels=observed.set_index('sector_id').sector.to_dict() if sector else {t:label_term(t) for t in observed.term}
    keep=list(labels)
    row0=[m.rows.iloc[0] for m in models]
    custom={'Estimator':['PyFixest feols '+str(r.estimator_version) for r in row0],
        'Controls':[info['controls']]*2,'Inference':[info['inference']]*2,
        'Clusters':[str(int(r.n_clusters)) if pd.notna(r.n_clusters) and float(r.n_clusters)>0 else '—' for r in row0],
        'Inference df':[int(r.inference_df) for r in row0],
        'Estimated parameters':[int(r.n_parameters) for r in row0],
        'Residual df':[int(r.n_obs-r.n_parameters) for r in row0],
        'First period':[str(r.start) for r in row0],'Last period':[str(r.end) for r in row0]}
    if not sector:
        custom['Mean Y (exact sample)']=[f'{r.mean_y:,.3f}' for r in row0]
        custom['SD Y (exact sample)']=[f'{r.sd_y:,.3f}' for r in row0]
        custom['R²']=[f'{r.r_squared:.3f}' for r in row0]
        custom['Time residual df']=[int(r.time_residual_df) for r in row0]
    else:custom['Reported slopes']=['Relative to reference' if str(info['source_model'])=='R1' else 'Absolute total effect βT + δᵢ' if not own else 'Own-series temperature slope']*2
    custom['COVID dates removed']=[0,int(info['n_removed_periods'])]
    et=mt.ETable(models,model_heads=['All eligible dates','Exclude 2020–2022'],head_order='dh',coef_fmt='b:.3f* \n (se:.3f)',
        signif_code=[.01,.05,.10],keep=keep,labels=labels,exact_match=True,
        felabels={'sector_season':'Sector × season FE','season':'Season FE','year':'Common year FE' if sector and not own else 'Year FE',
                  'sector_year':'Sector × year FE','date':'Full date FE'},
        feorder=['sector_season','season','year','sector_year','date'],model_stats=['N','units','periods'],
        model_stats_labels={'N':'Observed periods per series' if own else 'Observations','units':'Unit count per fit' if own else 'Sector / industry units','periods':'Months per series' if own and info['frequency']=='monthly' else 'Quarters per series' if own else 'Weather periods'},
        custom_model_stats=custom,stats_order='sc',
        notes='Identical outcome, controls and FE structure; only endpoint observations dated 2020–2022 are removed. Growth/lags are built before filtering. Stars use saved unrounded p: *** p<0.01, ** p<0.05, * p<0.10. Sector total-effect p-values test βT+δᵢ=0 with joint covariance; relative R1 tests differences from reference. No pointwise multiple-testing adjustment. Serial/cross-sector inference and short samples retain their stated limitations.')
    raw=et.make(type='gt',gt_style={'table_width':'100%','first_col_width':'270px','table_font_size_all':'13px','source_notes_font_size':'11px'}).as_raw_html()
    document=lh.fragment_fromstring(raw,create_parent='div');table=document.xpath('.//table')[0]
    # Keep actual ETable markup; one shared stylesheet avoids embedding hundreds of duplicate GT styles.
    for node in table.iter():
        node.attrib.pop('style',None)
    result=namespace_table_ids(lh.tostring(table,encoding='unicode'),info['pair_id'])
    return result


def build_covid_report():
    parts=[]
    for engine in ['common','sector']:
        path=OUT/engine/'estimates.csv';f=pd.read_csv(path,dtype={'pair_id':str,'sector_id':str,'source_model':str})
        f['engine']=engine
        for target,fallback in [('n_periods','n_quarters'),('n_clusters',None),('formula',None),('control_set',None),('family',None),('description',None)]:
            if target not in f:f[target]=f[fallback] if fallback and fallback in f else 0 if target=='n_clusters' else ''
        # main_inference marks unit CR1 for common slopes and baseline calendar bandwidth for unit-specific/national fits.
        if f.main_inference.dtype!=bool:f['main_inference']=f.main_inference.astype(str).str.lower().eq('true')
        parts.append(f)
    allrows=pd.concat(parts,ignore_index=True);main=allrows[allrows.main_inference].copy()
    specs=[];tables={};detail={}
    for pid,rows in main.groupby('pair_id',sort=False):
        assert set(rows.sample_variant)==set(VARIANTS),(pid,'Missing paired sample')
        row=rows.iloc[0];rfull=rows[rows.sample_variant.eq('all_eligible')].iloc[0];rex=rows[rows.sample_variant.eq('exclude_2020_2022')].iloc[0]
        sector=row.engine=='sector';key='sector_id' if sector else 'term'
        assert not rows.duplicated(['sample_variant',key]).any(),(pid,'Nonunique main inference')
        controls=controls_label(row.controls) if 'controls' in rows and pd.notna(row.controls) else str(row.control_set)
        if not controls or controls=='nan':controls=', '.join(label for term,label in [('rain_100mm','rainfall'),('rh_pct','humidity'),('wind_mps','wind'),('pm25_10ug','PM2.5')] if term in str(row.formula))
        inference='Sector / industry CR1 · t(S−1)' if row.covariance=='Sector cluster' else f'Calendar-HAC (h={int(row.hac_lags)}) · t(T−1)'
        if not sector and int(row.n_sectors)>1:assert row.covariance=='Sector cluster',(pid,'Pooled primary errors must cluster at unit i')
        nremoved=int(rfull.n_periods-rex.n_periods)
        info={'pair_id':pid,'engine':row.engine,'context':str(row.context),'frequency':str(row.frequency),'family':str(row.family),
            'source_csv':str(row.source_csv),'source_model':str(row.source_model),'outcome':str(row.outcome),
            'outcome_label':outcome_label(str(row.outcome),str(row.frequency)),'analysis':analysis_label(row),
            'description':str(row.description) if pd.notna(row.description) else str(row.source_model),
            'controls':controls,'control_set':str(row.control_set),'inference':inference,'n_removed_periods':nremoved,
            'n_units':int(rfull.n_sectors),'all_periods':int(rfull.n_periods),'excluded_periods':int(rex.n_periods),
            'all_start':str(rfull.start),'all_end':str(rfull.end),'excluded_start':str(rex.start),'excluded_end':str(rex.end),
            'formula':str(row.formula)}
        if sector:
            own=str(row.family)=='separate_series';model=str(row.source_model)
            weather='sector-specific weather slopes' if model in ['H2','H3','R1'] or own else 'shared other-weather slopes'
            years='sector × year FE' if model in ['H3','R1'] else 'year FE'
            info['formula']=(info['outcome_label']+' ~ '+('temperature + controls + season FE + year FE, fitted separately for each sector' if own else 'temperature + temperature × sector + '+weather+' + sector × season FE + '+years+(' + full date FE (relative effects only)' if model=='R1' else '')))
        specs.append(info);tables[pid]=native_table(rows,info)
        detail[pid]=clean_records(rows[[c for c in ['sample_variant','sector_id','sector','term','coefficient','std_error','ci_lower','ci_upper','p_value','mean_y','sd_y','temp_main','interaction_delta','delta_is_normalization'] if c in rows]])
    # Default to the baseline detailed-VA common-slope FE coefficient: unit-clustered main inference.
    default=next((s['pair_id'] for s in specs if s['engine']=='common' and 'granular/va-detailed/full/' in s['source_csv'] and s['source_model']=='G1'),specs[0]['pair_id'])
    extra={}
    for name in ['joint_tests','pairwise_tests','weather_coefficients']:
        frame=pd.read_csv(OUT/'sector'/f'{name}.csv',dtype={'pair_id':str,'sector_id':str,'first_sector_id':str,'second_sector_id':str})
        frame=frame[frame.main_inference]
        fields=['pair_id','sample_variant','sector_id','sector','term','first_sector_id','second_sector_id','comparison','coefficient','std_error','ci_lower','ci_upper','p_value','p_value_holm','multiplicity_family','test','restrictions','covariance_rank','status','wald_chi2','f_statistic','numerator_df','denominator_df','structural_rank_bound','covariance_condition','n_parameters','residual_df','n_obs']
        frame=frame[[c for c in fields if c in frame]]
        extra[name]={pid:clean_records(rows) for pid,rows in frame.groupby('pair_id',sort=False)}
    payload=json.dumps({'specs':specs,'tables':tables,'rows':detail,'default':default,**extra},ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    pd.DataFrame(specs).to_csv(OUT/'report_inventory.csv',index=False)
    downloads=' · '.join(download(OUT/e/'estimates.csv',e.title()+' model estimates and all inference checks') for e in ['common','sector'])
    downloads+=' · '+download(OUT/'report_inventory.csv','Comparison inventory')
    for name,label in [('joint_tests','Omnibus equality tests'),('pairwise_tests','All pairwise differences and Holm p-values'),('weather_coefficients','Own-series weather coefficients'),('interaction_components','Temperature main and interaction components'),('component_covariances','Full component covariance'),('slope_covariances','Full slope covariance')]:
        downloads+=' · '+download(OUT/'sector'/f'{name}.csv',label)
    for engine in ['common','sector']:
        path=OUT/engine/'pyfixest/estimator_manifest.json'
        downloads+=f' · <a href="data:application/json;base64,{base64.b64encode(path.read_bytes()).decode()}" download="{engine}_native_estimator_manifest.json">{engine.title()} native-estimator receipt ↓</a>'
    style='''*{box-sizing:border-box}body{margin:0;background:#f3f7fa;color:#15334a;font:14px/1.65 "Avenir Next","Segoe UI",Arial,sans-serif}main{max-width:1320px;margin:auto;padding:30px 32px}h1,h2,h3{font-family:"Avenir Next Condensed",Arial,sans-serif;line-height:1.15}h1{font-size:44px}h2{font-size:29px}h3{font-size:21px}a{color:#1f77b4}.eyebrow{font:11px Menlo,monospace;color:#64798b;letter-spacing:1px}.callout{background:#e9f3fa;border-left:3px solid #1f77b4;padding:16px 20px;margin:23px 0}.warning{background:#fff4e8;border-color:#c67b35}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0;padding:18px;background:white;border:1px solid #dbe4eb;border-radius:8px}.controls label{font-size:11px;color:#64798b}.controls select{display:block;font:13px inherit;max-width:440px;margin-top:5px;padding:10px;border:1px solid #c8d8e5;border-radius:5px;background:white;color:#15334a}.notes,.sub{font-size:12px;color:#64798b}.sub{display:block}.scroll{overflow:auto;background:white;border:1px solid #dbe4eb;border-radius:8px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:11px 12px;border-bottom:1px solid #dbe4eb;text-align:right;font-variant-numeric:tabular-nums}td:first-child,th:first-child{text-align:left;min-width:190px}.gt_table .gt_left{text-align:left}.gt_table .gt_center{text-align:center}.gt_table .gt_sourcenote{text-align:left;color:#64798b;font-size:11px;padding-top:15px}.gt_table .gt_group_heading{text-align:left;color:#64798b;font-size:10px;text-transform:uppercase;border-top:1px solid #a9bdcc}.gt_table .gt_col_headings{border-bottom:2px solid #a9bdcc}.gt_table .gt_heading{text-align:center}.gt_table td{white-space:pre-line}.badge{display:inline-block;padding:3px 6px;border-radius:4px;background:#f0f3f6;font-size:10px;color:#536b7c}.sig-1,.sig-5{background:#e5f2ec;color:#215b46}.sig-10{background:#fff1d9;color:#795719}.sample-cards{display:flex;gap:15px;flex-wrap:wrap}.sample-card{flex:1;min-width:230px;padding:18px;background:white;border:1px solid #dbe4eb;border-radius:8px}.sample-card strong{font-size:24px;display:block}.formula{font:12px/1.8 Menlo,monospace;overflow-wrap:anywhere;padding:15px;background:white;border:1px solid #dbe4eb}.downloads,footer{font-size:12px}footer{border-top:1px solid #dbe4eb;margin-top:30px;padding-top:20px}details{margin:22px 0}summary{cursor:pointer;font-weight:600}@media(max-width:720px){main{padding:20px 16px}h1{font-size:34px}}'''
    body=f'''<main><div class="eyebrow">Singapore / Paired sample comparisons</div><h1>All eligible dates versus excluding COVID years.</h1><p>Monthly and quarterly regression specifications have both samples here, retaining SA and NSA YoY versions. <a href="annual.html">Annual-frequency counterparts are reported separately →</a> Within a pair, the outcome, controls, fixed effects and estimation method are unchanged. Only observations dated <strong>2020–2022</strong> are removed. “All eligible” means the dates available under that specification’s control/coverage rules, not one common start date for every model.</p>
<div class="callout"><strong>Main pooled inference clusters at sector or industry i.</strong> The paired pooled FE coefficient uses CR1 and t(S−1), with the number of clusters shown in each column. National totals contain one unit and use time-series HAC. Individual industry slopes are supported by one unit, so their absolute/relative profiles retain joint calendar-HAC inference. These distinctions apply in both samples.</div>
<div class="controls"><label>Result type<select id="engine"><option value="common">Pooled FE / national totals</option><option value="sector">Sector FE profiles / own-series robustness</option></select></label><label>Economic series<select id="analysis"></select></label><label>Specification<select id="spec"></select></label></div>
<h2 id="spec-title"></h2><p class="notes" id="spec-details"></p><p id="outcome-definition"></p><div class="sample-cards" id="sample-cards"></div><p class="callout warning" id="sample-warning"></p><p class="formula" id="formula"></p><div class="scroll" id="native-table"></div>
<h3>Significance and the size of the estimate</h3><p class="notes">Significance labels use the saved unrounded p-values. Each sector total is βT + δᵢ; its p-value tests that sum against zero with the full covariance. Relative R1 effects instead test differences from the reference. Stars and sector tests are pointwise; a significant coefficient in one sample but not the other is not itself a formal test that the two estimates differ. Means use the outcome’s own units and sample.</p><div class="scroll"><table><thead><tr><th>Sector or coefficient</th><th>All eligible · β (SE)</th><th>p / significance</th><th>Exclude 2020–2022 · β (SE)</th><th>p / significance</th><th>Mean Y · all / excluded</th></tr></thead><tbody id="readout"></tbody></table></div>
<section id="joint-panel" hidden><h3>Joint test: do sector temperature effects differ?</h3><p class="notes">H₀: all sector slopes are equal; equivalently all nonreference temperature deviations are zero. The common temperature coefficient is unrestricted. This differs from testing a sector’s total effect against zero. Singular covariance gives an unavailable test, with no pseudoinverse p-value.</p><div class="scroll"><table><thead><tr><th>Sample</th><th>F statistic</th><th>p / significance</th><th>Restrictions / rank</th><th>df</th><th>Status</th></tr></thead><tbody id="joint-readout"></tbody></table></div></section>
<details id="pairwise-panel" hidden><summary>Sector-to-sector differences · both samples</summary><p class="notes">Each contrast uses the joint slope covariance. Holm p-values adjust within the reported comparison family separately for each model and sample. Significance here uses Holm p-values; individual-sector effects above remain pointwise.</p><div class="scroll"><table><thead><tr><th>Difference</th><th>All eligible · β (SE)</th><th>Raw p / Holm p / significance</th><th>Exclude 2020–2022 · β (SE)</th><th>Raw p / Holm p / significance</th></tr></thead><tbody id="pairwise-readout"></tbody></table></div></details>
<details id="weather-panel" hidden><summary>Own-series weather coefficients · both samples</summary><p class="notes">Separate time-series robustness uses each series’ own parameter count in its HAC finite-sample correction. These controls accompany its temperature slope.</p><div class="scroll"><table><thead><tr><th>Sector / weather regressor</th><th>All eligible · β (SE)</th><th>p / significance</th><th>Exclude 2020–2022 · β (SE)</th><th>p / significance</th></tr></thead><tbody id="weather-readout"></tbody></table></div></details>
<details><summary>How the samples and inference are paired</summary><p>Growth rates, YoY bases and temperature lags are built from the full underlying calendar before endpoint years are removed. A YoY observation dated2023 can retain a2022 base period; lagged weather in2023 can also come from2022. This is an endpoint-year exclusion, not a claim that every retained observation is pandemic-free.</p><p>The full-control baseline retains100months or29quarters during2017–2025, falling to66months/19quarters after exclusion. Long-history robustness is also paired, with its own dates and controls. Pre-2020-only specifications already contain no COVID dates and therefore have identical paired estimates, clearly marked. New industry-specific p-values still need the stated short-sample and stationarity cautions.</p><p>Clustering allows dependence across time within a sector; it does not cover arbitrary common shocks across sectors. Few-cluster and short-time inference remain approximate. Calendar-HAC alternatives and both bandwidths are retained in the downloads. Gapped exact-window unit-root screens are unavailable, not failed; full-history evidence does not clear a gapped subset. <a href="stationarity.html">Full stationarity audit →</a></p><p>All fitted coefficients use PyFixest. The new paired runs have isolated native estimator logs; the original source results are preserved and checked against source fingerprints.</p></details><div class="downloads">{downloads}</div><footer><a href="heat_economy.html">Complete analysis</a> · <a href="granular_va.html">Detailed VA</a> · <a href="manufacturing_detail.html">Manufacturing</a> · <a href="sector_heterogeneity.html">Sector interactions</a></footer></main>'''
    for a,b in [('dated2023','dated 2023'),('a2022','a 2022'),('in2023','in 2023'),('from2022','from 2022'),('retains100','retains 100'),('or29','or 29'),('during2017','during 2017'),('to66','to 66')]:body=body.replace(a,b)
    script=(ROOT/'templates/covid_comparison.js').read_text()
    path=ROOT/'outputs/covid_comparison.html';path.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full and no-COVID samples · Singapore heat and economy</title><style>'+style+'</style></head><body>'+body+'<script id="covid-data" type="application/json">'+payload+'</script><script>'+script+'</script></body></html>')
    print(f'Wrote paired COVID comparison: {len(specs)} native two-sample tables; {path.stat().st_size:,} bytes')
    return path

if __name__=='__main__':build_covid_report()
