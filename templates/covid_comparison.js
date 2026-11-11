const D=JSON.parse(document.getElementById('covid-data').textContent);
const E=id=>document.getElementById(id);
const safe=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=x=>x==null?'—':Number(x).toFixed(3).replace('-','−');
function badge(row,field='p_value') {
  return `<span class="badge ${safe(row[field+'_class'])}">${safe(row[field+'_significance'])}</span>`;
}
function pcell(row,holm=false) {
  return `${safe(row.p_value_display)}${holm?' / '+safe(row.p_value_holm_display):''}<br>${badge(row,holm?'p_value_holm':'p_value')}`;
}
function estimate(row){return `${num(row.coefficient)} (${num(row.std_error)})`;}
function pairedRows(rows,key,renderer) {
  const all=rows.filter(r=>r.sample_variant==='all_eligible');
  const lookup=new Map(rows.filter(r=>r.sample_variant==='exclude_2020_2022').map(r=>[key(r),r]));
  return all.map(r=>{
    const other=lookup.get(key(r));
    if(!other)throw Error('Missing paired result: '+key(r));
    return renderer(r,other);
  }).join('');
}
function choices(prefer) {
  const pool=D.specs.filter(s=>s.engine===E('engine').value);
  const options=[...new Set(pool.map(s=>s.analysis))];
  const old=prefer?.analysis||E('analysis').value;
  E('analysis').innerHTML=options.map(a=>`<option value="${safe(a)}">${safe(a)}</option>`).join('');
  if(options.includes(old))E('analysis').value=old;
  specs(prefer);
}
function specs(prefer) {
  const pool=D.specs.filter(s=>s.engine===E('engine').value&&s.analysis===E('analysis').value);
  E('spec').innerHTML=pool.map(s=>`<option value="${s.pair_id}">${safe(s.source_model+' · '+s.outcome_label+' · '+s.controls+' · '+s.all_start+'–'+s.all_end+' · '+s.family)}</option>`).join('');
  if(prefer&&pool.some(s=>s.pair_id===prefer.pair_id))E('spec').value=prefer.pair_id;
  render();
}
function render() {
  const selected=E('spec').value,s=D.specs.find(s=>s.pair_id===selected);
  if(!s)return;
  const rows=D.rows[selected],key=s.engine==='sector'?'sector_id':'term';
  E('spec-title').textContent=s.analysis+' · '+s.outcome_label;
  E('spec-details').textContent=s.source_model+' · '+s.family+' · '+s.controls+' · '+s.inference;
  const h=s.frequency==='monthly'?12:4,short=s.frequency==='monthly'?'MoM':'QoQ',isYoY=['y_yoy','y_yoy_nsa'].includes(s.outcome),adjustment=s.outcome.endsWith('_nsa')?'NSA':'SA';
  E('outcome-definition').textContent=s.outcome==='y_log_level'?'Y = 100 × ln(output). A coefficient of −1 is approximately 1% lower output per +1°C, conditional on controls and FE; this level sensitivity remains descriptive.':s.outcome==='y_raw_level'?'Y = output level: real VA in chained-2015 S$ million, or IIP index points (2025=100). Coefficients use those level units per +1°C. This level sensitivity remains descriptive.':['y_yoy_pct','y_yoy_pct_nsa'].includes(s.outcome)?'Y = 100 × [output(t)/output(t−h) − 1], h='+h+'. This is exact YoY percentage growth.':`Y = 100 × [ln output(t) − ln output(t−${isYoY?h:1})], ${isYoY?'YoY':short} log growth (${adjustment}), not annualised. This follows the Dell–Jones–Olken log-difference outcome form, adapted to our frequency. A coefficient of −1 is approximately one percentage point lower growth per +1°C, conditional on controls and FE.`;
  E('native-table').innerHTML=D.tables[selected];
  E('formula').textContent=s.formula;
  E('sample-cards').innerHTML=`<div class="sample-card"><span>All eligible dates</span><strong>${s.all_periods} ${s.frequency==='monthly'?'months':'quarters'}</strong><span class="sub">${safe(s.all_start)}–${safe(s.all_end)}</span></div><div class="sample-card"><span>Exclude 2020–2022</span><strong>${s.excluded_periods} ${s.frequency==='monthly'?'months':'quarters'}</strong><span class="sub">${safe(s.excluded_start)}–${safe(s.excluded_end)}</span></div>`;
  E('sample-warning').textContent=s.n_removed_periods===0?'Identical pair: this specification already contains no eligible observations dated 2020–2022.':`${s.n_removed_periods} eligible weather periods removed. Controls, outcome definition and FE are unchanged. Exact-window stationarity tests on the resulting calendar gaps are unavailable; this is not a failed unit-root test.`;
  const decomposition=r=>r.temp_main!=null&&s.source_model!=='R1'&&s.family!=='separate_series'?`<span class="sub">βT ${num(r.temp_main)} + δᵢ ${num(r.interaction_delta)}${r.delta_is_normalization?' · reference δ normalized':''}</span>`:'';
  E('readout').innerHTML=pairedRows(rows,r=>r[key],(r,t)=>`<tr><td>${safe(s.engine==='sector'?r.sector:r.term)}</td><td>${estimate(r)}${decomposition(r)}</td><td>${pcell(r)}</td><td>${estimate(t)}${decomposition(t)}</td><td>${pcell(t)}</td><td>${num(r.mean_y)} / ${num(t.mean_y)}</td></tr>`);
  const joint=D.joint_tests[selected]||[];
  E('joint-panel').hidden=!joint.length;
  E('joint-readout').innerHTML=joint.map(r=>`<tr><td>${r.sample_variant==='all_eligible'?'All eligible':'Exclude 2020–2022'}</td><td>${num(r.f_statistic)}</td><td>${pcell(r)}</td><td>${r.restrictions} / ${r.covariance_rank}</td><td>${r.numerator_df??'—'}, ${r.denominator_df??'—'}</td><td>${safe(r.status)}${r.status==='available'?'<span class="sub">Approximate finite-sample inference</span>':''}</td></tr>`).join('');
  const pairs=D.pairwise_tests[selected]||[];
  E('pairwise-panel').hidden=!pairs.length;
  E('pairwise-readout').innerHTML=pairedRows(pairs,r=>r.first_sector_id+'|'+r.second_sector_id,(r,t)=>`<tr><td>${safe(r.comparison)}</td><td>${estimate(r)}</td><td>${pcell(r,true)}</td><td>${estimate(t)}</td><td>${pcell(t,true)}</td></tr>`);
  const weather=(D.weather_coefficients[selected]||[]).filter(r=>r.term!=='temp_c');
  E('weather-panel').hidden=!weather.length;
  E('weather-readout').innerHTML=pairedRows(weather,r=>r.sector_id+'|'+r.term,(r,t)=>`<tr><td>${safe(r.sector)}<span class="sub">${safe(r.term)}</span></td><td>${estimate(r)}</td><td>${pcell(r)}</td><td>${estimate(t)}</td><td>${pcell(t)}</td></tr>`);
}
E('engine').addEventListener('change',()=>choices());
E('analysis').addEventListener('change',()=>specs());
E('spec').addEventListener('change',render);
const initial=D.specs.find(s=>s.pair_id===D.default);
E('engine').value=initial.engine;
choices(initial);
