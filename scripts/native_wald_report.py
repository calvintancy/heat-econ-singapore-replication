"""Presentation-only native-Wald overlay over immutable predecessor reports.

No estimator, covariance, sample, legacy CSV or old standalone HTML is changed.
The master is staged; integration into outputs/heat_economy.html is separate.
"""
from __future__ import annotations
import argparse,ast,base64,csv,gzip,hashlib,html,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from lxml import html as lh

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/native_wald'
TEMPLATE=ROOT/'templates/native_wald.html'
# Exact final-source routes, not native-stage archives or coefficient p values.
CHAPTER_SOURCE_RULES={
 'complete-crisis':('outputs/crisis_sensitivity/joint_tests.csv','outputs/crisis_sensitivity/annual/joint_tests.csv','outputs/complete_va_controls/joint_tests.csv'),
 'literature-models':('outputs/literature_sensitivity/tests.csv','outputs/literature_sensitivity/two_channel_span_tests.csv','outputs/literature_sensitivity/annual/tests.csv','outputs/literature_sensitivity/annual_seasonal/tests.csv','outputs/literature_sensitivity/full_controls/tests.csv','outputs/literature_sensitivity/full_controls/two_channel_span_tests.csv'),
 'referee-audit':('*',),
 'new-stationarity':(),
 'stata-check':('outputs/stata_key_results/joint_tests.csv',),
 'visitor-rigour':('outputs/visitor_rigour/monthly/tests.csv','outputs/visitor_rigour/annual/tests.csv','outputs/visitor_rigour/ppml/tests.csv','outputs/visitor_rigour/origin_year/tests.csv','outputs/stata_key_results/arrivals/joint_tests.csv'),
 'covid':('outputs/covid_comparison/sector/joint_tests.csv',),
 'annual':('outputs/annual/joint_tests.csv',),
 'aggregate':('outputs/aggregate_va/yoy_weather/temperature_difference_restriction',),
 'sector-va':('outputs/sector_slopes.csv',),
 'granular-va':('outputs/granular_interactions/va-detailed/',),
 'iip':('outputs/iip/monthly/sector_slopes.csv','outputs/iip/quarterly/sector_slopes.csv'),
 'manufacturing':('outputs/granular_interactions/iip-monthly-detailed/','outputs/granular_interactions/iip-quarterly-detailed/'),
 'visitors':('outputs/visitor_arrivals/joint_tests.json',),
 'growth-levels':('outputs/outcomes/va/yoy_weather/temperature_difference_restriction','outputs/outcomes/iip-monthly/yoy_weather/temperature_difference_restriction','outputs/outcomes/iip-quarterly/yoy_weather/temperature_difference_restriction'),
 'wind-pm':(),
 'heterogeneity':('outputs/heterogeneity/',),
 'cluster-sensitivity':('outputs/clustered_sector_sensitivity/joint_tests.csv',),
 'tables':(), 'reconciliation':(), 'stationarity':(),
}
COLLAPSE={
 'complete-crisis':['#joint'], 'literature-models':['#tests'],
 'annual':['#joint'], 'visitor-rigour':['#joint'],
 'covid':['#joint-panel'], 'granular-va':['.joint-readout'],
 'manufacturing':['.joint-readout'], 'visitors':['#region-test','#country-test'],
 'cluster-sensitivity':['#qcs-joint','#joint'],
}
LEGACY_NOTE=('Coefficient estimates, standard errors, intervals and coefficient t-test p-values in this preserved chapter are unchanged. '
 'Its joint-test p-values and F references are historical comparisons, not the current primary native χ² tests above. '
 'The joint-test selector above is independent of every coefficient selector below.')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(value):
 if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [clean(v) for v in value]
 if isinstance(value,np.generic):value=value.item()
 if isinstance(value,float) and not np.isfinite(value):return None
 return value

def first(row,*names,default=None):
 for name in names:
  value=row.get(name)
  if value is not None and not (isinstance(value,float) and np.isnan(value)) and str(value)!='':return value
 return default

def text(row,*names,default=''):
 value=first(row,*names,default=default)
 return str(value) if value is not None else default

def numeric(row,*names):
 value=first(row,*names)
 try:return float(value) if value is not None and np.isfinite(float(value)) else None
 except (ValueError,TypeError):return None

def family_label(source):
 if source.startswith('outputs/visitor_rigour/'):return 'Arrivals · '+source.split('/')[2].replace('_',' ')
 if source.startswith('outputs/visitor_arrivals/'):return 'Arrivals · original profiles'
 if source.startswith('outputs/literature_sensitivity/'):
  part=source.split('/')[2]
  return {'annual':'Literature · annual','annual_seasonal':'Literature · annual seasonal','full_controls':'Literature · wind / PM baseline'}.get(part,'Literature · subannual long history')
 if source.startswith('outputs/stata_key_results/'):return 'Stata · arrivals' if '/arrivals/' in source else 'Stata · heat / production'
 if source.startswith('outputs/heterogeneity/'):return 'Legacy · sector heterogeneity'
 if source.startswith('outputs/granular_interactions/'):return 'Legacy · detailed sector interactions'
 if source.startswith('outputs/complete_va_controls/'):return 'Complete GVA · five weather controls'
 if source.startswith('outputs/crisis_sensitivity/'):return 'Crisis sensitivity · annual' if '/annual/' in source else 'Crisis sensitivity · subannual'
 if source.startswith('outputs/covid_comparison/'):return 'Paired COVID sensitivity'
 if source.startswith('outputs/annual/'):return 'Legacy · annual analyses'
 if source.startswith('outputs/aggregate_va/'):return 'Aggregate VA · temperature restriction'
 if source.startswith('outputs/outcomes/'):return 'Growth / levels · temperature restriction'
 if source.startswith('outputs/clustered_sector_sensitivity/'):return 'Sector-CR1 interaction sensitivity'
 return 'Additional explicit restriction audit'

def display_hypothesis(row,raw):
 """Qualify a saved coefficient restriction without changing its null or p."""
 specifications=[text(row,key) for key in ['model_spec','specification']]
 dynamic=any(value.lower()=='dynamic' or value.upper().endswith('AR1') for value in specifications)
 lower=raw.lower()
 if dynamic:
  if 'static' in lower or 'level-channel' in lower:
   return 'Conditional direct-weather level-channel shape (growth lags held fixed; not an exact structural static-level test)'
  if 'equal' in lower and 'growth' in lower:
   return 'Conditional direct-weather equal-growth coefficient shape (growth lags held fixed)'
  if 'span' in lower and ('endpoint' in lower or 'growth' in lower):
   return 'Conditional direct-weather response in endpoint-difference plus growth-sum span (growth lags held fixed)'
 if 'two_channel' in specifications and 'static' in lower:
  return raw+' · same null as cumulative direct response zero'
 if 'DJO1' in specifications and 'static' in lower:
  return raw+' · same null as temperature coefficient sum zero'
 if any(value in {'benchmark','DJO0'} for value in specifications) and ('sum zero' in lower or 'cumulative direct' in lower):
  return raw+' · same null as the current temperature coefficient'
 return raw

def normalize(row):
 source=text(row,'source_csv','source_file').replace('\\','/')
 if source.startswith(str(ROOT)+'/'):source=source[len(str(ROOT))+1:]
 original_source=text(row,'original_source_csv','coefficient_source_csv','source_coefficient_csv').replace('\\','/')
 if original_source.startswith(str(ROOT)+'/'):original_source=original_source[len(str(ROOT))+1:]
 group=text(row,'group');parts=group.split('__')
 context=text(row,'context_label','context','panel',default=parts[0] if parts else '')
 if not context and source.startswith('outputs/heterogeneity/'):context=source.split('/')[2]
 if not context:context='Recorded source definition'
 outcome=text(row,'outcome_label','outcome',default=parts[1] if len(parts)>1 else 'Recorded outcome')
 adjustment=text(row,'output_adjustment','adjustment')
 if adjustment and adjustment not in outcome:outcome+=' · '+adjustment
 detail=[]
 for key,label in [('model_spec','Weather'),('specification','Weather'),('model','Model'),('structure','Model'),('history','History'),('control_set','Controls'),('window','Window')]:
  value=text(row,key)
  if value and not any(item.endswith(': '+value) for item in detail):detail.append(label+': '+value)
 if not detail:detail=[text(row,'source_model','spec_id','sample_id',default='Recorded equation')]
 sample=text(row,'sample_variant','sample',default=parts[2] if len(parts)>2 else text(row,'window',default='Recorded estimation sample'))
 cov=text(row,'covariance','method',default='Recorded joint covariance');band=numeric(row,'hac_lags','bandwidth')
 covariance=cov+((' · '+str(int(band))+' calendar lags') if band and band>0 else '')
 unit=text(row,'sector','unit','sector_id','unit_id',default='All sectors / origins in the null')
 raw_null=text(row,'test','null','hypothesis',default='Recorded explicit linear restriction')
 null=display_hypothesis(row,raw_null)
 status=text(row,'native_status','status',default='unavailable: native result not supplied')
 p=numeric(row,'p_value_native_chi2','native_p_value','p_native_chi2','chi2_p_value')
 if p is None and text(row,'reference_distribution','distribution').lower() in {'chi2','chi-square','chi_squared','chisq'}:p=numeric(row,'p_value')
 W=numeric(row,'chi2_statistic','wald_chi2','native_wald_chi2');q=numeric(row,'chi2_df','restrictions','numerator_df');rank=numeric(row,'covariance_rank')
 if p is None or W is None or q is None:
  status='Unavailable · '+status if not status.lower().startswith('unavailable') else status
 else:
  assert not any(word in status.lower() for word in ['unavailable','singular','guarded','error']), (row.get('test_id'),status)
  assert q>0 and (rank is None or rank>=q), (row.get('test_id'),q,rank)
  status='Computable · calibration not established'
 caution=[]
 if q and q>1:caution.append('Multi-restriction nominal diagnostic')
 if numeric(row,'time_residual_df') is not None and numeric(row,'time_residual_df')<10:caution.append('Few residual time df')
 if numeric(row,'covariance_condition','condition_number') and numeric(row,'covariance_condition','condition_number')>1e8:caution.append('Ill-conditioned restriction covariance')
 if first(row,'alias_of','canonical_test_id') and text(row,'alias_of','canonical_test_id')!=text(row,'test_id'):caution.append('Repeated alias; not independent evidence')
 if not caution:caution.append('Reference-law assumptions remain')
 metadata=['original_test_name = '+raw_null]
 for key in ['spec_id','sample_id','pair_id','source_model','original_source_csv','audit_role','controls','controls_label','formula','formula_rhs','test_role','model_spec','model','structure','history','control_set','window','sample_variant','start','end','native_kernel','native_kernel_version','covariance_source','source_npz','source_row']:
  value=text(row,key)
  if value:metadata.append(key+' = '+value)
 return clean({'test_id':text(row,'test_id'),'source_csv':source,'original_source_csv':original_source,'source_row':first(row,'source_row',default=''),
  'family':family_label(source),'context':context,'outcome':outcome,'specification':' · '.join(detail),'sample':sample,'covariance':covariance,'unit':unit,'hypothesis':null,
  'n_obs':numeric(row,'n_obs','N'),'n_periods':numeric(row,'n_periods','n_months','n_quarters','T'),
  'n_parameters':numeric(row,'n_parameters','K'),'time_residual_df':numeric(row,'time_residual_df','residual_time_df','df_time_resid'),
  'q':q,'rank':rank,'W':W,'p':p,'nominal_5pct':'Unavailable' if p is None else 'Yes' if p<.05 else 'No','legacy_F':numeric(row,'legacy_f_statistic','f_statistic'),
  'legacy_df1':numeric(row,'legacy_df1','numerator_df','restrictions'),'legacy_df2':numeric(row,'legacy_df2','denominator_df','inference_df'),
  'legacy_p':numeric(row,'legacy_p_value_F','legacy_p_value','p_value_legacy_F'),
  'status':status,'caution':'; '.join(caution),'metadata':'; '.join(metadata),'raw_chapter_ids':text(row,'chapter_ids')})

def belongs(row,key):
 explicit=row.get('raw_chapter_ids','')
 if explicit:
  try:ids=json.loads(explicit)
  except json.JSONDecodeError:ids=re.split('[;,|]',explicit)
  if isinstance(ids,list) and key in ids:return True
 sources=[row['source_csv'],row.get('original_source_csv','')]
 for rule in CHAPTER_SOURCE_RULES[key]:
  if rule=='*':return True
  if rule.endswith('.csv') or rule.endswith('.json'):
   if any(source==rule for source in sources):return True
  elif any(source.startswith(rule) for source in sources):return True
 return False

def encode_payload(rows):
 raw=json.dumps({'rows':rows},ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
 return json.dumps({'gzip':base64.b64encode(gzip.compress(raw,mtime=0)).decode()},separators=(',',':'))

def widget(rows,label,calibration_html=''):
 return TEMPLATE.read_text().replace('__NW_CHAPTER_LABEL__',html.escape(label)).replace('__NW_DATA__',encode_payload(rows)).replace('__NW_CALIBRATION__',calibration_html)

def calibration_evidence():
 path=OUT/'calibration.json';boundary=OUT/'calibration_boundary_verification.json'
 record=json.loads(path.read_text());guard=json.loads(boundary.read_text())
 assert record['status']=='passed' and guard['status']=='passed'
 bindings={**record['source_artifacts_sha256'],**record['verified_artifacts_sha256']}
 for name,digest in bindings.items():assert sha(ROOT/name)==digest,('Calibration binding changed',name)
 helper=ROOT/'scripts/native_wald_calibration.py'
 assert guard['source_sha256']==sha(helper)
 bindings['scripts/native_wald_calibration.py']=sha(helper)
 bindings['scripts/native_wald.py']=sha(ROOT/'scripts/native_wald.py')
 binds={'receipt_sha256':sha(path),'boundary_receipt_sha256':sha(boundary),'sources_and_artifacts_sha256':bindings}
 rows=[]
 heat=record['heat_reference_only']
 for result in heat['results']:
  rows.append(('Heat · '+heat['design'],result['restrictions'],result['hac_lags'],result['native_chi2_reference_rejections_at_05'],result['requested_draws'],'Reference-only recalculation on archived W; no new native calls'))
 for design in record['arrivals_exact_native_kernel_replay']:
  for result in design['results']:
   rows.append(('Arrivals · '+design['spec_id'],design['restrictions'],result['hac_lags'],result['native_chi2_rejections_at_05'],result['requested_draws'],'Actual native-kernel replay; fixed recorded seed'))
 cells=''.join('<tr><td>'+html.escape(label)+'</td><td>'+str(q)+'</td><td>'+str(h)+'</td><td>'+str(hits)+' / '+str(reps)+'</td><td>'+html.escape(scope)+'</td></tr>' for label,q,h,hits,reps,scope in rows)
 downloads=[]
 for file,label in [(path,'Calibration receipt'),(boundary,'Proof lifecycle fixture')]+[(ROOT/name,Path(name).name) for name in record['verified_artifacts_sha256']]:
  data=base64.b64encode(gzip.compress(file.read_bytes(),mtime=0)).decode()
  downloads.append('<a href="data:application/gzip;base64,'+data+'" download="'+html.escape(file.name)+'.gz">'+html.escape(label)+' · gzip ↓</a>')
 block='<details><summary>Selected checks under the current χ² reference: the calibration problem remains</summary><p>Fixed-X independent Gaussian null errors, not the observed error law or empirical replacement p-values. Arrivals use actual native-kernel replay; heat changes only the reference on previously archived Wald statistics. No economic model is refitted. Seeds, array bindings and every draw are in the downloads.</p><div class="nw-scroll"><table><thead><tr><th>Selected design</th><th>q</th><th>Calendar bandwidth</th><th>Nominal 5% χ² rejections</th><th>Scope</th></tr></thead><tbody>'+cells+'</tbody></table></div><p>'+' · '.join(downloads)+'</p></details>'
 return binds,block

def extract_master(path):
 source=path.read_text();match=re.search(r'(<script\b[^>]*\bid="collection-data"[^>]*>)(.*?)(</script>)',source,re.S)
 assert match,'No collection-data payload in predecessor master'
 prefix=source[:match.start(2)];suffix=source[match.end(2):]
 data=json.loads(match.group(2));return prefix,data,suffix

def historical_qualifier(doc,key):
 main=doc.find('.//main')
 if main is None:main=doc.find('body')
 assert main is not None
 legacy=lh.Element('details',{'class':'nw-legacy-chapter','open':'open','id':'nw-legacy-chapter'})
 summary=lh.Element('summary');summary.text='Preserved coefficient tables and historical joint comparisons';legacy.append(summary)
 for child in list(main):main.remove(child);legacy.append(child)
 notice=lh.Element('div',{'class':'nw-legacy-notice'});notice.text=LEGACY_NOTE
 main.append(notice);main.append(legacy)
 nodes=[]
 for selector in COLLAPSE.get(key,[]):
  if selector.startswith('#'):nodes.extend(legacy.xpath('.//*[@id=$value]',value=selector[1:]))
  else:nodes.extend(legacy.xpath('.//*[contains(concat(" ",normalize-space(@class)," "),$value)]',value=' '+selector[1:]+' '))
 # Static heterogeneity joint tables have no dedicated ID.
 if key=='heterogeneity':
  for table in legacy.xpath('.//table'):
   headings=' '.join(table.xpath('.//thead//text()'))
   if 'Restrictions' in headings and 'Covariance rank' in headings:nodes.append(table)
 wrapped=0
 for node in nodes:
  if any('nw-historical-joints' in parent.get('class','') for parent in node.iterancestors()):continue
  parent=node.getparent()
  if parent is None:continue
  details=lh.Element('details',{'class':'nw-historical-joints'});s=lh.Element('summary');s.text='Historical joint-test comparison · original F reference';details.append(s)
  parent.replace(node,details);details.append(node);wrapped+=1
 marked=0
 for row in legacy.xpath('.//tr'):
  text_value=' '.join(row.itertext()).lower()
  if ('joint heterogeneity' in text_value or 'joint equality' in text_value) and ('p' in text_value) and row.xpath('./td|./th'):
   row.set('class',(row.get('class','')+' nw-historical-joint-row').strip());cell=row.xpath('./td|./th')[0]
   badge=lh.Element('span',{'class':'nw-historical-badge'});badge.text='Historical F comparison';cell.insert(0,badge);marked+=1
 return main,wrapped,marked

BRIDGE="""<script>(()=>{if(parent===window)return;const channel='heat-economy-collection',key='native-wald';const send=()=>parent.postMessage({channel,key,type:'height',height:document.documentElement.scrollHeight},'*');new ResizeObserver(send).observe(document.documentElement);window.addEventListener('message',event=>{if(event.data?.channel!==channel)return;if(event.data.type==='print')window.print();send();});document.addEventListener('click',event=>{const a=event.target.closest('[data-report-target]');if(a){event.preventDefault();parent.postMessage({channel,key,type:'navigate',target:a.dataset.reportTarget,hash:a.dataset.reportAnchor||''},'*');}});parent.postMessage({channel,key,type:'ready'},'*');send();})();</script>"""

def current_copy(embedded,key,label,rows):
 doc=lh.fromstring(embedded,parser=lh.HTMLParser(huge_tree=True))
 main,wrapped,marked=historical_qualifier(doc,key)
 fragment=lh.fragment_fromstring(widget(rows,label),parser=lh.HTMLParser(huge_tree=True));main.insert(0,fragment)
 doc.find('head').append(lh.fragment_fromstring('<meta name="native-wald-view" content="current-native-chi2-with-preserved-custom-covariance">'))
 return '<!doctype html>\n'+lh.tostring(doc,encoding='unicode'),wrapped,marked

def build_native_wald_report():
 OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads((OUT/'predecessor_manifest.json').read_text())
 assert set(manifest['reports'])==set(CHAPTER_SOURCE_RULES)
 for key,record in manifest['reports'].items():
  assert sha(OUT/'predecessor_reports'/record['filename'])==record['sha256']
  assert sha(ROOT/record['source'])==record['sha256'],'Old standalone report changed'
 assert sha(OUT/'predecessor_master.html')==manifest['master']['sha256']
 source=OUT/'tests.csv'
 header=pd.read_csv(source,nrows=0).columns
 assert 'test_id' in header and 'p_value_native_chi2' in header, 'Explicit native chi-square identity/p fields required'
 wanted={'test_id','audit_role','test_role','controls','controls_label','formula','formula_rhs','source_csv','source_file','original_source_csv','coefficient_source_csv','source_coefficient_csv','source_row','context_label','context','panel','group','outcome_label','outcome','output_adjustment','adjustment','model_spec','specification','model','structure','history','control_set','window','source_model','spec_id','sample_id','pair_id','sample_variant','sample','covariance','method','hac_lags','bandwidth','sector','unit','sector_id','unit_id','test','null','hypothesis','native_status','status','p_value_native_chi2','native_p_value','p_native_chi2','chi2_p_value','reference_distribution','distribution','p_value','chi2_statistic','wald_chi2','native_wald_chi2','chi2_df','restrictions','numerator_df','covariance_rank','time_residual_df','residual_time_df','df_time_resid','covariance_condition','condition_number','alias_of','canonical_test_id','start','end','native_kernel','native_kernel_version','covariance_source','source_npz','n_obs','N','n_periods','n_months','n_quarters','T','n_parameters','K','legacy_f_statistic','f_statistic','legacy_df1','legacy_df2','denominator_df','inference_df','legacy_p_value_F','legacy_p_value','p_value_legacy_F','chapter_ids'}
 calibration,calibration_html=calibration_evidence()
 rows=[];seen=set()
 for chunk in pd.read_csv(source,usecols=lambda col:col in wanted,low_memory=False,dtype={'test_id':str},float_precision='round_trip',chunksize=5000):
  for raw in chunk.to_dict('records'):
   row=normalize(raw)
   assert row['test_id'] and row['test_id'] not in seen
   seen.add(row['test_id']);rows.append(row)
 prefix,data,suffix=extract_master(OUT/'predecessor_master.html')
 assert set(data['reports'])==set(manifest['reports'])
 output=OUT/'reports';output.mkdir(exist_ok=True)
 records={};coverage=[]
 for key,record in manifest['reports'].items():
  selected=[row for row in rows if belongs(row,key)]
  updated,wrapped,marked=current_copy(data['reports'][key],key,record['label'],selected)
  target=output/record['filename'];target.write_text(updated)
  data['reports'][key]=updated
  records[key]={'filename':record['filename'],'source_predecessor_sha256':record['sha256'],'current_html_sha256':sha(target),
    'rows':len(selected),'sources':sorted({row['source_csv'] for row in selected}),'historical_containers_collapsed':wrapped,'historical_table_rows_marked':marked}
  coverage.extend({'chapter_id':key,'test_id':row['test_id'],'source_csv':row['source_csv']} for row in selected)
 index='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Current native Wald inference</title><style>body{margin:0;background:#f3f7fa}main{max-width:1400px;margin:auto;padding:28px}</style></head><body><main>'+widget(rows,'All analysis families',calibration_html).replace('../current_master.html#native-wald','current_master.html#native-wald')+'</main>'+BRIDGE+'</body></html>'
 (OUT/'native_wald.html').write_text(index);data['reports']['native-wald']=index
 button='<button type="button" role="tab" id="tab-native-wald" aria-controls="panel-native-wald" aria-selected="false" data-tab="native-wald">Native Wald · current</button>'
 prefix,n=re.subn(r'(<button\b[^>]*data-tab="overview"[^>]*>.*?</button>)',lambda m:m.group(1)+button,prefix,count=1,flags=re.S);assert n==1
 panel='<section role="tabpanel" aria-labelledby="tab-native-wald" id="panel-native-wald" class="chapter" hidden><iframe title="Current native Wald inference" id="frame-native-wald" data-report="native-wald" sandbox="allow-scripts allow-downloads allow-modals allow-popups allow-popups-to-escape-sandbox" scrolling="no"></iframe></section>'
 prefix,n=re.subn(r'(<section\b[^>]*id="panel-methods"[^>]*>)',panel+r'\1<div class="nw-legacy-notice"><strong>Current joint-test convention:</strong> the Native Wald tab and chapter widgets use the native PyFixest χ² kernel with preserved custom covariance. Coefficient t-tests are unchanged. Source notes below retain the earlier F convention where stated; they are historical descriptions of that comparison. Changing the reference does not establish calibration.</div>',prefix,count=1);assert n==1
 payload=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
 (OUT/'current_master.html').write_text(prefix+payload+suffix)
 pd.DataFrame(coverage).to_csv(OUT/'chapter_coverage.csv',index=False)
 result={'status':'rendered_pending_independent_presentation_verification','staged_only':True,'coefficient_t_tests_changed':False,'native_model_fits':0,
  'calibration':calibration,'tests_csv_sha256':sha(source),'collector_rows':len(rows),'reports':records,'source_rules':CHAPTER_SOURCE_RULES,
  'native_index_sha256':sha(OUT/'native_wald.html'),'current_master_sha256':sha(OUT/'current_master.html'),
  'predecessor_manifest_sha256':sha(OUT/'predecessor_manifest.json'),'predecessor_master_sha256':manifest['master']['sha256'],
  'builder_source_sha256':sha(Path(__file__)),'template_sha256':sha(TEMPLATE),'master_embedded_reports':len(data['reports']),
  'claim':'Current native test-kernel presentation over unchanged coefficients, samples and custom covariance; legacy F is separately historical; calibration unresolved'}
 (OUT/'report_manifest.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ['status','collector_rows','master_embedded_reports','current_master_sha256']},indent=2),flush=True)
 return result

if __name__=='__main__':
 argparse.ArgumentParser(description=__doc__).parse_args();build_native_wald_report()
