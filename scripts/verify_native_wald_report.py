"""Source-backed verification of the staged native-Wald presentation only."""
from __future__ import annotations
import argparse,ast,base64,gzip,hashlib,json,re,subprocess,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from lxml import html as lh
from scipy.stats import chi2
from presentation_verification_state import presentation_run

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/native_wald';PROOF=OUT/'report_verification.json'

def sha(path):
 with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def payload(source):
 match=re.search(r'<script type="application/json" id="nw-data">(.*?)</script>',source,re.S);assert match
 encoded=json.loads(match.group(1));return json.loads(gzip.decompress(base64.b64decode(encoded['gzip'])))['rows']
def rowhash(row):return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def num(value):
 try:return float(value) if value is not None and np.isfinite(float(value)) else None
 except (ValueError,TypeError):return None

def expected_hypothesis(raw):
 def field(key):
  value=raw.get(key)
  return '' if value is None or isinstance(value,float) and np.isnan(value) else str(value)
 name=next((field(k) for k in ['test','null','hypothesis'] if field(k)),'Recorded explicit linear restriction')
 specs={field('model_spec'),field('specification')};low=name.lower()
 if any(s.lower()=='dynamic' or s.upper().endswith('AR1') for s in specs):
  if 'static' in low or 'level-channel' in low:return name,'Conditional direct-weather level-channel shape (growth lags held fixed; not an exact structural static-level test)'
  if 'equal' in low and 'growth' in low:return name,'Conditional direct-weather equal-growth coefficient shape (growth lags held fixed)'
  if 'span' in low and ('endpoint' in low or 'growth' in low):return name,'Conditional direct-weather response in endpoint-difference plus growth-sum span (growth lags held fixed)'
 if 'static' in low and 'two_channel' in specs:return name,name+' · same null as cumulative direct response zero'
 if 'static' in low and 'DJO1' in specs:return name,name+' · same null as temperature coefficient sum zero'
 if specs&{'benchmark','DJO0'} and ('sum zero' in low or 'cumulative direct' in low):return name,name+' · same null as the current temperature coefficient'
 return name,name

def extract_master(path):
 source=path.read_text();match=re.search(r'(<script[^>]*id="collection-data"[^>]*>)(.*?)(</script>)',source,re.S);assert match
 return source[:match.start(2)],json.loads(match.group(2)),source[match.end(2):]

def route_matches(row,key,rules):
 explicit=row.get('raw_chapter_ids','')
 if explicit:
  try:ids=json.loads(explicit)
  except json.JSONDecodeError:ids=re.split('[;,|]',explicit)
  if isinstance(ids,list) and key in ids:return True
 route=rules[key]
 if '*' in route:return True
 paths=[row.get('source_csv',''),row.get('original_source_csv','')]
 return any(path==pattern if pattern.endswith(('.csv','.json')) else path.startswith(pattern) for path in paths for pattern in route)

def _verify_native_wald_report(target,prototype):
 OUT.mkdir(exist_ok=True);entry={}
 def pin(path):
  path=Path(path);name=str(path.relative_to(ROOT));digest=sha(path)
  if name in entry:assert entry[name]==digest,('Artifact changed before its read',name)
  entry[name]=digest;return digest
 initial=[Path(__file__),ROOT/'scripts/native_wald_report.py',ROOT/'templates/native_wald.html',ROOT/'scripts/verify_native_wald_report.mjs',ROOT/'scripts/presentation_verification_state.py',ROOT/'scripts/verify_native_wald_analysis.py']
 initial += [OUT/name for name in ['report_manifest.json','predecessor_manifest.json','verification.json','tests.csv','native_wald.html','current_master.html','predecessor_master.html','chapter_coverage.csv','calibration.json','calibration_boundary_verification.json']]
 for path in initial:pin(path)
 try:
  builder_source=(ROOT/'scripts/native_wald_report.py').read_text();assert hashlib.sha256(builder_source.encode()).hexdigest()==entry['scripts/native_wald_report.py']
  tree=ast.parse(builder_source);rules=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHAPTER_SOURCE_RULES' for t in node.targets))
  manifest=json.loads((OUT/'report_manifest.json').read_text());pred=json.loads((OUT/'predecessor_manifest.json').read_text())
  assert manifest['builder_source_sha256']==sha(ROOT/'scripts/native_wald_report.py')
  assert manifest['template_sha256']==sha(ROOT/'templates/native_wald.html')
  assert manifest['tests_csv_sha256']==sha(OUT/'tests.csv')
  numeric=json.loads((OUT/'verification.json').read_text());assert numeric['status']=='passed'
  numerical_source_current=numeric['verifier_source_sha256']==entry['scripts/verify_native_wald_analysis.py']
  assert numerical_source_current or prototype, 'Current numerical verifier source differs from the prior prototype receipt; use --prototype only for a separately labelled UI trial'
  for name in numeric['files_sha256']:pin(OUT/name)
  for key,info in pred['reports'].items():
   pin(ROOT/info['source']);pin(OUT/'predecessor_reports'/info['filename']);pin(OUT/'reports'/info['filename'])
  assert numeric['files_sha256']['tests.csv']==manifest['tests_csv_sha256']
  assert sha(OUT/'predecessor_master.html')==pred['master']['sha256']==manifest['predecessor_master_sha256']
  for key,info in pred['reports'].items():
   assert sha(ROOT/info['source'])==info['sha256']
   assert sha(OUT/'predecessor_reports'/info['filename'])==info['sha256']
  calibration=manifest['calibration']
  assert sha(OUT/'calibration.json')==calibration['receipt_sha256']
  assert sha(OUT/'calibration_boundary_verification.json')==calibration['boundary_receipt_sha256']
  for name,digest in calibration['sources_and_artifacts_sha256'].items():assert pin(ROOT/name)==digest, name
  assert json.loads((OUT/'calibration_boundary_verification.json').read_text())['status']=='passed'
  index=(OUT/'native_wald.html').read_text();rows=payload(index);catalog={row['test_id']:row for row in rows}
  assert len(catalog)==len(rows)==manifest['collector_rows']==numeric['test_rows']
  # Independently bind displayed primitives to the explicit native/legacy CSV
  # columns; never infer a native p from the historical source p.
  counts=available=unavailable=qualified=aliases=0
  needed={'test_id','source_csv','original_source_csv','source_row','p_value_native_chi2','chi2_statistic','wald_chi2','chi2_df','restrictions','covariance_rank','legacy_f_statistic','legacy_df1','legacy_df2','legacy_p_value_F'}
  needed|={'test','null','hypothesis','model_spec','specification'}
  for chunk in pd.read_csv(OUT/'tests.csv',usecols=lambda c:c in needed,dtype={'test_id':str},float_precision='round_trip',chunksize=5000,low_memory=False):
   for raw in chunk.to_dict('records'):
    row=catalog[raw['test_id']];assert row['source_csv']==raw['source_csv']
    original,expected_label=expected_hypothesis(raw)
    assert row['hypothesis']==expected_label,(raw['test_id'],'Hypothesis qualification',row['hypothesis'],expected_label)
    assert 'original_test_name = '+original in row['metadata']
    qualified+=int(expected_label.startswith('Conditional direct-weather'))
    aliases+=int('same null as' in expected_label)
    for displayed,source in [('p','p_value_native_chi2'),('W','chi2_statistic'),('q','chi2_df'),('rank','covariance_rank'),('legacy_F','legacy_f_statistic'),('legacy_df1','legacy_df1'),('legacy_df2','legacy_df2'),('legacy_p','legacy_p_value_F')]:
     expected=num(raw.get(source))
     if displayed=='W' and expected is None:expected=num(raw.get('wald_chi2'))
     if displayed=='q' and expected is None:expected=num(raw.get('restrictions'))
     assert row[displayed]==expected,(raw['test_id'],displayed,row[displayed],expected)
    assert row['nominal_5pct']==('Unavailable' if row['p'] is None else 'Yes' if row['p']<.05 else 'No')
    if row['p'] is None:unavailable+=1;assert row['status'].startswith('Unavailable') or row['status'].lower().startswith('unavailable')
    else:
     available+=1;assert row['q']>0 and row['W']>=0
     assert abs(row['p']-float(chi2.sf(row['W'],row['q'])))<2e-12
     assert row['status']=='Computable · calibration not established'
    counts+=1
  assert (available,unavailable)==(numeric['available_native_tests'],numeric['unavailable_tests_retained'])
  assert counts==len(rows)
  expected_hashes={key:rowhash(row) for key,row in catalog.items()}
  # Only source identity and declared chapter-route policy are needed after
  # the global primitive comparison, reducing memory during large HTML reads.
  expected_ids={key:{row['test_id'] for row in rows if route_matches(row,key,rules)} for key in pred['reports']}
  del catalog,rows
  oldprefix,olddata,oldsuffix=extract_master(OUT/'predecessor_master.html')
  currentprefix,currentdata,currentsuffix=extract_master(OUT/'current_master.html')
  assert oldsuffix==currentsuffix,'Original collection navigation runtime changed'
  assert currentdata['overview']==olddata['overview'] and currentdata['complete_overview']==olddata['complete_overview']
  assert set(currentdata['reports'])==set(olddata['reports'])|{'native-wald'}
  assert currentdata['reports']['native-wald']==index
  assert 'id="tab-native-wald"' in currentprefix and 'id="panel-native-wald"' in currentprefix
  checked=static_rows=old_scripts=0;route_counts={}
  for key,info in pred['reports'].items():
   path=OUT/'reports'/info['filename'];source=path.read_text()
   assert sha(path)==manifest['reports'][key]['current_html_sha256']
   assert source==currentdata['reports'][key]
   selected=payload(source);ids={row['test_id'] for row in selected}
   assert ids==expected_ids[key] and len(ids)==len(selected),(key,len(ids),len(expected_ids[key]))
   assert all(rowhash(row)==expected_hashes[row['test_id']] for row in selected)
   current=lh.fromstring(source,parser=lh.HTMLParser(huge_tree=True));old=lh.fromstring(olddata['reports'][key],parser=lh.HTMLParser(huge_tree=True))
   main=current.find('.//main')
   if main is None:main=current.find('body')
   assert main[0].get('id')=='nw-root','Native panel must precede old chapter content'
   assert current.xpath('//*[@id="nw-title"]')[0].text_content()=='Current native Wald inference'
   assert 'own selections' in current.xpath('//*[@id="nw-root"]')[0].text_content()
   assert 'Native test kernel does not mean native default covariance' in source
   legacy=current.xpath('//*[@id="nw-legacy-chapter"]')[0]
   assert legacy.tag=='details' and 'open' in legacy.attrib
   assert current.xpath('//*[contains(@class,"nw-legacy-notice")]')
   # Original numerical script payloads/runtime are preserved exactly.
   oldscript=[(node.get('id'),node.text or '') for node in old.xpath('//script')]
   newscripts=[(node.get('id'),node.text or '') for node in current.xpath('//script') if node.get('id')!='nw-data' and 'window.__nativeWaldReady=initialize' not in (node.text or '')]
   assert oldscript==newscripts,(key,'Old coefficient/script data changed')
   old_scripts+=len(oldscript)
   for node in legacy.xpath('.//*[contains(@class,"nw-historical-badge")]'):node.getparent().remove(node)
   text=lambda node:' '.join(' '.join(node.itertext()).split())
   oldrows=[text(row) for row in old.xpath('//tr')];newrows=[text(row) for row in legacy.xpath('.//tr')]
   assert oldrows==newrows,(key,'Static old coefficient/table values changed')
   static_rows+=len(oldrows)
   assert all('open' not in node.attrib for node in legacy.xpath('.//details[contains(@class,"nw-historical-joints")]'))
   if not selected:assert manifest['reports'][key]['rows']==0
   assert not current.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[not(starts-with(@src,"data:"))]')
   route_counts[key]=len(selected);checked+=1
  # Actual widget JavaScript is exercised once with the complete inventory;
  # every copied chapter uses the identical verified widget script/data schema.
  node_out=OUT/'browser_verification.json'
  run=subprocess.run(['node',str(ROOT/'scripts/verify_native_wald_report.mjs'),str(OUT/'native_wald.html'),str(node_out)],capture_output=True,text=True)
  assert run.returncode==0,(run.stdout[-2000:],run.stderr[-4000:])
  node=json.loads(node_out.read_text());assert node['status']=='passed' and node['rows_loaded']==counts
  generated={str(node_out.relative_to(ROOT)):sha(node_out)}
  for name,digest in entry.items():assert sha(ROOT/name)==digest,('Tested source/artifact changed during presentation verification',name)
  files={str(path.relative_to(OUT)):sha(path) for path in [OUT/'report_manifest.json',OUT/'current_master.html',OUT/'native_wald.html',OUT/'chapter_coverage.csv',node_out,*[OUT/'reports'/v['filename'] for v in pred['reports'].values()]]}
  result={'status':'passed','prototype_only':prototype,'numerical_verifier_source_matches_current':numerical_source_current,'verifier_source_sha256':entry[str(Path(__file__).relative_to(ROOT))], 'source_artifacts_sha256':entry,'generated_artifacts_sha256':generated,'browser_verifier_source_sha256':sha(ROOT/'scripts/verify_native_wald_report.mjs'),
   'test_rows_checked':counts,'available_native_tests':available,'unavailable_rows':unavailable,'chapter_copies_checked':checked,
   'conditional_direct_weather_shape_rows_checked':qualified,'algebraic_alias_rows_labelled':aliases,'original_test_names_preserved_in_export_metadata':True,
   'chapter_row_counts':route_counts,'static_legacy_table_rows_preserved':static_rows,'old_script_payloads_preserved':old_scripts,
   'old_standalones_unchanged':True,'predecessor_master_sha256':pred['master']['sha256'],'overview_coefficient_t_rows_unchanged':True,
   'native_current_fields_and_chi2_probabilities_source_backed':True,'calibration_bindings':calibration,'numerical_verification_sha256':sha(OUT/'verification.json'),
   'files_sha256':files,'browser':node,'staged_only':True,
   'scope':'Current native-Wald presentation/data routing, preserved coefficient t data and old source scripts, all21 chapter copies, offline embedding and actual generic widget Node runtime. No model refit, empirical calibration or Chrome visual claim.'}
 except Exception as error:
  target.write_text(json.dumps({'status':'failed','error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
 target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['status','test_rows_checked','chapter_copies_checked','static_legacy_table_rows_preserved']},indent=2));return result

def verify_native_wald_report(*,prototype=False):
 target=OUT/('report_trial_verification.json' if prototype else 'report_verification.json')
 with presentation_run(target,'Native-Wald presentation trial' if prototype else 'Current native-Wald source-backed presentation verification'):
  return _verify_native_wald_report(target,prototype)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prototype',action='store_true',help='Separate UI trial receipt may use a historical numerical-code generation; never current release readiness')
 verify_native_wald_report(prototype=parser.parse_args().prototype)
