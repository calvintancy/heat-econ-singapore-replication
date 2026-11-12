"""Actual isolated headless Google Chrome verification of native-Wald views.

Optional runtime: uv run --locked --with playwright==1.65.0 python this_script.py.
No global/user browser profile is opened. --staged writes a separate trial proof;
the default current-release proof requires the integrated and staged masters equal.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.metadata,json,traceback
from pathlib import Path
import pandas as pd
from presentation_verification_state import presentation_run

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/native_wald';BROWSER=OUT/'browser'

def sha(path):
 with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def _verify_native_wald_browser(proof,staged):
 BROWSER.mkdir(parents=True,exist_ok=True)
 entry={}
 def pin(path):
  path=Path(path);name=str(path.relative_to(ROOT));digest=sha(path)
  if name in entry:assert entry[name]==digest,('Input changed before its read',name)
  entry[name]=digest;return digest
 ui_path=OUT/('report_trial_verification.json' if staged else 'report_verification.json')
 initial=[Path(__file__),ROOT/'scripts/presentation_verification_state.py',ROOT/'scripts/native_wald_report.py',ROOT/'templates/native_wald.html',ROOT/'scripts/verify_native_wald_report.py',ROOT/'scripts/verify_native_wald_report.mjs',ROOT/'scripts/verify_native_wald_analysis.py',ROOT/'outputs/heat_economy.html',ui_path]
 initial += [OUT/name for name in ['report_manifest.json','native_wald.html','current_master.html','tests.csv','predecessor_manifest.json','verification.json','calibration.json','calibration_boundary_verification.json']]
 for path in initial:pin(path)
 try:
  assert importlib.metadata.version('playwright')=='1.65.0','Use the pinned optional Playwright1.65.0 runtime'
  from playwright.sync_api import sync_playwright
  manifest=json.loads((OUT/'report_manifest.json').read_text())
  ui=json.loads(ui_path.read_text());assert ui['status']=='passed'
  assert ui['verifier_source_sha256']==entry['scripts/verify_native_wald_report.py']
  if not staged:assert not ui.get('prototype_only') and ui['numerical_verifier_source_matches_current']
  for name,digest in ui['source_artifacts_sha256'].items():assert pin(ROOT/name)==digest,name
  for name,digest in ui['files_sha256'].items():assert pin(OUT/name)==digest,name
  for name,digest in ui.get('generated_artifacts_sha256',{}).items():assert pin(ROOT/name)==digest,name
  numeric=json.loads((OUT/'verification.json').read_text());assert numeric['status']=='passed'
  if not staged:assert numeric['verifier_source_sha256']==entry['scripts/verify_native_wald_analysis.py']
  for name,digest in numeric['files_sha256'].items():assert pin(OUT/name)==digest,name
  for name,digest in manifest['calibration']['sources_and_artifacts_sha256'].items():assert pin(ROOT/name)==digest,name
  assert ui['files_sha256']['current_master.html']==sha(OUT/'current_master.html')==manifest['current_master_sha256']
  target=OUT/'current_master.html' if staged else ROOT/'outputs/heat_economy.html'
  if not staged:assert sha(target)==sha(OUT/'current_master.html'),'Integrate the verified staged master before the current-master browser gate'
  routes=manifest['reports'];checks=[];screens=[];downloads=[];external=[];page_errors=[];native_tables={}
  def record(page,label):
   page.on('pageerror',lambda error:page_errors.append({'page':label,'error':str(error)}))
  def ready(page):
   page.wait_for_function('window.__nativeWald && window.__nativeWald.ready === true',timeout=180000)
   page.wait_for_load_state('networkidle',timeout=180000)
  def assert_no_overflow(page,label):
   size=page.evaluate('({viewport:innerWidth,body:document.documentElement.scrollWidth})')
   assert size['body']<=size['viewport']+2,(label,size)
   return size
  with sync_playwright() as p:
   browser=p.chromium.launch(channel='chrome',headless=True,args=['--allow-file-access-from-files'])
   version=browser.version
   context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
   def offline(route):external.append(route.request.url);route.abort()
   context.route('http://**/*',offline);context.route('https://**/*',offline)
   try:
    # Every copied chapter is opened in a real page, with its own original JS.
    for key,info in routes.items():
     page=context.new_page();record(page,key)
     page.goto((OUT/'reports'/info['filename']).as_uri(),wait_until='domcontentloaded',timeout=180000);ready(page)
     assert page.locator('#nw-title').inner_text()=='Current native Wald inference'
     count=page.evaluate('window.__nativeWald.rows.length');assert count==info['rows']
     if count:
      before=page.locator('#nw-legacy-chapter select').evaluate_all('(els)=>els.map(e=>({id:e.id,value:e.value}))')
      sample=page.evaluate('window.__nativeWald.rows[0].sample')
      page.select_option('#nw-sample',sample)
      assert page.evaluate('(value)=>window.__nativeWald.selection().every(r=>r.sample===value)',sample)
      after=page.locator('#nw-legacy-chapter select').evaluate_all('(els)=>els.map(e=>({id:e.id,value:e.value}))')
      assert before==after,(key,'Native selector changed an old coefficient selection')
      assert 'Nominal 5%?' in page.locator('#nw-table').inner_text()
      assert 'Active joint-test selections' in page.locator('#nw-active').inner_text()
     else:
      assert page.locator('#nw-loading').is_visible()
      assert 'No explicit Wald restrictions' in page.locator('#nw-loading').inner_text()
      assert page.locator('#nw-controls').is_hidden()
     assert page.locator('details.nw-historical-joints[open]').count()==0
     if key in {'stata-check','literature-models','visitor-rigour'}:
      selector='#nw-legacy-chapter table:visible' if key=='stata-check' else '#nw-legacy-chapter table.gt_table:visible'
      page.locator(selector).first.wait_for(state='visible',timeout=180000)
      text=page.locator(selector).first.inner_text();assert len(text)>100
      assert any(char.isdigit() for char in text)
      native_tables[key]={'kind':'Native Stata esttab' if key=='stata-check' else 'MakeTables/GT coefficient table','visible_table_characters':len(text)}
     checks.append({'chapter':key,'native_rows':count,'no_joint_note':not bool(count),'native_selectors_independent':True})
     page.close()
    # Global widget: real selectors, legacy toggle, unavailable, and download.
    page=context.new_page();record(page,'global-native-index')
    page.goto((OUT/'native_wald.html').as_uri(),wait_until='domcontentloaded',timeout=180000);ready(page)
    chosen=page.evaluate("window.__nativeWald.rows.find(r=>r.p!==null&&r.legacy_p!==null&&Math.abs(r.p-r.legacy_p)>1e-6&&r.q>1)")
    assert chosen
    for field in ['family','context','outcome','specification','sample','covariance','unit','hypothesis']:
     page.select_option('#nw-'+field,chosen[field])
    selected=page.evaluate('window.__nativeWald.selection().map(r=>r.test_id)');assert chosen['test_id'] in selected
    assert 'Historical F comparison' not in page.locator('#nw-table').inner_text()
    page.check('#nw-show-legacy');assert 'Historical F comparison' in page.locator('#nw-table').inner_text()
    page.evaluate('(id)=>window.__nativeWald.inspect(id)',chosen['test_id'])
    details=page.locator('#nw-detail').inner_text();assert 'Nominal 5%?' in details and 'Historical F p' in details
    assert chosen['nominal_5pct'] in details
    screenshot=BROWSER/'native_desktop.png';page.screenshot(path=str(screenshot),full_page=False);screens.append(screenshot)
    with page.expect_download(timeout=120000) as info:
     page.locator('#nw-download').click()
    download=info.value;path=BROWSER/'selected_current_tests.csv.gz';download.save_as(path);downloads.append(path)
    records=list(csv.DictReader(gzip.decompress(path.read_bytes()).decode().splitlines()))
    assert {row['test_id'] for row in records}==set(selected)
    expected={}
    for chunk in pd.read_csv(OUT/'tests.csv',usecols=['test_id','p_value_native_chi2','chi2_statistic','chi2_df','legacy_p_value_F'],dtype={'test_id':str},chunksize=5000,float_precision='round_trip'):
     part=chunk.loc[chunk.test_id.isin(selected)]
     expected.update({row['test_id']:row for row in part.to_dict('records')})
     if len(expected)==len(selected):break
    for row in records:
     raw=expected[row['test_id']]
     assert float(row['p'])==float(raw['p_value_native_chi2'])
     assert float(row['W'])==float(raw['chi2_statistic']) and float(row['q'])==float(raw['chi2_df'])
     assert row['nominal_5pct']==('Yes' if float(row['p'])<.05 else 'No')
    conditional_shapes=[]
    for phrase in ['level-channel shape','equal-growth coefficient shape','growth-sum span']:
     page.click('#nw-reset')
     shape=page.evaluate('(phrase)=>window.__nativeWald.rows.find(r=>r.hypothesis.startsWith("Conditional direct-weather")&&r.hypothesis.includes(phrase))',phrase)
     assert shape,phrase
     for field in ['family','context','outcome','specification','sample','covariance','unit','hypothesis']:page.select_option('#nw-'+field,shape[field])
     assert shape['hypothesis'] in page.locator('#nw-table').inner_text()
     assert page.locator('#nw-hypothesis option:checked').inner_text()==shape['hypothesis']
     page.evaluate('(id)=>window.__nativeWald.inspect(id)',shape['test_id'])
     assert page.locator('#nw-detail h3').inner_text()==shape['hypothesis']
     assert 'original_test_name = ' in shape['metadata']
     assert page.evaluate('(value)=>window.__nativeWald.selectedCSV().includes(value)',shape['hypothesis'])
     conditional_shapes.append(shape['test_id'])
    page.click('#nw-reset')
    unavailable=page.evaluate('window.__nativeWald.rows.find(r=>r.p===null)');assert unavailable
    for field in ['family','context','outcome','specification','sample','covariance','unit','hypothesis']:
     page.select_option('#nw-'+field,unavailable[field])
    page.evaluate('(id)=>window.__nativeWald.inspect(id)',unavailable['test_id'])
    assert 'Unavailable' in page.locator('#nw-detail').inner_text()
    assert unavailable['nominal_5pct']=='Unavailable'
    page.set_viewport_size({'width':390,'height':844});mobile=assert_no_overflow(page,'global-native-mobile')
    screenshot=BROWSER/'native_mobile.png';page.screenshot(path=str(screenshot),full_page=False);screens.append(screenshot);page.close()
    # Current/staged master: all24 tabs and sandboxed bridge navigation.
    page=context.new_page();record(page,'current-master')
    page.goto(target.as_uri(),wait_until='domcontentloaded',timeout=180000)
    page.get_by_role('tab',name='Native Wald · current',exact=True).wait_for(timeout=180000)
    assert page.locator('[role="tab"]').count()==24
    screenshot=BROWSER/'master_desktop.png';page.screenshot(path=str(screenshot),full_page=False);screens.append(screenshot)
    page.get_by_role('tab',name='Native Wald · current',exact=True).click()
    frame=page.locator('#frame-native-wald').element_handle().content_frame();assert frame
    frame.wait_for_function('window.__nativeWald && window.__nativeWald.ready',timeout=180000)
    assert frame.locator('#nw-title').is_visible()
    page.wait_for_function("parseInt(document.getElementById('frame-native-wald').style.height||'0')>=500",timeout=30000)
    embedded=[]
    for key in ['stata-check','aggregate']:
     page.locator(f'[data-tab="{key}"]').click()
     child=page.locator('#frame-'+key).element_handle().content_frame();assert child
     child.wait_for_function('window.__nativeWald && window.__nativeWald.ready',timeout=180000)
     assert child.locator('#nw-title').is_visible();embedded.append(key)
    child.locator('[data-report-target="native-wald"]').first.click()
    page.wait_for_function("document.getElementById('tab-native-wald').getAttribute('aria-selected')==='true'",timeout=30000)
    page.set_viewport_size({'width':390,'height':844});master_mobile=assert_no_overflow(page,'master-mobile')
    screenshot=BROWSER/'master_mobile.png';page.screenshot(path=str(screenshot),full_page=False);screens.append(screenshot)
    page.close()
   finally:context.close();browser.close()
  assert not external,external
  assert not page_errors,page_errors
  artifacts=[*screens,*downloads,OUT/'report_manifest.json',ui_path,OUT/'current_master.html',OUT/'native_wald.html']
  artifacts += [OUT/'reports'/info['filename'] for info in routes.values()]
  for name,digest in entry.items():assert sha(ROOT/name)==digest,('Tested artifact changed during actual Chrome verification',name)
  result={'status':'passed','scope':'Staged-only actualChrome trial, not integrated-master release readiness' if staged else 'Actual isolated headless Google Chrome on the integrated current master and all21 native overlay chapter copies',
   'staged_only':staged,'master_sha256':sha(ROOT/'outputs/heat_economy.html'),'staged_master_sha256':sha(OUT/'current_master.html'),
   'tested_master_sha256':entry[str(target.relative_to(ROOT))],'verifier_source_sha256':entry[str(Path(__file__).relative_to(ROOT))],'playwright_version':'1.65.0','browser_version':version,
   'chapters':checks,'chapter_count':len(checks),'global_inventory_rows':manifest['collector_rows'],'actual_download_rows':len(records),
   'native_legacy_toggle':True,'nominal_5pct_readouts':True,'conditional_shape_selector_table_detail_export_cases':conditional_shapes,'unavailable_readout':True,'preserved_native_tables':native_tables,
   'master_tabs':24,'embedded_chapters_exercised':embedded,'bridge_navigation_and_height':True,'mobile':mobile,'master_mobile':master_mobile,
   'network_requests':external,'page_errors':page_errors,'native_model_refits':0,
   'files_sha256':{str(path.relative_to(ROOT)):sha(path) for path in artifacts},
   'source_artifacts_sha256':entry,'entry_end_artifact_identity':True}
 except Exception as error:
  proof.write_text(json.dumps({'status':'failed','staged_only':staged,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
 proof.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({key:result[key] for key in ['status','scope','chapter_count','master_tabs','actual_download_rows','browser_version']},indent=2));return result

def verify_native_wald_browser(*,staged=False):
 proof=BROWSER/('staged_verification.json' if staged else 'verification.json')
 with presentation_run(proof,'Staged actual Chrome trial' if staged else 'Current integrated native-Wald actual Chrome verification'):
  return _verify_native_wald_browser(proof,staged)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--staged',action='store_true',help='Trial against staged master; writes staged_verification.json, never the current-release proof')
 verify_native_wald_browser(staged=parser.parse_args().staged)
