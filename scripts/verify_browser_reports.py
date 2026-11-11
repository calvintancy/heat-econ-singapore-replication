"""Optional real-Chrome offline UI smoke checks.

Run after numerical/presentation verification:
uv run --locked --with playwright==1.65.0 python scripts/verify_browser_reports.py
Requires installed Google Chrome; set HEAT_CHROME_EXECUTABLE on another platform.
Receipts bind the exact HTML generation, and do not certify numerical validity.
"""
import os
from pathlib import Path
import json
import sys
import hashlib
from playwright.sync_api import sync_playwright
from presentation_verification_state import presentation_run, persist

root=Path(__file__).resolve().parents[1]
out=root/'outputs/browser_verification';out.mkdir(exist_ok=True)
names=sys.argv[1:] or ['crisis_sensitivity','literature_sensitivity','extension_stationarity','stata_key_results','visitor_arrivals_rigour','clustered_sector_sensitivity_qualified','aea_audit','heat_economy']
result=[]
previous=json.loads((out/'verification.json').read_text()).get('checks',[]) if (out/'verification.json').exists() else []
with presentation_run(out/'verification.json','Optional real-Chrome offline UI checks'), sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path=os.environ.get('HEAT_CHROME_EXECUTABLE','/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))
    for name in names:
        source=root/f'outputs/{name}.html'
        source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        receipt=out/f'{name}-verification.json'
        receipt.write_text(json.dumps({'status':'running','report':name,'html_sha256':source_sha})+'\n')
        page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((root/f'outputs/{name}.html').as_uri(),wait_until='networkidle',timeout=60000)
        native_id='native-table' if name=='crisis_sensitivity' else 'native' if name=='literature_sensitivity' else None
        if native_id:
            page.wait_for_selector('#'+native_id+' table.gt_table',timeout=60000)
            page.screenshot(path=str(out/f'{name}-desktop.png'),full_page=False)
            assert page.locator('#'+native_id+' table.gt_table').count()==1,name
            assert page.locator('select').count()>=6
            page.locator('#'+native_id).screenshot(path=str(out/f'{name}-native-table.png'))
            page.locator('select').first.select_option(index=0)
            assert page.locator('#'+native_id+' table.gt_table').count()==1
            assert not errors,errors
            download_id='download-selected' if name=='crisis_sensitivity' else 'download'
            with page.expect_download() as downloading:
                page.locator('#'+download_id).click()
            target=out/(name+'-selected.csv');downloading.value.save_as(str(target));assert target.stat().st_size>100
        if name=='literature_sensitivity':
            for branch in ['subannual','full_controls','annual','annual_seasonal']:
                page.locator('#branch').select_option(branch)
                page.locator('#context').select_option('va20')
                page.locator('#outcome').select_option('y_yoy_nsa' if branch in ['subannual','full_controls'] else 'y_nsa')
                page.locator('#model-spec').select_option('dynamic' if branch=='full_controls' else 'seasonal_current_prior_AR1' if branch=='annual_seasonal' else 'DJO10' if branch=='annual' else 'distributed')
                page.locator('#model').select_option('H1')
                if branch in ['full_controls','annual_seasonal']:
                    assert page.locator('#native table.gt_table').count()==0,branch
                    reason=page.locator('#unavailable').inner_text()
                    assert 'unavailable' in reason.lower() and ('rank' in reason.lower() or 'degrees' in reason.lower()),reason
                page.locator('#model-spec').select_option('benchmark' if branch=='full_controls' else 'seasonal_current' if branch=='annual_seasonal' else 'BHM_unit_quadratic_trends' if branch=='annual' else 'distributed')
                assert page.locator('#native table.gt_table').count()==1,branch
                if branch in ['full_controls','annual','annual_seasonal']:
                    assert 'fragile' in page.locator('#active-caution').inner_text()
            page.locator('#branch').select_option('subannual')
            page.locator('#context').select_option('va20')
            page.locator('#outcome').select_option('y_yoy_nsa')
            page.locator('#model-spec').select_option('distributed')
            page.locator('#model').select_option('H1')
        if name=='extension_stationarity':
            page.wait_for_selector('#body tr',timeout=60000)
            assert page.locator('#body tr').count()==100
            assert '22306 diagnostics' in page.locator('#status').inner_text()
            page.locator('#state').select_option('unit-root-concern')
            assert '310 diagnostics' in page.locator('#status').inner_text()
            page.locator('#frequency').select_option('M')
            assert page.locator('#body tr').count()>0
            page.locator('#search').fill('temperature')
            page.locator('#search').fill('')
            page.locator('#state').select_option('unavailable-gaps')
            assert page.locator('#body tr').count()>0
            with page.expect_download() as downloading:
                page.locator('[data-download="tests.csv"]').click()
            target=out/'extension-stationarity-tests.csv'
            downloading.value.save_as(str(target))
            assert target.read_bytes()==(root/'outputs/extension_stationarity/tests.csv').read_bytes()
            page.locator('#state').select_option('')
            page.locator('#frequency').select_option('')
        if name=='stata_key_results':
            page.wait_for_selector('#context',timeout=60000)
            assert page.locator('#native table').count()>0
            page.locator('#context').select_option('va20')
            page.locator('#outcome').select_option('y_yoy_nsa')
            page.locator('#control_set').select_option('full90')
            page.locator('#structure').select_option('H1')
            page.locator('#view').select_option('native')
            assert page.locator('#rows tr').count()==20
            assert 'unavailable' in page.locator('#rows').inner_text()
            page.locator('#view').select_option('main')
            assert 'Base' in page.locator('#rows').inner_text()
            assert 'calendar' in page.locator('#rows').inner_text().lower()
            page.locator('summary',has_text='Reproducibility and downloads').click()
            with page.expect_download() as downloading:
                page.locator('[data-file$="specifications.csv"]').click()
            target=out/'stata-key-specifications.csv'
            downloading.value.save_as(str(target))
            assert target.read_bytes()==(root/'outputs/stata_key_results/specifications.csv').read_bytes()
        if name=='clustered_sector_sensitivity_qualified':
            page.wait_for_selector('#qcs-native table.gt_table',timeout=60000)
            assert page.locator('#qcs-native table.gt_table').count()==1
            page.locator('#qcs-context').select_option(index=0)
            page.locator('#qcs-window').select_option('full')
            assert page.locator('#qcs-readout tbody tr').count()>0
            assert 'CR1' not in page.locator('#qcs-readout thead').inner_text()
            names=page.evaluate('Object.keys(window.qualifiedSensitivityData.downloads)')
            archive=next(n for n in names if n.endswith('.html'))
            with page.expect_download() as downloading:
                page.locator('[data-raw="'+archive+'"]').click()
            target=out/'original-sector-cr1-archive.html'
            downloading.value.save_as(str(target))
            assert target.read_bytes()==(root/'outputs/clustered_sector_sensitivity.html').read_bytes()
        if name=='visitor_arrivals_rigour':
            page.wait_for_selector('#native table.gt_table',timeout=60000)
            assert page.locator('#model-counts').inner_text().startswith('Recorded new OLS fits: 299;')
            page.locator('#outcome').select_option('y_yoy_nsa')
            page.locator('#model_spec').select_option('dynamic')
            assert 'residual time df' in page.locator('#caution').inner_text()
            assert 'strict_growth_window_clean' in page.locator('#unavailable').inner_text()
            page.locator('#sample').select_option('strict_growth_window_clean')
            assert 'Unavailable' in page.locator('#effects').inner_text()
            page.locator('#branch').select_option('annual')
            assert page.locator('#native table.gt_table').count()==0
            assert 'rank' in page.locator('#unavailable').inner_text()
            page.locator('#branch').select_option('origin_year')
            assert page.locator('#native table.gt_table').count()==1
            assert 'origin × year FE' in page.locator('#equation').inner_text()
            page.locator('#branch').select_option('ppml')
            page.locator('#panel').select_option('complete52')
            assert page.locator('#native table.gt_table').count()==1
            with page.expect_download() as downloading:
                page.locator('#download-selected').click()
            target=out/'visitor-rigour-selected.csv'
            downloading.value.save_as(str(target));assert target.stat().st_size>200
            page.locator('[data-tab="sources"]').click()
            assert page.locator('#aggregation-checks tbody tr').count()==8
            page.locator('[data-tab="diagnostics"]').click()
            page.locator('#diagnostic-state').select_option('unit-root-concern')
            assert '38 diagnostics' in page.locator('#diagnostic-status').inner_text()
            with page.expect_download() as downloading:
                page.locator('[data-diag="tests.csv"]').click()
            target=out/'visitor-rigour-diagnostic-tests.csv'
            downloading.value.save_as(str(target))
            assert target.read_bytes()==(root/'outputs/visitor_rigour/diagnostics/tests.csv').read_bytes()
            page.locator('[data-tab="stata"]').click()
            assert page.locator('#stata-native table').count()>0
            assert page.locator('#stata-table option').count()==22
            page.locator('[data-tab="results"]').click()
            page.locator('#branch').select_option('monthly')
            page.locator('#panel').select_option('country')
            page.locator('#outcome').select_option('y_mom_nsa')
            page.locator('#control_set').select_option('full')
            page.locator('#model_spec').select_option('own_current')
            page.locator('#model').select_option('H1')
            page.locator('#method').select_option('Calendar HAC · 12 months')
            page.locator('#sample').select_option('all_eligible')
        if name=='heat_economy':
            assert page.locator('[data-tab]').count()==23
            assert page.locator('#paired-overview-rows tr').count()==9
            assert page.locator('#complete-overview-rows tr').count()==3
            for key, selector in [('literature-models','#native table.gt_table'),('visitor-rigour','#native table.gt_table'),('stata-check','#native table'),('cluster-sensitivity','#qcs-native table.gt_table')]:
                print('Checking embedded chapter: '+key,flush=True)
                page.locator('#tab-'+key).click()
                frame=page.frame_locator('#frame-'+key)
                if key=='stata-check':
                    frame.locator('details:has(#native):not([open]) > summary').click()
                frame.locator(selector).first.wait_for(timeout=90000)
                assert page.locator('#tab-'+key).get_attribute('aria-selected')=='true'
            visitor=page.frame_locator('#frame-visitor-rigour')
            page.locator('#tab-visitor-rigour').click()
            visitor.locator('[data-tab="stata"]').click()
            assert visitor.locator('#stata-table option').count()==22
            visitor.locator('[data-tab="results"]').click()
            with page.expect_download() as downloading:
                visitor.locator('#download-selected').click()
            target=out/'master-visitor-selected.csv';downloading.value.save_as(str(target))
            assert target.stat().st_size>200
            page.locator('#tab-referee-audit').click()
            audit=page.frame_locator('#frame-referee-audit')
            audit.locator('#arrivals').wait_for(timeout=60000)
            assert '137 Stata runs' in audit.locator('#arrivals').inner_text()
            # Exercise the generated cross-chapter bridge. The separate closure
            # verifier checks the historical filename-to-qualified-tab alias.
            audit.locator('body').evaluate("el=>{const a=document.createElement('a');a.id='legacy-alias-check';a.href='#cluster-sensitivity';a.dataset.reportTarget='cluster-sensitivity';a.textContent='legacy alias';el.appendChild(a);}")
            audit.locator('#legacy-alias-check').click()
            page.wait_for_function("document.getElementById('tab-cluster-sensitivity').getAttribute('aria-selected')==='true'")
            page.locator('#tab-overview').click()
            page.locator('[data-window="long"]').click()
            page.locator('[data-growth="yoy_nsa"]').click()
            assert 'NSA' in page.locator('#complete-overview-rows').inner_text()
        page.set_viewport_size({'width':430,'height':930})
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(out/f'{name}-mobile.png'),full_page=False)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile horizontal overflow: '+name
        assert not errors,errors
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
        entry={'report':name,'page_errors':errors,'title':page.title(),'desktop_mobile_screenshots':True,'html_sha256':source_sha}
        receipt.write_text(json.dumps({'status':'passed',**entry},indent=2)+'\n')
        result.append(entry)
        page.close()
    browser.close()
    merged={row['report']:row for row in previous if hashlib.sha256((root/'outputs'/(row['report']+'.html')).read_bytes()).hexdigest()==row['html_sha256']}
    merged.update({row['report']:row for row in result})
    persist(out/'verification.json',{'status':'passed','checks':list(merged.values()),'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
print(json.dumps(result,indent=2))
