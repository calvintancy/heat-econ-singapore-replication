import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import crypto from 'node:crypto';

const root = path.resolve(import.meta.dirname, '..');
const reportPath = path.join(root, 'outputs/visitor_arrivals.html');
const source = fs.readFileSync(reportPath, 'utf8');
const payload = source.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = [...source.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
const data = JSON.parse(payload);
const ids = [...source.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
assert.equal(new Set(ids).size, ids.length, 'Duplicate HTML IDs');
assert(!/__([A-Z][A-Z_0-9]+)__/.test(source), 'Unfilled report placeholder');
assert(!/<script[^>]+src=|<link[^>]+stylesheet/i.test(source), 'External runtime dependency');
const elements = new Map(ids.map(id => [id, {innerHTML: '', textContent: '', value: '', classList: {add(){},remove(){}}}]));
elements.get('report-data').textContent = payload;
const selectors = new Map();
for (const panel of ['aggregate','region','country']) {
  for (const [name,value] of [['outcome','mom'],['bandwidth','12'],['profile-sample','full_clean']]) {
    selectors.set(`.${name}[data-panel="${panel}"]`, {value});
  }
}
for (const [id,value] of [['robust-panel','aggregate'],['robust-sample','full_clean'],['robust-bandwidth','12']]) elements.get(id).value=value;
for (const [id,value] of [['crisis-panel','aggregate'],['crisis-outcome','mom'],['crisis-window','long_all'],['crisis-inference','hac12']]) elements.get(id).value=value;
const document = {
  getElementById(id) {assert(elements.has(id), `Missing target ${id}`); return elements.get(id);},
  querySelector(selector) {assert(selectors.has(selector), `Unknown selector ${selector}`); return selectors.get(selector);},
  querySelectorAll() {return [];}
};
const context = vm.createContext({document, console});
vm.runInContext(script, context, {timeout: 5000});
const f = (value,n=3) => Number(value).toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n});
const panels=['aggregate','region','country'];
const outcomes=['mom','yoy','log_level','raw'];
const samples=['long_all','long_pre2020','long_clean','full_all','full_clean'];
let pooledViews=0, profileViews=0, robustnessViews=0, crisisViews=0;
const notBroken = value => assert(!/undefined|NaN|Infinity/.test(value), 'Invalid rendered value');
for (const panel of panels) for (const outcome of outcomes) for (const mode of panel==='aggregate'?[12,24]:[12,24,'origin']) {
  const bandwidth=mode==='origin'?12:mode;
  selectors.get(`.outcome[data-panel="${panel}"]`).value=outcome;
  selectors.get(`.bandwidth[data-panel="${panel}"]`).value=String(mode);
  context.panel=panel;
  vm.runInContext('drawPanel(panel)',context);
  const table=elements.get(`${panel}-table`).innerHTML;
  notBroken(table);
  assert(table.includes('class="gt_table"'));
  for (const sample of samples) {
    const base=data.results.find(r=>r.panel===panel&&r.outcome===outcome&&r.sample===sample&&r.variant==='main'&&r.hac_lags===bandwidth);
    const r=mode==='origin'?{...base,...data.clusterResults.find(c=>c.source==='core'&&c.model_id===base.model_id)}:base;
    assert(r);
    assert(table.includes(f(r.coefficient)+'<br>('+f(r.std_error)+')'));
    assert(table.includes(f(r.n_months,0)) && table.includes(f(r.n_panel_obs,0)));
    assert(elements.get(`${panel}-context`).innerHTML.includes('['+f(r.lower_0p1c)+', '+f(r.upper_0p1c)+']'));
  }
  pooledViews++;
  if(panel!=='aggregate' && ['mom','yoy'].includes(outcome)) {
    for (const sample of ['full_clean','long_pre2020']) {
      selectors.get(`.profile-sample[data-panel="${panel}"]`).value=sample;
      vm.runInContext('drawPanel(panel)',context);
      const profile=elements.get(`${panel}-profile-table`).innerHTML;
      const expected=data.profiles.filter(r=>r.panel===panel&&r.outcome===outcome&&r.sample===sample&&r.hac_lags===bandwidth);
      assert(profile.includes('class="gt_table"'));
      assert(profile.includes('Temperature') && profile.includes('Holm p'));
      notBroken(profile);
      for(const row of expected) assert(profile.includes(f(row.coefficient)) && profile.includes(f(row.std_error)));
      const joint=data.joint.find(r=>r.group===`${panel}__${outcome}__${sample}`&&r.hac_lags===bandwidth);
      if(joint.covariance_rank<joint.restrictions) assert(elements.get(`${panel}-profile-test`).innerHTML.includes('unavailable: singular covariance'));
      if(mode==='origin')assert(elements.get(`${panel}-profile-test`).innerHTML.includes('one origin cluster'));
      profileViews++;
    }
  }
}
for(const panel of panels)for(const outcome of outcomes)for(const sample of ['long_all','full_all'])for(const method of panel==='aggregate'?['hac12','hac24']:['hac12','hac24','origin']) {
  elements.get('crisis-panel').value=panel;elements.get('crisis-outcome').value=outcome;
  elements.get('crisis-window').value=sample;elements.get('crisis-inference').value=method;
  vm.runInContext('drawCrisis()',context);
  const full=elements.get('crisis-gt').innerHTML,summary=elements.get('crisis-summary').innerHTML;
  notBroken(full);notBroken(summary);assert(full.includes('class="gt_table"'));
  assert.equal((summary.match(/<tr>/g)||[]).length,6);
  for(const policy of ['none','covid','gfc_covid','buffered','stable_pre2020']) {
    const base=data.crisisResults.find(r=>r.panel===panel&&r.outcome===outcome&&r.sample===sample&&r.variant==='crisis_'+policy&&r.hac_lags===(method==='hac24'?24:12));
    const r=method==='origin'?{...base,...data.clusterResults.find(c=>c.source==='crisis'&&c.model_id===base.model_id)}:base;
    assert(full.includes(f(r.coefficient)+'<br>('+f(r.std_error)+')'));
    assert(summary.includes(f(r.effect_0p1c))&&summary.includes(f(r.n_months,0)));
  }
  crisisViews++;
}
for(const panel of panels) for(const sample of ['full_clean','long_pre2020','long_clean']) for(const bw of [12,24]) {
  elements.get('robust-panel').value=panel;
  elements.get('robust-sample').value=sample;
  elements.get('robust-bandwidth').value=String(bw);
  vm.runInContext('drawRobust()',context);
  for(const id of ['robust-table','dynamic-table','loo-table','ppml-table','robust-full-gt']) notBroken(elements.get(id).innerHTML);
  assert(elements.get('robust-full-gt').innerHTML.includes('class="gt_table"'));
  assert(elements.get('robust-table').innerHTML.includes('Growth in arrivals per day'));
  if(sample==='full_clean') assert(elements.get('robust-table').innerHTML.includes('80% coverage screen'));
  const ppml=elements.get('ppml-table').innerHTML;
  assert.equal((ppml.match(/<tr>/g)||[]).length,5);
  assert(ppml.includes('all observed dates') && !ppml.includes('log-admissible'));
  const expected=data.ppml.filter(r=>r.panel===panel&&r.hac_lags===bw);
  for(const row of expected) assert(ppml.includes(f(row.coefficient)) && ppml.includes(f(row.zero_cells_retained,0)));
  robustnessViews++;
}
const images=[...source.matchAll(/<img[^>]+src="data:image\/png;base64,([^"]+)"/g)];
assert.equal(images.length,6);
for(const image of images) assert.equal(Buffer.from(image[1],'base64').subarray(1,4).toString(),'PNG');
let downloads=0;
for(const match of source.matchAll(/download="([^"]+)" href="data:[^,]+;base64,([^"]+)" data-evidence-path="([^"]+)"/g)) {
  const bytes=Buffer.from(match[2],'base64');
  const target=path.join(root,match[3]);
  assert(target,`No saved evidence for download ${match[1]}`);
  assert(bytes.equals(fs.readFileSync(target)),`Download differs: ${match[1]}`);
  downloads++;
}
assert(downloads>=30);
for(const [key,html] of Object.entries(data.gt))assert(html.includes('gt_'+key)&&html.includes('class="gt_table"'));
const receipt={status:'passed',pooled_selector_views:pooledViews,profile_selector_views:profileViews,crisis_selector_views:crisisViews,robustness_selector_views:robustnessViews,native_gt_tables:Object.keys(data.gt).length,embedded_pngs:images.length,verified_evidence_downloads:downloads,report_sha256:crypto.createHash('sha256').update(source).digest('hex'),scope:'Source/runtime rendering and all selector combinations in an isolated JavaScript DOM harness; visual Safari review recorded separately. Numerical estimates verified by independent Python verifiers.'};
fs.writeFileSync(path.join(root,'outputs/visitor_arrivals/report_verification.json'),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt,null,2));
