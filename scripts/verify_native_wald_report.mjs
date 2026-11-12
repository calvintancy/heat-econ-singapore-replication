import fs from 'node:fs';
import vm from 'node:vm';
import zlib from 'node:zlib';
import assert from 'node:assert/strict';
const [htmlPath,outPath]=process.argv.slice(2);
const fail=error=>{fs.writeFileSync(outPath,JSON.stringify({status:'failed',error:String(error)})+'\n');console.error(error);process.exitCode=1;};process.on('uncaughtException',fail);process.on('unhandledRejection',fail);fs.writeFileSync(outPath,JSON.stringify({status:'running',claim:'No completed Node runtime claim'})+'\n');
const source=fs.readFileSync(htmlPath,'utf8');
const encoded=source.match(/<script type="application\/json" id="nw-data">([\s\S]*?)<\/script>/)[1];
const code=[...source.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).find(s=>s.includes('window.__nativeWaldReady=initialize'));
assert(code);assert(source.includes('Nominal 5%?'));
const elements=new Map();
class Element{
 constructor(id){this.id=id;this.innerHTML='';this.textContent='';this.hidden=false;this.value='';this.checked=false;this.disabled=false;this.listeners=new Map();this.dataset={};}
 addEventListener(name,fn){if(!this.listeners.has(name))this.listeners.set(name,[]);this.listeners.get(name).push(fn);}
 async fire(name){for(const fn of this.listeners.get(name)||[])await fn({target:this});}
 querySelectorAll(selector){if(selector!=='[data-nw-id]')return[];return [...this.innerHTML.matchAll(/data-nw-id="([^"]+)"/g)].map(m=>{const e=new Element('button');e.dataset.nwId=m[1];return e;});}
 click(){return this.fire('click');}
}
const E=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};E('nw-data').textContent=encoded;
const document={getElementById:E,createElement:()=>new Element('download')};
const window={};
const context=vm.createContext({document,window,console,atob,btoa,Uint8Array,Blob,Response,DecompressionStream,CompressionStream,ReadableStream,TextEncoder,TextDecoder,URL,setTimeout,clearTimeout,history:{pushState(){throw Error('No sandbox history writes');},replaceState(){throw Error('No sandbox history writes');}}});
vm.runInContext(code,context,{filename:'native_wald_widget.js'});
await window.__nativeWaldReady;
const ui=window.__nativeWald;assert(ui.ready);assert(ui.rows.length>0);
const count=ui.rows.length;
assert(E('nw-table').innerHTML.includes('Nominal native χ² p'));
assert(!E('nw-table').innerHTML.includes('<th>Historical F comparison</th>'));
assert(E('nw-active').textContent.startsWith('Active joint-test selections'));
let selectorViews=0,details=0;
for(const field of ui.fields){
 ui.reset();const row=ui.rows.find(r=>r[field]);
 E('nw-'+field).value=row[field];await E('nw-'+field).fire('change');
 const chosen=ui.selection(),expected=ui.rows.filter(r=>r[field]===row[field]).length;
 assert.equal(chosen.length,expected);assert(E('nw-count').textContent.includes(expected.toLocaleString('en-US')));
 assert(chosen.every(r=>r[field]===row[field]));assert(!/>(?:NaN|undefined)(?:<|$)/.test(E('nw-table').innerHTML));selectorViews++;
}
ui.reset();assert(!E('nw-next').disabled);await E('nw-next').fire('click');assert(E('nw-page').textContent.startsWith('Rows '+(ui.state().pageSize+1)+'–'));await E('nw-prev').fire('click');assert(E('nw-page').textContent.startsWith('Rows 1–'));
E('nw-show-legacy').checked=true;await E('nw-show-legacy').fire('change');assert(E('nw-table').innerHTML.includes('Historical F comparison'));
E('nw-show-legacy').checked=false;await E('nw-show-legacy').fire('change');
const picked=[];
for(const value of new Set(ui.rows.map(r=>r.family))){const candidates=ui.rows.filter(r=>r.family===value);picked.push(candidates.find(r=>r.p!==null)||candidates[0]);const unavailable=candidates.find(r=>r.p===null);if(unavailable)picked.push(unavailable);}
for(const row of picked){
 ui.inspect(row.test_id);assert(!E('nw-detail').hidden);assert(E('nw-detail').innerHTML.includes(row.test_id));
 assert(E('nw-detail').innerHTML.includes(row.p===null?'Unavailable':ui.probability(row.p).replace('<','&lt;')));
 assert.equal(row.nominal_5pct,row.p===null?'Unavailable':row.p<.05?'Yes':'No');const markup=ui.rowMarkup(row);assert(markup.includes('>'+row.nominal_5pct+'</td>'));assert(markup.includes(row.test_id));assert(markup.includes(row.p===null?'Unavailable':ui.probability(row.p).replace('<','&lt;')));assert(!/>(?:NaN|undefined)(?:<|$)/.test(markup),row.test_id);details++;
}
const target=ui.rows.find(r=>r.p!==null);ui.reset();for(const key of ui.fields)ui.choose(key,target[key]);
const csv=ui.selectedCSV();assert(csv.startsWith('test_id,source_csv,source_row,'));assert(csv.includes(target.test_id));
const blob=await ui.selectedGzip();const restored=zlib.gunzipSync(Buffer.from(await blob.arrayBuffer())).toString('utf8');assert.equal(restored,csv);
let conditionalShapes=0;
for(const phrase of ['level-channel shape','equal-growth coefficient shape','growth-sum span']){
 const row=ui.rows.find(r=>r.hypothesis.startsWith('Conditional direct-weather')&&r.hypothesis.includes(phrase));assert(row,phrase);
 ui.reset();for(const key of ui.fields)ui.choose(key,row[key]);
 assert(E('nw-hypothesis').innerHTML.includes(row.hypothesis));assert(ui.rowMarkup(row).includes(row.hypothesis));
 ui.inspect(row.test_id);assert(E('nw-detail').innerHTML.includes(row.hypothesis));
 assert(row.metadata.includes('original_test_name = '));assert(ui.selectedCSV().includes(row.hypothesis));conditionalShapes++;
}
const output={status:'passed',engine:process.version,execution:'Actual current widget JavaScript in Node VM with browser-control doubles; not a Chrome visual claim',rows_loaded:count,selector_fields_exercised:selectorViews,representative_family_available_unavailable_readouts:details,conditional_shape_surfaces_exercised:conditionalShapes,pagination:true,legacy_comparison_toggle:true,independent_active_selection_label:true,streamed_selected_csv_gzip:true,sandbox_history_guard:true,no_unrendered_numeric_tokens:true};
fs.writeFileSync(outPath,JSON.stringify(output,null,2)+'\n');console.log(JSON.stringify(output));
