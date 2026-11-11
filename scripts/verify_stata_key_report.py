"""Execute the actual embedded Stata viewer JavaScript and all downloads."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stata_key_results'
NODE=r'''
const fs=require('fs'),vm=require('vm');
const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),nodes=new Map(),blobs=new Map();
function node(id){if(!nodes.has(id)){let html='';const n={id,value:id==='view'?'main':'',textContent:'',dataset:{},
 get innerHTML(){return html},set innerHTML(v){html=v;if(['context','outcome','control_set','structure'].includes(id)){const opts=[...v.matchAll(/<option>(.*?)<\/option>/g)].map(m=>m[1]);n.value=opts[0]||''}},
 click(){},href:'',download:''};nodes.set(id,n)}return nodes.get(id)}
const document={getElementById:node,createElement:()=>({href:'',download:'',click(){}})};
const URLproxy={createObjectURL(b){const id='blob:'+blobs.size;blobs.set(id,b);return id},revokeObjectURL(){}};
const ctx=vm.createContext({document,Blob,Response,DecompressionStream,Uint8Array,atob,URL:URLproxy,setTimeout(){},console});
vm.runInContext(input.script,ctx);
(async()=>{for(let i=0;i<1000;i++){if(vm.runInContext('typeof data!=="undefined"&&!!data',ctx)&&node('rows').innerHTML)break;await new Promise(r=>setTimeout(r,5))}
const data=vm.runInContext('data',ctx);if(!data)throw Error('Embedded async data did not initialize');
const unique=new Map();data.specs.forEach(s=>unique.set([s.context,s.outcome,s.control_set,s.structure].join('|'),s));let selections=0,downloads=0;
for(const s of unique.values())for(const view of ['main','native','calendar_h1','calendar_h2','project_cluster']){
 for(const k of ['context','outcome','control_set','structure'])node(k).value=s[k];node('view').value=view;
 vm.runInContext('draw()',ctx);const html=node('rows').innerHTML;
 if(!html||!node('selection').textContent.includes(s.context))throw Error('Selection failed');
 if(s.structure==='H1'&&view==='main'&&!html.includes('Base '))throw Error('H1 baseline+delta omitted');
 if(s.structure==='H1'&&view==='native'&&(!html.includes('unavailable')||html.includes('***')))throw Error('Invalid native H1 inference shown');
 if(s.structure==='H1'&&view==='project_cluster'&&html.includes('***'))throw Error('Industry-cluster sensitivity given misleading stars');
 if(!node('native').innerHTML.includes('table'))throw Error('Native esttab table omitted');selections++;
}
for(const [name,packed]of Object.entries(data.downloads)){await node('downloads').onclick({target:{dataset:{file:name}}});
 const blob=[...blobs.values()].at(-1),text=await blob.text();if(text!==input.downloads[name])throw Error('Download differs:'+name);downloads++}
console.log(JSON.stringify({selections,downloads,native_tables:Object.keys(data.native).length,targets:data.specs.length}));
})().catch(e=>{console.error(e.stack);process.exitCode=1});
'''


def verify_stata_key_report():
    assert json.loads((OUT/'verification.json').read_text())['status']=='passed'
    path=ROOT/'outputs/stata_key_results.html';html=path.read_text()
    scripts=re.findall(r'<script>(.*?)</script>',html,re.S);assert len(scripts)==1
    names=['estimates.csv','control_coefficients.csv','joint_tests.csv','specifications.csv','singleton_sample_comparison.csv','vce_convention_comparison.csv','stata_python_coverage.csv','verification.json','do/key_results.do','do/postprocess.mata.do','native/master_stata_output.txt','mcp/environment.json','mcp/requirements.txt']
    payload={'script':scripts[0],'downloads':{name:(OUT/name).read_text() for name in names}}
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary)/'payload.json';p.write_text(json.dumps(payload))
        result=subprocess.run(['node','-e',NODE,str(p)],capture_output=True,text=True,check=True)
    actual=json.loads(result.stdout.strip());assert actual=={'selections':255,'downloads':13,'native_tables':51,'targets':99}
    report={'status':'passed','scope':'Actual embedded JavaScript in Node with DOM/Blob/download emulation; not a native-browser screenshot claim',
        **actual,'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'numerical_receipt_sha256':hashlib.sha256((OUT/'verification.json').read_bytes()).hexdigest()}
    (OUT/'report_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Stata viewer passed:255 selector/method states and13 exact embedded downloads.',flush=True)
    return report


if __name__=='__main__':verify_stata_key_report()
