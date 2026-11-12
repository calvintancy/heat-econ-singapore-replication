"""Integrity and current-release gates for the additive native-Wald revision.

The old 55-stage execution is pinned historical evidence. Native-Wald/current
presentation receipts are distinct; no old p-value equivalence is inferred.
"""
from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
import package_extensions as transport
from restore_replication import _safe_target

ROOT=Path(__file__).resolve().parents[1]
SOURCE_MANIFEST='replication/extensions/source_manifest_wald_v2.json'
PROVENANCE='replication/native_wald/baseline_provenance.json'
RECIPE='replication/native_wald/recipe.json'
RELEASE='v1.1.0-audit.2'


def sha(path): return transport.sha(path)

def read(path): return json.loads(Path(path).read_text())

def write_new(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as handle:handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def source_guard(root=ROOT,manifest=SOURCE_MANIFEST):
 root=Path(root).resolve();path=Path(manifest);path=path if path.is_absolute() else root/path
 return transport.check_source(root,path)


def baseline_guard(root=ROOT,*,hash_files=True):
 root=Path(root).resolve();policy=read(root/PROVENANCE)
 path=root/policy['baseline_manifest'];assert sha(path)==policy['baseline_manifest_sha256']
 baseline=read(path);old_seal=root/policy['predecessor_source_manifest']
 assert sha(old_seal)==policy['predecessor_source_manifest_sha256']==baseline['source_revision']['sha256']
 old_sources=read(old_seal)['sources_sha256'];assert len(old_sources)==policy['predecessor_source_count']==349
 archive=root/policy['source_preimages_archive'];assert sha(archive)==policy['source_preimages_archive_sha256']
 changed=[]
 with zipfile.ZipFile(archive) as z:
  for name,digest in old_sources.items():
   assert hashlib.sha256(z.read(name)).hexdigest()==digest,('Historical source preimage',name)
   current=_safe_target(root,name);assert current.is_file(),('Missing predecessor source',name)
   if sha(current)!=digest:
    assert name in policy['allowed_changed_predecessor_sources'],('Unapproved predecessor source change',name)
    changed.append(name)
 source_set=set(old_sources);checked=0;overrides={};groups={}
 for entry in baseline['files']:
  name=entry['path'];groups[name]=entry['groups']
  if name in source_set:
   checked+=1;continue
  target=policy['historical_path_overrides'].get(name,name)
  actual=_safe_target(root,target)
  assert actual.is_file() and actual.stat().st_size==entry['size_bytes'],('Missing/size-changed predecessor member',name,target)
  if hash_files:assert sha(actual)==entry['sha256'],('Changed predecessor member',name,target)
  if target!=name:overrides[name]={'path':target,'sha256':entry['sha256']}
  checked+=1
 assert checked==len(baseline['files'])==16579
 historical=policy['historical_execution'];run=read(root/historical['receipt'])
 assert sha(root/historical['receipt'])==historical['receipt_sha256']
 assert run['status']=='passed' and run['command']=='reproduce' and len(run['steps'])==55
 assert all(s['status']=='passed' and s['exit_code']==0 for s in run['steps'])
 for name,digest in historical['files_sha256'].items():assert sha(_safe_target(root,name))==digest
 original,identity=transport.original_identity(root,hash_files)
 return {'status':'passed_snapshot_guards' if hash_files else 'passed_source_and_size_inventory','release':policy['release'],'predecessor_release':policy['predecessor_release'],
  'baseline_manifest_sha256':sha(path),'predecessor_source_manifest_sha256':sha(old_seal),'old_members_checked':checked,
  'old_sources_checked':349,'explicit_changed_documentation_sources':changed,'historical_path_bindings':overrides,
  'historical_55_receipt_sha256':historical['receipt_sha256'],'frozen_original_inputs':identity['frozen_input_files'],
  'hashes_computed':hash_files,'scope':('Audit.1 source preimages and saved output bytes' if hash_files else 'Source preimages hashed; old output presence/size only, not full output-byte validation')+'; pinned historical 55-step pass, not a new 55-step run or native-Wald validation.'}


def recipe(root=ROOT,path=RECIPE):
 root=Path(root).resolve();p=Path(path);p=p if p.is_absolute() else root/p;value=read(p)
 assert value['release']==RELEASE and value['source_manifest']==SOURCE_MANIFEST
 for field in ['adapter_steps','producer_steps','verification_steps','calibration_steps','report_steps','report_verification_steps','browser_steps']:
  assert isinstance(value[field],list)
  for command in value[field]:
   assert command and isinstance(command,list) and command[0].startswith('scripts/')
   assert _safe_target(root,command[0]).is_file(),('Required native-Wald command absent',command[0])
 return value


def artifact_maps(proof,folder=None,root=ROOT):
 result={}
 for field in ['verified_artifacts_sha256','artifacts_sha256','source_artifacts_sha256','source_sha256','sources_sha256','inputs_sha256','input_sources_sha256','source_npz_sha256','source_files_sha256','source_files','checked_files','checked_files_sha256','files_sha256','outputs_sha256']:
  mapping=proof.get(field)
  if isinstance(mapping,dict):
   for name,digest in mapping.items():
    if isinstance(name,str) and isinstance(digest,str) and len(digest)==64:
     if not name.startswith(('outputs/','scripts/','replication/','templates/','data/')):
      if folder is None:continue
      name=str((Path(folder)/name).relative_to(Path(root)))
     if name in result:assert result[name]==digest
     result[name]=digest
 return result


def native_readiness(root=ROOT,config=None,*,require_browser=True,check_current=True):
 root=Path(root).resolve();config=config or recipe(root);records={};bindings={}
 for role in ['adapter','native','calibration','report']+(['browser'] if require_browser else []):
  descriptor=config['receipts'][role];path=_safe_target(root,descriptor['path']);proof=read(path)
  assert proof['status'] in descriptor.get('statuses',['passed']),(role,'Current native-Wald receipt not passed')
  mapping=artifact_maps(proof,path.parent,root)
  for name,digest in mapping.items():assert sha(_safe_target(root,name))==digest,(role,'Stale verified artifact',name)
  bindings.update(mapping);records[role]={'path':descriptor['path'],'sha256':sha(path),'status':proof['status']}
 tests=root/'outputs/native_wald/tests.csv';assert tests.is_file()
 assert 'outputs/native_wald/tests.csv' in bindings,'Native/report proof must bind the actual overlay tests.csv'
 production=read(root/'outputs/native_wald/production.json');assert production['status']=='passed'
 inventory=read(root/'outputs/native_wald/inventory.json');verification=read(root/config['receipts']['native']['path'])
 assert production['rows']==inventory['test_rows']==verification['test_rows']
 assert production['native_kernel_calls']==inventory['native_kernel_calls']==verification['native_kernel_calls']
 assert verification['available_native_tests']+verification['unavailable_tests_retained']==verification['test_rows']
 assert verification['available_native_tests']==verification['native_kernel_calls']
 assert verification['verifier_source_sha256']==sha(root/'scripts/verify_native_wald_analysis.py')
 algorithms=production['algorithm_sources_sha256']
 assert set(algorithms)=={'scripts/native_wald_analysis.py','scripts/native_wald.py','scripts/native_wald_legacy.py','scripts/native_wald_extensions.py'}
 assert all(sha(root/name)==digest for name,digest in algorithms.items())
 assert inventory['algorithm_sources_sha256']==algorithms
 assert production['adapter_sha256']==sha(root/'scripts/native_wald.py') and production['producer_sha256']==sha(root/'scripts/native_wald_analysis.py')
 for name,digest in production['outputs_sha256'].items():assert sha(root/'outputs/native_wald'/name)==digest
 adapter=read(root/config['receipts']['adapter']['path'])
 assert adapter['native_model_refits']==0
 assert adapter['source_sha256']['scripts/native_wald.py']==sha(root/'scripts/native_wald.py')
 assert adapter['source_sha256']['scripts/verify_native_wald.py']==sha(root/'scripts/verify_native_wald.py')
 calibration=read(root/config['receipts']['calibration']['path'])
 assert calibration['native_model_refits']==0 and calibration['economic_model_results_changed'] is False and calibration['historical_calibration_evidence_changed'] is False
 assert calibration['source_artifacts_sha256']['scripts/native_wald_calibration.py']==sha(root/'scripts/native_wald_calibration.py')
 assert calibration['source_artifacts_sha256']['scripts/native_wald.py']==sha(root/'scripts/native_wald.py')
 report_manifest=read(root/'outputs/native_wald/report_manifest.json')
 assert report_manifest['native_model_fits']==0 and report_manifest['coefficient_t_tests_changed'] is False
 assert report_manifest['builder_source_sha256']==sha(root/'scripts/native_wald_report.py')
 assert report_manifest['template_sha256']==sha(root/'templates/native_wald.html')
 assert report_manifest['tests_csv_sha256']==sha(tests)
 assert report_manifest['current_master_sha256']==sha(root/config['staged_master'])
 assert report_manifest['native_index_sha256']==sha(root/'outputs/native_wald/native_wald.html')
 for row in report_manifest['reports'].values():assert row['current_html_sha256']==sha(root/'outputs/native_wald/reports'/row['filename'])
 report=read(root/config['receipts']['report']['path'])
 assert report['prototype_only'] is False and report['numerical_verifier_source_matches_current'] is True,'Prototype presentation proof cannot establish current release readiness'
 assert report['old_standalones_unchanged'] is True and report['overview_coefficient_t_rows_unchanged'] is True
 assert report['verifier_source_sha256']==sha(root/'scripts/verify_native_wald_report.py')
 assert report['browser_verifier_source_sha256']==sha(root/'scripts/verify_native_wald_report.mjs')
 assert report['numerical_verification_sha256']==records['native']['sha256']
 assert bindings['outputs/native_wald/report_manifest.json']==sha(root/'outputs/native_wald/report_manifest.json')
 assert report_manifest['calibration']['receipt_sha256']==records['calibration']['sha256']
 boundary=root/'outputs/native_wald/calibration_boundary_verification.json'
 assert report_manifest['calibration']['boundary_receipt_sha256']==sha(boundary)
 boundary_proof=read(boundary);assert boundary_proof['status']=='passed' and boundary_proof['source_sha256']==sha(root/'scripts/native_wald_calibration.py')
 master=root/config['current_master'];staged=root/config['staged_master'];assert staged.is_file()
 if check_current:assert master.is_file() and sha(master)==sha(staged),'Current master is not the verified native-Wald stage'
 if require_browser:
  browser=read(root/config['receipts']['browser']['path'])
  assert browser['staged_only'] is False and browser['tested_master_sha256']==sha(master)
  assert browser['verifier_source_sha256']==sha(root/'scripts/verify_native_wald_browser.py')
  assert browser['master_sha256']==sha(master) and browser['staged_master_sha256']==sha(staged)
 assert str(staged.relative_to(root)) in bindings or str(master.relative_to(root)) in bindings,'Current native-Wald master is not receipt-bound'
 for role in ['native','report']:
  assert records[role]['path'].startswith('outputs/native_wald/'),'Legacy proof cannot stand in for native-Wald completion'
 return {'status':'passed_native_wald_release_gates','release':RELEASE,'receipts':records,'tests_sha256':sha(tests),
  'master_sha256':sha(master) if check_current else None,'staged_master_sha256':sha(staged),'verified_artifacts':len(bindings),'scope':'Current native-kernel overlay and current presentation only; historical coefficient/covariance/sample proofs remain separately pinned.'}
