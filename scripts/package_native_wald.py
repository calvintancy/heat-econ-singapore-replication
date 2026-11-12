"""Versioned native-Wald packages; old release seals/assets are never rewritten.

Uses the existing bounded ZIP transport and safe extractor, with a separate
current-native readiness profile and immutable audit.1 snapshot guards.
"""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import package_extensions as transport
import native_wald_release as release

ROOT=Path(__file__).resolve().parents[1]
NAME='heat-econ-native-wald-v2'


def collect(root,manifest,*,hash_files,require_ready):
 root=Path(root).resolve();baseline=release.baseline_guard(root,hash_files=hash_files)
 plan=transport.collect(root,manifest,hash_files=hash_files,require_ready=False)
 # This one expected stale display binding is historical and byte-pinned by
 # baseline_guard. Missing numerical/source artifacts remain fatal.
 allowed={'outputs/collection/verification.json: missing/current passed proof'}
 problems=[item for item in plan['not_ready'] if item not in allowed]
 if problems:raise ValueError('Legacy evidence unexpectedly invalid: '+json.dumps(problems))
 config=release.recipe(root)
 gates=release.native_readiness(root,config) if require_ready else None
 if require_ready:
  completion=release.read(root/'outputs/native_wald/completion.json')
  assert completion['status']=='passed' and completion['source_manifest_sha256']==release.sha(root/release.SOURCE_MANIFEST)
  assert completion['current_native_gates']['master_sha256']==gates['master_sha256']
  for role in ['adapter','native','calibration','report']:
   assert completion['current_native_gates']['receipts'][role]['sha256']==gates['receipts'][role]['sha256'],('Native completion became stale',role)
 plan.update(native_wald_baseline=baseline,current_native_wald_gates=gates,
  historical_legacy_receipts=plan['readiness_receipts'],historical_legacy_display_scope='Audit.1 only; original master is byte-bound to predecessor_master.html')
 return plan


def build(root,manifest,out_dir,name=NAME,limit=transport.DEFAULT_LIMIT):
 root,out_dir=Path(root).resolve(),Path(out_dir).resolve()
 if out_dir==root or (root in out_dir.parents and root/'dist' not in out_dir.parents and out_dir!=root/'dist'):raise ValueError('Write transport outside the project or under ignored dist')
 if not name or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for char in name):raise ValueError('Unsafe transport name')
 out_dir.mkdir(parents=True,exist_ok=True)
 if list(out_dir.glob(name+'-*')):raise ValueError('Refusing to overwrite release transport; use a fresh directory/name')
 plan=collect(root,manifest,hash_files=True,require_ready=True)
 tree={'schema_version':1,'package_kind':'heat-economy-local-extension','release':release.RELEASE,'readiness_profile':'native-wald-overlay-v2',
  'files':plan['files'],'excluded':plan['excluded'],'original_v1':plan['original_v1'],'source_revision':plan['source_revision'],
  'native_wald_baseline':plan['native_wald_baseline'],'current_native_wald_gates':plan['current_native_wald_gates'],
  'historical_legacy_receipts':plan['historical_legacy_receipts'],'historical_legacy_scope':plan['historical_legacy_display_scope'],
  'self_digest_excluded':transport.EMBEDDED_INVENTORY,
  'scope':'Original fitted coefficients, covariance and samples are byte-pinned. Current native chi2 Wald rows/p-values and current presentation are separately verified; no claim that old F p-values are unchanged or newly calibrated.',
  'reproduction_claim':'File/source/saved-result integrity and the specific recorded native-overlay execution; no new legacy55 or fresh-environment claim is inferred.'}
 embedded=transport.encoded(tree);extra={'path':transport.EMBEDDED_INVENTORY,'size_bytes':len(embedded),'sha256':hashlib.sha256(embedded).hexdigest(),'groups':['metadata']}
 groups=transport.parts([extra]+plan['files'],limit);outer={**tree,'files':[extra]+plan['files'],'archives':[],'max_archive_bytes':limit,'nonoverlapping_archive_members':True}
 with tempfile.TemporaryDirectory(prefix='.native-wald-package-',dir=out_dir) as temporary:
  stage=Path(temporary)
  for i,group in enumerate(groups,1):
   path=stage/f'{name}-part-{i:03d}.zip';transport.write_zip(root,path,group,embedded)
   if path.stat().st_size>limit:raise ValueError('Archive exceeds asset limit; lower --part-bytes')
   outer['archives'].append({'filename':path.name,'size_bytes':path.stat().st_size,'sha256':release.sha(path),'members':group,'n_files':len(group)})
  for entry in plan['files']:assert release.sha(root/entry['path'])==entry['sha256'],('Tree changed while packaging',entry['path'])
  release.source_guard(root,manifest);release.baseline_guard(root)
  path=stage/f'{name}-manifest.json';path.write_bytes(transport.encoded(outer))
  checks=[f"{r['sha256']}  {r['filename']}" for r in outer['archives']]+[f'{release.sha(path)}  {path.name}']
  (stage/f'{name}-SHA256SUMS').write_text('\n'.join(checks)+'\n')
  for p in stage.iterdir():os.replace(p,out_dir/p.name)
 return outer


def verify_tree(value,root):
 assert value['release']==release.RELEASE and value['readiness_profile']=='native-wald-overlay-v2'
 byte_result=transport.verify_tree(value,root)
 baseline=release.baseline_guard(root);gates=release.native_readiness(root)
 assert value['current_native_wald_gates']['master_sha256']==gates['master_sha256']
 return {'status':'passed_native_wald_package_integrity','file_integrity':byte_result,'baseline':baseline,'native_gates':gates,
  'scope':'Exact packaged bytes/current-source/native-overlay receipts; not a new economic refit or scientific-validity certificate'}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['inventory','build','verify','extract'])
 p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--source-manifest',type=Path,default=Path(release.SOURCE_MANIFEST))
 p.add_argument('--out-dir',type=Path,default=ROOT/'dist/native-wald-package');p.add_argument('--name',default=NAME)
 p.add_argument('--part-bytes',type=int,default=transport.DEFAULT_LIMIT);p.add_argument('--manifest',type=Path);p.add_argument('--archive-dir',type=Path);p.add_argument('--archives-only',action='store_true')
 a=p.parse_args();root=a.root.resolve();source=a.source_manifest if a.source_manifest.is_absolute() else root/a.source_manifest
 if a.command=='inventory':
  plan=collect(root,source,hash_files=False,require_ready=False)
  result={'status':'native_wald_inventory_only','files':len(plan['files']),'bytes':sum(row['size_bytes'] for row in plan['files']),
   'planned_parts':len(transport.parts(plan['files'],a.part_bytes)),'baseline':plan['native_wald_baseline'],'scope':'No source seal, transport build or model execution'}
 elif a.command=='build':
  value=build(root,source,a.out_dir,a.name,a.part_bytes);result={'status':'packaged_verified_native_wald_bytes','archives':[{k:r[k] for k in ['filename','size_bytes','sha256']} for r in value['archives']]}
 else:
  path=a.manifest or root/transport.EMBEDDED_INVENTORY;value=release.read(path)
  assert value['release']==release.RELEASE and value['readiness_profile']=='native-wald-overlay-v2','Not a native-Wald v2 transport'
  if a.archives_only:
   if not a.archive_dir:p.error('--archives-only needs --archive-dir')
   result=transport.verify_archives(value,a.archive_dir)
  elif a.command=='extract':
   if not a.archive_dir or not a.manifest:p.error('extract needs detached --manifest and --archive-dir')
   transport.extract(value,a.archive_dir,root);result=verify_tree(value,root)
  else:result=verify_tree(value,root)
 print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
