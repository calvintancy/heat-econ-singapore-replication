"""Verify the explicit guard-only source revision and its integrity boundary.

No fit or reference reseal. The fixture uses a disposable tree and the actual
full-controls lineage guard; it never substitutes dummy data in the project.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'replication/extensions/review_evidence/derived_proof_guard_revision.json'
OUT = ROOT / 'outputs/derived_proof_guards/verification.json'
REMOVED = {
 'scripts/literature_sensitivity.py': ['outputs/crisis_sensitivity/verification.json','outputs/complete_va_controls/verification.json'],
 'scripts/annual_seasonal_literature.py': ['outputs/crisis_sensitivity/verification.json'],
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RemoveGuard(ast.NodeTransformer):
 def __init__(self, kind, target): self.kind,self.target,self.found=kind,target,[]
 def visit_Assign(self,node):
  if self.kind=='assignment' and any(isinstance(t,ast.Name) and t.id==self.target for t in node.targets):
   self.found.append(node);return None
  return self.generic_visit(node)
 def visit_FunctionDef(self,node):
  if self.kind=='function' and node.name==self.target:
   self.found.append(node);return None
  return self.generic_visit(node)


def ast_proof(change):
 before=change['before_source_utf8'];after=(ROOT/change['path']).read_text()
 assert hashlib.sha256(before.encode()).hexdigest()==change['before_sha256']
 assert sha(ROOT/change['path'])==change['after_sha256']
 old,new=ast.parse(before),ast.parse(after)
 a,b=RemoveGuard(change['guard_kind'],change['guard_target']),RemoveGuard(change['guard_kind'],change['guard_target'])
 old_clean,new_clean=a.visit(copy.deepcopy(old)),b.visit(copy.deepcopy(new))
 assert len(a.found)==len(b.found)==1
 assert ast.dump(old_clean,include_attributes=False)==ast.dump(new_clean,include_attributes=False),change['path']
 if change['guard_kind']=='assignment':
  old_names=ast.literal_eval(a.found[0].value);new_names=ast.literal_eval(b.found[0].value)
  removed=REMOVED[change['path']]
  assert new_names==[name for name in old_names if name not in removed]
  assert set(old_names)-set(new_names)==set(removed)
 else: removed='Derived verification-prefixed/history and named presentation records only; actual behavior tested separately'
 unchanged_functions={}
 new_functions={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 for node in old.body:
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in new_functions:
   left=ast.get_source_segment(before,node);right=ast.get_source_segment(after,new_functions[node.name])
   if left==right:unchanged_functions[node.name]=hashlib.sha256(left.encode()).hexdigest()
 return {'path':change['path'],'before_sha256':change['before_sha256'],'after_sha256':change['after_sha256'],
  'excluded_guard_node':change['guard_target'],'all_other_AST_identical':True,'removed_derived_bindings':removed,'byte_identical_other_functions_sha256':unchanged_functions}


def fixture():
 import literature_full_controls as producer
 import verify_literature_full_controls as verifier
 old=(producer.ROOT,producer.OUT,verifier.ROOT,verifier.OUT)
 guarded=['coefficients.csv','tests.csv','tests_native_stage.csv','rank_correction_manifest.json','two_channel_span_manifest.json','designs/example.npz','pyfixest/fit_log.csv']
 with tempfile.TemporaryDirectory(prefix='proof-guard-fixture-') as directory:
  root=Path(directory).resolve();full=root/'outputs/literature_sensitivity/full_controls'
  def put(name,data=b'fixture analytical bytes\n'):
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
  # Populate the actual explicitly declared source/input list with fixture bytes.
  module=ast.parse((ROOT/'scripts/literature_full_controls.py').read_text())
  function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='protected_inventory')
  initial=next(n for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paths' for t in n.targets))
  for name in ast.literal_eval(initial.value.generators[0].iter):put(name)
  put('scripts/literature_full_controls.py');put('scripts/span_fixture.py')
  for name in guarded:put('outputs/literature_sensitivity/'+name)
  put('outputs/crisis_sensitivity/parents/example.csv')
  derived=['outputs/crisis_sensitivity/verification.json','outputs/complete_va_controls/verification.json',
    'outputs/literature_sensitivity/verification.json','outputs/literature_sensitivity/verification_details.csv',
    'outputs/literature_sensitivity/verification_history/run/proof.json','outputs/literature_sensitivity/report_inventory.json',
    'outputs/literature_sensitivity/report_verification.json','outputs/literature_sensitivity/payload_packaging.json']
  for name in derived:put(name,b'first derived generation\n')
  span=put('outputs/literature_sensitivity/full_controls/two_channel_span_tests.csv')
  full.mkdir(parents=True,exist_ok=True)
  (full/'two_channel_span_manifest.json').write_text(json.dumps({'source':'scripts/span_fixture.py','source_sha256':sha(root/'scripts/span_fixture.py'),
    'output':str(span.relative_to(root)),'output_sha256':sha(span),'frozen_inputs_sha256':{},'design_archives_sha256':{}}))
  try:
   producer.ROOT=verifier.ROOT=root;producer.OUT=verifier.OUT=full
   protected=producer.protected_inventory()
   assert not set(derived)&set(protected)
   for name in guarded:assert 'outputs/literature_sensitivity/'+name in protected
   manifest={'inputs_sha256':{},'protected_sources_sha256':protected,'producer_source':'scripts/literature_full_controls.py','producer_source_sha256':sha(root/'scripts/literature_full_controls.py')}
   verifier.assert_lineage(manifest)
   for name in derived:put(name,b'refreshed derived verification/presentation bytes\n')
   verifier.assert_lineage(manifest)
   rejected=[]
   for name in ['outputs/literature_sensitivity/'+n for n in guarded]+['outputs/crisis_sensitivity/parents/example.csv','scripts/crisis_sensitivity.py']:
    path=root/name;before=path.read_bytes();path.write_bytes(before+b'changed\n')
    try:
     try:verifier.assert_lineage(manifest)
     except AssertionError:rejected.append(name)
     else:raise AssertionError('Analytical tampering not rejected: '+name)
    finally:path.write_bytes(before)
   verifier.assert_lineage(manifest)
  finally:producer.ROOT,producer.OUT,verifier.ROOT,verifier.OUT=old
 return {'scope':'Disposable synthetic tree; actual production inventory and independent verifier lineage guard, zero native fits',
  'derived_files_refreshed_without_invalidating_analytical_lineage':derived,'genuine_analytical_tampers_rejected':rejected,'strict_analytical_members':len(protected)}


def verify_derived_proof_guards():
 ledgers={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'outputs').glob('**/pyfixest/fit_log.csv')}
 evidence=json.loads(EVIDENCE.read_text());changes=[ast_proof(c) for c in evidence['changes']];behavior=fixture()
 assert all(sha(ROOT/n)==h for n,h in ledgers.items())
 report={'status':'passed','scope':'Guard-only source AST equivalence and actual integrity-boundary fixture; no estimator/sample/formula/covariance/test changes or native fit/reference reset',
  'revision_evidence_sha256':sha(EVIDENCE),'sources':changes,'fixture':behavior,'native_fits_run':0,'existing_native_ledgers_unchanged':True,'verifier_sha256':sha(Path(__file__))}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('Derived-proof guard revision passed: three source ASTs and strict analytical tamper fixtures; zero native fits',flush=True)
 return report


if __name__=='__main__':verify_derived_proof_guards()
