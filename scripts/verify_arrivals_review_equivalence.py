"""Bind a later reproduction to the immutable numerics examined by Fable.

This is an exact semantic bridge, not a new AI review or causal certificate. It
never refreshes its baseline during verification and applies no numeric tolerance.
Capture is an explicit, one-time archival operation on the read-only review tree.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'replication/extensions/review_evidence/arrivals_review_baseline.json'
OUT = ROOT / 'outputs/visitor_rigour/review_equivalence.json'
FAMILIES = ('monthly', 'annual', 'ppml', 'origin_year')
SOURCES = ('scripts/visitor_rigour_monthly.py', 'scripts/visitor_rigour_companions.py',
 'scripts/visitor_origin_year.py', 'scripts/visitor_rigour_stationarity.py',
 'scripts/visitor_counts.py', 'scripts/pyfixest_backend.py', 'scripts/crisis_sensitivity.py',
 'scripts/literature_sensitivity.py', 'scripts/literature_full_controls.py',
 'scripts/annual_seasonal_literature.py', 'scripts/extended_controls.py',
 'notes/visitor_rigour_protocol.md', 'notes/visitor_origin_year_protocol.md',
 'notes/crisis_sensitivity_policy.json', 'pyproject.toml', 'uv.lock',
 'replication/extensions/supplementary_inputs/calendar/T2007e.txt',
 'replication/extensions/supplementary_inputs/calendar/sources.json')
REVISION = 'replication/extensions/review_evidence/derived_proof_guard_revision.json'
GUARD_VERIFIER = 'scripts/verify_derived_proof_guards.py'
APPROVED = {'scripts/literature_sensitivity.py': ('assignment', 'protected_paths'),
 'scripts/annual_seasonal_literature.py': ('assignment', 'protected_names'),
 'scripts/literature_full_controls.py': ('function', 'protected_inventory')}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_record(value):
 array = np.asarray(value)
 assert not array.dtype.hasobject, 'Pickled/object diagnostic arrays are not allowed'
 # Include shape/dtype to prevent aliases across differently interpreted bytes.
 header = {'shape': list(array.shape), 'dtype': array.dtype.str}
 raw = np.ascontiguousarray(array).tobytes()
 return {**header, 'sha256': hashlib.sha256(json.dumps(header,sort_keys=True).encode()+raw).hexdigest()}


def inventory(root):
 result = {}
 counts = {}
 for family in FAMILIES:
  folder = root / 'outputs/visitor_rigour' / family
  specs = list(csv.DictReader((folder/'specifications.csv').open(newline='')))
  logs = list(csv.DictReader((folder/'pyfixest/fit_log.csv').open(newline='')))
  counts[family] = {'requests': len(specs), 'native_fits': len(logs),
   'statuses': {state: sum(row['status']==state for row in specs) for state in sorted({row['status'] for row in specs})}}
  for path in sorted(folder.rglob('*')):
   if not path.is_file() or path.suffix not in {'.csv','.npz'}: continue
   if any(part in {'preflight','native_tables','verification_history','R-library'} for part in path.parts): continue
   if path.name.startswith('verification') or path.name.startswith('report_'): continue
   relative = str(path.relative_to(root))
   if path.suffix == '.npz':
    with np.load(path,allow_pickle=False) as archive:
     arrays = {name: array_record(archive[name]) for name in sorted(archive.files)}
    result[relative] = {'kind':'npz','arrays':arrays,'array_count':len(arrays)}
   else:
    with path.open(newline='') as handle:
     rows = list(csv.reader(handle))
    if not rows:
     result[relative] = {'kind':'csv','rows':0,'columns':[],'sha256':digest([])}; continue
    header, body = rows[0], rows[1:]
    assert len(header)==len(set(header)), relative
    assert all(len(row)==len(header) for row in body), relative
    # Hash columns in specification/export rows are provenance metadata; the
    # actual corresponding model/sample arrays and complete native ledgers are
    # checked here. Root source revision is proved separately, never resealed.
    ledger = path.name=='fit_log.csv' and path.parent.name=='pyfixest'
    retained = [i for i,name in enumerate(header) if ledger or not (name.endswith('sha256') or name.endswith('_hash'))]
    columns = [header[i] for i in retained]
    normalized = [[row[i] for i in retained] for row in body]
    result[relative] = {'kind':'csv','rows':len(body),'columns':columns,
      'excluded_provenance_columns':[name for i,name in enumerate(header) if i not in retained],
      'sha256':digest([columns,normalized])}
 return result, counts


def capture_arrivals_review_baseline(reviewed_root):
 assert not BASELINE.exists(), 'The reviewed baseline is immutable; capture refuses overwrite'
 reviewed_root = Path(reviewed_root).resolve()
 files, counts = inventory(reviewed_root)
 revision = json.loads((ROOT/REVISION).read_text())
 assert {change['path'] for change in revision['changes']}==set(APPROVED)
 approvals = []
 for change in revision['changes']:
  assert (change['guard_kind'],change['guard_target'])==APPROVED[change['path']]
  assert sha(reviewed_root/change['path'])==change['before_sha256']
  approvals.append({key:change[key] for key in ['path','guard_kind','guard_target','before_sha256','after_sha256']})
 payload = {'status':'immutable_reviewed_numeric_baseline_not_a_review_verdict','schema_version':1,
  'scope':'354 final arrival-rigour requests; model arrays, actual samples, semantic exports/statuses and all319 native ledger rows',
  'comparison_policy':'Exact dtype/shape/array bytes; exact canonical CSV cells except separately-bound provenance hash fields. No numeric tolerance, baseline refresh, model refit or source reseal.',
  'files':files,'counts':counts,'reviewed_sources_sha256':{name:sha(reviewed_root/name) for name in SOURCES},
  'post_snapshot_guard_revision_sha256':sha(ROOT/REVISION),'guard_verifier_sha256':sha(ROOT/GUARD_VERIFIER),
  'authorized_guard_only_changes':approvals,
  'limits':'A later pass establishes semantic identity with the reviewed generation, not that the AI reviewed later report/driver/privacy code or that inference/causality is validated.'}
 assert sum(value['requests'] for value in counts.values())==354
 assert sum(value['native_fits'] for value in counts.values())==319
 BASELINE.parent.mkdir(parents=True,exist_ok=True)
 BASELINE.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 return {'status':'captured_once','baseline_sha256':sha(BASELINE),'files':len(files),'counts':counts}


def verify_arrivals_review_equivalence():
 OUT.parent.mkdir(parents=True,exist_ok=True)
 OUT.write_text(json.dumps({'status':'running','claim':'No completed current-generation equivalence claim'})+'\n')
 try:
  baseline = json.loads(BASELINE.read_text())
  assert sha(ROOT/REVISION)==baseline['post_snapshot_guard_revision_sha256']
  assert sha(ROOT/GUARD_VERIFIER)==baseline['guard_verifier_sha256']
  from verify_derived_proof_guards import ast_proof
  revision = json.loads((ROOT/REVISION).read_text())
  changes = {row['path']:row for row in revision['changes']}
  assert set(changes)==set(APPROVED)
  source_changes=[]
  for name,expected in baseline['reviewed_sources_sha256'].items():
   current=sha(ROOT/name)
   if current==expected: continue
   assert name in APPROVED, f'Unreviewed source difference: {name}'
   change=changes[name]
   assert (change['guard_kind'],change['guard_target'])==APPROVED[name]
   assert expected==change['before_sha256'] and current==change['after_sha256']
   source_changes.append(ast_proof(change))
  current,counts=inventory(ROOT)
  assert set(current)==set(baseline['files']), {'missing':sorted(set(baseline['files'])-set(current)), 'extra':sorted(set(current)-set(baseline['files']))}
  differences=[]
  for name,expected in baseline['files'].items():
   if current[name]!=expected: differences.append(name)
  assert not differences, f'Post-review numeric/status/export difference: {differences}'
  assert counts==baseline['counts'], (counts,baseline['counts'])
  report={'status':'passed','scope':baseline['scope'],'comparison_policy':baseline['comparison_policy'],
   'baseline_sha256':sha(BASELINE),'verifier_source_sha256':sha(Path(__file__)),
   'verified_at_utc':datetime.now(timezone.utc).isoformat(),
   'current_native_fit_logs_sha256':{f'outputs/visitor_rigour/{family}/pyfixest/fit_log.csv':sha(ROOT/f'outputs/visitor_rigour/{family}/pyfixest/fit_log.csv') for family in FAMILIES},
   'current_native_estimator_manifests_sha256':{f'outputs/visitor_rigour/{family}/pyfixest/estimator_manifest.json':sha(ROOT/f'outputs/visitor_rigour/{family}/pyfixest/estimator_manifest.json') for family in FAMILIES},
   'semantic_files_checked':len(current),'npz_arrays_checked':sum(row.get('array_count',0) for row in current.values()),
   'counts':counts,'source_bindings_checked':len(baseline['reviewed_sources_sha256']),
   'authorized_post_snapshot_guard_only_source_changes':source_changes,
   'guard_revision_sha256':sha(ROOT/REVISION),'guard_verifier_sha256':sha(ROOT/GUARD_VERIFIER),
   'numeric_or_status_differences':differences,'native_fits_executed_by_this_verifier':0,
   'claim':'Current arrival model/sample/array/export/native-ledger semantics exactly match the immutable reviewed generation. Explicit guard-only source revisions preserve every other AST node.',
   'limits':baseline['limits']}
 except Exception as error:
  OUT.write_text(json.dumps({'status':'failed','error':repr(error),'baseline_refreshed':False},indent=2)+'\n')
  raise
 OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 return report


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--capture-reviewed',type=Path,help='Explicit one-time capture from the frozen review tree; refuses overwrite')
 args=parser.parse_args()
 result=capture_arrivals_review_baseline(args.capture_reviewed) if args.capture_reviewed else verify_arrivals_review_equivalence()
 print(json.dumps({key:value for key,value in result.items() if key not in {'files','authorized_post_snapshot_guard_only_source_changes'}},indent=2))
