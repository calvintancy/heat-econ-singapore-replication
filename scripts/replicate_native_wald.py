"""Verify/rebuild only the additive native-Wald layer over pinned audit.1 fits.

No original coefficient/VCE/sample refit is implied. Optional legacy-reproduce
executes the old 55 stages only in a separate prepared workspace.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,sys,time
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import replicate
import native_wald_release as release

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def plan(config,command,with_browser=False):
 if command=='reproduce':
  steps=config.get('adapter_steps',[])+config['producer_steps']+config['verification_steps']+config.get('calibration_steps',[])+config['report_steps']
 else:steps=config.get('adapter_steps',[])+config['verification_steps']+config.get('report_verification_steps',[])
 if with_browser:steps+=config['browser_steps']
 return steps


@contextmanager
def completion_run(path):
 if path.exists():
  previous=path.read_bytes();history=path.parent/'completion_history'/(hashlib.sha256(previous).hexdigest()+'.json')
  history.parent.mkdir(parents=True,exist_ok=True)
  if history.exists():assert history.read_bytes()==previous
  else:history.write_bytes(previous)
 started=datetime.now(timezone.utc).isoformat()
 save(path,{'status':'running','started_utc':started,'scope':'Replacement native-Wald release checks have not completed; prior completion is historical only.'})
 previous_signal=signal.getsignal(signal.SIGTERM)
 def interrupted(signum,frame):raise KeyboardInterrupt('SIGTERM')
 signal.signal(signal.SIGTERM,interrupted)
 try:
  yield
  assert release.read(path)['status'] in {'passed','staged_not_current'},'Execution returned without a terminal current receipt'
 except BaseException as error:
  save(path,{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed','started_utc':started,'error':repr(error),'scope':'No current passed native-Wald release claim'})
  raise
 finally:signal.signal(signal.SIGTERM,previous_signal)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('command',choices=['inventory','verify','reproduce','legacy-reproduce'])
 parser.add_argument('--manifest',default=release.SOURCE_MANIFEST)
 parser.add_argument('--recipe',default=release.RECIPE)
 parser.add_argument('--dry-run',action='store_true')
 parser.add_argument('--with-browser',action='store_true')
 parser.add_argument('--install-master',action='store_true',help='Explicitly replace only the master after native report verification')
 parser.add_argument('--workspace',type=Path,help='Separate prepared package root for optional legacy55 execution')
 args=parser.parse_args()
 if args.command in {'verify','reproduce'} and not args.dry_run:
  with completion_run(ROOT/'outputs/native_wald/completion.json'):
   execute(args,parser)
 else:execute(args,parser)


def execute(args,parser):
 config=release.recipe(ROOT,args.recipe)
 if args.command=='legacy-reproduce':
  if not args.workspace:parser.error('legacy-reproduce requires --workspace containing a separately extracted v2 package')
  workspace=args.workspace.resolve()
  if workspace==ROOT or ROOT in workspace.parents:parser.error('Legacy workspace must be outside the active project')
  command=[sys.executable,'-c',replicate.FROZEN_LAUNCHER,str(workspace/'scripts/replicate_extensions.py'),'reproduce','--manifest',str(workspace/args.manifest)]
  if args.dry_run:print(json.dumps({'command':command,'scope':'Old55 only in separate workspace; not a new native-Wald overlay validation'},indent=2));return
  release.source_guard(workspace,args.manifest);release.baseline_guard(workspace)
  policy=release.read(ROOT/release.PROVENANCE);baseline=release.read(ROOT/policy['baseline_manifest'])
  for entry in baseline['files']:
   active=ROOT/entry['path'];separate=workspace/entry['path']
   if active.is_file() and separate.is_file():assert not os.path.samefile(active,separate),('Legacy workspace shares an inode with active project; extract/copy without hardlinks',entry['path'])
  completed=subprocess.run(command,cwd=workspace,env=replicate.child_environment())
  if completed.returncode:raise RuntimeError('Separate legacy55 execution failed')
  release.source_guard(workspace,args.manifest)
  save(workspace/'outputs/native_wald/legacy_reproduction.json',{'status':'passed_legacy_reproduction','source_manifest_sha256':release.sha(workspace/args.manifest),
   'scope':'The unchanged legacy55 pipeline was executed in this separate workspace under the v2 source seal. Derived proof timestamps may change; this is not byte identity or current-native-Wald presentation validation.'})
  return
 steps=plan(config,args.command,args.with_browser)
 if args.dry_run:
  print(json.dumps({'command':args.command,'steps':steps,'manifest':args.manifest,'install_master':args.install_master,
   'scope':'Native-Wald/reference layer and presentation only; old55 reused by hash unless explicitly requested in separate workspace'},indent=2));return
 environment,problems=replicate.environment()
 if problems:raise RuntimeError('; '.join(problems))
 release.source_guard(ROOT,args.manifest);baseline=release.baseline_guard(ROOT)
 if args.command=='inventory':
  print(json.dumps({'status':'passed_inventory','baseline':baseline,'environment':environment,'steps':steps},indent=2));return
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');folder=ROOT/'outputs/native_wald/runs'/stamp
 result={'status':'running','command':args.command,'started_utc':stamp,'source_manifest_sha256':release.sha(ROOT/args.manifest),
  'baseline':baseline,'environment':environment,'steps':[],'scope':'Current native-Wald layer only. Old55 is pinned historical evidence; no original coefficient/covariance/sample refit in this command.'}
 destination=folder/'run.json';save(destination,result);started=time.monotonic()
 try:
  # Browser checks run after explicit master integration, below.
  browser_steps=config['browser_steps'] if args.with_browser else []
  work_steps=steps[:-len(browser_steps)] if browser_steps else steps
  for number,command in enumerate(work_steps,1):
   step={'command':command,'status':'running'};result['steps'].append(step);save(destination,result);t=time.monotonic()
   print(f'[{number}/{len(steps)}] '+ ' '.join(command),flush=True)
   with (folder/f'{number:02d}-{Path(command[0]).stem}.txt').open('w') as log:
    done=subprocess.run([sys.executable,'-c',replicate.FROZEN_LAUNCHER,str(ROOT/command[0]),*command[1:]],cwd=ROOT,env=replicate.child_environment(),stdout=log,stderr=subprocess.STDOUT)
   step.update(status='passed' if done.returncode==0 else 'failed',exit_code=done.returncode,seconds=time.monotonic()-t);save(destination,result)
   if done.returncode:raise RuntimeError('Native-Wald stage failed: '+command[0])
   stdout_target=config.get('stdout_json_outputs',{}).get(' '.join(command))
   if stdout_target:
    value=json.loads((folder/f'{number:02d}-{Path(command[0]).stem}.txt').read_text())
    assert value['status']=='passed';save(ROOT/stdout_target,value)
    step['stdout_json_artifact']={'path':stdout_target,'sha256':release.sha(ROOT/stdout_target)};save(destination,result)
  if args.install_master:
   proof=release.read(ROOT/config['receipts']['report']['path']);assert proof['status'] in config['receipts']['report'].get('statuses',['passed'])
   stage=ROOT/config['staged_master'];assert stage.is_file()
   release.native_readiness(ROOT,config,require_browser=False,check_current=False)
   release.baseline_guard(ROOT)
   old=ROOT/config['current_master'];temporary=old.with_suffix(old.suffix+'.native-wald.tmp')
   temporary.write_bytes(stage.read_bytes());assert release.sha(temporary)==release.sha(stage);os.replace(temporary,old)
   result['explicit_master_integration']={'staged_sha256':release.sha(stage),'current_sha256':release.sha(old),'predecessor':baseline['historical_path_bindings']['outputs/heat_economy.html']};save(destination,result)
  for command in browser_steps:
   step={'command':command,'status':'running'};result['steps'].append(step);save(destination,result)
   t=time.monotonic();number=len(result['steps'])
   with (folder/f'{number:02d}-{Path(command[0]).stem}.txt').open('w') as log:
    done=subprocess.run([sys.executable,'-c',replicate.FROZEN_LAUNCHER,str(ROOT/command[0]),*command[1:]],cwd=ROOT,env=replicate.child_environment(),stdout=log,stderr=subprocess.STDOUT)
   step.update(status='passed' if done.returncode==0 else 'failed',exit_code=done.returncode,seconds=time.monotonic()-t);save(destination,result)
   if done.returncode:raise RuntimeError('Native-Wald browser stage failed')
  release.source_guard(ROOT,args.manifest);after=release.baseline_guard(ROOT)
  current=(ROOT/config['current_master']).is_file() and release.sha(ROOT/config['current_master'])==release.sha(ROOT/config['staged_master'])
  if current:
   gates=release.native_readiness(ROOT,config,require_browser=args.with_browser)
   result.update(status='passed',current_native_gates=gates,baseline_after=after,elapsed_seconds=time.monotonic()-started);save(destination,result)
   save(ROOT/'outputs/native_wald/completion.json',result)
  else:
   result.update(status='passed_staged_not_current',baseline_after=after,elapsed_seconds=time.monotonic()-started);save(destination,result)
   save(ROOT/'outputs/native_wald/staged_completion.json',result)
   save(ROOT/'outputs/native_wald/completion.json',{**result,'status':'staged_not_current'})
  print(result['status'],str(destination),flush=True)
 except BaseException as error:
  result.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-started);save(destination,result);raise


if __name__=='__main__':main()
