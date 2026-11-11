"""Narrow isolated fixtures: proof lifecycle, stored-Stata portability and privacy.

No economic/native model fit and no production analytical artifact is changed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import verification_artifacts as artifacts
import verify_visitor_rigour as monthly
import verify_visitor_origin_year as origin_year
import verify_visitor_rigour_companions as companions
import verify_visitor_rigour_stationarity as diagnostics
import verify_stata_visitor_rigour as stata_arrivals
from stata_capture import verify_package_capture,portable_native_artifacts,assert_artifacts_unchanged
from sanitize_stata_transport import sanitize_stata_transport


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lifecycle_fixture(base):
    previous=(json.dumps({'status':'passed','fixture':'Previously passed proof; no economic values'},indent=2)+'\n').encode();expected=hashlib.sha256(previous).hexdigest();records=[]
    cases=[('monthly',monthly,'_verify_visitor_rigour',monthly.verify_visitor_rigour),('origin_year',origin_year,'_verify',origin_year.verify_visitor_origin_year),
        ('annual',companions,'_verify_annual',companions.verify_visitor_annual),('ppml',companions,'_verify_ppml',companions.verify_visitor_ppml),('diagnostics',diagnostics,'_verify_visitor_rigour_stationarity',diagnostics.verify_visitor_rigour_stationarity)]
    for name,module,math_name,run in cases:
        parent=base/name;folder=parent/name if module is companions else parent;folder.mkdir(parents=True)
        path=folder/'verification.json';path.write_bytes(previous)
        (folder/'specifications.csv').write_text('spec_id,design_npz\nfixture,declared_but_missing.npz\n')
        calls=[]
        def never(*args,**kwargs):calls.append(True);raise RuntimeError('Fixture reached economic math')
        with patch.object(module,'OUT',parent),patch.object(artifacts,'ROOT',base),patch.object(module,math_name,never):
            try:run()
            except AssertionError as error:assert 'Declared analytical input absent' in str(error)
            else:raise AssertionError('Missing-file fixture unexpectedly passed')
        current=json.loads(path.read_text());assert current['status']=='failed' and current['error_type']=='AssertionError' and not calls
        archived=folder/'verification_history'/f'{expected}.json';assert archived.read_bytes()==previous
        records.append({'branch':name,'missing_declared_artifact':'failed','previous_passed_proof_archived_byte_exact':True,'economic_math_calls':0})
    return records


def portability_fixture(base):
    base.mkdir(parents=True)
    manifest=json.loads((ROOT/'outputs/stata_key_results/preparation_manifest.json').read_text());environment=json.loads((ROOT/'outputs/stata_key_results/mcp/environment.json').read_text())
    unavailable=lambda path:base/'absent_ado'/Path(path).name
    result=verify_package_capture(manifest['packages'],environment,resolve_local=unavailable)
    assert result['historical_capture_consistent'] and result['current_local_sources_unavailable']==4 and result['current_local_sources_confirmed']==0
    altered=json.loads(json.dumps(environment));first=next(iter(manifest['packages'].values()))['path'];altered['package_sources'][first]['sha256']='0'*64
    try:verify_package_capture(manifest['packages'],altered,resolve_local=unavailable)
    except AssertionError:pass
    else:raise AssertionError('Historical manifest/capture mismatch was accepted')
    local=base/'wrong.ado';local.write_text('*! fixture changed source\n')
    try:verify_package_capture(manifest['packages'],environment,resolve_local=lambda path:local)
    except AssertionError:pass
    else:raise AssertionError('Present mismatching local source was accepted')
    arrival=base/'outputs/stata_key_results/arrivals';(arrival/'mcp').mkdir(parents=True)
    upstream=base/'outputs/stata_key_results/mcp';upstream.mkdir(parents=True)
    shutil.copyfile(ROOT/'outputs/stata_key_results/mcp/environment.json',upstream/'environment.json')
    (base/'scripts').mkdir()
    for name in ['verify_stata_visitor_rigour.py','stata_visitor_rigour.py','stata_mcp_client.py','stata_capture.py']:shutil.copyfile(ROOT/'scripts'/name,base/'scripts'/name)
    for filename in ['environment.json','requirements.txt']:shutil.copyfile(ROOT/'outputs/stata_key_results/arrivals/mcp'/filename,arrival/'mcp'/filename)
    before=sha(arrival/'mcp/environment.json')
    for directory in ['do','data','native','native_tables']:(arrival/directory).mkdir()
    for filename in ['specifications.json','specifications.csv','temperature_effects.csv','control_estimates.csv','joint_tests.csv','source_bridge.csv','relative_date_FE_identity.csv','stata_python_coverage.csv','vce_convention_comparison.csv','singleton_sample_comparison.csv']:(arrival/filename).write_text('{}\n')
    for filename in ['preparation_manifest.json','do/key_results.do','do/postprocess.mata.do','native/master_stata_output.txt','native/completed.txt','native/runs.csv','mcp/transport_privacy_redactions.json','mcp/arrival_key_results.json']:(arrival/filename).write_text('{}\n')
    protocol=json.loads((ROOT/'outputs/stata_key_results/arrivals/mcp/arrival_key_results.json').read_text());protocol['server_command']=sys.executable
    with patch.object(stata_arrivals,'OUT',arrival),patch.object(stata_arrivals,'ROOT',base):stata_arrivals.capture_runtime_and_artifacts(protocol,resolve_local=unavailable)
    local_check=json.loads((arrival/'mcp/local_environment_check.json').read_text())
    assert local_check['mcp_server_python_available'] and local_check['current_local_sources_confirmed']==0 and local_check['current_local_sources_unavailable']==37 and not local_check['fresh_native_estimation']
    assert sha(arrival/'mcp/environment.json')==before
    bound=portable_native_artifacts(arrival,root=base)
    assert not any(name.endswith('.log') or name.endswith(('local_environment_check.json','runtime_proof.json','verification.json')) or '/verification_history/' in name for name in bound)
    runtime=json.loads((arrival/'runtime_proof.json').read_text())
    assert runtime['portable_artifacts_sha256']==runtime['artifact_sha256']==bound
    assert_artifacts_unchanged(bound,root=base)
    required=arrival/'mcp/arrival_key_results.json';original=required.read_bytes();required.write_text('{"fixture_tamper":true}\n')
    try:assert_artifacts_unchanged(bound,root=base)
    except AssertionError:pass
    else:raise AssertionError('Changed mandatory native/MCP member was accepted')
    required.write_bytes(original)
    (arrival/'mcp/fixture_server_stderr.log').write_text('Optional fixture transport\n')
    (arrival/'mcp/local_environment_check.json').write_text('{"fixture_probe_changed":true}\n')
    assert portable_native_artifacts(arrival,root=base)==bound
    return {'heat_manifest_packages_historically_verified_without_local_ado':4,'arrival_same_python_no_ado_sources_unavailable':37,'historical_capture_preserved':True,'historical_capture_mismatch_rejected':True,'present_local_mismatch_rejected':True,'fresh_native_estimation_claim':False,
        'portable_native_artifact_members':len(bound),'mandatory_native_MCP_tamper_rejected':True,'runtime_mandatory_maps_equal_passed_map':True,'optional_transport_local_probe_changes_excluded':True}


def privacy_fixture(base):
    folder=base/'outputs/stata_key_results/mcp';folder.mkdir(parents=True)
    fake_serial=b'1'*12
    original=b'Unrelated transport line\r\n'+b'Serial'+b' number: '+fake_serial+b'\r\n  Licensed'+b' to: FIXTURE OWNER\r\n               FIXTURE ORGANIZATION\r\n\r\nUnrelated native output\r\n'
    path=folder/'fixture_server_stderr.log';path.write_bytes(original);native=base/'outputs/stata_key_results/native';native.mkdir();(native/'master_stata_output.txt').write_bytes(b'Native result unchanged\n');(folder/'response.json').write_text('{"native":"unchanged"}\n')
    result=sanitize_stata_transport(base);assert result['redacted_fields_this_run']==3
    record=json.loads((folder/'transport_privacy_redactions.json').read_text())['records'][0]
    preimage=base/record['private_preimage'];assert preimage.read_bytes()==original
    assert stat.S_IMODE(preimage.stat().st_mode)==0o600 and stat.S_IMODE(preimage.parent.stat().st_mode)==0o700 and stat.S_IMODE((base/'dist/local_private_stata_transport').stat().st_mode)==0o700
    assert fake_serial not in path.read_bytes() and b'FIXTURE OWNER' not in path.read_bytes() and b'FIXTURE ORGANIZATION' not in path.read_bytes()
    assert path.read_bytes().startswith(b'Unrelated transport line\r\n') and path.read_bytes().endswith(b'Unrelated native output\r\n')
    before={p.name:sha(p) for p in folder.iterdir() if p.is_file()};second=sanitize_stata_transport(base);assert second['redacted_fields_this_run']==0
    assert before=={p.name:sha(p) for p in folder.iterdir() if p.is_file()};sanitize_stata_transport(base,check_only=True)
    return {'only_startup_identifiers_redacted':3,'raw_preimage_byte_exact':True,'private_modes':'0700/0600','native_and_response_json_unchanged':True,'idempotent_second_run':True}


def verify_verification_boundaries():
    with tempfile.TemporaryDirectory(prefix='verification-boundary-fixtures-') as temporary:
        base=Path(temporary);result={'status':'passed','lifecycle':lifecycle_fixture(base/'lifecycle'),'stata_portability':portability_fixture(base/'portability'),'transport_privacy':privacy_fixture(base/'privacy'),'economic_or_native_model_fits':0}
    paths=['scripts/verification_artifacts.py','scripts/stata_capture.py','scripts/sanitize_stata_transport.py','scripts/verify_visitor_rigour.py','scripts/verify_visitor_origin_year.py','scripts/verify_visitor_rigour_companions.py','scripts/verify_visitor_rigour_stationarity.py','scripts/verify_stata_key_results.py','scripts/verify_stata_visitor_rigour.py','scripts/verify_verification_boundaries.py']
    result['source_sha256']={path:sha(ROOT/path) for path in paths}
    output=ROOT/'outputs/visitor_rigour/verification_boundaries.json';output.write_text(json.dumps(result,indent=2)+'\n')
    print('Verification boundaries passed:5missing-file lifecycles, stored-Stata absent-ado checks and idempotent private transport redaction;0model fits.',flush=True);return result


if __name__=='__main__':verify_verification_boundaries()
