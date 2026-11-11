"""Historical Stata package provenance with optional current local confirmation.

Stored native-result verification does not require a commercial installation.
Present local sources must match the frozen capture; absent local sources are
reported as unconfirmed, never relabelled as freshly executed Stata evidence.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_package_capture(packages,environment,*,resolve_local=None):
    captured=environment['package_sources'];checks=[]
    for name,record in packages.items():
        path=record['path'];assert path in captured,('Package absent from packaged historical capture',name)
        for field in ['sha256','header']:assert record[field]==captured[path][field],('Packaged manifest/capture mismatch',name,field)
        local=Path(path) if resolve_local is None else Path(resolve_local(path))
        present=local.is_file()
        if present:
            assert sha(local)==record['sha256'],('Installed package differs from historical capture',name)
            first=local.read_text(errors='replace').splitlines()[0] if local.stat().st_size else ''
            assert first==record['header'],('Installed package header differs from historical capture',name)
        checks.append({'package':name,'historical_path':path,'sha256':record['sha256'],'header':record['header'],
            'historical_capture_consistent':True,'current_local_source_available':present,
            'current_local_status':'Hash/header confirmed' if present else 'Unavailable; historical packaged capture only'})
    return {'historical_capture_consistent':True,'packages_checked':len(checks),
        'current_local_sources_confirmed':sum(row['current_local_source_available'] for row in checks),
        'current_local_sources_unavailable':sum(not row['current_local_source_available'] for row in checks),
        'packages':checks,'scope':'Historical native MCP/log evidence and packaged source hashes/headers remain authoritative for the stored run. Optional present local sources are confirmed separately; absence does not imply a fresh native environment or a new native estimation.'}


def verify_environment_sources(environment,*,resolve_local=None):
    packages={path:{'path':path,**record} for path,record in environment['package_sources'].items()}
    return verify_package_capture(packages,environment,resolve_local=resolve_local)


HEAT_RESULTS=['estimates.csv','control_coefficients.csv','joint_tests.csv','vce_convention_comparison.csv','singleton_sample_comparison.csv','stata_python_coverage.csv']
ARRIVAL_RESULTS=['temperature_effects.csv','control_estimates.csv','joint_tests.csv','vce_convention_comparison.csv','singleton_sample_comparison.csv','source_bridge.csv','relative_date_FE_identity.csv','stata_python_coverage.csv']


def portable_native_artifacts(folder,*,root,include_results=True):
    """Required distributed native evidence, without proof/transport cycles.

    Raw/safe stderr logs and local-environment probes are deliberately absent.
    The frozen environment and public privacy provenance remain mandatory.
    """
    folder=Path(folder);root=Path(root);arrival=folder.name=='arrivals';paths=set()
    for name in ['preparation_manifest.json','specifications.json','specifications.csv','do/key_results.do','do/postprocess.mata.do',
        'native/master_stata_output.txt','native/completed.txt','native/runs.csv','mcp/environment.json','mcp/requirements.txt',
        'mcp/transport_privacy_redactions.json','mcp/arrival_key_results.json' if arrival else 'mcp/heat_key_results.json']:
        path=folder/name;assert path.is_file(),('Required portable Stata artifact absent',str(path));paths.add(path)
    for directory,suffixes in [('do',{'.do'}),('data',{'.csv'}),('native',{'.csv','.txt'}),('native_tables',{'.html'})]:
        paths.update(path for path in (folder/directory).rglob('*') if path.is_file() and path.suffix in suffixes)
    paths.update(path for path in (folder/'mcp').iterdir() if path.is_file() and path.suffix=='.json' and path.name not in {'local_environment_check.json'})
    if (folder/'policy.json').is_file():paths.add(folder/'policy.json')
    if (folder/'README.md').is_file():paths.add(folder/'README.md')
    if include_results:
        for name in ARRIVAL_RESULTS if arrival else HEAT_RESULTS:
            path=folder/name;assert path.is_file(),('Verified Stata result export absent',str(path));paths.add(path)
    return {str(path.resolve().relative_to(root.resolve())):sha(path) for path in sorted(paths)}


def optional_transport_artifacts(folder,*,root):
    return {str(path.resolve().relative_to(Path(root).resolve())):sha(path) for path in sorted((Path(folder)/'mcp').glob('*_server_stderr.log')) if path.is_file()}


def assert_artifacts_unchanged(mapping,*,root):
    for name,digest in mapping.items():assert sha(Path(root)/name)==digest,('Verified native/MCP artifact changed during port',name)
