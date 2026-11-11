"""Portable byte bindings for independently checked visitor analytical artifacts."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def snapshot(folder):
    """Analysis files plus actual sample/design paths named by specifications.

    Excludes derived verification/presentation files and prior proof archives,
    so rerunning a verifier can update its own status without a digest cycle.
    """
    folder=Path(folder);paths=set()
    for path in folder.iterdir():
        if path.is_file() and path.suffix in {'.csv','.json','.npz'} and not path.name.startswith(('verification','report')):paths.add(path)
    for name in ['designs','samples','input_samples','pyfixest','native_checkpoints']:
        base=folder/name
        if base.exists():
            paths.update(path for path in base.rglob('*') if path.is_file() and path.suffix in {'.csv','.json','.npz'})
    specifications=folder/'specifications.csv'
    if specifications.exists():
        with specifications.open(newline='') as handle:
            for row in csv.DictReader(handle):
                for field in ['sample_csv','source_sample_csv','source_design_npz','design_npz','input_sample_csv']:
                    value=row.get(field,'')
                    if value:
                        path=ROOT/value;assert path.is_file(),('Declared analytical input absent',field,value);paths.add(path)
    return {str(path.resolve().relative_to(ROOT)):sha(path) for path in sorted(paths)}


def assert_unchanged(expected):
    for name,digest in expected.items():assert sha(ROOT/name)==digest,('Analytical artifact changed during independent math audit',name)


def archive_passed(path):
    path=Path(path)
    if path.exists():
        try:old=json.loads(path.read_text())
        except (ValueError,OSError):return
        if old.get('status')=='passed':
            directory=path.parent/'verification_history';directory.mkdir(exist_ok=True)
            archived=directory/(sha(path)+'.json')
            if archived.exists():assert archived.read_bytes()==path.read_bytes()
            else:archived.write_bytes(path.read_bytes())
