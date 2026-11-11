"""Redact Stata startup license identifiers from transport stderr only.

Native logs, MCP JSON, calculations and result exports are never changed.
Raw local preimages are private, ignored, and not replication inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT=Path(__file__).resolve().parents[1]
MARKER=b'[REDACTED STATA LICENSE IDENTIFIER]'
FIELDS=re.compile(rb'(?im)^([ \t]*)(Serial[ \t]+number|Licen[sc]e[ \t]+number|Licen[sc]ed[ \t]+to|Registered[ \t]+to)([ \t]*:[ \t]*)([^\r\n]*)(\r?\n|$)')


def digest(data):return hashlib.sha256(data).hexdigest()


def redact(data):
    counts={'serial_number':0,'license_number':0,'registered_license_identifier':0,'registered_continuation':0};lines=[];owner_active=False
    for number,line in enumerate(data.splitlines(keepends=True),1):
        match=FIELDS.fullmatch(line)
        if match:
            label=match[2].lower();value=match[4]
            owner=b'to' in label;owner_active=owner
            if value.strip() and value.strip()!=MARKER:
                kind='registered_license_identifier' if owner else 'serial_number' if b'serial' in label else 'license_number'
                counts[kind]+=1;lines.append(number)
                line=match[1]+match[2]+match[3]+MARKER+match[5]
        elif owner_active:
            if not line.strip():owner_active=False
            elif re.match(rb'^[ \t]{15,}\S',line):
                prefix=re.match(rb'^[ \t]+',line)[0];ending=b'\r\n' if line.endswith(b'\r\n') else b'\n' if line.endswith(b'\n') else b''
                if line.strip()!=MARKER:line=prefix+MARKER+ending;counts['registered_continuation']+=1;lines.append(number)
            else:owner_active=False
        yield line,counts,lines


def sanitized(data):
    chunks=[];counts={'serial_number':0,'license_number':0,'registered_license_identifier':0,'registered_continuation':0};lines=[]
    for chunk,counts,lines in redact(data):chunks.append(chunk)
    return b''.join(chunks),counts,lines


def sanitize_stata_transport(root=ROOT,*,check_only=False):
    root=Path(root).resolve();private=root/'dist/local_private_stata_transport'
    scopes=[root/'outputs/stata_key_results/mcp',root/'outputs/stata_key_results/arrivals/mcp'];summary=[]
    # Explicit immutable comparison envelope; no numerical/native/MCP response
    # file can be silently modified by this transport-only privacy operation.
    protected={str(path.relative_to(root)):digest(path.read_bytes()) for base in [root/'outputs/stata_key_results'] for path in base.rglob('*') if path.is_file() and path.suffix in {'.csv','.json','.do','.txt','.npz'} and path.name!='transport_privacy_redactions.json'}
    for folder in scopes:
        if not folder.is_dir():continue
        metadata=folder/'transport_privacy_redactions.json';old=json.loads(metadata.read_text()) if metadata.exists() else {'records':[]};records=list(old['records'])
        for path in sorted(folder.glob('*_server_stderr.log')):
            original=path.read_bytes();safe,counts,line_numbers=sanitized(original);relative=str(path.relative_to(root));total=sum(counts.values())
            if check_only:
                assert total==0,('Unsanitized startup identifiers remain; values withheld',relative,total)
                matching=[row for row in records if row['path']==relative and row['sanitized_sha256']==digest(original)]
                assert matching,('Sanitized transport lacks portable privacy provenance',relative)
                summary.append({'path':relative,'status':'sanitized_provenance_verified'});continue
            if total:
                private.mkdir(parents=True,exist_ok=True);os.chmod(private,0o700)
                group=private/('arrivals' if 'arrivals' in path.parts else 'heat');group.mkdir(exist_ok=True);os.chmod(group,0o700)
                preimage=group/(digest(original)+'__'+path.name)
                if preimage.exists():assert preimage.read_bytes()==original
                else:
                    descriptor=os.open(preimage,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                    with os.fdopen(descriptor,'wb') as handle:handle.write(original)
                os.chmod(preimage,0o600)
                assert stat.S_IMODE(private.stat().st_mode)==stat.S_IMODE(group.stat().st_mode)==0o700
                assert stat.S_IMODE(preimage.stat().st_mode)==0o600
                path.write_bytes(safe)
                record={'path':relative,'raw_sha256':digest(original),'sanitized_sha256':digest(safe),'raw_bytes':len(original),'sanitized_bytes':len(safe),
                    'redacted_fields':counts,'redacted_line_numbers':line_numbers,'private_preimage':str(preimage.relative_to(root)),
                    'private_preimage_distributed':False,'private_directory_mode':'0700','private_file_mode':'0600'}
                if record not in records:records.append(record)
            elif not any(row['path']==relative and row['sanitized_sha256']==digest(original) for row in records):
                records.append({'path':relative,'raw_sha256':digest(original),'sanitized_sha256':digest(original),'raw_bytes':len(original),'sanitized_bytes':len(original),'redacted_fields':counts,'redacted_line_numbers':[],'private_preimage':None,'private_preimage_distributed':False})
            summary.append({'path':relative,'redacted_fields':total,'status':'sanitized' if total else 'already_safe'})
        if not check_only:
            value={'schema_version':1,'scope':'Only startup serial/license-number and registered-license identifiers in MCP server stderr; native master logs, MCP response JSON and numerical results unchanged',
                'redaction_marker':MARKER.decode(),'private_preimages':'Local ignored dist/local_private_stata_transport only; excluded from all replication packages and not required for reproduction',
                'sanitizer_source':'scripts/sanitize_stata_transport.py','sanitizer_source_sha256':digest(Path(__file__).read_bytes()),'records':records}
            content=(json.dumps(value,indent=2)+'\n').encode()
            if not metadata.exists() or metadata.read_bytes()!=content:metadata.write_bytes(content)
    for name,expected in protected.items():assert digest((root/name).read_bytes())==expected,('Protected native/MCP/result file changed',name)
    return {'status':'passed','check_only':check_only,'transport_logs':len(summary),'redacted_fields_this_run':sum(row.get('redacted_fields',0) for row in summary),'protected_native_files_unchanged':len(protected),'files':summary}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');arguments=parser.parse_args()
    print(json.dumps(sanitize_stata_transport(check_only=arguments.check_only),indent=2))
