"""Create a new source-revision manifest without replacing an existing one."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"replication/extensions/source_manifest.json"
REFERENCES=["outputs/covid_comparison/original_inputs_reference.json",
            "outputs/pyfixest/migration_input_reference.json",
            "data/reference_before_nsa_inputs/manifest.json"]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(manifest):
    try:
        if not (ROOT/".git").exists():
            # An extracted project may sit inside another repository's ignored
            # directory. Its parent's Git index is not this project's inventory.
            raise OSError("No project-local Git metadata")
        names=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"],cwd=ROOT,stderr=subprocess.DEVNULL).decode().split("\0")
    except (OSError,subprocess.CalledProcessError):
        # Standalone archives intentionally omit .git. Retain explicitly tracked
        # files inside ignored data trees, and detect extra unignored source files.
        if not Path(manifest).is_file():
            raise RuntimeError("Creating a new source revision requires a Git checkout; a standalone archive must carry its sealed manifest")
        names=set(json.loads(Path(manifest).read_text())["sources_sha256"])
        patterns=[line.strip() for line in (ROOT/".gitignore").read_text().splitlines() if line.strip() and not line.startswith("#")]
        if any(pattern.startswith("!") for pattern in patterns):
            raise RuntimeError("Standalone source inventory does not support negated gitignore rules")
        def ignored(relative,is_dir):
            if relative==".git" or relative.startswith(".git/"):return True
            for pattern in patterns:
                directory=pattern.endswith("/");rule=pattern.strip("/")
                if pattern.startswith("/"):
                    if relative==rule or (directory and relative.startswith(rule+"/")):return True
                elif "/" not in rule:
                    if any(fnmatch.fnmatchcase(part,rule) for part in relative.split("/")):return True
                elif fnmatch.fnmatchcase(relative,rule):return True
            return False
        for directory,dirs,files in os.walk(ROOT):
            parent=Path(directory)
            dirs[:]=[name for name in dirs if not ignored((parent/name).relative_to(ROOT).as_posix(),True)]
            names.update((parent/name).relative_to(ROOT).as_posix() for name in files if not ignored((parent/name).relative_to(ROOT).as_posix(),False))
    target=Path(manifest).resolve()
    return sorted(name for name in set(names) if name and (ROOT/name).is_file() and (ROOT/name).resolve()!=target)


def seal(output):
    output=Path(output).resolve()
    if output.exists():raise RuntimeError("Manifest already exists; choose an explicit new revision path rather than resealing it")
    names=source_paths(output)
    value={"schema_version":1,"purpose":"Separately sealed source revision for the local AEA extension",
           "sources_sha256":{name:sha(ROOT/name) for name in names},
           "published_v1_manifest_sha256":sha(ROOT/"replication/manifest.json"),
           "original_references_sha256":{name:sha(ROOT/name) for name in REFERENCES},
           "policy":"Original input/result references and tolerances are unchanged; this source manifest is a new revision, not a replacement release benchmark."}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as handle:handle.write(json.dumps(value,indent=2)+"\n")
    print(f"Sealed {len(names)} source/evidence files: {output}")
    return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT)
    seal(parser.parse_args().output)
