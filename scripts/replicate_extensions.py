"""Reproduce or verify the local AEA audit extension without refreshing data.

The immutable v1.0.0 package remains a separate reference. This driver verifies
its input-member hashes and a separately sealed extension source manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import replicate
from seal_extension_sources import source_paths
from restore_replication import verify_files

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"replication/extensions/source_manifest.json"
HEAT_PRODUCERS=["crisis_sensitivity.py","annual_crisis_sensitivity.py","complete_va_controls.py",
           "annual_literature_sensitivity.py","literature_sensitivity.py","annual_seasonal_literature.py",
           "literature_rank_correction.py","literature_restrictions.py",
           "literature_full_controls.py","literature_full_controls_restrictions.py"]
ARRIVAL_PRODUCERS=["visitor_rigour_monthly.py","visitor_rigour_companions.py","visitor_origin_year.py"]
HEAT_NUMERICAL=["verify_crisis_sensitivity.py","verify_annual_crisis_sensitivity.py","verify_complete_va_controls.py",
           "verify_annual_literature_sensitivity.py","verify_literature_sensitivity.py","verify_annual_seasonal_literature.py","verify_literature_full_controls.py"]
ARRIVAL_NUMERICAL=["verify_visitor_rigour.py","verify_visitor_rigour_companions.py","verify_visitor_origin_year.py"]
PRESENTATION=["crisis_report.py","verify_crisis_report.py","literature_report.py","verify_literature_report.py",
              "extension_stationarity_report.py","verify_stationarity_report.py",
              "verify_visitor_rigour_stationarity_report.py","stata_key_report.py","verify_stata_key_report.py",
              "corrected_legacy_notes.py","qualified_clustered_sector_sensitivity.py","verify_qualified_clustered_sector_sensitivity.py",
              "visitor_rigour_report.py","verify_visitor_rigour_report.py",
              "aea_audit_report.py","combined_report.py","verify_collection.py","verify_baseline.py",
              "verify_extension_collection.py","verify_final_audit_closure.py"]


def execution_steps(command):
    """Refresh parents before diagnostics that bind their current receipts."""
    steps=["verify_verification_boundaries.py","verify_derived_proof_guards.py"]+(HEAT_PRODUCERS+ARRIVAL_PRODUCERS if command=="reproduce" else [])+HEAT_NUMERICAL
    # Rebuild diagnostic bindings after any legitimate native receipt refresh;
    # this computes no economic regression and never changes a fitted sample.
    steps+=["extension_stationarity.py","verify_extension_stationarity.py","selected_null_calibration.py","verify_calendar_hac.py"]
    steps+=ARRIVAL_NUMERICAL
    steps+=["verify_arrivals_review_equivalence.py","selected_arrivals_null_calibration.py"]
    # Diagnostic construction is not an economic refit. Refresh its bindings to
    # freshly executed parent receipts even in existing-output verification.
    steps+=["visitor_rigour_stationarity.py","verify_visitor_rigour_stationarity.py",
            "verify_stata_key_results.py","verify_stata_visitor_rigour.py"]
    return steps+PRESENTATION


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source(manifest):
    expected=json.loads(manifest.read_text())
    actual=set(source_paths(manifest))
    recorded=set(expected["sources_sha256"])
    if actual!=recorded:
        raise RuntimeError("Extension source/evidence file set changed: "+json.dumps({"added":sorted(actual-recorded),"missing":sorted(recorded-actual)}))
    failures=[name for name,digest in expected["sources_sha256"].items() if not (ROOT/name).is_file() or sha(ROOT/name)!=digest]
    if failures:raise RuntimeError("Extension source differs from its separate sealed manifest: "+", ".join(failures))
    if sha(ROOT/"replication/manifest.json")!=expected["published_v1_manifest_sha256"]:
        raise RuntimeError("Published v1.0.0 manifest changed")
    for name,digest in expected["original_references_sha256"].items():
        if sha(ROOT/name)!=digest:raise RuntimeError("Original reference changed: "+name)
    return expected


def check_frozen_inputs():
    # Source is checked against the separate extension revision above; the
    # original release remains authoritative for every frozen input member.
    original=json.loads((ROOT/"replication/manifest.json").read_text())
    return verify_files(original,ROOT,groups=("inputs",),include_sources=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["inventory","verify","reproduce"])
    parser.add_argument("--manifest",type=Path,default=MANIFEST)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    steps=execution_steps(args.command)
    if args.dry_run:
        print(json.dumps({"command":args.command,"source_manifest":str(args.manifest),"steps":steps,
            "data_policy":"Cached official data only; HTTP disabled; immutable v1.0.0 references are not resealed",
            "stata_policy":"Verify stored native MCP runs through independent Python ports; licensed native Stata reruns use the separate MCP commands."},indent=2));return
    environment,problems=replicate.environment()
    if problems:raise RuntimeError("; ".join(problems))
    source=check_source(args.manifest)
    check_frozen_inputs()
    if args.command=="inventory":
        print(json.dumps({"status":"passed","environment":environment,"source_files":len(source["sources_sha256"]),"steps":steps},indent=2));return
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder=ROOT/"replication/logs/extensions"/stamp;folder.mkdir(parents=True,exist_ok=False)
    result={"status":"running","command":args.command,"started_utc":stamp,"steps":[],
            "source_manifest_sha256":sha(args.manifest),"environment":environment,
            "claim_scope":"Computational/source/calendar/presentation checks, not causal identification or stationarity certification"}
    def save():(folder/"run.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    save();started=time.monotonic()
    print(f"Extension execution evidence: {folder/'run.json'}",flush=True)
    try:
        for i,name in enumerate(steps,1):
            print(f"[{i}/{len(steps)}] {name}",flush=True)
            step={"script":"scripts/"+name,"status":"running"};result["steps"].append(step);save()
            t=time.monotonic()
            with (folder/f"{i:02d}-{name}.log").open("w") as log:
                completed=subprocess.run([sys.executable,"-c",replicate.FROZEN_LAUNCHER,str(ROOT/"scripts"/name)],
                    cwd=ROOT,env=replicate.child_environment(),stdout=log,stderr=subprocess.STDOUT,check=False)
            step.update(exit_code=completed.returncode,seconds=time.monotonic()-t,status="passed" if completed.returncode==0 else "failed")
            save()
            if completed.returncode:raise RuntimeError(f"{name} failed; see {folder}")
        check_source(args.manifest)
        check_frozen_inputs()
        result.update(status="passed",elapsed_seconds=time.monotonic()-started);save()
    except BaseException as error:
        result.update(status="failed",error=str(error),elapsed_seconds=time.monotonic()-started);save();raise
    print(f"Extension {args.command} passed: {len(steps)} subprocesses; source/input references unchanged.",flush=True)


if __name__=="__main__":main()
