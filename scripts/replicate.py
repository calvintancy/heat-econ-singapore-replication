"""Run the frozen Singapore heat/economy replication from any working directory.

Bootstrap first: restore_replication.py, uv sync --locked, then run this file with
the locked Python environment. No command refreshes official data. Each existing
producer/verifier runs in a separate process; native estimator ledgers stay with
their analytical family. Existing-output verification and main-only reproduction
do not require R. Visitor reproduction uses the native R gt renderer.

``verify --quick`` runs representative main numerical checks, the complete
monolithic visitor audit, and every presentation verifier. ``verify --all``
(the default) additionally runs every main independent numerical verifier.
``reproduce`` rebuilds the selected analyses before the same verification plan.
Use ``--dry-run`` to inspect a plan without restoring, fitting or writing files.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CALENDAR = "data/raw/visitor_arrivals/calendar/extract_lunar_new_year.py"
RESTORE = "scripts/restore_replication.py"
GENERATED_NOTES = ("notes/pyfixest_migration_validation.md", "notes/covid_comparison_methods.md")

# Only HTTP requests are disabled; native libraries may use unrelated sockets.
# This wrapper changes no analytical implementation or source-vintage setting.
FROZEN_LAUNCHER = r"""
import runpy, sys
from pathlib import Path
import requests
def blocked_request(self, method, url, *args, **kwargs):
    raise RuntimeError(
        'Frozen replication attempted an HTTP request. Restore the complete '
        'release inputs with scripts/restore_replication.py before running; '
        'the replication entry point never fetches or refreshes source data. '
        f'Request: {method} {str(url).split("?")[0]}'
    )
requests.sessions.Session.request = blocked_request
script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name='__main__')
"""

MAIN_NUMERICAL = (
    "verify.py", "verify_iip.py", "verify_extended_controls.py",
    "verify_outcomes.py", "verify_heterogeneity.py", "verify_aggregate_va.py",
    "verify_granular.py", "verify_granular_interactions.py",
    "verify_sector_cluster.py", "verify_cluster_contrasts.py",
    "verify_clustered_sector_sensitivity.py",
    "verify_pyfixest_migration.py", "verify_stationarity_audit.py",
    "verify_annual.py", "verify_covid_comparison.py",
)
MAIN_PRESENTATION = (
    "verify_annual_report.py", "verify_covid_report.py",
    "verify_collection.py", "verify_baseline.py",
)
QUICK_MAIN = (
    ("scripts/verify.py",),
    ("scripts/verify_iip.py",),
    ("scripts/verify_extended_controls.py",),
    ("scripts/verify_sector_cluster.py", "--csv", "outputs/regression_results.csv",
     "--csv", "outputs/iip/monthly/regression_results.csv",
     "--csv", "outputs/granular/va-detailed/full/regression_results.csv"),
    ("scripts/verify_clustered_sector_sensitivity.py",),
)
VISITOR_VERIFIERS = (
    ("scripts/verify_visitor_arrivals.py",),
    ("scripts/verify_visitor_crisis.py",),
    ("scripts/verify_visitor_report.mjs",),
)


def families(only: str) -> tuple[bool, bool]:
    return only in {"all", "main"}, only in {"all", "visitors"}


def verification_plan(only: str, quick: bool) -> list[tuple[str, ...]]:
    main, visitors = families(only)
    steps: list[tuple[str, ...]] = []
    if main:
        steps.extend(QUICK_MAIN if quick else [(f"scripts/{name}",) for name in MAIN_NUMERICAL])
        steps.extend((f"scripts/{name}",) for name in MAIN_PRESENTATION)
    if visitors:
        steps.extend(VISITOR_VERIFIERS[:-1])
        # Numerical audits refresh timestamped receipts embedded as downloads.
        # Assemble the report only after those receipts are current, including
        # when checking a moved frozen package. Cached gt tables need no R here.
        steps.append(("scripts/visitor_report.py",))
        steps.extend(VISITOR_VERIFIERS[-1:])
    return steps


def production_plan(only: str) -> list[tuple[str, ...]]:
    main, visitors = families(only)
    steps: list[tuple[str, ...]] = []
    if main:
        steps.append(("scripts/run_all.py",))
    if visitors:
        steps.extend((name,) for name in (
            CALENDAR, "scripts/visitor_analysis.py", "scripts/visitor_counts.py",
            "scripts/visitor_crisis.py", "scripts/visitor_inference.py",
            "scripts/visitor_gt_data.py",
        ))
        steps.append(("scripts/visitor_gt.R", str(ROOT)))
    return steps


def environment() -> tuple[dict, list[str]]:
    problems: list[str] = []
    if sys.version_info[:2] != (3, 11):
        problems.append("Use Python 3.11: uv sync --locked --python 3.11; uv run --locked python scripts/replicate.py ...")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions: dict[str, set[str]] = {}
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    for package in lock["package"]:
        if "version" in package:
            versions.setdefault(normalize(package["name"]), set()).add(package["version"])
    dependencies = {}
    for requirement in project["project"]["dependencies"]:
        name = re.match(r"[A-Za-z0-9_.-]+", requirement).group()
        expected = sorted(versions.get(normalize(name), set()))
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        dependencies[name] = {"installed": actual, "locked_versions": expected}
        if actual is None or actual not in expected:
            problems.append(f"{name}: installed={actual!r}, lock={expected}; run uv sync --locked")
    node = shutil.which("node")
    node_version = subprocess.check_output([node, "--version"], text=True).strip() if node else None
    if not node_version or not re.match(r"v24\.", node_version):
        problems.append(f"Node.js 24 is required for the report runtime checks; found {node_version!r}")
    uv = shutil.which("uv")
    uv_version = subprocess.check_output([uv, "--version"], text=True).strip() if uv else None
    return {"python": sys.version.split()[0], "python_executable": sys.executable,
            "node": node_version, "uv": uv_version, "dependencies": dependencies}, problems


def helper_inventory(only: str) -> tuple[dict, list[str]]:
    planned = production_plan(only) + verification_plan(only, False)
    required = {"pyproject.toml", "uv.lock", RESTORE}
    required.update(step[0] for step in planned)
    if families(only)[1]:
        required.update(("scripts/setup_visitor_r.R", "replication/renv.lock"))
    problems = [f"Missing helper: {name}" for name in sorted(required) if not (ROOT / name).is_file()]
    scripts = list((ROOT / "scripts").glob("*.py"))
    if (ROOT / CALENDAR).is_file():
        scripts.append(ROOT / CALENDAR)
    local_modules = {path.stem for path in scripts}
    imports: set[str] = set()
    for path in scripts:
        try:
            tree = ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        except SyntaxError as error:
            problems.append(f"Invalid Python helper: {path.relative_to(ROOT)}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.add(node.module.split(".")[0])
    external = imports - local_modules - sys.stdlib_module_names - {"__future__"}
    missing = sorted(name for name in external if importlib.util.find_spec(name) is None)
    problems.extend(f"Missing imported dependency: {name}; run uv sync --locked" for name in missing)
    # Requiring explicit classification prevents a newly added verifier silently
    # falling out of the advertised exhaustive plan.
    classified = {step[0] for step in verification_plan("all", False)
                  if Path(step[0]).name.startswith("verify")}
    discovered = {str(path.relative_to(ROOT)) for path in (ROOT / "scripts").glob("verify*.py")}
    discovered.update(str(path.relative_to(ROOT)) for path in (ROOT / "scripts").glob("verify*.mjs"))
    unclassified = sorted(discovered - classified)
    problems.extend(f"Unclassified verifier: {name}; add it to the exhaustive replication plan" for name in unclassified)
    return {"required_helpers": sorted(required), "python_files_parsed": len(scripts),
            "local_modules": sorted(local_modules), "external_imports": sorted(external),
            "exhaustive_verifier_count": len(classified)}, problems


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["UV_OFFLINE"] = "1"
    env["HEAT_ECON_FROZEN_REPLICATION"] = "1"
    return env


def production_prerequisites(only: str) -> None:
    main, visitors = families(only)
    if main and not shutil.which("pdftotext"):
        raise RuntimeError("Main reproduction requires Poppler's pdftotext for the official IIP-weight PDF audit. Install poppler-utils (Linux) or poppler (macOS).")
    if visitors:
        rscript = shutil.which("Rscript")
        if not rscript:
            raise RuntimeError("Visitor reproduction requires Rscript and the locked gt/jsonlite packages. Existing-output verification and --only main do not require R.")
        subprocess.run([rscript, "--vanilla", str(ROOT / "scripts/setup_visitor_r.R"),
                        str(ROOT), "--check-only"], cwd=ROOT, env=child_environment(), check=True)


def run_python(script: str, arguments: tuple[str, ...] = (), *, frozen: bool = True) -> None:
    command = [sys.executable]
    command += ["-c", FROZEN_LAUNCHER] if frozen else []
    command += [str(ROOT / script), *arguments]
    subprocess.run(command, cwd=ROOT, env=child_environment(), check=True)


def check_inputs(manifest: Path, *, postflight: bool = False) -> None:
    """Delegate archive/member semantics to the single package restore helper."""
    if not manifest.is_file():
        raise RuntimeError(f"Missing release manifest: {manifest}. Run scripts/restore_replication.py first.")
    arguments = ("--manifest", str(manifest), "--root", str(ROOT),
                 "--verify-only", "--groups", "inputs")
    if postflight:
        arguments += ("--skip-source-check",)
    run_python(RESTORE, arguments, frozen=False)


def inventory(only: str, manifest: Path) -> dict:
    env, problems = environment()
    helpers, helper_problems = helper_inventory(only)
    problems.extend(helper_problems)
    report = {"root": str(ROOT), "scope": only, "environment": env, "helpers": helpers,
              "manifest": str(manifest), "problems": problems,
              "source_policy": "Frozen release inputs; HTTP disabled inside every analytical Python child. No refresh and no external EMA-directory fallback."}
    if problems:
        print(json.dumps({"status": "failed", **report}, indent=2), flush=True)
        raise RuntimeError("Replication inventory failed; resolve the listed environment/helper problems.")
    check_inputs(manifest)
    report["status"] = "passed"
    report["input_integrity"] = "Frozen input members verified by restore_replication.py"
    print(json.dumps(report, indent=2), flush=True)
    return report


def execute(steps: list[tuple[str, ...]]) -> None:
    for number, step in enumerate(steps, 1):
        runtime = "node" if step[0].endswith(".mjs") else "Rscript" if step[0].endswith(".R") else "python"
        print(f"[{number}/{len(steps)}] {runtime} {' '.join(step)}", flush=True)
        started = time.monotonic()
        if step[0].endswith(".mjs"):
            subprocess.run([shutil.which("node"), str(ROOT / step[0]), *step[1:]],
                           cwd=ROOT, env=child_environment(), check=True)
        elif step[0].endswith(".R"):
            subprocess.run([shutil.which("Rscript"), "--vanilla", str(ROOT / step[0]), *step[1:]],
                           cwd=ROOT, env=child_environment(), check=True)
        else:
            run_python(step[0], step[1:])
        print(f"  Passed in {time.monotonic()-started:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inventory", "verify", "reproduce"):
        sub = subparsers.add_parser(command, help={"inventory": "Check locked environment, helpers and frozen inputs",
            "verify": "Audit existing fitted outputs without fitting production models",
            "reproduce": "Rebuild selected analyses from the frozen inputs, then verify"}[command])
        sub.add_argument("--only", choices=("all", "main", "visitors"), default="all",
                         help="Select the analytical family (default: all; complete input package still required)")
        sub.add_argument("--manifest", type=Path, default=ROOT / "replication/manifest.json",
                         help="Frozen release manifest (default: repository replication/manifest.json)")
        if command != "inventory":
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument("--quick", action="store_true", help="Representative main numerical checks, full visitor audit and all HTML checks")
            mode.add_argument("--all", action="store_true", help="Every independent verifier (the default)")
            sub.add_argument("--dry-run", action="store_true", help="Print commands without checking inputs or running analyses")
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.only, args.manifest.resolve())
        return 0
    steps = production_plan(args.only) if args.command == "reproduce" else []
    steps += verification_plan(args.only, args.quick)
    if args.dry_run:
        print(json.dumps({"command": args.command, "scope": args.only,
              "verification": "quick (representative main; complete visitor; all presentation)" if args.quick else "all existing independent verifiers",
              "preflight_and_postflight": "Frozen input-group SHA256 verification; outputs may be regenerated",
              "prerequisites": {"node": "24", "main_reproduction": "pdftotext (Poppler)", "visitor_reproduction": "Rscript plus locked gt/jsonlite; no R needed for verify or main-only"},
              "steps": [{"runtime": "node24" if step[0].endswith(".mjs") else "Rscript (native gt)" if step[0].endswith(".R") else "python3.11 (HTTP disabled)",
                         "script": step[0], "arguments": step[1:]} for step in steps]}, indent=2))
        return 0
    started = time.monotonic()
    inventory(args.only, args.manifest.resolve())
    if args.command == "reproduce":
        production_prerequisites(args.only)
    notes = {ROOT / name: (ROOT / name).read_bytes() for name in GENERATED_NOTES if (ROOT / name).is_file()}
    try:
        execute(steps)
    finally:
        # These two verifiers append machine-dependent summary blocks to source
        # documentation. Keep sealed source bytes; fresh numerical JSON/CSV
        # receipts remain the authoritative record of this execution.
        for path, content in notes.items():
            if path.read_bytes() != content:
                path.write_bytes(content)
    check_inputs(args.manifest.resolve(), postflight=True)
    print(f"Frozen {args.command} passed: scope={args.only}, verification={'quick' if args.quick else 'all'}, "
          f"{len(steps)} subprocesses, {time.monotonic()-started:.1f}s. No source refresh performed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Replication stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
