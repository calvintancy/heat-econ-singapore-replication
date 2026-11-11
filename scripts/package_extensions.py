"""Package the local extension without resealing or changing v1.0.0 references.

Standard-library commands:
  inventory --root .                         # no archives and no source seal
  build --source-manifest PATH --out-dir dist/extension-package
  extract --manifest DIR/heat-econ-extension-manifest.json --archive-dir DIR --root NEW
  verify --manifest DIR/heat-econ-extension-manifest.json --root NEW

ZIP parts have nonoverlapping members and a conservative asset-size limit.
Extract into a separate directory, keeping transport files outside that tree.
These commands verify bytes, not clean-clone native analytical reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile

import package_replication as original_package
from restore_replication import _relative, _safe_target
import seal_extension_sources as source_inventory

ROOT = Path(__file__).resolve().parents[1]
V1_MANIFEST_SHA256 = "ccac81c8c83f9bec0cba2849f210adf1f732710bd6a72acab5c501702986cfcd"
REFERENCES = ("outputs/covid_comparison/original_inputs_reference.json", "outputs/pyfixest/migration_input_reference.json", "data/reference_before_nsa_inputs/manifest.json")
EMBEDDED_INVENTORY = "outputs/extension_package/files_manifest.json"
DEFAULT_NAME = "heat-econ-extension"
DEFAULT_LIMIT = 1_800_000_000
RESERVE = 16 * 1024 * 1024
EXTRA_DIRS = {"dist", ".codex", ".agents", ".Rproj.user", ".quarto", ".ipynb_checkpoints", ".mcp-venv", "mcp-venv", "stata-install"}
LICENSE_NAMES = {"stata.lic", "stata.license", "stata.licence", "license.dat", "stata.serial", "stata_auth.json"}
ALREADY_COMPRESSED = {".npz", ".gz", ".zip", ".png", ".jpg", ".jpeg", ".pdf", ".woff", ".woff2"}
PRIVATE_IDENTIFIERS = re.compile(
    rb"(?:serial[ _]+(?:number|no\.?)|licen[cs]e[ _]+(?:number|no\.?))\s*[:=]\s*(?:[0-9][ -]?){6,}"
    rb"|(?:\b|\\[rn])(?:sk-(?:proj-|ant-)[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}"
    rb"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b", re.I)
REQUIRED_RECEIPTS = ["outputs/collection/verification.json", "outputs/collection/extension_verification.json", "outputs/collection/final_audit_closure_verification.json",
    "outputs/crisis_sensitivity/report_verification.json", "outputs/literature_sensitivity/report_verification.json", "outputs/extension_stationarity/report_verification.json",
    "outputs/calendar_hac/verification.json", "outputs/clustered_sector_sensitivity_qualified/verification.json", "outputs/aea_audit/narrative_corrections.json",
    "outputs/derived_proof_guards/verification.json",
    "outputs/visitor_rigour/review_equivalence.json",
    "outputs/visitor_rigour/selected_review_benchmarks.json",
    "outputs/visitor_rigour/monthly/verification.json", "outputs/visitor_rigour/origin_year/verification.json", "outputs/visitor_rigour/annual/verification.json", "outputs/visitor_rigour/ppml/verification.json",
    "outputs/visitor_rigour/diagnostics/verification.json", "outputs/visitor_rigour/diagnostics/report_verification.json", "outputs/visitor_rigour/report_verification.json",
    "outputs/stata_key_results/verification.json", "outputs/stata_key_results/report_verification.json", "outputs/stata_key_results/arrivals/verification.json"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def encoded(value): return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def scan_distributable_text(path, name):
    """Reject concrete private identifiers without printing their contents."""
    if path.suffix.lower() in ALREADY_COMPRESSED: return
    previous = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            window = previous + block
            if original_package.SECRET_CONTENT.search(window) or PRIVATE_IDENTIFIERS.search(window):
                raise ValueError("Private credential/license identifier detected; refusing package: " + name)
            previous = window[-512:]


def excluded(path, relative, group):
    parts = Path(relative).parts
    if any(part in EXTRA_DIRS for part in parts): return "local runtime/cache/review scratch directory"
    if relative.startswith("outputs/extension_package/"): return "generated package inventory; a fresh bootstrap copy is supplied"
    if path.name.lower() in LICENSE_NAMES or path.name.lower().startswith("stata-license"): return "proprietary Stata license/credential artifact"
    if relative == "replication/manifest.json": return None  # preserve the immutable v1 identity, not the old source snapshot
    return original_package._excluded(path, relative, "source" if group == "sources" else group)


def walk(root, directory):
    blocked = original_package.BLOCKED_DIRS | EXTRA_DIRS
    if not directory.exists(): return
    for base, directories, files in os.walk(directory, followlinks=False):
        directories.sort();files.sort();parent = Path(base)
        for name in list(directories):
            item = parent / name
            if item.is_symlink(): raise ValueError("Symlink refused: " + str(item))
            if name in blocked:
                directories.remove(name);yield item
        for name in files:
            item = parent / name
            if item.is_symlink(): raise ValueError("Symlink refused: " + str(item))
            yield item


def discover_sources(root, source_manifest):
    previous = source_inventory.ROOT
    try:
        source_inventory.ROOT = root
        return source_inventory.source_paths(source_manifest or root / "replication/extensions/source_manifest.json")
    finally: source_inventory.ROOT = previous


def check_source(root, manifest_path):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest_path.relative_to(root)
    manifest = json.loads(manifest_path.read_text())
    actual, expected = set(discover_sources(root, manifest_path)), set(manifest["sources_sha256"])
    if actual != expected:
        raise ValueError("Sealed source set differs: " + json.dumps({"added": sorted(actual - expected), "missing": sorted(expected - actual)}))
    for name, digest in manifest["sources_sha256"].items():
        path = _safe_target(root, name)
        if sha(path) != digest: raise ValueError("Sealed source changed: " + name)
    if manifest["published_v1_manifest_sha256"] != V1_MANIFEST_SHA256: raise ValueError("Source revision is not bound to the immutable v1 manifest")
    for name, digest in manifest["original_references_sha256"].items():
        if sha(_safe_target(root, name)) != digest: raise ValueError("Original reference changed: " + name)
    return manifest


def original_identity(root, hash_inputs):
    path = root / "replication/manifest.json"
    if sha(path) != V1_MANIFEST_SHA256: raise ValueError("Immutable published v1.0.0 manifest changed")
    manifest = json.loads(path.read_text());members = manifest["archives"]["inputs"]["members"]
    expected = {row["path"]: row for group in ["inputs", "outputs"] for row in manifest["archives"][group]["members"]}
    for row in members:
        actual = _safe_target(root, row["path"])
        if not actual.is_file() or actual.stat().st_size != row["size_bytes"]: raise ValueError("Frozen original input missing/size changed: " + row["path"])
        if hash_inputs and sha(actual) != row["sha256"]: raise ValueError("Frozen original input changed: " + row["path"])
    references = {}
    for name in REFERENCES:
        digest = expected[name]["sha256"]
        if sha(_safe_target(root, name)) != digest: raise ValueError("Original frozen reference changed: " + name)
        references[name] = digest
    return manifest, {"manifest_path": "replication/manifest.json", "manifest_sha256": V1_MANIFEST_SHA256,
        "frozen_input_files": len(members), "frozen_input_hashes_checked": hash_inputs, "original_reference_sha256": references}


def readiness(root):
    records, problems = {}, []
    for name in REQUIRED_RECEIPTS:
        path = root / name
        if not path.is_file(): problems.append("missing " + name);continue
        try: value = json.loads(path.read_text())
        except (ValueError, OSError): problems.append("unreadable " + name);continue
        if name == "outputs/collection/verification.json":
            # The unchanged legacy verifier predates status fields.
            valid = value.get("html_sha256") == sha(root / "outputs/heat_economy.html") and value.get("overview_rows_checked") == 81
        else: valid = value.get("status") == "passed"
        if not valid: problems.append(name + ": missing/current passed proof")
        else: records[name] = sha(path)
    equivalence_name = "outputs/visitor_rigour/review_equivalence.json"
    if equivalence_name in REQUIRED_RECEIPTS and equivalence_name in records:
        try:
            proof = json.loads((root / equivalence_name).read_text())
            baseline = root / "replication/extensions/review_evidence/arrivals_review_baseline.json"
            revision = root / "replication/extensions/review_evidence/derived_proof_guard_revision.json"
            guard_helper = root / "scripts/verify_derived_proof_guards.py"
            assert proof["baseline_sha256"] == sha(baseline)
            assert proof["verifier_source_sha256"] == sha(root / "scripts/verify_arrivals_review_equivalence.py")
            assert isinstance(proof["verified_at_utc"], str) and proof["verified_at_utc"]
            for field, filename in [("current_native_fit_logs_sha256", "fit_log.csv"), ("current_native_estimator_manifests_sha256", "estimator_manifest.json")]:
                mapping = proof[field]
                assert set(mapping) == {f"outputs/visitor_rigour/{family}/pyfixest/{filename}" for family in ["monthly", "annual", "ppml", "origin_year"]}
                assert all(sha(root / name) == digest for name, digest in mapping.items())
            assert proof["guard_revision_sha256"] == sha(revision) and proof["guard_verifier_sha256"] == sha(guard_helper)
            assert proof["numeric_or_status_differences"] == [] and proof["native_fits_executed_by_this_verifier"] == 0
            assert proof["semantic_files_checked"] == 534 and proof["npz_arrays_checked"] == 8338 and proof["source_bindings_checked"] == 18
            assert sum(row["requests"] for row in proof["counts"].values()) == 354
            assert sum(row["native_fits"] for row in proof["counts"].values()) == 319
            closure_name = "outputs/collection/final_audit_closure_verification.json"
            if closure_name in records:
                closure = json.loads((root / closure_name).read_text())
                assert closure["arrival_review_equivalence"]["receipt_sha256"] == records[equivalence_name]
                assert closure["arrival_review_equivalence"]["baseline_sha256"] == sha(baseline)
                assert closure["verifier_sha256"] == sha(root / "scripts/verify_final_audit_closure.py")
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            problems.append(equivalence_name + ": stale/incomplete baseline, verifier, count or closure binding")
    selected_name = "outputs/visitor_rigour/selected_review_benchmarks.json"
    if selected_name in REQUIRED_RECEIPTS and selected_name in records:
        try:
            proof = json.loads((root / selected_name).read_text())
            helper = root / "replication/extensions/review_evidence/arrivals_selected_benchmarks.py"
            baseline = root / "replication/extensions/review_evidence/arrivals_review_baseline.json"
            assert proof["source_sha256"] == sha(helper) and proof["baseline_sha256"] == sha(baseline)
            assert proof["native_fits_executed"] == 0 and len(proof["designs"]) == 2
            expected = {"region": (8, 2026111102, [1045, 1948]), "country": (41, 2026111103, [4000, 4000])}
            seen = set()
            for design in proof["designs"]:
                kind = design["spec_id"].split("__")[0]
                restrictions, seed, rejected = expected[kind]
                assert kind not in seen;seen.add(kind)
                assert design["q"] == restrictions and design["t"] == 207 and design["reps"] == 4000 and design["seed"] == seed
                assert [row["hac_lags"] for row in design["results"]] == [12, 24]
                assert [row["reject_at_nominal_5pct"] for row in design["results"]] == rejected
                assert all(row["available"] == 4000 and row["unavailable"] == 0 for row in design["results"])
            assert seen == set(expected)
            closure_name = "outputs/collection/final_audit_closure_verification.json"
            if closure_name in records:
                closure = json.loads((root / closure_name).read_text())
                selected = closure["arrival_selected_reference_diagnostics"]
                assert selected["current_execution_sha256"] == records[selected_name]
                assert selected["helper_source_sha256"] == sha(helper) and selected["baseline_sha256"] == sha(baseline)
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            problems.append(selected_name + ": stale/incomplete selected-reference source, baseline, counts or closure binding")
    review = root / "replication/extensions/review_evidence/fable52-arrivals-receipt.json"
    if not review.is_file(): problems.append("missing completed exact arrivals review")
    else:
        value = json.loads(review.read_text())
        if not (value.get("status") == "complete" and value.get("exitCode") == 0 and value.get("modelVerified") is True
                and value.get("reportedModel") == value.get("model") == "claude-fable-5-2" and value.get("effort") == "max"
                and value.get("provider") == "claude" and value.get("modelEvidence") == "provider-report"
                and value.get("mode") == "isolated-write" and value.get("writerSandbox") is not None):
            problems.append("arrivals review incomplete/unverified")
        for name in ["fable52-arrivals-review.md", "arrivals_adjudication.md"]:
            if not (review.parent / name).is_file(): problems.append("missing " + name)
    return records, problems


def collect(root=ROOT, source_manifest=None, hash_files=False, require_ready=False):
    root = Path(root).resolve();source_manifest = Path(source_manifest).resolve() if source_manifest else None
    published, identity = original_identity(root, hash_files)
    if source_manifest:
        sealed = check_source(root, source_manifest)
        source_names = sorted(sealed["sources_sha256"])
    else: source_names = discover_sources(root, None)
    candidates, omitted = {}, []
    def add(name, group, mandatory=False):
        path = _safe_target(root, name)
        reason = excluded(path, name, group)
        if reason:
            omitted.append({"path": name, "group": group, "reason": reason})
            if mandatory: raise ValueError("Required analytical/source file cannot be omitted: " + name + " (" + reason + ")")
            return
        if not path.is_file():
            if mandatory: raise ValueError("Required package file missing: " + name)
            return
        if path.is_symlink(): raise ValueError("Symlink refused: " + name)
        scan_distributable_text(path, name)
        if name not in candidates: candidates[name] = {"path": name, "size_bytes": path.stat().st_size, "groups": []}
        if group not in candidates[name]["groups"]: candidates[name]["groups"].append(group)
    for name in source_names: add(name, "sources", mandatory=True)
    add("replication/manifest.json", "sources", mandatory=True)
    if source_manifest: add(source_manifest.relative_to(root).as_posix(), "metadata", mandatory=True)
    for row in published["archives"]["inputs"]["members"]: add(row["path"], "inputs", mandatory=True)
    required_original_outputs = {row["path"] for row in published["archives"]["outputs"]["members"]}
    for name in required_original_outputs: add(name, "outputs", mandatory=True)
    for path in walk(root, root / "outputs"):
        add(path.relative_to(root).as_posix(), "outputs")
    for name in ["replication/extensions/supplementary_inputs/calendar/T2007e.txt", "replication/extensions/supplementary_inputs/calendar/sources.json"]:
        add(name, "sources", mandatory=True)
    entries = sorted(candidates.values(), key=lambda row: row["path"])
    if hash_files:
        for entry in entries: entry["sha256"] = sha(root / entry["path"])
    receipts, problems = readiness(root)
    if require_ready and problems: raise ValueError("Package is not ready:\n" + "\n".join(problems))
    if require_ready:
        closure = json.loads((root / "outputs/collection/final_audit_closure_verification.json").read_text())
        if closure["master_sha256"] != sha(root / "outputs/heat_economy.html"): raise ValueError("Final master closure receipt is stale")
    return {"root": str(root), "files": entries, "excluded": sorted(omitted, key=lambda row: (row["path"], row["group"])),
        "original_v1": identity, "source_revision": {"path": source_manifest.relative_to(root).as_posix(), "sha256": sha(source_manifest)} if source_manifest else None,
        "readiness_receipts": receipts, "not_ready": problems, "hashes_computed": hash_files}


def parts(entries, limit):
    budget = limit - RESERVE
    if budget <= 0: raise ValueError("Part limit must exceed the metadata reserve")
    result, current, size = [], [], 0
    for entry in entries:
        if entry["size_bytes"] > budget: raise ValueError("Single file exceeds safe asset budget: " + entry["path"])
        if current and size + entry["size_bytes"] > budget: result.append(current);current, size = [], 0
        current.append(entry);size += entry["size_bytes"]
    if current: result.append(current)
    return result


def write_zip(root, destination, entries, embedded):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for entry in entries:
            info = zipfile.ZipInfo(entry["path"], date_time=(1980, 1, 1, 0, 0, 0));info.create_system = 3;info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED if Path(entry["path"]).suffix.lower() in ALREADY_COMPRESSED else zipfile.ZIP_DEFLATED
            digest, count = hashlib.sha256(), 0
            stream = io.BytesIO(embedded) if entry["path"] == EMBEDDED_INVENTORY else (root / entry["path"]).open("rb")
            with stream, archive.open(info, "w", force_zip64=True) as output:
                for block in iter(lambda: stream.read(1024 * 1024), b""): output.write(block);digest.update(block);count += len(block)
            if count != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]: raise ValueError("File changed during archive write: " + entry["path"])


def build(root, source_manifest, out_dir, name=DEFAULT_NAME, limit=DEFAULT_LIMIT):
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name): raise ValueError("Unsafe package name")
    if (out_dir == root or (root in out_dir.parents and out_dir != root / "dist" and root / "dist" not in out_dir.parents)):
        raise ValueError("Build transport files outside the project or below its ignored dist directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob(name + "-*")): raise ValueError("Package transport already exists; use a new output directory/name")
    plan = collect(root, source_manifest, hash_files=True, require_ready=True)
    tree = {"schema_version": 1, "package_kind": "heat-economy-local-extension", "files": plan["files"], "excluded": plan["excluded"],
        "original_v1": plan["original_v1"], "source_revision": plan["source_revision"], "readiness_receipts": plan["readiness_receipts"],
        "self_digest_excluded": EMBEDDED_INVENTORY, "reproduction_claim": "Byte/source integrity only. No claim of clean-clone native reproduction until a separate execution receipt establishes it",
        "scope": "Current sealed extension sources/evidence, unchanged original frozen inputs/references, official supplementary calendar and current analytical/report outputs; no upstream refresh, software installs, credentials or proprietary license"}
    embedded = encoded(tree);extra = {"path": EMBEDDED_INVENTORY, "size_bytes": len(embedded), "sha256": hashlib.sha256(embedded).hexdigest(), "groups": ["metadata"]}
    groups = parts([extra] + plan["files"], limit)
    manifest = {**tree, "files": [extra] + plan["files"], "archives": [], "max_archive_bytes": limit, "nonoverlapping_archive_members": True}
    with tempfile.TemporaryDirectory(prefix=".extension-package-", dir=out_dir) as temporary:
        stage = Path(temporary)
        for index, group in enumerate(groups, 1):
            path = stage / f"{name}-part-{index:03d}.zip";write_zip(root, path, group, embedded)
            if path.stat().st_size > limit: raise ValueError("Compressed ZIP exceeds asset limit; lower --part-bytes")
            manifest["archives"].append({"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha(path), "members": group, "n_files": len(group)})
        for entry in plan["files"]:
            if sha(root / entry["path"]) != entry["sha256"]: raise ValueError("Tree changed while packaging: " + entry["path"])
        check_source(root, source_manifest);original_identity(root, True)
        manifest_path = stage / f"{name}-manifest.json";manifest_path.write_bytes(encoded(manifest))
        checksum_rows = [f"{row['sha256']}  {row['filename']}" for row in manifest["archives"]] + [f"{sha(manifest_path)}  {manifest_path.name}"]
        (stage / f"{name}-SHA256SUMS").write_text("\n".join(checksum_rows) + "\n")
        for path in stage.iterdir(): os.replace(path, out_dir / path.name)
    return manifest


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("package_kind") != "heat-economy-local-extension": raise ValueError("Unsupported extension package manifest")
    seen = set()
    for entry in manifest["files"]:
        path = _relative(entry["path"]).as_posix()
        if path in seen: raise ValueError("Duplicate package member: " + path)
        seen.add(path)
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]): raise ValueError("Invalid package size/hash: " + path)
    if manifest["original_v1"]["manifest_sha256"] != V1_MANIFEST_SHA256: raise ValueError("Package does not retain the pinned original manifest")
    if "archives" in manifest:
        members, archive_names = [], set()
        for archive in manifest["archives"]:
            if Path(archive["filename"]).name != archive["filename"] or not archive["filename"].endswith(".zip"): raise ValueError("Unsafe archive name")
            if archive["filename"] in archive_names: raise ValueError("Duplicate archive name")
            archive_names.add(archive["filename"])
            if (not isinstance(archive["size_bytes"], int) or archive["size_bytes"] < 0
                    or archive["size_bytes"] > manifest["max_archive_bytes"]
                    or not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])): raise ValueError("Invalid archive size/hash")
            if archive["n_files"] != len(archive["members"]): raise ValueError("Archive member count differs")
            members.extend(archive["members"])
        if sorted(members, key=lambda row: row["path"]) != sorted(manifest["files"], key=lambda row: row["path"]): raise ValueError("Archive member partition differs from global file inventory")


def verify_tree(manifest, root):
    root = Path(root).resolve();validate_manifest(manifest)
    for entry in manifest["files"]:
        path = _safe_target(root, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["size_bytes"] or sha(path) != entry["sha256"]: raise ValueError("Extracted file integrity failed: " + entry["path"])
    original_identity(root, True)
    check_source(root, root / manifest["source_revision"]["path"])
    return {"status": "passed_file_integrity", "files": len(manifest["files"]), "original_references_unchanged": True,
        "scope": "Byte/hash/source integrity; no native analytical reproduction performed"}


def verify_archives(manifest, archive_dir, stage=None):
    validate_manifest(manifest)
    if not manifest.get("archives"): raise ValueError("Transport archive records require the detached package manifest")
    for record in manifest["archives"]:
        path = Path(archive_dir) / record["filename"]
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha(path) != record["sha256"]: raise ValueError("Archive integrity failed: " + str(path))
        expected = {row["path"]: row for row in record["members"]};seen = set()
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                rel = _relative(info.filename).as_posix();mode = info.external_attr >> 16
                if info.is_dir() or (mode and stat.S_IFMT(mode) not in {0, stat.S_IFREG}) or rel not in expected or rel in seen: raise ValueError("Unexpected/nonregular/duplicate ZIP member: " + rel)
                seen.add(rel);entry = expected[rel]
                if info.file_size != entry["size_bytes"]: raise ValueError("ZIP member size differs: " + rel)
                digest, size = hashlib.sha256(), 0
                target = _safe_target(stage, rel) if stage else None
                if target: target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source:
                    output = target.open("xb") if target else None
                    try:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block);size += len(block)
                            if output: output.write(block)
                    finally:
                        if output: output.close()
                if size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]: raise ValueError("ZIP member hash differs: " + rel)
        if seen != set(expected): raise ValueError("ZIP is missing declared members: " + record["filename"])
    return {"status": "passed_archive_integrity", "archives": len(manifest["archives"])}


def extract(manifest, archive_dir, root):
    root = Path(root).resolve();root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and any(root.iterdir()):
        # Idempotent verification is safe; never silently overwrite a checkout.
        return verify_tree(manifest, root)
    with tempfile.TemporaryDirectory(prefix=".extension-extract-", dir=root.parent) as temporary:
        stage = Path(temporary);verify_archives(manifest, archive_dir, stage)
        result = verify_tree(manifest, stage)
        root.mkdir(parents=True, exist_ok=True)
        for child in stage.iterdir(): shutil.move(str(child), root / child.name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "build", "verify", "extract"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist/extension-package")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--archives-only", action="store_true")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--part-bytes", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    if args.command == "inventory":
        plan = collect(args.root, args.source_manifest)
        counts = {group: sum(group in row["groups"] for row in plan["files"]) for group in ["sources", "inputs", "outputs", "metadata"]}
        result = {"status": "inventory_only_no_archives_no_source_seal", "counts": counts, "total_files": len(plan["files"]),
            "uncompressed_bytes": sum(row["size_bytes"] for row in plan["files"]), "estimated_parts": len(parts(plan["files"], args.part_bytes)),
            "hashes_computed": False, "original_v1": plan["original_v1"], "excluded": plan["excluded"], "not_ready": plan["not_ready"],
            "build_requires_source_manifest": True, "scope": "Planning only; final build checks all input/output/source hashes and readiness receipts"}
    elif args.command == "build":
        if not args.source_manifest: parser.error("build requires --source-manifest from a separately sealed revision; this command never seals")
        result = build(args.root, args.source_manifest, args.out_dir, args.name, args.part_bytes)
        result = {"status": "packaged_verified_files_only", "archives": [{key: row[key] for key in ["filename", "size_bytes", "sha256", "n_files"]} for row in result["archives"]],
            "out_dir": str(args.out_dir), "scope": "No clean-clone native analytical reproduction claim"}
    else:
        path = args.manifest or args.root / EMBEDDED_INVENTORY
        manifest = json.loads(path.read_text())
        if args.command == "extract":
            if not args.archive_dir: parser.error("extract requires --archive-dir and a detached --manifest")
            result = extract(manifest, args.archive_dir, args.root)
        elif args.archives_only:
            if not args.archive_dir: parser.error("--archives-only requires --archive-dir")
            result = verify_archives(manifest, args.archive_dir)
        else: result = verify_tree(manifest, args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
