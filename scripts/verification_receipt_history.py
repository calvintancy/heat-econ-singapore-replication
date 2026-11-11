"""Keep immutable original audit snapshots separate from fresh verification runs.

Only the eight named auxiliary JSON receipts may be restored. Data, estimates,
covariances, native ledgers and reference files are always checked, never restored.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Callable, Sequence
import uuid

from provenance_omissions import omitted_artifact

REFERENCE = "outputs/covid_comparison/original_inputs_reference.json"
RECEIPT_WRITERS = {
    "outputs/aggregate_va/verification.json": "verify_aggregate_va.py",
    "outputs/extended_controls/verification.json": "verify_extended_controls.py",
    "outputs/granular/verification.json": "verify_granular.py",
    "outputs/granular_interactions/verification.json": "verify_granular_interactions.py",
    "outputs/heterogeneity/diagnostics/verification.json": "stationarity_diagnostics.py",
    "outputs/heterogeneity/verification.json": "verify_heterogeneity.py",
    "outputs/iip/verification.json": "verify_iip.py",
    "outputs/outcomes/verification.json": "verify_outcomes.py",
}
PINNED_RECEIPTS = tuple(RECEIPT_WRITERS)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


class VerificationReceiptHistory:
    """A verification-phase transaction, begun only after any production phase."""

    def __init__(self, root: Path, *, command: str, scope: str,
                 expected_reference_sha256: str | None = None,
                 execution_manifest: Path | None = None):
        self.root = Path(root).resolve()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+"-"+uuid.uuid4().hex[:8]
        self.directory = self.root/"replication/logs/verification-runs"/run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.reference_bytes: bytes | None = None
        self.reference: dict = {}
        self.snapshots: dict[str, bytes] = {}
        self.index = {
            "schema_version": 1, "run_id": run_id, "status": "running",
            "command": command, "scope": scope, "started_utc": _utc(),
            "reference_path": REFERENCE, "reference_sha256": None,
            "expected_published_reference_sha256": expected_reference_sha256,
            "reference_origin": "published manifest" if expected_reference_sha256 else "post-production checkpoint",
            "receipt_policy": "Only eight explicitly named auxiliary verification JSONs are restored. Analytical inputs/results/references remain strict and are never restored.",
            "historical_receipts": {}, "current_rechecks": {}, "steps": [],
        }
        self._persist()
        try:
            if execution_manifest is not None:
                manifest_bytes = Path(execution_manifest).read_bytes()
                _write(self.directory/"execution_manifest.json", manifest_bytes)
                self.index["execution_manifest_sha256"] = _sha(manifest_bytes)
                self.index["execution_manifest_archive"] = "execution_manifest.json"
            self.reference_bytes = (self.root/REFERENCE).read_bytes()
            if expected_reference_sha256 and _sha(self.reference_bytes) != expected_reference_sha256:
                raise RuntimeError("Original-input checkpoint differs from the published manifest")
            self.reference = json.loads(self.reference_bytes)
            self.index["reference_sha256"] = _sha(self.reference_bytes)
            if not set(PINNED_RECEIPTS) <= self.reference["files"].keys():
                raise RuntimeError("Frozen checkpoint does not contain the eight required auxiliary receipts")
            self.index["precheck"] = self.check_checkpoint()
            for relative in PINNED_RECEIPTS:
                content = (self.root/relative).read_bytes()
                self.snapshots[relative] = content
                archived = self.directory/"historical"/relative
                _write(archived, content)
                self.index["historical_receipts"][relative] = {
                    "sha256": _sha(content), "archive": str(archived.relative_to(self.directory)),
                    "role": "Immutable historical checkpoint record; not the current execution's proof",
                }
                self.index["current_rechecks"][relative] = {"status": "not_run", "writer": RECEIPT_WRITERS[relative]}
            self._persist()
        except BaseException as error:
            self.fail(error)
            raise

    def check_checkpoint(self) -> dict:
        """Check every original file, including auxiliary bytes before/after steps."""
        if (self.root/REFERENCE).read_bytes() != self.reference_bytes:
            raise RuntimeError("Immutable original-input checkpoint changed during verification")
        omitted, checked = [], 0
        for relative, recorded in self.reference["files"].items():
            path = self.root/relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
                raise RuntimeError(f"Unsafe original checkpoint path: {relative}")
            omission = omitted_artifact(self.root, relative, recorded)
            if omission is not None:
                omitted.append(omission)
                continue
            if not path.is_file():
                raise RuntimeError(f"Frozen original input/result missing: {relative}")
            content = path.read_bytes()
            if len(content) != recorded["bytes"] or _sha(content) != recorded["sha256"]:
                raise RuntimeError(f"Frozen original input/result changed: {relative}")
            checked += 1
        return {"checked_files": checked, "omitted_unverified": omitted,
                "reference_sha256": self.index["reference_sha256"]}

    def _persist(self) -> None:
        _write(self.directory/"index.json", (json.dumps(self.index, indent=2, allow_nan=False)+"\n").encode())
        rows = []
        for path, entry in self.index["current_rechecks"].items():
            current = html.escape(entry["status"])
            if entry.get("archive"):
                current += ' · <a href="'+html.escape(entry["archive"], quote=True)+'">observed recheck receipt</a>'
            historical = self.index["historical_receipts"].get(path, {})
            link = '<a href="'+html.escape(historical.get("archive", ""), quote=True)+'">frozen historical record</a>'
            rows.append("<tr><td>"+html.escape(path)+"</td><td>"+current+"</td><td>"+link+"</td></tr>")
        document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Verification run</title>'
            '<style>body{font:16px system-ui;margin:3rem;max-width:1100px}td,th{padding:.6rem;border:1px solid #ccc;text-align:left}table{border-collapse:collapse}code{overflow-wrap:anywhere}</style>'
            '<h1>Verification run: '+html.escape(self.index["status"])+"</h1><p><code>"+html.escape(self.index["run_id"])+"</code></p>"
            '<p>Historical receipts retain their original checkpoint bytes. Only the separately labelled rechecks are evidence from this run. '
            'An observed receipt payload saying “passed” does not override a failed subprocess or run.</p>'
            '<p><a href="index.json">Commands, exit status, input/reference hashes and receipt history</a></p>'
            '<table><thead><tr><th>Receipt</th><th>Current recheck</th><th>Historical snapshot</th></tr></thead><tbody>'+"".join(rows)+"</tbody></table></html>")
        _write(self.directory/"index.html", document.encode())

    def run_step(self, command: Sequence[str], operation: Callable[[], None]) -> None:
        """Archive observed receipts and restore only the allowlist after one child."""
        if self.index["status"] != "running":
            raise RuntimeError("Cannot append verification to a terminal run")
        self.check_checkpoint()
        step_number = len(self.index["steps"])+1
        step = {"number": step_number, "command": list(command), "status": "running",
                "started_utc": _utc(), "exit_code": None, "error": None, "receipts": {}}
        self.index["steps"].append(step)
        self._persist()
        error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            operation()
        except BaseException as caught:
            error = caught
        try:
            for relative, before in self.snapshots.items():
                source = self.root/relative
                current = source.read_bytes() if source.is_file() and not source.is_symlink() else None
                if current is None:
                    cleanup_error = cleanup_error or RuntimeError(f"Verifier removed or replaced an auxiliary receipt: {relative}")
                archive = self.directory/"steps"/f"{step_number:03d}"/relative
                if current is not None:
                    _write(archive, current)
                step["receipts"][relative] = {
                    "historical_sha256": _sha(before),
                    "observed_sha256": None if current is None else _sha(current),
                    "bytes_changed": current != before,
                    "archive": None if current is None else str(archive.relative_to(self.directory)),
                    "expected_writer_step": Path(command[0]).name == RECEIPT_WRITERS[relative],
                }
        except BaseException as caught:
            cleanup_error = caught
        finally:
            # Atomic replace does not follow a child-created symlink into data.
            for relative, content in self.snapshots.items():
                try:
                    if not (self.root/relative).is_file() or (self.root/relative).is_symlink() or (self.root/relative).read_bytes() != content:
                        _write(self.root/relative, content)
                except BaseException as caught:
                    cleanup_error = cleanup_error or caught
        try:
            step["postcheck"] = self.check_checkpoint()
        except BaseException as caught:
            cleanup_error = cleanup_error or caught
        step["exit_code"] = (getattr(error, "returncode", 130 if isinstance(error, KeyboardInterrupt) else 1) if error else 0)
        step["error"] = None if error is None else f"{type(error).__name__}: {error}"
        step["integrity_or_archive_error"] = None if cleanup_error is None else f"{type(cleanup_error).__name__}: {cleanup_error}"
        step["status"] = "failed" if error or cleanup_error else "passed"
        step["finished_utc"] = _utc()
        for relative, record in step["receipts"].items():
            if record["expected_writer_step"]:
                self.index["current_rechecks"][relative] = {
                    "status": step["status"], "step": step_number,
                    "archive": record["archive"], "sha256": record["observed_sha256"],
                    "writer": RECEIPT_WRITERS[relative], "child_exit_code": step["exit_code"],
                }
        if error or cleanup_error:
            self.index["status"] = "failed"
            self.index["finished_utc"] = _utc()
        self._persist()
        if cleanup_error:
            raise cleanup_error from error
        if error:
            raise error.with_traceback(error.__traceback__)

    def fail(self, error: BaseException) -> None:
        self.index.update(status="failed", finished_utc=_utc(), error=f"{type(error).__name__}: {error}")
        self._persist()

    def complete(self) -> None:
        if self.index["status"] != "running":
            raise RuntimeError("A failed verification run cannot become passed")
        self.index["postcheck"] = self.check_checkpoint()
        self.index.update(status="passed", finished_utc=_utc())
        self._persist()
