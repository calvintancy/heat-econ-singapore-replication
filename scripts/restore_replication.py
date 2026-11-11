"""Restore frozen data/results from verified release assets; standard library only.

Examples::
    python3 scripts/restore_replication.py
    python3 scripts/restore_replication.py --archive-dir dist/replication
    python3 scripts/restore_replication.py --verify-only

The default remote operation uses the authenticated GitHub CLI.  It never fetches
current statistical data.  --verify-only checks files on disk (also after simply
unzipping replication-full.zip); --verify-archives checks archive bytes instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "calvintancy/heat-econ-singapore-replication"
DEFAULT_TAG = "v1.0.0"
MANIFEST_NAME = "replication-manifest.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Unsafe manifest/member path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in {".", "..", ""} for part in value.split("/")) or ":" in value:
        raise ValueError(f"Unsafe manifest/member path: {value!r}")
    return p


def _source_path(path: str) -> bool:
    p = _relative(path)
    return (p.parts[0] in {"scripts", "templates", "notes", "replication", ".github", "tests"}
            or len(p.parts) == 1
            or (path.startswith("data/raw/visitor_arrivals/calendar/") and path.endswith(".py")))


def _validate_manifest(manifest: dict):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("sources"), list):
        raise ValueError("Unsupported or incomplete replication manifest")
    for group in ("inputs", "outputs"):
        record = manifest.get("archives", {}).get(group)
        if not isinstance(record, dict) or Path(record.get("filename", "")).name != f"replication-{group}.tar.gz":
            raise ValueError(f"Missing/unsafe {group} archive metadata")
        _validate_entries(record.get("members"), group)
        if record.get("n_files") != len(record["members"]):
            raise ValueError(f"Incorrect member count for {group}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) or not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
            raise ValueError(f"Invalid archive checksum/size for {group}")
    _validate_entries(manifest["sources"], "source")
    every = manifest["sources"] + manifest["archives"]["inputs"]["members"] + manifest["archives"]["outputs"]["members"]
    if len({r["path"] for r in every}) != len(every):
        raise ValueError("Manifest contains overlapping source/input/output paths")


def _validate_entries(entries, group):
    if not isinstance(entries, list):
        raise ValueError(f"Invalid member inventory: {group}")
    seen = set()
    for row in entries:
        path = row.get("path")
        p = _relative(path)
        if path in seen:
            raise ValueError(f"Duplicate manifest member: {path}")
        seen.add(path)
        if group == "source" and not _source_path(path):
            raise ValueError(f"Unexpected source path: {path}")
        if group == "inputs" and (p.parts[0] != "data" or _source_path(path)):
            raise ValueError(f"Input archive cannot overwrite source: {path}")
        if group == "outputs" and p.parts[0] != "outputs":
            raise ValueError(f"Unexpected output archive path: {path}")
        if not isinstance(row.get("size_bytes"), int) or row["size_bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
            raise ValueError(f"Invalid file size/checksum: {path}")


def _safe_target(root: Path, rel: str) -> Path:
    p = root.joinpath(*_relative(rel).parts)
    current = root
    for part in _relative(rel).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing symlink in destination: {current}")
    return p


def verify_files(manifest: dict, root: Path = ROOT, groups=("inputs", "outputs"), include_sources=True) -> dict:
    """Check the exact frozen file hashes without needing archive assets."""
    root = Path(root).resolve()
    _validate_manifest(manifest)
    checks = {}
    if include_sources:
        checks["source"] = manifest["sources"]
    checks.update({group: manifest["archives"][group]["members"] for group in groups})
    for group, entries in checks.items():
        failures = []
        for row in entries:
            p = _safe_target(root, row["path"])
            if not p.is_file():
                failures.append(f"missing {row['path']}")
            elif p.stat().st_size != row["size_bytes"] or _sha(p) != row["sha256"]:
                failures.append(f"changed {row['path']}")
        if failures:
            raise ValueError(f"{group} integrity failed ({len(failures)} files):\n" + "\n".join(failures[:20]))
        print(f"Verified {group} hashes: {len(entries):,} frozen files")
    return {group: len(entries) for group, entries in checks.items()}


def _tracked_files(root: Path) -> set[str]:
    if not shutil.which("git"):
        return set()
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=False)
    if result.returncode:
        return set()
    return set(result.stdout.decode("utf-8").rstrip("\x00").split("\x00")) - {""}


def _download(repo: str, tag: str, files: list[str], directory: Path):
    if not shutil.which("gh"):
        raise RuntimeError("GitHub CLI is needed for private release downloads. Install gh and run `gh auth login`, or provide --archive-dir with downloaded assets.")
    command = ["gh", "release", "download", tag, "--repo", repo, "--dir", str(directory)]
    for filename in files:
        command += ["--pattern", filename]
    subprocess.run(command, check=True)


def _stage_archive(path: Path, record: dict, stage: Path | None = None):
    if not path.is_file() or path.stat().st_size != record["size_bytes"] or _sha(path) != record["sha256"]:
        raise ValueError(f"Archive checksum/size mismatch: {path}")
    expected = {row["path"]: row for row in record["members"]}
    seen = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            rel = _relative(member.name).as_posix()
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"Only regular archive files are permitted: {member.name}")
            if rel not in expected or rel in seen:
                raise ValueError(f"Unexpected or duplicate archive member: {rel}")
            row = expected[rel]
            if member.size != row["size_bytes"]:
                raise ValueError(f"Archive member size mismatch: {rel}")
            seen.add(rel)
            digest = hashlib.sha256()
            actual_size = 0
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archive member: {rel}")
            output = None
            try:
                if stage is not None:
                    target = _safe_target(stage, rel)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = target.open("xb")
                with stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        actual_size += len(block)
                        if output is not None:
                            output.write(block)
            finally:
                if output is not None:
                    output.close()
            if actual_size != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"Archive member checksum mismatch: {rel}")
    if seen != set(expected):
        raise ValueError(f"Archive is missing {len(set(expected)-seen)} expected members: {path}")
    print(f"Verified archive: {path.name} ({len(seen):,} members)")


def restore_replication(archive_dir: Path | None = None, root: Path = ROOT, repo: str = DEFAULT_REPO,
                        tag: str = DEFAULT_TAG, groups=("inputs", "outputs"), manifest_path: Path | None = None,
                        verify_only=False, verify_archives=False, skip_source_check=False) -> dict:
    """Verify then restore a fixed release; original source files are immutable."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Destination repository does not exist: {root}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not tag or tag.startswith("-"):
        raise ValueError("Invalid repository or release tag")
    if not groups or len(set(groups)) != len(groups) or any(g not in {"inputs", "outputs"} for g in groups):
        raise ValueError("Select inputs and/or outputs exactly once")
    if skip_source_check and not verify_only:
        raise ValueError("--skip-source-check is permitted only with --verify-only")
    with tempfile.TemporaryDirectory(prefix="heat-econ-assets-") as temporary:
        downloads = Path(temporary)
        assets = Path(archive_dir).resolve() if archive_dir is not None else downloads
        candidates = ([Path(manifest_path)] if manifest_path is not None else
                      [root / "replication/manifest.json", assets / MANIFEST_NAME])
        file = next((p for p in candidates if p.is_file()), None)
        if file is None:
            if manifest_path is not None or archive_dir is not None:
                raise FileNotFoundError("Replication manifest not found; supply --manifest or include replication-manifest.json beside archives")
            _download(repo, tag, [MANIFEST_NAME], downloads)
            file = downloads / MANIFEST_NAME
        manifest = json.loads(file.read_text(encoding="utf-8"))
        _validate_manifest(manifest)
        if manifest.get("tag") != tag or manifest.get("repo") != repo:
            raise ValueError("Manifest release/repository does not match requested --tag/--repo")
        # Source integrity is checked first and no extraction operation targets it.
        if not skip_source_check:
            verify_files(manifest, root, groups=(), include_sources=True)
        if verify_only:
            return {"status": "verified_files", "counts": verify_files(manifest, root, groups, include_sources=False)}
        if archive_dir is None:
            _download(repo, tag, [manifest["archives"][g]["filename"] for g in groups], downloads)
        if verify_archives:
            for group in groups:
                record = manifest["archives"][group]
                _stage_archive(assets / record["filename"], record)
            return {"status": "verified_archives", "groups": list(groups)}
        tracked = _tracked_files(root)
        for group in groups:
            for row in manifest["archives"][group]["members"]:
                if row["path"] in tracked or _source_path(row["path"]):
                    raise ValueError(f"Refusing to overwrite tracked/source file: {row['path']}")
                target = _safe_target(root, row["path"])
                if target.exists() and not target.is_file():
                    raise ValueError(f"Destination is not a regular file: {target}")
        # All archives are staged and verified before the first destination write.
        with tempfile.TemporaryDirectory(prefix=".heat-econ-restore-", dir=root.parent) as temporary_stage:
            stage = Path(temporary_stage)
            for group in groups:
                record = manifest["archives"][group]
                _stage_archive(assets / record["filename"], record, stage)
            for group in groups:
                for row in manifest["archives"][group]["members"]:
                    target = _safe_target(root, row["path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(stage / row["path"], target)
                    target.chmod(0o644)
        counts = verify_files(manifest, root, groups, include_sources=False)
        print(f"Restored frozen {tag}; no current data was fetched and source files were preserved.")
        return {"status": "restored", "counts": counts, "tag": tag, "repo": repo}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", "--assets-dir", dest="archive_dir", type=Path)
    parser.add_argument("--manifest", dest="manifest_path", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--groups", nargs="+", choices=("inputs", "outputs"), default=["inputs", "outputs"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Verify source/data/output files already on disk; archives are unnecessary")
    mode.add_argument("--verify-archives", action="store_true", help="Verify archive and member hashes without restoring files")
    parser.add_argument("--skip-source-check", action="store_true", help="Only with --verify-only: check requested frozen data/results after audits rewrite generated notes")
    args = parser.parse_args()
    restore_replication(**vars(args))


if __name__ == "__main__":
    main()
