"""Build deterministic frozen replication assets without changing original files.

Run only after all source and analytical outputs have been frozen.  The detached
manifest is the release's integrity inventory; it intentionally does not hash
itself.  The portable ZIP carries a bootstrap copy without its own ZIP entry.
This module uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "calvintancy/heat-econ-singapore-replication"
DEFAULT_TAG = "v1.0.0"
MANIFEST_NAME = "replication-manifest.json"
PORTABLE_MANIFEST = "replication/manifest.json"
SOURCE_DIRS = ("scripts", "templates", "notes", "replication", ".github", "tests")
SOURCE_NAMES = {"pyproject.toml", "uv.lock", ".gitignore", ".gitattributes", ".python-version", ".nvmrc", "Makefile", "CITATION.cff"}
BLOCKED_DIRS = {".git", ".venv", ".impeccable", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "R-library", "native-install", "native_install", "cache", ".cache", "logs", "tmp", "temp"}
BLOCKED_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".o", ".a", ".class", ".log", ".tmp", ".swp", ".pem", ".key", ".p12", ".pfx"}
SECRET_NAMES = {".netrc", ".npmrc", ".pypirc", "credentials", "credentials.json", "token", "token.json", "id_rsa", "id_ed25519", ".DS_Store"}
SECRET_CONTENT = re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _excluded(path: Path, relative: str, group: str) -> str | None:
    parts = Path(relative).parts
    name = path.name
    if any(part in BLOCKED_DIRS for part in parts):
        return "runtime/install/cache directory"
    if name in SECRET_NAMES or name.startswith("._") or name.startswith(".env"):
        return "local metadata or credentials"
    if any(word in name.lower() for word in ("credential", "service-account", "service_account")):
        return "credential filename"
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return "compiled/runtime/temporary/credential file"
    if relative in {PORTABLE_MANIFEST, "replication/manifest.sha256", "replication/SHA256SUMS"}:
        return "generated manifest excluded from source hashes to avoid self-reference"
    if "literature" in parts:
        if name != "manifest.json" or path.stat().st_size > 100_000:
            return "third-party literature content (citation manifest only)"
    if group == "inputs" and relative.startswith("data/raw/visitor_arrivals/calendar/") and path.suffix == ".py":
        return "calendar helper belongs to tracked source, not the input archive"
    return None


def _files_below(path: Path):
    if not path.exists():
        return
    if path.is_symlink():
        raise ValueError(f"Refusing symlink in package tree: {path}")
    if path.is_file():
        yield path
        return
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs.sort()
        files.sort()
        for name in list(dirs):
            item = Path(base) / name
            if item.is_symlink():
                raise ValueError(f"Refusing symlink in package tree: {item}")
            if name in BLOCKED_DIRS:
                # Yield a sentinel so that excluded directories remain recorded.
                dirs.remove(name)
                yield item
        for name in files:
            item = Path(base) / name
            if item.is_symlink():
                raise ValueError(f"Refusing symlink in package tree: {item}")
            yield item


def collect_inventory(root: Path = ROOT) -> dict:
    """Return source/input/output inventories and explicit exclusion reasons."""
    root = root.resolve()
    source_roots = [root / name for name in SOURCE_DIRS]
    source_roots += [p for p in root.iterdir() if p.is_file() and (p.name in SOURCE_NAMES or p.suffix == ".md" or p.name.startswith("LICENSE"))]
    source_roots += sorted((root / "data/raw/visitor_arrivals/calendar").glob("*.py"))
    data_roots = [root / "data/raw", root / "data/processed"]
    data_roots += sorted((root / "data").glob("reference_before*"))
    groups = {"source": source_roots, "inputs": data_roots, "outputs": [root / "outputs"]}
    result = {"source": [], "inputs": [], "outputs": [], "excluded": []}
    seen = set()
    for group, roots in groups.items():
        for folder in roots:
            for path in _files_below(folder):
                rel = path.relative_to(root).as_posix()
                key = (group, rel)
                if key in seen:
                    continue
                seen.add(key)
                reason = _excluded(path, rel, group)
                if reason:
                    result["excluded"].append({"path": rel, "reason": reason})
                    continue
                if not path.is_file():
                    continue
                # A high-confidence content guard complements filename exclusions.
                if path.stat().st_size < 5_000_000:
                    if SECRET_CONTENT.search(path.read_bytes()):
                        raise ValueError(f"Credential signature found; package refused: {rel}")
                result[group].append({"path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        result[group].sort(key=lambda row: row["path"])
    result["excluded"].sort(key=lambda row: row["path"])
    all_paths = [r["path"] for group in ("source", "inputs", "outputs") for r in result[group]]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("Source, inputs and outputs overlap")
    return result


class _CheckedReader:
    def __init__(self, file):
        self.file = file
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        data = self.file.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data

    def assert_matches(self, entry):
        if self.size != entry["size_bytes"] or self.digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"File changed while sealing package: {entry['path']}")


def _tar(root: Path, destination: Path, entries: list[dict]):
    with destination.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for entry in entries:
                info = tarfile.TarInfo(entry["path"])
                info.size = entry["size_bytes"]
                info.mode = 0o644
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                with (root / entry["path"]).open("rb") as f:
                    checked = _CheckedReader(f)
                    archive.addfile(info, checked)
                    checked.assert_matches(entry)


def _zip(root: Path, destination: Path, entries: list[dict], portable_manifest: bytes):
    all_entries = sorted(entries + [{"path": PORTABLE_MANIFEST, "size_bytes": len(portable_manifest), "sha256": hashlib.sha256(portable_manifest).hexdigest()}], key=lambda row: row["path"])
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
        for entry in all_entries:
            info = zipfile.ZipInfo(entry["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as target:
                if entry["path"] == PORTABLE_MANIFEST:
                    target.write(portable_manifest)
                else:
                    with (root / entry["path"]).open("rb") as source:
                        checked = _CheckedReader(source)
                        for block in iter(lambda: checked.read(1024 * 1024), b""):
                            target.write(block)
                        checked.assert_matches(entry)
    return all_entries


def _archive_record(path: Path, members: list[dict]) -> dict:
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "n_files": len(members), "uncompressed_bytes": sum(row["size_bytes"] for row in members), "members": members}


def build_replication(root: Path = ROOT, out_dir: Path | None = None, tag: str = DEFAULT_TAG, repo: str = DEFAULT_REPO) -> dict:
    """Seal the frozen tree into three assets and a detached integrity manifest."""
    root = Path(root).resolve()
    out_dir = Path(out_dir or root / "dist/replication").resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not tag or tag.startswith("-"):
        raise ValueError("Invalid repository or release tag")
    inventory = collect_inventory(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "repo": repo, "tag": tag,
        "deterministic_metadata": {"tar_mtime": 0, "zip_datetime": "1980-01-01T00:00:00", "file_mode": "0644", "owner": "0:0", "ordering": "lexicographic POSIX paths"},
        "sources": inventory["source"], "excluded": inventory["excluded"],
        "portable_manifest_self_digest_excluded": True,
        "policy": {"frozen_sources": True, "refresh_current_data": False, "literature": "Citation manifest JSON only; paper PDFs/text/replication code excluded", "source_helpers": "data/raw/visitor_arrivals/calendar/*.py are source-only", "restore": "Only inputs and outputs are restored; tracked source is never overwritten"},
        "archives": {},
    }
    with tempfile.TemporaryDirectory(prefix=".replication-seal-", dir=out_dir) as temp:
        stage = Path(temp)
        for group in ("inputs", "outputs"):
            file = stage / f"replication-{group}.tar.gz"
            _tar(root, file, inventory[group])
            manifest["archives"][group] = _archive_record(file, inventory[group])
        portable = json_bytes(manifest)
        full_file = stage / "replication-full.zip"
        full_members = _zip(root, full_file, inventory["source"] + inventory["inputs"] + inventory["outputs"], portable)
        manifest["archives"]["full"] = _archive_record(full_file, full_members)
        # Detect a concurrent producer rather than publishing mixed vintages.
        for group in ("source", "inputs", "outputs"):
            for row in inventory[group]:
                if sha256_file(root / row["path"]) != row["sha256"]:
                    raise RuntimeError(f"Tree changed during package build: {row['path']}")
        (stage / MANIFEST_NAME).write_bytes(json_bytes(manifest))
        checksums = [f"{row['sha256']}  {row['filename']}" for row in manifest["archives"].values()]
        checksums.append(f"{sha256_file(stage / MANIFEST_NAME)}  {MANIFEST_NAME}")
        (stage / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        for file in stage.iterdir():
            os.replace(file, out_dir / file.name)
    print(f"Sealed {len(inventory['source'])} source, {len(inventory['inputs'])} input and {len(inventory['outputs'])} output files in {out_dir}")
    for row in manifest["archives"].values():
        print(f"{row['filename']}: {row['size_bytes']:,} bytes; SHA256 {row['sha256']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--inventory-only", action="store_true", help="Print counts/exclusions without making archives")
    args = parser.parse_args()
    if args.inventory_only:
        inventory = collect_inventory(args.root)
        print(json.dumps({key: len(value) for key, value in inventory.items()}, indent=2))
        for row in inventory["excluded"]:
            print(f"EXCLUDED {row['path']}: {row['reason']}")
    else:
        build_replication(args.root, args.out_dir, args.tag, args.repo)


if __name__ == "__main__":
    main()
