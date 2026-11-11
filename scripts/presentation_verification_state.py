"""Fail-closed lifecycle for new presentation verification receipts only."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def persist(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def presentation_run(path: Path, scope: str):
    if path.exists():
        previous = path.read_bytes()
        archive = path.parent / "presentation_receipt_history" / (path.stem + "__" + hashlib.sha256(previous).hexdigest() + ".json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists(): assert archive.read_bytes() == previous
        else: archive.write_bytes(previous)
    started = datetime.now(timezone.utc).isoformat()
    persist(path, {"status": "running", "scope": scope, "started_utc": started,
        "claim": "Replacement presentation check has not completed; older receipt is archived, not current proof"})
    try:
        yield
        if json.loads(path.read_text()).get("status") != "passed":
            raise RuntimeError("Presentation checker returned without activating a passed receipt")
    except BaseException as error:
        persist(path, {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "scope": scope,
            "started_utc": started, "error_type": type(error).__name__, "error": str(error)[:2000]})
        raise
