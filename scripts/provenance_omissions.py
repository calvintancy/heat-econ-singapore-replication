"""Exact third-party citation artifacts omitted from the distributable package.

These files informed the literature review, not the Singapore estimates. Their
historical records stay in the frozen references. An absent artifact is labelled
omitted/unverified, never hash-verified. If present, callers verify it normally.
No directory-wide, suffix-wide or missing-input exception is permitted.
"""
from __future__ import annotations

from pathlib import Path

ARTIFACTS = {
    "data/raw/literature/djo_2012/maketable2.do": {
        "bytes": 7044,
        "sha256": "91d7e0ae4f0f0c2d0decdf52801f2fb117045911f45166d603fa8e524ab47302",
    },
    "data/raw/literature/djo_2012/maketable5.do": {
        "bytes": 53600,
        "sha256": "4217cebb195a95801a47bf0b9b187e8ed7f0952531e3be817ef80fb5cbaf1c36",
    },
    "data/raw/literature/djo_2012/replication.zip": {
        "bytes": 750614,
        "sha256": "1066ad0b36ba5ceb37f533372cf410773148dd9814f61b273372638402691548",
    },
}
REASON = "Third-party Dell–Jones–Olken replication material used only for literature-method provenance; excluded from redistribution, not an analytical input."


def omitted_artifact(root: Path, relative: str, recorded: dict) -> dict | None:
    """Return an explicit unverified record only for an absent, pinned artifact."""
    if (root / relative).exists() or relative not in ARTIFACTS:
        return None
    expected = ARTIFACTS[relative]
    assert recorded.get("bytes") == expected["bytes"] and recorded.get("sha256") == expected["sha256"], (
        relative, "Provenance-only omission does not match the original pinned artifact")
    return {"path": relative, "status": "omitted_unverified", "classification": "provenance_only",
            "reason": REASON, "recorded_bytes": recorded["bytes"], "recorded_sha256": recorded["sha256"],
            "current_artifact_hash_verified": False}
