"""Replay the archived selected-design arrivals diagnostics; no native fits."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(
        str(root / "replication/extensions/review_evidence/arrivals_selected_benchmarks.py"),
        run_name="__main__",
    )
