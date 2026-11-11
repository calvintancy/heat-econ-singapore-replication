"""Reproduce the isolated visitor analysis and native gt report."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-only", action="store_true", help="Reuse fitted estimates; rebuild and verify native gt/report output")
    args = parser.parse_args()
    programs = [] if args.reports_only else ["visitor_analysis.py", "visitor_counts.py", "verify_visitor_arrivals.py",
        "visitor_crisis.py", "visitor_inference.py"]
    programs += ["visitor_gt_data.py", "verify_visitor_crisis.py"]
    for program in programs:
        subprocess.run([sys.executable, str(ROOT / "scripts" / program)], cwd=ROOT, check=True)
    subprocess.run(["Rscript", str(ROOT / "scripts/visitor_gt.R"), str(ROOT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/visitor_report.py")], cwd=ROOT, check=True)
    subprocess.run(["node", str(ROOT / "scripts/verify_visitor_report.mjs")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
