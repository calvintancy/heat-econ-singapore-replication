"""Reproduce the Singapore heat/economy analyses, including SA/NSA YoY and annual models."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from fetch_data import fetch
from build_panel import build
from estimate import estimate
from report import report
from verify import verify
from html_report import build_html
from iip_analysis import run_iip
from verify_iip import verify_iip
from regression_tables import build_tables
from iip_report import build_iip_report
from extended_controls import build_controls, fit_controls, build_controls_report
from verify_extended_controls import verify_extended_controls
from outcome_analysis import run_outcomes
from outcome_report import build_outcome_report
from heterogeneity import run_heterogeneity
from stationarity_diagnostics import run_diagnostics
from verify_heterogeneity import verify_heterogeneity
from heterogeneity_report import build_heterogeneity_report
from aggregate_va import run_aggregate_va
from verify_aggregate_va import verify_aggregate_va
from aggregate_va_report import build_aggregate_va_report
from sample_context import build_sample_context
from combined_report import build_combined_report
from verify_collection import verify_collection
from verify_baseline import verify_baseline
from va_reconciliation import run_va_reconciliation
from iip_reconciliation import run_iip_reconciliation
from granular_analysis import run_granular
from granular_interactions import run_granular_interactions
from verify_granular import verify_granular
from verify_sector_cluster import verify_sector_cluster
from verify_cluster_contrasts import verify_cluster_contrasts
from reconciliation_report import build_reconciliation_report
from stationarity_audit import run_stationarity_audit
from stationarity_report import build_stationarity_report
from pyfixest_backend import reset_fit_log, write_fit_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download current official source vintages")
    args = parser.parse_args()
    reset_fit_log()
    fetch(refresh=args.refresh)
    build()
    estimate()
    report()
    verify()
    run_iip()
    verify_iip()
    build_controls()
    fit_controls()
    verify_extended_controls()
    run_outcomes()
    from verify_outcomes import verify_outcomes
    verify_outcomes()
    run_heterogeneity()
    run_diagnostics()
    verify_heterogeneity()
    run_aggregate_va()
    verify_aggregate_va()
    run_va_reconciliation()
    run_iip_reconciliation()
    run_granular()
    verify_granular()
    run_granular_interactions()
    write_fit_log()
    from verify_granular_interactions import verify_granular_interactions
    verify_granular_interactions()
    verify_sector_cluster()
    verify_cluster_contrasts()
    from verify_pyfixest_migration import verify_pyfixest_migration
    verify_pyfixest_migration()
    run_stationarity_audit()
    from verify_stationarity_audit import verify_stationarity_audit
    verify_stationarity_audit()
    from stationarity_context import clear_cache
    clear_cache()
    build_sample_context()
    build_controls_report()
    build_tables()
    build_iip_report()
    build_html()
    build_outcome_report()
    build_heterogeneity_report()
    build_aggregate_va_report()
    from granular_report import build_granular_reports
    build_granular_reports()
    build_reconciliation_report()
    build_stationarity_report()
    from annual_analysis import run_annual_analysis
    from verify_annual import verify_annual
    from annual_report import build_annual_report
    from verify_annual_report import verify_annual_report
    run_annual_analysis()
    verify_annual()
    build_annual_report()
    verify_annual_report()
    from covid_comparison import run_covid_comparison
    run_covid_comparison(refresh_reference=True,package=False)
    # Keep the explicitly requested sector-cluster sensitivity in its own native
    # estimator process/ledger; it must not pollute the original 1,510-fit audit.
    from replicate import FROZEN_LAUNCHER
    project_root = Path(__file__).resolve().parents[1]
    for helper in ("clustered_sector_sensitivity.py", "verify_clustered_sector_sensitivity.py"):
        subprocess.run([sys.executable, "-c", FROZEN_LAUNCHER,
                        str(project_root / "scripts" / helper)], cwd=project_root, check=True)
    build_combined_report()
    verify_collection()
    verify_baseline()
