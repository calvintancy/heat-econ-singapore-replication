"""Reproduce or render every full/no-COVID pair without replacing the original results."""
from __future__ import annotations
import argparse


def run_covid_comparison(*,estimate=True,refresh_reference=False,package=True):
    from verify_covid_comparison import capture_original_inputs_reference,verify_covid_comparison
    if estimate:
        # Deliberately capture before either isolated producer can mutate anything.
        capture_original_inputs_reference(refresh=refresh_reference)
        from covid_common import run_covid_common
        from covid_sector import run_covid_sector
        run_covid_common()
        run_covid_sector()
    verify_covid_comparison()
    from covid_report import build_covid_report
    from verify_covid_report import verify_covid_report
    build_covid_report()
    verify_covid_report()
    if package:
        from combined_report import build_combined_report
        from verify_collection import verify_collection
        from verify_baseline import verify_baseline
        build_combined_report()
        verify_collection()
        verify_baseline()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-only',action='store_true',help='Verify and render the already-fitted pairs')
    parser.add_argument('--refresh-reference',action='store_true',help='Capture the current original outputs before new paired fits')
    args=parser.parse_args()
    run_covid_comparison(estimate=not args.reports_only,refresh_reference=args.refresh_reference)
