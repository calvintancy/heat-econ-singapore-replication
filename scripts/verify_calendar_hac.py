"""Check actual-calendar HAC and unchanged legacy saved-calendar behavior.

No regression is fitted and no historical output/reference is rewritten. The
legacy comparison is an all-score algebraic identity on calendars where the old
grouping helper does not relabel; sparse-gap fixtures use independent pair sums.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum as old_covariance, weights_bartlett, weights_uniform

from calendar_hac import calendar_meat, cov_nw_groupsum
from presentation_verification_state import persist, presentation_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/calendar_hac"
CONSUMERS = ["estimate", "covid_common", "aggregate_va", "sector_temperature", "outcome_analysis", "iip_analysis"]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_meat(scores, dates, bandwidth, kernel):
    weights = kernel(bandwidth);result = np.zeros((scores.shape[1], scores.shape[1]))
    for first in range(len(scores)):
        for second in range(len(scores)):
            distance = abs(int(dates[first]) - int(dates[second]))
            if distance <= bandwidth: result += weights[distance] * np.outer(scores[first], scores[second])
    return result


def saved_samples():
    paths = set()
    for name in ["iip", "extended_controls", "outcomes", "aggregate_va", "granular", "heterogeneity", "covid_comparison/common"]:
        folder = ROOT / "outputs" / name
        for pattern in ["**/samples/*.csv", "**/common_sample.csv", "**/baseline_sample.csv", "**/sensitivity_80_sample.csv"]:
            paths.update(folder.glob(pattern))
    paths.update(ROOT.glob("outputs/*sample.csv"))
    paths.update(ROOT.glob("data/processed/panel_*.csv"))
    paths.update(ROOT.glob("data/processed/iip/panel_*.csv"))
    return sorted(paths)


def verify_calendar_hac():
    OUT.mkdir(parents=True, exist_ok=True)
    with presentation_run(OUT / "verification.json", "Actual-calendar HAC arithmetic and legacy saved-calendar compatibility"):
        protected_paths = [*ROOT.glob("outputs/**/pyfixest/fit_log.csv"), *ROOT.glob("outputs/**/pyfixest/estimator_manifest.json")]
        protected_paths += [ROOT / name for name in ["outputs/pyfixest/migration_input_reference.json", "outputs/covid_comparison/original_inputs_reference.json",
            "data/reference_before_nsa_inputs/manifest.json", "replication/manifest.json"]]
        protected = {str(path.relative_to(ROOT)): sha(path) for path in protected_paths}
        sources = {f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py") for name in CONSUMERS + ["calendar_hac", "verify_calendar_hac"]}
        for name in CONSUMERS:
            text = (ROOT / f"scripts/{name}.py").read_text()
            assert "from calendar_hac import cov_nw_groupsum" in text
            assert "from statsmodels.stats.sandwich_covariance import cov_nw_groupsum" not in text
        sample_records, cache = [], {}
        for path in saved_samples():
            frame = pd.read_csv(path, usecols=lambda column: column == "time")
            assert "time" in frame and len(frame) > 3, path
            raw = frame.time.to_numpy();assert np.isfinite(raw).all() and np.array_equal(raw, raw.astype(int))
            time = raw.astype(int) - int(raw.min());n = len(time);span = int(time.max()) + 1
            assert int(time.max()) <= 2 * n, (path, "A historical saved calendar triggers the old relabelling bug")
            key = hashlib.sha256(time.astype("<i8").tobytes()).hexdigest()
            if key not in cache:
                row = np.arange(n)
                scores = np.column_stack([np.sin(row * .13) + time * .01, np.cos(row * .31), (row % 7 - 3.) / 4])
                maximum = 0.
                for bandwidth in [0, 4, 12, 24]:
                    for correction in [False, "hac", "cluster"]:
                        before = old_covariance((scores, np.eye(3)), bandwidth, time, use_correction=correction)
                        after = cov_nw_groupsum((scores, np.eye(3)), bandwidth, time, use_correction=correction)
                        assert np.array_equal(before, after), (path, bandwidth, correction, np.max(np.abs(before - after)))
                        maximum = max(maximum, float(np.max(np.abs(before - after))))
                cache[key] = maximum
            sample_records.append({"sample_csv": str(path.relative_to(ROOT)), "sample_sha256": sha(path), "n_rows": n,
                "n_observed_dates": len(np.unique(time)), "calendar_span": span, "old_relabels": False,
                "normalized_calendar_sha256": key, "maximum_old_new_covariance_error": cache[key]})
        assert sample_records
        fixtures = []
        times = np.asarray([1001, 0, 1000, 0, 1001, 1002, 2000, 2000], int)
        scores = np.asarray([[1., -.2], [2., 1.], [.3, 2.], [-1., .4], [2., .1], [-.6, .8], [1.5, -2.], [.7, .1]])
        for kernel in [weights_bartlett, weights_uniform]:
            for bandwidth in [0, 1, 4, 12]:
                expected = pair_meat(scores, times, bandwidth, kernel)
                actual = calendar_meat(scores, times, bandwidth, kernel)
                assert np.allclose(actual, expected, atol=1e-13, rtol=1e-13)
                assert np.array_equal(actual, calendar_meat(scores, times + 10_000_000, bandwidth, kernel))
                for correction in [False, "hac", "cluster"]:
                    factor = 1. if not correction else len(scores) / (len(scores) - 2) if correction == "hac" else len(np.unique(times)) / (len(np.unique(times)) - 1) * (len(scores) - 1) / (len(scores) - 2)
                    result = cov_nw_groupsum((scores, np.eye(2)), bandwidth, times, kernel, correction)
                    assert np.allclose(result, expected * factor, atol=1e-13, rtol=1e-13)
                fixtures.append({"kernel": kernel.__name__, "bandwidth": bandwidth, "span": int(times.max()) + 1,
                    "n_rows": len(times), "maximum_pair_sum_error": float(np.max(np.abs(actual - expected)))})
        old_wrong = old_covariance((scores, np.eye(2)), 1, times)
        correct = cov_nw_groupsum((scores, np.eye(2)), 1, times)
        assert np.max(np.abs(old_wrong - correct)) > 1e-3, "Regression fixture no longer detects categorical relabelling"
        boundary = np.asarray([0, 16, 0, 16, 0, 16, 0, 16])
        assert np.array_equal(old_covariance((scores, np.eye(2)), 4, boundary), cov_nw_groupsum((scores, np.eye(2)), 4, boundary)), "Exact old max(time)==2N boundary changed"
        automatic_lags = int(np.floor(4 * ((times.max() + 1) / 100.) ** (2 / 9)))
        assert np.array_equal(calendar_meat(scores, times, None), calendar_meat(scores, times, automatic_lags))
        assert all(sha(ROOT / path) == digest for path, digest in protected.items()), "Protected ledger/reference changed"
        assert all(sha(ROOT / path) == digest for path, digest in sources.items()), "HAC source changed during verification"
        pd.DataFrame(sample_records).to_csv(OUT / "legacy_calendar_checks.csv", index=False)
        proof = {"status": "passed", "native_fits_run": 0, "consumer_imports": CONSUMERS, "sources_sha256": sources,
            "legacy_saved_calendars": len(sample_records), "distinct_row_calendar_layouts": len(cache), "currently_affected_saved_calendars": 0,
            "maximum_old_new_covariance_error": max(cache.values()), "large_gap_fixtures": fixtures,
            "old_no_relabel_2N_boundary_bit_identical": True,
            "protected_ledgers_and_references_sha256": protected, "legacy_calendar_checks_sha256": sha(OUT / "legacy_calendar_checks.csv"),
            "scope": "Six import-only consumer migrations. For every scanned saved calendar the old and new dense score grid/corrections are identical for arbitrary regression scores; numerical examples are bit-identical. Explicit sparse-gap, repeated/unsorted-date, origin-shift, kernel and correction fixtures match independent actual-date pair sums. No model refits or historical output changes",
            "limit": "This repairs time-distance bookkeeping. It does not calibrate HAC t/F tests or change their working finite-sample conventions"}
        persist(OUT / "verification.json", proof)
        print(f"Actual-calendar HAC passed: {len(sample_records)} saved calendars, {len(fixtures)} large-gap fixtures; zero native fits", flush=True)
        return proof


if __name__ == "__main__": verify_calendar_hac()
