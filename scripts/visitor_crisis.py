"""Calendar-defined crisis/recovery exclusions, independent of realised growth."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

import pyfixest_backend as native
from visitor_analysis import Analysis, BASE, FULL, OUT, load_prepared, save_json

CRISIS = OUT / "crisis"
POLICIES = {
    "none": {"label": "No crisis exclusion", "windows": []},
    "covid": {"label": "COVID 2020–2022", "windows": [["2020-01", "2022-12"]]},
    "gfc_covid": {"label": "GFC 2008–2009 + COVID 2020–2022", "windows": [["2008-01", "2009-12"], ["2020-01", "2022-12"]]},
    "buffered": {"label": "GFC/recovery 2008–2010 + COVID/recovery 2020–2023", "windows": [["2008-01", "2010-12"], ["2020-01", "2023-12"]]},
    "stable_pre2020": {"label": "Pre-COVID after GFC/recovery buffer", "windows": [["2008-01", "2010-12"], ["2020-01", "2099-12"]]},
}


def run():
    CRISIS.mkdir(parents=True, exist_ok=True)
    save_json(CRISIS / "policies.json", {"policies": POLICIES,
        "selection_rule": "Externally motivated calendar sensitivities. Never filter on coefficient signs, p-values or observed growth size.",
        "growth_rule": "Remove any observation whose complete [base,current] monthly interval intersects an exclusion window; transformations precede masks.",
        "interpretation": "Whole-year windows and one-year recovery buffers are analytical choices, not official crisis boundaries. Retained years are not certified crisis-free.",
        "sources": [
            {"url": "https://www.pmo.gov.sg/newsroom/speech-mr-goh-chok-tong-senior-minister-panel-discussion--financial-crisis-2007-09/", "supports": "Singapore recession began in 2008Q3; motivates GFC window rather than exact whole-year boundaries."},
            {"url": "https://www.mti.gov.sg/newsroom/minister-lim-hng-kiang-s-reply-to-parliament-questions-on-economic--employment-and-apec/", "supports": "Positive year-on-year growth by 2009Q4; 2010 is a recovery-buffer choice."},
            {"url": "https://isomer-user-content.by.gov.sg/166/2b9db1b9-8cd6-4b33-9c60-419b656135de/fullreport_2q13.pdf", "supports": "MTI historical tourism table reports sharp 2010 arrivals rebound; it is not a numerical input to these regressions."},
            {"url": "https://www.stb.gov.sg/about-stb/media-publications/media-centre/singapore-s-tourism-sector-emerges-from-2020-with-greater-resilience-and-reinvention/", "supports": "2020 border restrictions and tourism disruption."},
            {"url": "https://www.stb.gov.sg/about-stb/media-publications/media-centre/singapore-s-tourism-sector-posts-strong-recovery-in-2023--exceeds-forecasts-for-tourism-receipts/", "supports": "2023 strong tourism recovery, at about 71% of 2019 arrivals, motivates a separately labelled recovery buffer; it does not define an official crisis end."}]})
    native.LOG_DIR = CRISIS / "pyfixest"
    native.reset_fit_log()
    with threadpool_limits(limits=1):
        counts, outcomes, weather, _ = load_prepared()
        analysis = Analysis(counts, outcomes, weather, output_dir=CRISIS)
        for panel in ["aggregate", "region", "country"]:
            for outcome in ["mom", "yoy", "log_level", "raw"]:
                for sample in ["long_all", "full_all"]:
                    for policy, setting in POLICIES.items():
                        analysis.estimate(panel, outcome, sample,
                            terms=FULL if sample.startswith("full") else BASE,
                            variant="crisis_"+policy, exclusion_windows=setting["windows"])
        analysis.write_outputs()
        native.write_fit_log()
    results = pd.read_csv(CRISIS / "regression_results.csv")
    save_json(CRISIS / "manifest.json", {"status": "fitted", "n_models": results.model_id.nunique(),
        "source_results_preserved": True,
        "source_results_sha256": hashlib.sha256((OUT / "regression_results.csv").read_bytes()).hexdigest()})
    print(f"Fitted {results.model_id.nunique()} crisis specifications", flush=True)


if __name__ == "__main__":
    run()
