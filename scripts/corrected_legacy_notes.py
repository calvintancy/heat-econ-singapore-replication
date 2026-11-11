"""Render covariance-labelled historical notes without rewriting release artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import iip_report
import report

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/aea_audit"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_corrected_legacy_notes():
    OUT.mkdir(parents=True,exist_ok=True)
    guarded=["outputs/first_results.md","outputs/iip_results.md","outputs/heat_econ_results.html","outputs/iip_results.html",
        "outputs/regression_results.csv","outputs/iip/monthly/regression_results.csv","outputs/iip/quarterly/regression_results.csv",
        "outputs/pyfixest/fit_log.csv","outputs/annual/pyfixest/fit_log.csv","replication/manifest.json"]
    before={name:sha(ROOT/name) for name in guarded}
    va=OUT/"first_results_corrected.md";iip=OUT/"iip_results_corrected.md"
    report.report(destination=va,figures=False)
    iip_report.build_iip_report(markdown_only=True,destination=iip)
    text=va.read_text();assert "| Control | Covariance | Coefficient | SE | 95% interval | p | Reference df |" in text
    results=pd.read_csv(ROOT/"outputs/regression_results.csv")
    source=results.loc[results.model.eq("S3")&results.term.isin(["rain_100mm","rh_pct"])]
    assert len(source)==4
    for row in source.itertuples():
        method=row.covariance+(f" ({row.hac_lags} lags)" if row.hac_lags else "")
        expected=f"| {row.term} | {method} | {row.coefficient:.3f} | {row.std_error:.3f} | [{row.ci_lower:.3f}, {row.ci_upper:.3f}] | {row.p_value:.3f} | {row.inference_df} |"
        assert expected in text
    iip_text=iip.read_text();assert "common-slope headline/model rows use sector-cluster CR1" in iip_text and "t(5)" in iip_text
    assert all(sha(ROOT/name)==digest for name,digest in before.items()),"A preserved release artifact changed"
    proof={"status":"passed","scope":"Two source-rendered corrected historical Markdown notes; no economic re-estimation or modification of original release notes/results",
        "control_rows_labelled":len(source),"original_artifacts_sha256":before,
        "source_sha256":{str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),ROOT/"scripts/report.py",ROOT/"scripts/iip_report.py"]},
        "outputs_sha256":{str(path.relative_to(ROOT)):sha(path) for path in [va,iip]}}
    (OUT/"narrative_corrections.json").write_text(json.dumps(proof,indent=2)+"\n")
    print("Covariance-labelled narrative corrections passed; original notes and native ledgers unchanged.")
    return proof


if __name__=="__main__":build_corrected_legacy_notes()
