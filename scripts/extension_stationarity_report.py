"""Render the exact-calendar diagnostic appendix after independent validation."""
import json
from pathlib import Path

from extension_stationarity import render_extension_stationarity

ROOT=Path(__file__).resolve().parents[1]

if __name__=="__main__":
    receipt=json.loads((ROOT/"outputs/extension_stationarity/verification.json").read_text())
    assert receipt["status"]=="passed","Independent stationarity diagnostics check has not passed"
    render_extension_stationarity()
