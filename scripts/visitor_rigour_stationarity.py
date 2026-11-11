"""Exact-window marginal ADF/KPSS screens for the visitor-rigour companion.

No estimation, pretest selection or frozen appendix mutation. PPML count-level
screens are descriptive; an ordinary I(0) result is not an admission requirement
or validity certificate for a conditional Poisson model.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from stationarity_audit import LABELS, state, test_pair
from stationarity_diagnostics import seasonal_adjust

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/visitor_rigour/diagnostics"
FAMILIES = {"monthly": "outputs/visitor_rigour/monthly/specifications.csv",
            "annual": "outputs/visitor_rigour/annual/specifications.csv",
            "ppml": "outputs/visitor_rigour/ppml/specifications.csv",
            "origin_year": "outputs/visitor_rigour/origin_year/specifications.csv"}
EXPECTED = {"monthly": 270, "annual": 30, "ppml": 30, "origin_year": 24}
EXCLUDED = {"lny_window_days", "lny_lag1", "trend", "trend_sq", "time"}
LABELS = {**LABELS, "unavailable-missing": "Not established · missing values", "unavailable-constant": "Not established · constant series",
          "unavailable-empty": "Not established · empty calendar"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path): return str(Path(path).resolve().relative_to(ROOT))
def present(value): return isinstance(value, str) and bool(value) and value != "nan"
def items(value): return json.loads(value) if present(value) else []


def clean(value):
    if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def binary(values):
    values = np.asarray(values, dtype="<f8").copy();values[np.isnan(values)] = np.nan
    return values.tobytes()


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class VisitorDiagnostics:
    def __init__(self):
        self.hashes, self.frames, self.diagnostics = {}, {}, {}
        self.tests, self.series, self.links, self.models = [], [], [], []

    def bind(self, path):
        path = Path(path);key = relative(path)
        if key not in self.hashes: self.hashes[key] = sha(path)
        return key

    def frame(self, path):
        key = self.bind(path)
        if key not in self.frames:
            self.frames[key] = pd.read_csv(path, dtype={"unit_id": str, "sector_id": str}, float_precision="round_trip", low_memory=False)
        return self.frames[key]

    def diagnostic(self, dates, values, frequency, variable, role, count_level=False):
        dates = pd.PeriodIndex(dates, freq=frequency)
        order = np.argsort(dates.asi8);dates, values = dates[order], np.asarray(values, float)[order]
        assert dates.is_unique and len(dates) == len(values)
        annual = frequency == "Y-DEC";h = 1 if annual else 12
        is_ar, rain = variable.startswith("ar_l"), variable.startswith("rain")
        difference = variable == "temp_endpoint_difference"
        growth = role == "Y" and not count_level
        deterministic = "c" if growth or is_ar or rain or difference else "ct"
        # Every outcome is NSA. Apply the established season-offset screen with
        # trend retained to monthly raw bases, including lagged NSA growth.
        seasonal = not annual
        policy = {"frequency": frequency, "deterministic": deterministic,
                  "seasonal_rule": "season_offsets_with_trend_retained" if seasonal else "none"}
        calendar, raw = dates.asi8.astype("<i8").tobytes(), binary(values)
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode() + b"\0" + calendar + raw).hexdigest()
        ident = "vs_" + digest[:24]
        if ident in self.diagnostics:
            row = self.diagnostics[ident];assert row["input_sha256"] == digest
            row["_variables"].add(variable);row["_roles"].add(role);row["_count_screen"] |= count_level
            return ident
        n = len(values);gaps = int(dates[-1].ordinal - dates[0].ordinal + 1 - n) if n else 0
        missing = int((~np.isfinite(values)).sum());minimum = 20 if annual else 60
        row = {"diagnostic_id": ident, "input_sha256": digest, "calendar_sha256": hashlib.sha256(calendar).hexdigest(),
            "raw_values_sha256": hashlib.sha256(raw).hexdigest(), **policy, "frequency_horizon": h, "minimum_periods": minimum,
            "n_periods": n, "start": str(dates[0]) if n else "", "end": str(dates[-1]) if n else "", "calendar_gaps": gaps,
            "missing_values": missing, "primary_state": None, "legacy_state": None, "test_status": "not_run", "test_values_sha256": None,
            "_variables": {variable}, "_roles": {role}, "_count_screen": count_level}
        result = "unavailable-gaps" if gaps else "unavailable-missing" if missing else "unavailable-empty" if not n else "unavailable-short" if n < minimum else "unavailable-constant" if np.ptp(values) == 0 else None
        adjusted = component = None
        if result is None:
            source = pd.Series(values, index=dates)
            adjusted, component = seasonal_adjust(source) if seasonal else (source, source * 0)
            if np.ptp(adjusted.to_numpy()) == 0: result = "unavailable-constant"
        if result is None:
            tested = adjusted.to_numpy();row["test_values_sha256"] = hashlib.sha256(binary(tested)).hexdigest()
            for specification, sensitivity in [("primary", False), ("legacy_lag_sensitivity", True)]:
                pair = test_pair(tested, deterministic, h, sensitivity)
                self.tests.extend({"diagnostic_id": ident, "specification": specification, "deterministic": deterministic, "frequency": frequency, **test} for test in pair)
                classification = state(*pair, deterministic);stem = "legacy_" if sensitivity else "primary_"
                row["legacy_state" if sensitivity else "primary_state"] = classification
                for suffix, value in [("adf_p", pair[0]["p_value"]), ("kpss_p", pair[1]["p_value"]), ("kpss_relation", pair[1]["p_relation"]),
                                      ("adf_lags", pair[0]["lags"]), ("kpss_lags", pair[1]["lags"])]: row[stem + suffix] = value
            result = row["primary_state"] if row["primary_state"] == row["legacy_state"] else "sensitivity-dependent"
            row["test_status"] = "completed" if row["primary_state"] != "unavailable" and row["legacy_state"] != "unavailable" else "error"
        row.update(state=result, label=LABELS[result])
        for index, (date, value) in enumerate(zip(dates, values, strict=True)):
            self.series.append({"diagnostic_id": ident, "period": str(date), "ordinal": int(date.ordinal), "raw_value": float(value),
                "test_value": float(adjusted.iloc[index]) if row["test_values_sha256"] else None,
                "seasonal_component": float(component.iloc[index]) if row["test_values_sha256"] else None})
        self.diagnostics[ident] = row
        return ident

    def register(self, family, spec_path, index, record):
        sample_name = record["sample_csv"]
        actual_sample = (ROOT / sample_name).is_file()
        if not actual_sample:
            assert family == "ppml" and record["status"] != "estimated", (family, sample_name)
            sample_name = record["input_sample_csv"]
        sample = self.frame(ROOT / sample_name)
        declared_hash = record.get("sample_sha256" if actual_sample else "input_sample_sha256")
        if present(declared_hash): assert self.hashes[sample_name] == declared_hash, sample_name
        frequency = "Y-DEC" if family == "annual" else "M"
        period = "year" if family == "annual" else "month"
        units = sample.unit_id.to_numpy(dtype=str)
        dates = pd.PeriodIndex(sample[period].astype(str), freq=frequency)
        outcome = record["outcome"];y_field = "arrivals" if family == "ppml" else outcome
        y = sample[y_field].to_numpy(float)
        if family in {"monthly", "origin_year"}:
            requested = list(dict.fromkeys(items(record["temp_features"]) + items(record["control_features"]) + items(record["ar_features"])))
        else: requested = list(dict.fromkeys(items(record["temperature_terms"]) + items(record["controls"])))
        excluded, terms = [term for term in requested if term in EXCLUDED], [term for term in requested if term not in EXCLUDED]
        assert all(term in sample for term in terms) and len(set(zip(units, dates))) == len(sample)
        design_path = ROOT / record["design_npz"]
        design_sha, basis_validation = "", "No native matrix returned; intended request raw sample bases retained"
        if design_path.is_file():
            self.bind(design_path);design_sha = self.hashes[relative(design_path)]
            with np.load(design_path, allow_pickle=False) as packed:
                archived_units = packed["unit_ids" if "unit_ids" in packed else "units"].astype(str)
                archived_time = packed["time"].astype(int)
                expected_time = sample.year.to_numpy(int) if family == "annual" else dates.asi8
                assert np.array_equal(archived_units, units) and np.array_equal(archived_time, expected_time)
                assert np.allclose(packed["Y"], y, atol=1e-10, rtol=1e-12, equal_nan=True)
                x, names = packed["X"], packed["names"].tolist()
                for term in terms:
                    actual = x[:, names.index(term)]
                    raw = sample[term].to_numpy(float)
                    expected = raw if family == "ppml" else raw - raw.mean()
                    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-12, equal_nan=True), (family, record["spec_id"], term)
            basis_validation = "Raw sample bases verified against actual archived X before FE projection; declared OLS centering reversed, PPML raw bases unchanged"
        elif family != "ppml": raise AssertionError("Missing requested OLS design archive: " + str(design_path))
        identity = hashlib.sha256(json.dumps([family, spec_path, int(index), outcome], separators=(",", ":")).encode()).hexdigest()[:24]
        group = {"coverage_id": "vm_" + identity, "branch": family, "source_spec_csv": spec_path, "source_spec_row": int(index),
            "spec_id": record["spec_id"], "native_spec_id": record.get("native_spec_id", record["spec_id"]), "alias_of": record.get("alias_of", ""),
            "outcome": outcome, "model": record["model"], "model_spec": record["model_spec"], "context": record["context"],
            "control_set": record["control_set"], "history": record.get("history", "exact_sample"), "sample_variant": record["sample_variant"],
            "source_status": record["status"], "sample_csv": sample_name, "sample_sha256": self.hashes[sample_name],
            "sample_role": "Native retained sample" if family == "ppml" and actual_sample else "Intended unavailable PPML request" if family == "ppml" else "Exact requested OLS sample; includes unavailable designs",
            "design_npz": relative(design_path) if design_sha else "", "design_sha256": design_sha,
            "frequency": frequency, "n_obs": len(y), "n_units": len(set(units)), "n_periods": len(set(dates)),
            "basis_rule": basis_validation, "count_level_screen": family == "ppml"}
        unit_list = sorted(set(units));count = 0
        for unit in unit_list:
            keep = units == unit
            diagnostic = self.diagnostic(dates[keep], y[keep], frequency, outcome, "Y", family == "ppml")
            self.links.append({**group, "variable": outcome, "role": "Y", "unit_id": unit, "applies_to_units": json.dumps([unit]), "diagnostic_id": diagnostic});count += 1
        for term in terms:
            separated = [(unit, dates[units == unit], sample.loc[units == unit, term].to_numpy(float)) for unit in unit_list]
            common = len(separated) > 1 and all(np.array_equal(d.asi8, separated[0][1].asi8) and np.array_equal(v, separated[0][2], equal_nan=True) for _, d, v in separated[1:])
            for unit, date, values in separated[:1] if common else separated:
                diagnostic = self.diagnostic(date, values, frequency, term, "X")
                self.links.append({**group, "variable": term, "role": "X", "unit_id": "common" if common else unit,
                    "applies_to_units": json.dumps(unit_list if common else [unit]), "diagnostic_id": diagnostic});count += 1
        self.models.append({**group, "n_variable_links": count, "requested_basis_terms": json.dumps(terms),
            "excluded_deterministic_terms": json.dumps(excluded), "growth_outcome": family != "ppml",
            "interpretation": "Descriptive count-level screen; not an I(0) prerequisite or validity certificate for Poisson" if family == "ppml" else "Marginal exact-window screen; no pretest sample/model selection"})


def build_report_payload():
    summary = json.loads((OUT / "summary.json").read_text())
    rows = pd.read_csv(OUT / "diagnostics.csv").fillna("")
    for stem in ["primary", "legacy"]:
        for field in [stem + "_kpss_relation", stem + "_kpss_p", stem + "_adf_p"]:
            if field not in rows: rows[field] = ""
        rows[stem + "_kpss_bound"] = [str(symbol) + f"{float(value):.4g}" if value != "" else "" for symbol, value in zip(rows[stem + "_kpss_relation"], rows[stem + "_kpss_p"])]
    columns = ["diagnostic_id", "branches", "variables", "roles", "frequency", "n_periods", "calendar_gaps", "state", "primary_adf_p", "primary_kpss_bound", "legacy_adf_p", "legacy_kpss_bound", "count_level_descriptive_screen"]
    downloads = {name: base64.b64encode(gzip.compress((OUT / name).read_bytes(), mtime=0)).decode() for name in ["diagnostics.csv", "tests.csv", "model_inventory.csv", "source_hashes.json"]}
    payload = {"schema_version": 1, "counts": summary["counts"], "states": summary["diagnostic_states"], "display_columns": columns,
        "rows": rows[columns].to_dict("records"), "downloads": downloads,
        "source_hashes_sha256": sha(OUT / "source_hashes.json"), "scope": summary["scope"], "limits": summary["limits"]}
    save_json(OUT / "report_payload.json", payload)
    packed = base64.b64encode(gzip.compress(json.dumps(clean(payload), separators=(",", ":"), allow_nan=False).encode(), mtime=0)).decode()
    counts = summary["counts"]
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visitor-arrival exact-window diagnostics</title><style>body{font:14px/1.6 system-ui;margin:30px auto;padding:0 20px;max-width:1320px;background:#f4f7fb;color:#18344b}h1{font-size:30px}p{max-width:1100px}.note{padding:15px;background:#fff0d8;border-left:4px solid #bd8327}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px;border-bottom:1px solid #d7e1e8;font-size:11px;text-align:left}th{background:#e6eef6}select,input,button{font:inherit;padding:7px;margin:5px 10px 12px 0}#vd-wrap{overflow:auto}code{overflow-wrap:anywhere}</style><h1>Visitor arrivals · exact-window unit-root diagnostics</h1><p class="note">These marginal ordinary ADF/KPSS screens do not certify stationarity, identification, inference calibration or Poisson validity. PPML raw-count screens are descriptive; I(0) clearance is not a precondition imposed on the count model. No sample or model is selected by these tests. Gaps are never compressed.</p><p>__COUNTS__</p><p>Monthly minimum:60 observations; annual minimum:20 complete growth years. The source supplies17 annual growth years, so annual windows cannot clear this screen. Primary ADF uses AIC with a two-year lag cap; KPSS automatic bandwidth is uncapped. The one-year lag-cap convention is retained as sensitivity. Monthly NSA series have calendar-season offsets removed with trend retained. LNY calendar controls, FE/trend indicators and zero-padded interaction columns are excluded. Raw counts retain actual zeros.</p><div id="vd-downloads">Loading embedded evidence…</div><label>Evidence<select id="vd-state"><option value="">All states</option></select></label><label>Frequency<select id="vd-frequency"><option value="">All</option><option>M</option><option>Y-DEC</option></select></label><label>Family<select id="vd-family"><option value="">All</option><option>monthly</option><option>annual</option><option>ppml</option><option>origin_year</option></select></label><label>Find<input id="vd-search" placeholder="Variable or diagnostic ID"></label><button id="vd-back">Previous</button><button id="vd-next">Next</button><p id="vd-status"></p><div id="vd-wrap"><table><thead><tr>__HEADERS__</tr></thead><tbody id="vd-body"></tbody></table></div><p>The standalone file embeds the diagnostic table, tests, model inventory and source-hash manifest. Full raw arrays are saved in <code>series.csv</code>; per-model/per-unit links are saved in <code>model_variable_coverage.csv</code>. Those two larger audit ledgers are not embedded. Repeated source arrays and PPML mask aliases are deduplicated; logical model counts are not independent tests.</p><script>const visitorPacked="__PACKED__";let visitorData,visitorOffset=0;const vd=id=>document.getElementById('vd-'+id);const escapeVisitor=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));async function inflateVisitor(text){return new Response(new Blob([Uint8Array.from(atob(text),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text()}function drawVisitorDiagnostics(){const state=vd('state').value,freq=vd('frequency').value,family=vd('family').value,q=vd('search').value.toLowerCase();const rows=visitorData.rows.filter(r=>(!state||r.state===state)&&(!freq||r.frequency===freq)&&(!family||JSON.parse(r.branches).includes(family))&&(!q||(r.variables+' '+r.diagnostic_id).toLowerCase().includes(q)));visitorOffset=Math.max(0,Math.min(visitorOffset,Math.floor(Math.max(rows.length-1,0)/100)*100));vd('body').innerHTML=rows.slice(visitorOffset,visitorOffset+100).map(r=>'<tr>'+visitorData.display_columns.map(c=>'<td>'+escapeVisitor(typeof r[c]==='number'&&!Number.isInteger(r[c])?r[c].toPrecision(4):r[c])+'</td>').join('')+'</tr>').join('');vd('status').textContent=`${rows.length} diagnostics; showing ${rows.length?visitorOffset+1:0}–${Math.min(visitorOffset+100,rows.length)}. KPSS bounds and warnings are in tests.csv.`}async function initializeVisitorDiagnostics(){visitorData=JSON.parse(await inflateVisitor(visitorPacked));for(const value of [...new Set(visitorData.rows.map(r=>r.state))].sort())vd('state').add(new Option(value,value));vd('downloads').innerHTML=Object.keys(visitorData.downloads).map(name=>'<button data-download="'+name+'">'+name+'</button>').join('');vd('downloads').onclick=async e=>{const name=e.target.dataset.download;if(!name)return;const text=await inflateVisitor(visitorData.downloads[name]);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};for(const id of ['state','frequency','family','search'])vd(id).oninput=()=>{visitorOffset=0;drawVisitorDiagnostics()};vd('back').onclick=()=>{visitorOffset-=100;drawVisitorDiagnostics()};vd('next').onclick=()=>{visitorOffset+=100;drawVisitorDiagnostics()};drawVisitorDiagnostics()}window.visitorDiagnosticsReady=initializeVisitorDiagnostics();window.visitorDiagnosticsReady.catch(e=>{vd('status').textContent='Diagnostics unavailable: '+e.message;throw e});</script></html>'''
    line = f"{counts['logical_model_outcomes']:,} logical model/outcome requests · {counts['model_variable_links']:,} variable links · {counts['distinct_diagnostics']:,} distinct arrays · {counts['completed_diagnostics']:,} completed screens · {counts['unavailable_diagnostics']:,} unavailable"
    page = page.replace("__COUNTS__", line).replace("__HEADERS__", "".join("<th>" + html.escape(column) + "</th>" for column in columns)).replace("__PACKED__", packed)
    (OUT / "index.html").write_text(page)
    return payload


def run_visitor_rigour_stationarity():
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "summary.json", {"status": "running", "claim": "No completed diagnostic or independent verification claim"})
    audit = VisitorDiagnostics()
    for path in [Path(__file__), ROOT / "scripts/stationarity_audit.py", ROOT / "scripts/stationarity_diagnostics.py",
                 ROOT / "scripts/visitor_rigour_monthly.py", ROOT / "scripts/visitor_rigour_companions.py", ROOT / "scripts/visitor_origin_year.py",
                 ROOT / "notes/visitor_origin_year_protocol.md",
                 ROOT / "outputs/extension_stationarity/summary.json", ROOT / "outputs/extension_stationarity/verification.json"]: audit.bind(path)
    parent_manifest = ROOT / "outputs/visitor_rigour/monthly/manifest.json"
    audit.bind(parent_manifest)
    monthly = json.loads(parent_manifest.read_text())
    assert monthly["status"] == "computed_not_yet_independently_verified"
    for path, expected in monthly["raw_inputs_sha256"].items():
        audit.bind(ROOT / path);assert audit.hashes[path] == expected, path
    baseline = None
    with threadpool_limits(limits=1):
        for family, path in FAMILIES.items():
            if family == "origin_year":
                baseline = {"models": len(audit.models), "diagnostic_ids": set(audit.diagnostics), "test_rows": len(audit.tests), "series_rows": len(audit.series)}
            source = audit.frame(ROOT / path)
            assert len(source) == EXPECTED[family], (family, len(source))
            for name in ["fit_log.csv", "estimator_manifest.json"]: audit.bind(ROOT / Path(path).parent / "pyfixest" / name)
            for index, row in source.iterrows(): audit.register(family, path, index, row.to_dict())
            if family == "origin_year":
                assert baseline["models"] == 330 and len(audit.models) == 354
                assert baseline["diagnostic_ids"] == set(audit.diagnostics), "FE-only comparison unexpectedly changes raw stochastic arrays or diagnostic policy"
                assert baseline["test_rows"] == len(audit.tests) and baseline["series_rows"] == len(audit.series)
            print(f"Visitor diagnostics {family}: {len(source)} model requests, {len(audit.diagnostics)} unique arrays so far", flush=True)
    branches = {}
    for link in audit.links: branches.setdefault(link["diagnostic_id"], set()).add(link["branch"])
    diagnostic_rows = []
    for ident, row in audit.diagnostics.items():
        diagnostic_rows.append({**{key: value for key, value in row.items() if not key.startswith("_")},
            "variables": json.dumps(sorted(row["_variables"])), "roles": json.dumps(sorted(row["_roles"])),
            "branches": json.dumps(sorted(branches[ident])), "count_level_descriptive_screen": row["_count_screen"]})
    frames = {"diagnostics": pd.DataFrame(diagnostic_rows), "tests": pd.DataFrame(audit.tests), "series": pd.DataFrame(audit.series),
              "model_inventory": pd.DataFrame(audit.models), "model_variable_coverage": pd.DataFrame(audit.links)}
    for name, frame in frames.items(): frame.to_csv(OUT / f"{name}.csv", index=False)
    assert all(sha(ROOT / path) == value for path, value in audit.hashes.items()), "Analytical source/sample/design changed during diagnostics"
    save_json(OUT / "source_hashes.json", audit.hashes)
    diagnostics = frames["diagnostics"]
    assert len(audit.models) == 354 and not any(term in EXCLUDED for term in frames["model_variable_coverage"].variable)
    counts = {"source_spec_rows": 354, "logical_model_outcomes": len(audit.models), "model_variable_links": len(audit.links),
        "distinct_diagnostics": len(diagnostics), "completed_diagnostics": int(diagnostics.test_status.eq("completed").sum()),
        "unavailable_diagnostics": int(diagnostics.test_status.eq("not_run").sum()), "error_diagnostics": int(diagnostics.test_status.eq("error").sum()),
        "test_rows": len(audit.tests), "series_rows": len(audit.series)}
    summary = {"status": "computed_not_yet_independently_verified", "families": FAMILIES, "counts": counts,
        "logical_models_by_family": frames["model_inventory"].branch.value_counts().to_dict(), "diagnostic_states": diagnostics.state.value_counts().to_dict(),
        "source_hashes_sha256": sha(OUT / "source_hashes.json"), "native_fits_run": 0,
        "coverage_extension": {"previous_logical_requests": baseline["models"], "added_origin_year_requests": 24,
            "identical_diagnostic_ids": True, "additional_test_rows": 0, "additional_series_rows": 0},
        "policy": {"primary": "ADF AIC cap2h; uncapped automatic KPSS", "sensitivity": "ADF AIC caph; automatic KPSS cappedh", "minimum_periods": {"M": 60, "Y-DEC": 20},
            "deterministic": "c for growth/AR/rain/temperature difference; ct for count levels, temperature/RH/wind/PM and temperature exposure sum",
            "seasonality": "Monthly NSA actual raw bases: remove calendar-season offsets with trend retained; annual:none",
            "gaps": "Any gaps/missingness unavailable before short/constant screens; no compression or longest-run replacement",
            "excluded": sorted(EXCLUDED), "basis": "Raw uncentered sample stochastic bases, validated against archived non-FE X before projection; no padded sector interactions",
            "dedup": "SHA256(sorted compact policy JSON + null byte + little-endian int64 period ordinals + little-endian float64 raw values with canonical NaN bits)"},
        "scope": "All354 logical visitor-rigour requests, including24origin-year FE comparisons, intended unavailable designs and PPML aliases; per-unit Y and actual stochastic bases; exact saved calendars; no new regression fits",
        "limits": "Marginal ordinary zero-frequency diagnostics, not stationarity/causal/inference-calibration or Poisson validity certificates. PPML count levels are descriptive screens, not an imposed I(0) prerequisite. Annual source has17growth years<20; short/gapped windows remain unresolved. No pretest selection",
        "outputs_sha256": {name + ".csv": sha(OUT / (name + ".csv")) for name in frames}}
    save_json(OUT / "summary.json", summary)
    build_report_payload()
    print(json.dumps(counts, indent=2), flush=True)
    return summary


if __name__ == "__main__": run_visitor_rigour_stationarity()
