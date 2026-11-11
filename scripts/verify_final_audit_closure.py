"""Bind final audit closures and new arrival chapters to the offline master.

No fit, covariance recomputation or frozen heat-checker edits. Deep numerical
receipts are bound to every audited artifact byte; native counts come from the
actual ledgers, keeping model aliases, diagnostics and Stata targets separate.
"""
from __future__ import annotations
import base64
import gzip
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from pathlib import PurePosixPath

import pandas as pd
from lxml import html
from presentation_verification_state import persist, presentation_run
from verify_extension_collection import NODE_HARNESS, offline, verify_selected_null_claim

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EXPECTED = {"monthly": (270, 260, "Feols"), "origin_year": (24, 24, "Feols"),
            "annual": (30, 15, "Feols"), "ppml": (30, 20, "Fepois")}
CHAPTERS = {"visitor-rigour": "visitor_arrivals_rigour.html", "stata-check": "stata_key_results.html",
            "cluster-sensitivity": "clustered_sector_sensitivity_qualified.html"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def checked_json(path):
    value = json.loads(Path(path).read_text());assert value["status"] == "passed", str(path)
    return value


def source_map(mapping, cache):
    for name, expected in mapping.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in name, (name, "Nonportable artifact path")
        if name in cache: assert cache[name] == expected, (name, "Conflicting verified generations")
        else: assert sha(ROOT / name) == expected, (name, "Verified artifact changed");cache[name] = expected


def verify_arrival_equivalence(record, artifact_hashes):
    path = OUT / "visitor_rigour/review_equivalence.json"
    proof = checked_json(path)
    baseline_path = ROOT / "replication/extensions/review_evidence/arrivals_review_baseline.json"
    baseline = json.loads(baseline_path.read_text())
    guard_path = ROOT / "replication/extensions/review_evidence/derived_proof_guard_revision.json"
    guard_helper = ROOT / "scripts/verify_derived_proof_guards.py"
    guard_proof = checked_json(OUT / "derived_proof_guards/verification.json")
    assert record["status"] == "passed" and record["receipt_sha256"] == sha(path)
    assert record["baseline_sha256"] == proof["baseline_sha256"] == sha(baseline_path)
    assert record["semantic_files"] == proof["semantic_files_checked"] == 534
    assert record["npz_arrays"] == proof["npz_arrays_checked"] == 8338
    assert proof["numeric_or_status_differences"] == [] and proof["native_fits_executed_by_this_verifier"] == 0
    assert proof["verifier_source_sha256"] == sha(ROOT / "scripts/verify_arrivals_review_equivalence.py")
    assert isinstance(proof["verified_at_utc"], str) and proof["verified_at_utc"]
    generation = {}
    for field, filename in [("current_native_fit_logs_sha256", "fit_log.csv"), ("current_native_estimator_manifests_sha256", "estimator_manifest.json")]:
        mapping = proof[field]
        assert set(mapping) == {f"outputs/visitor_rigour/{family}/pyfixest/{filename}" for family in EXPECTED}
        source_map(mapping, artifact_hashes);generation[field] = mapping
    assert proof["guard_revision_sha256"] == baseline["post_snapshot_guard_revision_sha256"] == guard_proof["revision_evidence_sha256"] == sha(guard_path)
    assert proof["guard_verifier_sha256"] == baseline["guard_verifier_sha256"] == guard_proof["verifier_sha256"] == sha(guard_helper)
    assert proof["counts"] == baseline["counts"] and set(proof["counts"]) == set(EXPECTED)
    for family, (requests, fits, _) in EXPECTED.items():
        assert proof["counts"][family]["requests"] == requests and proof["counts"][family]["native_fits"] == fits
    assert sum(row["requests"] for row in proof["counts"].values()) == 354
    assert sum(row["native_fits"] for row in proof["counts"].values()) == 319
    approved = {row["path"]: row for row in guard_proof["sources"]}
    changes = {row["path"]: row for row in proof["authorized_post_snapshot_guard_only_source_changes"]}
    assert changes == approved
    baseline_approvals = {row["path"]: row for row in baseline["authorized_guard_only_changes"]}
    assert set(baseline_approvals) == set(approved)
    assert proof["source_bindings_checked"] == len(baseline["reviewed_sources_sha256"]) == 18
    current_sources = {}
    for name, reviewed in baseline["reviewed_sources_sha256"].items():
        current = sha(ROOT / name)
        if current != reviewed:
            assert name in changes and changes[name]["before_sha256"] == reviewed and changes[name]["after_sha256"] == current
            assert baseline_approvals[name]["before_sha256"] == reviewed and baseline_approvals[name]["after_sha256"] == current
            assert changes[name]["all_other_AST_identical"] is True
        current_sources[name] = current
    source_map(current_sources, artifact_hashes)
    source_map({str(p.relative_to(ROOT)): sha(p) for p in [path, baseline_path, guard_path, guard_helper,
        ROOT / "scripts/verify_arrivals_review_equivalence.py"]}, artifact_hashes)
    return {"status": "passed", "receipt_sha256": sha(path), "baseline_sha256": sha(baseline_path),
        "semantic_files": 534, "npz_arrays": 8338, "logical_requests": 354, "native_ledger_rows": 319,
        "source_bindings": 18, "approved_guard_only_source_changes": sorted(changes),
        "verified_at_utc": proof["verified_at_utc"], **generation,
        "native_fits_executed": 0, "scope": proof["scope"], "limits": proof["limits"]}


def verify_arrival_selected_reference(record, artifact_hashes, execution_path=None):
    path = Path(execution_path) if execution_path is not None else OUT / "visitor_rigour/selected_review_benchmarks.json"
    proof = checked_json(path)
    evidence = ROOT / "replication/extensions/review_evidence"
    helper = evidence / "arrivals_selected_benchmarks.py"
    reference_path = evidence / "arrivals_selected_benchmarks.json"
    reference = checked_json(reference_path)
    baseline_path = evidence / "arrivals_review_baseline.json"
    baseline = json.loads(baseline_path.read_text())
    assert record["status"] == "passed_selected_reference_law_replay"
    assert record["current_execution_sha256"] == sha(path)
    assert record["helper_source_sha256"] == proof["source_sha256"] == reference["source_sha256"] == sha(helper)
    assert record["baseline_sha256"] == proof["baseline_sha256"] == reference["baseline_sha256"] == sha(baseline_path)
    assert proof["native_fits_executed"] == reference["native_fits_executed"] == 0
    assert record["limits"] == proof["limits"] == reference["limits"] and proof["claim"] == reference["claim"]
    expected = {"region": (8, 2026111102, [1045, 1948]), "country": (41, 2026111103, [4000, 4000])}
    canonical = {row["spec_id"]: row for row in reference["designs"]}
    assert len(proof["designs"]) == len(canonical) == 2
    seen, rows = set(), []
    for design in proof["designs"]:
        kind = design["spec_id"].split("__")[0]
        q, seed, rejections = expected[kind]
        assert kind not in seen;seen.add(kind)
        assert design["spec_id"] == kind + "__y_mom_nsa__long__all_eligible__H1__own_current"
        assert design["q"] == q and design["t"] == 207 and design["reps"] == 4000 and design["seed"] == seed
        assert design["dgp"] == canonical[design["spec_id"]]["dgp"] and design["null"] == canonical[design["spec_id"]]["null"]
        assert design["nominal_reference"] == f"F({q},206)"
        relative = "outputs/visitor_rigour/monthly/designs/" + design["spec_id"] + ".npz"
        archive = ROOT / relative
        assert design["archive_sha256"] == sha(archive)
        saved_arrays = baseline["files"][relative]["arrays"]
        assert set(design["array_bindings"]) == {"X", "Y", "time", "unit_ids", "names", "component_positions", "beta", "resid"}
        assert all(binding == saved_arrays[name] for name, binding in design["array_bindings"].items())
        assert [design["n"], design["k"]] == saved_arrays["X"]["shape"]
        assert [row["hac_lags"] for row in design["results"]] == [12, 24]
        for band, result, rejected in zip([12, 24], design["results"], rejections, strict=True):
            assert result["available"] == 4000 and result["unavailable"] == 0 and result["minimum_uncapped_rank"] == q
            assert result["reject_at_nominal_5pct"] == rejected
            assert result["rejection_fraction_of_requested"] == result["rejection_fraction_of_available"] == rejected / 4000
            rows.append({"design": design["spec_id"], "restrictions": q, "dates": 207, "seed": seed, "hac_lags": band,
                "draws": 4000, "nominal_5pct_rejections": rejected, "rejection_fraction": rejected / 4000, "numerically_unavailable": 0})
        source_map({relative: sha(archive)}, artifact_hashes)
    assert seen == set(expected) and len(rows) == 4 and record["designs"] == rows
    source_map({str(p.relative_to(ROOT)): sha(p) for p in [path, reference_path, helper, baseline_path,
        ROOT / "scripts/selected_arrivals_null_calibration.py"]}, artifact_hashes)
    return {"status": record["status"], "current_execution_sha256": sha(path), "helper_source_sha256": sha(helper),
        "baseline_sha256": sha(baseline_path), "designs": rows, "selected_designs": 2, "reference_law_cases": 4,
        "native_fits_executed": 0, "limits": proof["limits"]}


def verify_final_audit_closure():
    destination = OUT / "collection/final_audit_closure_verification.json"
    with presentation_run(destination, "Final closure/arrival/native-evidence/offline-master binding"):
        master = OUT / "heat_economy.html";master_sha = sha(master)
        document = html.fromstring(master.read_text(), parser=html.HTMLParser(huge_tree=True))
        data = json.loads(document.get_element_by_id("collection-data").text)
        manifest = json.loads((OUT / "collection/manifest.json").read_text())
        old_heat_check = checked_json(OUT / "collection/extension_verification.json")
        assert old_heat_check["master_sha256"] == master_sha
        inventory_path = OUT / "aea_audit/evidence_inventory.json";inventory = json.loads(inventory_path.read_text())
        artifact_hashes, embedded = {}, {}
        guard_path = OUT / "derived_proof_guards/verification.json"
        guard_revision = checked_json(guard_path)
        assert guard_revision["native_fits_run"] == 0 and guard_revision["existing_native_ledgers_unchanged"] is True
        assert guard_revision["verifier_sha256"] == sha(ROOT / "scripts/verify_derived_proof_guards.py")
        assert guard_revision["revision_evidence_sha256"] == sha(ROOT / "replication/extensions/review_evidence/derived_proof_guard_revision.json")
        assert {row["path"] for row in guard_revision["sources"]} == {"scripts/literature_sensitivity.py", "scripts/annual_seasonal_literature.py", "scripts/literature_full_controls.py"}
        assert all(row["all_other_AST_identical"] and sha(ROOT / row["path"]) == row["after_sha256"] for row in guard_revision["sources"])
        assert len(guard_revision["fixture"]["genuine_analytical_tampers_rejected"]) >= 9
        for key, filename in CHAPTERS.items():
            for prefix in ["panel-", "tab-", "frame-"]:
                assert len(document.xpath('//*[@id=$identifier]', identifier=prefix + key)) == 1, (key, "Missing actual master navigation element", prefix)
            record = manifest[key];assert record["file"] == filename
            assert record["source_sha256"] == sha(OUT / filename)
            assert record["embedded_sha256"] == hashlib.sha256(data["reports"][key].encode()).hexdigest()
            for name, text in [(filename, (OUT / filename).read_text()), (key + " embedded", data["reports"][key])]:
                parsed = html.fromstring(text, parser=html.HTMLParser(huge_tree=True));offline(parsed, name)
                assert parsed.find("body") is not None, (name, "Missing body for master bridge")
                assert not any("@import" in node.text_content() or "https://" in node.text_content() or "http://" in node.text_content() for node in parsed.xpath('//style')), (name, "External CSS asset")
            embedded[key] = record
        from combined_report import DESTINATIONS
        for key, filename in CHAPTERS.items(): assert DESTINATIONS[filename] == key
        assert DESTINATIONS["clustered_sector_sensitivity.html"] == "cluster-sensitivity"
        closures = inventory["implementation_closures"]
        assert set(closures) == {"calendar_hac", "historical_narratives", "qualified_sector_CR1"}
        closure_records = {}
        for key, record in closures.items():
            path = ROOT / record["receipt"];proof = checked_json(path)
            assert record["status"] == "passed" and record["receipt_sha256"] == sha(path)
            if key == "calendar_hac":
                assert proof["native_fits_run"] == 0 and proof["currently_affected_saved_calendars"] == 0
                assert proof["maximum_old_new_covariance_error"] == 0 and proof["old_no_relabel_2N_boundary_bit_identical"] is True
                assert proof["legacy_saved_calendars"] >= 685 and len(proof["large_gap_fixtures"]) == 8
                source_map(proof["sources_sha256"], artifact_hashes);source_map(proof["protected_ledgers_and_references_sha256"], artifact_hashes)
            elif key == "historical_narratives":
                source_map(proof["source_sha256"], artifact_hashes);source_map(proof["outputs_sha256"], artifact_hashes);source_map(proof["original_artifacts_sha256"], artifact_hashes)
                assert proof["control_rows_labelled"] == 4
            else:
                assert proof["html_sha256"] == sha(OUT / CHAPTERS["cluster-sensitivity"])
                assert proof["native_fits_run"] == 0 and proof["individual_CR1_p_CI_stars_suppressed"] is True
                assert proof["native_DK_tables"] == proof["selector_runtime"]["selector_views"] == 180
                assert proof["shared_algebraic_CR1_numbers"] == 90
                source_map(proof["original_evidence_unchanged"], artifact_hashes)
            closure_records[key] = {"receipt": record["receipt"], "receipt_sha256": sha(path)}
        arrival = inventory["arrival_extension"];assert set(arrival["families"]) == set(EXPECTED)
        native_counts, request_total, aliases = {"Feols": 0, "Fepois": 0}, 0, 0
        family_records = {}
        for family, (requests, fits, native_class) in EXPECTED.items():
            record = arrival["families"][family];folder = ROOT / record["folder"]
            assert folder == OUT / "visitor_rigour" / family
            proof = checked_json(folder / "verification.json");specs = pd.read_csv(folder / "specifications.csv")
            ledger = pd.read_csv(folder / "pyfixest/fit_log.csv")
            assert len(specs) == record["logical_requests"] == requests
            assert len(ledger) == record["native_fits"] == fits
            assert ("fepois" if native_class == "Fepois" else "feols") in record["estimator"].lower()
            assert record["verification_sha256"] == sha(folder / "verification.json")
            assert record["native_ledger_sha256"] == sha(folder / "pyfixest/fit_log.csv") == proof["native_fit_log_sha256"]
            assert ledger.fit_id.tolist() == list(range(1, fits + 1)) and ledger.native_class.eq(native_class).all()
            assert ledger.native_module.str.startswith("pyfixest.").all()
            if native_class == "Feols":
                assert ledger.native_dropped_rows.eq(0).all() and ledger.native_dropped_coefficients.eq(0).all()
            else:
                assert ledger.native_converged.all() and ledger.dropped_row_positions.eq("[]").all()
                alias_rows = specs.loc[specs.alias_of.fillna("").ne("")];assert len(alias_rows) == 10
                assert alias_rows.native_fit_executed.eq(False).all();aliases += len(alias_rows)
                assert specs.native_spec_id.nunique() == 20
            assert proof.get("verified_artifacts_sha256"), (family, "Missing whole-array artifact binding")
            assert proof["artifact_binding_helper_sha256"] == sha(ROOT / "scripts/verification_artifacts.py")
            source_map(proof["verified_artifacts_sha256"], artifact_hashes)
            native_counts[native_class] += fits;request_total += requests
            family_records[family] = {"native_fits": fits, "logical_requests": requests, "native_class": native_class,
                "verification_sha256": sha(folder / "verification.json"), "whole_analytical_artifacts": len(proof["verified_artifacts_sha256"])}
        assert native_counts == {"Feols": 299, "Fepois": 20}
        assert arrival["native_OLS_fits"] == 299 and arrival["native_PPML_fits"] == 20 and arrival["logical_requests"] == request_total == 354
        assert arrival["aliases"] == aliases == 10
        arrival_equivalence = verify_arrival_equivalence(arrival["review_equivalence"], artifact_hashes)
        arrival_reference_diagnostics = verify_arrival_selected_reference(inventory["arrival_selected_reference_diagnostics"], artifact_hashes)
        assert inventory["combined_recorded_OLS_including_arrivals"] == old_heat_check["original_OLS_fits"] + old_heat_check["native_extension_fits"] + 299 == 16188
        assert inventory["combined_recorded_PPML_including_arrivals"] == old_heat_check["original_PPML_fits"] + 20 == 32
        diagnostics = OUT / "visitor_rigour/diagnostics"
        diagnostic_proof = checked_json(diagnostics / "verification.json")
        diagnostic_ui = checked_json(diagnostics / "report_verification.json")
        assert arrival["diagnostics"]["logical_requests"] == diagnostic_proof["counts"]["logical_model_outcomes"] == 354
        assert arrival["diagnostics"]["verification_sha256"] == diagnostic_ui["numerical_receipt_sha256"] == sha(diagnostics / "verification.json")
        assert diagnostic_ui["payload_sha256"] == sha(diagnostics / "report_payload.json") and diagnostic_ui["html_sha256"] == sha(diagnostics / "index.html")
        source_map(diagnostic_proof["verified_artifacts_sha256"], artifact_hashes)
        stata_records = {}
        for name, targets, folder in [("heat", 99, OUT / "stata_key_results"), ("arrivals", 38, OUT / "stata_key_results/arrivals")]:
            proof = checked_json(folder / "verification.json");record = inventory["stata_key_results"][name]
            assert record["targets"] == proof["targets"] == targets and record["verification_sha256"] == sha(folder / "verification.json")
            mandatory = proof["verified_native_artifacts_sha256"]
            assert len(mandatory) > targets, (name, "Incomplete native/MCP artifact binding")
            prefix = folder.relative_to(ROOT).as_posix() + "/"
            assert all(key.startswith(prefix) and not key.endswith(".log") and not key.endswith("/local_environment_check.json") for key in mandatory)
            for member in ["preparation_manifest.json", "specifications.json", "specifications.csv", "do/key_results.do", "do/postprocess.mata.do",
                           "native/master_stata_output.txt", "native/completed.txt", "native/runs.csv", "mcp/environment.json", "mcp/requirements.txt",
                           "mcp/transport_privacy_redactions.json", "mcp/arrival_key_results.json" if name == "arrivals" else "mcp/heat_key_results.json"]:
                assert prefix + member in mandatory, (name, "Required native evidence is unbound", member)
            assert any(key.startswith(prefix + "native_tables/") and key.endswith(".html") for key in mandatory)
            source_map(mandatory, artifact_hashes)
            runtime_binding = None
            if name == "arrivals":
                runtime_path = folder / "runtime_proof.json";runtime = json.loads(runtime_path.read_text())
                assert proof["runtime_proof_sha256"] == sha(runtime_path)
                assert runtime["portable_artifacts_sha256"] == runtime["artifact_sha256"] == mandatory
                required_sources = {"scripts/verify_stata_visitor_rigour.py", "scripts/stata_visitor_rigour.py", "scripts/stata_mcp_client.py", "scripts/stata_capture.py"}
                assert set(runtime["source_files"]) == required_sources and runtime["status"] == "captured"
                assert runtime["native_transport"] == "MCP stdio" and runtime["local_environment_check"]["fresh_native_estimation"] is False
                source_map(runtime["source_files"], artifact_hashes)
                runtime_binding = {"sha256": sha(runtime_path), "portable_artifacts": len(mandatory), "source_files": runtime["source_files"],
                    "scope": "Historical native capture with mandatory portable bytes; optional local probes/stderr are not required and no fresh Stata execution is claimed"}
            stata_records[name] = {"targets": targets, "verification_sha256": sha(folder / "verification.json"),
                "mandatory_native_artifacts": len(mandatory), "mandatory_map_sha256": hashlib.sha256(json.dumps(mandatory, sort_keys=True).encode()).hexdigest(),
                "runtime_proof": runtime_binding, "scope": "Selected cross-software audit targets, not extra PyFixest production fits"}
        stata_ui = checked_json(OUT / "stata_key_results/report_verification.json")
        assert stata_ui["html_sha256"] == sha(OUT / "stata_key_results.html")
        assert stata_ui["numerical_receipt_sha256"] == sha(OUT / "stata_key_results/verification.json")
        visitor_ui = checked_json(OUT / "visitor_rigour/report_verification.json")
        assert visitor_ui["html_sha256"] == sha(OUT / "visitor_arrivals_rigour.html")
        source_map(visitor_ui["verified_artifacts_sha256"], artifact_hashes)
        visitor_inventory = json.loads((OUT / "visitor_rigour/report_inventory.json").read_text())
        source_map(visitor_inventory["source_sha256"], artifact_hashes)
        visitor_doc = html.fromstring((OUT / "visitor_arrivals_rigour.html").read_text(), parser=html.HTMLParser(huge_tree=True))
        packed = visitor_doc.get_element_by_id("visitor-rigour-data").text
        decoded = gzip.decompress(base64.b64decode(packed));assert hashlib.sha256(decoded).hexdigest() == visitor_inventory["payload_sha256"] == visitor_ui["payload_sha256"]
        visitor_data = json.loads(decoded)
        assert visitor_data["diagnostics"] == json.loads((diagnostics / "report_payload.json").read_text())
        assert visitor_data["counts"]["origin_year"]["native_fits"] == 24
        selected_null = verify_selected_null_claim(inventory)
        record = inventory["arrival_external_review"]
        evidence = ROOT / "replication/extensions/review_evidence"
        receipt_path = evidence / "fable52-arrivals-receipt.json"
        report_path = evidence / "fable52-arrivals-review.md"
        adjudication_path = evidence / "arrivals_adjudication.md"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == record["status"] == "complete" and receipt["exitCode"] == 0 and receipt["error"] is None
        assert receipt["reportedModel"] == receipt["model"] == record["reported_model"] == "claude-fable-5-2"
        assert receipt["modelVerified"] is True and receipt["modelEvidence"] == "provider-report"
        assert receipt["provider"] == "claude" and receipt["effort"] == "max"
        assert receipt["mode"] == "isolated-write" and receipt["writerSandbox"] is not None
        assert receipt["preflight"]["status"] == "passed" and receipt["completedAt"] >= receipt["startedAt"] and receipt["elapsedMs"] > 0
        argv = receipt["argv"]
        assert argv[argv.index("--model") + 1] == "claude-fable-5-2" and argv[argv.index("--effort") + 1] == "max"
        assert record["receipt_sha256"] == sha(receipt_path) and record["report_sha256"] == sha(report_path)
        assert record["adjudication_sha256"] == sha(adjudication_path)
        assert len(report_path.read_text().strip()) > 1000 and len(adjudication_path.read_text().strip()) > 100
        arrival_review = {"receipt_sha256": sha(receipt_path), "review_sha256": sha(report_path),
            "adjudication_sha256": sha(adjudication_path), "reported_model": receipt["reportedModel"],
            "scope": "Completed exact-model review of the explicitly recorded snapshot, with separately bound native adjudication; no later-source or causal-validity certificate"}
        script = document.xpath('//script[not(@type="application/json")]')[0].text_content()
        prefixes = ('const DATA=', 'const number=', 'function renderOverview()', 'function renderPairedOverview()', 'function ensureFrame(', 'function showTab(', 'function renderCompleteOverview()',
            "document.addEventListener('click',", "document.querySelectorAll('[data-window]').forEach", "document.querySelectorAll('[data-window],[data-growth]').forEach")
        selected = '\n'.join(line for line in script.splitlines() if line.startswith(prefixes))
        code = NODE_HARNESS.replace("['complete-crisis','literature-models','new-stationarity','referee-audit']", "['visitor-rigour','stata-check','cluster-sensitivity']")
        assert "for(const target of ['visitor-rigour','stata-check','cluster-sensitivity'])" in code
        with tempfile.TemporaryDirectory(prefix="final-audit-master-") as temporary:
            path = Path(temporary) / "input.json";path.write_text(json.dumps({"script": selected, "data": {"overview": data["overview"], "complete_overview": data["complete_overview"],
                "reports": {key: "<html><body>independently source-bound chapter " + key + "</body></html>" for key in data["reports"]}}}))
            run = subprocess.run(["node", "-e", code, str(path)], text=True, capture_output=True)
            assert run.returncode == 0, run.stderr[-6000:]
            runtime = json.loads(run.stdout)
        assert sha(master) == master_sha
        source_map(artifact_hashes.copy(), {})
        report = {"status": "passed", "master_sha256": master_sha, "new_embedded_chapters": embedded, "implementation_closures": closure_records,
            "arrival_families": family_records, "arrival_native_OLS_fits": 299, "arrival_native_PPML_fits": 20, "arrival_logical_requests": 354, "arrival_count_mask_aliases": 10,
            "arrival_review_equivalence": arrival_equivalence,
            "arrival_selected_reference_diagnostics": arrival_reference_diagnostics,
            "whole_verified_artifact_files": len(artifact_hashes), "artifact_bindings_sha256": hashlib.sha256(json.dumps(artifact_hashes, sort_keys=True).encode()).hexdigest(),
            "diagnostic_requests": 354, "diagnostic_numerical_receipt_sha256": sha(diagnostics / "verification.json"), "diagnostic_presentation_receipt_sha256": sha(diagnostics / "report_verification.json"),
            "selected_Stata_audits": stata_records, "selected_known_null_claim": selected_null, "arrival_snapshot_review": arrival_review, "master_navigation_runtime": runtime,
            "derived_proof_guard_revision_sha256": sha(guard_path),
            "audit_inventory_sha256": sha(inventory_path), "verifier_sha256": sha(Path(__file__)),
            "scope": "Current byte/source/native-ledger/receipt/offline embedding and actual master-navigation binding. No source-physics rerun; diagnostics/calibration/Stata audits are not extra production fit families; no scientific-validity certificate"}
        persist(destination, report)
        print(f"Final audit closure passed: three chapters, 354 arrival requests, {len(artifact_hashes)} whole artifact bindings", flush=True)
        return report


if __name__ == "__main__": verify_final_audit_closure()
