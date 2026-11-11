"""Source-backed scope and claim audit, kept distinct from model estimates."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import urlsplit

import markdown
import numpy as np
import pandas as pd
from lxml import html as lhtml

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"replication/extensions/review_evidence"
FAMILIES=[("Long weather / complete GVA / crises","outputs/crisis_sensitivity",2655),
          ("Complete GVA with wind / PM2.5","outputs/complete_va_controls",750),
          ("Annual crisis companion","outputs/crisis_sensitivity/annual",1656),
          ("Annual distributed / quadratic literature models","outputs/literature_sensitivity/annual",2428),
          ("Subannual dynamics / YoY channels","outputs/literature_sensitivity",2361),
          ("Annual calendar-quarter exposure","outputs/literature_sensitivity/annual_seasonal",684),
          ("Subannual literature models with wind / PM2.5","outputs/literature_sensitivity/full_controls",574)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(path,label):
    return '<a download="'+html.escape(path.name)+'" href="data:text/plain;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'+html.escape(label)+' ↓</a>'


def review_html(text):
    document=lhtml.fragment_fromstring(markdown.markdown(text,extensions=["tables","fenced_code"]),create_parent="article")
    for node in document.xpath('.//script|.//iframe|.//object|.//embed|.//link|.//style|.//form'):
        node.drop_tree()
    for node in document.iter():
        for attribute in list(node.attrib):
            if attribute.lower().startswith("on"):
                del node.attrib[attribute]
    for node in document.xpath('.//a[@href]'):
        href=node.get("href")
        if urlsplit(href).scheme not in {"http","https"} and not href.startswith("#"):
            node.drop_tag()
    return lhtml.tostring(document,encoding="unicode")


def origin_year_mean_identity(folder):
    base=ROOT/"outputs/visitor_rigour"
    parent=pd.read_csv(base/"monthly/temperature_effects.csv",dtype={"unit_id":str},float_precision="round_trip")
    child=pd.read_csv(base/"origin_year/temperature_effects.csv",dtype={"unit_id":str},float_precision="round_trip")
    specs=pd.read_csv(base/"origin_year/specifications.csv")
    parent=parent.loc[parent.lag.eq(0)&parent.main_inference.eq(True)]
    child=child.loc[child.lag.eq(0)&child.main_inference.eq(True)]
    rows=[]
    for row in specs.itertuples():
        old=parent.loc[parent.spec_id.eq(row.source_spec_id)].set_index("unit_id").coefficient
        new=child.loc[child.spec_id.eq(row.spec_id)].set_index("unit_id").coefficient
        assert set(old.index)==set(new.index) and len(old)==row.n_sectors
        assert np.isclose(old.mean(),new.mean(),rtol=1e-10,atol=1e-8)
        rows.append({"spec_id":row.spec_id,"source_spec_id":row.source_spec_id,"origins":len(old),
            "shared_year_mean_slope":old.mean(),"origin_year_mean_slope":new.mean(),"difference":new.mean()-old.mean()})
    assert len(rows)==24
    path=folder/"origin_year_mean_identity.csv";pd.DataFrame(rows).to_csv(path,index=False)
    return path,{"cases":24,"maximum_absolute_mean_difference":max(abs(row["difference"]) for row in rows),
        "sha256":sha(path),"scope":"Equal-origin mean point slope only; individual slopes and covariance can change. Not independent national confirmation or a finite-sample inference identity."}


def build():
    folder=ROOT/"outputs/aea_audit";folder.mkdir(exist_ok=True)
    rows=[];receipts={};native_total=0
    for label,relative,expected in FAMILIES:
        path=ROOT/relative/"verification.json";proof=json.loads(path.read_text())
        assert proof["status"]=="passed",path
        log=ROOT/relative/"pyfixest/fit_log.csv"
        count=len(pd.read_csv(log,usecols=["fit_id"]))
        assert count==expected,(label,count,expected)
        native_total+=count
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{count:,}</td><td class='pass'>Independent check passed</td><td><code>{relative}/verification.json</code></td></tr>")
        receipts[relative]={"status":proof["status"],"sha256":sha(path),"native_fits":count,"native_ledger_sha256":sha(log)}
    review_sections=[];review_meta=[]
    for lane in ["design","inference","new-methods"]:
        receipt=EVIDENCE/f"fable52-{lane}-receipt.json";text=EVIDENCE/f"fable52-{lane}-review.md"
        info=json.loads(receipt.read_text())
        assert info["status"]=="complete" and info["modelVerified"] is True,receipt
        assert info["reportedModel"]=="claude-fable-5-2",receipt
        review_meta.append({"lane":lane,"receipt_sha256":sha(receipt),"report_sha256":sha(text),"reported_model":info["reportedModel"],"status":info["status"]})
        review_sections.append('<details><summary>Fable 5.2 · '+html.escape(lane)+' review</summary><p>'+download(text,"Download original reviewer report")+' · '+download(receipt,"Exact served-model receipt")+'</p>'+review_html(text.read_text())+"</details>")
    acceptance=EVIDENCE/"implementation_acceptance.json"
    fixed=json.loads(acceptance.read_text());assert fixed["status"]=="passed"
    scope={"status":"numerical_checks_passed_scientific_limits_remain","extension_native_fits":native_total,
           "original_native_OLS_fits":4781,"original_native_PPML_fits":12,
           "cumulative_recorded_OLS_fits":4781+native_total,"numerical_receipts":receipts,"reviews":review_meta,
           "baseline_reverification":fixed,
           "scope":"Existing release independently reverified twice; new families independently reconstructed. This is not a causal, stationarity, finite-sample coverage or AEA-acceptance certificate."}
    adjudication=EVIDENCE/"design_adjudication.md"
    diagnostic=EVIDENCE/"design_time_level_vs_project_DK_CR1.csv"
    assert adjudication.is_file() and diagnostic.is_file()
    df=pd.read_csv(diagnostic)
    assert len(df)==42
    convention_rows=[]
    for r in df.to_dict("records"):
        prob=lambda value:"<0.001" if value<.001 else f"{value:.3f}"
        convention_rows.append('<tr><td>'+html.escape(str(r["sample"]))+'</td><td>'+html.escape(str(r["model"]))+'</td><td>'+str(int(r["T"]))+'</td><td>'+str(int(r["time_df"]))+'</td><td>'+f'{r["b_panel"]:.3f}'+'</td><td>'+prob(r["cr1_p"])+"</td><td>"+prob(r["p_project_t(T-1)"])+"</td><td>"+prob(r["p_time_t(T-K)"])+"</td></tr>")
    scope["design_review_adjudication"]={"status":"selected_probes_replayed_broad_claims_qualified",
        "note_sha256":sha(adjudication),"diagnostic_sha256":sha(diagnostic),"historical_diagnostic_rows":42,
        "meaning":"Finite-sample-convention sensitivity on original static models; no change to requested main inference and no uniquely exact HAC reference claimed."}
    stationarity_folder=ROOT/"outputs/extension_stationarity"
    stationarity_receipt=stationarity_folder/"verification.json"
    stationarity=json.loads(stationarity_receipt.read_text())
    assert stationarity["status"]=="passed",stationarity_receipt
    stationarity_summary=json.loads((stationarity_folder/"summary.json").read_text())
    unitroot_counts=stationarity_summary["counts"]
    scope["extension_stationarity"]={"verification_sha256":sha(stationarity_receipt),
        "summary_sha256":sha(stationarity_folder/"summary.json"),"counts":unitroot_counts,
        "scope":stationarity_summary["scope"],"states":stationarity_summary["diagnostic_states"]}
    new_methods_note=EVIDENCE/"new_methods_adjudication.md"
    new_methods_proof=EVIDENCE/"new_methods_selected_fixed_design.json"
    new_methods_comparison=EVIDENCE/"new_methods_inference_comparison.csv"
    null_proof=json.loads(new_methods_proof.read_text())
    assert null_proof["status"]=="passed_selected_fixed_X_null_replay"
    assert null_proof["matches_independently_replayed_reference_counts"] is True
    assert null_proof["design"]=="va12__y_yoy_nsa__all_eligible__H1__distributed"
    assert null_proof["seed"]==20261119 and null_proof["repetitions"]==2000
    assert [row["rejections"] for row in null_proof["rejections"]]==[2000,712,2000,1091]
    current_null_path=ROOT/"outputs/aea_audit_calibration/summary.json"
    current_null=json.loads(current_null_path.read_text())
    assert current_null["status"]==null_proof["status"]
    assert current_null["array_bindings"]==null_proof["array_bindings"]
    assert current_null["rejections"]==null_proof["rejections"]
    assert current_null["seed"]==null_proof["seed"] and current_null["repetitions"]==2000
    for relative,digest in current_null["frozen_sources_sha256"].items():
        assert sha(ROOT/relative)==digest,(relative,"Current selected-null execution became stale")
    with np.load(ROOT/"outputs/literature_sensitivity/designs"/(null_proof["design"]+".npz"),allow_pickle=False) as archive:
        for key,binding in null_proof["array_bindings"].items():
            value=np.ascontiguousarray(archive[key])
            assert list(value.shape)==binding["shape"] and str(value.dtype)==binding["dtype"]
            assert hashlib.sha256(value.tobytes()).hexdigest()==binding["sha256"]
    current_null_copy=folder/"selected_null_current_execution.json"
    current_null_copy.write_bytes(current_null_path.read_bytes())
    scope["new_methods_review_adjudication"]={"status":"current_design_known_null_calibration_failure_reproduced",
        "note_sha256":sha(new_methods_note),"proof_sha256":sha(new_methods_proof),
        "comparison_sha256":sha(new_methods_comparison),"design":null_proof["design"],
        "draws":2000,"seed":20261119,"scope":null_proof["limits"],
        "current_execution_sha256":sha(current_null_copy),
        "historical_binding_policy":"Archived whole-file/ledger hashes identify the original replay; exact design arrays and the fresh execution bind the current result. Legitimate new native receipts need not retain historical timestamps."}
    closures={}
    for key,relative,copy_name in [
        ("calendar_hac","outputs/calendar_hac/verification.json","calendar_hac_verification.json"),
        ("historical_narratives","outputs/aea_audit/narrative_corrections.json","narrative_corrections.json"),
        ("qualified_sector_CR1","outputs/clustered_sector_sensitivity_qualified/verification.json","qualified_CR1_verification.json")]:
        path=ROOT/relative;record=json.loads(path.read_text());assert record["status"]=="passed"
        for field in ["sources_sha256","source_sha256","outputs_sha256","original_artifacts_sha256","protected_ledgers_and_references_sha256"]:
            for name,digest in record.get(field,{}).items():assert sha(ROOT/name)==digest,(key,name,"Closure became stale")
        if key=="qualified_sector_CR1":
            assert record["html_sha256"]==sha(ROOT/"outputs/clustered_sector_sensitivity_qualified.html")
            assert record["individual_CR1_p_CI_stars_suppressed"] is True
        copy=folder/copy_name
        if copy!=path:copy.write_bytes(path.read_bytes())
        closures[key]={"status":"passed","receipt":relative,"receipt_sha256":sha(path),"copy":str(copy.relative_to(ROOT))}
    scope["implementation_closures"]=closures
    arrival={"families":{},"native_OLS_fits":0,"native_PPML_fits":0,"logical_requests":0,"aliases":0}
    for branch in ["monthly","origin_year","annual","ppml"]:
        relative="outputs/visitor_rigour/"+branch;path=ROOT/relative/"verification.json";record=json.loads(path.read_text())
        assert record["status"]=="passed"
        for name,digest in record["verified_artifacts_sha256"].items():assert sha(ROOT/name)==digest,(branch,name,"Verified artifact changed")
        ledger=ROOT/relative/"pyfixest/fit_log.csv";native=pd.read_csv(ledger)
        expected_class="Fepois" if branch=="ppml" else "Feols"
        assert native.native_class.eq(expected_class).all()
        specs=pd.read_csv(ROOT/relative/"specifications.csv")
        item={"folder":relative,"verification_sha256":sha(path),"native_ledger_sha256":sha(ledger),
            "native_fits":len(native),"logical_requests":len(specs),"estimator":expected_class}
        arrival["families"][branch]=item;arrival["logical_requests"]+=len(specs)
        arrival["native_PPML_fits" if branch=="ppml" else "native_OLS_fits"]+=len(native)
        if branch=="ppml":arrival["aliases"]=int(specs.alias_of.notna().sum())
    assert (arrival["native_OLS_fits"],arrival["native_PPML_fits"],arrival["logical_requests"],arrival["aliases"])==(299,20,354,10)
    arrival_diagnostic=ROOT/"outputs/visitor_rigour/diagnostics/verification.json"
    diagnostic_proof=json.loads(arrival_diagnostic.read_text());assert diagnostic_proof["status"]=="passed"
    for name,digest in diagnostic_proof["verified_artifacts_sha256"].items():assert sha(ROOT/name)==digest
    arrival["diagnostics"]={"verification_sha256":sha(arrival_diagnostic),"logical_requests":354}
    scope["arrival_extension"]=arrival
    origin_mean_path,scope["origin_year_mean_identity"]=origin_year_mean_identity(folder)
    scope["combined_recorded_OLS_including_arrivals"]=4781+native_total+arrival["native_OLS_fits"]
    scope["combined_recorded_PPML_including_arrivals"]=12+arrival["native_PPML_fits"]
    scope["stata_key_results"]={}
    for branch,relative,targets in [("heat","outputs/stata_key_results",99),("arrivals","outputs/stata_key_results/arrivals",38)]:
        path=ROOT/relative/"verification.json";record=json.loads(path.read_text());assert record["status"]=="passed"
        checker="verify_stata_visitor_rigour.py" if branch=="arrivals" else "verify_stata_key_results.py"
        assert record["verifier_sha256"]==sha(ROOT/"scripts"/checker),(branch,"Stata port source changed")
        artifacts=record["verified_native_artifacts_sha256"];assert artifacts
        for name,digest in artifacts.items():assert sha(ROOT/name)==digest,(branch,name,"Native Stata evidence changed")
        item={"targets":targets,"verification_sha256":sha(path),"folder":relative,"verified_native_artifacts":len(artifacts),
            "native_artifact_bindings_sha256":hashlib.sha256(json.dumps(artifacts,sort_keys=True).encode()).hexdigest()}
        if branch=="arrivals":
            runtime_path=ROOT/relative/"runtime_proof.json";assert sha(runtime_path)==record["runtime_proof_sha256"]
            runtime=json.loads(runtime_path.read_text());assert runtime["artifact_sha256"]==runtime["portable_artifacts_sha256"]
            for name,digest in {**runtime["source_files"],**runtime["portable_artifacts_sha256"]}.items():
                assert sha(ROOT/name)==digest,(name,"Portable native runtime evidence changed")
            item["runtime_proof_sha256"]=sha(runtime_path)
        scope["stata_key_results"][branch]=item
    inference_adjudication=EVIDENCE/"inference_adjudication.md";assert inference_adjudication.is_file()
    scope["inference_review_adjudication"]={"note_sha256":sha(inference_adjudication),"scope":"Concrete source/display closures and qualified scientific claims; the full historical permutation grid was not independently rerun."}
    boundary_path=ROOT/"outputs/visitor_rigour/verification_boundaries.json"
    boundary=json.loads(boundary_path.read_text());assert boundary["status"]=="passed"
    assert len(boundary["lifecycle"])==5 and boundary["economic_or_native_model_fits"]==0
    for name,digest in boundary["source_sha256"].items():assert sha(ROOT/name)==digest
    boundary_copy=folder/"verification_boundaries.json";boundary_copy.write_bytes(boundary_path.read_bytes())
    scope["verification_boundary_checks"]={"status":"passed","receipt":str(boundary_path.relative_to(ROOT)),"sha256":sha(boundary_path),"scope":"Missing-file proof lifecycle, historical native-runtime portability and transport-startup privacy fixtures; no economic fits"}
    guard_path=ROOT/"outputs/derived_proof_guards/verification.json";guard=json.loads(guard_path.read_text())
    assert guard["status"]=="passed" and guard["native_fits_run"]==0 and guard["existing_native_ledgers_unchanged"] is True
    assert guard["verifier_sha256"]==sha(ROOT/"scripts/verify_derived_proof_guards.py")
    assert guard["revision_evidence_sha256"]==sha(EVIDENCE/"derived_proof_guard_revision.json")
    for change in guard["sources"]:
        assert change["all_other_AST_identical"] is True and sha(ROOT/change["path"])==change["after_sha256"]
    guard_copy=folder/"derived_proof_guard_verification.json";guard_copy.write_bytes(guard_path.read_bytes())
    scope["derived_proof_guard_revision"]={"status":"passed","receipt_sha256":sha(guard_path),"revision_evidence_sha256":guard["revision_evidence_sha256"],"scope":guard["scope"]}
    equivalence_path=ROOT/"outputs/visitor_rigour/review_equivalence.json"
    equivalence=json.loads(equivalence_path.read_text());baseline_path=EVIDENCE/"arrivals_review_baseline.json"
    baseline=json.loads(baseline_path.read_text())
    assert equivalence["status"]=="passed" and equivalence["numeric_or_status_differences"]==[]
    assert equivalence["baseline_sha256"]==sha(baseline_path)
    assert equivalence["verifier_source_sha256"]==sha(ROOT/"scripts/verify_arrivals_review_equivalence.py")
    assert equivalence["verified_at_utc"]
    for field in ["current_native_fit_logs_sha256","current_native_estimator_manifests_sha256"]:
        assert len(equivalence[field])==4
        for name,digest in equivalence[field].items():assert sha(ROOT/name)==digest,(name,"Reviewed-snapshot generation proof became stale")
    assert equivalence["guard_revision_sha256"]==guard["revision_evidence_sha256"]
    assert equivalence["guard_verifier_sha256"]==guard["verifier_sha256"]
    allowed={change["path"]:change for change in equivalence["authorized_post_snapshot_guard_only_source_changes"]}
    for name,digest in baseline["reviewed_sources_sha256"].items():
        current=sha(ROOT/name)
        if current!=digest:
            change=allowed[name];assert change["before_sha256"]==digest and change["after_sha256"]==current and change["all_other_AST_identical"] is True
    assert sum(row["requests"] for row in equivalence["counts"].values())==354
    assert sum(row["native_fits"] for row in equivalence["counts"].values())==319
    equivalence_copy=folder/"arrivals_review_equivalence.json";equivalence_copy.write_bytes(equivalence_path.read_bytes())
    scope["arrival_extension"]["review_equivalence"]={"status":"passed","receipt_sha256":sha(equivalence_path),"baseline_sha256":sha(baseline_path),
        "semantic_files":equivalence["semantic_files_checked"],"npz_arrays":equivalence["npz_arrays_checked"],"scope":equivalence["limits"]}
    arrival_benchmark_path=ROOT/"outputs/visitor_rigour/selected_review_benchmarks.json"
    arrival_benchmark=json.loads(arrival_benchmark_path.read_text())
    assert arrival_benchmark["status"]=="passed" and arrival_benchmark["native_fits_executed"]==0
    assert arrival_benchmark["source_sha256"]==sha(EVIDENCE/"arrivals_selected_benchmarks.py")
    assert arrival_benchmark["baseline_sha256"]==sha(baseline_path)
    expected_benchmarks={"region":(8,2026111102,[1045,1948]),"country":(41,2026111103,[4000,4000])}
    benchmark_rows=[]
    for design in arrival_benchmark["designs"]:
        kind=design["spec_id"].split("__")[0];q,seed,rejections=expected_benchmarks[kind]
        assert design["q"]==q and design["seed"]==seed and design["reps"]==4000 and design["t"]==207
        assert [row["reject_at_nominal_5pct"] for row in design["results"]]==rejections
        with np.load(ROOT/"outputs/visitor_rigour/monthly/designs"/(design["spec_id"]+".npz"),allow_pickle=False) as archive:
            for key,binding in design["array_bindings"].items():
                value=np.ascontiguousarray(archive[key]);header={"shape":list(value.shape),"dtype":value.dtype.str}
                assert header["shape"]==binding["shape"] and header["dtype"]==binding["dtype"]
                assert hashlib.sha256(json.dumps(header,sort_keys=True).encode()+value.tobytes()).hexdigest()==binding["sha256"]
        for row in design["results"]:
            assert row["unavailable"]==0 and row["available"]==4000 and row["minimum_uncapped_rank"]==q
            benchmark_rows.append({"design":design["spec_id"],"restrictions":q,"dates":207,"seed":seed,
                "hac_lags":row["hac_lags"],"draws":4000,"nominal_5pct_rejections":row["reject_at_nominal_5pct"],
                "rejection_fraction":row["rejection_fraction_of_requested"],"numerically_unavailable":0})
    assert len(benchmark_rows)==4 and len(arrival_benchmark["designs"])==2
    arrival_benchmark_copy=folder/"arrivals_selected_reference_diagnostics.json";arrival_benchmark_copy.write_bytes(arrival_benchmark_path.read_bytes())
    arrival_benchmark_csv=folder/"arrivals_selected_reference_diagnostics.csv";pd.DataFrame(benchmark_rows).to_csv(arrival_benchmark_csv,index=False)
    scope["arrival_selected_reference_diagnostics"]={"status":"passed_selected_reference_law_replay","current_execution_sha256":sha(arrival_benchmark_path),
        "helper_source_sha256":arrival_benchmark["source_sha256"],"baseline_sha256":sha(baseline_path),"designs":benchmark_rows,"limits":arrival_benchmark["limits"]}
    relocated_base=EVIDENCE/"relocated_native";relocated_inventory_path=relocated_base/"inventory_after_eq.json"
    relocated_inventory=json.loads(relocated_inventory_path.read_text())
    for member in relocated_inventory["files"]:
        path=(relocated_base/member["path"]).resolve();assert path.is_relative_to(relocated_base.resolve())
        assert path.stat().st_size==member["size_bytes"] and sha(path)==member["sha256"]
    relocated_run=json.loads((relocated_base/"guard_revision_run.json").read_text())
    assert relocated_run["status"]=="passed" and relocated_run["current_saved_native_fits"]==11427
    assert relocated_run["frozen_526_inputs_unchanged"] is True and relocated_run["relocated_sources_unchanged"] is True
    assert relocated_inventory["historical_inventory_sha256"]==sha(relocated_base/"inventory.json")
    relocated_eq=json.loads((relocated_base/relocated_inventory["post_execution_eq_receipt"]).read_text())
    assert relocated_eq["status"]=="passed" and relocated_eq["baseline_sha256"]==sha(baseline_path) and relocated_eq["numeric_or_status_differences"]==[]
    for source,name in [("guard_revision_run.json","relocated_native_execution.json"),("inventory_after_eq.json","relocated_native_evidence_inventory.json"),("README_after_eq.md","relocated_native_scope.md")]:
        (folder/name).write_bytes((relocated_base/source).read_bytes())
    scope["relocated_native_replay"]={"status":"passed","receipt_sha256":sha(relocated_base/"guard_revision_run.json"),
        "inventory_sha256":sha(relocated_inventory_path),"saved_native_fits":11427,"scope":relocated_run["scope"],
        "post_refit_exact_equivalence_sha256":sha(relocated_base/relocated_inventory["post_execution_eq_receipt"]),"post_refit_equivalence_scope":relocated_inventory["scope"]}
    arrival_review_receipt=EVIDENCE/"fable52-arrivals-receipt.json"
    arrival_review_report=EVIDENCE/"fable52-arrivals-review.md"
    arrival_review_html='<p>The focused international-arrivals external review has not yet been recorded as complete.</p>'
    if arrival_review_receipt.exists() and arrival_review_report.exists():
        record=json.loads(arrival_review_receipt.read_text())
        assert record["status"]=="complete" and record["modelVerified"] is True and record["reportedModel"]=="claude-fable-5-2"
        scope["arrival_external_review"]={"status":"complete","reported_model":record["reportedModel"],
            "receipt_sha256":sha(arrival_review_receipt),"report_sha256":sha(arrival_review_report)}
        arrival_review_html='<details><summary>Fable 5.2 · focused international-arrivals review</summary><p>'+download(arrival_review_report,'Original arrivals reviewer report')+' · '+download(arrival_review_receipt,'Exact served-model receipt')+'</p>'+review_html(arrival_review_report.read_text())+'</details>'
        arrival_adjudication=EVIDENCE/"arrivals_adjudication.md"
        if arrival_adjudication.exists():
            scope["arrival_external_review"]["adjudication_sha256"]=sha(arrival_adjudication)
            arrival_review_html='<details><summary>Arrivals review adjudication</summary><p>'+download(arrival_adjudication,'Accepted findings and scientific limits')+'</p>'+review_html(arrival_adjudication.read_text())+'</details>'+arrival_review_html
    (folder/"evidence_inventory.json").write_text(json.dumps(scope,indent=2)+"\n")
    body=f'''<header><div class="eyebrow">Singapore heat & economy · evidence audit</div><h1>What the checks establish—and what remains open</h1><p>All new regression families have passed independent source/calendar/design/numerical checks. Three exact-model Fable 5.2 review lanes and native independent checks examine the original collection and the literature extension. The reviewed snapshot and final corrected result are distinguished below.</p></header>
<div class="warning"><strong>Assessment:</strong> a comprehensive, reproducible exploratory analysis. Causal identification, small-sample test calibration and conference acceptance remain scientific judgments; no numerical verifier certifies them.</div>
<div class="warning"><strong>A demonstrated inference problem:</strong> the unchanged VA12 long-history NSA-YoY distributed-lag H1 design rejects a true 60-coefficient zero null in 2,000 of 2,000 fixed-X IID Gaussian draws at both calendar bandwidths. The 11-restriction equality-of-cumulative-responses test rejects 35.6% and 54.55% at a nominal 5%. This is a verified selected-design counterexample, not the estimated rejection rate of every model or proof that the observed GDP effect is zero. Full numerical rank does not validate the reference F law. These multi-restriction p-values are retained as nominal/asymptotic diagnostics, without headline heterogeneity or mechanism claims. <a href="#calibration">Inspect the portable replay</a>.</div>
<nav><a href="heat_economy.html#complete-crisis">Complete GVA / crises</a><a href="heat_economy.html#literature-models">Literature specifications</a><a href="#reviewers">Reviewer reports</a><a href="#reproduce">Reproduce</a></nav>
<section><h2>Checks and coverage</h2><p>The heat/VA/IIP extension records <strong>{native_total:,} native OLS fits</strong>. The original package records 4,781 OLS fits and 12 separate PPML models. Repeated fits, aliases, alternative covariances and common data are not independent hypotheses or independent weather realizations. The original 4,781 ledger records contain 3,984 distinct exact X/Y hash pairs.</p><div class="table-wrap"><table><thead><tr><th>Extension family</th><th>Native fits</th><th>Numerical status</th><th>Machine-readable proof</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>Independent checks rebuild official SA/NSA real-output transformations, actual-calendar lags and crisis masks, complete sector definitions including dwellings, fixed-effect and trend designs, native coefficient/residual mappings, covariance matrices and cross terms, cumulative/marginal effects and rank-checked restrictions. The HTML has separate source-backed presentation/runtime tests. “Passed” concerns that stated computational scope.</p></section>
<section><h2>Corrections made during this audit</h2><div class="table-wrap"><table><thead><tr><th>Finding</th><th>Response and evidence</th><th>Meaning for results</th></tr></thead><tbody>
<tr><td>Historical receipt versus fresh verification collision</td><td>Six unchanged CSVs had stale hashes inside a historical auxiliary receipt. A narrow eight-path receipt-history mechanism archives each fresh subprocess receipt and preserves the immutable original checkpoint. Eight mutation/failure lifecycle fixtures and two consecutive 23-step existing-output verification runs passed.</td><td>Original numerical inputs/results and v1.0.0 references were unchanged. Historical receipts are not represented as current proof. This was a real verification-workflow defect, not a correction to economic coefficients.</td></tr>
<tr><td>Avoidable AR-history truncation at comparison boundaries</td><td>The initial new subannual run looked up lagged growth in a cropped parent. Final construction uses full raw output-growth histories, then explicit target-calendar alignment and crisis masks. Superseded evidence is archived separately; the final estimates are independently rechecked.</td><td>Valid raw observations are not discarded merely because a comparison parent starts later. Necessary missing history, explicit matching and crisis exclusions are separate causes of sample loss.</td></tr>
<tr><td>Complete GVA component coverage</td><td>New broad 12-component and detailed 20-component panels include official ownership of dwellings. Earlier 11/19-unit definitions remain labelled comparison sets. Nominal accounting and real-chain nonadditivity are shown separately.</td><td>Coverage is complete; chained real components need not add. Official total regressions always use published aggregate GVA. Equal-component panel growth is a different estimand.</td></tr>
<tr><td>Current-only YoY did not contain the unrestricted history</td><td>Added all intermediate temperature/rainfall lags, separate dynamic-growth sensitivity, joint endpoint/accumulation model, and exact nested restrictions. Annual lag and nonlinear/seasonal companions retain their own explicit adaptations.</td><td>Existing current/difference/rolling/endpoint comparisons remain. The new specifications answer different conditional timing questions and are not selected from their significance.</td></tr>
<tr><td>Overly restrictive multi-feature joint-test rank limit</td><td>A one-temperature contrast-rank bound was incorrectly extended to stacked temperature lags. Corrected multi-feature tests use the actual covariance rank with the universal calendar-score bound T−1, or G−1 for sector CR1. Thirty-two subannual availability decisions change, including twelve global lag-vector tests; seasonal multi-feature restrictions are rechecked under the same principle.</td><td>Native coefficients and covariance matrices are unchanged. Newly computable tests can be extremely high-dimensional (234 restrictions from 237 dates) and are explicitly flagged as fragile; computability is not a small-sample validity certificate.</td></tr>
<tr><td>Calendar-gap handling and historical prose</td><td>The actual-calendar HAC wrapper is independently checked on all 685 saved original calendars and eight gap/boundary fixtures. Current covariance matrices are bit-identical. Corrected Markdown notes explicitly identify each row’s covariance and reference degrees of freedom.</td><td>Original numerical results are preserved. The latent group-index shortcut can no longer compress a sparse calendar. Corrected notes are separate, reviewable artifacts.</td></tr>
<tr><td>Unsupported sector-CR1 interaction inference</td><td>The qualified sensitivity retains all 90 native fits and coefficients, reports the common algebraic CR1 SE once, and suppresses unsupported individual CR1 p-values, intervals and stars. Original raw outputs remain labelled historical downloads.</td><td>Small nonzero clustered SEs do not establish reliable inference for an industry slope supported by one industry. The calendar-based comparison remains separately identified.</td></tr>
<tr><td>Native reproduction order and derived-proof dependencies</td><td>A relocated native rebuild found full-controls fitting was scheduled before its parents’ rank/span postprocessors. That order is corrected. Three producers also froze derived verification or presentation records, preventing a legitimate replacement audit. Only those guard lists/functions changed; executable AST and source comparisons establish all other model code unchanged.</td><td>Actual data, formulas, samples, coefficients, covariance, native ledgers and substantive test/rank/span archives remain protected. A new full run regenerates provenance under the explicitly sealed source revision; original v1 references are preserved.</td></tr>
</tbody></table></div></section>
<section id="implementation"><h2>Replayed implementation closures</h2><p>{download(guard_copy,'Derived-proof guard AST and tamper fixtures')} · {download(EVIDENCE/'derived_proof_guard_revision.json','Archived preimages, source diffs and revision hashes')}</p><p>{download(folder/'calendar_hac_verification.json','Calendar-HAC identity and gap checks')} · {download(folder/'narrative_corrections.json','Corrected-note source and coefficient bindings')} · {download(folder/'qualified_CR1_verification.json','Qualified interaction-CR1 presentation proof')}</p><p>{download(folder/'first_results_corrected.md','Corrected initial VA note')} · {download(folder/'iip_results_corrected.md','Corrected IIP note')}</p><p>Five missing-file lifecycle fixtures establish that a failed replacement audit archives the previous passed receipt and marks the current check failed before any economic calculation. Stored-native Stata verification checks packaged historical runtime metadata without requiring this machine’s ado paths; present local mismatches are rejected and unavailable local sources are labelled. This is distinct from a new licensed native run. Transport startup identifiers are redacted in distributable copies with private raw preimages and explicit provenance; native calculations and MCP response JSON remain unchanged. {download(boundary_copy,'Inspect lifecycle, portability and privacy fixtures')}</p><details><summary>Inference-review adjudication</summary><p>{download(inference_adjudication,'Download inference adjudication')}</p>{review_html(inference_adjudication.read_text())}</details></section>
<section id="arrivals"><h2>International arrivals and independent Stata implementation</h2><div class="warning"><strong>Direct arrivals calibration counterexample:</strong> two long-history MoM H1 designs reject nominal 5% equal-origin-slope nulls far too often under 4,000 fixed-design Gaussian draws: nine regions reject 1,045/4,000 at HAC12 and 1,948/4,000 at HAC24; 42 named markets reject 4,000/4,000 at both bandwidths. Every draw passes the declared rank/HC0 availability gates. This is not the empirical error law of observed arrivals, a universal restriction-count cutoff, a verdict on all scalar tests, or a replacement covariance/reference rule. Nominal multi-DF p-values remain exploratory. {download(arrival_benchmark_copy,'Fresh selected-design diagnostic')} · {download(arrival_benchmark_csv,'Four exact rejection-count summaries')}</div><p>The new arrivals extension records <strong>299 native OLS fits and 20 native PPML fits</strong>: 260 monthly OLS, 24 origin-specific-year OLS, 15 annual OLS and 20 Poisson count models. Its 354 logical requests also include ten exact count-mask aliases and 25 unavailable OLS designs. Combined with the original release and heat extension, the recorded totals are <strong>16,188 OLS fits and 32 PPML fits</strong>. These counts describe execution, not independent statistical tests.</p><p>Growth is computed on the full unadjusted count calendar before weather alignment or exclusions. Zeros remain in Poisson count models; no pseudocount enters a log-growth series. Complete 10-region and 52-market partitions reconcile to the published total in every source month. Eight PPML complete-partition aggregation equalities follow from the shared-regressor likelihood and are accounting/algebra checks, not independent evidence of a heat effect. The 24 origin-specific-year sensitivities absorb origin-specific annual shocks while retaining within-year weather variation. Calendar-HAC 12/24 is primary for these new arrivals comparisons; unit-clustered common-slope results remain a labelled sensitivity.</p><p>The exact-window arrivals appendix covers all 354 requests and 6,623 distinct outcome/regressor arrays. It completes 65 descriptive screens on contiguous Poisson input histories; 6,558 arrays are untestable on their actual short or gapped calendars. It does not clear the growth equations as stationary or impose ordinary I(0) tests as a prerequisite for a count-model conditional-mean specification. Short full-control dynamic YoY fits can have only one to three residual time degrees of freedom; some requested designs are unavailable with every requested control retained.</p><p>Across all 24 matched origin × year FE comparisons, the equal-origin mean temperature point slope equals the shared-year H1 mean to numerical precision. This follows from the balanced design: the added origin-year contrasts do not change its common mean channel. Individual slopes and their covariance can change; the mean equality is not independent national robustness. {download(origin_mean_path,'All 24 mean-slope identity checks')}</p><p>Actual Stata MCP execution independently re-estimated <strong>99 heat keys and 38 arrivals keys</strong> using Stata 17/IC and <code>reghdfe</code>, with native logs, reusable do-files, native tables and separate Python ports. Native Stata and project covariance finite-sample factors remain explicitly distinct. Calendar covariance and sector-total interaction effects use all covariance cross terms. Eight arrivals and twelve heat interaction keys have degenerate native unit-cluster covariance; no supported p-value or interval is manufactured. The 137 Stata runs are independent implementation checks of selected keys, not additions to the economic-model counts above. The other lag/nonlinear/PPML models are not claimed to have been replicated in Stata.</p><p><a href="heat_economy.html#visitor-rigour">Open arrivals models, diagnostics and Stata tables</a> · <a href="heat_economy.html#stata-check">Open heat Stata checks and downloads</a></p><p>The fresh exact-semantic bridge compares 534 model/sample/export files and 8,338 native archive arrays with the immutable reviewed generation, including every one of its 354 requests and 319 native ledger rows. The three approved guard-only source changes preserve every other model AST node. This extends numeric coverage of the reviewed equations without claiming the external reviewer examined later report, driver or privacy code. {download(equivalence_copy,'Current reviewed-snapshot semantic bridge')} · {download(baseline_path,'Immutable reviewed numeric baseline')}</p>{arrival_review_html}</section>
<section><h2>Scientific limitations that additional regression fits do not remove</h2>
<h3>One national weather realization per date</h3><p>Temperature is identical across sectors in each date. Across the 27 independently probed old common-weather designs, pooled temperature coefficients equal the corresponding equal-sector-mean time-series coefficients to numerical precision. Full actual-date FE absorb the common national temperature effect. Sector × calendar-season and year FE leave temperature variation but do not remove every national macroeconomic shock. A defensible causal argument must address that remaining temporal confounding.</p>
<h3>Clustering, short samples and high-dimensional tests</h3><p>Requested sector/industry CR1 appears for common pooled slopes, beside calendar-based alternatives. H1 absolute sector slopes rely on one sector each, so sector clustering does not create independent supporting units. The separately retained 90-fit sector-CR1 interaction sensitivity has no reliable equality-test covariance: rank collapse is reported, not hidden. Calendar HAC/DK also relies on its own assumptions and adequate temporal information.</p><p>The full-control quarterly sample contains 29 weather dates and 12 residual time df; event-date exclusion leaves 19 dates and five df, and strict growth-window cleaning leaves 17 dates and four df. Some available annual/profile Wald tests are ill-conditioned or close to their time-rank bound. An available p-value is not evidence of good finite-sample size.</p>
<h3>Calendar sensitivity and pollution controls</h3><p>In the old 11-sector short-growth bridge, the long weather coefficient is −1.43; using the 29 full-control dates without adding wind/PM gives −26.72; adding those controls gives −27.05; excluding 2020–2022 current dates gives −0.03. The large change is predominantly a sample-date change. It does not establish that pollution controls uncovered a stable causal heat effect. The causal role of contemporaneous pollution/humidity is not proven by these regressions.</p>
<h3>Outcome support, stationarity and interpretation</h3><p>Current-date exclusions can retain YoY growth whose denominator is a crisis year. The strict variants remove the full declared output support, including growth lags where present. Gaps remain actual calendar gaps. Growth is primary, but ordinary ADF/KPSS screens do not prove stationarity. Existing mixed/short/gapped screens and descriptive level regressions remain clearly qualified.</p><p>The independently checked new-model appendix maps {unitroot_counts['logical_model_outcomes']:,} requested model/outcome combinations to {unitroot_counts['distinct_diagnostics']:,} distinct actual-calendar arrays. It completes {unitroot_counts['completed_diagnostics']:,} diagnostic screens; {unitroot_counts['unavailable_diagnostics']:,} arrays are untestable because of gaps or short histories. These arrays include outcomes and actual stochastic weather/AR bases, with identical arrays deduplicated. They are not independent tests or independent economic series. All {unitroot_counts['test_rows']:,} ADF/KPSS rows and the exact source/design bindings are independently reconstructed. No observation is deleted to obtain stationarity clearance. <a href="heat_economy.html#new-stationarity">Inspect exact-calendar diagnostics and download their evidence</a>.</p><p>Cumulative temperature coefficients include every covariance term. A direct sum in a model with lagged Y is not an unconditional long-run multiplier, and a finite lag horizon cannot establish a permanent climate effect. Quadratic derivatives are local to observed Singapore temperatures. Unit-specific dynamic FE bias and climate extrapolation remain unresolved.</p>
<h3>Multiple testing and changing economic partitions</h3><p>This is a post-exploration research grid. Pointwise stars and within-table Holm adjustments do not control selection over all outcomes, cuts, samples, lag lengths and inference choices. A significant coefficient beside an insignificant coefficient is not itself a significant difference. National totals, equal-sector means and finer partitions change weights and sometimes history; matched calendars do not make those estimands identical.</p><p>Published IIP totals are used directly. Exact historical component reconstruction remains limited by available compilation weights and source vintages; contemporary fixed weights are not imposed on the entire history. Existing reconciliation exceptions and rounding bounds remain disclosed rather than force-fitted.</p></section>
<section id="calibration"><h2>Accepted calibration failure and bounded reference sensitivities</h2><p>The portable fixed-design diagnostic reads the actual current NPZ and uses seed 20261119, 2,000 draws and independent QR/calendar-score calculations. It reproduces the original reviewer experiment’s four rejection counts. It runs no native regression fits and changes no economic result. The selected 60 restrictions are fewer than the design’s 116 residual time df, so a new q≤time-df rank guard would not repair the demonstrated failure.</p><p>{download(new_methods_proof,'Archived current-design known-null proof')} · {download(current_null_copy,'Fresh portable execution')} · {download(new_methods_comparison,'Observed-reference and null diagnostics')} · {download(EVIDENCE/'new_methods_replay_receipt.json','Independent replay receipt')}</p><pre>uv run --locked python scripts/selected_null_calibration.py</pre><details><summary>New-methods adjudication: accepted failures, interpretation fixes and limits</summary><p>{download(new_methods_note,'Download adjudication')}</p>{review_html(new_methods_note.read_text())}</details><p>Hotelling reference calculations apply only under their stated balanced iid Gaussian sector-vector model; they are not a universal cluster correction. Changing calendar variance multipliers or Student degrees of freedom is reported as a sensitivity rather than a uniquely exact HAC law. Dynamic static-level labels and duplicate algebraic nulls are qualified in the results chapter. No rule requires both CR1 and calendar p-values to cross a threshold.</p></section>
<section id="reviewers"><h2>Independent reviewer reports and scope</h2><p>Design/inference lanes examine the immutable original e47edee snapshot. The new-methods lane examines the initial literature implementation snapshot. Later raw-history/rank corrections, the seasonal/full-control companions and final HTML are covered by the separately identified final numerical/presentation checks; a snapshot review is not mislabelled as a review of later source changes. Model identity is checked from provider receipts, not inferred from a rolling alias.</p><h3>Adjudication before accepting reviewer conclusions</h3><p>Selected original-model coefficients, panel/mean-series equivalence, CR1 dispersion identities and YoY decompositions reproduce. The selected fixed-X Gaussian null failure above is accepted. Broader weather-block permutation claims need exchangeability assumptions; changing HAC finite-sample conventions does not make one Student reference exact, and neither check establishes absence of every effect. Shared national exposure and sample leverage remain substantial concerns. Requested sector CR1 stays main, with labelled calendar alternatives.</p><details><summary>Original-model accepted, qualified and unestablished claims</summary><p>{download(adjudication,'Download adjudication')} · {download(EVIDENCE/'design_replay_receipt.json','Exact replay proof')}</p>{review_html(adjudication.read_text())}</details><details><summary>Historical static-model finite-sample-convention sensitivity · 42 comparisons</summary><p>This is an independently replayed audit diagnostic on the original snapshot, not a replacement baseline or a uniquely valid finite-sample test. Temperature coefficients stay identical; variance correction and reference degrees of freedom differ. Ten project-DK 5% decisions cross that threshold under the alternative time-level convention.</p><p>{download(diagnostic,'Download complete diagnostic with standard errors')}</p><div class="table-wrap"><table><thead><tr><th>Original sample</th><th>Y</th><th>Dates</th><th>Time residual df</th><th>β</th><th>Sector CR1 p</th><th>Project DK p</th><th>Time-level convention p</th></tr></thead><tbody>{''.join(convention_rows)}</tbody></table></div></details><p>The unedited reviewer reports below retain their own evaluative wording; the adjudications above state what was accepted, qualified or not established.</p>{''.join(review_sections)}<details><summary>Native scientific probes and receipt-history acceptance</summary><p>{download(EVIDENCE/'native_findings.json','Native findings')} · {download(acceptance,'Verify-only acceptance evidence')}</p><p>The native claim review is scoped to the original release. Its historical statement about an initial authentication failure predates the successful exact Fable 5.2 reviews and is not the current authentication status.</p></details></section>
<section id="reproduce"><h2>Reproduce and inspect</h2><p>A relocated native execution and independent mathematical replay passed for all 11,427 new saved fits, with the 526 frozen inputs unchanged. This used the existing locked environment; four unaffected numerical passes were reused after the guard revision and six were rerun. Preserved failures document the corrected dependency order and bookkeeping defects. It is not a fresh-environment or final-source-seal certificate. {download(folder/'relocated_native_execution.json','Exact relocated execution receipt')} · {download(folder/'relocated_native_evidence_inventory.json','Portable transcript and numerical-proof inventory')} · {download(folder/'relocated_native_scope.md','Execution scope and failure history')}</p><pre>uv sync --locked --python 3.11
uv run --locked python scripts/replicate_extensions.py inventory
uv run --locked python scripts/replicate_extensions.py verify
uv run --locked python scripts/replicate_extensions.py reproduce</pre><p>The separate extension source manifest and original frozen input hashes are checked. HTTP is disabled; no upstream data refresh or silent reference reset occurs. A failed subprocess produces a failed execution receipt. The subannual and literature families retain full design/residual/covariance archives. The annual crisis companion retains samples, formulas, reported temperature/profile covariance and native design/response hashes; its verifier independently reconstructs the required full designs and reported inference. Masks, unavailable requests, native ledgers and numerical verifiers remain available.</p><p>{download(folder/'evidence_inventory.json','Download audit evidence inventory')}</p></section>'''
    css='''body{margin:0;background:#f3f7fa;color:#15334a;font:14px/1.65 "Avenir Next","Segoe UI",Arial,sans-serif}main{max-width:1280px;margin:auto;padding:32px}header{border-bottom:4px solid #087e7a;padding-bottom:22px}h1{font-size:37px;line-height:1.2}h2{font-size:25px}h3{font-size:18px}.eyebrow{font:11px Menlo,monospace;letter-spacing:1.5px;text-transform:uppercase;color:#5d7381}section{background:white;border:1px solid #d6e1e6;border-radius:9px;padding:24px;margin:22px 0}.warning{background:#fff3df;border-left:4px solid #bf851e;padding:16px;margin:20px 0}nav{display:flex;flex-wrap:wrap;gap:20px;margin:20px 0}a{color:#126f97}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:10px;text-align:left;border-bottom:1px solid #d6e1e6;vertical-align:top}th{background:#eaf1f4}.table-wrap{overflow:auto}code,pre{font:11px/1.7 Menlo,monospace;overflow-wrap:anywhere}pre{background:#eef3f6;padding:15px;white-space:pre-wrap}.pass{color:#08745b}details{margin:20px 0}summary{font-weight:700;cursor:pointer}article{border-left:2px solid #d6e1e6;padding-left:18px}article h1{font-size:26px}article h2{font-size:22px}article table{display:block;overflow:auto}@media(max-width:700px){main{padding:16px 12px}section{padding:16px}h1{font-size:29px}}'''
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Heat economy evidence audit</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>'
    path=ROOT/"outputs/aea_audit.html";path.write_text(document)
    print(f"Wrote {path}; {native_total} extension fits and {len(review_meta)} exact-model review receipts")
    return path


if __name__=="__main__":build()
