"""Probe 01 - scope counts, native-ledger binding and source generation.

Independent recount of logical requests / native estimates from the saved
specification tables and native fit logs; binds every saved design archive
(Y/beta/residual bytes) to its native receipt hash; re-hashes every file that a
family manifest pins.  Read-only; writes audit_probes/probe01_result.json.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe01_scope_counts_source_generation.py
"""
from _probe_common import BASE, HERE, ROOT, sha, write

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HEX = re.compile(r"^[0-9a-f]{64}$")


def backend_hash(array):
    values = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()


def plain_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pinned(node, found):
    """Collect every {relative path: sha256} pin anywhere inside a manifest."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and HEX.match(value) and ("/" in key or key.endswith((".py", ".md", ".json", ".csv"))):
                found.setdefault(key, set()).add(value)
            else:
                pinned(value, found)
    elif isinstance(node, list):
        for value in node:
            pinned(value, found)
    return found


result = {"probe": "01 scope counts / native ledger binding / source generation", "families": {}}
total_requests = total_native = 0
all_ids = []
for family in ["monthly", "annual", "ppml", "origin_year"]:
    folder = BASE / family
    specs = pd.read_csv(folder / "specifications.csv", low_memory=False)
    log = pd.read_csv(folder / "pyfixest/fit_log.csv", low_memory=False)
    receipt = json.loads((folder / "pyfixest/estimator_manifest.json").read_text())
    entry = {"requests": len(specs), "unique_spec_ids": int(specs.spec_id.nunique()),
             "status_counts": {str(k)[:60]: int(v) for k, v in specs.status.value_counts(dropna=False).items()},
             "fit_log_rows": len(log), "fit_log_sha256_matches_receipt": sha(folder / "pyfixest/fit_log.csv") == receipt.get("fit_log_sha256"),
             "receipt_native_count": receipt.get("native_fits_recorded", receipt.get("native_calls")),
             "native_classes": sorted(log.native_class.astype(str).unique()), "estimator_versions": sorted(log.estimator_version.astype(str).unique())}
    all_ids += list(family + "::" + specs.spec_id)
    estimated = specs[specs.status.eq("estimated")]
    binding = {"checked": 0, "y_hash_mismatch": 0, "beta_hash_mismatch": 0, "residual_hash_mismatch": 0, "label_or_shape_mismatch": 0,
               "missing_design_npz": 0, "unestimated_with_finite_beta": 0, "sample_sha_mismatch": 0, "sample_sha_checked": 0}
    for column in ["sample_sha256", "input_sample_sha256"]:
        path_column = column.replace("_sha256", "_csv")
        if column in specs and path_column in specs:
            pairs = specs[[path_column, column]].dropna().drop_duplicates()
            for path, value in pairs.itertuples(index=False):
                binding["sample_sha_checked"] += 1
                binding["sample_sha_mismatch"] += int(sha(ROOT / path) != value)
    if family == "ppml":
        native_specs = specs[specs.native_fit_executed.astype(str).eq("True")]
        aliases = specs[specs.alias_of.notna() & specs.alias_of.astype(str).ne("")]
        alias_table = pd.read_csv(folder / "aliases.csv")
        entry.update(native_executed_specs=len(native_specs), alias_specs=len(aliases), alias_table_rows=len(alias_table),
                     alias_variants=sorted(aliases.sample_variant.unique()), alias_targets_all_native_event=bool(
                         aliases.alias_of.isin(native_specs.spec_id).all() and aliases.alias_of.str.endswith("event_dates_excluded").all()),
                     alias_rows_marked_not_fitted=bool(aliases.native_fit_executed.astype(str).eq("False").all()),
                     log_spec_ids_equal_native_specs=sorted(log.spec_id) == sorted(native_specs.spec_id),
                     log_all_converged=bool(log.native_converged.astype(str).eq("True").all()),
                     log_postprocessing_complete=bool(log.postprocessing_status.eq("complete").all()),
                     log_dropped_rows_all_empty=bool(log.dropped_row_positions.astype(str).eq("[]").all()))
        # alias rows must carry byte-identical numbers to their native parent in every numeric ledger
        alias_numeric = {}
        for table in ["temperature_effects", "control_estimates", "tests"]:
            frame = pd.read_csv(folder / f"{table}.csv", low_memory=False)
            worst = 0.0
            compared = 0
            for alias_id, parent_id in aliases[["spec_id", "alias_of"]].itertuples(index=False):
                a = frame[frame.spec_id.eq(alias_id)].reset_index(drop=True)
                b = frame[frame.spec_id.eq(parent_id)].reset_index(drop=True)
                assert len(a) == len(b) and len(a) > 0, (table, alias_id)
                numeric = [c for c in a.columns if a[c].dtype.kind == "f" and c not in ("native_fit_executed",)]
                diff = (a[numeric].to_numpy(float) - b[numeric].to_numpy(float))
                both_nan = np.isnan(a[numeric].to_numpy(float)) & np.isnan(b[numeric].to_numpy(float))
                assert np.array_equal(np.isnan(a[numeric].to_numpy(float)), np.isnan(b[numeric].to_numpy(float)))
                worst = max(worst, float(np.nanmax(np.where(both_nan, 0.0, np.abs(diff)))) if diff.size else 0.0)
                compared += int((~both_nan).sum())
            alias_numeric[table] = {"numeric_cells_compared": compared, "max_abs_difference": worst}
        entry["alias_ledger_rows_equal_parent"] = alias_numeric
        for row in log.itertuples():
            spec = native_specs[native_specs.spec_id.eq(row.spec_id)].iloc[0]
            path = ROOT / spec.design_npz
            if not path.exists():
                binding["missing_design_npz"] += 1
                continue
            with np.load(path) as z:
                binding["checked"] += 1
                binding["y_hash_mismatch"] += int(plain_hash(z["Y"].astype(float)) != row.response_sha256 or row.response_sha256 != row.native_response_sha256)
                binding["beta_hash_mismatch"] += int(plain_hash(z["beta"]) != row.native_coefficients_sha256)
                binding["residual_hash_mismatch"] += int(plain_hash(z["residual"]) != row.native_residual_sha256 or plain_hash(z["mu"]) != row.native_mu_sha256)
                binding["label_or_shape_mismatch"] += int(len(z["Y"]) != row.n_obs or int(spec.n_obs) != row.n_obs)
        entry["native_estimates"] = len(log)
    else:
        log_by_id = log.set_index("fit_id")
        entry["estimated_specs_with_fit_id"] = int(estimated.fit_id.notna().sum())
        entry["fit_ids_unique_and_complete"] = bool(sorted(estimated.fit_id.astype(int)) == sorted(log.fit_id.astype(int)))
        entry["native_dropped_rows_total"] = int(log.native_dropped_rows.sum())
        entry["native_dropped_coefficients_total"] = int(log.native_dropped_coefficients.sum())
        entry["max_native_residual_identity_error"] = float(log.original_design_residual_identity_max_abs_error.max())
        beta_key = "native_input_coefficients" if family == "origin_year" else "beta"
        resid_key = "residual" if family == "annual" else "resid"
        for spec in specs.itertuples():
            path = ROOT / spec.design_npz
            if not path.exists():
                binding["missing_design_npz"] += 1
                continue
            with np.load(path) as z:
                if spec.status != "estimated":
                    has_beta = "beta" in z.files and np.isfinite(z["beta"]).any()
                    binding["unestimated_with_finite_beta"] += int(bool(has_beta))
                    continue
                row = log_by_id.loc[int(spec.fit_id)]
                binding["checked"] += 1
                binding["label_or_shape_mismatch"] += int(row.label != spec.native_label or int(row.n_obs) != int(spec.n_obs) or int(row.n_parameters) != int(spec.n_parameters))
                binding["y_hash_mismatch"] += int(backend_hash(z["Y"]) != row.outcome_sha256)
                binding["beta_hash_mismatch"] += int(backend_hash(z[beta_key]) != row.coefficients_sha256)
                binding["residual_hash_mismatch"] += int(backend_hash(z[resid_key]) != row.residuals_sha256)
        entry["native_estimates"] = len(log)
    entry["design_archive_binding"] = binding
    entry["design_npz_files"] = len(list((folder / "designs").glob("*.npz")))
    entry["sample_csv_files"] = len(list((folder / "samples").glob("*.csv"))) if (folder / "samples").exists() else 0
    manifest = json.loads((folder / "manifest.json").read_text())
    pins = pinned(manifest, {})
    mismatched, missing, matched = [], [], 0
    for name, values in sorted(pins.items()):
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        elif len(values) != 1 or sha(path) not in values:
            mismatched.append(name)
        else:
            matched += 1
    entry["manifest_status"] = manifest.get("status")
    entry["manifest_pins"] = {"pinned_files": len(pins), "matched_current_bytes": matched, "mismatched": mismatched[:20], "missing": missing[:20]}
    result["families"][family] = entry
    total_requests += len(specs)
    total_native += len(log)

monthly = pd.read_csv(BASE / "monthly/specifications.csv", low_memory=False)
unavailable = monthly[monthly.status.ne("estimated")]
result["monthly_unavailable_requests"] = unavailable[["spec_id", "n_obs", "n_periods", "n_parameters", "design_rank", "time_design_rank", "time_residual_df"]].assign(
    status=unavailable.status.str.slice(0, 80)).to_dict("records")
annual = pd.read_csv(BASE / "annual/specifications.csv", low_memory=False)
result["annual_status_by_control_set"] = {f"{c}|{str(s)[:40]}": int(v) for (c, s), v in annual.groupby(["control_set", "status"]).size().items()}
result["annual_full_controls_listed"] = sorted(annual[annual.control_set.eq("full")].controls.unique())
origin = pd.read_csv(BASE / "origin_year/specifications.csv", low_memory=False)
result["origin_year_parent_specs_all_exist_in_monthly"] = bool(origin.source_spec_id.isin(monthly.spec_id).all()) if "source_spec_id" in origin else None
result["origin_year_parent_filter"] = monthly[monthly.spec_id.isin(origin.source_spec_id)].groupby(["panel", "model", "model_spec"]).size().reset_index(name="n").to_dict("records") if "source_spec_id" in origin else None
result["totals"] = {"logical_requests": total_requests, "native_estimates": total_native, "spec_ids_globally_unique": len(set(all_ids)) == len(all_ids),
                    "expected": {"logical_requests": 354, "native_estimates": 319}}
scope = json.loads((HERE / "parent_snapshot_scope.json").read_text())
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
result["snapshot"] = {"parent_scope": scope, "git_head": head, "head_equals_declared_base_commit": head == scope.get("base_commit")}
generation = {}
for name in ["scripts/visitor_rigour_monthly.py", "scripts/visitor_rigour_companions.py", "scripts/visitor_origin_year.py", "notes/visitor_rigour_protocol.md",
             "notes/visitor_origin_year_protocol.md", "notes/crisis_sensitivity_policy.json", "scripts/pyfixest_backend.py", "scripts/crisis_sensitivity.py",
             "scripts/literature_sensitivity.py", "scripts/literature_full_controls.py", "scripts/visitor_counts.py", "data/raw/visitor_arrivals/nsa.csv",
             "data/raw/visitor_arrivals/nsa_singstat_metadata.json", "data/processed/visitor_arrivals/all_source_counts.csv",
             "data/processed/visitor_arrivals/lunar_new_year.csv", "data/processed/extended_controls/monthly_controls.csv",
             "replication/extensions/supplementary_inputs/calendar/T2007e.txt", "replication/extensions/supplementary_inputs/calendar/sources.json",
             "data/raw/temperature_monthly.csv", "data/raw/humidity_monthly.csv", "data/raw/rainfall_monthly.csv"]:
    generation[name] = sha(ROOT / name)
result["source_generation_sha256"] = generation
ok = (total_requests == 354 and total_native == 319
      and all(f["manifest_pins"]["mismatched"] == [] for f in result["families"].values())
      and all(sum(v for k, v in f["design_archive_binding"].items() if k.endswith("mismatch") or k.startswith("missing") or k.startswith("unestimated")) == 0 for f in result["families"].values()))
absent = {family: len(json.loads((BASE / family / "manifest.json").read_text()) and [n for n in pinned(json.loads((BASE / family / "manifest.json").read_text()), {}) if not (ROOT / n).is_file()])
          for family in result["families"]}
absent_names = sorted({n for family in result["families"] for n in pinned(json.loads((BASE / family / "manifest.json").read_text()), {}) if not (ROOT / n).is_file()})
result["pinned_files_absent_from_snapshot"] = {"per_family": absent, "distinct": len(absent_names),
    "by_folder": pd.Series([str(Path(n).parent) for n in absent_names]).value_counts().to_dict() if absent_names else {}}
result["verdict"] = ("counts, native-ledger binding and every pinned file present in the snapshot reconcile (0 byte mismatches); "
                     f"{len(absent_names)} run-time pinned files are absent from the reviewed snapshot and cannot be re-hashed here") if ok else "DISCREPANCY - inspect"
write("probe01_result.json", result)
print(json.dumps({"totals": result["totals"], "per_family": {k: {"requests": v["requests"], "status": v["status_counts"], "native": v["native_estimates"],
      "binding": v["design_archive_binding"], "pins": {a: (b if not isinstance(b, list) else len(b)) for a, b in v["manifest_pins"].items()}} for k, v in result["families"].items()},
      "ppml_alias": {k: result["families"]["ppml"][k] for k in ["native_executed_specs", "alias_specs", "alias_table_rows", "alias_targets_all_native_event", "alias_ledger_rows_equal_parent"]},
      "absent_pins": result["pinned_files_absent_from_snapshot"], "git_head": result["snapshot"]["git_head"], "verdict": result["verdict"]}, indent=1, default=str))
