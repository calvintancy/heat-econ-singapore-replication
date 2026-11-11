"""Probe 10 - claim layer: origin x year reporting, labels, multiplicity, cluster rows, joint tests, diagnostics.

Read-only tabulation of the saved ledgers, set against what Probes 01-09
established numerically. Nothing is re-estimated.

A  All 24 origin x year requests: saved, estimated, one-to-one with their
   declared parents, same rows, same covariance menu, same labels; descriptive
   change table (HAC12 total origin slopes) with nominal and Holm counts in
   parent and child - to see whether anything was retained or dropped on p.
B  Vocabulary of method/status/unit labels per family.
C  Nominal 5% and within-profile Holm counts on the declared main covariance,
   split by residual time df.
D  Origin-cluster comparison rows against calendar HAC for every pooled slope
   (monthly, annual, PPML): standard-error ratios and p-value conflicts, plus
   the main_inference flag those rows carry.
E  Joint Wald rows: availability, restriction counts, fragile flags, small p.
F  ADF/KPSS coverage actually achieved, by family and role.
G  Reported mean/SD of Y against the archived estimation sample; PPML unit
   conversions for all 90 rows.

Command: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe10_claim_layer_origin_year_reporting.py
"""
from _probe_common import BASE, ROOT, write

import json

import numpy as np
import pandas as pd

STR = {"unit_id": str, "sector_id": str, "reference_unit_id": str}


def read(path):
    return pd.read_csv(path, dtype=STR, low_memory=False, float_precision="round_trip")


def counts(series):
    return {str(k): int(v) for k, v in series.fillna("<missing>").astype(str).value_counts().items()}


def quantiles(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return None if not len(values) else {"n": int(len(values)), "min": float(values.min()), "q10": float(np.quantile(values, 0.1)), "median": float(np.median(values)),
                                         "q90": float(np.quantile(values, 0.9)), "max": float(values.max())}


frames = {}
for family in ["monthly", "origin_year", "annual", "ppml"]:
    frames[family] = {name: read(BASE / family / f"{name}.csv") for name in ["specifications", "temperature_effects", "tests", "control_estimates"]}
    path = BASE / family / "channel_estimates.csv"
    frames[family]["channel_estimates"] = read(path) if path.exists() and path.stat().st_size > 5 else pd.DataFrame()
for family in ["annual"]:
    for name in ["temperature_effects", "tests", "control_estimates"]:
        frame = frames[family][name]
        if "p_value_holm_within_profile" in frame:
            frame["p_value_holm_profile"] = frame.p_value_holm_within_profile
        if "fragile" in frame and "joint_test_fragile" not in frame:
            frame["joint_test_fragile"] = frame.fragile
ppml_tests = frames["ppml"]["tests"]
if "fragile" in ppml_tests and "joint_test_fragile" not in ppml_tests:
    ppml_tests["joint_test_fragile"] = ppml_tests.fragile
MAIN = {"monthly": 12, "origin_year": 12, "annual": 1, "ppml": 12}

# ------------------------------------------------------------------ A
monthly, child = frames["monthly"], frames["origin_year"]
child_specs = child["specifications"]
parent_specs = monthly["specifications"].set_index("spec_id")
part_a = {"requests": len(child_specs), "status": counts(child_specs.status), "nuisance_spec": counts(child_specs.nuisance_spec),
          "source_ids_unique": bool(child_specs.source_spec_id.is_unique), "source_ids_all_in_monthly": bool(child_specs.source_spec_id.isin(parent_specs.index).all()),
          "parents_all_H1_own_current": bool((parent_specs.loc[child_specs.source_spec_id, "model"].eq("H1") & parent_specs.loc[child_specs.source_spec_id, "model_spec"].eq("own_current")).all()),
          "child_id_is_parent_plus_suffix": bool((child_specs.spec_id == child_specs.source_spec_id + "__origin_year").all()),
          "all_monthly_H1_own_current_parents_covered": int((parent_specs.model.eq("H1") & parent_specs.model_spec.eq("own_current")).sum()),
          "child_covariances": counts(child["temperature_effects"].covariance + " " + child["temperature_effects"].hac_lags.astype(str)),
          "child_main_inference_by_method": {f"{k[0]} {k[1]}": counts(v.main_inference) for k, v in child["temperature_effects"].groupby(["covariance", "hac_lags"])},
          "change_or_selection_columns_in_child_ledgers": sorted({c for name in ["specifications", "temperature_effects", "tests"] for c in child[name].columns if any(key in c.lower() for key in ["change", "delta_vs", "selected", "retain", "parent_coef"])}),
          "label_sets_equal_parent": {}}
parent_effects = monthly["temperature_effects"].loc[monthly["temperature_effects"].spec_id.isin(child_specs.source_spec_id)]
for column in ["calibration_status", "finite_sample_correction_formula", "interpretation", "inference_status", "multiplicity_family_size"]:
    part_a["label_sets_equal_parent"][column] = {"child": sorted(child["temperature_effects"][column].dropna().astype(str).unique().tolist())[:6],
                                                 "parent": sorted(parent_effects.loc[parent_effects.covariance.eq("Calendar HAC"), column].dropna().astype(str).unique().tolist())[:6]}
key = ["unit_id", "effect", "lag", "term", "covariance", "hac_lags"]
change_rows = []
for row in child_specs.itertuples():
    mine = child["temperature_effects"].loc[child["temperature_effects"].spec_id.eq(row.spec_id)]
    base = parent_effects.loc[parent_effects.spec_id.eq(row.source_spec_id) & parent_effects.covariance.eq("Calendar HAC")]
    merged = mine.merge(base, on=key, suffixes=("_oy", "_parent"), validate="one_to_one")
    main = merged.loc[merged.hac_lags.eq(12) & merged.effect.eq("cumulative_direct")]
    assert len(merged) == len(mine) == len(base) and len(main) == int(row.n_sectors)
    change = main.coefficient_oy - main.coefficient_parent
    change_rows.append({"spec_id": row.spec_id, "panel": row.panel, "outcome": row.outcome, "control_set": row.control_set, "sample_variant": row.sample_variant,
                        "origins": len(main), "rows_child": len(mine), "rows_parent_calendar": len(base), "n_periods": int(row.n_periods), "time_residual_df": int(row.time_residual_df),
                        "time_residual_df_parent": int(parent_specs.loc[row.source_spec_id, "time_residual_df"]), "n_parameters": int(row.n_parameters), "n_parameters_parent": int(parent_specs.loc[row.source_spec_id, "n_parameters"]),
                        "max_abs_change": float(change.abs().max()), "median_abs_change": float(change.abs().median()), "mean_change_across_origins": float(change.mean()),
                        "sign_changes": int((np.sign(main.coefficient_oy) != np.sign(main.coefficient_parent)).sum()), "median_se_ratio_oy_over_parent": float((main.std_error_oy / main.std_error_parent).median()),
                        "nominal_p05_parent": int((main.p_value_parent < 0.05).sum()), "nominal_p05_oy": int((main.p_value_oy < 0.05).sum()), "nominal_p05_both": int(((main.p_value_parent < 0.05) & (main.p_value_oy < 0.05)).sum()),
                        "holm_p05_parent": int((main.p_value_holm_profile_parent < 0.05).sum()), "holm_p05_oy": int((main.p_value_holm_profile_oy < 0.05).sum()),
                        "finite_sample_correction_oy": float(main.finite_sample_correction_oy.iloc[0]), "finite_sample_correction_parent": float(main.finite_sample_correction_parent.iloc[0])})
changes = pd.DataFrame(change_rows)
changes.to_csv(ROOT / "audit_probes" / "probe10_origin_year_change_table.csv", index=False)
part_a.update(change_table_rows=len(changes), all_rows_matched_one_to_one=True, origin_slopes=int(changes.origins.sum()),
              nominal_p05_parent=int(changes.nominal_p05_parent.sum()), nominal_p05_oy=int(changes.nominal_p05_oy.sum()), nominal_p05_both=int(changes.nominal_p05_both.sum()),
              holm_p05_parent=int(changes.holm_p05_parent.sum()), holm_p05_oy=int(changes.holm_p05_oy.sum()), sign_changes=int(changes.sign_changes.sum()),
              mean_change_across_origins_max_abs=float(changes.mean_change_across_origins.abs().max()), max_abs_change_range=[float(changes.max_abs_change.min()), float(changes.max_abs_change.max())],
              median_se_ratio_range=[float(changes.median_se_ratio_oy_over_parent.min()), float(changes.median_se_ratio_oy_over_parent.max())],
              time_df_same_as_parent=bool((changes.time_residual_df == changes.time_residual_df_parent).all()), correction_same_as_parent=bool(np.allclose(changes.finite_sample_correction_oy, changes.finite_sample_correction_parent, rtol=0, atol=1e-15)),
              min_time_residual_df=int(changes.time_residual_df.min()), specs_with_time_df_below_10=int((changes.time_residual_df < 10).sum()))

# ------------------------------------------------------------------ B
part_b = {}
for family, book in frames.items():
    effects, tests, specs = book["temperature_effects"], book["tests"], book["specifications"]
    entry = {"spec_status": counts(specs.status), "effect_rows": len(effects), "test_rows": len(tests),
             "methods": {f"{k[0]} {k[1]} main={k[2]}": int(v) for k, v in effects.groupby(["covariance", "hac_lags", "main_inference"]).size().items()}}
    for column in ["calibration_status", "finite_sample_correction_formula", "inference_status", "interpretation", "finite_sample_rule", "units", "coefficient_units", "primary_inference", "inference_reference_scope", "native_point_guard", "zero_retention_status"]:
        for name, frame in [("effects", effects), ("tests", tests), ("specifications", specs)]:
            if column in frame:
                entry[f"{name}.{column}"] = counts(frame[column])
    if "multiplicity_family" in effects:
        entry["multiplicity_family_examples"] = sorted(effects.multiplicity_family.dropna().astype(str).unique().tolist())[:3]
        entry["multiplicity_family_sizes"] = counts(effects.multiplicity_family_size)
    part_b[family] = entry

# ------------------------------------------------------------------ C
part_c = {}
for family, book in frames.items():
    effects = book["temperature_effects"]
    main = effects.loc[effects.covariance.eq("Calendar HAC") & effects.hac_lags.eq(MAIN[family])].copy()
    if "model" not in main:
        main["model"] = "common"
    if "effect" not in main:
        main["effect"] = "current"
    main["thin"] = np.where(main.time_residual_df < 10, "time_df<10", "time_df>=10")
    table = []
    for group, rows in main.groupby(["model", "effect", "thin"]):
        available = rows.loc[rows.p_value.notna()]
        holm = rows.p_value_holm_profile.dropna() if "p_value_holm_profile" in rows else pd.Series(dtype=float)
        table.append({"model": group[0], "effect": group[1], "time_df": group[2], "rows": len(rows), "with_p": len(available), "nominal_p05": int((available.p_value < 0.05).sum()),
                      "share_nominal_p05": float((available.p_value < 0.05).mean()) if len(available) else None, "with_holm": len(holm), "holm_p05": int((holm < 0.05).sum()), "specs": int(rows.spec_id.nunique())})
    part_c[family] = table

# ------------------------------------------------------------------ D
part_d = {}
for family, book in frames.items():
    pieces = [book["temperature_effects"]] + ([book["channel_estimates"]] if len(book["channel_estimates"]) else [])
    out = []
    for effects in pieces:
        cluster = effects.loc[effects.covariance.eq("Origin cluster")]
        if not len(cluster):
            continue
        calendar = effects.loc[effects.covariance.eq("Calendar HAC") & effects.hac_lags.eq(MAIN[family])]
        on = [c for c in ["spec_id", "unit_id", "effect", "lag", "term", "channel"] if c in effects]
        merged = cluster.merge(calendar, on=on, suffixes=("_cluster", "_calendar"), validate="one_to_one")
        merged["ratio"] = merged.std_error_cluster / merged.std_error_calendar
        out.append(merged)
    if not out:
        part_d[family] = {"origin_cluster_rows": 0}
        continue
    merged = pd.concat(out, ignore_index=True)
    conflict = merged.loc[(merged.p_value_cluster < 0.05) & (merged.p_value_calendar >= 0.05)]
    reverse = merged.loc[(merged.p_value_cluster >= 0.05) & (merged.p_value_calendar < 0.05)]
    part_d[family] = {"origin_cluster_rows": len(merged), "specs": int(merged.spec_id.nunique()), "main_inference_flag_cluster_rows": counts(merged.main_inference_cluster),
                      "models_with_cluster_rows": counts(merged["model_cluster"]) if "model_cluster" in merged else counts(merged["model"]) if "model" in merged else None,
                      "se_ratio_cluster_over_calendar": quantiles(merged.ratio), "share_ratio_below_half": float((merged.ratio < 0.5).mean()), "share_ratio_below_one": float((merged.ratio < 1).mean()),
                      "cluster_p05": int((merged.p_value_cluster < 0.05).sum()), "calendar_p05": int((merged.p_value_calendar < 0.05).sum()),
                      "cluster_p05_but_calendar_not": len(conflict), "calendar_p05_but_cluster_not": len(reverse), "cluster_p001_but_calendar_not_p05": int((conflict.p_value_cluster < 0.001).sum()),
                      "strongest_conflicts": [{"spec_id": r.spec_id, "term": r.term, "coefficient": round(float(r.coefficient_cluster), 4), "se_cluster": round(float(r.std_error_cluster), 4), "p_cluster": float(r.p_value_cluster),
                                               "se_calendar": round(float(r.std_error_calendar), 4), "p_calendar": round(float(r.p_value_calendar), 4), "df_cluster": int(r.inference_df_cluster)} for r in conflict.sort_values("p_value_cluster").head(8).itertuples()],
                      "calibration_status_cluster_rows": counts(merged["calibration_status_cluster"]) if "calibration_status_cluster" in merged else None}

# ------------------------------------------------------------------ E
part_e = {}
for family, book in frames.items():
    tests = book["tests"].copy()
    if not len(tests):
        continue
    multi = tests.loc[tests.restrictions > 1]
    available = multi.loc[multi.p_value.notna()]
    fragile = available.joint_test_fragile.astype(str).str.lower().eq("true")
    small = available.loc[available.p_value < 0.05]
    robust_small = available.loc[(available.p_value < 0.05) & ~fragile]
    part_e[family] = {"rows": len(tests), "single_restriction_rows": int((tests.restrictions <= 1).sum()), "multi_restriction_rows": len(multi), "multi_with_p": len(available), "multi_without_p": int(multi.p_value.isna().sum()),
                      "status": counts(tests.status) if "status" in tests else None, "calibration_status": counts(tests.calibration_status) if "calibration_status" in tests else None,
                      "restrictions": quantiles(available.restrictions), "flagged_fragile": int(fragile.sum()), "p05": len(small), "p05_and_fragile": int(((available.p_value < 0.05) & fragile).sum()),
                      "p05_not_fragile": len(robust_small), "p05_not_fragile_restrictions_ge_20": int((robust_small.restrictions >= 20).sum()), "p001_not_fragile": int((robust_small.p_value < 0.001).sum()),
                      "share_p05_among_available": float((available.p_value < 0.05).mean()) if len(available) else None,
                      "by_method": {f"{k[0]} {k[1]}": {"with_p": len(v), "p05": int((v.p_value < 0.05).sum()), "p05_not_fragile": int(((v.p_value < 0.05) & ~v.joint_test_fragile.astype(str).str.lower().eq("true")).sum())}
                                    for k, v in available.groupby(["covariance", "hac_lags"])},
                      "smallest_p_not_fragile": [{"spec_id": r.spec_id, "test": r.test, "covariance": f"{r.covariance} {r.hac_lags}", "restrictions": int(r.restrictions), "bound": int(r.structural_rank_bound), "f": round(float(r.f_statistic), 3), "p": float(r.p_value)}
                                                 for r in robust_small.sort_values("p_value").head(6).itertuples()]}

# ------------------------------------------------------------------ F
diagnostics = pd.read_csv(BASE / "diagnostics/diagnostics.csv", low_memory=False)
coverage = pd.read_csv(BASE / "diagnostics/model_variable_coverage.csv", dtype={"unit_id": str}, low_memory=False, usecols=["branch", "spec_id", "variable", "role", "diagnostic_id", "sample_variant", "source_status"])
coverage = coverage.merge(diagnostics[["diagnostic_id", "state", "n_periods", "calendar_gaps", "frequency"]], on="diagnostic_id", how="left", validate="many_to_one")
coverage["completed"] = ~coverage.state.str.startswith("unavailable")
completed = diagnostics.loc[~diagnostics.state.str.startswith("unavailable")]
part_f = {"distinct_diagnostics": len(diagnostics), "completed": len(completed), "states": counts(diagnostics.state), "links": len(coverage),
          "links_by_branch_role": {f"{k[0]}/{k[1]}": {"links": len(v), "completed": int(v.completed.sum()), "specs": int(v.spec_id.nunique()), "specs_with_any_completed": int(v.loc[v.completed].spec_id.nunique())} for k, v in coverage.groupby(["branch", "role"])},
          "completed_by_variables_and_state": {f"{k[0]} | {k[1]} | {k[2]}": int(v) for k, v in completed.groupby(["variables", "branches", "state"]).size().items()},
          "completed_growth_outcome_diagnostics": int(completed.variables.str.contains("y_mom_nsa|y_yoy_nsa|annual", regex=True).sum()),
          "gapped_all_eligible_growth_example": diagnostics.loc[diagnostics.variables.str.contains("y_mom_nsa") & diagnostics.n_periods.eq(207), ["n_periods", "calendar_gaps", "state", "label"]].head(1).to_dict("records")}

# ------------------------------------------------------------------ G
mean_rows = []
for family, book in frames.items():
    for row in book["specifications"].itertuples():
        native = getattr(row, "native_spec_id", row.spec_id) if family == "ppml" else row.spec_id
        path = BASE / family / "designs" / f"{native}.npz"
        if not path.exists() or not np.isfinite(float(row.mean_y)):
            mean_rows.append({"family": family, "spec_id": row.spec_id, "archive": path.exists(), "mean_diff": None, "sd_ddof1_diff": None, "sd_ddof0_diff": None})
            continue
        y = np.load(path)["Y"].astype(float)
        mean_rows.append({"family": family, "spec_id": row.spec_id, "archive": True, "mean_diff": abs(y.mean() - float(row.mean_y)), "sd_ddof1_diff": abs(y.std(ddof=1) - float(row.sd_y)) if len(y) > 1 else None,
                          "sd_ddof0_diff": abs(y.std(ddof=0) - float(row.sd_y))})
means = pd.DataFrame(mean_rows)
ppml_effects = frames["ppml"]["temperature_effects"]
part_g = {"mean_sd_by_family": {family: {"specs": len(rows), "with_archive_and_mean": int(rows.mean_diff.notna().sum()), "mean_max_abs_diff": float(rows.mean_diff.max()), "sd_ddof1_max_abs_diff": float(rows.sd_ddof1_diff.max()), "sd_ddof0_max_abs_diff": float(rows.sd_ddof0_diff.max())}
                                for family, rows in means.groupby("family")},
          "ppml_rows": len(ppml_effects), "ppml_coefficient_is_100_raw_max_abs": float((ppml_effects.coefficient - 100 * ppml_effects.raw_native_coefficient).abs().max()),
          "ppml_effect_0p1c_is_10_raw_max_abs": float((ppml_effects.effect_0p1c - 10 * ppml_effects.raw_native_coefficient).abs().max()),
          "ppml_exact_percent_is_100_expm1_max_abs": float((ppml_effects.exact_percent_0p1c - 100 * np.expm1(0.1 * ppml_effects.raw_native_coefficient)).abs().max()),
          "ppml_alias_rows_equal_native_rows_max_abs": float(max((ppml_effects.loc[ppml_effects.spec_id.eq(a), ["coefficient", "std_error", "p_value"]].to_numpy() - ppml_effects.loc[ppml_effects.spec_id.eq(n), ["coefficient", "std_error", "p_value"]].to_numpy()).__abs__().max()
                                                             for a, n in frames["ppml"]["specifications"].loc[frames["ppml"]["specifications"].alias_of.notna(), ["spec_id", "alias_of"]].itertuples(index=False))),
          "ppml_main_p05_rows": int((ppml_effects.loc[ppml_effects.covariance.eq("Calendar HAC")].p_value < 0.05).sum())}
manifest_status = {}
for name in ["monthly/manifest.json", "origin_year/manifest.json", "annual/manifest.json", "ppml/manifest.json", "companions_manifest.json", "monthly/verification.json", "origin_year/verification.json", "annual/verification.json", "ppml/verification.json", "diagnostics/verification.json", "diagnostics/summary.json"]:
    manifest_status[name] = json.loads((BASE / name).read_text()).get("status")

result = {"probe": "10 claim layer and origin x year reporting", "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe10_claim_layer_origin_year_reporting.py",
          "A_origin_year_reporting": part_a, "B_label_vocabulary": part_b, "C_main_inference_counts": part_c, "D_origin_cluster_vs_calendar": part_d, "E_joint_tests": part_e, "F_diagnostic_coverage": part_f,
          "G_means_units": part_g, "status_fields": manifest_status, "origin_year_change_table": change_rows}
write("probe10_result.json", result)
print("A", json.dumps({k: v for k, v in part_a.items() if k != "label_sets_equal_parent"}, default=str))
print("A labels", json.dumps(part_a["label_sets_equal_parent"], default=str)[:3000])
print(changes[["panel", "outcome", "control_set", "sample_variant", "origins", "n_periods", "time_residual_df", "max_abs_change", "sign_changes", "median_se_ratio_oy_over_parent", "nominal_p05_parent", "nominal_p05_oy", "nominal_p05_both", "holm_p05_parent", "holm_p05_oy"]].round(3).to_string())
for family in part_b:
    print("B", family, json.dumps(part_b[family], default=str)[:6000])
for family in part_c:
    print("C", family)
    for row in part_c[family]:
        print("    ", row)
for family in part_d:
    print("D", family, json.dumps(part_d[family], default=str))
for family in part_e:
    print("E", family, json.dumps(part_e[family], default=str))
print("F", json.dumps(part_f, default=str))
print("G", json.dumps(part_g, default=str))
print("status", json.dumps(manifest_status))
