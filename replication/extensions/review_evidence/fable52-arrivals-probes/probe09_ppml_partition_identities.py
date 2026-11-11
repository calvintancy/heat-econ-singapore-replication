"""Probe 09 - replay all eight complete-partition PPML aggregation identities.

Theorem replayed (derivation in the report): with exhaustive component counts,
identical dates for every component, regressors that vary by date only,
component x calendar-month effects and year effects shared by all components,
    mu_it = exp(a_{i,s(t)} + g_{y(t)} + x_t'b)   =>   sum_i mu_it = exp(A_{s(t)} + g_{y(t)} + x_t'b)
and the Poisson likelihood is the product of the likelihood of the date totals
and a multinomial for the within-month-of-year shares that is free of (b, g).
Consequences checked here on the saved native quantities, for complete10 and
complete52 x long/full x all_eligible/event_dates_excluded against the
official-total fit:
  1 component counts add to the official total on every retained date;
  2 all slopes equal; 3 fitted means add to the total fitted mean;
  4 mu_it / mu_t is the observed within-cell share Y_{i,s} / Y_{.,s};
  5 weighted-FE residualised regressors equal the aggregate ones row by row;
  6 information and bread equal; 7 date-summed scores and influence equal;
  8 raw and corrected calendar HAC12/24 equal for the full slope matrix;
  9 T, K_time, T/(T-K_time) and df equal because every component has exactly
    the aggregate's dates and regressors (identical local time design).
Row-level HC0 and the origin-cluster matrix are NOT part of the identity and
are reported to show that. The non-exhaustive region and named-market cuts are
replayed through the same code to show the identity fails there.

Command: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe09_ppml_partition_identities.py
"""
from _probe_common import BASE, ROOT, bartlett_calendar, maxabs, maxrel, scaled_rank, write

import csv
import json

import numpy as np
import pandas as pd

PPML = BASE / "ppml"
MONTHS = {name: number for number, name in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
STR = {"unit_id": str, "sector_id": str}
specs = pd.read_csv(PPML / "specifications.csv", dtype=STR, low_memory=False)
effects = pd.read_csv(PPML / "temperature_effects.csv", dtype=STR, float_precision="round_trip", low_memory=False)
controls = pd.read_csv(PPML / "control_estimates.csv", dtype=STR, float_precision="round_trip", low_memory=False)
saved_checks = pd.read_csv(PPML / "aggregation_checks.csv")


def ordinal(month):
    return (int(month[:4]) - 1970) * 12 + int(month[5:7]) - 1


def load(spec_id):
    spec = specs.loc[specs.spec_id.eq(spec_id)].iloc[0]
    sample = pd.read_csv(ROOT / spec["sample_csv"], dtype=STR, float_precision="round_trip")
    archive = np.load(PPML / "designs" / f"{spec_id}.npz")
    time = np.array([ordinal(m) for m in sample.month])
    assert np.array_equal(sample.arrivals.to_numpy(), archive["Y"]) and np.array_equal(time, archive["time"])
    return spec, sample, archive, time


def relative(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) / np.max(np.abs(np.asarray(b, float))))


def local_design(months, x):
    year = np.array([int(m[:4]) for m in months])
    season = np.array([int(m[5:7]) for m in months])
    return np.column_stack([np.column_stack([(year == v).astype(float) for v in sorted(set(year.tolist()))]),
                            np.column_stack([(season == s).astype(float) for s in range(1, 13)]), x])


# ---- raw official counts and the hierarchy of the partition
meta = json.loads((ROOT / "data/raw/visitor_arrivals/nsa_singstat_metadata.json").read_text())["Data"]["records"]
by_name = {row["rowText"].strip(): str(row["seriesNo"]) for row in meta["row"]}
with open(ROOT / "data/raw/visitor_arrivals/nsa.csv", newline="") as handle:
    raw_rows = list(csv.reader(handle))
header = [f"{text[:4]}-{MONTHS[text[4:]]:02d}" for text in raw_rows[0][1:]]
raw = {by_name[row[0].strip()]: np.array([int(v) for v in row[1:]]) for row in raw_rows[1:]}
order = np.argsort(header)
months_sorted = [header[k] for k in order]
raw = {key: value[order] for key, value in raw.items()}
regions = sorted(u for u in raw if u.count(".") == 1 and not u.endswith(".999"))
named = sorted(u for u in raw if u.count(".") == 2 and not u.endswith(".999"))
residual = sorted(u for u in raw if u.endswith(".999"))
complete10 = regions + ["1.999"]
complete52 = named + residual
hierarchy = {"series": len(raw), "regions": len(regions), "named_markets": len(named), "residual_categories": len(residual), "months": len(months_sorted),
             "complete10_minus_total_max_abs": int(np.max(np.abs(sum(raw[u] for u in complete10) - raw["1"]))),
             "complete52_minus_total_max_abs": int(np.max(np.abs(sum(raw[u] for u in complete52) - raw["1"]))),
             "region_children_plus_residual_minus_region_max_abs": int(max(np.max(np.abs(sum(raw[u] for u in named + residual if u.startswith(r + ".")) - raw[r])) for r in regions)),
             "selected_regions_share_of_total": float(sum(raw[u] for u in regions).sum() / raw["1"].sum()),
             "named_markets_share_of_total": float(sum(raw[u] for u in named).sum() / raw["1"].sum()),
             "named_markets_share_min_max_by_month": [float(np.min(sum(raw[u] for u in named) / raw["1"])), float(np.max(sum(raw[u] for u in named) / raw["1"]))],
             "zero_cells": {"complete10": int(sum((raw[u] == 0).sum() for u in complete10)), "complete52": int(sum((raw[u] == 0).sum() for u in complete52)),
                            "regions": int(sum((raw[u] == 0).sum() for u in regions)), "named": int(sum((raw[u] == 0).sum() for u in named)), "total": int((raw["1"] == 0).sum())}}
reconciliation = pd.read_csv(PPML / "count_partition_reconciliation.csv")
hierarchy["saved_reconciliation_rows"] = len(reconciliation)
hierarchy["saved_reconciliation_identities"] = int(reconciliation.identity.nunique())
hierarchy["saved_reconciliation_max_abs_difference"] = float(reconciliation.difference.abs().max())

rows = []
for panel in ["complete10", "complete52", "region", "country"]:
    for control in ["long", "full"]:
        for variant in ["all_eligible", "event_dates_excluded"]:
            aggregate_id, component_id = f"ppml__aggregate__{control}__{variant}", f"ppml__{panel}__{control}__{variant}"
            spec_a, sample_a, a, time_a = load(aggregate_id)
            spec_p, sample_p, p, time_p = load(component_id)
            terms = a["terms"].tolist()
            assert terms == p["terms"].tolist() and np.all(np.diff(time_a) > 0)
            position = {t: k for k, t in enumerate(time_a.tolist())}
            index = np.array([position[t] for t in time_p.tolist()])
            t, n = len(time_a), len(time_p)
            unit = sample_p.unit_id.to_numpy()
            units = sorted(set(unit.tolist()))
            season = np.array([int(m[5:7]) for m in sample_p.month])
            season_a = np.array([int(m[5:7]) for m in sample_a.month])
            y_p, y_a, mu_p, mu_a = p["Y"].astype(float), a["Y"].astype(float), p["mu"], a["mu"]
            cell_labels = sorted({(u, s) for u, s in zip(unit, season)})
            cell_number = {label: k for k, label in enumerate(cell_labels)}
            cell = np.array([cell_number[(u, s)] for u, s in zip(unit, season)])
            cell_total = np.bincount(cell, weights=y_p)
            season_total = np.array([y_a[season_a == s].sum() for _, s in cell_labels])
            observed_share = (cell_total / season_total)[cell]
            fitted_share = mu_p / mu_a[index]
            spread = np.array([np.ptp(fitted_share[cell == k]) / fitted_share[cell == k].mean() for k in range(len(cell_labels))])
            date_sum = lambda values: np.column_stack([np.bincount(index, weights=values[:, j], minlength=t) for j in range(values.shape[1])]) if values.ndim == 2 else np.bincount(index, weights=values, minlength=t)
            local_a = local_design(sample_a.month.tolist(), a["X"])
            same_local = all(np.array_equal(local_design(sample_p.month[unit == u].tolist(), p["X"][unit == u]), local_a) for u in units)
            k_time = scaled_rank(local_a)
            record = {"panel": panel, "control_set": control, "sample_variant": variant, "exhaustive_partition": panel.startswith("complete"), "n_obs": n, "n_units": len(units), "n_months": t,
                      "balanced_same_dates_every_unit": bool(all(np.array_equal(np.sort(time_p[unit == u]), time_a) for u in units)),
                      "regressors_equal_aggregate_row_by_row_max_abs": maxabs(p["X"], a["X"][index]),
                      "counts_sum_minus_aggregate_max_abs": float(np.max(np.abs(date_sum(y_p) - y_a))), "zero_cells_retained": int((y_p == 0).sum()), "min_origin_season_cell_total": float(cell_total.min()),
                      "share_of_aggregate_count": float(y_p.sum() / y_a.sum()),
                      "beta_max_abs_diff": maxabs(p["beta"], a["beta"]), "beta_max_rel_diff": float(np.max(np.abs(p["beta"] - a["beta"]) / np.abs(a["beta"]))),
                      "temperature_100beta_component": float(100 * p["beta"][0]), "temperature_100beta_aggregate": float(100 * a["beta"][0]),
                      "fitted_sum_max_abs_diff": float(np.max(np.abs(date_sum(mu_p) - mu_a))), "fitted_sum_max_rel_diff": float(np.max(np.abs(date_sum(mu_p) - mu_a) / mu_a)),
                      "fitted_share_spread_within_cell_max": float(spread.max()), "fitted_share_vs_observed_cell_share_max_rel": float(np.max(np.abs(fitted_share - observed_share) / observed_share)),
                      "mu_from_aggregate_fit_times_observed_share_max_rel": float(np.max(np.abs(observed_share * mu_a[index] - mu_p) / mu_p)),
                      "xr_equals_aggregate_xr_max_abs": maxabs(p["xr"], a["xr"][index]), "information_max_rel": relative(p["information"], a["information"]), "bread_max_rel": relative(p["bread"], a["bread"]),
                      "date_summed_scores_max_rel": relative(date_sum(p["scores"]), a["scores"]), "date_summed_influence_max_rel": relative(date_sum(p["influence"]), a["influence"]),
                      "row_level_hc0_temperature_ratio_component_over_aggregate": float(p["hc0"][0, 0] / a["hc0"][0, 0]),
                      "local_time_design_identical_for_every_unit_and_aggregate": bool(same_local), "k_time_mine": k_time, "k_time_component": int(spec_p["time_design_rank"]), "k_time_aggregate": int(spec_a["time_design_rank"]),
                      "local_ranks_saved_component_unique": sorted(set(p["local_ranks"].tolist())), "corrections_component": p["corrections"].tolist(), "corrections_aggregate": a["corrections"].tolist(),
                      "inference_df_component": p["inference_df"].tolist(), "inference_df_aggregate": a["inference_df"].tolist(), "covariance_labels_component": p["covariance_labels"].tolist(),
                      "covariance_labels_aggregate": a["covariance_labels"].tolist()}
            for k, bandwidth in enumerate([12, 24]):
                mine = bartlett_calendar(p["influence"], time_p, bandwidth)
                record[f"raw_hac{bandwidth}_component_vs_aggregate_max_rel"] = relative(p[f"raw_calendar_covariance_{k}"], a[f"raw_calendar_covariance_{k}"])
                record[f"raw_hac{bandwidth}_independent_rebuild_from_component_influence_vs_aggregate_max_rel"] = relative(mine, a[f"raw_calendar_covariance_{k}"])
                record[f"corrected_hac{bandwidth}_component_vs_aggregate_max_rel"] = relative(p[f"covariance_{k}"], a[f"covariance_{k}"])
                record[f"corrected_hac{bandwidth}_rebuilt_times_T_over_T_minus_K_vs_aggregate_max_rel"] = relative(mine * t / (t - k_time), a[f"covariance_{k}"])
            ledger = 0.0
            for book, wanted in [(effects, ["temp_c"]), (controls, [term for term in terms if term != "temp_c"])]:
                for bandwidth in [12, 24]:
                    for term in wanted:
                        one = book.loc[book.spec_id.eq(component_id) & book.covariance.eq("Calendar HAC") & book.hac_lags.eq(bandwidth) & book.term.eq(term)]
                        two = book.loc[book.spec_id.eq(aggregate_id) & book.covariance.eq("Calendar HAC") & book.hac_lags.eq(bandwidth) & book.term.eq(term)]
                        assert len(one) == len(two) == 1
                        for column in ["coefficient", "std_error", "p_value", "ci_lower", "ci_upper"]:
                            ledger = max(ledger, abs(float(one.iloc[0][column]) - float(two.iloc[0][column])))
            record["ledger_calendar_rows_max_abs_diff_all_terms"] = ledger
            if len(units) > 1 and "Origin cluster" in p["covariance_labels"].tolist():
                k = p["covariance_labels"].tolist().index("Origin cluster")
                origin = np.vstack([p["influence"][unit == u].sum(axis=0) for u in units])
                record["origin_cluster_scores_sum_over_origins_max_abs_over_scale"] = float(np.max(np.abs(origin.sum(axis=0))) / np.max(np.abs(origin)))
                record["origin_cluster_rank"] = int(np.linalg.matrix_rank(p[f"covariance_{k}"]))
                record["temperature_se_origin_cluster_over_calendar_hac12"] = float(np.sqrt(p[f"covariance_{k}"][0, 0] / p["covariance_0"][0, 0]))
            saved = saved_checks.loc[saved_checks.component_spec_id.eq(component_id)]
            record["saved_aggregation_check_rows"] = len(saved)
            if len(saved):
                one = saved.iloc[0]
                record["saved_status"] = one["status"]
                record["saved_diffs_beta_mu_covariance"] = [float(one["max_abs_shared_coefficient_difference"]), float(one["max_abs_aggregated_fitted_count_difference"]), float(one["max_abs_calendar_covariance_difference"])]
                record["saved_same_time_correction"] = bool(one["same_time_correction"])
                record["saved_interpretation"] = one["interpretation"]
            rows.append(record)

exact = [r for r in rows if r["exhaustive_partition"]]
other = [r for r in rows if not r["exhaustive_partition"]]
identity_keys = ["counts_sum_minus_aggregate_max_abs", "regressors_equal_aggregate_row_by_row_max_abs", "beta_max_abs_diff", "beta_max_rel_diff", "fitted_sum_max_rel_diff", "fitted_share_spread_within_cell_max",
                 "fitted_share_vs_observed_cell_share_max_rel", "mu_from_aggregate_fit_times_observed_share_max_rel", "xr_equals_aggregate_xr_max_abs", "information_max_rel", "bread_max_rel",
                 "date_summed_scores_max_rel", "date_summed_influence_max_rel", "raw_hac12_component_vs_aggregate_max_rel", "raw_hac24_component_vs_aggregate_max_rel",
                 "raw_hac12_independent_rebuild_from_component_influence_vs_aggregate_max_rel", "raw_hac24_independent_rebuild_from_component_influence_vs_aggregate_max_rel",
                 "corrected_hac12_component_vs_aggregate_max_rel", "corrected_hac24_component_vs_aggregate_max_rel", "corrected_hac12_rebuilt_times_T_over_T_minus_K_vs_aggregate_max_rel",
                 "corrected_hac24_rebuilt_times_T_over_T_minus_K_vs_aggregate_max_rel", "ledger_calendar_rows_max_abs_diff_all_terms"]
summary = {"exhaustive_pairs": len(exact), "worst_over_8_exhaustive_pairs": {key: max(r[key] for r in exact) for key in identity_keys},
           "worst_over_8_nonexhaustive_pairs": {key: max(r[key] for r in other) for key in identity_keys},
           "exhaustive_all_balanced": all(r["balanced_same_dates_every_unit"] for r in exact), "exhaustive_all_local_designs_identical": all(r["local_time_design_identical_for_every_unit_and_aggregate"] for r in exact),
           "exhaustive_k_time_equal": all(r["k_time_mine"] == r["k_time_component"] == r["k_time_aggregate"] for r in exact),
           "exhaustive_corrections_equal": all(np.allclose(r["corrections_component"][:2], r["corrections_aggregate"][:2], rtol=0, atol=1e-15) for r in exact),
           "exhaustive_df_equal": all(r["inference_df_component"][:2] == r["inference_df_aggregate"][:2] for r in exact),
           "exhaustive_min_cell_total": min(r["min_origin_season_cell_total"] for r in exact), "exhaustive_zero_cells_retained": {f"{r['panel']}/{r['control_set']}/{r['sample_variant']}": r["zero_cells_retained"] for r in exact},
           "exhaustive_saved_rows_all_verified": all(r["saved_aggregation_check_rows"] == 1 and r["saved_status"] == "verified within native numerical tolerance" and r["saved_same_time_correction"] for r in exact),
           "nonexhaustive_saved_rows": sum(r["saved_aggregation_check_rows"] for r in other),
           "row_level_hc0_ratio_range_exhaustive": [min(r["row_level_hc0_temperature_ratio_component_over_aggregate"] for r in exact), max(r["row_level_hc0_temperature_ratio_component_over_aggregate"] for r in exact)],
           "origin_cluster_over_calendar_se_ratio": {f"{r['panel']}/{r['control_set']}/{r['sample_variant']}": round(r["temperature_se_origin_cluster_over_calendar_hac12"], 4) for r in rows},
           "origin_cluster_score_sum_exhaustive_max": max(r["origin_cluster_scores_sum_over_origins_max_abs_over_scale"] for r in exact),
           "origin_cluster_score_sum_nonexhaustive_max": max(r["origin_cluster_scores_sum_over_origins_max_abs_over_scale"] for r in other),
           "nonexhaustive_temperature_100beta": {f"{r['panel']}/{r['control_set']}/{r['sample_variant']}": [round(r["temperature_100beta_component"], 6), round(r["temperature_100beta_aggregate"], 6), round(r["share_of_aggregate_count"], 4)] for r in other}}
write("probe09_result.json", {"probe": "09 complete-partition PPML aggregation identities (8 exhaustive pairs) and 8 non-exhaustive comparators",
                              "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe09_ppml_partition_identities.py",
                              "production_reference": "scripts/visitor_rigour_companions.py:446-471 aggregation_checks; :310-443 ppml_models", "hierarchy": hierarchy, "summary": summary, "pairs": rows})
print("hierarchy", json.dumps(hierarchy))
print("summary", json.dumps(summary, indent=1))
for r in rows:
    print(r["panel"], r["control_set"], r["sample_variant"], "N", r["n_obs"], "G", r["n_units"], "T", r["n_months"], "zeros", r["zero_cells_retained"], "beta diff", f"{r['beta_max_abs_diff']:.3g}",
          "mu sum rel", f"{r['fitted_sum_max_rel_diff']:.3g}", "xr", f"{r['xr_equals_aggregate_xr_max_abs']:.3g}", "hac12", f"{r['corrected_hac12_component_vs_aggregate_max_rel']:.3g}",
          "K_time", r["k_time_component"], r["k_time_aggregate"], "hc0 ratio", f"{r['row_level_hc0_temperature_ratio_component_over_aggregate']:.4g}")
