"""Probe 08 - zero-retaining PPML: receipts, row map, literal controls, independent IRLS and sandwich.

No native model is fitted. Part A audits the 20 saved Fepois receipts and the
10 aliases. Part B takes the zero-retaining request
ppml__country__long__all_eligible (42 named markets x 225 months, 50 zero
cells) and
  * maps every retained row to the raw official count (integer equality, so no
    +1, no deleted zero, no deleted market, no growth-history column);
  * checks the regressors are the literal current values;
  * refits the Poisson pseudo-likelihood by an independent IRLS in which the
    origin x calendar-month effects are absorbed exactly by weighted cell
    means and the year effects enter as dummies; a dense 527-column score and
    information matrix cross-check the absorbed algebra;
  * rebuilds the weighted-FE-residualised regressors, bread, scores,
    influence, HC0, calendar HAC12/24 with zero bins, T/(T-K_time), the
    pooled-origin CR1 matrix and every saved ledger number;
  * reports descriptive facts that bear on interpretation: overdispersion, the
    share of the temperature score variance carried by crisis months, and how
    much the slope moves under the two things the pipeline must NOT do
    (deleting zero cells, adding one), each as an explicit comparator fit.

Command: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe08_ppml_zero_retention_irls.py
"""
from _probe_common import BASE, ROOT, bartlett_calendar, maxabs, maxrel, scaled_rank, write

import csv
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats

PPML = BASE / "ppml"
MONTHS = {name: number for number, name in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
STR = {"unit_id": str, "sector_id": str}
SPEC = "ppml__country__long__all_eligible"


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def ordinal(year, month):
    return (year - 1970) * 12 + month - 1


# ------------------------------------------------------------- A. receipts
specs = pd.read_csv(PPML / "specifications.csv", dtype=STR, low_memory=False)
log = pd.read_csv(PPML / "pyfixest/fit_log.csv")
manifest = json.loads((PPML / "pyfixest/estimator_manifest.json").read_text())
aliases = pd.read_csv(PPML / "aliases.csv")
dropped = pd.read_csv(PPML / "dropped_observations.csv")
receipt_rows = []
for row in log.to_dict("records"):
    archive = np.load(PPML / "designs" / f"{row['spec_id']}.npz")
    spec = specs.loc[specs.spec_id.eq(row["spec_id"])].iloc[0]
    sample = pd.read_csv(ROOT / spec["sample_csv"], dtype=STR)
    given = pd.read_csv(ROOT / spec["input_sample_csv"], dtype=STR)
    receipt_rows.append({
        "spec_id": row["spec_id"], "api": row["api"], "native_class": row["native_class"], "native_module": row["native_module"], "version": row["estimator_version"],
        "converged": bool(row["native_converged"]), "postprocessing": row["postprocessing_status"], "dropped_positions": row["dropped_row_positions"],
        "formula_has_no_offset_or_transform": ("log(" not in row["formula"]) and ("+ 1" not in row["formula"]) and row["formula"].startswith("arrivals ~ "),
        "response_hash_native_equals_sample": row["native_response_sha256"] == row["response_sha256"],
        "response_hash_recomputed_from_sample_csv": array_hash(sample.arrivals.to_numpy(dtype=float)) == row["response_sha256"],
        "response_hash_recomputed_from_npz": array_hash(archive["Y"].astype(float)) == row["response_sha256"],
        "coefficient_hash_from_npz": array_hash(archive["beta"]) == row["native_coefficients_sha256"],
        "mu_hash_from_npz": array_hash(archive["mu"]) == row["native_mu_sha256"], "residual_hash_from_npz": array_hash(archive["residual"]) == row["native_residual_sha256"],
        "n_obs_receipt_equals_sample": int(row["n_obs"]) == len(sample) == len(archive["Y"]),
        "input_rows": len(given), "retained_rows": len(sample), "input_row_map_is_identity": bool(np.array_equal(sample.input_row.to_numpy(), np.arange(len(given)))),
        "input_zero_cells": int((given.arrivals == 0).sum()), "retained_zero_cells": int((sample.arrivals == 0).sum()),
        "ledger_zero_counts_ok": int(spec["input_zero_cells"]) == int((given.arrivals == 0).sum()) and int(spec["retained_zero_cells"]) == int((sample.arrivals == 0).sum()),
        "units_in": int(given.unit_id.nunique()), "units_retained": int(sample.unit_id.nunique()),
        "growth_or_lag_columns_in_sample": [c for c in sample.columns if any(key in c for key in ["y_mom", "y_yoy", "growth", "_ar", "lag_y"])],
        "balanced": bool(len(sample) == sample.unit_id.nunique() * sample.month.nunique()),
        "mu_positive_finite": bool(np.isfinite(archive["mu"]).all() and (archive["mu"] > 0).all()),
        "mu_plus_residual_equals_y_max_abs": maxabs(archive["mu"] + archive["residual"], archive["Y"].astype(float)),
        "exp_eta_equals_mu_max_rel": float(np.max(np.abs(np.exp(archive["eta"]) - archive["mu"]) / archive["mu"])),
        "dropped_by_stage": row["dropped_by_stage"]})
receipts = pd.DataFrame(receipt_rows)
bool_columns = ["converged", "formula_has_no_offset_or_transform", "response_hash_native_equals_sample", "response_hash_recomputed_from_sample_csv", "response_hash_recomputed_from_npz",
                "coefficient_hash_from_npz", "mu_hash_from_npz", "residual_hash_from_npz", "n_obs_receipt_equals_sample", "input_row_map_is_identity", "ledger_zero_counts_ok", "balanced", "mu_positive_finite"]
alias_ok = 0
for row in aliases.itertuples():
    a, b = specs.loc[specs.spec_id.eq(row.spec_id)].iloc[0], specs.loc[specs.spec_id.eq(row.native_spec_id)].iloc[0]
    alias_ok += int(a["alias_of"] == row.native_spec_id and not bool(a["native_fit_executed"]) and bool(b["native_fit_executed"]) and a["design_npz"] == b["design_npz"]
                    and row.spec_id.replace("strict_growth_window_clean", "event_dates_excluded") == row.native_spec_id)
part_a = {"receipts": len(receipts), "native_calls_manifest": manifest["native_calls"], "classes": sorted(receipts.native_class.unique().tolist()), "apis": sorted(receipts.api.unique().tolist()),
          "modules": sorted(receipts.native_module.unique().tolist()), "versions": sorted(receipts.version.astype(str).unique().tolist()),
          "failed": {c: int((~receipts[c]).sum()) for c in bool_columns}, "postprocessing": receipts.postprocessing.value_counts().to_dict(),
          "dropped_positions_values": sorted(receipts.dropped_positions.unique().tolist()), "dropped_observation_ledger_rows": len(dropped),
          "total_input_zero_cells": int(receipts.input_zero_cells.sum()), "total_retained_zero_cells": int(receipts.retained_zero_cells.sum()),
          "specs_with_zero_cells": int((receipts.input_zero_cells > 0).sum()), "units_dropped_anywhere": int((receipts.units_in != receipts.units_retained).sum()),
          "growth_or_lag_columns_anywhere": sorted({c for row in receipts.growth_or_lag_columns_in_sample for c in row}),
          "mu_plus_residual_max_abs": float(receipts.mu_plus_residual_equals_y_max_abs.max()), "exp_eta_vs_mu_max_rel": float(receipts.exp_eta_equals_mu_max_rel.max()),
          "aliases": len(aliases), "aliases_consistent": alias_ok, "fit_log_sha_matches_manifest": hashlib.sha256((PPML / "pyfixest/fit_log.csv").read_bytes()).hexdigest() == manifest["fit_log_sha256"]}

# ------------------------------------------------------------- B. one design
spec = specs.loc[specs.spec_id.eq(SPEC)].iloc[0]
sample = pd.read_csv(ROOT / spec["sample_csv"], dtype=STR, float_precision="round_trip")
archive = np.load(PPML / "designs" / f"{SPEC}.npz")
terms = archive["terms"].tolist()

meta = json.loads((ROOT / "data/raw/visitor_arrivals/nsa_singstat_metadata.json").read_text())["Data"]["records"]
by_name = {row["rowText"].strip(): str(row["seriesNo"]) for row in meta["row"]}
with open(ROOT / "data/raw/visitor_arrivals/nsa.csv", newline="") as handle:
    raw_rows = list(csv.reader(handle))
header = [f"{text[:4]}-{MONTHS[text[4:]]:02d}" for text in raw_rows[0][1:]]
raw = {(by_name[row[0].strip()], month): int(text) for row in raw_rows[1:] for month, text in zip(header, row[1:])}
named = sorted(u for u in set(by_name.values()) if u.count(".") == 2 and not u.endswith(".999"))
raw_values = np.array([raw[(u, m)] for u, m in zip(sample.unit_id, sample.month)])
raw_zero_cells = sorted((u, m) for (u, m), v in raw.items() if u in named and v == 0)
sample_zero_cells = sorted((u, m) for u, m, v in zip(sample.unit_id, sample.month, sample.arrivals) if v == 0)
weather_raw = {}
for stem, column in [("temperature", "mean_temp"), ("humidity", "mean_rh"), ("rainfall", "total_rainfall")]:
    frame = pd.read_csv(ROOT / f"data/raw/{stem}_monthly.csv")
    weather_raw[stem] = dict(zip(frame.month, frame[column].astype(float)))
lunar = pd.read_csv(PPML / "lunar_calendar_extended.csv").set_index("month")
x = archive["X"]
literal = {"temp_c": maxabs(x[:, terms.index("temp_c")], [weather_raw["temperature"][m] for m in sample.month]),
           "rain_100mm": maxabs(x[:, terms.index("rain_100mm")], [weather_raw["rainfall"][m] / 100 for m in sample.month]),
           "rh_pct": maxabs(x[:, terms.index("rh_pct")], [weather_raw["humidity"][m] for m in sample.month]),
           "lny_window_days": maxabs(x[:, terms.index("lny_window_days")], [lunar.loc[m, "lny_window_days"] for m in sample.month]),
           "lny_lag1": maxabs(x[:, terms.index("lny_lag1")], [lunar.loc[m, "lny_lag1"] for m in sample.month])}
january_2008 = sample.loc[sample.month.eq("2008-01")]
row_map = {"rows": len(sample), "expected_rows_42_markets_x_225_months": len(named) * len(header), "units_equal_named_markets": sorted(sample.unit_id.unique()) == named,
           "months_equal_all_225": sorted(sample.month.unique()) == sorted(header), "arrivals_equal_raw_integer_counts": bool(np.array_equal(sample.arrivals.to_numpy(), raw_values)),
           "npz_y_equal_raw": bool(np.array_equal(archive["Y"], raw_values)), "npz_y_dtype": str(archive["Y"].dtype),
           "raw_zero_cells_named_markets": len(raw_zero_cells), "sample_zero_cells": len(sample_zero_cells), "zero_cell_sets_equal": raw_zero_cells == sample_zero_cells,
           "zero_cell_months": sorted({m for _, m in sample_zero_cells}), "zero_cell_units": sorted({u for u, _ in sample_zero_cells}),
           "january_2008_rows": len(january_2008), "january_2008_lny_lag1_values": sorted(january_2008.lny_lag1.unique().tolist()),
           "january_2008_lny_window_days": sorted(january_2008.lny_window_days.unique().tolist()), "literal_control_max_abs_diff": literal,
           "x_columns_uncentered_means": {term: float(x[:, j].mean()) for j, term in enumerate(terms)}, "sample_columns": sample.columns.tolist()}

# ---- independent IRLS with exact absorption of origin x calendar-month cells
y = raw_values.astype(float)
unit = sample.unit_id.to_numpy()
season = np.array([int(m[5:7]) for m in sample.month])
year = np.array([int(m[:4]) for m in sample.month])
time = np.array([ordinal(int(m[:4]), int(m[5:7])) for m in sample.month])
cell_labels = sorted({(u, s) for u, s in zip(unit, season)})
cell_index = {label: number for number, label in enumerate(cell_labels)}
cell = np.array([cell_index[(u, s)] for u, s in zip(unit, season)])
years = sorted(set(year.tolist()))
year_dummies = np.column_stack([(year == value).astype(float) for value in years[1:]])
inner = np.column_stack([year_dummies, x])


def absorb(values, weights):
    total = np.bincount(cell, weights=weights)
    if values.ndim == 1:
        return values - (np.bincount(cell, weights=weights * values) / total)[cell]
    return np.column_stack([values[:, j] - (np.bincount(cell, weights=weights * values[:, j]) / total)[cell] for j in range(values.shape[1])])


def irls(outcome, rows=None, tolerance=1e-13, limit=200):
    keep = np.arange(len(outcome)) if rows is None else rows
    cells = cell[keep]
    block, target = inner[keep], outcome[keep]
    totals = np.bincount(cells, weights=target, minlength=len(cell_labels))
    counts = np.bincount(cells, minlength=len(cell_labels))
    assert (totals[counts > 0] > 0).all(), "an origin x month cell is all zero: separation"
    mu = (totals / np.maximum(counts, 1))[cells]
    eta = np.log(mu)
    for iteration in range(1, limit + 1):
        z = eta + (target - mu) / mu
        weight_total = np.bincount(cells, weights=mu, minlength=len(cell_labels))
        z_tilde = z - (np.bincount(cells, weights=mu * z, minlength=len(cell_labels)) / weight_total)[cells]
        d_tilde = np.column_stack([block[:, j] - (np.bincount(cells, weights=mu * block[:, j], minlength=len(cell_labels)) / weight_total)[cells] for j in range(block.shape[1])])
        root = np.sqrt(mu)
        theta, *_ = np.linalg.lstsq(d_tilde * root[:, None], z_tilde * root, rcond=None)
        new_eta = z - (z_tilde - d_tilde @ theta)
        change = float(np.max(np.abs(new_eta - eta)))
        eta, mu = new_eta, np.exp(new_eta)
        if change < tolerance:
            break
    return theta[-x.shape[1]:], mu, eta, iteration, change


beta, mu, eta, iterations, last_change = irls(y)
residual = y - mu
dense = np.column_stack([np.stack([(cell == number).astype(float) for number in range(len(cell_labels))], axis=1), year_dummies, x])
dense_rank = scaled_rank(dense)
score = dense.T @ residual
information_full = (dense * mu[:, None]).T @ dense
scale = np.sqrt(np.diag(information_full))
inverse_full = np.linalg.inv(information_full / np.outer(scale, scale)) / np.outer(scale, scale)
k_beta = x.shape[1]
xr = x - dense[:, :-k_beta] @ np.linalg.lstsq(dense[:, :-k_beta] * np.sqrt(mu)[:, None], x * np.sqrt(mu)[:, None], rcond=None)[0]
information = xr.T @ (mu[:, None] * xr)
bread = np.linalg.inv(information)
scores = xr * residual[:, None]
influence = scores @ bread
hc0 = influence.T @ influence
fit = {"iterations": iterations, "last_max_abs_eta_change": last_change, "dense_design_columns": dense.shape[1], "dense_design_rank": dense_rank,
       "n_parameters_ledger": int(spec["n_parameters"]), "min_cell_total": float(np.bincount(cell, weights=y).min()), "min_year_total": float(min(y[year == value].sum() for value in years)),
       "beta_mine": beta, "beta_saved": archive["beta"], "beta_max_abs_diff": maxabs(beta, archive["beta"]), "beta_max_rel_diff": float(np.max(np.abs(beta - archive["beta"]) / np.abs(archive["beta"]))),
       "mu_max_rel_diff": float(np.max(np.abs(mu - archive["mu"]) / archive["mu"])), "eta_max_abs_diff": maxabs(eta, archive["eta"]), "residual_max_abs_diff": maxabs(residual, archive["residual"]),
       "score_all_527_columns_max_abs_over_sum_y": float(np.max(np.abs(score)) / y.sum()), "saved_score_slopes_max_abs_over_sum_y": float(np.max(np.abs(x.T @ archive["residual"])) / y.sum()),
       "ledger_relative_score_error": float(spec["relative_score_error"]),
       "xr_max_abs_diff_vs_saved_alternating_projection": maxabs(xr, archive["xr"]), "information_max_rel_diff": maxrel(information, archive["information"]),
       "bread_max_rel_diff": maxrel(bread, archive["bread"]), "bread_equals_dense_inverse_block_max_rel": maxrel(inverse_full[-k_beta:, -k_beta:], bread),
       "scores_max_rel_diff": maxrel(scores, archive["scores"]), "influence_max_rel_diff": maxrel(influence, archive["influence"]), "hc0_max_rel_diff": maxrel(hc0, archive["hc0"]),
       "weights_equal_mu_max_rel": float(np.max(np.abs(archive["weights"] - archive["mu"]) / archive["mu"])),
       "mu_at_zero_cells_min_max": [float(mu[y == 0].min()), float(mu[y == 0].max())], "deviance": float(2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu))),
       "pearson_chi2_over_df": float(np.sum(residual ** 2 / mu) / (len(y) - dense.shape[1]))}

# ---- covariance layers
units = sorted(set(unit.tolist()))
local = []
for u in units:
    rows = unit == u
    block = np.column_stack([np.column_stack([(year[rows] == value).astype(float) for value in years]), np.column_stack([(season[rows] == s).astype(float) for s in range(1, 13)]), x[rows]])
    local.append(scaled_rank(block))
t, g, n = len(set(time.tolist())), len(units), len(y)
k_time = max(local)
factor = t / (t - k_time)
labels = archive["covariance_labels"].tolist()
covariances = {}
layers = {"k_time_mine": k_time, "k_time_ledger": int(spec["time_design_rank"]), "local_ranks_all_equal": len(set(local)) == 1, "factor_mine": factor,
          "corrections_saved": archive["corrections"].tolist(), "inference_df_saved": archive["inference_df"].tolist(), "labels": labels,
          "time_axis_matches_month_ordinal": bool(np.array_equal(time, archive["time"])), "calendar_gaps": int(time.max() - time.min() + 1 - t)}
for index, (label, bandwidth) in enumerate(zip(labels, archive["bandwidths"].tolist())):
    if label == "Calendar HAC":
        raw_mine = bartlett_calendar(influence, time, int(bandwidth))
        layers[f"raw_hac{bandwidth}_max_rel_diff"] = maxrel(raw_mine, archive[f"raw_calendar_covariance_{index}"])
        covariances[(label, int(bandwidth))] = (raw_mine * factor, t - 1)
    else:
        origin_scores = np.vstack([influence[unit == u].sum(axis=0) for u in units])
        cluster_factor = g / (g - 1) * (n - 1) / (n - int(spec["n_parameters"]))
        covariances[(label, int(bandwidth))] = (origin_scores.T @ origin_scores * cluster_factor, g - 1)
        layers["cluster_factor_mine"] = cluster_factor
    layers[f"covariance_{index}_{label}_{bandwidth}_max_rel_diff"] = maxrel(covariances[(label, int(bandwidth))][0], archive[f"covariance_{index}"])
effects = pd.read_csv(PPML / "temperature_effects.csv", dtype=STR, float_precision="round_trip", low_memory=False)
control_rows = pd.read_csv(PPML / "control_estimates.csv", dtype=STR, float_precision="round_trip", low_memory=False)
ledger_diff = {"coefficient": 0.0, "variance_rel": 0.0, "p": 0.0, "ci": 0.0, "effect_0p1c": 0.0, "exact_percent_0p1c": 0.0}
table = []
for (label, bandwidth), (covariance, df) in covariances.items():
    for j, term in enumerate(terms):
        book = effects if term == "temp_c" else control_rows
        saved = book.loc[book.spec_id.eq(SPEC) & book.covariance.eq(label) & book.hac_lags.eq(bandwidth) & book.term.eq(term)]
        assert len(saved) == 1
        saved = saved.iloc[0]
        coefficient, variance = 100 * beta[j], 1e4 * covariance[j, j]
        se = float(np.sqrt(variance))
        p = float(2 * stats.t.sf(abs(coefficient / se), df))
        critical = float(stats.t.ppf(0.975, df))
        ledger_diff["coefficient"] = max(ledger_diff["coefficient"], abs(coefficient - saved.coefficient))
        ledger_diff["variance_rel"] = max(ledger_diff["variance_rel"], abs(variance - saved.variance) / saved.variance)
        ledger_diff["p"] = max(ledger_diff["p"], abs(p - saved.p_value))
        ledger_diff["ci"] = max(ledger_diff["ci"], abs(coefficient - critical * se - saved.ci_lower), abs(coefficient + critical * se - saved.ci_upper))
        if term == "temp_c":
            ledger_diff["effect_0p1c"] = max(ledger_diff["effect_0p1c"], abs(10 * beta[j] - saved.effect_0p1c))
            ledger_diff["exact_percent_0p1c"] = max(ledger_diff["exact_percent_0p1c"], abs(100 * np.expm1(0.1 * beta[j]) - saved.exact_percent_0p1c))
            table.append({"covariance": label, "hac_lags": bandwidth, "coefficient_100_log_points_per_c": float(saved.coefficient), "std_error": float(saved.std_error), "p_value": float(saved.p_value),
                          "ci": [float(saved.ci_lower), float(saved.ci_upper)], "inference_df": int(saved.inference_df), "main_inference": bool(saved.main_inference),
                          "model_based_poisson_se": float(100 * np.sqrt(bread[j, j])), "hc0_se": float(100 * np.sqrt(hc0[j, j]))})
layers["ledger_max_abs_diff"] = ledger_diff
layers["temperature_rows"] = table

# ---- descriptive facts
policy = json.loads((ROOT / "notes/crisis_sensitivity_policy.json").read_text())
event = np.zeros(len(y), bool)
for item in policy["events"]:
    event |= (time >= ordinal(int(item["start"][:4]), int(item["start"][5:7]))) & (time <= ordinal(int(item["end"][:4]), int(item["end"][5:7])))
first = int(time.min())
bins = np.zeros((int(time.max()) - first + 1, k_beta))
np.add.at(bins, time - first, influence)
event_bins = np.zeros(len(bins), bool)
event_bins[np.unique(time[event]) - first] = True
covid_bins = np.zeros(len(bins), bool)
covid_bins[ordinal(2020, 1) - first: ordinal(2022, 12) - first + 1] = True
j = terms.index("temp_c")
facts = {"event_months_in_sample": int(event_bins.sum()), "covid_months_in_sample": int(covid_bins.sum()), "months": int(len(bins)),
         "share_of_date_summed_temperature_influence_squares_in_event_months": float(np.sum(bins[event_bins, j] ** 2) / np.sum(bins[:, j] ** 2)),
         "share_in_covid_months_2020_2022": float(np.sum(bins[covid_bins, j] ** 2) / np.sum(bins[:, j] ** 2)),
         "share_of_poisson_weight_mu_in_event_months": float(mu[event].sum() / mu.sum()),
         "share_of_information_for_temperature_in_event_months": float(np.sum(mu[event] * xr[event, j] ** 2) / np.sum(mu * xr[:, j] ** 2)),
         "sum_of_date_influence_all_months_abs": float(abs(bins[:, j].sum())), "first_order_event_month_contribution_to_beta_100": float(100 * bins[event_bins, j].sum())}
beta_no_zero, *_ = irls(y, rows=np.flatnonzero(y > 0))
beta_plus_one, *_ = irls(y + 1.0)
event_archive = np.load(PPML / "designs" / "ppml__country__long__event_dates_excluded.npz")
beta_event, *_ = irls(y, rows=np.flatnonzero(~event))
facts.update(comparator_temperature_coefficient_100={"saved_zero_retaining": float(100 * archive["beta"][j]), "mine_zero_retaining": float(100 * beta[j]),
                                                     "NOT_DONE_zero_cells_deleted": float(100 * beta_no_zero[j]), "NOT_DONE_arrivals_plus_one": float(100 * beta_plus_one[j]),
                                                     "event_dates_excluded_saved": float(100 * event_archive["beta"][j]), "event_dates_excluded_mine": float(100 * beta_event[j])},
             saved_minus_zero_deleted_abs_100=float(abs(100 * (archive["beta"][j] - beta_no_zero[j]))), saved_minus_plus_one_abs_100=float(abs(100 * (archive["beta"][j] - beta_plus_one[j]))),
             event_excluded_refit_max_abs_diff_vs_saved_beta=maxabs(beta_event, event_archive["beta"]))
all_temperature = effects.loc[effects.term.eq("temp_c") & effects.covariance.eq("Calendar HAC") & effects.hac_lags.eq(12),
                              ["spec_id", "coefficient", "std_error", "p_value", "exact_percent_0p1c", "native_fit_executed"]]
facts["all_30_temperature_rows_hac12"] = [{"spec_id": r.spec_id, "coefficient": round(float(r.coefficient), 4), "std_error": round(float(r.std_error), 4), "p_value": round(float(r.p_value), 4),
                                           "exact_percent_0p1c": round(float(r.exact_percent_0p1c), 4), "native_fit_executed": bool(r.native_fit_executed)} for r in all_temperature.itertuples()]

write("probe08_result.json", {"probe": "08 PPML zero retention, receipts, independent IRLS and sandwich", "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe08_ppml_zero_retention_irls.py",
                              "production_reference": "scripts/visitor_rigour_companions.py:310-443 ppml_models; scripts/visitor_counts.py:15-27 weighted_residualize",
                              "A_receipts": part_a, "B_row_map": row_map, "B_independent_fit": fit, "B_covariance_layers": layers, "B_descriptive_facts": facts, "receipt_rows": receipt_rows})
print("A", json.dumps(part_a, default=str))
print("B row map", json.dumps({k: v for k, v in row_map.items() if k not in ["sample_columns"]}, default=str))
print("B fit", json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}, default=str))
print("B layers", json.dumps(layers, default=str))
print("B facts", json.dumps({k: v for k, v in facts.items() if k != "all_30_temperature_rows_hac12"}, default=str))
for row in facts["all_30_temperature_rows_hac12"]:
    print("   ", row)
