"""Probe 07 - annual companion: counts, day-weighted weather, growth, masks, all 30 statuses.

Independent of every production module.
  A. 12-month sums from the raw official file (integer arithmetic), complete
     years only, annual log growth; compare annual_counts.csv for all 62 series.
  B. Day-weighted annual temperature / RH / wind / PM2.5 and rain sums from the
     raw monthly files with calendar.monthrange weights; all-12-month 90 %
     eligibility; compare annual_weather.csv. Unweighted means are reported as a
     sensitivity only.
  C. Annual event / strict masks from the policy file; compare calendar_masks.csv.
  D. All 30 requests: rebuild the literal design from the saved sample, compare
     X/names, recompute N, K, scaled rank, local ranks, K_time, status and the
     separate availability flags; confirm the 15 full-control requests keep the
     wind and PM columns and carry no coefficient.
  E. The 15 estimated requests: least-squares refit, calendar-year Bartlett
     HAC1/HAC2 with zero bins for removed years, T/(T-K_time), pooled-origin
     CR1, ledger coefficient/variance/p/CI/Holm and joint tests.
  F. Descriptive design facts needed to describe the annual model accurately:
     trend/temperature collinearity, residual temperature variation, influence
     of single years (leave-one-year-out, arithmetic only), the mean-equation
     identity for pooled slopes, the annual LNY constant, and the exact
     fixed-design iid-Gaussian null law of the saved t ratio (Craig identity
     from probe 05; a favourable benchmark, not a law under dependence).

Command: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe07_annual_reconstruction.py
"""
from _probe_common import BASE, ROOT, bartlett_calendar, maxabs, maxrel, scaled_rank, write

import calendar
import csv
import json
import math

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

ANNUAL = BASE / "annual"
MONTHS = {name: number for number, name in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
STR = {"unit_id": str, "sector_id": str, "reference_unit_id": str}
BASE_TERMS, FULL_TERMS = ["temp_c", "rain_100mm", "rh_pct"], ["temp_c", "rain_100mm", "rh_pct", "wind_mps", "pm25_10ug"]
VARIANTS = ["all_eligible", "event_dates_excluded", "strict_growth_window_clean"]


def ledger(path):
    return pd.read_csv(path, dtype=STR, float_precision="round_trip", low_memory=False)


# ---------------------------------------------------------------- A. counts
meta = json.loads((ROOT / "data/raw/visitor_arrivals/nsa_singstat_metadata.json").read_text())["Data"]["records"]
catalog = [(str(row["seriesNo"]), row["rowText"].strip()) for row in meta["row"]]
by_name = {name: unit for unit, name in catalog}
with open(ROOT / "data/raw/visitor_arrivals/nsa.csv", newline="") as handle:
    raw_rows = list(csv.reader(handle))
header = [(int(text[:4]), MONTHS[text[4:]]) for text in raw_rows[0][1:]]
counts = {by_name[row[0].strip()]: {ym: int(text) for ym, text in zip(header, row[1:])} for row in raw_rows[1:]}
ids = [unit for unit, _ in catalog]
cuts = {"aggregate": ["1"], "region": sorted(u for u in ids if u.count(".") == 1 and u != "1.999"),
        "country": sorted(u for u in ids if u.count(".") == 2 and not u.endswith(".999"))}
years_present = sorted({year for year, _ in header})
complete_years = [year for year in years_present if all((year, month) in counts["1"] for month in range(1, 13))]
annual = {unit: {year: sum(counts[unit][(year, month)] for month in range(1, 13)) for year in complete_years} for unit in ids}


def annual_growth(unit, year):
    now, before = annual[unit].get(year), annual[unit].get(year - 1)
    if now is None or before is None or now <= 0 or before <= 0:
        return float("nan")
    return 100.0 * (math.log(now) - math.log(before))


saved_counts = ledger(ANNUAL / "annual_counts.csv")
count_diff, previous_diff, growth_diff, growth_nan_mismatch = 0.0, 0.0, 0.0, 0
for row in saved_counts.itertuples():
    count_diff = max(count_diff, abs(row.arrivals - annual[row.unit_id][row.year]))
    previous = annual[row.unit_id].get(row.year - 1)
    if previous is None:
        growth_nan_mismatch += int(not (pd.isna(row.previous_arrivals) and pd.isna(row.y_annual_nsa)))
    else:
        previous_diff = max(previous_diff, abs(row.previous_arrivals - previous))
        mine = annual_growth(row.unit_id, row.year)
        if math.isnan(mine) or pd.isna(row.y_annual_nsa):
            growth_nan_mismatch += int(not (math.isnan(mine) and pd.isna(row.y_annual_nsa)))
        else:
            growth_diff = max(growth_diff, abs(mine - row.y_annual_nsa))
growth_years = [year for year in complete_years if year - 1 in complete_years]
zero_annual = [(unit, year) for unit in ids for year in complete_years if annual[unit][year] == 0]
zero_months_in_complete_years = sum(1 for unit in cuts["country"] for year in complete_years for month in range(1, 13) if counts[unit][(year, month)] == 0)
partial_year_months = {year: sum(1 for y, _ in header if y == year) for year in years_present if year not in complete_years}
part_a = {"complete_level_years": [complete_years[0], complete_years[-1], len(complete_years)], "incomplete_years_excluded_with_month_counts": partial_year_months,
          "growth_years": [growth_years[0], growth_years[-1], len(growth_years)], "saved_rows": len(saved_counts), "expected_rows": len(ids) * len(complete_years),
          "arrivals_max_abs_diff": count_diff, "previous_arrivals_max_abs_diff": previous_diff, "growth_max_abs_diff": growth_diff, "growth_nan_mismatches": growth_nan_mismatch,
          "annual_totals_equal_zero": zero_annual, "zero_months_inside_complete_years_named_markets": zero_months_in_complete_years,
          "aggregate_annual_growth": {str(year): round(annual_growth("1", year), 3) for year in growth_years},
          "aggregate_annual_arrivals": {str(year): annual["1"][year] for year in complete_years}}

# ---------------------------------------------------------------- B. weather
monthly = {}
for stem, column, name in [("temperature", "mean_temp", "temp_c"), ("humidity", "mean_rh", "rh_pct"), ("rainfall", "total_rainfall", "rain_mm")]:
    frame = pd.read_csv(ROOT / f"data/raw/{stem}_monthly.csv")
    monthly[name] = {(int(m[:4]), int(m[5:7])): float(v) for m, v in zip(frame.month, frame[column])}
controls = pd.read_csv(ROOT / "data/processed/extended_controls/monthly_controls.csv")
for name in ["wind_mps", "pm25_10ug", "wind_coverage", "pm_coverage"]:
    monthly[name] = {(int(m[:4]), int(m[5:7])): float(v) for m, v in zip(controls.month, controls[name])}
monthly["eligible_90"] = {(int(m[:4]), int(m[5:7])): bool(v) for m, v in zip(controls.month, controls.eligible_90)}
saved_weather = ledger(ANNUAL / "annual_weather.csv").set_index("year")
weather_diff, unweighted_gap, eligible_mismatch, coverage_rule_mismatch = {}, {}, 0, 0
annual_weather = {}
for year in complete_years:
    days = [calendar.monthrange(year, month)[1] for month in range(1, 13)]
    row = {}
    for name in ["temp_c", "rh_pct", "wind_mps", "pm25_10ug"]:
        values = [monthly[name].get((year, month), float("nan")) for month in range(1, 13)]
        row[name] = sum(v * d for v, d in zip(values, days)) / sum(days) if all(not math.isnan(v) for v in values) else float("nan")
        row[name + "_unweighted"] = sum(values) / 12 if all(not math.isnan(v) for v in values) else float("nan")
    rain = [monthly["rain_mm"].get((year, month), float("nan")) for month in range(1, 13)]
    row["rain_100mm"] = sum(v / 100 for v in rain) if all(not math.isnan(v) for v in rain) else float("nan")
    flags = [monthly["eligible_90"].get((year, month), False) for month in range(1, 13)]
    row["eligible_90"] = all(flags)
    rule = all(monthly["wind_coverage"].get((year, month), 0.0) >= 0.9 and monthly["pm_coverage"].get((year, month), 0.0) >= 0.9 for month in range(1, 13))
    coverage_rule_mismatch += int(rule != row["eligible_90"])
    row["months_eligible_90"] = sum(flags)
    annual_weather[year] = row
    for name in ["temp_c", "rh_pct", "wind_mps", "pm25_10ug", "rain_100mm"]:
        saved = saved_weather.loc[year, name]
        if math.isnan(row[name]) or pd.isna(saved):
            eligible_mismatch += int(not (math.isnan(row[name]) and pd.isna(saved)))
        else:
            weather_diff[name] = max(weather_diff.get(name, 0.0), abs(row[name] - saved))
            if name != "rain_100mm":
                unweighted_gap[name] = max(unweighted_gap.get(name, 0.0), abs(row[name] - row[name + "_unweighted"]))
    eligible_mismatch += int(bool(saved_weather.loc[year, "eligible_90"]) != row["eligible_90"])
part_b = {"max_abs_diff_vs_saved": weather_diff, "nan_or_eligibility_mismatches": eligible_mismatch, "eligible_flag_equals_both_coverages_ge_090_all_12_months_mismatches": coverage_rule_mismatch,
          "max_abs_gap_day_weighted_vs_unweighted_mean": unweighted_gap,
          "eligible_90_years": [year for year in complete_years if annual_weather[year]["eligible_90"]],
          "months_eligible_by_year": {str(year): annual_weather[year]["months_eligible_90"] for year in complete_years if not math.isnan(annual_weather[year]["wind_mps"]) or annual_weather[year]["months_eligible_90"]},
          "wind_and_pm_finite_years": [year for year in complete_years if not math.isnan(annual_weather[year]["wind_mps"]) and not math.isnan(annual_weather[year]["pm25_10ug"])]}

# annual LNY constant
lunar = pd.read_csv(BASE / "ppml/lunar_calendar_extended.csv")
lny_by_year = {}
for month, days in zip(lunar.month, lunar.lny_window_days):
    lny_by_year[int(month[:4])] = lny_by_year.get(int(month[:4]), 0) + int(days)
part_b["annual_lny_window_days_2008_2025"] = sorted({lny_by_year[year] for year in complete_years})

# ---------------------------------------------------------------- C. masks
policy = json.loads((ROOT / "notes/crisis_sensitivity_policy.json").read_text())
event_years = set()
for event in policy["events"]:
    event_years.update(range(int(event["start"][:4]), int(event["end"][:4]) + 1))
masks = ledger(ANNUAL / "calendar_masks.csv")
mask_errors, retained_sets = 0, {}
for (panel, control_set, variant), group in masks.groupby(["panel", "control_set", "sample_variant"], sort=False):
    terms = BASE_TERMS if control_set == "long" else FULL_TERMS
    eligible = [year for year in growth_years if all(not math.isnan(annual_growth(u, year)) for c in cuts.values() for u in c)
                and all(not math.isnan(annual_weather[year][term]) for term in terms) and (control_set == "long" or annual_weather[year]["eligible_90"])]
    mask_errors += int(sorted(group.year.tolist()) != eligible)
    for row in group.itertuples():
        current, prior = row.year in event_years, row.year - 1 in event_years
        keep = True if variant == "all_eligible" else (not current if variant == "event_dates_excluded" else not (current or prior))
        mask_errors += int(bool(row.current_event) != current) + int(bool(row.prior_year_event) != prior) + int(bool(row.retained) != keep) + int(row.output_support_start != row.year - 1)
    retained_sets[f"{panel}|{control_set}|{variant}"] = sorted(group.loc[group.retained.astype(bool), "year"].tolist())
part_c = {"mask_rows": len(masks), "mask_errors": mask_errors, "policy_event_years_in_window": sorted(y for y in event_years if 2008 <= y <= 2025),
          "retained_years_aggregate": {key.split("|", 1)[1]: value for key, value in retained_sets.items() if key.startswith("aggregate|")},
          "retained_sets_identical_across_panels": all(retained_sets[f"aggregate|{c}|{v}"] == retained_sets[f"{p}|{c}|{v}"] for p in ["region", "country"] for c in ["long", "full"] for v in VARIANTS)}


# ---------------------------------------------------------------- D/E. designs
def build(sample, terms, model):
    units = sorted(sample.unit_id.unique())
    names = [f"origin[{u}]" for u in units] + ["trend"] + terms[1:] + ["temp_c"]
    columns = [(sample.unit_id == u).to_numpy(float) for u in units]
    columns.append((sample.year - sample.year.mean()).to_numpy(float))
    for term in terms[1:]:
        columns.append((sample[term] - sample[term].mean()).to_numpy(float))
    temperature = (sample.temp_c - sample.temp_c.mean()).to_numpy(float)
    columns.append(temperature)
    if model == "H1":
        for u in units[1:]:
            names.append(f"temp_c:C(unit_id)[T.{u}]")
            columns.append(temperature * (sample.unit_id == u).to_numpy(float))
    return np.column_stack(columns), names, units


def craig(weights):
    weights = np.asarray(weights, float)
    weights = weights[weights > 0]
    if not len(weights):
        return 1.0
    function = lambda phi: float(np.exp(-0.5 * np.sum(np.log1p(weights / max(np.sin(phi) ** 2, 1e-300)))))
    return float(2 / np.pi * integrate.quad(function, 0, np.pi / 2, limit=400, epsabs=1e-13, epsrel=1e-11)[0])


def exact_law(matrix, years, vector, bandwidth, factor, degrees, t_observed):
    """Null law of c'b / sqrt(factor * c' HAC c) with the saved design fixed and iid Gaussian errors."""
    pinv = np.linalg.pinv(matrix)
    m = pinv.T @ vector
    first = int(years.min())
    span = int(years.max()) - first + 1
    spread = np.zeros((len(m), span))
    spread[np.arange(len(m)), years - first] = m
    q, _ = np.linalg.qr(matrix)
    projected = spread - q @ (q.T @ spread)
    lag = np.abs(np.subtract.outer(np.arange(span), np.arange(span)))
    kernel = np.clip(1.0 - lag / (bandwidth + 1.0), 0.0, None)
    values, vectors = np.linalg.eigh(kernel)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    eigen = np.clip(np.linalg.eigvalsh(root.T @ (projected.T @ projected) @ root), 0.0, None)
    eigen = eigen[eigen > eigen.max() * 1e-13]
    scaled = factor * eigen / float(m @ m)
    critical = float(stats.t.ppf(0.975, degrees))
    return {"exact_null_rejection_at_nominal_5pct": craig(critical ** 2 * scaled), "nominal_critical": critical,
            "exact_5pct_critical_abs_t": float(optimize.brentq(lambda value: craig(value ** 2 * scaled) - 0.05, 1e-3, 1e7, xtol=1e-10, rtol=1e-12)),
            "p_exact_benchmark": craig(t_observed ** 2 * scaled), "expected_variance_ratio": float(scaled.sum()),
            "satterthwaite_df": float(eigen.sum() ** 2 / (eigen ** 2).sum())}


specs = ledger(ANNUAL / "specifications.csv")
effects = ledger(ANNUAL / "temperature_effects.csv")
tests = ledger(ANNUAL / "tests.csv")
status_rows, fits, design_facts = [], [], []
for spec in specs.to_dict("records"):
    spec_id = spec["spec_id"]
    sample = pd.read_csv(ROOT / spec["sample_csv"], dtype={"unit_id": str}, float_precision="round_trip")
    terms = BASE_TERMS if spec["control_set"] == "long" else FULL_TERMS
    matrix, names, units = build(sample, terms, spec["model"])
    archive = np.load(ANNUAL / "designs" / f"{spec_id}.npz")
    n, k = matrix.shape
    years = sample.year.to_numpy()
    t, g = len(set(years.tolist())), len(units)
    unit = sample.unit_id.to_numpy()
    local = [scaled_rank(matrix[unit == u]) for u in units]
    k_time, rank = max(local), scaled_rank(matrix)
    status = "estimated" if n > k and rank == k else "unavailable"
    sample_growth_diff = max(abs(annual_growth(u, int(y)) - v) for u, y, v in zip(sample.unit_id, sample.year, sample.y_annual_nsa))
    sample_weather_diff = max(abs(annual_weather[int(y)][term] - v) for term in terms for y, v in zip(sample.year, sample[term]))
    own = effects.loc[effects.spec_id.eq(spec_id)]
    row = {"spec_id": spec_id, "n": n, "k": k, "t": t, "g": g, "rank_mine": rank, "k_time_mine": k_time, "time_df_mine": t - k_time, "status_mine": status,
           "status_ledger": spec["status"][:11], "status_ok": spec["status"].startswith(status), "x_max_abs_diff": maxabs(matrix, archive["X"]),
           "names_equal": names == archive["names"].tolist(), "y_max_abs_diff": maxabs(sample.y_annual_nsa.to_numpy(), archive["Y"]),
           "k_ledger_ok": int(spec["n_parameters"]) == k, "rank_ledger_ok": int(spec["design_rank"]) == rank, "k_time_ledger_ok": int(spec["time_design_rank"]) == k_time,
           "time_df_ledger_ok": int(spec["time_residual_df"]) == t - k_time, "t_ledger_ok": int(spec["n_periods"]) == t,
           "time_correction_available_ok": bool(spec["time_correction_available"]) == (t > k_time),
           "primary_available_ok": bool(spec["primary_calendar_inference_available"]) == (status == "estimated" and t > k_time),
           "sample_growth_max_abs_diff_vs_raw": sample_growth_diff, "sample_weather_max_abs_diff_vs_raw": sample_weather_diff,
           "balanced": bool(n == t * g), "years": sorted(set(years.tolist())), "wind_pm_columns_present": all(term in names for term in ["wind_mps", "pm25_10ug"]),
           "archive_has_beta": "beta" in archive.files, "ledger_rows": len(own), "ledger_rows_with_coefficient": int(own.coefficient.notna().sum()),
           "ledger_rows_with_p": int(own.p_value.notna().sum()), "ledger_inference_status": sorted(own.inference_status.astype(str).unique().tolist()),
           "flags_no_year_fe_no_lny": [bool(spec["year_fixed_effects"]), bool(spec["origin_fixed_effects"]), bool(spec["linear_trend"]), bool(spec["annual_LNY_included"])]}
    status_rows.append(row)
    if status != "estimated":
        continue
    outcome = sample.y_annual_nsa.to_numpy()
    beta, *_ = np.linalg.lstsq(matrix, outcome, rcond=None)
    residual = outcome - matrix @ beta
    pinv = np.linalg.pinv(matrix)
    influence = pinv.T * residual[:, None]
    factor = t / (t - k_time)
    labels = archive["covariance_labels"].tolist()
    fit = {"spec_id": spec_id, "beta_max_abs_diff": maxabs(beta, archive["beta"]), "residual_max_abs_diff": maxabs(residual, archive["residual"]),
           "influence_max_rel_diff": maxrel(influence, archive["influence"]), "hc0_max_rel_diff": maxrel(influence.T @ influence, archive["hc0"]),
           "labels": labels, "bandwidths": archive["bandwidths"].tolist(), "inference_df": archive["inference_df"].tolist(), "corrections_saved": archive["corrections"].tolist(),
           "correction_mine": factor, "cluster_presence_ok": ("Origin cluster" in labels) == (spec["model"] == "common" and g > 1)}
    covariances = {}
    for index, (label, bandwidth) in enumerate(zip(labels, archive["bandwidths"].tolist())):
        if label == "Calendar HAC":
            mine = bartlett_calendar(influence, years, int(bandwidth)) * factor
            df = t - 1
        else:
            scores = np.vstack([influence[unit == u].sum(axis=0) for u in units])
            mine = scores.T @ scores * (g / (g - 1) * (n - 1) / (n - k))
            df = g - 1
        covariances[(label, int(bandwidth))] = (mine, df)
        fit[f"covariance_{index}_{label}_{bandwidth}_max_rel_diff"] = maxrel(mine, archive[f"covariance_{index}"])
        fit[f"covariance_{index}_df_ok"] = int(archive["inference_df"][index]) == df
    # ledger rows
    worst = {"coefficient": 0.0, "variance_rel": 0.0, "p": 0.0, "ci": 0.0, "holm": 0.0}
    position = names.index("temp_c")
    for (label, bandwidth), (covariance, df) in covariances.items():
        rows = own.loc[own.covariance.eq(label) & own.hac_lags.eq(bandwidth)]
        assert len(rows) == (g if spec["model"] == "H1" else 1)
        ps = {}
        for saved in rows.itertuples():
            vector = np.zeros(k)
            vector[position] = 1.0
            if spec["model"] == "H1" and saved.unit_id != units[0]:
                vector[names.index(f"temp_c:C(unit_id)[T.{saved.unit_id}]")] = 1.0
            estimate, variance = float(vector @ beta), float(vector @ covariance @ vector)
            se = math.sqrt(variance)
            p = float(2 * stats.t.sf(abs(estimate / se), df))
            critical = float(stats.t.ppf(0.975, df))
            worst["coefficient"] = max(worst["coefficient"], abs(estimate - saved.coefficient))
            worst["variance_rel"] = max(worst["variance_rel"], abs(variance - saved.variance) / abs(saved.variance))
            worst["p"] = max(worst["p"], abs(p - saved.p_value))
            worst["ci"] = max(worst["ci"], abs(estimate - critical * se - saved.ci_lower), abs(estimate + critical * se - saved.ci_upper))
            ps[saved.unit_id] = (p, saved.p_value_holm_within_profile)
        if spec["model"] == "H1":
            order = sorted(ps, key=lambda u: ps[u][0])
            running = 0.0
            for number, u in enumerate(order):
                running = max(running, min(1.0, ps[u][0] * (len(order) - number)))
                worst["holm"] = max(worst["holm"], abs(running - ps[u][1]))
    fit["ledger_max_abs_diff"] = worst
    # joint tests
    joint_rows = []
    if spec["model"] == "H1":
        profile = np.zeros((g, k))
        profile[:, position] = 1.0
        for number, u in enumerate(units[1:], start=1):
            profile[number, names.index(f"temp_c:C(unit_id)[T.{u}]")] = 1.0
        for (label, bandwidth), (covariance, df) in covariances.items():
            for name, restriction in [("All origin temperature slopes equal", profile[1:] - profile[[0]]), ("All origin temperature slopes zero", profile)]:
                saved = tests.loc[tests.spec_id.eq(spec_id) & tests.covariance.eq(label) & tests.hac_lags.eq(bandwidth) & tests.test.eq(name)]
                assert len(saved) == 1
                saved = saved.iloc[0]
                q, bound = len(restriction), t - 1
                v = restriction @ covariance @ restriction.T
                singular = np.linalg.svd((v + v.T) / 2, compute_uv=False)
                numerical_rank = int(np.sum(singular > singular[0] * 1e-10))
                available = min(numerical_rank, bound) == q
                item = {"covariance": label, "hac_lags": bandwidth, "test": name, "q": q, "rank_bound": bound, "numerical_rank_mine": numerical_rank,
                        "available_mine": available, "status_ledger": saved.status, "available_ok": available == str(saved.status).startswith("computable"),
                        "fragile_ledger": bool(saved.fragile), "calibration_status": saved.calibration_status}
                if available:
                    effect = restriction @ beta
                    value = float(effect @ np.linalg.solve(v, effect)) / q
                    item.update(f_mine=value, f_ledger=float(saved.f_statistic), f_rel_diff=abs(value - saved.f_statistic) / abs(saved.f_statistic),
                                p_mine=float(stats.f.sf(value, q, df)), p_ledger=float(saved.p_value))
                joint_rows.append(item)
    fit["joint_tests"] = joint_rows
    fits.append(fit)

    # F. descriptive design facts for the pooled / single-series slope
    if spec["model"] == "common":
        vector = np.zeros(k)
        vector[position] = 1.0
        others = np.delete(matrix, position, axis=1)
        temp = matrix[:, position]
        temp_resid = temp - others @ np.linalg.lstsq(others, temp, rcond=None)[0]
        first = unit == units[0]
        year_list = years[first]
        share = {int(y): float(np.sum(temp_resid[years == y] ** 2) / np.sum(temp_resid ** 2)) for y in year_list}
        trend = matrix[:, names.index("trend")]
        loo = {}
        for y in sorted(set(years.tolist())):
            keep = years != y
            sub = sample.loc[keep]
            sub_matrix, sub_names, _ = build(sub, terms, "common")
            sub_beta = np.linalg.lstsq(sub_matrix, sub.y_annual_nsa.to_numpy(), rcond=None)[0]
            loo[int(y)] = float(sub_beta[sub_names.index("temp_c")])
        saved = own.loc[own.covariance.eq("Calendar HAC") & own.hac_lags.eq(1)].iloc[0]
        law = exact_law(matrix, years, vector, 1, factor, t - 1, float(saved.coefficient / saved.std_error))
        law2 = exact_law(matrix, years, vector, 2, factor, t - 1, float(saved.coefficient / saved.std_error))
        mean_y = np.array([outcome[years == y].mean() for y in year_list])
        mean_x = np.column_stack([np.ones(len(year_list))] + [matrix[first][:, names.index(name)] for name in ["trend"] + terms[1:] + ["temp_c"]])
        mean_beta = np.linalg.lstsq(mean_x, mean_y, rcond=None)[0]
        fact = {"spec_id": spec_id, "t": t, "k_time": k_time, "time_df": t - k_time, "correction": factor, "temp_coefficient": float(beta[position]),
                "ledger_hac1": {"std_error": float(saved.std_error), "p_value": float(saved.p_value), "ci": [float(saved.ci_lower), float(saved.ci_upper)]},
                "sd_annual_temperature_c": float(np.std(temp[first], ddof=1)), "corr_temperature_trend": float(np.corrcoef(temp[first], trend[first])[0, 1]),
                "r2_temperature_on_other_regressors": float(1 - np.sum(temp_resid[first] ** 2) / np.sum((temp[first] - temp[first].mean()) ** 2)),
                "sd_residual_temperature_c": float(np.sqrt(np.sum(temp_resid[first] ** 2) / (t - 1))),
                "largest_year_shares_of_residual_temperature_variation": sorted(share.items(), key=lambda item: -item[1])[:4],
                "leave_one_year_out_min_max": [min(loo.values()), max(loo.values())], "leave_one_year_out_extreme_years": [min(loo, key=loo.get), max(loo, key=loo.get)],
                "leave_one_year_out_sign_changes": int(sum(np.sign(v) != np.sign(beta[position]) for v in loo.values())),
                "sd_outcome": float(np.std(outcome, ddof=1)), "residual_sd": float(np.sqrt(residual @ residual / (n - k))),
                "pooled_slope_equals_mean_equation_slope_abs_diff": abs(float(mean_beta[-1]) - float(beta[position])),
                "exact_fixed_design_gaussian_hac1": law, "exact_fixed_design_gaussian_hac2_rejection": law2["exact_null_rejection_at_nominal_5pct"]}
        if "Origin cluster" in labels:
            cluster = own.loc[own.covariance.eq("Origin cluster")].iloc[0]
            fact["origin_cluster"] = {"std_error": float(cluster.std_error), "p_value": float(cluster.p_value), "se_ratio_cluster_over_hac1": float(cluster.std_error / saved.std_error),
                                      "main_inference": bool(cluster.main_inference), "inference_df": int(cluster.inference_df)}
        design_facts.append(fact)

status_frame = pd.DataFrame(status_rows)
checks = ["status_ok", "names_equal", "k_ledger_ok", "rank_ledger_ok", "k_time_ledger_ok", "time_df_ledger_ok", "t_ledger_ok", "time_correction_available_ok", "primary_available_ok", "balanced"]
full = status_frame.loc[status_frame.spec_id.str.contains("__full__")]
part_d = {"requests": len(status_frame), "status_counts_mine": status_frame.status_mine.value_counts().to_dict(), "failed_checks": {c: int((~status_frame[c]).sum()) for c in checks},
          "x_max_abs_diff": float(status_frame.x_max_abs_diff.max()), "y_max_abs_diff": float(status_frame.y_max_abs_diff.max()),
          "sample_growth_max_abs_diff_vs_raw": float(status_frame.sample_growth_max_abs_diff_vs_raw.max()), "sample_weather_max_abs_diff_vs_raw": float(status_frame.sample_weather_max_abs_diff_vs_raw.max()),
          "full_control_requests": len(full), "full_control_unavailable": int(full.status_mine.eq("unavailable").sum()), "full_control_wind_pm_columns_present": int(full.wind_pm_columns_present.sum()),
          "full_control_archives_with_beta": int(full.archive_has_beta.sum()), "full_control_ledger_rows": int(full.ledger_rows.sum()),
          "full_control_ledger_rows_with_coefficient": int(full.ledger_rows_with_coefficient.sum()), "full_control_ledger_rows_with_p": int(full.ledger_rows_with_p.sum()),
          "full_control_years": {v: full.loc[full.spec_id.str.contains(v), "years"].iloc[0] for v in VARIANTS},
          "full_control_n_k_rank": sorted({(int(r.n), int(r.k), int(r.rank_mine), int(r.t), int(r.k_time_mine)) for r in full.itertuples()}),
          "long_requests_estimated": int(status_frame.loc[status_frame.spec_id.str.contains("__long__"), "status_mine"].eq("estimated").sum()),
          "flags_year_fe_origin_fe_trend_lny": sorted({tuple(v) for v in status_frame.flags_no_year_fe_no_lny}),
          "unavailable_ledger_inference_status": sorted({s for r in full.itertuples() for s in r.ledger_inference_status})}
fit_frame = pd.DataFrame(fits)
covariance_columns = [c for c in fit_frame.columns if c.endswith("_max_rel_diff") and c.startswith("covariance_")]
joint_all = [dict(item, spec_id=f["spec_id"]) for f in fits for item in f["joint_tests"]]
part_e = {"estimated": len(fits), "beta_max_abs_diff": float(fit_frame.beta_max_abs_diff.max()), "residual_max_abs_diff": float(fit_frame.residual_max_abs_diff.max()),
          "influence_max_rel_diff": float(fit_frame.influence_max_rel_diff.max()), "hc0_max_rel_diff": float(fit_frame.hc0_max_rel_diff.max()),
          "covariance_max_rel_diff": float(np.nanmax(fit_frame[covariance_columns].to_numpy(float))),
          "cluster_presence_ok": bool(fit_frame.cluster_presence_ok.all()),
          "ledger_max_abs_diff": {key: max(f["ledger_max_abs_diff"][key] for f in fits) for key in ["coefficient", "variance_rel", "p", "ci", "holm"]},
          "joint_tests": len(joint_all), "joint_available_mismatch": int(sum(not item["available_ok"] for item in joint_all)),
          "joint_available": int(sum(item["available_mine"] for item in joint_all)),
          "joint_f_max_rel_diff": max((item["f_rel_diff"] for item in joint_all if item["available_mine"]), default=None),
          "joint_p_max_abs_diff": max((abs(item["p_mine"] - item["p_ledger"]) for item in joint_all if item["available_mine"]), default=None),
          "joint_by_panel": {panel: {"rows": sum(1 for i in joint_all if f"__{panel}__" in i["spec_id"]), "available": sum(1 for i in joint_all if f"__{panel}__" in i["spec_id"] and i["available_mine"]),
                                     "fragile": sum(1 for i in joint_all if f"__{panel}__" in i["spec_id"] and i["fragile_ledger"]),
                                     "available_p_lt_05": sum(1 for i in joint_all if f"__{panel}__" in i["spec_id"] and i["available_mine"] and i["p_ledger"] < 0.05)} for panel in ["region", "country"]},
          "joint_calibration_status": sorted({item["calibration_status"] for item in joint_all})}
h1_effects = effects.loc[effects.model.eq("H1") & effects.covariance.eq("Calendar HAC") & effects.hac_lags.eq(1) & effects.p_value.notna()]
part_e["h1_hac1_rows"] = {"rows": len(h1_effects), "p_lt_05": int((h1_effects.p_value < 0.05).sum()), "holm_lt_05": int((h1_effects.p_value_holm_within_profile < 0.05).sum())}

write("probe07_result.json", {"probe": "07 annual reconstruction", "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe07_annual_reconstruction.py",
                              "production_reference": "scripts/visitor_rigour_companions.py:155-168 annual_data, 171-179 annual_design, 182-284 annual_models",
                              "A_counts": part_a, "B_weather": part_b, "C_masks": part_c, "D_statuses": part_d, "E_fits": part_e, "F_design_facts": design_facts,
                              "status_rows": status_rows, "fits": fits})
print("A", json.dumps({k: v for k, v in part_a.items() if k not in ["aggregate_annual_arrivals"]}, default=str))
print("B", json.dumps(part_b, default=str))
print("C", json.dumps(part_c, default=str))
print("D", json.dumps(part_d, default=str))
print("E", json.dumps(part_e, default=str))
for fact in design_facts:
    law = fact["exact_fixed_design_gaussian_hac1"]
    print("\nF", fact["spec_id"], "| T", fact["t"], "timeDF", fact["time_df"], "factor", round(fact["correction"], 4))
    print("   temp coef", round(fact["temp_coefficient"], 3), "HAC1 se", round(fact["ledger_hac1"]["std_error"], 3), "p", round(fact["ledger_hac1"]["p_value"], 4), "CI", [round(v, 2) for v in fact["ledger_hac1"]["ci"]],
          "| origin cluster", fact.get("origin_cluster"))
    print("   sd temp", round(fact["sd_annual_temperature_c"], 4), "corr(temp,trend)", round(fact["corr_temperature_trend"], 3), "R2 temp|others", round(fact["r2_temperature_on_other_regressors"], 3),
          "sd resid temp", round(fact["sd_residual_temperature_c"], 4), "| top shares", [(y, round(s, 3)) for y, s in fact["largest_year_shares_of_residual_temperature_variation"]])
    print("   leave-one-year-out min/max", [round(v, 2) for v in fact["leave_one_year_out_min_max"]], "years", fact["leave_one_year_out_extreme_years"], "sign changes", fact["leave_one_year_out_sign_changes"],
          "| sd y", round(fact["sd_outcome"], 2), "resid sd", round(fact["residual_sd"], 2), "| mean-eq identity", fact["pooled_slope_equals_mean_equation_slope_abs_diff"])
    print("   exact fixed-design Gaussian: rejection@5%", round(law["exact_null_rejection_at_nominal_5pct"], 4), "exact crit", round(law["exact_5pct_critical_abs_t"], 3), "nominal crit", round(law["nominal_critical"], 3),
          "benchmark p", round(law["p_exact_benchmark"], 4), "E[V]/V", round(law["expected_variance_ratio"], 3), "Satt df", round(law["satterthwaite_df"], 2), "| HAC2 rejection", round(fact["exact_fixed_design_gaussian_hac2_rejection"], 4))
