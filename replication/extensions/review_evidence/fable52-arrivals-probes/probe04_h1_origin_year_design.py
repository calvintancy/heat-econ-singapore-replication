"""Probe 04 - normal H1 and origin-by-year designs rebuilt from the frozen parent samples.

Independent of the production design/fit code: the sample CSV is parsed with the
csv module, the LSDV designs are rebuilt column by column, coefficients come
from (a) an SVD pseudo-inverse / QR on the rebuilt design and (b) a separate
mean-equation + per-origin deviation-equation route (G+1 small time-series
regressions) that never forms the pooled design.

Checks
  A. three selected parent/origin-year pairs: design, names, coefficients,
     residuals, total = base + deviation, full cross-term variance, HAC12/24
     variances, p/CI/Holm, joint tests, reference-origin dependence of deltas.
  B. all 24 origin-year requests: bitwise parent equality of Y/time/unit/rows
     and of every stochastic column, nesting of the deterministic space,
     QR-bridge identities, coefficients/residuals, K_time and convention.
  C. the exact mean/deviation identity: the cross-origin mean slope and all
     shared control slopes are algebraically identical in parent and
     origin-year fits (and equal to the pooled common-slope model).

Command: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe04_h1_origin_year_design.py
"""
from _probe_common import BASE, ROOT, bartlett_calendar, maxabs, maxrel, scaled_rank, sha, write

import csv
import time as clock

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import stats

MONTHLY, OY = BASE / "monthly", BASE / "origin_year"
OTHER, ADDED = ["rh_pct", "lny_window_days", "lny_lag1"], ["wind_mps", "pm25_10ug"]
STR = {"unit_id": str, "sector_id": str, "reference_unit_id": str, "reference_sector_id": str}


def ledger(path):
    return pd.read_csv(path, dtype=STR, float_precision="round_trip", low_memory=False)


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def my_design(rows, control_set, nuisance):
    unit = np.array([row["unit_id"] for row in rows])
    season = np.array([int(row["season"]) for row in rows])
    year = np.array([int(row["year"]) for row in rows])
    units, seasons, years = sorted(set(unit.tolist())), sorted(set(season.tolist())), sorted(set(year.tolist()))
    controls = ["rain_l0"] + OTHER + (ADDED if control_set == "full" else [])
    columns, names, kinds = [], [], []
    for u in units:
        for s in seasons:
            columns.append(((unit == u) & (season == s)).astype(float)); names.append(f"origin_season[{u},{s}]"); kinds.append("det")
    if nuisance == "year":
        for y in years[1:]:
            columns.append((year == y).astype(float)); names.append(f"year[{y}]"); kinds.append("det")
    else:
        for u in units:
            for y in years[1:]:
                columns.append(((unit == u) & (year == y)).astype(float)); names.append(f"origin_year[{u},{y}]"); kinds.append("det")
    for control in controls:
        values = np.array([float(row[control]) for row in rows])
        columns.append(values - values.mean()); names.append(control); kinds.append("control")
    values = np.array([float(row["temp_l0"]) for row in rows])
    values = values - values.mean()
    columns.append(values); names.append("temp_l0"); kinds.append("temp")
    for u in units[1:]:
        columns.append(np.where(unit == u, values, 0.0)); names.append(f"temp_l0:C(unit_id)[T.{u}]"); kinds.append("temp")
    return np.column_stack(columns), names, np.array(kinds), units, seasons, years, controls, unit


def qr_fit(matrix, outcome):
    scale = np.linalg.norm(matrix, axis=0)
    q, r = np.linalg.qr(matrix / scale, mode="reduced")
    beta = scipy.linalg.solve_triangular(r, q.T @ outcome) / scale
    return beta, outcome - matrix @ beta


def decomposition(rows, outcome, control_set, nuisance):
    """Mean equation + per-origin deviation equations; never forms the pooled design."""
    units = sorted({row["unit_id"] for row in rows})
    dates = sorted({int(row["time"]) for row in rows})
    place, column = {d: k for k, d in enumerate(dates)}, {u: k for k, u in enumerate(units)}
    controls = ["rain_l0"] + OTHER + (ADDED if control_set == "full" else [])
    growth = np.full((len(dates), len(units)), np.nan)
    common = np.full((len(dates), len(controls) + 3), np.nan)
    spread = 0.0
    for row in rows:
        t = place[int(row["time"])]
        growth[t, column[row["unit_id"]]] = float(row[outcome])
        values = np.array([float(row[name]) for name in controls] + [float(row["temp_l0"]), float(row["season"]), float(row["year"])])
        if np.isfinite(common[t, 0]):
            spread = max(spread, float(np.max(np.abs(common[t] - values))))
        common[t] = values
    assert np.isfinite(growth).all() and np.isfinite(common).all(), "unbalanced or missing"
    season, year = common[:, -2].astype(int), common[:, -1].astype(int)
    seasonal = np.column_stack([(season == s).astype(float) for s in sorted(set(season.tolist()))])
    years = sorted(set(year.tolist()))
    annual = np.column_stack([(year == y).astype(float) for y in years[1:]]) if len(years) > 1 else np.zeros((len(dates), 0))
    weather, temperature = common[:, :len(controls)], common[:, [len(controls)]]
    mean_design = np.column_stack([seasonal, annual, weather, temperature])
    mean_coef = np.linalg.lstsq(mean_design, growth.mean(axis=1), rcond=None)[0]
    deviation_design = np.column_stack([seasonal, annual, temperature]) if nuisance == "origin_year" else np.column_stack([seasonal, temperature])
    deviation = np.array([np.linalg.lstsq(deviation_design, growth[:, k] - growth.mean(axis=1), rcond=None)[0][-1] for k in range(len(units))])
    return {"mean_slope": float(mean_coef[-1]), "control_slopes": mean_coef[-(len(controls) + 1):-1], "deviation": deviation,
            "common_regressor_cross_origin_spread": spread, "mean_design_rank": scaled_rank(mean_design), "n_dates": len(dates)}


def local_ranks(matrix, unit, time):
    ranks, blocks = {}, []
    for u in sorted(set(unit.tolist())):
        rows = np.flatnonzero(unit == u)
        rows = rows[np.argsort(time[rows], kind="stable")]
        local = matrix[rows]
        ranks[u] = scaled_rank(local)
        blocks.append(local[:, np.any(local != 0, axis=0)])
    return ranks, scaled_rank(np.column_stack(blocks))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return adjusted


def deep(parent_id, spec, effects, tests, components, family_dir, nuisance):
    """Full independent reconstruction of one saved H1 equation."""
    spec_id = parent_id if nuisance == "year" else parent_id + "__origin_year"
    row = spec.loc[spec.spec_id.eq(spec_id)].iloc[0]
    rows = read_rows(ROOT / (row["sample_csv"] if nuisance == "year" else row["source_sample_csv"]))
    matrix, names, kinds, units, seasons, years, controls, unit = my_design(rows, row["control_set"], nuisance)
    outcome = np.array([float(item[row["outcome"]]) for item in rows])
    time = np.array([int(item["time"]) for item in rows])
    archive = np.load(family_dir / "designs" / f"{spec_id}.npz")
    out = {"spec_id": spec_id, "n_obs": len(rows), "n_parameters": matrix.shape[1], "n_units": len(units), "n_periods": len(set(time.tolist())),
           "years": [years[0], years[-1], len(years)], "reference_unit": units[0],
           "names_equal": list(archive["names"]) == names, "design_max_abs_diff": maxabs(matrix, archive["X"]),
           "design_bitwise_equal_columns": int(sum(np.array_equal(matrix[:, j], archive["X"][:, j]) for j in range(matrix.shape[1]))),
           "outcome_bitwise_equal": bool(np.array_equal(outcome, archive["Y"])), "controls_equal_saved": list(archive["control_features"]) == controls,
           "ledger_controls": row["controls"]}
    pinv = np.linalg.pinv(matrix)
    beta = pinv @ outcome
    resid = outcome - matrix @ beta
    out.update(beta_max_abs_diff=maxabs(beta, archive["beta"]), resid_max_abs_diff=maxabs(resid, archive["resid"]),
               design_rank_mine=scaled_rank(matrix), normal_equation_max_abs=float(np.max(np.abs(matrix.T @ resid))))
    # FWL: partial the deterministic space and controls out of Y and of the temperature block.
    nuisance_block, temp_block = matrix[:, kinds != "temp"], matrix[:, kinds == "temp"]
    project = lambda target: target - nuisance_block @ np.linalg.lstsq(nuisance_block, target, rcond=None)[0]
    fwl = np.linalg.lstsq(project(temp_block), project(outcome), rcond=None)[0]
    out["fwl_temperature_block_max_abs_diff"] = maxabs(fwl, archive["beta"][kinds == "temp"])
    # Separate-slope reparameterisation: one own slope per origin instead of base + deviations.
    centred = matrix[:, names.index("temp_l0")]
    separate = np.column_stack([nuisance_block] + [np.where(unit == u, centred, 0.0) for u in units])
    own = np.linalg.lstsq(separate, outcome, rcond=None)[0][-len(units):]
    base, deltas = archive["beta"][names.index("temp_l0")], np.array([0.0] + [archive["beta"][names.index(f"temp_l0:C(unit_id)[T.{u}]")] for u in units[1:]])
    out["separate_slope_vs_base_plus_deviation_max_abs_diff"] = maxabs(own, base + deltas)
    route = decomposition(rows, row["outcome"], row["control_set"], nuisance)
    out.update(decomposition_total_max_abs_diff=maxabs(route["mean_slope"] + route["deviation"], base + deltas),
               decomposition_control_slope_max_abs_diff=maxabs(route["control_slopes"], archive["beta"][kinds == "control"]),
               decomposition_mean_slope=route["mean_slope"], mean_of_saved_origin_totals=float(np.mean(base + deltas)),
               decomposition_deviation_sum=float(np.sum(route["deviation"])), common_regressor_cross_origin_spread=route["common_regressor_cross_origin_spread"])
    # Working time rank and correction.
    ranks, span = local_ranks(matrix, unit, time)
    k_time, periods = max(ranks.values()), len(set(time.tolist()))
    factor = periods / (periods - k_time)
    out.update(k_time_mine=k_time, k_time_ledger=int(row["time_design_rank"]), full_span_mine=span, full_span_ledger=int(row["full_cross_origin_temporal_span_rank"]),
               local_rank_min=min(ranks.values()), correction_mine=factor, corrections_saved=archive["corrections"].tolist(),
               covariance_labels=archive["covariance_labels"].tolist(), bandwidths=archive["bandwidths"].tolist(), inference_df_saved=archive["inference_df"].tolist(),
               calendar_gaps=int(time.max() - time.min() + 1 - periods))
    selected = np.flatnonzero(kinds != "det")
    assert np.array_equal(selected, archive["selected_positions"])
    influence = pinv[selected].T * resid[:, None]
    temp_local = [list(selected).index(j) for j in np.flatnonzero(kinds == "temp")]
    totals = np.zeros((len(units), len(selected)))
    totals[:, temp_local[0]] = 1.0
    for number in range(1, len(units)):
        totals[number, temp_local[number]] = 1.0
    beta_selected = beta[selected]
    mine = {}
    for index, bandwidth in enumerate([12, 24]):
        raw = bartlett_calendar(influence, time, bandwidth)
        saved_raw = archive[f"raw_calendar_covariance_{index}"][np.ix_(selected, selected)]
        saved = archive[f"covariance_{index}"][np.ix_(selected, selected)]
        mine[bandwidth] = raw * factor
        out[f"hac{bandwidth}_raw_selected_max_rel_diff"] = maxrel(raw, saved_raw)
        out[f"hac{bandwidth}_corrected_selected_max_rel_diff"] = maxrel(raw * factor, saved)
    out["hc0_selected_max_rel_diff"] = maxrel(influence.T @ influence, archive["hc0_full"][np.ix_(selected, selected)])
    # Ledger: total = base + deviation with the full cross term, p/CI/Holm.
    table = effects.loc[effects.spec_id.eq(spec_id)]
    degrees, checks, example = periods - 1, {}, []
    worst = {key: 0.0 for key in ["coefficient", "base_plus_delta", "variance_identity_rel", "variance_vs_mine_rel", "std_error_rel", "p_value", "ci", "holm", "correction"]}
    for bandwidth in [12, 24]:
        cov = mine[bandwidth]
        for effect in ["lag", "cumulative_direct"]:
            part = table.loc[table.hac_lags.eq(bandwidth) & table.effect.eq(effect) & table.covariance.eq("Calendar HAC")].set_index("unit_id").loc[units]
            assert len(part) == len(units)
            p_mine = []
            for number, u in enumerate(units):
                item, vector = part.loc[u], totals[number]
                estimate, variance = float(vector @ beta_selected), float(vector @ cov @ vector)
                se = np.sqrt(variance); p = float(2 * stats.t.sf(abs(estimate / se), degrees)); crit = float(stats.t.ppf(0.975, degrees))
                p_mine.append(p)
                worst["coefficient"] = max(worst["coefficient"], abs(item.coefficient - estimate))
                worst["variance_vs_mine_rel"] = max(worst["variance_vs_mine_rel"], abs(item.variance - variance) / variance)
                worst["std_error_rel"] = max(worst["std_error_rel"], abs(item.std_error - se) / se)
                worst["p_value"] = max(worst["p_value"], abs(item.p_value - p))
                worst["ci"] = max(worst["ci"], abs(item.ci_lower - (estimate - crit * se)), abs(item.ci_upper - (estimate + crit * se)))
                worst["correction"] = max(worst["correction"], abs(item.finite_sample_correction - factor))
                assert int(item.inference_df) == degrees and int(item.multiplicity_family_size) == len(units)
                if effect == "lag":
                    worst["base_plus_delta"] = max(worst["base_plus_delta"], abs(item.temp_main + item.interaction_delta - item.coefficient))
                    identity = item.var_temp_main + item.var_interaction_delta + 2 * item.cov_temp_main_delta
                    worst["variance_identity_rel"] = max(worst["variance_identity_rel"], abs(identity - item.variance) / item.variance)
                    if bandwidth == 12:
                        example.append({"unit_id": u, "unit": item.unit, "total": item.coefficient, "se": item.std_error, "p_total": item.p_value, "p_holm": item.p_value_holm_profile,
                                        "base": item.temp_main, "delta": item.interaction_delta, "var_base": item.var_temp_main, "var_delta": item.var_interaction_delta,
                                        "cov_base_delta": item.cov_temp_main_delta, "naive_variance_without_cross_term": item.var_temp_main + item.var_interaction_delta})
            worst["holm"] = max(worst["holm"], float(np.max(np.abs(holm(p_mine) - part.p_value_holm_profile.to_numpy()))))
    out["ledger_total_effect_checks_max_abs"] = worst
    # Deviation (relative) rows: a different null from the total-effect rows.
    comp = components.loc[components.spec_id.eq(spec_id) & components.hac_lags.eq(12) & components.normalization.eq(False)].set_index("unit_id").loc[units]
    cov, relative = mine[12], []
    alt_reference = len(units) - 1
    total_cov = totals @ cov @ totals.T
    total_beta = totals @ beta_selected
    worst_component = 0.0
    for number, u in enumerate(units):
        item = comp.loc[u]
        vector = np.zeros(len(selected)); vector[temp_local[number]] = 1.0
        estimate, variance = float(vector @ beta_selected), float(vector @ cov @ vector)
        p = float(2 * stats.t.sf(abs(estimate) / np.sqrt(variance), degrees))
        worst_component = max(worst_component, abs(item.coefficient - estimate), abs(item.p_value - p))
        contrast = np.zeros(len(units)); contrast[number] = 1.0; contrast[alt_reference] -= 1.0
        alt_p = None
        if number != alt_reference:
            alt_p = float(2 * stats.t.sf(abs(contrast @ total_beta) / np.sqrt(contrast @ total_cov @ contrast), degrees))
        relative.append({"unit_id": u, "component": item.component, "p_total_zero": float(2 * stats.t.sf(abs(total_beta[number]) / np.sqrt(total_cov[number, number]), degrees)),
                         "p_vs_saved_reference": None if number == 0 else p, "p_vs_alternative_reference": alt_p})
    out["interaction_component_ledger_max_abs_diff"] = worst_component
    reject = lambda key: [item["unit_id"] for item in relative if item[key] is not None and item[key] < 0.05]
    out["different_nulls_hac12"] = {"saved_reference": units[0], "alternative_reference": units[alt_reference], "units_total_effect_p_lt_05": reject("p_total_zero"),
                                    "units_deviation_from_saved_reference_p_lt_05": reject("p_vs_saved_reference"),
                                    "units_deviation_from_alternative_reference_p_lt_05": reject("p_vs_alternative_reference")}
    # Joint tests recomputed; the equality statistic must not depend on the reference origin.
    joint = tests.loc[tests.spec_id.eq(spec_id) & tests.unit_id.eq("all")]
    joint_out = []
    for bandwidth in [12, 24]:
        cov_total = totals @ mine[bandwidth] @ totals.T
        for name, restriction in [("All origin cumulative direct effects equal", np.eye(len(units))[1:] - np.eye(len(units))[[0]]),
                                  ("All origin temperature response coefficients jointly zero", np.eye(len(units)))]:
            estimate, middle = restriction @ total_beta, restriction @ cov_total @ restriction.T
            chi2 = float(estimate @ np.linalg.solve(middle, estimate)); q = len(restriction)
            item = joint.loc[joint.test.eq(name) & joint.hac_lags.eq(bandwidth)].iloc[0]
            record = {"test": name, "hac_lags": bandwidth, "q": q, "denominator_df": degrees, "status_saved": item.status if "status" in item else None,
                      "chi2_mine": chi2, "chi2_saved": item.wald_chi2, "p_mine": float(stats.f.sf(chi2 / q, q, degrees)), "p_saved": item.p_value,
                      "fragile_saved": bool(item.joint_test_fragile), "fragility_reason": item.fragility_reason if isinstance(item.fragility_reason, str) else "",
                      "calibration_status": item.calibration_status, "covariance_rank_saved": int(item.covariance_rank), "bound_saved": int(item.structural_rank_bound)}
            if "equal" in name:
                other = np.delete(np.eye(len(units)), alt_reference, axis=0); other[:, alt_reference] -= 1.0
                alt = other @ total_beta
                record["chi2_alternative_reference"] = float(alt @ np.linalg.solve(other @ cov_total @ other.T, alt))
            joint_out.append(record)
    out["joint_tests"] = joint_out
    out["example_rows_hac12"] = example if len(units) <= 9 else example[:4]
    return out, {"units": units, "total": base + deltas, "controls": archive["beta"][kinds == "control"], "mean_slope": route["mean_slope"]}


def light(source, own, parent_specs, effects_parent, effects_own):
    """Checks applied to every one of the 24 origin-year requests."""
    parent_id, spec_id = source["source_spec_id"], source["spec_id"]
    parent_row = parent_specs.loc[parent_specs.spec_id.eq(parent_id)].iloc[0]
    parent, child = np.load(ROOT / parent_row["design_npz"]), np.load(ROOT / source["design_npz"])
    rows = read_rows(ROOT / source["source_sample_csv"])
    matrix, names, kinds, units, seasons, years, controls, unit = my_design(rows, source["control_set"], "origin_year")
    outcome = np.array([float(item[source["outcome"]]) for item in rows])
    parent_names, child_names = list(parent["names"]), list(child["names"])
    stochastic_parent = [name for name in parent_names if not name.startswith(("origin_season[", "year["))]
    stochastic_child = [name for name in child_names if not name.startswith(("origin_season[", "origin_year["))]
    seasonal = [name for name in parent_names if name.startswith("origin_season[")]
    year_names = [name for name in parent_names if name.startswith("year[")]
    px, cx = parent["X"], child["X"]
    nesting = 0.0
    for name in year_names:
        year = name[5:-1]
        nesting = max(nesting, float(np.max(np.abs(px[:, parent_names.index(name)] - sum(cx[:, child_names.index(f"origin_year[{u},{year}]")] for u in units)))))
    beta, resid = qr_fit(matrix, outcome)
    bridge = child["coefficient_transform"]
    native_design = child["native_input_design"]
    temp_parent = parent["beta"][parent_names.index("temp_l0")] + np.array([0.0] + [parent["beta"][parent_names.index(f"temp_l0:C(unit_id)[T.{u}]")] for u in units[1:]])
    temp_child = child["beta"][child_names.index("temp_l0")] + np.array([0.0] + [child["beta"][child_names.index(f"temp_l0:C(unit_id)[T.{u}]")] for u in units[1:]])
    common_id = parent_id.replace("__H1__", "__common__")
    common = np.load(MONTHLY / "designs" / f"{common_id}.npz")
    common_slope = float(common["beta"][list(common["names"]).index("temp_l0")])
    control_parent = np.array([parent["beta"][parent_names.index(name)] for name in controls])
    control_child = np.array([child["beta"][child_names.index(name)] for name in controls])
    control_common = np.array([common["beta"][list(common["names"]).index(name)] for name in controls])
    # Ledger-level descriptive change, primary HAC12 rows.
    key = lambda table, identifier: table.loc[table.spec_id.eq(identifier) & table.hac_lags.eq(12) & table.effect.eq("lag")].set_index("unit_id").loc[units]
    a, b = key(effects_parent, parent_id), key(effects_own, spec_id)
    change = b.coefficient.to_numpy() - a.coefficient.to_numpy()
    return {"spec_id": spec_id, "n_obs": len(rows), "n_units": len(units), "n_periods": int(source["n_periods"]),
            "k_parent": int(px.shape[1]), "k_origin_year": int(cx.shape[1]), "k_expected": int(px.shape[1] + (len(units) - 1) * (len(years) - 1)),
            "sample_sha_equal_parent_ledger": sha(ROOT / source["source_sample_csv"]) == parent_row["sample_sha256"] == source["source_sample_sha256"],
            "same_sample_file_as_parent": source["source_sample_csv"] == parent_row["sample_csv"],
            "y_bitwise": bool(np.array_equal(parent["Y"], child["Y"]) and np.array_equal(child["Y"], outcome)),
            "time_bitwise": bool(np.array_equal(parent["time"], child["time"])), "unit_bitwise": bool(np.array_equal(parent["unit_ids"], child["unit_ids"])),
            "periods_bitwise": bool(np.array_equal(parent["periods"], child["periods"])),
            "stochastic_names_equal": stochastic_parent == stochastic_child,
            "stochastic_columns_bitwise": bool(all(np.array_equal(px[:, parent_names.index(name)], cx[:, child_names.index(name)]) for name in stochastic_parent)),
            "origin_season_columns_bitwise": bool(all(np.array_equal(px[:, parent_names.index(name)], cx[:, child_names.index(name)]) for name in seasonal)),
            "parent_year_equals_sum_of_origin_year_max_abs": nesting,
            "rebuilt_names_equal": child_names == names, "rebuilt_design_max_abs_diff": maxabs(matrix, cx),
            "qr_beta_max_abs_diff": maxabs(beta, child["beta"]), "qr_resid_max_abs_diff": maxabs(resid, child["resid"]),
            "bridge_design_max_abs": float(np.max(np.abs(cx @ bridge - native_design))),
            "bridge_orthonormality_max_abs": float(np.max(np.abs(native_design.T @ native_design - np.eye(native_design.shape[1])))),
            "bridge_beta_max_abs": maxabs(bridge @ child["native_input_coefficients"], child["beta"]),
            "k_time_parent": int(parent_row["time_design_rank"]), "k_time_origin_year": int(source["time_design_rank"]),
            "corrections_equal": bool(np.array_equal(parent["corrections"], child["corrections"])), "inference_df_equal": bool(np.array_equal(parent["inference_df"], child["inference_df"])),
            "labels_origin_year": child["covariance_labels"].tolist(), "labels_parent": parent["covariance_labels"].tolist(),
            "mean_total_slope_parent": float(temp_parent.mean()), "mean_total_slope_origin_year": float(temp_child.mean()), "common_model_slope": common_slope,
            "mean_slope_identity_max_abs": float(max(abs(temp_parent.mean() - temp_child.mean()), abs(temp_parent.mean() - common_slope))),
            "control_slope_identity_max_abs": float(max(np.max(np.abs(control_parent - control_child)), np.max(np.abs(control_parent - control_common)))),
            "ledger_change": {"mean_change": float(change.mean()), "max_abs_change": float(np.max(np.abs(change))), "sd_parent_totals": float(a.coefficient.std()),
                              "sd_origin_year_totals": float(b.coefficient.std()), "correlation_of_deviations": float(np.corrcoef(a.coefficient - a.coefficient.mean(), b.coefficient - b.coefficient.mean())[0, 1]),
                              "sign_changes": int(np.sum(np.sign(a.coefficient.to_numpy()) != np.sign(b.coefficient.to_numpy()))),
                              "median_se_ratio": float(np.median(b.std_error.to_numpy() / a.std_error.to_numpy())),
                              "p_lt_05_parent": int((a.p_value < 0.05).sum()), "p_lt_05_origin_year": int((b.p_value < 0.05).sum()), "p_lt_05_both": int(((a.p_value < 0.05) & (b.p_value < 0.05)).sum()),
                              "holm_lt_05_parent": int((a.p_value_holm_profile < 0.05).sum()), "holm_lt_05_origin_year": int((b.p_value_holm_profile < 0.05).sum())}}


def main():
    started = clock.time()
    parent_specs, own_specs = ledger(MONTHLY / "specifications.csv"), ledger(OY / "specifications.csv")
    effects_parent, effects_own = ledger(MONTHLY / "temperature_effects.csv"), ledger(OY / "temperature_effects.csv")
    tests_parent, tests_own = ledger(MONTHLY / "tests.csv"), ledger(OY / "tests.csv")
    components_parent, components_own = ledger(MONTHLY / "interaction_components.csv"), ledger(OY / "interaction_components.csv")
    deep_ids = ["region__y_mom_nsa__long__all_eligible__H1__own_current", "region__y_yoy_nsa__full__event_dates_excluded__H1__own_current",
                "country__y_mom_nsa__long__all_eligible__H1__own_current"]
    deep_out = []
    for parent_id in deep_ids:
        first, saved_parent = deep(parent_id, parent_specs, effects_parent, tests_parent, components_parent, MONTHLY, "year")
        second, saved_child = deep(parent_id, own_specs, effects_own, tests_own, components_own, OY, "origin_year")
        pair = {"parent": first, "origin_year": second,
                "pair_mean_total_slope_difference": float(np.mean(saved_parent["total"]) - np.mean(saved_child["total"])),
                "pair_control_slope_max_abs_difference": maxabs(saved_parent["controls"], saved_child["controls"]),
                "pair_max_abs_total_slope_change": maxabs(saved_parent["total"], saved_child["total"])}
        deep_out.append(pair)
        print(parent_id, "deep done", round(clock.time() - started, 1), "s", flush=True)
    light_out = []
    for source in own_specs.to_dict("records"):
        light_out.append(light(source, own_specs, parent_specs, effects_parent, effects_own))
        print(source["spec_id"], "light done", round(clock.time() - started, 1), "s", flush=True)
    frame = pd.DataFrame(light_out)
    flags = ["sample_sha_equal_parent_ledger", "same_sample_file_as_parent", "y_bitwise", "time_bitwise", "unit_bitwise", "periods_bitwise", "stochastic_names_equal",
             "stochastic_columns_bitwise", "origin_season_columns_bitwise", "rebuilt_names_equal", "corrections_equal", "inference_df_equal"]
    summary = {"origin_year_requests": len(frame), **{f"all_{flag}": bool(frame[flag].all()) for flag in flags},
               "k_matches_expected": bool((frame.k_origin_year == frame.k_expected).all()),
               "k_time_equal_parent": bool((frame.k_time_parent == frame.k_time_origin_year).all()),
               "max_parent_year_minus_sum_origin_year": float(frame.parent_year_equals_sum_of_origin_year_max_abs.max()),
               "max_rebuilt_design_abs_diff": float(frame.rebuilt_design_max_abs_diff.max()),
               "max_qr_beta_abs_diff": float(frame.qr_beta_max_abs_diff.max()), "max_qr_resid_abs_diff": float(frame.qr_resid_max_abs_diff.max()),
               "max_bridge_design_abs": float(frame.bridge_design_max_abs.max()), "max_bridge_orthonormality_abs": float(frame.bridge_orthonormality_max_abs.max()),
               "max_bridge_beta_abs": float(frame.bridge_beta_max_abs.max()),
               "max_mean_slope_identity_abs": float(frame.mean_slope_identity_max_abs.max()), "max_control_slope_identity_abs": float(frame.control_slope_identity_max_abs.max()),
               "labels_origin_year": sorted({tuple(item) for item in frame.labels_origin_year}), "labels_parent": sorted({tuple(item) for item in frame.labels_parent}),
               "max_abs_mean_ledger_change": float(max(abs(item["mean_change"]) for item in frame.ledger_change))}
    payload = {"probe": "04 H1 and origin-year design reconstruction", "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe04_h1_origin_year_design.py",
               "summary_all_24": summary, "deep_pairs": deep_out, "all_24": light_out, "seconds": round(clock.time() - started, 1),
               "identity_statement": "Balanced panel + destination-only (common-in-origin) regressors: OLS splits into a cross-origin mean equation (seasons, years, shared controls, temperature) and per-origin deviation equations (origin seasons, [origin years], temperature only). Mean slope and shared control slopes are therefore identical in parent H1, origin-year H1 and the pooled common model; origin-year FE can only move the deviations."}
    write("probe04_result.json", payload)
    print(pd.Series(summary).to_string())
    for pair in deep_out:
        for side in ["parent", "origin_year"]:
            item = pair[side]
            keep = {key: item[key] for key in item if key not in ["example_rows_hac12", "joint_tests", "ledger_controls", "inference_df_saved", "corrections_saved", "covariance_labels", "bandwidths"]}
            print("\n", side, pd.Series(keep).to_string())
            for record in item["joint_tests"]:
                print("   ", {key: record[key] for key in ["test", "hac_lags", "q", "chi2_mine", "chi2_saved", "p_mine", "p_saved", "fragile_saved", "chi2_alternative_reference"] if key in record})
        print({key: pair[key] for key in pair if key.startswith("pair_")})
    print(pd.DataFrame([{"spec": item["spec_id"].replace("__H1__own_current__origin_year", ""), **item["ledger_change"], "mean_parent": item["mean_total_slope_parent"], "common": item["common_model_slope"]} for item in light_out]).to_string())


if __name__ == "__main__":
    main()
