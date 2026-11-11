"""Probe 05 - working time-rank convention and its reference law.

Part A (all 270 monthly + 24 origin-year archives, arithmetic only): recompute
every per-origin local design rank, K_time = max local rank and the full
cross-origin temporal-span rank from the saved X; check the saved correction
T/(T-K_time), reference df T-1, origin-cluster factor, and that native point
availability and temporal-correction availability are recorded separately.

Part B (13 selected designs): exact null law of the production calendar-HAC t
ratio under the most favourable textbook benchmark - saved design held fixed,
errors iid Gaussian. Then c'b is independent of the residuals and
    t^2 = z0^2 |m|^2 / (factor * sum_k lambda_k chi2_k),
with lambda_k the eigenvalues of M D W D M (M residual maker, D = diag(m),
m = pinv(X)'c, W = actual-calendar Bartlett weights). Craig's identity gives
    P(|t| > c) = (2/pi) int_0^{pi/2} prod_k (1 + c^2 factor lambda_k /(|m|^2 sin^2 phi))^(-1/2) dphi,
a smooth one-dimensional integral: no simulation is used. A 20,000-draw check
of one contrast only validates the formula. For lagged-growth designs the
"fixed design" is an algebraic benchmark, not a coherent dynamic DGP.

Part C: the aggregate YoY full event-date dynamic equation has one residual
degree of freedom (N=T=61, K=60), so every HAC matrix is a fixed matrix times
one chi2_1; exact null rejection rates of its saved joint tests follow from the
same identity.

Command: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe05_time_rank_reference_law.py
"""
from _probe_common import BASE, HERE, maxabs, scaled_rank, write

import time as clock

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import integrate, optimize, stats

MONTHLY, OY = BASE / "monthly", BASE / "origin_year"
STR = {"unit_id": str, "sector_id": str, "reference_unit_id": str, "reference_sector_id": str}


def ledger(path):
    return pd.read_csv(path, dtype=STR, float_precision="round_trip", low_memory=False)


def rank_parts(matrix, unit, time):
    ranks, blocks = {}, []
    for u in sorted(set(unit.tolist())):
        rows = np.flatnonzero(unit == u)
        rows = rows[np.argsort(time[rows], kind="stable")]
        local = matrix[rows]
        ranks[u] = scaled_rank(local)
        blocks.append(local[:, np.any(local != 0, axis=0)])
    return ranks, scaled_rank(np.column_stack(blocks))


def part_a():
    rows = []
    for family, folder in [("monthly", MONTHLY), ("origin_year", OY)]:
        specs = ledger(folder / "specifications.csv")
        for spec in specs.to_dict("records"):
            archive = np.load(folder / "designs" / f"{spec['spec_id']}.npz")
            matrix, unit, time = archive["X"], archive["unit_ids"].astype(str), archive["time"]
            n, k = matrix.shape
            t, g = len(set(time.tolist())), len(set(unit.tolist()))
            ranks, span = rank_parts(matrix, unit, time)
            k_time = max(ranks.values())
            labels, corrections, degrees = archive["covariance_labels"].tolist(), archive["corrections"], archive["inference_df"]
            calendar = [i for i, label in enumerate(labels) if label == "Calendar HAC"]
            cluster = [i for i, label in enumerate(labels) if label == "Origin cluster"]
            expected = t / (t - k_time) if t > k_time else np.nan
            correction_ok = all((np.isnan(expected) and np.isnan(corrections[i])) or np.isclose(corrections[i], expected, rtol=1e-13, atol=0) for i in calendar)
            cluster_ok = all(np.isclose(corrections[i], g / (g - 1) * (n - 1) / (n - k), rtol=1e-13, atol=0) if n > k else np.isnan(corrections[i]) for i in cluster)
            estimated = spec["status"] == "estimated"
            rows.append({"family": family, "spec_id": spec["spec_id"], "panel": spec["panel"], "model": spec["model"], "model_spec": spec["model_spec"],
                         "outcome": spec["outcome"], "control_set": spec["control_set"], "sample_variant": spec["sample_variant"],
                         "n": n, "k": k, "t": t, "g": g, "estimated": estimated, "design_rank_mine": scaled_rank(matrix) if not estimated else k,
                         "k_time_mine": k_time, "k_time_ledger": int(spec["time_design_rank"]), "local_rank_min": min(ranks.values()),
                         "span_mine": span, "span_ledger": int(spec["full_cross_origin_temporal_span_rank"]),
                         "time_df_ledger": int(spec["time_residual_df"]), "time_df_mine": t - k_time,
                         "correction_ok": bool(correction_ok), "cluster_factor_ok": bool(cluster_ok),
                         "calendar_df_ok": bool(all(int(degrees[i]) == t - 1 for i in calendar)), "cluster_df_ok": bool(all(int(degrees[i]) == g - 1 for i in cluster)),
                         "has_cluster": bool(cluster), "cluster_expected": bool(spec["model"] == "common" and g > 1),
                         "time_correction_available_ledger": bool(spec["time_correction_available"]),
                         "primary_available_ledger": bool(spec["primary_calendar_inference_available"]),
                         "primary_available_npz": bool(archive["primary_inference_available"]), "point_available_npz": bool(archive["point_fit_available"])})
    return pd.DataFrame(rows)


def craig(weights):
    """P(chi2_1 > sum_k weights_k chi2_1k) for independent chi-squares, weights >= 0."""
    weights = np.asarray(weights, float)
    weights = weights[weights > 0]
    if not len(weights):
        return 1.0
    function = lambda phi: float(np.exp(-0.5 * np.sum(np.log1p(weights / max(np.sin(phi) ** 2, 1e-300)))))
    return float(2 / np.pi * integrate.quad(function, 0, np.pi / 2, limit=400, epsabs=1e-13, epsrel=1e-11)[0])


def bartlett_matrix(span, bandwidth):
    lag = np.abs(np.subtract.outer(np.arange(span), np.arange(span)))
    return np.clip(1.0 - lag / (bandwidth + 1.0), 0.0, None)


class Design:
    def __init__(self, folder, spec_id, spec):
        archive = np.load(folder / "designs" / f"{spec_id}.npz")
        self.spec_id, self.spec = spec_id, spec
        self.x, self.y, self.resid, self.beta = archive["X"], archive["Y"], archive["resid"], archive["beta"]
        self.names, self.time, self.unit = list(archive["names"]), archive["time"], archive["unit_ids"].astype(str)
        self.raw = {12: archive["raw_calendar_covariance_0"], 24: archive["raw_calendar_covariance_1"]}
        self.corrections = archive["corrections"]
        self.scale = np.linalg.norm(self.x, axis=0)
        self.q, self.r = np.linalg.qr(self.x / self.scale, mode="reduced")
        self.bins = (self.time - self.time.min()).astype(int)
        self.span = int(self.bins.max()) + 1
        self.n, self.k = self.x.shape
        self.t = len(set(self.time.tolist()))
        self.units = sorted(set(self.unit.tolist()))
        self.roots = {}
        for bandwidth in [12, 24]:
            values, vectors = np.linalg.eigh(bartlett_matrix(self.span, bandwidth))
            assert values.min() > 0
            self.roots[bandwidth] = vectors * np.sqrt(values)
        self.lags = sorted(int(name[6:]) for name in self.names if name.startswith("temp_l") and ":" not in name)

    def contrast(self, unit=None, lags=None, average=False):
        vector = np.zeros(self.k)
        for lag in (self.lags if lags is None else lags):
            vector[self.names.index(f"temp_l{lag}")] = 1.0
            if average:
                for u in self.units[1:]:
                    vector[self.names.index(f"temp_l{lag}:C(unit_id)[T.{u}]")] = 1.0 / len(self.units)
            elif unit is not None and unit != self.units[0]:
                vector[self.names.index(f"temp_l{lag}:C(unit_id)[T.{unit}]")] = 1.0
        return vector

    def law(self, vector, bandwidth, factor, degrees):
        m = self.q @ scipy.linalg.solve_triangular(self.r, vector / self.scale, trans="T")
        mm = float(m @ m)
        spread = np.zeros((self.n, self.span))
        spread[np.arange(self.n), self.bins] = m
        projected = spread - self.q @ (self.q.T @ spread)
        root = self.roots[bandwidth]
        eigen = np.clip(np.linalg.eigvalsh(root.T @ (projected.T @ projected) @ root), 0.0, None)
        eigen = eigen[eigen > eigen.max() * 1e-13]
        sums = np.zeros(self.span)
        np.add.at(sums, self.bins, m * self.resid)
        kernel = bartlett_matrix(self.span, bandwidth)
        raw = float(sums @ kernel @ sums)
        saved_raw = float(vector @ self.raw[bandwidth] @ vector)
        estimate = float(vector @ self.beta)
        t_observed = estimate / np.sqrt(raw * factor)
        critical = float(stats.t.ppf(0.975, degrees))
        scaled = factor * eigen / mm
        size = craig(critical ** 2 * scaled)
        exact_critical = float(optimize.brentq(lambda value: craig(value ** 2 * scaled) - 0.05, 1e-3, 1e7, xtol=1e-10, rtol=1e-12))
        return {"estimate": estimate, "raw_variance_rel_diff_vs_saved": abs(raw - saved_raw) / abs(saved_raw), "t_observed": float(t_observed),
                "p_nominal_t": float(2 * stats.t.sf(abs(t_observed), degrees)), "p_exact_benchmark": craig(t_observed ** 2 * scaled),
                "nominal_critical": critical, "exact_null_rejection_at_nominal_5pct": size, "exact_5pct_critical_abs_t": exact_critical,
                "expected_variance_ratio": float(scaled.sum()), "satterthwaite_df": float(eigen.sum() ** 2 / (eigen ** 2).sum()),
                "nonzero_eigenvalues": int(len(eigen)), "m_norm2": mm, "scaled_eigenvalues": scaled}


SELECTED = [
    ("aggregate__y_yoy_nsa__full__event_dates_excluded__common__dynamic", "timeDF 1, single series"),
    ("aggregate__y_yoy_nsa__full__all_eligible__common__dynamic", "timeDF 3, single series"),
    ("region__y_yoy_nsa__full__event_dates_excluded__H1__dynamic", "timeDF 1, H1 full controls"),
    ("region__y_yoy_nsa__full__all_eligible__H1__dynamic", "timeDF 3, H1 full controls"),
    ("country__y_yoy_nsa__full__event_dates_excluded__H1__dynamic", "timeDF 1, H1 full controls, 42 markets"),
    ("region__y_yoy_nsa__full__event_dates_excluded__common__dynamic", "timeDF 1, pooled slope"),
    ("aggregate__y_yoy_nsa__full__event_dates_excluded__common__distributed", "timeDF 13, single series"),
    ("aggregate__y_yoy_nsa__long__all_eligible__common__dynamic", "long YoY dynamic, span = T"),
    ("region__y_mom_nsa__long__all_eligible__H1__dynamic", "long MoM dynamic, span > K_time"),
    ("country__y_mom_nsa__long__all_eligible__H1__dynamic", "long MoM dynamic, 42 markets"),
    ("aggregate__y_mom_nsa__long__all_eligible__common__own_current", "benign comparator, single series"),
    ("region__y_mom_nsa__long__all_eligible__H1__own_current", "benign comparator, H1"),
    ("country__y_mom_nsa__long__all_eligible__H1__own_current", "benign comparator, H1 42 markets"),
]


def part_b(specs, effects, ranks):
    out, validation = [], None
    for spec_id, why in SELECTED:
        spec = specs.loc[specs.spec_id.eq(spec_id)].iloc[0]
        design = Design(MONTHLY, spec_id, spec)
        row = ranks.loc[ranks.spec_id.eq(spec_id)].iloc[0]
        factor, degrees = design.t / (design.t - row.k_time_mine), design.t - 1
        table = effects.loc[effects.spec_id.eq(spec_id) & effects.covariance.eq("Calendar HAC")]
        record = {"spec_id": spec_id, "why": why, "n": design.n, "k": design.k, "t": design.t, "g": len(design.units), "residual_df": design.n - design.k,
                  "k_time": int(row.k_time_mine), "time_df": int(row.time_df_mine), "full_span": int(row.span_mine), "calendar_span_months": design.span,
                  "correction": factor, "reference_df": degrees, "lags": len(design.lags)}
        contrasts = []
        if spec["model"] == "H1":
            for u in design.units:
                contrasts.append((f"cumulative total, origin {u}", design.contrast(unit=u), u, "cumulative_direct", -1))
            contrasts.append(("cumulative cross-origin mean slope", design.contrast(average=True), None, None, None))
        else:
            unit = design.units[0] if len(design.units) == 1 else "pooled"
            contrasts.append(("cumulative", design.contrast(), unit, "cumulative_direct", -1))
            if len(design.lags) > 1:
                for lag in design.lags:
                    contrasts.append((f"lag {lag}", design.contrast(lags=[lag]), unit, "lag", lag))
        results = []
        for label, vector, unit, effect, lag in contrasts:
            for bandwidth in [12, 24]:
                if spec["model"] == "H1" and bandwidth == 24 and label.startswith("cumulative total") and unit != design.units[0]:
                    continue
                item = design.law(vector, bandwidth, factor, degrees)
                item.update(contrast=label, hac_lags=bandwidth)
                if unit is not None:
                    saved = table.loc[table.unit_id.eq(unit) & table.effect.eq(effect) & table.lag.eq(lag) & table.hac_lags.eq(bandwidth)]
                    assert len(saved) == 1, (spec_id, unit, effect, lag, len(saved))
                    saved = saved.iloc[0]
                    item.update(ledger_p=float(saved.p_value), ledger_p_abs_diff=abs(float(saved.p_value) - item["p_nominal_t"]),
                                ledger_holm=float(saved.p_value_holm_profile), ledger_inference_status=saved.inference_status,
                                ledger_calibration_status=saved.calibration_status, ledger_correction_abs_diff=abs(float(saved.finite_sample_correction) - factor))
                results.append(item)
        if spec_id == "region__y_yoy_nsa__full__event_dates_excluded__H1__dynamic":
            # Formula validation only: 20,000 iid Gaussian draws on this one saved design and one contrast.
            vector = design.contrast(unit=design.units[0])
            m = design.q @ scipy.linalg.solve_triangular(design.r, vector / design.scale, trans="T")
            generator, kernel, hits, draws = np.random.default_rng(20261111), bartlett_matrix(design.span, 12), 0, 20000
            critical = stats.t.ppf(0.975, degrees)
            for start in range(0, draws, 2000):
                noise = generator.standard_normal((2000, design.n))
                residual = noise - (noise @ design.q) @ design.q.T
                sums = np.zeros((2000, design.span))
                np.add.at(sums, (slice(None), design.bins), residual * m)
                variance = np.einsum("ij,jk,ik->i", sums, kernel, sums) * factor
                hits += int(np.sum(np.abs(noise @ m) / np.sqrt(variance) > critical))
            exact = [item for item in results if item["contrast"].endswith(design.units[0]) and item["hac_lags"] == 12][0]["exact_null_rejection_at_nominal_5pct"]
            validation = {"spec_id": spec_id, "contrast": "cumulative total, reference origin, HAC12", "draws": draws, "simulated_rejection": hits / draws,
                          "exact_formula": exact, "binomial_se": float(np.sqrt(exact * (1 - exact) / draws))}
        frame = pd.DataFrame([{key: value for key, value in item.items() if key != "scaled_eigenvalues"} for item in results])
        primary = frame.loc[frame.hac_lags.eq(12)]
        record.update(max_raw_variance_rel_diff_vs_saved=float(frame.raw_variance_rel_diff_vs_saved.max()),
                      max_ledger_p_abs_diff=float(frame.ledger_p_abs_diff.max()) if "ledger_p_abs_diff" in frame else None,
                      hac12_exact_null_rejection_min_median_max=[float(primary.exact_null_rejection_at_nominal_5pct.min()), float(primary.exact_null_rejection_at_nominal_5pct.median()), float(primary.exact_null_rejection_at_nominal_5pct.max())],
                      hac12_exact_5pct_critical_min_median_max=[float(primary.exact_5pct_critical_abs_t.min()), float(primary.exact_5pct_critical_abs_t.median()), float(primary.exact_5pct_critical_abs_t.max())],
                      hac12_expected_variance_ratio_min_median_max=[float(primary.expected_variance_ratio.min()), float(primary.expected_variance_ratio.median()), float(primary.expected_variance_ratio.max())],
                      hac12_satterthwaite_df_min_median_max=[float(primary.satterthwaite_df.min()), float(primary.satterthwaite_df.median()), float(primary.satterthwaite_df.max())],
                      hac24_exact_null_rejection_min_median_max=[float(frame.loc[frame.hac_lags.eq(24)].exact_null_rejection_at_nominal_5pct.min()), float(frame.loc[frame.hac_lags.eq(24)].exact_null_rejection_at_nominal_5pct.median()), float(frame.loc[frame.hac_lags.eq(24)].exact_null_rejection_at_nominal_5pct.max())],
                      nominal_critical=float(frame.nominal_critical.iloc[0]),
                      ledger_rows_p_lt_05=int((primary.ledger_p < 0.05).sum()) if "ledger_p" in primary else None,
                      benchmark_rows_p_lt_05=int((primary.loc[primary.ledger_p.notna(), "p_exact_benchmark"] < 0.05).sum()) if "ledger_p" in primary else None,
                      ledger_rows_compared=int(primary.ledger_p.notna().sum()) if "ledger_p" in primary else 0)
        keep = ["contrast", "hac_lags", "estimate", "t_observed", "p_nominal_t", "ledger_p", "p_exact_benchmark", "exact_null_rejection_at_nominal_5pct",
                "exact_5pct_critical_abs_t", "expected_variance_ratio", "satterthwaite_df", "nonzero_eigenvalues"]
        ordered = frame.sort_values("p_nominal_t")
        record["rows"] = frame[[key for key in keep if key in frame]].to_dict("records") if len(frame) <= 30 else ordered[[key for key in keep if key in frame]].head(8).to_dict("records")
        out.append(record)
        print(spec_id, "done", flush=True)
    return out, validation


def part_c(specs, tests):
    spec_id = "aggregate__y_yoy_nsa__full__event_dates_excluded__common__dynamic"
    archive = np.load(MONTHLY / "designs" / f"{spec_id}.npz")
    x, resid, beta, names, time = archive["X"], archive["resid"], archive["beta"], list(archive["names"]), archive["time"]
    n, k = x.shape
    assert n - k == 1
    pinv = np.linalg.pinv(x)
    u = scipy.linalg.null_space(x.T)[:, 0]
    lags = sorted(int(name[6:]) for name in names if name.startswith("temp_l"))
    rows = [names.index(f"temp_l{lag}") for lag in lags]
    p_sel = pinv[rows]
    bins = (time - time.min()).astype(int)
    span = int(bins.max()) + 1
    t = len(set(time.tolist()))
    k_time = scaled_rank(x)
    factor, degrees = t / (t - k_time), t - 1
    length, eye = len(lags), np.eye(len(lags))
    endpoint = length - 1
    requested = [("All temperature-lag coefficients zero", eye), ("Cumulative direct temperature response zero", np.ones((1, length))),
                 ("Conditional direct-weather level-channel shape; not exact structural static level with AR", np.vstack([eye[j] for j in range(1, length) if j != endpoint] + [eye[0] + eye[endpoint]])),
                 ("Conditional direct-weather equal-growth coefficient shape; growth lags held fixed", np.vstack([eye[j] - eye[0] for j in range(1, length - 1)] + [eye[-1]])),
                 ("Conditional direct-weather response lies in endpoint-difference plus growth-sum span; growth lags held fixed", np.vstack([eye[j] - eye[1] for j in range(2, length - 1)] + [eye[0] + eye[-1] - eye[1]]))]
    table = tests.loc[tests.spec_id.eq(spec_id) & tests.covariance.eq("Calendar HAC")]
    out = {"spec_id": spec_id, "n": n, "k": k, "residual_df": n - k, "t": t, "k_time": k_time, "correction": factor, "reference_df": degrees,
           "residual_is_rank_one_max_abs": maxabs(resid, u * (u @ archive["Y"])), "tests": []}
    for bandwidth, index in [(12, 0), (24, 1)]:
        kernel = bartlett_matrix(span, bandwidth)
        base = np.zeros((span, length))
        base[bins] = (p_sel * u).T
        fixed = base.T @ kernel @ base
        saved = archive[f"covariance_{index}"][np.ix_(rows, rows)]
        out[f"hac{bandwidth}_fixed_matrix_times_chi_rel_diff"] = float(np.max(np.abs(fixed * (u @ archive["Y"]) ** 2 * factor - saved)) / np.max(np.abs(saved)))
        for name, restriction in requested:
            q = len(restriction)
            gram, middle = restriction @ p_sel @ p_sel.T @ restriction.T, restriction @ fixed @ restriction.T
            ratios = np.clip(scipy.linalg.eigh(gram, middle, eigvals_only=True), 0.0, None)
            critical = float(stats.f.ppf(0.95, q, degrees))
            size = 1.0 - craig(ratios / (q * factor * critical))
            item = table.loc[table.test.eq(name) & table.hac_lags.eq(bandwidth)].iloc[0]
            estimate = restriction @ beta[rows]
            statistic = float(estimate @ np.linalg.solve(restriction @ saved @ restriction.T, estimate))
            out["tests"].append({"test": name, "hac_lags": bandwidth, "q": q, "status_saved": item.status, "fragile_saved": bool(item.joint_test_fragile),
                                 "fragility_reason": item.fragility_reason if isinstance(item.fragility_reason, str) else "", "calibration_status": item.calibration_status,
                                 "wald_mine": statistic, "wald_saved": None if pd.isna(item.wald_chi2) else float(item.wald_chi2), "p_saved": None if pd.isna(item.p_value) else float(item.p_value),
                                 "p_exact_benchmark": 1.0 - craig(ratios / (factor * statistic)), "exact_null_rejection_at_nominal_5pct": size})
    return out


def main():
    started = clock.time()
    ranks = part_a()
    monthly, own = ranks.loc[ranks.family.eq("monthly")], ranks.loc[ranks.family.eq("origin_year")]
    estimated = monthly.loc[monthly.estimated]
    summary_a = {"archives": len(ranks), "monthly": len(monthly), "origin_year": len(own),
                 "k_time_mismatches": int((ranks.k_time_mine != ranks.k_time_ledger).sum()), "span_mismatches": int((ranks.span_mine != ranks.span_ledger).sum()),
                 "time_df_mismatches": int((ranks.time_df_mine != ranks.time_df_ledger).sum()), "correction_mismatches": int((~ranks.correction_ok).sum()),
                 "cluster_factor_mismatches": int((~ranks.cluster_factor_ok).sum()), "calendar_df_mismatches": int((~ranks.calendar_df_ok).sum()),
                 "cluster_df_mismatches": int((~ranks.cluster_df_ok).sum()), "cluster_presence_mismatches": int((ranks.has_cluster != ranks.cluster_expected).sum()),
                 "local_ranks_differ_across_origins": int((ranks.local_rank_min != ranks.k_time_mine).sum()),
                 "monthly_estimated": int(monthly.estimated.sum()), "monthly_unestimated": int((~monthly.estimated).sum()),
                 "estimated_but_time_correction_unavailable": int((monthly.estimated & (monthly.time_df_mine <= 0)).sum()),
                 "unestimated_with_time_df_le_0": int((~monthly.estimated & (monthly.time_df_mine <= 0)).sum()),
                 "unestimated_rank_deficit": sorted({int(v) for v in (monthly.loc[~monthly.estimated, "k"] - monthly.loc[~monthly.estimated, "design_rank_mine"])}),
                 "availability_flag_mismatches": int(((monthly.time_correction_available_ledger != (monthly.time_df_mine > 0)) | (monthly.primary_available_ledger != (monthly.estimated & (monthly.time_df_mine > 0))) | (monthly.primary_available_npz != monthly.primary_available_ledger) | (monthly.point_available_npz != monthly.estimated)).sum()),
                 "estimated_time_df_distribution": {"le_3": int((estimated.time_df_mine <= 3).sum()), "4_to_9": int(((estimated.time_df_mine > 3) & (estimated.time_df_mine < 10)).sum()),
                                                    "10_to_29": int(((estimated.time_df_mine >= 10) & (estimated.time_df_mine < 30)).sum()), "ge_30": int((estimated.time_df_mine >= 30).sum())},
                 "estimated_specs_time_df_le_3": estimated.loc[estimated.time_df_mine <= 3, ["spec_id", "t", "k_time_mine", "time_df_mine", "span_mine", "n", "k"]].to_dict("records"),
                 "estimated_span_exceeds_k_time": int((estimated.span_mine > estimated.k_time_mine).sum()), "estimated_span_equals_t": int((estimated.span_mine == estimated.t).sum()),
                 "span_exceeds_k_time_by_model_spec": estimated.assign(wider=estimated.span_mine > estimated.k_time_mine).groupby("model_spec").wider.sum().astype(int).to_dict(),
                 "correction_range_estimated": [float((estimated.t / estimated.time_df_mine).min()), float((estimated.t / estimated.time_df_mine).max())]}
    print(pd.Series(summary_a).to_string(), flush=True)
    specs, effects, tests = ledger(MONTHLY / "specifications.csv"), ledger(MONTHLY / "temperature_effects.csv"), ledger(MONTHLY / "tests.csv")
    # Ledger exposure at very small working time df.
    small = estimated.loc[estimated.time_df_mine <= 3, "spec_id"]
    exposed = effects.loc[effects.spec_id.isin(small) & effects.covariance.eq("Calendar HAC") & effects.hac_lags.eq(12)]
    small_tests = tests.loc[tests.spec_id.isin(small) & tests.covariance.eq("Calendar HAC") & tests.hac_lags.eq(12)]
    exposure = {"specs": int(small.nunique()), "temperature_effect_rows_hac12": len(exposed), "rows_with_p": int(exposed.p_value.notna().sum()),
                "rows_p_lt_05": int((exposed.p_value < 0.05).sum()), "rows_holm_lt_05": int((exposed.p_value_holm_profile < 0.05).sum()),
                "inference_status_values": exposed.inference_status.value_counts().to_dict(), "calibration_status_values": exposed.calibration_status.value_counts().to_dict(),
                "joint_test_rows_hac12": len(small_tests), "joint_available": int(small_tests.status.eq("available").sum()), "joint_p_lt_05": int((small_tests.p_value < 0.05).sum()),
                "joint_flagged_fragile": int(small_tests.joint_test_fragile.sum()), "joint_fragility_reasons": small_tests.fragility_reason.value_counts().to_dict()}
    print(pd.Series(exposure).to_string(), flush=True)
    selected, validation = part_b(specs, effects, ranks)
    exact_joint = part_c(specs, tests)
    payload = {"probe": "05 working time-rank convention and reference law", "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe05_time_rank_reference_law.py",
               "benchmark_scope": "Exact law of the saved statistic with the saved design held fixed and iid Gaussian errors (favourable benchmark; lagged-growth columns are treated as fixed, which is an algebraic device rather than a dynamic DGP). It is not the law under serial or cross-origin dependence and is not a causal or calibration certificate.",
               "part_a_summary": summary_a, "small_time_df_ledger_exposure": exposure, "part_b_selected_designs": selected, "formula_validation": validation,
               "part_c_exact_joint_tests_one_residual_df": exact_joint, "seconds": round(clock.time() - started, 1)}
    write("probe05_result.json", payload)
    ranks.to_csv(HERE / "probe05_rank_table.csv", index=False)
    view = pd.DataFrame([{key: (np.round(value, 4).tolist() if isinstance(value, list) else value) for key, value in item.items() if key not in ["rows", "why"]} for item in selected])
    pd.set_option("display.width", 320, "display.max_columns", 40, "display.max_colwidth", 70)
    print(view.drop(columns=["spec_id"]).assign(spec=[item[0].replace("__y_", " ").replace("_nsa__", " ").replace("__", " ") for item in SELECTED]).to_string())
    print("validation", validation)
    for item in selected[:2]:
        print(item["spec_id"]); print(pd.DataFrame(item["rows"]).round(4).to_string())
    print(pd.DataFrame(exact_joint["tests"]).drop(columns=["calibration_status", "fragility_reason"]).round(5).to_string())
    print({key: value for key, value in exact_joint.items() if key != "tests"})


if __name__ == "__main__":
    main()
