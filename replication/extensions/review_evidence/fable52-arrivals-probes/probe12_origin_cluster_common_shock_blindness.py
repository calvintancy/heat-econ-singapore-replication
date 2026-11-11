"""Probe 12 - what the pooled origin-cluster (CR1) covariance can and cannot see.

Exact algebra on saved designs; no native estimation and no simulation of the data.

In a balanced panel whose regressors are destination-level series (weather, holiday) and whose
nuisance space is origin x season cells plus common year dummies (monthly) or origin intercepts
plus a common trend (annual), each row of pinv(X) for a common slope is the same date weight a_t
for every origin. Two consequences follow:

  (1) slope error    = sum_t a_t * (sum_i u_it)      -> only the cross-origin SUM of errors matters;
  (2) origin score_i = sum_t a_t * e_it = sum_t a_t * (e_it - ebar_t), because sum_t a_t ebar_t = 0,

so the CR1 meat sum_i score_i score_i' is a function of within-date deviations only. It is exactly
invariant to ANY common date shock c_t added to every origin, while the estimate moves one-for-one
with G * sum_t a_t c_t. Part A checks the identity on every saved common design that carries an
origin-cluster covariance; Part B demonstrates the invariance by exact refits (QR) of four selected
static designs under injected common date shocks; Part C tabulates the affected ledger rows.
Models with lagged-outcome regressors (dynamic) have origin-specific regressors: the identity is
then not exact and those designs are reported separately.
"""
import time as clock

from _probe_common import BASE, maxrel, write

import numpy as np
import pandas as pd
import scipy.linalg

MONTHLY, ANNUAL = BASE / "monthly", BASE / "annual"
COMMAND = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe12_origin_cluster_common_shock_blindness.py"


def bartlett_matrix(span, bandwidth):
    lag = np.abs(np.subtract.outer(np.arange(span), np.arange(span)))
    return np.clip(1.0 - lag / (bandwidth + 1.0), 0.0, None)


class Panel:
    def __init__(self, folder, spec_id, annual):
        archive = np.load(folder / "designs" / f"{spec_id}.npz")
        self.spec_id, self.annual, self.archive = spec_id, annual, archive
        self.labels = [str(value) for value in archive["covariance_labels"]]
        self.x, self.y, self.names = archive["X"], archive["Y"], [str(value) for value in archive["names"]]
        self.resid = archive["residual" if annual else "resid"] if ("residual" if annual else "resid") in archive.files else None
        self.time = archive["time"].astype(np.int64)
        self.units = archive["units" if annual else "unit_ids"].astype(str)
        self.n, self.k = self.x.shape
        self.unit_list, self.dates = sorted(set(self.units.tolist())), np.unique(self.time)
        self.g, self.t = len(self.unit_list), len(self.dates)
        self.order = np.lexsort((self.time, self.units))
        self.balanced = self.n == self.g * self.t and bool(np.all(self.time[self.order].reshape(self.g, self.t) == self.dates))
        self.static = not any(name.startswith("ar_l") for name in self.names)
        self.temperature = [position for position, name in enumerate(self.names) if name.startswith("temp_")]

    def factorize(self):
        self.scale = np.linalg.norm(self.x, axis=0)
        self.q, self.r = np.linalg.qr(self.x / self.scale, mode="reduced")
        w = scipy.linalg.solve_triangular(self.r, np.eye(self.k)[:, self.temperature], trans="T")
        self.p_sel = (self.q @ w) / self.scale[self.temperature]
        self.weights = self.p_sel[self.order].reshape(self.g, self.t, len(self.temperature))

    def grid(self, values):
        return values[..., self.order].reshape(values.shape[:-1] + (self.g, self.t))


def part_a_row(folder, spec, annual):
    panel = Panel(folder, spec.spec_id, annual)
    if "Origin cluster" not in panel.labels or panel.resid is None or not np.isfinite(panel.resid).all():
        return None
    index = panel.labels.index("Origin cluster")
    panel.factorize()
    archive, weights = panel.archive, panel.weights
    errors = panel.grid(panel.resid)
    scores = np.einsum("gtc,gt->gc", weights, errors)
    factor = float(archive["corrections"][index])
    saved = archive[f"covariance_{index}"][np.ix_(panel.temperature, panel.temperature)]
    deviations = errors - errors.mean(axis=0)
    scores_deviation = np.einsum("gtc,gt->gc", weights, deviations)
    correlation = np.corrcoef(errors)
    off = correlation[~np.eye(panel.g, dtype=bool)]
    calendar = archive["covariance_0"][np.ix_(panel.temperature, panel.temperature)]
    ratio = np.sqrt(np.diag(saved) / np.diag(calendar)) if np.isfinite(calendar).all() else np.full(len(panel.temperature), np.nan)
    return {"family": folder.name, "spec_id": spec.spec_id, "model_spec": str(spec.model_spec), "outcome": str(spec.outcome), "static": panel.static, "balanced": panel.balanced,
            "n": panel.n, "k": panel.k, "g": panel.g, "t": panel.t, "temperature_terms": len(panel.temperature),
            "date_weight_spread_across_origins_rel": float(np.max(np.abs(weights - weights.mean(axis=0))) / np.max(np.abs(weights))),
            "cluster_covariance_rebuilt_rel_diff_vs_archive": maxrel(factor * scores.T @ scores, saved),
            "scores_full_vs_within_date_deviation_rel_diff": float(np.max(np.abs(scores - scores_deviation)) / np.max(np.abs(scores))),
            "date_mean_residual_variance_over_independence_value": float(panel.g * errors.mean(axis=0).var() / errors.var(axis=1).mean()),
            "mean_pairwise_residual_correlation": float(off.mean()), "share_pairwise_correlations_positive": float((off > 0).mean()),
            "cluster_over_calendar_se_min": float(np.nanmin(ratio)), "cluster_over_calendar_se_median": float(np.nanmedian(ratio)), "cluster_over_calendar_se_max": float(np.nanmax(ratio))}


def part_b(folder, spec_id, annual, draws=200, seed=20261112):
    panel = Panel(folder, spec_id, annual)
    assert panel.static and panel.balanced
    panel.factorize()
    archive, weights = panel.archive, panel.weights
    index = panel.labels.index("Origin cluster")
    cluster_factor, calendar_factor, bandwidth = float(archive["corrections"][index]), float(archive["corrections"][0]), int(archive["bandwidths"][0])
    calendar_bins = (panel.dates - panel.dates.min()).astype(int)
    kernel = bartlett_matrix(int(calendar_bins.max()) + 1, bandwidth)[np.ix_(calendar_bins, calendar_bins)]
    date_index = np.searchsorted(panel.dates, panel.time)
    beta = archive["beta"][panel.temperature]
    saved_cluster = np.sqrt(np.diag(archive[f"covariance_{index}"])[panel.temperature])
    saved_calendar = np.sqrt(np.diag(archive["covariance_0"])[panel.temperature])
    errors = panel.grid(panel.resid)
    common_sd = float(errors.mean(axis=0).std())
    exposure = weights.sum(axis=0)                      # d beta / d c_t = sum_i a_it  (t x c)
    generator, out = np.random.default_rng(seed), []
    for multiple in [1.0, 10.0]:
        shock = generator.normal(0.0, common_sd * multiple, (draws, panel.t))
        outcome = panel.y[None, :] + shock[:, date_index]
        beta_star = outcome @ panel.p_sel
        residual = panel.grid(outcome - (outcome @ panel.q) @ panel.q.T)
        scores = np.einsum("gtc,dgt->dgc", weights, residual)
        cluster_se = np.sqrt(cluster_factor * np.einsum("dgc,dgc->dc", scores, scores))
        sums = np.einsum("gtc,dgt->dtc", weights, residual)
        calendar_se = np.sqrt(calendar_factor * np.einsum("dtc,ts,dsc->dc", sums, kernel, sums))
        out.append({"injected_common_date_shock_sd": common_sd * multiple, "multiple_of_saved_date_mean_residual_sd": multiple, "draws": draws,
                    "estimate_shift_matches_exposure_formula_max_abs": float(np.max(np.abs((beta_star - beta) - shock @ exposure))),
                    "sd_of_estimate_across_draws": (beta_star - beta).std(axis=0), "analytic_sd_of_estimate": common_sd * multiple * np.sqrt((exposure ** 2).sum(axis=0)),
                    "cluster_se_max_rel_change_vs_saved": float(np.max(np.abs(cluster_se / saved_cluster - 1.0))),
                    "calendar_se_over_saved_min_median_max": [float(np.min(calendar_se / saved_calendar)), float(np.median(calendar_se / saved_calendar)), float(np.max(calendar_se / saved_calendar))],
                    "share_of_draws_cluster_t_abs_gt_2": float(np.mean(np.abs(beta_star / cluster_se) > 2.0))})
    first = {"spec_id": spec_id, "n": panel.n, "k": panel.k, "g": panel.g, "t": panel.t, "terms": [panel.names[position] for position in panel.temperature],
             "saved_estimate": beta, "saved_origin_cluster_se": saved_cluster, "saved_calendar_hac_se": saved_calendar, "calendar_bandwidth": bandwidth,
             "saved_date_mean_residual_sd": common_sd, "perturbations": out}
    print(f"B  {spec_id}: G={panel.g} T={panel.t} terms={len(panel.temperature)}", flush=True)
    for item in out:
        print(f"     common shock sd x{item['multiple_of_saved_date_mean_residual_sd']:<4.0f}: estimate sd (first term) {item['sd_of_estimate_across_draws'][0]:.4f} (analytic {item['analytic_sd_of_estimate'][0]:.4f}),"
              f" saved cluster SE {saved_cluster[0]:.4f} changes by at most {item['cluster_se_max_rel_change_vs_saved']:.1e} (relative),"
              f" calendar SE/saved median {item['calendar_se_over_saved_min_median_max'][1]:.3f}, share |beta*/cluster SE|>2: {item['share_of_draws_cluster_t_abs_gt_2']:.3f}", flush=True)
    return first


def main():
    started = clock.time()
    rows = []
    monthly = pd.read_csv(MONTHLY / "specifications.csv", low_memory=False)
    monthly = monthly.loc[monthly.model.eq("common") & monthly.n_sectors.gt(1) & monthly.status.eq("estimated")]
    annual = pd.read_csv(ANNUAL / "specifications.csv", low_memory=False)
    annual = annual.loc[annual.model.eq("common") & annual.n_units.gt(1) & annual.status.eq("estimated")]
    for folder, table, flag in [(MONTHLY, monthly, False), (ANNUAL, annual, True)]:
        for spec in table.itertuples():
            row = part_a_row(folder, spec, flag)
            if row is not None:
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(BASE.parent.parent / "audit_probes" / "probe12_cluster_identity_table.csv", index=False)
    summary = {}
    for name, part in [("static_balanced_designs", table.loc[table.static & table.balanced]), ("dynamic_designs_with_lagged_outcome", table.loc[~table.static])]:
        summary[name] = {"designs": len(part), "by_family": part.family.value_counts().to_dict(), "all_balanced": bool(part.balanced.all()),
                         "max_date_weight_spread_across_origins_rel": float(part.date_weight_spread_across_origins_rel.max()),
                         "max_cluster_covariance_rebuilt_rel_diff_vs_archive": float(part.cluster_covariance_rebuilt_rel_diff_vs_archive.max()),
                         "max_scores_full_vs_within_date_deviation_rel_diff": float(part.scores_full_vs_within_date_deviation_rel_diff.max()),
                         "median_scores_full_vs_within_date_deviation_rel_diff": float(part.scores_full_vs_within_date_deviation_rel_diff.median()),
                         "date_mean_residual_variance_over_independence_value_min_median_max": [float(part.date_mean_residual_variance_over_independence_value.min()), float(part.date_mean_residual_variance_over_independence_value.median()), float(part.date_mean_residual_variance_over_independence_value.max())],
                         "mean_pairwise_residual_correlation_min_median_max": [float(part.mean_pairwise_residual_correlation.min()), float(part.mean_pairwise_residual_correlation.median()), float(part.mean_pairwise_residual_correlation.max())],
                         "cluster_over_calendar_se_median_min_median_max": [float(part.cluster_over_calendar_se_median.min()), float(part.cluster_over_calendar_se_median.median()), float(part.cluster_over_calendar_se_median.max())]}
        print("A ", name, summary[name], flush=True)
    by_panel = []
    for (family, g), part in table.groupby(["family", "g"]):
        by_panel.append({"family": family, "origins": int(g), "designs": len(part), "variance_ratio_median": float(part.date_mean_residual_variance_over_independence_value.median()),
                         "mean_pairwise_correlation_median": float(part.mean_pairwise_residual_correlation.median()), "cluster_over_calendar_se_median": float(part.cluster_over_calendar_se_median.median())})
        print("A ", by_panel[-1], flush=True)

    selected = [(MONTHLY, "country__y_yoy_nsa__long__event_dates_excluded__common__own_current", False),
                (MONTHLY, "region__y_mom_nsa__long__all_eligible__common__own_current", False),
                (MONTHLY, "country__y_yoy_nsa__full__all_eligible__common__distributed", False),
                (ANNUAL, "annual__region__long__all_eligible__common", True)]
    refits = [part_b(folder, spec_id, flag) for folder, spec_id, flag in selected]

    # C. Ledger rows that carry this covariance (plain tabulation).
    ledger = []
    for family, folder in [("monthly", MONTHLY), ("annual", ANNUAL)]:
        effects = pd.read_csv(folder / "temperature_effects.csv", low_memory=False)
        cluster = effects.loc[effects.covariance.eq("Origin cluster") & effects.p_value.notna()]
        ledger.append({"family": family, "origin_cluster_rows_with_p": len(cluster), "p_lt_05": int((cluster.p_value < 0.05).sum()), "p_lt_001": int((cluster.p_value < 0.001).sum()),
                       "main_inference_true": int(cluster.main_inference.astype(str).eq("True").sum()),
                       "calibration_status_values": cluster.calibration_status.value_counts().to_dict() if "calibration_status" in cluster else {},
                       "rows_in_dynamic_specs": int(cluster.model_spec.eq("dynamic").sum()) if "model_spec" in cluster else 0})
        print("C ", ledger[-1], flush=True)

    payload = {"probe": 12, "title": "Origin-cluster CR1 covariance is exactly blind to common date shocks in balanced static common models", "command": COMMAND,
               "identity": "score_i = sum_t a_t e_it = sum_t a_t (e_it - ebar_t); slope error = sum_t a_t sum_i u_it. Exact for balanced panels with destination-level regressors, "
                           "origin x season (or origin) intercepts and common time effects; not exact when lagged outcomes enter (dynamic).",
               "A_summary": summary, "A_by_panel": by_panel, "A_table_csv": "audit_probes/probe12_cluster_identity_table.csv",
               "B_injected_common_shock_refits": refits, "C_ledger_rows": ledger, "seconds": round(clock.time() - started, 1)}
    write("probe12_result.json", payload)
    print("wrote probe12_result.json in", payload["seconds"], "s")


if __name__ == "__main__":
    main()
