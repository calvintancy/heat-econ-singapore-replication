"""Probe 11 - selected fixed-design reference-law benchmarks for saved test statistics.

Deliberately NOT a calibration grid and NOT a size certificate for every arrival equation:
five saved static H1 designs for the cross-origin joint Wald tests (Part A) and three saved
single-series YoY designs for the calendar-HAC t ratio (Part B). Nothing is re-estimated
natively; saved X, residuals, coefficients, covariances and ledger rows are read only.

Part A  The selected designs contain no lagged outcome: every regressor is the destination
        weather history or a deterministic dummy, so conditioning on the saved design is
        coherent. Draw u ~ N(0, I) on the saved rows, beta_hat - beta = pinv(X) u,
        e = (I - P_X) u, and rebuild the production statistic
            F = (R b)' [f * R V_calendarHAC(e) R']^{-1} (R b) / q
        with the restrictions of scripts/visitor_rigour_monthly.py:436-442 and
        scripts/visitor_rigour_companions.py:269-271, f = T/(T-K_time) as archived. The same
        code path first reproduces the ledger F from the saved residuals. iid Gaussian rows
        are the favourable case: no serial dependence, heteroskedasticity or crisis outliers.
Part B  Exact law (Craig identity, no simulation) of the calendar-HAC t ratio on aggregate YoY
        designs under (1) iid Gaussian errors and (2) the MA(11) overlap that a 12-month log
        difference carries when month-on-month errors are white noise. Both are reference DGPs
        chosen a priori from the outcome construction, not estimated truths; a Monte Carlo
        check validates the formula on one contrast. Saved-residual autocorrelations are
        reported descriptively next to the MA(11) pattern.
"""
import time as clock

from _probe_common import BASE, maxabs, maxrel, write

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import integrate, optimize, stats

MONTHLY, ANNUAL, ORIGIN_YEAR = BASE / "monthly", BASE / "annual", BASE / "origin_year"
COMMAND = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe11_selected_reference_law_benchmarks.py"
EQUAL, ZERO = "All origin cumulative direct effects equal", "All origin temperature response coefficients jointly zero"


def bartlett_matrix(span, bandwidth):
    lag = np.abs(np.subtract.outer(np.arange(span), np.arange(span)))
    return np.clip(1.0 - lag / (bandwidth + 1.0), 0.0, None)


def craig(weights):
    """P(chi2_1 > sum_k weights_k chi2_1k) for independent chi-squares, weights >= 0."""
    weights = np.asarray(weights, float)
    weights = weights[weights > 0]
    if not len(weights):
        return 1.0
    function = lambda phi: float(np.exp(-0.5 * np.sum(np.log1p(weights / max(np.sin(phi) ** 2, 1e-300)))))
    return float(2 / np.pi * integrate.quad(function, 0, np.pi / 2, limit=400, epsabs=1e-13, epsrel=1e-11)[0])


def monthly_restrictions(archive, k):
    """Mirror of visitor_rigour_monthly.py:386-397 and :436-442 in full coefficient space."""
    positions, features, lags = archive["component_positions"], list(archive["temperature_features"]), archive["lags"]
    g = positions.shape[1]
    two = features == ["temp_endpoint_difference", "temp_growth_sum"]
    profiles, basis_profiles = [], []
    for number in range(g):
        basis = np.zeros((len(features), k))
        for feature_number, indexes in enumerate(positions):
            basis[feature_number, indexes[0]] = 1.0
            if number:
                basis[feature_number, indexes[number]] = 1.0
        basis_profiles.append(basis)
        contrast = basis
        if two:
            horizon = len(lags) - 1
            assert horizon == 12
            expansion = np.zeros((horizon + 1, 2))
            expansion[0, 0], expansion[-1, 0] = 1.0, -1.0
            expansion[:-1, 1] = 1.0
            contrast = expansion @ basis
        profiles.append(contrast)
    cumulative = np.vstack([profile.sum(axis=0) for profile in profiles])
    difference = np.eye(g)[1:] - np.eye(g)[[0]]
    return [(EQUAL, difference @ cumulative), (ZERO, np.vstack(basis_profiles if two else profiles))]


def annual_restrictions(archive, k):
    """Mirror of visitor_rigour_companions.py:253-255 and :269-271."""
    positions = archive["component_positions"]
    profile = np.zeros((len(positions), k))
    profile[:, positions[0]] = 1.0
    for number in range(1, len(positions)):
        profile[number, positions[number]] = 1.0
    return [("All origin temperature slopes equal", profile[1:] - profile[[0]]), ("All origin temperature slopes zero", profile)]


def joint_benchmark(label, folder, spec_id, kind, draws, seed, overlap=False, batch=200):
    started = clock.time()
    archive = np.load(folder / "designs" / f"{spec_id}.npz")
    annual = kind == "annual"
    x, names = archive["X"], list(archive["names"])
    time = archive["time"].astype(np.int64)
    beta, resid = archive["beta"], archive["residual" if annual else "resid"]
    unit_codes = pd.Categorical(archive["units" if annual else "unit_ids"].astype(str)).codes
    n, k = x.shape
    tests = annual_restrictions(archive, k) if annual else monthly_restrictions(archive, k)
    assert list(archive["covariance_labels"][:2]) == ["Calendar HAC", "Calendar HAC"]
    bandwidths = [int(value) for value in archive["bandwidths"][:2]]
    corrections = [float(value) for value in archive["corrections"][:2]]
    order = np.argsort(time, kind="stable")
    sorted_time = time[order]
    starts = np.flatnonzero(np.r_[True, np.diff(sorted_time) > 0])
    dates = sorted_time[starts]
    t = len(dates)
    calendar = (dates - dates.min()).astype(int)
    span = int(calendar.max()) + 1
    kernels = {bandwidth: bartlett_matrix(span, bandwidth)[np.ix_(calendar, calendar)] for bandwidth in bandwidths}
    scale = np.linalg.norm(x, axis=0)
    q_, r_ = np.linalg.qr(x / scale, mode="reduced")
    used = np.flatnonzero(np.any(np.vstack([restriction for _, restriction in tests]) != 0, axis=0))
    w = scipy.linalg.solve_triangular(r_, np.eye(k)[:, used], trans="T")
    p_sel = (q_ @ w) / scale[used]
    p_sorted = np.ascontiguousarray(p_sel[order])
    reduced = [(name, np.ascontiguousarray(restriction[:, used])) for name, restriction in tests]
    columns = [(index, bandwidth, name, restriction) for index, bandwidth in enumerate(bandwidths) for name, restriction in reduced]

    def statistics(error_sorted, coefficient_error):
        sums = np.add.reduceat(p_sorted * error_sorted[:, None], starts, axis=0)
        cores = [sums.T @ (kernels[bandwidth] @ sums) for bandwidth in bandwidths]
        out = []
        for index, _, _, restriction in columns:
            covariance = corrections[index] * (restriction @ cores[index] @ restriction.T)
            estimate = restriction @ coefficient_error
            out.append(float(estimate @ np.linalg.solve(covariance, estimate)) / len(restriction))
        return out, cores

    saved_statistics, saved_cores = statistics(resid[order], beta[used])
    covariance_check = max(maxrel(corrections[index] * saved_cores[index], archive[f"covariance_{index}"][np.ix_(used, used)]) for index in range(2))
    ledger = pd.read_csv(folder / "tests.csv", low_memory=False)
    ledger = ledger.loc[ledger.spec_id.eq(spec_id) & ledger.covariance.eq("Calendar HAC")]
    if not annual:
        ledger = ledger.loc[ledger.unit_id.astype(str).eq("all")]

    # Reference error laws. "iid": independent N(0,1) rows. "ma11_overlap" (YoY designs only): each origin's error is
    # the 12-month moving sum of its own white-noise monthly shocks on the actual calendar, independent across origins.
    g, row_bins = int(unit_codes.max()) + 1, (time - time.min()).astype(int)
    generator = np.random.default_rng(seed)
    laws = ["iid"] + (["ma11_overlap"] if overlap else [])
    values = {law: np.empty((draws, len(columns))) for law in laws}
    for law in laws:
        done = 0
        while done < draws:
            size = min(batch, draws - done)
            if law == "iid":
                noise = generator.standard_normal((size, n))
            else:
                cumulative = np.cumsum(generator.standard_normal((size, g, span + 11)), axis=2)
                moving = cumulative[:, :, 11:] - np.concatenate([np.zeros((size, g, 1)), cumulative[:, :, :span - 1]], axis=2)
                noise = moving[:, unit_codes, row_bins]
            coefficient = noise @ p_sel
            error = (noise - (noise @ q_) @ q_.T)[:, order]
            for row in range(size):
                values[law][done + row] = statistics(error[row], coefficient[row])[0]
            done += size

    degrees, results = t - 1, []
    for column, (index, bandwidth, name, restriction) in enumerate(columns):
        q = len(restriction)
        row = ledger.loc[ledger.test.eq(name) & ledger.hac_lags.eq(bandwidth)]
        assert len(row) == 1, (spec_id, name, bandwidth, len(row))
        row = row.iloc[0]
        assert int(row.numerator_df) == q and int(row.denominator_df) == degrees
        saved_f = float(row.f_statistic)
        nominal5, nominal1 = float(stats.f.ppf(0.95, q, degrees)), float(stats.f.ppf(0.99, q, degrees))
        item = {"test": name, "hac_lags": bandwidth, "q": q, "reference": f"F({q},{degrees})",
                "saved_f": saved_f, "saved_nominal_p": float(row.p_value),
                "saved_fragile": bool(row["fragile" if annual else "joint_test_fragile"]), "saved_status": str(row.status),
                "saved_calibration_status": str(row.calibration_status),
                "f_rebuilt_from_saved_residual": saved_statistics[column], "f_rel_diff_vs_ledger": abs(saved_statistics[column] - saved_f) / abs(saved_f),
                "nominal_5pct_critical_f": nominal5, "nominal_1pct_critical_f": nominal1, "null_laws": {}}
        for law in laws:
            f = values[law][:, column]
            rejection, exceed = float(np.mean(f > nominal5)), int(np.sum(f >= saved_f))
            item["null_laws"][law] = {"null_rejection_at_nominal_5pct": rejection, "null_rejection_at_nominal_1pct": float(np.mean(f > nominal1)),
                                      "mc_se_of_5pct_rejection": float(np.sqrt(rejection * (1 - rejection) / draws)),
                                      "null_mean_f": float(f.mean()), "null_min_f": float(f.min()), "null_5pct_f": float(np.quantile(f, 0.05)),
                                      "null_median_f": float(np.median(f)), "null_95pct_f": float(np.quantile(f, 0.95)),
                                      "null_99pct_f": float(np.quantile(f, 0.99)), "null_max_f": float(f.max()),
                                      "null_draws_with_f_ge_saved": exceed, "null_share_f_ge_saved": exceed / draws,
                                      "benchmark_upper_p_of_saved_f": (exceed + 1) / (draws + 1), "saved_f_over_null_95pct_f": saved_f / float(np.quantile(f, 0.95))}
        results.append(item)
    out = {"label": label, "spec_id": spec_id, "family": folder.name, "n": n, "k": k, "t": t, "g": g,
           "calendar_span": span, "calendar_gaps": span - t, "residual_df": n - k, "correction": corrections[0], "reference_df": degrees, "draws_per_law": draws, "seed": seed,
           "has_lagged_outcome_regressor": bool(any(name.startswith("ar_l") for name in names)),
           "selected_coefficients": len(used), "beta_from_pinv_max_abs_diff": maxabs(p_sel.T @ archive["Y"], beta[used]),
           "calendar_covariance_rel_diff_vs_archive": covariance_check,
           "max_f_rel_diff_vs_ledger": max(item["f_rel_diff_vs_ledger"] for item in results), "tests": results, "seconds": round(clock.time() - started, 1)}
    print(f"A  {spec_id}: N={n} K={k} T={t} G={g} gaps={span - t} f={corrections[0]:.4f} | checks: beta {out['beta_from_pinv_max_abs_diff']:.1e}, "
          f"HAC {covariance_check:.1e}, ledger F {out['max_f_rel_diff_vs_ledger']:.1e} ({out['seconds']}s)", flush=True)
    for item in results:
        for law in laws:
            null = item["null_laws"][law]
            print(f"     {item['test'][:36]:36s} HAC{item['hac_lags']:<2d} q={item['q']:<3d} saved F={item['saved_f']:8.3f} nominal p={item['saved_nominal_p']:.1e} fragile={item['saved_fragile']!s:5s}"
                  f" | {law:12s} rej@5%={null['null_rejection_at_nominal_5pct']:.3f} @1%={null['null_rejection_at_nominal_1pct']:.3f} F 5%={null['null_5pct_f']:7.2f}"
                  f" med={null['null_median_f']:7.2f} 95%={null['null_95pct_f']:8.2f} max={null['null_max_f']:8.2f} share>=saved={null['null_share_f_ge_saved']:.4f}", flush=True)
    return out


def sign_probability(matrix):
    """P(w'Hw > 0), w ~ N(0, I), for H = rank-one PSD minus PSD (at most one positive eigenvalue)."""
    values = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    top = float(values[-1])
    if top <= 0:
        return 0.0
    assert values[-2] <= 1e-9 * top, (values[-2], top)
    rest = np.clip(-values[:-1], 0.0, None)
    return craig(rest[rest > top * 1e-14] / top)


def t_ratio_laws(spec_id, validate=None):
    archive = np.load(MONTHLY / "designs" / f"{spec_id}.npz")
    x, names, time, beta, resid = archive["X"], list(archive["names"]), archive["time"].astype(np.int64), archive["beta"], archive["resid"]
    n, k = x.shape
    assert len(set(time.tolist())) == n and not any(name.startswith("ar_l") for name in names)
    factor, degrees = float(archive["corrections"][0]), n - 1
    scale = np.linalg.norm(x, axis=0)
    q_, r_ = np.linalg.qr(x / scale, mode="reduced")
    bins = (time - time.min()).astype(int)
    span = int(bins.max()) + 1
    annihilator = np.eye(n) - q_ @ q_.T
    moving = np.zeros((n, span + 11))
    for row, position in enumerate(bins):
        moving[row, position:position + 12] = 1.0
    generators = {"iid Gaussian": np.eye(n), "MA(11): 12-month moving sum of white-noise monthly shocks on the actual calendar": moving}
    temperature = [name for name in names if name.startswith("temp_")]
    contrasts = [(name, np.eye(k)[names.index(name)]) for name in (temperature if len(temperature) <= 2 else temperature[:1])]
    if len(temperature) > 2:
        vector = np.zeros(k)
        vector[[names.index(name) for name in temperature]] = 1.0
        contrasts.append((f"cumulative direct sum of {len(temperature)} temperature lags", vector))
    critical = float(stats.t.ppf(0.975, degrees))
    rows, validation = [], None
    for label, vector in contrasts:
        m = q_ @ scipy.linalg.solve_triangular(r_, vector / scale, trans="T")
        estimate = float(vector @ beta)
        for index, bandwidth in enumerate([12, 24]):
            kernel = bartlett_matrix(span, bandwidth)
            spread = np.zeros((span, n))
            spread[bins, np.arange(n)] = m
            spread = spread @ annihilator
            quadratic = spread.T @ kernel @ spread
            saved_variance = float(vector @ archive[f"covariance_{index}"] @ vector)
            t_saved = estimate / np.sqrt(saved_variance)
            item = {"contrast": label, "hac_lags": bandwidth, "estimate": estimate, "saved_std_error": float(np.sqrt(saved_variance)), "t_saved": float(t_saved),
                    "p_nominal_t": float(2 * stats.t.sf(abs(t_saved), degrees)), "nominal_critical_abs_t": critical,
                    "variance_rebuilt_rel_diff_vs_archive": abs(factor * float(resid @ quadratic @ resid) - saved_variance) / saved_variance, "laws": {}}
            for name, generator in generators.items():
                gm, core = generator.T @ m, generator.T @ quadratic @ generator
                outer = np.outer(gm, gm)
                law = {"true_variance_of_estimator": float(gm @ gm), "expected_variance_estimate_over_truth": float(factor * np.trace(core) / (gm @ gm)),
                       "exact_null_rejection_at_nominal_5pct": sign_probability(outer - critical ** 2 * factor * core),
                       "exact_5pct_critical_abs_t": float(optimize.brentq(lambda value: sign_probability(outer - value ** 2 * factor * core) - 0.05, 0.3, 500.0, xtol=1e-9, rtol=1e-11)),
                       "benchmark_p_of_saved_t": sign_probability(outer - t_saved ** 2 * factor * core)}
                item["laws"]["iid" if name.startswith("iid") else "ma11_overlap"] = law
            rows.append(item)
            if validate == (label, bandwidth):
                draws, hits, generator = 40000, 0, np.random.default_rng(20261112)
                for start in range(0, draws, 4000):
                    error_full = generator.standard_normal((4000, span + 11)) @ moving.T
                    residual = error_full - (error_full @ q_) @ q_.T
                    sums = np.zeros((4000, span))
                    sums[:, bins] = residual * m
                    variance = factor * np.einsum("ij,jk,ik->i", sums, kernel, sums)
                    hits += int(np.sum(np.abs(error_full @ m) / np.sqrt(variance) > critical))
                exact = item["laws"]["ma11_overlap"]["exact_null_rejection_at_nominal_5pct"]
                validation = {"spec_id": spec_id, "contrast": label, "hac_lags": bandwidth, "dgp": "ma11_overlap", "draws": draws, "simulated_rejection": hits / draws,
                              "exact_formula": exact, "binomial_se": float(np.sqrt(exact * (1 - exact) / draws))}
    out = {"spec_id": spec_id, "n": n, "k": k, "t": n, "calendar_span": span, "calendar_gaps": span - n, "correction": factor, "reference_df": degrees, "rows": rows}
    print(f"B  {spec_id}: T={n} K={k} gaps={span - n} f={factor:.4f}", flush=True)
    for item in rows:
        iid, ma = item["laws"]["iid"], item["laws"]["ma11_overlap"]
        print(f"     {item['contrast'][:46]:46s} HAC{item['hac_lags']} t={item['t_saved']:7.3f} p={item['p_nominal_t']:.4f} | iid rej={iid['exact_null_rejection_at_nominal_5pct']:.4f} crit={iid['exact_5pct_critical_abs_t']:.3f}"
              f" E[V]/V={iid['expected_variance_estimate_over_truth']:.3f} | MA11 rej={ma['exact_null_rejection_at_nominal_5pct']:.4f} crit={ma['exact_5pct_critical_abs_t']:.3f}"
              f" E[V]/V={ma['expected_variance_estimate_over_truth']:.3f} p_bench={ma['benchmark_p_of_saved_t']:.4f}", flush=True)
    return out, validation


def residual_autocorrelation(folder, spec_id, lags=(1, 2, 3, 6, 9, 11, 12, 13, 18, 24)):
    """Calendar-aware autocorrelation of the saved date-mean residual; gaps are never closed.

    For single-series designs the same statistic is also evaluated on the population residual
    covariance (I-P)Sigma(I-P) implied by each reference law, so that saved residuals are compared
    with what the fixed effects and controls would leave of an iid or MA(11) error.
    """
    archive = np.load(folder / "designs" / f"{spec_id}.npz")
    time = archive["time"].astype(np.int64)
    series = pd.Series(archive["resid"]).groupby(time).mean()
    bins = (series.index.to_numpy() - series.index.min()).astype(int)
    span = int(bins.max()) + 1
    full = np.full(span, np.nan)
    full[bins] = series.to_numpy()
    scale = float(np.nanmean(full ** 2))
    out = {"spec_id": spec_id, "dates": int(len(series)), "calendar_gaps": int(span - len(series)), "series": "cross-origin date mean of saved residuals", "lags": {}}
    implied = {}
    if len(series) == len(time):
        x = archive["X"]
        q_ = np.linalg.qr(x / np.linalg.norm(x, axis=0), mode="reduced")[0]
        annihilator = np.eye(len(time)) - q_ @ q_.T
        moving = np.zeros((len(time), span + 11))
        for row, position in enumerate((time - time.min()).astype(int)):
            moving[row, position:position + 12] = 1.0
        position_of = np.full(span, -1)
        position_of[(time - time.min()).astype(int)] = np.arange(len(time))
        for law, sigma in [("iid", np.eye(len(time))), ("ma11_overlap", moving @ moving.T)]:
            covariance = annihilator @ sigma @ annihilator
            implied[law] = (covariance, float(np.mean(np.diag(covariance))), position_of)
    for lag in lags:
        products = full[lag:] * full[:-lag]
        valid = np.isfinite(products)
        item = {"rho_saved_residual": float(products[valid].mean() / scale), "pairs": int(valid.sum()), "ma11_overlap_before_projection": max(0.0, (12 - lag) / 12)}
        for law, (covariance, level, position_of) in implied.items():
            later, earlier = position_of[lag:], position_of[:-lag]
            keep = (later >= 0) & (earlier >= 0)
            item[f"rho_implied_after_projection_{law}"] = float(covariance[later[keep], earlier[keep]].mean() / level)
        out["lags"][str(lag)] = item
    print(f"C  {spec_id}:", flush=True)
    print("     saved residual      " + " ".join(f"r{lag}={out['lags'][str(lag)]['rho_saved_residual']:+.2f}" for lag in lags), flush=True)
    for law in implied:
        print(f"     implied {law:12s}" + " ".join(f"r{lag}={out['lags'][str(lag)][f'rho_implied_after_projection_{law}']:+.2f}" for lag in lags), flush=True)
    return out


def main():
    started = clock.time()
    designs = [
        ("annual region H1 (T=17, G=9)", ANNUAL, "annual__region__long__all_eligible__H1", "annual", 20000, 2026111101, False),
        ("monthly region MoM own-current H1 (G=9), low-dimensional comparator", MONTHLY, "region__y_mom_nsa__long__all_eligible__H1__own_current", "monthly", 4000, 2026111102, False),
        ("monthly country MoM own-current H1 (G=42)", MONTHLY, "country__y_mom_nsa__long__all_eligible__H1__own_current", "monthly", 4000, 2026111103, False),
        ("monthly country YoY two-channel H1 (G=42); smallest non-fragile joint p in the monthly ledger", MONTHLY, "country__y_yoy_nsa__long__all_eligible__H1__two_channel", "monthly", 4000, 2026111104, True),
        ("origin×year country YoY own-current H1; smallest joint p in the origin×year ledger", ORIGIN_YEAR, "country__y_yoy_nsa__long__all_eligible__H1__own_current__origin_year", "monthly", 4000, 2026111105, True),
    ]
    part_a = [joint_benchmark(*design) for design in designs]

    part_b, validation = [], None
    for spec_id, validate in [("aggregate__y_yoy_nsa__long__all_eligible__common__own_current", None),
                              ("aggregate__y_yoy_nsa__long__all_eligible__common__two_channel", ("temp_growth_sum", 12)),
                              ("aggregate__y_yoy_nsa__long__all_eligible__common__distributed", None),
                              ("aggregate__y_yoy_nsa__long__event_dates_excluded__common__own_current", None),
                              ("aggregate__y_yoy_nsa__long__event_dates_excluded__common__two_channel", None),
                              ("aggregate__y_yoy_nsa__long__event_dates_excluded__common__distributed", None)]:
        result, check = t_ratio_laws(spec_id, validate)
        part_b.append(result)
        validation = check or validation
    print("B  Monte Carlo validation of the exact formula:", validation, flush=True)

    part_c = [residual_autocorrelation(MONTHLY, spec_id) for spec_id in [
        "aggregate__y_yoy_nsa__long__all_eligible__common__own_current", "aggregate__y_yoy_nsa__long__event_dates_excluded__common__own_current",
        "aggregate__y_yoy_nsa__long__strict_growth_window_clean__common__own_current",
        "region__y_yoy_nsa__long__all_eligible__H1__own_current", "country__y_yoy_nsa__long__all_eligible__H1__own_current",
        "country__y_yoy_nsa__long__event_dates_excluded__H1__own_current",
        "aggregate__y_mom_nsa__long__all_eligible__common__own_current", "aggregate__y_mom_nsa__long__event_dates_excluded__common__own_current",
        "country__y_mom_nsa__long__all_eligible__H1__own_current"]]

    # D. Plain ledger tabulation (no new arithmetic): main-method t ratios by outcome, time df >= 10.
    part_d = []
    for family, folder in [("monthly", MONTHLY), ("origin_year", ORIGIN_YEAR)]:
        effects = pd.read_csv(folder / "temperature_effects.csv", low_memory=False)
        main = effects.loc[effects.covariance.eq("Calendar HAC") & effects.hac_lags.eq(12) & effects.time_residual_df.ge(10) & effects.p_value.notna()]
        ratio = (main.coefficient / main.std_error).abs()
        for (outcome, model), rows in main.groupby(["outcome", "model"]):
            t_abs = ratio.loc[rows.index]
            part_d.append({"family": family, "outcome": outcome, "model": model, "rows": len(rows), "p_lt_05": int((rows.p_value < 0.05).sum()), "p_lt_01": int((rows.p_value < 0.01).sum()),
                           "p_lt_001": int((rows.p_value < 0.001).sum()), "share_p_lt_05": float((rows.p_value < 0.05).mean()),
                           "abs_t_ge_2p74": int((t_abs >= 2.74).sum()), "abs_t_ge_3p37": int((t_abs >= 3.37).sum())})
            print("D ", part_d[-1], flush=True)

    payload = {"probe": 11, "title": "Selected fixed-design reference-law benchmarks (five joint-test designs, six single-series YoY designs)", "command": COMMAND,
               "scope_statement": "Selected saved designs only. Fixed-design Gaussian reference DGPs; not a calibration grid, not an estimate of the size of every arrival test, "
                                  "not a true-null experiment on the data. iid rows are the favourable case; the MA(11) case is the mechanical overlap of a 12-month log difference under white-noise monthly errors.",
               "A_joint_wald_fixed_design_gaussian_reference_laws": part_a, "B_yoy_t_ratio_exact_laws": part_b, "B_monte_carlo_validation": validation,
               "C_saved_residual_autocorrelation_descriptive": part_c, "D_ledger_main_method_t_ratios_by_outcome": part_d,
               "D_note": "abs_t thresholds 2.74 and 3.37 are the range of exact 5% critical values of the three all-eligible aggregate YoY designs under the MA(11) reference law (HAC12); "
                         "they are design-specific and are shown only to locate ledger rows, not as corrected critical values.",
               "seconds": round(clock.time() - started, 1)}
    write("probe11_result.json", payload)
    print("wrote probe11_result.json in", payload["seconds"], "s")


if __name__ == "__main__":
    main()
