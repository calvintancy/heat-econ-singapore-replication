"""Independently reconstruct visitor outcomes, calendars, FE estimates and HAC.

Does not import the production visitor preparation, estimation or inference code.
Writes only the visitor verification receipt; reviewed source artifacts are read-only.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.stats import f as f_dist, t as t_dist
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/visitor_arrivals"
DATA = ROOT / "data/processed/visitor_arrivals"
PANELS = ["aggregate", "region", "country"]
METRICS: dict[str, float] = {}
COUNTS: dict[str, int] = {}


def compare(name, actual, expected, *, atol=1e-8, rtol=2e-7):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(f"{name}: independent calculation differs")
    finite = np.isfinite(a) & np.isfinite(b)
    error = float(np.max(np.abs(a[finite] - b[finite]) / np.maximum(1., np.abs(b[finite])))) if finite.any() else 0.
    METRICS[name] = max(METRICS.get(name, 0.), error)


def hac(psi, time, bandwidth, correction):
    """Direct Bartlett sum on actual calendar lags, with absent months zero-filled."""
    psi = np.asarray(psi)
    if psi.ndim == 1:
        psi = psi[:, None]
    time = np.asarray(time, dtype=int)
    calendar = np.arange(time.min(), time.max() + 1)
    score = np.zeros((len(calendar), psi.shape[1]))
    np.add.at(score, time-time.min(), psi)
    value = score.T @ score
    for lag in range(1, min(bandwidth+1, len(score))):
        cross = score[lag:].T @ score[:-lag]
        value += (cross+cross.T) * (1-lag/(bandwidth+1))
    return (value+value.T) * correction / 2


def reference_sources():
    manifest = json.loads((ROOT / "data/raw/visitor_arrivals/manifest.json").read_text())
    for filename, record in manifest.items():
        assert hashlib.sha256((ROOT / "data/raw/visitor_arrivals" / filename).read_bytes()).hexdigest() == record["sha256"]
    meta = json.loads((ROOT / "data/raw/visitor_arrivals/nsa_singstat_metadata.json").read_text())["Data"]["records"]
    catalog = pd.DataFrame(meta["row"])[["seriesNo", "rowText"]].rename(columns={"seriesNo": "id", "rowText": "unit"})
    catalog["panel"] = "residual"
    catalog.loc[catalog.id.eq("1"), "panel"] = "aggregate"
    catalog.loc[catalog.id.str.count(r"\.").eq(1) & ~catalog.id.eq("1.999"), "panel"] = "region"
    catalog.loc[catalog.id.str.count(r"\.").eq(2) & ~catalog.id.str.endswith(".999"), "panel"] = "country"
    source = pd.read_csv(ROOT / "data/raw/visitor_arrivals/nsa.csv")
    source.DataSeries = source.DataSeries.str.strip()
    assert source.DataSeries.is_unique
    values = source.set_index("DataSeries").T
    values.index = pd.PeriodIndex(pd.to_datetime(values.index, format="%Y%b"), freq="M")
    values = values.sort_index().astype(float)
    assert values.index.equals(pd.period_range(values.index.min(), values.index.max(), freq="M"))
    assert values.size == meta["total"] and values.notna().all().all() and values.ge(0).all().all()
    counts = {p: values[catalog.loc[catalog.panel.eq(p), "unit"]] for p in PANELS}
    assert [len(v.columns) for v in counts.values()] == [1, 9, 42]
    compare("source_total_reconciliation", counts["aggregate"].iloc[:, 0], counts["region"].sum(axis=1)+values.Others, atol=0, rtol=0)
    for row in catalog.loc[catalog.panel.eq("region")].itertuples():
        children = catalog.loc[catalog.id.str.startswith(row.id+"."), "unit"]
        compare("source_region_reconciliation", values[row.unit], values[children].sum(axis=1), atol=0, rtol=0)
    outcomes = {}
    for p, level in counts.items():
        log = np.log(level.where(level.gt(0)))
        daily = log.sub(np.log(level.index.days_in_month), axis=0)
        outcomes[p] = {"mom": 100*(log-log.shift(1)), "yoy": 100*(log-log.shift(12)),
                       "log_level": 100*log, "raw": level, "daily_mom": 100*(daily-daily.shift(1))}
    source = pd.read_csv(ROOT / "data/raw/visitor_arrivals/sa.csv")
    source.DataSeries = source.DataSeries.str.strip()
    assert source.iloc[0, 0].startswith("Total International Visitor Arrivals")
    sa = source.set_index("DataSeries").T
    sa.index = pd.PeriodIndex(pd.to_datetime(sa.index, format="%Y%b"), freq="M")
    sa = sa.sort_index().astype(float)
    assert sa.iloc[:, 0].gt(0).all()
    log = np.log(sa.iloc[:, 0])
    outcomes["aggregate_sa"] = {"mom": (100*(log-log.shift(1))).to_frame(),
                                "yoy": (100*(log-log.shift(12))).to_frame()}
    weather = pd.read_csv(DATA / "weather.csv", index_col="month")
    weather.index = pd.PeriodIndex(weather.index, freq="M")
    assert weather.index.equals(values.index)
    for stem, column, target in [("temperature", "mean_temp", "temp_c"), ("humidity", "mean_rh", "rh_pct"),
                                 ("rainfall", "total_rainfall", "rain_mm")]:
        raw = pd.read_csv(ROOT / f"data/raw/{stem}_monthly.csv", index_col="month")
        raw.index = pd.PeriodIndex(raw.index, freq="M")
        compare("raw_weather_alignment", weather[target], raw[column].reindex(weather.index))
    compare("rainfall_scaling", weather.rain_100mm, weather.rain_mm/100)
    compare("quadratic_temperature", weather.temp_sq, (weather.temp_c-28)**2)
    compare("calendar_time", weather.time, weather.index.asi8, atol=0, rtol=0)
    compare("calendar_trend", weather.trend, (weather.index.asi8-weather.index.asi8.min())/12)
    for variable in ["temp_c", "rain_100mm", "rh_pct", "wind_mps", "pm25_10ug"]:
        for lag in [1, 2, 3, 12]:
            compare("calendar_weather_lags", weather[f"{variable}_lag{lag}"], weather[variable].shift(lag))
        for lag in [1, 12]:
            compare("calendar_weather_differences", weather[f"{variable}_diff{lag}"], weather[variable]-weather[variable].shift(lag))
    compare("calendar_weather_lead", weather.temp_lead1, weather.temp_c.shift(-1))
    compare("calendar_lunar_control_lag", weather.lny_lag1, weather.lny_window_days.shift(1))
    extended = pd.read_csv(ROOT / "data/processed/extended_controls/monthly_controls.csv", index_col="month")
    extended.index = pd.PeriodIndex(extended.index, freq="M")
    for variable in ["wind_mps", "pm25_10ug", "wind_coverage", "pm_coverage"]:
        compare("extended_control_alignment", weather[variable], extended[variable].reindex(weather.index))
    for threshold in ["eligible_80", "eligible_90"]:
        assert weather[threshold].eq(True).equals(extended[threshold].reindex(weather.index).eq(True))
    COUNTS.update(source_months=len(values), named_countries=42, regions=9,
                  source_country_zero_cells=int(counts["country"].eq(0).sum().sum()))
    return counts, outcomes, weather, values


def independent_mask(sp, weather, outcomes):
    own = sp["own_dates"]
    terms = sp.get("support_terms") or sp["terms"]
    mask = weather[terms].notna().all(axis=1)
    panel, outcome, sample = sp["panel"], sp["outcome"], sp["sample"]
    if not own:
        for p in PANELS:
            mask &= outcomes[p][outcome].notna().all(axis=1)
    mask &= outcomes[panel][outcome].notna().all(axis=1)
    if sample.startswith("full"):
        threshold = "eligible_80" if "80" in sample else "eligible_90"
        mask &= weather[threshold].eq(True) & weather.index.year.isin(range(2017, 2026))
    if "pre2020" in sample:
        mask &= weather.index.year < 2020
    if "endpoints" in sample or "clean" in sample:
        mask &= ~weather.index.year.isin(range(2020, 2023))
    if "clean" in sample:
        horizon = {"mom": 1, "daily_mom": 1, "yoy": 12}.get(outcome, 0)
        mask &= ~(weather.index-horizon).year.isin(range(2020, 2023))
        if sp["variant"] in ["lags", "lag_matched_current"]:
            for lag in [1, 2, 3]:
                mask &= ~(weather.index-lag).year.isin(range(2020, 2023))
        if sp["variant"] == "lead":
            mask &= ~(weather.index+1).year.isin(range(2020, 2023))
    if sp.get("omitted_year") is not None:
        mask &= weather.index.year != sp["omitted_year"]
    if sample.startswith("full") and sp["variant"] == "weather_differences":
        horizon = 12 if outcome == "yoy" else 1
        mask &= weather.eligible_90.eq(True).shift(horizon, fill_value=False)
    return mask


def ols_verification(counts, outcomes, weather):
    specs = json.loads((OUT / "specifications.json").read_text())
    results = pd.read_csv(OUT / "regression_results.csv")
    coefficients = pd.read_csv(OUT / "all_weather_coefficients.csv")
    samples = pd.read_csv(OUT / "model_samples.csv")
    lag_results = pd.read_csv(OUT / "lag_results.csv")
    for sp in specs:
        name = sp["model_id"]
        z = np.load(OUT / "designs" / f"{name}.npz")
        dates = pd.PeriodIndex(z["months"], freq="M")
        mask = independent_mask(sp, weather, outcomes)
        assert dates.equals(weather.index[mask]), f"Calendar: {name}"
        assert samples.loc[samples.model_id.eq(name), "month"].tolist() == list(dates.astype(str))
        units = outcomes[sp["panel"]][sp["outcome"]].loc[dates]
        compare("raw_outcome_transforms", z["Y_units"], units)
        if sp["weighted"]:
            weights = counts[sp["panel"]].loc["2017-01":"2019-12"].sum()
            weights /= weights.sum()
            y = (units @ weights).to_numpy()
        else:
            y = units.mean(axis=1).to_numpy()
        compare("collapsed_outcome_and_weights", z["y"], y)
        X = z["X"]
        assert X.shape[0] == len(dates) and X.shape[1] == len(sp["columns"])
        assert list(z["columns"]) == sp["columns"]
        for term in sp["terms"]:
            compare("design_weather_columns", X[:, list(z["columns"]).index(term)], weather.loc[dates, term])
        for column, label in enumerate(z["columns"]):
            if label == "Intercept":
                expected = np.ones(len(dates))
            elif match := re.fullmatch(r"C\(season\)\[T\.(\d+)\]", label):
                expected = dates.month == int(match[1])
            elif match := re.fullmatch(r"C\(year\)\[T\.(\d+)\]", label):
                expected = dates.year == int(match[1])
            elif match := re.fullmatch(r"C\(season\)\[(\d+)\]:trend", label):
                expected = (dates.month == int(match[1]))*weather.loc[dates, "trend"]
            else:
                continue
            compare("design_fixed_effect_columns", X[:, column], expected)
        assert np.linalg.matrix_rank(X) == X.shape[1]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y-X@beta
        compare("ols_coefficients", z["beta"], beta, atol=2e-7)
        compare("ols_residuals", z["residual"], residual, atol=2e-7)
        # QR bread avoids the precision loss of directly inverting X'X.
        _, R = np.linalg.qr(X, mode="reduced")
        inverse = np.linalg.solve(R, np.eye(R.shape[0]))
        bread = inverse@inverse.T
        inf = (X@bread)*residual[:, None]
        T, K = X.shape
        rows = results.loc[results.model_id.eq(name)]
        assert len(rows) == 2
        for row in rows.itertuples():
            covariance = hac(inf, z["time"], row.hac_lags, T/(T-K))
            j = list(z["columns"]).index(row.temperature_term)
            compare("ols_reported_temperature", row.coefficient, beta[j])
            compare("ols_hac_variance", row.std_error**2, covariance[j, j], atol=2e-6)
            compare("ols_mean_sd", [row.mean_y, row.sd_y], [y.mean(), y.std(ddof=1)])
            compare("ols_p_values", row.p_value, 2*t_dist.sf(abs(beta[j])/np.sqrt(covariance[j, j]), T-1), atol=1e-10)
            critical = t_dist.ppf(.975, T-1)
            se = np.sqrt(covariance[j, j])
            compare("ols_intervals_and_scaling", [row.ci_lower, row.ci_upper, row.effect_0p1c, row.lower_0p1c, row.upper_0p1c],
                    [beta[j]-critical*se, beta[j]+critical*se, beta[j]/10, (beta[j]-critical*se)/10, (beta[j]+critical*se)/10])
            for lag in [1, 12]:
                lookup = dict(zip(z["time"], residual))
                pairs = [(residual[i], lookup[time-lag]) for i, time in enumerate(z["time"]) if time-lag in lookup]
                acf = np.corrcoef(np.asarray(pairs).T)[0, 1]
                compare("calendar_residual_acf", getattr(row, f"residual_acf_calendar_{lag}"), acf, atol=1e-10)
            subset = coefficients.loc[coefficients.model_id.eq(name) & coefficients.hac_lags.eq(row.hac_lags)]
            assert set(subset.term) == set(sp["terms"])
            for cr in subset.itertuples():
                jj = list(z["columns"]).index(cr.term)
                compare("all_weather_coefficients", cr.coefficient, beta[jj], atol=2e-7)
                compare("all_weather_hac_variance", cr.std_error**2, covariance[jj, jj], atol=2e-6)
            lag = lag_results.loc[lag_results.model_id.eq(name) & lag_results.hac_lags.eq(row.hac_lags)]
            if len(lag):
                terms = ["temp_c", "temp_lead1"] if sp["variant"] == "lead" else ["temp_c"]+[f"temp_c_lag{i}" for i in [1, 2, 3]]
                positions = [list(z["columns"]).index(term) for term in terms]
                for lr in lag.itertuples():
                    jj = positions if lr.term == "sum_lag0_to3" else [list(z["columns"]).index(lr.term)]
                    compare("lag_coefficient_or_sum", lr.coefficient, beta[jj].sum())
                    compare("lag_joint_hac_variance", lr.std_error**2, covariance[np.ix_(jj, jj)].sum(), atol=2e-6)
        for panel in PANELS:
            if sp["panel"] != panel or sp["variant"] != "main":
                continue
            same = [s for s in specs if s["variant"] == "main" and s["sample"] == sp["sample"] and s["outcome"] == sp["outcome"]]
            assert len(same) == 3
            for other in same:
                assert samples.loc[samples.model_id.eq(other["model_id"]), "month"].tolist() == list(dates.astype(str))
        COUNTS["ols_models"] = COUNTS.get("ols_models", 0)+1
        COUNTS["exact_calendar_masks"] = COUNTS.get("exact_calendar_masks", 0)+1
        if sp.get("omitted_year") is not None:
            COUNTS["leave_one_year_out_models"] = COUNTS.get("leave_one_year_out_models", 0)+1
    for sp in specs:
        if sp["variant"] == "lags":
            a = samples.loc[samples.model_id.eq(sp["model_id"]), "month"].tolist()
            name = sp["model_id"].rsplit("__", 1)[0]+"__lag_matched_current"
            assert samples.loc[samples.model_id.eq(name), "month"].tolist() == a


def explicit_stacked_fe():
    receipts = []
    for panel in ["region", "country"]:
        z = np.load(OUT / "designs" / f"{panel}__mom__long_pre2020__main.npz")
        X, Y, columns = z["X"], z["Y_units"], z["columns"]
        T, S = Y.shape
        seasons = pd.PeriodIndex(z["months"], freq="M").month.to_numpy()-1
        origin_month = np.zeros((T*S, S*12))
        origin_month[np.arange(T*S), np.repeat(np.arange(S), T)*12+np.tile(seasons, S)] = 1
        shared = [i for i, c in enumerate(columns) if c != "Intercept" and not c.startswith("C(season)")]
        stacked = np.column_stack([origin_month, np.tile(X[:, shared], (S, 1))])
        beta, _, rank, _ = lstsq(stacked, Y.T.ravel(), lapack_driver="gelsy")
        assert rank == stacked.shape[1]
        j = S*12+list(columns[shared]).index("temp_c")
        expected = z["beta"][list(columns).index("temp_c")]
        compare("explicit_stacked_fe_equality", beta[j], expected, atol=1e-9)
        receipts.append({"panel": panel, "observations": len(stacked), "parameters": len(beta),
                         "temperature_coefficient": float(beta[j]), "collapsed_difference": float(beta[j]-expected)})
    return receipts


def profile_verification():
    slopes = pd.read_csv(OUT / "market_slopes.csv")
    tests = json.loads((OUT / "joint_tests.json").read_text())
    for path in (OUT / "designs").glob("profiles__*.npz"):
        z = np.load(path)
        X, Y = z["X"], z["Y"]
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        residual = Y-X@beta
        compare("profile_coefficients", z["beta"], beta, atol=2e-7)
        compare("profile_residuals", z["residual"], residual, atol=2e-7)
        _, R = np.linalg.qr(X, mode="reduced")
        inverse = np.linalg.solve(R, np.eye(R.shape[0]))
        bread = inverse@inverse.T
        j = int(z["temperature_index"])
        psi = (X@bread[:, j])[:, None]*residual
        bandwidth = int(path.stem.split("__")[-1])
        group = path.stem.removeprefix("profiles__").rsplit("__", 1)[0]
        covariance = hac(psi, z["time"], bandwidth, len(X)/(len(X)-X.shape[1]))
        compare("profile_joint_hac_covariance", z["covariance"], covariance, atol=1e-8)
        rows = slopes.loc[slopes.group.eq(group) & slopes.hac_lags.eq(bandwidth)].set_index("unit").loc[list(z["units"])]
        compare("profile_reported_slopes", rows.coefficient, beta[j])
        compare("profile_reported_variances", rows.std_error**2, np.diag(covariance))
        p = 2*t_dist.sf(abs(beta[j])/np.sqrt(np.diag(covariance)), len(X)-1)
        compare("profile_holm_multiplicity", rows.holm_p_value, multipletests(p, method="holm")[1], atol=1e-10)
        C = np.column_stack([-np.ones(Y.shape[1]-1), np.eye(Y.shape[1]-1)])
        V, delta = C@covariance@C.T, C@beta[j]
        rank = np.linalg.matrix_rank(V)
        record = next(t for t in tests if t["group"] == group and t["hac_lags"] == bandwidth)
        assert record["covariance_rank"] == rank
        assert bool(record["p_value"] is not None) == (rank == len(delta))
        if rank == len(delta):
            statistic = delta@np.linalg.solve(V, delta)/len(delta)
            compare("profile_joint_test", record["statistic"], statistic, atol=1e-7)
            compare("profile_joint_test_p", record["p_value"], f_dist.sf(statistic, len(delta), len(X)-1), atol=1e-10)
        COUNTS["profile_model_bandwidths"] = COUNTS.get("profile_model_bandwidths", 0)+1


def explicit_weighted_projection(X, mu, groups):
    G, H = groups[:, 0], groups[:, 1]
    d = np.zeros((len(X), G.max()+1+H.max()))
    d[np.arange(len(X)), G] = 1.
    for year in range(1, H.max()+1):
        d[:, G.max()+year] = H == year
    weights = np.sqrt(mu)
    coefficients, _, rank, _ = lstsq(d*weights[:, None], X*weights[:, None], lapack_driver="gelsy")
    assert rank == d.shape[1]
    return X-d@coefficients, d


def ppml_verification(counts, weather):
    results = pd.read_csv(OUT / "ppml_results.csv")
    receipts = json.loads((OUT / "ppml/estimator_receipts.json").read_text())
    explicit = []
    for receipt in receipts:
        name, panel, sample = receipt["model_id"], receipt["panel"], receipt["sample"]
        frame = pd.read_csv(OUT / "ppml" / f"{name}_sample.csv")
        variables = ["temp_c", "rain_100mm", "rh_pct"]+(["wind_mps", "pm25_10ug"] if sample.startswith("full") else [])
        mask = weather[variables].notna().all(axis=1)
        if sample.startswith("full"):
            mask &= weather.eligible_90.eq(True) & weather.index.year.isin(range(2017, 2026))
        if sample.endswith("pre2020"):
            mask &= weather.index.year < 2020
        if sample.endswith("clean"):
            mask &= ~weather.index.year.isin(range(2020, 2023))
        expected = counts[panel].loc[weather.index[mask]]
        observed = frame.pivot(index="month", columns="unit", values="arrivals")
        observed.index = pd.PeriodIndex(observed.index, freq="M")
        observed = observed.reindex(index=expected.index, columns=expected.columns)
        compare("ppml_all_counts_and_zeros", observed, expected, atol=0, rtol=0)
        assert len(frame) == expected.size and frame.unit.nunique() == expected.shape[1]
        assert int(frame.arrivals.eq(0).sum()) == receipt["zero_cells_retained"] == int(expected.eq(0).sum().sum())
        z = np.load(OUT / "ppml" / f"{name}__12.npz")
        X, y, mu, groups, xr = z["X"], z["y"], z["mu"], z["groups"], z["xr"]
        compare("ppml_saved_inputs", X, frame[variables])
        compare("ppml_saved_outcome", y, frame.arrivals, atol=0, rtol=0)
        assert np.isfinite(mu).all() and np.min(mu) > 0
        u = y-mu
        scale = y.sum()
        score_error = float(max(np.max(abs(X.T@u)), max(np.max(abs(np.bincount(groups[:, j], weights=u))) for j in range(2)))/scale)
        assert score_error < 1e-6, f"PPML score: {name}"
        METRICS["ppml_relative_score_error"] = max(METRICS.get("ppml_relative_score_error", 0.), score_error)
        projected_error = max(np.max(abs(np.bincount(groups[:, g], weights=mu*xr[:, j]))) for g in range(2) for j in range(X.shape[1]))/scale
        assert projected_error < 1e-9
        METRICS["ppml_relative_projection_orthogonality"] = max(METRICS.get("ppml_relative_projection_orthogonality", 0.), float(projected_error))
        if name in ["region__full_all", "country__full_all"]:
            independent_xr, nuisance = explicit_weighted_projection(X, mu, groups)
            compare("ppml_explicit_weighted_fe_projection", xr, independent_xr, atol=1e-7, rtol=1e-7)
            # Fitted log means must equal weather coefficients plus the FE space.
            target = np.log(mu)-X@z["beta"]
            fe_beta = lstsq(nuisance, target, lapack_driver="gelsy")[0]
            compare("ppml_link_and_coefficient_reconstruction", target, nuisance@fe_beta, atol=2e-6)
            explicit.append({"model_id": name, "n_observations": len(X), "n_fe_parameters": nuisance.shape[1],
                             "max_abs_projected_X_difference": float(np.max(abs(xr-independent_xr)))})
            projected = independent_xr
        else:
            projected = xr
        information = projected.T@(mu[:, None]*projected)
        bread = np.linalg.solve(information, np.eye(len(variables)))
        newton_step = bread@(projected.T@u)
        assert np.max(abs(newton_step)) < 1e-7
        METRICS["ppml_max_newton_step"] = max(METRICS.get("ppml_max_newton_step", 0.), float(np.max(abs(newton_step))))
        influence = (projected*u[:, None])@bread
        T = frame.month.nunique()
        K = 12+frame.year.nunique()-1+len(variables)
        compare("ppml_temporal_correction", z["correction"], T/(T-K))
        for bandwidth in [12, 24]:
            covariance = hac(influence, z["time"], bandwidth, T/(T-K))
            saved = np.load(OUT / "ppml" / f"{name}__{bandwidth}.npz")
            compare("ppml_joint_hac_covariance", saved["covariance"], covariance, atol=1e-10, rtol=1e-6)
            row = results.loc[results.model_id.eq(name) & results.hac_lags.eq(bandwidth)].iloc[0]
            compare("ppml_reported_slope", row.coefficient, 100*z["beta"][0])
            compare("ppml_reported_variance", row.std_error**2, 10000*covariance[0, 0], atol=1e-8, rtol=1e-6)
        COUNTS["ppml_models"] = COUNTS.get("ppml_models", 0)+1
    return explicit


def run():
    with threadpool_limits(limits=1):
        counts, outcomes, weather, _ = reference_sources()
        ols_verification(counts, outcomes, weather)
        stacked = explicit_stacked_fe()
        profile_verification()
        weighted = ppml_verification(counts, weather)
    hashes = json.loads((OUT / "input_hashes.json").read_text())
    economic_path = "outputs/heat_economy.html"
    checked_inputs = 0
    for path, expected in hashes.items():
        if path == economic_path:
            continue
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, f"Input changed: {path}"
        checked_inputs += 1
    economic_file = ROOT/economic_path
    current_hash = hashlib.sha256(economic_file.read_bytes()).hexdigest()
    before_hash = hashes[economic_path]
    unchanged = current_hash == before_hash
    isolation = {"path": economic_path, "analytical_input": False,
                 "before_sha256": before_hash, "current_sha256": current_hash,
                 "unchanged": unchanged, "current_bytes": economic_file.stat().st_size,
                 "current_modified_utc": datetime.fromtimestamp(economic_file.stat().st_mtime, timezone.utc).isoformat(),
                 "status": "unchanged" if unchanged else "concurrent_change_observed",
                 "interpretation": "This economic-report HTML is not a visitor analytical input. Concurrent project work changed it after the original snapshot; numerical verification does not require it to remain immutable. The verifier does not write this file, and the hash comparison alone does not identify the writer." if not unchanged else "The economic-report HTML still matches its original snapshot; the verifier does not write it."}
    receipt = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
               "implementation": "Independent raw-source/calendar reconstruction; NumPy/SciPy QR/lstsq; direct calendar Bartlett sums; statsmodels Holm. No production visitor functions imported.",
               "counts": COUNTS, "max_scaled_errors": METRICS,
               "stacked_FE_checks": stacked, "PPML_explicit_weighted_FE_checks": weighted,
               "analytical_input_hashes": {"status": "passed", "inputs_checked": checked_inputs},
               "main_economic_report_unchanged": unchanged,
               "main_economic_report_isolation_observation": isolation,
               "interpretation_limits": ["Common weather panel cuts are different origin-weighting estimands, not independent exposure replication.",
                   "Separate-origin profiles allow origin-specific weather slopes, year effects and seasonality.",
                   "Short country equality tests have many restrictions relative to residual time support; rank checks do not make their approximate F inference reliable.",
                   "PPML is a count-level log-conditional-mean sensitivity, not a growth model. Its clean exclusion has no lagged base.",
                   "Raw-count and raw-count variance tolerances are relative to scale; max_scaled_errors divides discrepancies by max(1, absolute reference value)."]}
    path = OUT / "verification.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": receipt["status"], "counts": COUNTS, "receipt": str(path)}, indent=2))


if __name__ == "__main__":
    run()
