"""Separate, reproducible monthly heat and visitor-arrivals analysis.

Balanced common-weather FE slopes use their exact equal-origin mean reduction.
Heterogeneity uses a joint system with origin-specific nuisance parameters.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy.stats import f as f_dist, t as t_dist
from statsmodels.tsa.stattools import adfuller, kpss
from threadpoolctl import threadpool_limits

import pyfixest_backend as native
from visitor_fetch import fetch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/visitor_arrivals"
DATA = ROOT / "data/processed/visitor_arrivals"
OUT = ROOT / "outputs/visitor_arrivals"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
FULL = BASE + ["wind_mps", "pm25_10ug"]
OUTCOMES = ["mom", "yoy", "log_level", "raw"]
PANELS = ["aggregate", "region", "country"]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def prepare():
    fetch()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RAW / "nsa_singstat_metadata.json").read_text())["Data"]["records"]
    catalog = pd.DataFrame(meta["row"])[["seriesNo", "rowText"]].rename(
        columns={"seriesNo": "unit_id", "rowText": "unit"})
    catalog["panel"] = np.select(
        [catalog.unit_id.eq("1"), catalog.unit_id.str.count(r"\.").eq(1) & ~catalog.unit_id.eq("1.999"),
         catalog.unit_id.str.count(r"\.").eq(2) & ~catalog.unit_id.str.endswith(".999")],
        PANELS, default="residual")
    catalog["region_id"] = catalog.unit_id.map(lambda s: ".".join(s.split(".")[:2]) if "." in s else "")
    source = pd.read_csv(RAW / "nsa.csv")
    source["DataSeries"] = source.DataSeries.str.strip()
    assert source.DataSeries.is_unique and set(source.DataSeries) == set(catalog.unit)
    wide = source.set_index("DataSeries").T
    wide.index = pd.PeriodIndex(pd.to_datetime(wide.index, format="%Y%b"), freq="M")
    wide = wide.sort_index().apply(pd.to_numeric, errors="raise")
    wide.index.name = "month"
    assert len(wide) * len(wide.columns) == meta["total"]
    assert wide.notna().all().all() and wide.ge(0).all().all()
    assert wide.index.equals(pd.period_range(wide.index.min(), wide.index.max(), freq="M", name="month"))
    counts = {key: wide[catalog.loc[catalog.panel.eq(key), "unit"].tolist()].copy() for key in PANELS}
    assert [d.shape[1] for d in counts.values()] == [1, 9, 42]
    reconciliation = pd.DataFrame(index=wide.index)
    total = counts["aggregate"].iloc[:, 0]
    reconciliation["total_minus_regions_and_others"] = total - counts["region"].sum(axis=1) - wide.Others
    for r in catalog.loc[catalog.panel.eq("region")].itertuples():
        children = catalog.loc[catalog.unit_id.str.startswith(r.unit_id + "."), "unit"]
        reconciliation[r.unit] = wide[r.unit] - wide[children].sum(axis=1)
    assert reconciliation.eq(0).all().all(), "Arrival hierarchy failed to reconcile"
    reconciliation.to_csv(DATA / "hierarchy_reconciliation.csv")
    catalog.to_csv(DATA / "market_catalog.csv", index=False)
    wide.to_csv(DATA / "all_source_counts.csv")
    weather_frames = []
    input_paths = [RAW / "nsa.csv", RAW / "sa.csv"]
    for name, old, new in [("temperature", "mean_temp", "temp_c"),
                           ("humidity", "mean_rh", "rh_pct"),
                           ("rainfall", "total_rainfall", "rain_mm")]:
        path = ROOT / f"data/raw/{name}_monthly.csv"
        frame = pd.read_csv(path).set_index("month").rename(columns={old: new})
        frame.index = pd.PeriodIndex(frame.index, freq="M")
        assert frame.index.is_unique
        weather_frames.append(frame[[new]])
        input_paths.append(path)
    weather = pd.concat(weather_frames, axis=1).reindex(wide.index)
    weather["rain_100mm"] = weather.rain_mm / 100
    weather["year"] = weather.index.year
    weather["season"] = weather.index.month
    weather["time"] = weather.index.asi8
    weather["trend"] = (weather.time - weather.time.min()) / 12
    weather["temp_sq"] = (weather.temp_c - 28) ** 2
    path = ROOT / "data/processed/extended_controls/monthly_controls.csv"
    controls = pd.read_csv(path, index_col="month")
    controls.index = pd.PeriodIndex(controls.index, freq="M")
    weather = weather.join(controls[["wind_mps", "pm25_10ug", "eligible_90", "eligible_80", "wind_coverage", "pm_coverage"]])
    input_paths.append(path)
    calendar_path = DATA / "lunar_new_year.csv"
    holidays = pd.read_csv(calendar_path, index_col="month")
    holidays.index = pd.PeriodIndex(holidays.index, freq="M")
    weather = weather.join(holidays[["lny_window_days"]])
    weather["lny_lag1"] = weather.lny_window_days.shift(1)
    input_paths.append(calendar_path)
    for term in FULL:
        for lag in [1, 2, 3, 12]:
            weather[f"{term}_lag{lag}"] = weather[term].shift(lag)
        for horizon in [1, 12]:
            weather[f"{term}_diff{horizon}"] = weather[term] - weather[term].shift(horizon)
    weather["temp_lead1"] = weather.temp_c.shift(-1)
    weather.to_csv(DATA / "weather.csv")
    outcomes = {}
    for panel, values in counts.items():
        logs = np.log(values.where(values.gt(0)))
        outcomes[panel] = {"mom": 100 * logs.diff(), "yoy": 100 * logs.diff(12),
                           "log_level": 100 * logs, "raw": values,
                           "daily_mom": 100 * logs.sub(np.log(values.index.days_in_month), axis=0).diff()}
        weights = values.loc["2017-01":"2019-12"].sum()
        (weights / weights.sum()).rename("fixed_2017_19_share").to_csv(DATA / f"{panel}_weights.csv")
    sa_source = pd.read_csv(RAW / "sa.csv")
    sa_source["DataSeries"] = sa_source.DataSeries.str.strip()
    sa = sa_source.set_index("DataSeries").T
    sa.index = pd.PeriodIndex(pd.to_datetime(sa.index, format="%Y%b"), freq="M")
    sa = sa.sort_index().apply(pd.to_numeric, errors="raise")
    a = np.log(sa.iloc[:, 0])
    outcomes["aggregate_sa"] = {"mom": 100 * a.diff().to_frame("Official SA aggregate"),
                                "yoy": 100 * a.diff(12).to_frame("Official SA aggregate")}
    zero_rows = wide.stack().loc[lambda s: s.eq(0)].rename("arrivals").reset_index()
    zero_rows.to_csv(DATA / "reported_zeros.csv", index=False)
    coverage = pd.DataFrame({"total": total, "named": counts["country"].sum(axis=1), "others": wide.Others})
    coverage["named_share"] = coverage.named / coverage.total
    coverage.to_csv(DATA / "coverage.csv")
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    main_report = ROOT / "outputs/heat_economy.html"
    inputs[str(main_report.relative_to(ROOT))] = hashlib.sha256(main_report.read_bytes()).hexdigest()
    save_json(OUT / "input_hashes.json", inputs)
    summary = {"source_start": str(wide.index.min()), "source_end": str(wide.index.max()),
               "source_months": len(wide), "source_series": len(wide.columns),
               "source_updated": meta["dataLastUpdated"], "panels": {p: v.shape[1] for p, v in counts.items()},
               "nsa_metadata_adjustment_discrepancy": meta["adjustmentType"],
               "hierarchy_max_abs_difference": int(reconciliation.abs().max().max()),
               "country_zero_cells": int(counts["country"].eq(0).sum().sum()),
               "zero_country_months": list(counts["country"].index[counts["country"].eq(0).any(axis=1)].astype(str)),
               "sa_series": len(sa.columns),
               "annual_named_coverage": {str(y): float(g.named.sum() / g.total.sum()) for y, g in coverage.groupby(coverage.index.year)},
               "annual_total_arrivals": {str(y): int(g.total.sum()) for y, g in coverage.groupby(coverage.index.year)}}
    save_json(OUT / "data_checks.json", summary)
    return counts, outcomes, weather, catalog


def load_prepared():
    """Read the frozen visitor inputs without refreshing or rewriting any file."""
    wide = pd.read_csv(DATA / "all_source_counts.csv", index_col="month")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    catalog = pd.read_csv(DATA / "market_catalog.csv", dtype={"unit_id": str})
    counts = {panel: wide[catalog.loc[catalog.panel.eq(panel), "unit"]] for panel in PANELS}
    outcomes = {}
    for panel, levels in counts.items():
        logs = np.log(levels.where(levels.gt(0)))
        outcomes[panel] = {"mom": 100*logs.diff(), "yoy": 100*logs.diff(12),
            "log_level": 100*logs, "raw": levels,
            "daily_mom": 100*logs.sub(np.log(levels.index.days_in_month), axis=0).diff()}
    source = pd.read_csv(RAW / "sa.csv")
    source.DataSeries = source.DataSeries.str.strip()
    sa = source.set_index("DataSeries").T
    sa.index = pd.PeriodIndex(pd.to_datetime(sa.index, format="%Y%b"), freq="M")
    logs = np.log(sa.sort_index().iloc[:, 0].astype(float))
    outcomes["aggregate_sa"] = {"mom": (100*logs.diff()).to_frame("Official SA aggregate"),
                                "yoy": (100*logs.diff(12)).to_frame("Official SA aggregate")}
    weather = pd.read_csv(DATA / "weather.csv", index_col="month")
    weather.index = pd.PeriodIndex(weather.index, freq="M")
    return counts, outcomes, weather, catalog


def design(frame, terms, seasonal_trends=False):
    formula = "1 + " + " + ".join(terms) + " + C(season) + C(year)"
    if seasonal_trends:
        formula += " + C(season):trend"
    x = patsy.dmatrix(formula, frame, return_type="dataframe", NA_action="raise")
    if np.linalg.matrix_rank(x) < x.shape[1]:
        # Common linear trend is already spanned by year and month effects.
        keep, rank = [], 0
        for col in x.columns:
            trial = x[keep + [col]].to_numpy()
            new_rank = np.linalg.matrix_rank(trial)
            if new_rank > rank:
                keep.append(col)
                rank = new_rank
        x = x[keep]
    return x


def influence_hac(influence, times, bandwidth, correction=1.0):
    """Calendar Bartlett covariance of coefficient influence rows."""
    influence = np.asarray(influence)
    if influence.ndim == 1:
        influence = influence[:, None]
    grid = np.zeros((int(times.max() - times.min() + 1), influence.shape[1]))
    grid[np.asarray(times - times.min(), int)] = influence
    meat = grid.T @ grid
    for lag in range(1, bandwidth + 1):
        if lag >= len(grid):
            break
        cross = grid[lag:].T @ grid[:-lag]
        meat += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
    return (meat + meat.T) * (correction / 2)


def inference(beta, variance, df):
    assert variance >= -1e-10
    se = np.sqrt(max(0, variance))
    critical = t_dist.ppf(.975, df)
    return {"coefficient": float(beta), "std_error": float(se),
            "ci_lower": float(beta - critical * se), "ci_upper": float(beta + critical * se),
            "p_value": float(2 * t_dist.sf(abs(beta / se), df)) if se > 0 else 0.,
            "effect_0p1c": float(beta * .1), "lower_0p1c": float((beta - critical * se) * .1),
            "upper_0p1c": float((beta + critical * se) * .1)}


def sample_mask(weather, matrices, outcome, sample, terms):
    mask = weather[terms].notna().all(axis=1)
    # Identical dates and an unchanged balanced origin set in the three cuts.
    for panel in PANELS:
        mask &= matrices[panel][outcome].notna().all(axis=1)
    if sample.startswith("full"):
        threshold = "eligible_80" if "80" in sample else "eligible_90"
        mask &= weather[threshold].fillna(False) & weather.year.between(2017, 2025)
    if "pre2020" in sample:
        mask &= weather.year.lt(2020)
    if "endpoints" in sample or "clean" in sample:
        mask &= ~weather.year.between(2020, 2022)
    if "clean" in sample:
        horizon = 12 if outcome == "yoy" else 1 if outcome in ["mom", "daily_mom"] else 0
        mask &= ~pd.Series((weather.index - horizon).year, index=weather.index).between(2020, 2022)
    return mask


class Analysis:
    def __init__(self, counts, outcomes, weather, output_dir=None):
        self.counts, self.outcomes, self.weather = counts, outcomes, weather
        self.output_dir = Path(output_dir) if output_dir else OUT
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results, self.specs, self.samples, self.diagnostics, self.heterogeneity = [], [], [], [], []
        self.fits = {}
        self.lag_rows, self.joint_tests = [], []
        self.coefficients = []

    def estimate(self, panel, outcome, sample, terms=None, variant="main", weights=False,
                 seasonal_trends=False, own_dates=False, support_terms=None, omitted_year=None,
                 exclusion_windows=None):
        terms = terms or (FULL if sample.startswith("full") else BASE)
        source_panel = "aggregate" if panel == "aggregate_sa" else panel
        ymat = self.outcomes[panel][outcome]
        if own_dates:
            mask = self.weather[terms].notna().all(axis=1) & ymat.notna().all(axis=1)
            if sample.startswith("full"):
                mask &= self.weather.eligible_90.fillna(False) & self.weather.year.between(2017, 2025)
        else:
            mask = sample_mask(self.weather, self.outcomes, outcome, sample, support_terms or terms)
            mask &= ymat.notna().all(axis=1)
        if "clean" in sample and variant in ["lags", "lag_matched_current"]:
            for lag in [1, 2, 3]:
                mask &= ~pd.Series((self.weather.index-lag).year, index=self.weather.index).between(2020, 2022)
        if "clean" in sample and variant == "lead":
            mask &= ~pd.Series((self.weather.index+1).year, index=self.weather.index).between(2020, 2022)
        if sample.startswith("full") and variant == "weather_differences":
            horizon = 12 if outcome == "yoy" else 1
            eligibility = self.weather.eligible_90.eq(True)
            mask &= eligibility.shift(horizon, fill_value=False)
        if omitted_year is not None:
            mask &= self.weather.year.ne(omitted_year)
        if exclusion_windows:
            horizon = 12 if outcome == "yoy" else 1 if outcome in ["mom", "daily_mom"] else 0
            for start, end in exclusion_windows:
                # Reject any growth interval that intersects the calendar window.
                mask &= ~((self.weather.index >= pd.Period(start, freq="M")) &
                          (self.weather.index-horizon <= pd.Period(end, freq="M")))
        wf, ym = self.weather.loc[mask], ymat.loc[mask]
        assert len(wf) > 30
        if weights:
            w = pd.read_csv(DATA / f"{source_panel}_weights.csv", index_col=0).iloc[:, 0].reindex(ym.columns)
            y = ym @ w
        else:
            y = ym.mean(axis=1)
        x = design(wf, terms, seasonal_trends)
        model_id = f"{panel}__{outcome}__{sample}__{variant}"
        fitted = native.fit_matrix(y, x, list(x.columns), model_id)
        j = x.columns.get_loc("temp_c" if "temp_c" in terms else terms[0])
        bread = fitted.normalized_cov_params
        influence = np.asarray(x) @ bread[:, j] * fitted.wresid
        t, k = x.shape
        correction = t / (t - k)
        common = {"model_id": model_id, "panel": panel, "outcome": outcome, "sample": sample,
                  "variant": variant, "weighted": weights, "n_months": t, "n_units": ym.shape[1],
                  "n_panel_obs": t * ym.shape[1], "n_temporal_parameters": k, "df_time_resid": t - k,
                  "inference_df": t - 1, "start": str(wf.index.min()), "end": str(wf.index.max()),
                  "mean_y": float(y.mean()), "sd_y": float(y.std()), "r_squared": fitted.rsquared,
                  "temperature_term": x.columns[j], "correction": correction,
                  "calendar_gaps": int(wf.index[-1].ordinal - wf.index[0].ordinal + 1 - t)}
        weather_nuisance = np.delete(np.asarray(x), j, axis=1)
        residual_t = x.iloc[:, j].to_numpy() - weather_nuisance @ np.linalg.lstsq(weather_nuisance, x.iloc[:, j], rcond=None)[0]
        common["sd_residual_temperature"] = float(residual_t.std(ddof=1))
        for lag in [1, 12]:
            residual_series = pd.Series(fitted.wresid, index=wf.index)
            lagged = residual_series.reindex(wf.index-lag).to_numpy()
            valid = np.isfinite(lagged)
            common[f"residual_acf_calendar_{lag}"] = float(np.corrcoef(fitted.wresid[valid], lagged[valid])[0, 1])
        for bandwidth in [12, 24]:
            variance = influence_hac(influence, wf.time.to_numpy(), bandwidth, correction)[0, 0]
            self.results.append({**common, "hac_lags": bandwidth, **inference(fitted.params.iloc[j], variance, t - 1)})
            positions = [x.columns.get_loc(term) for term in terms]
            inf_all = x.to_numpy() @ bread[:, positions] * fitted.wresid[:, None]
            cov_all = influence_hac(inf_all, wf.time.to_numpy(), bandwidth, correction)
            for a, term in enumerate(terms):
                term_inference = inference(fitted.params.iloc[positions[a]], cov_all[a, a], t-1)
                self.coefficients.append({"model_id": model_id, "term": term, "hac_lags": bandwidth,
                    **{key: term_inference[key] for key in ["coefficient", "std_error", "ci_lower", "ci_upper", "p_value"]}})
        self.specs.append({**common, "terms": terms, "seasonal_trends": seasonal_trends,
                           "own_dates": own_dates, "support_terms": support_terms, "omitted_year": omitted_year,
                           "exclusion_windows": exclusion_windows or [],
                           "columns": list(x.columns)})
        for month in wf.index:
            self.samples.append({"model_id": model_id, "month": str(month)})
        folder = self.output_dir / "designs"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / f"{model_id}.npz", X=x.to_numpy(), y=y.to_numpy(),
                            months=wf.index.astype(str).to_numpy(dtype="U7"), time=wf.time.to_numpy(),
                            beta=fitted.params.to_numpy(), residual=fitted.wresid,
                            columns=np.asarray(x.columns, dtype=str), Y_units=ym.to_numpy())
        self.fits[model_id] = (fitted, x, wf, ym)
        if variant in ["lags", "lead"]:
            selected = ["temp_c"] + ([f"temp_c_lag{i}" for i in [1, 2, 3]] if variant == "lags" else ["temp_lead1"])
            indices = [x.columns.get_loc(name) for name in selected]
            inf = x.to_numpy() @ bread[:, indices] * fitted.wresid[:, None]
            for bandwidth in [12, 24]:
                cov = influence_hac(inf, wf.time.to_numpy(), bandwidth, correction)
                beta = fitted.params.iloc[indices].to_numpy()
                for idx, name in enumerate(selected):
                    self.lag_rows.append({"model_id": model_id, "term": name, "hac_lags": bandwidth,
                                          **inference(beta[idx], cov[idx, idx], t - 1)})
                if variant == "lags":
                    self.lag_rows.append({"model_id": model_id, "term": "sum_lag0_to3", "hac_lags": bandwidth,
                                          **inference(beta.sum(), cov.sum(), t - 1)})
        return model_id

    def profile(self, panel, outcome, sample):
        terms = FULL if sample.startswith("full") else BASE
        mask = sample_mask(self.weather, self.outcomes, outcome, sample, terms)
        wf, ym = self.weather.loc[mask], self.outcomes[panel][outcome].loc[mask]
        x = design(wf, terms)
        models = native.fit_many_matrix(ym.to_numpy(), x.to_numpy(), x.columns,
            f"profiles__{panel}__{outcome}__{sample}", outcome_names=ym.columns)
        j = x.columns.get_loc("temp_c")
        betas = np.array([m.params.iloc[j] for m in models])
        residuals = np.column_stack([m.wresid for m in models])
        influence = (x.to_numpy() @ models[0].normalized_cov_params[:, j])[:, None] * residuals
        t, k = x.shape
        group = f"{panel}__{outcome}__{sample}"
        for bandwidth in [12, 24]:
            covariance = influence_hac(influence, wf.time.to_numpy(), bandwidth, t / (t - k))
            rows = []
            for i, unit in enumerate(ym.columns):
                rows.append({"group": group, "panel": panel, "outcome": outcome, "sample": sample,
                    "unit": unit, "hac_lags": bandwidth, "n_months": t, "df_time_resid": t-k,
                    "mean_y": float(ym.iloc[:, i].mean()), "sd_y": float(ym.iloc[:, i].std()),
                    **inference(betas[i], covariance[i, i], t - 1)})
            p = np.array([r["p_value"] for r in rows])
            order = np.argsort(p)
            adjusted = np.empty(len(p))
            adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
            for row, value in zip(rows, adjusted):
                row["holm_p_value"] = float(value)
            self.heterogeneity.extend(rows)
            contrast = np.column_stack([-np.ones(len(betas)-1), np.eye(len(betas)-1)])
            v = contrast @ covariance @ contrast.T
            delta = contrast @ betas
            rank = int(np.linalg.matrix_rank(v))
            q = len(delta)
            test = {"group": group, "hac_lags": bandwidth, "restrictions": q, "covariance_rank": rank,
                    "status": "available" if rank == q else "unavailable: singular covariance", "p_value": None,
                    "condition_number": float(np.linalg.cond(v)), "residual_time_df": t-k,
                    "high_dimensional_warning": bool(q > (t-k)/2)}
            if rank == q:
                statistic = float(delta @ np.linalg.solve(v, delta) / q)
                test.update(statistic=statistic, p_value=float(f_dist.sf(statistic, q, t-1)))
            self.joint_tests.append(test)
            np.savez_compressed(self.output_dir / "designs" / f"profiles__{group}__{bandwidth}.npz",
                X=x.to_numpy(), Y=ym.to_numpy(), beta=np.column_stack([m.params.to_numpy() for m in models]),
                time=wf.time.to_numpy(), covariance=covariance, residual=residuals, units=np.array(ym.columns, str),
                temperature_index=j, correction=t/(t-k), columns=np.asarray(x.columns, str))

    def run(self):
        for panel in PANELS:
            for outcome in OUTCOMES:
                for sample in ["long_all", "long_pre2020", "long_endpoints", "long_clean", "full_all", "full_endpoints", "full_clean"]:
                    self.estimate(panel, outcome, sample)
            for sample in ["long_pre2020", "long_clean", "full_clean"]:
                terms = FULL if sample.startswith("full") else BASE
                self.estimate(panel, "mom", sample, terms + ["lny_window_days", "lny_lag1"], "lunar")
                self.estimate(panel, "daily_mom", sample, variant="daily")
                self.estimate(panel, "mom", sample, terms, "season_trends", seasonal_trends=True)
                lag_terms = terms + [f"temp_c_lag{i}" for i in [1, 2, 3]]
                self.estimate(panel, "mom", sample, lag_terms, "lags")
                self.estimate(panel, "mom", sample, terms, "lag_matched_current", support_terms=lag_terms)
                self.estimate(panel, "mom", sample, terms + ["temp_lead1"], "lead")
                self.estimate(panel, "mom", sample, terms + ["temp_sq"], "quadratic_at28")
                for outcome, horizon in [("mom", 1), ("yoy", 12)]:
                    self.estimate(panel, outcome, sample, [f"{v}_diff{horizon}" for v in terms], "weather_differences")
                    if panel != "aggregate":
                        self.estimate(panel, outcome, sample, variant="fixed_weights", weights=True)
            for outcome in ["mom", "yoy"]:
                self.estimate(panel, outcome, "full_clean", BASE, "matched_base_controls")
                self.estimate(panel, outcome, "full80_clean", FULL, "coverage80")
                if panel != "aggregate":
                    for sample in ["long_pre2020", "full_clean"]:
                        self.profile(panel, outcome, sample)
            for sample in ["long_pre2020", "full_clean"]:
                terms = FULL if sample.startswith("full") else BASE
                mask = sample_mask(self.weather, self.outcomes, "mom", sample, terms)
                for year in sorted(self.weather.loc[mask, "year"].unique()):
                    self.estimate(panel, "mom", sample, variant=f"loo_{year}", omitted_year=int(year))
            if panel in ["aggregate", "region"]:
                for outcome in OUTCOMES:
                    for sample in ["long_all", "full_all"]:
                        self.estimate(panel, outcome, sample, variant="own_positive_dates", own_dates=True)
        for outcome in ["mom", "yoy"]:
            for sample in ["long_all", "long_pre2020", "long_clean", "full_all", "full_clean"]:
                self.estimate("aggregate_sa", outcome, sample, variant="official_sa")
        self.write_outputs()

    def write_outputs(self):
        pd.DataFrame(self.results).to_csv(self.output_dir / "regression_results.csv", index=False)
        pd.DataFrame(self.coefficients).to_csv(self.output_dir / "all_weather_coefficients.csv", index=False)
        pd.DataFrame(self.samples).to_csv(self.output_dir / "model_samples.csv", index=False)
        if self.heterogeneity:
            pd.DataFrame(self.heterogeneity).to_csv(self.output_dir / "market_slopes.csv", index=False)
        if self.lag_rows:
            pd.DataFrame(self.lag_rows).to_csv(self.output_dir / "lag_results.csv", index=False)
        save_json(self.output_dir / "joint_tests.json", self.joint_tests)
        save_json(self.output_dir / "specifications.json", self.specs)


def stationarity(outcomes, weather):
    rows = []
    for context in ["pre2020_contiguous_reference", "all_history_reference"]:
        for panel in PANELS:
            for outcome in ["mom", "yoy", "log_level"]:
                for unit, raw in outcomes[panel][outcome].items():
                    series = raw[raw.index.year < 2020] if context.startswith("pre2020") else raw
                    series = series.dropna()
                    result = {"context": context, "panel": panel, "variable": outcome, "unit": unit,
                              "n_months": len(series), "status": "not tested: calendar gaps"}
                    if len(series) < 50 or series.index[-1].ordinal - series.index[0].ordinal + 1 != len(series):
                        rows.append(result)
                        continue
                    tested = series - series.groupby(series.index.month).transform("mean")
                    regression = "ct" if outcome == "log_level" else "c"
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        adf = adfuller(tested, maxlag=min(24, len(tested)//2-3), regression=regression, autolag="AIC")
                        kp = kpss(tested, regression=regression, nlags="auto")
                    state = "compatible" if adf[1] < .05 and kp[1] > .05 else "unit-root concern" if adf[1] > .05 and kp[1] < .05 else "mixed or inconclusive"
                    result.update(status=state, adf_stat=float(adf[0]), adf_p=float(adf[1]), adf_lags=int(adf[2]),
                                  kpss_stat=float(kp[0]), kpss_p=float(kp[1]), kpss_lags=int(kp[2]),
                                  deterministic=regression, warnings=" | ".join(str(w.message) for w in captured))
                    rows.append(result)
        for variable in FULL:
            series = weather.loc[weather.index.year < 2020, variable] if context.startswith("pre2020") else weather[variable]
            series = series.dropna()
            result = {"context": context, "panel": "weather", "variable": variable, "unit": variable,
                      "n_months": len(series), "status": "not tested: calendar gaps"}
            if len(series) >= 50 and series.index[-1].ordinal-series.index[0].ordinal+1 == len(series):
                tested = series - series.groupby(series.index.month).transform("mean")
                regression = "ct" if variable in ["temp_c", "rh_pct"] else "c"
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    adf = adfuller(tested, maxlag=min(24, len(tested)//2-3), regression=regression, autolag="AIC")
                    kp = kpss(tested, regression=regression, nlags="auto")
                result.update(status="compatible" if adf[1]<.05 and kp[1]>.05 else "unit-root concern" if adf[1]>.05 and kp[1]<.05 else "mixed or inconclusive",
                    adf_stat=float(adf[0]), adf_p=float(adf[1]), adf_lags=int(adf[2]),
                    kpss_stat=float(kp[0]), kpss_p=float(kp[1]), kpss_lags=int(kp[2]),
                    deterministic=regression, warnings=" | ".join(str(w.message) for w in captured))
            rows.append(result)
    pd.DataFrame(rows).to_csv(OUT / "stationarity.csv", index=False)


def run():
    native.LOG_DIR = OUT / "pyfixest"
    native.reset_fit_log()
    with threadpool_limits(limits=1):
        counts, outcomes, weather, _ = prepare()
        analysis = Analysis(counts, outcomes, weather)
        analysis.run()
        stationarity(outcomes, weather)
        native.write_fit_log()
    print(f"Visitor analysis: {len(analysis.specs)} specifications, separate outputs at {OUT}", flush=True)


if __name__ == "__main__":
    run()
