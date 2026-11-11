"""Build and estimate monthly and quarterly manufacturing IIP panels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum

from build_panel import quarter_period
from estimate import FE, WEATHER, estimates, fit
from cumulative_inference import cluster_sum
from sector_temperature import sector_temperature_slopes

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed/iip"
OUT = ROOT / "outputs/iip"
BROAD = [f"1.{i}" for i in range(1, 7)]


def source_frame(name: str, frequency: str) -> pd.DataFrame:
    payload = json.loads((RAW / f"{name}_tabledata.json").read_text())["Data"]
    meta = json.loads((RAW / f"{name}_metadata.json").read_text())["Data"]["records"]
    rows = [{"sector_id": r["seriesNo"], "sector": r["rowText"], "period_label": c["key"],
             name: float(c["value"])} for r in payload["row"] for c in r["columns"]]
    df = pd.DataFrame(rows)
    assert len(df) == meta["total"], f"Incomplete {name} source"
    if frequency == "M":
        df["month"] = pd.to_datetime(df.pop("period_label"), format="%Y %b").dt.to_period("M")
        period = "month"
    else:
        df["quarter"] = df.pop("period_label").map(quarter_period)
        period = "quarter"
    assert not df.duplicated(["sector_id", period]).any()
    assert df[name].gt(0).all()
    for _, group in df.groupby("sector_id"):
        assert list(group.sort_values(period)[period]) == list(pd.period_range(group[period].min(), group[period].max(), freq=frequency))
    return df.sort_values(["sector_id", period]).reset_index(drop=True)


def build_iip() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    monthly = source_frame("iip_sa", "M").merge(source_frame("iip_nsa", "M"),
              on=["sector_id", "sector", "month"], validate="one_to_one")
    ids = set(monthly.sector_id)
    detailed = sorted(s for s in ids if any(s.startswith(b + ".") for b in BROAD)
                      and not any(other.startswith(s + ".") for other in ids))
    assert len(detailed) == 18
    catalog = monthly[["sector_id", "sector"]].drop_duplicates()
    catalog["broad_panel"] = catalog.sector_id.isin(BROAD)
    catalog["detailed_panel"] = catalog.sector_id.isin(detailed)
    catalog.to_csv(PROCESSED / "sector_catalog.csv", index=False)
    monthly["quarter"] = monthly.month.dt.asfreq("Q")
    quarterly = monthly.groupby(["sector_id", "sector", "quarter"], as_index=False).agg(
        iip_sa=("iip_sa", "mean"), iip_nsa=("iip_nsa", "mean"), months=("month", "count"))
    quarterly = quarterly[quarterly.months.eq(3)].sort_values(["sector_id", "quarter"])
    published = source_frame("iip_quarterly_nsa", "Q")
    reconciliation = quarterly.merge(published, on=["sector_id", "sector", "quarter"], validate="one_to_one")
    assert len(reconciliation) == len(published), "Missing published quarterly IIP observations"
    rounding_gap = float((reconciliation.iip_nsa-reconciliation.iip_quarterly_nsa).abs().max())
    assert rounding_gap < .002, f"Quarterly average disagrees with official table: {rounding_gap}"
    weather_monthly = pd.read_csv(ROOT / "data/processed/weather_monthly.csv")
    weather_monthly["month"] = pd.PeriodIndex(weather_monthly.month, freq="M")
    weather_monthly = weather_monthly.rename(columns={"temperature": "temp_c", "humidity": "rh_pct", "rainfall": "rain_mm"})
    weather_monthly = weather_monthly[["month", "temp_c", "rh_pct", "rain_mm"]].dropna()
    weather_quarterly = pd.read_csv(ROOT / "data/processed/weather_quarterly.csv")
    weather_quarterly["quarter"] = pd.PeriodIndex(weather_quarterly.quarter, freq="Q")
    summary = {"official_quarterly_reconciliation_cells": len(reconciliation),
               "official_quarterly_max_rounding_gap": rounding_gap,
               "index_base": "2025=100", "classification": "SSIC 2025", "panels": {}}
    for frequency, levels, weather, period, freq, max_lag in [
        ("monthly", monthly.drop(columns="quarter"), weather_monthly, "month", "M", 12),
        ("quarterly", quarterly, weather_quarterly, "quarter", "Q", 4),
    ]:
        levels = levels.copy().sort_values(["sector_id", period])
        for adjustment in ["sa", "nsa"]:
            levels[f"g_iip_{adjustment}"] = 100*np.log(levels[f"iip_{adjustment}"]).groupby(levels.sector_id).diff()
        weather = weather.copy().sort_values(period)
        for lag in range(1, max_lag+1):
            weather[f"temp_l{lag}"] = weather.temp_c.shift(lag)
        panel = levels.merge(weather, on=period, validate="many_to_one")
        panel["year"] = panel[period].dt.year
        panel["season"] = panel[period].dt.month if frequency == "monthly" else panel[period].dt.quarter
        origin = pd.Period("1982-01" if frequency == "monthly" else "1982Q1", freq=freq).ordinal
        panel["time"] = panel[period].astype("int64") - origin
        panel["rain_100mm"] = panel.rain_mm/100
        for name, sector_ids in [("broad", BROAD), ("detailed", detailed)]:
            frame = panel[panel.sector_id.isin(sector_ids)].dropna(subset=["g_iip_sa", "g_iip_nsa"]).copy()
            start = frame.groupby("sector_id")[period].min().max()
            end = frame.groupby("sector_id")[period].max().min()
            frame = frame[frame[period].between(start, end)].sort_values(["sector_id", period]).reset_index(drop=True)
            assert frame.groupby(period).sector_id.nunique().eq(len(sector_ids)).all()
            assert frame[period].nunique() == len(pd.period_range(start, end, freq=freq))
            frame.to_csv(PROCESSED / f"panel_{frequency}_{name}.csv", index=False)
            summary["panels"][f"{frequency}_{name}"] = {"start": str(start), "end": str(end),
                "n_obs": len(frame), "n_sectors": len(sector_ids), "n_periods": frame[period].nunique()}
        aggregate = panel[panel.sector_id.eq("1")].dropna(subset=["g_iip_sa"]).copy()
        aggregate.to_csv(PROCESSED / f"aggregate_{frequency}.csv", index=False)
    (PROCESSED / "data_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def read_iip(frequency: str, name: str) -> pd.DataFrame:
    period, freq = ("month", "M") if frequency == "monthly" else ("quarter", "Q")
    df = pd.read_csv(PROCESSED / f"{name}_{frequency}.csv" if name == "aggregate" else
                     PROCESSED / f"panel_{frequency}_{name}.csv", dtype={"sector_id": str})
    df[period] = pd.PeriodIndex(df[period], freq=freq)
    return df.sort_values(["sector_id", "time"]).reset_index(drop=True)


def estimate_iip(frequency: str) -> None:
    out = OUT / frequency
    out.mkdir(parents=True, exist_ok=True)
    period, bandwidth, doubled = ("month", 12, 24) if frequency == "monthly" else ("quarter", 4, 8)
    broad, detailed = read_iip(frequency, "broad"), read_iip(frequency, "detailed")
    primary = f"g_iip_sa ~ temp_c + {WEATHER} + {FE}"
    matched = broad[broad[period].between(detailed[period].min(), detailed[period].max())].copy()
    specs = [
        ("S1", "No weather controls", broad, f"g_iip_sa ~ temp_c + {FE}"),
        ("S2", "Rainfall control", broad, f"g_iip_sa ~ temp_c + rain_100mm + {FE}"),
        ("S3", "Baseline: rainfall and humidity", broad, primary),
        ("S4", "Pre-COVID: through 2019", broad[broad.year.lt(2020)].copy(), primary),
        ("S5", "Exclude 2020-2022", broad[~broad.year.between(2020, 2022)].copy(), primary),
        ("S6", "Six clusters: matched detailed sample", matched, primary),
        ("S7", "18 detailed industries: matched sample", detailed, primary),
        ("S8", "NSA IIP log growth", broad, f"g_iip_nsa ~ temp_c + {WEATHER} + {FE}"),
    ]
    if frequency == "quarterly":
        specs.append(("S9", "Equal-weight monthly temperature", broad,
                      f"g_iip_sa ~ temp_c_unweighted + {WEATHER} + {FE}"))
    rows, all_coefficients, fitted = [], [], {}
    for model_id, label, df, formula in specs:
        df = df.reset_index(drop=True)
        model = fit(df, formula, frequency + " IIP " + label)
        temp = "temp_c_unweighted" if model_id == "S9" else "temp_c"
        terms = [temp] + [term for term in ["rain_100mm", "rh_pct"] if term in model.params.index]
        result = estimates(model, df, model_id, label, formula, terms, lags=bandwidth, period_col=period)
        rows.extend(result)
        fitted[model_id] = (model, df, formula)
        all_coefficients.extend({"model": model_id, "term": term, "coefficient": val}
                                for term, val in model.params.items())
        print(f"IIP {frequency} {model_id}: beta={result[0]['coefficient']:.4f}, SE={result[0]['std_error']:.4f}", flush=True)
    base, df, formula = fitted["S3"]
    for lags, covariance in [(0, "Quarter cluster"), (doubled, "Driscoll-Kraay")]:
        results = estimates(base, df, "S3", "Baseline inference check", formula, ["temp_c"],
                            lags=lags, covariance=covariance, period_col=period)
        if frequency == "monthly" and covariance == "Quarter cluster":
            for row in results:
                row["covariance"] = "Month cluster"
        rows.extend(results)
    pd.DataFrame(rows).to_csv(out / "regression_results.csv", index=False)
    pd.DataFrame(all_coefficients).to_csv(out / "all_coefficients.csv", index=False)
    df.to_csv(out / "baseline_sample.csv", index=False)
    temp_residual = fit(df, f"temp_c ~ {WEATHER} + {FE}", "IIP temperature residual").resid
    growth_residual = fit(df, f"g_iip_sa ~ {WEATHER} + {FE}", "IIP growth residual").resid
    partial = df[[period, "sector_id", "sector", "temp_c", "g_iip_sa"]].copy()
    partial["temp_residual"], partial["growth_residual"] = temp_residual, growth_residual
    partial.groupby(period)[["temp_residual", "growth_residual"]].mean().to_csv(out / "partial_regression_period_means.csv")
    (out / "estimation_checks.json").write_text(json.dumps({
        "frequency": frequency, "temperature_residual_sd_c": float(temp_residual.std()),
        "temperature_min_c": float(df.temp_c.min()), "temperature_max_c": float(df.temp_c.max()),
        "n_parameters": base.model.exog.shape[1],
        "fwl_beta_difference": float(np.dot(temp_residual, growth_residual)/np.dot(temp_residual, temp_residual)-base.params["temp_c"]),
    }, indent=2) + "\n")

    lag_terms = ["temp_c"] + [f"temp_l{i}" for i in range(1, bandwidth+1)]
    lag_sample = broad.dropna(subset=lag_terms).reset_index(drop=True)
    lag_formula = "g_iip_sa ~ " + " + ".join(lag_terms) + f" + {WEATHER} + {FE}"
    lag_model = fit(lag_sample, lag_formula, "IIP distributed lag " + frequency)
    lag_rows = estimates(lag_model, lag_sample, "L1", f"Current plus {bandwidth} lagged periods", lag_formula,
                         lag_terms, lags=bandwidth, period_col=period)
    current = fit(lag_sample, primary, "IIP current-only matched lag sample")
    lag_rows.extend(estimates(current, lag_sample, "L0", "Current only on matched lag sample", primary,
                             ["temp_c"], lags=bandwidth, period_col=period))
    times = (lag_sample.time-lag_sample.time.min()).to_numpy(dtype=int)
    cov = cov_nw_groupsum(lag_model, nlags=bandwidth, time=times, use_correction="cluster")
    indices = [lag_model.model.exog_names.index(term) for term in lag_terms]
    beta_sum = float(lag_model.params[lag_terms].sum())
    se = float(np.sqrt(cov[np.ix_(indices, indices)].sum()))
    dof = lag_sample[period].nunique()-1
    critical = student_t.ppf(.975, dof)
    pd.DataFrame(lag_rows).to_csv(out / "distributed_lag_results.csv", index=False)
    (out / "distributed_lag_sum.json").write_text(json.dumps({
        "sum_lag_0_to_max": beta_sum, "max_lag": bandwidth, "std_error": se,
        "sector_cluster": cluster_sum(lag_model,lag_sample,lag_terms,"sum_lag_0_to_max"),
        "ci_lower": beta_sum-critical*se, "ci_upper": beta_sum+critical*se,
        "p_value": float(2*student_t.sf(abs(beta_sum/se), dof)),
        "n_periods": lag_sample[period].nunique(), "start": str(lag_sample[period].min()),
        "end": str(lag_sample[period].max()), "frequency": frequency,
    }, indent=2) + "\n")
    sectors = sector_temperature_slopes(broad, "g_iip_sa", WEATHER, FE, period=period, lags=bandwidth,
                                        label="Manufacturing FE: temperature + industry interactions")
    pd.DataFrame(sectors).to_csv(out / "sector_slopes.csv", index=False)
    aggregate = read_iip(frequency, "aggregate")
    aggregate = aggregate[aggregate[period].between(broad[period].min(), broad[period].max())].reset_index(drop=True)
    model = fit(aggregate, primary, "Official aggregate IIP " + frequency)
    pd.DataFrame(estimates(model, aggregate, "A1", "Official total manufacturing: matched sample", primary,
                           ["temp_c", "rain_100mm", "rh_pct"], lags=bandwidth, period_col=period)).to_csv(out / "aggregate_results.csv", index=False)


def run_iip() -> None:
    build_iip()
    for frequency in ["monthly", "quarterly"]:
        estimate_iip(frequency)


if __name__ == "__main__":
    run_iip()
