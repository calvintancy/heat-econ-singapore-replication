"""Quarterly time-series regressions for the official real aggregate GVA series."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum

from estimate import estimates, fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/aggregate_va"
PROCESSED = ROOT / "data/processed/aggregate_va"
FE = "C(season) + C(year)"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
ADDED = ["wind_mps", "pm25_10ug"]
OUTCOMES = [("G1", "QoQ log growth (SA)", "y_short"),
            ("G2N", "YoY log growth (NSA)", "y_yoy_nsa"),
            ("G2", "YoY log growth (SA)", "y_yoy"),
            ("L1", "100 × log aggregate real GVA (SA)", "y_log_level"),
            ("L3", "Aggregate real GVA: chained-2015 S$ million (SA)", "y_raw_level"),
            ("G3N", "YoY exact percentage growth (NSA)", "y_yoy_pct_nsa"),
            ("G3", "YoY exact percentage growth (SA)", "y_yoy_pct")]


def prepare() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "data/processed/aggregate_gva.csv", dtype={"sector_id": str})
    assert frame.sector_id.eq("1.4").all()
    assert frame.sector.eq("Gross Value Added At Basic Prices").all()
    frame["quarter"] = pd.PeriodIndex(frame.quarter, freq="Q")
    frame = frame.sort_values("quarter").reset_index(drop=True)
    assert not frame.quarter.duplicated().any()
    assert list(frame.quarter) == list(pd.period_range(frame.quarter.min(), frame.quarter.max(), freq="Q"))
    frame["y_short"] = frame.g_va_sa_qoq
    frame["y_yoy"] = frame.g_va_sa_yoy
    frame["y_yoy_pct"] = 100*np.expm1(frame.y_yoy/100)
    frame['y_yoy_nsa']=frame.g_va_nsa_yoy
    frame['y_yoy_pct_nsa']=100*np.expm1(frame.y_yoy_nsa/100)
    frame["y_short_nsa"] = frame.g_va_nsa_qoq
    frame["y_log_level"] = 100*np.log(frame.va_sa)
    frame["y_raw_level"] = frame.va_sa
    weather = pd.read_csv(ROOT / "data/processed/weather_quarterly.csv")
    weather["quarter"] = pd.PeriodIndex(weather.quarter, freq="Q")
    weather = weather.sort_values("quarter").set_index("quarter")
    weather["rain_100mm"] = weather.rain_mm/100
    assert list(weather.index) == list(pd.period_range(weather.index.min(), weather.index.max(), freq="Q"))
    transformed = pd.DataFrame(index=weather.index)
    for col in BASE:
        label = "temp" if col == "temp_c" else col
        transformed[f"{label}_year_lag"] = weather[col].shift(4)
        transformed[f"{label}_year_change"] = weather[col]-weather[col].shift(4)
        transformed[f"{label}_year_mean"] = weather[col].rolling(4, min_periods=4).mean()
    for lag in range(1, 5):
        transformed[f"temperature_lag_{lag}"] = weather.temp_c.shift(lag)
    frame = frame.merge(transformed.reset_index(), on="quarter", validate="one_to_one")
    controls = pd.read_csv(ROOT / "data/processed/extended_controls/quarterly_controls.csv")
    controls["quarter"] = pd.PeriodIndex(controls.quarter, freq="Q")
    frame = frame.merge(controls[["quarter"]+ADDED+["eligible_90", "wind_coverage", "pm_coverage"]],
                        on="quarter", how="left", validate="one_to_one")
    frame["eligible_90"] = frame.eligible_90.eq(True)
    assert frame[[outcome for _, _, outcome in OUTCOMES]+BASE].notna().all().all()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "panel.csv", index=False)
    return frame


def fit_one(sample: pd.DataFrame, outcome: str, terms: list[str], model_id: str,
            label: str) -> tuple[list[dict], list[dict], object]:
    sample = sample.reset_index(drop=True)
    formula = outcome+" ~ "+" + ".join(terms)+" + "+FE
    model = fit(sample, formula, "Aggregate GVA: "+label)
    if model.df_resid < 1:
        raise ValueError(f"No aggregate residual time degrees of freedom: {label}")
    rows = []
    for bandwidth in [4, 8]:
        records = estimates(model, sample, model_id, label, formula, terms, lags=bandwidth)
        for row in records:
            row.update(outcome=outcome, output_adjustment='NSA' if outcome.endswith('_nsa') else 'SA', time_residual_df=int(model.df_resid),
                       n_parameters=model.model.exog.shape[1], mean_y=float(sample[outcome].mean()),
                       sd_y=float(sample[outcome].std()),
                       mean_qoq_log=float(sample.y_short.mean()), sd_qoq_log=float(sample.y_short.std()),
                       mean_yoy_log=float(sample.y_yoy.mean()), sd_yoy_log=float(sample.y_yoy.std()),
                       mean_yoy_pct=float(sample.y_yoy_pct.mean()),
                       effect_0p1c=row["coefficient"]*.1 if row["term"].startswith("temp") else np.nan,
                       se_0p1c=row["std_error"]*.1 if row["term"].startswith("temp") else np.nan,
                       ci_lower_0p1c=row["ci_lower"]*.1 if row["term"].startswith("temp") else np.nan,
                       ci_upper_0p1c=row["ci_upper"]*.1 if row["term"].startswith("temp") else np.nan,
                       covariance_description="Singleton-panel Driscoll–Kraay equals calendar-aware Bartlett time-series HAC")
        rows.extend(records)
    coefficients = [{"model": model_id, "term": term, "coefficient": float(value)}
                    for term, value in model.params.items()]
    return rows, coefficients, model


def save_group(group: str, specs: list[tuple]) -> dict:
    folder = OUT / group
    (folder / "samples").mkdir(parents=True, exist_ok=True)
    rows, coefficients, models, definitions = [], [], {}, []
    for model_id, label, frame, outcome, terms in specs:
        frame = frame.reset_index(drop=True)
        result, all_coefficients, model = fit_one(frame, outcome, terms, model_id, label)
        rows.extend(result)
        coefficients.extend(all_coefficients)
        frame.to_csv(folder / "samples" / f"{model_id}.csv", index=False)
        models[model_id] = model
        definitions.append({"model": model_id, "label": label, "outcome": outcome, "terms": terms,
                            "start": str(frame.quarter.min()), "end": str(frame.quarter.max()),
                            "quarters": frame.quarter.nunique(), "mean_y": float(frame[outcome].mean()),
                            "sd_y": float(frame[outcome].std()), "residual_time_df": int(model.df_resid)})
    pd.DataFrame(rows).to_csv(folder / "regression_results.csv", index=False)
    pd.DataFrame(coefficients).to_csv(folder / "all_coefficients.csv", index=False)
    (folder / "specifications.json").write_text(json.dumps(definitions, indent=2)+"\n")
    return models


def diagnose_aggregate(frame: pd.DataFrame) -> None:
    from stationarity_diagnostics import autocorrelations, paired_evidence, run_tests

    folder = OUT / "diagnostics"
    folder.mkdir(exist_ok=True)
    assert len(frame) > 20 and np.diff(frame.quarter.astype("int64")).min() == 1
    tests, summaries, observations = [], [], []
    for variable, deterministic in [("y_short", "c"), ("y_yoy", "c"), ('y_yoy_nsa','c'), ("y_log_level", "ct")]:
        values = frame[variable].to_numpy()
        metadata = {"context": "aggregate_va", "kind": "outcome", "sector_id": "1.4",
                    "sector": "Official aggregate real GVA", "variable": variable, "window": "long",
                    "start": str(frame.quarter.min()), "end": str(frame.quarter.max()),
                    "n_periods": len(frame), "horizon": 4, "deterministic": deterministic}
        results = run_tests(values, deterministic, 4)
        tests.extend({**metadata, **test} for test in results)
        adf, kpss_result = results
        summaries.append({**metadata, **autocorrelations(values, 4, deterministic),
            "adf_statistic": adf["statistic"], "adf_p_value": adf["p_value"], "adf_lags": adf["used_lags"],
            "kpss_statistic": kpss_result["statistic"], "kpss_p_value": kpss_result["p_value"],
            "kpss_p_value_relation": kpss_result["p_value_relation"], "kpss_lags": kpss_result["used_lags"],
            "evidence_5pct": paired_evidence(adf, kpss_result)})
        observations.extend({**metadata, "period": str(period), "raw_value": float(value),
                             "test_value": float(value), "seasonal_component": 0.}
                            for period, value in zip(frame.quarter, values))
    pd.DataFrame(tests).to_csv(folder / "tests.csv", index=False)
    pd.DataFrame(summaries).to_csv(folder / "summary.csv", index=False)
    pd.DataFrame(observations).to_csv(folder / "observations.csv", index=False)


def run_aggregate_va() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = prepare()
    full = frame[frame.eligible_90].copy()
    no_covid = ~frame.year.between(2020, 2022)
    baseline = [
        ("S1", "Temperature only", frame, "y_short", ["temp_c"]),
        ("S2", "Add rainfall", frame, "y_short", ["temp_c", "rain_100mm"]),
        ("S3", "Baseline: rainfall and humidity", frame, "y_short", BASE),
        ("S4", "Pre-2020 observations", frame[frame.year.lt(2020)], "y_short", BASE),
        ("S5", "Exclude observations dated 2020–2022", frame[no_covid], "y_short", BASE),
        ("S8", "NSA QoQ log growth", frame, "y_short_nsa", BASE),
        ("S9", "Equal-month temperature", frame, "y_short", ["temp_c_unweighted", "rain_100mm", "rh_pct"]),
    ]
    save_group("baseline", baseline)
    for group, sample, terms in [("weather", frame, BASE), ("full", full, BASE+ADDED)]:
        specs = []
        for model_id, label, outcome in OUTCOMES:
            specs.append((model_id, label, sample, outcome, terms))
            specs.append((model_id+"X", label+": exclude 2020–2022 endpoints",
                          sample[~sample.year.between(2020, 2022)], outcome, terms))
        save_group(group, specs)
    controls = []
    for suffix, sample in [("", full), ("X", full[~full.year.between(2020, 2022)])]:
        for number, added in [(1, []), (2, ["wind_mps"]), (3, ["pm25_10ug"]), (4, ADDED)]:
            controls.append((f"C{number}{suffix}", "Matched control comparison"+(
                ": exclude 2020–2022 endpoints" if suffix else ": all retained dates"),
                sample, "y_short", BASE+added))
    save_group("matched_controls", controls)
    lagged = ["temp_year_lag", "rain_100mm_year_lag", "rh_pct_year_lag"]
    changes = ["temp_year_change", "rain_100mm_year_change", "rh_pct_year_change"]
    means = ["temp_year_mean", "rain_100mm_year_mean", "rh_pct_year_mean"]
    annual = frame.dropna(subset=BASE+lagged+changes+means).copy()
    weather_specs=[("Y1", "Current-period weather", BASE),
                                      ("Y2", "Trailing-year mean weather", means),
                                      ("Y3", "YoY changes in all weather", changes),
                                      ("Y4", "Current and same-period-last-year weather", BASE+lagged)]
    models = save_group("yoy_weather", [(model_id+suffix, label+' · '+adjustment, annual, outcome, terms)
        for suffix,outcome,adjustment in [('', 'y_yoy','SA'),('N','y_yoy_nsa','NSA')]
        for model_id,label,terms in weather_specs])
    for suffix,outcome in [('', 'y_yoy'),('N','y_yoy_nsa')]:
        model = models['Y4'+suffix]
        time = (annual.time-annual.time.min()).to_numpy(dtype=int)
        covariance = cov_nw_groupsum(model, nlags=4, time=time, use_correction="cluster")
        indices = [model.model.exog_names.index(term) for term in ["temp_c", "temp_year_lag"]]
        summed = float(model.params.iloc[indices].sum())
        se = float(np.sqrt(covariance[np.ix_(indices, indices)].sum()))
        (OUT / ('yoy_weather/temperature_difference_restriction'+('_nsa' if suffix else '')+'.json')).write_text(json.dumps({
        'outcome':outcome,
        "null": "Current and year-lagged temperature coefficients sum to zero; temperature-only restriction",
        "sum": summed, "std_error": se,
        "p_value": float(2*student_t.sf(abs(summed/se), len(annual)-1))}, indent=2)+"\n")

    temperature_lags = [f"temperature_lag_{i}" for i in range(1, 5)]
    lag_sample = frame.dropna(subset=temperature_lags).copy()
    lag_models = save_group("distributed_lag", [
        ("D0", "Current temperature on the matched lag sample", lag_sample, "y_short", BASE),
        ("D1", "Current plus four lagged quarters", lag_sample, "y_short", BASE+temperature_lags)])
    lag_model = lag_models["D1"]
    times = (lag_sample.time-lag_sample.time.min()).to_numpy(dtype=int)
    covariance = cov_nw_groupsum(lag_model, nlags=4, time=times, use_correction="cluster")
    indices = [lag_model.model.exog_names.index(term) for term in ["temp_c"]+temperature_lags]
    summed = float(lag_model.params.iloc[indices].sum())
    se = float(np.sqrt(covariance[np.ix_(indices, indices)].sum()))
    critical = student_t.ppf(.975, len(lag_sample)-1)
    (OUT / "distributed_lag/cumulative_response.json").write_text(json.dumps({
        "sum_lag_0_to_4": summed, "std_error": se, "ci_lower": summed-critical*se,
        "ci_upper": summed+critical*se, "effect_0p1c": summed*.1, "se_0p1c": se*.1,
        "p_value": float(2*student_t.sf(abs(summed/se), len(lag_sample)-1)),
        "mean_qoq_log": float(lag_sample.y_short.mean()), "sd_qoq_log": float(lag_sample.y_short.std()),
        "quarters": len(lag_sample), "start": str(lag_sample.quarter.min()), "end": str(lag_sample.quarter.max()),
        "interpretation": "Sum of five quarterly growth responses to one quarter of +1°C temperature, conditional on the stated model; not an identified permanent-warming effect."}, indent=2)+"\n")
    summary = {"source_series": "1.4 Gross Value Added At Basic Prices",
               "source_tables": {"sa": "M015662", "nsa": "M015661"},
               "unit": "Million Singapore dollars, chained 2015",
               "construction": "Official published aggregate, never a sum of chain-linked sector series",
               "long": {"start": str(frame.quarter.min()), "end": str(frame.quarter.max()), "quarters": len(frame)},
               "full_controls": {"start": str(full.quarter.min()), "end": str(full.quarter.max()), "quarters": len(full)},
               "full_controls_excluding_2020_22": int((~full.year.between(2020, 2022)).sum()),
               "mean_qoq_log_long": float(frame.y_short.mean()), "mean_yoy_log_long": float(frame.y_yoy.mean())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    diagnose_aggregate(frame)
    print("Aggregate real GVA analysis complete: official series 1.4; long sample "
          f"{len(frame)} quarters, full controls {len(full)} quarters, "
          f"excluding 2020–2022: {summary['full_controls_excluding_2020_22']} quarters.")
    return summary


if __name__ == "__main__":
    run_aggregate_va()
