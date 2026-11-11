"""Compare one-period growth, YoY growth and output levels without replacing earlier fits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum

from build_panel import parse_va
from cumulative_inference import cluster_sum
from estimate import FE, estimates, fit, read_panel
from iip_analysis import read_iip, source_frame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/outcomes"
PROCESSED = ROOT / "data/processed/outcomes"
BASE = ["temp_c", "rain_100mm", "rh_pct"]
ADDED = ["wind_mps", "pm25_10ug"]
CONTEXTS = [("va", "quarterly", "quarter", "Q", 4),
            ("iip-monthly", "monthly", "month", "M", 12),
            ("iip-quarterly", "quarterly", "quarter", "Q", 4)]


def prepare(key: str, frequency: str, period: str, freq: str, horizon: int) -> pd.DataFrame:
    panel = read_panel("broad") if key == "va" else read_iip(frequency, "broad")
    if key == "va":
        levels = parse_va("va_sa").rename(columns={"va_sa": "output_sa"})
        nsa = parse_va("va_nsa").rename(columns={"va_nsa": "output_nsa"})
        levels = levels.merge(nsa[['sector_id',period,'output_nsa']],on=['sector_id',period],how='left',validate='one_to_one')
        old_outcome = "g_va_sa_qoq"
    else:
        levels = source_frame("iip_sa", "M").rename(columns={"iip_sa": "output_sa"})
        nsa = source_frame("iip_nsa", "M").rename(columns={"iip_nsa": "output_nsa"})
        levels = levels.merge(nsa[['sector_id','month','output_nsa']],on=['sector_id','month'],how='left',validate='one_to_one')
        if frequency == "quarterly":
            levels["quarter"] = levels.month.dt.asfreq("Q")
            levels = levels.groupby(["sector_id", "sector", "quarter"], as_index=False).agg(
                output_sa=("output_sa", "mean"), output_nsa=('output_nsa','mean'), nsa_count=('output_nsa','count'), count=("month", "size"))
            levels = levels[levels["count"].eq(3)]
            levels.loc[levels.nsa_count.ne(3),'output_nsa']=np.nan
        old_outcome = "g_iip_sa"
    levels = levels.sort_values(["sector_id", period]).copy()
    loglevel = np.log(levels.output_sa)
    levels["y_log_level"] = 100*loglevel
    levels["y_short"] = 100*loglevel.groupby(levels.sector_id).diff()
    levels["y_yoy"] = 100*loglevel.groupby(levels.sector_id).diff(horizon)
    lag_level = levels.output_sa.groupby(levels.sector_id).shift(horizon)
    levels["y_yoy_pct"] = 100*(levels.output_sa/lag_level-1)
    nsa_history=levels.set_index(['sector_id',period]).output_nsa
    nsa_base=nsa_history.reindex(pd.MultiIndex.from_arrays([levels.sector_id,levels[period]-horizon])).to_numpy()
    levels['y_yoy_nsa']=100*(np.log(levels.output_nsa)-np.log(nsa_base))
    levels['y_yoy_pct_nsa']=100*(levels.output_nsa/nsa_base-1)
    levels["y_raw_level"] = levels.output_sa
    outcomes = ["y_short", "y_yoy", "y_yoy_pct", "y_yoy_nsa", "y_yoy_pct_nsa", "y_log_level", "y_raw_level"]
    merged = panel.merge(levels[["sector_id", period,'output_nsa']+outcomes], on=["sector_id", period], validate="one_to_one")
    assert np.allclose(merged.y_short, merged[old_outcome], atol=1e-10, rtol=0)
    assert merged.loc[merged.y_yoy.notna(),['y_yoy_nsa','y_yoy_pct_nsa']].notna().all().all(), 'NSA coverage must not change existing SA eligibility'

    weather = pd.read_csv(ROOT / f"data/processed/weather_{frequency}.csv")
    weather[period] = pd.PeriodIndex(weather[period], freq=freq)
    if frequency == "monthly":
        weather = weather.rename(columns={"temperature": "temp_c", "humidity": "rh_pct", "rainfall": "rain_mm"})
    weather["rain_100mm"] = weather.rain_mm/100
    weather = weather.set_index(period).sort_index()[BASE]
    full_index = pd.period_range(weather.index.min(), weather.index.max(), freq=freq)
    weather = weather.reindex(full_index)
    weather.index.name = period
    transformed = pd.DataFrame(index=weather.index)
    for col in BASE:
        lag = weather[col].shift(horizon)
        transformed[f"{col}_year_lag"] = lag
        transformed[f"{col}_year_change"] = weather[col]-lag
        transformed[f"{col}_year_mean"] = weather[col].rolling(horizon, min_periods=horizon).mean()
    transformed = transformed.rename(columns={f"temp_c_year_{kind}": f"temp_year_{kind}" for kind in ["lag", "change", "mean"]})
    merged = merged.merge(transformed.reset_index(), on=period, validate="many_to_one")
    controls = pd.read_csv(ROOT / f"data/processed/extended_controls/{frequency}_controls.csv")
    controls[period] = pd.PeriodIndex(controls[period], freq=freq)
    merged = merged.merge(controls[[period]+ADDED+["eligible_90"]], on=period, how="left", validate="many_to_one")
    merged["eligible_90"] = merged.eligible_90.astype("boolean").fillna(False).astype(bool)
    # A common linear trend is already spanned by year and season FE. These slopes
    # instead let the within-year seasonal pattern evolve over calendar years.
    for season in sorted(merged.season.unique())[1:]:
        merged[f"season_trend_{int(season)}"] = (merged.season.eq(season)*(merged.year-2000)).astype(float)
    merged = merged.sort_values(["sector_id", "time"]).reset_index(drop=True)
    assert not merged.duplicated(["sector_id", period]).any()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PROCESSED / f"{key}.csv", index=False)
    return merged


def balanced_dates(sample: pd.DataFrame, period: str) -> pd.DataFrame:
    start = sample.groupby("sector_id")[period].min().max()
    end = sample.groupby("sector_id")[period].max().min()
    sample = sample[sample[period].between(start, end)].reset_index(drop=True)
    assert sample.groupby(period).sector_id.nunique().eq(sample.sector_id.nunique()).all()
    return sample


def save_fit(sample: pd.DataFrame, *, outcome: str, terms: list[str], model_id: str,
             label: str, period: str, horizon: int, trend: bool = False) -> tuple[list[dict], list[dict], object]:
    trend_terms = [c for c in sample if c.startswith("season_trend_")] if trend else []
    formula = f"{outcome} ~ " + " + ".join(terms+trend_terms) + " + " + FE
    model = fit(sample, formula, label)
    # Equivalent national-time regression size, since weather is common to sectors.
    grouped = sample.groupby("time").first()
    tx = np.column_stack([np.ones(len(grouped)), grouped[terms+trend_terms].to_numpy(),
                          pd.get_dummies(grouped.year, drop_first=True, dtype=float),
                          pd.get_dummies(grouped.season, drop_first=True, dtype=float)])
    residual_df = len(grouped)-np.linalg.matrix_rank(tx)
    if residual_df < 1:
        raise ValueError(f"No residual national-time degrees of freedom: {label}")
    rows = []
    for bandwidth in [horizon, 2*horizon]:
        fitted_rows = estimates(model, sample, model_id, label, formula, terms, lags=bandwidth, period_col=period)
        for row in fitted_rows:
            row.update(outcome=outcome, output_adjustment='NSA' if outcome.endswith('_nsa') else 'SA', season_trend=trend, time_residual_df=int(residual_df))
        rows.extend(fitted_rows)
    coefficients = [{"model": model_id, "term": term, "coefficient": float(value)} for term, value in model.params.items()]
    return rows, coefficients, model


def run_context(key: str, frequency: str, period: str, freq: str, horizon: int) -> dict:
    panel = prepare(key, frequency, period, freq, horizon)
    summary = {}
    for control_set, terms in [("weather", BASE), ("full", BASE+ADDED)]:
        common = panel.dropna(subset=["y_short", "y_yoy", "y_log_level", "y_raw_level"]+terms).copy()
        if control_set == "full":
            common = common[common.eligible_90].copy()
        common = balanced_dates(common, period)
        specs = [("G1", "MoM growth" if frequency == "monthly" else "QoQ growth", "y_short", False),
                 ("G2N", "YoY log growth (NSA)", "y_yoy_nsa", False),
                 ("G2", "YoY log growth (SA)", "y_yoy", False),
                 ("L1", "Output log level × 100", "y_log_level", False),
                 ("L2", "Log level: changing seasonality", "y_log_level", True),
                 ("L3", "Output level in original units", "y_raw_level", False),
                 ("L4", "Raw level: changing seasonality", "y_raw_level", True),
                 ("G3N", "YoY exact percentage growth (NSA)", "y_yoy_pct_nsa", False),
                 ("G3", "YoY exact percentage growth (SA)", "y_yoy_pct", False)]
        folder = OUT / key / control_set
        folder.mkdir(parents=True, exist_ok=True)
        rows, coefficients = [], []
        for model_id, label, outcome, trend in specs:
            fitted, all_coefficients, _ = save_fit(common, outcome=outcome, terms=terms, model_id=model_id,
                label=label, period=period, horizon=horizon, trend=trend)
            rows.extend(fitted)
            coefficients.extend(all_coefficients)
        no_covid = common[~common.year.between(2020, 2022)].reset_index(drop=True)
        for model_id, outcome in [("G2NX", "y_yoy_nsa"), ("G2X", "y_yoy"), ("L1X", "y_log_level"), ("L3X", "y_raw_level")]:
            fitted, all_coefficients, _ = save_fit(no_covid, outcome=outcome, terms=terms, model_id=model_id,
                label="Exclude observations dated 2020–2022", period=period, horizon=horizon)
            rows.extend(fitted)
            coefficients.extend(all_coefficients)
        pd.DataFrame(rows).to_csv(folder / "regression_results.csv", index=False)
        pd.DataFrame(coefficients).to_csv(folder / "all_coefficients.csv", index=False)
        common.to_csv(folder / "common_sample.csv", index=False)
        summary[control_set] = {"start": str(common[period].min()), "end": str(common[period].max()),
                                "periods": common[period].nunique(), "observations": len(common)}
        print(key, control_set, summary[control_set], flush=True)

    # Alternative YoY weather definitions are compared on one common long sample.
    # Applying the full-control coverage rule to entire rolling years would leave
    # too few quarterly periods for the unrestricted weather-lag model.
    lag_terms = ["temp_year_lag", "rain_100mm_year_lag", "rh_pct_year_lag"]
    delta_terms = ["temp_year_change", "rain_100mm_year_change", "rh_pct_year_change"]
    mean_terms = ["temp_year_mean", "rain_100mm_year_mean", "rh_pct_year_mean"]
    common = balanced_dates(panel.dropna(subset=["y_yoy"]+BASE+lag_terms+delta_terms+mean_terms), period)
    folder = OUT / key / "yoy_weather"
    folder.mkdir(parents=True, exist_ok=True)
    rows, coefficients, models = [], [], {}
    weather_specs=[("Y1", "Current-period weather", BASE),
                                   ("Y2", "Trailing-year mean of period weather", mean_terms),
                                   ("Y3", "YoY changes in all weather variables", delta_terms),
                                   ("Y4", "Current and same-period-last-year weather", BASE+lag_terms)]
    for suffix,outcome,adjustment in [('', 'y_yoy','SA'),('N','y_yoy_nsa','NSA')]:
        for model_id,label,terms in weather_specs:
            fitted, all_coefficients, model = save_fit(common, outcome=outcome, terms=terms, model_id=model_id+suffix,
                label=label+' · '+adjustment, period=period, horizon=horizon)
            rows.extend(fitted);coefficients.extend(all_coefficients);models[model_id+suffix]=model
        unrestricted = models['Y4'+suffix]
        cov = cov_nw_groupsum(unrestricted, nlags=horizon,
            time=(common.time-common.time.min()).to_numpy(dtype=int), use_correction="cluster")
        indices = [unrestricted.model.exog_names.index(term) for term in ["temp_c", "temp_year_lag"]]
        beta_sum = float(unrestricted.params.iloc[indices].sum())
        se = float(np.sqrt(cov[np.ix_(indices, indices)].sum()))
        test = {"outcome":outcome,"output_adjustment":adjustment,"null": "Current and lagged temperature coefficients sum to zero; temperature-only restriction, not a joint test of all weather restrictions",
            "sum": beta_sum, "std_error": se,
            "p_value": float(2*student_t.sf(abs(beta_sum/se), common[period].nunique()-1))}
        test["sector_cluster"] = cluster_sum(unrestricted, common, ["temp_c", "temp_year_lag"])
        (folder / ('temperature_difference_restriction'+('_nsa' if suffix else '')+'.json')).write_text(json.dumps(test, indent=2)+"\n")
    pd.DataFrame(rows).to_csv(folder / "regression_results.csv", index=False)
    pd.DataFrame(coefficients).to_csv(folder / "all_coefficients.csv", index=False)
    common.to_csv(folder / "common_sample.csv", index=False)
    summary["yoy_weather"] = {"start": str(common[period].min()), "end": str(common[period].max()),
                              "periods": common[period].nunique(), "observations": len(common)}
    return summary


def run_outcomes() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {key: run_context(key, frequency, period, freq, horizon)
               for key, frequency, period, freq, horizon in CONTEXTS}
    (OUT / "samples.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    run_outcomes()
