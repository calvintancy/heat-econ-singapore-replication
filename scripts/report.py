"""Write a source-backed results note and Stata-style exploratory figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
    "axes.linewidth": .6, "axes.edgecolor": "0.2", "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True,
})
BLUE, RED, GRID = "#1f77b4", "#d62728", "#e3e6e8"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def coefficient_plot(results: pd.DataFrame) -> None:
    frame = results[results.model.isin(["S1", "S2", "S3", "S4", "S5"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, r in frame.iterrows():
        color = RED if r.model == "S3" else BLUE
        ax.errorbar(r.coefficient, i, xerr=[[r.coefficient-r.ci_lower], [r.ci_upper-r.coefficient]],
                    fmt="o", color=color, capsize=3, markersize=4, linewidth=1)
    ax.set_yticks(range(len(frame)), frame.label)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", linewidth=.7)
    ax.grid(axis="x", color=GRID, linewidth=.6)
    ax.tick_params(length=2.5, color="0.4")
    ax.set_xlabel("Quarterly VA growth log points per 1°C (95% interval)")
    ax.set_title("Broad sector panel: sensitivity to controls and COVID sample", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "temperature_coefficients.png", dpi=300)
    plt.close(fig)


def partial_plot(beta: float) -> None:
    frame = pd.read_csv(OUT / "baseline_partial_regression.csv")
    q = frame.groupby("quarter")[["temp_residual", "growth_residual"]].mean()
    q["covid"] = q.index.str[:4].astype(int).isin([2020, 2021, 2022])
    q.to_csv(OUT / "partial_regression_quarter_means.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    for covid, label, color in [(False, "Other quarters", BLUE), (True, "2020–2022", RED)]:
        f = q[q.covid.eq(covid)]
        ax.scatter(f.temp_residual, f.growth_residual, s=14, color=color, alpha=.75, label=label, zorder=3)
    x = np.array([q.temp_residual.min(), q.temp_residual.max()])
    ax.plot(x, beta*x, color="0.3", linewidth=1, label="Baseline fitted slope")
    for quarter in ["2020Q2", "2020Q3"]:
        ax.annotate(quarter, (q.loc[quarter, "temp_residual"], q.loc[quarter, "growth_residual"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="0.7", linewidth=.5)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.tick_params(length=2.5, color="0.4")
    ax.set_xlabel("Temperature residual after controls and fixed effects (°C)")
    ax.set_ylabel("Mean sector growth residual (log points)")
    ax.set_title(f"{len(q)} quarterly observations underlying the pooled temperature slope", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "partial_regression.png", dpi=300)
    plt.close(fig)


def sector_plot() -> None:
    frame = pd.read_csv(OUT / "sector_slopes.csv").sort_values("coefficient").reset_index(drop=True)
    names = frame.sector.replace({"Real Estate, Professional Services And Administrative & Support Services":
                                 "Real estate, professional & admin services"})
    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    ax.errorbar(frame.coefficient, np.arange(len(frame)),
                xerr=[frame.coefficient-frame.ci_lower, frame.ci_upper-frame.coefficient],
                fmt="o", color=BLUE, capsize=2, markersize=3, linewidth=.8)
    ax.set_yticks(np.arange(len(frame)), names)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", linewidth=.7)
    ax.grid(axis="x", color=GRID, linewidth=.6)
    ax.tick_params(length=2.5, color="0.4")
    ax.set_xlabel("Quarterly growth log points per 1°C (95% interval)")
    ax.set_title("Exploratory sector slopes", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sector_slopes.png", dpi=300)
    plt.close(fig)


def report() -> None:
    results = pd.read_csv(OUT / "regression_results.csv")
    temp = results[results.term.str.startswith("temp")].copy()
    main = temp[temp.covariance.eq("Sector cluster")]
    base = main[main.model.eq("S3")].iloc[0]
    covid = main[main.model.eq("S5")].iloc[0]
    checks = json.loads((OUT / "estimation_checks.json").read_text())
    data_checks = json.loads((ROOT / "data/processed/data_checks.json").read_text())
    weather_coverage = data_checks["weather_sources"]
    weather_coverage_text = "; ".join(
        f"{name}: {item['observed_start']} to {item['observed_end']} ({item['months']} months)"
        for name, item in weather_coverage.items())
    metadata_mismatches = [name for name, item in weather_coverage.items()
                           if item["metadata_coverage_end"] and item["metadata_coverage_end"][:7] != item["observed_end"]]
    mismatch_note = (" Metadata advertise a newer last month than the retrieved CSV for " +
                     ", ".join(metadata_mismatches) + ". The estimation window uses actual observations only."
                     if metadata_mismatches else "")
    sources = json.loads((ROOT / "data/raw/manifest.json").read_text())
    lag = json.loads((OUT / "distributed_lag_sum.json").read_text())
    coefficient_plot(main)
    partial_plot(base.coefficient)
    sector_plot()
    rows = [[r.model + ": " + r.label, f"{r.coefficient:.3f}",
             f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]", f"{r.p_value:.3f}", f"{r.n_obs:,}", str(r.n_quarters)]
            for r in main.itertuples()]
    table = markdown_table(["Specification", "Temperature coefficient", "95% interval", "p", "Sector-quarters", "Quarters"], rows)
    inference = temp[temp.model.eq("S3")]
    inference_table = markdown_table(["Covariance", "SE", "95% interval", "p"], [
        [r.covariance + (f" ({r.hac_lags} lags)" if r.hac_lags else ""), f"{r.std_error:.3f}",
         f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]", f"{r.p_value:.3f}"] for r in inference.itertuples()])
    weather_controls = results[(results.model.eq("S3")) & results.term.isin(["rain_100mm", "rh_pct"])]
    controls_table = markdown_table(["Control", "Coefficient", "95% interval"], [
        [r.term, f"{r.coefficient:.3f}", f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]"]
        for r in weather_controls.itertuples()])
    significance_note = ("None of these temperature estimates rejects zero at 5%." if main.p_value.ge(.05).all()
                         else "Some pointwise temperature tests reject zero at 5%; compare the full sensitivity table.")
    sector_results = pd.read_csv(OUT / "sector_slopes.csv")
    sector_zero_count = int((sector_results.ci_lower.le(0) & sector_results.ci_upper.ge(0)).sum())
    source_table = markdown_table(["Variable", "Official source", "Definition"], [
        ["Real VA (SA levels)", "[SingStat M015662](https://tablebuilder.singstat.gov.sg/table/TS/M015662)", "Quarterly, S$m chained (2015), SSIC 2020"],
        ["Real VA (NSA levels)", "[SingStat M015661](https://tablebuilder.singstat.gov.sg/table/TS/M015661)", "Same industry classification; robustness outcome"],
        ["Mean air temperature", "[NEA/data.gov.sg](https://data.gov.sg/datasets/d_755290a24afe70c8f9e8bcbf9f251573/view)", "Monthly mean, Changi, °C"],
        ["Rainfall", "[NEA/data.gov.sg](https://data.gov.sg/datasets/d_b16d06b83473fdfcc92ed9d37b66ba58/view)", "Monthly total, Changi, mm"],
        ["Relative humidity", "[NEA/data.gov.sg](https://data.gov.sg/datasets/d_e3b1111a26b2aefef4f386c068d4ca5d/view)", "Monthly mean, Changi, %"],
    ])
    text = fr"""# Heat and Singapore sectoral real value-added growth: first specification

The first panel does **not establish a reliable temperature effect on real VA growth**. The baseline estimate is **{base.coefficient:.2f} growth log points per 1°C**, with a 95% interval of **[{base.ci_lower:.2f}, {base.ci_upper:.2f}]** (p = {base.p_value:.3f}). Excluding 2020–2022 gives **{covid.coefficient:.2f}**, with an interval of **[{covid.ci_lower:.2f}, {covid.ci_upper:.2f}]**. The result is imprecise and sensitive to the pandemic period. These are exploratory conditional associations, not identified causal damage estimates.

## Sample and first regression

The baseline is a balanced panel of **{base.n_sectors} non-overlapping sectors × {base.n_quarters} quarters = {base.n_obs:,} sector-quarter observations**, from **{base.start} to {base.end}**. The economic outcome is sectoral real value added from the GDP-by-industry tables, excluding taxes, ownership of dwellings and aggregate rows. Each sector receives equal weight; this is an average sector-growth relationship, not an estimate of aggregate GDP growth.

The outcome is **100 × [ln(VA at t) − ln(VA at t−1)]**, using SingStat's seasonally adjusted real levels. It is quarterly log growth, not annualised. Log points approximate percentage points for small changes. The first 1982Q1 growth observation uses a real VA level from 1981Q4.

$$g_{{it}} = \beta T_t + \gamma_1 (Rain_t/100) + \gamma_2 RH_t + \alpha_{{i,s(t)}} + \lambda_{{y(t)}} + \varepsilon_{{it}}.$$

- **Temperature:** quarter mean air temperature, day-weighted across the three monthly means.
- **Controls:** quarterly total rainfall (per 100 mm) and day-weighted mean relative humidity (percentage points).
- **Fixed effects:** sector × quarter-of-year (each sector's seasonal pattern), plus calendar year. The implementation expands these as sector FE, quarter-of-year FE and their interaction, plus year FE.
- **Main inference:** Sector-clustered CR1 covariance and t(S−1), S=11. Calendar-aware Driscoll–Kraay is retained as a sensitivity: Bartlett kernel, four quarterly lags. The finite-sample multiplier is T/(T−1) × (N−1)/(N−K); confidence intervals and p-values use a t reference with T−1 degrees of freedom, a conventional approximation. N is sector-quarters, T is unique observed quarters, K includes all fixed-effect coefficients. A quarter-clustered comparator and eight-lag HAC check are also reported.

National temperature is identical across sectors within a quarter. Full **year-quarter FE would absorb it entirely**. Separate year and seasonal effects preserve within-year departures from the usual seasonal pattern. After the baseline controls and FEs, the temperature residual SD is **{checks['temperature_residual_sd_c']:.3f}°C**, so a 1°C coefficient is a scaling convention considerably larger than a typical residual movement. The raw quarter mean spans {checks['temperature_min_c']:.2f}–{checks['temperature_max_c']:.2f}°C.

The {base.n_obs:,} rows contain **{base.n_quarters} national weather observations**. The common temperature slope can also be recovered by regressing the equal-sector quarterly mean growth on the corresponding common controls and year/season effects. Uncertainty must accommodate common sector shocks and serial dependence, rather than treating the replicated temperature values as independent.

## Results and sensitivity

{table}

All common-slope rows use sector × season and year FEs and sector-clustered inference. S2 adds rainfall; S3–S9 include rainfall and humidity. S4 uses only 1982–2019; S5 omits all quarters in 2020–2022, retaining actual calendar spacing in the HAC calculation. S6 and S7 share the same {data_checks['detailed']['n_quarters']} quarters, separating the effect of a later sample from finer sector aggregation. S8 changes to non-seasonally-adjusted quarterly log growth. S9 uses the simple average of monthly temperature means; the humidity construction remains day-weighted. {significance_note}

![Temperature coefficient sensitivity](temperature_coefficients.png)

The baseline weather-control coefficients are conditional on temperature and all fixed effects:

{controls_table}

Rainfall and humidity are correlated with temperature. Adding rainfall materially changes the temperature estimate and its precision; the baseline coefficient describes temperature holding these controls fixed, rather than the combined effect of all changes in thermal conditions.

![Partial regression, quarter means](partial_regression.png)

Each point is one quarter's mean across sectors after removing the baseline controls and fixed effects. Pandemic quarters are highlighted. The fitted slope equals the panel temperature coefficient. The large COVID contraction and rebound are a reason to retain the pandemic-sample checks alongside the full-sample estimate.

## Inference and timing checks

The baseline point estimate is identical across covariance choices:

{inference_table}

The separate current-plus-four-lag model uses a common estimation sample beginning 1983Q1. Its sum of temperature coefficients is {lag['sum_lag_0_to_4']:.2f} (95% interval [{lag['ci_lower']:.2f}, {lag['ci_upper']:.2f}], p = {lag['p_value']:.3f}). It remains imprecise. This sum describes the five-quarter growth response to a one-quarter temperature pulse in that linear model; it does not establish a permanent-warming effect. Individual lag coefficients and a current-only estimate on the same sample are in [distributed_lag_results.csv](distributed_lag_results.csv).

## Exploratory sector slopes

The pooled interaction model allows one temperature slope per broad sector, with the same common weather controls, year FE and sector × season FE. {sector_zero_count} of {len(sector_results)} individual 95% intervals span zero. Intervals are pointwise and do not adjust for multiple testing; the ordering should not be interpreted as a ranking of causal heat exposure.

![Sector slopes](sector_slopes.png)

## Data and construction

{source_table}

The real VA tables were last updated **{sources['va_sa_tabledata.json']['source_last_updated']}** (day/month/year). Raw files, API metadata, retrieval timestamps and SHA-256 hashes are saved under [data/raw](../data/raw/manifest.json). Retrieved weather CSV coverage is {weather_coverage_text}.{mismatch_note} The estimation window ends at {base.end}. Every retained weather quarter has all three months for all three variables; incomplete quarters are excluded, and no missing weather observations are interpolated.

Temperature and humidity are aggregated using calendar days, including leap years. Rainfall totals are summed. Monthly temperature is the actual published monthly mean, not the midpoint of daily maximum and minimum temperatures. Official monthly means are published to one decimal place, which limits exposure precision.

The long panel uses manufacturing, construction, utilities, other goods industries, wholesale & retail trade, transportation & storage, accommodation & food services, information & communications, finance & insurance, real estate/professional/administrative services, and other services industries. Detailed splits begin in 2005Q1, giving 19 sectors and first quarterly growth in 2005Q2. Parent and child industries are never pooled together. [sector_catalog.csv](../data/processed/sector_catalog.csv) records the exact selection. Chain-linked sector values need not add to aggregate real GVA, so no aggregate is constructed by summing them. The separately published aggregate GVA series is retained for future work.

## Reference to the earlier electricity study

The earlier EMA electricity-demand project is documented in [the reference note](../notes/prior_electricity_reference.md). Its July 2026 robustness annex uses half-hourly EMA system demand, 1°C temperature bins, rainfall, wind, humidity and PM2.5; half-hour/day-of-week/year-month/holiday/school-holiday FEs; and calendar-date clustered SE. The canonical weather exposure averages within station × half-hour, then equally across active stations. Frozen wind inputs are bundled with this project's replication release, so the earlier workstation is not a runtime dependency.

This first VA specification carries over the weather-control and calendar-FE approach while adapting the outcome, frequency and inference. It uses a linear quarterly mean-temperature term as requested. Changi provides a much longer, consistent station series; it replaces the earlier varying-station average. Wind and PM2.5 are omitted from this long-sample baseline because the earlier complete weather/control sample only covers 2017–2025. Fine calendar FEs become seasonal and annual effects at quarterly frequency. The earlier national electricity model is a time series; the new model adds an actual sector dimension.

## Interpretation and next identification work

Year FE remove annual macro conditions and annual temperature shifts, but leave quarterly demand shocks, sector-specific shocks, pandemic dynamics and other confounders. This first specification includes weather controls, not a complete model of economic demand. It cannot yet support a causal estimate of heat damage or a projection under permanent warming. A useful next extension is genuinely external demand (partner/world activity), selected with a clear confounding rationale, and sector heat-exposure interactions. Contemporaneous Singapore GDP or exports should not be added mechanically: GDP includes the outcome and exports may themselves respond to heat. The Changi exposure also measures one climate station, not sector-specific workplace heat.

## Reproduce

From the project root: `uv sync --locked`, then `uv run python scripts/run_all.py`. The default uses cached source files. `uv run python scripts/run_all.py --refresh` explicitly retrieves current data vintages and rebuilds all results. Machine-readable coefficients, complete baseline sample, fixed-effect estimates and verification records are in this output folder.
"""
    (OUT / "first_results.md").write_text(text)
    print("Wrote outputs/first_results.md and three figures")


if __name__ == "__main__":
    report()
