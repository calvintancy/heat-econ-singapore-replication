"""Create the monthly/quarterly IIP results report, with native ETable output."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regression_tables import tables_for
from inference_display import main_rows
from significance import significance_label,significance_stars,p_format
from sector_readout import sector_readout

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BLUE, RED, GRID = "#1f77b4", "#d62728", "#dbe4eb"


def fmt(value, digits=3):
    return f"{value:.{digits}f}".replace("-", "−")


def interval(row):
    return f"[{fmt(row['ci_lower'], 2)}, {fmt(row['ci_upper'], 2)}]"


def download(path: Path, label: str) -> str:
    mime = "application/json" if path.suffix == ".json" else "text/csv"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<a href="data:{mime};base64,{data}" download="iip_{path.parent.name}_{path.name}">{html.escape(label)} ↓</a>'


def forest(frame: pd.DataFrame, label: str, *, sectors=False) -> str:
    width, left, right, row_height = 1180, 345, 820, 47
    low = np.floor(frame.ci_lower.min())-1
    high = np.ceil(frame.ci_upper.max())+1
    height = 65+len(frame)*row_height
    x = lambda v: left+(v-low)/(high-low)*(right-left)
    ticks = np.linspace(low, high, 6)
    if not any(abs(t) < 1e-9 for t in ticks):
        ticks = np.sort(np.append(ticks, 0))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}"><title>{html.escape(label)}. Dots show coefficients and bars show 95 percent confidence intervals. Filled dots mark p below 0.05; open dots are not significant at 5 percent.</title>']
    for tick in ticks:
        parts.append(f'<line x1="{x(tick):.2f}" x2="{x(tick):.2f}" y1="18" y2="{height-45}" stroke="{GRID if tick else "#64798b"}" stroke-width="{.7 if tick else 1.3}"/><text x="{x(tick):.2f}" y="{height-24}" text-anchor="middle" class="tick">{fmt(tick,1)}</text>')
    labels = {"S1": "No weather controls", "S2": "Rainfall control", "S3": "Baseline: rainfall + humidity",
              "S4": "Before COVID", "S5": "Exclude 2020–2022"}
    for i, (_, row) in enumerate(frame.iterrows()):
        y = 35+i*row_height
        name = row.sector.replace(" Cluster", "") if sectors else labels[row.model]
        color = RED if not sectors and row.model == "S3" else BLUE
        parts.append(f'<text x="18" y="{y+4}" class="label">{html.escape(name)}</text><line x1="{x(row.ci_lower):.2f}" x2="{x(row.ci_upper):.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2"/><circle cx="{x(row.coefficient):.2f}" cy="{y}" r="{5.5 if row.p_value<.05 else 4.5}" fill="{color if row.p_value<.05 else "white"}" stroke="{color}" stroke-width="1.5"><title>{html.escape(name)}: {fmt(row.coefficient)}, CI {interval(row)}</title></circle><text x="1160" y="{y+4}" text-anchor="end" class="label">{fmt(row.coefficient,2)}{significance_stars(row.p_value)} · p={html.escape(p_format(row.p_value))} · {"Sig. 5%" if row.p_value<.05 else "10% only" if row.p_value<.1 else "Not sig."}</text>')
    parts.append(f'<text x="{(left+right)/2}" y="{height-2}" text-anchor="middle" class="axis">Growth log points per 1°C · 95% intervals</text></svg>')
    return "".join(parts)


def scatter(path: Path, beta: float, period: str) -> str:
    frame = pd.read_csv(path)
    xmin, xmax = frame.temp_residual.min(), frame.temp_residual.max()
    ymin, ymax = frame.growth_residual.min(), frame.growth_residual.max()
    xpad, ypad = (xmax-xmin)*.06, (ymax-ymin)*.08
    xmin, xmax, ymin, ymax = xmin-xpad, xmax+xpad, ymin-ypad, ymax+ypad
    x = lambda v: 65+(v-xmin)/(xmax-xmin)*890
    y = lambda v: 315-(v-ymin)/(ymax-ymin)*280
    parts = ['<svg viewBox="0 0 1000 365" role="img" aria-label="IIP partial regression, one point per period"><title>Growth and temperature residuals after weather controls and fixed effects. Pandemic periods 2020–2022 are red.</title>']
    for tick in np.linspace(ymin, ymax, 5):
        parts.append(f'<line x1="65" x2="955" y1="{y(tick):.2f}" y2="{y(tick):.2f}" stroke="{GRID}"/><text x="53" y="{y(tick)+4:.2f}" text-anchor="end" class="tick">{fmt(tick,1)}</text>')
    for tick in np.linspace(xmin, xmax, 6):
        parts.append(f'<text x="{x(tick):.2f}" y="337" text-anchor="middle" class="tick">{fmt(tick,1)}</text>')
    parts.append(f'<line x1="{x(xmin):.2f}" x2="{x(xmax):.2f}" y1="{y(beta*xmin):.2f}" y2="{y(beta*xmax):.2f}" stroke="#15334a" stroke-width="1.5"/>')
    for _, row in frame.iterrows():
        covid = 2020 <= int(row[period][:4]) <= 2022
        parts.append(f'<circle cx="{x(row.temp_residual):.2f}" cy="{y(row.growth_residual):.2f}" r="3.5" fill="{RED if covid else BLUE}" fill-opacity=".7" stroke="white" stroke-width=".6"><title>{row[period]}: temperature residual {fmt(row.temp_residual)}°C; growth residual {fmt(row.growth_residual)} log points</title></circle>')
    parts.append('<text x="65" y="16" class="axis">Mean cluster growth residual · log points</text><text x="510" y="361" text-anchor="middle" class="axis">Temperature residual after controls and fixed effects · °C</text></svg>')
    return "".join(parts)


def build_iip_report() -> Path:
    summary = json.loads((ROOT / "data/processed/iip/data_checks.json").read_text())
    sections, cards, markdown = [], [], ["# Heat and manufacturing IIP: monthly and quarterly analysis\n"]
    for frequency, period, bandwidth in [("monthly", "month", 12), ("quarterly", "quarter", 4)]:
        folder = OUT / "iip" / frequency
        results = pd.read_csv(folder / "regression_results.csv")
        main = main_rows(results,frequency)
        main = main[main.term.str.startswith("temp")]
        base = main[main.model.eq("S3")].iloc[0]
        exclusion = main[main.model.eq("S5")].iloc[0]
        title = frequency.capitalize()
        growth_label = "MoM" if frequency == "monthly" else "QoQ"
        cards.append(f'<div class="result-card"><span class="eyebrow">{growth_label} IIP log growth · six clusters</span><div class="big">{fmt(base.coefficient,2)}<small>growth log points /1°C</small></div><p>95% interval {interval(base)} · p = {html.escape(p_format(base.p_value))}</p><p><strong>{significance_label(base.p_value)}</strong></p><p class="muted">{base.start}–{base.end} · {int(base[f"n_{period}s"]):,} {period}s · {int(base.n_obs):,} observations</p></div>')
        native_main, native_robustness = tables_for(folder / "regression_results.csv", f"{growth_label} IIP log growth (SA)", frequency, f"iip-{frequency}")
        sector = pd.read_csv(folder / "sector_slopes.csv").sort_values("coefficient")
        lag = json.loads((folder / "distributed_lag_sum.json").read_text())["sector_cluster"]
        checks = json.loads((folder / "estimation_checks.json").read_text())
        aggregate = pd.read_csv(folder / "aggregate_results.csv")
        agg = aggregate[aggregate.term.eq("temp_c")].iloc[0]
        inference = results[results.model.eq("S3") & results.term.eq("temp_c")]
        inference_html = "".join(f'<tr><td>{html.escape(r.covariance)}{f" ({int(r.hac_lags)} lags)" if r.hac_lags else ""}</td><td>{fmt(r.std_error)}</td><td>{interval(r)}</td><td>{fmt(r.p_value)}</td></tr>' for _, r in inference.iterrows())
        sections.append(f'''<section class="frequency" id="{frequency}" {"" if frequency == "monthly" else "hidden"}>
<div class="section-heading"><h2>{title} regressions</h2>{download(folder / "regression_results.csv", "Download coefficients")}</div>
<p class="intro">Dependent variable: 100 × {"month-on-month" if frequency == "monthly" else "quarter-on-quarter"} change in log seasonally adjusted IIP. Sector × {"month-of-year" if frequency == "monthly" else "quarter-of-year"} and year fixed effects. Sector-clustered CR1 standard errors across six manufacturing clusters; t(5) inference. Calendar-HAC alternatives are retained below. Growth stationarity is mixed for some series; the complete audit includes published totals, all industries and weather regressors. Nominal significance does not resolve persistence. <a href="stationarity.html">Inspect all stationarity flags →</a></p>
<div class="panel etable-scroll">{native_main}</div>
<details class="panel robustness"><summary>Detailed industries and measurement checks</summary><div class="etable-scroll">{native_robustness}</div><p class="caption">The broad and detailed matched samples begin at {summary['panels'][frequency+'_detailed']['start']}. S8 uses non-seasonally-adjusted IIP growth with the same seasonal and year effects. {"The final column averages monthly temperature equally; the baseline weights it by calendar days." if frequency == 'quarterly' else "No equal-month averaging check is added at monthly frequency, since it would duplicate the baseline exposure."}</p></details>
<div class="section-heading"><h2>Controls and pandemic sensitivity</h2></div><figure class="panel"><div class="chart-scroll">{forest(main[main.model.isin(['S1','S2','S3','S4','S5'])], title+' IIP sensitivity')}</div><figcaption class="caption">Baseline: {fmt(base.coefficient,2)}, CI {interval(base)}. Excluding 2020–2022: {fmt(exclusion.coefficient,2)}, CI {interval(exclusion)}. Sector clusters retain every observed date within each industry; the HAC benchmark preserves calendar gaps.</figcaption></figure>
<div class="section-heading"><h2>Manufacturing clusters</h2>{download(folder/'sector_slopes.csv', 'Download cluster slopes')}</div><figure class="panel"><div class="chart-scroll">{forest(sector, title+' IIP cluster slopes', sectors=True)}</div><figcaption class="caption">Separate temperature slopes in a pooled model, with shared rainfall and humidity controls and the same fixed effects. Sector-specific slopes use calendar-HAC because each slope has support in only one sector cluster. Intervals are pointwise, without adjustment for multiple testing. Filled dots: significant at 5%; open dots: not significant at 5%.</figcaption></figure>{sector_readout(sector,folder/"regression_results.csv")}
<div class="section-heading"><h2>The underlying {period} observations</h2></div><figure class="panel"><div class="chart-scroll">{scatter(folder/'partial_regression_period_means.csv',base.coefficient,period)}</div><figcaption class="caption">One point per {period}, after averaging the sector residuals. Temperature and growth are residualised on the controls and fixed effects. Red points are 2020–2022. Temperature residual SD: {fmt(checks['temperature_residual_sd_c'])}°C.</figcaption></figure>
<div class="two-up"><div class="panel padded"><h3>Inference choices</h3><div class="table-scroll"><table><thead><tr><th>Covariance</th><th>SE</th><th>95% interval</th><th>p</th></tr></thead><tbody>{inference_html}</tbody></table></div></div><div class="panel padded"><span class="eyebrow">Current + {bandwidth} lagged {period}s</span><h3>The cumulative response · sector CR1</h3><div class="lag-value">{fmt(lag['sum_lag_0_to_max'],2)}</div><p>95% interval {interval(lag)}; p = {fmt(lag['p_value'])}.</p><p class="muted">The sum describes the fitted growth response over {bandwidth+1} {period}s to a one-{period} temperature pulse. It is not a permanent-warming estimate.</p>{download(folder/'distributed_lag_results.csv','Download lag coefficients')}</div></div>
<div class="panel padded aggregate"><h3>The separately published manufacturing total</h3><p>Using the official aggregate IIP on the same dates gives a temperature coefficient of <strong>{fmt(agg.coefficient,2)}</strong>, SE {fmt(agg.std_error)}, 95% interval {interval(agg)}, p = {fmt(agg.p_value)}. This uses the official manufacturing aggregate, rather than an average of cluster indices.</p>{download(folder/'aggregate_results.csv','Download aggregate results')}</div>
</section>''')
        markdown += [f"## {title}\n", f"Balanced broad panel: {base.start}–{base.end}, six clusters, {int(base[f'n_{period}s'])} periods, {int(base.n_obs)} observations.\n",
                     f"Baseline temperature coefficient: {base.coefficient:.6f}; SE {base.std_error:.6f}; 95% CI {interval(base)}; p={base.p_value:.6f}. Exclude 2020–2022: {exclusion.coefficient:.6f}, SE {exclusion.std_error:.6f}, p={exclusion.p_value:.6f}.\n",
                     "| Model | Temperature | SE | 95% interval | p |\n|---|---:|---:|---:|---:|\n"]
        markdown += [f"| {r.model}: {r.label} | {r.coefficient:.3f} | {r.std_error:.3f} | {interval(r)} | {r.p_value:.3f} |\n" for _, r in main.iterrows()]
    template = (ROOT / "templates/iip_results.html").read_text()
    report = template.replace("__RESULT_CARDS__", "".join(cards)).replace("__FREQUENCY_SECTIONS__", "\n".join(sections))
    report = report.replace("__SOURCE_RECONCILIATION__", f"{summary['official_quarterly_reconciliation_cells']:,} published quarterly observations reconcile within {summary['official_quarterly_max_rounding_gap']:.6f} index points, consistent with source rounding.")
    path = OUT / "iip_results.html"
    path.write_text(report)
    markdown += ["\n## Construction and interpretation\n", "Current linked IIP, 2025=100, SSIC 2025. Broad six-cluster levels begin1992Jan and detailed18-industry levels begin1996Jan. Monthly SA and NSA: SingStat M355382/M355381. Published quarterly NSA: M355391. Quarterly SA is constructed as the arithmetic average of monthly SA levels; growth is taken after aggregation, not averaged from monthly growth. All retained quarters contain three months. Weather uses the existing NEA Changi mean temperature, rainfall total and mean humidity, aggregated with the VA project's rules. Monthly controls use their published monthly readings.\n",
                     "Year and industry-specific season effects preserve within-year national temperature variation. Inference uses12monthly or4quarterly Bartlett lags, with24/8lag checks and contemporaneous period-clustered comparisons. P-values use t(T−1); finite-sample factor T/(T−1)×(N−1)/(N−K). These are conditional associations with potential quarterly/monthly macro confounding; not causal heat impacts or permanent climate damage. The equal-cluster panel is distinct from the official aggregate manufacturing IIP. Growth coefficients at different frequencies are not directly comparable.\n",
                     "\nSources: https://tablebuilder.singstat.gov.sg/table/TS/M355381 ; https://tablebuilder.singstat.gov.sg/table/TS/M355382 ; https://tablebuilder.singstat.gov.sg/table/TS/M355391 ; https://www.edb.gov.sg/en/manufacturing-statistics .\n"]
    (OUT / "iip_results.md").write_text("\n".join(markdown))
    print(f"Wrote {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    build_iip_report()
