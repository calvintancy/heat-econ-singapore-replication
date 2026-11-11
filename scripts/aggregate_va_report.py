"""Native estimation tables for the official aggregate real GVA analysis."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import maketables as mt
import pandas as pd
from lxml import html as lxml_html

from regression_tables import SavedResult, namespace_table_ids
from stationarity_context import model_assessment
from significance import significance_label,p_format

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/aggregate_va"
LABELS = {"temp_c": "Mean air temperature (°C)", "rain_100mm": "Rainfall (100 mm)",
          "rh_pct": "Relative humidity (pp)", "wind_mps": "Wind speed (m/s)", "pm25_10ug": "PM2.5 (10 µg/m³)",
          "temp_year_lag": "Temperature: same quarter last year (°C)",
          "temp_year_change": "YoY temperature difference (°C)",
          "temp_year_mean": "Trailing-year mean temperature (°C)",
          "rain_100mm_year_lag": "Rainfall: same quarter last year (100 mm)",
          "rain_100mm_year_change": "YoY rainfall difference (100 mm)",
          "rain_100mm_year_mean": "Trailing-year mean rainfall (100 mm/quarter)",
          "rh_pct_year_lag": "Humidity: same quarter last year (pp)",
          "rh_pct_year_change": "YoY humidity difference (pp)",
          "rh_pct_year_mean": "Trailing-year mean humidity (pp)"}
LABELS.update({f"temperature_lag_{i}": f"Temperature: {i}-quarter lag (°C)" for i in range(1, 5)})
DEPENDENTS = {"G1": "QoQ log growth (SA)", "G2": "YoY log growth (SA)", 'G2N':'YoY log growth (NSA)',
              "L1": "100 × log real GVA (SA)", "L3": "Real GVA: chained-2015 S$ million (SA)",
              "G3": "YoY exact percentage growth (SA)", 'G3N':'YoY exact percentage growth (NSA)'}


class AggregateResult(SavedResult):
    @property
    def __maketables_fixef_string__(self):
        return "season+year"


def download(path: Path, label: str) -> str:
    mime = "text/csv" if path.suffix == ".csv" else "application/json"
    encoded = base64.b64encode(path.read_bytes()).decode()
    filename = "aggregate_va_"+path.parent.name+"_"+path.name
    return f'<a href="data:{mime};base64,{encoded}" download="{filename}">{html.escape(label)} ↓</a>'


def table(group: str, models: list[str], heads: list[str], dependents: list[str], tag: str, note: str) -> str:
    source = OUT / group / "regression_results.csv"
    rows = pd.read_csv(source)
    rows = rows[rows.hac_lags.eq(4)].copy()
    saved = [AggregateResult(rows[rows.model.eq(model)].copy(), dependent)
             for model, dependent in zip(models, dependents, strict=True)]
    assert all(not result.rows.empty for result in saved)
    observed = list(dict.fromkeys(term.replace("temp_c_unweighted", "temp_c") for result in saved for term in result.rows.term))
    assert rows.estimator.str.startswith("PyFixest").all()
    customs = {
        "Estimator": [str(result.rows.iloc[0].estimator)+" "+str(result.rows.iloc[0].estimator_version) for result in saved],
        "Mean Y (own units)": [f"{result.rows.iloc[0].mean_y:,.3f}" for result in saved],
        "SD of Y (own units)": [f"{result.rows.iloc[0].sd_y:,.3f}" for result in saved],
        "Residual time df": [int(result.rows.iloc[0].time_residual_df) for result in saved],
        "First quarter": [result.rows.iloc[0].start for result in saved],
        "Last quarter": [result.rows.iloc[0].end for result in saved],
        "DJO log-growth form": ["Yes · adapted / rolling horizon" if "log growth" in result.dependent else "No · level / exact-percent sensitivity" for result in saved],
        "Inference": ["Time-series HAC · one national unit"]*len(saved),
    }
    assessments = [model_assessment(source, model) for model in models]
    customs["Y stationarity screen (exact window)"] = [a["y_label"] for a in assessments]
    customs["Y + weather screens (exact window)"] = [a["all_label"] for a in assessments]
    temps=[r.rows[r.rows.term.str.startswith("temp")].iloc[0] for r in saved]
    customs["Temperature p-value"] = [p_format(r.p_value) for r in temps]
    customs["Temperature significance"] = [significance_label(r.p_value) for r in temps]
    notes = ("Bartlett time-series HAC standard errors in parentheses, four quarterly lags; "
             "the singleton-panel Driscoll–Kraay implementation uses T/(T−K). "
             "Saved two-sided t(T−1) p-values determine nominal stars: * p<0.10, ** p<0.05, *** p<0.01. "
             "R² includes fixed effects. Mean Y and SD use the exact estimation dates. Stationarity screens are evidence, not proof; short/gapped windows remain unestablished, and levels are descriptive sensitivities. Nominal significance does not validate a persistent specification. "+note)
    etable = mt.ETable(saved, model_heads=heads, head_order="dh", coef_fmt="b:.3f* \n (se:.3f)",
        signif_code=[.01, .05, .1], keep=observed, exact_match=True, labels=LABELS,
        felabels={"season": "Quarter-of-year FE", "year": "Year FE"}, feorder=["season", "year"],
        model_stats=["N", "r2"], model_stats_labels={"N": "Quarters", "r2": "R² (including FE)"},
        custom_model_stats=customs, stats_order="sc", notes=notes)
    gt = etable.make(type="gt", gt_style={"table_width": "100%", "first_col_width": "230px",
        "table_font_size_all": "13px", "source_notes_font_size": "11px", "data_row_padding": "8px"})
    fragment = namespace_table_ids(gt.as_raw_html(), "aggregate-"+tag)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / f"{tag}.html").write_text(fragment)
    (OUT / "tables" / f"{tag}.tex").write_text(etable.make(type="tex"))
    return '<div class="etable-scroll">'+fragment+'</div><p class="download">'+download(source, "Download estimates, mean Y and both HAC bandwidths")+'</p>'


def growth_context() -> str:
    rows = []
    for group, model, label in [("full", "G1", "Baseline · all controls · QoQ"),
                                ('full','G2N','Baseline · all controls · YoY (NSA)'),
                                ("full", "G2", "Baseline · all controls · YoY (SA)"),
                                ("full", "G1X", "Wind + PM2.5 · exclude 2020–22 · QoQ"),
                                ('full','G2NX','Wind + PM2.5 · exclude 2020–22 · YoY (NSA)'),
                                ("full", "G2X", "Wind + PM2.5 · exclude 2020–22 · YoY (SA)"),
                                ("weather", "G1", "Long-history robustness · no wind/PM2.5 · QoQ"),
                                ('weather','G2N','Long-history robustness · no wind/PM2.5 · YoY (NSA)'),
                                ("weather", "G2", "Long-history robustness · no wind/PM2.5 · YoY (SA)")]:
        frame = pd.read_csv(OUT / group / "regression_results.csv")
        row = frame[frame.model.eq(model)&frame.hac_lags.eq(4)&frame.term.eq("temp_c")].iloc[0]
        rows.append(f'<tr><td>{html.escape(label)}</td><td>{row.mean_y:.3f}</td><td>{row.sd_y:.3f}</td>'
                    f'<td>{row.effect_0p1c:+.3f}</td><td>[{row.ci_lower_0p1c:+.3f}, {row.ci_upper_0p1c:+.3f}]</td>'
                    f'<td>{int(row.n_quarters)}</td></tr>')
    return ('<div class="etable-scroll"><table class="compact"><thead><tr><th>Sample and outcome</th>'
            '<th>Mean growth</th><th>SD of growth</th><th>β × 0.1°C</th><th>95% interval × 0.1°C</th><th>Quarters</th>'
            '</tr></thead><tbody>'+"".join(rows)+'</tbody></table></div>'
            '<p class="notes">Growth entries are log points: 100 × the relevant log ratio, not annualised. '
            'The +0.1°C column rescales the fitted temperature coefficient and its interval; it is not a causal forecast. '
            'Means are the actual sample mean of aggregate growth and are not implied by the temperature regression.</p>')


def bandwidth_table() -> str:
    rows = []
    for group, suffix, label in [("weather", "", "Long sample"), ("full", "X", "Full controls · exclude 2020–22")]:
        source = pd.read_csv(OUT / group / "regression_results.csv")
        for model, outcome in [("G1", "QoQ (SA)"), ('G2N','YoY (NSA)'),("G2", "YoY (SA)"), ("L1", "Log level")]:
            estimates = source[source.model.eq(model+suffix)&source.term.eq("temp_c")].set_index("hac_lags")
            four, eight = estimates.loc[4], estimates.loc[8]
            rows.append(f'<tr><td>{label} · {outcome}</td><td>{four.coefficient:.3f}</td>'
                        f'<td>{four.std_error:.3f}</td><td>{eight.std_error:.3f}</td>'
                        f'<td>[{eight.ci_lower:.3f}, {eight.ci_upper:.3f}]</td></tr>')
    return ('<div class="etable-scroll"><table class="compact"><thead><tr><th>Specification</th><th>β per 1°C</th>'
            '<th>SE: 4 lags</th><th>SE: 8 lags</th><th>8-lag 95% interval</th></tr></thead><tbody>'
            +"".join(rows)+'</tbody></table></div>')


def stationarity_table() -> str:
    summary = pd.read_csv(OUT / "diagnostics/summary.csv")
    first = summary.iloc[0]
    names = {"y_short": "QoQ log growth (SA)", "y_yoy": "YoY log growth (SA)", 'y_yoy_nsa':'YoY log growth (NSA)', "y_log_level": "100 × log GVA (SA)"}
    rows = []
    for row in summary.itertuples():
        relation = html.escape(str(row.kpss_p_value_relation))
        rows.append(f'<tr><td>{names[row.variable]}</td><td>{row.deterministic}</td>'
                    f'<td>{row.adf_statistic:.3f}</td><td>{row.adf_p_value:.3f}</td><td>{int(row.adf_lags)}</td>'
                    f'<td>{row.kpss_statistic:.3f}</td><td>{relation}{row.kpss_p_value:.3f}</td><td>{int(row.kpss_lags)}</td>'
                    f'<td>{html.escape(row.evidence_5pct)}</td></tr>')
    return ('<div class="etable-scroll"><table class="compact"><thead><tr><th>Series</th><th>Terms</th>'
            '<th>ADF statistic</th><th>ADF p</th><th>Lags</th><th>KPSS statistic</th><th>KPSS p</th><th>Lags</th>'
            '<th>Paired evidence at 5%</th></tr></thead><tbody>'+"".join(rows)+'</tbody></table></div>'
            f'<p class="notes">All tests use the contiguous {int(first.n_periods)}-quarter official aggregate histories with SA/NSA as labelled, {first.start}–{first.end}. '
            'c means a constant; ct means a constant and linear trend. ADF has a unit-root null and selects lags by AIC '
            'up to four. KPSS has a level/trend-stationarity null, with automatic bandwidth capped at four; '
            'inequalities preserve the tabulated p-value bounds. Opposite nulls, structural breaks, overlap and lag choice '
            'limit interpretation. No gapped wind/PM2.5 sample or fitted-regression residual is tested. '
            'Neither test validates a causal levels model. '+download(OUT / 'diagnostics/tests.csv', 'Download diagnostic details')+'</p>')


def build_aggregate_va_report() -> Path:
    summary = json.loads((OUT / "summary.json").read_text())
    verification = json.loads((OUT / "verification.json").read_text())
    heads = ["QoQ\nlog growth (SA)", 'YoY\nlog growth (NSA)',"YoY\nlog growth (SA)", "Log\nlevel (SA)", "Raw\nlevel (SA)"]
    ids = ["G1", 'G2N',"G2", "L1", "L3"]
    dependents = [DEPENDENTS[key] for key in ids]
    note = ("QoQ is 100 Δln A; YoY is 100[ln A(t)−ln A(t−4)]; log level is 100 ln A; "
            "raw level is chained-2015 S$ million. Every column in this table uses identical dates. "
            "Temperature and other controls enter in current-period levels. Growth is not annualised.")
    primary = table("full", [key+"X" for key in ids], heads, dependents, "full-excluded", note+
        " Observations dated 2020–2022 are excluded; YoY observations dated 2023 retain 2022 base periods.")
    long = table("weather", ids, heads, dependents, "long", note)
    short_all = table("full", ids, heads, dependents, "full-all", note)
    control_steps = table("matched_controls", ["C1X", "C2X", "C3X", "C4X"],
        ["Rain +\nhumidity", "+ Wind", "+ PM2.5", "+ Both"], [DEPENDENTS["G1"]]*4,
        "control-steps-excluded", "All four columns use the same retained quarters and exclude observations dated 2020–2022.")
    baseline = table("baseline", ["S1", "S2", "S3", "S4", "S5"],
        ["Temperature\nonly", "+ Rainfall", "+ Humidity", "Before\n2020", "Exclude\n2020–22"],
        [DEPENDENTS["G1"]]*5, "baseline", "The final two columns change dates as labelled; exact-sample growth means are reported below each model.")
    measurement = table("baseline", ["S3", "S8", "S9"], ["SA baseline", "NSA growth", "SA · equal-month\ntemperature"],
        ["QoQ real GVA log growth"]*3, "measurement", "The NSA column uses non-seasonally adjusted real levels; other columns use SA levels. All dates match.")
    annual = table("yoy_weather", ["Y1N", "Y2N", "Y3N", "Y4N"],
        ["Current\nweather", "Trailing-year\nmean weather", "YoY weather\ndifferences", "Current +\nlast-year weather"],
        [DEPENDENTS['G2N']]*4, "annual-weather-nsa",
        "All columns use identical dates. All weather variables follow the indicated transformation; annual means are equal-weight means of four quarterly exposures, including the current quarter. Rainfall means are per-quarter amounts, not annual totals.")
    lags = table("distributed_lag", ["D0", "D1"], ["Current only", "Current +\n4 lagged quarters"],
        [DEPENDENTS["G1"]]*2, "distributed-lag", "Columns use the same 174 quarters beginning 1983Q1. All other weather controls are current-period rainfall and humidity.")
    annual_sa=table('yoy_weather',['Y1','Y2','Y3','Y4'],['Current\nweather','Trailing-year\nmean weather','YoY weather\ndifferences','Current +\nlast-year weather'],[DEPENDENTS['G2']]*4,'annual-weather-sa','Retained SA YoY comparison on identical dates and weather definitions.')
    exact = table("full", ['G2NX','G3NX',"G2X", "G3X"], ['NSA · log\nannual ratio','NSA · exact\npercentage',"SA · log\nannual ratio", "SA · exact\npercentage"],
        [DEPENDENTS['G2N'],DEPENDENTS['G3N'],DEPENDENTS["G2"], DEPENDENTS["G3"]], "exact-yoy-excluded", "All columns use the same 19 retained quarters and exclude observations dated 2020–2022.")
    cumulative = json.loads((OUT / "distributed_lag/cumulative_response.json").read_text())
    restriction = json.loads((OUT / "yoy_weather/temperature_difference_restriction_nsa.json").read_text())
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heat and the Singapore economy · Official aggregate real GVA</title>
<style>body{{margin:0;background:#f5f5f1;color:#202722;font:15px/1.6 Arial,sans-serif}}main{{max-width:1280px;margin:auto;padding:38px 25px 70px}}h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;letter-spacing:-1.4px;margin:12px 0 18px}}h2{{font-size:24px;margin:36px 0 10px}}h3{{font-size:18px}}p{{max-width:1100px}}a{{color:#205c47}}.eyebrow{{text-transform:uppercase;letter-spacing:2px;font-size:11px}}.callout{{background:#fff4da;border-left:4px solid #ad7926;padding:15px 20px;margin:22px 0}}.etable-scroll{{overflow-x:auto;margin:15px 0}}details{{background:#fff;border:1px solid #d9dfd7;padding:15px 18px;margin:18px 0}}summary{{cursor:pointer;font-weight:bold}}.notes,.download{{font-size:13px;color:#505c53}}.compact{{width:100%;border-collapse:collapse;background:white;font-size:13px}}.compact th,.compact td{{padding:10px 12px;border-bottom:1px solid #d8dfd7;text-align:right}}.compact th:first-child,.compact td:first-child{{text-align:left}}button{{float:right;cursor:pointer;padding:8px 14px;background:white;border:1px solid #cbd4c8}}footer{{margin-top:40px;border-top:1px solid #cbd4c8;padding-top:20px;font-size:13px}}@media print{{button,summary,.download{{display:none}}details{{display:block}}main{{padding:0}}body{{background:white}}}}</style></head><body><main>
<button type="button" onclick="window.print()">Print / save PDF</button><div class="eyebrow">Heat &amp; the Singapore economy</div>
<h1>Official aggregate real value added.</h1>
<p>This is Singapore's published gross value added at basic prices, series <strong>1.4</strong>, in chained-2015 S$ million. It is an aggregate time series, distinct from the equal-sector panel average. The aggregate includes ownership of dwellings; the earlier 11-sector panel omits that category. Product taxes are outside basic-price GVA. Chain-linked sector levels are never summed to construct this outcome.</p>
<div class="callout"><strong>Stationarity is not established for every specification.</strong> The expanded audit covers every fitted outcome and weather regressor. Growth has mixed results in some series; levels remain descriptive, and the short screened samples have calendar gaps. Nominal significance does not override these concerns. <a href="stationarity.html">Inspect the complete unit-root audit →</a></div><h2>Mean growth and a +0.1°C association</h2>{growth_context()}
<h2>Baseline · rainfall, humidity, wind and PM2.5</h2><p>All 29 eligible quarters during 2017–2025. PM2.5 is included in every baseline column. Calendar gaps make exact-window standard unit-root screens unavailable; this is not a rejection of stationarity.</p>{short_all}
<h2>Robustness · baseline controls, observations dated 2020–2022 excluded</h2>
<div class="callout"><strong>19 retained quarters; only five residual time degrees of freedom.</strong> All five outcomes below include rainfall, humidity, wind and PM2.5. Nominal stars and intervals are fragile in this short time series. Removing endpoint years 2020–2022 does not remove the 2022 base period from YoY observations dated 2023. These estimates are conditional associations, not causal heat damages.</div>
{primary}
<details><summary>Adding wind and PM2.5 on those same excluded-year dates</summary>{control_steps}</details>
<details><summary>Log YoY growth versus the exact annual percentage change</summary>{exact}</details>

<details><summary>Long-history robustness · rainfall and humidity, without wind/PM2.5</summary><p class="notes">{summary['long']['start']}–{summary['long']['end']}, {summary['long']['quarters']} quarters. Current-period temperature and controls enter every column.</p>{long}</details>
<details><summary>Initial specifications, pre-2020 and excluded-year checks</summary>{baseline}</details>
<details><summary>Seasonal adjustment and temperature averaging</summary>{measurement}</details>
<details><summary>Serial-correlation sensitivity: four versus eight HAC lags</summary>{bandwidth_table()}<p class="notes">Annual outcomes overlap and levels may be persistent. The longer bandwidth changes estimated uncertainty, not the fitted coefficient; neither bandwidth guarantees valid small-sample inference.</p></details>
<details><summary>Persistence diagnostics on the contiguous official aggregate history</summary>{stationarity_table()}</details>
<details><summary>For annual growth, compare current, annual-window and changed weather</summary><p class="notes">Temperature need not automatically become a YoY difference. A static output-level equation implies weather differences after annual differencing; accumulating period-growth effects implies the annual exposure window. These are distinct conditional specifications. This comparison uses long-sample rainfall and humidity, with fresh quarter and year effects in every column.</p><h3>YoY growth from NSA levels</h3>{annual}<h3>Retained YoY growth from SA levels</h3>{annual_sa}<p class="notes">In the NSA model, current and year-lagged temperature coefficients in the unrestricted model sum to {restriction['sum']:.3f} (SE {restriction['std_error']:.3f}; p={restriction['p_value']:.3f}). This tests the temperature-only opposite-coefficient restriction, not all weather restrictions or a complete structural model.</p></details>
<details><summary>Temperature dynamics: current plus four lagged quarters</summary>{lags}<p class="notes">The sum of temperature coefficients is {cumulative['sum_lag_0_to_4']:.3f} (SE {cumulative['std_error']:.3f}; 95% interval [{cumulative['ci_lower']:.3f}, {cumulative['ci_upper']:.3f}]). Per +0.1°C pulse the fitted sum is {cumulative['effect_0p1c']:.3f}. It describes accumulated growth responses over five quarters in the fitted model, not permanent-warming damage.</p></details>
<footer><p>Every regression uses quarter-of-year and year fixed effects. There is one national observation per quarter. Full quarter-date effects would absorb national weather. Mean Y and SD are exact-sample descriptive statistics. Log-level coefficients are output log points per °C; raw-level coefficients are chained-2015 S$ million per °C. Neither is automatically a permanent growth effect.</p>
<p>Sources: <a href="https://tablebuilder.singstat.gov.sg/table/TS/M015662">SingStat SA real GVA</a>, <a href="https://tablebuilder.singstat.gov.sg/table/TS/M015661">NSA real GVA</a>; existing verified NEA Changi weather and wind/PM2.5 controls. Tables use native <a href="https://py-econometrics.github.io/maketables/docs/ETable.html">MakeTables ETable</a>, the PyFixest table engine. Independent raw-source and manual-HAC verification passed {verification['model_bandwidth_fits_checked']} model/bandwidth fits.</p>
<p>{download(ROOT / 'data/processed/aggregate_va/panel.csv', 'Download prepared aggregate data')} · {download(OUT / 'verification.json', 'Download independent verification')}</p></footer>
</main></body></html>'''
    document = lxml_html.fromstring(body)
    identifiers = document.xpath('//*[@id]/@id')
    assert len(identifiers) == len(set(identifiers)), "Duplicate aggregate table IDs"
    for node in document.xpath('//*[@headers]'):
        assert all(identifier in identifiers for identifier in node.get("headers").split())
    path = ROOT / "outputs/aggregate_va.html"
    path.write_text(body)
    print(f"Wrote {path.relative_to(ROOT)} with native regression tables and mean-growth context.")
    return path


if __name__ == "__main__":
    build_aggregate_va_report()
