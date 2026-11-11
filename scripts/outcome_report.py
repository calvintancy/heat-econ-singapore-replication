"""Render clearly labelled growth and level comparisons through the existing ETable layout."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd
from lxml import html as lxml_html

from outcome_analysis import BASE, CONTEXTS, OUT, ROOT
from regression_tables import table_fragment


def download(path: Path, label: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    mime = "text/csv" if path.suffix == ".csv" else "text/markdown"
    return f'<a href="data:{mime};base64,{encoded}" download="{path.parent.name}_{path.name}">{html.escape(label)} ↓</a>'


def comparison(key: str, frequency: str, period: str, control_set: str) -> str:
    folder = OUT / key / control_set
    results = pd.read_csv(folder / "regression_results.csv")
    bandwidth = 12 if frequency == "monthly" else 4
    current = results[results.term.eq("temp_c") & results.hac_lags.eq(bandwidth)].set_index("model")
    short = "MoM" if frequency == "monthly" else "QoQ"
    unit = "S$ million, chained 2015" if key == "va" else "IIP index points, 2025=100"
    dependents = [f"{short} log growth (SA)", 'YoY log growth (NSA)', "YoY log growth (SA)", "100 × log output (SA)",
                  "100 × log output (SA)", f"Output level ({unit}; SA)"]
    note = (f"{short} growth = 100[ln A(t)−ln A(t−1)]. YoY log growth = 100[ln A(t)−ln A(t−{bandwidth})]. "
            f"Log level = 100 ln A(t); raw level is in {unit}. Growth is not annualised. "
            "Every column uses identical sectors and dates. Temperature and controls enter in current-period levels. "
            "Season-specific time slopes let within-year seasonality change linearly over years; they do not guarantee stationarity.")
    main = table_fragment(folder / "regression_results.csv", dependents, frequency,
        ["G1", 'G2N', "G2", "L1", "L2", "L3"], [f"{short}\ngrowth (SA)", 'YoY\ngrowth (NSA)', "YoY\ngrowth (SA)", "Log\nlevel", "Log level\nseason slopes", "Raw\nlevel"],
        suffix=f"{key}_{control_set}_outcomes", note=note)
    checks = table_fragment(folder / "regression_results.csv",
        ['YoY log growth (NSA)', "YoY log growth (SA)", "100 × log output (SA)", f"Output level ({unit}; SA)", f"Output level ({unit}; SA)"], frequency,
        ['G2NX', "G2X", "L1X", "L3X", "L4"], ['YoY NSA: exclude\n2020–22 endpoints', "YoY SA: exclude\n2020–22 endpoints", "Log level: exclude\n2020–22 endpoints",
         "Raw level: exclude\n2020–22 endpoints", "Raw level\nseason slopes"],
        suffix=f"{key}_{control_set}_outcome_checks",
        note=note.replace("Every column uses identical sectors and dates.", "The first four columns exclude 2020–2022 endpoint dates; the final column uses the full common sample.")
             +" Endpoint exclusions leave 2023 YoY observations whose base period is in 2022; this is not a fully pandemic-free sample.")
    exact = table_fragment(folder / "regression_results.csv", ['YoY log growth (NSA)', 'YoY exact % growth (NSA)', "YoY log growth (SA)", "YoY exact % growth (SA)"], frequency,
        ['G2N','G3N',"G2", "G3"], ['NSA · log\nratio','NSA · exact\npercentage',"SA · log\nratio", "SA · exact\npercentage"], suffix=f"{key}_{control_set}_yoy_exact",
        note=f"Both outcomes compare A(t) with A(t−{bandwidth}), with identical current weather controls, sectors and dates. The exact percentage and log growth differ for large changes.")
    row = current.loc["G2"]
    cov_rows = []
    for model_id, name in [('G2N','YoY log growth (NSA)'),("G2", "YoY log growth (SA)"), ("L1", "Log level"), ("L3", "Raw level")]:
        low = current.loc[model_id]
        high = results[results.model.eq(model_id) & results.term.eq("temp_c") & results.hac_lags.eq(2*bandwidth)].iloc[0]
        cov_rows.append(f'<tr><td>{name}</td><td>{low.coefficient:.3f}</td><td>{low.std_error:.3f}</td><td>{high.std_error:.3f}</td>'
                        f'<td>[{high.ci_lower:.3f}, {high.ci_upper:.3f}]</td></tr>')
    inference = ('<table class="compact"><thead><tr><th>Outcome</th><th>Temperature β</th><th>SE: one-year bandwidth</th>'
                 '<th>SE: two-year bandwidth</th><th>Two-year bandwidth 95% interval</th></tr></thead><tbody>'+"".join(cov_rows)+'</tbody></table>')
    warning = ('<p class="callout"><strong>Short weather sample.</strong> Wind and PM2.5 require the retained 2017–2025 dates. '
               'Quarterly models have only 29 weather periods and 12 residual time degrees of freedom, or nine with changing-seasonality slopes. '
               'Endpoint exclusions leave only five residual time degrees of freedom. Nominal stars and intervals are fragile.</p>' if control_set == "full" and frequency == "quarterly" else "")
    return (f'<p class="notes"><strong>{row.start}–{row.end}</strong> · {int(row[f"n_{period}s"])} {period}s · '
            f'{int(row.n_sectors)} sectors · {int(row.n_obs):,} observations. All six main columns use identical dates. '
            f'{download(folder / "regression_results.csv", "Download all coefficients and both HAC bandwidths")} · '
            f'{download(folder / "common_sample.csv", "Download exact sample")}</p>{warning}'
            f'<div class="etable-scroll">{main}</div>'
            f'<details><summary>Endpoint exclusions and raw-level seasonality</summary><div class="etable-scroll">{checks}</div></details>'
            f'<details><summary>YoY log growth versus the exact percentage change</summary><div class="etable-scroll">{exact}</div></details>'
            f'<details><summary>Serial-correlation sensitivity: double the HAC bandwidth</summary>{inference}'
            '<p class="notes">YoY outcomes overlap across adjacent observations. Bartlett Driscoll–Kraay uses four/eight quarterly or 12/24 monthly lags. '
            'The longer bandwidth changes uncertainty, not the fitted coefficient. Neither bandwidth guarantees valid small-sample inference.</p></details>')


def yoy_weather(key: str, frequency: str, adjustment='NSA') -> str:
    folder = OUT / key / "yoy_weather"
    terms = ["temp_c", "temp_year_mean", "temp_year_change", "temp_year_lag", "rain_100mm", "rh_pct",
             "rain_100mm_year_mean", "rh_pct_year_mean", "rain_100mm_year_change", "rh_pct_year_change",
             "rain_100mm_year_lag", "rh_pct_year_lag"]
    suffix="N" if adjustment=="NSA" else ""
    table = table_fragment(folder / "regression_results.csv", "YoY log growth ("+adjustment+")", frequency,
        ["Y1"+suffix, "Y2"+suffix, "Y3"+suffix, "Y4"+suffix], ["Current\nweather", "Trailing-year\nweather mean", "YoY change\nin weather", "Current +\nlast-year weather"],
        suffix=f"{key}_yoy_weather_{adjustment}", terms=terms,
        note="All four columns use identical dates and the long-sample rainfall/humidity control set. Wind and PM2.5 are not included here. "
             "All weather variables use the transformation indicated in the column. The trailing year is an equal-weight mean of four quarterly or 12 monthly period exposures, including the current period. "
             "Rainfall is mean rainfall per period, not a trailing annual total. YoY changes are differences in °C, mm or humidity points, never Celsius percentage growth.")
    test = json.loads((folder / ("temperature_difference_restriction"+("_nsa" if suffix else "")+".json")).read_text())["sector_cluster"]
    return (f'<details class="weather-design"><summary>For YoY growth ({adjustment}), which temperature variable? Four matched-sample specifications</summary>'
            '<p class="notes">A growth model can use temperature levels. Differencing a static output-level model instead implies differences in temperature and all controls; '
            'time-invariant sector-by-season effects would cancel, while annual effects become annual-effect differences. '
            'Column 3 allows fresh year and sector-by-season effects, so it is a related conditional specification, not an exactly differenced copy of the levels fit. '
            'Column 4 avoids imposing the temperature-difference restriction.</p>'
            f'<div class="etable-scroll">{table}</div><p class="notes">In column 4, current-temperature plus last-year-temperature coefficients sum to '
            f'{test["sum"]:.3f} (sector-CR1 SE {test["std_error"]:.3f}; p={test["p_value"]:.3f} for a zero sum). '
            'This tests only the temperature restriction within that model; it does not test every static-level assumption or establish a growth mechanism. '
            f'{download(folder / "regression_results.csv", "Download YoY weather specifications")}</p></details>')


def build_outcome_report() -> Path:
    panels = []
    for key, frequency, period, _, _ in CONTEXTS:
        label = "Real VA · quarterly" if key == "va" else "IIP · "+frequency
        parts = []
        for control_set in ["full", "weather"]:
            parts.append(f'<div class="control-panel" id="{key}-{control_set}" {"" if control_set == "full" else "hidden"}>'
                         + comparison(key, frequency, period, control_set)+"</div>")
        panels.append(f'<section id="{key}" class="reg-panel" aria-labelledby="tab-{key}" {"" if key == "va" else "hidden"}>'
                      f'<h2>{label}</h2><div class="control-tabs" role="group" aria-label="Weather controls">'
                      f'<button type="button" data-context="{key}" data-controls="weather" aria-pressed="false">Long-history robustness · no wind / PM2.5</button>'
                      f'<button type="button" data-context="{key}" data-controls="full" aria-pressed="true">Baseline · rain, humidity, wind + PM2.5</button></div>'
                      +"".join(parts)+yoy_weather(key, frequency,"NSA")+yoy_weather(key, frequency,"SA")+"</section>")
    template = (ROOT / "templates/regression_tables.html").read_text()
    template = template.replace("The regression tables.", "Growth and output levels.")
    template = template.replace("Models side by side, with coefficients, standard errors, fixed effects and fit statistics. Select an outcome and frequency to inspect the full specification.",
        "The original MoM/QoQ models are retained. Here, compare YoY growth and output levels on identical dates, then inspect alternative temperature definitions. The results depend on the outcome and weather window; they do not establish a stable causal heat effect.")
    template = template.replace("__TABLE_PANELS__", "\n".join(panels))
    doc = lxml_html.fromstring(template)
    doc.xpath('//title')[0].text = "Heat & the Singapore economy · Growth and levels"
    for button in doc.xpath('//button[@data-panel]'):
        button.text = {"va": "Real VA · quarterly", "iip-monthly": "IIP · monthly", "iip-quarterly": "IIP · quarterly"}[button.get("data-panel")]
    definitions = lxml_html.fragment_fromstring('''<div class="definitions"><h2>What is Y?</h2><table class="compact"><thead><tr><th>Outcome</th><th>Definition using labelled SA or NSA output A</th><th>Meaning of temperature coefficient</th></tr></thead><tbody>
<tr><td>MoM / QoQ log growth</td><td>100[ln A(t) − ln A(t−1)]</td><td>Growth log points per 1°C; approximately percentage points</td></tr>
<tr><td>YoY log growth · NSA and SA</td><td>100[ln A(t) − ln A(t−h)]</td><td>YoY growth log points per 1°C; h=12 months or 4 quarters</td></tr>
<tr><td>Log output level</td><td>100 ln A(t)</td><td>Approximately percent difference in output per 1°C</td></tr>
<tr><td>Raw output level</td><td>A(t)</td><td>VA: chained-2015 S$ million per sector; IIP: index points (2025=100)</td></tr></tbody></table>
<p>No growth rate is annualised. A 100→102 change is 1.98 log points or exactly 2%. Quarterly IIP uses the average of three monthly SA index levels before computing growth. YoY uses the same month/quarter last year and is reported for both NSA and SA output. NSA retains the official unadjusted levels; same-season differencing removes a stable seasonal pattern, while moving holidays and changing seasonality can remain. The separate Annual tab aggregates output and weather to calendar years. Original-unit VA coefficients are average sector amounts, not Singapore-wide dollar losses.</p>
<p><strong>Does temperature also need to be YoY?</strong> No. Current temperature asks about hotter periods; a trailing-year mean asks about exposure over the growth window; a YoY temperature difference asks about change from the base period. These are distinct specifications. Temperature changes are in °C, not percentages.</p></div>''')
    nav = doc.xpath('//nav')[0]
    nav.addprevious(definitions)
    notes = doc.xpath('//div[@class="notes"]')[0]
    notes.clear()
    notes.set("class", "notes")
    notes.append(lxml_html.fragment_fromstring('''<div><h2>What the AEA papers do</h2>
<p><a href="https://doi.org/10.1257/mac.4.3.66">Dell, Jones &amp; Olken (2012), AEJ: Macroeconomics</a>: annual output growth on temperature levels and lags. Their framework distinguishes an output-level channel from a growth channel; it does not require temperature change whenever the outcome is growth.</p>
<p><a href="https://doi.org/10.1257/app.20200547">LoPalo (2023), AEJ: Applied Economics</a>: daily interviewer productivity levels on same-day wet-bulb temperature bins. <a href="https://doi.org/10.1257/pol.20180612">Park et al. (2020), AEJ: Economic Policy</a>: test-score levels on heat during the preceding school year. These micro studies illustrate level outcomes and exposure windows; they do not validate a macro VA-level regression.</p>
<p><a href="https://doi.org/10.1257/aer.97.1.354">Deschênes &amp; Greenstone (2007), AER</a>: dollar profits per acre on growing-season degree-days and rainfall, with county and state-year effects. Its numerical findings were subsequently corrected in the 2012 AER exchange. We use it only as an example of a level specification.</p>
<h2>Interpretation and limits</h2><p>The log-level model asks whether output is higher or lower in hotter periods after the stated controls; it does not estimate a permanent growth effect. Annual and seasonal effects remove specified trends and seasonality, but do not guarantee stationary residuals or remove all national shocks. Changing-seasonality slopes are a limited descriptive sensitivity, not a stationarity test. Sector-specific linear trends would not be a useful test of the common-temperature coefficient in this balanced panel: the common linear trend is already spanned by year and season effects.</p>
<p>A YoY log change sums four quarterly or 12 monthly log changes. If heat affects each period’s growth, the accumulated exposure may matter; the trailing-year mean column represents that window. A static log-level model instead implies a difference in all weather variables when differenced. The paired-temperature column relaxes the imposed opposite-coefficient restriction. Neither is a universal test of all dynamic effects.</p>
<p>Fixed effects remain sector × month-of-year and year for monthly IIP, or sector × quarter-of-year and year for quarterly VA/IIP. Full date effects would absorb national weather. Main common-slope inference clusters by sector, with CR1 and t(S−1). Calendar-aware Bartlett Driscoll–Kraay benchmarks use approximate t(T−1) tests and one/two-year bandwidths; repeated sector rows do not increase the number of weather periods. Level persistence and overlapping YoY outcomes make serial dependence especially relevant.</p>
<p>Long-sample controls are Changi rainfall and relative humidity. The shorter comparison adds verified wind and PM2.5 with the existing 90% monthly coverage rules. All transformations are constructed before filtering dates, preserving actual base periods and calendar gaps. Sources and vintages are unchanged from the original reports. These are equal-sector conditional associations, not identified causal climate damages.</p></div>'''))
    literature = ROOT / "notes/aea_temperature_literature.md"
    notes.append(lxml_html.fragment_fromstring('<p>'+download(literature, "Download the four-paper literature and specification note")+'</p>'))
    style = doc.xpath('//style')[0]
    style.text += '''
.definitions{margin:25px 0;background:white;border:1px solid #dbe4eb;border-radius:12px;padding:22px 26px}.definitions h2{font-size:24px;margin:0 0 12px}.definitions p{font-size:12px;color:#64798b;margin:12px 0 0}.compact{width:100%;border-collapse:collapse;font-size:12px}.compact th,.compact td{padding:10px 12px;text-align:left;border-bottom:1px solid #dbe4eb;vertical-align:top}.compact th{color:#64798b;font-weight:600}.control-tabs{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.control-tabs button{border:1px solid #dbe4eb;border-radius:6px;background:white;padding:10px 13px;color:#64798b;cursor:pointer}.control-tabs button[aria-pressed=true]{background:#e9f3fa;border-color:#1f77b4;color:#15334a}.control-panel[hidden]{display:none}details{margin-top:20px;border-top:1px solid #dbe4eb;padding-top:14px}summary{cursor:pointer;font-size:13px;font-weight:600}.callout{font-size:12px;background:#fff4e8;border-left:3px solid #c67b35;padding:13px 16px}.weather-design{margin-top:30px}.notes h2{font-size:24px;color:#15334a;margin-top:25px}
@media(max-width:700px){.definitions{padding:16px 12px;overflow:auto}.compact{min-width:620px}.control-panel>details{overflow:auto}}
'''
    script = doc.xpath('//script')[0]
    script.text += '''
document.querySelectorAll('[data-controls]').forEach(b=>b.addEventListener('click',()=>{const key=b.dataset.context;document.querySelectorAll(`[data-context="${key}"]`).forEach(x=>x.setAttribute('aria-pressed',String(x===b)));['weather','full'].forEach(c=>document.getElementById(`${key}-${c}`).hidden=c!==b.dataset.controls);}));
'''
    path = ROOT / "outputs/outcome_comparisons.html"
    path.write_text("<!doctype html>\n"+lxml_html.tostring(doc, encoding="unicode"))
    print(f"Wrote {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    build_outcome_report()
