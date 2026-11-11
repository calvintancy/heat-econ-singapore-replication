"""Render the separate, self-contained visitor-arrivals research report."""
from __future__ import annotations

import base64
import html
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visitor_analysis import DATA, OUT, RAW, ROOT, save_json

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.linewidth": .6,
    "axes.edgecolor": "0.2", "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})
BLUE, RED, GRAY, GRID = "#1f77b4", "#d62728", "0.62", "#e3e6e8"
LABEL = {"aggregate": "Official total", "region": "Nine regions, equal weights", "country": "42 named markets, equal weights"}


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def make_figures(results, profiles):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    counts = pd.read_csv(DATA / "all_source_counts.csv", index_col="month")
    months = pd.PeriodIndex(counts.index, freq="M").to_timestamp()
    weather = pd.read_csv(DATA / "weather.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
    axes[0].plot(months, counts.iloc[:, 0]/1e6, color=BLUE, lw=1)
    axes[0].set(ylabel="Monthly arrivals (million)", title="The closure and reopening dominate the arrivals history")
    axes[1].plot(months, weather.temp_c, color=BLUE, lw=.9)
    axes[1].set(ylabel="Changi mean temperature (°C)", xlabel="Month")
    for ax in axes:
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01"), color="0.93", zorder=-1)
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.tick_params(length=2.5, color="0.4")
    axes[0].set_ylim(bottom=0)
    fig.savefig(folder / "history.png", dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    chosen = results[(results.variant == "main") & (results.outcome == "mom") & results["sample"].isin(["long_pre2020", "full_clean"])]
    for i, panel in enumerate(LABEL):
        for sample, offset, color, marker in [("long_pre2020", .14, BLUE, "o"), ("full_clean", -.14, RED, "s")]:
            r = chosen[(chosen.panel == panel) & (chosen["sample"] == sample) & (chosen.hac_lags == 12)].iloc[0]
            ax.errorbar(r.effect_0p1c, 2-i+offset,
                xerr=[[r.effect_0p1c-r.lower_0p1c], [r.upper_0p1c-r.effect_0p1c]],
                fmt=marker, color=color, capsize=3, ms=5,
                label=("2008–2019, rain/humidity" if sample == "long_pre2020" else "2017–2025, full controls, strict COVID exclusion") if i==0 else None)
    ax.axvline(0, color="0.35", lw=.7)
    ax.set(yticks=[2,1,0], yticklabels=list(LABEL.values()),
           xlabel="MoM growth log points per +0.1°C (95% calendar-HAC interval, 12 lags)",
           title="The sign changes between the pre-2020 history and the short full-control window")
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5,-.17), fontsize=8)
    fig.savefig(folder / "comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    variants = [("main", "Current weather"), ("lunar", "Lunar New Year: current + lag"),
                ("season_trends", "Changing seasonal pattern"), ("matched_base_controls", "Rain/humidity on same dates"),
                ("official_sa", "Official SA outcome"), ("weather_differences", "Weather changes (different X)"),
                ("lag_matched_current", "Current only on lag-model dates"), ("lags", "Current + three temperature lags")]
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    for i, (variant, label) in enumerate(variants):
        panel = "aggregate_sa" if variant=="official_sa" else "aggregate"
        r = results[(results.panel==panel)&(results.outcome=="mom")&(results["sample"]=="full_clean")&(results.variant==variant)&(results.hac_lags==12)].iloc[0]
        ax.errorbar(r.effect_0p1c, len(variants)-1-i,
            xerr=[[r.effect_0p1c-r.lower_0p1c],[r.upper_0p1c-r.effect_0p1c]], fmt="o", color=BLUE, capsize=2, ms=4)
    ax.set(yticks=range(len(variants)), yticklabels=[v[1] for v in variants][::-1],
        xlabel="MoM growth log points per +0.1°C (95% calendar-HAC interval, 12 lags)",
        title="The short aggregate result weakens with changing seasonality")
    ax.axvline(0, color="0.35", lw=.7)
    ax.grid(axis="x", color=GRID, lw=.6)
    fig.savefig(folder / "robustness.png", dpi=250)
    plt.close(fig)
    for panel in ["region", "country"]:
        chosen = profiles[(profiles.panel==panel)&(profiles.outcome=="mom")&(profiles["sample"]=="full_clean")&(profiles.hac_lags==12)].sort_values("unit")
        fig, ax = plt.subplots(figsize=(8, 4 if panel=="region" else 11), layout="constrained")
        for i, row in enumerate(chosen.itertuples()):
            ax.errorbar(row.effect_0p1c, len(chosen)-i-1,
                xerr=[[row.effect_0p1c-row.lower_0p1c],[row.upper_0p1c-row.effect_0p1c]],
                fmt="o", color=BLUE, markerfacecolor=BLUE if row.holm_p_value<.05 else "white", ms=4, capsize=2)
        ax.set(yticks=range(len(chosen)), yticklabels=list(chosen.unit)[::-1],
            xlabel="MoM growth log points per +0.1°C (pointwise 95% calendar-HAC interval)",
            title="Market profiles: full controls and strict COVID exclusion")
        ax.axvline(0, color="0.35", lw=.7)
        ax.grid(axis="x", color=GRID, lw=.6)
        fig.savefig(folder / f"{panel}_slopes.png", dpi=250)
        plt.close(fig)
    crisis = pd.read_csv(OUT / "crisis/regression_results.csv")
    policies = ["none", "covid", "gfc_covid", "buffered", "stable_pre2020"]
    policy_labels = ["No crisis exclusion", "COVID 2020–2022", "GFC + COVID", "Crisis + recovery buffers", "Pre-COVID after GFC buffer"]
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for sample, color, offset, marker in [("long_all", BLUE, .12, "o"), ("full_all", RED, -.12, "s")]:
        for i, policy in enumerate(policies):
            row = crisis[(crisis.panel=="aggregate")&(crisis.outcome=="mom")&(crisis["sample"]==sample)&(crisis.variant=="crisis_"+policy)&(crisis.hac_lags==12)].iloc[0]
            ax.errorbar(row.effect_0p1c,len(policies)-i-1+offset,
                xerr=[[row.effect_0p1c-row.lower_0p1c],[row.upper_0p1c-row.effect_0p1c]],
                fmt=marker,color=color,ms=4,capsize=2,
                label=("Long controls" if sample=="long_all" else "Full controls, 2017–2025") if i==0 else None)
    ax.set(yticks=range(len(policies)),yticklabels=policy_labels[::-1],
        xlabel="MoM growth log points per +0.1°C (95% calendar-HAC interval, 12 lags)",
        title="Crisis exclusions do not produce a stable negative long-history association")
    ax.axvline(0,color=".35",lw=.7)
    ax.grid(axis="x",color=GRID,lw=.6)
    ax.legend(frameon=False,fontsize=8)
    fig.savefig(folder / "crisis.png",dpi=250)
    plt.close(fig)
    return {p.stem: image_uri(p) for p in folder.glob("*.png")}


def download(path, title=None, filename=None):
    mime = "application/zip" if path.suffix==".zip" else "application/json" if path.suffix==".json" else "text/csv;charset=utf-8" if path.suffix==".csv" else "text/plain;charset=utf-8"
    return f'<a class="download" download="{html.escape(filename or path.name)}" href="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}" data-evidence-path="{html.escape(str(path.relative_to(ROOT)))}">{html.escape(title or path.name)}</a>'


def build():
    r = pd.read_csv(OUT / "regression_results.csv")
    profiles = pd.read_csv(OUT / "market_slopes.csv")
    coefficients = pd.read_csv(OUT / "all_weather_coefficients.csv")
    ppml = pd.read_csv(OUT / "ppml_results.csv")
    diagnostic = pd.read_csv(OUT / "stationarity.csv")
    checks = json.loads((OUT / "data_checks.json").read_text())
    verified = json.loads((OUT / "verification.json").read_text()) if (OUT / "verification.json").exists() else {"status": "pending"}
    figures = make_figures(r, profiles)
    main = r[(r.variant=="main")&(r.outcome=="mom")&(r.hac_lags==12)]
    short = main[(main.panel=="aggregate")&(main["sample"]=="full_clean")].iloc[0]
    pre = main[(main.panel=="aggregate")&(main["sample"]=="long_pre2020")].iloc[0]
    payload = {"results": json.loads(r.to_json(orient="records")),
               "coefficients": json.loads(coefficients.to_json(orient="records")),
               "profiles": json.loads(profiles.to_json(orient="records")),
               "ppml": json.loads(ppml.to_json(orient="records")),
               "lags": json.loads(pd.read_csv(OUT / "lag_results.csv").to_json(orient="records")),
               "joint": json.loads((OUT / "joint_tests.json").read_text()),
               "gt": json.loads((OUT / "gt/tables.json").read_text()),
               "clusterResults": json.loads(pd.read_csv(OUT / "origin_cluster/regression_results.csv").to_json(orient="records")),
               "crisisResults": json.loads(pd.read_csv(OUT / "crisis/regression_results.csv").to_json(orient="records"))}
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    rows=[]
    for panel in LABEL:
        a=main[(main.panel==panel)&(main["sample"]=="long_pre2020")].iloc[0]
        b=main[(main.panel==panel)&(main["sample"]=="full_clean")].iloc[0]
        rows.append(f'<tr><th>{LABEL[panel]}</th><td>{a.effect_0p1c:+.3f}<br><small>[{a.lower_0p1c:+.3f}, {a.upper_0p1c:+.3f}] · p={a.p_value:.3f}</small></td>'
                    f'<td>{b.effect_0p1c:+.3f}<br><small>[{b.lower_0p1c:+.3f}, {b.upper_0p1c:+.3f}] · p={b.p_value:.3f}</small></td></tr>')
    overview_table='<div class="scroll"><table><thead><tr><th>Estimand</th><th>Pre-2020 · rain/humidity<br>143 months</th><th>2017–2025 · full controls<br>strict exclusion · 66 months</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    panels=[]
    for panel,title in [("aggregate","1 · Official aggregate"),("region","2 · Region fixed effects"),("country","3 · Country / origin-market fixed effects")]:
        explanation = {"aggregate": "The published national total measures all covered visitor-arrival events. Its log growth is distinct from the average growth of regional or country series.",
            "region": "Nine nonoverlapping geographic regions enter with equal weights. Regional totals include their own residual markets; the small global ‘Others’ bucket is excluded from the regional FE panel and retained in the accounting audit.",
            "country": "The fixed market set contains 42 named origins. These are places of residence: Hong Kong SAR and Taiwan are separately published, and Belgium & Luxembourg is one combined market. Every named market has equal weight; ‘Other markets’ residuals are excluded."}[panel]
        extra = '' if panel=="aggregate" else f'''<h3>Fully interacted market profiles</h3><p>These robustness estimates allow every weather slope, year effect and seasonal effect to differ by market. Their joint covariance permits correlation across markets and dates. They are distinct from the shared-slope panel above.</p>
        <div class="controls"><label>Profile sample <select class="profile-sample" data-panel="{panel}"><option value="full_clean">Full controls · strict COVID exclusion</option><option value="long_pre2020">Pre-2020 · long controls</option></select></label></div>
        <div id="{panel}-profile-test" class="callout"></div><div id="{panel}-profile-table" class="scroll"></div>
        <details><summary>Show all market intervals for the full-control MoM model</summary><img class="figure" src="{figures[panel+'_slopes']}" alt="All {panel} temperature slopes and pointwise confidence intervals"><p class="caption">Fully origin-specific controls and FE; 66 months; 12-lag HAC. Open markers do not pass the within-table Holm 5% threshold. No market passes it for this MoM view. The chart is fixed to this stated specification; selectors change the table above.</p></details>'''
        if panel=="country":
            extra += '<p class="caution">The short-window country omnibus compares 41 slopes with only 44 residual time degrees of freedom for MoM. Its covariance is ill-conditioned, so its tiny nominal p-value is fragile and does not establish reliable market differences. For short-window YoY the contrast covariance is singular and no omnibus p-value is supplied.</p>'
        inference_options = '<option value="12">Calendar HAC · 12 months</option><option value="24">Calendar HAC · 24 months</option>' + ('<option value="origin">Origin-clustered CR1 · comparison</option>' if panel!='aggregate' else '')
        panels.append(f'''<section class="tab" id="{panel}"><h2>{title}</h2><p>{explanation}</p><div class="controls"><label>Outcome <select class="outcome" data-panel="{panel}"><option value="mom">MoM log growth</option><option value="yoy">YoY log growth</option><option value="log_level">100 × log arrivals · descriptive</option><option value="raw">Raw arrivals · descriptive</option></select></label><label>Inference <select class="bandwidth" data-panel="{panel}">{inference_options}</select></label></div>
        <p id="{panel}-interpretation" class="callout"></p><div id="{panel}-table" class="scroll"></div><p class="caption">Coefficients per +1°C, standard errors in parentheses, nominal pointwise p-values. Month-of-year and year FE are included. Panel columns correspond exactly to origin × month-of-year FE plus common year FE on a balanced market set. ‘All log-admissible dates’ excludes dates invalid for any named market; it does not mean every pandemic month. Exact masks are downloadable.</p>
        <details><summary>+0.1°C effects, sample context and identification diagnostics</summary><div id="{panel}-context" class="scroll"></div></details>{extra}</section>''')
    diagnostic_table=diagnostic.groupby(["panel","variable","status"]).size().rename("series_windows").reset_index().to_html(index=False,escape=True)
    downloads=''.join(download(path,title) for path,title in [
        (OUT/"regression_results.csv","All pooled estimates · both bandwidths"),
        (OUT/"all_weather_coefficients.csv","All weather coefficients"),
        (OUT/"market_slopes.csv","Every market slope and Holm p-value"),
        (OUT/"model_samples.csv","Exact dates of every OLS model"),
        (OUT/"specifications.json","Specification manifest"),
        (OUT/"ppml_results.csv","Zero-retaining PPML estimates"),
        (OUT/"lag_results.csv","Temperature lags and coefficient sums"),
        (OUT/"joint_tests.json","Joint market tests and covariance rank"),
        (OUT/"stationarity.csv","ADF/KPSS results and warnings"),
        (DATA/"all_source_counts.csv","All official counts"),
        (DATA/"market_catalog.csv","Market hierarchy"),
        (DATA/"coverage.csv","Named-market coverage by month"),
        (DATA/"reported_zeros.csv","Every reported zero"),
        (DATA/"hierarchy_reconciliation.csv","Monthly accounting reconciliation"),
        (DATA/"weather.csv","Weather and coverage screens"),
        (DATA/"lunar_new_year.csv","Lunar New Year timing"),
        (RAW/"manifest.json","Visitor source hashes and provenance"),
        (OUT/"input_hashes.json","Frozen input hashes"),
        (OUT/"pyfixest/estimator_manifest.json","Native OLS estimator provenance"),
        (OUT/"ppml/estimator_receipts.json","Native PPML estimator provenance")])
    if (OUT/"verification.json").exists(): downloads += download(OUT/"verification.json","Independent numerical verification")
    for path, title, filename in [
        (OUT/"crisis/regression_results.csv","Crisis and recovery exclusions · all estimates","crisis_regression_results.csv"),
        (OUT/"crisis/all_weather_coefficients.csv","Crisis models · all weather coefficients","crisis_weather_coefficients.csv"),
        (OUT/"crisis/model_samples.csv","Crisis models · exact monthly samples","crisis_model_samples.csv"),
        (OUT/"crisis/policies.json","Calendar exclusion windows and rationale","crisis_policies.json"),
        (OUT/"origin_cluster/regression_results.csv","Origin-clustered CR1 comparison","origin_cluster_results.csv"),
        (OUT/"origin_cluster/all_weather_coefficients.csv","Origin-clustered errors · all weather coefficients","origin_cluster_coefficients.csv"),
        (OUT/"gt/profile_all_weather.csv","Market profiles · every weather coefficient","profile_all_weather.csv"),
        (OUT/"gt/manifest.json","Native R gt table provenance","gt_manifest.json")]:
        downloads += download(path,title,filename)
    if (OUT/"crisis/verification.json").exists():
        downloads += download(OUT/"crisis/verification.json","Independent crisis and CR1 verification","crisis_verification.json")
    archive = OUT/"gt/full_regression_gt_tables.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((OUT/"gt").glob("*.html")):
            bundle.write(path,path.name)
        for path in [OUT/"gt/input.json",OUT/"gt/manifest.json",ROOT/"scripts/visitor_gt.R"]:
            bundle.write(path,path.name)
    downloads += download(archive,"All native gt tables and rendering source")
    template=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Heat and Singapore visitor arrivals</title><style>
    :root{color-scheme:light;--ink:#1c2b38;--muted:#52606a;--blue:#1b567e;--line:#d8e0e5;--wash:#f1f6f8}*{box-sizing:border-box}body{margin:0;background:#f7f8f9;color:var(--ink);font:16px/1.6 Arial,Helvetica,sans-serif}header,main{max-width:1200px;margin:auto;padding:32px 38px}header{padding-top:52px;padding-bottom:20px}h1{font-size:clamp(30px,4vw,46px);line-height:1.12;max-width:960px;letter-spacing:-.8px;margin:10px 0 18px}h2{font-size:28px;line-height:1.25;margin:8px 0 22px}h3{font-size:21px;margin:30px 0 14px}.eyebrow{font-size:12px;letter-spacing:1.5px;text-transform:uppercase;color:var(--blue);font-weight:bold}.subtitle{color:var(--muted);max-width:950px}.summary{border-left:4px solid var(--blue);padding:12px 24px;background:white;margin:28px 0}.summary li{margin:9px 0}.summary h2{font-size:23px;margin-bottom:12px}nav{display:flex;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding:10px 0 20px;position:sticky;top:0;background:#f7f8f9f5;z-index:5}button,select{font:inherit;color:inherit;border:1px solid #bdcad3;border-radius:5px;background:white;padding:9px 13px}button{cursor:pointer;font-size:14px}button.active{background:var(--blue);color:white;border-color:var(--blue)}button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #71acd3;outline-offset:3px}.tab{display:none;padding:28px 0}.tab.active{display:block}.figure{width:100%;max-width:1000px;height:auto;display:block;margin:18px auto;background:white}.caption,small{font-size:13px;color:var(--muted)}.caption{margin:8px 0 25px}.scroll{overflow:auto;background:white;border:1px solid var(--line);border-radius:5px;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:11px 13px;border-bottom:1px solid #e5e9ed;vertical-align:top;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{background:var(--wash);font-weight:bold;color:#223b4b}tbody th{font-weight:normal}tr:last-child td,tr:last-child th{border-bottom:none}.controls{display:flex;gap:24px;flex-wrap:wrap;align-items:end;margin:20px 0}.controls label{font-size:13px;color:var(--muted);display:flex;flex-direction:column;gap:7px}select{font-size:14px;max-width:100%;color:var(--ink)}.callout{padding:15px 20px;background:#eaf2f7;border-left:3px solid #5a87a4}.caution{padding:15px 20px;background:#fff3dc;border-left:3px solid #bd8c37}details{border-top:1px solid var(--line);padding:18px 0;margin-top:15px}summary{font-weight:bold;cursor:pointer;color:var(--blue)}.downloads{display:flex;gap:10px;flex-wrap:wrap}.download{display:inline-block;text-decoration:none;border:1px solid #c8d8e4;border-radius:4px;padding:9px 13px;background:white;color:var(--blue);font-size:13px}.tag{font-size:12px;padding:4px 8px;background:#e7edf1;border-radius:3px;color:#445769}code{font-size:13px;white-space:normal;overflow-wrap:anywhere}a{color:var(--blue)}footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding:25px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}.two>div{min-width:0}.scroll .dataframe{font-size:12px}@media(max-width:700px){header,main{padding:24px 16px}header{padding-top:30px}h2{font-size:24px}.summary{padding:12px 17px}.summary ul{padding-left:20px}.two{display:block}nav{position:static}button{padding:8px 10px}th,td{padding:10px}.controls{gap:12px}}@media print{nav,.controls,.downloads{display:none}.tab{display:block!important;break-before:page}header{padding-top:0}body{background:white}.scroll{overflow:visible}table{font-size:10px}th,td{white-space:normal}.figure{max-width:800px}}
    </style></head><body><header><div class="eyebrow">Separate research report · Singapore · monthly</div><h1>Heat and international visitor arrivals</h1><p class="subtitle">Official aggregate, nine geographic regions and 42 named origin markets. Monthly source history: January 2008–September 2026. These estimates describe conditional weather associations, not an identified causal climate effect.</p>
    <div class="summary"><h2>Executive summary</h2><ul>
    <li><strong>A stable negative heat relationship is not established.</strong> In the 66-month full-control sample excluding pandemic endpoints and growth bases, +0.1°C is associated with <strong>__SHORT__ MoM growth log points</strong> in total arrivals (95% interval __SHORTCI__, nominal p=__SHORTP__). The pre-2020 estimate is __PRE__ (__PRECI__); the sign changes across windows.</li>
    <li><strong>The short-sample result is sensitive to the specification.</strong> Changing seasonal patterns weakens the aggregate coefficient and its nominal significance. Dropping the GFC, COVID and recovery buffers leaves no clear negative association in the long aggregate history. Region/country averages, YoY growth, controls and bandwidths tell different stories.</li>
    <li><strong>The market cuts describe different averages.</strong> Region and country FE give each unit equal weight; they do not estimate a national tourism loss. All markets face the same Singapore weather, so the panels do not create additional independent temperature observations.</li>
    <li><strong>Zeros and the pandemic are handled explicitly.</strong> The 42-market source has 50 zero cells, all in 2020–2021. Balanced log comparisons omit invalid dates; separate PPML models retain all zeros and show no clear temperature association in expected count levels. No individual market MoM slope survives Holm correction in either the pre-2020 or short full-control profile table at 12 lags.</li>
    </ul></div></header><main><nav aria-label="Report sections"><button class="active" data-tab="overview">Overview</button><button data-tab="aggregate">Aggregate</button><button data-tab="region">Region FE</button><button data-tab="country">Country FE</button><button data-tab="crisis">Crisis exclusions</button><button data-tab="robustness">Robustness</button><button data-tab="methods">Methods & data</button></nav>
    <section id="overview" class="tab active"><h2>Similar questions, three different estimands</h2><p>The outcome is a visitor-arrival event classified by place of residence, rather than nationality or a unique person. The national outcome uses the published total. Regional and named-market models use a balanced, unchanged set of units, origin-specific seasonal effects and common year effects. The common-slope FE estimate equals a regression of the equal-origin mean outcome on the shared weather and calendar effects; that equivalence has been checked independently.</p>
    <img class="figure" src="__COMPARISON__" alt="Aggregate, regional and country heat coefficients change sign across sample windows">__OVERVIEW_TABLE__<p class="caption">MoM growth log points per +0.1°C; 95% intervals use calendar-aware Bartlett HAC with 12 lags and t(T−1). The long pre-2020 model has 117 residual time degrees of freedom; the short model has 44. These are nominal comparisons within a larger specification exercise.</p>
    <h3>The national arrivals series is dominated by the travel disruption</h3><img class="figure" src="__HISTORY__" alt="Monthly national arrivals and temperature from 2008 to September 2026"><p class="caption">Published, not seasonally adjusted arrivals and Changi monthly mean temperature. Shading denotes 2020–2022 endpoint years, not exact legal closure dates. Every regression uses full calendar months; the 2026 source is January–September only.</p>
    <p>The 42 named origins cover __COVER2019__ of 2019 arrivals and __COVER2025__ of 2025 arrivals. Nine regional totals plus global ‘Others’ reconcile exactly to the official total in every month; each region also reconciles exactly to its named markets and regional residual. The country panel excludes residual categories, so its weights and coverage differ from the aggregate.</p></section>
    __PANELS__
    __CRISIS_SECTION__
    <section id="robustness" class="tab"><h2>The association depends on the window and outcome</h2><p>The full-control window adds wind and PM2.5 but uses only the eligible 2017–2025 months. It is a useful continuity comparison with the economic project, not automatically the strongest evidence for tourism. The longer pre-2020 and strict-exclusion histories are shown alongside it.</p><img class="figure" src="__ROBUSTNESS__" alt="Sensitivity of the short aggregate MoM temperature coefficient"><p class="caption">All estimates refer to the aggregate MoM outcome, with the exact dates of each model retained. The weather-change model uses Δ temperature and changes in all controls, so its X interpretation differs. ‘Current + three lags’ plots the contemporaneous coefficient; the dynamic table below also reports the full-covariance sum.</p>
    <div class="controls"><label>Analysis <select id="robust-panel"><option value="aggregate">Official aggregate</option><option value="region">Region FE</option><option value="country">Country FE</option></select></label><label>Sample <select id="robust-sample"><option value="full_clean">Full controls · strict COVID exclusion</option><option value="long_pre2020">Pre-2020 · long controls</option><option value="long_clean">Long controls · strict COVID exclusion</option></select></label><label>Bandwidth <select id="robust-bandwidth"><option value="12">12 months</option><option value="24">24 months</option></select></label></div><div id="robust-table" class="scroll"></div>
    <details><summary>Full regression gt tables for the sensitivity models above</summary><div id="robust-full-gt"></div></details>
    <h3>Dynamic response and future-temperature diagnostic</h3><div id="dynamic-table" class="scroll"></div><p class="caption">The sum of coefficients at lags 0–3 describes the fitted accumulation of monthly growth responses to a one-month temperature pulse, conditional on this model. It is not permanent-warming damage. Future-month temperature is an exploratory timing diagnostic; persistence and seasonality can make it correlated with current temperature, so it is not a definitive causal placebo.</p>
    <h3>Leaving out one year</h3><div id="loo-table" class="scroll"></div><p class="caption">Each row refits the same specification after omitting one retained year. This assesses sensitivity to a single year; it is not a confidence interval or an independent out-of-sample test.</p>
    <h3>A count model that keeps genuine zeros</h3><p>PPML models expected count levels with origin × month-of-year and year effects, retaining all named markets and all reported zeros. It estimates a proportional association in expected arrival levels, a different estimand from growth. Calendar score covariance allows dependence across all markets and over time; 12- and 24-month bandwidths are available.</p><div id="ppml-table" class="scroll"></div><p class="caption">Table coefficients are 100β log points per +1°C in the conditional mean, with 95% time-HAC intervals. For a finite temperature change d, the fitted proportional difference is exp(βd)−1. PPML does not resolve omitted travel-demand factors or prove stationary level relationships.</p></section>
    <section id="methods" class="tab"><h2>Design, sources and independent checks</h2><h3>The regression question</h3><p><code>100[ln(Aᵢₜ) − ln(Aᵢ,ₜ₋ₕ)] = βTₜ + γ′Xₜ + origin × calendar-month FE + year FE + εᵢₜ</code></p><p>h=1 for MoM and h=12 for YoY. Growth is not annualised. Arrivals are NSA throughout the three-way comparison; origin-specific month effects absorb regular seasonality. A month-length-normalized arrivals sensitivity and current/lagged Lunar New Year-window controls are included. YoY growth does not automatically require YoY temperature; current-weather and weather-change specifications answer different questions. Log and raw levels remain descriptive sensitivities.</p><p>Temperature is the Changi mean in °C; rainfall is monthly total per 100 mm; relative humidity is in percentage points; wind is m/s; PM2.5 is per 10 μg/m³. The long model controls for rain and humidity. Full controls add wind and PM2.5 using the economic project's frozen, verified input snapshot and 90% within-month temporal coverage; an 80% check is separate. Matched-control estimates hold dates constant. Wind is an equal-active-station mean and PM2.5 is the mean of five published regions. These are destination-wide proxies, not each visitor's experienced heat.</p>
    <h3>Time support and interpretation</h3><p>National weather is common to every origin. Full date FE would absorb its coefficient. Region/country FE remove persistent origin and seasonal differences, and year effects remove annual common shifts; origin demand, exchange rates, air capacity, travel restrictions, events and changing holidays can still confound month-to-month estimates. Realised monthly weather may also occur after trips were planned. No causal tourism-loss, permanent-growth or future climate projection is identified.</p><p>For the balanced shared-slope OLS panels, collapsing to the equal-origin mean is an exact coefficient identity, not an approximation. Fixed 2017–2019 arrival-share weights provide a different, explicitly reweighted benchmark; they do not reconstruct official national log growth. Their fixed historical portfolio is a sensitivity, not an ex ante forecast. Fully interacted profiles allow all weather and calendar parameters to differ by origin and use a joint system covariance.</p><p>HAC aggregates coefficient-influence scores at their real monthly calendar positions, adds Bartlett-weighted covariances at lags 1–12 or 1–24, and leaves missing months as gaps. The finite-sample factor is T/(T−k), with k the time-series design rank; intervals use an approximate t(T−1) reference. More origins do not turn T weather observations into N×T independent observations. PPML uses the analogous time-score correction and weighted FE projection. These finite-sample inferences are approximate; 66 retained months and only 44 residual time df warrant caution.</p>
    <h3>Pandemic dates, zeros and matched samples</h3><p>Every log change, weather change and lag is constructed on the complete raw calendar before filtering. An endpoint-only exclusion removes observations dated 2020–2022. The strict exclusion also removes the growth base when it falls in those years: January 2023 for MoM and all of 2023 for YoY. The coverage screen already removes January 2023 from the short MoM model, so its endpoint-only and strict versions coincide; that is verified rather than assumed. The strict dynamic check also excludes pandemic weather-lag months.</p><p>The source has 50 zero cells among named markets, all in 2020–2021. No constant is added before logging. Each three-cut log comparison uses identical dates on which every named market has valid endpoints, preserving balanced panels. The all-date log samples therefore retain 207 MoM months and 189 YoY months, rather than the 224/213 months potentially available for the national series. Separate national/region estimates use all their own positive dates; PPML retains all 225 source months (100 eligible full-control months) and all zeros. The strict full-control MoM and YoY models retain 66 and 56 months respectively. Exact samples and exclusions are downloadable.</p>
    <h3>Persistence diagnostics are reference screens</h3><p>ADF uses AIC selection up to 24 lags; KPSS uses its automatic bandwidth. Series are first adjusted for fixed calendar-month means; growth uses a constant and log levels a constant plus trend. The pre-2020 contiguous histories and usable full histories are reference screens. Estimated seasonal removal, low power and breaks limit interpretation; non-rejection is not proof. Gapped exact full-control/strict-exclusion samples receive no stationarity clearance. The CSV preserves bounded KPSS p-value warnings and cases not tested. Level regressions and PPML remain descriptive.</p><details><summary>Diagnostic status counts across series and reference windows</summary><div class="scroll">__DIAGNOSTIC_TABLE__</div><p class="caption">Counts are series × reference-window results, not the number of fitted regressions or an accuracy score. ‘Compatible’ means the two tests align at nominal 5%; it does not certify causal validity or every shorter estimation window.</p></details>
    <h3>Source definitions and a metadata discrepancy</h3><ul><li><a href="https://data.gov.sg/datasets/d_d1c33009b674dcf70b8e8c790b793f28/view">Current arrivals by place of residence</a> / <a href="https://tablebuilder.singstat.gov.sg/table/TS/M550001">SingStat M550001</a>: 62 series, 225 months; source update 30 October 2026; retrieved 11 November 2026. Includes the total, nine regions, 42 named markets and ten residual buckets.</li><li><a href="https://data.gov.sg/datasets/d_c6d3f73a65d1fa3a92f1f4e70e619a66/view">Explicitly seasonally adjusted M550002</a> contains only 22 series. It supports an aggregate sensitivity, not a comprehensive SA country/region comparison. M550001's metadata field incorrectly says ‘Seasonally Adjusted’; its plain monthly title and distinct observed values establish its identity. Both metadata snapshots are preserved. January 2008 total: 885,290 NSA versus 896,038 SA.</li><li><a href="https://www.singstat.gov.sg/find-data/explore-data-themes/industry/tourism/our-data-explained">Official tourism definitions</a>: visitors stay less than a year; residence is not nationality. Exclusions include Malaysian citizens arriving by land, returning Singapore residents/pass holders, crew, military and transit/transfer passengers. The Malaysia series therefore does not measure all Malaysian cross-border travel.</li><li>NEA Changi <a href="https://data.gov.sg/datasets/d_755290a24afe70c8f9e8bcbf9f251573/view">temperature</a>, <a href="https://data.gov.sg/datasets/d_b16d06b83473fdfcc92ed9d37b66ba58/view">rainfall</a> and <a href="https://data.gov.sg/datasets/d_e3b1111a26b2aefef4f386c068d4ca5d/view">humidity</a>; existing frozen <a href="https://data.gov.sg/datasets/d_7677738484067741bf3b56ab5d69c7e9/view">wind</a> and <a href="https://data.gov.sg/datasets/d_6c0d5fc34838b12472475fdb73c0af29/view">PM2.5</a> snapshot. New visitor analyses reuse, and do not refresh, the economic report's climate inputs.</li><li><a href="https://www.hko.gov.hk/en/gts/time/conversion1_text.htm">Hong Kong Observatory annual lunar-calendar tables</a> supply the Lunar New Year date. The monthly control counts days in the inclusive ±7-day window; annual source calendars, hashes and extraction code are retained.</li></ul>
    <h3>Verification and reproduction</h3><p id="verification-status">__VERIFICATION__</p><p>The analysis is isolated in <code>data/{raw,processed}/visitor_arrivals</code>, <code>outputs/visitor_arrivals</code> and new visitor-specific scripts. These scripts do not write the main economic HTML or add this report to its pipeline. The economic HTML changed during concurrent annual-analysis work in this folder; the receipt retains both observed hashes. That separate HTML is not an input to the visitor estimates, and all actual visitor/weather input hashes still match.</p><p><code>uv run python scripts/visitor_analysis.py<br>uv run python scripts/visitor_counts.py<br>uv run python scripts/verify_visitor_arrivals.py<br>uv run python scripts/visitor_report.py<br>node scripts/verify_visitor_report.mjs</code></p><p>Raw files are cached with hashes; reruns reuse the same vintage. Native PyFixest estimates are independently reconstructed; no model is selected by a significance or stationarity pretest. Nominal p-values across the specification grid are exploratory. Market p-values receive Holm adjustment within each panel × outcome × sample × bandwidth family; no adjustment across all specifications is claimed.</p><div class="downloads">__DOWNLOADS__</div></section>
    <footer>Standalone visitor-arrivals analysis · source observations through September 2026 · built from saved estimates · works offline. Sources and methods are specific to this report.</footer></main><script id="report-data" type="application/json">__PAYLOAD__</script><script>
    const D=JSON.parse(document.getElementById('report-data').textContent);
    const samples=['long_all','long_pre2020','long_clean','full_all','full_clean'];
    const sampleLabels={long_all:'Long controls · all admissible dates',long_pre2020:'Long controls · before 2020',long_clean:'Long controls · strict exclusion',full_all:'Full controls · all admissible dates',full_clean:'Full controls · strict exclusion'};
    const termLabels={temp_c:'Temperature (°C)',rain_100mm:'Rainfall (100 mm)',rh_pct:'Humidity (pp)',wind_mps:'Wind (m/s)',pm25_10ug:'PM2.5 (10 μg/m³)'};
    const f=(x,n=3)=>x==null?'—':Number(x).toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n});
    const p=x=>x==null?'Unavailable':x<.001?'&lt;0.001':f(x);
    const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
    const table=(heads,rows)=>'<table><thead><tr>'+heads.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map((v,i)=>'<'+(i===0?'th':'td')+'>'+v+'</'+(i===0?'th':'td')+'>').join('')+'</tr>').join('')+'</tbody></table>';
    function drawPanel(panel){
      const outcome=document.querySelector('.outcome[data-panel="'+panel+'"]').value;
      const mode=document.querySelector('.bandwidth[data-panel="'+panel+'"]').value;
      const bw=mode==='origin'?12:+mode,method=mode==='origin'?'origin':'hac'+mode;
      const chosen=samples.map(s=>{const base=D.results.find(r=>r.panel===panel&&r.outcome===outcome&&r.sample===s&&r.variant==='main'&&r.hac_lags===bw);return mode==='origin'?{...base,...D.clusterResults.find(r=>r.source==='core'&&r.model_id===base.model_id)}:base;});
      document.getElementById(panel+'-table').innerHTML=D.gt['core__'+panel+'__'+outcome+'__'+method];
      document.getElementById(panel+'-context').innerHTML=table(['Sample','+0.1°C effect','95% interval','Residual temperature SD (°C)','Calendar gaps','Residual ACF(1)','Residual ACF(12)'],chosen.map(r=>[sampleLabels[r.sample],f(r.effect_0p1c),'['+f(r.lower_0p1c)+', '+f(r.upper_0p1c)+']',f(r.sd_residual_temperature),f(r.calendar_gaps,0),f(r.residual_acf_calendar_1),f(r.residual_acf_calendar_12)]));
      document.getElementById(panel+'-interpretation').textContent=outcome==='mom'?'MoM: 100 × [ln arrivals(t) − ln arrivals(t−1)]. Temperature coefficients are growth log points per +1°C. Seasonal adjustment is handled with fixed effects.':outcome==='yoy'?'YoY: 100 × [ln arrivals(t) − ln arrivals(t−12)]. Rolling annual outcomes overlap; HAC uses real calendar lags. Strict exclusion removes both pandemic endpoints and pandemic base years.':outcome==='log_level'?'Descriptive log levels: 100 × ln arrivals. A temperature slope is approximately a percent level difference per +1°C for small changes. Persistence and confounding remain unresolved.':'Descriptive count levels: visitor-arrival events per month. Region/country coefficients average counts per unit, not total Singapore arrivals.';
      if(panel!=='aggregate')drawProfiles(panel,outcome,bw);
    }
    function drawProfiles(panel,outcome,bw){
      const target=document.getElementById(panel+'-profile-table'),test=document.getElementById(panel+'-profile-test');
      if(!['mom','yoy'].includes(outcome)){target.innerHTML='';test.textContent='Market profiles are supplied for MoM and YoY growth. Choose either growth outcome above.';return;}
      const sample=document.querySelector('.profile-sample[data-panel="'+panel+'"]').value;
      const rows=D.profiles.filter(r=>r.panel===panel&&r.outcome===outcome&&r.sample===sample&&r.hac_lags===bw).sort((a,b)=>a.unit.localeCompare(b.unit));
      target.innerHTML=D.gt['profile__'+panel+'__'+outcome+'__'+sample+'__hac'+bw];
      const joint=D.joint.find(r=>r.group===panel+'__'+outcome+'__'+sample&&r.hac_lags===bw);
      test.innerHTML='Every row uses '+rows[0].n_months+' months; fully origin-specific weather, seasonal and year effects. '+rows.filter(r=>r.holm_p_value<.05).length+' of '+rows.length+' slopes pass the within-table Holm 5% threshold. Omnibus equality: '+esc(joint.status)+', covariance rank '+joint.covariance_rank+'/'+joint.restrictions+', nominal p='+p(joint.p_value)+(joint.high_dimensional_warning?'. <strong>High-dimensional and fragile: restrictions exceed half the residual time df.</strong>':'.');
      if(document.querySelector('.bandwidth[data-panel="'+panel+'"]').value==='origin')test.innerHTML='Profile tables retain 12-month calendar HAC: a separate-origin slope has only one origin cluster, so origin-cluster inference is unavailable for it. '+test.innerHTML;
    }
    function drawCrisis(){
      const panel=document.getElementById('crisis-panel').value,outcome=document.getElementById('crisis-outcome').value,sample=document.getElementById('crisis-window').value;
      const control=document.getElementById('crisis-inference');
      if(panel==='aggregate'&&control.value==='origin')control.value='hac12';
      const method=control.value,bw=method==='origin'?12:+method.slice(3);
      const chosen=['none','covid','gfc_covid','buffered','stable_pre2020'].map(policy=>{const r=D.crisisResults.find(r=>r.panel===panel&&r.outcome===outcome&&r.sample===sample&&r.variant==='crisis_'+policy&&r.hac_lags===bw);return method==='origin'?{...r,...D.clusterResults.find(c=>c.source==='crisis'&&c.model_id===r.model_id)}:r;});
      document.getElementById('crisis-gt').innerHTML=D.gt['crisis__'+panel+'__'+outcome+'__'+sample+'__'+method];
      const names=['No crisis exclusion','COVID only','GFC + COVID','Crisis + recovery buffers','Pre-COVID after GFC buffer'];
      document.getElementById('crisis-summary').innerHTML=table(['Exclusion policy','Months','Residual time df','+0.1°C effect','95% interval','Nominal p'],chosen.map((r,i)=>[names[i],f(r.n_months,0),f(r.df_time_resid,0),f(r.effect_0p1c),'['+f(r.lower_0p1c)+', '+f(r.upper_0p1c)+']',p(r.p_value)]));
      document.getElementById('crisis-note').textContent=(method==='origin'?'Origin CR1 comparison: '+(panel==='region'?'nine clusters, t(8)':'42 clusters, t(41)')+'. Assumes independent origin markets and misses common shocks; calendar HAC remains primary. ':'Calendar HAC is primary and preserves gaps after exclusion. ')+(sample==='full_all'?'GFC exclusions predate this window, so COVID-only and GFC+COVID columns match. The pre-COVID full-control check has only 32 months and 13 residual time df.':'The long history starts in 2008; no Asian financial crisis or SARS dates are present.');
    }
    const variantLabels={main:'Current weather',lunar:'Lunar New Year current + lag',daily:'Growth in arrivals per day',season_trends:'Changing seasonal pattern',fixed_weights:'Fixed 2017–2019 share weights',matched_base_controls:'Base controls on full-control dates',coverage80:'80% coverage screen',weather_differences:'Changes in all weather controls',lag_matched_current:'Current only, matched lag dates',lags:'Current + temperature lags 1–3',lead:'Current + future temperature',quadratic_at28:'Quadratic temperature, slope at 28°C',official_sa:'Official SA aggregate'};
    function drawRobust(){
      const panel=document.getElementById('robust-panel').value,sample=document.getElementById('robust-sample').value,bw=+document.getElementById('robust-bandwidth').value;
      let selected=D.results.filter(r=>r.panel===panel&&r.sample===sample&&r.outcome==='mom'&&r.hac_lags===bw&&!r.variant.startsWith('loo_')&&!['own_positive_dates'].includes(r.variant));
      selected=selected.concat(D.results.filter(r=>r.panel===panel&&r.sample===sample&&r.outcome==='daily_mom'&&r.variant==='daily'&&r.hac_lags===bw));
      if(sample==='full_clean')selected=selected.concat(D.results.filter(r=>r.panel===panel&&r.sample==='full80_clean'&&r.outcome==='mom'&&r.variant==='coverage80'&&r.hac_lags===bw));
      if(panel==='aggregate')selected=selected.concat(D.results.filter(r=>r.panel==='aggregate_sa'&&r.sample===sample&&r.outcome==='mom'&&r.hac_lags===bw));
      selected.sort((a,b)=>Object.keys(variantLabels).indexOf(a.variant)-Object.keys(variantLabels).indexOf(b.variant));
      document.getElementById('robust-table').innerHTML=table(['Specification','Months','Coefficient per +1°C','SE','+0.1°C effect','95% interval, +0.1°C','Nominal p'],selected.map(r=>[variantLabels[r.variant]||r.variant,f(r.n_months,0),f(r.coefficient),f(r.std_error),f(r.effect_0p1c),'['+f(r.lower_0p1c)+', '+f(r.upper_0p1c)+']',p(r.p_value)]));
      document.getElementById('robust-full-gt').innerHTML=Object.keys(D.gt).filter(key=>key.startsWith('robust__'+panel+'__'+sample+'__hac'+bw+'__')).sort().map(key=>'<div class="scroll">'+D.gt[key]+'</div>').join('');
      const dynamic=D.lags.filter(r=>r.model_id.startsWith(panel+'__mom__'+sample+'__')&&r.hac_lags===bw);
      document.getElementById('dynamic-table').innerHTML=table(['Model','Term','Coefficient per +1°C','SE','95% interval','Nominal p'],dynamic.map(r=>[r.model_id.endsWith('__lead')?'Lead diagnostic':'Distributed lags',esc(r.term),f(r.coefficient),f(r.std_error),'['+f(r.ci_lower)+', '+f(r.ci_upper)+']',p(r.p_value)]));
      const loo=D.results.filter(r=>r.panel===panel&&r.sample===sample&&r.outcome==='mom'&&r.hac_lags===bw&&r.variant.startsWith('loo_'));
      document.getElementById('loo-table').innerHTML=loo.length?table(['Omitted year','Months','Coefficient per +1°C','95% interval','Nominal p'],loo.map(r=>[r.variant.slice(4),f(r.n_months,0),f(r.coefficient),'['+f(r.ci_lower)+', '+f(r.ci_upper)+']',p(r.p_value)])):'<p>Leave-one-year checks are supplied for the pre-2020 and full-control strict-exclusion windows.</p>';
      const pp=D.ppml.filter(r=>r.panel===panel&&r.hac_lags===bw);
      document.getElementById('ppml-table').innerHTML=table(['Sample','Months','Origin units','Zero cells retained','100β per +1°C','95% interval','Nominal p'],pp.map(r=>[(sampleLabels[r.sample]||r.sample).replace('all admissible dates','all observed dates'),f(r.n_months,0),f(r.n_units,0),f(r.zero_cells_retained,0),f(r.coefficient),'['+f(r.ci_lower)+', '+f(r.ci_upper)+']',p(r.p_value)]));
    }
    document.querySelectorAll('nav button').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('nav button,.tab').forEach(e=>e.classList.remove('active'));button.classList.add('active');document.getElementById(button.dataset.tab).classList.add('active');}));
    document.querySelectorAll('.outcome,.bandwidth,.profile-sample').forEach(s=>s.addEventListener('change',()=>drawPanel(s.dataset.panel)));
    document.querySelectorAll('#robust-panel,#robust-sample,#robust-bandwidth').forEach(s=>s.addEventListener('change',drawRobust));
    document.querySelectorAll('#crisis-panel,#crisis-outcome,#crisis-window,#crisis-inference').forEach(s=>s.addEventListener('change',drawCrisis));
    ['aggregate','region','country'].forEach(drawPanel);drawRobust();drawCrisis();
    </script></body></html>'''
    verification_text = ("Independent numerical verification: " + html.escape(str(verified.get("status", "pending"))) +
        ". The saved receipt records scope, numerical tolerances and remaining limits; report rendering is checked separately.")
    replacements = {"SHORT": f"{short.effect_0p1c:+.3f}", "SHORTCI": f"[{short.lower_0p1c:+.3f}, {short.upper_0p1c:+.3f}]",
        "SHORTP": f"{short.p_value:.3f}", "PRE": f"{pre.effect_0p1c:+.3f}", "PRECI": f"[{pre.lower_0p1c:+.3f}, {pre.upper_0p1c:+.3f}]",
        "COMPARISON": figures["comparison"], "HISTORY": figures["history"], "ROBUSTNESS": figures["robustness"],
        "OVERVIEW_TABLE": overview_table, "PANELS": ''.join(panels), "DIAGNOSTIC_TABLE": diagnostic_table,
        "CRISIS_SECTION": (ROOT/"templates/visitor_crisis.html").read_text().replace("__CRISIS_FIGURE__",figures["crisis"]),
        "COVER2019": f"{checks['annual_named_coverage']['2019']:.2%}", "COVER2025": f"{checks['annual_named_coverage']['2025']:.2%}",
        "VERIFICATION": verification_text, "DOWNLOADS": downloads, "PAYLOAD": encoded}
    for key, value in replacements.items():
        template = template.replace(f"__{key}__", value)
    p_notes = '<h3>Nominal p, Holm p and the error structure</h3><p>A nominal p-value is the probability, under a zero-coefficient null and the stated statistical assumptions, of a test statistic at least as extreme as the observed one. It is not the probability that the null is true. Holm p adjusts the temperature tests across the nine regions or 42 named markets in a particular profile table to control family-wise false positives. It does not adjust every model in the specification grid. Holm does not require the tests to be independent when their individual p-values are valid.</p><p>Calendar HAC is the primary inference here, rather than clustering by origin. Common Singapore temperature and tourism shocks affect many origins together. Region/country clustering permits arbitrary dependence within an origin over time but assumes independent origins, which omits that common-shock dependence. The new CR1 comparison uses nine regional clusters with t(8), or 42 origin clusters with t(41), and counts all explicit FE parameters in its small-sample factor. Nine clusters is a small number. Fixed effects and the clustering choice address different problems.</p><p>Full regression tables are generated with native R gt 1.4.0. They show all included weather regressors with robust SE and nominal p-values, FE indicators and sample/inference details. Fully interacted profile tables now show every weather coefficient, with a separate Holm column for temperature; those profiles retain calendar HAC because an individual origin slope has just one origin cluster.</p>'
    template = template.replace('<h3>The regression question</h3>',p_notes+'<h3>The regression question</h3>')
    template = template.replace('uv run python scripts/visitor_analysis.py<br>uv run python scripts/visitor_counts.py<br>uv run python scripts/verify_visitor_arrivals.py<br>uv run python scripts/visitor_report.py<br>node scripts/verify_visitor_report.mjs',
                                'uv run python scripts/visitor_run.py<br>uv run python scripts/visitor_run.py --reports-only')
    target = ROOT / "outputs/visitor_arrivals.html"
    target.write_text(template)
    save_json(OUT / "report_manifest.json", {"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
        "pooled_specifications": r.model_id.nunique(), "profile_rows": len(profiles), "ppml_models": ppml.model_id.nunique(),
        "crisis_ols_specifications": len(pd.read_csv(OUT/"crisis/regression_results.csv").model_id.unique()),
        "origin_cluster_comparisons": len(payload["clusterResults"]), "native_gt_tables": len(payload["gt"]),
        "source_end": checks["source_end"], "self_contained": True, "main_economic_report_included": False})
    print(f"Wrote {target}: {target.stat().st_size:,} bytes", flush=True)
    return target


if __name__ == "__main__":
    build()
