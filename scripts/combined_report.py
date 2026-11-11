"""Package every analysis, its reference notes and sample context in one offline HTML."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
import pandas as pd
from lxml import html as lxml_html
from inference_display import main_rows
from significance import significance_label, significance_class, p_format
from stationarity_context import model_assessment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORTS = [
    ("covid", "Full vs no COVID", "covid_comparison.html"),
    ('annual','Annual analyses','annual.html'),
    ("aggregate", "Aggregate VA", "aggregate_va.html"),
    ("sector-va", "VA: long history", "heat_econ_results.html"),
    ("granular-va", "Detailed VA", "granular_va.html"),
    ("iip", "IIP: long history", "iip_results.html"),
    ("manufacturing", "Manufacturing detail", "manufacturing_detail.html"),
    ("growth-levels", "Growth & levels", "outcome_comparisons.html"),
    ("wind-pm", "Wind / PM2.5", "extended_controls.html"),
    ("heterogeneity", "Heterogeneity", "sector_heterogeneity.html"),
    ("cluster-sensitivity", "Sector-cluster sensitivity", "clustered_sector_sensitivity.html"),
    ("tables", "Regression tables", "regression_tables.html"),
    ("reconciliation", "Aggregate checks", "reconciliation.html"),
    ("stationarity", "Stationarity", "stationarity.html"),
]
DESTINATIONS = {filename: key for key, _, filename in REPORTS}
DESTINATIONS["heat_economy.html"] = "overview"
UNITS = {"short": "MoM/QoQ growth log points", "yoy": "YoY growth log points"}


def download(path: Path, label: str) -> str:
    mime = {".csv": "text/csv", ".md": "text/markdown", ".json": "application/json"}.get(path.suffix, "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<a href="data:{mime};base64,{encoded}" download="{html.escape(path.name)}">{html.escape(label)} ↓</a>'


def overview_data() -> list[dict]:
    from sample_context import model_statistics
    rows = []
    analyses = [("aggregate", "Aggregate VA", "quarterly"), ("va", "Sector VA panel", "quarterly"),
                ("iip-monthly", "IIP sector panel", "monthly"), ("iip-quarterly", "IIP sector panel", "quarterly")]
    for context, label, frequency in analyses:
        for growth in ["short", "yoy_nsa", "yoy"]:
            for window in ["long", "full", "exclude"]:
                if context == "aggregate":
                    group = "weather" if window == "long" else "full"
                    path = OUT / "aggregate_va" / group / "regression_results.csv"
                    model = ("G1" if growth == "short" else 'G2N' if growth=='yoy_nsa' else "G2")+("X" if window == "exclude" else "")
                elif growth == "short":
                    path = ((OUT / "regression_results.csv" if context == "va" else OUT / "iip" / frequency / "regression_results.csv")
                            if window == "long" else OUT / "extended_controls" / context / "regression_results.csv")
                    model = {"long": "S3", "full": "C4", "exclude": "C5"}[window]
                else:
                    path = OUT / "outcomes" / context / ("weather" if window == "long" else "full") / "regression_results.csv"
                    model = ('G2N' if growth=='yoy_nsa' else 'G2')+('X' if window=='exclude' else '')
                result = pd.read_csv(path)
                bandwidth = 12 if frequency == "monthly" else 4
                selected = main_rows(result, frequency)
                selected = selected[selected.model.eq(model)&selected.term.eq("temp_c")]
                assert len(selected) == 1, (context, growth, window, path, model)
                row = selected.iloc[0]
                stats = row.to_dict() if "mean_y" in selected else model_statistics(path, model)
                assert stats and pd.notna(stats["mean_y"]), (path, model)
                periods = int(row.get("n_periods", row.get("n_months", row.get("n_quarters"))))
                record = {"context": context, "label": label, "frequency": frequency,
                          "growth": growth, "window": window,
                          "outcome_label": ("YoY" if growth.startswith('yoy') else "MoM" if frequency == "monthly" else "QoQ")+" log growth ("+('NSA' if growth=='yoy_nsa' else 'SA')+")",
                          "mean_y": float(stats["mean_y"]), "sd_y": float(stats["sd_y"]),
                          "coefficient": float(row.coefficient), "std_error": float(row.std_error),
                          "ci_lower": float(row.ci_lower), "ci_upper": float(row.ci_upper), "p_value": float(row.p_value),
                          "effect_0p1c": float(.1*row.coefficient), "ci_0p1c_lower": float(.1*row.ci_lower),
                          "ci_0p1c_upper": float(.1*row.ci_upper), "periods": periods,
                          "n_obs": int(row.n_obs), "start": str(row.start), "end": str(row.end),
                          "source_csv": str(path.relative_to(ROOT)), "model": model,
                          "covariance": row.covariance, "n_clusters": int(row.n_sectors) if row.covariance == "Sector cluster" else 0,
                          "target": "aggregate" if context == "aggregate" else "sector-va" if context == "va" else "iip"}
                rows.append(record)
    for context, label, frequency, target in [
        ("va-detailed", "VA · 19-sector mean benchmark", "quarterly", "granular-va"),
        ("iip-monthly-total", "IIP · official weighted total", "monthly", "manufacturing"),
        ("iip-quarterly-total", "IIP · official weighted total", "quarterly", "manufacturing"),
        ("iip-monthly-detailed", "IIP · 18-industry mean benchmark", "monthly", "manufacturing"),
        ("iip-quarterly-detailed", "IIP · 18-industry mean benchmark", "quarterly", "manufacturing")]:
        for growth, model in [("short", "G1"), ('yoy_nsa','G2N'),("yoy", "G2")]:
            for window in ["long", "full", "exclude"]:
                path = OUT / "granular" / context / window / "regression_results.csv"
                selected = main_rows(pd.read_csv(path), frequency)
                row = selected[selected.model.eq(model)&selected.term.eq("temp_c")].iloc[0]
                rows.append({"context": context, "label": label, "frequency": frequency, "growth": growth, "window": window,
                    "outcome_label": ("YoY" if growth.startswith('yoy') else "MoM" if frequency == "monthly" else "QoQ")+" log growth ("+('NSA' if growth=='yoy_nsa' else 'SA')+")",
                    **{col: float(row[col]) for col in ["mean_y", "sd_y", "coefficient", "std_error", "ci_lower", "ci_upper", "p_value"]},
                    "effect_0p1c": .1*row.coefficient, "ci_0p1c_lower": .1*row.ci_lower, "ci_0p1c_upper": .1*row.ci_upper,
                    "periods": int(row["n_months" if frequency == "monthly" else "n_quarters"]), "n_obs": int(row.n_obs),
                    "start": row.start, "end": row.end, "source_csv": str(path.relative_to(ROOT)), "model": model, "target": target,
                    "covariance": row.covariance, "n_clusters": int(row.n_sectors) if row.covariance == "Sector cluster" else 0})
    for record in rows:
        assessment = model_assessment(ROOT/record["source_csv"], record["model"])
        record.update(is_baseline=record["window"]=="full", significance=significance_label(record["p_value"]), significance_class=significance_class(record["p_value"]),
            p_display=p_format(record["p_value"]), stationarity_y=assessment["y_label"],stationarity_all=assessment["all_label"],
            analysis_role=assessment["analysis_role"])
    return rows


BRIDGE = r'''
(()=>{
 const key=__REPORT_KEY__, channel='heat-economy-collection';
 let lastHeight=0, scheduled=false;
 function measure(){scheduled=false;const main=document.querySelector('main')||document.body;
  const height=Math.ceil(main.getBoundingClientRect().height+Math.max(0,main.offsetTop)+28);
  if(Math.abs(height-lastHeight)>2){lastHeight=height;parent.postMessage({channel,type:'height',key,height},'*');}}
 function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(measure);}}
 function anchor(hash){const id=(hash||'').replace(/^#/,'');if(!id){schedule();return;}
  const element=document.getElementById(id);if(!element)return;
  if(element.classList.contains('reg-panel')&&typeof showPanel==='function')showPanel(id);
  if(element.classList.contains('frequency')&&typeof showFrequency==='function')showFrequency(id);
  schedule();requestAnimationFrame(()=>parent.postMessage({channel,type:'scroll',key,top:element.getBoundingClientRect().top+window.scrollY},'*'));}
 document.addEventListener('click',event=>{const link=event.target.closest('a');if(!link)return;
  if(link.dataset.reportTarget){event.preventDefault();parent.postMessage({channel,type:'navigate',key,target:link.dataset.reportTarget,hash:link.dataset.reportAnchor||''},'*');}
  else if(link.getAttribute('href')?.startsWith('#')){event.preventDefault();anchor(link.getAttribute('href'));}
 },true);
 window.addEventListener('message',event=>{if(event.source!==parent||event.data?.channel!==channel)return;
  if(event.data.type==='show'){schedule();anchor(event.data.hash||'');}
  if(event.data.type==='print')window.print();});
 new ResizeObserver(schedule).observe(document.querySelector('main')||document.body);
 window.addEventListener('load',()=>{schedule();parent.postMessage({channel,type:'ready',key},'*');});
 document.addEventListener('toggle',schedule,true);document.addEventListener('click',schedule);
 schedule();
})();
'''


def embedded_report(filename: str, key: str) -> tuple[str, dict]:
    path = OUT / filename
    source = path.read_text()
    doc = lxml_html.fromstring(source, parser=lxml_html.HTMLParser(huge_tree=True))
    if key not in {'covid','annual','reconciliation','stationarity','cluster-sensitivity'}:
        container=doc.find('.//main')
        if container is None:container=doc.find('body')
        banner=lxml_html.fragment_fromstring('<p style="padding:12px 17px;background:#e9f3fa;border-left:3px solid #1f77b4;font-size:13px"><a href="covid_comparison.html"><strong>Full versus excluding 2020–2022: paired regression tables for every specification →</strong></a></p>')
        container.insert(0,banner)
    for node in doc.xpath('//a[@href]'):
        href = node.get("href")
        parsed = urlsplit(href)
        basename = Path(unquote(parsed.path)).name
        if basename in DESTINATIONS and parsed.scheme in {"", "file"}:
            target = parsed.fragment if basename == "heat_economy.html" and parsed.fragment else DESTINATIONS[basename]
            node.set("href", "#"+target)
            node.set("data-report-target", target)
            if parsed.fragment and basename != "heat_economy.html":
                node.set("data-report-anchor", parsed.fragment)
        elif parsed.scheme in {"http", "https"}:
            node.set("target", "_blank")
            node.set("rel", "noopener")
        elif not parsed.scheme and parsed.path:
            local = (path.parent/unquote(parsed.path)).resolve()
            assert local.exists(), (path, href)
            replacement = lxml_html.fragment_fromstring(download(local, node.text_content()))
            node.set("href", replacement.get("href"))
            node.set("download", local.name)
    assert not doc.xpath('//script[@src] | //link[@rel="stylesheet"] | //img[not(starts-with(@src,"data:"))]'), filename
    style = lxml_html.Element("style")
    style.text = '''
html,body{background:#f3f7fa!important}body{margin:0!important}main,.wrap{max-width:1320px!important}main{padding-top:20px!important}h1{font-size:39px!important;line-height:1.13!important}.topbar,.top-links,.topline,.print-button,.print{display:none!important}.hero{padding-top:12px!important}.hero h1{font-size:45px!important}.hero-intro{font-size:15px!important}header p{font-size:14px!important}.footer{margin-bottom:0!important}a[data-report-target]{cursor:pointer}
'''
    doc.find("head").append(style)
    script = lxml_html.Element("script")
    script.text = BRIDGE.replace("__REPORT_KEY__", json.dumps(key))
    doc.find("body").append(script)
    embedded = "<!doctype html>\n"+lxml_html.tostring(doc, encoding="unicode")
    return embedded, {"file": filename, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "embedded_sha256": hashlib.sha256(embedded.encode()).hexdigest(), "bytes": len(embedded.encode())}


def methods_html() -> str:
    sections = []
    titles = {"aea_temperature_literature.md": "AEA literature and the log-growth definition",
              "aggregate_va_methods.md": "Aggregate VA: source, models and uncertainty",
              "sector_cluster_inference.md": "Sector-clustered inference and time-series exceptions",
              "va_reconciliation_methods.md": "VA aggregation identities and granular taxonomy",
              "iip_reconciliation_methods.md": "IIP hierarchy, official weights and annual links",
              "granular_methods.md": "Detailed-sector analysis and matched samples",
              "granular_interactions_methods.md": "Main detailed-sector FE interactions and time-series robustness",
              "baseline_specification.md": "Baseline controls, PM2.5 and stationarity interpretation",
              "pyfixest_backend_methods.md": "Native PyFixest estimator, FE design and provenance",
              "pyfixest_migration_validation.md": "Independent estimator-migration checks",
              "sample_context_methods.md": "Sample means, magnitude comparisons and COVID exclusions",
              "covid_comparison_methods.md": "Complete full/no-COVID pairs, unit-clustered main inference and verification",
              'nsa_yoy_provenance_methods.md':'Retained SA and added NSA YoY: official sources, transformations and validation',
              'annual_methods.md':'Annual output and weather: AEA comparison, fixed effects and sample limits',
              'temperature_horizon_review.md':'Further literature: quarterly YoY growth, temperature differences and exposure lags',
              "heterogeneity_review.md": "Sector heterogeneity: models and identification",
              "stationarity_methods.md": "Legacy stationarity and persistence diagnostics",
              "stationarity_expanded_methods.md": "Expanded all-series stationarity audit and inference labels",
              "stationarity_audit_review.md": "Independent review of stationarity coverage and claims",
              "outcome_specification_review.md": "QoQ / MoM, YoY and levels",
              "prior_electricity_reference.md": "Connection to the earlier electricity study"}
    for name, title in titles.items():
        path = ROOT / "notes" / name
        assert path.exists(), path
        source = path.read_text()
        source = re.sub(r"\\\[(.*?)\\\]", lambda m: '\n<pre class="equation">'+html.escape(m.group(1).strip())+'</pre>\n', source, flags=re.S)
        source = re.sub(r"\\\((.*?)\\\)", lambda m: '<code>'+html.escape(m.group(1))+'</code>', source)
        rendered = markdown.markdown(source, extensions=["tables", "fenced_code"])
        fragment = lxml_html.fragment_fromstring(rendered, create_parent="div")
        for link in fragment.xpath('.//a[@href]'):
            parsed = urlsplit(link.get("href"))
            if parsed.scheme or not parsed.path:
                continue
            local = (path.parent/unquote(parsed.path)).resolve()
            assert local.is_file(), (path, link.get("href"))
            replacement = lxml_html.fragment_fromstring(download(local, link.text_content()))
            link.set("href", replacement.get("href"))
            link.set("download", local.name)
        rendered = "".join(lxml_html.tostring(child, encoding="unicode") for child in fragment)
        sections.append(f'<details class="source-note"><summary>{html.escape(title)}</summary><article>{rendered}</article>'
                        f'<p>{download(path, "Download this note")}</p></details>')
    for path, label in [(ROOT / "data/raw/manifest.json", "Official-data source manifest"),
                         (ROOT / "data/raw/extended_controls/manifest.json", "Wind / PM2.5 source manifest"),
                         (OUT / "aggregate_va/verification.json", "Aggregate VA verification"),
                         (OUT / "granular/verification.json", "Detailed-sector outcome and HAC verification"),
                         (OUT / "granular_interactions/verification.json", "Main detailed-sector FE interaction verification"),
                         (OUT / "sector_cluster_verification.json", "Sector CR1 covariance verification"),
                         (OUT / "pyfixest/estimator_manifest.json", "PyFixest native-estimation provenance"),
                         (OUT / "pyfixest/migration_verification.json", "Independent PyFixest migration verification"),
                         (OUT / "pyfixest/sector_total_effect_null_checks.csv", "Sector total-effect tests: one-restriction Wald audit"),
                         (OUT / "pyfixest/omnibus_null_checks.csv", "Joint heterogeneity tests: null, rank and basis audit"),
                         (OUT / "stationarity/verification.json", "Expanded unit-root audit numerical verification"),
                         (OUT / "sector_cluster_contrasts_verification.json", "Sector CR1 cumulative and paired-weather contrasts"),
                         (OUT / "heterogeneity/verification.json", "Sector heterogeneity verification"),
                         (OUT / "sample_context/verification.json", "Sample means verification"),
                         (OUT / "sample_context/sample_means.csv", "All sample growth means"),
                         (OUT / "covid_comparison/verification.json", "Independent full/no-COVID coefficient, covariance and sample audit"),
                         (OUT / "covid_comparison/report_verification.json", "Paired native table and presentation verification"),
                         (OUT/'annual/verification.json','Independent annual aggregation, coefficients, covariance and stationarity verification'),
                         (OUT/'annual/presentation_verification.json','Annual native tables and presentation verification'),
                         (OUT/'annual/reconciliation_summary.json','Annual output versus official annual series cross-check'),
                         (OUT / "sample_context/temperature_magnitudes.csv", "Temperature coefficients and +0.1°C magnitudes")]:
        sections.append('<p>'+download(path, label)+'</p>')
    return "\n".join(sections)


def build_combined_report() -> Path:
    reports, manifest = {}, {}
    for key, _, filename in REPORTS:
        reports[key], manifest[key] = embedded_report(filename, key)
    overview = overview_data()
    folder = OUT / "collection"
    folder.mkdir(exist_ok=True)
    pd.DataFrame(overview).to_csv(folder / "overview.csv", index=False)
    manifest["overview"] = {"rows": len(overview), "source": "Exact saved coefficients and sample statistics"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    payload = json.dumps({"reports": reports, "overview": overview}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    tabs = '<button type="button" role="tab" id="tab-overview" aria-controls="panel-overview" aria-selected="true" data-tab="overview">Overview & scale</button>'
    tabs += "".join(f'<button type="button" role="tab" id="tab-{key}" aria-controls="panel-{key}" aria-selected="false" data-tab="{key}">{label}</button>' for key, label, _ in REPORTS)
    tabs += '<button type="button" role="tab" id="tab-methods" aria-controls="panel-methods" aria-selected="false" data-tab="methods">Methods & sources</button>'
    panels = "".join(f'<section role="tabpanel" aria-labelledby="tab-{key}" id="panel-{key}" class="chapter" hidden><iframe title="{label} analysis" id="frame-{key}" data-report="{key}" sandbox="allow-scripts allow-downloads allow-modals allow-popups allow-popups-to-escape-sandbox" scrolling="no"></iframe></section>' for key, label, _ in REPORTS)
    template = (ROOT / "templates/collection.html").read_text()
    replacements = {"__TABS__": tabs, "__REPORT_PANELS__": panels, "__METHOD_NOTES__": methods_html(),
                    "__REPORT_PAYLOAD__": payload, "__OVERVIEW_DOWNLOAD__": download(folder / "overview.csv", "Download overview estimates and sample means")}
    for token, value in replacements.items():
        template = template.replace(token, value)
    assert not re.search(r"__[A-Z_]+__", template)
    path = OUT / "heat_economy.html"
    path.write_text(template)
    print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes; all analysis documents embedded)")
    return path


if __name__ == "__main__":
    build_combined_report()
