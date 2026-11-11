"""Render saved estimates through MakeTables, the engine used by PyFixest etable."""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

import maketables as mt
import pandas as pd
from lxml import html as lxml_html
from inference_display import main_rows, djo_form
from significance import significance_label, p_format
from stationarity_context import model_assessment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def namespace_table_ids(raw: str, prefix: str) -> str:
    """Keep GT header IDs unique when several estimation tables share a page."""
    fragment = lxml_html.fragment_fromstring(raw, create_parent="div")
    mapping = {node.get("id"): f"{prefix}-{node.get('id')}" for node in fragment.xpath(".//*[@id]")}
    for node in fragment.iter():
        if node.get("id") in mapping:
            node.set("id", mapping[node.get("id")])
        for attr in ["headers", "aria-labelledby", "aria-describedby", "for"]:
            if node.get(attr):
                node.set(attr, " ".join(mapping.get(token, token) for token in node.get(attr).split()))
        if node.get("href", "").startswith("#") and node.get("href")[1:] in mapping:
            node.set("href", "#"+mapping[node.get("href")[1:]])
        if node.tag == "style" and node.text:
            for old, new in mapping.items():
                node.text = re.sub(r"#"+re.escape(old)+r"(?![\w-])", "#"+new, node.text)
    return "".join(lxml_html.tostring(child, encoding="unicode") for child in fragment)


@dataclass
class SavedResult:
    rows: pd.DataFrame
    dependent: str

    @property
    def __maketables_coef_table__(self):
        frame = self.rows.set_index("term")[["coefficient", "std_error", "p_value", "ci_lower", "ci_upper"]].rename(
            columns={"coefficient": "b", "std_error": "se", "p_value": "p", "ci_lower": "ci95l", "ci_upper": "ci95u"},
            index={"temp_c_unweighted": "temp_c"})
        assert frame.index.is_unique
        return frame

    @property
    def __maketables_depvar__(self):
        return self.dependent

    @property
    def __maketables_fixef_string__(self):
        trend = "season_trend" in self.rows and bool(self.rows.iloc[0].season_trend)
        return "sector_season+year" + ("+season_trend" if trend else "")

    def __maketables_stat__(self, key):
        row = self.rows.iloc[0]
        period = "n_months" if "n_months" in self.rows else "n_quarters"
        value = {"N": int(row.n_obs), "r2": row.r_squared,
                 "sectors": int(row.n_sectors), "periods": int(row[period])}.get(key)
        return value


def table_fragment(csv_path: Path, outcome: str | list[str], frequency: str, model_ids: list[str],
                   headers: list[str], *, suffix: str, terms: list[str] | None = None,
                   note: str | None = None) -> str:
    raw = pd.read_csv(csv_path)
    bandwidth = 12 if frequency == "monthly" else 4
    rows = main_rows(raw, frequency)
    outcomes = outcome if isinstance(outcome, list) else [outcome]*len(model_ids)
    models = [SavedResult(rows[rows.model.eq(model_id)].copy(), dependent)
              for model_id, dependent in zip(model_ids, outcomes, strict=True)]
    assert all(not model.rows.empty for model in models)
    period_label = "Months" if frequency == "monthly" else "Quarters"
    season_label = "month-of-year" if frequency == "monthly" else "quarter-of-year"
    growth_label = "MoM" if frequency == "monthly" else "QoQ"
    notes = ("Sector-clustered CR1 standard errors in parentheses; t(S−1) tests with S sector units. "
             "Each cluster permits arbitrary dependence across that sector’s dates; cross-sector dependence is not covered. "
             "The saved calendar-HAC benchmarks allow shared national shocks and serial dependence. "
             "Stars use the saved two-sided t-test p-values: * p<0.10, ** p<0.05, *** p<0.01. "
             f"{note or f'{growth_label} growth is 100 × one-period change in log output, seasonally adjusted unless labelled NSA; not annualised.'} R² includes fixed effects. "
             "x indicates included fixed effects. Blank cells indicate an omitted control.")
    assert rows.estimator.str.startswith("PyFixest").all(), "Non-PyFixest FE results cannot be labelled migrated"
    custom = {"Estimator": [str(m.rows.iloc[0].estimator)+" "+str(m.rows.iloc[0].estimator_version) for m in models],
              "First period": [m.rows.iloc[0].start for m in models],
              "Last period": [m.rows.iloc[0].end for m in models],
              "Clustering unit": ["Sector / industry" if m.rows.iloc[0].covariance == "Sector cluster" else "Time-series HAC" for m in models],
              "Clusters": [str(int(m.rows.iloc[0].n_sectors)) if m.rows.iloc[0].covariance == "Sector cluster" else "—" for m in models],
              "Inference df": [int(m.rows.iloc[0].inference_df) for m in models],
              "DJO log-growth form": [djo_form(m.dependent) for m in models]}
    assessments = [model_assessment(csv_path, model_id) for model_id in model_ids]
    custom["Y stationarity screen (exact window)"] = [a["y_label"] for a in assessments]
    custom["Y + weather screens (exact window)"] = [a["all_label"] for a in assessments]
    temp_rows = [m.rows[m.rows.term.str.startswith("temp")].iloc[0] for m in models]
    custom["Temperature p-value"] = [p_format(r.p_value) for r in temp_rows]
    custom["Temperature significance"] = [significance_label(r.p_value) for r in temp_rows]
    notes += " Stationarity rows count concordant ADF/KPSS screens under both prespecified lag designs; they are evidence, not proof. Gap/short windows remain unestablished; levels are descriptive. Significance is nominal conditional inference, not a stationarity clearance. "
    from sample_context import model_statistics
    sample_stats = [model_statistics(csv_path, model_id) for model_id in model_ids]
    if "mean_y" in rows:
        sample_stats = [m.rows.iloc[0].to_dict() for m in models]
    if all(sample_stats):
        custom["Mean Y (outcome units)"] = [f"{s['mean_y']:,.3f}" for s in sample_stats]
        custom["SD Y (outcome units)"] = [f"{s['sd_y']:,.3f}" for s in sample_stats]
        notes += " DJO denotes the 100 × log-difference outcome form, not a replication of its annual multicountry design. Mean Y and SD Y use each column's exact sample and displayed outcome units; for log growth these are log points."
    if "time_residual_df" in rows:
        custom["Residual time df"] = [int(m.rows.iloc[0].time_residual_df) for m in models]
    fe_labels = {"sector_season": f"Sector × {season_label} FE", "year": "Year FE", "season_trend": "Season-specific time slopes"}
    fe_order = ["sector_season", "year"] + (["season_trend"] if any("season_trend" in m.__maketables_fixef_string__ for m in models) else [])
    table = mt.ETable(models, model_heads=headers, head_order="dh", coef_fmt="b:.3f* \n (se:.3f)",
        signif_code=[.01, .05, .10], keep=terms or ["temp_c", "rain_100mm", "rh_pct", "wind_mps", "pm25_10ug"], exact_match=True,
        labels={"temp_c": "Mean air temperature (°C)", "rain_100mm": "Rainfall (100 mm)", "rh_pct": "Relative humidity (pp)",
                "wind_mps": "Wind speed (m/s)", "pm25_10ug": "PM2.5 (10 µg/m³)",
                "temp_year_change": "YoY temperature change (°C)", "temp_year_mean": "Trailing-year mean temperature (°C)",
                "temp_year_lag": "Same period last year temperature (°C)",
                "rain_100mm_year_change": "YoY rainfall change (100 mm)", "rain_100mm_year_mean": "Trailing-year mean rainfall (100 mm/period)",
                "rain_100mm_year_lag": "Same period last year rainfall (100 mm)",
                "rh_pct_year_change": "YoY humidity change (pp)", "rh_pct_year_mean": "Trailing-year mean humidity (%)",
                "rh_pct_year_lag": "Same period last year humidity (%)"},
        felabels=fe_labels, feorder=fe_order,
        model_stats=["N", "sectors", "periods", "r2"],
        model_stats_labels={"N": "Observations", "sectors": "Sectors", "periods": period_label, "r2": "R² (including FE)"},
        custom_model_stats=custom,
        stats_order="sc", notes=notes)
    gt = table.make(type="gt", gt_style={"table_width": "100%", "first_col_width": "235px",
        "table_font_size_all": "13px", "source_notes_font_size": "11px", "data_row_padding": "8px",
        "column_labels_padding": "8px"})
    html = namespace_table_ids(gt.as_raw_html(), suffix)
    table_dir = OUT / "tables"
    table_dir.mkdir(exist_ok=True)
    (table_dir / f"{suffix}.html").write_text(html)
    (table_dir / f"{suffix}.tex").write_text(table.make(type="tex"))
    return html


def tables_for(csv_path: Path, outcome: str, frequency: str, key: str) -> tuple[str, str]:
    main = table_fragment(csv_path, outcome, frequency, ["S1", "S2", "S3", "S4", "S5"],
        ["Temperature\nonly", "+ Rainfall", "+ Humidity\nLong-history check", "Before\nCOVID", "Exclude\n2020–2022"], suffix=f"{key}_main")
    ids = ["S6", "S7", "S8"] + (["S9"] if frequency == "quarterly" else [])
    headers = ["Broad sectors\nmatched sample (SA)", "Detailed sectors\nmatched sample (SA)", "NSA growth"]
    if frequency == "quarterly":
        headers.append("Equal-month\ntemperature (SA)")
    # Adjustment varies by column; keep the common outcome heading neutral and
    # label SA/NSA in each column. GT repeats IDs for recurring split spanners.
    robustness = table_fragment(csv_path, outcome.replace(" (SA)", ""), frequency, ids, headers, suffix=f"{key}_robustness")
    return main, robustness


def tables_with_controls(csv_path: Path, outcome: str, frequency: str, key: str) -> tuple[str, str]:
    main = table_fragment(csv_path, outcome, frequency, ["C4", "C1", "C2", "C3", "C0"],
        ["Baseline\nall controls", "Same dates\nrain + humidity", "+ Wind\nno PM2.5", "+ PM2.5\nno wind", "Long history\nno wind / PM2.5"], suffix=f"{key}_controls")
    checks = table_fragment(csv_path, outcome, frequency, ["C4", "C5", "C6"],
        ["Baseline\n90% coverage", "Exclude 2020–22\n90% coverage", "Full controls\n80% coverage"], suffix=f"{key}_controls_checks")
    return main, checks


def build_tables() -> Path:
    panels = []
    contexts = [("va", "Real VA · QoQ growth", OUT / "regression_results.csv", "QoQ real VA log growth (SA)", "quarterly"),
                ("iip-monthly", "IIP · MoM growth", OUT / "iip/monthly/regression_results.csv", "MoM IIP log growth (SA)", "monthly"),
                ("iip-quarterly", "IIP · QoQ growth", OUT / "iip/quarterly/regression_results.csv", "QoQ IIP log growth (SA)", "quarterly")]
    for key, label, path, outcome, frequency in contexts:
        main, robustness = tables_for(path, outcome, frequency, key)
        download = "data:text/csv;base64," + base64.b64encode(path.read_bytes()).decode()
        controls_path = OUT / "extended_controls" / key / "regression_results.csv"
        if controls_path.exists():
            controlled, checks = tables_with_controls(controls_path, outcome, frequency, key)
            extended_download = "data:text/csv;base64,"+base64.b64encode(controls_path.read_bytes()).decode()
            body = (f'<p class="notes">Wind and PM2.5 controls. The baseline includes rainfall, humidity, wind and PM2.5; '
                    f'columns 1–4 use identical retained dates and the final column is long-history robustness. <a href="extended_controls.html">Coverage, sources and interpretation →</a></p>'
                    f'<div class="etable-scroll">{controlled}</div><h3>Pandemic and coverage checks</h3><div class="etable-scroll">{checks}</div>'
                    f'<details><summary>Original long-sample specifications and robustness</summary><div class="etable-scroll">{main}</div>'
                    f'<h3>Sample, sector and measurement checks</h3><div class="etable-scroll">{robustness}</div>'
                    f'<a href="{download}" download="{key}_original_regressions.csv">Download original coefficients ↓</a></details>')
            download = extended_download
        else:
            body = f'<div class="etable-scroll">{main}</div><h3>Sample, sector and measurement checks</h3><div class="etable-scroll">{robustness}</div>'
        panels.append(f'<section id="{key}" class="reg-panel" aria-labelledby="tab-{key}" {"" if key == "va" else "hidden"}>'
                      f'<div class="section-title"><h2>{label}</h2><a href="{download}" download="{key}_regressions.csv">Download coefficients ↓</a></div>'
                      f'{body}</section>')
    template = (ROOT / "templates/regression_tables.html").read_text()
    html = template.replace("__TABLE_PANELS__", "\n".join(panels))
    path = OUT / "regression_tables.html"
    path.write_text(html)
    print(f"Wrote {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    build_tables()
