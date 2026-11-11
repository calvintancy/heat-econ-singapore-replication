"""Build an offline sector-heterogeneity report with native estimation tables."""
from __future__ import annotations

import base64
import html
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import maketables as mt
import numpy as np
import pandas as pd

from heterogeneity import CONTEXTS, OUT, OUTCOMES, ROOT
from regression_tables import namespace_table_ids
from sample_context import model_statistics
from significance import p_format, significance_class, significance_label, significance_stars
from stationarity_context import model_assessment, unit_assessment

BLUE, RED, GRAY, GRID = "#1f77b4", "#d62728", "#64798b", "#dbe4eb"
MODEL_NAMES = {"H1": "Shared controls / year", "H2": "Sector weather / shared year",
               "H3": "Sector weather / sector year", "R1": "Relative to reference sector"}
WEATHER_LABELS = {"temp_c": "Mean air temperature (°C)", "rain_100mm": "Rainfall (100 mm)",
                  "rh_pct": "Relative humidity (pp)", "wind_mps": "Wind speed (m/s)", "pm25_10ug": "PM2.5 (10 µg/m³)"}


def esc(value) -> str:
    return html.escape(str(value))


def fmt(value, digits=3) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}".replace("-", "−")


def pformat(value) -> str:
    return p_format(value)


def significance_badge(value: float) -> str:
    bucket = significance_class(value)
    five_percent = "Unavailable" if bucket == "unavailable" else "Yes" if value < .05 else "No"
    return (f'<span class="sig-tag {bucket}" data-p-value="{value}"'
            f' data-significance="{esc(significance_label(value))}">5%: {five_percent}</span>'
            f'<span class="sig-description">{esc(significance_label(value))}</span>')


def stationarity_cells(assessment: dict) -> tuple[str, str]:
    state = assessment["y_state"]
    category = "screen-compatible" if state in {"stationarity-compatible", "trend-stationarity-compatible"} else "screen-unavailable" if state.startswith("unavailable") else "screen-caution"
    y_label = ("Unavailable · "+assessment["y_label"]) if state.startswith("unavailable") else assessment["y_label"]
    y = (f'<span class="stationarity-flag {category}" data-y-state="{esc(state)}" data-y-label="{esc(assessment["y_label"])}">{esc(y_label)}</span>'
         f'<span class="stationarity-detail">Source-history background: {esc(assessment["background_label"])}</span>')
    x_category = "screen-compatible" if assessment["x_issues"] == 0 else "screen-unavailable" if state.startswith("unavailable") else "screen-caution"
    x = (f'<span class="stationarity-flag {x_category}" data-x-label="{esc(assessment["x_label"])}">{esc(assessment["x_label"])}</span>'
         '<span class="stationarity-detail">Exact estimation window</span>')
    return y, x


def stationarity_summary(assessment: dict) -> str:
    y, x = stationarity_cells(assessment)
    return '<div class="stationarity-cell"><strong>Y:</strong> '+y+'<div class="x-summary"><strong>X:</strong> '+x+'</div></div>'


def uri(path: Path, label: str) -> str:
    mime = "text/csv" if path.suffix == ".csv" else "text/markdown" if path.suffix == ".md" else "application/json"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<a href="data:{mime};base64,{data}" download="{path.parent.name}_{path.name}">{esc(label)} ↓</a>'


def outcome_label(outcome: str, frequency: str) -> str:
    return {"y_short": ("MoM" if frequency == "monthly" else "QoQ")+" log growth (SA)",
            "y_yoy": "YoY log growth (SA)", "y_yoy_nsa": "YoY log growth (NSA)",
            "y_log_level": "100 × log output level (SA)"}[outcome]


@dataclass
class HeterogeneityResult:
    rows: pd.DataFrame
    dependent: str
    model_id: str

    @property
    def __maketables_coef_table__(self):
        return self.rows.set_index("sector_id")[["coefficient", "std_error", "p_value", "ci_lower", "ci_upper"]].rename(
            columns={"coefficient": "b", "std_error": "se", "p_value": "p", "ci_lower": "ci95l", "ci_upper": "ci95u"})

    @property
    def __maketables_depvar__(self):
        return self.dependent

    @property
    def __maketables_fixef_string__(self):
        return "sector_season+"+({"H1": "year", "H2": "year", "H3": "sector_year", "R1": "sector_year+date"}[self.model_id])

    def __maketables_stat__(self, key):
        row = self.rows.iloc[0]
        return {"N": int(row.n_obs), "r2": row.r_squared, "sectors": int(row.n_sectors),
                "periods": int(row.n_periods)}.get(key)


@dataclass
class SeriesResult:
    rows: pd.DataFrame
    dependent: str

    @property
    def __maketables_coef_table__(self):
        return self.rows.set_index("term")[["coefficient", "std_error", "p_value", "ci_lower", "ci_upper"]].rename(
            columns={"coefficient": "b", "std_error": "se", "p_value": "p", "ci_lower": "ci95l", "ci_upper": "ci95u"})

    @property
    def __maketables_depvar__(self):
        return self.dependent

    @property
    def __maketables_fixef_string__(self):
        return "season+year"

    def __maketables_stat__(self, key):
        row = self.rows.iloc[0]
        return {"N": int(row.n_obs), "r2": row.r_squared, "resid_df": int(row.residual_df)}.get(key)


def render_native(table, suffix: str, *, width="100%", first="255px", font="12px") -> str:
    result = table.make(type="gt", gt_style={"table_width": width, "first_col_width": first,
        "table_font_size_all": font, "source_notes_font_size": "11px", "data_row_padding": "7px", "column_labels_padding": "8px"})
    fragment = namespace_table_ids(result.as_raw_html(), suffix)
    folder = OUT / "tables"
    folder.mkdir(exist_ok=True)
    (folder / f"{suffix}.html").write_text(fragment)
    (folder / f"{suffix}.tex").write_text(table.make(type="tex"))
    return fragment


def sector_table(slopes: pd.DataFrame, joint: pd.DataFrame, label: str, frequency: str, suffix: str,
                 *, csv_path: Path, outcome: str, relative=False) -> str:
    ids = ["R1"] if relative else ["H1", "H2", "H3"]
    models = [HeterogeneityResult(slopes[slopes.model.eq(model)], label, model) for model in ids]
    names = slopes[slopes.model.isin(ids)].drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    stats = {"Estimator": [str(m.rows.iloc[0].estimator)+" "+str(m.rows.iloc[0].estimator_version) for m in models],
             "DJO log-growth form": ["Yes · frequency adapted" if outcome in {"y_short", "y_yoy", "y_yoy_nsa"} else "No · log level"]*len(ids),
             "Inference": ["Joint calendar HAC · sector slopes"]*len(ids),
             "Controls": ["Rainfall, humidity, wind, PM2.5" if model.rows.iloc[0].control_set == "full" else "Rainfall, humidity" for model in models],
             "Weather-control slopes": ["Shared" if m == "H1" else "Sector-specific" for m in ids],
             "Panel parameters": [int(model.rows.iloc[0].n_parameters) for model in models],
             "Joint heterogeneity · equal slopes: p": [pformat(joint[joint.model.eq(m)].iloc[0].p_value) for m in ids]}
    means = [model_statistics(csv_path, model, outcome=outcome) for model in ids]
    assert all(means), (csv_path, outcome)
    stats["Mean observed Y (all sectors)"] = [fmt(s["mean_y"]) for s in means]
    stats["SD observed Y (all sectors)"] = [fmt(s["sd_y"]) for s in means]
    assessments = [model_assessment(csv_path, model, outcome=outcome) for model in ids]
    stats["Y stationarity screen · exact dates"] = [assessment["y_label"] for assessment in assessments]
    stats["Y + X screens · exact dates"] = [assessment["all_label"] for assessment in assessments]
    stats["Analysis role"] = [assessment["analysis_role"] for assessment in assessments]
    time = "Months" if frequency == "monthly" else "Quarters"
    lag = 12 if frequency == "monthly" else 4
    note = (f"Bartlett Driscoll–Kraay standard errors, {lag} calendar lags; approximate t(T−1) tests. "
            "* p<0.10, ** p<0.05, *** p<0.01. Sector intervals/stars are pointwise, without multiple-testing adjustment. "
            "The equality row is an approximate F(q,T−1) Wald test of all nonreference deviations being zero, using joint cross-sector covariance. "
            "Absolute sector p-values test betaT + delta_i = 0 with the main/interaction covariance included; relative R1 p-values test delta_i = 0. "
            "All columns use identical sectors and dates. R² includes all fixed effects. "
            "Sector-specific slopes retain calendar HAC: each absolute slope is supported by one clustering unit, so sector-clustered inference would be degenerate. "
            "The DJO label refers to log-difference outcome form only, not the annual multicountry design. "
            "Significance is nominal conditional inference; unresolved stationarity means exploratory. "
            "Ordinary ADF/KPSS screens cannot prove stationarity; gap/short-history screens are unavailable. "
            "Source-history background does not clear the exact window; levels remain descriptive. ")
    if relative:
        reference = slopes[slopes.model.eq("R1")].iloc[0].reference_sector
        note += f"Every displayed coefficient is a difference from {reference}. Its omitted zero is a normalization, not evidence of no heat response. R1 re-expresses H3 differences; finite-parameter covariance corrections differ."
    else:
        note += "H1 shares other weather slopes and year effects. H2 allows sector-specific weather slopes. H3 additionally allows sector-specific year effects and matches separate OLS point estimates."
    table = mt.ETable(models, model_heads=["Date FE\nrelative slopes"] if relative else ["Temperature\ninteractions", "Sector weather\ninteractions", "Fully\ninteracted"],
        head_order="dh", coef_fmt="b:.3f* \n (se:.3f)", signif_code=[.01, .05, .10],
        keep=list(names), exact_match=True, labels=names,
        felabels={"sector_season": "Sector × season FE", "year": "Common year FE",
                  "sector_year": "Sector × year FE", "date": "Full date FE"},
        feorder=["sector_season"]+(["sector_year", "date"] if relative else ["year", "sector_year"]),
        model_stats=["N", "sectors", "periods", "r2"],
        model_stats_labels={"N": "Sector-period observations", "sectors": "Sectors", "periods": time, "r2": "R² (including FE)"},
        custom_model_stats=stats, stats_order="cs", notes=note)
    return render_native(table, suffix)


def series_table(frame: pd.DataFrame, label: str, suffix: str, lag: int, *, csv_path: Path, outcome: str) -> str:
    models = [SeriesResult(g, label) for _, g in frame.groupby("sector_id", sort=True)]
    means = [model_statistics(csv_path, "separate", outcome=outcome, sector_id=m.rows.iloc[0].sector_id) for m in models]
    assert all(means)
    # Separate equations use the same outcome, dates and weather series as H3;
    # the audit explicitly registers those source series through sector_slopes.
    assessments = [unit_assessment(csv_path.parent/"sector_slopes.csv", "H3", model.rows.iloc[0].sector_id, outcome=outcome)
                   for model in models]
    table = mt.ETable(models, model_heads=[textwrap.fill(m.rows.iloc[0].sector, width=18) for m in models], head_order="dh",
        coef_fmt="b:.3f* \n (se:.3f)", signif_code=[.01, .05, .10], labels=WEATHER_LABELS,
        keep=list(WEATHER_LABELS), exact_match=True,
        felabels={"season": "Season FE", "year": "Year FE"}, feorder=["season", "year"],
        model_stats=["N", "resid_df", "r2"], model_stats_labels={"N": "Time observations", "resid_df": "Residual df", "r2": "R²"},
        custom_model_stats={"Estimator": [str(m.rows.iloc[0].estimator)+" "+str(m.rows.iloc[0].estimator_version) for m in models],
                            "DJO log-growth form": ["Yes · frequency adapted" if outcome in {"y_short", "y_yoy", "y_yoy_nsa"} else "No · log level"]*len(models),
                            "Controls": ["Rainfall, humidity, wind, PM2.5" if csv_path.parent.name == "full" else "Rainfall, humidity"]*len(models),
                            "Mean Y (outcome units)": [fmt(s["mean_y"]) for s in means],
                            "SD Y (outcome units)": [fmt(s["sd_y"]) for s in means],
                            "Y screen · exact dates": [assessment["y_label"] for assessment in assessments],
                            "X screens · exact dates": [assessment["x_label"] for assessment in assessments]},
        notes=f"Each column is one sector's own time-series regression on weather, season and year effects. Calendar-aware Bartlett HAC uses {lag} lags; covariance correction T/(T−K) and approximate t(T−1) inference. Temperature point estimates equal H3; SEs differ slightly because H3 uses the full-panel K correction. * p<0.10, ** p<0.05, *** p<0.01; pointwise comparisons. Significance is nominal conditional inference; unresolved stationarity means exploratory. Screens use the corresponding H3 source series and exact dates, not a full-history clearance; levels remain descriptive.")
    return render_native(table, suffix, width="100%", first="200px", font="12px")


def forest(frame: pd.DataFrame, label: str, *, relative=False) -> str:
    model_ids = ["R1"] if relative else ["H1", "H2", "H3"]
    df = frame[frame.model.isin(model_ids)]
    sectors = sorted(df.sector_id.unique())
    names = df.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    left, right, width, row_height = 365, 1000, 1510, 72
    height = 95+len(sectors)*row_height
    low, high = min(0, df.ci_lower.min()), max(0, df.ci_upper.max())
    pad = max((high-low)*.07, .1)
    low, high = low-pad, high+pad
    scale = 10**np.floor(np.log10((high-low)/5))
    step = next(value*scale for value in [1, 2, 2.5, 5, 10] if value*scale >= (high-low)/5)
    low, high = np.floor(low/step)*step, np.ceil(high/step)*step
    x = lambda value: left+(value-low)/(high-low)*(right-left)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(label)}"><title>{esc(label)}</title>',
             '<style>text{font-family:Arial,sans-serif;fill:#15334a;font-size:12px}.tick{fill:#64798b;font-size:11px}.forest-p-label{font-size:11px}</style>',
             '<text x="1020" y="22" class="tick">Model · coefficient · saved p-value · pointwise significance</text>']
    for tick in np.arange(low, high+step/2, step):
        parts.append(f'<line x1="{x(tick):.2f}" x2="{x(tick):.2f}" y1="36" y2="{height-45}" stroke="{GRID}"/><text x="{x(tick):.2f}" y="{height-25}" text-anchor="middle" class="tick">{fmt(tick,1)}</text>')
    parts.append(f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="36" y2="{height-45}" stroke="{GRAY}"/>')
    colors = {"H1": BLUE, "H2": GRAY, "H3": RED, "R1": BLUE}
    for i, sector_id in enumerate(sectors):
        y = 65+i*row_height
        name_lines = textwrap.wrap(names[sector_id], width=43)
        for line_index, line in enumerate(name_lines):
            parts.append(f'<text x="12" y="{y-5+line_index*14}">{esc(line)}</text>')
        for j, model in enumerate(model_ids):
            row = df[df.sector_id.eq(sector_id)&df.model.eq(model)].iloc[0]
            yj = y+(j-(len(model_ids)-1)/2)*17
            significant = pd.notna(row.p_value) and row.p_value < .05
            fill = colors[model] if significant else "white"
            category = significance_class(row.p_value)
            significance_text = significance_label(row.p_value)
            description = (f"{names[sector_id]}; {MODEL_NAMES[model]}: {fmt(row.coefficient)}, "
                           f"CI [{fmt(row.ci_lower)}, {fmt(row.ci_upper)}]; p={pformat(row.p_value)}; "
                           f"{significance_text}; significant at 5%: {'Yes' if significant else 'No'}")
            parts.append(f'<line x1="{x(row.ci_lower):.2f}" x2="{x(row.ci_upper):.2f}" y1="{yj}" y2="{yj}" stroke="{colors[model]}" stroke-width="1.5"/>'
                         f'<circle cx="{x(row.coefficient):.2f}" cy="{yj}" r="{4.2 if significant else 3.2}" fill="{fill}" '
                         f'stroke="{colors[model]}" stroke-width="2" class="forest-mark {category}" '
                         f'data-sector-id="{esc(sector_id)}" data-model="{model}" data-p-value="{row.p_value}"><title>{esc(description)}</title></circle>'
                         f'<text x="1020" y="{yj+4}" class="forest-p-label" font-weight="{600 if significant else 400}">'
                         f'{model}: {fmt(row.coefficient,2)}{significance_stars(row.p_value)} · p={esc(pformat(row.p_value))} · {esc(significance_text)}</text>')
    unit = "Output log points per 1°C" if "level" in label.lower() else "Growth log points per 1°C"
    parts.append(f'<text x="{(left+right)/2}" y="{height-4}" text-anchor="middle" class="tick">{unit}{" relative to reference sector" if relative else ""} · pointwise 95% intervals</text></svg>')
    return "".join(parts)


def compact(headers: list[str], rows: list[list[str]]) -> str:
    return '<div class="compact-scroll"><table class="compact"><thead><tr>'+"".join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+"".join('<tr>'+"".join(f'<td>{cell}</td>' for cell in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def individual_h3_table(slopes: pd.DataFrame, *, csv_path: Path, outcome: str) -> str:
    rows = []
    for row in slopes.loc[slopes.model.eq("H3")].sort_values("sector_id").itertuples():
        assessment = unit_assessment(csv_path, "H3", row.sector_id, outcome=outcome)
        y_screen, x_screen = stationarity_cells(assessment)
        rows.append([esc(row.sector), fmt(row.coefficient)+significance_stars(row.p_value), fmt(row.std_error),
                     esc(pformat(row.p_value)), significance_badge(row.p_value),
                     f'[{fmt(row.ci_lower)}, {fmt(row.ci_upper)}]',
                     '<div class="stationarity-cell">'+y_screen+'</div>', '<div class="stationarity-cell">'+x_screen+'</div>'])
    return ('<div class="individual-significance" data-significance-model="H3">'
            +compact(["Sector", "H3 β per +1°C", "Calendar-HAC SE", "Pointwise p-value", "Pointwise significance", "95% interval", "Y: exact-window screen", "X: weather screen"], rows)
            +'<p class="small">Each p-value tests whether that sector’s H3 slope equals zero, using the saved calendar-HAC covariance. '
            'These are pointwise results without multiple-testing adjustment. They are distinct from the joint equal-slope test below '
            'and from Holm-adjusted pairwise differences. Significance uses unrounded p-values; rounded interval endpoints do not determine the label. '
            '<strong>Significance is nominal conditional inference; unresolved stationarity means exploratory.</strong> '
            'Ordinary ADF/KPSS tests cannot prove stationarity. Screens for gaps or short histories are unavailable. '
            'Source-history background is not clearance of this exact estimation window; levels remain descriptive.</p></div>')


def diagnostics(key: str, outcome: str, folder: Path) -> str:
    summary = pd.read_csv(OUT / "diagnostics/summary.csv", dtype={"sector_id": str})
    df = summary[summary.context.eq(key)&summary.variable.eq(outcome)&summary.specification.eq("primary")]
    parts = []
    for window in ["long", "pre_2020"]:
        rows = []
        for _, row in df[df.window.eq(window)].sort_values("sector_id").iterrows():
            kp = ("" if row.kpss_p_value_relation == "=" else str(row.kpss_p_value_relation))+f"{row.kpss_p_value:.3f}"
            rows.append([esc(row.sector), str(int(row.n_periods)), esc(pformat(row.adf_p_value)), esc(kp),
                         str(int(row.adf_lags)), str(int(row.kpss_lags)), fmt(row.acf_1, 2), esc(row.evidence_5pct)])
        table = compact(["Sector", "N", "ADF p", "KPSS p", "ADF lags", "KPSS lags", "ACF(1)", "Evidence at 5%"], rows)
        if window == "long":
            parts.append(table)
        else:
            parts.append('<details><summary>Pre-2020 contiguous sensitivity</summary>'+table+'</details>')
    persistence = pd.read_csv(folder / "residual_persistence.csv", dtype={"sector_id": str})
    selected = persistence[persistence.outcome.eq(outcome)&persistence.model.eq("H3")]
    p_rows = []
    for _, group in selected.groupby("sector_id", sort=True):
        one = group[group.lag.eq(1)].iloc[0]
        annual = group[group.lag.ne(1)].iloc[0]
        p_rows.append([esc(one.sector), fmt(one.residual_correlation, 2), fmt(annual.residual_correlation, 2), str(int(annual.matched_pairs))])
    return ('<details class="diagnostics"><summary>Stationarity and persistence diagnostics</summary>'
            '<p>ADF tests a unit-root null; KPSS tests level/trend stationarity. Neither proves its null when not rejected. '
            'The paired screen uses the exact contiguous long-sample dates. Output levels include a deterministic trend in the tests; growth tests include a constant. '
            'The ACF is descriptive. YoY overlap and structural breaks affect these diagnostics.</p>'+"".join(parts)
            +'<h3>Fully interacted model residual correlations</h3>'+compact(["Sector", "Lag 1", "Lag one year", "Annual-lag pairs"], p_rows)
            +'<p>Residual correlations preserve actual calendar gaps. They are not unit-root or cointegration tests. '
            'The screened wind/PM2.5 subset has gaps, so ordinary ADF/KPSS is not run after compressing those dates. '
            +uri(OUT / "diagnostics/summary.csv", "Download paired diagnostics")+' · '+uri(OUT / "diagnostics/tests.csv", "Download full test results and warnings")+'</p></details>')


def result_panel(key: str, frequency: str, horizon: int, controls: str, outcome: str) -> str:
    folder = OUT / key / controls
    all_slopes = pd.read_csv(folder / "sector_slopes.csv", dtype={"sector_id": str})
    all_joint = pd.read_csv(folder / "joint_tests.csv")
    slopes = all_slopes[all_slopes.outcome.eq(outcome)&all_slopes.hac_lags.eq(horizon)]
    joint = all_joint[all_joint.outcome.eq(outcome)&all_joint.hac_lags.eq(horizon)]
    main = slopes[slopes.model.eq("H3")].iloc[0]
    controls_label = ("Baseline · rainfall, humidity, wind and PM2.5 · all retained 2017–2025 dates" if controls == "full"
                      else "Long-history robustness · rainfall and humidity; no wind/PM2.5")
    label = outcome_label(outcome, frequency)
    suffix = f"{key}_{controls}_{outcome}"
    native = sector_table(slopes, joint, label, frequency, suffix+"_absolute", csv_path=folder/"sector_slopes.csv", outcome=outcome)
    relative = sector_table(slopes, joint, label+" · difference from reference", frequency, suffix+"_relative", relative=True,
                            csv_path=folder/"sector_slopes.csv", outcome=outcome)
    graphic = forest(slopes, f"{label}: sector temperature slopes")
    relative_graphic = forest(slopes, f"{label}: differences from {main.reference_sector}", relative=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    (figures / (suffix+"_absolute.svg")).write_text(graphic)
    (figures / (suffix+"_relative.svg")).write_text(relative_graphic)
    h1, h3 = joint[joint.model.eq("H1")].iloc[0], joint[joint.model.eq("H3")].iloc[0]
    lead = (f'The joint equal-slope test gives p={pformat(h1.p_value)} ({significance_label(h1.p_value)}) with shared controls/year effects and '
            f'p={pformat(h3.p_value)} ({significance_label(h3.p_value)}) when weather responses and year effects vary by sector. '
            'This tests differences between sectors, rather than whether any individual coefficient has a star.')
    warning = ""
    if controls == "full" and frequency == "quarterly":
        warning += ('<aside class="caution"><strong>Only 29 weather periods.</strong> Each fully flexible sector equation has 12 residual time degrees of freedom. '
                    f'The equality test has {int(h3.restrictions)} restrictions; nominal p-values can be highly unstable. '
                    'After the endpoint exclusion there are only five residual time degrees of freedom. Do not treat tiny p-values as precise evidence.</aside>')
    if outcome == "y_log_level":
        warning += ('<aside class="caution"><strong>Output-level persistence.</strong> The ADF/KPSS screens flag persistence concerns in most sectors. '
                    'These level coefficients are descriptive; year effects and HAC covariance do not establish stationarity or a valid long-run heat relationship.</aside>')
    joint_rows = []
    for model in ["H1", "H2", "H3", "R1"]:
        one = joint[joint.model.eq(model)].iloc[0]
        doubled = all_joint[all_joint.outcome.eq(outcome)&all_joint.model.eq(model)&all_joint.hac_lags.eq(2*horizon)].iloc[0]
        joint_rows.append([esc(model+": "+MODEL_NAMES[model]), fmt(one.f_statistic), str(int(one.restrictions)),
                           esc(pformat(one.p_value)), significance_badge(one.p_value), esc(pformat(doubled.p_value)),
                           significance_badge(doubled.p_value), f'{int(one.covariance_rank)}/{int(one.restrictions)}', f'{one.covariance_condition:.2g}'])
    pairs = pd.read_csv(folder / "pairwise_tests.csv")
    pair_rows = []
    for _, row in pairs[pairs.outcome.eq(outcome)&pairs.model.eq("H3")&pairs.hac_lags.eq(horizon)].iterrows():
        pair_rows.append([esc(row.comparison), fmt(row.coefficient), f'[{fmt(row.ci_lower)}, {fmt(row.ci_upper)}]',
                          esc(pformat(row.p_value)), significance_badge(row.p_value),
                          esc(pformat(row.p_value_holm)), significance_badge(row.p_value_holm)])
    excluded = slopes[slopes.model.eq("H3X")]
    excluded_joint = joint[joint.model.eq("H3X")].iloc[0]
    endpoint_rows = []
    for _, row in excluded.iterrows():
        base = slopes[slopes.model.eq("H3")&slopes.sector_id.eq(row.sector_id)].iloc[0]
        base_assessment = unit_assessment(folder/"sector_slopes.csv", "H3", row.sector_id, outcome=outcome)
        excluded_assessment = unit_assessment(folder/"sector_slopes.csv", "H3X", row.sector_id, outcome=outcome)
        endpoint_rows.append([esc(row.sector), f'{fmt(base.coefficient)}{significance_stars(base.p_value)} ({fmt(base.std_error)})',
                              esc(pformat(base.p_value)), significance_badge(base.p_value),
                              f'{fmt(row.coefficient)}{significance_stars(row.p_value)} ({fmt(row.std_error)})',
                              esc(pformat(row.p_value)), significance_badge(row.p_value), f'[{fmt(row.ci_lower)}, {fmt(row.ci_upper)}]',
                              stationarity_summary(base_assessment), stationarity_summary(excluded_assessment)])
    standalone = pd.read_csv(folder / "separate_sector_coefficients.csv", dtype={"sector_id": str})
    standalone = standalone[standalone.outcome.eq(outcome)&standalone.hac_lags.eq(horizon)]
    own = series_table(standalone, label, suffix+"_separate", horizon, csv_path=folder/"separate_sector_coefficients.csv", outcome=outcome)
    scale_rows = []
    for _, row in slopes[slopes.model.eq("H3")].iterrows():
        stats = model_statistics(folder/"sector_slopes.csv", "H3", outcome=outcome, sector_id=row.sector_id)
        assert stats
        scale_rows.append([esc(row.sector), fmt(stats["mean_y"]), fmt(stats["sd_y"]), fmt(.1*row.coefficient),
                           f'[{fmt(.1*row.ci_lower)}, {fmt(.1*row.ci_upper)}]'])
    scale_table = compact(["Sector", "Mean Y", "SD Y", "+0.1°C association (H3)", "95% interval"], scale_rows)
    covariance_note = ('Joint Wald covariance is singular: no omnibus p-value is reported.' if excluded_joint.status != "available" else
                       'The remaining covariance is full rank, but the short-sample p-value remains approximate.')
    return (f'<section class="result-panel" id="{suffix}" data-control-set="{controls}" {"" if suffix == "va_full_y_short" else "hidden"}>'
            f'<div class="panel-heading"><h2>{esc(label)}</h2><span>{main.start}–{main.end} · {int(main.n_sectors)} sectors · '
            f'{int(main.n_periods)} dates · {int(main.n_obs):,} rows</span></div><p class="baseline-controls">{esc(controls_label)}</p>'
            f'<p class="lead-result">{esc(lead)}</p>{warning}'
            '<div class="view" data-view="table"><div class="etable-scroll">'+native+'</div></div>'
            '<div class="view" data-view="plot" hidden><div class="legend"><span style="color:#1f77b4">● H1: shared controls/year</span>'
            '<span style="color:#64798b">● H2: sector weather</span><span style="color:#d62728">● H3: sector weather/year</span></div>'
            '<p class="significance-legend">● Filled, larger dot: significant at 5% (p&lt;0.05). '
            '○ Open dot: not significant at 5%. Each plotted estimate also has its p-value, stars and significance label in text.</p>'
            '<figure>'+graphic+'<figcaption>Pointwise 95% intervals; sectors are kept in catalog order. '
            '*** p&lt;0.01; ** p&lt;0.05; * p&lt;0.10. Intervals and labels are not adjusted for multiple comparisons.</figcaption></figure></div>'
            '<h3>Is each sector’s H3 slope different from zero?</h3>'+individual_h3_table(slopes, csv_path=folder/"sector_slopes.csv", outcome=outcome)
            +'<h3>Joint test: are all sector slopes equal?</h3>'+compact(["Model", "F", "Restrictions", "p: one-year HAC", "Joint-test significance", "p: two-year HAC", "Joint-test significance · 2 years", "Covariance rank", "Covariance condition"], joint_rows)
            +'<details><summary>Sector means and the size of the +0.1°C association</summary>'+scale_table
            +'<p>Means use the same dates as each sector coefficient. Growth is in log points; log-level means use 100 ln output. '
            'The +0.1°C column rescales the fitted coefficient and interval without dividing by mean growth.</p></details>'
            +'<p class="small">R1 expresses H3 differences with full date fixed effects. It has the same point contrasts and uncorrected covariance; '
            'its p-value differs because the full-panel parameter correction counts the extra date effects. It is not independent corroboration.</p>'
            '<details><summary>Selected sector differences, with multiple-comparison adjustment</summary>'
            +compact(["First sector − second sector", "Difference", "Pointwise 95% interval", "Pointwise p", "Pointwise significance", "Holm-adjusted p", "Holm-adjusted significance"], pair_rows)
            +'<p>Pairs were chosen for economic relevance before comparing results. Holm adjusts the selected family within this model, outcome and bandwidth; '
            'the displayed intervals remain pointwise. The adjusted significance label uses the saved Holm p-value, '
            'not the unadjusted interval. All models and both bandwidths are in the download.</p></details>'
            '<details><summary>Relative sensitivity with full date fixed effects</summary>'
            f'<p>Reference sector: <strong>{esc(main.reference_sector)}</strong>. These are coefficient differences from that sector, '
            'not absolute heat responses. Common national weather terms are absorbed by date effects; the model estimates nonreference sector weather interactions, '
            'sector × season and sector × year effects. Filled dots mark p&lt;0.05 for that coefficient difference; '
            'open dots mark p≥0.05. The text gives the pointwise p-value and significance bucket.</p><div class="etable-scroll">'+relative+'</div><figure>'+relative_graphic+'</figure></details>'
            '<details><summary>Each sector’s own time series: full regression table</summary><p>The fully interacted panel reproduces these temperature point estimates. '
            'Separate fits alone do not provide the cross-sector covariance needed for comparing sectors; the joint panel inference does.</p>'
            '<div class="etable-scroll wide-series">'+own+'</div></details>'
            '<details><summary>Exclude observations dated 2020–2022</summary>'
            f'<p>Fully flexible H3: {int(excluded.iloc[0].n_periods)} dates remain. Equality-test p={esc(pformat(excluded_joint.p_value))}; '
            f'covariance rank {int(excluded_joint.covariance_rank)}/{int(excluded_joint.restrictions)}. {esc(covariance_note)} '
            'For YoY, 2023 observations still compare with 2022, so this is not a fully pandemic-free sample.</p>'
            +compact(["Sector", "Full sample: β (SE)", "Full-sample p", "Full-sample significance", "Exclude endpoint dates: β (SE)", "Excluded-sample p", "Excluded-sample significance", "Excluded-sample 95% interval", "Full-sample stationarity screens", "Excluded-sample stationarity screens"], endpoint_rows)
            +'<p>These sector-specific H3X tests retain calendar HAC and are pointwise. The equality-test p-value above concerns all slopes jointly; '
            'it does not determine any sector’s individual significance.</p></details>'
            +diagnostics(key, outcome, folder)+'<div class="downloads">'
            +uri(folder / "sector_slopes.csv", "Sector slopes")+' · '+uri(folder / "joint_tests.csv", "Joint tests")+' · '
            +uri(folder / "pairwise_tests.csv", "Sector differences")+' · '+uri(folder / "separate_sector_coefficients.csv", "Separate-series regressions")
            +'</div></section>')


def weather_diagnostics() -> str:
    summary = pd.read_csv(OUT / "diagnostics/summary.csv")
    df = summary[summary.kind.eq("weather")&summary.window.eq("long")&summary.specification.eq("primary")]
    rows = []
    for _, row in df.iterrows():
        kp = ("" if row.kpss_p_value_relation == "=" else str(row.kpss_p_value_relation))+f'{row.kpss_p_value:.3f}'
        rows.append([esc(row.context), esc(row.description), "Constant + trend" if row.deterministic == "ct" else "Constant",
                     str(int(row.n_periods)), esc(pformat(row.adf_p_value)), esc(kp), esc(row.evidence_5pct)])
    return compact(["Context", "Weather series", "Test deterministics", "N", "ADF p", "KPSS p", "Evidence at 5%"], rows)


def outcome_template(template: str) -> str:
    """Add NSA beside SA YoY without changing the shared source template."""
    pattern = r'(<button\b[^>]*data-value="y_yoy"[^>]*>).*?(</button>)'
    match = re.search(pattern, template, flags=re.DOTALL)
    if match is None:
        raise ValueError("The heterogeneity template has no SA YoY selector")
    selector = match.group(1)+"YoY log growth (SA)"+match.group(2)
    if 'data-value="y_yoy_nsa"' not in template:
        selector += '<button data-kind="outcome" data-value="y_yoy_nsa" aria-pressed="false">YoY log growth (NSA)</button>'
    template = template[:match.start()]+selector+template[match.end():]
    template = template.replace("Output is seasonally adjusted; growth is not annualised.",
                                "SA and NSA YoY outcomes are labelled separately; other outcomes use SA output. Growth is not annualised.")
    return template.replace("Official SA output is used directly.",
                            "Official SA output and the added NSA YoY growth series are used as labelled; neither YoY definition replaces the other.")


def build_heterogeneity_report() -> Path:
    panels = []
    for key, frequency, _, _, horizon in CONTEXTS:
        for controls in ["full", "weather"]:
            for outcome in OUTCOMES:
                panels.append(result_panel(key, frequency, horizon, controls, outcome))
    template = outcome_template((ROOT / "templates/heterogeneity.html").read_text())
    replacements = {"__PANELS__": "\n".join(panels), "__WEATHER_DIAGNOSTICS__": weather_diagnostics(),
                    "__DIAGNOSTIC_NOTE__": uri(ROOT / "notes/stationarity_methods.md", "Diagnostic methods and limitations"),
                    "__VERIFICATION__": uri(OUT / "verification.json", "Independent heterogeneity verification")}
    for token, value in replacements.items():
        template = template.replace(token, value)
    assert "__PANELS__" not in template
    path = ROOT / "outputs/sector_heterogeneity.html"
    path.write_text(template)
    print(f"Wrote {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    build_heterogeneity_report()
