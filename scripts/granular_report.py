"""Standalone reports for detailed VA sectors and manufacturing production."""
from __future__ import annotations

import base64
import html
import textwrap
from pathlib import Path

import maketables as mt
import pandas as pd
from lxml import html as lxml_html

from aggregate_va_report import AggregateResult, LABELS
from heterogeneity_report import HeterogeneityResult
from inference_display import main_rows
from regression_tables import namespace_table_ids, table_fragment
from significance import p_format, significance_class, significance_label, significance_stars
from stationarity_context import model_assessment, unit_assessment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = OUT / "granular"
INTERACTIONS = OUT / "granular_interactions"
MODEL_OUTCOMES = {"G1": "y_short", "G2": "y_yoy", "G2N": "y_yoy_nsa", "L1": "y_log_level", "L3": "y_raw_level"}
MODELS = list(MODEL_OUTCOMES)
WINDOWS = {"full": "Baseline · rainfall, humidity, wind + PM2.5",
           "exclude": "Robustness · baseline excluding 2020–2022",
           "long": "Long-history robustness · no wind/PM2.5"}
VA_CONTEXTS = {"va-detailed": "19 detailed sectors",
               "va-broad-matched": "11 broad sectors · matched dates"}
IIP_CONTEXTS = {"iip-monthly-detailed": "Monthly · 18 industries",
                "iip-monthly-total": "Monthly · published total",
                "iip-monthly-broad-matched": "Monthly · 6 clusters, matched",
                "iip-quarterly-total": "Quarterly · published total",
                "iip-quarterly-detailed": "Quarterly · 18 industries",
                "iip-quarterly-broad-matched": "Quarterly · 6 clusters, matched"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def significance_badge(p_value: float) -> str:
    bucket = significance_class(p_value)
    five_percent = "Unavailable" if bucket == "unavailable" else "Yes" if p_value < .05 else "No"
    return (f'<span class="sig-tag {bucket}" data-p-value="{p_value}"'
            f' data-significance="{esc(significance_label(p_value))}">5%: {five_percent}</span>'
            f'<span class="sig-description">{esc(significance_label(p_value))}</span>')


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


def labels(context: str) -> tuple[list[str], list[str]]:
    period = "MoM" if "monthly" in context else "QoQ"
    family = "real VA" if context.startswith("va-") else "IIP"
    raw = "Chained-2015 S$ million (SA)" if context.startswith("va-") else "IIP index points · 2025=100 (SA)"
    dependents = [f"{period} {family} log growth (SA)", f"YoY {family} log growth (SA)",
                  f"YoY {family} log growth (NSA)", f"100 × log {family} (SA)", raw]
    return dependents, [f"{period}\nlog growth (SA)", "YoY\nlog growth (SA)", "YoY\nlog growth (NSA)", "Log\nlevel (SA)", "Raw\nlevel (SA)"]


def download(path: Path, label: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    filename = "_".join(path.relative_to(OUT).parts)
    return f'<a class="download" href="data:text/csv;base64,{encoded}" download="{esc(filename)}">{esc(label)} ↓</a>'


def number(value: float, *, signed: bool = False) -> str:
    return "Unavailable" if pd.isna(value) else f"{value:+.3f}" if signed else f"{value:.3f}"


def compact_table(columns: list[str], rows: list[list[str]], *, table_class: str = "") -> str:
    return ('<div class="table-scroll"><table class="compact '+table_class+'"><thead><tr>'
            +"".join('<th>'+esc(column)+'</th>' for column in columns)+'</tr></thead><tbody>'
            +"".join('<tr>'+"".join('<td>'+cell+'</td>' for cell in row)+'</tr>' for row in rows)+'</tbody></table></div>')


def interaction_native(slopes: pd.DataFrame, joint: pd.DataFrame, context: str, window: str, path: Path) -> str:
    dependents, heads = labels(context)
    outcomes = list(MODEL_OUTCOMES.values())
    models = [HeterogeneityResult(slopes.loc[slopes.outcome.eq(outcome)].copy(), dependent, "H1")
              for outcome, dependent in zip(outcomes, dependents, strict=True)]
    names = slopes.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
    benchmark = pd.read_csv(path)
    assessments = [model_assessment(path, model) for model in MODELS]
    tests = [joint.loc[joint.outcome.eq(outcome)].iloc[0] for outcome in outcomes]
    assert slopes.estimator.str.startswith("PyFixest").all()
    custom = {"Estimator": [str(m.rows.iloc[0].estimator)+" "+str(m.rows.iloc[0].estimator_version) for m in models],
              "Other weather slopes": ["Shared"]*len(models), "Year effects": ["Common"]*len(models),
              "Reference industry": [str(models[0].rows.iloc[0].sector)]*len(models),
              "Controls": ["Rainfall, humidity, wind, PM2.5" if window != "long" else "Rainfall, humidity"]*len(models),
              "Panel parameters": [int(model.rows.iloc[0].n_parameters) for model in models],
              "First period": [model.rows.iloc[0].start for model in models],
              "Last period": [model.rows.iloc[0].end for model in models],
              "Inference df": [int(model.rows.iloc[0].inference_df) for model in models],
              "Joint equal-slope test: p": [p_format(test.p_value) for test in tests],
              "Joint-test significance": [significance_label(test.p_value) for test in tests],
              "Mean Y (outcome units)": [f"{benchmark.loc[benchmark.model.eq(model)].iloc[0].mean_y:,.3f}" for model in MODELS],
              "Y screen · exact dates": [assessment["y_label"] for assessment in assessments],
              "Y + X screens · exact dates": [assessment["all_label"] for assessment in assessments],
              "Analysis role": [assessment["analysis_role"] for assessment in assessments]}
    h = 12 if "monthly" in context else 4
    note = (f"Main panel fixed-effects model: sector × temperature interactions, sector × seasonal effects and common year effects. "
            f"Other weather slopes are shared. Joint calendar Driscoll–Kraay HAC uses {h} lags and t(T−1) sector-slope tests. "
            "Each βi is supported by one sector, so sector clustering is not valid for these individual slopes. "
            "Pointwise stars: * p<0.10, ** p<0.05, *** p<0.01; they are not adjusted across sectors. "
            "Each sector p-value tests its total effect: H0 betaT + delta_i = 0 (reference: betaT = 0). The sum SE includes Cov(betaT,delta_i). "
            "The heterogeneity joint row instead tests all nonreference deviations delta_i = 0, using their full cross-sector covariance; it does not test each total slope against zero. "
            "Short quarterly tests have 17/18 restrictions and only 29 or19 dates; even nonsingular nominal F-tests can be unreliable and poorly conditioned. "
            "Stationarity flags reuse the exact Y/X series and dates from the common-slope model. "
            "Significance is nominal conditional inference; unresolved stationarity means exploratory. Levels remain descriptive. "+outcome_note(context))
    etable = mt.ETable(models, model_heads=heads, head_order="dh", coef_fmt="b:.3f* \n (se:.3f)",
                      signif_code=[.01, .05, .1], keep=list(names), exact_match=True, labels=names,
                      felabels={"sector_season": "Sector × season FE", "year": "Common year FE"},
                      feorder=["sector_season", "year"], model_stats=["N", "sectors", "periods", "r2"],
                      model_stats_labels={"N": "Sector-period observations", "sectors": "Sectors", "periods": "Months" if "monthly" in context else "Quarters", "r2": "R² (including FE)"},
                      custom_model_stats=custom, stats_order="sc", notes=note)
    gt = etable.make(type="gt", gt_style={"table_width": "100%", "first_col_width": "255px",
                    "table_font_size_all": "12px", "source_notes_font_size": "11px", "data_row_padding": "7px"})
    suffix = f"granular-interactions-{context}-{window}"
    fragment = namespace_table_ids(gt.as_raw_html(), suffix)
    (OUT / "tables" / f"{suffix}.html").write_text(fragment)
    (OUT / "tables" / f"{suffix}.tex").write_text(etable.make(type="tex"))
    return fragment


def interaction_forest(rows: pd.DataFrame, dependent: str) -> str:
    ordered = rows.sort_values("sector_id")
    left, right, width, row_height = 350, 870, 1400, 47
    height = 85+len(ordered)*row_height
    low, high = min(0, ordered.ci_lower.min()), max(0, ordered.ci_upper.max())
    pad = max((high-low)*.06, .1)
    low, high = low-pad, high+pad
    position = lambda value: left+(value-low)/(high-low)*(right-left)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(dependent)}: main FE interaction slopes">',
             f'<title>{esc(dependent)}: main FE interaction slopes, pointwise 95% intervals</title>',
             '<style>.interaction-forest-label{font:12px Arial,sans-serif;fill:#15334a}.interaction-forest-tick{font:11px Arial,sans-serif;fill:#64798b}</style>',
             '<text x="890" y="21" class="interaction-forest-tick">β · saved p-value · pointwise significance</text>']
    for tick in [low+i*(high-low)/4 for i in range(5)]:
        parts.append(f'<line x1="{position(tick):.2f}" x2="{position(tick):.2f}" y1="31" y2="{height-40}" stroke="#dbe4eb"/>'
                     f'<text x="{position(tick):.2f}" y="{height-23}" text-anchor="middle" class="interaction-forest-tick">{tick:.1f}</text>')
    parts.append(f'<line x1="{position(0):.2f}" x2="{position(0):.2f}" y1="31" y2="{height-40}" stroke="#64798b"/>')
    for index, row in enumerate(ordered.itertuples()):
        y = 58+index*row_height
        for line_index, line in enumerate(textwrap.wrap(row.sector, width=40)):
            parts.append(f'<text x="8" y="{y-5+line_index*14}" class="interaction-forest-label">{esc(line)}</text>')
        significant = pd.notna(row.p_value) and row.p_value < .05
        parts.append(f'<line x1="{position(row.ci_lower):.2f}" x2="{position(row.ci_upper):.2f}" y1="{y}" y2="{y}" stroke="#1f77b4" stroke-width="1.5"/>'
                     f'<circle cx="{position(row.coefficient):.2f}" cy="{y}" r="{4.3 if significant else 3.2}" fill="{"#1f77b4" if significant else "white"}" '
                     f'stroke="#1f77b4" stroke-width="2" class="interaction-forest-mark {significance_class(row.p_value)}" '
                     f'data-sector-id="{esc(row.sector_id)}" data-p-value="{row.p_value}"><title>{esc(row.sector)}; '
                     f'β={row.coefficient:.3f}; p={esc(p_format(row.p_value))}; {esc(significance_label(row.p_value))}</title></circle>'
                     f'<text x="890" y="{y+4}" class="interaction-forest-label">{row.coefficient:+.3f}{significance_stars(row.p_value)} · '
                     f'p={esc(p_format(row.p_value))} · {esc(significance_label(row.p_value))}</text>')
    parts.append(f'<text x="{(left+right)/2}" y="{height-4}" text-anchor="middle" class="interaction-forest-tick">Outcome units per +1°C · pointwise 95% intervals</text></svg>')
    return "".join(parts)


def interaction_results(path: Path, context: str, window: str) -> str:
    folder = INTERACTIONS / context / window
    h = 12 if "monthly" in context else 4
    slopes = pd.read_csv(folder/"sector_slopes.csv", dtype={"sector_id": str})
    slopes = slopes.loc[slopes.model.eq("H1") & slopes.hac_lags.eq(h)].copy()
    joint = pd.read_csv(folder/"joint_tests.csv")
    joint = joint.loc[joint.model.eq("H1") & joint.hac_lags.eq(h)]
    pairs = pd.read_csv(folder/"pairwise_tests.csv")
    pairs = pairs.loc[pairs.model.eq("H1") & pairs.hac_lags.eq(h)]
    components = pd.read_csv(folder/"interaction_components.csv", dtype={"sector_id": str, "reference_sector_id": str})
    components = components.loc[components.model.eq("H1") & components.hac_lags.eq(h)]
    source = pd.read_csv(folder/"common_sample.csv", dtype={"sector_id": str})
    if window != "long":
        assert slopes.formula.str.contains("pm25_10ug", regex=False).all() and slopes.formula.str.contains("wind_mps", regex=False).all(), "Full-control FE view omits wind or PM2.5"
    benchmark_sample = pd.read_csv(path.parent/"samples/G1.csv", dtype={"sector_id": str})
    period = "month" if "monthly" in context else "quarter"
    keys = ["sector_id", period]
    assert source.sort_values(keys)[keys].reset_index(drop=True).equals(benchmark_sample.sort_values(keys)[keys].reset_index(drop=True)), "Interaction and benchmark sample dates differ"
    assert set(slopes.outcome) == set(MODEL_OUTCOMES.values())
    own = pd.read_csv(path.parent/"separate_sector_coefficients.csv", dtype={"sector_id": str})
    dependents, _ = labels(context)
    options = "".join(f'<option value="{model}">{esc(dependent)}</option>' for model, dependent in zip(MODELS, dependents, strict=True))
    panels = []
    for model, outcome, dependent in zip(MODELS, MODEL_OUTCOMES.values(), dependents, strict=True):
        selected = slopes.loc[slopes.outcome.eq(outcome)].sort_values("sector_id")
        series = own.loc[own.model.eq(model)].set_index("sector_id")
        assert selected.sector_id.is_unique and set(selected.sector_id) == set(series.index)
        test = joint.loc[joint.outcome.eq(outcome)].iloc[0]
        selected_components = components.loc[components.outcome.eq(outcome)]
        base = selected_components.loc[selected_components.component.eq("base")].iloc[0]
        deviations = selected_components.loc[selected_components.component.eq("interaction")].set_index("sector_id")
        readout, comparison = [], []
        for row in selected.itertuples():
            assessment = unit_assessment(path, model, row.sector_id)
            y_screen, x_screen = stationarity_cells(assessment)
            ts = series.loc[row.sector_id]
            assert int(ts.n_obs) == int(row.n_periods) and ts.start == row.start and ts.end == row.end, "FE and own-series dates differ"
            delta = deviations.loc[row.sector_id]
            assert abs(float(base.coefficient)+float(delta.coefficient)-row.coefficient) < max(1e-7,abs(row.coefficient)*1e-9), "Reported impact is not temperature + interaction"
            mapping = (f'<span class="stationarity-detail">βT={number(base.coefficient,signed=True)}; δᵢ={number(delta.coefficient,signed=True)}'
                       +(' (reference normalization)' if bool(delta.normalization) else '')+'</span>')
            mean_y = float(source.loc[source.sector_id.eq(row.sector_id), outcome].mean())
            readout.append([f'<span class="sector-name">{esc(row.sector)}</span><span class="series-id">{esc(row.sector_id)}</span>',
                            number(mean_y), number(row.coefficient, signed=True)+significance_stars(row.p_value)+mapping, number(row.std_error),
                            esc(p_format(row.p_value)), significance_badge(row.p_value),
                            f'[{number(row.ci_lower,signed=True)}, {number(row.ci_upper,signed=True)}]', number(.1*row.coefficient,signed=True),
                            '<div class="stationarity-cell">'+y_screen+'</div>', '<div class="stationarity-cell">'+x_screen+'</div>'])
            comparison.append([esc(row.sector), number(row.coefficient,signed=True)+significance_stars(row.p_value), number(row.std_error), esc(p_format(row.p_value)),
                               number(ts.coefficient,signed=True)+significance_stars(ts.p_value), number(ts.std_error), esc(p_format(ts.p_value)), significance_badge(ts.p_value)])
        pair_rows = []
        family = pairs.loc[pairs.outcome.eq(outcome)]
        for row in family.itertuples():
            pair_rows.append([esc(row.comparison), number(row.coefficient,signed=True), f'[{number(row.ci_lower,signed=True)}, {number(row.ci_upper,signed=True)}]',
                              esc(p_format(row.p_value)), significance_badge(row.p_value), esc(p_format(row.p_value_holm)), significance_badge(row.p_value_holm)])
        joint_info = (f'<div class="joint-readout"><strong>Joint heterogeneity test · H₀: all δᵢ=0:</strong> F({int(test.numerator_df)}, {int(test.denominator_df)})={number(test.f_statistic)}, '
                      f'{int(test.restrictions)} restrictions; p={esc(p_format(test.p_value))}. {significance_badge(test.p_value)}'
                      f'<span class="stationarity-detail">Status: {esc(test.status)} · restriction-covariance rank {int(test.covariance_rank)}/{int(test.restrictions)}. '
                      'This tests whether all sector temperature slopes are equal; it does not determine which individual slopes differ from zero.</span></div>')
        if "monthly" not in context and int(test.n_periods) <= 29:
            joint_info += (f'<p class="callout"><strong>Short quarterly joint test:</strong> {int(test.restrictions)} restrictions with only {int(test.n_periods)} weather dates. '
                           'Even a full-rank covariance can be poorly conditioned and its nominal F-test unreliable in this small sample. '
                           'A tiny p-value is not precise finite-sample evidence. Singular cases have no omnibus p-value.</p>')
        panels.append(f'<div class="interaction-panel" data-interaction-model="{model}" data-outcome="{outcome}" {"" if model == "G1" else "hidden"}>'
                      f'<p class="units">Y = {esc(dependent)}. Reference industry: <strong>{esc(base.sector)}</strong>; βT is its temperature slope. Every other reported impact is βT + δᵢ. β and the +0.1°C response use these outcome units.</p>'+joint_info
                      +compact_table(["Sector / industry", "Mean Y", "Total βT + δᵢ per +1°C", "Joint-HAC SE", "Total-effect p · H₀: βT + δᵢ=0", "Pointwise significance", "95% CI", "β × +0.1°C", "Y: exact-window screen", "X: weather screen"], readout, table_class="interaction-readout")
                      +'<details class="evidence"><summary>Coefficient plot · main FE interactions</summary><p class="notes">Filled, larger dots mean p&lt;0.05; open dots mean p≥0.05. '
                      'The p-value and significance label are also printed. Intervals are pointwise.</p><figure class="interaction-forest">'+interaction_forest(selected,dependent)+'</figure></details>'
                      +'<details class="evidence"><summary>Compare FE interactions with separate time-series robustness</summary>'
                      +compact_table(["Sector", "Main FE β", "Main FE HAC SE", "Main FE p", "Own-series β", "Own-series HAC SE", "Own-series p", "Own-series significance"], comparison, table_class="interaction-ts-comparison")
                      +'<p class="notes">The main FE panel shares the other weather slopes and common year effects. Each separate time-series regression estimates its own weather slopes and year effects. '
                      'The periods and sector series match exactly. These are different restrictions on the same data; each retains its own saved calendar-HAC inference.</p></details>'
                      +'<details class="evidence"><summary>Pairwise sector differences · Holm-adjusted family</summary>'
                      +compact_table(["First sector − second sector", "FE slope difference", "Pointwise 95% CI", "Pointwise p", "Pointwise significance", "Holm-adjusted p", "Holm-adjusted significance"],pair_rows)
                      +f'<p class="notes">Holm adjusts the {len(family)} displayed contrasts within this outcome, FE specification and bandwidth. '
                      'Adjusted labels use the saved Holm p-values; intervals remain pointwise. These compare two slopes, rather than testing one slope against zero.</p></details></div>')
    selector_id = f"interaction-outcome-{context}-{window}"
    return ('<div class="interaction-results"><h3>Main analysis · sector × temperature interactions in a panel FE model</h3>'
            '<p class="interaction-formula">Yᵢₜ = βT Tₜ + Σₛ≠ref δₛ 1(i=s)Tₜ + γ′Wₜ + αᵢ,season(t) + λyear(t) + εᵢₜ</p>'
            +f'<p class="interaction-controls" data-control-window="{window}"><strong>Controls:</strong> '
            +('rainfall, relative humidity, wind speed and PM2.5.' if window != "long" else 'rainfall and relative humidity; wind and PM2.5 are omitted for this long-history robustness check.')+'</p>'
            +'<p class="notes">'+('The baseline uses all retained quality-screened dates in 2017–2025; COVID dates remain in the baseline. '
              if window == "full" else 'This is a robustness window relative to the full-control baseline. ')
            +'PyFixest estimates temperature plus temperature × industry interactions. The reference industry effect is βT; every other industry effect is βT + δᵢ. The native table and sector readout report these total effects. For industry i, H₀: βT + δᵢ=0 is one restriction; its standard error uses Var(βT)+Var(δᵢ)+2Cov(βT,δᵢ). The deviation δᵢ alone compares that industry with the reference. The joint heterogeneity box tests all such deviations together. Other weather slopes and year effects are shared; seasonal fixed effects vary by sector. '
            'This is the main detailed-sector heterogeneity specification. Separate time-series equations are a robustness check. '
            'Individual βᵢ coefficients cannot use sector-clustered inference because each is supported by one unit; joint calendar HAC retains cross-sector covariance.</p>'
            '<div class="etable-scroll interaction-native">'+interaction_native(slopes,joint,context,window,path)+'</div>'
            f'<label class="select-label" for="{selector_id}">Inspect significance, sector comparisons and the plot</label>'
            f'<select id="{selector_id}" class="interaction-select">{options}</select>'+"".join(panels)
            +'<p class="notes"><strong>Significance is nominal conditional inference; unresolved stationarity means exploratory.</strong> '
            'Exact-window stationarity flags reuse the same Y/X series and dates as the common-slope benchmark; they do not certify the FE model or causal identification. '
            'Full-history background is not exact-window clearance. Level specifications remain descriptive sensitivities.</p>'
            '<div class="downloads">'+download(folder/"sector_slopes.csv","Download main FE slopes, p-values and HAC bands")
            +' · '+download(folder/"interaction_components.csv","Native temperature and interaction components")+' · '+download(folder/"component_covariances.csv","Component joint covariance")
            +' · '+download(folder/"joint_tests.csv","Joint equality tests")+' · '+download(folder/"pairwise_tests.csv","Pairwise tests and Holm adjustment")+'</div></div>')


def outcome_note(context: str) -> str:
    monthly = "monthly" in context
    horizon = 12 if monthly else 4
    raw = "chained-2015 S$ million" if context.startswith("va-") else "IIP index points (2025=100)"
    return (f"All columns use identical dates. G1: 100[ln A_SA(t)−ln A_SA(t−1)]; G2: 100[ln A_SA(t)−ln A_SA(t−{horizon})]; "
            f"G2N: 100[ln A_NSA(t)−ln A_NSA(t−{horizon})]. L1: 100 ln A_SA(t); L3: A_SA(t) in {raw}. "
            "SA YoY is retained and NSA YoY is shown alongside it. Growth is not annualised. "
            "Temperature and the other weather controls enter in current-period levels. "
            "DJO identifies the log-growth transformation with frequency adapted, not the full annual multicountry design.")


def national_table(path: Path, context: str, window: str) -> str:
    frequency = "monthly" if "monthly" in context else "quarterly"
    h = 12 if frequency == "monthly" else 4
    raw = pd.read_csv(path)
    rows = main_rows(raw, frequency)
    dependents, heads = labels(context)
    models = [AggregateResult(rows.loc[rows.model.eq(model)].copy(), dependent)
              for model, dependent in zip(MODELS, dependents, strict=True)]
    season = "Month-of-year" if frequency == "monthly" else "Quarter-of-year"
    custom = {"Estimator": [str(m.rows.iloc[0].estimator)+" "+str(m.rows.iloc[0].estimator_version) for m in models],
              "First period": [m.rows.iloc[0].start for m in models],
              "Last period": [m.rows.iloc[0].end for m in models],
              "Mean Y (outcome units)": [f"{m.rows.iloc[0].mean_y:,.3f}" for m in models],
              "SD Y (outcome units)": [f"{m.rows.iloc[0].sd_y:,.3f}" for m in models],
              "Residual time df": [int(m.rows.iloc[0].time_residual_df) for m in models],
              "Inference df": [int(m.rows.iloc[0].inference_df) for m in models],
              "DJO log-growth form": ["Yes · frequency adapted" if model in {"G1", "G2", "G2N"} else "No · levels" for model in MODELS]}
    assessments = [model_assessment(path, model) for model in MODELS]
    custom["Y stationarity screen · exact dates"] = [assessment["y_label"] for assessment in assessments]
    custom["Y + X screens · exact dates"] = [assessment["all_label"] for assessment in assessments]
    custom["Analysis role"] = [assessment["analysis_role"] for assessment in assessments]
    note = (f"Bartlett time-series HAC standard errors in parentheses, {h} calendar lags; t(T−1) tests. "
            "The official weighted manufacturing total is one national series: sector clustering is undefined. "
            "The singleton-panel implementation uses the T/(T−K) correction. "
            "Stars: * p<0.10, ** p<0.05, *** p<0.01. R² includes fixed effects. "+outcome_note(context)
            +" Significance is nominal conditional inference; unresolved stationarity makes the result exploratory. "
            "Ordinary ADF/KPSS screens cannot prove stationarity; gap/short-window screens are unavailable. "
            "Source-history background does not clear the exact estimation window. Levels remain descriptive sensitivities.")
    table = mt.ETable(models, model_heads=heads, head_order="dh", coef_fmt="b:.3f* \n (se:.3f)",
                     signif_code=[.01, .05, .1], keep=list(LABELS)[:5], exact_match=True, labels=LABELS,
                     felabels={"season": season+" FE", "year": "Year FE"}, feorder=["season", "year"],
                     model_stats=["N", "r2"], model_stats_labels={"N": "Observed periods", "r2": "R² (including FE)"},
                     custom_model_stats=custom, stats_order="sc", notes=note)
    gt = table.make(type="gt", gt_style={"table_width": "100%", "first_col_width": "235px",
                    "table_font_size_all": "13px", "source_notes_font_size": "11px", "data_row_padding": "8px"})
    suffix = f"granular-{context}-{window}"
    fragment = namespace_table_ids(gt.as_raw_html(), suffix)
    (OUT / "tables" / f"{suffix}.html").write_text(fragment)
    (OUT / "tables" / f"{suffix}.tex").write_text(table.make(type="tex"))
    return fragment


def inference_comparison(raw: pd.DataFrame, context: str) -> str:
    frequency = "monthly" if "monthly" in context else "quarterly"
    h = 12 if frequency == "monthly" else 4
    primary = main_rows(raw, frequency)
    national = context.endswith("total")
    dependents, _ = labels(context)
    body = []
    for model, dependent in zip(MODELS, dependents, strict=True):
        point = primary.loc[primary.model.eq(model) & primary.term.eq("temp_c")].iloc[0]
        hac = raw.loc[raw.model.eq(model) & raw.term.eq("temp_c") & raw.covariance.eq("Driscoll-Kraay")]
        base = hac.loc[hac.hac_lags.eq(h)].iloc[0]
        doubled = hac.loc[hac.hac_lags.eq(2*h)].iloc[0]
        cells = [esc(dependent), f"{point.coefficient:+.3f}{significance_stars(point.p_value)}", f"{point.std_error:.3f}",
                 esc(p_format(point.p_value)), significance_badge(point.p_value),
                 f"[{point.ci_lower:+.3f}, {point.ci_upper:+.3f}]", str(int(point.inference_df))]
        if not national:
            cells += [f"{base.std_error:.3f}", f"[{base.ci_lower:+.3f}, {base.ci_upper:+.3f}]"]
        cells += [f"{doubled.std_error:.3f}", f"[{doubled.ci_lower:+.3f}, {doubled.ci_upper:+.3f}]"]
        body.append("<tr>"+"".join(f"<td>{cell}</td>" for cell in cells)+"</tr>")
    columns = ["Outcome", "β per +1°C", f"HAC SE · {h} lags" if national else "Sector CR1 SE",
               "Main p-value", "Main significance", "Main 95% CI", "Main df"]
    if not national:
        columns += [f"DK SE · {h} lags", "DK 95% CI"]
    columns += [f"HAC SE · {2*h} lags" if national else f"DK SE · {2*h} lags", "Doubled-lag 95% CI"]
    return ('<div class="table-scroll"><table class="compact"><thead><tr>'+"".join(f"<th>{esc(c)}</th>" for c in columns)
            +"</tr></thead><tbody>"+"".join(body)+"</tbody></table></div>"
            '<p class="notes">Every row compares the same fitted coefficient on the same observations. '
            'Only the covariance estimator or bandwidth changes. DK sums scores across sectors within each calendar date before applying HAC; '
            'sector CR1 sums across dates within each sector. These address different dependence assumptions.</p>')


def separate_sector_tables(path: Path, context: str, window: str) -> str:
    source = path.parent / "separate_sector_coefficients.csv"
    if not source.exists():
        detailed = "va-detailed" if context.startswith("va-") else ("iip-monthly-detailed" if "monthly" in context else "iip-quarterly-detailed")
        return ('<div class="individual-link"><p>The detailed-sector view leads with panel FE temperature interactions and retains individual time-series regressions as robustness.</p>'
                f'<button type="button" class="text-button" data-open-context="{detailed}">Open detailed-sector estimates →</button></div>')
    rows = pd.read_csv(source, dtype={"sector_id": str})
    assert not rows.covariance.eq("Sector cluster").any()
    dependents, _ = labels(context)
    uid = f"{context}-{window}-individual"
    options = "".join(f'<option value="{model}">{esc(dependent)}</option>' for model, dependent in zip(MODELS, dependents, strict=True))
    panels = []
    for model, dependent in zip(MODELS, dependents, strict=True):
        selected = rows.loc[rows.model.eq(model)].sort_values("sector_id")
        assert selected.sector_id.is_unique
        body = []
        for row in selected.itertuples():
            assessment = unit_assessment(source, model, row.sector_id)
            y_screen, x_screen = stationarity_cells(assessment)
            body.append(f'<tr data-sector-id="{esc(row.sector_id)}"><td><span class="sector-name">{esc(row.sector)}</span><span class="series-id">{esc(row.sector_id)}</span></td>'
                        f'<td>{row.mean_y:,.3f}</td><td>{row.coefficient:+.3f}{significance_stars(row.p_value)}</td><td>{row.std_error:.3f}</td>'
                        f'<td>{esc(p_format(row.p_value))}</td><td>{significance_badge(row.p_value)}</td>'
                        f'<td>[{row.ci_lower:+.3f}, {row.ci_upper:+.3f}]</td><td>{.1*row.coefficient:+.3f}</td>'
                        f'<td>{int(row.time_residual_df)}</td><td class="stationarity-cell">{y_screen}</td>'
                        f'<td class="stationarity-cell">{x_screen}</td></tr>')
        panels.append(f'<div class="individual-panel" data-individual-model="{model}" {"" if model == "G1" else "hidden"}>'
                      f'<p class="units">Y = {esc(dependent)}. Mean Y, β and the +0.1°C response use these same outcome units.</p>'
                      '<div class="table-scroll"><table class="compact sector-table"><thead><tr><th>Sector / industry</th>'
                      '<th>Mean Y</th><th>β per +1°C</th><th>HAC SE</th><th>HAC p-value</th><th>Pointwise significance</th>'
                      '<th>95% CI</th><th>β × +0.1°C</th><th>Residual df</th><th>Y: exact-window screen</th><th>X: weather screen</th>'
                      '</tr></thead><tbody>'+"".join(body)+'</tbody></table></div></div>')
    return (f'<div class="individual-tables"><label class="select-label" for="{uid}">Choose the dependent variable</label>'
            f'<select id="{uid}" class="individual-select">{options}</select>'+"".join(panels)
            +'<p class="notes">Each row is a separate time-series regression with its own weather coefficients, seasonal effects and year effects. '
            'HAC uses 12 monthly or four quarterly calendar lags and t(T−1) intervals. One sector cannot support sector-clustered inference; '
            'these are the explicit individual-series exception. Residual df counts the observations left after estimating all regressors. '
            'Short quarterly samples can have only five residual degrees of freedom. Significance uses the unrounded saved HAC p-value, not the rounded interval. '
            '*** p&lt;0.01 = significant at 1%; ** p&lt;0.05 = significant at 5%; * p&lt;0.10 = 10% only; otherwise not significant at 10%. '
            'The 5% badge is Yes only when p&lt;0.05. These are pointwise tests of each sector’s slope against zero; '
            'neither p-values nor intervals are adjusted for testing many sectors. '
            '<strong>Significance is nominal conditional inference; unresolved stationarity means exploratory.</strong> '
            'Ordinary ADF/KPSS tests cannot prove stationarity. Screens for gaps or short histories are unavailable; '
            'full source-history background is not clearance of the exact estimation window. Level specifications remain descriptive.</p>'
            +download(source, "Download every individual-sector outcome and HAC interval")+'</div>')


def result_view(context: str, name: str, window: str, first: bool) -> str:
    path = DATA / context / window / "regression_results.csv"
    raw = pd.read_csv(path)
    frequency = "monthly" if "monthly" in context else "quarterly"
    rows = main_rows(raw, frequency)
    point = rows.loc[rows.model.eq("G1") & rows.term.eq("temp_c")].iloc[0]
    national = context.endswith("total")
    detailed = context.endswith("detailed")
    dependents, heads = labels(context)
    suffix = f"granular-{context}-{window}"
    table = national_table(path, context, window) if national else table_fragment(
        path, dependents, frequency, MODELS, heads, suffix=suffix, note=outcome_note(context))
    count_col = "n_months" if frequency == "monthly" else "n_quarters"
    period = "months" if frequency == "monthly" else "quarters"
    short = "MoM" if frequency == "monthly" else "QoQ"
    inference = (f"Main FE interactions · joint calendar HAC" if detailed else
                 f"Calendar HAC · {12 if frequency == 'monthly' else 4} lags" if national else f"Sector CR1 · {int(point.n_sectors)} clusters")
    exclusions = ('<p class="sample-note">Observations dated 2020–2022 are excluded after growth is computed. '
                  'A YoY observation dated 2023 can still use a 2022 base period; retained calendar gaps remain in HAC lag calculations.</p>') if window == "exclude" else ""
    weighting = ('<p class="sample-note">This is the published weighted national manufacturing index. Its long history starts before the detailed panel. '
                 'Use the matched six-cluster and 18-industry views for taxonomy comparisons.</p>') if national else (
                 '<p class="sample-note">The pooled panel gives each sector-period observation equal weight. '
                 'Splitting a broad sector into several detailed sectors changes its implicit weight, even on matched dates. '
                 'This coefficient is an average sector association; it does not aggregate to the national response.</p>')
    metrics = ('<div class="metric-grid"><div class="metric"><span>Mean '+short+' growth</span>'
               f'<strong>{point.mean_y:+.3f}</strong><small>log points per {"month" if frequency == "monthly" else "quarter"}</small></div>'
               f'<div class="metric"><span>Common-slope response to +0.1°C · {short}</span><strong>{.1*point.coefficient:+.3f}</strong>'
               '<small>log points, conditional association</small></div>'
               f'<div class="metric"><span>95% interval · +0.1°C response</span><strong class="interval">[{.1*point.ci_lower:+.3f}, {.1*point.ci_upper:+.3f}]</strong>'
               '<small>same outcome units; current-period temperature</small></div></div>')
    common_table = ('<h3>Clearly labelled dependent variables · SA and NSA YoY retained</h3><div class="etable-scroll">'+table+'</div>'
                    '<div class="downloads">'+download(path, "Download common-slope coefficients, CR1 and both HAC bandwidths")+'</div>'
                    '<details class="evidence"><summary>Compare common-slope inference with calendar HAC</summary>'+inference_comparison(raw,context)+'</details>')
    if detailed:
        body = (exclusions+interaction_results(path,context,window)
                +'<details class="evidence common-slope-benchmark"><summary>Benchmark · common temperature slope across sectors, clustered by sector</summary>'
                +weighting+metrics+common_table+'</details>'
                '<details class="evidence time-series-robustness"><summary>Robustness · each sector’s own time-series regression</summary>'
                +separate_sector_tables(path,context,window)+'</details>')
    else:
        body = (metrics+weighting+exclusions+common_table
                +'<details class="evidence"><summary>Detailed-sector heterogeneity and time-series robustness</summary>'
                +separate_sector_tables(path,context,window)+'</details>')
    return (f'<section id="view-{context}-{window}" class="result-view" data-context="{context}" data-window="{window}" '
            f'aria-label="{esc(name+"; "+WINDOWS[window])}" {"" if first else "hidden"}>'
            '<div class="view-heading"><div><p class="eyebrow">'+esc(WINDOWS[window])+'</p><h2>'+esc(name)+'</h2></div>'
            '<span class="inference-badge">'+esc(inference)+'</span></div>'
            f'<div class="sample-strip"><span><strong>{esc(point.start)}–{esc(point.end)}</strong> estimation dates</span>'
            f'<span><strong>{int(point[count_col])}</strong> observed {period}</span><span><strong>{int(point.n_sectors)}</strong> '
            f'{"national series" if national else "sector units"}</span><span><strong>{int(point.n_obs):,}</strong> observations</span></div>'
            +body+'</section>')


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f3f7fa;color:#15334a;font:15px/1.65 'Avenir Next','Segoe UI',Arial,sans-serif}a{color:#1f77b4;text-underline-offset:4px}main{max-width:1360px;margin:auto;padding:36px 35px 60px}.topline{display:flex;justify-content:space-between;align-items:center;gap:20px;border-bottom:1px solid #dbe4eb;padding-bottom:18px;margin-bottom:30px}.topline a{font-size:12px;text-decoration:none}.eyebrow{font:11px Menlo,Consolas,monospace;letter-spacing:1.4px;text-transform:uppercase;color:#64798b;margin:0 0 9px}h1{font:600 48px/1.1 'Avenir Next Condensed','Arial Narrow',Arial,sans-serif;letter-spacing:-1px;max-width:950px;margin:12px 0 17px}header .lede{color:#64798b;max-width:940px;font-size:16px;margin:0 0 24px}.intro-grid{display:grid;grid-template-columns:1.3fr 1fr;gap:28px;background:#eaf2f8;border-left:4px solid #1f77b4;padding:20px 25px;margin:25px 0}.intro-grid p{margin:0}.intro-grid h2{font-size:15px;margin:0 0 8px}.intro-grid .small{font-size:13px}.controls{display:flex;gap:22px;flex-wrap:wrap;background:white;border:1px solid #dbe4eb;border-radius:12px;padding:18px 22px;margin:28px 0 16px;position:sticky;top:0;z-index:3;box-shadow:0 5px 17px #15334a08}.control{display:flex;flex-direction:column;gap:5px}.control label,.select-label{font-size:11px;font-weight:700;letter-spacing:.4px;color:#64798b;text-transform:uppercase}select{font:600 13px 'Avenir Next',Arial,sans-serif;color:#15334a;border:1px solid #c8d8e5;border-radius:6px;padding:10px 34px 10px 11px;background:white;max-width:100%}button{font:600 12px 'Avenir Next',Arial,sans-serif;cursor:pointer}.print{border:1px solid #c8d8e5;border-radius:6px;padding:9px 12px;background:white;color:#15334a;white-space:nowrap}.result-view{background:white;border:1px solid #dbe4eb;border-radius:12px;padding:26px 28px 30px}.result-view[hidden],.individual-panel[hidden]{display:none}.view-heading{display:flex;justify-content:space-between;align-items:center;gap:18px}.view-heading h2{font:600 28px 'Avenir Next Condensed',Arial,sans-serif;margin:0}.inference-badge{font-size:11px;font-weight:600;white-space:nowrap;padding:6px 11px;background:#eaf2f8;border-radius:20px;color:#1a5f8b}.sample-strip{display:flex;gap:18px;flex-wrap:wrap;padding:17px 0;border-bottom:1px solid #dbe4eb;font-size:12px;color:#64798b}.sample-strip strong{font-weight:600;color:#15334a}.metric-grid{display:grid;grid-template-columns:1fr 1fr 1.25fr;gap:1px;background:#dbe4eb;border:1px solid #dbe4eb;border-radius:9px;overflow:hidden;margin:23px 0 16px}.metric{background:#f8fbfd;padding:17px 19px}.metric span,.metric small{display:block;color:#64798b;font-size:11px}.metric strong{display:block;font-size:28px;line-height:1.3;margin:5px 0;color:#1f77b4;font-weight:600;font-variant-numeric:tabular-nums}.metric .interval{font-size:23px}.sample-note{font-size:12px;color:#64798b;max-width:1060px}h3{font-size:18px;margin:28px 0 14px}.etable-scroll,.table-scroll{overflow-x:auto;padding:3px}.etable-scroll>div{min-width:840px}.gt_table td,.gt_table th{line-height:1.6!important}.compact{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}.compact th{background:#eff5f9;text-align:right;font-size:11px;font-weight:600;padding:10px 12px;border-top:1px solid #dbe4eb;border-bottom:1px solid #c8d8e5;color:#536f83}.compact td{text-align:right;padding:11px 12px;border-bottom:1px solid #e7edf2}.compact th:first-child,.compact td:first-child{text-align:left;white-space:normal;min-width:205px}.compact tbody tr:hover{background:#f3f8fc}.sector-name{display:block;font-weight:600}.series-id{display:block;color:#8294a3;font:10px Menlo,Consolas,monospace}.notes,.units{font-size:12px;color:#64798b;max-width:1080px}.units{margin-top:17px}.downloads{display:flex;gap:20px;flex-wrap:wrap;margin:15px 0}.download{font-size:12px}.evidence{margin:24px 0 0;border-top:1px solid #dbe4eb;padding-top:17px}.evidence summary{cursor:pointer;font-weight:600;font-size:14px;color:#15334a;margin-bottom:17px}.individual-select{display:block;margin-top:7px;min-width:290px}.text-button{background:transparent;border:0;color:#1f77b4;padding:0}.individual-link p{font-size:13px}.methods{margin:28px 0 0;padding:23px 28px;background:white;border:1px solid #dbe4eb;border-radius:12px}.methods h2{font-size:19px;margin:0 0 13px}.methods p{font-size:13px;color:#536f83;margin:12px 0}.definition-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.definition{padding:15px 17px;background:#f5f8fb;border:1px solid #e2eaf0;border-radius:7px}.definition h3{font-size:13px;margin:0 0 7px}.definition p{font-size:12px;margin:0}.footer{font-size:11px;color:#64798b;border-top:1px solid #dbe4eb;padding-top:20px;margin-top:30px}.footer-links{display:flex;gap:20px;flex-wrap:wrap}.footer p{max-width:1100px}.taxonomy-list{columns:2;column-gap:35px;font-size:12px;color:#536f83;padding-left:20px}.taxonomy-list li{break-inside:avoid;padding:3px 0}a:focus-visible,select:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #1f77b4;outline-offset:4px}
.sig-tag{display:inline-block;padding:3px 8px;border-radius:4px;font-size:10px;font-weight:700;border:1px solid transparent}.sig-tag.sig-1{background:#dceee7;color:#145442;border-color:#b5d9ca}.sig-tag.sig-5{background:#e6f1eb;color:#215e49;border-color:#c6dfd2}.sig-tag.sig-10{background:#fff3d9;color:#795719;border-color:#e8d6ae}.sig-tag.not-sig,.sig-tag.unavailable{background:#f0f3f6;color:#536b7c;border-color:#dbe4eb}.sig-description{display:block;font-size:10px;line-height:1.5;color:#536b7c;margin-top:3px}.sector-table th:nth-child(6),.sector-table td:nth-child(6){text-align:left;min-width:155px}.compact .stationarity-cell{white-space:normal;text-align:left;min-width:210px;max-width:280px}.stationarity-flag{display:inline-block;font-size:11px;font-weight:600;line-height:1.45;padding:4px 7px;border:1px solid #dbe4eb;border-radius:4px}.screen-compatible{background:#e8f2ed;color:#245c48}.screen-caution{background:#fff4df;color:#795719}.screen-unavailable{background:#f0f3f6;color:#536b7c}.stationarity-detail{display:block;font-size:10px;line-height:1.45;color:#64798b;margin-top:4px}
.interaction-formula{padding:13px 16px;background:#eaf2f8;border-left:3px solid #1f77b4;font-size:17px;letter-spacing:.2px}.interaction-select{display:block;min-width:330px;margin:7px 0 17px}.interaction-results>.select-label{display:block;margin-top:25px}.interaction-results>h3{font-size:21px;margin-top:22px}.interaction-panel[hidden]{display:none}.interaction-readout th:nth-child(6),.interaction-readout td:nth-child(6){text-align:left;min-width:155px}.joint-readout{font-size:12px;background:#f5f8fb;border:1px solid #dbe4eb;border-radius:6px;padding:13px 15px;margin:14px 0}.joint-readout .sig-tag{margin-left:9px}.joint-readout>.sig-description{display:inline;margin-left:7px}.interaction-forest{overflow:auto;margin:20px 0}.interaction-forest svg{display:block;width:100%;min-width:1250px}.interaction-ts-comparison td:last-child{text-align:left;min-width:160px}.common-slope-benchmark>summary,.time-series-robustness>summary{font-size:15px}.interaction-native>div{min-width:900px}
@media(max-width:760px){main{padding:23px 14px 40px}h1{font-size:36px}.intro-grid,.definition-grid{grid-template-columns:1fr}.intro-grid{gap:15px;padding:17px 19px}.controls{position:static;gap:14px;padding:16px}.control{width:100%}.control select{width:100%}.result-view{padding:20px 12px}.view-heading{align-items:flex-start;flex-direction:column;gap:10px}.view-heading h2{font-size:25px}.metric-grid{grid-template-columns:1fr}.metric{padding:12px 16px}.metric strong{font-size:25px}.sample-strip{gap:9px 17px}.methods{padding:20px 17px}.topline .print{display:none}.taxonomy-list{columns:1}.individual-select{min-width:0;width:100%}}
@media print{@page{size:A4 landscape;margin:12mm}body{background:white}main{max-width:none;padding:0}.topline,.controls,.downloads,.download,.print,.footer-links{display:none}.result-view{border:0;padding:0}.etable-scroll,.table-scroll{overflow:visible}.etable-scroll>div{min-width:0}.intro-grid,.metric-grid,.methods{break-inside:avoid}.evidence{break-before:auto}.compact{font-size:10px}.compact th,.compact td{padding:7px}.metric strong{font-size:22px}h1{font-size:31px}.methods{border:0;padding:18px 0}h3{break-after:avoid}}
"""

SCRIPT = """
const contextSelect=document.getElementById('context-select');
const windowSelect=document.getElementById('window-select');
function showView(updateHash=true){
 const key=contextSelect.value+'--'+windowSelect.value;
 document.querySelectorAll('.result-view').forEach(view=>view.hidden=!(view.dataset.context===contextSelect.value&&view.dataset.window===windowSelect.value));
 if(updateHash){try{if(window.parent===window)location.hash='#'+key;}catch(error){}}
}
contextSelect.addEventListener('change',()=>showView());
windowSelect.addEventListener('change',()=>showView());
document.querySelectorAll('.individual-select').forEach(select=>select.addEventListener('change',()=>{
 select.closest('.individual-tables').querySelectorAll('.individual-panel').forEach(panel=>panel.hidden=panel.dataset.individualModel!==select.value);
}));
document.querySelectorAll('.interaction-select').forEach(select=>select.addEventListener('change',()=>{
 select.closest('.interaction-results').querySelectorAll('.interaction-panel').forEach(panel=>panel.hidden=panel.dataset.interactionModel!==select.value);
}));
document.querySelectorAll('[data-open-context]').forEach(button=>button.addEventListener('click',()=>{
 contextSelect.value=button.dataset.openContext;showView();
 document.querySelector('.result-view:not([hidden]) .interaction-results')?.scrollIntoView({behavior:'smooth',block:'start'});
}));
const initial=decodeURIComponent(location.hash.slice(1)).split('--');
if([...contextSelect.options].some(option=>option.value===initial[0]))contextSelect.value=initial[0];
if([...windowSelect.options].some(option=>option.value===initial[1]))windowSelect.value=initial[1];
showView(false);
"""


def methods(family: str) -> str:
    va = family == "va"
    unit = "real VA" if va else "IIP"
    raw = "chained-2015 S$ million" if va else "index points (2025=100)"
    definition = ('<div class="definition-grid"><div class="definition"><h3>G1 · One-period log growth · DJO form</h3>'
                  '<p>100 × [ln A(t) − ln A(t−1)]. MoM for monthly IIP; QoQ for quarterly data. '
                  'β = −1 means about one percentage point lower period growth per +1°C, conditional on the fitted model.</p></div>'
                  '<div class="definition"><h3>G2 · SA year-on-year log growth · DJO form</h3>'
                  '<p>100 × [ln A_SA(t) − ln A_SA(t−12)] monthly, or t−4 quarterly. The existing SA definition is retained. Adjacent observations overlap. '
                  'The model keeps current-period temperature in °C; changing Y to YoY growth does not require differencing temperature.</p></div>'
                  '<div class="definition"><h3>G2N · NSA year-on-year log growth · DJO form</h3>'
                  '<p>100 × [ln A_NSA(t) − ln A_NSA(t−12)] monthly, or t−4 quarterly, using official non-seasonally adjusted output levels. '
                  'This is added beside SA YoY on the same dates. Temperature and other weather controls still enter in current-period levels.</p></div>'
                  '<div class="definition"><h3>L1 · Log level · sensitivity</h3>'
                  f'<p>100 × ln {unit}. β is approximately a percent difference in the output level per +1°C. '
                  'This is a different estimand from growth and requires particular care about persistence and common trends.</p></div>'
                  '<div class="definition"><h3>L3 · Raw level · sensitivity</h3>'
                  f'<p>The dependent variable is output in {raw}. β has those units per +1°C. '
                  'Levels from large and small sectors have different scales, so pooled raw-level estimates need cautious interpretation.</p></div></div>')
    specific = ('The 19-sector VA panel replaces combined wholesale/retail with two sectors; accommodation/food services with two; '
                'real estate/professional/administrative services with three; and other services with five. '
                'All displayed Y variables share the YoY-ready sample beginning 2006Q1. The matched 11-sector panel uses exactly the same dates. '
                'Sectoral chain-linked real VA need not sum to published aggregate GVA; see the reconciliation report.') if va else (
                'The 18-industry panel and six-cluster comparator share dates from January 1997 (1997Q1 quarterly), once all displayed outcomes are available. '
                'The published total has a longer YoY-ready history beginning January 1984 (1984Q1 quarterly). '
                'Quarterly IIP is the mean of all three monthly index levels; growth is calculated after that aggregation. '
                'The national analysis uses the official weighted total directly. Equal-weight sector regressions and sums of index numbers do not reconstruct it. '
                'The reconciliation report distinguishes the official annual chain-link weights from a static base-year weighted approximation.')
    return ('<section class="methods"><h2>How to read these specifications</h2>'+definition
            +'<p><strong>The Dell, Jones and Olken (DJO) label refers to the log-growth form.</strong> '
            'Their annual cross-country design is not replicated by these monthly or quarterly Singapore regressions. '
            'Temperature, rainfall, humidity and, where included, wind and PM2.5 enter in current-period levels. '
            'A coefficient rescaled to +0.1°C describes the fitted conditional association, not a forecast or a permanent growth loss.</p>'
            '<p><strong>Detailed-sector heterogeneity uses a panel FE interaction model.</strong> Temperature has a separate coefficient βᵢ for each sector; '
            'rainfall, humidity and any wind/PM2.5 slopes are shared, with sector-specific seasonality and common year effects. '
            'These main slopes retain jointly estimated calendar-HAC covariance for valid cross-sector comparisons. '
            'Separate time-series fits, with each sector’s own controls and year effects, are robustness results. '
            'Sector clustering is inappropriate for an individual βᵢ supported by one unit.</p>'
            '<p><strong>Common-slope benchmarks cluster at sector i.</strong> They use CR1 with S/(S−1) × (N−1)/(N−K), '
            'counting explicit fixed effects in K, and t(S−1) tests. Sector clusters allow dependence across dates within each sector. '
            'They do not cover cross-sector dependence from shared national shocks. Six, 11, 18 or 19 clusters are small numbers; '
            'nominal stars and intervals remain approximate. Calendar DK estimates at the original and doubled bandwidths are retained for comparison. '
            'Published national totals and individual-sector regressions have one unit, so they retain time-series HAC.</p>'
            '<p><strong>Coverage and comparability.</strong> '+specific+' The short wind/PM2.5 sample requires at least 90% source coverage; '
            'no missing weather observations are imputed. This all-retained-date 2017–2025 full-control window is the baseline. '
            'Excluding observations dated 2020–2022 and the longer history without wind/PM2.5 are robustness checks. '
            'The exclusion is applied after constructing growth. '
            'Year and seasonal fixed effects remain in every model. Very short quarterly samples leave limited residual degrees of freedom.</p>'
            '<p><strong>Why the baseline stationarity screen is unavailable.</strong> Quality screening of wind and PM2.5 leaves calendar gaps. '
            'Ordinary ADF/KPSS tests require evenly spaced observations, so the retained dates are not compressed and tested as if consecutive. '
            'Unavailable means the exact-window test was not run; it does not mean every series failed a test. '
            'Contiguous source-history tests describe a different window and cannot certify the gapped baseline.</p>'
            '<p><strong>Levels remain exploratory.</strong> Persistent output, temperature trends, structural breaks and overlapping YoY observations '
            'can complicate inference. <a href="sector_heterogeneity.html">Earlier heterogeneity and stationarity diagnostics</a> provide context; '
            'they do not automatically establish stationarity for every new detailed series or validate a causal levels model.</p></section>')


def validate_document(document: str, contexts: dict[str, str]) -> None:
    tree = lxml_html.fromstring(document)
    ids = tree.xpath("//@id")
    assert len(ids) == len(set(ids)), "Duplicate HTML identifiers"
    views = tree.xpath("//section[contains(concat(' ',normalize-space(@class),' '),' result-view ')]")
    assert len(views) == len(contexts)*len(WINDOWS)
    assert len([view for view in views if "hidden" not in view.attrib]) == 1
    assert [view for view in views if "hidden" not in view.attrib][0].get("data-window") == "full"
    assert not tree.xpath("//script[@src] | //link[@rel='stylesheet'] | //img"), "Reports must be self-contained"
    expected_native = len(views)+len(WINDOWS)*sum(context.endswith("detailed") for context in contexts)
    assert len(tree.xpath("//table[contains(@class,'gt_table')]")) == expected_native
    for node in tree.xpath("//*[@headers]"):
        assert set(node.get("headers").split()) <= set(ids), "Unresolved table header reference"
    assert len(tree.xpath("//a[@download]")) >= len(views)


def build_report(family: str) -> Path:
    va = family == "va"
    contexts = VA_CONTEXTS if va else IIP_CONTEXTS
    default = next(iter(contexts))
    title = "Heat and the finer sectoral structure of real VA." if va else "Heat, manufacturing industries and the weighted IIP total."
    lede = ("Nineteen detailed sectors, with sector-specific temperature effects estimated in one panel FE interaction model. "
            "The earlier 11-sector taxonomy remains on matched dates; separate time-series fits are robustness checks.") if va else (
            "Monthly and quarterly evidence for the official weighted manufacturing total, 18 detailed industries and six broad clusters. "
            "Detailed-sector heterogeneity leads with panel FE temperature interactions; separate time-series fits remain robustness checks.")
    intro = ('<h2>The granular VA sample begins in 2006Q1</h2><p>The source adds finer sector cuts from 2005. '
             'Requiring a year-earlier observation makes 2006Q1 the common starting point for the displayed outcomes, including both SA and NSA YoY. '
             'Matched broad-sector estimates separate the shorter history from the choice of taxonomy.</p>') if va else (
             '<h2>Use the published weighted total for national manufacturing</h2><p>The national index carries the official aggregation weights. '
             'The industry panels measure average sector responses. Compare the six- and 18-industry views on matched dates; '
             'the official total retains its longer history.</p>')
    options = "".join(f'<option value="{key}">{esc(name)}</option>' for key, name in contexts.items())
    windows = "".join(f'<option value="{key}">{esc(name)}</option>' for key, name in WINDOWS.items())
    views = "".join(result_view(context, name, window, context == default and window == "full")
                    for context, name in contexts.items() for window in WINDOWS)
    other_file = "manufacturing_detail.html" if va else "granular_va.html"
    other_name = "Manufacturing detail" if va else "Detailed real VA"
    source_id = "M015662" if va else "M355382"
    source_name = "Quarterly real VA by industry" if va else "Monthly IIP by industry cluster"
    content = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<title>{esc(title)} · Heat & the Singapore economy</title><style>{STYLE}</style></head><body><main>'
               '<div class="topline"><a href="heat_economy.html">← Complete tabbed analysis</a>'
               '<button class="print" type="button" onclick="window.print()">Print selected view / save PDF ↗</button></div>'
               f'<header><p class="eyebrow">Heat & the Singapore economy · {"Detailed VA" if va else "Manufacturing"}</p><h1>{title}</h1><p class="lede">{lede}</p></header>'
               '<div class="intro-grid"><div>'+intro+'</div><div class="small"><h2>Growth is the main outcome</h2>'
               '<p>The baseline includes rainfall, humidity, wind and PM2.5 on all retained 2017–2025 dates. '
               'One-period growth and the SA/NSA YoY columns use 100 × log differences, following the DJO outcome form with frequency adapted. '
               'Log and raw levels remain visible as sensitivities. Current-period temperature stays in °C in every column.</p></div></div>'
               '<div class="controls" aria-label="Select analysis"><div class="control"><label for="context-select">Series and frequency</label>'
               f'<select id="context-select">{options}</select></div><div class="control"><label for="window-select">Sample and weather controls</label>'
               f'<select id="window-select">{windows}</select></div></div>'+views+methods(family)
               +'<footer class="footer"><div class="footer-links"><a href="heat_economy.html">Complete analysis</a>'
               f'<a href="{other_file}">{other_name}</a><a href="reconciliation.html">VA totals, IIP weights and reconciliation</a>'
               '<a href="sector_heterogeneity.html">Sector heterogeneity and diagnostics</a></div>'
               f'<p>Sources: <a href="https://tablebuilder.singstat.gov.sg/table/TS/{source_id}">SingStat · {source_name}</a>; '
               'official Singapore weather, wind and PM2.5 series from data.gov.sg. '
               'The linked reconciliation report and downloadable coefficients retain the measurement and aggregation details. '
               'Estimation outputs, exact samples and formulas are saved under outputs/granular; no values are fabricated or imputed. '
               'Native tables use MakeTables ETable, the table engine used by PyFixest etable.</p></footer></main>'
               f'<script>{SCRIPT}</script></body></html>')
    validate_document(content, contexts)
    destination = OUT / ("granular_va.html" if va else "manufacturing_detail.html")
    destination.write_text(content)
    print(f"Wrote {destination.relative_to(ROOT)} ({len(content):,} characters)", flush=True)
    return destination


def build_granular_reports() -> list[Path]:
    (OUT / "tables").mkdir(exist_ok=True)
    return [build_report("va"), build_report("iip")]


if __name__ == "__main__":
    build_granular_reports()
