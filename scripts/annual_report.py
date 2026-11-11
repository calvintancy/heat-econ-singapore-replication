"""Offline annual-growth report rendered from saved native PyFixest results.

Only the selected two-column MakeTables fragment is inserted in the document.
Unavailable specifications remain explicit columns rather than disappearing.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path

import maketables as mt
import numpy as np
import pandas as pd
from lxml import html as lh

from regression_tables import namespace_table_ids
from significance import p_format, significance_class, significance_label, significance_stars

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/annual"
REPORT = ROOT/"outputs/annual.html"
VARIANTS = ["all_eligible", "exclude_2020_2022"]
CONTEXTS = {"aggregate_va": "Official aggregate GVA", "va11": "11 broad VA sectors",
            "va19": "19 detailed VA sectors", "iip_total": "Published manufacturing IIP total",
            "iip6": "Six manufacturing clusters", "iip18": "18 manufacturing industries"}
MODELS = {"H1": "Main heterogeneity · sector × temperature FE", "common": "Common-slope benchmark",
          "separate": "Robustness · own-sector time series"}
TERMS = {"temp_c": "Annual mean air temperature (°C)", "rain_100mm": "Annual rainfall (100 mm)",
         "rh_pct": "Annual mean relative humidity (pp)", "wind_mps": "Annual mean wind speed (m/s)",
         "pm25_10ug": "Annual mean PM2.5 (10 µg/m³)", "trend": "Linear calendar-year trend"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def number(value, signed=False) -> str:
    return "Unavailable" if pd.isna(value) else f"{value:+.3f}" if signed else f"{value:,.3f}"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[rows.status.ne("estimated") | rows.main_inference.eq(True)].copy()


def badge(p_value) -> str:
    bucket = significance_class(p_value)
    five = "Unavailable" if bucket == "unavailable" else "Yes" if float(p_value)<.05 else "No"
    return (f'<span class="sig {bucket}" data-p="{p_value}">5%: {five}</span>'
            f'<span class="small sig-label">{esc(significance_label(p_value))}</span>')


def screen_label(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "Unavailable · no matching annual screen"
    unavailable = rows.adf_status.ne("ok") | rows.kpss_status.ne("ok")
    if unavailable.all():
        reasons = " ".join(rows.reason.fillna("").astype(str))
        pieces = []
        if "Gapped" in reasons:
            pieces.append("calendar gaps")
        if "Fewer than" in reasons:
            pieces.append("fewer than 20 years")
        return "Unavailable · "+("; ".join(pieces) or "tests not completed")
    compatible = rows.evidence_5pct.str.startswith("Compatible", na=False)
    if len(rows)==1:
        return str(rows.iloc[0].evidence_5pct)
    return f"{int(compatible.sum())}/{len(rows)} screens compatible; {int(unavailable.sum())} unavailable"


def screens(diagnostics: pd.DataFrame, info: dict, sector_id: str | None = None) -> dict:
    selected = diagnostics.loc[diagnostics.context.eq(info["context"])
        & diagnostics.control_set.eq(info["control_set"])
        & diagnostics.sample_variant.eq(info["sample_variant"])]
    y = selected.loc[selected.variable.eq(info["outcome"]) & selected.adjustment.eq(info["adjustment"])]
    if sector_id is not None:
        y = y.loc[y.sector_id.eq(str(sector_id))]
    x = selected.loc[selected.adjustment.eq("common") & selected.sector_id.eq("common")]
    return {"y": screen_label(y), "x": screen_label(x)}


@dataclass
class AnnualResult:
    rows: pd.DataFrame
    info: dict
    dependent: str
    profile: bool

    @property
    def __maketables_coef_table__(self):
        key = "sector_id" if self.profile else "term"
        frame = self.rows.set_index(key)[["coefficient", "std_error", "p_value", "ci_lower", "ci_upper"]]
        assert frame.index.is_unique
        return frame.rename(columns={"coefficient": "b", "std_error": "se", "p_value": "p", "ci_lower": "ci95l", "ci_upper": "ci95u"})

    @property
    def __maketables_depvar__(self):
        return self.dependent

    @property
    def __maketables_fixef_string__(self):
        return "sector" if self.info["model"]!="separate" and int(self.info["n_sectors"])>1 else ""

    def __maketables_stat__(self, key):
        return {"N": int(self.info["n_obs"]), "years": int(self.info["n_periods"])}.get(key)


def status_text(info: dict) -> str:
    return "Estimated · fragile annual sample" if info["status"]=="estimated" and bool(info["fragile_short_sample"]) else "Estimated" if info["status"]=="estimated" else "Unavailable · underidentified"


def native_table(models: list[AnnualResult], view_id: str, names: dict[str,str], assessments: list[dict]) -> str:
    profile = models[0].profile
    own = models[0].info["model"]=="separate"
    keys = list(models[0].rows.sector_id) if profile else list(models[0].rows.term)
    labels = {key: names.get(key,key) for key in keys} if profile else {key: TERMS.get(key,key) for key in keys}
    info = [model.info for model in models]
    covariance = []
    for model in models:
        if model.info["status"]!="estimated":
            covariance.append("Unavailable")
        else:
            row = model.rows.iloc[0]
            covariance.append("Sector CR1" if row.covariance=="Sector cluster" else "Calendar HAC · 1 annual lag")
    custom = {"Status": [status_text(row) for row in info],
              "Estimator": ["PyFixest "+str(row["estimator_version"]) if row["status"]=="estimated" else "Not estimated" for row in info],
              "Inference": covariance,
              "Unit equations / sectors": [int(row["profile_units"]) for row in info],
              "Inference df": [str(int(model.rows.iloc[0].inference_df)) if model.info["status"]=="estimated" else "Unavailable" for model in models],
              "Annual time residual df": [str(int(row["time_residual_df"]))+" (nominal "+str(int(row["nominal_time_residual_df"]))+")" for row in info],
              "Requested parameters": [int(row["n_parameters"]) for row in info],
              "Full design rank": [str(int(row["design_rank"]))+"/"+str(int(row["n_parameters"])) for row in info],
              "Annual time design rank": [str(int(row["time_design_rank"]))+"/"+str(int(row["n_time_parameters"])) for row in info],
              "First year": [int(row["start"]) for row in info], "Last year": [int(row["end"]) for row in info],
              "Calendar gaps": [int(row["calendar_gaps"]) for row in info],
              "Mean Y (log points)": ["Sector means below" if profile else number(row["mean_y"]) for row in info],
              "SD Y (log points)": ["Sector SDs in download" if profile else number(row["sd_y"]) for row in info],
              "Y screen · exact window": [assessment["y"] for assessment in assessments],
              "Weather screens · exact window": [assessment["x"] for assessment in assessments],
              "Year fixed effects": ["No · would absorb temperature"]*2,
              "Linear time trend": ["Own-sector" if own else "Common"]*2}
    note = ("Saved native PyFixest coefficients; standard errors in parentheses. Annual Y is 100 times the log change in complete-year output. "
            "NSA is primary; the SA annual definition is retained as a comparison. Current-year temperature and weather enter in levels. "
            "Sector FE plus a linear calendar-year trend replace the subannual seasonal/year effects. Annual year effects would absorb national temperature. "
            "Common pooled slopes use sector CR1 with t(S−1); national and sector-specific coefficients use calendar HAC with approximate t(T−1). "
            "The HAC2 sensitivity and full covariance are retained in downloads. Stars: * p<0.10, ** p<0.05, *** p<0.01. "
            "Sector significance is pointwise; joint equality and Holm-adjusted pairwise tests are separate. "
            "Unavailable columns retain requested controls; blanks are not zero estimates. Short/gapped stationarity screens are unavailable, not failed tests.")
    if own:
        note += " Each row is a separate own-sector equation; N and parameter/rank counts are per equation, not a pooled regression."
    table = mt.ETable(models, model_heads=["All eligible annual years", "Exclude observations dated 2020–2022"],
        head_order="dh", coef_fmt="b:.3f* \n (se:.3f)", signif_code=[.01,.05,.1], keep=keys,
        exact_match=True, labels=labels, felabels={"sector":"Sector fixed effects"}, feorder=["sector"],
        model_stats=["N","years"], model_stats_labels={"N":"Observations per equation" if own else "Sector-year observations", "years":"Annual weather observations"},
        custom_model_stats=custom, stats_order="sc", notes=note)
    gt = table.make(type="gt",gt_style={"table_width":"100%","first_col_width":"255px","table_font_size_all":"12px",
                    "source_notes_font_size":"11px","data_row_padding":"7px"})
    return namespace_table_ids(gt.as_raw_html(),"annual-"+view_id)


def impact_cell(row: dict, info: dict) -> str:
    if info["status"]!="estimated":
        return '<strong>Unavailable · not estimated</strong><span class="small">'+esc(info["reason"])+'</span>'
    value = (f'<strong>{number(row["coefficient"],True)}{significance_stars(row["p_value"])}</strong>'
             f'<span class="small">SE {number(row["std_error"])} · 95% CI [{number(row["ci_lower"],True)}, {number(row["ci_upper"],True)}]</span>'
             f'<span class="small">p={esc(p_format(row["p_value"]))}</span>'+badge(row["p_value"])
             +f'<span class="small">+0.1°C: {number(.1*row["coefficient"],True)} log points</span>')
    if info["model"]=="H1":
        value += (f'<span class="small">βT={number(row["temperature_main"],True)}; δᵢ={number(row["temperature_interaction"],True)}'
                  +(" (reference normalization)" if row["sector_id"]==row["reference_sector_id"] else "")+'</span>')
    return value


def readout(models: list[AnnualResult], diagnostics: pd.DataFrame, names: dict) -> str:
    profile = models[0].profile
    paired = []
    for model in models:
        frame = model.rows if profile else model.rows.loc[model.rows.term.eq("temp_c")]
        paired.append(frame.set_index("sector_id" if profile else "term").to_dict("index"))
    keys = list(paired[0])
    assert set(keys)==set(paired[1])
    body = []
    for key in keys:
        cells = [esc(names.get(key,key) if profile else "Temperature effect per +1°C")]
        for index, model in enumerate(models):
            row = paired[index][key]
            if profile:
                row["sector_id"] = key
            info = model.info
            assessment = screens(diagnostics,info,key if profile else None)
            cells += [number(row["mean_y"]),impact_cell(row,info)
                      +'<span class="screen">Y: '+esc(assessment["y"])+'<br>Weather: '+esc(assessment["x"])+"</span>"]
        body.append("<tr>"+"".join("<td>"+cell+"</td>" for cell in cells)+"</tr>")
    return ('<div class="scroll"><table class="compact"><thead><tr><th>Sector / effect</th><th>Mean Y · all years</th>'
            '<th>All eligible years</th><th>Mean Y · exclusion</th><th>Exclude 2020–2022</th></tr></thead><tbody>'
            +"".join(body)+"</tbody></table></div>")


def joint_readout(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    body = []
    for variant in VARIANTS:
        selected = rows.loc[rows.sample_variant.eq(variant)]
        selected = selected.loc[selected.status.ne("estimated") | selected.main_inference.eq(True)]
        assert len(selected)==1
        row = selected.iloc[0]
        available = row.test_status=="available"
        statistic = f"F({int(row.numerator_df)}, {int(row.denominator_df)})={number(row.f_statistic)}" if available else "Unavailable"
        rank = (str(int(row.covariance_rank))+"/"+str(int(row.restrictions))) if pd.notna(row.covariance_rank) else "Unavailable"
        body.append('<tr><td>'+esc("All eligible years" if variant==VARIANTS[0] else "Exclude 2020–2022")+'</td>'
                    '<td>'+esc(statistic)+'</td><td>'+esc(p_format(row.p_value))+'</td><td>'+badge(row.p_value)+'</td>'
                    '<td>'+esc(rank)+'</td><td>'+esc(row.test_status)+'</td></tr>')
    return ('<details><summary>Joint test: are all annual sector slopes equal?</summary><div class="scroll"><table class="compact"><thead><tr>'
            '<th>Sample</th><th>Joint F</th><th>Joint p</th><th>Joint-test significance</th><th>Covariance rank / restrictions</th><th>Status</th>'
            '</tr></thead><tbody>'+"".join(body)+'</tbody></table></div><p class="notes">The joint test uses cross-sector covariance. '
            'It tests equality of all slopes, rather than whether any single sector’s total βT+δᵢ equals zero. '
            'Singular restriction covariances have no omnibus p-value; nominal full-rank tests can also be fragile with few annual years.</p></details>')


def pairwise_readout(rows: pd.DataFrame, names: dict) -> str:
    if rows.empty:
        return '<p class="notes">Pairwise tests are unavailable when the requested interaction model is underidentified.</p>'
    selected = rows.loc[rows.main_inference.eq(True)]
    body = []
    for row in selected.itertuples():
        body.append('<tr><td>'+esc("All years" if row.sample_variant==VARIANTS[0] else "Exclude COVID dates")+'</td><td>'
                    +esc(names.get(row.first_sector_id,row.first_sector_id)+" − "+names.get(row.second_sector_id,row.second_sector_id))+'</td>'
                    f'<td>{number(row.coefficient,True)}</td><td>{esc(p_format(row.p_value))}</td><td>{esc(p_format(row.p_value_holm))}</td><td>{badge(row.p_value_holm)}</td></tr>')
    return ('<details><summary>All pairwise sector differences · Holm-adjusted inference</summary><div class="scroll"><table class="compact"><thead><tr>'
            '<th>Sample</th><th>First sector − second sector</th><th>Difference per +1°C</th><th>Pointwise p</th><th>Holm-adjusted p</th><th>Adjusted significance</th>'
            '</tr></thead><tbody>'+"".join(body)+'</tbody></table></div><p class="notes">Holm adjusts all S(S−1)/2 sector pairs separately '
            'within context, controls, adjustment, sample variant and HAC bandwidth. These are slope differences, not individual-sector heat effects. '
            +('The excluded-year model is underidentified, so its pairwise tests are unavailable.' if "exclude_2020_2022" not in set(selected.sample_variant) else '')+'</p></details>')


def compressed_download(path: Path, label: str) -> str:
    data = base64.b64encode(gzip.compress(path.read_bytes(),mtime=0)).decode()
    return '<a href="data:application/gzip;base64,'+data+'" download="annual_'+esc(path.name)+'.gz">'+esc(label)+' · CSV.gz ↓</a>'


def build_views() -> tuple[list[dict],dict]:
    files = {name:OUT/(name+".csv") for name in ["specs","estimates","sector_estimates","separate_estimates","joint_tests","pairwise_tests"]}
    files["diagnostics"] = OUT/"diagnostics/summary.csv"
    frames = {name:pd.read_csv(path,dtype={"sector_id":str,"first_sector_id":str,"second_sector_id":str,"reference_sector_id":str},low_memory=False) for name,path in files.items()}
    specs,diagnostics = frames["specs"],frames["diagnostics"]
    names_by_context = {context:group.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict()
                        for context,group in frames["sector_estimates"].groupby("context")}
    views = []
    keys = specs[["context","control_set","adjustment","model"]].drop_duplicates()
    for context,control_set,adjustment,kind in keys.itertuples(index=False,name=None):
        names = names_by_context.get(context,{})
        chosen = specs.loc[specs.context.eq(context)&specs.control_set.eq(control_set)&specs.adjustment.eq(adjustment)&specs.model.eq(kind)]
        source_name = "estimates" if kind=="common" else "sector_estimates" if kind=="H1" else "separate_estimates"
        rows = primary(frames[source_name].loc[frames[source_name].spec_id.isin(chosen.spec_id)])
        if kind=="separate":
            rows = rows.loc[rows.term.eq("temp_c")].copy()
        dependent = "Annual "+adjustment+" log growth (100 × Δln annual output)"
        models, assessments, statuses = [],[],[]
        for variant in VARIANTS:
            metadata = chosen.loc[chosen.sample_variant.eq(variant)]
            assert len(metadata)==(metadata.sector_id.nunique() if kind=="separate" else 1)
            info = metadata.iloc[0].to_dict()
            assert metadata.status.nunique()==metadata.n_periods.nunique()==1
            part = rows.loc[rows.sample_variant.eq(variant)].copy()
            if kind!="common":part = part.sort_values("sector_id")
            info["profile_units"] = part.sector_id.nunique() if kind!="common" else int(info["n_sectors"])
            models.append(AnnualResult(part,info,dependent,kind!="common"))
            assessments.append(screens(diagnostics,info))
            statuses.append({"sample_variant":variant,"status":info["status"],"reason":"" if pd.isna(info["reason"]) else str(info["reason"]),
                             "n_periods":int(info["n_periods"]),"n_obs":int(info["n_obs"]),"time_residual_df":int(info["time_residual_df"]),
                             "nominal_time_residual_df":int(info["nominal_time_residual_df"]),"calendar_gaps":int(info["calendar_gaps"]),
                             "fragile":bool(info["fragile_short_sample"]),"y_screen":assessments[-1]["y"],"x_screen":assessments[-1]["x"]})
        view_id = "annual_"+hashlib.sha256("|".join([context,control_set,adjustment,kind]).encode()).hexdigest()[:14]
        spec_ids = list(chosen.spec_id)
        joint = frames["joint_tests"].loc[frames["joint_tests"].spec_id.isin(spec_ids)]
        pairs = frames["pairwise_tests"].loc[frames["pairwise_tests"].spec_id.isin(spec_ids)]
        item = {"id":view_id,"context":context,"context_label":CONTEXTS[context],"control_set":control_set,
                "adjustment":adjustment,"model":kind,"model_label":MODELS[kind],"outcome":models[0].info["outcome"],
                "dependent":dependent,"source_csv":str(files[source_name].relative_to(ROOT)),"source_spec_ids":spec_ids,
                "source_rows":rows.astype(object).where(pd.notna(rows),None).to_dict(orient="records"),"statuses":statuses,
                "native_html":native_table(models,view_id,names,assessments),"readout_html":readout(models,diagnostics,names),
                "joint_html":joint_readout(joint) if kind=="H1" else "",
                "pairwise_html":pairwise_readout(pairs,names) if kind=="H1" else ""}
        views.append(item)
    assert len(views)==56
    return views,{str(path.relative_to(ROOT)):sha(path) for path in files.values()}


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f3f7fa;color:#15334a;font:14px/1.65 'Avenir Next','Segoe UI',Arial,sans-serif}main{max-width:1320px;margin:auto;padding:30px 30px 55px}a{color:#1f77b4;text-underline-offset:4px}.top{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid #dbe4eb;padding-bottom:16px;font-size:12px}.eyebrow{font:11px Menlo,Consolas,monospace;color:#64798b;letter-spacing:1.2px;text-transform:uppercase;margin-top:28px}h1{font:600 46px/1.1 'Avenir Next Condensed','Arial Narrow',Arial,sans-serif;letter-spacing:-.8px;margin:13px 0 18px}header>p{max-width:1000px;color:#64798b;font-size:16px}.limit{padding:16px 20px;border-left:4px solid #b77a2a;background:#fff4e2;font-size:13px;margin:22px 0}.limit p{margin:5px 0}.controls{display:grid;grid-template-columns:1.1fr 1fr 1fr 1.2fr;gap:13px;background:white;border:1px solid #dbe4eb;border-radius:10px;padding:18px;margin:23px 0 17px}.control label{display:block;font-size:10px;letter-spacing:.4px;text-transform:uppercase;color:#64798b;font-weight:700;margin-bottom:6px}select{width:100%;max-width:100%;font:600 12px 'Avenir Next',Arial,sans-serif;border:1px solid #c9d9e5;color:#15334a;background:white;border-radius:5px;padding:10px}.paper{padding:25px 26px;background:white;border:1px solid #dbe4eb;border-radius:11px}.heading{display:flex;justify-content:space-between;gap:20px;align-items:center}.heading h2{font:600 27px 'Avenir Next Condensed',Arial,sans-serif;margin:0}.role{font-size:11px;font-weight:600;padding:5px 9px;background:#eaf2f8;color:#1a5f8b;border-radius:20px;white-space:nowrap}.notes,.small{font-size:11px;color:#64798b}.small{display:block;line-height:1.55;margin-top:3px}.formula{background:#f0f6fa;border-left:3px solid #1f77b4;padding:12px 15px;font-size:15px;margin:17px 0}.status-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}.status-card{border:1px solid #dbe4eb;border-radius:7px;padding:14px 16px;background:#f8fbfd;font-size:12px}.status-card h3{font-size:13px;margin:0 0 8px}.status-card strong{color:#15334a}.status-card.unavailable{background:#fff4e9;border-color:#ebd5b8}.native-scroll,.scroll{overflow:auto;padding:2px}.native-scroll>div{min-width:760px}.gt_table td,.gt_table th{line-height:1.6!important}h3{font-size:18px;margin:27px 0 13px}.compact{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}.compact th,.compact td{padding:11px 12px;border-bottom:1px solid #e0e8ee;text-align:right;vertical-align:top;min-width:110px}.compact th{font-weight:600;color:#64798b;font-size:11px;background:#f4f8fb}.compact th:first-child,.compact td:first-child{text-align:left;min-width:200px}.compact td:nth-child(3),.compact td:nth-child(5){text-align:left;min-width:260px}.compact strong{font-weight:600;color:#15334a}.sig{display:inline-block;margin-top:5px;padding:3px 7px;border:1px solid #dbe4eb;border-radius:4px;font-size:10px;font-weight:700;white-space:nowrap}.sig-1{background:#dceee7;color:#145442}.sig-5{background:#e6f1eb;color:#215e49}.sig-10{background:#fff0d5;color:#795719}.not-sig,.sig.unavailable{background:#f0f3f6;color:#536b7c}.screen{display:block;font-size:10px;line-height:1.5;color:#64798b;border-top:1px dotted #dbe4eb;margin-top:7px;padding-top:6px}details{border-top:1px solid #dbe4eb;padding-top:16px;margin-top:24px}summary{font-size:13px;font-weight:600;cursor:pointer}.methods{background:white;border:1px solid #dbe4eb;border-radius:10px;padding:23px 26px;margin-top:25px}.methods h2{font-size:21px;margin:0 0 15px}.methods p{font-size:13px;color:#536b7c;max-width:1100px}.downloads{display:flex;gap:10px 20px;flex-wrap:wrap;margin:17px 0;font-size:11px}.footer{border-top:1px solid #dbe4eb;padding-top:19px;margin-top:27px;font-size:11px;color:#64798b}.links{display:flex;gap:17px;flex-wrap:wrap}.print{font:11px 'Avenir Next',Arial,sans-serif;border:1px solid #c9d9e5;background:white;color:#15334a;border-radius:5px;padding:7px 11px;cursor:pointer}a:focus-visible,select:focus-visible,summary:focus-visible,button:focus-visible{outline:3px solid #1f77b4;outline-offset:4px}
@media(max-width:760px){main{padding:21px 13px 35px}h1{font-size:35px}.controls{grid-template-columns:1fr 1fr;padding:14px;gap:13px}.paper{padding:19px 12px}.heading{align-items:flex-start;flex-direction:column;gap:8px}.heading h2{font-size:24px}.status-grid{grid-template-columns:1fr}.methods{padding:20px 17px}.print{display:none}}
@media print{@page{size:A4 landscape;margin:12mm}body{background:white}main{padding:0;max-width:none}.controls,.top,.print,.downloads{display:none}.paper,.methods{border:0;padding:0}.native-scroll,.scroll{overflow:visible}.native-scroll>div{min-width:0}.status-grid{break-inside:avoid}h1{font-size:31px}.compact{font-size:10px}.compact th,.compact td{padding:7px}.compact table{break-inside:avoid}}
"""

SCRIPT = """
const book=JSON.parse(document.getElementById('annual-data').textContent);
const controls={context:document.getElementById('context'),control_set:document.getElementById('controls'),adjustment:document.getElementById('adjustment'),model:document.getElementById('model')};
function show(){
 const available=[...new Set(book.views.filter(v=>v.context===controls.context.value).map(v=>v.model))];
 [...controls.model.options].forEach(o=>o.disabled=!available.includes(o.value));
 if(!available.includes(controls.model.value))controls.model.value='common';
 const view=book.views.find(v=>Object.entries(controls).every(([k,input])=>v[k]===input.value));
 if(!view)return;
 document.getElementById('selected-title').textContent=view.context_label+' · '+view.model_label;
 document.getElementById('selected-role').textContent=view.adjustment==='NSA'?'Primary · annual NSA growth':'Comparison · annual SA growth';
 document.getElementById('selected-outcome').textContent='Y = '+view.dependent+'. Controls: '+(view.control_set==='full'?'rainfall, humidity, wind and PM2.5':'rainfall and humidity; long-history robustness')+'.';
 const formula=view.model==='H1'?'Yᵢᵧ = βT Tᵧ + Σ δₛ 1(i=s)Tᵧ + γ′Wᵧ + αᵢ + λ·year + εᵢᵧ':view.model==='separate'?'Yᵢᵧ = βᵢ Tᵧ + γᵢ′Wᵧ + αᵢ + λᵢ·year + εᵢᵧ':'Yᵢᵧ = β Tᵧ + γ′Wᵧ + αᵢ + λ·year + εᵢᵧ';
 document.getElementById('formula').textContent=formula;
 document.getElementById('status-grid').innerHTML=view.statuses.map(s=>'<div class="status-card '+(s.status==='estimated'?'':'unavailable')+'"><h3>'+(s.sample_variant==='all_eligible'?'All eligible annual years':'Exclude observations dated 2020–2022')+'</h3><strong>'+(s.status==='estimated'?'Estimated'+(s.fragile?' · fragile short sample':''):'Unavailable · underidentified')+'</strong><span class="small">'+s.n_periods+' annual years · '+s.n_obs+' observations per model/equation · time residual df '+s.time_residual_df+' (nominal '+s.nominal_time_residual_df+')</span><span class="small">'+(s.reason||'')+'</span></div>').join('');
 document.getElementById('native-table').innerHTML=view.native_html;
 document.getElementById('readout').innerHTML=view.readout_html;
 document.getElementById('joint').innerHTML=view.joint_html+view.pairwise_html;
 document.getElementById('annual-selection').dataset.viewId=view.id;
 try{if(window.parent===window)location.hash='#'+[view.context,view.control_set,view.adjustment,view.model].join('--');}catch(error){}
}
Object.values(controls).forEach(input=>input.addEventListener('change',show));
const requested=decodeURIComponent(location.hash.slice(1)).split('--');
['context','control_set','adjustment','model'].forEach((key,i)=>{if([...controls[key].options].some(o=>o.value===requested[i]))controls[key].value=requested[i];});
show();
"""


def build_annual_report() -> Path:
    views,hashes = build_views()
    default = next(view for view in views if (view["context"],view["control_set"],view["adjustment"],view["model"])==("va19","full","NSA","H1"))
    source_files = [OUT/(name+".csv") for name in ["estimates","sector_estimates","separate_estimates","specs","joint_tests","pairwise_tests","interaction_components","coefficient_covariances","year_fe_collinearity"]]
    source_files += [OUT/"diagnostics/summary.csv",OUT/"diagnostics/tests.csv"]
    downloads = "".join(compressed_download(path,path.stem.replace("_"," ").title()) for path in source_files)
    reconciliation = ""
    reconciliation_path = OUT/"reconciliation.csv"
    if reconciliation_path.exists():
        hashes[str(reconciliation_path.relative_to(ROOT))] = sha(reconciliation_path)
        reconciliation = ('<p>Annual NSA sums/means match the published annual tables within rounding on the checked overlap. '
                          'The analysis uses the latest cached quarterly/monthly sources throughout; SA sums/means remain a comparison. '
                          '<a href="https://tablebuilder.singstat.gov.sg/table/TS/M015721">Published annual VA</a> · '
                          '<a href="https://tablebuilder.singstat.gov.sg/table/TS/M355401">Published annual IIP</a>.</p><div class="downloads">'
                          +compressed_download(reconciliation_path,"Annual NSA aggregation reconciliation")+'</div>')
    inventory = {"report":"outputs/annual.html","rendering":"One selected native MakeTables ETable fragment in the DOM",
                 "paired_views":len(views),"default_view":default["id"],"contexts":CONTEXTS,"models":MODELS,
                 "sample_variants":VARIANTS,"source_hashes":hashes,"views":views}
    OUT.mkdir(exist_ok=True)
    (OUT/"report_inventory.json").write_text(json.dumps(inventory,ensure_ascii=False,allow_nan=False)+"\n")
    payload = json.dumps({"views":views},ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    contexts = "".join(f'<option value="{key}" {"selected" if key=="va19" else ""}>{esc(label)}</option>' for key,label in CONTEXTS.items())
    models = "".join(f'<option value="{key}" {"selected" if key=="H1" else ""}>{esc(label)}</option>' for key,label in MODELS.items())
    content = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
      '<title>Heat and annual output growth · Singapore</title><style>'+STYLE+'</style></head><body><main>'
      '<div class="top"><a href="heat_economy.html">← Complete tabbed analysis</a><button class="print" onclick="window.print()">Print selected view / save PDF</button></div>'
      '<header><div class="eyebrow">Singapore · annual frequency · NSA primary</div><h1>Heat and annual output growth.</h1>'
      '<p>Complete-year output and weather, with all eligible years beside the same model excluding observations dated 2020–2022. '
      'NSA annual log growth is primary; annual growth of SA output remains visible as a comparison.</p></header>'
      '<aside class="limit"><p><strong>The PM2.5 baseline has nine annual observations.</strong> The all-year specification has only two residual annual time degrees of freedom. '
      'Excluding 2020–2022 leaves six years and an underidentified requested model. The unavailable column is retained; controls are not dropped to manufacture an estimate.</p></aside>'
      '<div class="controls"><div class="control"><label for="context">Economic series</label><select id="context">'+contexts+'</select></div>'
      '<div class="control"><label for="adjustment">Annual output definition</label><select id="adjustment"><option value="NSA">NSA log growth · primary</option><option value="SA">SA log growth · comparison</option></select></div>'
      '<div class="control"><label for="controls">Weather controls</label><select id="controls"><option value="full">Baseline · rainfall, RH, wind + PM2.5</option><option value="weather">Long-history robustness · rain + RH</option></select></div>'
      '<div class="control"><label for="model">Model</label><select id="model">'+models+'</select></div></div>'
      '<section class="paper" id="annual-selection" data-view-id="'+default["id"]+'"><div class="heading"><h2 id="selected-title">'+esc(default["context_label"]+" · "+default["model_label"])+'</h2>'
      '<span id="selected-role" class="role">Primary · annual NSA growth</span></div><p class="notes" id="selected-outcome"></p><div class="formula" id="formula"></div>'
      '<div class="status-grid" id="status-grid"></div><div id="native-table" class="native-scroll">'+default["native_html"]+'</div>'
      '<h3>Means, p-values and the size of the heat association</h3><div id="readout">'+default["readout_html"]+'</div>'
      '<p class="notes">Growth is in annual log points, not an annualised monthly/quarterly rate. β=−1 means about one percentage point lower annual growth per +1°C, conditional on the fitted model. '
      'The +0.1°C figure rescales the same coefficient; it is not a causal forecast. Labels use unrounded saved p-values: *** p&lt;0.01; ** p&lt;0.05; * p&lt;0.10. '
      'A 5%: Yes badge means p&lt;0.05. Unavailable is different from not significant.</p><div id="joint">'+default["joint_html"]+default["pairwise_html"]+'</div></section>'
      '<section class="methods"><h2>Annual definitions and identification</h2><p><strong>Y = 100[ln annual output(y) − ln annual output(y−1)].</strong> '
      'VA annual levels sum all four quarterly real levels; IIP annual levels average all twelve monthly index levels. Growth is computed after aggregation, on complete source years. '
      'The current year’s air temperature, humidity, wind and PM2.5 are annual means, and rainfall is the annual total. Weather is not converted to a YoY difference. '
      'The published weighted manufacturing total is used directly; an equal-weight industry panel is not the national index.</p>'
      '<p><strong>Annual FE are sector effects plus a linear calendar-year trend.</strong> No seasonal effects are needed at annual frequency. '
      'Year effects would absorb the national annual temperature series, so they are not included. The sector-interaction model estimates temperature plus nonreference sector × temperature deviations; '
      'reported total effects are βT+δᵢ with their full covariance cross term. Other weather slopes and the trend are shared. Own-series regressions allow each sector its own coefficients and trend.</p>'
      '<p><strong>Uncertainty depends on the unit.</strong> Common-slope sector panels cluster by sector using CR1 and t(S−1). National totals and individual-sector slopes use one-/two-year calendar HAC and approximate t(T−1). '
      'Each individual sector slope is supported by one sector, making sector-clustered errors degenerate. Shared national shocks and small cluster counts remain concerns for common-slope CR1; repeated sector rows do not create more annual weather observations.</p>'
      '<p><strong>Stationarity is not certified.</strong> Exact-window ADF/KPSS screens require at least 20 annual observations and a consecutive annual calendar. '
      'The nine-year PM2.5 baseline is too short; the COVID exclusion also leaves calendar gaps. These screens are unavailable rather than failed. '
      'Long-history screens use a constant for growth, and a constant/trend where specified for weather. ADF has a unit-root null; KPSS has a stationarity null. '
      'Compatible screens are evidence, not proof of stationarity, cointegration or causal identification. Nominal regression significance remains conditional and exploratory when stationarity is unresolved.</p>'
      '<p>The exclusion removes observations dated 2020–2022 after growth is constructed. Annual growth dated 2023 still compares with 2022. '
      'Control eligibility, model definitions and actual calendar indices are retained. These annual regressions share the log-growth outcome form used by Dell, Jones and Olken; '
      'a single-country annual analysis is not a replication of their multicountry design.</p>'+reconciliation
      +'<details><summary>Download saved estimates, statuses, covariance and diagnostics</summary><div class="downloads">'+downloads+'</div></details></section>'
      '<footer class="footer"><div class="links"><a href="heat_economy.html">Complete analysis</a><a href="outcome_comparisons.html">Monthly / quarterly SA and NSA YoY</a>'
      '<a href="granular_va.html">Detailed VA</a><a href="manufacturing_detail.html">Manufacturing detail</a><a href="sector_heterogeneity.html">Subannual heterogeneity</a></div>'
      '<p>Estimates come from saved native PyFixest output. Tables use MakeTables ETable, the table engine used by PyFixest etable. '
      'All tables, readouts and downloads are embedded for offline use. Presentation checks are separate from the independent numerical audit.</p></footer>'
      '<script id="annual-data" type="application/json">'+payload+'</script><script>'+SCRIPT+'</script></main></body></html>')
    tree = lh.fromstring(content)
    assert len(tree.xpath('//table[contains(@class,"gt_table")]'))==1
    ids = tree.xpath('//@id');assert len(ids)==len(set(ids))
    REPORT.write_text(content)
    print(f"Wrote outputs/annual.html: {len(views)} paired views; one selected native table",flush=True)
    return REPORT


if __name__=="__main__":
    build_annual_report()
