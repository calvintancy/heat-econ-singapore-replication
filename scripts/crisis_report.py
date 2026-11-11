"""Offline, native regression tables for complete-GVA and crisis extensions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path

import maketables as mt
import pandas as pd
from lxml import html as lh

from regression_tables import namespace_table_ids

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/crisis_sensitivity"
VARIANTS = ["all_eligible","event_dates_excluded","strict_growth_window_clean"]
HEADS = ["All eligible dates","Crisis dates excluded","Strict growth window clean"]
LABELS = {"temp_c":"Mean air temperature (°C)","rain_100mm":"Rainfall (100 mm)",
          "rh_pct":"Relative humidity (pp)","wind_mps":"Wind (m/s)","pm25_10ug":"PM2.5 (10 µg/m³)"}


def esc(x):
    return html.escape(str(x))


def clean(frame):
    return json.loads(frame.to_json(orient="records",double_precision=15))


@dataclass
class Result:
    rows: pd.DataFrame
    dependent: str
    profile: bool
    annual: bool
    own: bool

    @property
    def __maketables_coef_table__(self):
        key="sector_id" if self.profile else "term"
        f=self.rows.set_index(key)[["coefficient","std_error","p_value","ci_lower","ci_upper"]]
        assert f.index.is_unique
        return f.rename(columns={"coefficient":"b","std_error":"se","p_value":"p","ci_lower":"ci95l","ci_upper":"ci95u"})

    @property
    def __maketables_depvar__(self):
        return self.dependent

    @property
    def __maketables_fixef_string__(self):
        row=self.rows.iloc[0]
        if self.annual:
            return "sector" if row.n_sectors>1 else ""
        return ("sector_season" if row.n_sectors>1 else "season")+"+year"

    def __maketables_stat__(self,key):
        row=self.rows.iloc[0]
        return {"N":int(row.n_obs),"periods":int(row.n_periods),"r2":float(row.r_squared)}.get(key)


def native_table(rows, info, key):
    profile=info["model"]!="common"; own=info["model"]=="separate"
    models=[Result(rows[rows.sample_variant.eq(v)].copy(),info["outcome_label"],profile,info["annual"],own) for v in VARIANTS]
    assert all(len(m.rows) for m in models),key
    first=[m.rows.iloc[0] for m in models]
    names=rows.drop_duplicates("sector_id").set_index("sector_id").sector.to_dict() if profile else {t:LABELS.get(t,t) for t in rows.term.unique()}
    stats={"Estimator":["Native PyFixest "+str(r.estimator_version) for r in first],
           "Controls":[info["controls"]]*3,
           "Inference":[info["inference"]]*3,
           "Inference df":[int(r.inference_df) for r in first],
           "Estimated parameters per fit":[int(r.n_parameters) for r in first],
           "Time residual df":[int(r.time_residual_df) for r in first],
           "Missing calendar dates":[int(r.calendar_gaps) for r in first],
           "First period":[str(r.start) for r in first],"Last period":[str(r.end) for r in first],
           "Linear time trend":["Yes" if info["annual"] else "Absorbed by year FE" for _ in first]}
    if not profile:
        stats["Mean observed Y"]=[f"{r.mean_y:.3f}" for r in first]
        stats["SD observed Y"]=[f"{r.sd_y:.3f}" for r in first]
    if not own:
        stats["R² including FE"]=[f"{r.r_squared:.3f}" for r in first]
    else:
        stats["Units represented"]= [len(m.rows) for m in models]
    stats["Stationarity clearance"]=["Not established; see diagnostics limitations"]*3
    table=mt.ETable(models,model_heads=HEADS,head_order="dh",coef_fmt="b:.3f* \n (se:.3f)",
        signif_code=[.01,.05,.10],keep=list(names),exact_match=True,labels=names,
        felabels={"sector":"Sector FE","sector_season":"Sector × calendar-season FE","season":"Calendar-season FE","year":"Year FE"},
        feorder=["sector","sector_season","season","year"],model_stats=["N","periods"],
        model_stats_labels={"N":"Observations per equation" if own else "Observations","periods":"Distinct weather dates"},
        custom_model_stats=stats,stats_order="sc",notes="Stars: *** p<0.01, ** p<0.05, * p<0.10. Exact p-values and explicit 5% decisions appear below. Sector FE rows show total effects βT + δᵢ, including the covariance cross-term. Own-series rows are separate equations. All tests are conditional and pointwise; this grid is exploratory. Table engine: MakeTables ETable, used by PyFixest. Annual tables are temperature excerpts with included controls stated above.")
    raw=table.make(type="gt",gt_style={"table_width":"100%","table_font_size_all":"13px","first_col_width":"290px"}).as_raw_html()
    doc=lh.fragment_fromstring(raw,create_parent="div"); element=doc.xpath(".//table")[0]
    for node in element.iter():
        node.attrib.pop("style",None)
    return namespace_table_ids(lh.tostring(element,encoding="unicode"),key)


def build():
    views=[]; sources={}; verifications={}
    branches=[("long","Long history · temperature, rain, humidity",OUT,False),
              ("full","Complete GVA · wind and PM2.5 baseline",ROOT/"outputs/complete_va_controls",False),
              ("annual","Annual · temperature, rain, humidity",OUT/"annual",True)]
    for branch,label,folder,is_annual in branches:
        verify=folder/"verification.json"
        receipt=json.loads(verify.read_text())
        assert receipt["status"]=="passed",verify
        verifications[branch]={"source":str(verify.relative_to(ROOT)),"sha256":hashlib.sha256(verify.read_bytes()).hexdigest(),"status":receipt["status"]}
        jpath=folder/"joint_tests.csv"; joints=pd.read_csv(jpath)
        sources[str(jpath.relative_to(ROOT))]=hashlib.sha256(jpath.read_bytes()).hexdigest()
        for filename,model in [("common_estimates" if is_annual else "estimates","common"),("sector_estimates","H1"),("separate_estimates","separate")]:
            path=folder/f"{filename}.csv"; f=pd.read_csv(path,dtype={"sector_id":str},float_precision="round_trip",low_memory=False)
            sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            if "term" not in f:
                f["term"]="temp_c" if model=="common" else "total_temperature_effect"
            if model=="separate":
                f=f[f.term.isin(["temp_c","total_temperature_effect"])].copy()
            if is_annual:
                f["frequency"]="annual"
                f["outcome_label"]=f.outcome_label.str.replace("Annual log growth","Annual log growth",regex=False)
            else:
                short=f.outcome.eq("y_short")
                f.loc[short,"outcome_label"]=f.loc[short,"frequency"].map({"quarterly":"QoQ log growth (SA; not annualised)","monthly":"MoM log growth (SA; not annualised)"})
            if "context_label" not in f:
                f["context_label"]=f.context.str.replace("_"," ")
            for cols,rows in f.groupby(["context","outcome","covariance","hac_lags"],sort=False):
                context,outcome,covariance,band=cols
                info={"branch":branch,"branch_label":label,"context":context,"context_label":rows.context_label.iloc[0],
                      "outcome":outcome,"outcome_label":rows.outcome_label.iloc[0],"model":model,"annual":is_annual,
                      "frequency":rows.frequency.iloc[0],"covariance":covariance,"hac_lags":int(band),
                      "n_sectors":int(rows.n_sectors.iloc[0]),
                      "controls":"Temperature, rainfall, humidity, wind, PM2.5" if branch=="full" else "Temperature, rainfall, humidity",
                      "inference":covariance+(f" · {int(band)} calendar lags" if band else " · CR1"),
                      "source_csv":str(path.relative_to(ROOT)),"includes_dwellings":bool(rows.includes_dwellings.iloc[0])}
                key="r"+hashlib.sha256(json.dumps([branch,model,*map(str,cols)]).encode()).hexdigest()[:16]
                keep=[c for c in ["sample_variant","sector_id","sector","term","coefficient","std_error","ci_lower","ci_upper","p_value","p_value_holm_within_table","mean_y","sd_y","n_obs","n_periods","time_residual_df","r_squared","start","end","calendar_gaps","stationarity_status","n_parameters"] if c in rows]
                relevant=joints[joints.context.eq(context)&joints.outcome.eq(outcome)&joints.model.eq(model)&joints.covariance.eq(covariance)&joints.hac_lags.eq(band)]
                jkeep=[c for c in ["sample_variant","status","p_value","restrictions","covariance_rank","covariance_condition","denominator_df","test","null"] if c in relevant]
                views.append({**info,"id":key,"table":native_table(rows,info,key),"rows":clean(rows[keep]),"joint":clean(relevant[jkeep])})
    payload={"views":views,"sources":sources,"verification":verifications}
    (OUT/"report_inventory.json").write_text(json.dumps({"views":len(views),"sources":sources,"verification":verifications},indent=2)+"\n")
    template=(ROOT/"templates/crisis_sensitivity.html").read_text()
    document=template.replace("__RESULTS_PAYLOAD__",json.dumps(payload,ensure_ascii=False,allow_nan=False).replace("<","\\u003c"))
    assert "__RESULTS_PAYLOAD__" not in document
    destination=ROOT/"outputs/crisis_sensitivity.html";destination.write_text(document)
    print(f"Wrote {destination}: {len(views)} native three-column tables",flush=True)
    return destination


if __name__=="__main__":
    build()
