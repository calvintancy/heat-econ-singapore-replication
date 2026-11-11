"""Native regression tables and source-to-model guide for literature extensions."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import gzip
import hashlib
import json
from pathlib import Path

import maketables as mt
import pandas as pd
from lxml import html

from crisis_report import VARIANTS, HEADS, clean
from regression_tables import namespace_table_ids

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/literature_sensitivity"
SUBANNUAL_BRANCHES={"subannual","full_controls"}
FIELDS={"spec_id","context","context_label","frequency","outcome","outcome_label","structure","specification","model","model_spec","history",
        "sample_variant","sector_id","sector","effect","lag","term","coefficient","std_error","ci_lower","ci_upper","p_value",
        "mean_y","sd_y","n_obs","n_sectors","n_periods","n_parameters","time_residual_df","calendar_gaps","start","end","r_squared",
        "covariance","hac_lags","inference_df","main_inference","controls","formula","formula_rhs","includes_dwellings","estimator_version",
        "history_loss_periods","weather_history_loss_periods","maximum_output_support_horizon","parent_n_periods","parent_periods",
        "maximum_history_n_periods","raw_history_loss_periods","raw_AR_history_loss_periods","target_alignment_loss_periods","raw_model_history_eligible_n_periods","matched_history_loss_periods","control_raw_missing_loss_periods","control_coverage_loss_periods","control_total_loss_periods","crisis_loss_periods",
        "status","reason","test","null","restrictions","covariance_rank","covariance_condition","f_statistic","evaluation_temperature","effect_units","mean_temperature","joint_test_fragile","fragility_reason"}
MODEL_LABELS={"benchmark":"Contemporaneous weather · matched dates","distributed":"Unrestricted temperature + rainfall lags",
              "dynamic":"Distributed weather + lagged growth","two_channel":"YoY · endpoint difference + accumulated exposure",
              "DJO0":"DJO-inspired · current annual weather","DJO1":"DJO-inspired · current + 1 annual weather lag",
              "DJO5":"DJO-inspired · current + 5 annual weather lags","DJO10":"DJO-inspired · current + 10 annual weather lags",
              "BHM_common_trend":"BHM-inspired quadratic · common linear trend",
              "BHM_unit_quadratic_trends":"BHM-inspired quadratic · sector linear/quadratic trends",
              "seasonal_current":"Calendar Q1–Q4 temperature · current year",
              "seasonal_current_prior":"Calendar Q1–Q4 · current and prior year",
              "seasonal_current_AR1":"Current Q1–Q4 weather + annual growth AR1",
              "seasonal_current_prior_AR1":"Current/prior Q1–Q4 weather + annual growth AR1"}
ANNUAL_LABELS={"aggregate_va":"Official aggregate GVA · annual",
              "va11":"Legacy 11 broad VA sectors · excludes dwellings",
              "va19":"Legacy 19 detailed VA sectors · excludes dwellings",
              "va12":"Complete 12 broad VA components · includes dwellings",
              "va20":"Complete 20 detailed VA components · includes dwellings",
              "iip_total":"Published weighted total IIP · annual",
              "iip6":"6 manufacturing clusters · annual",
              "iip18":"18 manufacturing industries · annual",
              "aggregate_va_matched_va20":"Official aggregate GVA · detailed-component dates",
              "va11_matched_va19":"Legacy 11 VA sectors · 19-sector dates",
              "va12_matched_va20":"Complete 12 VA components · 20-component dates",
              "iip_total_matched_iip6":"Published IIP total · 6-cluster dates",
              "iip_total_matched_iip18":"Published IIP total · 18-industry dates",
              "iip6_matched_iip18":"6 manufacturing clusters · 18-industry dates"}


def read(path):
    return pd.read_csv(path,usecols=lambda c:c in FIELDS,dtype={"sector_id":str},float_precision="round_trip",low_memory=False)


def normalize(frame,branch):
    f=frame.copy()
    if "sector_id" not in f:
        f["sector_id"]="pooled"
    f["branch"]=branch
    if branch in SUBANNUAL_BRANCHES:
        f=f.rename(columns={old:new for old,new in {"specification":"model_spec","structure":"model"}.items() if new not in f})
        f["history"]="full_controls_max_AR_history" if branch=="full_controls" else "matched_maximum_AR_history"
        short=f.outcome.eq("y_short")
        if "outcome_label" in f:
            f.loc[short,"outcome_label"]=f.loc[short,"frequency"].map({"quarterly":"QoQ log growth (SA)","monthly":"MoM log growth (SA)"})
    else:
        f["frequency"]="annual"
        f["context_label"]=f.context.map(ANNUAL_LABELS)
        assert f.context_label.notna().all()
        f["outcome_label"]=f.outcome.map({"y_nsa":"Annual log growth (NSA)","y_sa":"Annual log growth (SA)"})
        if "null" in f:
            f=f.rename(columns={"null":"test"})
    if branch=="annual" and "parent_periods" in f:
        # This non-AR family requires both SA/NSA outcomes to be finite together.
        # Weather-history eligibility is identical across them, including rows
        # that store one unavailable design request before outcome expansion.
        group=["context","history","model_spec","model","sector_id"]
        f["maximum_history_n_periods"]=f.groupby(group,dropna=False).n_periods.transform("max")
        f["raw_history_loss_periods"]=f.parent_periods-f.maximum_history_n_periods
        f["target_alignment_loss_periods"]=0  # current parents already match
    if "target_alignment_loss_periods" not in f and "matched_history_loss_periods" in f:
        f["target_alignment_loss_periods"]=f.matched_history_loss_periods
    if "raw_history_loss_periods" not in f and "raw_model_history_eligible_n_periods" in f and "parent_n_periods" in f:
        f["raw_history_loss_periods"]=f.parent_n_periods-f.raw_model_history_eligible_n_periods
    return f


def term_label(term):
    if term.startswith("temp_Q"):
        quarter,lag=term.removeprefix("temp_").split("_l")
        return f"{quarter} mean temperature · "+("current year" if lag=="0" else "prior year")
    if term.endswith("_current_plus_prior"):
        return term.split("_")[0]+" · current + prior coefficient sum"
    if term.startswith("temp_l"):
        return "Temperature · lag "+term.removeprefix("temp_l")
    if term.startswith("T_lag_"):
        return "Temperature · lag "+term.removeprefix("T_lag_")
    return {"sum_temperature_lags":"Sum of direct temperature coefficients","temperature_lag_sum":"Sum of temperature coefficients",
            "temp_endpoint_difference":"Endpoint temperature difference coefficient","temp_growth_sum":"Accumulated temperature SUM coefficient",
            "temperature_linear_centered":"Temperature · linear, centered","temperature_quadratic":"Temperature · squared, centered",
            "marginal_effect_p5":"Marginal effect at observed T p5","marginal_effect_p50":"Marginal effect at observed T median",
            "marginal_effect_p95":"Marginal effect at observed T p95","sum_all_temperature_features":"Sum of all direct seasonal coefficients",
            "sum_current_Q1_Q4":"Sum of current Q1–Q4 coefficients","sum_prior_Q1_Q4":"Sum of prior Q1–Q4 coefficients"}.get(term,term)


@dataclass
class Result:
    rows:pd.DataFrame
    annual:bool
    dependent:str

    @property
    def __maketables_coef_table__(self):
        f=self.rows.set_index("row_key")[["coefficient","std_error","p_value","ci_lower","ci_upper"]]
        assert f.index.is_unique
        return f.rename(columns={"coefficient":"b","std_error":"se","p_value":"p","ci_lower":"ci95l","ci_upper":"ci95u"})

    @property
    def __maketables_depvar__(self):return self.dependent

    @property
    def __maketables_fixef_string__(self):
        r=self.rows.iloc[0]
        return ("sector" if r.n_sectors>1 else "") if self.annual else ("sector_season" if r.n_sectors>1 else "season")+"+year"

    def __maketables_stat__(self,key):
        r=self.rows.iloc[0]
        return {"N":int(r.n_obs),"periods":int(r.n_periods)}.get(key)


def native_table(rows,info,view_id):
    if rows.empty:return ""
    rows=rows[rows.coefficient.notna()].copy()
    if rows.empty:return ""
    if info["model"]=="common":
        selected=rows.copy()
    elif info["model_spec"]=="two_channel":
        selected=rows[rows.term.isin(["temp_endpoint_difference","temp_growth_sum"])].copy()
    elif info["model_spec"].startswith("BHM"):
        selected=rows[rows.term.eq("marginal_effect_p50")].copy()
    else:
        selected=rows[rows.term.isin(["sum_temperature_lags","temperature_lag_sum","sum_all_temperature_features"])].copy()
    if selected.empty:return ""
    profile=info["model"]!="common"
    selected["row_key"]=selected.sector_id+"|"+selected.term if profile else selected.term
    names={r.row_key:(str(r.sector)+" · "+term_label(r.term) if profile and info["model_spec"]=="two_channel" else str(r.sector) if profile else term_label(r.term)) for r in selected.drop_duplicates("row_key").itertuples()}
    included=[v for v in VARIANTS if selected.sample_variant.eq(v).any()]
    models=[Result(selected[selected.sample_variant.eq(v)],info["branch"] not in SUBANNUAL_BRANCHES,info["outcome_label"]) for v in included]
    first=[m.rows.iloc[0] for m in models]
    stats={"Weather controls":[info["controls_label"]]*len(models),"Inference":[info["inference"]]*len(models),
           "Parameters per equation":[int(r.n_parameters) for r in first],
           "Residual time df":[int(r.time_residual_df) for r in first],"Inference df":[int(r.inference_df) for r in first],
           "Missing calendar dates":[int(r.calendar_gaps) for r in first],"First period":[str(r.start) for r in first],"Last period":[str(r.end) for r in first],
           "Lagged growth":["Lags 1–h; shared in H1" if info["model_spec"]=="dynamic" else "AR1; shared in H1" if info["model_spec"].endswith("AR1") else "None"]*len(models),
           "Time trends":["Sector linear + quadratic" if info["model_spec"]=="BHM_unit_quadratic_trends" else "Common linear" if info["branch"] not in SUBANNUAL_BRANCHES else "Year FE"]*len(models)}
    if not profile:
        stats["Mean Y"]=[f"{r.mean_y:.3f}" for r in first]
        stats["SD Y"]=[f"{r.sd_y:.3f}" for r in first]
    if info["model_spec"].startswith("BHM"):
        stats["Temperature basis center (°C)"]=[f"{r.mean_temperature:.3f}" for r in first]
        if profile:
            stats["Derivative evaluation T (°C)"]=[f"{r.evaluation_temperature:.3f}" for r in first]
    if info["model"]!="separate":stats["R² including FE"]=[f"{r.r_squared:.3f}" for r in first]
    table=mt.ETable(models,model_heads=[HEADS[VARIANTS.index(v)] for v in included],head_order="dh",coef_fmt="b:.3f* \n (se:.3f)",
        signif_code=[.01,.05,.10],keep=list(names),exact_match=True,labels=names,
        felabels={"sector":"Sector FE","sector_season":"Sector × calendar-season FE","season":"Calendar-season FE","year":"Common year FE"},
        feorder=["sector","sector_season","season","year"],model_stats=["N","periods"],
        model_stats_labels={"N":"Observations per equation" if info["model"]=="separate" else "Observations","periods":"Distinct weather dates"},
        custom_model_stats=stats,stats_order="sc",notes="Native PyFixest estimates; MakeTables ETable presentation. Temperature excerpts: controls and deterministic terms are included as labelled. Sector totals include main/interaction covariance. Cumulative direct coefficients are not unconditional long-run dynamic multipliers. Quadratic sector tables show the derivative at the observed median temperature; other percentiles and every lag appear in the readout. Missing sample columns are unavailable equations, explicitly listed above. Stars: *** p<0.01, ** p<0.05, * p<0.10; nominal pointwise inference, without full-grid multiplicity adjustment.")
    raw=table.make(type="gt",gt_style={"table_width":"100%","first_col_width":"300px","table_font_size_all":"13px"}).as_raw_html()
    element=html.fragment_fromstring(raw,create_parent="div").xpath(".//table")[0]
    for node in element.iter():node.attrib.pop("style",None)
    return namespace_table_ids(html.tostring(element,encoding="unicode"),view_id)


def build():
    views=[];source_hashes={};receipts={}
    dimensions=["context","history","outcome","model_spec","model"]
    for branch,folder in [("subannual",OUT),("full_controls",OUT/"full_controls"),("annual",OUT/"annual"),("annual_seasonal",OUT/"annual_seasonal")]:
        receipt_path=folder/"verification.json"
        receipt=json.loads(receipt_path.read_text())
        assert receipt["status"]=="passed",receipt_path
        receipts[branch]={"status":receipt["status"],"source":str(receipt_path.relative_to(ROOT)),"sha256":hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
        names=["temperature_effects","channel_estimates","tests","two_channel_span_tests","specifications"] if branch in SUBANNUAL_BRANCHES else ["temperature_effects","tests","specifications"] if branch=="annual_seasonal" else ["estimates","tests","specifications"]
        frames={}
        for name in names:
            path=folder/f"{name}.csv";source_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            frames[name]=normalize(read(path),branch)
        if branch=="full_controls":
            metadata=frames["specifications"]
            assert metadata.spec_id.is_unique
            for name in names:
                if name=="specifications":continue
                missing=[c for c in metadata if c not in frames[name]]
                frames[name]=frames[name].merge(metadata[["spec_id"]+missing],on="spec_id",how="left",validate="many_to_one")
        estimates=pd.concat([frames["temperature_effects"],frames["channel_estimates"]],ignore_index=True) if branch in SUBANNUAL_BRANCHES else frames["temperature_effects"] if branch=="annual_seasonal" else frames["estimates"]
        estimates["term_label"]=estimates.term.map(term_label)
        tests=pd.concat([frames["tests"],frames["two_channel_span_tests"]],ignore_index=True) if branch in SUBANNUAL_BRANCHES else frames["tests"]
        specs=frames["specifications"]
        if branch=="annual":
            missing=specs[specs.outcome.isna()].copy();specs=specs[specs.outcome.notna()].copy()
            for outcome in ["y_nsa","y_sa"]:
                add=missing.copy();add["outcome"]=outcome;add["outcome_label"]="Annual log growth ("+("NSA" if outcome=="y_nsa" else "SA")+")"
                specs=pd.concat([specs,add],ignore_index=True)
        e_groups={key:frame for key,frame in estimates.groupby(dimensions,sort=False)}
        t_groups={key:frame for key,frame in tests.groupby(dimensions,sort=False)}
        for key, requested in specs.groupby(dimensions,sort=False):
            context,history,outcome,model_spec,model=key
            e=e_groups.get(key,pd.DataFrame())
            covariances=e[["covariance","hac_lags"]].drop_duplicates().itertuples(index=False,name=None) if len(e) else [("Unavailable",0)]
            for covariance,band in covariances:
                rows=e[e.covariance.eq(covariance)&e.hac_lags.eq(band)].copy() if len(e) else e
                selected_tests=t_groups.get(key,pd.DataFrame())
                if len(selected_tests):selected_tests=selected_tests[selected_tests.covariance.eq(covariance)&selected_tests.hac_lags.eq(band)]
                first=requested.iloc[0]
                info={"branch":branch,"context":context,"context_label":str(first.context_label),"history":history,
                      "outcome":outcome,"outcome_label":str(first.outcome_label),"model_spec":model_spec,"model_spec_label":MODEL_LABELS[model_spec],
                      "model":model,"frequency":first.frequency,"covariance":covariance,"hac_lags":int(band),
                      "inference":covariance+(f" · {int(band)} calendar lags" if band else " · CR1" if covariance!="Unavailable" else ""),
                      "controls_label":"Current Q1–Q4 rainfall + annual humidity" if model_spec in {"seasonal_current","seasonal_current_AR1"} else "Current/prior Q1–Q4 rainfall + annual humidity" if model_spec.startswith("seasonal") else ("Current rainfall + humidity" if model_spec in {"benchmark","DJO0"} else "Rainfall + rainfall² + humidity" if model_spec.startswith("BHM") else "Current and lagged rainfall + current humidity")+(" + current wind + PM2.5" if branch=="full_controls" else ""),
                      "includes_dwellings":bool(first.includes_dwellings),"source_csv":str((folder/("estimates.csv" if branch=="annual" else "temperature_effects.csv")).relative_to(ROOT))}
                view_id="l"+hashlib.sha256(json.dumps([branch,*key,covariance,int(band)]).encode()).hexdigest()[:16]
                statuscols=[c for c in ["sample_variant","sector_id","status","reason","n_obs","n_periods","n_parameters","time_residual_df","start","end","parent_n_periods","parent_periods","history_loss_periods","raw_history_loss_periods","raw_AR_history_loss_periods","weather_history_loss_periods","control_raw_missing_loss_periods","control_coverage_loss_periods","control_total_loss_periods","target_alignment_loss_periods","maximum_history_n_periods","maximum_output_support_horizon","controls","formula","formula_rhs"] if c in requested]
                rowcols=[c for c in ["sample_variant","sector_id","sector","term","term_label","coefficient","std_error","ci_lower","ci_upper","p_value","mean_y","sd_y","n_obs","n_periods","n_parameters","time_residual_df","r_squared","start","end","evaluation_temperature","mean_temperature","effect_units","history_loss_periods","maximum_output_support_horizon"] if c in rows]
                testcols=[c for c in ["sample_variant","sector_id","sector","test","status","p_value","restrictions","covariance_rank","covariance_condition","f_statistic","n_periods","time_residual_df","inference_df","joint_test_fragile","fragility_reason"] if c in selected_tests]
                views.append({**info,"id":view_id,"table":native_table(rows,info,view_id),"rows":clean(rows[rowcols]) if len(rows) else [],
                              "requests":clean(requested[statuscols]),"tests":clean(selected_tests[testcols]) if len(selected_tests) else []})
        print(f"Literature report: {branch} prepared",flush=True)
    inventory={"views":len(views),"sources":source_hashes,"verification":receipts}
    (OUT/"report_inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
    raw=json.dumps({**inventory,"views":views},ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    packed=base64.b64encode(gzip.compress(raw,compresslevel=6,mtime=0)).decode()
    document=(ROOT/"templates/literature_sensitivity.html").read_text().replace("__RESULTS_PAYLOAD__",packed)
    path=ROOT/"outputs/literature_sensitivity.html";path.write_text(document)
    (OUT/"payload_packaging.json").write_text(json.dumps({"status":"fresh_generation","views":len(views),
        "raw_payload_bytes":len(raw),"packed_payload_bytes":len(packed),"html_bytes":path.stat().st_size,
        "decoded_payload_sha256":hashlib.sha256(raw).hexdigest(),"encoding":"gzip-base64"},indent=2)+"\n")
    print(f"Wrote {path}: {len(views)} views",flush=True)
    return path


def repackage_existing():
    """Packaging-only conversion; native tables and numerical payload unchanged."""
    path=ROOT/"outputs/literature_sensitivity.html"
    previous=path.read_bytes()
    doc=html.fromstring(previous,parser=html.HTMLParser(huge_tree=True))
    node=doc.get_element_by_id("literature-data")
    data=json.loads(gzip.decompress(base64.b64decode(node.text))) if node.get("data-encoding")=="gzip-base64" else json.loads(node.text)
    raw=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    packed=base64.b64encode(gzip.compress(raw,compresslevel=6,mtime=0)).decode()
    target=ROOT/"dist/aea_audit_20261111/literature_before_payload_compression.html"
    if not target.exists():target.write_bytes(previous)
    document=(ROOT/"templates/literature_sensitivity.html").read_text().replace("__RESULTS_PAYLOAD__",packed)
    path.write_text(document)
    proof={"status":"packaging_only","views":len(data["views"]),"before_bytes":len(previous),"after_bytes":path.stat().st_size,
           "decoded_payload_sha256":hashlib.sha256(raw).hexdigest(),"all_views_tables_rows_tests_requests_retained":True}
    (OUT/"payload_packaging.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps(proof,indent=2),flush=True)
    return proof


if __name__=="__main__":build()
