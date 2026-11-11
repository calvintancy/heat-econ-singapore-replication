"""Portable native-table guide to the separately verified arrival extension."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import maketables as mt
import numpy as np
import pandas as pd
from lxml import html
import markdown

from regression_tables import namespace_table_ids

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/visitor_rigour"
VARIANTS=["all_eligible","event_dates_excluded","strict_growth_window_clean"]
HEADS=["All eligible","Exclude event dates","Clean output support"]
BRANCHES={"monthly":"Monthly growth / timing","origin_year":"Matched origin × year FE","annual":"Annual growth","ppml":"Count levels / zero retention"}
LABELS={"aggregate":"Official national total","region":"Nine selected regions","country":"42 named markets",
        "complete10":"Complete 10-bucket partition","complete52":"Complete 52-bucket partition",
        "own_current":"Current weather · own history","benchmark":"Current weather · matched history",
        "distributed":"Unrestricted weather lags","dynamic":"Weather lags + lagged growth",
        "two_channel":"YoY endpoint difference + exposure SUM","common":"Common slope","H1":"Temperature × origin FE",
        "current":"Current weather",
        "long":"Long weather controls","full":"Full controls, including wind / PM2.5"}
KEEP={"spec_id","panel","outcome","outcome_label","control_set","history","sample_variant","model_spec","model","status","reason",
      "unit_id","unit","sector_id","sector","term","effect","channel","lag","feature","component","estimated","normalization",
      "coefficient","std_error","ci_lower","ci_upper","p_value","p_value_holm_profile","mean_y","sd_y","n_obs","n_periods","n_sectors",
      "n_parameters","time_residual_df","calendar_gaps","start","end","covariance","hac_lags","inference_df","finite_sample_correction",
      "finite_sample_correction_formula","main_inference","inference_status","calibration_status","temp_main","interaction_delta",
      "delta_is_normalization","var_temp_main","var_interaction_delta","cov_temp_main_delta","restrictions","covariance_rank",
      "structural_rank_bound","covariance_condition","f_statistic","test","joint_test_fragile","fragility_reason","zero_cells_retained",
      "output_support_horizon","parent_n_periods","current_own_output_n_periods","own_output_history_n_periods","output_history_loss_periods",
      "weather_history_loss_periods","shared_panel_alignment_loss_periods","control_raw_missing_loss_periods","control_coverage_loss_periods",
      "crisis_loss_periods","sample_csv","formula","controls","temp_features","control_features","ar_features","time_design_rank",
      "local_time_design_ranks","full_cross_origin_temporal_span_rank","sample_id","alias_of","alias_request","effect_units","interpretation"}
KEEP|={"raw_native_coefficient","exact_percent_0p1c","effect_0p1c","native_spec_id","native_fit_executed","retained_zero_cells",
       "dropped_rows","dropped_units","native_converged","native_warnings","zero_retention_status","input_n_obs","input_zero_cells",
       "fit_id","estimator","estimator_version","native_label","sample_sha256","nuisance_spec","source_spec_id"}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if value is pd.NA:return None
    return value


def records(frame):return clean(frame[[c for c in frame if c in KEEP]].to_dict("records"))


def label(term):
    if str(term).startswith("temp_l"):return "Temperature · lag "+str(term)[6:]
    if str(term).startswith("rain_l"):return "Rainfall / 100 mm · lag "+str(term)[6:]
    if str(term).startswith("ar_l"):return "Lagged growth · "+str(term)[4:]
    return {"temp_c":"Temperature (°C)","rain_100mm":"Rainfall (100 mm)","rh_pct":"Relative humidity (pp)",
        "wind_mps":"Wind (m/s)","pm25_10ug":"PM2.5 (10 μg/m³)","lny_window_days":"Lunar New Year window days",
        "lny_lag1":"Prior-month holiday window days","sum_temperature_lags":"Sum of direct temperature coefficients",
        "temp_endpoint_difference":"Endpoint temperature difference","temp_growth_sum":"Trailing-year temperature SUM"}.get(str(term),str(term))


@dataclass
class NativeTable:
    rows:pd.DataFrame
    metadata:dict
    branch:str

    @property
    def __maketables_coef_table__(self):
        selected=self.rows.set_index("row_key")
        assert selected.index.is_unique
        return selected[["coefficient","std_error","p_value","ci_lower","ci_upper"]].rename(columns={
            "coefficient":"b","std_error":"se","p_value":"p","ci_lower":"ci95l","ci_upper":"ci95u"})

    @property
    def __maketables_depvar__(self):return self.metadata.get("outcome_label",self.metadata["outcome"])

    @property
    def __maketables_fixef_string__(self):
        if self.branch=="annual":return "origin" if self.metadata["n_sectors"]>1 else ""
        if self.branch=="origin_year":return "origin_season+origin_year"
        return ("origin_season" if self.metadata["n_sectors"]>1 else "season")+"+year"

    def __maketables_stat__(self,key):return {"N":int(self.metadata["n_obs"]),"periods":int(self.metadata["n_periods"])}.get(key)


def native_table(effects,channels,controls,requests,info,view_id):
    if effects.empty and channels.empty:return ""
    if info["model"]=="H1":
        selected=channels.copy() if info["model_spec"]=="two_channel" else effects.loc[effects.effect.eq("current" if info["branch"]=="annual" else "cumulative_direct")].copy()
    elif info["branch"]=="ppml":selected=effects.copy()
    else:
        selected=channels.copy() if info["model_spec"]=="two_channel" else effects.loc[effects.effect.eq("current" if info["branch"]=="annual" else "lag")].copy()
        selected=pd.concat([selected,controls],ignore_index=True)
    selected=selected.loc[selected.coefficient.notna()].copy()
    if selected.empty:return ""
    selected["row_key"]=selected.unit_id.astype(str)+"|"+selected.term if info["model"]=="H1" else selected.term
    names={row.row_key:(str(row.unit)+(" · "+label(row.term) if info["model_spec"]=="two_channel" else "")
            if info["model"]=="H1" else label(row.term)) for row in selected.drop_duplicates("row_key").itertuples()}
    variants=[v for v in VARIANTS if selected.sample_variant.eq(v).any()]
    models=[]
    for variant in variants:
        meta=requests.loc[requests.sample_variant.eq(variant)].iloc[0].to_dict()
        models.append(NativeTable(selected.loc[selected.sample_variant.eq(variant)],meta,info["branch"]))
    first=[model.metadata for model in models]
    stats={"Inference":[info["method"]]*len(models),"Weather controls":[LABELS.get(info["control_set"],info["control_set"])]*len(models),
        "Parameters":[int(r["n_parameters"]) for r in first],"Residual time df":[int(r["time_residual_df"]) for r in first],
        "First period":[str(r["start"]) for r in first],"Last period":[str(r["end"]) for r in first],
        "Time controls":["Common linear trend" if info["branch"]=="annual" else "Origin × year FE" if info["branch"]=="origin_year" else "Common year FE"]*len(models)}
    if info["model"]=="common":
        stats["Mean Y"]=[f'{r["mean_y"]:.3f}' for r in first];stats["SD Y"]=[f'{r["sd_y"]:.3f}' for r in first]
    table=mt.ETable(models,model_heads=[HEADS[VARIANTS.index(v)] for v in variants],head_order="dh",coef_fmt="b:.3f* \n (se:.3f)",
        signif_code=[.01,.05,.10],keep=list(names),exact_match=True,labels=names,
        felabels={"origin":"Origin FE","origin_season":"Origin × calendar-month FE","origin_year":"Origin × year FE","season":"Calendar-month FE","year":"Common year FE"},
        feorder=["origin","origin_season","origin_year","season","year"],model_stats=["N","periods"],
        model_stats_labels={"N":"Observations","periods":"Distinct weather dates"},custom_model_stats=stats,stats_order="sc",
        notes="Native PyFixest estimates; MakeTables ETable presentation. H1 rows are total origin effects, including main/interaction covariance. H1 timing tables show direct coefficient sums; the complete lag vector and shared controls are below. Growth-lag sums are conditional, not unconditional long-run multipliers. Stars are nominal pointwise *** p<0.01, ** p<0.05, * p<0.10. Missing sample columns have explicit unavailable requests above. PPML is a count-level model with separate units.")
    raw=table.make(type="gt",gt_style={"table_width":"100%","first_col_width":"290px","table_font_size_all":"13px"}).as_raw_html()
    element=html.fragment_fromstring(raw,create_parent="div").xpath(".//table")[0]
    for node in element.iter():node.attrib.pop("style",None)
    return namespace_table_ids(html.tostring(element,encoding="unicode"),view_id)


def read(path):
    return pd.read_csv(path,dtype={"unit_id":str,"sector_id":str},low_memory=False,float_precision="round_trip")


def load_branch(branch,sources):
    folder=OUT/branch
    proof=folder/"verification.json"
    receipt=json.loads(proof.read_text());assert receipt["status"]=="passed",proof
    for name,digest in receipt["verified_artifacts_sha256"].items():
        assert sha(ROOT/name)==digest,(branch,name,"Verified analytical artifact changed")
    frames={}
    for name in ["specifications","temperature_effects","control_estimates","channel_estimates","interaction_components","tests"]:
        path=folder/(name+".csv")
        if path.exists():
            frame=pd.DataFrame() if name in {"channel_estimates","interaction_components"} and path.stat().st_size<=1 else read(path)
            sources[str(path.relative_to(ROOT))]=sha(path)
        else:frame=pd.DataFrame()
        frames[name]=frame
    sources[str(proof.relative_to(ROOT))]=sha(proof)
    specs=frames["specifications"]
    for field,value in [("model_spec","current"),("model","common"),("outcome","count_level"),("outcome_label","Expected arrival counts")]:
        if field not in specs:specs[field]=value
    if branch=="annual":specs["outcome_label"]="Annual log arrivals growth (NSA; 100 log points)"
    if branch=="ppml":specs["outcome_label"]="Expected arrival count levels (PPML; 100 log-mean points)"
    for name,frame in frames.items():
        if name=="specifications" or frame.empty:continue
        if "spec_id" in frame:
            missing=[c for c in specs if c not in frame and c in KEEP]
            frame=frame.merge(specs[["spec_id"]+missing],on="spec_id",how="left",validate="many_to_one")
        if "unit_id" not in frame and "sector_id" in frame:frame["unit_id"]=frame.sector_id
        if "unit" not in frame and "sector" in frame:frame["unit"]=frame.sector
        if "p_value_holm_within_profile" in frame:frame["p_value_holm_profile"]=frame.p_value_holm_within_profile
        if branch=="annual" and name=="interaction_components" and "feature" not in frame:frame["feature"]=frame.term
        if "fragile" in frame and "joint_test_fragile" not in frame:frame["joint_test_fragile"]=frame.fragile
        aliases={"parent_growth_periods":"parent_n_periods","shared_cut_alignment_loss_periods":"shared_panel_alignment_loss_periods",
            "control_raw_missing_periods":"control_raw_missing_loss_periods","control_coverage_excluded_periods":"control_coverage_loss_periods",
            "crisis_removed_periods":"crisis_loss_periods","source_count_months":"parent_n_periods","removed_for_event_dates":"crisis_loss_periods"}
        for old,new in aliases.items():
            if old in frame and new not in frame:frame[new]=frame[old]
        frames[name]=frame
    for old,new in {"parent_growth_periods":"parent_n_periods","shared_cut_alignment_loss_periods":"shared_panel_alignment_loss_periods",
            "control_raw_missing_periods":"control_raw_missing_loss_periods","control_coverage_excluded_periods":"control_coverage_loss_periods",
            "crisis_removed_periods":"crisis_loss_periods","source_count_months":"parent_n_periods","removed_for_event_dates":"crisis_loss_periods"}.items():
        if old in specs and new not in specs:specs[new]=specs[old]
    return frames


def build_visitor_rigour_report():
    OUT.mkdir(exist_ok=True)
    sources={};views=[];counts={}
    for branch in BRANCHES:
        frames=load_branch(branch,sources);specs=frames["specifications"]
        effects=frames["temperature_effects"]
        counts[branch]={"requests":len(specs),"estimated_requests":int(specs.status.eq("estimated").sum()),
            "native_fits":int(specs.native_fit_executed.sum()) if branch=="ppml" else int(specs.fit_id.dropna().nunique()),
            "aliases":int(specs.alias_of.notna().sum()) if "alias_of" in specs else 0}
        groups=["panel","outcome","control_set","model_spec","model"]
        for group,requests in specs.groupby(groups,sort=False,dropna=False):
            info=dict(zip(groups,group,strict=True));info["branch"]=branch
            ids=set(requests.spec_id)
            selected=effects.loc[effects.spec_id.isin(ids)]
            methods=selected[["covariance","hac_lags"]].drop_duplicates().to_dict("records")
            if not methods:
                methods=[{"covariance":"Calendar HAC","hac_lags":band} for band in ([1,2] if branch=="annual" else [12,24])]
            for method in methods:
                view={**info,**method};band=int(method["hac_lags"])
                view["method"]=str(method["covariance"])+(f' · {band} '+("years" if branch=="annual" else "months") if band else "")
                view["id"]="v"+str(len(views));view["requests"]=records(requests)
                parts={}
                for name,frame in frames.items():
                    if name=="specifications" or frame.empty:parts[name]=pd.DataFrame();continue
                    chosen=frame.loc[frame.spec_id.isin(ids)]
                    if "covariance" in chosen:chosen=chosen.loc[chosen.covariance.eq(method["covariance"]) & chosen.hac_lags.eq(band)]
                    parts[name]=chosen
                    view[name]=records(chosen)
                for name in ["temperature_effects","channel_estimates","control_estimates","interaction_components","tests"]:
                    view.setdefault(name,[])
                view["outcome_label"]=str(requests.iloc[0].outcome_label)
                view["table"]=native_table(parts["temperature_effects"],parts["channel_estimates"],parts["control_estimates"],requests,view,view["id"])
                views.append(view)
    diagnostics=OUT/"diagnostics/report_payload.json";diagnostic_proof=OUT/"diagnostics/verification.json"
    diagnostic_receipt=json.loads(diagnostic_proof.read_text());assert diagnostic_receipt["status"]=="passed"
    for name,digest in diagnostic_receipt["verified_artifacts_sha256"].items():assert sha(ROOT/name)==digest
    sources[str(diagnostics.relative_to(ROOT))]=sha(diagnostics);sources[str(diagnostic_proof.relative_to(ROOT))]=sha(diagnostic_proof)
    stata=ROOT/"outputs/stata_key_results/arrivals"
    stata_proof=stata/"verification.json";assert json.loads(stata_proof.read_text())["status"]=="passed"
    stata_tables={path.stem:path.read_text() for path in sorted((stata/"native_tables").glob("*.html"))}
    downloads={}
    for branch in BRANCHES:
        for path in sorted((OUT/branch).glob("*.csv")):
            if path.stat().st_size>1 and path.name in {"specifications.csv","temperature_effects.csv","control_estimates.csv","channel_estimates.csv","tests.csv","sample_inventory.csv","aliases.csv","aggregation_checks.csv"}:
                downloads[str(path.relative_to(OUT))]=base64.b64encode(gzip.compress(path.read_bytes(),mtime=0)).decode()
    for name in ["specifications.csv","temperature_effects.csv","verification.json","stata_python_coverage.csv","do/key_results.do","do/postprocess.mata.do","native/master_stata_output.txt"]:
        path=stata/name
        if path.exists():downloads["stata/"+name]=base64.b64encode(gzip.compress(path.read_bytes(),mtime=0)).decode();sources[str(path.relative_to(ROOT))]=sha(path)
    protocol=ROOT/"notes/visitor_rigour_protocol.md";sources[str(protocol.relative_to(ROOT))]=sha(protocol)
    relative_identity=OUT/"monthly/verification_static_relative_identity.csv"
    relative_frame=read(relative_identity)
    identities={"static_cases":relative_frame.spec_id.nunique(),"static_contrasts":len(relative_frame),
        "static_max_error":float((relative_frame.native_interaction-relative_frame.independent_difference_slope).abs().max())}
    aggregation=OUT/"ppml/aggregation_checks.csv";identities["count_aggregation"]=clean(read(aggregation).to_dict("records"))
    for path in [relative_identity,aggregation]:
        sources[str(path.relative_to(ROOT))]=sha(path)
        downloads[str(path.relative_to(OUT))]=base64.b64encode(gzip.compress(path.read_bytes(),mtime=0)).decode()
    year_protocol=ROOT/"notes/visitor_origin_year_protocol.md";sources[str(year_protocol.relative_to(ROOT))]=sha(year_protocol)
    payload={"views":views,"counts":counts,"sources":sources,"downloads":downloads,"diagnostics":json.loads(diagnostics.read_text()),
        "identities":identities,
        "stata":{"proof":json.loads(stata_proof.read_text()),"tables":stata_tables},"labels":{**LABELS,**BRANCHES},
        "default":{"branch":"monthly","panel":"country","outcome":"y_mom_nsa","control_set":"full","model_spec":"own_current","model":"H1","method":"Calendar HAC · 12 months"}}
    encoded=json.dumps(clean(payload),ensure_ascii=False,separators=(',',':'),allow_nan=False)
    packed=base64.b64encode(gzip.compress(encoded.encode(),mtime=0)).decode()
    template=(ROOT/"templates/visitor_rigour.html").read_text()
    document=template.replace("__VISITOR_PAYLOAD__",packed).replace("__PROTOCOL__",markdown.markdown(protocol.read_text()+"\n\n"+year_protocol.read_text(),extensions=["tables","fenced_code"]))
    assert all(sha(ROOT/name)==digest for name,digest in sources.items()),"Report source changed during rendering"
    path=ROOT/"outputs/visitor_arrivals_rigour.html";path.write_text(document)
    inventory={"status":"built_from_independently_verified_results","views":len(views),"native_tables":sum(bool(v["table"]) for v in views),
        "counts":counts,"source_sha256":sources,"payload_sha256":hashlib.sha256(encoded.encode()).hexdigest(),"html_sha256":sha(path),
        "scope":"Source-backed reporting; numerical, presentation and actual browser checks remain separately identified"}
    (OUT/"report_inventory.json").write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+"\n")
    print(f"Built {path}: {len(views)} views",flush=True)
    return path


if __name__=="__main__":build_visitor_rigour_report()
