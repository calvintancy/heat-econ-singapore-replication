"""Qualified display of the preserved 90-fit sector-cluster experiment.

No fits or raw-output edits. Rank-one shared-nuisance CR1 numbers are algebraic
diagnostics, without sector-specific p-values, confidence intervals or stars.
"""
from __future__ import annotations
import base64
import gzip
import hashlib
import json
from pathlib import Path

import maketables as mt
import numpy as np
import pandas as pd

from heterogeneity_report import HeterogeneityResult
from regression_tables import namespace_table_ids
from significance import p_format, significance_stars

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/clustered_sector_sensitivity"
OUT = ROOT / "outputs/clustered_sector_sensitivity_qualified"
REPORT = ROOT / "outputs/clustered_sector_sensitivity_qualified.html"
CONTEXTS = {"va": "11 broad VA sectors", "va-detailed": "19 detailed VA sectors",
    "iip-monthly": "6 manufacturing clusters · monthly", "iip-quarterly": "6 manufacturing clusters · quarterly",
    "iip-monthly-detailed": "18 manufacturing industries · monthly", "iip-quarterly-detailed": "18 manufacturing industries · quarterly"}
WINDOWS = {"full": "Full controls · rainfall, humidity, wind + PM2.5", "exclude": "Full controls · excluding 2020–2022 endpoints", "long": "Long history · rainfall + humidity"}
OUTCOMES = {"y_short": "One-period log growth (SA)", "y_yoy_nsa": "YoY log growth (NSA)", "y_yoy": "YoY log growth (SA)",
    "y_log_level": "100 × log output level (SA; descriptive)", "y_raw_level": "Raw output level (SA; descriptive)"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qualified sector-cluster sensitivity</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f7fb;color:#18354e;font:14px/1.6 system-ui}header{background:#142e4b;color:white;padding:32px max(20px,calc((100vw - 1260px)/2))}header p{max-width:1100px;color:#dce8f5}h1{font-size:31px;line-height:1.2}h2{font-size:23px}main{max-width:1300px;padding:24px 20px;margin:auto}.panel{background:white;border:1px solid #d8e3eb;border-radius:12px;padding:22px;margin:0 0 20px;min-width:0}.warning{background:#fff1da;border-left:5px solid #bc8120;padding:15px 18px;margin:18px 0}.selectors{display:grid;grid-template-columns:1fr 1.5fr 1.3fr .7fr;gap:15px}label{font-size:12px;font-weight:700;min-width:0}select{display:block;width:100%;max-width:100%;padding:9px;margin-top:5px;color:#18354e;background:white;border:1px solid #b9cbd9;border-radius:6px;font:inherit}.scroll{overflow:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #dce5ed;text-align:right;white-space:nowrap}th{background:#e9f1f8}th:first-child,td:first-child{text-align:left;white-space:normal;min-width:210px}.meta{display:flex;gap:12px 24px;flex-wrap:wrap;color:#567083;margin:14px 0}.number{font-size:29px;font-weight:750;color:#9a610d}.muted{font-size:12px;color:#587084}.button{font:inherit;font-size:12px;background:#eef5fb;border:1px solid #b8cbdc;padding:8px 12px;border-radius:6px;margin:4px 8px 5px 0;cursor:pointer;overflow-wrap:anywhere}a{color:#14679b}header a{color:#d7e9fc}code{overflow-wrap:anywhere}@media(max-width:800px){.selectors{grid-template-columns:1fr 1fr}}@media(max-width:480px){.selectors{grid-template-columns:1fr}main{padding:16px 12px}.panel{padding:15px}h1{font-size:27px}}
</style></head><body><header><p>Singapore heat & economy · preserved algebra, qualified inference</p><h1>Sector-cluster sensitivity: read the shared-nuisance number carefully</h1><p>This page qualifies the retained 90-fit H1 experiment. Each sector-specific temperature slope is supported by one sector cluster. The resulting rank-one CR1 covariance reflects shared nuisance estimation; it cannot supply sector-specific uncertainty or an informative equality-of-slopes test.</p><a href="heat_economy.html#overview">Open the complete results collection</a></header><main>
<div class="warning"><strong>Individual CR1 p-values, confidence intervals and stars are suppressed.</strong> Nonzero CR1 standard errors in the raw files do not establish supported sector inference. They remain inspectable algebraic quantities. The calendar-DK comparison is retained with explicitly nominal, approximate inference. Neither covariance convention resolves identification or guarantees finite-sample calibration.</div>
<section class="panel"><div class="selectors"><label>Sector definition<select id="qcs-context"></select></label><label>Controls / dates<select id="qcs-window"></select></label><label>Dependent variable<select id="qcs-outcome"></select></label><label>DK bandwidth<select id="qcs-band"></select></label></div><p id="qcs-description"></p><div class="meta" id="qcs-meta"></div><p class="muted">These are the original 11/19-unit VA definitions and 6/18-unit manufacturing cuts. Complete 12/20-component GVA models, including dwellings, remain separate in the main collection. No sample, coefficient, covariance or frozen reference is changed here.</p></section>
<section class="panel"><h2>One shared algebraic CR1 number</h2><div class="number" id="qcs-shared">Loading…</div><p id="qcs-ranks"></p><p class="muted">This reference-row CR1 SE is common to sector total effects up to numerical roundoff. It describes the shared-nuisance component only. It is reported once, without a t reference, sector confidence interval or significance test. Sector-difference uncertainty collapses, so this experiment supplies no CR1 heterogeneity test.</p></section>
<section class="panel"><h2>Retained calendar-DK regression table</h2><div id="qcs-native" class="scroll">Loading native table…</div><p class="muted">MakeTables ETable, used by PyFixest etable, displays saved native PyFixest point estimates. Parentheses contain calendar-DK SEs; stars use saved DK p-values. They are nominal pointwise labels. Levels remain descriptive and exact-window stationarity is unresolved.</p></section>
<section class="panel"><h2>Sector totals and nominal calendar inference</h2><p>Total βᵢ = βT + δᵢ. Calendar variance includes Var(βT) + Var(δᵢ) + 2Cov(βT,δᵢ). The reference δ is a normalization, without an estimated zero SE.</p><div class="scroll" id="qcs-readout"></div><h3>Calendar equality test</h3><div class="warning">Multi-restriction Wald p-values are nominal asymptotic diagnostics. Selected fixed-design checks elsewhere in the collection show severe over-rejection despite full covariance rank. That diagnostic does not estimate every equation’s size; a computable F statistic does not validate the reference distribution or establish heterogeneity.</div><div id="qcs-joint"></div></section>
<section class="panel"><h2>Preserved raw algebra and audit trail</h2><p>The original CSVs and original HTML are unchanged. Their historical CR1 p/CI fields are unsupported for individual sector inference and are not displayed as results here. Raw downloads retain those fields solely to preserve the experiment and permit algebraic inspection. The numerical audit establishes reproducible arithmetic, not inferential support.</p><div id="qcs-downloads"></div><p class="muted">All downloads are embedded and work offline. The original raw display is included as an archival HTML download, preserving its bytes; this qualified page supplies the current interpretation.</p></section>
</main><script type="application/octet-stream" data-encoding="gzip-base64" id="qcs-data">__PAYLOAD__</script><script>
async function inflateQualified(text){return new Response(new Blob([Uint8Array.from(atob(text),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text()}
async function initializeQualifiedSensitivity(){const packed=document.getElementById('qcs-data');const D=JSON.parse(await inflateQualified(packed.textContent.trim()));packed.textContent='';const E=id=>document.getElementById('qcs-'+id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),f=(x,d=3)=>x==null?'—':Number(x).toLocaleString('en',{minimumFractionDigits:d,maximumFractionDigits:d});
for(const[id,items,value]of[['context',D.contexts,'va-detailed'],['window',D.windows,'full'],['outcome',D.outcomes,'y_short']]){E(id).innerHTML=Object.entries(items).map(([k,v])=>'<option value="'+k+'">'+esc(v)+'</option>').join('');E(id).value=value}
function table(head,rows){return '<table><thead><tr>'+head.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function show(){const context=E('context').value,window=E('window').value,outcome=E('outcome').value,h=context.includes('monthly')?12:4,previous=Number(E('band').value);E('band').innerHTML=[h,2*h].map(b=>'<option value="'+b+'">'+b+(h===12?' months':' quarters')+'</option>').join('');E('band').value=String([h,2*h].includes(previous)?previous:h);const band=Number(E('band').value),key=context+'__'+window+'|'+outcome,spec=D.specs.find(r=>r.sample_id===context+'__'+window&&r.outcome===outcome),shared=D.shared[key];const rows=D.rows.filter(r=>r.sample_id===spec.sample_id&&r.outcome===outcome&&r.hac_lags===band);E('native').innerHTML=D.tables[key+'|'+band];E('shared').textContent=f(shared.algebraic_shared_se,6);E('ranks').textContent='Absolute total-effect covariance rank '+shared.absolute_rank+' / '+spec.n_sectors+'; sector-difference covariance rank '+shared.difference_rank+' / '+(spec.n_sectors-1)+'.';E('description').textContent=D.outcomes[outcome].replace('One-period',h===12?'MoM':'QoQ')+(outcome==='y_raw_level'?(context.startsWith('va')?' · S$million real VA':' · IIP index points'):'');E('meta').innerHTML=[spec.n_sectors+' sector units',spec.n_periods+' distinct weather dates','N='+spec.n_obs+'; K='+spec.n_parameters,spec.start+' – '+spec.end,'Nominal calendar df '+(spec.n_periods-1)].map(v=>'<span>'+esc(v)+'</span>').join('');E('readout').innerHTML=table(['Sector','βT','δᵢ','Total','Calendar SE','Calendar 95% CI','Nominal calendar p','5%?','Mean Y','SD Y'],rows.map(r=>[esc(r.sector),f(r.temp_main),r.reference_delta_normalization?'0 · normalization':f(r.interaction_delta),f(r.coefficient)+esc(r.stars),f(r.std_error),'['+f(r.ci_lower)+', '+f(r.ci_upper)+']',esc(r.p_display),r.p_value==null?'Unavailable':r.p_value<.05?'Yes':'No',f(r.mean_y),f(r.sd_y)]));const joint=D.joints.find(r=>r.sample_id===spec.sample_id&&r.outcome===outcome&&r.hac_lags===band);E('joint').textContent='Restrictions '+joint.restrictions+'; covariance rank '+joint.covariance_rank+'; nominal Wald p '+joint.p_display+'; '+(joint.status==='available'?'Computable (calibration not established)':joint.status)+'.'}
for(const id of ['context','window','outcome','band'])E(id).addEventListener('change',show);E('downloads').innerHTML=Object.keys(D.downloads).map(name=>'<button class="button" data-raw="'+esc(name)+'">Raw diagnostic · '+esc(name)+'</button>').join('');E('downloads').onclick=async e=>{const name=e.target.dataset.raw;if(!name)return;const text=await inflateQualified(D.downloads[name]);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};show();window.qualifiedSensitivityData=D;window.showQualifiedSensitivity=show}
window.qualifiedSensitivityReady=initializeQualifiedSensitivity();window.qualifiedSensitivityReady.catch(e=>{document.getElementById('qcs-native').textContent='Report unavailable: '+e.message;throw e});
</script></body></html>'''


def build_qualified_clustered_sector_sensitivity():
    OUT.mkdir(parents=True, exist_ok=True)
    source_names = ["specifications.csv", "sector_slopes.csv", "slope_covariances.csv", "joint_tests.csv", "interaction_components.csv", "component_covariances.csv", "verification.json"]
    protected = {str((RAW / name).relative_to(ROOT)): sha(RAW / name) for name in source_names}
    for path in [ROOT / "outputs/clustered_sector_sensitivity.html", RAW / "pyfixest/fit_log.csv", RAW / "pyfixest/estimator_manifest.json"]:
        protected[str(path.relative_to(ROOT))] = sha(path)
    specs = pd.read_csv(RAW / "specifications.csv", dtype={"reference_sector_id": str}, float_precision="round_trip")
    slopes = pd.read_csv(RAW / "sector_slopes.csv", dtype={"sector_id": str, "reference_sector_id": str}, float_precision="round_trip")
    joints = pd.read_csv(RAW / "joint_tests.csv", float_precision="round_trip")
    assert len(specs) == 90 and specs.cluster_absolute_covariance_rank.eq(1).all()
    dk = slopes.loc[slopes.covariance.eq("Driscoll-Kraay")].copy()
    cr = slopes.loc[slopes.covariance.eq("Sector cluster")].copy()
    dk["p_display"] = dk.p_value.map(p_format);dk["stars"] = dk.p_value.map(significance_stars)
    shared, tables = {}, {}
    for spec in specs.to_dict("records"):
        mask = cr.sample_id.eq(spec["sample_id"]) & cr.outcome.eq(spec["outcome"])
        algebra = cr.loc[mask].sort_values("sector_id")
        reference = algebra.loc[algebra.sector_id.eq(spec["reference_sector_id"])].iloc[0]
        collapsed = joints.loc[joints.sample_id.eq(spec["sample_id"]) & joints.outcome.eq(spec["outcome"]) & joints.covariance.eq("Sector cluster")]
        assert len(collapsed) == 1 and int(collapsed.iloc[0].covariance_rank) == 0
        key = spec["sample_id"] + "|" + spec["outcome"]
        shared[key] = {"algebraic_shared_se": float(reference.std_error), "reference_sector_id": spec["reference_sector_id"],
            "absolute_rank": 1, "difference_rank": 0, "scope": "Shared nuisance only; no individual CR1 inference"}
        horizon = 12 if spec["frequency"] == "monthly" else 4
        for band in [horizon, 2*horizon]:
            selected = dk.loc[dk.sample_id.eq(spec["sample_id"]) & dk.outcome.eq(spec["outcome"]) & dk.hac_lags.eq(band)].copy()
            assert len(selected) == spec["n_sectors"]
            assert np.array_equal(selected.set_index("sector_id").coefficient.sort_index(), algebra.set_index("sector_id").coefficient.sort_index())
            labels = selected.set_index("sector_id").sector.to_dict()
            dependent = spec["outcome_label"].replace("One-period", "MoM" if horizon == 12 else "QoQ")
            result = HeterogeneityResult(selected, dependent, "H1")
            controls = ", ".join({"rain_100mm": "Rainfall", "rh_pct": "humidity", "wind_mps": "wind", "pm25_10ug": "PM2.5"}[term] for term in json.loads(spec["terms"]) if term != "temp_c")
            native = mt.ETable([result], model_heads=[f"Calendar DK · {band} lags"], head_order="dh", coef_fmt="b:.3f* \n (se:.3f)",
                signif_code=[.01, .05, .10], keep=list(labels), exact_match=True, labels=labels,
                felabels={"sector_season": "Sector × season FE", "year": "Common year FE"}, feorder=["sector_season", "year"],
                model_stats=["N", "sectors", "periods", "r2"], model_stats_labels={"N": "Observations", "sectors": "Sector units", "periods": "Weather periods", "r2": "R² including FE"},
                custom_model_stats={"Estimator": ["PyFixest feols " + str(spec["estimator_version"])], "Shared controls": [controls],
                    "Nominal calendar df": [int(spec["n_periods"])-1], "Estimated parameters": [int(spec["n_parameters"])],
                    "Mean observed Y": [f"{spec['mean_y_all']:.3f}"], "SD observed Y": [f"{spec['sd_y_all']:.3f}"]}, stats_order="sc",
                notes="Saved native PyFixest H1 totals betaT+delta_i. Only calendar-DK SEs/p-values/stars appear here; all inference is nominal and approximate. Individual CR1 inference is unsupported and suppressed. No new fit or stationarity clearance. * p<.10, ** p<.05, *** p<.01.")
            tables[key + "|" + str(band)] = namespace_table_ids(native.make(type="gt", gt_style={"table_width": "100%", "first_col_width": "300px", "table_font_size_all": "12px"}).as_raw_html(), "qualified-" + key.replace("|", "-") + "-" + str(band))
    display = ["sample_id", "outcome", "sector_id", "sector", "covariance", "hac_lags", "temp_main", "interaction_delta", "reference_delta_normalization",
               "coefficient", "std_error", "ci_lower", "ci_upper", "p_value", "p_display", "stars", "mean_y", "sd_y"]
    retained_joints = joints.loc[joints.covariance.eq("Driscoll-Kraay"), ["sample_id", "outcome", "hac_lags", "restrictions", "covariance_rank", "p_value", "status"]].copy()
    retained_joints["p_display"] = retained_joints.p_value.map(p_format)
    payload = clean({"specs": specs.to_dict("records"), "rows": dk[display].to_dict("records"), "shared": shared, "tables": tables,
        "joints": retained_joints.to_dict("records"), "contexts": CONTEXTS, "windows": WINDOWS, "outcomes": OUTCOMES,
        "downloads": {name: base64.b64encode(gzip.compress(path.read_bytes(), mtime=0)).decode() for name, path in
            {**{name: RAW / name for name in source_names}, "original_raw_display.html": ROOT / "outputs/clustered_sector_sensitivity.html"}.items()}})
    packed = base64.b64encode(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0)).decode()
    REPORT.write_text(TEMPLATE.replace("__PAYLOAD__", packed))
    assert all(sha(ROOT / path) == digest for path, digest in protected.items()), "Original algebraic evidence changed"
    inventory = {"status": "built_not_yet_presentation_verified", "html_sha256": sha(REPORT), "native_fits_run": 0,
        "source_hashes": protected, "builder_sha256": sha(Path(__file__)), "models": len(specs), "native_DK_tables": len(tables),
        "retained_calendar_rows": len(dk), "shared_CR1_numbers": len(shared), "suppressed_individual_CR1_inference": True,
        "scope": "Pure display qualification of immutable raw sensitivity results; calendar comparison retained, shared CR1 algebra reported once without sector p/CI/stars"}
    (OUT / "report_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Built qualified sensitivity: {len(specs)} models/{len(tables)} native DK tables; no fits", flush=True)
    return REPORT


if __name__ == "__main__": build_qualified_clustered_sector_sensitivity()
