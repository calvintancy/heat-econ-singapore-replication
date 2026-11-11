"""Prepare full regression tables from verified estimates for native R gt."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from visitor_analysis import OUT, influence_hac, inference, save_json
from visitor_crisis import POLICIES

PANELS = {"aggregate": "Official aggregate", "region": "Nine-region FE", "country": "42-origin-market FE"}
OUTCOMES = {"mom": "MoM log growth", "yoy": "YoY log growth", "log_level": "100 × log arrivals (descriptive)", "raw": "Raw arrivals (descriptive)"}
SAMPLES = ["long_all", "long_pre2020", "long_clean", "full_all", "full_clean"]
SAMPLE_LABELS = {"long_all": "Long controls<br>All admissible dates", "long_pre2020": "Long controls<br>Before 2020", "long_clean": "Long controls<br>Strict COVID exclusion", "full_all": "Full controls<br>All admissible dates", "full_clean": "Full controls<br>Strict COVID exclusion"}
TERMS = {"temp_c": "Temperature (°C)", "rain_100mm": "Rainfall (100 mm)", "rh_pct": "Relative humidity (pp)", "wind_mps": "Wind (m/s)", "pm25_10ug": "PM2.5 (10 μg/m³)", "temp_sq": "(Temperature − 28°C)²", "temp_lead1": "Next-month temperature", "lny_window_days": "Lunar New Year window days", "lny_lag1": "Prior-month LNY window days"}
VARIANTS = {"main": "Current weather", "lunar": "LNY current + lag", "daily": "Arrivals/day growth", "season_trends": "Changing seasonality", "matched_base_controls": "Base controls<br>Matched dates", "fixed_weights": "Fixed arrival weights", "coverage80": "80% coverage", "weather_differences": "Weather differences", "lag_matched_current": "Current only<br>Matched lag dates", "lags": "Current + 3 lags", "lead": "Current + lead", "quadratic_at28": "Quadratic temperature", "official_sa": "Official SA aggregate"}


def label(term):
    if term in TERMS:
        return TERMS[term]
    for base in ["temp_c", "rain_100mm", "rh_pct", "wind_mps", "pm25_10ug"]:
        if term.startswith(base+"_diff"):
            return f"Δ{term.rsplit('diff',1)[1]} " + TERMS[base]
        if term.startswith(base+"_lag"):
            return TERMS[base] + " · lag " + term.rsplit("lag", 1)[1]
    return term


def number(value, digits=3):
    return f"{value:,.{digits}f}"


def pvalue(value):
    return "&lt;0.001" if value < .001 else number(value)


def cell(row):
    return f"{number(row['coefficient'])}<br>({number(row['std_error'])})<br>p={pvalue(row['p_value'])}"


def run():
    folder = OUT / "gt"
    folder.mkdir(exist_ok=True)
    datasets = {name: (pd.read_csv(path / "regression_results.csv"), pd.read_csv(path / "all_weather_coefficients.csv"),
                         {s["model_id"]: s for s in json.loads((path / "specifications.json").read_text())})
                for name, path in [("core", OUT), ("crisis", OUT / "crisis")]}
    cluster = pd.read_csv(OUT / "origin_cluster/regression_results.csv")
    cluster_c = pd.read_csv(OUT / "origin_cluster/all_weather_coefficients.csv")
    jobs = []

    def regression_job(key, model_ids, labels, method, source, title):
        rr, cc, specs = datasets[source]
        if method == "origin":
            frame = cluster[cluster.source.eq(source)]
            coefs = cluster_c[cluster_c.source.eq(source)]
            note = "Origin-clustered CR1; t(G−1). Allows arbitrary serial dependence within an origin, assumes independent origins, and misses common cross-origin shocks. Comparison only; calendar HAC remains primary."
        else:
            bandwidth = int(method.replace("hac", ""))
            frame = rr[rr.hac_lags.eq(bandwidth)]
            coefs = cc[cc.hac_lags.eq(bandwidth)]
            note = f"Calendar-aware Bartlett HAC, {bandwidth} months; T/(T−k) correction; approximate t(T−1). Allows common origin shocks and serial dependence; real calendar gaps are preserved."
        chosen = [{**rr.loc[rr.model_id.eq(mid)].iloc[0].to_dict(),
                   **frame.loc[frame.model_id.eq(mid)].iloc[0].to_dict()} for mid in model_ids]
        columns = {f"m{i+1}": title for i, title in enumerate(labels)}
        terms = list(dict.fromkeys(term for mid in model_ids for term in specs[mid]["terms"]))
        rows=[]
        bindings=[]
        for term in terms:
            row={"term": label(term)}
            for i, mid in enumerate(model_ids):
                found = coefs[coefs.model_id.eq(mid) & coefs.term.eq(term)]
                row[f"m{i+1}"] = cell(found.iloc[0]) if len(found) else "—"
                if len(found):
                    bindings.append({"column": f"m{i+1}", "term": term,
                        **{k: float(found.iloc[0][k]) for k in ["coefficient", "std_error", "p_value"]}})
            rows.append(row)
        metadata = [("Origin × calendar-month FE", ["Yes" if c["panel"]!="aggregate" else "Calendar-month FE" for c in chosen]),
            ("Common year FE", ["Yes"]*len(chosen)),
            ("Month-specific linear trends", ["Yes" if specs[mid]["seasonal_trends"] else "No" for mid in model_ids]),
            ("Distinct calendar months T", [number(c["n_months"],0) for c in chosen]),
            ("Origins G", [number(c["n_units"],0) for c in chosen]),
            ("Origin-month observations N", [number(c["n_panel_obs"],0) for c in chosen]),
            ("Residual time degrees of freedom", [number(c["df_time_resid"],0) for c in chosen]),
            ("Inference reference df", [number(c["inference_df"] if method=="origin" else c["n_months"]-1,0) for c in chosen]),
            ("R² of mean-outcome model", [number(c["r_squared"]) for c in chosen]),
            ("Mean outcome", [number(c["mean_y"]) for c in chosen]),
            ("SD of mean outcome across months", [number(c["sd_y"]) for c in chosen]),
            ("First / last month", [c["start"]+"<br>"+c["end"] for c in chosen])]
        for name, values in metadata:
            rows.append({"term":name, **{f"m{i+1}": value for i,value in enumerate(values)}})
        note += " Coefficients per stated regressor unit; robust SE in parentheses and nominal p-values underneath. All weather/holiday regressors shown; deterministic FE and time-trend coefficients are represented by indicators. P-values are not adjusted across model choices."
        if any(c["df_time_resid"]<20 for c in chosen):
            note += " At least one column has fewer than 20 residual time df: inference is fragile."
        jobs.append({"key":key,"title":title,"subtitle":method.upper()+" · full regression table", "labels":columns,
            "rows":rows,"note":note,"model_ids":model_ids,"inference":method,"source":source,"bindings":bindings})

    for panel in PANELS:
        methods=["hac12","hac24"]+(["origin"] if panel!="aggregate" else [])
        for outcome in OUTCOMES:
            ids=[f"{panel}__{outcome}__{sample}__main" for sample in SAMPLES]
            for method in methods:
                regression_job(f"core__{panel}__{outcome}__{method}",ids,[SAMPLE_LABELS[s] for s in SAMPLES],method,"core",PANELS[panel]+" · "+OUTCOMES[outcome])
            for sample in ["long_all","full_all"]:
                ids=[f"{panel}__{outcome}__{sample}__crisis_{policy}" for policy in POLICIES]
                for method in methods:
                    regression_job(f"crisis__{panel}__{outcome}__{sample}__{method}",ids,
                        [POLICIES[p]["label"] for p in POLICIES],method,"crisis",PANELS[panel]+" · crisis/recovery exclusions · "+OUTCOMES[outcome])
        rr=datasets["core"][0]
        for sample in ["long_pre2020","long_clean","full_clean"]:
            selected=rr[(rr.panel.eq(panel))&rr["sample"].eq(sample)&rr.hac_lags.eq(12)&rr.outcome.isin(["mom","daily_mom"])&~rr.variant.str.startswith("loo_")&~rr.variant.eq("own_positive_dates")]
            if sample=="full_clean":
                selected=pd.concat([selected,rr[rr.panel.eq(panel)&rr["sample"].eq("full80_clean")&rr.variant.eq("coverage80")&rr.outcome.eq("mom")&rr.hac_lags.eq(12)]])
            if panel=="aggregate":
                selected=pd.concat([selected,rr[rr.panel.eq("aggregate_sa")&rr["sample"].eq(sample)&rr.variant.eq("official_sa")&rr.outcome.eq("mom")&rr.hac_lags.eq(12)]])
            selected=selected.sort_values("variant")
            for method in methods:
                current=selected if method!="origin" else selected[selected.panel.ne("aggregate_sa")]
                # Several readable tables instead of one extremely wide table.
                for start in range(0,len(current),5):
                    chunk=current.iloc[start:start+5]
                    regression_job(f"robust__{panel}__{sample}__{method}__{start//5}",chunk.model_id.tolist(),
                        [VARIANTS[v] for v in chunk.variant],method,"core",PANELS[panel]+" · sensitivity regressions · "+SAMPLE_LABELS[sample].replace("<br>"," · "))
    profile_rows=[]
    slopes=pd.read_csv(OUT / "market_slopes.csv")
    for (group,bandwidth), view in slopes.groupby(["group","hac_lags"],sort=False):
        z=np.load(OUT / "designs" / f"profiles__{group}__{bandwidth}.npz")
        x,y,res=z["X"],z["Y"],z["residual"]
        q,r=np.linalg.qr(x,mode="reduced")
        a=np.linalg.solve(r,q.T).T
        positions=[j for j,term in enumerate(z["columns"]) if term in TERMS and term not in ["temp_sq","temp_lead1","lny_window_days","lny_lag1"]]
        terms=[str(z["columns"][j]) for j in positions]
        rows=[]
        for u,unit in enumerate(z["units"]):
            variance=influence_hac(a[:,positions]*res[:,u,None],z["time"],int(bandwidth),float(z["correction"]))
            row={"term":str(unit)}
            for j,term in enumerate(terms):
                value=inference(z["beta"][positions[j],u],variance[j,j],len(x)-1)
                row[term]=cell(value)
                profile_rows.append({"group":group,"hac_lags":int(bandwidth),"unit":str(unit),"term":term,
                    **{k:value[k] for k in ["coefficient","std_error","ci_lower","ci_upper","p_value"]}})
            sr=view.loc[view.unit.eq(str(unit))].iloc[0]
            row["holm"] = pvalue(sr.holm_p_value)
            row["scaled_ci"] = f"{number(sr.effect_0p1c)}<br>[{number(sr.lower_0p1c)}, {number(sr.upper_0p1c)}]"
            row["mean_y"] = number(sr.mean_y)
            row["sd_y"] = number(sr.sd_y)
            rows.append(row)
        jobs.append({"key":f"profile__{group}__hac{bandwidth}","title":"Fully interacted origin regressions · "+group.replace("__"," · "),
            "subtitle":f"Calendar HAC {bandwidth} · {len(x)} months per origin · {len(x)-x.shape[1]} residual time df",
            "labels":{**{term:label(term) for term in terms},"scaled_ci":"Temperature +0.1°C<br>Effect [95% interval]","holm":"Temperature<br>Holm p","mean_y":"Mean growth","sd_y":"SD growth"},
            "rows":rows,"note":"Each row is one origin's complete weather regression with its own seasonal and year FE. Cells contain coefficient (SE) and nominal p. Temperature Holm p adjusts the nine or 42 temperature tests in this exact panel/outcome/sample/bandwidth table, not all model choices. Origin clustering is unavailable for a slope supported by one origin cluster.",
            "source":"profiles","inference":f"hac{bandwidth}","model_ids":[group]})
    pd.DataFrame(profile_rows).to_csv(folder / "profile_all_weather.csv",index=False)
    save_json(folder / "input.json", jobs)
    print(f"Prepared {len(jobs)} native gt tables",flush=True)


if __name__ == "__main__":
    run()
