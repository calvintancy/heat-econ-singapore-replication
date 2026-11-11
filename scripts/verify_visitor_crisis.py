"""Independent crisis-calendar and origin-cluster checks for visitor arrivals.

No production visitor functions are imported. The verifier reads raw counts,
constructs calendars separately, fits independent QR regressions and compares
origin-cluster covariance with representative complete stacked FE designs.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/visitor_arrivals"
CRISIS = OUT / "crisis"
DATA = ROOT / "data/processed/visitor_arrivals"
PANELS = ["aggregate", "region", "country"]
POLICIES = {
    "none": [],
    "covid": [("2020-01", "2022-12")],
    "gfc_covid": [("2008-01", "2009-12"), ("2020-01", "2022-12")],
    "buffered": [("2008-01", "2010-12"), ("2020-01", "2023-12")],
    "stable_pre2020": [("2008-01", "2010-12"), ("2020-01", "2099-12")],
}
ERRORS: dict[str, float] = {}


def check(name, actual, expected, atol=1e-8, rtol=3e-7):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    assert a.shape == b.shape, f"{name}: shape mismatch"
    assert np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), f"{name}: independent values differ"
    finite = np.isfinite(a) & np.isfinite(b)
    error = float(np.max(np.abs(a[finite]-b[finite])/np.maximum(1., np.abs(b[finite])))) if finite.any() else 0.
    ERRORS[name] = max(ERRORS.get(name, 0.), error)


def read_sources():
    raw = pd.read_csv(ROOT / "data/raw/visitor_arrivals/nsa.csv")
    raw.DataSeries = raw.DataSeries.str.strip()
    assert raw.DataSeries.is_unique
    values = raw.set_index("DataSeries").T.astype(float)
    values.index = pd.PeriodIndex(pd.to_datetime(values.index, format="%Y%b"), freq="M")
    values = values.sort_index()
    assert values.shape == (225, 62)
    assert values.index.equals(pd.period_range("2008-01", "2026-09", freq="M"))
    metadata = json.loads((ROOT / "data/raw/visitor_arrivals/nsa_singstat_metadata.json").read_text())["Data"]["records"]
    rows = pd.DataFrame(metadata["row"])
    names = {"aggregate": rows.loc[rows.seriesNo.eq("1"), "rowText"].tolist(),
             "region": rows.loc[rows.seriesNo.str.match(r"^1\.[1-9]$"), "rowText"].tolist(),
             "country": rows.loc[rows.seriesNo.str.match(r"^1\.[1-9]\.\d+$") & ~rows.seriesNo.str.endswith(".999"), "rowText"].tolist()}
    assert list(map(len, names.values())) == [1, 9, 42]
    levels, outcomes = {}, {}
    for panel in PANELS:
        levels[panel] = values[names[panel]]
        logs = np.log(levels[panel].where(levels[panel]>0))
        daily = logs.sub(np.log(logs.index.days_in_month), axis=0)
        outcomes[panel] = {"raw": levels[panel], "mom": 100*(logs-logs.shift(1)),
                           "yoy": 100*(logs-logs.shift(12)), "log_level": 100*logs,
                           "daily_mom": 100*(daily-daily.shift(1))}
    weather = pd.read_csv(DATA / "weather.csv", index_col="month")
    weather.index = pd.PeriodIndex(weather.index, freq="M")
    assert weather.index.equals(values.index)
    for name, col, target in [("temperature", "mean_temp", "temp_c"), ("humidity", "mean_rh", "rh_pct"), ("rainfall", "total_rainfall", "rain_mm")]:
        frame = pd.read_csv(ROOT / f"data/raw/{name}_monthly.csv", index_col="month")
        frame.index = pd.PeriodIndex(frame.index, freq="M")
        check("raw_weather_alignment", weather[target], frame[col].reindex(weather.index))
    check("rainfall_units", weather.rain_100mm, weather.rain_mm/100)
    extended = pd.read_csv(ROOT / "data/processed/extended_controls/monthly_controls.csv", index_col="month")
    extended.index = pd.PeriodIndex(extended.index, freq="M")
    for target in ["wind_mps", "pm25_10ug"]:
        check("extended_control_alignment", weather[target], extended[target].reindex(weather.index))
    assert weather.eligible_90.eq(True).equals(extended.eligible_90.reindex(weather.index).eq(True))
    return levels, outcomes, weather, names


def expected_mask(spec, outcomes, weather):
    outcome, sample = spec["outcome"], spec["sample"]
    valid = weather[spec["terms"]].notna().all(axis=1)
    for panel in PANELS:
        valid &= outcomes[panel][outcome].notna().all(axis=1)
    if sample == "full_all":
        valid &= weather.eligible_90.eq(True) & weather.index.year.isin(range(2017, 2026))
    else:
        assert sample == "long_all"
    policy = spec["variant"].removeprefix("crisis_")
    assert policy in POLICIES
    assert list(map(tuple, spec["exclusion_windows"])) == POLICIES[policy]
    bad = set()
    for start, end in POLICIES[policy]:
        bad.update(pd.period_range(start, end, freq="M").asi8.tolist())
    horizon = {"mom": 1, "yoy": 12}.get(outcome, 0)
    allowed = []
    for current in weather.index.asi8:
        interval = set(range(int(current-horizon), int(current)+1))
        allowed.append(interval.isdisjoint(bad))
    return valid & pd.Series(allowed, index=weather.index)


def expected_design(columns, frame):
    result = []
    for name in columns:
        if name == "Intercept":
            value = np.ones(len(frame))
        elif match := re.fullmatch(r"C\(season\)\[T\.(\d+)\]", name):
            value = (frame.index.month == int(match.group(1))).astype(float)
        elif match := re.fullmatch(r"C\(year\)\[T\.(\d+)\]", name):
            value = (frame.index.year == int(match.group(1))).astype(float)
        elif match := re.fullmatch(r"C\(season\)\[(\d+)\]:trend", name):
            value = (frame.index.month == int(match.group(1))) * frame.trend.to_numpy()
        else:
            value = frame[name].to_numpy()
        result.append(value)
    return np.column_stack(result)


def independent_fit(x, y):
    q, r = qr(x, mode="economic")
    assert np.linalg.matrix_rank(r) == len(r)
    beta = solve_triangular(r, q.T@y)
    residual = y-x@beta
    influence_design = solve_triangular(r, q.T).T
    return beta, residual, influence_design


def calendar_hac(influences, time, bandwidth, correction):
    influences = np.asarray(influences)
    if influences.ndim == 1:
        influences = influences[:, None]
    lookup = {int(value): i for i, value in enumerate(time)}
    covariance = influences.T@influences
    for lag in range(1, bandwidth+1):
        cross = np.zeros((influences.shape[1], influences.shape[1]))
        for value, i in lookup.items():
            j = lookup.get(value-lag)
            if j is not None:
                cross += np.outer(influences[i], influences[j])
        covariance += (1-lag/(bandwidth+1))*(cross+cross.T)
    return covariance*correction


def check_inference(name, row, beta, variance, df, scaled=False):
    se = np.sqrt(max(variance, 0.))
    critical = student_t.ppf(.975, df)
    p = 2*student_t.sf(abs(beta/se), df) if se>0 else 0.
    check(name, [row.coefficient, row.std_error, row.ci_lower, row.ci_upper, row.p_value],
          [beta, se, beta-critical*se, beta+critical*se, p], atol=2e-7)
    if scaled:
        check(name+"_temperature_scale", [row.effect_0p1c, row.lower_0p1c, row.upper_0p1c],
              [.1*beta, .1*(beta-critical*se), .1*(beta+critical*se)], atol=2e-7)


def verify_crisis(outcomes, weather):
    specs = json.loads((CRISIS / "specifications.json").read_text())
    results = pd.read_csv(CRISIS / "regression_results.csv")
    coefficients = pd.read_csv(CRISIS / "all_weather_coefficients.csv")
    saved_samples = pd.read_csv(CRISIS / "model_samples.csv")
    assert len(specs)==120 and results.model_id.nunique()==120
    assert len(results)==240
    boundaries = []
    for spec in specs:
        name=spec["model_id"]
        frame=weather.loc[expected_mask(spec, outcomes, weather)]
        months=frame.index
        z=np.load(CRISIS / "designs" / f"{name}.npz")
        assert list(z["months"])==list(months.astype(str))
        assert saved_samples.loc[saved_samples.model_id.eq(name), "month"].tolist()==list(months.astype(str))
        assert len(months)==spec["n_months"]
        x=expected_design(list(z["columns"]), frame)
        check("crisis_design", z["X"], x)
        units=outcomes[spec["panel"]][spec["outcome"]].loc[months]
        check("crisis_raw_outcomes", z["Y_units"], units.to_numpy(), atol=2e-7)
        y=units.mean(axis=1).to_numpy()
        check("crisis_equal_origin_mean", z["y"], y, atol=2e-7)
        beta, residual, influence_design=independent_fit(x, y)
        check("crisis_coefficients", z["beta"], beta, atol=2e-7)
        check("crisis_residuals", z["residual"], residual, atol=2e-7)
        t,k=x.shape
        assert spec["n_temporal_parameters"]==k and spec["df_time_resid"]==t-k
        check("crisis_mean_sd", [spec["mean_y"],spec["sd_y"]], [y.mean(),y.std(ddof=1)])
        positions=[list(z["columns"]).index(v) for v in spec["terms"]]
        influence=influence_design[:,positions]*residual[:,None]
        for bandwidth in [12,24]:
            covariance=calendar_hac(influence, months.asi8, bandwidth, t/(t-k))
            row=results.loc[results.model_id.eq(name)&results.hac_lags.eq(bandwidth)].iloc[0]
            check_inference("crisis_reported_temperature",row,beta[positions[0]],covariance[0,0],t-1,scaled=True)
            for j, term in enumerate(spec["terms"]):
                cr=coefficients.loc[coefficients.model_id.eq(name)&coefficients.hac_lags.eq(bandwidth)&coefficients.term.eq(term)].iloc[0]
                check_inference("crisis_all_weather_inference",cr,beta[positions[j]],covariance[j,j],t-1)
        if spec["panel"]=="aggregate" and spec["sample"]=="long_all" and spec["variant"]=="crisis_buffered" and spec["outcome"] in ["mom","yoy"]:
            first=months[0]
            post=months[months.year>=2020][0]
            expected=("2011-02","2024-02") if spec["outcome"]=="mom" else ("2012-01","2025-01")
            assert (str(first),str(post))==expected
            boundaries.append({"outcome":spec["outcome"],"first_after_GFC_buffer":str(first),"first_after_COVID_buffer":str(post)})
    comparisons=[]
    core=pd.read_csv(OUT / "regression_results.csv")
    for panel in PANELS:
        for outcome in ["mom","yoy","log_level","raw"]:
            for bw in [12,24]:
                a=results.loc[results.panel.eq(panel)&results.outcome.eq(outcome)&results["sample"].eq("full_all")&results.variant.eq("crisis_covid")&results.hac_lags.eq(bw)].iloc[0]
                b=results.loc[results.panel.eq(panel)&results.outcome.eq(outcome)&results["sample"].eq("full_all")&results.variant.eq("crisis_gfc_covid")&results.hac_lags.eq(bw)].iloc[0]
                check("GFC_outside_full_control_window_identity",a[["coefficient","std_error","n_months"]],b[["coefficient","std_error","n_months"]],atol=0,rtol=0)
            for sample in ["long_all","full_all"]:
                for bw in [12,24]:
                    a=results.loc[results.panel.eq(panel)&results.outcome.eq(outcome)&results["sample"].eq(sample)&results.variant.eq("crisis_none")&results.hac_lags.eq(bw)].iloc[0]
                    b=core.loc[core.panel.eq(panel)&core.outcome.eq(outcome)&core["sample"].eq(sample)&core.variant.eq("main")&core.hac_lags.eq(bw)].iloc[0]
                    check("no_exclusion_reproduces_core",a[["coefficient","std_error","n_months"]],b[["coefficient","std_error","n_months"]],atol=2e-7)
    full_pre=results.loc[results.panel.eq("aggregate")&results.outcome.eq("mom")&results["sample"].eq("full_all")&results.variant.eq("crisis_stable_pre2020")&results.hac_lags.eq(12)].iloc[0]
    assert full_pre.n_months==32 and full_pre.df_time_resid==13
    return specs, boundaries


def verify_origin_clusters(outcomes, weather, names):
    result=pd.read_csv(OUT / "origin_cluster/regression_results.csv")
    coefficients=pd.read_csv(OUT / "origin_cluster/all_weather_coefficients.csv")
    expected_ids=set()
    explicit=[]
    representative={
        "region__mom__long_pre2020__main",
        "country__mom__full_clean__main",
        "country__mom__long_clean__fixed_weights",
        "region__mom__long_all__crisis_buffered",
        "country__yoy__full_all__crisis_buffered",
    }
    counts={"core":0,"crisis":0}
    for source,folder in [("core",OUT),("crisis",CRISIS)]:
        specs=json.loads((folder / "specifications.json").read_text())
        for spec in specs:
            if spec["panel"] not in ["region","country"]:
                continue
            name=spec["model_id"];expected_ids.add(name);counts[source]+=1
            z=np.load(folder / "designs" / f"{name}.npz")
            months=pd.PeriodIndex(z["months"],freq="M")
            frame=weather.loc[months]
            x=expected_design(list(z["columns"]),frame)
            check("cluster_saved_design",z["X"],x)
            units=outcomes[spec["panel"]][spec["outcome"]].loc[months].to_numpy()
            check("cluster_raw_origin_outcomes",z["Y_units"],units,atol=2e-7)
            t,g=units.shape
            if spec["weighted"]:
                old=outcomes[spec["panel"]]["raw"].loc["2017-01":"2019-12"].sum()
                weights=(old/old.sum()).to_numpy()
            else:
                weights=np.ones(g)/g
            y=units@weights
            check("cluster_weighted_mean",z["y"],y,atol=2e-7)
            beta,residual,influence_design=independent_fit(x,y)
            deviations=units-y[:,None]
            residual_units=deviations.copy()
            for season in set(months.month):
                in_season=months.month==season
                residual_units[in_season]-=deviations[in_season].mean(axis=0)
            residual_units+=residual[:,None]
            positions=[list(z["columns"]).index(v) for v in spec["terms"]]
            group_scores=np.stack([(influence_design[:,positions]*residual_units[:,i,None]).sum(axis=0)*weights[i] for i in range(g)])
            n=t*g;k_full=x.shape[1]+(g-1)*len(set(months.month))
            correction=g/(g-1)*(n-1)/(n-k_full)
            covariance=group_scores.T@group_scores*correction
            saved=np.load(OUT / "origin_cluster/designs" / f"{name}.npz")
            check("cluster_residuals",saved["residual_units"],residual_units,atol=2e-7)
            check("cluster_weights",saved["weights"],weights)
            check("cluster_scores",saved["cluster_scores"],group_scores,atol=2e-7)
            check("cluster_covariance",saved["covariance"],covariance,atol=2e-6)
            rr=result.loc[result.model_id.eq(name)].iloc[0]
            assert rr.source==source and rr.n_clusters==g and rr.inference_df==g-1
            assert rr.full_fe_parameters==k_full and rr.n_panel_obs==n and rr.n_months==t
            check("cluster_finite_sample_correction",rr.finite_sample_correction,correction)
            ti=spec["terms"].index(spec["temperature_term"])
            check_inference("cluster_temperature_inference",rr,beta[positions[ti]],covariance[ti,ti],g-1,scaled=True)
            for j,term in enumerate(spec["terms"]):
                cr=coefficients.loc[coefficients.model_id.eq(name)&coefficients.term.eq(term)].iloc[0]
                check_inference("cluster_all_weather_inference",cr,beta[positions[j]],covariance[j,j],g-1)
            if name in representative:
                full=np.tile(x,(g,1))
                seasons=np.array(sorted(set(months.month)))
                dummies=(months.month.to_numpy()[:,None]==seasons).astype(float)
                extras=np.zeros((n,(g-1)*len(seasons)))
                for i in range(1,g):
                    extras[i*t:(i+1)*t,(i-1)*len(seasons):i*len(seasons)]=dummies
                full=np.column_stack([full,extras])
                wrows=np.repeat(np.sqrt(weights),t)
                full_y=units.T.ravel()
                full_beta,full_residual,full_influence=independent_fit(full*wrows[:,None],full_y*wrows)
                fs=np.stack([(full_influence[i*t:(i+1)*t,positions]*full_residual[i*t:(i+1)*t,None]).sum(axis=0) for i in range(g)])
                fc=fs.T@fs*correction
                # Unit-specific seasonal intercepts use a different normalization;
                # only the common weather slopes must equal the collapsed fit.
                check("explicit_stacked_FE_coefficients",full_beta[positions],beta[positions],atol=2e-7)
                check("explicit_stacked_CR1_covariance",fc,covariance,atol=2e-6)
                explicit.append({"model_id":name,"n_obs":n,"n_fe_and_weather_parameters":k_full,
                    "temperature_coefficient":float(full_beta[positions[ti]]),"origin_CR1_se":float(np.sqrt(fc[ti,ti])),
                    "reference_df":g-1,"weighted":spec["weighted"]})
    assert set(result.model_id)==expected_ids and len(result)==len(expected_ids)==254
    assert not any("profiles__" in name for name in result.model_id)
    assert len(explicit)==len(representative)
    return counts, explicit


def verify_profile_weather(outcomes, weather):
    rows=pd.read_csv(OUT / "gt/profile_all_weather.csv")
    assert len(rows)==1632 and not rows.duplicated(["group","hac_lags","unit","term"]).any()
    old=pd.read_csv(OUT / "market_slopes.csv")
    systems=0
    for (group,bandwidth),view in rows.groupby(["group","hac_lags"]):
        panel,outcome,sample=group.split("__")
        assert panel in ["region","country"] and outcome in ["mom","yoy"]
        z=np.load(OUT / "designs" / f"profiles__{group}__{int(bandwidth)}.npz")
        months=pd.PeriodIndex([pd.Period(ordinal=int(t),freq="M") for t in z["time"]])
        x=expected_design(list(z["columns"]),weather.loc[months])
        y=outcomes[panel][outcome].loc[months]
        assert list(y.columns)==list(z["units"])
        check("profile_all_weather_design",z["X"],x)
        check("profile_all_weather_outcomes",z["Y"],y.to_numpy())
        beta,residual,a=independent_fit(x,y.to_numpy())
        check("profile_all_weather_coefficients",z["beta"],beta)
        check("profile_all_weather_residuals",z["residual"],residual)
        terms=[v for v in ["temp_c","rain_100mm","rh_pct","wind_mps","pm25_10ug"] if v in z["columns"]]
        assert len(view)==len(y.columns)*len(terms)
        t,k=x.shape;lookup={int(value):i for i,value in enumerate(months.asi8)}
        for term in terms:
            j=list(z["columns"]).index(term)
            psi=a[:,j,None]*residual
            variance=np.square(psi).sum(axis=0)
            for lag in range(1,int(bandwidth)+1):
                cross=np.zeros(y.shape[1])
                for value,i in lookup.items():
                    previous=lookup.get(value-lag)
                    if previous is not None:
                        cross+=psi[i]*psi[previous]
                variance+=2*(1-lag/(int(bandwidth)+1))*cross
            variance*=t/(t-k)
            for unit_index,unit in enumerate(y.columns):
                row=view.loc[view.unit.eq(unit)&view.term.eq(term)].iloc[0]
                check_inference("profile_all_weather_inference",row,beta[j,unit_index],variance[unit_index],t-1)
                if term=="temp_c":
                    prior=old.loc[old.group.eq(group)&old.hac_lags.eq(bandwidth)&old.unit.eq(unit)].iloc[0]
                    check("profile_temperature_matches_existing",row[["coefficient","std_error","p_value"]],prior[["coefficient","std_error","p_value"]])
        systems+=1
    assert systems==16
    return {"rows":len(rows),"profile_bandwidth_systems":systems}


def verify_gt_bindings():
    jobs=json.loads((OUT / "gt/input.json").read_text())
    assert len(jobs)==168 and len(set(job["key"] for job in jobs))==168
    specs={}
    for source,folder in [("core",OUT),("crisis",CRISIS)]:
        specs[source]={s["model_id"]:s for s in json.loads((folder / "specifications.json").read_text())}
    tables={"core":pd.read_csv(OUT / "all_weather_coefficients.csv"),
            "crisis":pd.read_csv(CRISIS / "all_weather_coefficients.csv"),
            "origin":pd.read_csv(OUT / "origin_cluster/all_weather_coefficients.csv")}
    profiles=pd.read_csv(OUT / "gt/profile_all_weather.csv")
    def number(value):
        return f"{float(value):,.3f}"
    def pvalue(value):
        return "&lt;0.001" if value<.001 else number(value)
    def cell(row):
        return f"{number(row['coefficient'])}<br>({number(row['std_error'])})<br>p={pvalue(row['p_value'])}"
    bindings=0;nonprofile_tables=0;profile_tables=0
    for job in jobs:
        if job["source"]=="profiles":
            group=job["model_ids"][0];bandwidth=int(job["inference"].removeprefix("hac"))
            frame=profiles.loc[profiles.group.eq(group)&profiles.hac_lags.eq(bandwidth)]
            assert set(row["term"] for row in job["rows"])==set(frame.unit)
            terms=set(frame.term)
            assert set(job["labels"])-{"holm","scaled_ci","mean_y","sd_y"}==terms
            z=np.load(OUT / "designs" / f"profiles__{group}__{bandwidth}.npz")
            for row in job["rows"]:
                unit_index=list(z["units"]).index(row["term"])
                assert row["mean_y"]==number(z["Y"][:,unit_index].mean())
                assert row["sd_y"]==number(z["Y"][:,unit_index].std(ddof=1))
                for term in terms:
                    value=frame.loc[frame.unit.eq(row["term"])&frame.term.eq(term)].iloc[0]
                    assert row[term]==cell(value),f"Profile GT cell differs: {job['key']} {row['term']} {term}"
            profile_tables+=1
            continue
        source,method=job["source"],job["inference"]
        assert source in ["core","crisis"] and method in ["hac12","hac24","origin"]
        expected=set()
        for i,model_id in enumerate(job["model_ids"]):
            assert model_id in specs[source]
            spec=specs[source][model_id]
            if method=="origin":
                assert spec["panel"] in ["region","country"]
                frame=tables["origin"].loc[tables["origin"].source.eq(source)&tables["origin"].model_id.eq(model_id)]
            else:
                frame=tables[source].loc[tables[source].model_id.eq(model_id)&tables[source].hac_lags.eq(int(method[3:]))]
            column=f"m{i+1}"
            trend_row=next(row for row in job["rows"] if row["term"]=="Month-specific linear trends")
            assert trend_row[column]==("Yes" if spec["seasonal_trends"] else "No")
            for term in spec["terms"]:
                expected.add((column,term))
                selected=frame.loc[frame.term.eq(term)]
                assert len(selected)==1
                value=selected.iloc[0]
                matches=[b for b in job["bindings"] if b["column"]==column and b["term"]==term]
                assert len(matches)==1
                binding=matches[0]
                check("GT_source_bindings",[binding[v] for v in ["coefficient","std_error","p_value"]],value[["coefficient","std_error","p_value"]],atol=1e-12,rtol=1e-12)
                assert any(row.get(column)==cell(value) for row in job["rows"]),f"GT rendered cell missing: {job['key']} {column} {term}"
                bindings+=1
        actual={(b["column"],b["term"]) for b in job["bindings"]}
        assert actual==expected and len(actual)==len(job["bindings"]),f"GT omits/duplicates a regressor: {job['key']}"
        assert "FE" in job["note"] and "nominal" in job["note"]
        if method=="origin":
            assert "t(G−1)" in job["note"] and "independent origins" in job["note"]
        else:
            assert f"{method[3:]} months" in job["note"] and "t(T−1)" in job["note"]
        nonprofile_tables+=1
    assert (nonprofile_tables,profile_tables)==(152,16)
    return {"tables":len(jobs),"nonprofile_tables":nonprofile_tables,"profile_tables":profile_tables,
        "nonprofile_source_bindings":bindings,"all_non_FE_regressors_included":True,
        "month_trend_indicators_checked":True,"profile_mean_SD_checked":True,
        "input_json_sha256":hashlib.sha256((OUT / "gt/input.json").read_bytes()).hexdigest()}


def run():
    manifest=json.loads((CRISIS / "manifest.json").read_text())
    core_results=OUT / "regression_results.csv"
    assert hashlib.sha256(core_results.read_bytes()).hexdigest()==manifest["source_results_sha256"], "Original core results changed"
    assert pd.read_csv(core_results).model_id.nunique()==269
    original_hashes=json.loads((OUT / "input_hashes.json").read_text())
    input_count=0
    for name,expected in original_hashes.items():
        if name=="outputs/heat_economy.html":
            continue
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, f"Analytical input changed: {name}"
        input_count+=1
    with threadpool_limits(limits=1):
        levels,outcomes,weather,names=read_sources()
        specs,boundaries=verify_crisis(outcomes,weather)
        cluster_counts,explicit=verify_origin_clusters(outcomes,weather,names)
        profile_weather=verify_profile_weather(outcomes,weather)
        gt_bindings=verify_gt_bindings()
    receipt={"status":"passed","verified_utc":datetime.now(timezone.utc).isoformat(),
        "implementation":"Independent raw-count outcomes, month-set interval masks, reconstructed designs, SciPy QR and explicit calendar Bartlett sums. No production visitor function imported.",
        "counts":{"crisis_OLS_models":len(specs),"crisis_model_bandwidths":240,"core_models_preserved":269,
            "origin_CR1_models":254,"origin_CR1_by_source":cluster_counts,"analytical_input_hashes":input_count,
            "explicit_stacked_FE_CR1_checks":len(explicit)},
        "max_scaled_errors":ERRORS,"buffered_growth_boundaries":boundaries,
        "profile_all_weather_reporting":profile_weather,"GT_input_bindings":gt_bindings,
        "short_full_control_preCOVID":{"months":32,"residual_time_df":13,"status":"fragile short-sample inference"},
        "explicit_stacked_FE_checks":explicit,
        "core_results_sha256":hashlib.sha256(core_results.read_bytes()).hexdigest(),
        "main_economic_report_is_input":False,
        "limits":["Calendar exclusions and recovery buffers change the retained-period estimand, not causal identification.",
            "GFC dates lie outside the full-control window: identical COVID and GFC+COVID columns there are expected.",
            "Origin CR1 assumes independent origin clusters and permits within-origin temporal dependence; it misses cross-origin common-shock dependence.",
            "Region CR1 uses only nine clusters and approximate t(8); country CR1 uses 42 clusters and t(41). Calendar HAC remains primary.",
            "Origin-specific profiles have no origin-cluster inference; their fully interacted coefficients cannot obtain meaningful one-origin-cluster standard errors."]}
    (CRISIS / "verification.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":receipt["status"],"counts":receipt["counts"],"max_scaled_error":max(ERRORS.values())},indent=2))


if __name__=="__main__":
    run()
