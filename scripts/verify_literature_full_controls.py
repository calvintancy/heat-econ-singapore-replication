"""Independent raw-calendar and numerical audit of full-controls sensitivities.

The audit uses raw official output/weather, an independent raw wind/PM parser,
QR estimates and calendar-keyed Bartlett/cluster influence scores. Production
preparation, coefficient and covariance functions are never called. Unestimated
requests are audited as complete retained designs, not silently omitted models.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_literature_sensitivity as prior
from verify_granular import _raw_extended

independent = prior.independent
OUT = ROOT / "outputs/literature_sensitivity/full_controls"
ADDED = ["wind_mps", "pm25_10ug"]
HISTORY = "full_controls_max_AR_history"
RANK_CACHE: OrderedDict = OrderedDict()


def read(path):
    return prior.read(path)


def assert_lineage(manifest):
    protected = {**manifest["inputs_sha256"], **manifest["protected_sources_sha256"],
                 manifest["producer_source"]: manifest["producer_source_sha256"]}
    for relative, digest in protected.items():
        assert independent.sha256(ROOT / relative) == digest, ("Frozen input/source changed", relative)
    span = json.loads((OUT/"two_channel_span_manifest.json").read_text())
    extra = {span["source"]:span["source_sha256"], span["output"]:span["output_sha256"],
             **span["frozen_inputs_sha256"], **span["design_archives_sha256"]}
    for relative,digest in extra.items():
        assert independent.sha256(ROOT/relative) == digest, ("Span supplement lineage",relative)
    protected.update(extra)
    return protected


def build_samples(policy):
    """Rebuild raw support, current coverage, target alignment and crisis masks."""
    raw_parents, source_audit = independent.audit_raw_parents()
    growth, weather, raw_hashes = prior.raw_histories()
    added, coverage_audit = _raw_extended()
    # The canonical quarterly coverage columns report the worst constituent
    # month, while eligibility requires all three months to pass separately.
    for date, values in added["Q"].items():
        months = [date.asfreq("M", how="start") + lag for lag in range(3)]
        for field in ("wind_coverage", "pm_coverage"):
            values[field] = min(added["M"][m][field] for m in months if m in added["M"])
    features, own_raw, own_full, details = {}, {}, {}, {}
    for context in independent.CONTEXTS:
        freq = "M" if "monthly" in context else "Q"
        period, h = ("month", 12) if freq == "M" else ("quarter", 4)
        family = "iip" if context.startswith("iip") else "va"
        path = ROOT / f"outputs/crisis_sensitivity/parents/{context}.csv"
        parent = read(path).sort_values(["time", "sector_id"]).reset_index(drop=True)
        independent.compare_sample(parent, raw_parents[context], period)
        dates = pd.PeriodIndex(parent[period], freq=freq)
        for outcome in prior.YS:
            frame = parent.copy()
            for lag in range(h + 1):
                for field, name in (("temp_c", "temp"), ("rain_100mm", "rain")):
                    frame[f"{name}_l{lag}"] = [weather[freq].get(date-lag, {}).get(field, np.nan) for date in dates]
            for lag in range(1, h + 1):
                frame[f"ar_l{lag}"] = [growth[family, freq, outcome, sid].get(date-lag, np.nan)
                    for sid, date in zip(frame.sector_id, dates)]
            current = [growth[family, freq, outcome, sid].get(date, np.nan) for sid, date in zip(frame.sector_id, dates)]
            independent.check(frame[outcome].to_numpy(), current, "Current output growth from uncropped raw calendar", atol=3e-10, rtol=2e-12)
            wcols = [f"{field}_l{lag}" for field in ("temp", "rain") for lag in range(h+1)]
            acols = [f"ar_l{lag}" for lag in range(1, h+1)]
            wvalid = np.isfinite(frame[wcols]).all(axis=1)
            avalid = np.isfinite(frame[acols]).all(axis=1)
            valid = wvalid & avalid & np.isfinite(frame[[outcome, "rh_pct"]]).all(axis=1)
            counts = frame.loc[valid].groupby("time").sector_id.nunique()
            raw_dates = set(counts.index[counts.eq(parent.sector_id.nunique())])
            for field in ADDED + ["wind_coverage", "pm_coverage", "eligible_90"]:
                frame[field] = [added[freq].get(date, {}).get(field, np.nan) for date in dates]
            finite = np.isfinite(frame[ADDED]).all(axis=1)
            covered = frame.eligible_90.fillna(False).astype(bool)
            raw = frame.loc[frame.time.isin(raw_dates)]
            bydate = pd.DataFrame({"time": frame.time, "finite": finite, "covered": covered}).groupby("time").all().loc[sorted(raw_dates)]
            full_dates = set(bydate.index[bydate.finite & bydate.covered])
            own_raw[context, outcome], own_full[context, outcome] = raw_dates, full_dates
            features[context, outcome] = frame
            details[context, outcome] = {
                "parent_n_periods": parent.time.nunique(), "parent_n_obs": len(parent),
                "raw_model_history_n_periods": len(raw_dates), "raw_history_n_periods": len(raw_dates),
                "raw_history_n_obs": len(raw), "raw_history_loss_periods": parent.time.nunique()-len(raw_dates),
                "raw_AR_history_loss_periods": frame.loc[~avalid, "time"].nunique(),
                "weather_history_loss_periods": frame.loc[~wvalid, "time"].nunique(),
                "control_raw_missing_loss_periods": int((~bydate.finite).sum()),
                "control_coverage_loss_periods": int((bydate.finite & ~bydate.covered).sum()),
                "control_total_loss_periods": len(raw_dates)-len(full_dates),
                "full_control_eligible_n_periods": len(full_dates),
                "full_control_eligible_n_obs": len(full_dates)*parent.sector_id.nunique()}
    inventory, masks = read(OUT/"sample_inventory.csv"), read(OUT/"masks.csv")
    assert len(inventory) == 180 and not inventory.duplicated(["context", "outcome", "sample_variant"]).any()
    assert not masks.duplicated(["context", "outcome", "time"]).any()
    samples, records, mask_count = {}, [], 0
    for context in independent.CONTEXTS:
        freq = "M" if "monthly" in context else "Q"
        period, h = ("month", 12) if freq == "M" else ("quarter", 4)
        event_sets = [(event["event_id"], {month.ordinal if freq == "M" else month.asfreq("Q").ordinal
            for month in pd.period_range(event["start"], event["end"], freq="M")}) for event in policy["events"]]
        for outcome in prior.YS:
            target = prior.MATCHES.get(context)
            own = own_full[context, outcome]
            aligned = own if target is None else own & own_full[target, outcome]
            frame = features[context, outcome]
            dates = pd.PeriodIndex(frame.loc[frame.time.isin(aligned), period].drop_duplicates(), freq=freq)
            support = h+1 if outcome == "y_short" else 2*h
            current, intersecting = [], []
            for date in dates:
                current.append([name for name, values in event_sets if date.ordinal in values])
                needed = set(range(date.ordinal-support, date.ordinal+1))
                intersecting.append([name for name, values in event_sets if not needed.isdisjoint(values)])
            saved = masks[masks.context.eq(context) & masks.outcome.eq(outcome)].set_index("period")
            assert set(saved.index) == set(dates.astype(str))
            saved = saved.loc[dates.astype(str)]
            assert saved.history.eq(HISTORY).all() and saved.output_support_horizon.eq(support).all()
            assert saved.output_support_start.tolist() == [str(date-support) for date in dates]
            assert saved.event_ids_at_t.map(independent._event_list).tolist() == current
            assert saved.event_ids_in_output_support.map(independent._event_list).tolist() == intersecting
            assert saved.retain_event_dates.map(independent._truth).tolist() == [not values for values in current]
            assert saved.retain_strict.map(independent._truth).tolist() == [not values for values in intersecting]
            mask_count += len(saved)
            info = {**details[context, outcome], "target_alignment_loss_periods": len(own)-len(aligned),
                "maximum_history_n_periods": len(aligned), "maximum_history_n_obs": len(aligned)*independent.CONTEXTS[context],
                "history_loss_periods": frame.time.nunique()-len(aligned)}
            assert info["control_total_loss_periods"] == info["control_raw_missing_loss_periods"]+info["control_coverage_loss_periods"]
            assert info["history_loss_periods"] == info["raw_history_loss_periods"]+info["control_total_loss_periods"]+info["target_alignment_loss_periods"]
            for variant in prior.VARIANTS:
                retain = np.ones(len(dates), bool) if variant == "all_eligible" else np.array([not values for values in (current if variant == "event_dates_excluded" else intersecting)])
                labels = set(dates[retain].astype(str))
                expected = frame.loc[frame[period].isin(labels)].copy()
                path = OUT/"samples"/f"{context}__{outcome}__{variant}.csv"
                sample = read(path)
                independent.compare_sample(sample, expected, period)
                observed = inventory.loc[inventory.context.eq(context)&inventory.outcome.eq(outcome)&inventory.sample_variant.eq(variant)]
                assert len(observed) == 1
                row = observed.iloc[0]
                values = {**info, "crisis_loss_periods": len(aligned)-len(labels), "retained_n_periods": len(labels),
                          "n_periods": len(labels), "n_sectors": independent.CONTEXTS[context]}
                for field, value in values.items():
                    assert int(row[field]) == value, (context, outcome, variant, field, row[field], value)
                assert row.history == HISTORY and row.control_set == "full90" and "Not cleared" in row.stationarity_status
                for field in ("parent", "weather_history", "control_history", "sample", "policy_source"):
                    path_field = field+"_csv" if field != "policy_source" else field
                    assert independent.sha256(ROOT/row[path_field]) == row[field+"_sha256"]
                assert json.loads(row.current_weather_variables) == ["temp_c", "rain_100mm", "rh_pct", *ADDED]
                if target:
                    assert row.matched_to == target and row.matched_template_n_periods == len(own_full[target, outcome])
                else:
                    assert pd.isna(row.matched_to) and pd.isna(row.matched_template_n_periods)
                samples[context, outcome, variant] = sample
                records.append({"context": context, "outcome": outcome, "sample_variant": variant, **values,
                    "start": str(min(labels)), "end": str(max(labels)), "output_support_horizon": support})
    assert mask_count == len(masks)
    for (context, outcome, variant), sample in samples.items():
        if context in prior.MATCHES:
            assert set(sample.time) == set(samples[prior.MATCHES[context], outcome, variant].time)
    return samples, records, raw_hashes, source_audit, coverage_audit


def scaled_rank(matrix):
    # Removing exact-zero columns changes no rank and makes per-unit checks
    # independent of the number of other units' explicit FE columns.
    matrix = matrix[:, np.any(matrix != 0, axis=0)]
    key = independent.array_sha256(matrix)
    if key in RANK_CACHE:
        RANK_CACHE.move_to_end(key)
        return RANK_CACHE[key]
    scale = np.linalg.norm(matrix, axis=0)
    singular = np.linalg.svd(matrix/scale, compute_uv=False)
    # The producer's tolerance uses the ORIGINAL matrix shape, including zero
    # nuisance columns. Supply that original dimension below where relevant.
    result = singular
    RANK_CACHE[key] = result
    if len(RANK_CACHE) > 256: RANK_CACHE.popitem(last=False)
    return result


def rank(matrix):
    singular = scaled_rank(matrix)
    threshold = singular[0]*max(matrix.shape)*np.finfo(float).eps
    return int(np.sum(singular > threshold))


def no_estimate(rows):
    if rows.empty: return
    for column in ("coefficient", "variance", "std_error", "ci_lower", "ci_upper", "p_value",
                   "hc0_reference_variance", "zero_variance_threshold", "r_squared", "finite_sample_correction"):
        if column in rows:
            assert rows[column].isna().all(), ("Unavailable design contains estimate", column)
    assert rows.inference_status.eq("unavailable: complete design/time support").all()


def verify_native(expected):
    path = OUT/"pyfixest/fit_log.csv"
    log = read(path)
    receipt = json.loads((OUT/"pyfixest/estimator_manifest.json").read_text())
    assert len(log) == len(expected) == 574 and log.api_call_id.nunique() == 574
    assert receipt["native_fits_recorded"] == 574 and receipt["fit_log_sha256"] == independent.sha256(path)
    lookup = log.set_index(["label", "outcome"])
    assert lookup.index.is_unique and set(lookup.index) == {(row["label"],row["outcome"]) for row in expected}
    for row in expected:
        observed = lookup.loc[(row["label"], row["outcome"]), :]
        for field, value in row.items():
            if field not in {"label", "outcome"}: assert observed[field] == value, (row["label"], field)
    assert log.native_class.eq("Feols").all() and log.native_module.str.startswith("pyfixest").all()
    assert log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all() and log.native_residuals_sha256.eq(log.residuals_sha256).all()
    for field in ("native_mapped_coefficient_max_abs_error", "native_residual_max_abs_error", "native_response_max_abs_error", "native_outcome_max_abs_error", "native_design_max_abs_error"):
        assert log[field].eq(0).all()
    assert log.native_design_basis.eq("orthonormal SVD preconditioned full LSDV").all()
    assert log.native_design_orthogonality_max_abs_error.lt(1e-10).all()
    return {"native_fits": len(log), "fit_log_sha256": independent.sha256(path)}


def _verify_literature_full_controls():
    independent.ERRORS.clear(); prior.GEOMETRY.clear(); RANK_CACHE.clear()
    manifest = json.loads((OUT/"manifest.json").read_text())
    assert (manifest["requested_specs"], manifest["identified_specs"], manifest["unavailable_specs"], manifest["native_fits"]) == (1056,574,482,574)
    assert manifest["history"] == HISTORY and manifest["control_set"] == "full90"
    protected = assert_lineage(manifest)
    policy = json.loads((OUT/"policy.json").read_text())
    assert policy == json.loads((ROOT/"notes/crisis_sensitivity_policy.json").read_text())
    samples, histories, raw_hashes, source_audit, coverage_audit = build_samples(policy)
    assert all(manifest["inputs_sha256"][path] == digest for path,digest in raw_hashes.items())
    specs = read(OUT/"specifications.csv")
    assert len(specs) == 1056 and specs.spec_id.is_unique
    expected = set()
    for (context,outcome,variant), sample in samples.items():
        for specification in ["benchmark","distributed","dynamic"]+(["two_channel"] if outcome != "y_short" else []):
            for structure in ["common"]+(["H1"] if sample.sector_id.nunique()>1 else []):
                expected.add((context,outcome,variant,structure,specification))
    assert set(specs[["context","outcome","sample_variant","structure","specification"]].itertuples(index=False,name=None)) == expected
    assert specs.structure.eq(specs.model).all() and specs.specification.eq(specs.model_spec).all()
    # Avoid reading repeated source metadata into every large effect table.
    keep = {"spec_id","sector_id","term","coefficient","std_error","p_value","ci_lower","ci_upper","variance",
        "covariance","hac_lags","inference_df","n_clusters","main_inference","r_squared","finite_sample_correction",
        "n_obs","n_periods","n_parameters","mean_y","sd_y","effect","lag","channel","effect_units","temp_main",
        "interaction_delta","delta_is_normalization","var_temp_main","var_interaction_delta","cov_temp_main_delta",
        "component","estimated","normalization","inference_status","degenerate","hc0_reference_variance","zero_variance_threshold",
        "test","restrictions","covariance_rank","uncapped_covariance_rank","structural_rank_bound","covariance_condition",
        "hc0_covariance_scale","zero_covariance_threshold","rank_cutoff","status","wald_chi2","f_statistic","numerator_df",
        "denominator_df","joint_test_fragile","fragility_reason","time_residual_df","p_value_holm_within_sector_family"}
    names_of_tables = ["coefficients","temperature_effects","control_estimates","interaction_components","channel_estimates","tests","two_channel_span_tests"]
    tables = {name: prior.read(OUT/f"{name}.csv",keep) for name in names_of_tables}
    groups = {name:{str(key):part for key,part in table.groupby("spec_id",sort=False)} for name,table in tables.items()}
    counts = {name:0 for name in names_of_tables}
    counts.update(residual_values=0,full_covariance_cells=0,unavailable_designs=0,nested_designs=0)
    native_expected, fit_records, test_records, nesting_records = [], [], [], []
    for number, spec in enumerate(specs.itertuples(),1):
        sample = samples[spec.context,spec.outcome,spec.sample_variant]
        h = int(spec.horizon); period = "month" if spec.frequency == "monthly" else "quarter"
        path = ROOT/spec.design_npz
        with np.load(path,allow_pickle=False) as archive:
            assert all(archive[key].dtype.kind != "O" for key in archive.files)
            units,times = archive["units"].astype(str),archive["time"].astype(int)
            assert set(zip(sample.sector_id,sample.time)) == set(zip(units,times)) and len(sample) == len(units)
            sample = sample.set_index(["sector_id","time"]).loc[list(zip(units,times))].reset_index()
            assert sample[period].astype(str).tolist() == archive["periods"].astype(str).tolist()
            names = list(archive["coef_names"].astype(str)); assert names == list(archive["names"].astype(str)) and names[-2:] == ADDED
            x, controls, features = prior.expected_design(sample,spec.specification,spec.structure,h,names[:-2])
            x = np.column_stack([x,(sample[ADDED]-sample[ADDED].mean()).to_numpy()]); controls += ADDED
            y = sample[spec.outcome].to_numpy(float)
            independent.check(archive["X"],x,"All full-controls logical design cells",atol=2e-10,rtol=2e-12)
            independent.check(archive["Y"],y,"Every retained native response",atol=0.,rtol=0.)
            n,k = x.shape; t=sample.time.nunique(); sectors=sorted(sample.sector_id.unique())
            full_rank=rank(x); unit_ranks=[rank(x[sample.sector_id.eq(sid)]) for sid in sectors]
            available=full_rank==k and n>k and t>max(unit_ranks)
            assert available == (spec.status=="estimated"), (spec.spec_id,full_rank,k,t,unit_ranks,spec.status)
            for field,value in {"n_obs":n,"n_parameters":k,"n_periods":t,"n_sectors":len(sectors),"design_rank":full_rank,
                "residual_df":n-full_rank,"time_design_rank":max(unit_ranks),"time_design_rank_min":min(unit_ranks),"time_residual_df":t-max(unit_ranks)}.items():
                assert int(getattr(spec,field))==value,(spec.spec_id,field,getattr(spec,field),value)
            assert json.loads(spec.controls)==controls and json.loads(spec.temperature_terms)==features
            assert spec.history==HISTORY and spec.control_set=="full90" and "Not cleared" in spec.stationarity_status
            assert spec.start==sample[period].min() and spec.end==sample[period].max()
            dates=pd.PeriodIndex(sample[period],freq="M" if period=="month" else "Q")
            assert spec.calendar_gaps==dates.max().ordinal-dates.min().ordinal+1-t
            independent.check([spec.mean_y,spec.sd_y],[y.mean(),y.std(ddof=1)],"Exact full-controls outcome mean/SD",atol=3e-10,rtol=2e-12)
            selected=archive["selected_positions"].astype(int)
            selected_expected=[index for index,name in enumerate(names) if not name.startswith(("sector_season[","year["))]
            assert selected.tolist()==selected_expected
            component_expected=[[names.index(feature)]+([names.index(f"{feature}:C(sector_id)[T.{sid}]") for sid in sectors[1:]] if spec.structure=="H1" else []) for feature in features]
            assert archive["component_positions"].tolist()==component_expected and list(archive["temperature_features"])==features
            expected_lags=[0] if spec.specification=="benchmark" else list(range(h+1))
            assert archive["lags"].tolist()==expected_lags
            methods=([("Sector cluster",0,len(sectors)-1)] if spec.structure=="common" and len(sectors)>1 else [])
            methods += [("Calendar HAC" if len(sectors)==1 else "Driscoll-Kraay",band,t-1) for band in (h,2*h)]
            assert list(archive["covariance_labels"])==[value[0] for value in methods]
            assert archive["bandwidths"].tolist()==[value[1] for value in methods] and archive["inference_df"].tolist()==[value[2] for value in methods]
            profiles=prior.profile_contrasts(names,features,sectors,spec.structure,h,spec.specification)
            if available:
                fit=prior.qr_fit(x,y,names); beta=fit["beta"]; residual=fit["residual"]
                independent.check(archive["coefficients"],beta,"All native full-controls coefficients",atol=2e-7,rtol=2e-7)
                independent.check(archive["residuals"],residual,"All native full-controls residuals",atol=2e-8,rtol=1e-8)
                assert np.array_equal(archive["beta"],archive["coefficients"]) and np.array_equal(archive["residual"],archive["residuals"])
                rows=groups["coefficients"][spec.spec_id].set_index("term")
                assert len(rows)==k and set(rows.index)==set(names)
                independent.check(rows.loc[names,"coefficient"].to_numpy(),beta,"All exported native coefficients",atol=2e-7)
                counts["coefficients"]+=k; counts["residual_values"]+=n
                native_expected.append({"label":spec.native_label,"outcome":spec.outcome,"n_obs":n,"n_parameters":k,
                    "design_sha256":independent.array_sha256(archive["X"]),"outcome_sha256":independent.array_sha256(archive["Y"]),
                    "coefficients_sha256":independent.array_sha256(archive["coefficients"]),"residuals_sha256":independent.array_sha256(archive["residuals"]),
                    "coefficient_names_sha256":hashlib.sha256(json.dumps(names,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
                individual,covariances=prior.full_covariances(fit,sample,h,spec.structure)
                hc0=individual[:,selected].T@individual[:,selected]
                independent.check(archive["hc0_selected"],hc0,"Full-controls HC0 reference",atol=1e-7,rtol=2e-7)
            else:
                assert spec.spec_id not in groups["coefficients"]
                assert not any(key in archive.files for key in ["beta","coefficients","residual","residuals","hc0_selected"])
                assert not any(key.startswith("covariance_") and key!="covariance_labels" for key in archive.files)
                assert np.isnan(archive["corrections"]).all()
                fit=beta=hc0=None;counts["unavailable_designs"]+=1
            for method_number,(method,bandwidth,degrees) in enumerate(methods):
                main=method=="Sector cluster" if spec.structure=="common" and len(sectors)>1 else bandwidth==h
                current={name:part.loc[part.covariance.eq(method)&part.hac_lags.eq(bandwidth)] for name in names_of_tables[1:]
                    if (part:=groups[name].get(spec.spec_id)) is not None}
                point=current["temperature_effects"]; tests=current["tests"]
                assert point.main_inference.eq(main).all() and point.inference_df.eq(degrees).all()
                if available:
                    _,_,full_v,factor,actual_df=covariances[method_number];assert actual_df==degrees
                    independent.check(archive[f"covariance_{method_number}"],full_v,"Every full-controls nuisance-inclusive covariance cell",atol=2e-7,rtol=2e-7)
                    independent.check(archive["corrections"][method_number],factor,"Full-controls finite sample multiplier",atol=1e-12,rtol=1e-12)
                    counts["full_covariance_cells"]+=k*k
                    v=full_v[np.ix_(selected,selected)]; b=beta[selected]
                    independent.check(point.r_squared,np.full(len(point),fit["r_squared"]),"Centered native full-controls R squared",atol=5e-10)
                else:
                    full_v=v=b=None
                    for name,rows in current.items():
                        if name not in {"tests","interaction_components","two_channel_span_tests"}:no_estimate(rows)
                bound=len(sectors)-1 if method=="Sector cluster" else t-1
                basis_rows=[];cumulative_rows=[]
                def audit_effect(row, vector):
                    if available:
                        independent.inference(row,float(vector@b),float(vector@v@vector),degrees,float(vector@hc0@vector))
                def audit_test(row, restriction, table="tests"):
                    assert int(row.restrictions)==len(restriction) and row.structural_rank_bound==bound
                    if available:
                        ok=prior.verify_test(row,b,v,hc0,restriction,degrees,bound)
                    else:
                        assert row.status=="unavailable: complete design/time support"
                        assert pd.isna(row.p_value) and pd.isna(row.wald_chi2) and pd.isna(row.f_statistic) and pd.isna(row.covariance_rank)
                        assert row.denominator_df==degrees and row.numerator_df==len(restriction)
                        ok=False
                    test_records.append({"spec_id":spec.spec_id,"covariance":method,"hac_lags":bandwidth,"sector_id":row.sector_id,"test":row.test,"point_available":available,"restriction_available":ok})
                    counts[table]+=1
                for sid,(lag_full,basis_full) in profiles.items():
                    lag,basis=lag_full[:,selected],basis_full[:,selected]
                    unit=sample if sid=="pooled" else sample[sample.sector_id.eq(sid)]
                    observed=point[point.sector_id.eq(sid)]
                    assert len(observed)==len(expected_lags)+1
                    for i,lag_number in enumerate(expected_lags+[-1]):
                        row=observed.loc[observed.lag.eq(lag_number)];assert len(row)==1;row=row.iloc[0]
                        vector=lag[i] if lag_number>=0 else lag.sum(axis=0)
                        audit_effect(row,vector)
                        independent.check([row.mean_y,row.sd_y],[unit[spec.outcome].mean(),unit[spec.outcome].std(ddof=1)],"Every sector exact-sample outcome mean/SD",atol=3e-10,rtol=2e-12)
                        if available and spec.structure=="H1" and spec.specification!="two_channel" and lag_number>=0:
                            main_pos=names.index(f"temp_l{lag_number}");delta=None if sid==sectors[0] else names.index(f"temp_l{lag_number}:C(sector_id)[T.{sid}]")
                            independent.check([row.temp_main,row.interaction_delta,row.var_temp_main,row.var_interaction_delta,row.cov_temp_main_delta],
                                [beta[main_pos],0. if delta is None else beta[delta],full_v[main_pos,main_pos],0. if delta is None else full_v[delta,delta],0. if delta is None else full_v[main_pos,delta]],
                                "Full-controls main-plus-industry lag decomposition",atol=2e-7)
                        counts["temperature_effects"]+=1
                    if spec.specification=="two_channel":
                        channels=current["channel_estimates"].loc[lambda d:d.sector_id.eq(sid)].set_index("channel")
                        assert len(channels)==2 and set(channels.index)==set(features)
                        for j,feature in enumerate(features):audit_effect(channels.loc[feature],basis[j]);counts["channel_estimates"]+=1
                        operators={"All temperature-lag coefficients zero":np.eye(2),"Cumulative direct temperature response zero":np.array([[0.,h]]),
                            "Exact contemporaneous static-output-level restriction":np.array([[0.,1.]]),"Equal one-period growth responses over YoY accumulation window":np.array([[1.,0.]])}
                        restrictions={name:r@basis for name,r in operators.items()}
                    else:
                        restrictions={name:r@lag for name,r in prior.restriction_operators(len(expected_lags),spec.outcome).items()}
                    for name,restriction in restrictions.items():
                        row=tests.loc[tests.sector_id.eq(sid)&tests.test.eq(name)];assert len(row)==1
                        audit_test(row.iloc[0],restriction)
                    if spec.specification in {"distributed","dynamic"} and spec.outcome!="y_short":
                        span_rows=current["two_channel_span_tests"]
                        row=span_rows.loc[span_rows.sector_id.eq(sid)];assert len(row)==1
                        eye=np.eye(h+1)
                        span_restriction=np.vstack([eye[2:h]-eye[1],eye[0]+eye[h]-eye[1]])
                        assert span_restriction.shape==(h-1,h+1)
                        audit_test(row.iloc[0],span_restriction@lag,"two_channel_span_tests")
                    cumulative_rows.append(lag.sum(axis=0));basis_rows.append(basis)
                controls_rows=current["control_estimates"].set_index("term")
                assert len(controls_rows)==len(controls) and set(controls_rows.index)==set(controls)
                for name in controls:
                    vector=np.eye(len(selected))[list(selected).index(names.index(name))]
                    audit_effect(controls_rows.loc[name],vector);counts["control_estimates"]+=1
                if spec.structure=="H1":
                    components=current["interaction_components"]
                    assert len(components)==len(features)*(len(sectors)+1)
                    for j,feature in enumerate(features):
                        lag_label=j if spec.specification!="two_channel" else -2-j
                        for i,sid in enumerate(sectors):
                            term=feature if i==0 else f"{feature}:C(sector_id)[T.{sid}]"
                            row=components[components.term.eq(term)&~components.normalization.map(independent._truth)]
                            assert len(row)==1 and row.lag.iloc[0]==lag_label and row.sector_id.iloc[0]==sid
                            assert bool(independent._truth(row.estimated.iloc[0]))==available
                            if available:
                                vector=np.eye(len(selected))[list(selected).index(names.index(term))];audit_effect(row.iloc[0],vector)
                            else:no_estimate(row)
                            counts["interaction_components"]+=1
                        normalization=components[components.lag.eq(lag_label)&components.normalization.map(independent._truth)]
                        assert len(normalization)==1 and normalization.sector_id.iloc[0]==sectors[0]
                        assert normalization.coefficient.eq(0).all() and normalization[["std_error","p_value","ci_lower","ci_upper"]].isna().all().all()
                        counts["interaction_components"]+=1
                    global_restrictions={
                        "All sector cumulative direct temperature responses equal":np.vstack([row-cumulative_rows[0] for row in cumulative_rows[1:]]),
                        "All sector temperature-lag response vectors equal":np.vstack([row-basis_rows[0] for row in basis_rows[1:]]),
                        "All sector temperature-lag responses jointly zero":np.vstack(basis_rows)}
                    for name,restriction in global_restrictions.items():
                        row=tests.loc[tests.sector_id.eq("all")&tests.test.eq(name)];assert len(row)==1
                        audit_test(row.iloc[0],restriction)
            fit_records.append({"spec_id":spec.spec_id,"context":spec.context,"outcome":spec.outcome,"sample_variant":spec.sample_variant,
                "structure":spec.structure,"specification":spec.specification,"status":spec.status,"n_obs":n,"n_periods":t,"n_parameters":k,
                "full_rank":full_rank,"minimum_unit_rank":min(unit_ranks),"maximum_unit_rank":max(unit_ranks),"residual_time_df":t-max(unit_ranks)})
        if number%75==0: print(f"Full-controls independent design/QR/VCV audit: {number}/1056 requests",flush=True)
    for name in names_of_tables:assert counts[name]==len(tables[name]),("Every exported row audited",name,counts[name],len(tables[name]))
    assert counts["unavailable_designs"]==482
    # A reduced model may be estimable while its unrestricted DL parent is not.
    # Design nesting is still exact; only estimate the identified reduced X.
    for spec in specs.loc[specs.specification.eq("two_channel")].itertuples():
        full_id=spec.spec_id.replace("__two_channel","__distributed")
        with np.load(OUT/"designs"/f"{full_id}.npz",allow_pickle=False) as full,np.load(ROOT/spec.design_npz,allow_pickle=False) as reduced:
            assert np.array_equal(full["Y"],reduced["Y"]) and np.array_equal(full["time"],reduced["time"]) and np.array_equal(full["units"],reduced["units"])
            mapping=prior.nesting_embedding(list(full["coef_names"]),list(reduced["coef_names"]),int(spec.horizon))
            nested=full["X"]@mapping
            independent.check(nested,reduced["X"],"Two-channel constrained full-controls DL embedding",atol=2e-10,rtol=2e-12)
            if spec.status=="estimated":
                fit=prior.qr_fit(nested,full["Y"],list(reduced["coef_names"]))
                independent.check(fit["beta"],reduced["coefficients"],"Identified constrained-DL / native two-channel coefficients",atol=2e-7)
                independent.check(fit["residual"],reduced["residuals"],"Constrained-DL / native two-channel residuals",atol=2e-8)
            nesting_records.append({"spec_id":spec.spec_id,"distributed_spec_id":full_id,"two_channel_status":spec.status,
                "unrestricted_parameters":full["X"].shape[1],"constrained_parameters":reduced["X"].shape[1]})
    assert len(nesting_records)==192;counts["nested_designs"]=len(nesting_records)
    # Familywise inference includes every displayed industry in each family;
    # missing tests retain a slot but are never displayed as significant.
    effects=read(OUT/"temperature_effects.csv")
    family=["context","outcome","sample_variant","model_spec","covariance","hac_lags","effect","lag","term"]
    for _,part in effects.loc[effects.model.eq("H1")].groupby(family,sort=False):
        adjusted=independent.holm(part.p_value.fillna(1).to_numpy())
        expected=np.where(part.p_value.notna(),adjusted,np.nan)
        independent.check(part.p_value_holm_within_sector_family.to_numpy(),expected,"Holm industry-family adjusted p values",atol=2e-13,rtol=2e-12)
    native=verify_native(native_expected)
    assert assert_lineage(manifest)==protected
    pd.DataFrame(histories).to_csv(OUT/"verification_history.csv",index=False)
    pd.DataFrame(fit_records).to_csv(OUT/"verification_fits.csv",index=False)
    pd.DataFrame(test_records).to_csv(OUT/"verification_tests.csv",index=False)
    pd.DataFrame(nesting_records).to_csv(OUT/"verification_nesting.csv",index=False)
    return {"status":"passed","contexts":20,"samples":180,"requests":1056,"native_fits":574,"unavailable_requests":482,
        "history":HISTORY,"counts":counts,"raw_output_source_hashes":raw_hashes,"raw_parent_audit":source_audit,
        "raw_wind_pm_audit":coverage_audit,"native_receipt":native,"producer_source_sha256":manifest["producer_source_sha256"],
        "verifier_source_sha256":independent.sha256(Path(__file__)),"protected_files_checked":len(protected),"numerical_checks":independent.ERRORS,
        "scope":"Raw output and weather calendars, wind/PM means and actual coverage, all support/coverage/alignment/event loss stages, every retained and unavailable X/Y request, full QR native coefficients/residuals and CR1/calendar-Bartlett covariances, controls, baseline-plus-interaction totals, lag sums, two channels, exact shape restrictions, industry-Holm inference and constrained-design nesting.",
        "span_supplement_manifest_sha256":independent.sha256(OUT/"two_channel_span_manifest.json"),
        "limits":"An estimable sparse model is not a stationarity or causal certificate. Small residual time df, conditional dynamic-FE bias, high-dimensional or ill-conditioned Wald tests and contemporaneous pollution endogeneity remain substantive limits. Unavailable models keep every requested control and have no invented estimates or p values. General span tests in dynamic models are conditional weather-shape restrictions, not comparisons to the separately fitted no-AR two-channel equation."}


def verify_literature_full_controls():
    """Write a fail-closed independent receipt and return its complete payload."""
    path=OUT/"verification.json"
    initial={"status":"running","verifier_source_sha256":independent.sha256(Path(__file__))}
    path.write_text(json.dumps(initial,indent=2)+"\n")
    try:
        with threadpool_limits(limits=1): report=_verify_literature_full_controls()
    except BaseException as error:
        path.write_text(json.dumps({**initial,"status":"failed","error_type":type(error).__name__,"error":str(error)},indent=2)+"\n")
        raise
    temporary=path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");os.replace(temporary,path)
    print(f"Full-controls independent verification passed: 1056 requests, 574 native fits, 482 unavailable designs, {report['counts']['full_covariance_cells']:,} covariance cells.",flush=True)
    return report


if __name__=="__main__":verify_literature_full_controls()
