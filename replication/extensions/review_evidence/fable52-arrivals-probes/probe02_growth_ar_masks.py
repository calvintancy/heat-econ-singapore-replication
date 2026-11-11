"""Probe 02 - raw-count growth / AR lags / common calendar / event and strict masks.

Independent of every production module: raw official counts are parsed with a
hand-written month parser, growth is formed with math.log on integers, masks
with integer month arithmetic.  All 72 saved monthly samples and all 5,400
saved mask rows are compared.  Writes audit_probes/probe02_result.json.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe02_growth_ar_masks.py
"""
from _probe_common import BASE, ROOT, write

import csv
import json
import math

import numpy as np
import pandas as pd

MONTHS = {name: number for number, name in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
NAN = float("nan")


def ordinal(year, month):
    return (year - 1970) * 12 + month - 1


def label(o):
    return f"{1970 + o // 12:04d}-{o % 12 + 1:02d}"


# ---- raw counts ---------------------------------------------------------
meta = json.loads((ROOT / "data/raw/visitor_arrivals/nsa_singstat_metadata.json").read_text())["Data"]["records"]
catalog = [(str(row["seriesNo"]), row["rowText"].strip()) for row in meta["row"]]
with open(ROOT / "data/raw/visitor_arrivals/nsa.csv", newline="") as handle:
    rows = list(csv.reader(handle))
header, body = rows[0], rows[1:]
ordinals = [ordinal(int(text[:4]), MONTHS[text[4:]]) for text in header[1:]]
assert sorted(ordinals) == list(range(ordinal(2008, 1), ordinal(2026, 9) + 1)) and len(ordinals) == 225
names_csv = [row[0].strip() for row in body]
assert len(set(names_csv)) == len(names_csv) == len(catalog) == 62
order_mapping_equals_name_mapping = names_csv == [name for _, name in catalog]
by_name = {name: unit for unit, name in catalog}
counts = {}
for row in body:
    unit = by_name[row[0].strip()]
    values = {}
    for o, text in zip(ordinals, row[1:]):
        value = int(text)  # raises on blanks / "na" / decimals -> no imputation possible
        assert value >= 0
        values[o] = value
    counts[unit] = values
FIRST, LAST = min(ordinals), max(ordinals)
DATES = list(range(FIRST, LAST + 1))
ids = [unit for unit, _ in catalog]
cuts = {"aggregate": [u for u in ids if u == "1"],
        "region": sorted(u for u in ids if u.count(".") == 1 and u != "1.999"),
        "country": sorted(u for u in ids if u.count(".") == 2 and not u.endswith(".999"))}
assert [len(cuts[c]) for c in cuts] == [1, 9, 42]
HORIZON = {"y_mom_nsa": (1, 3), "y_yoy_nsa": (12, 12)}


def growth(unit, o, h):
    a, b = counts[unit].get(o), counts[unit].get(o - h)
    if a is None or b is None or a <= 0 or b <= 0:
        return NAN
    return 100.0 * (math.log(a) - math.log(b))


G = {y: {u: {o: growth(u, o, h) for o in range(FIRST - 30, LAST + 1)} for c in cuts for u in cuts[c]} for y, (h, _) in HORIZON.items()}

# ---- weather / controls (only finiteness + lag values are needed here) --
weather = {}
for stem, column in [("temperature", "mean_temp"), ("rainfall", "total_rainfall"), ("humidity", "mean_rh")]:
    frame = pd.read_csv(ROOT / f"data/raw/{stem}_monthly.csv")
    weather[stem] = {ordinal(int(m[:4]), int(m[5:7])): float(v) for m, v in zip(frame.month, frame[column])}
controls = pd.read_csv(ROOT / "data/processed/extended_controls/monthly_controls.csv")
ctrl = {ordinal(int(m[:4]), int(m[5:7])): (w, p, wc, pc) for m, w, p, wc, pc in zip(controls.month, controls.wind_mps, controls.pm25_10ug, controls.wind_coverage, controls.pm_coverage)}
lunar = pd.read_csv(ROOT / "data/processed/visitor_arrivals/lunar_new_year.csv")
lny = {ordinal(int(m[:4]), int(m[5:7])): int(v) for m, v in zip(lunar.month, lunar.lny_window_days)}
lny[ordinal(2007, 12)] = 0  # verified from the HKO 2007 table in probe 03

policy = json.loads((ROOT / "notes/crisis_sensitivity_policy.json").read_text())
events = [(e["event_id"], ordinal(int(e["start"][:4]), int(e["start"][5:7])), ordinal(int(e["end"][:4]), int(e["end"][5:7]))) for e in policy["events"]]


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def eligible_calendar(panel, outcome, control_set, history):
    h, H = HORIZON[outcome]
    matched = history == "matched_maximum_growth_history"
    stages = {}
    out = {}
    for o in DATES:
        own_current = all(finite(G[outcome][u][o]) for u in cuts[panel])
        def hist_ok(cut):
            lags = range(0, H + 1) if matched else [0]
            return all(finite(G[outcome][u][o - lag]) for u in cuts[cut] for lag in lags)
        own_hist = own_current and hist_ok(panel)
        lag_window = H if matched else 0
        w_ok = all(finite(weather[s].get(o - lag)) for s in ["temperature", "rainfall"] for lag in range(lag_window + 1)) and finite(weather["humidity"].get(o)) \
            and finite(lny.get(o)) and finite(lny.get(o - 1))
        shared = all(hist_ok(cut) for cut in cuts)
        added = True
        cover = True
        if control_set == "full":
            c = ctrl.get(o)
            added = c is not None and finite(c[0]) and finite(c[1]) and not math.isnan(c[0]) and not math.isnan(c[1])
            cover = c is not None and (c[2] >= .9) and (c[3] >= .9)
        out[o] = dict(own_current=own_current, own_hist=own_hist, weather=w_ok, shared=shared, added=added, cover=cover,
                      eligible=own_hist and w_ok and shared and added and cover)
    return out


def masks_for(o, support):
    at = [name for name, s, e in events if s <= o <= e]
    overlap = [name for name, s, e in events if not (o < s or o - support > e)]  # [o-support,o] intersects [s,e]
    return at, overlap


result = {"probe": "02 growth / AR / common calendar / masks", "order_mapping_equals_name_mapping": order_mapping_equals_name_mapping,
          "source_months": len(DATES), "first": label(FIRST), "last": label(LAST)}

# ---- zero cells and the losses they cause -------------------------------
zero_cells = {c: {u: [label(o) for o in DATES if counts[u][o] == 0] for u in cuts[c] if any(counts[u][o] == 0 for o in DATES)} for c in cuts}
names = dict(catalog)
result["zero_cells_per_cut"] = {c: sum(len(v) for v in zero_cells[c].values()) for c in cuts}
result["zero_cells_country_by_market"] = {f"{u} {names[u]}": v for u, v in zero_cells["country"].items()}
losses = {}
for outcome, (h, H) in HORIZON.items():
    own = [o for o in DATES if o - h >= FIRST]
    lost = [o for o in own if not all(finite(G[outcome][u][o]) for c in cuts for u in cuts[c])]
    agg_defined_but_lost = [o for o in lost if finite(G[outcome]["1"][o])]
    region_defined_but_lost = [o for o in lost if all(finite(G[outcome][u][o]) for u in cuts["region"])]
    losses[outcome] = {"raw_growth_months": len(own), "common_calendar_lost_months": len(lost), "lost_months": [label(o) for o in lost],
                       "lost_although_official_total_growth_is_finite": len(agg_defined_but_lost),
                       "lost_although_all_nine_region_growths_finite": len(region_defined_but_lost),
                       "official_total_growth_in_lost_months_min_max": [min(G[outcome]["1"][o] for o in agg_defined_but_lost), max(G[outcome]["1"][o] for o in agg_defined_but_lost)] if agg_defined_but_lost else None,
                       "official_total_growth_all_months_min_max": [min(G[outcome]["1"][o] for o in own), max(G[outcome]["1"][o] for o in own)]}
result["common_calendar_losses_own_current"] = losses

# ---- one zero market traced end to end ----------------------------------
trace_unit = max(zero_cells["country"], key=lambda u: len(zero_cells["country"][u]))
zeros_o = [o for o in DATES if counts[trace_unit][o] == 0]
trace = {"unit": f"{trace_unit} {names[trace_unit]}", "zero_months": [label(o) for o in zeros_o],
         "mom_nan_months": [label(o) for o in DATES if o > FIRST and not finite(G["y_mom_nsa"][trace_unit][o])],
         "yoy_nan_months": [label(o) for o in DATES if o - 12 >= FIRST and not finite(G["y_yoy_nsa"][trace_unit][o])]}
expected_mom = sorted({o for z in zeros_o for o in (z, z + 1) if FIRST < o <= LAST})
expected_yoy = sorted({o for z in zeros_o for o in (z, z + 12) if FIRST + 12 <= o <= LAST})
trace["mom_nan_equals_zero_month_and_next"] = trace["mom_nan_months"] == [label(o) for o in expected_mom]
trace["yoy_nan_equals_zero_month_and_plus12"] = trace["yoy_nan_months"] == [label(o) for o in expected_yoy]
result["zero_market_trace"] = trace

# ---- all 72 saved samples ------------------------------------------------
inventory = pd.read_csv(BASE / "monthly/sample_inventory.csv")
saved_masks = pd.read_csv(BASE / "monthly/masks.csv")
sample_report = []
worst = {"y": 0.0, "ar": 0.0, "temp_lag": 0.0, "rain_lag": 0.0, "arrivals": 0.0, "channel": 0.0, "lny": 0.0}
mask_mismatch = {"rows": 0, "eligible": 0, "retain_event": 0, "retain_strict": 0, "support": 0, "stage_flags": 0, "event_id_lists": 0}
trace_presence = []
for (panel, outcome, control_set, history), group in inventory.groupby(["panel", "outcome", "control_set", "history"], sort=False):
    h, H = HORIZON[outcome]
    matched = history == "matched_maximum_growth_history"
    support = (H + h) if matched else 12
    calendar = eligible_calendar(panel, outcome, control_set, history)
    saved = saved_masks[(saved_masks.panel == panel) & (saved_masks.outcome == outcome) & (saved_masks.control_set == control_set) & (saved_masks.history == history)]
    assert len(saved) == 225
    for row in saved.itertuples():
        o = int(row.time)
        at, overlap = masks_for(o, support)
        mask_mismatch["rows"] += 1
        c = calendar[o]
        mask_mismatch["eligible"] += int(bool(row.eligible_before_event_masks) != c["eligible"])
        mask_mismatch["retain_event"] += int(bool(row.retain_event_dates) != (not at))
        mask_mismatch["retain_strict"] += int(bool(row.retain_strict) != (not overlap))
        mask_mismatch["support"] += int(int(row.output_support_horizon) != support or row.output_support_start != label(o - support) or row.month != label(o))
        mask_mismatch["stage_flags"] += int(bool(row.current_output_finite) != c["own_current"] or bool(row.own_output_history_finite) != c["own_hist"]
                                            or bool(row.weather_history_finite) != c["weather"] or bool(row.all_three_cuts_output_history_finite) != c["shared"]
                                            or bool(row.current_added_controls_finite) != c["added"] or bool(row.current_control_coverage_pass) != c["cover"])
        mask_mismatch["event_id_lists"] += int(json.loads(row.event_ids_at_t) != at or json.loads(row.event_ids_in_output_support) != overlap)
    for item in group.itertuples():
        variant = item.sample_variant
        keep = []
        for o in DATES:
            if not calendar[o]["eligible"]:
                continue
            at, overlap = masks_for(o, support)
            if variant == "event_dates_excluded" and at:
                continue
            if variant == "strict_growth_window_clean" and overlap:
                continue
            keep.append(o)
        sample = pd.read_csv(ROOT / item.sample_csv, dtype={"unit_id": str, "sector_id": str})
        saved_dates = sorted(sample.time.unique())
        units = sorted(sample.unit_id.unique())
        balanced = len(sample) == len(saved_dates) * len(cuts[panel]) and units == cuts[panel] and not sample.duplicated(["time", "unit_id"]).any()
        o_arr, u_arr = sample.time.to_numpy(int), sample.unit_id.to_numpy(str)
        mine_y = np.array([G[outcome][u][o] for o, u in zip(o_arr, u_arr)])
        worst["y"] = max(worst["y"], float(np.max(np.abs(mine_y - sample[outcome].to_numpy(float)))))
        worst["arrivals"] = max(worst["arrivals"], float(np.max(np.abs(np.array([counts[u][o] for o, u in zip(o_arr, u_arr)]) - sample.arrivals.to_numpy(float)))))
        ar_checked = 0
        for lag in range(1, H + 1):
            mine = np.array([G[outcome][u][o - lag] for o, u in zip(o_arr, u_arr)])
            theirs = sample[f"ar_l{lag}"].to_numpy(float)
            if matched:
                assert np.isfinite(theirs).all() and np.isfinite(mine).all(), (item.sample_id, lag)
            same_nan = np.array_equal(np.isnan(mine), np.isnan(theirs))
            assert same_nan, (item.sample_id, lag)
            if np.isfinite(mine).any():
                worst["ar"] = max(worst["ar"], float(np.nanmax(np.abs(mine - theirs))))
            ar_checked += int(np.isfinite(mine).sum())
        for lag in range(0, H + 1):
            for stem, key, scale in [("temperature", "temp", 1.0), ("rainfall", "rain", 100.0)]:
                mine = np.array([weather[stem][o - lag] / scale for o in o_arr])
                worst[f"{key}_lag"] = max(worst[f"{key}_lag"], float(np.max(np.abs(mine - sample[f"{key}_l{lag}"].to_numpy(float)))))
        temp = lambda o, lag: weather["temperature"][o - lag]
        mine_diff = np.array([temp(o, 0) - temp(o, H) for o in o_arr])
        mine_sum = np.array([sum(temp(o, lag) for lag in range(H)) for o in o_arr])
        worst["channel"] = max(worst["channel"], float(np.max(np.abs(mine_diff - sample.temp_endpoint_difference.to_numpy(float)))),
                               float(np.max(np.abs(mine_sum - sample.temp_growth_sum.to_numpy(float)))))
        worst["lny"] = max(worst["lny"], float(np.max(np.abs(np.array([lny[o] for o in o_arr]) - sample.lny_window_days.to_numpy(float)))),
                           float(np.max(np.abs(np.array([lny[o - 1] for o in o_arr]) - sample.lny_lag1.to_numpy(float)))))
        # masked/pre-mask history really enters: count regressors whose source month is itself an excluded event month
        ar_from_event_months = 0
        if matched and variant == "event_dates_excluded":
            ar_from_event_months = int(sum(1 for o in keep for lag in range(1, H + 1) if masks_for(o - lag, support)[0]))
        y_base_in_event = int(sum(1 for o in keep if masks_for(o - h, 0)[0])) if variant == "event_dates_excluded" else 0
        if panel == "aggregate":
            trace_presence.append({"sample": item.sample_id, "retained_months_where_trace_market_growth_is_nan": int(sum(1 for o in keep if not finite(G[outcome][trace_unit][o])))})
        sample_report.append({"sample_id": item.sample_id, "T_independent": len(keep), "T_saved": len(saved_dates), "dates_equal": keep == [int(v) for v in saved_dates],
                              "balanced_all_units": bool(balanced), "n_obs": len(sample), "T_inventory": int(item.n_periods), "gaps": int(keep[-1] - keep[0] + 1 - len(keep)),
                              "start": label(keep[0]), "end": label(keep[-1]), "support": support, "ar_values_checked": ar_checked,
                              "retained_obs_whose_AR_lag_month_is_event_month": ar_from_event_months, "retained_obs_whose_growth_base_month_is_event_month": y_base_in_event})
result["samples_checked"] = len(sample_report)
result["samples_dates_equal"] = int(sum(r["dates_equal"] for r in sample_report))
result["samples_balanced_all_units"] = int(sum(r["balanced_all_units"] for r in sample_report))
result["max_abs_difference"] = worst
result["mask_rows_checked"] = mask_mismatch["rows"]
result["mask_mismatches"] = {k: v for k, v in mask_mismatch.items() if k != "rows"}
result["support_by_history_outcome"] = sorted({(r["sample_id"].split("__")[1], r["sample_id"].split("__")[3], r["support"]) for r in sample_report})
result["aggregate_samples_never_retain_month_with_nan_market_growth"] = all(r["retained_months_where_trace_market_growth_is_nan"] == 0 for r in trace_presence)
result["pre_mask_history_examples"] = [r for r in sample_report if r["sample_id"] in (
    "aggregate__y_yoy_nsa__long__matched_maximum_growth_history__event_dates_excluded", "aggregate__y_mom_nsa__long__matched_maximum_growth_history__event_dates_excluded",
    "aggregate__y_yoy_nsa__long__own_current_history__event_dates_excluded", "aggregate__y_yoy_nsa__long__own_current_history__strict_growth_window_clean",
    "aggregate__y_yoy_nsa__long__matched_maximum_growth_history__strict_growth_window_clean", "aggregate__y_mom_nsa__long__matched_maximum_growth_history__strict_growth_window_clean")]
result["sample_table"] = sample_report
ok = (result["samples_dates_equal"] == 72 and result["samples_balanced_all_units"] == 72 and all(v == 0 for v in result["mask_mismatches"].values())
      and max(worst.values()) < 1e-9)
result["verdict"] = "independent reconstruction matches all 72 samples and 5,400 mask rows" if ok else "DISCREPANCY - inspect"
write("probe02_result.json", result)
print(json.dumps({k: v for k, v in result.items() if k not in ("sample_table",)}, indent=1)[:9000])
