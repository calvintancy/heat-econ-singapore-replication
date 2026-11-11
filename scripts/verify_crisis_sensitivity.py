"""Independent raw-calendar, QR and score-sandwich crisis sensitivity audit.

No crisis producer, production estimator, or production covariance helper is
imported. Five outcomes share independently rebuilt designs; exclusion windows
are checked against every calendar period rather than compressed row positions.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
from collections import Counter

for _name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.linalg import qr, solve_triangular
from scipy.stats import f as f_distribution, t as student_t
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/crisis_sensitivity"
PROBES = ROOT / "dist/aea_audit_20261111/new_crisis_checks"
POLICY = ROOT / "notes/crisis_sensitivity_policy.json"
OUTCOMES = ["y_short", "y_yoy_nsa", "y_yoy", "y_log_level", "y_raw_level"]
WEATHER = ["temp_c", "rain_100mm", "rh_pct"]
VARIANTS = ["all_eligible", "event_dates_excluded", "strict_growth_window_clean"]
CONTEXTS = {"aggregate_va": 1, "va": 11, "va-broad-matched": 11, "va-detailed": 19,
            "iip-monthly-total": 1, "iip-monthly": 6, "iip-monthly-broad-matched": 6, "iip-monthly-detailed": 18,
            "iip-quarterly-total": 1, "iip-quarterly": 6, "iip-quarterly-broad-matched": 6, "iip-quarterly-detailed": 18}
MATCHED_TOTALS = {"aggregate_va-matched-va19": ("aggregate_va", "va-detailed"),
                  "iip-monthly-total-matched-broad": ("iip-monthly-total", "iip-monthly"),
                  "iip-monthly-total-matched-detailed": ("iip-monthly-total", "iip-monthly-detailed"),
                  "iip-quarterly-total-matched-broad": ("iip-quarterly-total", "iip-quarterly"),
                  "iip-quarterly-total-matched-detailed": ("iip-quarterly-total", "iip-quarterly-detailed")}
CONTEXTS.update({name: 1 for name in MATCHED_TOTALS})
COMPLETE_VA = {"va12": "va", "va12-matched": "va-broad-matched", "va20": "va-detailed"}
CONTEXTS.update({"va12": 12, "va12-matched": 12, "va20": 20})
ID_COLUMNS = {"sector_id", "unit_id", "sector_i", "sector_j", "first_sector_id", "second_sector_id", "reference_sector_id"}
ERRORS: dict[str, dict] = {}


def sha256(path: Path) -> str:
    with Path(path).open("rb") as f:
        digest = hashlib.file_digest(f, "sha256")
    return digest.hexdigest()


def array_sha256(array) -> str:
    value = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={name: str for name in ID_COLUMNS}, float_precision="round_trip", low_memory=False)


def check(actual, expected, name: str, *, atol=2e-7, rtol=2e-7):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), (name, "unavailable masks differ")
    assert np.array_equal(np.isinf(a), np.isinf(b)), (name, "infinite masks differ")
    finite = np.isfinite(a) & np.isfinite(b)
    error = np.abs(a[finite] - b[finite])
    tolerance = atol + rtol * np.abs(b[finite])
    fraction = np.divide(error, tolerance, out=np.where(error == 0, 0., np.inf), where=tolerance > 0)
    assert np.all(error <= tolerance), (name, float(error.max()), float(np.max(fraction)))
    record = ERRORS.setdefault(name, {"values": 0, "maximum_absolute_error": 0., "maximum_scaled_error": 0., "maximum_fraction_of_tolerance": 0.})
    record["values"] += int(finite.sum())
    if error.size:
        record["maximum_absolute_error"] = max(record["maximum_absolute_error"], float(error.max()))
        record["maximum_scaled_error"] = max(record["maximum_scaled_error"], float(np.max(error / np.maximum(1., np.abs(b[finite])))))
        record["maximum_fraction_of_tolerance"] = max(record["maximum_fraction_of_tolerance"], float(np.max(fraction)))


def independent_qr(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Estimate every response jointly from QR, never a production fit helper."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    n, k = x.shape
    assert n > k and np.isfinite(x).all() and np.isfinite(y).all()
    scale = np.linalg.norm(x, axis=0)
    assert np.all(scale > 0)
    with threadpool_limits(limits=1):
        q, r = qr(x / scale, mode="economic", pivoting=False)
        singular = np.linalg.svd(r, compute_uv=False)
        rank = int(np.sum(singular > singular[0] * 1e-10))
        assert rank == k, ("Independent QR rank", rank, k)
        scaled_beta = solve_triangular(r, q.T @ y)
        beta = scaled_beta / scale[:, None]
        residual = y - q @ (q.T @ y)
    return {"x": x, "q": q, "r": r, "scale": scale, "beta": beta, "residual": residual,
            "names": names, "n": n, "k": k, "rank": rank,
            "r_squared": 1 - (residual**2).sum(axis=0) / ((y-y.mean(axis=0))**2).sum(axis=0)}


def influences(fit: dict, contrast: np.ndarray) -> np.ndarray:
    """Full OLS coefficient influences via triangular solve, avoiding raw bread."""
    c = np.atleast_2d(contrast)
    with threadpool_limits(limits=1):
        return fit["q"] @ solve_triangular(fit["r"].T, (c / fit["scale"]).T, lower=True)


def bartlett_scores(score: np.ndarray, dates: np.ndarray, bandwidth: int) -> np.ndarray:
    """Lag pairs are joined by actual calendar ordinals, including missing dates."""
    score, dates = np.asarray(score, float), np.asarray(dates, dtype=np.int64)
    assert len(score) == len(dates) and len(set(dates)) == len(dates)
    lookup = {int(value): i for i, value in enumerate(dates)}
    covariance = score.T @ score
    for lag in range(1, bandwidth + 1):
        cross = np.zeros_like(covariance)
        for current, i in lookup.items():
            previous = lookup.get(current - lag)
            if previous is not None:
                cross += np.outer(score[i], score[previous])
        covariance += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
    return (covariance + covariance.T) / 2


def score_covariance(fit: dict, sample: pd.DataFrame, contrast: np.ndarray, outcome: int,
                     method: str, bandwidth: int) -> tuple[np.ndarray, int, float]:
    influence = influences(fit, contrast) * fit["residual"][:, [outcome]]
    period = "month" if "month" in sample and sample["month"].notna().all() else "quarter"
    freq = "M" if period == "month" else "Q"
    dates = pd.PeriodIndex(sample[period], freq=freq).asi8
    unique = np.unique(dates)
    n, k, periods = fit["n"], fit["k"], len(unique)
    if method == "Sector cluster":
        units = sample.sector_id.astype(str).to_numpy()
        sectors = sorted(set(units))
        assert len(sectors) > 1
        scores = np.vstack([influence[units == sector].sum(axis=0) for sector in sectors])
        correction = len(sectors)/(len(sectors)-1) * (n-1)/(n-k)
        covariance, degrees = scores.T @ scores, len(sectors)-1
    else:
        scores = np.vstack([influence[dates == date].sum(axis=0) for date in unique])
        correction = periods/(periods-1) * (n-1)/(n-k)
        covariance, degrees = bartlett_scores(scores, unique, bandwidth), periods-1
        if sample.sector_id.nunique() == 1:
            check(correction, periods/(periods-k), "Singleton correction identity", atol=1e-12, rtol=1e-12)
    return covariance * correction, degrees, correction


def inference(row, coefficient: float, variance: float, degrees: int, hc0: float):
    threshold = 1e-12 * max(hc0, 0.)
    assert np.isfinite(variance) and variance >= -max(threshold, np.finfo(float).tiny), ("Invalid reported-effect variance", variance)
    se = math.sqrt(max(variance, 0.))
    degenerate = abs(variance) <= threshold or degrees <= 0 or hc0 <= 0
    critical = student_t.ppf(.975, degrees)
    probability = np.nan if degenerate else float(2 * student_t.sf(abs(coefficient/se), degrees))
    expected = [coefficient, se, probability, np.nan if degenerate else coefficient-critical*se, np.nan if degenerate else coefficient+critical*se]
    check(row[["coefficient", "std_error", "p_value", "ci_lower", "ci_upper"]].to_numpy(float), expected,
          "Coefficient / SE / t p / CI", atol=5e-7)
    if "inference_df" in row:
        assert row.inference_df == degrees
    if "degenerate" in row:
        assert _truth(row.degenerate) == degenerate
    if "hc0_reference_variance" in row:
        check(row.hc0_reference_variance, hc0, "HC0 numerical scale reference", atol=1e-7)
        check(row.zero_variance_threshold, threshold, "Numerical zero-variance threshold", atol=1e-20)
    if "effect_0p1c" in row and pd.notna(row.effect_0p1c):
        check(row.effect_0p1c, .1*coefficient, "+0.1C scale")


def holm(probabilities):
    values = np.asarray(probabilities, float)
    order = np.argsort(values, kind="stable")
    result = np.empty_like(values)
    result[order] = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return result


def parent_path(context: str) -> Path:
    if context in COMPLETE_VA:
        return parent_path(COMPLETE_VA[context])
    if context in MATCHED_TOTALS:
        return parent_path(MATCHED_TOTALS[context][0])
    if context == "aggregate_va":
        return ROOT / "outputs/aggregate_va/weather/samples/G1.csv"
    if context in {"va", "iip-monthly", "iip-quarterly"}:
        return ROOT / f"outputs/outcomes/{context}/weather/common_sample.csv"
    return ROOT / f"outputs/granular/{context}/long/samples/G1.csv"


def calendar_policy(policy: dict, dates: pd.PeriodIndex) -> dict:
    assert policy["intervals_inclusive"] is True and policy["variants"] == VARIANTS
    assert policy["controls"] == WEATHER and policy["outcomes"] == OUTCOMES
    freq = dates.freqstr
    horizon = 12 if freq == "M" else 4
    events = []
    for event in policy["events"]:
        start, end = pd.Period(event["start"], "M"), pd.Period(event["end"], "M")
        assert start <= end and event["source_urls"]
        covered = {month.ordinal if freq == "M" else month.asfreq("Q").ordinal
                   for month in pd.period_range(start, end, freq="M")}
        events.append((event["event_id"], covered))
    endpoint, window = [], []
    for date in dates:
        endpoint.append([event for event, covered in events if date.ordinal in covered])
        interval = set(range(date.ordinal-horizon, date.ordinal+1))
        window.append([event for event, covered in events if not interval.isdisjoint(covered)])
    return {"event_ids": endpoint, "growth_event_ids": window,
            "all_eligible": np.ones(len(dates), dtype=bool),
            "event_dates_excluded": np.array([not value for value in endpoint]),
            "strict_growth_window_clean": np.array([not value for value in window]),
            "base_period": [str(date-horizon) for date in dates], "horizon": horizon}


def audit_raw_parents() -> tuple[dict, dict]:
    """Reuse only existing independent source parsers, not producer transforms."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_granular import _raw_levels, _quarterly_levels, _raw_weather
    sa_va, va_names, va_meta = _raw_levels("va", "sa")
    nsa_va, _, _ = _raw_levels("va", "nsa")
    sa_iip, _, iip_meta = _raw_levels("iip", "sa")
    nsa_iip, _, _ = _raw_levels("iip", "nsa")
    level_sets = {("va", "Q"): (sa_va, nsa_va), ("iip", "M"): (sa_iip, nsa_iip),
                  ("iip", "Q"): (_quarterly_levels(sa_iip), _quarterly_levels(nsa_iip))}
    weather = _raw_weather()
    parents, summary = {}, {"va": va_meta, "iip": iip_meta, "contexts": []}
    for context, count in CONTEXTS.items():
        freq = "M" if "monthly" in context else "Q"
        period, horizon = ("month", 12) if freq == "M" else ("quarter", 4)
        if context in COMPLETE_VA:
            # Build the extra component independently from the raw row and raw
            # weather, using the original panel only for its eligibility dates.
            base = parents[COMPLETE_VA[context]].copy()
            cols = ["sector_id", "sector", period, "time", "year", "season"] + WEATHER + OUTCOMES
            frame = base[cols].copy()
            frame["output_sa"] = frame.y_raw_level
            frame["output_nsa"] = [nsa_va[str(row.sector_id)][getattr(row, period)] for row in frame.itertuples()]
            dwelling_rows = []
            for date in sorted(base[period].unique()):
                sa, nsa = sa_va["1.3"], nsa_va["1.3"]
                current, unadjusted = sa[date], nsa[date]
                time = int(base.loc[base[period].eq(date), "time"].iloc[0])
                # Propagate the already independently raw-verified parent
                # weather numbers exactly, avoiding artificial 1e-14 differences
                # between equivalent summation algorithms across components.
                common_weather = base.loc[base[period].eq(date), WEATHER].iloc[0].to_dict()
                dwelling_rows.append({"sector_id": "1.3", "sector": va_names["1.3"], period: date,
                    "time": time, "year": date.year, "season": date.quarter, **common_weather,
                    "y_short": 100*math.log(current/sa[date-1]),
                    "y_yoy": 100*math.log(current/sa[date-4]),
                    "y_yoy_nsa": 100*math.log(unadjusted/nsa[date-4]),
                    "y_log_level": 100*math.log(current), "y_raw_level": current,
                    "output_sa": current, "output_nsa": unadjusted})
            frame = pd.concat([frame, pd.DataFrame(dwelling_rows)], ignore_index=True)
        else:
            frame = read(parent_path(context))
        frame[period] = pd.PeriodIndex(frame[period], freq=freq)
        frame = frame.sort_values(["sector_id", period]).reset_index(drop=True)
        assert not frame.duplicated(["sector_id", period]).any()
        assert frame.sector_id.nunique() == count
        is_va = context.startswith("va") or context.startswith("aggregate_va")
        catalog = read(ROOT / ("data/processed/sector_catalog.csv" if is_va else "data/processed/iip/sector_catalog.csv"))
        if count == 1:
            expected_ids = {"1.4" if is_va else "1"}
        else:
            detailed = context.endswith("detailed") or context == "va20"
            expected_ids = set(catalog.loc[catalog["detailed_panel" if detailed else "broad_panel"], "sector_id"])
            if context in COMPLETE_VA:
                expected_ids.add("1.3")
        assert set(frame.sector_id) == expected_ids, (context, "Taxonomy does not match official leaf selection")
        source = level_sets[("va" if is_va else "iip", freq)]
        expected = {outcome: [] for outcome in OUTCOMES}
        expected_weather = {name: [] for name in WEATHER}
        for row in frame.itertuples():
            sid, date = str(row.sector_id), getattr(row, period)
            sa, nsa = source[0][sid], source[1][sid]
            current, unadjusted = sa[date], nsa[date]
            expected["y_short"].append(100*math.log(current/sa[date-1]) if date-1 in sa else np.nan)
            expected["y_yoy"].append(100*math.log(current/sa[date-horizon]) if date-horizon in sa else np.nan)
            expected["y_yoy_nsa"].append(100*math.log(unadjusted/nsa[date-horizon]) if date-horizon in nsa else np.nan)
            expected["y_log_level"].append(100*math.log(current))
            expected["y_raw_level"].append(current)
            for name in WEATHER:
                expected_weather[name].append(weather[freq][date][name])
        for name, values in expected.items():
            check(frame[name].to_numpy(float), values, "Raw calendar-lookup outcome: " + name, atol=3e-10, rtol=2e-12)
        for name, values in expected_weather.items():
            check(frame[name].to_numpy(float), values, "Raw NEA weather: " + name, atol=5e-12, rtol=1e-12)
        # Parent eligibility may precede complete YoY histories; require a common
        # finite mask across all sectors/outcomes before applying event calendars.
        finite = np.isfinite(frame[OUTCOMES + WEATHER].to_numpy(float)).all(axis=1)
        eligible = frame.loc[finite].copy()
        dates = eligible.groupby(period).sector_id.nunique()
        complete = dates.index[dates.eq(count)]
        eligible = eligible[eligible[period].isin(complete)].reset_index(drop=True)
        if context in MATCHED_TOTALS:
            template = parents[MATCHED_TOTALS[context][1]]
            eligible = eligible[eligible[period].isin(template[period])].reset_index(drop=True)
        assert eligible.groupby(period).sector_id.nunique().eq(count).all()
        assert eligible.groupby(period)[WEATHER].nunique().eq(1).all().all(), (context, "Weather differs across sectors")
        assert np.ptp(eligible.time.to_numpy() - eligible[period].astype("int64").to_numpy()) == 0
        parents[context] = eligible
        summary["contexts"].append({"context": context, "parent_source": str(parent_path(context).relative_to(ROOT)),
                                   "parent_sha256": sha256(parent_path(context)), "parent_rows": len(frame),
                                   "complete_rows": len(eligible), "complete_periods": eligible[period].nunique(),
                                   "match_template": str(parent_path(MATCHED_TOTALS[context][1]).relative_to(ROOT)) if context in MATCHED_TOTALS else "",
                                   "includes_dwellings": context.startswith("aggregate_va") or context in COMPLETE_VA,
                                   "dwellings_independently_derived_from_raw": context in COMPLETE_VA,
                                   "start": str(eligible[period].min()), "end": str(eligible[period].max())})
    return parents, summary


def design_formula(model: str, sectors: int) -> str:
    weather = "temp_c * C(sector_id) + rain_100mm + rh_pct" if model == "H1" else "temp_c + rain_100mm + rh_pct"
    nuisance = "C(sector_id) * C(season) + C(year)" if model != "separate" and sectors > 1 else "C(season) + C(year)"
    return weather + " + " + nuisance


def rebuild_design(sample: pd.DataFrame, model: str, names: list[str]) -> np.ndarray:
    frame = sample.copy()
    frame["sector_id"] = frame.sector_id.astype(str)
    independent = dmatrix(design_formula(model, frame.sector_id.nunique()), frame,
                          return_type="dataframe", NA_action="raise")
    if names[0].startswith("sector_season["):
        # Interpret named coordinates one column at a time. The producer uses
        # tensor-product arrays; this audit also checks the raw Patsy FE span.
        centered = {term: frame[term].to_numpy(float)-frame[term].mean() for term in WEATHER}
        columns = []
        for name in names:
            if match := re.fullmatch(r"sector_season\[([^,]+),(\d+)\]", name):
                columns.append((frame.sector_id.eq(match[1]) & frame.season.eq(int(match[2]))).to_numpy(float))
            elif match := re.fullmatch(r"year\[(\d+)\]", name):
                columns.append(frame.year.eq(int(match[1])).to_numpy(float))
            elif name in WEATHER:
                columns.append(centered[name])
            elif match := re.fullmatch(r"temp_c:C\(sector_id\)\[T\.(.+)\]", name):
                columns.append(centered["temp_c"] * frame.sector_id.eq(match[1]).to_numpy(float))
            else:
                raise AssertionError(("Unrecognized logical design coordinate", name))
        x = np.column_stack(columns)
        assert x.shape[1] == independent.shape[1]
        with threadpool_limits(limits=1):
            q, _ = qr(x/np.linalg.norm(x, axis=0), mode="economic")
        check(q @ (q.T @ independent.to_numpy(float)), independent.to_numpy(float),
              "Raw-formula Patsy FE / centered complete-cell span", atol=2e-9, rtol=1e-10)
        return x
    assert set(independent.columns) == set(names) and len(names) == len(independent.columns), (model, "Unexpected/omitted design terms", set(names)^set(independent.columns))
    return independent[names].to_numpy(float)


def total_temperature_contrast(names: list[str], sectors: list[str]) -> np.ndarray:
    contrast = np.zeros((len(sectors), len(names)))
    contrast[:, names.index("temp_c")] = 1
    for i, sector in enumerate(sectors[1:], start=1):
        term = f"temp_c:C(sector_id)[T.{sector}]"
        assert term in names
        contrast[i, names.index(term)] = 1
    return contrast


def verify_joint_pairs(slopes: np.ndarray, covariance: np.ndarray, sectors: list[str],
                       joint: pd.Series, pairs: pd.DataFrame, degrees: int, bound: int, hc0: np.ndarray):
    constraints = np.eye(len(sectors))[1:] - np.eye(len(sectors))[[0]]
    difference = constraints @ slopes
    restriction_covariance = constraints @ covariance @ constraints.T
    singular = np.linalg.svd(restriction_covariance, compute_uv=False)
    reference_scale = float(np.linalg.norm(constraints @ hc0 @ constraints.T, 2))
    zero_threshold = 1e-12 * reference_scale
    cutoff = max(float(singular.max()) * 1e-10, zero_threshold)
    numerical_rank = int(np.sum(singular > cutoff)) if singular.size and singular.max() > 0 else 0
    rank = min(numerical_rank, bound)
    for field, value in [("restrictions", len(sectors)-1), ("numerator_df", len(sectors)-1),
                         ("denominator_df", degrees), ("covariance_rank", rank), ("structural_rank_bound", bound)]:
        if field in joint and pd.notna(joint[field]):
            assert int(joint[field]) == value, ("Joint rank/DF metadata", field, joint[field], value)
    if rank < len(sectors)-1:
        assert "unavailable" in str(joint.status)
        for field in ("p_value", "wald_chi2", "f_statistic"):
            assert pd.isna(joint[field]), ("Invalid singular Wald inference", field, joint[field])
    else:
        assert str(joint.status) in {"available", "estimated"}
        statistic = float(difference @ np.linalg.solve(restriction_covariance, difference))
        assert statistic >= 0
        check(joint.wald_chi2, statistic, "Omnibus Wald statistic", atol=2e-5, rtol=3e-6)
        check(joint.f_statistic, statistic/(len(sectors)-1), "Omnibus F statistic", atol=2e-6, rtol=3e-6)
        check(joint.p_value, f_distribution.sf(statistic/(len(sectors)-1), len(sectors)-1, degrees), "Omnibus F p", atol=2e-8)
    for field, value in [("reference_covariance_scale", reference_scale), ("zero_covariance_threshold", zero_threshold), ("rank_cutoff", cutoff)]:
        if field in joint:
            check(joint[field], value, "Omnibus numerical rank scale", atol=1e-12, rtol=2e-6)
    first = "first_sector_id" if "first_sector_id" in pairs else "sector_i"
    second = "second_sector_id" if "second_sector_id" in pairs else "sector_j"
    observed = {(str(row[first]), str(row[second])) for _, row in pairs.iterrows()}
    assert observed == set(itertools.combinations(sectors, 2))
    probabilities = []
    for _, row in pairs.iterrows():
        i, j = sectors.index(str(row[first])), sectors.index(str(row[second]))
        delta = float(slopes[i]-slopes[j])
        variance = float(covariance[i, i]+covariance[j, j]-2*covariance[i, j])
        contrast = np.eye(len(sectors))[i]-np.eye(len(sectors))[j]
        reference_variance = float(contrast @ hc0 @ contrast)
        threshold = 1e-12*max(reference_variance, 0.)
        assert variance >= -max(threshold, np.finfo(float).tiny)
        se = math.sqrt(max(variance, 0.))
        degenerate = abs(variance) <= threshold or reference_variance <= 0
        probability = np.nan if degenerate else float(2*student_t.sf(abs(delta/se), degrees))
        probabilities.append(probability)
        point_field = "difference" if "difference" in row else "coefficient"
        check(row[point_field], delta, "Pairwise slope difference")
        check(row.std_error, se, "Pairwise difference SE")
        check(row.p_value, probability, "Pairwise t p", atol=3e-8)
        if "ci_lower" in row:
            critical = student_t.ppf(.975, degrees)
            check(row[["ci_lower", "ci_upper"]].to_numpy(float), [np.nan, np.nan] if degenerate else [delta-critical*se, delta+critical*se], "Pairwise confidence intervals", atol=5e-7)
    field = "p_value_holm" if "p_value_holm" in pairs else "holm_p_value"
    adjusted = holm(np.nan_to_num(np.asarray(probabilities), nan=1.))
    adjusted[np.isnan(probabilities)] = np.nan
    check(pairs[field].to_numpy(float), adjusted, "All-pairs within-family Holm", atol=3e-8)
    return {"rank": rank, "bound": bound, "restrictions": len(sectors)-1, "available": rank == len(sectors)-1}


def _truth(value) -> bool:
    if isinstance(value, str):
        assert value.lower() in {"true", "false"}, value
        return value.lower() == "true"
    return bool(value)


def _event_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    text = str(value)
    if text.startswith("["):
        return json.loads(text)
    return text.split("|") if "|" in text else text.split(";") if ";" in text else text.split(",")


def compare_sample(actual: pd.DataFrame, expected: pd.DataFrame, period: str):
    keys = ["sector_id", "time"]
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    assert len(actual) == len(expected) and not actual.duplicated(keys).any()
    assert set(expected.columns) <= set(actual.columns), ("Missing parent sample columns", set(expected)-set(actual))
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]) and column not in ID_COLUMNS:
            check(actual[column].to_numpy(float), expected[column].to_numpy(float), "Exact parent/sample numeric values", atol=5e-11, rtol=2e-12)
        else:
            a = actual[column].fillna("__NA__").astype(str).to_numpy()
            b = expected[column].fillna("__NA__").astype(str).to_numpy()
            assert np.array_equal(a, b), ("Parent/sample strings differ", column)
    freq = "M" if period == "month" else "Q"
    assert np.ptp(actual.time.to_numpy()-pd.PeriodIndex(actual[period], freq=freq).asi8) == 0


def audit_samples(parents: dict, policy: dict) -> tuple[dict, list]:
    assert json.loads((OUT / "policy.json").read_text()) == policy
    mask_rows = read(OUT / "masks.csv")
    assert set(mask_rows.context) == set(CONTEXTS)
    samples, records = {}, []
    for context, parent in parents.items():
        period, freq = ("month", "M") if "monthly" in context else ("quarter", "Q")
        dates = pd.PeriodIndex(parent[period].drop_duplicates().sort_values(), freq=freq)
        policy_masks = calendar_policy(policy, dates)
        saved = mask_rows[mask_rows.context.eq(context)].copy()
        assert not saved.period.duplicated().any()
        saved = saved.set_index("period").loc[dates.astype(str)]
        assert len(saved) == len(dates)
        assert saved.base_period.astype(str).tolist() == policy_masks["base_period"]
        assert saved.event_ids_at_t.map(_event_list).tolist() == policy_masks["event_ids"]
        assert saved.intersecting_event_ids_in_growth_window.map(_event_list).tolist() == policy_masks["growth_event_ids"]
        assert np.array_equal(saved.retain_event_dates.map(_truth), policy_masks["event_dates_excluded"])
        assert np.array_equal(saved.retain_strict.map(_truth), policy_masks["strict_growth_window_clean"])
        date_time = parent.drop_duplicates(period).set_index(period).time.reindex(dates).to_numpy()
        check(saved.time.to_numpy(), date_time, "Mask original calendar time", atol=0., rtol=0.)
        for variant in VARIANTS:
            sample_id = context+"__"+variant
            expected_dates = dates[policy_masks[variant]]
            expected = parent[parent[period].isin(expected_dates)].copy()
            path = OUT / "samples" / f"{sample_id}.csv"
            actual = read(path)
            compare_sample(actual, expected, period)
            assert set(actual[period]) == set(expected_dates.astype(str))
            assert actual.groupby(period).sector_id.nunique().eq(CONTEXTS[context]).all()
            actual = actual.sort_values(["sector_id", "time"]).reset_index(drop=True)
            span = expected_dates[-1].ordinal-expected_dates[0].ordinal+1
            samples[sample_id] = actual
            records.append({"sample_id": sample_id, "context": context, "sample_variant": variant,
                            "n_periods": len(expected_dates), "n_obs": len(actual), "n_sectors": CONTEXTS[context],
                            "start": str(expected_dates[0]), "end": str(expected_dates[-1]),
                            "removed_periods": len(dates)-len(expected_dates), "calendar_span_periods": span,
                            "calendar_gaps": span-len(expected_dates), "source_sample_sha256": sha256(parent_path(context)),
                            "sample_sha256": sha256(path), "stationarity_scope": "unresolved: gapped exact sample; no compressed-calendar screen" if span>len(expected_dates) else "not certified by this numerical audit"})
    return samples, records


def metadata(row, sample: pd.DataFrame, fit: dict, outcome: str, *, sector_id: str | None = None):
    period = "month" if "monthly" in str(row.context) else "quarter"
    freq = "M" if period == "month" else "Q"
    dates = pd.PeriodIndex(sample[period], freq=freq)
    nperiods, sectors = len(dates.unique()), sample.sector_id.nunique()
    expected = {"n_obs": len(sample), "n_sectors": sectors, "n_periods": nperiods,
                "n_parameters": fit["k"], "calendar_span_periods": dates.max().ordinal-dates.min().ordinal+1,
                "calendar_gaps": dates.max().ordinal-dates.min().ordinal+1-nperiods}
    for field, value in expected.items():
        if field in row and pd.notna(row[field]):
            assert int(row[field]) == value, ("Model sample/rank metadata", field, row[field], value)
    if "start" in row:
        assert str(row.start) == str(dates.min()) and str(row.end) == str(dates.max())
    y = sample[outcome] if sector_id is None else sample.loc[sample.sector_id.eq(sector_id), outcome]
    if "mean_y" in row:
        check(row.mean_y, y.mean(), "Exact model outcome mean", atol=3e-10, rtol=2e-12)
    if "sd_y" in row:
        check(row.sd_y, y.std(ddof=1), "Exact model outcome SD", atol=3e-10, rtol=2e-12)
    if "r_squared" in row:
        check(row.r_squared, fit["r_squared"][OUTCOMES.index(outcome)], "Full centered R squared", atol=5e-10, rtol=2e-10)
    if "policy_id" in row:
        assert row.policy_id == json.loads((OUT / "policy.json").read_text())["policy_id"]
    for field in ("policy_sha256", "policy_hash"):
        if field in row:
            assert row[field] == sha256(OUT / "policy.json")
    if "estimator_version" in row:
        import importlib.metadata
        assert str(row.estimator_version) == importlib.metadata.version("pyfixest")


def _cov_value(frame: pd.DataFrame) -> str:
    for name in ("covariance_value", "value", "covariance_estimate", "covariance"):
        if name in frame and pd.api.types.is_numeric_dtype(frame[name]):
            return name
    raise AssertionError("No covariance value field")


def _method_field(frame: pd.DataFrame) -> str:
    return "covariance_type" if "covariance_type" in frame else "covariance"


def covariance_matrix(frame: pd.DataFrame, labels: list[str], first: str, second: str) -> np.ndarray:
    assert not frame.duplicated([first, second]).any()
    assert set(frame[first]) == set(labels) == set(frame[second]) and len(frame) == len(labels)**2
    return frame.pivot(index=first, columns=second, values=_cov_value(frame)).loc[labels, labels].to_numpy(float)


def audit_protected_files() -> dict:
    candidates = [OUT / name for name in ("protected_inputs_reference.json", "protected_files.json", "original_inputs_reference.json")]
    path = next((value for value in candidates if value.exists()), None)
    if path is None:
        path = OUT / "manifest.json"
    payload = json.loads(path.read_text())
    files = payload.get("files", payload.get("protected_files", payload.get("protected_sources_sha256", {})))
    assert files, "Empty protected reference"
    for relative, record in files.items():
        current = ROOT / relative
        assert current.is_file(), ("Protected file missing", relative)
        expected_hash = record if isinstance(record, str) else record["sha256"]
        assert sha256(current) == expected_hash, ("Protected file changed", relative)
        length = None if isinstance(record, str) else record.get("bytes", record.get("size_bytes"))
        if length is not None:
            assert current.stat().st_size == length
    return {"reference_sha256": sha256(path), "protected_files_unchanged": len(files)}


def verify_source_metadata(specifications: pd.DataFrame, parents: dict) -> list[dict]:
    """Check declared matching templates separately from the sample-value audit."""
    records=[]
    for context,parent in parents.items():
        rows=specifications[specifications.context.eq(context)]
        source=read(parent_path(context))
        period="month" if "monthly" in context else "quarter"
        assert rows.source_parent_n_obs.eq(len(source)).all()
        assert rows.source_parent_n_periods.eq(source[period].nunique()).all()
        applied=context in MATCHED_TOTALS
        assert rows.calendar_match_applied.map(_truth).eq(applied).all()
        if applied:
            template_context=MATCHED_TOTALS[context][1]
            template_path=parent_path(template_context)
            assert rows.match_template_source_csv.eq(str(template_path.relative_to(ROOT))).all()
            assert rows.match_template_sha256.eq(sha256(template_path)).all()
            assert set(parent[period])==set(parents[template_context][period])
        else:
            assert rows.match_template_source_csv.fillna("").eq("").all()
            assert rows.match_template_sha256.fillna("").eq("").all()
        target=next((value for value in rows.matched_to.dropna().unique()),None)
        if target:
            assert target in parents and set(parent[period])==set(parents[target][period])
        records.append({"context":context,"source_rows":len(source),"source_periods":source[period].nunique(),
                        "eligible_rows":len(parent),"eligible_periods":parent[period].nunique(),
                        "additional_calendar_match":applied,"matched_to":target})
    return records


def _take_group(groups: dict, key: str) -> pd.DataFrame:
    assert key in groups, ("Missing output specification", key)
    return groups[key]


def _sample_for_design(sample: pd.DataFrame, archive, unit_id: str | None):
    result = sample if unit_id is None else sample[sample.sector_id.eq(unit_id)]
    units = archive["units"].astype(str)
    times = archive["time"].astype(int)
    assert len(units) == len(result) and len(set(zip(units, times))) == len(result)
    assert set(zip(units, times)) == set(zip(result.sector_id, result.time))
    result = result.set_index(["sector_id", "time"]).loc[list(zip(units, times))].reset_index()
    period = "month" if "monthly" in str(archive["periods"][0]) else None
    assert np.array_equal(result.sector_id.to_numpy(str), units)
    assert np.array_equal(result.time.to_numpy(), times)
    return result


def _verify_specification(row, sample: pd.DataFrame, fit: dict, parent: pd.DataFrame):
    metadata(row, sample, fit, row.outcome)
    context = row.context
    period, freq = ("month", "M") if "monthly" in context else ("quarter", "Q")
    t = sample.time.nunique()
    time_frame = sample.drop_duplicates("time").sort_values("time")
    temporal = dmatrix("temp_c + rain_100mm + rh_pct + C(season) + C(year)", time_frame,
                       return_type="dataframe", NA_action="raise").to_numpy(float)
    assert np.linalg.matrix_rank(temporal) == temporal.shape[1]
    assert int(row.time_design_rank) == temporal.shape[1] and int(row.time_residual_df) == t-temporal.shape[1]
    assert int(row.residual_df) == fit["n"]-fit["k"] and int(row.design_rank) == fit["k"]
    effective_parent = parent[parent.sector_id.eq(row.unit_id)] if row.model == "separate" else parent
    assert int(row.parent_n_obs) == len(effective_parent)
    assert int(row.parent_n_periods) == effective_parent.time.nunique()
    assert int(row.n_removed_obs) == len(effective_parent)-len(sample)
    assert int(row.n_removed_periods) == effective_parent.time.nunique()-t
    assert row.source_sample_csv == str(parent_path(context).relative_to(ROOT))
    assert row.source_sample_sha256 == sha256(parent_path(context))
    assert row.sample_sha256 == sha256(ROOT / row.sample_csv)
    assert row.eligible_parent_sha256 == sha256(ROOT / row.eligible_parent_csv)
    assert row.eligible_parent_csv == f"outputs/crisis_sensitivity/parents/{context}.csv"
    assert json.loads(row.controls) == WEATHER and row.control_set == "weather"
    assert all(term in row.formula for term in WEATHER) and "C(year)" in row.formula
    assert "wind" not in row.formula and "pm25" not in row.formula and "trend" not in row.formula
    assert ("* C(sector_id)" in row.formula.split("+ rain_100mm")[0]) == (row.model == "H1")
    assert str(row.status) == "estimated"
    if context.startswith(("va", "aggregate_va")):
        included = context in COMPLETE_VA or context.startswith("aggregate_va")
        assert _truth(row.includes_dwellings) == included
        assert _truth(row.dwellings_added_from_raw) == (context in COMPLETE_VA)
        if context in COMPLETE_VA:
            assert str(row.dwellings_source_sector_id) == "1.3"
            extra = json.loads(row.additional_raw_source_hashes)
            assert set(extra) == {f"data/raw/va_{adjustment}_{suffix}.json" for adjustment in ("sa", "nsa") for suffix in ("tabledata", "metadata")}
            assert all(sha256(ROOT/path) == digest for path, digest in extra.items())
    # A gapped history must never be represented as passing a conventional
    # contiguous ADF/KPSS screen by silently concatenating retained observations.
    if row.calendar_gaps > 0:
        assert "Not cleared" in str(row.stationarity_status) and "gap" in str(row.stationarity_status)


def audit_native_receipt(expected: list[dict]) -> dict:
    log_path = OUT / "pyfixest/fit_log.csv"
    log = read(log_path)
    receipt = json.loads((OUT / "pyfixest/estimator_manifest.json").read_text())
    assert receipt["fit_log_sha256"] == sha256(log_path)
    assert receipt["native_fits_recorded"] == len(expected) == len(log)
    expected_frame = pd.DataFrame(expected)
    assert not log.duplicated(["label", "outcome"]).any()
    assert Counter(log[["label", "outcome"]].itertuples(index=False, name=None)) == Counter(expected_frame[["label", "outcome"]].itertuples(index=False, name=None))
    log = log.set_index(["label", "outcome"])
    for row in expected:
        actual = log.loc[(row["label"], row["outcome"])]
        for name in ("n_obs", "n_parameters", "design_sha256", "outcome_sha256", "coefficients_sha256", "residuals_sha256", "coefficient_names_sha256"):
            assert actual[name] == row[name], ("Native receipt identity", row["label"], row["outcome"], name)
    assert log.native_class.eq("Feols").all() and log.native_module.str.startswith("pyfixest").all()
    assert log.native_dropped_rows.eq(0).all() and log.native_dropped_coefficients.eq(0).all()
    for name in ("native_mapped_coefficient_max_abs_error", "native_residual_max_abs_error", "native_outcome_max_abs_error", "native_response_max_abs_error", "native_design_max_abs_error"):
        assert log[name].eq(0).all(), ("Native provenance discrepancy", name)
    assert log.native_outcome_sha256.eq(log.outcome_sha256).all()
    assert log.native_residuals_sha256.eq(log.residuals_sha256).all()
    assert log.groupby("api_call_id").size().eq(5).all()
    assert log.native_design_basis.eq("orthonormal SVD preconditioned full LSDV").all()
    assert (log.native_design_orthogonality_max_abs_error < 1e-10).all()
    return {"native_fits": len(log), "native_api_calls": int(log.api_call_id.nunique()),
            "native_fit_log_sha256": sha256(log_path), "no_native_row_or_coefficient_drops": True}


def verify_crisis_sensitivity() -> dict:
    """Audit every new crisis fit without invoking any analytical writer."""
    ERRORS.clear()
    producer_manifest = json.loads((OUT / "manifest.json").read_text())
    assert producer_manifest["contexts"] == len(CONTEXTS) == 20
    assert producer_manifest["status"] != "running", "Wait for the complete producer output set"
    policy = json.loads((OUT / "policy.json").read_text())
    assert policy == json.loads(POLICY.read_text())
    assert producer_manifest["policy_source_sha256"] == sha256(POLICY)
    assert producer_manifest["policy_sha256"] == sha256(OUT / "policy.json")
    protected = audit_protected_files()
    parents, raw_summary = audit_raw_parents()
    for context, parent in parents.items():
        compare_sample(read(OUT / "parents" / f"{context}.csv"), parent, "month" if "monthly" in context else "quarter")
    samples, sample_records = audit_samples(parents, policy)
    specifications = read(OUT / "specifications.csv")
    source_metadata = verify_source_metadata(specifications, parents)
    assert specifications.spec_id.is_unique and set(specifications.outcome) == set(OUTCOMES)
    assert set(specifications.context) == set(CONTEXTS) and set(specifications.sample_variant) == set(VARIANTS)
    assert specifications.status.eq("estimated").all(), "Every requested long-history core design should be identified"
    assert len(specifications) == 2655 and specifications.fit_group_id.nunique() == 531
    tables = {name: read(OUT / f"{name}.csv") for name in ("estimates", "sector_estimates", "separate_estimates", "all_coefficients",
              "coefficient_covariances", "sector_covariances", "interaction_components", "component_covariances", "joint_tests", "pairwise_tests")}
    groups = {name: {str(key): group for key, group in frame.groupby("spec_id", sort=False)} for name, frame in tables.items()}
    native_expected, fit_records, joint_records = [], [], []
    separate_cache, common_slopes, h1_slopes = {}, {}, {}
    coefficient_rows = covariance_cells = 0
    with threadpool_limits(limits=1):
        for group_id, specs in specifications.groupby("fit_group_id", sort=True):
            assert len(specs) == 5 and set(specs.outcome) == set(OUTCOMES)
            first = specs.iloc[0]
            context, variant, model = first.context, first.sample_variant, first.model
            period, horizon = ("month", 12) if "monthly" in context else ("quarter", 4)
            full_sample = samples[context+"__"+variant]
            unit_id = str(first.unit_id) if model == "separate" else None
            path = OUT / "designs" / f"{group_id}.npz"
            with np.load(path, allow_pickle=False) as archive:
                assert list(archive["outcomes"].astype(str)) == OUTCOMES
                sample = _sample_for_design(full_sample, archive, unit_id)
                assert list(archive["periods"].astype(str)) == sample[period].astype(str).tolist()
                names = list(archive["coef_names"].astype(str))
                x = rebuild_design(sample, model, names)
                y = sample[OUTCOMES].to_numpy(float)
                check(archive["X"], x, "Every saved logical design cell", atol=2e-11, rtol=1e-12)
                check(archive["Y"], y, "Every saved response cell", atol=0., rtol=0.)
                fit = independent_qr(x, y, names)
                saved_beta, saved_residual = archive["coefficients"], archive["residuals"]
                check(saved_beta, fit["beta"], "All native / nuisance coefficients", atol=3e-7, rtol=2e-7)
                check(saved_residual, fit["residual"], "Every native residual", atol=5e-7, rtol=1e-8)
                design_hash = array_sha256(archive["X"])
                name_hash = hashlib.sha256(json.dumps(names, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                for j, outcome in enumerate(OUTCOMES):
                    native_expected.append({"label": "Crisis sensitivity "+group_id, "outcome": outcome,
                        "n_obs": len(sample), "n_parameters": len(names), "design_sha256": design_hash,
                        "outcome_sha256": array_sha256(archive["Y"][:, j]), "coefficients_sha256": array_sha256(saved_beta[:, j]),
                        "residuals_sha256": array_sha256(saved_residual[:, j]), "coefficient_names_sha256": name_hash})
            sectors, dates = sorted(sample.sector_id.unique()), sample.drop_duplicates("time").sort_values("time")
            t = len(dates)
            selected_names = [name for name in names if name in WEATHER or name.startswith("temp_c:C(sector_id)")]
            selected_positions = [names.index(name) for name in selected_names]
            contrast = np.eye(len(names))[selected_positions]
            influence = influences(fit, contrast)
            if model == "separate":
                separate_cache.setdefault((context, variant), {})[unit_id] = {
                    "temperature_beta": fit["beta"][names.index("temp_c")], "residuals": fit["residual"],
                    "temperature_influence": influences(fit, np.eye(len(names))[[names.index("temp_c")]])[:, 0],
                    "time": sample.time.to_numpy(), "k": fit["k"]}
            for j, outcome in enumerate(OUTCOMES):
                spec = specs[specs.outcome.eq(outcome)].iloc[0]
                _verify_specification(spec, sample, fit, parents[context])
                full_coefficients = _take_group(groups["all_coefficients"], spec.spec_id).set_index("term")
                assert set(full_coefficients.index) == set(names) and len(full_coefficients) == len(names)
                check(full_coefficients.loc[names, "coefficient"].to_numpy(), fit["beta"][:, j], "Every exported coefficient", atol=3e-7)
                coefficient_rows += len(names)
                if model == "common":
                    common_slopes[(context, variant, outcome)] = fit["beta"][names.index("temp_c"), j]
                covariance_rows = _take_group(groups["coefficient_covariances"], spec.spec_id)
                expected_bands = {(0, "Sector cluster"), (horizon, "Driscoll-Kraay"), (2*horizon, "Driscoll-Kraay")} if model == "common" and len(sectors)>1 else {
                    (horizon, "Calendar HAC" if len(sectors)==1 else "Driscoll-Kraay"),
                    (2*horizon, "Calendar HAC" if len(sectors)==1 else "Driscoll-Kraay")}
                method_column = _method_field(covariance_rows)
                assert set(zip(covariance_rows.hac_lags, covariance_rows[method_column])) == expected_bands
                for (method, bandwidth), saved_cov in covariance_rows.groupby([method_column, "hac_lags"], sort=False):
                    bandwidth = int(bandwidth)
                    covariance, degrees, correction = score_covariance(fit, sample, contrast, j, method, bandwidth)
                    scores = influence * fit["residual"][:, [j]]
                    hc0 = scores.T @ scores
                    check(covariance_matrix(saved_cov, selected_names, "term_i", "term_j"), covariance,
                          "Every selected coefficient covariance", atol=5e-6, rtol=3e-7)
                    check(saved_cov.pivot(index="term_i", columns="term_j", values="hc0_reference_value").loc[selected_names, selected_names].to_numpy(), hc0,
                          "Every HC0 numerical-scale cell", atol=2e-6, rtol=3e-7)
                    check(saved_cov.finite_sample_correction.to_numpy(), np.full(len(saved_cov), correction), "Method-specific finite correction", atol=1e-12, rtol=1e-12)
                    covariance_cells += len(saved_cov)
                    if model != "H1":
                        filename = "separate_estimates" if model == "separate" else "estimates"
                        points = _take_group(groups[filename], spec.spec_id)
                        points = points[points.covariance.eq(method)&points.hac_lags.eq(bandwidth)].set_index("term")
                        assert set(points.index) == set(WEATHER) and len(points) == 3
                        for term in WEATHER:
                            index = selected_names.index(term)
                            row = points.loc[term]
                            inference(row, fit["beta"][names.index(term), j], covariance[index, index], degrees, hc0[index, index])
                            metadata(row, sample, fit, outcome)
                            assert _truth(row.main_inference) == (method == "Sector cluster" if model=="common" and len(sectors)>1 else bandwidth==horizon)
                    else:
                        totals = total_temperature_contrast(names, sectors)
                        transform = totals[:, selected_positions]
                        beta = totals @ fit["beta"][:, j]
                        total_cov, total_hc0 = transform @ covariance @ transform.T, transform @ hc0 @ transform.T
                        h1_slopes[(context, variant, outcome)] = beta
                        slopes = _take_group(groups["sector_estimates"], spec.spec_id)
                        slopes = slopes[slopes.hac_lags.eq(bandwidth)].set_index("sector_id").loc[sectors]
                        assert len(slopes) == len(sectors)
                        for i, sector in enumerate(sectors):
                            row = slopes.loc[sector]
                            inference(row, beta[i], total_cov[i, i], degrees, total_hc0[i, i])
                            metadata(row, sample, fit, outcome, sector_id=sector)
                            base_index = selected_names.index("temp_c")
                            delta_index = None if i==0 else selected_names.index(f"temp_c:C(sector_id)[T.{sector}]")
                            check(row.temp_main, fit["beta"][names.index("temp_c"), j], "H1 temperature baseline")
                            check(row.interaction_delta, 0. if i==0 else fit["beta"][names.index(selected_names[delta_index]), j], "H1 increment")
                            check([row.var_temp_main, row.var_interaction_delta, row.cov_temp_main_delta],
                                  [covariance[base_index, base_index], 0. if i==0 else covariance[delta_index, delta_index], 0. if i==0 else covariance[base_index, delta_index]],
                                  "H1 total-effect variance components", atol=5e-6)
                            check(row.variance, row.var_temp_main+row.var_interaction_delta+2*row.cov_temp_main_delta, "Baseline plus increment total variance", atol=5e-6)
                        sector_cov = _take_group(groups["sector_covariances"], spec.spec_id)
                        sector_cov = sector_cov[sector_cov.hac_lags.eq(bandwidth)]
                        check(covariance_matrix(sector_cov, sectors, "sector_i", "sector_j"), total_cov,
                              "H1 joint total-slope covariance", atol=5e-6, rtol=3e-7)
                        components = _take_group(groups["interaction_components"], spec.spec_id)
                        components = components[components.hac_lags.eq(bandwidth)]
                        estimated = components[components.estimated.map(_truth)].set_index("term")
                        component_names = ["temp_c"]+[f"temp_c:C(sector_id)[T.{sector}]" for sector in sectors[1:]]
                        assert set(estimated.index) == set(component_names)
                        for term in component_names:
                            index = selected_names.index(term)
                            inference(estimated.loc[term], fit["beta"][names.index(term), j], covariance[index,index], degrees, hc0[index,index])
                        normalized = components[~components.estimated.map(_truth)]
                        assert len(normalized)==1 and normalized.coefficient.eq(0).all() and normalized[["std_error","p_value","ci_lower","ci_upper"]].isna().all().all()
                        component_cov = _take_group(groups["component_covariances"], spec.spec_id)
                        component_cov = component_cov[component_cov.hac_lags.eq(bandwidth)]
                        positions = [selected_names.index(term) for term in component_names]
                        check(covariance_matrix(component_cov, component_names, "term_i", "term_j"), covariance[np.ix_(positions,positions)], "Native baseline / increment full covariance", atol=5e-6)
                        joint = _take_group(groups["joint_tests"], spec.spec_id)
                        joint = joint[joint.hac_lags.eq(bandwidth)]
                        pairs = _take_group(groups["pairwise_tests"], spec.spec_id)
                        pairs = pairs[pairs.hac_lags.eq(bandwidth)]
                        assert len(joint)==1
                        result = verify_joint_pairs(beta, total_cov, sectors, joint.iloc[0], pairs, degrees, t-sample.season.nunique()-1, total_hc0)
                        joint_records.append({"spec_id": spec.spec_id, "model": model, "hac_lags": bandwidth, **result})
            fit_records.append({"fit_group_id": group_id, "context": context, "sample_variant": variant, "model": model,
                                "n_obs": len(sample), "n_periods": t, "n_parameters": fit["k"], "outcomes": 5})
            if len(fit_records)%30==0:
                print(f"Crisis independent QR / covariance audit: {len(fit_records)}/531 designs", flush=True)
    # Separate regressions have correlated national-time shocks. Reconstruct the
    # entire joint slope covariance, not independent-error pairwise differences.
    for (context, variant), units in separate_cache.items():
        sectors = sorted(units)
        sample = samples[context+"__"+variant]
        horizon = 12 if "monthly" in context else 4
        first = units[sectors[0]]
        dates, k = first["time"], first["k"]
        assert all(np.array_equal(item["time"], dates) and item["k"]==k for item in units.values())
        t = len(dates)
        for j, outcome in enumerate(OUTCOMES):
            beta = np.array([units[sector]["temperature_beta"][j] for sector in sectors])
            scores = np.column_stack([units[sector]["temperature_influence"]*units[sector]["residuals"][:,j] for sector in sectors])
            hc0 = scores.T @ scores
            spec_id = context+"__"+variant+"__separate_joint__"+outcome
            for bandwidth in (horizon, 2*horizon):
                covariance = bartlett_scores(scores, dates, bandwidth)*t/(t-k)
                rows = _take_group(groups["sector_covariances"], spec_id)
                rows = rows[rows.hac_lags.eq(bandwidth)]
                check(covariance_matrix(rows, sectors, "sector_i", "sector_j"), covariance,
                      "Joint separate-series temperature covariance", atol=5e-6, rtol=3e-7)
                joint = _take_group(groups["joint_tests"], spec_id)
                joint = joint[joint.hac_lags.eq(bandwidth)]
                pairs = _take_group(groups["pairwise_tests"], spec_id)
                pairs = pairs[pairs.hac_lags.eq(bandwidth)]
                assert len(joint)==1
                result = verify_joint_pairs(beta, covariance, sectors, joint.iloc[0], pairs, t-1, t-k, hc0)
                joint_records.append({"spec_id": spec_id, "model": "separate", "hac_lags": bandwidth, **result})
    assert set(common_slopes) >= set(h1_slopes)
    for key, values in h1_slopes.items():
        check(values.mean(), common_slopes[key], "Balanced H1 mean equals common-slope identity", atol=2e-7)
    receipt = audit_native_receipt(native_expected)
    assert receipt["native_fits"]==2655 and receipt["native_api_calls"]==531
    assert not len(read(OUT / "unavailable_specs.csv"))
    protected_after = audit_protected_files()
    assert protected == protected_after
    report = {"status": "passed", "scope": "Independent cached-raw SA/NSA calendar growth and NEA weather; explicit inclusive event windows; named-design reconstruction plus raw Patsy FE span; full QR; unit CR1 and actual-calendar Bartlett covariance; baseline-plus-increment total effects; joint separate-series covariance; direct full-rank Wald and all-pairs Holm; native receipts and protected hashes. Not a Fable execution claim, causal claim, or stationarity certificate.",
              "contexts": len(CONTEXTS), "samples": len(samples), "requested_specs": len(specifications),
              "independent_designs": len(fit_records), "exported_coefficients_checked": coefficient_rows,
              "selected_covariance_cells_checked": covariance_cells, "joint_profile_tests": len(joint_records),
              "unavailable_joint_tests": sum(not row["available"] for row in joint_records),
              "raw_source_audit": raw_summary, "native_receipt": receipt, "protected_hashes": protected,
              "source_and_matching_metadata": source_metadata,
              "numerical_checks": ERRORS, "policy_sha256": sha256(OUT / "policy.json"),
              "policy_interpretation": policy["interpretation"], "stationarity": "Gapped exclusion windows remain explicitly unresolved; no observations selected by outcome, residual, significance or stationarity pretests."}
    pd.DataFrame(fit_records).to_csv(OUT / "verification_fits.csv", index=False)
    pd.DataFrame(sample_records).to_csv(OUT / "verification_samples.csv", index=False)
    pd.DataFrame(joint_records).to_csv(OUT / "verification_joint_tests.csv", index=False)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Crisis independent verification passed: {len(CONTEXTS)} contexts, {len(samples)} samples, {len(specifications)} native fits and every exported slope/covariance.", flush=True)
    return report


if __name__ == "__main__":
    verify_crisis_sensitivity()
