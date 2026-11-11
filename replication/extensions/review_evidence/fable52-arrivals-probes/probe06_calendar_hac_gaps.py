"""Probe 06 - actual-calendar Bartlett HAC12/24 on gapped and short/dynamic designs.

For three saved monthly designs (one explicitly gapped single series, one short
dynamic H1 panel, one short pooled two-channel panel that also carries the
pooled-origin CR1 comparison) this probe

  1. decodes the saved integer time axis against the saved period strings and
     lists every calendar gap;
  2. refits by an SVD pseudoinverse (not the production scaled-QR mapping) and
     rebuilds the influence rows pinv(X)' * residual;
  3. computes the full K x K Bartlett HAC12/24 twice - as an explicit
     autocovariance sum over the zero-filled calendar array and as the dense
     quadratic form A' W A with the Toeplitz Bartlett matrix - and compares
     both to raw_calendar_covariance_0/1 and, after T/(T-K_time), to
     covariance_0/1;
  4. computes the deliberately wrong "compressed" version (gaps closed, lags
     counted over observed dates) and counts the observed-date pairs whose
     kernel weight differs, to show production does not lag across gaps;
  5. tabulates alternative finite-sample scalings / reference laws for the
     selected temperature contrasts strictly as conventions (none is exact);
  6. for the pooled panel, rebuilds the origin-cluster CR1 matrix.

Command: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe06_calendar_hac_gaps.py
"""
from _probe_common import BASE, bartlett_calendar, bartlett_compressed, maxabs, maxrel, scaled_rank, write

import numpy as np
import pandas as pd
from scipy import stats

MONTHLY = BASE / "monthly"
SELECTED = [
    ("aggregate__y_mom_nsa__long__event_dates_excluded__common__own_current", "explicitly gapped single series (event dates removed)"),
    ("region__y_mom_nsa__full__strict_growth_window_clean__H1__dynamic", "short dynamic H1 panel with many gaps"),
    ("region__y_yoy_nsa__full__event_dates_excluded__common__two_channel", "short pooled two-channel panel; also origin-cluster CR1"),
]


def month_label(ordinal):
    return f"{1970 + ordinal // 12:04d}-{ordinal % 12 + 1:02d}"


def gap_map(time):
    observed = np.unique(time)
    runs = []
    for before, after in zip(observed[:-1], observed[1:]):
        if after - before > 1:
            runs.append({"last_observed_before": month_label(int(before)), "first_observed_after": month_label(int(after)), "missing_months": int(after - before - 1)})
    return observed, runs


def pair_weights(observed, bandwidth):
    """Kernel weight attached to each ordered pair of observed dates under calendar and compressed lagging."""
    calendar = np.clip(1.0 - np.abs(np.subtract.outer(observed, observed)) / (bandwidth + 1.0), 0.0, None)
    order = np.arange(len(observed))
    compressed = np.clip(1.0 - np.abs(np.subtract.outer(order, order)) / (bandwidth + 1.0), 0.0, None)
    upper = np.triu_indices(len(observed), 1)
    return {"observed_date_pairs_with_positive_calendar_weight": int(np.sum(calendar[upper] > 0)),
            "observed_date_pairs_with_positive_compressed_weight": int(np.sum(compressed[upper] > 0)),
            "pairs_where_weights_differ": int(np.sum(np.abs(calendar[upper] - compressed[upper]) > 1e-15)),
            "pairs_given_weight_only_by_compressed": int(np.sum((compressed[upper] > 0) & (calendar[upper] == 0))),
            "sum_abs_weight_difference": float(np.sum(np.abs(calendar[upper] - compressed[upper])))}


def local_k_time(matrix, unit, time):
    ranks, blocks = [], []
    for u in sorted(set(unit.tolist())):
        rows = np.flatnonzero(unit == u)
        rows = rows[np.argsort(time[rows], kind="stable")]
        local = matrix[rows]
        ranks.append(scaled_rank(local))
        blocks.append(local[:, np.any(local != 0, axis=0)])
    return max(ranks), scaled_rank(np.column_stack(blocks))


def contrasts_for(archive, spec):
    names = archive["names"].tolist()
    units = sorted(set(archive["unit_ids"].astype(str).tolist()))
    out = []
    if spec["model_spec"] == "two_channel":
        for feature in archive["temperature_features"].tolist():
            vector = np.zeros(len(names)); vector[names.index(feature)] = 1.0
            out.append((feature, vector))
        return out
    lags = sorted(int(name[6:]) for name in names if name.startswith("temp_l") and ":" not in name)
    vector = np.zeros(len(names))
    for lag in lags:
        vector[names.index(f"temp_l{lag}")] = 1.0
    out.append((f"cumulative direct, {'reference origin ' + units[0] if spec['model'] == 'H1' else 'single/pooled slope'}", vector.copy()))
    single = np.zeros(len(names)); single[names.index("temp_l0")] = 1.0
    out.append(("temp_l0 main", single))
    if spec["model"] == "H1":
        last = units[-1]
        total = vector.copy()
        for lag in lags:
            total[names.index(f"temp_l{lag}:C(unit_id)[T.{last}]")] = 1.0
        out.append((f"cumulative direct total, origin {last}", total))
    return out


def run(spec_id, why, specs):
    spec = specs.loc[specs.spec_id.eq(spec_id)].iloc[0]
    archive = np.load(MONTHLY / "designs" / f"{spec_id}.npz")
    matrix, outcome, time, unit = archive["X"], archive["Y"], archive["time"], archive["unit_ids"].astype(str)
    periods = archive["periods"].tolist()
    n, k = matrix.shape
    decoded = np.array([(int(text[:4]) - 1970) * 12 + int(text[5:7]) - 1 for text in periods])
    observed, runs = gap_map(time)
    t, g = len(observed), len(set(unit.tolist()))
    span_months = int(observed[-1] - observed[0] + 1)
    out = {"spec_id": spec_id, "why": why, "n": n, "k": k, "t": t, "g": g, "start": month_label(int(observed[0])), "end": month_label(int(observed[-1])),
           "time_axis_is_month_ordinal": bool(np.array_equal(decoded, time)), "calendar_span_months": span_months,
           "calendar_gaps_mine": span_months - t, "calendar_gaps_ledger": int(spec["calendar_gaps"]), "gap_runs": runs,
           "balanced_dates": bool(n == t * g and len({(u, d) for u, d in zip(unit.tolist(), time.tolist())}) == n)}

    # Independent point fit by SVD pseudoinverse of the column-scaled design.
    scale = np.linalg.norm(matrix, axis=0)
    u_svd, singular, vt = np.linalg.svd(matrix / scale, full_matrices=False)
    assert singular[-1] > singular[0] * max(n, k) * np.finfo(float).eps
    pinv = (vt.T / singular) @ u_svd.T / scale[:, None]
    beta = pinv @ outcome
    residual = outcome - matrix @ beta
    out.update(beta_max_abs_diff_vs_native=maxabs(beta, archive["beta"]), residual_max_abs_diff_vs_native=maxabs(residual, archive["resid"]),
               scaled_condition_number=float(singular[0] / singular[-1]), normal_equation_max_abs=float(np.max(np.abs(matrix.T @ residual))))
    influence = pinv.T * residual[:, None]
    out["hc0_max_rel_diff"] = maxrel(influence.T @ influence, archive["hc0_full"])

    k_time, span_rank = local_k_time(matrix, unit, time)
    factor = t / (t - k_time)
    labels, corrections, degrees = archive["covariance_labels"].tolist(), archive["corrections"], archive["inference_df"]
    out.update(k_time_mine=k_time, k_time_ledger=int(spec["time_design_rank"]), full_span_rank_mine=span_rank,
               full_span_rank_ledger=int(spec["full_cross_origin_temporal_span_rank"]), factor_mine=factor,
               factor_saved=[float(value) for value in corrections], inference_df_saved=[int(value) for value in degrees], labels=labels)

    bins = (time - observed[0]).astype(int)
    filled = np.zeros((span_months, k))
    np.add.at(filled, bins, influence)
    contrasts = contrasts_for(archive, spec)
    out["bandwidths"] = {}
    for index, bandwidth in enumerate([12, 24]):
        mine = bartlett_calendar(influence, time, bandwidth)
        lag = np.abs(np.subtract.outer(np.arange(span_months), np.arange(span_months)))
        toeplitz = np.clip(1.0 - lag / (bandwidth + 1.0), 0.0, None)
        dense = filled.T @ toeplitz @ filled
        wrong = bartlett_compressed(influence, time, bandwidth)
        raw_saved, corrected_saved = archive[f"raw_calendar_covariance_{index}"], archive[f"covariance_{index}"]
        record = {"raw_full_matrix_max_rel_diff_autocovariance_sum": maxrel(mine, raw_saved), "raw_full_matrix_max_rel_diff_dense_quadratic_form": maxrel(dense, raw_saved),
                  "corrected_full_matrix_max_rel_diff": maxrel(mine * factor, corrected_saved),
                  "corrected_equals_raw_times_saved_factor_max_abs": maxabs(raw_saved * corrections[index], corrected_saved),
                  "compressed_full_matrix_max_rel_diff_vs_saved": maxrel(wrong, raw_saved),
                  "min_eigenvalue_over_max_raw": float(np.linalg.eigvalsh((mine + mine.T) / 2)[0] / np.linalg.eigvalsh((mine + mine.T) / 2)[-1]),
                  "pair_weights": pair_weights(observed, bandwidth), "contrasts": []}
        for label, vector in contrasts:
            estimate = float(vector @ beta)
            raw = float(vector @ mine @ vector)
            saved = float(vector @ corrected_saved @ vector)
            compressed = float(vector @ wrong @ vector)
            hc0 = float(vector @ (influence.T @ influence) @ vector)
            date_cluster = float(vector @ (filled.T @ filled) @ vector)
            alternatives = {"production: raw x T/(T-K_time), t(T-1)": (raw * factor, t - 1),
                            "no correction, t(T-1)": (raw, t - 1),
                            "raw x N/(N-K), t(T-1)": (raw * n / (n - k), t - 1),
                            "raw x T/(T-K_time), t(T-K_time)": (raw * factor, t - k_time),
                            "raw x T/(T-K_time), normal": (raw * factor, np.inf)}
            if span_rank < t and span_rank != k_time:
                alternatives["raw x T/(T-full span rank), t(T-1)"] = (raw * t / (t - span_rank), t - 1)
            table = {}
            for name, (variance, df) in alternatives.items():
                se = float(np.sqrt(variance))
                p = float(2 * stats.norm.sf(abs(estimate / se))) if not np.isfinite(df) else float(2 * stats.t.sf(abs(estimate / se), df))
                table[name] = {"std_error": se, "t": estimate / se, "p": p}
            record["contrasts"].append({"contrast": label, "estimate": estimate, "raw_variance_mine": raw, "corrected_variance_saved": saved,
                                        "corrected_variance_rel_diff": abs(raw * factor - saved) / abs(saved),
                                        "se_ratio_compressed_over_calendar": float(np.sqrt(compressed / raw)) if compressed > 0 else None,
                                        "se_ratio_hc0_over_calendar_raw": float(np.sqrt(hc0 / raw)), "se_ratio_date_cluster_only_over_calendar_raw": float(np.sqrt(date_cluster / raw)),
                                        "conventions": table})
        out["bandwidths"][str(bandwidth)] = record

    if "Origin cluster" in labels:
        index = labels.index("Origin cluster")
        units = sorted(set(unit.tolist()))
        scores = np.zeros((g, k))
        for position, u in enumerate(unit.tolist()):
            scores[units.index(u)] += influence[position]
        cluster = scores.T @ scores * (g / (g - 1) * (n - 1) / (n - k))
        saved = archive[f"covariance_{index}"]
        primary = archive["covariance_0"]
        rows = []
        for label, vector in contrasts:
            rows.append({"contrast": label, "se_origin_cluster": float(np.sqrt(vector @ saved @ vector)), "se_calendar_hac12": float(np.sqrt(vector @ primary @ vector)),
                         "se_ratio_cluster_over_calendar_hac12": float(np.sqrt((vector @ saved @ vector) / (vector @ primary @ vector)))})
        out["origin_cluster"] = {"full_matrix_max_rel_diff": maxrel(cluster, saved), "factor_mine": g / (g - 1) * (n - 1) / (n - k), "factor_saved": float(corrections[index]),
                                 "inference_df_saved": int(degrees[index]), "matrix_rank_mine": int(np.linalg.matrix_rank(cluster)), "rank_bound_g_minus_1": g - 1,
                                 "sum_of_origin_scores_max_abs": float(np.max(np.abs(scores.sum(axis=0)))), "contrasts": rows}
    return out


def main():
    specs = pd.read_csv(MONTHLY / "specifications.csv", low_memory=False)
    results = [run(spec_id, why, specs) for spec_id, why in SELECTED]
    worst = {"beta_max_abs_diff_vs_native": max(r["beta_max_abs_diff_vs_native"] for r in results),
             "residual_max_abs_diff_vs_native": max(r["residual_max_abs_diff_vs_native"] for r in results),
             "raw_hac_full_matrix_max_rel_diff": max(max(b["raw_full_matrix_max_rel_diff_autocovariance_sum"], b["raw_full_matrix_max_rel_diff_dense_quadratic_form"]) for r in results for b in r["bandwidths"].values()),
             "corrected_hac_full_matrix_max_rel_diff": max(b["corrected_full_matrix_max_rel_diff"] for r in results for b in r["bandwidths"].values()),
             "compressed_gap_version_min_rel_diff_vs_saved": min(b["compressed_full_matrix_max_rel_diff_vs_saved"] for r in results for b in r["bandwidths"].values()),
             "all_time_axes_month_ordinal": all(r["time_axis_is_month_ordinal"] for r in results),
             "all_gap_counts_match_ledger": all(r["calendar_gaps_mine"] == r["calendar_gaps_ledger"] for r in results),
             "all_k_time_match": all(r["k_time_mine"] == r["k_time_ledger"] for r in results)}
    write("probe06_result.json", {"probe": "06 actual-calendar Bartlett HAC with gaps; scaling conventions", "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/probe06_calendar_hac_gaps.py",
                                  "production_reference": "scripts/literature_sensitivity.py:197-208 calendar_covariance; scripts/visitor_rigour_monthly.py:358-369",
                                  "summary": worst, "designs": results})
    print("SUMMARY", worst)
    for r in results:
        print("\n==", r["spec_id"], "|", r["why"])
        print("  N,K,T,G:", r["n"], r["k"], r["t"], r["g"], "| span months", r["calendar_span_months"], "| gaps mine/ledger", r["calendar_gaps_mine"], r["calendar_gaps_ledger"],
              "| K_time mine/ledger", r["k_time_mine"], r["k_time_ledger"], "| span rank", r["full_span_rank_mine"], "| factor", round(r["factor_mine"], 6), r["factor_saved"], r["inference_df_saved"])
        print("  gap runs:", [(x["last_observed_before"], x["first_observed_after"], x["missing_months"]) for x in r["gap_runs"]])
        print("  beta/resid diff vs native:", r["beta_max_abs_diff_vs_native"], r["residual_max_abs_diff_vs_native"], "| hc0 rel", r["hc0_max_rel_diff"], "| cond", round(r["scaled_condition_number"], 1))
        for bandwidth, b in r["bandwidths"].items():
            print(f"  HAC{bandwidth}: raw rel diff (sum, dense) {b['raw_full_matrix_max_rel_diff_autocovariance_sum']:.2e} {b['raw_full_matrix_max_rel_diff_dense_quadratic_form']:.2e}"
                  f" | corrected {b['corrected_full_matrix_max_rel_diff']:.2e} | compressed-vs-saved {b['compressed_full_matrix_max_rel_diff_vs_saved']:.3f} | pairs {b['pair_weights']}")
            for c in b["contrasts"]:
                print(f"    {c['contrast']}: est {c['estimate']:.4f}; SE ratio compressed/calendar {c['se_ratio_compressed_over_calendar']:.3f}; HC0/calendar {c['se_ratio_hc0_over_calendar_raw']:.3f}; date-cluster/calendar {c['se_ratio_date_cluster_only_over_calendar_raw']:.3f}")
                for name, v in c["conventions"].items():
                    print(f"        {name:40s} se {v['std_error']:.4f} t {v['t']:.3f} p {v['p']:.4f}")
        if "origin_cluster" in r:
            o = r["origin_cluster"]
            print("  origin cluster: rel diff", o["full_matrix_max_rel_diff"], "factor", o["factor_mine"], o["factor_saved"], "df", o["inference_df_saved"], "rank", o["matrix_rank_mine"], "bound", o["rank_bound_g_minus_1"],
                  "sum scores", o["sum_of_origin_scores_max_abs"])
            for c in o["contrasts"]:
                print("    ", c)


if __name__ == "__main__":
    main()
