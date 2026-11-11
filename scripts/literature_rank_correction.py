"""Reassess new multi-feature Wald tests using universal score-rank bounds.

Native fits and their covariance matrices are unchanged. Original test CSV bytes
are retained alongside the corrected tests and in a content-addressed audit
archive. Calendar scores have rank at most T−1; unit-cluster scores at most G−1.
Stronger bounds for a single exposure do not extend to stacked lag/season vectors.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from threadpoolctl import threadpool_limits

import annual_seasonal_literature as seasonal
import literature_sensitivity as subannual

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/literature_sensitivity"
VERSION = "universal_score_rank_v1"


def assess(beta, covariance, reference, restriction, metadata):
    r = np.atleast_2d(restriction)
    effect = r @ beta
    matrix = r @ covariance @ r.T;matrix = (matrix + matrix.T) / 2
    singular = np.linalg.svd(matrix, compute_uv=False)
    scale = float(np.linalg.norm(r @ reference @ r.T, 2))
    zero = subannual.ZERO * scale
    cutoff = max(float(singular[0]) * subannual.RANK, zero)
    uncapped = int(np.sum(singular > cutoff))
    bound = int(metadata["n_sectors"] - 1 if metadata["covariance"] == "Sector cluster" else metadata["n_periods"] - 1)
    rank = min(uncapped, bound)
    q, degrees = len(effect), int(metadata["inference_df"])
    available = rank == q and singular[0] > zero and degrees > 0
    statistic = float(effect @ np.linalg.solve(matrix, effect)) if available else np.nan
    assert not available or statistic >= -1e-8, (metadata["spec_id"], metadata["test"], statistic)
    condition = float(np.linalg.cond(matrix))
    flags = []
    if q >= .8 * bound: flags.append("Restriction count approaches the universal score-rank bound")
    if condition > 1e8: flags.append("Ill-conditioned restriction covariance")
    if metadata["time_residual_df"] < 10: flags.append("Fewer than ten residual time degrees of freedom")
    return {**metadata, "restrictions": q, "uncapped_covariance_rank": uncapped, "covariance_rank": rank,
        "structural_rank_bound": bound, "covariance_condition": condition, "hc0_covariance_scale": scale,
        "zero_covariance_threshold": zero, "rank_cutoff": cutoff,
        "status": "available" if available else "unavailable: rank-deficient or numerically degenerate restriction covariance",
        "wald_chi2": max(statistic, 0.) if available else None, "f_statistic": max(statistic, 0.) / q if available else None,
        "numerator_df": q, "denominator_df": degrees,
        "p_value": float(f_distribution.sf(max(statistic, 0.) / q, q, degrees)) if available else None,
        "rank_reassessment_version": VERSION,
        "rank_bound_rule": "G−1 unit-cluster score bound" if metadata["covariance"] == "Sector cluster" else "T−1 universal calendar-score bound; no single-exposure seasonal/OLS-residual-df cap",
        "joint_test_fragile": bool(flags), "fragility_reason": "; ".join(flags)}


def profile_bases(archive, structure, specification, *, annual):
    selected, components = archive["selected_positions"], archive["component_positions"]
    units = sorted(set(archive["sector_id" if annual else "units"]))
    positions = np.asarray([[list(selected).index(index) for index in row] for row in components])
    profiles = {}
    ids = units if structure == "H1" else [units[0] if len(units) == 1 and not annual else "pooled"]
    for number, unit in enumerate(ids):
        basis = np.eye(len(selected))[positions[:, 0]]
        if structure == "H1" and number: basis = basis + np.eye(len(selected))[positions[:, number]]
        profiles[unit] = basis
    return profiles, units


def subannual_restriction(row, profiles):
    bases = list(profiles.values())
    if row.test == "All sector cumulative direct temperature responses equal":
        return np.vstack([(basis - bases[0]).sum(axis=0) for basis in bases[1:]]) if row.specification != "two_channel" else np.vstack([(basis - bases[0])[1] * row.horizon for basis in bases[1:]])
    if row.test == "All sector temperature-lag responses jointly zero": return np.vstack(bases)
    basis = profiles[row.sector_id]
    if row.specification == "two_channel":
        choices = {"All temperature-lag coefficients zero": np.eye(2),
            "Cumulative direct temperature response zero": np.array([[0., row.horizon]]),
            "Exact contemporaneous static-output-level restriction": np.array([[0., 1.]]),
            "Equal one-period growth responses over YoY accumulation window": np.array([[1., 0.]])}
        return choices[row.test] @ basis
    choices = dict(subannual.restrictions(len(basis), row.outcome))
    return choices[row.test] @ basis


def seasonal_restriction(row, profiles):
    bases = list(profiles.values())
    if row.test == "All sector seasonal temperature response vectors equal": return np.vstack([basis - bases[0] for basis in bases[1:]])
    if row.test == "All sector sums of direct seasonal temperature coefficients equal": return np.vstack([(basis - bases[0]).sum(axis=0) for basis in bases[1:]])
    return dict(seasonal.nulls(profiles[row.sector_id], row.model_spec))[row.test]


def correct_branch(branch):
    annual = branch == "annual_seasonal"
    folder = BASE / branch if annual else BASE
    path, native_path = folder / "tests.csv", folder / "tests_native_stage.csv"
    current = pd.read_csv(path, dtype={"sector_id": str, "unit_id": str}, float_precision="round_trip", low_memory=False)
    if "rank_reassessment_version" in current:
        assert native_path.exists(), "Corrected tests require preserved native-stage test bytes"
        original = pd.read_csv(native_path, dtype={"sector_id": str, "unit_id": str}, float_precision="round_trip", low_memory=False)
    else:
        original = current
        if native_path.exists() and subannual.shared.sha(native_path) != subannual.shared.sha(path):
            old_sha = subannual.shared.sha(native_path)
            old_archive = ROOT / "dist/aea_audit_20261111/test_rank_before_general_bound" / branch / old_sha
            old_archive.mkdir(parents=True, exist_ok=True);shutil.copy2(native_path, old_archive / "tests.csv")
        shutil.copy2(path, native_path)
    original_sha = subannual.shared.sha(native_path)
    archive_folder = ROOT / "dist/aea_audit_20261111/test_rank_before_general_bound" / branch / original_sha
    archive_folder.mkdir(parents=True, exist_ok=True)
    archive_path = archive_folder / "tests.csv"
    if archive_path.exists(): assert subannual.shared.sha(archive_path) == original_sha
    else: shutil.copy2(native_path, archive_path)
    protected_paths = [folder / "pyfixest/fit_log.csv", folder / "pyfixest/estimator_manifest.json", folder / "manifest.json",
        folder / "coefficients.csv", folder / "temperature_effects.csv", folder / "interaction_components.csv", folder / "specifications.csv"]
    protected = {str(item.relative_to(ROOT)): subannual.shared.sha(item) for item in protected_paths}
    corrected, design_hashes = {}, {}
    with threadpool_limits(limits=1):
        for spec_id, group in original.groupby("spec_id", sort=False):
            archive_path = folder / "designs" / f"{spec_id}.npz"
            design_hashes[str(archive_path.relative_to(ROOT))] = subannual.shared.sha(archive_path)
            with np.load(archive_path, allow_pickle=False) as archive:
                if ("beta" if annual else "coefficients") not in archive:
                    for index, row in group.iterrows():
                        value = row.to_dict()
                        value.update(structural_rank_bound=int(row.n_sectors - 1 if row.covariance == "Sector cluster" else row.n_periods - 1),
                            rank_reassessment_version=VERSION, rank_bound_rule="No covariance estimated: complete point/time design unavailable",
                            uncapped_covariance_rank=None, joint_test_fragile=True, fragility_reason="Point/time design unavailable")
                        corrected[index] = value
                    continue
                structure = group.iloc[0]["model" if annual else "structure"]
                specification = group.iloc[0]["model_spec" if annual else "specification"]
                profiles, _ = profile_bases(archive, structure, specification, annual=annual)
                selected = archive["selected_positions"]
                beta = archive["beta" if annual else "coefficients"][selected]
                reference = archive["hc0_selected"]
                for number, (method, bandwidth) in enumerate(zip(archive["covariance_labels"], archive["bandwidths"], strict=True)):
                    covariance = archive[f"covariance_{number}"][np.ix_(selected, selected)]
                    for index, row in group.loc[group.covariance.eq(method) & group.hac_lags.eq(bandwidth)].iterrows():
                        restriction = seasonal_restriction(row, profiles) if annual else subannual_restriction(row, profiles)
                        corrected[index] = assess(beta, covariance, reference, restriction, row.to_dict())
    assert len(corrected) == len(original)
    result = pd.DataFrame([corrected[index] for index in range(len(original))])
    result.to_csv(path, index=False)
    assert all(subannual.shared.sha(ROOT / name) == value for name, value in protected.items()), "Rank postprocessing changed native fits or covariance inputs"
    changed = original.status.ne(result.status)
    receipt = {"status": "reassessed_not_yet_independently_verified", "version": VERSION, "branch": branch,
        "source": "scripts/literature_rank_correction.py", "source_sha256": subannual.shared.sha(Path(__file__)),
        "native_stage_tests": str(native_path.relative_to(ROOT)), "native_stage_tests_sha256": original_sha,
        "archive": str(archive_folder.relative_to(ROOT)), "corrected_tests": str(path.relative_to(ROOT)), "corrected_tests_sha256": subannual.shared.sha(path),
        "test_rows": len(result), "rank_fields_changed": int(original.covariance_rank.fillna(-1).ne(result.covariance_rank.fillna(-1)).sum()),
        "statuses_changed": int(changed.sum()), "newly_available": int((changed & result.status.eq("available")).sum()),
        "newly_unavailable": int((changed & ~result.status.eq("available")).sum()),
        "fragile_available_tests": int((result.status.eq("available") & result.joint_test_fragile).sum()),
        "protected_native_inputs_sha256": protected, "design_archives_sha256": design_hashes,
        "claim": "Only Wald rank/status/statistic/p metadata reassessed; full native beta, residuals, covariance, samples, receipts and frozen original references are unchanged",
        "thresholds": {"relative_singular_value": subannual.RANK, "relative_to_HC0_zero": subannual.ZERO},
        "bound": "Universal T−1 calendar-score /G−1 unit-cluster-score rank upper bounds; actual SVD rank can be lower"}
    (folder / "rank_correction_manifest.json").write_text(json.dumps(subannual.shared.clean(receipt), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Rank correction {branch}: {len(result)} tests; {receipt['rank_fields_changed']} rank fields, {receipt['statuses_changed']} availability changes; no fits", flush=True)
    return receipt


def run_literature_rank_correction(branch="all"):
    return {name: correct_branch(name) for name in (["subannual", "annual_seasonal"] if branch == "all" else [branch])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["all", "subannual", "annual_seasonal"], default="all")
    run_literature_rank_correction(parser.parse_args().branch)
