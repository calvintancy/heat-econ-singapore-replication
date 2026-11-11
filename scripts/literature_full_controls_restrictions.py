"""Test the general two-channel shape from frozen full-control lag estimates.

Only the supplement and its lineage manifest are written. Saved native fits,
designs, samples, model CSVs and all earlier references remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import literature_full_controls as producer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/literature_sensitivity/full_controls"
TEST = "Unrestricted YoY temperature response lies in endpoint-difference plus growth-sum span"


def run_literature_full_controls_restrictions():
    source = Path(__file__)
    own_sha = producer.shared.sha(source)
    fitted_manifest = json.loads((OUT/"manifest.json").read_text())
    assert producer.shared.sha(ROOT/fitted_manifest["producer_source"]) == fitted_manifest["producer_source_sha256"]
    protected = {**fitted_manifest["inputs_sha256"], **fitted_manifest["protected_sources_sha256"]}
    fitted_paths = [OUT/name for name in ["manifest.json", "specifications.csv", "coefficients.csv", "temperature_effects.csv", "channel_estimates.csv",
        "control_estimates.csv", "interaction_components.csv", "tests.csv", "sample_inventory.csv", "masks.csv", "pyfixest/fit_log.csv", "pyfixest/estimator_manifest.json"]]
    fitted_paths += list((OUT/"samples").glob("*.csv"))
    frozen = {producer.relative(path): producer.shared.sha(path) for path in fitted_paths}
    assert all(producer.shared.sha(ROOT/path) == value for path, value in protected.items())
    specs = pd.read_csv(OUT/"specifications.csv", dtype={"sector_id": str}, float_precision="round_trip", low_memory=False)
    specs = specs.loc[specs.specification.isin(["distributed", "dynamic"]) & specs.outcome.isin(["y_yoy", "y_yoy_nsa"]) & specs.structure.isin(["common", "H1"])]
    effects = pd.read_csv(OUT/"temperature_effects.csv", dtype={"sector_id": str}, float_precision="round_trip", low_memory=False)
    lookup = {(row.spec_id, row.sector_id, row.covariance, int(row.hac_lags)): row
              for row in effects.loc[effects.effect.eq("cumulative_direct")].itertuples(index=False)}
    rows, designs = [], {}
    with threadpool_limits(limits=1):
        for specification in specs.itertuples(index=False):
            path = ROOT/specification.design_npz
            designs[producer.relative(path)] = producer.shared.sha(path)
            with np.load(path, allow_pickle=False) as archive:
                selected, components = archive["selected_positions"], archive["component_positions"]
                sectors = sorted(set(archive["units"]))
                horizon = len(components)-1
                assert horizon == specification.horizon
                eye = np.eye(horizon+1)
                restriction = np.vstack([eye[j]-eye[1] for j in range(2,horizon)] + [eye[0]+eye[-1]-eye[1]])
                assert restriction.shape == (horizon-1,horizon+1) and np.linalg.matrix_rank(restriction) == horizon-1
                positions = np.asarray([[list(selected).index(index) for index in part] for part in components])
                estimated = specification.status == "estimated"
                assert ("coefficients" in archive) == estimated
                beta = archive["coefficients"][selected] if estimated else None
                hc0 = archive["hc0_selected"] if estimated else None
                profile_units = sectors if specification.structure == "H1" else [sectors[0] if len(sectors)==1 else "pooled"]
                for number, (method, band, degrees, correction) in enumerate(zip(archive["covariance_labels"], archive["bandwidths"], archive["inference_df"], archive["corrections"], strict=True)):
                    covariance = archive[f"covariance_{number}"][np.ix_(selected,selected)] if estimated else None
                    for unit_number, unit in enumerate(profile_units):
                        basis = np.eye(len(selected))[positions[:,0]]
                        if specification.structure == "H1" and unit_number:
                            basis = basis + np.eye(len(selected))[positions[:,unit_number]]
                        row = lookup[specification.spec_id,unit,str(method),int(band)]
                        metadata = {**specification._asdict(), "sector_id": unit, "sector": row.sector,
                            "mean_y": row.mean_y, "sd_y": row.sd_y, "covariance": str(method), "hac_lags": int(band), "inference_df": int(degrees),
                            "finite_sample_correction": float(correction) if estimated else None,
                            "main_inference": bool(row.main_inference), "n_clusters": int(row.n_clusters), "underlying_point_status": specification.status,
                            "restriction_definition": "beta_j−beta_1=0 for j=2,...,h−1; beta_0+beta_h−beta_1=0; h−1 independent restrictions",
                            "interpretation": "Two-channel span nested in the unrestricted full-control no-AR distributed equation" if specification.specification == "distributed" else "Conditional two-channel weather shape given AR1..h and all current controls; not a nesting test against the no-AR fitted model"}
                        full_restriction = restriction@basis
                        rows.append(producer.wald(beta,covariance,hc0,full_restriction,metadata,TEST) if estimated else producer.unavailable_test(metadata,TEST,full_restriction))
    destination = OUT/"two_channel_span_tests.csv"
    pd.DataFrame(rows).to_csv(destination,index=False)
    assert all(producer.shared.sha(ROOT/path)==value for path,value in protected.items()), "Earlier protected analytical file changed"
    assert all(producer.shared.sha(ROOT/path)==value for path,value in frozen.items()), "Full-control fitted input or receipt changed"
    assert all(producer.shared.sha(ROOT/path)==value for path,value in designs.items()), "Full-control native design/covariance changed"
    assert producer.shared.sha(source)==own_sha, "Postprocessor source changed during run"
    receipt = {"status":"computed_not_yet_independently_verified", "scope":"Pure full-control general two-channel-span Wald supplement; no fits, sample/control changes, fit-CSV or native-ledger writes",
        "source":producer.relative(source), "source_sha256":own_sha, "producer_source":fitted_manifest["producer_source"], "producer_source_sha256":fitted_manifest["producer_source_sha256"],
        "requested_specs":len(specs), "fitted_specs":int(specs.status.eq("estimated").sum()), "unavailable_point_specs":int(specs.status.ne("estimated").sum()),
        "test_rows":len(rows), "available_tests":sum(row["status"]=="available" for row in rows), "unavailable_tests":sum(row["status"]!="available" for row in rows),
        "fragile_available_tests":sum(row["status"]=="available" and row["joint_test_fragile"] for row in rows),
        "restriction":"h−1 restrictions on unrestricted YoY coefficients: all intermediate lags equal and beta0+betah equals that common intermediate coefficient",
        "dynamic_interpretation":"Conditional weather-shape restriction given fixed AR1..h controls, not a test comparing the separately fitted no-AR two-channel model",
        "rank_bound":"Actual numerical SVD rank with T−1 calendar-score/G−1 sector-score bounds; no inverses or p-values for deficient matrices",
        "output":producer.relative(destination), "output_sha256":producer.shared.sha(destination), "frozen_inputs_sha256":frozen,
        "design_archives_sha256":designs, "earlier_protected_hashes_verified":len(protected)}
    (OUT/"two_channel_span_manifest.json").write_text(json.dumps(producer.shared.clean(receipt),indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(f"Computed full-control two-channel span: {len(rows)} tests from {len(specs)} requests; no native fits",flush=True)
    return receipt


if __name__=="__main__":run_literature_full_controls_restrictions()
