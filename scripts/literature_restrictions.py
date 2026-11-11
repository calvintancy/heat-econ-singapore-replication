"""Test the two-channel shape inside saved unrestricted YoY lag equations.

Pure postprocessing of already fitted native models. No coefficient fitting,
sample selection, source/reference rewriting or native-ledger writing occurs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import literature_sensitivity as producer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/literature_sensitivity"


def run_literature_restrictions() -> dict:
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert manifest["implementation_version"] == "raw_history_v2"
    assert manifest["native_fits"] == 2361
    specs = pd.read_csv(OUT / "specifications.csv", dtype={"unit_id": str, "sector_id": str}, float_precision="round_trip", low_memory=False)
    effects = pd.read_csv(OUT / "temperature_effects.csv", dtype={"unit_id": str, "sector_id": str}, float_precision="round_trip", low_memory=False)
    effect_lookup = {(row.spec_id, row.sector_id, row.covariance, int(row.hac_lags)): row
                     for row in effects.loc[effects.effect.eq("cumulative_direct")].itertuples(index=False)}
    specs = specs.loc[specs.specification.isin(["distributed", "dynamic"]) & specs.outcome.isin(["y_yoy", "y_yoy_nsa"]) & specs.structure.isin(["common", "H1"])]
    sources = {name: producer.shared.sha(ROOT / name) for name in ["scripts/literature_sensitivity.py", "outputs/literature_sensitivity/manifest.json",
        "outputs/literature_sensitivity/specifications.csv", "outputs/literature_sensitivity/temperature_effects.csv",
        "outputs/literature_sensitivity/pyfixest/fit_log.csv", "outputs/literature_sensitivity/pyfixest/estimator_manifest.json"]}
    rows, design_hashes = [], {}
    with threadpool_limits(limits=1):
        for _, specification in specs.iterrows():
            path = OUT / "designs" / f"{specification.spec_id}.npz"
            design_hashes[str(path.relative_to(ROOT))] = producer.shared.sha(path)
            with np.load(path, allow_pickle=False) as archive:
                selected, components = archive["selected_positions"], archive["component_positions"]
                beta, hc0 = archive["coefficients"][selected], archive["hc0_selected"]
                sectors = sorted(set(archive["units"]))
                n_lags = len(components);horizon = n_lags - 1
                eye = np.eye(n_lags)
                restriction = np.vstack([eye[j] - eye[1] for j in range(2, horizon)] + [eye[0] + eye[-1] - eye[1]])
                assert restriction.shape == (horizon - 1, horizon + 1)
                positions = np.asarray([[list(selected).index(index) for index in item] for item in components])
                for number, (method, band, degrees, correction) in enumerate(zip(archive["covariance_labels"], archive["bandwidths"], archive["inference_df"], archive["corrections"], strict=True)):
                    covariance = archive[f"covariance_{number}"][np.ix_(selected, selected)]
                    profile_units = sectors if specification.structure == "H1" else [sectors[0] if len(sectors) == 1 else "pooled"]
                    for unit_number, sector in enumerate(profile_units):
                        basis = np.eye(len(selected))[positions[:, 0]]
                        if specification.structure == "H1" and unit_number:
                            basis += np.eye(len(selected))[positions[:, unit_number]]
                        source = effect_lookup[(specification.spec_id, sector, method, int(band))]
                        metadata = {**specification.to_dict(), "sector_id": sector, "sector": source.sector,
                            "mean_y": source.mean_y, "sd_y": source.sd_y,
                            "covariance": str(method), "hac_lags": int(band), "inference_df": int(degrees),
                            "finite_sample_correction": float(correction), "main_inference": bool(source.main_inference),
                            "n_clusters": int(source.n_clusters),
                            "restriction_definition": "beta_j−beta_1=0 for j=2,...,h−1; beta_0+beta_h−beta_1=0; h−1 independent restrictions",
                            "interpretation": "Two-channel regressor span is nested in the unrestricted no-AR distributed equation" if specification.specification == "distributed" else "Conditional two-channel weather shape given AR1..h; not a nesting test against the no-AR fitted model"}
                        bound = len(sectors) - 1 if method == "Sector cluster" else int(specification.n_periods) - 1
                        rows.append(producer.test_record(basis @ beta, basis @ covariance @ basis.T, basis @ hc0 @ basis.T,
                            restriction, metadata, "Unrestricted YoY temperature response lies in endpoint-difference plus growth-sum span", bound))
    destination = OUT / "two_channel_span_tests.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    assert all(producer.shared.sha(ROOT / name) == value for name, value in sources.items()), "Postprocessing mutated a fitted input or receipt"
    receipt = {"status": "computed_not_yet_independently_verified", "scope": "Additional restrictions from saved full covariance; no native fits or sample/ref/ledger changes",
        "source": "scripts/literature_restrictions.py", "source_sha256": producer.shared.sha(Path(__file__)),
        "fitted_specs": len(specs), "test_rows": len(rows), "unavailable_tests": sum(row["status"] != "available" for row in rows),
        "output": str(destination.relative_to(ROOT)), "output_sha256": producer.shared.sha(destination),
        "frozen_inputs_sha256": sources, "design_archives_sha256": design_hashes}
    (OUT / "two_channel_span_manifest.json").write_text(json.dumps(producer.shared.clean(receipt), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Computed {len(rows)} rank-safe two-channel-span tests from {len(specs)} saved native fits; no refits", flush=True)
    return receipt


if __name__ == "__main__": run_literature_restrictions()
