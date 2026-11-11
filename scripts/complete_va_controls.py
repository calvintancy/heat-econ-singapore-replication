"""Complete VA12/VA20 panels with rainfall, humidity, wind and PM2.5.

The already audited crisis engine supplies native fitting, calendar masks, QR
geometry, CR1/HAC covariance, component propagation and rank-safe tests. This
companion configures it only within this process and restores every module
binding afterwards. No long-history or baseline artifact is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crisis_sensitivity as engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/complete_va_controls"
TERMS = ["temp_c", "rain_100mm", "rh_pct", "wind_mps", "pm25_10ug"]


def contexts() -> list[engine.Context]:
    return [engine.Context("va12", "12 complete broad VA components · full controls",
                "outputs/outcomes/va/full/common_sample.csv", "quarterly", add_dwellings=True),
            engine.Context("va12-matched", "12 complete broad VA components · detailed-matched full controls",
                "outputs/granular/va-broad-matched/full/samples/G1.csv", "quarterly", "va20", add_dwellings=True),
            engine.Context("va20", "20 complete detailed VA components · full controls",
                "outputs/granular/va-detailed/full/samples/G1.csv", "quarterly", add_dwellings=True)]


def run_complete_va_controls(policy_path: Path = engine.DEFAULT_POLICY) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    source_policy = json.loads(Path(policy_path).read_text())
    assert source_policy["controls"] == ["temp_c", "rain_100mm", "rh_pct"]
    policy = dict(source_policy)
    policy.update(policy_id=source_policy["policy_id"]+"__complete_va_full_controls",
        label="Complete VA coverage · full weather controls · same event/rebound calendar",
        controls=TERMS, source_crisis_policy=str(Path(policy_path).resolve().relative_to(ROOT)),
        source_crisis_policy_sha256=engine.sha(Path(policy_path)),
        control_scope="All five weather variables included: temperature, rainfall, humidity, wind speed and PM2.5",
        eligibility="Exact frozen 90%-coverage parent samples; no new coverage rule, imputation or missing-value deletion")
    policy_file = OUT / "full_control_policy.json"
    policy_file.write_text(json.dumps(policy, indent=2, ensure_ascii=False)+"\n")
    protected_names = ["scripts/crisis_sensitivity.py", "outputs/crisis_sensitivity/manifest.json",
        "outputs/crisis_sensitivity/specifications.csv", "outputs/crisis_sensitivity/estimates.csv",
        "outputs/crisis_sensitivity/sector_estimates.csv", "outputs/crisis_sensitivity/separate_estimates.csv",
        "outputs/crisis_sensitivity/pyfixest/fit_log.csv", "outputs/crisis_sensitivity/pyfixest/estimator_manifest.json",
        "data/processed/extended_controls/quarterly_controls.csv", "data/raw/extended_controls/manifest.json"]
    protected = {name: engine.sha(ROOT / name) for name in protected_names}
    old = {name: getattr(engine, name) for name in ["OUT", "TERMS", "contexts", "design", "base_metadata", "fit_model"]}
    original_design, original_metadata, original_fit = old["design"], old["base_metadata"], old["fit_model"]

    def full_design(sample: pd.DataFrame, model: str):
        # The original common design already creates every configured weather
        # column; extend its reported weather positions from three to five.
        common, common_names, _, _ = original_design(sample, "common")
        prefix = common.shape[1]-len(TERMS)
        if model != "H1":
            return common, common_names, list(range(prefix, common.shape[1])), []
        weather = common[:, prefix:]
        sectors = sorted(sample.sector_id.unique())
        nonreference = np.column_stack([sample.sector_id.eq(sector) for sector in sectors[1:]]).astype(float)
        temperature = np.column_stack([weather[:, 0], nonreference*weather[:, [0]]])
        matrix = np.column_stack([common[:, :prefix], weather[:, 1:], temperature])
        names = common_names[:prefix]+TERMS[1:]+["temp_c"]+[f"temp_c:C(sector_id)[T.{sector}]" for sector in sectors[1:]]
        component_positions = list(range(prefix+len(TERMS)-1, matrix.shape[1]))
        # The audited H1 output adapter expects two displayed control columns
        # before the component block. The FIT includes all four controls.
        # Its full non-FE covariance is expanded below from the same QR map.
        display_selection = [prefix, prefix+1]+component_positions
        return matrix, names, display_selection, component_positions

    def full_metadata(*args, **kwargs):
        metadata = original_metadata(*args, **kwargs)
        metadata.update(control_set="full", controls=json.dumps(TERMS),
            controls_label="Temperature, rainfall, humidity, wind speed and PM2.5",
            coverage_rule="Frozen eligible_90 parent dates; every constituent month passed original wind/PM2.5 coverage rules",
            baseline_role="Complete-coverage full-control baseline; long-history weather-only versions are separate")
        if metadata["calendar_gaps"]:
            metadata["stationarity_status"] = "Not cleared: exact full-control calendar contains gaps; ordinary contiguous ADF/KPSS screens are not applied"
        return metadata

    def full_fit(context, parent, sample, variant, used_policy, sample_path, model, records, unit_id="pooled"):
        before = {name: len(rows) for name, rows in records.items()}
        result = original_fit(context, parent, sample, variant, used_policy, sample_path, model, records, unit_id)
        for name, rows in records.items():
            for row in rows[before[name]:]:
                if "formula" in row:
                    row["formula"] = row["formula"].replace("rh_pct +", "rh_pct + wind_mps + pm25_10ug +")
        if result is None or model != "H1": return result
        prefix = next(i for i, name in enumerate(result["coef_names"]) if name == "rain_100mm")
        selected = list(range(prefix, len(result["coef_names"])))
        selected_names = [result["coef_names"][i] for i in selected]
        # Replace only this call's reported selected covariance block. Native
        # point estimates/residuals and all sector/component results stay intact.
        del records["coefficient_covariances"][before["coefficient_covariances"]:]
        h1_rows = records["sector_estimates"][before["sector_estimates"]:]
        for j, outcome in enumerate(engine.OUTCOMES):
            influence = result["mapping"][selected].T*result["residuals"][:, [j]]
            hc0 = influence.T@influence
            for band in [context.horizon, 2*context.horizon]:
                meta = next(row for row in h1_rows if row["outcome"] == outcome and row["hac_lags"] == band).copy()
                meta.update(mean_y=float(sample[outcome].mean()), sd_y=float(sample[outcome].std()))
                covariance = engine.covariance_from_scores(influence, sample.time.to_numpy(), band, meta["finite_sample_correction"])
                engine.store_matrix(records, "coefficient_covariances", meta, covariance, selected_names, hc0)
                for index, term in enumerate(TERMS[1:]):
                    records["shared_control_estimates"].append({**meta, "sector_id": "shared", "sector": "Shared weather response",
                        "term": term, **engine.inference(result["beta"][selected[index], j], covariance[index, index], meta["inference_df"], hc0[index, index])})
        return result

    # The shared collector also serializes this additional H1 control output.
    shared_controls = []
    def fitting_adapter(*args, **kwargs):
        records = args[7] if len(args) > 7 else kwargs["records"]
        records.setdefault("shared_control_estimates", shared_controls)
        return full_fit(*args, **kwargs)

    try:
        engine.OUT, engine.TERMS, engine.contexts = OUT, TERMS, contexts
        engine.design, engine.base_metadata, engine.fit_model = full_design, full_metadata, fitting_adapter
        manifest = engine.run_crisis_sensitivity(policy_file)
    finally:
        for name, value in old.items(): setattr(engine, name, value)
    assert all(engine.sha(ROOT / name) == expected for name, expected in protected.items()), "Core long-history source/output changed"
    specs = pd.read_csv(OUT / "specifications.csv")
    assert len(specs) == manifest["native_fits"] == 750
    expected_periods = {"all_eligible": 29, "event_dates_excluded": 19, "strict_growth_window_clean": 17}
    assert all(specs.loc[specs.sample_variant.eq(variant), "n_periods"].eq(periods).all()
               for variant, periods in expected_periods.items())
    assert specs.control_set.eq("full").all() and specs.includes_dwellings.eq(True).all()
    manifest.update(analysis="Complete VA12/VA20 full-control baseline and event-window sensitivities",
        controls=TERMS, expected_quarters=expected_periods, long_history_reference="outputs/crisis_sensitivity/",
        protected_core_sources_sha256=protected,
        shared_engine="scripts/crisis_sensitivity.py; native fitting and pure calendar/covariance/inference routines reused through bounded process-local configuration",
        H1_covariance_reporting="Full five-control native design throughout; the reported non-FE covariance block includes all four shared controls and temperature components",
        interpretation="Includes ownership of dwellings. Equal-component panel estimates are distinct from the separately published official aggregate. Full-control short/gapped inference remains fragile.")
    (OUT / "manifest.json").write_text(json.dumps(engine.clean(manifest), indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=engine.DEFAULT_POLICY)
    args = parser.parse_args()
    run_complete_va_controls(args.policy.resolve())
