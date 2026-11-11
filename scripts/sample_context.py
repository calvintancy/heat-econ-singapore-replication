"""Attach exact-sample dependent-variable means to saved estimates, without refitting."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/sample_context"
CONTEXTS = [("va", "quarterly", "quarter", "Q", 4),
            ("iip-monthly", "monthly", "month", "M", 12),
            ("iip-quarterly", "quarterly", "quarter", "Q", 4)]
ALIASES = {"g_va_sa_qoq": ("short", "sa"), "g_va_nsa_qoq": ("short", "nsa"),
           "g_iip_sa": ("short", "sa"), "g_iip_nsa": ("short", "nsa"),
           "y_short": ("short", "sa"), "y_yoy": ("yoy", "sa"),
           'y_yoy_nsa':('yoy','nsa'),'y_yoy_pct_nsa':('yoy_pct','nsa'),
           "y_yoy_pct": ("yoy_pct", "sa"), "y_log_level": ("log_level", "sa"),
           "y_raw_level": ("raw_level", "sa")}
TEMPERATURE_TERMS = {"temp_c", "temp_c_unweighted", "temp_year_mean", "temp_year_change", "temp_year_lag"}


def relative(path: str | Path) -> str:
    path = Path(path)
    return str(path.resolve().relative_to(ROOT)) if path.is_absolute() else str(path)


def outcome_units(context: str, outcome: str) -> str:
    transform, adjustment = ALIASES[outcome]
    frequency = "month" if context == "iip-monthly" else "quarter"
    if transform == "short":
        return f"{frequency}-on-{frequency} log points (100 × Δ ln output; {adjustment.upper()}; not annualised)"
    if transform == "yoy":
        return f"year-on-year log points (100 × Δyear ln output; {adjustment.upper()})"
    if transform == "yoy_pct":
        return f"year-on-year percent change (exact ratio; {adjustment.upper()})"
    if transform == "log_level":
        return "100 × ln seasonally adjusted output"
    return "chained-2015 S$ million (SA)" if context == "va" else "IIP index points, 2025=100 (SA)"


class SampleBook:
    def __init__(self):
        self.frames: dict[str, pd.DataFrame] = {}
        self.input_hashes: dict[str, str] = {}
        self.rows: list[dict] = []
        self.magnitudes: list[dict] = []
        self.raw_errors: list[dict] = []
        self.inventory: list[dict] = []
        self.levels: dict[tuple[str, str], dict[tuple[str, pd.Period], float]] = {}
        self._load_raw_levels()

    def _hash(self, path: Path) -> None:
        self.input_hashes[relative(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def _load_raw_levels(self) -> None:
        for family in ["va", "iip"]:
            for adjustment in ["sa", "nsa"]:
                path = ROOT / f"data/raw/{family}_{adjustment}_tabledata.json"
                self._hash(path)
                payload = json.loads(path.read_text())["Data"]
                values = {}
                for series in payload["row"]:
                    sector = str(series["seriesNo"])
                    for cell in series["columns"]:
                        number = str(cell["value"]).replace(",", "")
                        if number.lower() in {"na", "-", ""}:
                            continue
                        if family == "va":
                            year, quarter = re.fullmatch(r"(\d{4}) ([1-4])Q", cell["key"]).groups()
                            period = pd.Period(f"{year}Q{quarter}", freq="Q")
                        else:
                            period = pd.Period(pd.to_datetime(cell["key"], format="%Y %b"), freq="M")
                        assert (sector, period) not in values
                        values[sector, period] = float(number)
                self.levels[family, adjustment] = values
                if family == "iip":
                    quarters = defaultdict(list)
                    for (sector, month), value in values.items():
                        quarters[sector, month.asfreq("Q")].append((month, value))
                    self.levels["iip-quarterly", adjustment] = {
                        key: math.fsum(value for _, value in observations)/3
                        for key, observations in quarters.items()
                        if len(observations) == 3 and len({month for month, _ in observations}) == 3}

    def read(self, path: str | Path, period: str | None = None, freq: str | None = None) -> pd.DataFrame:
        path = ROOT / path if not Path(path).is_absolute() else Path(path)
        key = relative(path)
        if key not in self.frames:
            self._hash(path)
            frame = pd.read_csv(path, dtype={"sector_id": str})
            if period:
                frame[period] = pd.PeriodIndex(frame[period], freq=freq)
                assert not frame.duplicated(["sector_id", period]).any(), key
                frame = frame.sort_values(["sector_id", period]).reset_index(drop=True)
            self.frames[key] = frame
        return self.frames[key].copy()

    def independent_values(self, frame: pd.DataFrame, context: str, outcome: str,
                           period: str, horizon: int) -> np.ndarray:
        transform, adjustment = ALIASES[outcome]
        family = "va" if context == "va" else "iip-quarterly" if context == "iip-quarterly" else "iip"
        source = self.levels[family, adjustment]
        values = []
        for sector, date in zip(frame.sector_id, frame[period]):
            current = source.get((sector, date), np.nan)
            if transform == "raw_level":
                value = current
            elif transform == "log_level":
                value = 100*math.log(current)
            else:
                lag = 1 if transform == "short" else horizon
                prior = source.get((sector, date-lag), np.nan)
                value = 100*(current/prior-1) if transform == "yoy_pct" else 100*(math.log(current)-math.log(prior))
            values.append(value)
        return np.asarray(values, dtype=float)

    def statistics(self, frame: pd.DataFrame, context: str, outcome: str, period: str,
                   horizon: int, *, source_sample: str, sample_rule: str) -> dict:
        independent = self.independent_values(frame, context, outcome, period, horizon)
        observed = frame[outcome].to_numpy(dtype=float) if outcome in frame else independent
        assert np.isfinite(observed).all(), (context, outcome, "Missing dependent variable")
        assert np.isfinite(independent).all(), (context, outcome, "Unavailable raw level or calendar lag")
        maximum = float(np.max(np.abs(observed-independent)))
        assert np.allclose(observed, independent, rtol=1e-12, atol=1e-9), (context, outcome, maximum)
        mean, sd = float(observed.mean()), float(observed.std(ddof=1))
        raw_mean = math.fsum(independent)/len(independent)
        raw_sd = math.sqrt(math.fsum((float(x)-raw_mean)**2 for x in independent)/(len(independent)-1))
        assert np.isclose(mean, raw_mean, rtol=1e-12, atol=1e-9)
        assert np.isclose(sd, raw_sd, rtol=1e-12, atol=1e-9)
        self.raw_errors.append({"context": context, "outcome": outcome, "source_sample": source_sample,
            "sample_rule": sample_rule, "n_values": len(frame), "max_abs_value_error": maximum,
            "mean_abs_error": abs(mean-raw_mean), "sd_abs_error": abs(sd-raw_sd)})
        return {"context": context, "outcome": outcome, "mean_y": mean, "sd_y": sd,
            "n_obs": len(frame), "n_periods": frame[period].nunique(), "n_sectors": frame.sector_id.nunique(),
            "start": str(frame[period].min()), "end": str(frame[period].max()),
            "period_column": period, "outcome_units": outcome_units(context, outcome),
            "source_sample": source_sample, "sample_rule": sample_rule,
            "mean_definition": "Signed arithmetic mean of observed dependent-variable values; equal observation weights",
            "sd_definition": "Sample standard deviation across observed dependent-variable values, ddof=1"}

    def register(self, csv_path: str | Path, model: str, outcome: str, sample: pd.DataFrame,
                 context: str, period: str, horizon: int, *, source_sample: str, sample_rule: str,
                 per_sector: bool = False, separate: bool = False) -> None:
        coefficients = self.read(csv_path)
        selected = coefficients[coefficients.model.eq(model)] if "model" in coefficients else coefficients
        if "outcome" in selected:
            selected = selected[selected.outcome.eq(outcome)]
        assert not selected.empty, (csv_path, model, outcome)
        stats = self.statistics(sample, context, outcome, period, horizon,
                                source_sample=source_sample, sample_rule=sample_rule)
        if not separate:
            for field, expected in [("n_obs", len(sample)), ("n_sectors", sample.sector_id.nunique()),
                                    ("n_periods", sample[period].nunique()), ("n_months", sample[period].nunique()),
                                    ("n_quarters", sample[period].nunique())]:
                if field in selected:
                    assert selected[field].dropna().eq(expected).all(), (csv_path, model, field, expected)
            for field, expected in [("start", stats["start"]), ("end", stats["end"])]:
                if field in selected:
                    assert selected[field].dropna().eq(expected).all(), (csv_path, model, field, expected)
        key = {"coefficient_csv": relative(csv_path), "model": model}
        if not separate:
            self.rows.append({**key, **stats, "scope": "pooled", "sector_id": "pooled", "sector": "Pooled sample"})
        if per_sector or separate:
            for sector_id, group in sample.groupby("sector_id", sort=True):
                sector_stats = self.statistics(group, context, outcome, period, horizon,
                    source_sample=source_sample, sample_rule=sample_rule+f"; sector_id={sector_id}")
                if separate:
                    assert selected[selected.sector_id.eq(sector_id)].n_obs.eq(len(group)).all()
                self.rows.append({**key, **sector_stats, "scope": "sector", "sector_id": sector_id,
                                  "sector": group.sector.iloc[0]})
        # Retain primary-bandwidth magnitudes. Each exposure and estimand stays explicit.
        bandwidth = 12 if context == "iip-monthly" else 4
        magnitude = selected[selected.hac_lags.eq(bandwidth)] if "hac_lags" in selected else selected
        if "covariance" in magnitude:
            magnitude = magnitude[magnitude.covariance.eq("Driscoll-Kraay")]
        for row in magnitude.to_dict("records"):
            term = row.get("term", "temp_c")
            if term not in TEMPERATURE_TERMS and not str(term).endswith(":temp_c"):
                continue
            sector_id = str(row.get("sector_id", "pooled"))
            scoped = sample[sample.sector_id.eq(sector_id)] if sector_id != "pooled" else sample
            values = self.independent_values(scoped, context, outcome, period, horizon)
            self.magnitudes.append({**key, "outcome": outcome, "sector_id": sector_id,
                "sector": row.get("sector", "Pooled sample"), "term": term,
                "beta_1c": float(row["coefficient"]), "effect_0_1c": float(row["coefficient"])*.1,
                "mean_y": float(values.mean()), "sd_y": float(values.std(ddof=1)),
                "n_obs_for_mean": len(scoped), "n_periods": scoped[period].nunique(),
                "outcome_units": stats["outcome_units"], "estimand": row.get("estimand", "Conditional association"),
                "exposure": term, "source_sample": source_sample, "sample_rule": sample_rule,
                "interpretation": "Coefficient and 0.1 × coefficient in dependent-variable units; no division by mean growth"})
        if model in {"C5", "G2X", 'G2NX', "L1X", "L3X", "H3X"}:
            self.inventory.append({**key, **stats, "exclusion": "Observation dates in 2020–2022 excluded",
                "caution": "YoY observations in 2023 can retain 2022 base periods; not a pandemic-free history"})


def _outcome(frame: pd.DataFrame) -> str:
    if "outcome" in frame:
        values = frame.outcome.dropna().unique()
    else:
        values = frame.formula.str.split("~").str[0].str.strip().unique()
    assert len(values) == 1, values
    return str(values[0])


def build_sample_context() -> dict:
    book = SampleBook()
    catalog = []
    for context, frequency, period, freq, horizon in CONTEXTS:
        broad_path = "data/processed/panel_broad.csv" if context == "va" else f"data/processed/iip/panel_{frequency}_broad.csv"
        detailed_path = "data/processed/panel_detailed.csv" if context == "va" else f"data/processed/iip/panel_{frequency}_detailed.csv"
        broad, detailed = book.read(broad_path, period, freq), book.read(detailed_path, period, freq)
        main_path = "outputs/regression_results.csv" if context == "va" else f"outputs/iip/{frequency}/regression_results.csv"
        main = book.read(main_path)
        for model, rows in main.groupby("model", sort=False):
            sample, source, rule = broad, broad_path, "All original broad-panel estimation dates"
            if model == "S4":
                sample, rule = broad[broad.year.lt(2020)], "Observation dates before 2020"
            elif model == "S5":
                sample, rule = broad[~broad.year.between(2020, 2022)], "Exclude observation dates in 2020–2022"
            elif model == "S6":
                start = pd.Period("2005Q2", freq="Q") if context == "va" else detailed[period].min()
                sample = broad[broad[period].between(start, detailed[period].max())]
                rule = "Broad sectors on matched detailed-panel dates"
            elif model == "S7":
                sample, source, rule = detailed, detailed_path, "All original detailed-panel estimation dates"
            book.register(main_path, model, _outcome(rows), sample, context, period, horizon,
                          source_sample=source, sample_rule=rule)

        # Earlier timing, sector-slope and aggregate-IIP annexes also have exact samples.
        folder = "outputs" if context == "va" else f"outputs/iip/{frequency}"
        timing_path = f"{folder}/distributed_lag_results.csv"
        timing = book.read(timing_path)
        lags = [f"temp_l{i}" for i in range(1, horizon+1)]
        lag_sample = broad.dropna(subset=["temp_c"]+lags)
        for model, rows in timing.groupby("model", sort=False):
            book.register(timing_path, model, _outcome(rows), lag_sample, context, period, horizon,
                source_sample=broad_path, sample_rule="Complete current and distributed-lag temperature observations")
        slope_path = f"{folder}/sector_slopes.csv"
        slope = book.read(slope_path)
        book.register(slope_path, "H1", _outcome(slope), broad, context, period, horizon,
            source_sample=broad_path, sample_rule="Original broad-panel slope sample", per_sector=True)
        if context != "va":
            aggregate_path = f"data/processed/iip/aggregate_{frequency}.csv"
            aggregate = book.read(aggregate_path, period, freq)
            aggregate = aggregate[aggregate[period].between(broad[period].min(), broad[period].max())]
            book.register(f"{folder}/aggregate_results.csv", "A1", "g_iip_sa", aggregate,
                context, period, horizon, source_sample=aggregate_path, sample_rule="Official aggregate IIP on broad-panel dates")

        extended_path = f"outputs/extended_controls/{context}/regression_results.csv"
        extended = book.read(extended_path)
        for model, rows in extended.groupby("model", sort=False):
            if model == "C0":
                sample, source, rule = broad, broad_path, "Original long-sample broad-panel baseline"
            else:
                source = f"outputs/extended_controls/{context}/"+("sensitivity_80_sample.csv" if model == "C6" else "baseline_sample.csv")
                sample = book.read(source, period, freq)
                rule = "80% coverage dates" if model == "C6" else "90% coverage dates"
                if model == "C5":
                    sample = sample[~sample.year.between(2020, 2022)]
                    rule += "; exclude observation dates in 2020–2022"
            book.register(extended_path, model, _outcome(rows), sample, context, period, horizon,
                          source_sample=source, sample_rule=rule)

        for controls in ["weather", "full", "yoy_weather"]:
            source = f"outputs/outcomes/{context}/{controls}/common_sample.csv"
            sample = book.read(source, period, freq)
            csv_path = f"outputs/outcomes/{context}/{controls}/regression_results.csv"
            results = book.read(csv_path)
            for model, rows in results.groupby("model", sort=False):
                selected = sample[~sample.year.between(2020, 2022)] if model.endswith("X") else sample
                rule = "Exact common outcome-comparison dates"+("; exclude observation dates in 2020–2022" if model.endswith("X") else "")
                book.register(csv_path, model, _outcome(rows), selected, context, period, horizon,
                              source_sample=source, sample_rule=rule)
            if controls in {"weather", "full"}:
                windows = [("base" if controls == "weather" else "full90", sample)]
                if controls == "full":
                    windows.append(("full90_exclude_2020_2022", sample[~sample.year.between(2020, 2022)]))
                for label, selected in windows:
                    for outcome in ["y_short", "y_yoy", 'y_yoy_nsa', "y_log_level"]:
                        scopes = [("pooled", "Pooled sample", selected)]+[(s, g.sector.iloc[0], g) for s, g in selected.groupby("sector_id", sort=True)]
                        for sector_id, sector, group in scopes:
                            stats = book.statistics(group, context, outcome, period, horizon,
                                source_sample=source, sample_rule=label)
                            catalog.append({"sample_window": label, "sector_id": sector_id, "sector": sector, **stats})

        for controls in ["weather", "full"]:
            source = f"outputs/heterogeneity/{context}/{controls}/common_sample.csv"
            sample = book.read(source, period, freq)
            slope_path = f"outputs/heterogeneity/{context}/{controls}/sector_slopes.csv"
            slope = book.read(slope_path)
            for (model, outcome), _ in slope.groupby(["model", "outcome"], sort=False):
                selected = sample[~sample.year.between(2020, 2022)] if model == "H3X" else sample
                rule = "Exact heterogeneity common sample"+("; exclude observation dates in 2020–2022" if model == "H3X" else "")
                book.register(slope_path, model, outcome, selected, context, period, horizon,
                    source_sample=source, sample_rule=rule, per_sector=True)
            for name in ["separate_series", "separate_sector_coefficients"]:
                csv_path = f"outputs/heterogeneity/{context}/{controls}/{name}.csv"
                for outcome in book.read(csv_path).outcome.unique():
                    book.register(csv_path, "separate", outcome, sample, context, period, horizon,
                        source_sample=source, sample_rule="Separate-sector fits on exact common dates", separate=True)

    OUT.mkdir(parents=True, exist_ok=True)
    lookup = pd.DataFrame(book.rows)
    key = ["coefficient_csv", "model", "outcome", "sector_id"]
    assert not lookup.duplicated(key).any()
    assert lookup[["mean_y", "sd_y"]].notna().all().all()
    lookup.to_csv(OUT / "model_statistics.csv", index=False)
    pd.DataFrame(catalog).to_csv(OUT / "sample_means.csv", index=False)
    pd.DataFrame(book.magnitudes).to_csv(OUT / "temperature_magnitudes.csv", index=False)
    pd.DataFrame(book.inventory).to_csv(OUT / "covid_variant_inventory.csv", index=False)
    checks = pd.DataFrame(book.raw_errors)
    checks.to_csv(OUT / "raw_level_checks.csv", index=False)
    summary = {"n_lookup_rows": len(lookup), "n_coefficient_files": lookup.coefficient_csv.nunique(),
        "n_model_outcome_samples": len(lookup[["coefficient_csv", "model", "outcome"]].drop_duplicates()),
        "n_independent_raw_checks": len(checks), "max_abs_value_error": float(checks.max_abs_value_error.max()),
        "max_abs_mean_error": float(checks.mean_abs_error.max()), "max_abs_sd_error": float(checks.sd_abs_error.max()),
        "source_hashes": book.input_hashes,
        "mean_definition": "Signed unweighted arithmetic mean of the model's observed dependent-variable sample",
        "sd_definition": "Descriptive sample standard deviation across observations (ddof=1), not uncertainty in the mean",
        "covid_caution": "Endpoint-date exclusion; YoY observations dated 2023 may contain 2022 base periods",
        "refits_performed": 0}
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Sample context: {len(lookup)} lookup rows across {lookup.coefficient_csv.nunique()} coefficient files; no refits", flush=True)
    return summary


def model_statistics(csv_path: str | Path, model_id: str, *, outcome: str | None = None,
                     sector_id: str | None = None) -> dict | None:
    """Read one saved mean/SD record; return None when a path/model is not registered."""
    path = OUT / "model_statistics.csv"
    if not path.exists():
        return None
    lookup = pd.read_csv(path, dtype={"sector_id": str})
    rows = lookup[lookup.coefficient_csv.eq(relative(csv_path)) & lookup.model.eq(model_id)
                  & lookup.sector_id.eq("pooled" if sector_id is None else str(sector_id))]
    if outcome is not None:
        rows = rows[rows.outcome.eq(outcome)]
    if rows.empty:
        return None
    if len(rows) != 1:
        raise ValueError(f"Specify the outcome for sample context: {csv_path}/{model_id}")
    return rows.iloc[0].to_dict()


get_model_stats = model_statistics


if __name__ == "__main__":
    build_sample_context()
