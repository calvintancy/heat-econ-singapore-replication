# Sample means and the existing COVID-exclusion variants

`scripts/sample_context.py` computes descriptive context for saved regressions without estimating models or changing coefficient files. Run `.venv/bin/python scripts/sample_context.py` after the outcome/heterogeneity outputs exist, or call `build_sample_context()` from the pipeline.

## Meaning of the statistics

`mean_y` is the **signed arithmetic mean of the actual dependent variable on that model's exact observation sample**. `sd_y` is its descriptive sample standard deviation, with denominator N−1. Neither is residual variation or uncertainty in the mean. Pooled sector-panel statistics give every sector-period observation equal weight; they describe an average across sectors and do not estimate growth of aggregate Singapore GDP. Sector-specific records retain each sector's own signed mean, including negative values.

Growth outcomes remain `100 × log changes`: MoM/QoQ are not annualised, and YoY compares the same month/quarter of the preceding year. These are growth log points, rather than exact percentage changes. The separately saved exact YoY percentage outcome has its own mean and units. Level models retain means of their own dependent variable: `100 × ln output`, real VA in chained-2015 S$ million, or IIP index points (2025=100). A mean of log output is not labelled mean output in original units.

The lookup distinguishes original long samples, common outcome-comparison samples, detailed-panel dates, control-coverage dates and endpoint exclusions. For example, the original monthly broad IIP regressions use 416 months, while the later common-outcome comparisons use 405 months. Their means should not be interchanged. Counts and first/last dates are checked against every registered coefficient file. The endpoints do not imply that coverage-screened or COVID-excluded dates are contiguous.

## Files and lookup interface

- `outputs/sample_context/model_statistics.csv`: keyed by relative coefficient-CSV path, model ID, outcome variable and sector ID. `sector_id=pooled` denotes the pooled sample. It covers the original S specifications, earlier distributed lags and sector slopes, official aggregate IIP, extended controls, outcome/YoY-weather comparisons and sector heterogeneity.
- `sample_means.csv`: a clean long-format table for each context and outcome, pooled and by sector. Its windows are `base` (the long common outcome sample), `full90` (90% wind/PM2.5 coverage dates), and `full90_exclude_2020_2022`.
- `temperature_magnitudes.csv`: the saved temperature coefficient and `0.1 × coefficient` alongside the matching mean/SD. The exposure term and relative/absolute estimand are preserved. A trailing-year-mean or YoY-change coefficient does not describe an isolated current-period temperature pulse. No coefficient/mean-growth ratio is calculated.
- `covid_variant_inventory.csv`: exact samples for already estimated C5, G2X/L1X/L3X and H3X models.
- `raw_level_checks.csv`, `verification.json`: independent raw-level reconstruction checks, source hashes and coverage of the lookup.

The module has no imports from the model-fitting or table-rendering modules, preventing a circular dependency. The table adapter can use:

```python
from sample_context import model_statistics

pooled = model_statistics(csv_path, model_id)
sector = model_statistics(
    "outputs/heterogeneity/va/full/sector_slopes.csv",
    "H3", outcome="y_short", sector_id="1.1.2",
)
```

The function reads the generated lookup and returns one dictionary, or `None` for an unregistered path/model. It requires an explicit outcome when a model has multiple outcomes. `get_model_stats` is an alias. The individual-sector coefficient files have no saved model column; their lookup ID is `separate` and requires `sector_id`. Aggregate VA extensions can use statistics saved directly in their own coefficient rows; no aggregate-VA refit or result file is introduced here.

## COVID variants already available

The existing short-window wind/PM2.5 regressions already include the requested observation-date exclusion:

| Result family | Existing model | Scope |
|---|---|---|
| `outputs/extended_controls/{context}/regression_results.csv` | C5 | Full weather controls; MoM/QoQ log growth; excludes observations dated 2020–2022 |
| `outputs/outcomes/{context}/full/regression_results.csv` | G2X, L1X, L3X | Full controls; YoY log growth, log output level and raw output level; same endpoint exclusion |
| `outputs/heterogeneity/{context}/full/sector_slopes.csv` | H3X | Full controls; sector-specific weather slopes and year effects, for all three heterogeneity outcomes; same endpoint exclusion |

The original S5 and long-weather outcome/heterogeneity families also have analogous exclusions; their sample statistics remain separately registered. The exact-percentage YoY G3 sensitivity does not have a separately saved endpoint-exclusion variant, which is not needed to supply the existing requested C5 comparison.

In the current cached snapshot, C5 retains **19 quarters** for VA and quarterly IIP and **66 months** for monthly IIP, rather than 29 quarters/100 months in the corresponding full-coverage samples. The observation dates run from 2017 to 2025 with intervening exclusions. This is an **endpoint-date exclusion**, not a pandemic-free exposure history: a YoY outcome dated in 2023 can still compare against 2022. All growth transformations are formed before filtering the sample; gaps are not crossed by recomputing growth after deletion.

## Independent validation

The script reads the cached official SingStat SA and NSA level tables independently, rather than reusing the outcome-building functions. It maps observations to real calendar periods, forms quarterly IIP levels as the arithmetic average of all three monthly index levels, and reconstructs the requested short/YoY/log/raw outcomes with `math.log`. Means and SDs are cross-checked using `math.fsum`. This validates every registered sample, including NSA, detailed sectors and exclusion dates. It also verifies N, sector counts, distinct-period counts and date bounds against the saved estimation metadata. No regressions are refitted.
