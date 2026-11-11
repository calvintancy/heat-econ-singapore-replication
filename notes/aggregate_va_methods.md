# Official aggregate real value added

Implemented 11 November 2026. The aggregate analysis is additional work; it does not replace any earlier sector-panel, IIP, outcome-comparison or heterogeneity estimate.

## Source and estimand

The outcome is SingStat series `1.4`, **Gross Value Added At Basic Prices**, from the cached [seasonally adjusted M015662 table](https://tablebuilder.singstat.gov.sg/table/TS/M015662) and [NSA M015661 table](https://tablebuilder.singstat.gov.sg/table/TS/M015661). The source unit is million Singapore dollars in chained-2015 prices. The already verified `data/processed/aggregate_gva.csv` supplies the weather-aligned series; its growth values are independently checked against complete raw histories.

This is the published national aggregate. It includes ownership of dwellings, which is separately reported as series `1.3` and is outside the earlier 11-sector panel. The source separately reports taxes on products as `1.5`; those are outside basic-price GVA. It is therefore important to label the outcome aggregate GVA rather than silently calling it GDP at market prices. The analysis never sums chain-linked sector values or treats the equal-sector growth average as aggregate growth.

The cached aggregate sample contains 178 quarters, 1982Q1–2026Q2. No sector dimension is manufactured. Fixed effects are quarter-of-year and year indicators. Full quarter-date effects would absorb national temperature and all other common weather variables.

## Outcomes and comparisons

| Outcome | Construction |
| --- | --- |
| QoQ log growth | `100[ln A(t) − ln A(t−1)]` |
| YoY log growth | `100[ln A(t) − ln A(t−4)]` |
| Exact YoY percent growth | `100[A(t)/A(t−4) − 1]` |
| Log level | `100 ln A(t)` |
| Raw level | Official chained-2015 S$ million |

Growth is not annualised. The 1982Q1 observations use the required pre-1982 official raw levels. The first outcome values are not calculated by shifting an already cropped weather sample.

The `baseline` group mirrors the existing specification sequence: temperature alone, rainfall, humidity, pre-2020 dates, exclusion of observations dated 2020–2022, NSA QoQ growth and equal-month temperature. The `weather` group compares all outcomes on the same long-sample dates with current temperature, rainfall and humidity. The `full` group repeats that comparison with current wind and PM2.5, both on all 29 retained quarters and after excluding the 2020–2022 endpoint years. The latter is the default short-sample table and has 19 retained quarters. Every growth definition has its own explicitly saved exclusion fit.

The `matched_controls` group adds wind and PM2.5 step by step on identical dates, with both all-date and excluded-year versions. The existing 90% monthly temporal-coverage rule and requirement that all three months pass are retained. Wind is in m/s and PM2.5 is scaled per 10 µg/m³. Changi quarterly temperature and humidity are day weighted, and rainfall is the quarterly total in 100 mm.

Each result row carries the exact-sample `mean_y` and `sd_y`, sample QoQ/YoY mean and SD, and the temperature coefficient, SE and confidence interval scaled to +0.1°C. Mean growth is an observed descriptive statistic, not a fitted causal benchmark. No ratio to the mean is presented as a damage share.

## Annual weather and dynamic checks

The separate `yoy_weather` group compares current weather, trailing-four-quarter means, annual differences in all weather variables and unrestricted current plus same-quarter-last-year weather. All four models share 174 dates beginning 1983Q1. Annual means weight quarters equally, matching the algebraic accumulation of equal-weight quarterly log growth. The temperature-only opposite-coefficient restriction is tested explicitly; it is not labeled a joint test of every static-level assumption.

The `distributed_lag` group compares current temperature with current plus four quarterly temperature lags, on the same 174-quarter sample. Rainfall and humidity remain current controls. The coefficient sum and its full covariance-based uncertainty describe the fitted sum of growth responses over five quarters to a one-quarter pulse. They do not identify permanent-warming damage.

## Inference and persistence

The singleton-panel Driscoll–Kraay implementation reduces to calendar-aware Bartlett time-series HAC. Its finite-sample multiplier is exactly `T/(T−K)`: `T/(T−1) × (T−1)/(T−K)`. Both four- and eight-quarter bandwidths are saved. Confidence intervals and two-sided nominal p-values use the project's approximate `t(T−1)` reference. Missing dates after coverage screens or exclusions retain their actual calendar positions.

The 19-quarter full-control fits have only five residual time degrees of freedom. Their stars and intervals are fragile. Excluding observations dated 2020–2022 does not make annual growth dated 2023 independent of a 2022 base period. The report states that distinction rather than describing the sample as pandemic free.

ADF and KPSS diagnostics use only the contiguous 178-quarter official SA aggregate history. QoQ/YoY growth uses a constant; log levels use a constant and linear trend. ADF selects lags by AIC up to four, and KPSS automatic bandwidth is capped at four. P-value bounds are preserved. No gapped full-control sample or estimated fixed-effect residual is treated as a regular raw series. Growth is the main specification; levels remain descriptive, and the diagnostics do not establish causal validity or a permanent growth mechanism.

## Files and verification

Entry points are `run_aggregate_va()` in `scripts/aggregate_va.py`, `verify_aggregate_va()` in `scripts/verify_aggregate_va.py` and `build_aggregate_va_report()` in `scripts/aggregate_va_report.py`. Preparation, fits and verification are separate from report generation so the consolidated pipeline can attach sample context before rendering.

Results are under `outputs/aggregate_va/{baseline,weather,full,matched_controls,yoy_weather,distributed_lag}/regression_results.csv`. Each group also saves all coefficients, a specification manifest and the exact sample of every model. The offline native MakeTables report is `outputs/aggregate_va.html`; HTML and LaTeX table fragments are retained under `outputs/aggregate_va/tables`.

For the master overview, filter `hac_lags == 4` and `term == 'temp_c'`. Long QoQ/YoY are `weather/G1` and `weather/G2`; short full-control all-date estimates are `full/G1` and `full/G2`; the default excluded-year versions are `full/G1X` and `full/G2X`.

Independent verification reconstructs aggregate outcomes from the official raw SA/NSA row, uses calendar lookups for annual and lagged values, reconstructs weather transformations, and fits separate NumPy regressions with manually assembled Bartlett covariance. All 82 model/bandwidth fits passed, including mean/SD and +0.1°C scaling, coefficient-sum tests and exact estimation dates. The largest coefficient difference was below 2.4×10⁻¹⁰; the largest SE difference was below 7.0×10⁻⁸ in an original-currency-unit model. ADF, KPSS and ACF statistics also pass direct numerical verification. The record is `outputs/aggregate_va/verification.json`.
