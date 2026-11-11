# Heat and international visitor arrivals

The separate entry point is `outputs/visitor_arrivals.html`. It is not embedded in the economic report or added to its production pipeline. All new raw, processed and result files are in visitor-specific directories.

## Revision: crisis windows, origin-cluster comparison and native gt tables

The report now includes 120 additional OLS crisis/recovery specifications, without changing the original 269 models. Five calendar policies retain the no-exclusion reference, exclude COVID 2020–2022, exclude GFC 2008–2009 plus COVID, extend those to recovery buffers 2008–2010 and 2020–2023, and isolate pre-COVID dates after the GFC buffer. Growth is computed on the full calendar and the entire [base,current] interval must avoid excluded months. These are transparent sensitivity windows motivated by official historical context, not exact official crisis boundaries or filters on observed growth. In the buffered sample MoM starts February 2011/February 2024 and YoY January 2012/January 2025. Crises before 2008 are outside this source. GFC exclusions have no effect on models beginning in 2017.

The long buffered aggregate MoM model has 139 months: beta −0.7778 per +1°C, HAC(12) SE 1.2037 and nominal p=0.519. The short buffered model has 55 months: beta −4.8325, SE 2.2047, p=0.0327. The full-control pre-COVID check has only 32 months and 13 residual time degrees of freedom, beta −0.5473 and p=0.734. Excluding crises therefore does not establish a stable negative long-history association. The added crisis grid concerns OLS, not new crisis-specific PPML fits.

Primary inference remains common-shock calendar HAC. A separate origin-cluster CR1 comparison is available for 254 pooled panel models (174 original plus 80 crisis fits). Its clusters are nine regions or 42 named markets, with t(G−1) and multiplier G/(G−1)×(N−1)/(N−K_full). K_full counts all explicit origin×month and common-year FE. This allows within-origin serial dependence but assumes independent origins, missing common national shocks. It is not applied to separate-origin slopes, whose scores degenerate with one supporting cluster. Explicit stacked-FE/WLS checks independently reproduce the efficient cluster implementation.

Nominal p tests one coefficient under a zero-coefficient null and the stated inference assumptions. It is not the posterior probability that the null is true. Holm p controls family-wise false positives across the nine/42 temperature tests in an exact profile-table family; it does not adjust all specification choices. Holm does not require independent tests when marginal p-values are valid.

All principal, crisis and sensitivity regressions now have native R `gt` tables, including weather/holiday coefficients, robust SE, nominal p, FE/trend indicators, calendar months, origin-month observations, reference df and outcome context. Fully interacted market tables show all weather coefficients and temperature Holm p. There are 168 native tables, with standalone HTML and rendering source in a downloadable archive. The private R library is in `outputs/visitor_arrivals/R-library`; the HTML report works without R or an internet connection.

## Source and outcomes

The current data.gov.sg M550001 mirror supplies 62 rows over January 2008–September 2026: the official total, nine regions, 42 named markets, nine regional residuals and global Others. Every regional and national counting identity reconciles exactly. The 42-market panel is labelled named origin markets, rather than 42 sovereign countries, and excludes all residual buckets. Named markets represent 96.66% of 2019 and 95.86% of 2025 arrivals.

Source definitions concern residence and arrival events, not nationality or unique visitors. Malaysian citizens arriving by land are excluded. M550001's `adjustmentType` metadata incorrectly says seasonally adjusted; its plain monthly identity and values differ from explicit SA M550002. The primary analysis consistently uses NSA counts; the SA national total is a sensitivity. SA coverage is insufficient for a complete origin panel.

MoM and YoY outcomes are 100 log differences, computed on the full calendar before filtering. Log and raw levels are descriptive. No arbitrary constant is added to zero counts. The named-market source has 50 zeros in 2020–2021. Each pooled log comparison uses a common balanced date mask across all three cuts. Separate national/region positive-date fits and PPML retain broader support.

## Estimands and inference

Common weather plus origin × calendar-month and shared year effects identify a conditional temporal association. For a balanced origin panel, the common-slope coefficient is exactly the coefficient from the equal-origin mean time series. The implementation uses that reduction; explicit original stacked FE fits independently verify it. Aggregate log growth is a distinct estimand. Fixed 2017–2019 arrival-share weights provide a labelled reweighting sensitivity rather than an accounting reconstruction or ex ante forecast.

Primary pooled uncertainty is calendar-aware Bartlett HAC with 12 and 24 months. Coefficient-influence vectors are placed on the real calendar, preserving gaps. The small-sample multiplier is T/(T−k), with the temporal design rank k, and intervals/p-values use approximate t(T−1). The number of markets does not enlarge the number of independent destination-weather observations.

Fully interacted market profiles allow every weather slope and seasonal/year nuisance parameter to vary by origin. Native multi-outcome PyFixest fits produce separate-origin slopes, with a joint HAC covariance across the system. Holm corrections apply within each panel × outcome × sample × bandwidth family. Omnibus equality tests use only full-rank contrast covariance; none is generated using a pseudoinverse. The short country MoM omnibus is high dimensional and ill-conditioned, and short country YoY is singular. They are not headline evidence.

Zero-retaining native PyFixest PPML fits expected count levels with origin × month and shared year effects. Weighted FE projection and month-level score sums produce the robust covariance. Its 100β coefficient is in log points of the expected count, a different outcome model from growth. Each market and every zero remains in the PPML samples. All PPML score equations and representative explicit weighted FE projections are independently checked.

## Controls and robustness

The longer window uses temperature, monthly rainfall per 100 mm and relative humidity. The 2017–2025 comparison adds wind and PM2.5 using the unchanged economic project's frozen inputs and monthly 90% coverage rules; an 80% check is separate. The base-control comparison uses identical full-control dates. Weather differences require valid controls at both weather endpoints, including coverage at both endpoints for added controls.

The grid includes current and three lagged temperatures with a matched-current comparator; covariance-based lag sums; future-temperature timing diagnostics; changing seasonal patterns; a quadratic temperature term centered at 28°C; arrivals per day; current/previous-month Lunar New Year window counts; current versus differenced weather; official SA aggregate outcomes; and leave-one-year-out checks. Lunar calendars come from 19 cached official HKO annual tables.

Endpoint exclusions remove 2020–2022 observation dates. Strict growth exclusions also remove contaminated base dates, including January 2023 for MoM and all 2023 for YoY. The coverage screen already removes January 2023 in the short MoM sample. The short strict samples have 66 MoM months and 56 YoY months. Dynamic strict models also exclude pandemic lagged-weather months. All transformations precede filters.

ADF/KPSS reference screens use contiguous usable histories after fixed month-mean adjustment. ADF uses AIC up to 24 lags, KPSS automatic bandwidth, and warnings/bounded p-values are saved. Growth uses a constant and log levels a trend. Exact gapped samples are not assigned stationarity clearance. These exploratory reference screens are not formal causal or cointegration validation.

## Findings and limits

The 66-month full-control aggregate MoM coefficient is −4.0757 per +1°C (12-lag HAC SE 1.9324); +0.1°C corresponds to −0.4076 growth log points, with nominal 95% interval [−0.7935, −0.0216]. Before 2020 the coefficient is +0.7644 (SE 0.8882), with +0.1°C interval [−0.0992, +0.2520]. Changing seasonality weakens the short estimate. Region and country averaging, bandwidth, outcome and sample choices matter. PPML count-level coefficients show no clear association, and no individual MoM origin slope passes Holm at 12 lags in the two profile windows. The evidence does not establish a stable causal negative heat relationship.

Origin travel demand, capacity, exchange rates, events and restrictions remain plausible confounders; destination weather is often realised after travel is planned. Positive growth coefficients do not establish beneficial warming, and negative coefficients do not identify a national tourism loss or permanent damage. Nominal p-values across the 269-model OLS grid are exploratory rather than a model-selection criterion.

## Run and verify

```sh
uv run python scripts/visitor_run.py
# Reuse fitted estimates and rebuild/check the gt tables and report:
uv run python scripts/visitor_run.py --reports-only
```

First-time preparation also needs the cached lunar-calendar extraction under `data/raw/visitor_arrivals/calendar/`; its script rebuilds the monthly timing CSV. Sources are cached and hash-checked; reruns do not silently refresh vintages. Independent numerical verification covers every OLS fit, mask, HAC and profile system, zero retention and PPML covariance. The HTML runtime harness checks every selector combination, embedded images and downloadable evidence. A native Safari check confirms the report and MoM/YoY controls render, including the singular-country-test warning. The economic HTML changed during concurrent annual-analysis work elsewhere in the folder. It is not an input to these visitor estimates; the receipt preserves before/current hashes, and actual analytical input hashes still match. Visitor scripts do not modify or integrate with that separate report.
