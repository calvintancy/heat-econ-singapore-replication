# Outcome and temperature specifications: independent review

Reviewed 10 November 2026. This note records design advice before interpreting the new estimates. Existing MoM/QoQ regressions should remain intact; comparisons on a common sample are additional estimates.

## The temperature regressor does not automatically have to become YoY

Let `a_it = ln(A_it)` and let `h = 12` for monthly data or `h = 4` for quarterly data. The outcome definitions are:

| Outcome | Definition | Coefficient units for a 1°C increase |
| --- | --- | --- |
| MoM/QoQ log growth | `100(a_it − a_i,t−1)` | Monthly/quarterly growth log points |
| YoY log growth | `100(a_it − a_i,t−h)` | Year-on-year growth log points |
| Exact YoY percentage growth | `100(A_it/A_i,t−h − 1)` | Year-on-year percentage points |
| Log output level | `100 a_it` | Output log points, approximately percent for small changes |
| Raw output level | `A_it` | VA: chained-2015 S$ million; IIP: index points |

These are different outcomes. Neither a change of outcome nor a wish to compare with last year determines the weather transformation by itself. Temperature is measured in degrees, so a YoY temperature change is a degree difference, not a percentage growth rate.

### Independent algebra: a level model

If the proposed structural relationship is `a_it = nuisance_it + β T_t + δ X_t + u_it`, differencing it over one year produces

`a_it − a_i,t−h = Δ_h nuisance_it + β(T_t − T_t−h) + δ(X_t − X_t−h) + (u_it − u_i,t−h)`.

Thus a YoY temperature difference follows from this particular level model. Its other weather controls must also be differenced. The unrestricted endpoint model contains both current and same-period-last-year weather. The temperature difference restriction is `β_current + β_lag = 0`; a complete test of the level-model restriction must impose corresponding restrictions jointly on every included weather variable. A temperature-only test is useful when labeled as such.

Sector and same-season fixed effects cancel in an exact annual difference; year effects transform into differences of year effects. Maintaining the project's original fixed-effect structure in the empirical comparison is permissible, but it gives a more flexible nuisance model rather than reproducing the exact differenced level equation.

### Independent algebra: a period-growth model

If instead `a_it − a_i,t−1 = nuisance_it + β T_t + δ X_t + e_it`, then

`a_it − a_i,t−h = Σ[j=0,…,h−1] nuisance_i,t−j + β Σ[j=0,…,h−1] T_t−j + δ Σ[j=0,…,h−1] X_t−j + Σ[j=0,…,h−1] e_i,t−j`.

The implied temperature exposure is the annual window of period temperatures, not the endpoint difference. If its regressor is the equal-period mean, the implied coefficient is `hβ`. A day-weighted annual mean is a different descriptive exposure; it is not this exact algebraic sum when the underlying period-growth equation weights months/quarters equally. Use all annual-window weather means together for a coherent comparison. With a distributed-lag growth model, annual exposures have convolution weights rather than necessarily uniform weights.

A YoY outcome on current temperature alone is also a valid descriptive regression. It asks how annual growth covaries with current conditions; it does not estimate the same quantity as either restricted model above.

## Small useful comparison set

1. Preserve the original MoM/QoQ results and their original samples.
2. On common dates within each outcome/frequency/control set, compare current-temperature regressions for short log growth, YoY log growth, log level and raw level. Keep exact YoY percentage growth as an explicitly labeled companion. Use the same rainfall/RH controls in the long sample and a separately labeled wind/PM2.5 sample.
3. On a common long sample, compare YoY growth on current weather, trailing-year mean weather, YoY changes in weather and unrestricted current/lagged weather. Keep this four-way exercise to rainfall/RH controls; the sparse quarterly full-control sample cannot support an elaborate annual-exposure design reliably.
4. For levels, a common season-by-centered-year slope sensitivity can reveal sensitivity to an evolving within-year shape. Drop a reference season because the common linear year trend is already in year fixed effects. It does not resolve stochastic trends, omitted common shocks or climate-effect identification.

Sector-specific linear trends do not provide a meaningful sensitivity test for the common temperature coefficient in this balanced, equal-sector-weight panel. With sector-by-season and year fixed effects, their common component is already absorbed and their cross-sector deviations have zero cross-sector mean. Their addition leaves the common weather coefficients unchanged by construction.

## Available samples and implementation constraints

Build annual outcomes from the complete raw level series before weather merges, sample restrictions or exclusions. Use calendar `t−h` lookups or complete contiguous raw series, not a row shift after dropping observations.

Expected balanced core comparison samples are:

| Context | Long-sample common start | Last period | National periods |
| --- | --- | --- | --- |
| VA quarterly | 1982Q1 | 2026Q2 | 178 |
| IIP monthly | January 1993 | September 2026 | 405 |
| IIP quarterly | 1993Q1 | 2026Q3 | 135 |

The endpoint-lag weather comparison moves VA to 1983Q1 (174 quarters), because the weather series starts in 1982. IIP has sufficient earlier weather history and retains the starts above. The full-control comparison retains the existing 100 months or 29 quarters under the 90% coverage rule. Early IIP histories differ by cluster: trim to the latest sector-specific first valid annual outcome so every retained date has all six clusters.

The 29-quarter full-control regression has 17 parameters in its equivalent national-time regression, leaving 12 residual time degrees of freedom. Adding the three changing-seasonality slopes leaves nine. Excluding observations dated 2020–2022 leaves 19 quarters and only five residual time degrees of freedom before adding season trends. Rank checks and these time degrees of freedom should be reported; repeated sector rows do not create additional weather observations.

YoY outcomes dated 2023 still compare against 2022 levels. Consequently, a sensitivity excluding only endpoint observations dated 2020–2022 must be labeled exactly that way, not as a pandemic-free sample. A stricter exclusion can remove 2023 endpoints too, but would further reduce the quarterly sample.

## Inference and interpretation

Annual outcomes overlap. Even independent one-period growth errors generate annual-growth errors sharing up to `h−1` observations. Retain the actual calendar spacing after exclusions. The existing 12-month/four-quarter Bartlett covariance covers that mechanical overlap; a 24-month/eight-quarter sensitivity is useful because economic persistence may extend beyond the overlap. This is not a proof that either bandwidth is optimal, especially for persistent output levels or the 29-quarter sample. The `t(T−1)` reference remains approximate.

Log-level fits are preferable to raw-level fits for cross-sector comparison because proportional responses are interpretable across differently sized sectors. A common raw-VA coefficient imposes an equal absolute S$ million response across sectors. It is not an aggregate GDP coefficient. IIP index-point coefficients depend on the index normalization, whereas log specifications are invariant to a constant sector-specific rescaling. Sector IIP levels should not be summed into a national index.

Year fixed effects remove annual national temperature shifts and annual economic shifts. The identified association comes from within-year weather variation left after seasonality and controls. This design cannot estimate permanent national warming effects; full month/quarter-date fixed effects would absorb every common national weather regressor. In levels, annual fixed effects and changing-seasonality slopes also do not establish cointegration or eliminate all nonstationarity concerns. Treat those estimates as descriptive level associations and compare them with growth results rather than selecting the strongest significance.

## AEA references checked

[Dell, Jones and Olken (2012), AEJ: Macroeconomics](https://www.aeaweb.org/articles?id=10.1257/mac.4.3.66), including the [published paper hosted by MIT](https://economics.mit.edu/sites/default/files/publications/120626%20Climate%20Published.pdf), equations (1)–(4), pp. 72–74, distinguishes output-level and growth mechanisms. Its growth regression uses temperature levels and lags; the distinction is investigated through dynamics, not by requiring temperature changes whenever growth is the outcome. Footnote 10 discusses nonstationarity concerns with integrated output-level models. That multicountry design has geographic weather variation that this Singapore sector panel lacks.

[Dell, Jones and Olken (2014), Journal of Economic Literature](https://www.aeaweb.org/articles?id=10.1257/jel.52.3.740) reviews identification from weather fluctuations within geographic units. Its general design lesson does not make common Singapore temperature exogenous to every quarterly macro shock. The paper supports using a clearly stated weather-economy model rather than mechanically matching transformation names.

## Completed independent implementation audit

The added implementation passed [the independent verifier](../scripts/verify_outcomes.py), which imports none of the estimation helpers. Its record is [outputs/outcomes/verification.json](../outputs/outcomes/verification.json). The audit reconstructed 5,282 processed sector-period rows directly from raw level and monthly weather sources, using calendar endpoint lookups rather than shifts of restricted panels. It verified log growth, exact percentage growth, log/raw levels, the annual-growth telescoping identity, current/lagged/differenced weather and equal-period annual-window means.

All 144 model/bandwidth fits were checked through separate NumPy QR regressions on collapsed national-time means and manually assembled calendar-aware Bartlett score covariance, with the stated full-panel finite-sample factor. Checks included every reported weather coefficient, SE, p-value and confidence interval, the changing-seasonality columns, recovered total panel R², reported national-time residual degrees of freedom and the temperature-only endpoint restriction. The largest absolute coefficient difference was 1.4×10⁻¹⁰; the largest absolute SE difference was 7.0×10⁻⁹, in a VA original-unit model. No material estimation or outcome-transformation error remained.

Verified long core dates are 1982Q1–2026Q2 for VA, January 1993–September 2026 for monthly IIP and 1993Q1–2026Q3 for quarterly IIP. The annual-weather comparison uses 174, 405 and 135 national periods respectively. Full controls use 29 quarters or 100 months. Main comparisons share dates; endpoint-exclusion checks are separately labeled and do not claim that 2023 annual outcomes are free of 2022 base-period effects.

The audit identified and the report author corrected one presentation issue: a note asserting identical dates across an auxiliary table containing both endpoint-excluded fits and a full-common-sample raw-level fit. The five-column main comparison itself had correctly matched dates. Remaining limitations are methodological interpretation and short-time-sample inference, as stated above, rather than an unresolved coding error.
