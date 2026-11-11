# Stationarity coverage and claim audit

Read-only methodological review, 11 November 2026. Applied the `audit-the-claim-layer` skill. This note assesses the existing diagnostics and the proposed unified registry; it does not change outcomes, select a preferred regression, or certify that the proposed expansion has already run.

**The existing evidence does not support saying that all series or all regressions are stationary.** Computation verified, diagnostic coverage complete, stationarity evidence compatible, and coefficient statistically significant are four different claims. They need separate fields and labels.

## Existing coverage and actual evidence

`scripts/stationarity_diagnostics.py:19` defines only three output variables: SA one-period log growth, SA YoY log growth and `100 ln` SA output. Its weather list at line 22 contains only temperature, rainfall and humidity. The context loop at line 244 covers 11 broad VA sectors and six broad IIP clusters, monthly and quarterly. It aligns these series to the earlier contiguous long comparison samples and also tests contiguous pre-2020 histories. The current saved output contains **324 test rows / 162 ADF–KPSS pairs**. These counts include windows and temperature deterministic-specification sensitivity; they are not 162 independent economic series.

`scripts/aggregate_va.py:115` adds diagnostics for the national SA aggregate, but the loop at line 122 still includes only those same three outcome definitions. It has three long-history pairs, not diagnostics for every aggregate model or control. The new detailed VA19/IIP18 industries, published IIP totals, raw levels, NSA and exact-percent growth, wind, PM2.5 and alternative weather exposures are outside the old diagnostic inventory. Their regression arithmetic can be verified without their stationarity having been examined.

The old primary long-sample pairs already show:

| Series group | Compatible with chosen stationarity null | Mixed | Unit-root concern | Inconclusive |
| --- | ---: | ---: | ---: | ---: |
| VA11 one-period log growth | 6 | 5 | 0 | 0 |
| VA11 YoY log growth | 5 | 5 | 1 | 0 |
| Six-cluster monthly IIP one-period growth | 6 | 0 | 0 | 0 |
| Six-cluster monthly IIP YoY growth | 5 | 1 | 0 | 0 |
| Six-cluster quarterly IIP one-period growth | 6 | 0 | 0 | 0 |
| Six-cluster quarterly IIP YoY growth | 5 | 1 | 0 | 0 |
| All 23 broad-context log-level series | 0 | 2 | 20 | 1 |

These counts are from `outputs/heterogeneity/diagnostics/evidence_counts.csv`. Monthly and quarterly IIP are related aggregations, not independent replications. Temperature and rainfall are compatible under their chosen specifications; humidity has mixed evidence in VA and a unit-root concern in both IIP contexts. Temperature/humidity use a trend specification, so compatibility means **trend-stationarity**, not stationarity around a constant mean.

The national GVA results in `outputs/aggregate_va/diagnostics/summary.csv` are: QoQ growth compatible (ADF p≈1.26×10⁻²²; KPSS p≈0.077); YoY growth mixed (ADF p≈0.0023; KPSS p≈0.033); log level a concern (ADF p≈0.939; KPSS p<0.01). A universal stationary-growth claim would therefore be wrong even before adding the finer sectors.

## Test design and defensible extension

ADF tests a unit-root null. KPSS tests stationarity around the specified constant or linear trend. Non-rejection proves neither null; a paired screen cannot establish an exact integration order. This interpretation follows the version-matched [statsmodels 0.14.6 ADF documentation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.adfuller.html) and [KPSS documentation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.kpss.html). KPSS p-values outside its tabulated 0.01–0.10 range must retain `<`/`>` bounds, not be displayed as exact probabilities.

The current implementation at `stationarity_diagnostics.py:103` selects ADF lags by AIC with a one-year cap. **All 46 existing YoY ADF runs, across long and pre-2020 sector histories, select that cap.** Across all 162 ADF runs, 60 bind; 95 of 162 KPSS bandwidth selections are truncated by the legacy cap at line 113. This is a material reason to report lag/bandwidth sensitivity. It is not evidence that the series are necessarily nonstationary.

The agreed extension is defensible: predeclare ADF AIC selection up to `2h` (24 months/eight quarters, subject to sample feasibility) and uncapped automatic KPSS as the new primary diagnostic pair; preserve the old ADF `h`/KPSS capped-`h` pair as a named sensitivity. Save requested/feasible caps, actual lags, binding flags, effective n, deterministics, warnings and critical values. Do not choose the specification that produces the most favourable classification. A robustness label can require compatible evidence across both declared pairs, while displaying disagreements rather than flattening them into a pass/fail.

For every actual stochastic Y and weather X, the registry should identify source series, sector, frequency, transformation, exact model date mask, coverage, deterministic specification and diagnostic result. Include NSA/raw/exact-percent outcomes and the fitted lag, annual-change and annual-window weather variants. Derive transformations before cropping. Include the national totals and new detailed sectors explicitly. Repeated common weather across sectors is one exposure series, and a harmless unit rescaling is not a new independent test. Deterministic FE indicators, constants, time trends and sector multipliers should have an explicit “deterministic basis / inherited underlying series” disposition rather than meaningless unit-root tests.

Keep **exact estimation-window evidence** separate from **background-history evidence**. A canonical long history can support context, but its rejection pattern is not a test of every shorter matched sample. All unavailable, constant, too-short and gapped cases must stay in the registry with reasons; a completed audit is a complete inventory of dispositions, not a claim that every series was testable.

The existing full-control sample has 100 months over 108 calendar months, or 29 quarters over 36 calendar quarters. Its longest uninterrupted stretches are only **32 months / 10 quarters**. Removing 2020–2022 leaves additional gaps. Ordinary ADF/KPSS must not be applied after compressing those dates, and no interpolation should be introduced to enable a test. KPSS does not handle missing values. The existing five-seasonal-cycle minimum is a transparent eligibility rule, not a guarantee of useful power; these exact full-control windows are unavailable under that rule. Wind can have a longer unfiltered monthly background history, but low-coverage months in that history do not validate the stricter 90%-eligible regression window.

## Seasonality, overlap and residual interpretation

Fixed month/quarter offsets remove deterministic seasonality, not stochastic seasonal unit roots. ADF rejection at the ordinary zero frequency does not rule out unit roots at seasonal frequencies. That distinction is explicit in [Hylleberg, Engle, Granger and Yoo (1990)](https://doi.org/10.1016/0304-4076(90)90080-D) and the monthly extension examined by [Beaulieu and Miron (1993)](https://doi.org/10.1016/0304-4076(93)90018-Z). The current screen should be labelled “ordinary ADF/KPSS after stated seasonal treatment”; it should not be advertised as a comprehensive seasonal-unit-root test. Dedicated seasonal-root tests would be a separately specified extension if that claim were required.

The theoretical literature permits deterministic seasonal dummies in KPSS without changing its limit distribution; see [The KPSS test with seasonal dummies](https://doi.org/10.1016/S0165-1765(02)00127-1). That result does not resolve finite-sample power, structural breaks, changing seasonality, or seasonal roots. The current weather preprocessing estimates seasonal offsets with a trend present and subtracts only the offsets. Its exact choices should remain visible rather than being described as an official adjustment of all weather variables.

YoY log growth is a sum of `h` consecutive short-period log growth rates. Even if those short growth rates are independent, adjacent annual growth observations share `h−1` components and are serially dependent. Overlap is therefore compatible with stationarity. Larger lag/bandwidth sensitivities address part of that dependence; they do not make nominal finite-sample inference exact. [The Federal Reserve’s original long-horizon inference paper](https://www.federalreserve.gov/pubs/ifdp/2006/853/revision/ifdp853r.htm) explains the overlap problem in its own predictive-regression setting; its specialised corrections must not be transplanted mechanically into this contemporaneous heat specification.

Season/year effects and robust covariance do not themselves establish stationarity, cointegration, or causal identification. Conversely, a concerning raw humidity screen does not automatically prove that every conditional year-FE growth regression is invalid. The raw-series tests do not determine the integration order of the conditional disturbances. Do not use ordinary residual ADF critical values as a cointegration certificate. A formal cointegration test has a different null and assumptions; for example, [statsmodels’ Engle–Granger implementation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.coint.html) assumes I(1) variables and tests no cointegration. Applying it indiscriminately to mixed-order weather, panel FEs or short gapped samples is not a solution.

## No test-selected “stationary analysis”

Retain the originally requested growth outcomes and report the evidence beside them. Repeatedly differencing until ADF rejects, adding/removing trends or controls until KPSS does not reject, dropping sectors with unfavourable tests, or choosing the sample/covariance with the smallest regression p-value changes the estimand and introduces selection. Filtering sectors also changes the equal-sector population represented by the panel.

This is a statistical issue beyond presentation. [Elliott and Stock’s original time-series pretesting research](https://www.nber.org/system/files/working_papers/t0122/t0122.pdf) shows that unit-root pretests followed by conventional second-stage inference can distort test size because the stages are dependent and pretests make errors. More generally, [Leeb and Pötscher (2005)](https://doi.org/10.1017/S0266466605050036) explain why ignoring data-driven model selection invalidates conventional post-selection inference. A series-screening result cannot certify the already estimated coefficient p-values. The new larger-cap diagnostic plan is a declared sensitivity motivated by binding caps, not an ex ante untouched replication design; preserve that history honestly.

Use evidence labels with their nulls visible:

| ADF at 5% | KPSS at 5% | Defensible label |
| --- | --- | --- |
| Reject | Do not reject | Evidence compatible with level/trend stationarity under the stated specification |
| Do not reject | Reject | Unit-root/nonstationarity concern; not proof of a unit root |
| Reject | Reject | Mixed evidence |
| Do not reject | Do not reject | Inconclusive |
| Unavailable | Any | Not tested / unavailable, with the precise reason |

For trend specifications the label must explicitly retain “trend.” Sensitivity disagreement should be visible. Passing numerical verification can mean the statistics were computed correctly; it must never be converted into a green “all stationary” badge.

## Claim-layer and sector-significance checks

At inspection, these existing statements are appropriately narrow and should remain so:

- `stationarity_diagnostics.py:322` saves `all_tests_completed=True`. It means the 324 enumerated legacy tests completed; it does not mean all fitted variables are covered or all tests favour stationarity. The line-327 printed count is similarly scoped.
- `aggregate_va.py:122` enumerates three diagnostics. A claim of aggregate diagnostic coverage must name those three transformations until the new registry adds the others.
- `granular_report.py:290` explicitly says earlier diagnostics do not establish stationarity for each new detailed series. Do not remove that qualification merely because a new registry has run; replace it with the actual new coverage/evidence states.
- `heterogeneity_report.py:296` says log-level persistence concerns affect most sectors. The saved broad-context results support that statement: 20 concerns, two mixed cases and one inconclusive case among 23 log-level context-series.
- `outcome_report.py:133` correctly states that seasonal/year effects do not guarantee stationary residuals. That limitation remains after expanded testing.

For clearer sector results, display the temperature coefficient, 95% interval, numeric p-value and an explicit “significant at 5% / not significant at 5%” status computed from the stored unrounded p-value. Show the covariance and sample count alongside it. Explain `*` at 10% separately from 5% significance; avoid printing small p-values as `0.000`. Keep stationarity-evidence labels separate because ADF/KPSS test different nulls from the coefficient’s zero-effect null.

For multiple sector slopes, retain unadjusted p-values and, if adding an adjustment, define the family in advance (for example, all 19 sector temperature slopes in one outcome/control/window). Holm-adjusted p-values can be useful, but should not be labelled an across-all-analysis correction. An individual sector’s coefficient being significant is not a test that it differs from another sector; use the existing pairwise/joint contrast machinery for that claim. Pooled user-requested sector-CR1 inference and individual-sector time-HAC inference must remain clearly distinguished. A one-sector equation cannot acquire a sector-clustered SE. Neither significance convention cures the five residual time degrees of freedom in the excluded-year short-quarter regressions.

No false universal “all stationary” claim was found in the inspected existing report generators. The risk is expanding their old computational-success counts into an unsupported universal claim. A defensible consolidated heading is **Stationarity coverage and evidence**: state how many fitted series were inventoried, how many eligible pairs were computed, how many were compatible/mixed/inconclusive/concerning, and how many exact windows were untestable. The assembled conclusion must explicitly retain the mixed evidence and unavailable short-window cases.
