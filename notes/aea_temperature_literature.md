# Temperature, output growth and output levels: four AEA studies

Checked against primary papers on 10 November 2026; the Dell–Jones–Olken empirical growth transformation was additionally verified against its author-provided replication code on 11 November 2026. Equations below use simplified notation; they are not verbatim transcriptions. Published journal details are linked separately from the full-text version inspected. For the two AEA-hosted prepublication manuscripts, page references refer to that manuscript, not an assumed journal page number.

The direct answer is that **a year-on-year output-growth dependent variable does not automatically require year-on-year temperature on the right-hand side**. The transformation follows the economic model. A static output-level model implies a temperature difference after differencing output; a model in which temperature affects growth can use temperature levels and their lags. These are different restrictions, not interchangeable robustness labels.

## 1. Dell, Jones and Olken (2012): aggregate output growth

Melissa Dell, Benjamin F. Jones and Benjamin A. Olken, *Temperature Shocks and Economic Growth: Evidence from the Last Half Century*, **AEJ: Macroeconomics 4(3), 66–95**. [DOI](https://doi.org/10.1257/mac.4.3.66); [published PDF, pp. 72–74, equations (1)–(4)](https://economics.mit.edu/sites/default/files/publications/120626%20Climate%20Published.pdf#page=7).

Their empirical equation (4) is

\[
g_{it}=\theta_i+\theta_{rt}+\sum_{j=0}^{L}\rho_j W_{i,t-j}+\varepsilon_{it},
\]

where the outcome is annual real GDP-per-capita growth, and the weather vector contains annual average temperature **levels**, precipitation, and optional lags. Country effects accompany region-by-year and initial-poverty-status-by-year effects. Standard errors are two-way clustered by country and region-year. Table 5 also examines agricultural and industrial VA growth.

Their theoretical equation (3) makes the distinction explicit:

\[
g_{it}=g_i+(\beta+\gamma)T_{it}-\beta T_{i,t-1}.
\]

Here \(\beta\) represents an output-level channel and \(\gamma\) a growth channel. With a level channel alone, the temperature coefficients cancel in the sum. Footnote 10 discusses nonstationarity as a reason to prefer a growth specification when studying both channels.

**The empirical dependent variable is also log growth, confirmed from the replication code.** In `maketable2.do`, lines 57–62 take the natural log of the WDI GDP level, subtract its one-year lag and multiply by 100. Lines 66–71 construct agricultural and industrial VA growth the same way. `maketable5.do` repeats this construction for the sector regressions. Thus the article's annual-growth label corresponds to `100 × [ln A(year) − ln A(year−1)]`, rather than importing an ordinary WDI percentage-growth indicator unchanged. This matches the transformation used in our project, with a different economic outcome and monthly/quarterly frequency.

Source: [official AEA replication archive](https://assets.aeaweb.org/asset-server/articles-attachments/aej/mac/data/2010-0092_data.zip), also catalogued as [ICPSR E114251V1](https://doi.org/10.3886/E114251V1). The ZIP, inspected do-files and a source/hash manifest are cached under `data/raw/literature/djo_2012/`. No replication code was executed, and our estimates were not changed.

## 2. Deschênes and Greenstone (2007): profit levels

Olivier Deschênes and Michael Greenstone, *The Economic Impacts of Climate Change: Evidence from Agricultural Output and Random Fluctuations in Weather*, **AER 97(1), 354–385**. [DOI](https://doi.org/10.1257/aer.97.1.354); [author-hosted published PDF, p. 367, equation (4)](https://olivierdeschenes.weebly.com/uploads/1/3/5/0/135068654/the-economic-impact-of-climate-change.pdf#page=14).

Their panel specification is

\[
\pi_{ct}/A_{ct}=\alpha_c+\delta_{st}+X_{ct}'\lambda+f(GDD_{ct},P_{ct})+u_{ct}.
\]

The dependent variable is agricultural profits **in dollars per acre**, not its growth rate or log. Annual growing-season degree-days and precipitation enter in levels and quadratics, with effects allowed to differ by irrigation status. Preferred fixed effects are county and state-by-year. The growing season is April–September; degree-days use daily mean temperature, a 46.4°F base and 89.6°F ceiling. The paper reports heteroskedasticity-robust inference and Conley spatial standard errors for preferred models (pp. 366, 375). Profit regressions use farmland-acreage weights.

This is a precedent for a weather–economic-outcome **level** specification, not proof that trending Singapore VA levels are stationary. The original numerical findings were subsequently corrected: read the [2012 AER Reply](https://doi.org/10.1257/aer.102.7.3761) and [Comment](https://doi.org/10.1257/aer.102.7.3749) before using its climate-damage estimates.

## 3. LoPalo (2023): daily productivity levels

Melissa LoPalo, *Temperature, Worker Productivity, and Adaptation: Evidence from Survey Data Production*, **AEJ: Applied Economics 15(1), 192–229**. [DOI](https://doi.org/10.1257/app.20200547); [AEA-hosted December 2021 manuscript, printed pp. 15–17, equation (1)](https://www.aeaweb.org/content/file?id=16102#page=15).

The estimating equation is

\[
y_{icprd}=\sum_b\beta_b 1\{WB_{cpd}\in b\}
+\alpha E[WB]_{cpd}+Z_{icprd}'\lambda+\theta_p+\mu_i+\varepsilon_{icprd}.
\]

The main outcome is interviews completed per hour **in levels** at interviewer-day frequency. The exposure is the day's average wet-bulb temperature in bins, relative to 50–60°F; it is not temperature growth. Controls include usual local calendar-date wet-bulb temperature, precipitation, daylight, respondent characteristics and days elapsed in the survey round. Fixed effects are interviewer and country-region, with regions defined within survey round. Standard errors are two-way clustered by region and interview date. This directly illustrates that contemporaneous productivity levels and nonlinear heat exposure can be paired, while controlling for expected climate separately. It does not estimate macroeconomic growth or a long-run VA elasticity.

## 4. Park, Goodman, Hurwitz and Smith (2020): human-capital production

R. Jisung Park, Joshua Goodman, Michael Hurwitz and Jonathan Smith, *Heat and Learning*, **AEJ: Economic Policy 12(2), 306–339**. [DOI](https://doi.org/10.1257/pol.20180612); [AEA-hosted July 2019 manuscript, printed pp. 10–11, equations (1)–(2)](https://assets.aeaweb.org/asset-server/files/9992.pdf#page=10).

The baseline equation is

\[
Score_{iscyn}=\beta Heat_{sy}+\eta_i+\gamma_{cyn}+\varepsilon_{iscyn}.
\]

The outcome is a standardized test-score **level**; within-student comparisons identify the relationship across repeated tests. Exposure is average daily maximum temperature on school days in the preceding year, or counts of days in temperature bins; neither is a year-on-year temperature difference. Fixed effects are student and cohort-by-test-year-by-take-number. Standard errors are clustered by weather station, with broader spatial clustering checked. Additional specifications add precipitation, test-day weather and pollution. This concerns learning and cumulative exposure, not contemporaneous GDP; it shows why the relevant exposure window follows the production mechanism rather than the outcome's label.

## Implications for the Singapore specifications — our methodological assessment

### A. A static level effect gives a precise year-on-year restriction

Let \(h=4\) for quarterly VA/IIP and \(h=12\) for monthly IIP. Suppose the maintained level model is

\[
\log Y_{st}=m_{st}+\beta T_t+\lambda'X_t+u_{st}.
\]

Then algebra, not an empirical convention, implies

\[
100(\log Y_{st}-\log Y_{s,t-h})
=100\Delta_h m_{st}+100\beta(T_t-T_{t-h})
+100\lambda'(X_t-X_{t-h})+100\Delta_h u_{st}.
\]

Consequently, **YoY temperature in °C is the coherent regressor when estimating the differenced counterpart of this static log-level model**. The control variables and deterministic terms must also be transformed consistently to claim exact equivalence. Reusing the same fixed effects in both regressions is not, by itself, exact differencing of the level model.

A useful less restrictive comparison is YoY log growth on both \(T_t\) and \(T_{t-h}\), followed by a test that their coefficients sum to zero. That tests the particular static-level restriction; it is not a universal test of all possible level versus growth mechanisms. Persistent effects can require intermediate lags.

For nonlinear exposure, difference the basis functions: \(f(T_t)-f(T_{t-h})\), or bin shares in each period. In general, \(f(T_t-T_{t-h})\) is a different model. Celsius percentage growth is inappropriate because the Celsius zero is arbitrary; use temperature differences in °C.

### B. Temperature levels in a growth regression remain a valid specification

The first paper supplies a direct published example. Such a regression asks whether growth is different in hotter periods, conditional on the included terms; it does not impose the static-level restriction. Our current-temperature-only YoY regression is therefore not mechanically erroneous, but omitting base-period temperature can confound current and denominator-period exposure when weather is persistent. Retain it as a distinct model and compare it with the paired-temperature or distributed-lag specification.

Annual GDP growth in a cross-country study is not the same object as overlapping quarterly/monthly YoY growth. YoY log growth sums \(h\) consecutive one-period log changes. If the mechanism is an effect on each period's growth, the relevant weather history may span those \(h\) periods and further lags, not merely the current quarter/month. Serial-correlation inference must accommodate overlapping growth windows and any remaining persistence.

### C. Output levels can be estimated, with a different interpretation

A log-VA or log-IIP level regression with a modeled trend can estimate a temporary proportional deviation in output associated with heat. With log output on the left, \(100\beta\) is approximately the percent output difference per 1°C. Raw dollar VA levels instead produce sector-size-dependent dollar effects and are less convenient for a pooled sector coefficient.

Before interpreting a macro level regression, address its trend and residual persistence: seasonality, sector-specific trends, year effects and meaningful stationarity diagnostics/sensitivity. Fixed effects alone do not guarantee stationarity, and the cited micro-level studies do not validate a trending national level regression. A significant levels coefficient is not automatically evidence of a permanent growth effect or of long-run climate damage.

In this one-country panel, national temperature is identical across sectors in each date. Full date fixed effects absorb its common coefficient. Sector-by-season effects plus year effects leave identifying variation, but cannot remove every omitted national shock. Repeated sector rows also do not create additional independent weather observations. Preserve calendar-aware inference and compare transformations on common samples.

Finally, an anomaly relative to a fixed month/quarter climatology is **not** a YoY temperature difference. With seasonal fixed effects already present, replacing a linear temperature level with that fixed-climatology anomaly only reparameterizes the regression; it does not solve the base-period or dynamic-specification question.

## Scope and source boundaries

All four main studies are published in AEA journals. Deryugina and Hsiang's *Does the Environment Still Matter? Daily Temperature and Income in the United States* is [NBER Working Paper 20750](https://www.nber.org/papers/w20750), not an AEA journal article verified in this search. Hsiang (2010, PNAS), Burke–Hsiang–Miguel (2015, Nature), and Colacito–Hoffmann–Phan (2019, JMCB) can inform a broader review, but should not be labelled AEA publications.

No model code or regression results were changed for this literature note.
