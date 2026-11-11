# Independent review: crisis exclusions and origin-cluster inference

Reviewed 11 November 2026. The executable audit is `scripts/verify_visitor_crisis.py`; its result is `outputs/visitor_arrivals/crisis/verification.json`. The verifier imports no production visitor preparation, regression, masking or inference functions. It writes only its own verification receipt. The separate economic HTML is not an analytical input and may change concurrently.

## Scope and result

All checks passed: 120 additional crisis OLS specifications, 240 model/bandwidth results, 254 origin-CR1 comparisons (174 original specifications and 80 crisis specifications), and every reported weather coefficient, standard error, interval and p-value. Seven actual analytical input hashes match. The original 269-model regression-results file matches its preserved hash.

The audit reconstructs arrival outcomes from the original 225-month NSA source, raw weather alignment, the balanced region/country support, and every crisis calendar. The exclusion calculation enumerates every month in the complete growth interval and tests intersection against the excluded-month set. Design columns are reconstructed separately, regressions use SciPy QR, and Bartlett covariance is calculated by explicit calendar-lag pair matching. Existing 12/24-month HAC estimates are retained as primary inference.

No-exclusion crisis models reproduce the corresponding original models. GFC-plus-COVID and COVID-only full-control columns are exactly identical, as they should be: the 2017–2025 control window contains no 2008–2009 observations. The buffered growth boundaries are verified as MoM February 2011/February 2024 and YoY January 2012/January 2025.

The reporting extension is also independently checked. All 1,632 rows in `gt/profile_all_weather.csv` reproduce raw-source origin outcomes, QR coefficients and 12/24-month calendar-HAC uncertainty for the 16 joint profile/bandwidth systems. This covers rain, humidity, wind and PM2.5 coefficients as well as temperature, using the stated t(T−1) reference. Temperature rows agree with the previously verified profile results. Every nonprofile `gt/input.json` binding is checked against its core, crisis or origin-CR1 coefficient source; all weather and holiday regressors specified by each model are included, and the formatted coefficient/SE/p cells agree. Each model's month-specific linear-trend indicator is verified separately. Profile-row mean/SD growth is checked against its exact saved, independently verified origin outcome series. All 168 table inputs are covered, including 16 profile tables. These checks validate reporting data and bindings; native R rendering and browser behavior are verified separately by the parent task.

Origin-cluster coefficients and covariance were checked against five complete stacked FE designs, including a fixed-weight country fit. Common weather coefficients agree to approximately 10⁻¹³, and scaled CR1 covariance differences are below 1.4 × 10⁻¹³. Unit seasonal intercepts have different normalizations in the stacked and collapsed designs; equality is required for the common weather slopes, not those nuisance intercepts.

## How to interpret the windows

The whole-year crisis windows and one-year recovery buffers are transparent analytical sensitivities, not official exact recession or legal-border dates. [PMO](https://www.pmo.gov.sg/newsroom/speech-mr-goh-chok-tong-senior-minister-panel-discussion--financial-crisis-2007-09/) dates Singapore's recession onset to 2008Q3. The [2011 Budget](https://www.mof.gov.sg/docs/default-source/default-document-library/singapore-budget/budget-archives/2011/fy2011_budget_statement.pdf) describes the exceptional 2010 rebound and recovery during that year. [STB's 2023 review](https://www.stb.gov.sg/about-stb/media-publications/media-centre/singapore-s-tourism-sector-posts-strong-recovery-in-2023--exceeds-forecasts-for-tourism-receipts/) identifies 2023 as a strong recovery year and also anticipates continuing recovery in 2024. A 2023 buffer therefore does not certify that 2024 was unaffected.

Keeping the full-history columns prevents a selected exclusion from defining the conclusion. Filtering out large realised growth values would select on the dependent variable; the implementation does not do that. The filtered estimand is the association in the retained noncrisis/recovery-buffer months, not an automatically causal heat effect. The 2008 source start also prevents the Asian financial crisis and SARS from entering this common three-cut sample.

## Verified aggregate MoM results

Coefficients below are growth log points per +1°C, with 12-lag calendar HAC. P-values are nominal.

| Controls / exclusion | Months | Coefficient | SE | p |
| --- | ---: | ---: | ---: | ---: |
| Long controls, GFC + COVID | 163 | −0.9129 | 1.0576 | 0.389 |
| Long controls, buffered GFC + COVID | 139 | −0.7778 | 1.2037 | 0.519 |
| Long controls, stable pre-2020 after GFC buffer | 107 | +0.3278 | 0.9952 | 0.743 |
| Full controls, GFC + COVID | 66 | −4.0757 | 1.9324 | 0.039 |
| Full controls, buffered GFC + COVID | 55 | −4.8325 | 2.2047 | 0.033 |
| Full controls, stable pre-2020 | 32 | −0.5473 | 1.5975 | 0.734 |

The short buffered aggregate result remains negative and nominally significant. The long-history exclusions and stable pre-COVID checks do not establish a stable negative relationship. The 32-month full-control pre-COVID fit has only 13 residual time degrees of freedom and fragile inference. These comparisons must not be described as proving zero heat effects or as selecting the preferred model by p-value.

## Origin-cluster interpretation

The CR1 comparison uses arbitrary within-origin time dependence and assumes independent origin clusters. The correction is `G/(G−1) × (N−1)/(N−K_full)`, counting the full origin × calendar-month nuisance space. Region models have nine clusters and approximate t(8) inference; country models have 42 clusters and t(41). This comparison does not handle cross-origin common shocks as calendar HAC does.

Origin-cluster errors can be smaller than HAC errors; they are not automatically more conservative. Fully origin-specific profile slopes are excluded from origin-cluster inference because a coefficient supported by one origin cluster cannot receive meaningful conventional origin-cluster uncertainty.

Nominal p-values concern one coefficient under its null and inference assumptions; they are not probabilities that the null is true. The existing Holm corrections control familywise error for the defined nine or 42 market tests within one outcome, sample and bandwidth table. They do not adjust every model or exclusion choice across the specification grid.

The new crisis grid concerns OLS growth and descriptive level outcomes. It does not add crisis-specific PPML models or claim that the existing count-model sensitivities were re-estimated under every new exclusion policy.
