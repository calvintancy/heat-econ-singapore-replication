# Expanded stationarity audit and significance presentation

This audit maps each actual stochastic outcome Y and weather regressor X to its sector, frequency, transformation and exact estimation calendar. It does **not** certify that every series or regression is stationary. It leaves all regression coefficients, covariance methods and samples unchanged.

The registry covers all main pooled and national regressions, detailed-sector separate series, broad-sector interactions and their equivalent separate-series estimates, distributed temperature lags, alternative YoY weather definitions, raw levels, NSA growth, exact percentage growth, wind and PM2.5. Equivalent model files may share diagnostic links; the independent verifier checks those equivalences. Common national weather is tested as one time series per window, not repeated as independent sector observations.

## Tests and decisions

The primary pair uses ADF with AIC-selected lags capped at two seasonal cycles (24 months or eight quarters, reduced only for sample feasibility), and KPSS with its uncapped automatic Bartlett bandwidth. A prespecified sensitivity retains the earlier ADF one-year cap and KPSS one-year cap. Selection at an ADF upper cap is recorded. Both designs are shown; no favourable result is selected after the tests. The ADF and KPSS nulls differ, as documented by [statsmodels ADF](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.adfuller.html) and [KPSS](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.kpss.html).

At 5%, ADF rejection with KPSS non-rejection is labelled stationarity-compatible, or **trend-stationarity-compatible** when the tests include a constant and trend. The label requires concordance in both lag designs. ADF non-rejection with KPSS rejection is a unit-root concern. Both rejecting is mixed; neither rejecting is inconclusive. Disagreement between lag designs is lag-sensitive. Non-rejection never proves a null, and agreement never certifies the regression or exact integration order.

Growth, rainfall and annual weather differences use a constant. Raw/log output levels and non-rainfall weather levels or means/lags use a constant and trend. For weather and NSA growth, deterministic seasonal offsets are estimated alongside a constant and trend and then only those offsets are removed. SA output retains the official adjustment. Estimated deseasonalisation makes usual critical values approximate and is **not** a test for seasonal unit roots. There is no seasonal-root, structural-break, panel-unit-root, cointegration or residual-stationarity claim.

KPSS p-values outside its tabulated range retain inequalities. All calculations save statistics, actual lags, deterministic terms, warnings and tested observations. The independent verifier reconstructs statistics and source/calendars; passing numerical verification is a different claim from compatible stationarity evidence.

## Exact windows versus source histories

The audit separately labels exact estimation windows, available source histories and pre-2020 histories. Full-source tests do not clear a model estimated on a short or gapped subset. Gaps are never compressed; missing values are not imputed. At least five seasonal cycles are required by this diagnostic policy. If the exact calendar has gaps or insufficient continuous observations, standard ADF/KPSS are explicitly unavailable.

For added-control source-history screens, 90% temporal coverage is required separately for wind and PM. Exact-window screens honor the model's actual retained data, including the labelled 80% sensitivity, rather than silently imposing stricter screening. Full-control and COVID-excluded calendars remain gapped, so their stationarity is not established by this audit. The absence of a test is not evidence that the series is nonstationary.

## How results should be used

Growth remains the main research outcome. Levels are retained as clearly labelled descriptive sensitivities. Any growth specification with mixed, inconclusive, lag-sensitive or unavailable evidence for Y or an X remains exploratory rather than “confirmed stationary”. Year/season FE, HAC and sector clustering do not themselves cure nonstationarity.

This screen is not an automatic model-selection rule. We do not drop sectors, keep only favourable tests, repeatedly difference until a test passes or present the selected coefficient as if no pretesting occurred. Further persistence remedies require a defensible model of dynamics, breaks and the economic question. Differencing growth again would change Y to growth acceleration; differencing temperature changes the exposure estimand. Those changes are not silently substituted for the agreed models.

## Significance labels

Every displayed classification uses the saved **unrounded** p-value from the stated sector-CR1 or calendar-HAC covariance. The disjoint buckets are p<0.01 (significant at 1%), 0.01≤p<0.05 (significant at 5%), 0.05≤p<0.10 (10% only) and p≥0.10 (not significant at 10%). Forest dots are filled for p<0.05 and open otherwise, with text providing the same distinction. Conventional stars remain *** p<0.01, ** p<0.05, * p<0.10.

These are nominal pointwise tests unless explicitly Holm-adjusted. They are not corrected for searching many sectors, persistence pretests or selecting outcomes. Significant versus insignificant coefficients are not necessarily significantly different; use joint/contrast tests for sector differences. A statistically significant coefficient cannot override an unresolved stationarity flag, and an insignificant estimate is not proof of zero effect.
