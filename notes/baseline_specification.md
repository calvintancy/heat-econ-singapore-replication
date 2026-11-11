# Baseline specification and PM2.5

The baseline is the **all-retained-date full-control model** during 2017–2025. It includes period mean air temperature, rainfall, relative humidity, wind speed and PM2.5. The 90% monthly coverage rules retain 100 months or 29 quarters. Those are observed, screened dates, not an imputed balanced calendar.

For sector heterogeneity the main equation is

`growth_it = β_i temperature_t + γ_r rain_t + γ_h humidity_t + γ_w wind_t + γ_p PM2.5_t + sector×season FE + year FE + error_it`.

Temperature × sector interactions give each sector an absolute temperature slope. Other weather slopes and year effects are shared. Separate-series fits are robustness checks. Aggregate VA and total IIP use the corresponding national time-series equation with seasonal and year FE. PM2.5 is scaled in10µg/m³ units in saved regressions; this scaling does not remove it from the model.

The baseline Y is one-period log growth; YoY growth is an alternative outcome, while log and raw levels remain descriptive sensitivities. All four outcomes are available with baseline controls. COVID endpoint exclusion is a robustness check on the same control set, retaining66months/19quarters. The longer history without wind and PM2.5 is **long-history robustness**, not the current baseline. Historical model identifiers and cached metadata may retain “original baseline” labels for traceability; they do not define the current baseline.

## Why stationarity is not confirmed for every specification

Two different states must remain separate:

1. **Mixed or lag-sensitive evidence:** some growth series and weather controls do not obtain concordant ADF/KPSS evidence. Humidity is mixed or lag-sensitive across contexts; monthly PM2.5 also has mixed evidence in its available history. Output levels often raise persistence concerns.
2. **Unavailable exact-window tests:** quality screening leaves calendar gaps in the full-control sample. Standard ADF/KPSS code is not run on an artificially compressed calendar. “Not tested” does not mean the data rejected stationarity. Longer continuous-history evidence is shown separately and does not certify the exact short window.

ADF tests a unit-root null; KPSS tests a level/trend-stationarity null. Non-rejection is not proof of either null. [ADF documentation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.adfuller.html), [KPSS documentation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.kpss.html).

Fixed effects and robust standard errors do not by themselves establish stationary disturbances. Conversely, an unresolved univariate screen is not sufficient to declare the regression invalid: deterministic effects, dynamic specification and possible common stochastic components matter. Our conservative audit is an evidence screen, not a theorem that every marginal series must pass both tests for an OLS coefficient to exist or for all economic relationships to be valid. No cointegration or residual-stationarity clearance is claimed.

The reporting correction changes defaults, ordering and interpretation, not estimates. The full-control models, interaction slopes, covariance matrices, stationarity tests and significance values were already computed and verified. Baseline tables now open on those existing full-control estimates, and unavailable tests are labelled explicitly rather than presented as zero compatible units.
