# Main detailed-sector FE interactions and time-series robustness

The main detailed-sector heterogeneity specification is

`Y_it = Σ_s β_s 1(i=s) T_t + γ'X_t + α_i,season(t) + λ_year(t) + ε_it`.

PyFixest now fits a common temperature term βT and temperature × sector deviations δ_s for all nonreference sectors. The tables report **absolute total sector effects βT + δ_s**, rather than the interaction deviation alone; the reference effect is βT and its omitted δ is zero by normalization. This is equivalent to the previous all-sector absolute-slope parameterization. The model includes sector-specific seasonal fixed effects, common year fixed effects and shared rainfall/humidity controls; the short samples also include shared wind and PM2.5 coefficients. Common year effects preserve within-year national temperature variation. Saturated date fixed effects would absorb the common temperature component.

All FE coefficients and residuals—including fixed-effect nuisance parameters—come from native PyFixest `feols`. The saved explicit full-rank FE basis and independently verified custom covariance are documented in [pyfixest_backend_methods.md](pyfixest_backend_methods.md). Native temperature/deviation coefficients and their joint covariance are retained in each folder.

The model covers 19 detailed VA sectors and 18 manufacturing industries at monthly and quarterly frequencies. G1 one-period growth, G2 rolling YoY growth, L1 log levels and L3 raw levels use the same dates within each context/window. Long, full-control and endpoint-excluded samples match the existing detailed-series analysis exactly. Growth remains the main outcome; levels are descriptive sensitivities.

## Main FE versus robustness

Separate time-series fits are robustness checks. They also include season and year effects, but estimate every weather slope and year effect separately for each sector. Their point estimates equal a fully interacted panel with sector-specific weather and sector × year effects (the broad-sector H3 specification). They do not equal the main FE interaction slopes in general, because the main specification shares controls and year effects. In this balanced common-weather panel, shared controls and common year effects cancel when comparing two sectors: their main slope difference equals the temperature coefficient from the difference in their outcomes regressed on temperature and seasonal indicators. The main contrast therefore does not remove differential sector-year shocks; allowing sector-specific year effects in the separate-series/H3 robustness is a substantive sensitivity. Neither design by itself establishes causality.

The equal-sector common-slope FE panel is retained as a benchmark. In this balanced design, the arithmetic mean of the main β_s equals that common-slope coefficient, up to numerical precision. This is a regression identity, not a claim that equal-sector growth equals aggregate economic growth. The published national VA and weighted IIP totals remain separate estimands.

## Inference and sector comparisons

For a nonreference sector, the reported total variance is Var(βT) + Var(δ_s) + 2 Cov(βT, δ_s). The table’s p-value tests that total effect, while a raw interaction p-value instead tests the difference from the reference.

Each sector-specific coefficient is supported in only one sector cluster. Conventional unit-clustered uncertainty for those slopes is degenerate or lacks the many-supporting-clusters justification; it is not reported as precise sector-CR1 inference. The main common-slope panel continues to cluster at sector i. Main interaction slopes instead use **joint calendar-aware Driscoll–Kraay covariance**, Bartlett four/twelve quarterly/monthly lags and doubled-bandwidth sensitivity, with approximate t(T−1) intervals. Scores sum across sectors by actual date; excluded calendar gaps remain gaps. The finite-sample factor is T/(T−1) × (N−1)/(N−K), including the full FE parameter count.

The omnibus null is equality of all β_s. In a balanced common-weather panel, differencing two sectors cancels shared year effects and controls. The contrast time-series design contains temperature and seasonal intercepts, so the contrast influence rank cannot exceed T−Q−1 (Q=4 or 12). Singular restriction covariance has no omnibus p-value; no pseudoinverse is used to manufacture one. The short excluded quarterly samples have 19 dates and 18/17 contrasts, exceeding that bound.

All S(S−1)/2 sector pairs are retained (171 for VA, 153 for IIP). Holm correction is applied within the predeclared context/window/outcome/bandwidth family. These adjusted tests address differences between sector slopes. Pointwise slope significance answers whether one β_s differs from zero; “significant in one sector but not another” is not itself evidence of heterogeneity.

## Stationarity and interpretation

Main FE interactions and their time-series robustness use the same marginal Y/X data and calendars, so they share stationarity evidence links. Compatibility is evidence under declared tests, not proof of stationary disturbances or a causal relationship. Mixed/lag-sensitive growth, persistent levels and untestable gapped short windows remain clearly flagged. No sector or specification is selected because it yields favourable pretests or significance.

A growth β_s of −1 means roughly one percentage point lower period growth for a +1°C current-period mean temperature, conditional on the shared controls and FE. It does not establish permanent growth damage. Log-level/raw-level coefficients have different units and remain descriptive. All labels follow the same saved unrounded p-values and covariance as their displayed intervals.


## Exact nulls checked in the final audit

The sector-specific total-effect test imposes one restriction: `betaT + delta_i = 0` (reference: `betaT = 0`). Its squared t statistic equals the one-restriction Wald F statistic with the same denominator degrees of freedom. It is not the two-restriction test that the temperature and interaction coefficients are both individually zero. The reported total-effect p-value uses the full component covariance.

The separate omnibus heterogeneity test imposes `delta_2 = ... = delta_S = 0`, equivalently equality of all absolute sector slopes. It uses the full joint covariance and the declared rank/df rules. This is different from a joint hypothesis that every sector’s absolute temperature effect is zero. The independent migration audit checks these distinctions and the linear transformation between absolute and reference-coded covariance matrices.
