# Paired regressions with and without 2020–2022

Every original regression analysis receives two samples: its full eligible sample and that same sample after removing observations dated in calendar years 2020, 2021 and 2022. These are additional saved results; the original estimates and source data are retained. Existing exclusion specifications (`S5`, `C5`, `H3X`, other `X` suffixes and detailed `exclude` folders) are linked to the corresponding excluded side of their original parent. Pre-2020 specifications are already outside the excluded years, so their two samples are identical and their native fit can be reused.

## Exact sample and outcome construction

The full side starts from the original specification's exact eligibility, including its outcome definition, seasonally adjusted or unadjusted output, weather construction, control coverage and any lag-related missing-value rule. The excluded side is a row filter applied **after** output growth, annual comparisons, weather differences and distributed lags were constructed. All remaining outcome, exposure, control and calendar-index values must match the full side. This prevents a three-year gap from being treated as a one-month or one-quarter growth observation.

The year effects and other fixed effects retain the same specification. Categories absent after the exclusion have no observations, so the number of estimated indicators can fall. This is recorded through each sample's actual parameter count and residual degrees of freedom. The original calendar indices are retained; HAC lag products span actual calendar months or quarters, with empty periods contributing zero scores. The exclusion does not join 2019 directly to 2023 in calendar time.

The primary wind/PM2.5-controlled windows retain the original 90% coverage screen. Their 100 eligible months or 29 eligible quarters become 66 months or 19 quarters after removing the specified endpoint years. Long-history weather models and the 80% coverage sensitivity retain their own original eligibility rules. Controls are never dropped merely to increase the excluded sample.

An observation dated in 2023 can still have a **2022 base** in a year-on-year outcome or annual weather difference. This is an endpoint-year exclusion; it does not also exclude observations whose lagged comparison year falls in 2020–2022.

## Estimation and uncertainty

Both samples use actual native `pyfixest.feols` estimates through the same full-rank fixed-effect adapter as the original analysis. The native fit receipts are isolated under `common/pyfixest` and `sector/pyfixest`; they do not replace the main source-analysis receipt. That receipt had 1,194 fits at the initial migration and has 1,510 after adding NSA YoY companions; its current sealed manifest is authoritative. The adapter's invertible, design-only orthonormal reparameterization improves numerical conditioning and does not change the statistical model. While fitting, each adapter result retains the actual native Feols object, native coefficients and residuals. Saved receipts record the version, logical/native design and coefficient-transform hashes, and native coefficient/response/residual identity checks.

For pooled **common-slope panels**, the main uncertainty estimate remains sector CR1, with multiplier `S/(S−1) × (N−1)/(N−K)` and a `t(S−1)` reference distribution. Original time-cluster sensitivities and calendar Driscoll–Kraay/Bartlett results remain available. National aggregate series use calendar HAC because they contain one output series.

**Sector-specific temperature slopes and own-series regressions use calendar covariance.** A slope supported by one sector cannot receive useful sector-cluster inference. Joint sector-FE covariance sums coefficient scores across sectors within each actual date, then applies Bartlett HAC with multiplier `T/(T−1) × (N−1)/(N−K)`. Own-series regressions use their own `T/(T−K)` correction. These two corrections can differ even when a flexible pooled specification and separate regressions share a point estimate; their inference is checked separately. The main bandwidth is four quarters or twelve months, with twice that bandwidth as a sensitivity. Calendar-based tests use `t(T−1)`.

## Sector effects, joint tests and controls

In an identified absolute-slope interaction model, the reported sector effect is the main temperature coefficient plus that sector's nonreference interaction. Its variance includes the cross term:

```text
beta_sector = beta_temperature + delta_sector
Var(beta_sector) = Var(beta_temperature) + Var(delta_sector)
                   + 2 Cov(beta_temperature, delta_sector)
```

The sector p-value tests this **one linear restriction**, `beta_sector = 0`. Its squared t statistic equals the one-restriction Wald/F statistic; the corresponding two-sided t and F tail probabilities agree. This differs from jointly requiring both component coefficients to be zero. The reference interaction is a zero normalization with no estimated standard error or p-value. R1 retains only relative sector effects because full date effects absorb common national temperature.

The heterogeneity omnibus tests equal sector slopes: every nonreference interaction equals zero while the common temperature effect remains unrestricted. Full-rank restriction covariance is inverted directly. A singular restriction covariance retains its rank and an unavailable joint p-value; it is not pseudoinverted to manufacture significance. Pairwise differences use the full joint covariance. Holm adjustment is performed separately for each specification, sample variant and bandwidth, using the original displayed broad-sector families or every detailed-sector pair.

The saved paired own-series weather-coefficient table includes temperature, rainfall, relative humidity and, where specified, wind and PM2.5. These are checked as part of the source-row accounting, alongside the displayed temperature results. Distributed-lag sums and the unrestricted current-plus-annual-lag temperature restriction use their full coefficient covariance.

Each coefficient table retains the outcome's original units. Exact-sample signed means and sample standard deviations use the actual regression rows; sector results use their own sector's outcome sample. A temperature effect for +0.1°C is one tenth of the per-°C coefficient. A signed growth mean is context for magnitude and is not used as a denominator for a coefficient-to-mean ratio.

## Independent verification and limits

`scripts/verify_covid_comparison.py` constructs its own original-source inventory, resolves exact samples independently, estimates full-design QR benchmarks, and computes CR1 or actual-calendar HAC from explicit score sums. It does not import either paired producer, the native adapter, or production covariance/contrast helpers. It checks coefficients, full sector/component covariance matrices, standard errors, confidence intervals, p-values, R², means, standard deviations, sample dates/counts, fixed-effect parameter counts, omnibus rank/availability and pairwise Holm families. Original source rows, including their existing exclusion aliases, must match the corresponding paired result. Algebraic partial-regression plots remain diagnostics of their covered parent regression; they are not counted as additional economic specifications. The numerical comparison rule is `abs(new − reference) <= 1e−7 + 1e−7 × abs(reference)`; exact sample identities, counts, aliases and unavailable-test masks receive separate checks.

The explicit pre-producer checkpoint is `capture_original_inputs_reference(refresh=True)`. The verifier never refreshes that checkpoint. Its hashes protect the original raw/processed/exact samples, analytical CSV/JSON results and original native receipt. `outputs/covid_comparison/verification.json` and companion check/coverage tables state the completed audit's scope and counts.

These checks establish numerical and sample-accounting consistency. They do not establish a causal temperature effect or remove short-sample inference limits. Sector CR1 can have few clusters, while common national weather does not become an independent exposure for every sector. The excluded windows can have very few effective time-series degrees of freedom.

Existing stationarity diagnostics describe the corresponding original samples and variables. The excluded samples often contain calendar gaps. Standard ADF/KPSS routines cannot be applied to a series after silently compressing those gaps, so this exercise does **not** grant new stationarity clearance to the excluded models. Original marginal tests are not residual-stationarity, cointegration, structural-break or causal-validity certificates.

<!-- completed-audit:begin -->
## Completed independent audit

All 424 model pairs passed, covering 13,716 temperature/common-coefficient or contrast rows and 11,744 paired own-series weather-coefficient rows. The two isolated native receipts contain 2,204 actual models and match the independent model/outcome/unit inventory. All 562 original input/result/receipt hashes remained unchanged.

The coverage inventory accounts exactly once for 9,146 original coefficient/covariance rows across 89 source CSVs, every row of 75 original component/covariance/joint/pairwise companion CSVs, and 21 structured linear-contrast results. The common branch has 4 unchanged pre-2020 pairs. Of 516 sector heterogeneity tests, 480 are full rank and 36 correctly omit p-values because restriction covariance is singular. All 20,400 pairwise comparisons and Holm families passed.
<!-- completed-audit:end -->
