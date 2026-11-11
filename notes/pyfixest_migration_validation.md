# Native Pyfixest migration validation

The production FE regressions use native `pyfixest.feols`. Validation compares the regenerated estimates with the 177 result files saved under `data/reference_before_pyfixest/outputs`, including six heterogeneity nuisance-coefficient files added to the original 171-file snapshot before the rerun. Independent NumPy/statistical verification remains separate from production estimation.

## Estimation and uncertainty

The logical design contains explicit fixed-effect indicators, weather controls and the requested temperature terms. For numerical conditioning, a design-only SVD of the scaled full-rank design `X/D = U S V'` supplies orthonormal coordinates `U` to native `feols`; reported coefficients are recovered as `D⁻¹ V S⁻¹ theta_native`. This invertible reparameterization preserves the model and sample. No outcome enters the construction of the transform, and no manual `U'Y` regression replaces native estimation. Native coefficients and residuals are retained through the adapter. Source/native formulas, logical and native design hashes, coefficient-transform hashes, installed version and fit provenance are logged. The explicit-indicator implementation is an LSDV FE regression rather than an absorbed-FE algorithm.

The estimator change preserves the previously checked uncertainty calculations: sector CR1 for eligible common-slope panels, and calendar-aware Driscoll–Kraay/Bartlett covariance for sector-specific and time-series estimates. Missing calendar periods remain gaps. The corresponding degrees of freedom, finite-sample multipliers, contrasts and Holm families remain unchanged. Native package default standard errors do not replace those specified calculations.

The original migration contained **1,194 native FE models**. Adding the NSA YoY companions expanded the main source-analysis receipt to **1,510 native models**, while retaining the original SA estimates. Annual and paired-COVID analyses have their own isolated receipts. The current `outputs/pyfixest/fit_log.csv` and its sealed manifest are authoritative for the main-stage count; the verifier independently reconstructs the expected count from current model/outcome/unit inventories. Counts treat individual outcomes in multi-outcome calls separately and include the original partial-regression fits. Auxiliary unit-root regressions and independent verification calculations are excluded.

## Main temperature and sector interactions

The estimable absolute-slope models contain a main temperature coefficient and nonreference sector interactions. With the first sorted sector as reference:

```text
beta_reference = beta_temperature
beta_sector = beta_temperature + delta_sector
Var(beta_sector) = Var(beta_temperature) + Var(delta_sector)
                   + 2 Cov(beta_temperature, delta_sector)
```

The native component covariance is transformed jointly to the full covariance of reported absolute sector slopes. `interaction_components.csv` and `component_covariances.csv` retain the underlying estimates and cross terms. The reference interaction is recorded as a zero normalization with `estimated=False`; its standard error, interval and p-value are unavailable. It is not an estimated coefficient with zero uncertainty.

The R1 model has full date effects, which absorb common national temperature. It therefore retains nonreference relative interactions without an identified main temperature coefficient or absolute reference-sector slope. This identification limit is preserved during migration.

## Which significance null is tested?

For a nonreference sector, its reported total-effect p-value tests the **one linear restriction** `beta_temperature + delta_sector = 0`. The test uses the variance of that sum, including its covariance cross term. Its squared t statistic equals the one-restriction Wald/F statistic, and the two-sided `t(T−1)` p-value equals the `F(1,T−1)` tail probability. For the reference sector, the same test reduces to `beta_temperature = 0` because its interaction is fixed to zero. This does not jointly set both the main coefficient and an interaction to zero, which would be a different two-restriction hypothesis.

The heterogeneity omnibus tests **all nonreference deltas equal zero**: sector slopes are equal while their common level remains unrestricted. The verifier computes that statistic in both the main/deviation basis and the absolute-sector contrast basis, checks their full covariance transformation, restriction count, finite-sample degrees of freedom and residual-score rank bounds. Singular omnibus covariance produces no p-value; no inverse or pseudoinverse is used for such a test. R1 tests its relative deltas, without claiming an identified common main effect. These checks do not introduce a different omnibus hypothesis that all temperature slopes are zero.

`sector_total_effect_null_checks.csv` records every sector linear-effect null and its t/F equivalence, including the 23 original-sector slope rows. `omnibus_null_checks.csv` records each equal-slopes omnibus null, basis-invariance calculation and rank/availability decision.

## Independent checks

`scripts/verify_pyfixest_migration.py` checks all 177 pre-existing result files, covering every original SA row's coefficients, standard errors, intervals, p-values, Holm-adjusted p-values, R², covariance matrices, sample dates, units and observation/parameter counts. Additional result rows must be explicitly identified as NSA and receive their own raw-source and regression verification. Original counts and unavailable-test masks must match exactly. Changed formula text is accepted only after checking that the old and new full-rank design matrices span the same column space.

The numerical tolerance is:

```text
abs(new − reference) <= 1e−7 + 1e−7 × abs(reference)
```

Raw-level coefficients and covariance entries have larger physical units than growth coefficients; the same relative rule applies. Actual absolute errors, errors scaled by `max(1, abs(reference))`, and fractions of the allowed tolerance are retained. Covariance condition numbers near singularity are excluded from equality because their SVD values are unstable under tiny floating-point changes. Every covariance entry, restriction rank, test availability and reported inferential statistic remains checked.

A separate manifest captured 262 raw, processed and exact-estimation sample files before the migration. Their original hashes and sizes are checked after the rerun. For NSA expansion, an immutable pre-NSA archive also permits a narrowly checked extension: explicitly named NSA-derived columns or new source-manifest entries. Every old row, old column and old manifest entry must remain unchanged, and each permitted extension is enumerated in the receipt. This protects the retained PM2.5/wind baselines, coverage rules and COVID exclusions independently of coefficient agreement. The primary controlled samples must still contain 100 months or 29 quarters, falling to 66 months or 19 quarters after excluding 2020–2022 endpoints.

The verifier checks the native fit log and scans production estimation modules for remaining statsmodels OLS or NumPy least-squares estimation calls. NumPy covariance geometry, linear transformations, Wald solves and standalone independent verification remain permitted. Existing independent regression verifiers continue to check the production results through their separate OLS/covariance implementations.

## Validation artifacts

Run `.venv/bin/python scripts/verify_pyfixest_migration.py` after the complete estimator rerun and `write_fit_log()`. It writes `outputs/pyfixest/migration_verification.json` and `migration_comparison.csv`. The input baseline is `outputs/pyfixest/migration_input_reference.json`, and native fit provenance is `outputs/pyfixest/fit_log.csv`.

The generated verification JSON records the pass/fail result and numerical maxima. Agreement validates this estimator migration; it does not establish stationarity, reliable small-sample inference or a causal heat effect.

<!-- validation-run:begin -->
## Completed verification

The completed audit passed for all 177 reference result files and 307,764 numerical values. All 1,510 native models were recorded by Pyfixest 0.70.0 across 1,246 actual native API calls. Native response, coefficient mapping and residual checks agree exactly, with no dropped rows or coefficients.

The maximum numerical error used 0.125931 of the stated tolerance. All original source/processed/exact-sample rows and columns remain unchanged; any added NSA columns and source-manifest entries are separately enumerated. All 330 model/outcome/bandwidth component groups passed the coefficient-sum and full covariance transformation checks, including 282 explicitly marked reference zero normalizations without standard errors or p-values. The three original interaction fits also passed independent full-design QR OLS and calendar-HAC checks for all 23 reported absolute sector slopes.

The significance audit checked 3,145 absolute-sector total-effect tests and 320 identified relative R1 tests. The largest reported-p discrepancy was 3.59e-11; two-sided t and one-restriction F probabilities differed by at most 3.19e-16. All 302 available equal-slopes omnibus tests agreed across the two coefficient bases; the other 28 had singular restriction covariance and correctly reported no p-value.
<!-- validation-run:end -->
