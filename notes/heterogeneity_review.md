# Sector heterogeneity: independent specification review

Reviewed 10 November 2026. This note records the estimands, rank restrictions and covariance requirements before interpreting heterogeneous sector slopes. Existing pooled estimates and outcome comparisons remain unchanged.

## Absolute and relative slopes answer different questions

Write `Y` for the `T × S` matrix of sector outcomes, `W` for the `T × p` matrix containing temperature and all weather controls, `B` for season indicators and `A` for year indicators. The panel is balanced: all sectors have the same retained dates and weather. Each outcome/control comparison should use the already defined common sample. Absolute slopes remain conditional weather associations rather than identified causal responses.

The proposed three absolute-slope models are:

| Model | Temperature | Other weather | Seasonal effects | Annual effects |
| --- | --- | --- | --- | --- |
| H1 | Sector-specific | Shared | Sector-specific | Shared |
| H2 | Sector-specific | Sector-specific | Sector-specific | Shared |
| H3 | Sector-specific | Sector-specific | Sector-specific | Sector-specific |

H1 can confound heterogeneous temperature responses with heterogeneous responses to omitted sector-specific rainfall, humidity or other weather coefficients. H2 relaxes that restriction. H3 additionally allows every sector its own annual shocks; it has the same coefficient estimates as separate sector regressions on identical regressors and dates. Its uncertainty must still account jointly for contemporaneous and serial dependence across sectors.

The independent least-squares decomposition is:

- H1: regress the cross-sector mean outcome on all weather, season and year effects; regress each sector's deviation from that mean on temperature and season effects. Add the mean temperature coefficient to each deviation coefficient.
- H2: use the same mean regression; regress each deviation on all weather and season effects. Add mean and deviation weather coefficients.
- H3: regress each sector directly on all weather, season and year effects.

These decompositions exploit a balanced equal-weight panel. They must not be silently reused for an unbalanced panel or sector-specific regressor availability without rederiving the projection.

## Full-date fixed effects identify relative slopes only

R1 should use full date effects, sector-specific season/year effects and weather interactions for every sector except a declared reference sector. Every reference-sector weather interaction must be omitted. Full date effects absorb all common weather effects; including all sectors' interactions would create an exact linear dependence. The reference coefficient is a normalization, not evidence that the reference sector is unaffected by heat.

R1's reported temperature slope for sector `s` is `β_s − β_reference`. It is not an absolute heat effect and cannot be aggregated into national damage. Differences between any two non-reference sectors remain estimable by subtracting their reported coefficients. Reference-sector rows should be marked as references rather than displaying a zero estimate with a zero standard error.

For the balanced panel, R1 with sector-specific year effects yields exactly the same point estimates as H3 sector-minus-reference contrasts. Omitting sector-specific year effects makes R1 point estimates equal H2 contrasts. The uncorrected calendar-HAC contrast covariances also agree. The full-model finite-sample correction may differ because full date effects change the parameter count. Thus R1 makes the relative identification explicit; it is not an independent robustness discovery. Equality tests and pairwise differences should be invariant to the choice of reference, apart from normalization and numerical precision.

Full date effects remove arbitrary common date shocks. They do not remove sector-specific shocks or establish that differential sector outcomes are unrelated to temperature. H3/R1 are useful adjustments for sector-specific annual shocks but do not solve omitted time-varying confounding at the monthly/quarterly frequency.

## Exact model ranks

Let `q` be the number of observed seasons and `y` the number of observed calendar years. If the observed year-season incidence graph is connected, the single-sector additive season/year design has rank `q+y−1`. Verify this with the actual matrix, particularly after exclusions.

With `S` sectors, `T` retained dates and `p` weather variables including temperature, the full panel parameter counts are:

| Model | Full-model parameter count K |
| --- | --- |
| H1 | `S q + (y−1) + S + (p−1)` |
| H2 | `S(q+p) + (y−1)` |
| H3 | `S(q+y−1+p)` |
| R1 with sector year effects | `T + (S−1)(q+y−1+p)` |
| R1 without sector year effects | `T + (S−1)(q+p)` |

In particular, `K_R1 − K_H3 = T − (q+y−1+p)`, the residual degrees of freedom of one H3 time regression. Full date effects and ordinary sector-season dummy formulas have redundant common seasonal components. An explicit design can use all date indicators plus non-reference sector indicators interacted with season/year contrasts and weather; alternatively, use a verified FWL projection. A full-rank reduced design is required before fitting or inverting a Wald covariance.

## Joint covariance is essential

The mean and deviation regressions in H1/H2 have correlated estimation errors. Combine their per-date influence vectors first, then form one joint covariance matrix for all sector temperature slopes. Do not add separately corrected covariance matrices or treat the auxiliary regressions as independent.

For H3, use the common residualized temperature regressor and each sector's regression residual to construct a joint per-date influence matrix. For R1, construct the corresponding sector-minus-reference residual influences. Form Bartlett HAC over these vectors using the actual calendar index: omitted dates contribute zero scores and do not become adjacent observations.

Apply the project's common finite-sample multiplier only once:

`T/(T−1) × (S T−1)/(S T−K_model)`.

For R1 this uses the original full panel's `S T` observations and its full-date-effect parameter count. A set of reference-difference regressions is an estimation device; its smaller stacked observation count must not silently replace the full panel correction. H3 differences and R1 have the same uncorrected covariance; any remaining corrected covariance difference should be explainable entirely by the stated multiplier.

Retain both one-year and two-year bandwidths. YoY overlap, persistent levels and short full-control samples remain relevant. A pooled sector panel supplies more outcomes, not more independent dates for weather inference.

## Equality and pairwise tests

Test all temperature slopes equal with a rank `S−1` contrast matrix, for example sector-minus-reference rows. For selected pairwise tests, use `Var(β_i−β_j)=V_ii+V_jj−2V_ij`; summing separate standard errors or variances drops the covariance term and is incorrect. Select meaningful pairs before inspecting rankings. If reporting all pairs, distinguish exploratory multiple comparisons and disclose any adjustment policy.

The joint Wald statistic is `d' (R V R')⁻¹ d`, where `d=Rβ`. Check the rank and numerical condition of `R V R'` explicitly. Report unavailable tests when it is singular; a pseudoinverse with unchanged contrast degrees of freedom would fabricate a valid omnibus test. A Wald chi-square reference is asymptotic. Using `W/(S−1)` against `F(S−1,T−1)` is also only an approximation under Driscoll–Kraay inference; it is not an exact small-sample F test.

There is a useful feasibility bound for H3/R1. If one sector regression has only `d=T−rank([B,A,W])` residual time degrees of freedom, its cross-sector temperature-score covariance has rank at most `d`. An omnibus equality test requiring `S−1>d` therefore cannot have an invertible covariance. The 29-quarter full-control sample has `d=12`: ten VA equality contrasts are possible in principle but potentially unstable. Excluding 2020–2022 leaves `d=5`: the ten-contrast VA omnibus test is necessarily singular. Five IIP contrasts are only borderline feasible. Actual covariance rank checks remain necessary even when this bound permits the test.

## Interpretation alongside stationarity diagnostics

Heterogeneous log-level coefficients remain descriptive level associations. Their statistical significance does not establish persistent growth damage. Stationarity diagnostics should be reported separately for outcomes and regression residuals, with the deterministic components and lag selection stated; passing a test does not validate a causal specification, and failure to reject a unit root is not proof of a unit root. Multiple sector diagnostics and short annual samples have limited power.

The distinction between a level mechanism and a growth mechanism requires a dynamic economic model. [Dell, Jones and Olken (2012), AEJ: Macroeconomics](https://www.aeaweb.org/articles?id=10.1257/mac.4.3.66), equations (1)–(4), treats the two mechanisms through current and lagged weather effects. Sector heterogeneity and stationarity checks complement that reasoning rather than replacing it.

## Completed independent implementation audit

[verify_heterogeneity.py](../scripts/verify_heterogeneity.py) passed all 180 model/outcome/bandwidth groups across three contexts and both control sets. It imports none of the heterogeneity-estimation helpers. Instead of the estimator's full-panel FWL fit, it uses independent national-mean/deviation regressions for H1/H2 and separate sector regressions for H3/H3X, then constructs joint calendar-date influence functions and Bartlett covariance directly.

The audit checks all sector slopes and full covariance matrices, model parameter counts, total panel R², confidence intervals and p-values, equality-test availability and F statistics, selected pairwise differences and Holm adjustments. It confirms the R1/H3 reference-contrast identity and uncorrected covariance identity at both bandwidths. Every separate-sector temperature estimate and its own HAC standard error was checked. The additional separate-sector tables' 1,104 weather coefficient rows, including every control coefficient, SE, interval, p-value and sector R², were verified from the same independent multi-outcome regressions.

The largest absolute sector-slope difference was below 3.8×10⁻¹², and the largest covariance-entry difference below 1.42×10⁻⁹. The full-control VA H3X equality tests are correctly unavailable for all three outcomes and both bandwidths: their covariance has at most rank five for ten restrictions. No material estimation, covariance or multiplicity-adjustment error remained. [The verification record](../outputs/heterogeneity/verification.json) preserves these checks.

Full-control quarterly omnibus significance remains particularly fragile: H3 has only 12 residual time degrees of freedom, while the VA equality test asks ten restrictions. Very small nominal p-values in that design should be presented beside this limitation rather than treated as unusually precise evidence. H3X retains endpoint exclusions dated 2020–2022 and does not make YoY observations dated 2023 free of a 2022 base-period effect.
