# Rank and interpretation of temperature restriction tests

Audit correction, 11 November 2026. Coefficients, residuals and covariance matrices
are unchanged; this note concerns the rank metadata and availability of new
multi-feature Wald tests. It does not reset any original release reference.

For original-coordinate observation influences, aggregate within each actual
calendar date to obtain a T × K score matrix S. OLS first-order conditions imply
that the rows sum to zero. Consequently rank(S) ≤ T−1. A calendar sandwich and
any contrast covariance RVR′ cannot have larger rank. For unit-cluster scores,
the analogous universal upper bound is G−1. Actual rank can be smaller. Bartlett
lags use the true calendar, including gaps; they do not turn a missing date into
an adjacent observation.

For a balanced **single-temperature equality-of-sector-slopes** model, a stronger
bound can be established. Shared controls and shared time effects cancel from
sector differences. Those differences solve a time-series regression on sector
seasonal intercepts and one temperature term. Their contrast-score columns have
the same scalar temperature weight multiplying residual-difference columns.
Thus their rank is at most T−Q−1 for Q seasonal intercepts; annually Q=1 gives
T−2. This is a property of that particular contrast and design, not a universal
residual-time-degree-of-freedom rule.

Stacking several temperature features creates different temporal weights, so the
single-feature bound does not carry over to the stacked response vector. The
new lag/seasonal postprocessor therefore computes the actual restriction
covariance SVD and uses the universal T−1 (or G−1) bound. It retains the original
relative singular-value threshold 1e−10 and zero-scale threshold 1e−12 relative
to the HC0 reference. No tolerance is loosened to obtain an available test.

This corrects 172 reported subannual rank fields and 448 seasonal rank fields.
Thirty-two subannual tests become numerically available, including twelve global
lag-vector tests with 234 restrictions and 237 weather dates. Seasonal
availability does not change. The original native-stage test CSVs are preserved
with SHA lineage. Independent checkers recompute restrictions from full saved
designs and covariance, rather than adopting the former cap.

Numerical availability is not a validity certificate. Tests with many
restrictions relative to the score-rank bound, ill-conditioned contrast
covariance, or fewer than ten residual time df receive explicit fragility flags.
The newly computable tests are not headline causal evidence. Rank-deficient or
numerically degenerate tests retain no p-value; no pseudoinverse statistic is
substituted.

For a displayed sector total βT+δi, H0: βT+δi=0 is **one** linear restriction,
with variance Var(βT)+Var(δi)+2Cov(βT,δi). A cumulative lag effect is also one
linear contrast, using every covariance among its lags. “All sector lag
coefficients zero”, “all sector cumulative effects equal”, a zero cumulative
response, a static-level shape, and a two-channel span are distinct hypotheses.
The general YoY two-channel shape has h−1 restrictions:

```
βj − β1 = 0, j = 2,...,h−1
β0 + βh − β1 = 0.
```

It embeds the lag vector [a+b,b,...,b,−a]. In a dynamic equation this is a
conditional weather-shape test. It does not make that equation identical to a
non-AR two-channel equation without also restricting the AR coefficients.

Lag order and growth aggregation also matter. If one-period growth genuinely has
temperature lags 0 through L, summing h consecutive growth equations generally
produces YoY temperature lags 0 through h−1+L, with accumulated control histories
as well. Short-growth DL0–4 and quarterly-YoY DL0–4 therefore need not have the
same current coefficient: they are differently truncated projections. Our stated
one-year YoY lag window follows the inspected quarterly specification and nests
the two simple contemporaneous channels; it is not a claim to encompass every
longer-lived mechanism or to be the exact aggregation of each short-growth fit.
