# Native Wald inference revision · 12 November 2026

The current displayed Wald inference uses the real PyFixest 0.70.0
`pyfixest.estimation.post_estimation.wald.wald_test` kernel with a chi-square
reference. This is the same kernel called by the public `Feols.wald_test` method.
For custom/subset restrictions the public method switches from F to chi-square.
The current implementation calls the kernel directly because the joint-system
coefficients, original coefficient basis, calendar covariance and declared
restriction matrix are supplied explicitly. It does not fabricate a fitted model
or describe a custom covariance matrix as native covariance estimation.

The test is H0: R beta = 0, with

    W = (R beta)' (R V R')^-1 (R beta)

and a nominal chi-square reference with q independent restrictions. The earlier
F statistic W/q, its denominator degrees of freedom and p-value remain explicit
historical comparisons. A one-restriction chi-square p-value can differ from the
retained coefficient t-test for the same null. Equality of sector effects tests
their differences; it does not test every effect against zero. A sector's total
response is beta_T + delta_i, including both variances and twice their covariance.

The source inventory contains 223,933 recorded test requests, including existing
sample/covariance choices, coefficient-audit checks and unavailable requests.
There are 166,747 recorded production native-kernel calls and 57,186 unavailable requests retained without a native test. These are not new regressions or independent hypotheses. The original fitted
coefficients, covariance, samples, outcome definitions, controls, fixed effects
and recorded model census remain byte-pinned to audit.1. Genuine native Stata
MCP/reghdfe evidence remains 99 heat and 38 arrivals keys. New PyFixest tests on
those saved Stata arrays are separately labelled.

The new layer reconstructs real R matrices and saved beta/V arrays, retains old
model/time/covariance unavailability, and runs rank, HC0-scale, structural-bound
and positive-definiteness guards before the native pseudoinverse is called.
No unavailable covariance is made valid merely because the native kernel can
return a number. Dynamic shape restrictions retain their conditional meaning;
two-channel restrictions use their minimal coefficient basis. Native arrays,
restriction matrices and source bindings are reviewable in the contrast archive.

Independent verification uses direct linear solves and the chi-square survival
function, checks the historical statistic/reference for comparability, verifies
source-row coverage and binds the exact output bytes before and after execution.
The official native API arithmetic check, full current-layer verification and
Gaussian calibration experiments have separate receipts and distinct scopes.

Switching references is not a calibration repair. The selected heat experiments
reuse archived Gaussian-null W draws: the 60-restriction test rejects 2,000/2,000
under either reference at both bandwidths; the 11-restriction equality test's
chi-square rejections are 753/2,000 and 1,134/2,000. These are reference-only
recalculations, with no new heat simulations or native economic fits.

The arrivals replay generates 4,000 fixed-design Gaussian-null draws for each of
two static designs, with the original seeds. It makes 16,000 actual native Wald
kernel calls across two bandwidths, while reproducing the historical F counts.
The eight-restriction chi-square test rejects 1,084/4,000 and 1,995/4,000; the
41-restriction test rejects 4,000/4,000 at both bandwidths. These selected laws
are not estimates of the actual economic error process, evidence of no real
effect, a verdict on every scalar test, or replacement empirical critical values.

The HTML adds current native-Wald panels and a global tab. Historical joint-test
content is labelled/collapsed for comparison; coefficient t-tests and intervals
are identified separately. Current Wald selections do not silently synchronize
with historical coefficient-table selections. Every displayed request identifies
its model, outcome, sample, covariance and restriction. Current exported Wald
CSV fields explicitly distinguish native chi-square and historical F p-values.

The original v1.0.0 and audit.1 GitHub releases and source seals are immutable.
The native revision uses a separately created source seal and release. The old
55-step reproduction is retained as historical model evidence; the new driver
rebuilds/verifies the additive Wald and presentation layer. Source/byte integrity,
mathematical agreement and browser checks do not establish causal identification,
universal stationarity, calibrated finite-sample inference or conference acceptance.
