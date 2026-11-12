# PyFixest Wald-function check

The reported project tests use custom post-estimation code, with native PyFixest coefficients and explicitly constructed covariance/restriction matrices. Native `feols` also runs an automatic all-coefficient Wald test internally; that automatic result is not used for the project's sector/temperature hypotheses.

Installed version: **PyFixest 0.70.0**. Its installed public implementation and Wald kernel were checked against the official tagged source by an independent read-only reviewer.

- [Official public-method documentation](https://pyfixest.org/reference/estimation.models.feols_.Feols.html#feolswald_test)
- [Official v0.70.0 implementation](https://github.com/py-econometrics/pyfixest/blob/v0.70.0/pyfixest/estimation/models/feols_.py)
- [Official v0.70.0 Wald kernel](https://github.com/py-econometrics/pyfixest/blob/v0.70.0/pyfixest/estimation/post_estimation/wald.py)

## Actual execution

`check.py` refits three saved designs through genuine `feols`: complete VA12 NSA YoY distributed-lag H1, nine-region arrivals MoM H1, and 42-country arrivals MoM H1. It checks 18 restrictions/bandwidth combinations, including 12 existing reported joint-test rows.

For public-method parity, it explicitly substitutes the saved project calendar covariance in each refitted model, transforming both covariance and restrictions into the native orthonormal coefficient basis. This isolates Wald arithmetic. It does **not** claim PyFixest's built-in covariance estimator produced the archived calendar covariance. The lower-level native kernel is also called directly with identical original beta/V/R and the project's denominator degrees of freedom.

The maximum absolute difference in W was **1.43e−11** for the actual public method and **3.62e−11** for the native kernel. All 12 saved joint-test rows matched. No selected quadratic-form or W/r scaling error was found.

## Important API difference

The public method uses F only for the identity restriction on **every retained coefficient**, with a zero right-hand side. Arbitrary or subset restrictions switch to chi-square with a warning, even when `distribution="F"` is requested. The project reports its explicitly labelled approximate F(r, T−1) calendar convention.

| Saved equality test | Project F p | Native public chi-square p, same beta/V/R |
|---|---:|---:|
| VA12 YoY distributed response, 4-quarter bandwidth | 3.080111e−12 | 1.044446e−15 |
| 42-country arrivals MoM, 12-month bandwidth | 3.269123e−5 | 1.392706e−6 |

Thus native public p-values are not the existing reported F p-values. A bare `model.wald_test()` in the adapter would also test the entire orthonormal full-LSDV basis, including nuisance/FE coordinates, using the adapter's native iid covariance. That is a different hypothesis and covariance.

Additional executed API fixtures confirm native CRV1's default identity F test uses G−1 degrees of freedom, while a subset switches to chi-square. A singular-covariance fixture confirms that the native pseudoinverse can return a number with two requested restrictions and contrast-covariance rank one. The production rank/HC0 guards deliberately withhold such tests; a native returned p-value alone is not validation.

## Calibration remains unresolved

`reference_comparison.json` reuses archived Gaussian-null simulation draws/counts; it runs no new simulations. Applying the documented chi-square reference still gives 2,000/2,000 rejections for the selected heat 60-restriction zero test at both bandwidths. For the 11-restriction heat equality test, the archived F rejection counts of 712 and 1,091 become 753 and 1,134 under chi-square, out of 2,000 each.

The country-arrivals 41-restriction benchmarks also necessarily retain 4,000/4,000 rejections at both bandwidths: every archived W exceeds the F critical value, which is greater than the corresponding chi-square critical value. This is an implication of the stored counts and critical values, not a fresh empirical calibration.

These are conditional fixed-X Gaussian reference experiments, not an estimated error law for the actual economy. They demonstrate that changing the function or F/chi-square reference alone does not repair the selected failures. They establish no universal verdict on other models or scalar tests.

## Evidence and preservation

- `summary.json`: scope, actual refits, numerical differences and protected output hashes.
- `comparisons.csv`: all 18 native/project comparisons.
- `reference_comparison.json`: reuse of existing null draws/counts.
- `execution.txt`: actual diagnostic stdout.
- `installed_*.py.txt`: exact inspected installed implementations.

All 349 sealed sources and protected model arrays, test CSVs, baseline ledger, final closure receipt and combined HTML remained unchanged. Published package and numerical references were not modified. This is a bounded API/arithmetic check, not a full-grid native-Wald migration or a repair of the affected statistical inference.
