# Independent review of the visitor-arrivals analysis

Reviewed 11 November 2026. This review concerns only the separate visitor-arrivals analysis. The verifier does not import the production visitor preparation, estimation, projection or covariance functions. It reads the official cached inputs and saved designs, then reconstructs the results with NumPy/SciPy linear algebra and direct Bartlett sums. The persisted receipt is `outputs/visitor_arrivals/verification.json`.

## Reproduction

From the project root:

```sh
uv run python scripts/verify_visitor_arrivals.py
```

The verifier writes only `outputs/visitor_arrivals/verification.json`. It checks the hashes of all seven analytical inputs and the raw-source manifest before recording success. It does not rebuild or amend the economic report.

The economic HTML is an isolation observation rather than an analytical input. Concurrent project work changed `outputs/heat_economy.html` after the initial visitor-run snapshot. The receipt therefore records `main_economic_report_unchanged: false`, preserves both the original and current hashes, and records `concurrent_change_observed`. The original snapshot is not reset. This unrelated output change does not invalidate visitor calculations; any actual visitor/weather input mismatch still fails verification. The hash comparison does not establish who wrote the changed HTML.

## Source, outcome and sample checks

The official NSA source contains 225 months, January 2008–September 2026, and 62 series. The three analysis cuts contain the published total, nine nonoverlapping region totals, and 42 individually published country/market series. The latter is a selected-market panel rather than an exhaustive list of sovereign countries. Region totals plus the top-level Others category exactly equal the national total in every month; each region and its published children also reconcile exactly. Individually named markets do not exhaust the regional totals.

The audit reconstructs monthly log growth, twelve-month log growth, log levels, raw counts and daily-normalized monthly log growth directly from the raw counts. It verifies both aggregate SA transformations from the separate official SA file, verifies fixed 2017–19 arrival-share weights, and independently aligns raw weather, extended controls, calendar lags/leads and weather differences.

All 269 model masks are reconstructed, including all 54 leave-one-year-out estimates. The checks cover the shared balanced market support, each own-positive-date sensitivity, 90%/80% coverage, pre-2020 dates, endpoint exclusions, and strict exclusions involving growth bases. Full-control differenced-weather models must pass coverage at both weather endpoints. Lagged-temperature models and their matched-current comparisons have identical calendars. Missing months retain their original month numbers.

The country source contains 50 genuine zero cells. Log transforms leave their affected growth observations unavailable and the shared balanced log support reflects that explicitly; they are never replaced with arbitrary positive counts. The zero-retaining PPML checks are separate from this shared log support.

## Linear estimates and uncertainty

All 269 OLS designs are fitted afresh with independent least squares. Every saved coefficient and residual, every weather coefficient, both 12- and 24-month HAC variances, confidence intervals, p-values, mean/SD context, calendar residual autocorrelations, and the distributed-lag sums are checked. QR-based bread avoids the loss of precision from a direct inversion of the unscaled normal equations. Raw-count coefficients and variances are verified with scale-aware tolerances.

The joint market profiles are independently fitted and their entire cross-market HAC covariance matrices are reconstructed. Holm adjustments are checked against `statsmodels.stats.multitest.multipletests`, and the covariance rank, available equality-test statistics and p-values are checked independently. There are 16 profile model/bandwidth combinations.

Two original stacked FE regressions separately verify the equal-origin mean reduction. For pre-2020 MoM growth, the region design has 1,287 observations and 122 parameters; the country design has 6,006 observations and 518 parameters. Their temperature coefficients reproduce the collapsed coefficients to approximately 10⁻¹³. This confirms a mathematical equivalence, not independent confirmation from extra temperature observations. Equal-origin panels and national total growth answer different weighting questions.

## Zero-retaining PPML sensitivity

All 12 PPML estimation samples are reconstructed from the original counts and weather eligibility. The checks confirm that every intended origin and date is retained, with no country dropped or zero count altered. Country `long_all` retains all 50 zero cells; `full_all` retains 46. The pre-2020 and excluded-period PPML samples contain no zero cells.

The audit checks score equations, weighted orthogonality to each FE group, temporal finite-sample corrections and all reported covariance matrices. For representative region and country `full_all` models it independently builds the complete explicit FE design and solves the weighted projection using SciPy least squares. These designs have 116 and 512 FE parameters, respectively. The projected weather matrices agree within approximately 2.1×10⁻¹². The weather coefficients and FE space reconstruct the fitted log means, and the largest one-step Newton update across all PPML models is below 4×10⁻¹⁵.

PPML estimates a count-level log conditional mean. Its temperature coefficient is not a monthly growth effect, and its clean sample has no lagged outcome base to exclude. Its natural count weighting also differs from the equal-origin log-growth regressions.

## Limits that numerical verification does not remove

Common Singapore weather is shared across origin markets. Origin panels do not provide independent exposure replication. Calendar score covariance handles common disturbances and serial correlation under its assumptions, but cannot resolve omitted demand, policy, airline, event or origin-weather confounding. Contemporaneous realized temperature also differs from weather information available when a trip was booked.

Market-profile fits allow origin-specific controls, year effects and seasonal effects. They are fully interacted/separate-origin robustness estimates, rather than temperature-only interactions with shared nuisance slopes.

The country MoM `full_clean` equality test has 41 restrictions against only 44 residual time degrees of freedom. Its covariance condition number is around 1.4×10⁷. Very large Wald/F statistics and small nominal p-values in this setting are fragile high-dimensional inference; passing a rank check does not make them reliable evidence for a headline. The country YoY `full_clean` contrast covariance has rank 35 for 41 restrictions, and its omnibus p-value is correctly unavailable.

Stationarity screens are reference-history diagnostics, not clearance of gapped model windows, proof of stationary errors, or causal identification. The verification receipt validates the computations listed above and does not claim that those statistical assumptions have been established.
