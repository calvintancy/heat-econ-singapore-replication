> This note documents the preserved **legacy** diagnostics. The expanded audit in [stationarity_expanded_methods.md](stationarity_expanded_methods.md) adds all detailed series, totals, transformations and weather regressors, explicit exact-window coverage, two-year ADF and uncapped KPSS primary settings. Legacy computation success does not establish universal stationarity.

# Stationarity and persistence diagnostics

`scripts/stationarity_diagnostics.py` reads the cached outcome panels and writes `outputs/heterogeneity/diagnostics/`. It does not replace the regressions or download data. Run `.venv/bin/python scripts/stationarity_diagnostics.py`, or call `run_diagnostics()` after the outcome panels have been prepared.

## Series and dates

Each broad sector is tested separately for seasonally adjusted output's `100 × log level`, one-period log growth (`100 × Δ log`, MoM/QoQ), and YoY log growth. The three common national weather series are tested once per analysis context, rather than once per sector. The contexts have different history start dates, so their weather tests need not be numerically identical.

The primary dates match `outputs/outcomes/{context}/weather/common_sample.csv` exactly for every variable and sector: 178 quarters for VA, 405 months for monthly IIP and 135 quarters for quarterly IIP in the current snapshot. Values come from the processed panels and are checked against those comparison files; both inputs are hashed. The long comparison samples must be balanced and contiguous. Reindexing also exposes missing values before choosing the longest finite **contiguous calendar run**; missing observations are never joined across time or filled, and ties select the earliest run. A separate contiguous pre-2020 sample checks sensitivity to the pandemic period without removing middle dates and compressing the remaining calendar. This is not a formal structural-break test. `sample_audit.csv` reports source span, missing dates, selected dates, observation counts and discarded finite observations.

The added-control regressions use a shorter, gapped 90%-coverage sample. Standard ADF/KPSS are not run on that filtered sample: treating adjacent retained rows as adjacent calendar dates would be wrong, and the short quarterly history is weakly informative even aside from gaps. `full_control_windows.csv` reports eligible counts, calendar gaps and the longest contiguous stretch. Long-history diagnostics should not be presented as tests specifically validating that restricted sample.

## Deterministic components

The primary level-output tests include a constant and linear trend (`ct`). Growth tests include a constant (`c`). Output is already seasonally adjusted by the source; we do not seasonally difference it again.

Weather retains its physical units. Its deterministic month-of-year or quarter-of-year offsets are first estimated jointly with a constant and linear time trend. Only the seasonal component is subtracted, centered to preserve the series mean; the time trend is not removed at this stage. Temperature and humidity use `ct` because the question permits a deterministic trend. Rainfall uses `c`. Temperature also has a `c` sensitivity on the same adjusted observations. This seasonal adjustment is a descriptive preprocessing choice, not a seasonal-unit-root test. Usual univariate critical values remain approximate after estimated preprocessing and in the presence of structural breaks.

## Complementary tests and reported uncertainty

ADF's null is a unit root; its alternative depends on the specified deterministic terms. Augmenting lags are selected by AIC over 0–12 months or 0–4 quarters. The selected lag, cap, effective regression observation count, statistic, approximate p-value and critical values are saved. Selection at the upper bound is flagged; these caps are transparent choices, not a guarantee of white-noise innovations. See the [official ADF documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html).

KPSS reverses the null: stationarity around a constant (`c`) or linear trend (`ct`). Its long-run variance bandwidth starts with the package's Hobijn automatic selector and is capped at 12 months or four quarters. Both automatic and final bandwidths are saved; when the cap binds, the test is recomputed using the capped bandwidth. The implementation uses installed statsmodels 0.14.6; its signature and local documentation were inspected. See the [official KPSS documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html).

KPSS reports p-values only over its tabulated 0.01–0.10 range. Values outside are saved as `<0.01` or `>0.10` using separate numeric and relation fields; package warnings are retained. A boundary value is not an exact p-value. `tests.csv` retains the raw statistics, critical values, warnings and method details. Numerical errors are not silently converted into stationarity conclusions. A minimum of five seasonal cycles is required by this diagnostic screen; no current long/pre-2020 series hits that minimum.

At 5%, paired descriptions are deliberately neutral:

- ADF rejects / KPSS does not reject: evidence compatible with stationarity under the chosen deterministic specification.
- ADF does not reject / KPSS rejects: a nonstationarity concern under that specification.
- Both reject: mixed evidence, potentially involving breaks, lag choices or other misspecification.
- Neither rejects: inconclusive, including possible low power.

These are individual exploratory tests, not a family-wise inference procedure, definitive integration-order classification or automatic model-selection rule. Borderline cases should be read alongside the saved critical values. Trend versus constant choices, finite samples, revisions, pandemic shocks and other breaks can change the conclusions. A panel does not cure nonstationarity, and ordinary ADF critical values must not be applied to fitted regression residuals as a cointegration test. No residual cointegration test is performed here.

## Persistence and output files

`summary.csv` pairs the tests with sample information and descriptive ACF at lag one and one year (four quarters or 12 months). `acf_1`/`acf_h` use the tested series: seasonally adjusted weather or published SA output transformations. `raw_acf_1`/`raw_acf_h` retain the original weather seasonality. `detrended_acf_*` remove the specified constant/trend solely for a supplementary persistence description; they are not unit-root tests. ACF uses the usual sample-mean-centered, unadjusted normalization. High ACF in trending levels alone does not establish a unit root; overlapping YoY growth also creates persistence mechanically.

`series.csv` supplies raw values, seasonal components and tested values for charts. `outcome_summary.csv` and `weather_summary.csv` are convenient subsets. `evidence_counts.csv`/`summary.json` summarize the primary long-sample pairs; the latter also retains cautions and the full-control sample audit. `verification.json` records input hashes, software versions, transformation checks and the calendar-gap sentinel check. Independent NumPy reconstructions of the ADF t-statistics, KPSS cumulative-residual statistics and descriptive ACF are compared with every saved result. The scripts verify log-level and growth transformations against the cached output history wherever its required lag is available; the initial observations can legitimately use pre-panel history and are not falsely reconstructed by differencing truncated data.

Humidity persistence is a specification warning, not an automatic rejection of the year-fixed-effect regressions: those models condition on deterministic calendar patterns, whereas these univariate screens do not establish the order of integration of regression disturbances. No ordinary residual ADF is used to supply that missing argument.
