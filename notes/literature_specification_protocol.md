# Literature-motivated Singapore specification extension

Implementation protocol, 11 November 2026. This is an explicitly post-exploration
extension, not a preregistration. The existing estimates and frozen v1.0.0
replication reference remain available. Results must be labelled exploratory.

The extension uses real VA and IIP growth. It preserves NSA YoY as the primary
YoY outcome and SA growth as a comparison. Complete GVA component panels contain
12 broad or 20 detailed units, including ownership of dwellings; the earlier
11/19-unit panels remain comparison definitions. National GVA is always the
published real total, not the sum of chain-linked component levels. National IIP
uses the official weighted index. Equal-unit panel slopes do not reconstruct
either national growth rate.

## Subannual dynamics

[Nguyen and Pienknagura, IMF WP24/202, equation (6), printed p.9](https://www.imf.org/-/media/files/publications/wp/2024/english/wpiea2024202-print-pdf.pdf)
motivates current temperature and four quarterly temperature lags, matching
precipitation history and four lagged YoY growth rates. The inspected version is
the working paper; the final 2025 ERE article and its data definitions must not be
silently substituted. Our monthly extension uses twelve months. Three models
compare contemporaneous weather, distributed temperature/rainfall lags, and
distributed weather plus lagged dependent variables. Current relative humidity
is retained as an additional Singapore control. The latter model estimates a
conditional dynamic association. FE with lagged outcomes can have finite-T bias;
the extension is not an Arellano–Bond or bias-corrected estimator.

Growth and all histories are constructed on the original calendar before
exclusions. Within context and outcome, all three models use the same maximum
history eligibility. The broad crisis policy is fixed in
`notes/crisis_sensitivity_policy.json`. Event-date exclusion removes the current
date only. Strict exclusion removes any crisis period from the complete output
support needed by the largest lagged-growth model: t−h−1 through t for one-period
growth, and t−2h through t for YoY growth, h=4 quarters or 12 months. Thus the
strict dynamic sample differs deliberately from the static core's one-year-safe
sample. Past weather during a crisis is not itself an economic-output outlier
and is not a separate exclusion criterion.

Temperature-lag sums use the complete covariance matrix. For a YoY equation with
T(t),…,T(t−h), the precise contemporaneous static-level restriction is jointly
zero intermediate lags and cancelling endpoints. A zero lag sum alone is a
weaker, different restriction. Equal one-period growth responses imply equal
coefficients on lags 0,…,h−1 and zero lag h. These are conditional restrictions
within the selected equation, not proofs of a physical mechanism. With lagged Y,
the direct weather-coefficient sum is not an unconditional long-run multiplier.
No division by one minus the autoregressive coefficient sum is reported.

## Annual dynamics and nonlinearity

[Dell, Jones and Olken (2012), Table 3 and equation (4)](https://economics.mit.edu/sites/default/files/publications/120626%20Climate%20Published.pdf)
motivates annual growth on weather levels and distributed temperature and
precipitation lags. We implement lag lengths 0, 1, 5 and 10, including current
weather in every case, rather than choosing a lag length from significance.
Current humidity is an additional control. The dependent variable is
100 × [ln annual real output(t) − ln annual real output(t−1)], matching their
log-growth transformation but not their per-capita outcome or multicountry
identification. Sector FE and a common linear trend are the annual Singapore
adaptation. Five- and ten-lag models can be unavailable in short histories; no
control or lag is silently dropped to manufacture a result.

[Burke, Hsiang and Miguel (2015), Supplement B.2, equation (15)](https://web.stanford.edu/~mburke/climate/BurkeHsiangMiguel2015_SI.pdf)
motivates annual quadratic temperature and precipitation, with unit-specific
linear and quadratic trends. We show this trend version and a common-linear-trend
comparison. Temperature and precipitation are centered for numerical stability;
centering changes neither the fitted quadratic function nor its marginal effect.
Current humidity is again retained. Temperature profiles interact both the linear
and quadratic terms with sector. Marginal effects and delta-method intervals
are evaluated at the observed 5th, 50th and 95th temperature percentiles, with the
linear/quadratic covariance included. No global optimum or out-of-support
Singapore damage curve is inferred.

Annual model comparisons use the same maximum weather-history eligibility (ten
years) within each context, with a separately labelled model-own-history set to
retain information from shorter records. Neither is selected by the estimates.
The annual strict crisis mask requires current and prior output years to be
event-free; there are no lagged dependent variables in these annual models.
Historical weather dates in the distributed-lag regressors do not enlarge this
output-based crisis mask.

## FE, inference and reporting invariants

All FE coefficients are fitted by native PyFixest through the existing audited
backend. Monthly/quarterly models use sector × calendar-season FE and common
year FE; separate series use season and year indicators. Annual models omit
full year FE because national weather would be absorbed. The original papers'
country/date FE cannot be reproduced with one national Singapore weather series.

Common panel slopes report requested sector CR1, alongside calendar-aware
Driscoll–Kraay sensitivity. National, sector-specific and separate-series effects
use joint calendar HAC with actual calendar gaps. Repeated sectors do not create
new independent temperature dates. Rank-deficient Wald restrictions have no
invented p-values; small effective time degrees of freedom and ill-conditioned
available tests remain visibly flagged. Mean Y, N, distinct dates, parameter
counts, weather support, masks and all unavailable requests are exported.

The extension concerns the timing and functional form of conditional weather
associations. It does not by itself resolve omitted national shocks, potential
pollution mediators, weak temporal replication, multiple testing, or weather-versus-
climate extrapolation. No ordinary unit-root screen certifies stationarity, and
gapped crisis samples must not be treated as contiguous time series.
