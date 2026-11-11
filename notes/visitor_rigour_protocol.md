# International arrivals: additional referee checks

This is an explicit extension of the preserved visitor analysis. It is exploratory,
not a preregistration. The original models, samples, native tables and verification
receipts remain available. No outcome-, residual-, significance- or stationarity-
based deletion is permitted. Frozen official inputs are reused without a vintage
refresh. One additional frozen historical HKO 2007 calendar, stored separately under
`replication/extensions/supplementary_inputs/calendar/`, supplies the preceding
December holiday history. Its URL/raw hash are retained; none of the old inputs
is overwritten. Reconstruct the ±7-day inclusive event windows from official
event dates, verify all existing monthly holiday counts, and look up lagged
holidays before arrival-calendar cropping. This avoids an artificial January
2008 PPML loss. Computational reproduction and causal identification are separate claims.

## Source and counting partitions

Use the cached official M550001 NSA monthly arrival counts and their series IDs,
not labels inferred from geography. Arrival events are classified by place of
residence; the series does not count unique people or necessarily sovereign states.
The source runs January 2008–September 2026; weather availability and quality rules
can end a regression earlier. Preserve the separately published SA national series
as an existing sensitivity, without inventing SA origin series.

The nine region totals omit top-level `Others`. The 42 named markets omit regional
residual buckets and top-level `Others`. Neither is an exhaustive national partition.
Two exhaustive, mutually exclusive partitions are ten region/other buckets and
52 named-market/residual buckets. Verify both monthly sums against the official
total, plus every regional parent/child identity. These are count identities:
component log growth and equal-component averages do not add to national growth.

Small residual buckets contain real zeros. Do not add one, interpolate, or quietly
drop their units. Keep the established positive-count log-growth support explicit;
use a zero-retaining count model to check exhaustive partitions. Report any PPML
separation, dropped observation or unavailable finite estimate rather than claiming
that every zero/unit was retained without examining the native estimation sample.

## Monthly FE equations

Use native PyFixest for every new FE regression. The main origin-heterogeneity
equation interacts temperature features with origins, with origin × calendar-month
FE, common year FE and shared non-temperature controls. Report total slopes
`beta_T + delta_i`, full linear-combination covariance, pointwise tests and defined
within-profile Holm adjustments. Retain the original fully origin-specific equations
as robustness; they allow different nuisance slopes and answer a different question.

Report three existing cuts: official total, nine regions and 42 named markets.
The two selected panels keep their established common positive-output calendar
across these cuts before model-history restrictions, making comparisons explicit.
New exhaustive partitions are primarily count-model checks because their zero
patterns otherwise remove many dates from balanced log panels.

New growth outcomes are `100[ln(A_t)-ln(A_(t-1))]` and
`100[ln(A_t)-ln(A_(t-12))]`, both NSA. Existing SA national, log/raw-level, daily-rate,
weighting, seasonality, future-weather and recovery-buffer sensitivities are retained.
Log points are not exact percentage changes; an exact growth-rate conversion needs
the relevant base and exponentiation. A growth coefficient is not a count-level
elasticity or a loss of visitor numbers.

Each monthly equation includes rainfall per 100 mm, relative humidity, and current
and prior-month Lunar New Year window days. Long controls omit wind/PM2.5. Full
controls add current wind and PM2.5 under the unchanged 90% monthly coverage rule.
Full histories for the included temperature/rainfall and growth lags are constructed
before current-control eligibility and exclusions. Historical PM is not required
when only current PM enters. Calendar controls distinguish this new companion
from the preserved weather-only baseline.

Define H=3 for MoM travel-timing sensitivities and H=12 for YoY overlap:

- Own-history current temperature/rainfall baseline.
- Current temperature/rainfall benchmark on the maximum-history comparison dates.
- Unrestricted current through H-lag temperature and rainfall; current other controls.
- The same weather history plus growth lags 1 through H, shared across origins.
- For YoY, endpoint-difference and trailing-year temperature SUM jointly, with the
  unrestricted current-through-twelve-lag rainfall and current other controls.
  This retains the distributed model’s rainfall nuisance space, so the temperature
  channel-span restriction is a valid nested comparison without also restricting rain.

The last four use the same maximum growth/weather-history calendar within each
outcome/control cut before crisis masks. No previous comparison-panel boundary may
truncate a valid raw growth lag. Report missing outcome/history, shared-panel
alignment, control coverage and exclusions separately. Require enough rank and
residual temporal information; display unavailable requests rather than dropping
controls to manufacture a fitted equation. Lag sums are direct conditional effects;
with lagged growth they are not unconditional long-run multipliers. Endpoint and
sum channels impose a testable lag shape, not an automatic consequence of using YoY.

## Calendars and inference

Use the same fixed broad crisis policy as the economy extension. Only 2008–2009
and 2020–2022 overlap the frozen arrival history. Report all eligible dates,
current-event-date exclusion, and full-output-support exclusion. For static current
models the strict comparison uses the common YoY-safe interval t−12 through t.
For matched dynamic comparisons it uses t−4 through t for MoM and t−24 through t
for YoY. Preserve earlier 2010/2023 recovery-buffer sensitivities separately.
Actual gaps remain gaps in lagged regressors and covariance calculations.

Destination weather is shared across origins. Calendar Bartlett HAC, at 12 and 24
months, remains primary for arrival slopes/profiles; origin CR1 is a labelled pooled
comparison under independent-origin assumptions. It cannot create independent
Singapore-weather realizations. Save each origin’s local design rank and the rank of the full cross-origin
temporal span. Define the primary working time rank as the maximum local rank,
which is invariant to reference-origin coding; use T/(T−K_time) and approximate
t(T−1). This is a stated finite-sample convention, not an exact theorem. If the
primary correction is undefined, retain an otherwise identified native panel
point fit and mark its primary calendar inference unavailable; do not silently
drop controls or substitute another covariance as the primary result. State all
finite-sample multipliers and approximate reference df. Report calendar/time-level
convention sensitivity without calling any HAC Student reference exact. Joint
tests require full contrast rank without a pseudoinverse; high-dimensional or
ill-conditioned computable tests remain visibly fragile. Significance across a
large, dependent grid is not evidence selected independently of that grid.

## Annual and count companions

Annual arrivals sum twelve complete monthly counts, excluding the partial final
calendar year. Annual temperature/humidity are day weighted and rainfall is summed.
Full-control years require all twelve months to pass the stated coverage rule.
Use annual log growth with origin intercepts and a common linear trend, with
current shared controls and common/H1 structures. Full year FE would absorb the
national weather regressor. Annual strict masks require current and prior years
outside crisis blocks. This short single-country history is not the cross-country
design in Dell–Jones–Olken. Report rank-short or unavailable annual equations.

For count levels, use native PPML with origin × calendar-month and common year FE,
current controls, and actual-calendar joint score covariance. Include official
total, selected and exhaustive cuts. Level equations have no lagged growth base;
strict/current-date exclusions can therefore be aliases. Record aliases without
counting them as independent models. Expected-count semi-elasticities and equal-
origin growth slopes are different estimands and have different implicit weights.

## Diagnostics, external checks and limits

Check exact saved Y/stochastic-X calendars with ordinary ADF/KPSS reference screens.
Do not compress gaps or test FE-demeaned arrays to manufacture stationary inputs.
Retain short, gapped, constant, mixed and lag-sensitive statuses. The annual source
contains fewer than twenty complete growth years, so the declared annual unit-root
screen cannot clear those equations. No pretest selects the reported samples.

Reconstruct samples, transformations, FE spaces, coefficients, residuals, full
covariance cross terms, total effects and restrictions independently. Check the
real HTML selector combinations and exports. Add separately labelled native Stata
`reghdfe` key-result comparisons with their own e(sample), FE/df and VCE conventions.

The [JRC tourism-demand study](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC131508/JRC131508_01.pdf)
uses monthly regional bed nights and a tourism-climate index, with economic
controls. Its outcome, exposure and many-destination variation differ from ours.
[The daily alpine study](https://onlinelibrary.wiley.com/doi/10.1002/jtr.2459)
distinguishes short trips and forecast-sensitive timing. It motivates timing
checks, not an assumption that realised monthly Singapore heat caused earlier
international booking decisions. These are tourism-specific references outside
AEA journals, not replications.

Origin demand, prices/exchange rates, airline capacity, events, restrictions,
origin weather and booking information remain potential confounders. Origin/year
and seasonal FE do not remove every monthly common shock. Adding more origins,
lags, covariance conventions or verified code does not resolve that identification
problem. Current weather coefficients cannot by themselves identify permanent
climate damage, destination substitution or a national tourism-revenue effect.
