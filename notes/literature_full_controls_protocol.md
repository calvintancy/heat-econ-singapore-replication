# Wind and PM2.5 companion to the subannual literature models

Implemented post-exploration robustness, 11 November 2026. This companion preserves
the requested full-control baseline while showing which expanded models the data
can identify. It adds no new upstream observations or imputed control histories.

Twenty VA/IIP contexts, three log-growth outcomes and three calendar masks give
180 exact samples. Common/H1 current, distributed-weather, dynamic-growth and
YoY two-channel models generate 1,056 requested designs. There are 574 native
PyFixest fits and 482 explicit unavailable designs. No requested term is dropped
to obtain an estimate. Separate-series expansion is not requested here; existing
full static and long distributed-lag own-series checks remain labelled robustness.

Every design includes current wind speed, current PM2.5, current humidity and
current rainfall. Distributed/dynamic/two-channel models also include rainfall
lags matching the temperature window (h=4 quarters or 12 months). Wind/PM enter
currently; their missing past values are neither imputed nor required. Current
eligibility follows the original 90% coverage rule. The loss tables distinguish
missing required raw histories, missing/insufficient-coverage controls, explicit
target-calendar alignment and fixed crisis exclusions.

Growth/AR histories use the complete raw official output calendar; temperature
and rainfall lags use the complete cached weather calendar. Comparison contexts
intersect the target's corresponding eligibility before masks. All specifications
within context/outcome share maximum-history dates. Strict masks use the complete
output support t−h−1 through t for short growth, t−2h through t for YoY. They leave
16 quarters for short growth and 13 for YoY; all-date and endpoint-excluded
versions retain 29 and 19 quarters respectively. These differ intentionally from
the shorter-support static complete-GVA strict comparison (17 quarters).

All quarterly dynamic requests are unavailable with all controls retained.
Monthly data identify more models, though some fitted samples have only one
residual time degree of freedom. Available effects and tests remain explicitly
fragile where support is small, covariance is ill-conditioned or restrictions
approach the score-rank bound. Requested common-panel sector CR1 appears beside
calendar HAC/DK. H1 total effects and cumulative/channel effects include complete
covariance cross terms; multi-feature tests use actual rank with T−1/G−1 bounds.

The separate full-control span postprocessor computes the h−1 two-channel-shape
restrictions from saved native coefficients and covariance, preserving unavailable
placeholders and all native fit files. Dynamic shape tests are conditional and do
not make a non-AR and AR equation the same model. Independent verification checks
every retained design, all reported weather effects/controls, masks, covariance,
rank decisions and 192 two-channel linear embeddings.

These are conditional growth projections, not a replication of multicountry
identification, a stationarity certificate or an unconditional long-run multiplier.
