# SA and NSA year-on-year output comparisons

The original seasonally adjusted (SA) year-on-year analyses are retained. New, explicitly labelled non-seasonally-adjusted (NSA) companions use the official NSA output levels; NSA growth is not inferred from SA growth or from annual aggregate growth.

The cached SingStat sources are:

| Output | SA table | NSA table | Native frequency and units |
|---|---|---|---|
| Value added by industry, including official aggregate GVA | M015662 | M015661 | Quarterly chained (2015) million dollars |
| Industrial production by cluster and detailed industry | M355382 | M355381 | Monthly volume index, 2025 = 100 |

The corresponding metadata and original TableBuilder payloads are retained under `data/raw/{va,iip}_{sa,nsa}_{metadata,tabledata}.json`. The VA cache was last updated by SingStat on 11 August 2026, and the IIP cache on 26 October 2026. Existing source hashes and all original SA estimation values remain protected by the migration audit. These cached vintages are used consistently throughout the comparisons.

For output level `V` and calendar horizon `h = 4` quarters or `12` months:

```text
YoY log growth = 100 × [log(V[t]) − log(V[t−h])]
YoY ordinary percentage growth = 100 × [V[t]/V[t−h] − 1]
```

The NSA formulas use NSA levels at both endpoints. Exact sector/calendar lookups determine the annual base; a positional row shift after filtering is not a substitute. Quarterly IIP first averages the three official monthly NSA index observations in each complete calendar quarter, then compares the two quarterly means. It does not average monthly percentage growth rates. VA already has quarterly flow observations.

NSA levels are left-joined to the existing SA preparation by sector and calendar period. Existing SA columns, dates, weather controls and coverage masks must remain unchanged. Current source coverage supports the same SA/NSA samples; this is asserted independently. New NSA regressions use the same controls, fixed-effect structure, uncertainty calculation and exact comparison window as their SA counterparts. Added NSA models do not replace the original SA model IDs. Table metadata and visible descriptions distinguish SA from NSA.

The common-slope additions are `G2N` (NSA YoY log growth), `G3N` (NSA ordinary YoY percentage growth), and the applicable excluded-sample `N...X` variants. The annual-weather comparison adds `Y1N`–`Y4N`, using the same current, annual-average, annual-change or current-plus-annual-lag weather definitions as the corresponding SA model. Granular and sector-interaction analyses add NSA YoY log-growth outcomes alongside the existing SA outcomes. The `Y4N` temperature restriction, like `Y4`, tests the sum of current and annual-lag temperature coefficients using their full covariance.

COVID exclusions are applied to endpoint dates after growth and weather lags were constructed. Retained 2023 YoY observations can therefore use 2022 output as their base. Calendar indices remain uncompressed in all covariance calculations. Seasonal indicators remain in quarterly/monthly NSA specifications; switching to an NSA outcome does not remove the original seasonal controls.

Independent verifiers reconstruct SA and NSA growth directly from the two official raw-level payloads, including every retained broad and detailed sector and official aggregate. They verify exact calendars, unchanged SA masks, regression formulas and adjustment labels, coefficients, sample means and appropriate CR1 or calendar HAC inference. The frozen 177-file original-result comparison still requires every original SA row and numerical value to match. Added result rows must be explicitly identified as NSA; added input columns are enumerated and every pre-existing SA row/column is compared with the immutable pre-NSA checkpoint.

NSA is a companion outcome choice, not an automatic cure for trends, persistence, breaks or endogeneity. The existing stationarity qualifications and short-sample uncertainty limits remain applicable.
