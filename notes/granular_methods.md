# Detailed sectors and published manufacturing total

Implemented 11 November 2026 as additional analyses. `scripts/granular_analysis.py` generates the results; `scripts/verify_granular.py` independently checks them through `verify_granular()` or its alias `run_verify_granular()`. Earlier estimates are retained.

## Outcomes, sources and samples

The VA extension uses the 19 non-overlapping industry leaves in official [SingStat SA table M015662](https://tablebuilder.singstat.gov.sg/table/TS/M015662). Wholesale and retail, accommodation and food services, three business-service branches, and five other-service branches are separately observed from 2005Q1. The seven unsplit industries retain their full earlier histories. Dwellings are outside both the 11-sector and 19-sector panel.

Manufacturing uses the 18 non-overlapping industry leaves, six parent clusters and **published total row `1`** in official [monthly SA IIP table M355382](https://tablebuilder.singstat.gov.sg/table/TS/M355382). Quarterly SA IIP is the arithmetic mean of the three monthly indices, requiring a complete quarter. Index levels are averaged before log transformations. The published total incorporates the official index weights; it is not reconstructed as an equal-weight mean of clusters, subclusters, growth rates or regression coefficients.

Four outcomes share one sample within each context/window:

| ID | Definition |
| --- | --- |
| G1 | `100[ln A(t) − ln A(t−1)]`, MoM or QoQ log growth |
| G2 | `100[ln A(t) − ln A(t−h)]`, YoY log growth; `h=12` monthly or `4` quarterly |
| L1 | `100 ln A(t)` |
| L3 | Raw official level: chained-2015 S$ million for VA; 2025=100 points for IIP |

Growth is not annualised. These growth equations use the log-growth functional form associated with DJO; using that form does not inherit another study’s identification assumptions. Levels remain descriptive because of persistence, trends and the observational design. Current weather in a YoY growth equation is a distinct exposure specification; the outcome’s annual difference does not itself require temperature to be differenced.

All growth transformations use complete consecutive raw histories before merging weather, balancing sectors or excluding dates. Requiring all four outcomes shortens even G1/level samples to the dates where YoY is also observed. The broad-sector comparisons are restricted to exactly the detailed-panel dates:

| Context | Long-sample dates | Dates | Units |
| --- | --- | ---: | ---: |
| VA detailed / broad matched | 2006Q1–2026Q2 | 82 | 19 / 11 sectors |
| Monthly IIP detailed / broad matched | 1997Jan–2026Sep | 357 | 18 industries / 6 clusters |
| Quarterly IIP detailed / broad matched | 1997Q1–2026Q3 | 119 | 18 industries / 6 clusters |
| Monthly published IIP total | 1984Jan–2026Sep | 513 | 1 national series |
| Quarterly published IIP total | 1984Q1–2026Q3 | 171 | 1 national series |

The `long` window uses current temperature, rainfall and humidity. Monthly controls are actual NEA monthly measures; quarterly temperature/humidity are day weighted and quarterly rainfall is summed. The `full` window adds wind in m/s and PM2.5 per 10 µg/m³. It requires at least 90% temporal coverage in each month for both added variables; all three months must pass for a quarter. It retains 100 months or 29 quarters within 2017–2025. The `exclude` window additionally removes observations dated 2020–2022, leaving 66 months or 19 quarters. Annual outcomes still use their actual prior-year bases: for example, 2023 YoY outcomes can contain a 2022 base. This is an endpoint-date exclusion, not a pandemic-free exposure history.

## Estimands, fixed effects and weighting

Pooled panel regressions have sector-by-calendar-season effects and common year effects: a separate recurring monthly/quarterly pattern for each sector, plus annual macro shifts shared across sectors. Full date effects would absorb national temperature and the common weather controls. Published-total regressions reduce to an ordinary national time series with calendar-season and year effects. Individual detailed-sector regressions use calendar-season and year effects separately for each industry.

The pooled coefficient gives equal weight to every included industry on each balanced date. Moving from 11 to 19 VA industries changes the representation of their parent categories: other services, for example, now contributes five equally weighted leaves. In manufacturing, the six parent clusters contain 4, 4, 2, 2, 3 and 3 detailed leaves. An equal-weight 18-industry panel consequently puts more weight on parents with more leaves than an equal-weight six-cluster panel. Restricting dates controls the sample difference; it does not make these two weighting schemes equivalent.

Neither pooled coefficient is a national output-weighted temperature response. The published manufacturing total is the appropriate additional national index outcome. For VA, the existing national regression uses the separately published GVA aggregate including dwellings; chain-linked sector levels should not be summed to reconstruct it. The separate reconciliation audit documents that distinction.

## Inference and practical limits

Pooled common slopes have sector-cluster CR1 inference as requested, with calendar-HAC estimates retained for comparison. There are 19/11 VA clusters and 18/6 manufacturing clusters. Sector clustering permits serial dependence within a sector; it does not generally protect against residual shocks shared across sectors. The six-cluster comparison has particularly few clusters. Calendar-HAC is therefore an informative benchmark under common time shocks.

National totals and individual-sector regressions have one series. Their uncertainty uses calendar-aware Bartlett time-series HAC, which is the singleton-panel Driscoll–Kraay calculation. Sector clustering cannot supply an independent variance estimate for a regression containing only one sector. Primary bandwidths are 12 months or four quarters; pooled/total temperature slopes also retain doubled-bandwidth checks of 24 months or eight quarters, including for overlapping YoY growth. Nominal HAC inference follows the project’s approximate `t(T−1)` reference.

HAC uses actual calendar positions with empty dates inserted as zero scores; excluded months/quarters are not compressed into neighbouring observations. Full-control quarterly fits have 29 observed dates over 36 calendar quarters and only 12 residual time degrees of freedom. Excluding 2020–2022 leaves 19 dates and **five** residual time degrees of freedom. Individual short-quarter estimates and levels are correspondingly fragile, regardless of a conventional significance star. The fitted relationships are observational associations conditional on the included weather controls and fixed effects.

## Independent verification

The verifier imports no production preparation or estimation helper. It re-parses all 4,534 VA source cells and 11,334 IIP source cells, checks cached hashes and metadata completeness, derives leaf membership from the raw hierarchy and checks the saved catalogues. It independently reconstructs quarterly indices from complete monthly levels and all four outcomes from raw calendar lookups, including the pre-sample and excluded-year bases.

Weather is rebuilt from raw NEA monthly files. Wind is reconstructed from 155,115 cached half-hour values with the knots-to-m/s conversion. PM2.5 is rebuilt from the five-region hourly file using SGT hour-ending timestamps shifted back one hour. The independent pass excludes one invalid row and three conflicting timestamp-hours, leaving 109,577 hours; all five regions are required and no missing periods are imputed. Raw reconstructed monthly means, coverage flags and day-weighted quarterly controls match the saved files.

For pooled fits, balanced panels with common regressors permit a distinct verification route: regress the date-mean outcome on weather, calendar seasons and years using QR. Recover full panel residuals by adding each sector’s season-demeaned deviation from the date mean. This reproduces the common point coefficient and full-panel R² without constructing the production dummy matrix. Panel HAC uses the date-mean influence scores with the panel finite-sample multiplier `T/(T−1) × (N−1)/(N−K_panel)`. Individual sector/outcome fits share one independent QR factorisation and use manual Bartlett score products with multiplier `T/(T−K_time)`.

All **24 sample windows, 96 pooled outcome fits, 660 individual fits and 21,358 exact-sample rows** pass. This includes exact calendars, all four outcome definitions, balancing and matching, mean/SD of Y, R², point coefficients, both pooled HAC bandwidths, individual HAC SEs, confidence intervals and p-values. Maximum raw outcome error is below 1.8×10⁻¹³, maximum coefficient difference below 1.5×10⁻¹⁰, and maximum HAC SE difference below 3.5×10⁻⁸. The sector-CR1 covariance is covered by the project’s separate sector-cluster verifier; this file checks its pooled point estimates and descriptive fields but does not claim to independently validate that covariance.

The machine-readable audit is `outputs/granular/verification.json`. Model results, exact samples and individual-sector coefficients remain under `outputs/granular/<context>/<window>/`; `summary.csv` supplies context/sample counts and exact mean/SD.
