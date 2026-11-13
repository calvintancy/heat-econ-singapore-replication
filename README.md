# Heat and the Singapore economy

## View the results

- **[Download heat_economy.html](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html)** — the current complete report (325 MB). Save the file, then open it in a desktop browser.
- **[Download the smaller ZIP](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat-econ-report.zip)** — 143 MB. Extract it and open `heat-econ-report/heat_economy.html`.
- **[All replication files and checksums](https://github.com/calvintancy/heat-econ-singapore-replication/releases/tag/v1.1.0-audit.2)** — full source, frozen data, estimates and verification evidence.

The generated HTML and data are distributed as release downloads. They are not files in GitHub's code tree. Analysis links in this write-up download the same complete HTML; select the named chapter using its tabs. GitHub does not display this interactive report inside its source viewer.

## Native Wald revision · 12 November 2026

For viewing, download the [combined HTML bundle](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat-econ-report.zip). The separate [extension prerelease](https://github.com/calvintancy/heat-econ-singapore-replication/releases/tag/v1.1.0-audit.2) carries all replication ZIP parts and checksums. The repository and release downloads are public. Standalone report and data paths shown below refer to files inside the full replication package.

The current Wald readout uses the actual **PyFixest 0.70.0 native Wald kernel with a chi-square reference**, supplied with each saved coefficient vector, joint covariance and explicit restriction matrix. It is separate from native covariance estimation. Current Wald panels cover the legacy and extended heat/VA/IIP/arrivals analyses, Stata-derived contrasts and temperature-sum tests; historical approximate F results remain labelled comparisons. Coefficient t-tests, confidence intervals, samples, controls and fitted models are unchanged. The native function does not repair the selected calibration failures. See [the current native-Wald results](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) and [the revision protocol](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/notes/native_wald_revision.md).

The updated [single-file HTML](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) adds complete GVA/crisis comparisons, literature/dynamic models, exact-calendar diagnostics, an evidence audit, international-arrivals robustness and independent Stata checks. Complete broad/detailed VA panels contain 12/20 components including ownership of dwellings; the earlier 11/19 definitions remain explicit comparisons. Official aggregate GVA remains its separately published real total. The overview includes complete-component common slopes and exact-sample growth means.

The literature extension implements unrestricted quarterly/monthly temperature and rainfall histories, lagged-growth sensitivity, joint YoY endpoint-difference/accumulated-exposure models, annual 0/1/5/10-weather-lag models, annual quadratic weather/trend comparisons, and annual calendar-quarter exposure with current/prior weather and AR1 sensitivities. NSA YoY/annual growth is primary and SA remains alongside. Sample-history alignment, full crisis-window rules, rank-based unavailable equations/tests and all interpretation boundaries are stated in the HTML. These are post-exploration conditional associations, not a causal or stationarity certificate.

The arrivals extension adds 299 OLS and 20 PPML fits, complete regional/market count reconciliation, raw-calendar growth and lag construction, annual aggregation, origin-specific year effects, long/full-control windows and fixed crisis masks. Its current-date and strict-growth-window exclusions are distinct. Growth histories with calendar gaps or too few observations remain untestable by ordinary ADF/KPSS. The new arrivals comparisons lead with calendar-HAC uncertainty.

Actual **Stata MCP** runs re-estimated **99 heat and 38 arrivals keys with `reghdfe`**, retaining reusable do-files, native esttab tables, MCP/native logs and separately executed Python ports. Open [the Stata heat checks](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) and the Stata section of [the arrivals chapter](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html). These selected implementation checks do not claim Stata coverage of the entire grid.

The audit found a substantive inference limitation: a selected high-dimensional joint Wald test rejects a known zero null in all 2,000 simulated draws. Its nominal p-values are therefore qualified rather than presented as established heterogeneity evidence. The [audit chapter](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) distinguishes accepted findings, corrected source/display defects and remaining identification/calibration concerns.

For this extension, use the separate [replication instructions](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/replication/extensions/README.md), extension package/manifest and the current `scripts/replicate_native_wald.py` driver. The published v1.0.0 release below is an immutable earlier snapshot; its assets and frozen numerical references are unchanged, and it does not contain these new families. [History-alignment correction](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/notes/literature_history_alignment_v2.md) distinguishes unavailable raw lags from explicit target-calendar matching.

## Replicate or check the results

The [original GitHub release v1.0.0](https://github.com/calvintancy/heat-econ-singapore-replication/releases/tag/v1.0.0) contains a portable **[replication-full.zip](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.0.0/replication-full.zip)** with the source, frozen inputs, saved results and offline HTML reports. Extract it, then open `outputs/heat_economy.html` or `outputs/visitor_arrivals.html` to inspect the results without installing software. The requested industry-clustered interaction sensitivity is also available as `outputs/clustered_sector_sensitivity.html`. The latest results are in the current release linked above.

For a code checkout and independently checked rebuild, follow **[REPRODUCING.md](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/REPRODUCING.md)**. It covers the original v1.0.0 release, locked Python/Node/R setup, quick verification and the full frozen-input reproduction command; use the extension instructions above for the current revision. [DATA_SOURCES.md](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/DATA_SOURCES.md) identifies the source vintages; [THIRD_PARTY_NOTICES.md](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/THIRD_PARTY_NOTICES.md) explains redistribution boundaries. SHA-256 inventories distinguish published snapshots from regenerated files.

The [completed handoff validation](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/replication/VALIDATION.md) records the separate-checkout full rebuild, its environment, numerical checks and execution log.

## Analysis overview

**YoY growth now has both NSA and SA versions.** The added NSA version uses official unadjusted output levels; the existing SA estimates remain separately labelled and retained on their original dates. MoM/QoQ growth continues to use SA output. The **Annual analyses** tab adds annual output growth on annual weather, with NSA primary and SA alongside, across aggregate VA, broad/detailed VA, published total IIP and manufacturing clusters/industries.

Annual VA is the sum of four quarterly levels; annual IIP is the mean of twelve monthly index levels. Annual growth is `100 × [ln A(year) − ln A(year−1)]`, calculated before exclusions. Annual temperature/humidity are day-weighted and rain is summed. The annual Singapore panel uses sector FE and a linear time trend: common year FE would absorb annual national temperature. This matches the Dell–Jones–Olken annual log-growth transformation/frequency while retaining distinct single-country identification. Full controls have only nine years (two residual time degrees of freedom); excluding 2020–2022 leaves six years and underidentified full-control specifications, displayed as unavailable. All requested controls stay in those specifications.


The original v1.0.0 regression matrix is paired in **[Full vs no COVID](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html)** and embedded in [the complete report](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html). Native PyFixest/MakeTables tables show all eligible dates beside the same specification excluding endpoint dates in **2020–2022**. This covers its long and full-control samples, growth and levels, distributed lags, sector FE profiles, separate series, joint tests and pairwise comparisons. Means, significance and sample counts are reported for both columns. Pre-2020-only models have identical columns. The newer crisis/literature/arrivals chapters separately report their declared all/event-date/strict-window masks; those comparisons are not all in the original COVID tab.

**Main common-slope VA/IIP panel errors cluster at sector/industry i** (CR1, t(S−1)). National totals and individual-sector profiles retain their labelled calendar-HAC exceptions; the new arrivals chapter leads with calendar HAC. The main heterogeneity model remains sector temperature interactions with FE; separate-series models are robustness. Growth and weather lags are computed before excluding dates, preserving actual calendar gaps and YoY bases. These newly gapped exclusions do not receive a stationarity clearance from the full history.


**Main entry point: [the complete tabbed HTML analysis](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html).** This single portable file embeds the aggregate VA, sectoral VA, monthly/quarterly IIP, growth/level comparisons, wind/PM2.5 controls, heterogeneity, diagnostics and full source/method notes. It works offline; links between analyses switch tabs within the file. The standalone reports below remain available.

The opening overview puts each coefficient beside **mean growth and its standard deviation in the exact estimation sample**, with a +0.1°C coefficient/interval rescaling. It has explicit long-sample, full-control and **short full-control excluding 2020–2022** selections, for both one-period and YoY log growth. Means are signed means of the observed dependent variable; no coefficient-to-mean ratio is used. Model tables also include mean Y and SD Y, and the heterogeneity report supplies sector-specific means.

The [aggregate VA analysis](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) uses SingStat's separately published **gross value added at basic prices**, including ownership of dwellings and excluding product taxes. It does not sum chain-linked sector values or substitute the equal-sector panel mean. Its quarterly time-series specification uses quarter-of-year and year effects, with calendar-aware Bartlett HAC. It repeats the main controls, outcomes and sample checks, including all 29 full-control quarters versus the 19 quarters retained after removing 2020–2022 endpoints. The short excluded model has only five residual time degrees of freedom; its nominal inference is fragile.


The [detailed VA results](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) add the non-overlapping 19-sector cut, including wholesale and retail separately, with the 11-sector model on the same dates. The [manufacturing detail results](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) add published weighted total IIP and 18 individual industries, at monthly and quarterly frequencies. Each includes one-period growth, YoY growth, log levels, raw levels, long/full controls, COVID exclusion, sample mean/SD and individual-sector HAC estimates. Five-outcome matched histories start 2006Q1 for detailed VA and 1997 for detailed IIP; source transforms precede all filters. Finer panels change equal-parent representation as well as classification, even on matching dates.

The [aggregate checks](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) audit VA accounting boundaries and IIP annual weights. Complete current-price VA identities reconcile within rounding; real chained volumes need not sum. IIP annual-overlap and two-stage 18-industry checks are rounding-compatible in recovered matching years 2022–24 and 2026; 2025 uses mismatched old-base release weights and is explicitly excluded from that claim. A complete historical compilation-weight panel was not recovered. National regressions use official published totals rather than a fitted-weight reconstruction.

Native tables label the **Dell–Jones–Olken log-growth form**: 100×log differences. Our short-period/rolling-YoY frequencies and single-country identification differ from its annual multicountry design. A growth coefficient of −1 is approximately one percentage point less growth per +1°C, conditional on controls/FE; it does not establish permanent growth damage. Levels and exact-percent outcomes are labelled as separate sensitivities.


The **main detailed-sector heterogeneity model** now uses temperature × sector interactions, sector × season fixed effects, shared weather controls and common year effects. The detailed VA and manufacturing tabs lead with these FE interaction slopes. Separate time-series estimates, with sector-specific controls and season/year effects, sit under robustness; the common-slope sector-CR1 panel is a benchmark. Sector-specific interaction slopes use joint calendar-aware HAC, with omnibus equality and all-pair Holm-adjusted comparisons. The excluded quarterly omnibus covariance is singular and is not replaced by a pseudoinverse or artificial p-value.

The requested [qualified industry-clustered interaction sensitivity](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) applies sector/industry CR1 to the same H1 fixed-effect design and exact parent common samples, alongside the calendar-DK baseline. It covers broad/detailed VA and monthly/quarterly IIP, five output definitions, and long/full/full-excluding-2020–2022 windows. Each industry's slope is supported by one industry. All 90 profiles have sector total SEs equal to the reference SE to numerical roundoff; shared nuisance estimation does not supply reliable sector-specific clustered inference. The qualified viewer reports that algebraic SE once and suppresses individual CR1 p-values, intervals and stars. Raw historical outputs remain explicitly labelled downloads. Its native fits have a separate ledger under `outputs/clustered_sector_sensitivity/pyfixest/`; the main 1,510-fit ledger is retained.

The **Stationarity** tab maps every fitted Y and weather X to its exact calendar and separately labelled full/pre-2020 histories. ADF (two-year AIC cap) and uncapped automatic KPSS are primary, with the former one-year settings retained as sensitivity. Compatible, trend-compatible, mixed, inconclusive, lag-sensitive, unit-root-concern and unavailable states are explicit. **The audit does not establish that all specifications are stationary.** Some growth series are mixed, humidity raises persistence concerns, levels remain descriptive and gapped short samples cannot be cleared by standard tests. A compatible full history does not clear a different short model window.

Sector tables/plots now display unrounded-p-based significance categories and explicit p-values. Filled versus open dots show 5% significance, with text duplicating the cue. These remain nominal pointwise tests unless explicitly Holm-adjusted; significance does not override an unresolved stationarity screen. Regression samples, coefficients and covariance choices are not changed by diagnostic pretesting.


Production linear/OLS FE regressions use native **PyFixest `feols`** (version recorded in the lock and receipts), including national/individual time-series models containing season/year effects; count models use native `fepois`. The linear adapter supplies a full-rank FE dummy design, preserving samples and parameter counts. Reported sector-CR1/calendar-HAC covariance is explicitly calculated and independently verified. The separate Stata implementation uses `reghdfe`. Diagnostics and independent verification retain separate numerical implementations.

Sectoral temperature interactions now use an explicit reference industry: `temperature + temperature × industry`. The reported effect is **βT for the reference and βT + δᵢ for every other sector**, with the sum’s joint-covariance SE. The reference δ=0 is a normalization, not an estimated precise zero. Component coefficients and covariances are downloadable; the full-date-FE R1 check identifies only relative differences. Native estimator provenance is in `outputs/pyfixest/fit_log.csv`; independent old/new comparisons are in `outputs/pyfixest/migration_verification.json`.

The default baseline includes **rainfall, humidity, wind and PM2.5**, using all eligible 2017–2025 dates. COVID exclusion and the longer history without wind/PM2.5 are robustness checks. Unavailable exact-window stationarity screens are now labelled “Not tested · calendar gaps”, rather than displayed as zero compatible units. This separates lack of a test from evidence against stationarity.

First project specification: quarterly sectoral real value-added growth on mean air temperature, rainfall, humidity, sector-by-season fixed effects and year fixed effects. This adapts the earlier EMA heat/electricity-demand analysis to a genuine sector panel.

Open [the interactive HTML report](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html), read [the corrected results note](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/first_results_corrected.md) or [the prior-study reference](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/notes/prior_electricity_reference.md), or inspect [the regression table](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/regression_results.csv). The HTML report is self-contained, works offline, and includes sample comparisons, charts and downloadable estimates. The audit retains the historical note and separately supplies corrected covariance labels for its control rows.

For conventional side-by-side estimation tables, open [the regression tables](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html). These use MakeTables ETable, the engine used by PyFixest, with coefficients, robust standard errors in parentheses, significance stars, fixed-effect indicators, sample counts and R². The exact saved inference is retained.

The existing growth results are **MoM** (monthly IIP) or **QoQ** (quarterly VA/IIP): `100 × [ln A(t) − ln A(t−1)]`, using seasonally adjusted output and not annualised. Their point estimates are retained; main pooled-panel uncertainty now clusters at sector i, with the earlier calendar-HAC results retained for comparison. The [growth and levels comparison](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) adds:

- **YoY log growth (NSA and retained SA):** `100 × [ln A(t) − ln A(t−h)]`, with h=12 months or four quarters. An exact percentage-change sensitivity uses `100 × [A(t)/A(t−h) − 1]`.
- **Log output level:** `100 × ln A(t)`. Temperature coefficients approximate percent output differences per 1°C, rather than growth percentage points.
- **Raw output level:** seasonally adjusted real VA in chained-2015 S$ million, or IIP in index points (2025=100). Pooled VA coefficients are average sector amounts, not aggregate Singapore GDP losses.

Each six-column comparison uses identical sectors and dates across outcomes. YoY is calculated from the full underlying level history before cropping the sample: the common long IIP samples begin in January 1993 / 1993Q1. Rainfall/humidity comparisons retain the long sample; the separate wind/PM2.5 comparison retains the existing 90% coverage dates. Level models also have a changing-seasonality sensitivity using common season-specific linear slopes over years. This is not a stationarity test; sector-specific linear trends would not diagnose common-temperature confounding in this balanced panel because the common linear trend is already spanned by year and season effects.

**YoY output does not automatically require YoY temperature.** The new YoY table compares current weather levels, trailing-year equal-period weather means, YoY differences in all weather variables, and current plus same-period-last-year weather, on identical long-sample dates. These have different interpretations. A differenced static output-level model implies weather differences; a period-growth model can imply accumulated exposure across the growth window. The difference specification retains fresh seasonal/year controls and is therefore not an exactly differenced copy of the fitted level model. Temperature changes are in °C, never percentage growth in Celsius. See [the four-paper AEA literature note](https://github.com/calvintancy/heat-econ-singapore-replication/blob/v1.1.0-audit.2/notes/aea_temperature_literature.md) for the primary sources and algebra.

YoY outcomes overlap across observations. These growth/weather comparisons save calendar-aware Driscoll–Kraay sensitivity results with four/eight quarterly or 12/24 monthly bandwidths; their main inference remains separately labelled. The shorter quarterly full-control models have only 12 residual time degrees of freedom, or nine with changing-seasonality slopes; endpoint exclusions leave five. Approximate stars and intervals are fragile. The new exclusion check removes observation dates in 2020–2022; 2023 YoY outcomes still use 2022 base periods, so this is not labelled a pandemic-free sample. Neither output levels nor a significant coefficient establish a causal or permanent climate-growth effect.

The [sector heterogeneity report](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) adds comparisons for MoM/QoQ growth, YoY growth and log output levels, using the same 178/405/135-date common samples and a separate full-control view. Native regression tables and coefficient plots show:

- **H1:** sector-specific temperature slopes; shared other weather slopes and year effects; sector-specific seasonality.
- **H2:** every weather response is sector-specific; year effects remain shared.
- **H3:** weather responses and year effects are sector-specific. Temperature point estimates match separate sector OLS regressions; the report also contains each sector's full regression table with its own calendar-aware HAC uncertainty.
- **R1:** full date effects, sector-specific year/season effects and nonreference weather interactions. These identify differences from manufacturing (VA) or electronics (IIP). In this balanced design they reproduce H3 contrasts, so they are an interpretation check, not independent corroboration. The uncorrected joint contrast covariance also agrees; full-panel parameter corrections differ.

Historical equality tests use joint cross-sector covariance with an approximate F(q,T−1) reference. The current native-Wald panels report chi-square p-values from the same preserved covariance and actual explicit restrictions, with the historical F result alongside. Selected economically motivated pair differences include pointwise intervals and Holm-adjusted p-values. Both one- and two-year HAC bandwidths are saved. H3 also has a 2020–2022 endpoint exclusion. The short full-control VA exclusion has ten equality restrictions but covariance rank at most five; its omnibus test is explicitly unavailable, rather than calculated with a pseudoinverse. Other short-sample omnibus p-values remain fragile even when their covariances are full rank.

ADF/KPSS diagnostics use the exact contiguous long-sample dates and a pre-2020 contiguous sensitivity. Log levels include a deterministic trend; growth includes a constant. Weather seasonality is removed while retaining its trend, with temperature/humidity trend tests and rainfall constant tests. ADF uses AIC-selected lags up to one year; KPSS automatic bandwidths are capped at one year. Censored KPSS p-values, warnings, lag choices and raw/tested series are retained. Non-rejection proves neither stationarity nor a unit root. Levels show substantial persistence concerns; growth evidence is stronger but can be mixed. Humidity warrants persistence/break sensitivity. Fixed effects and HAC do not by themselves establish stationarity or causal validity.

The full-control samples have seven quarterly/eight monthly gaps. Standard unit-root tests are not applied after compressing those dates. Model residual correlations match actual calendar lags and are reported as descriptive persistence checks. Independent NumPy reconstructions verify the ADF, KPSS and ACF statistics; a separate analytical mean/deviation estimator verifies every heterogeneity model, full joint covariance, equality/pair tests and separate-series estimates.

[Wind and PM2.5 control comparisons](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) add both controls alongside rainfall and humidity for all three outcomes/frequencies. These use a shorter 2017–2025 window, with the original long sample shown separately and four specifications on identical retained dates. Primary quality screening requires 90% temporal coverage in every month for each added variable; quarterly models require all three months to pass. An 80% coverage sensitivity and COVID exclusion are included. The large negative short-quarterly estimates are highly pandemic-sensitive and should not be treated as stable causal heat effects.

Wind comes from the earlier electricity study's derived NEA station-average half-hours, converted from knots to m/s. PM2.5 comes from the official hourly bulk series, averaged across all five regions, with hour-ending timestamps assigned to the preceding hour. Inputs and hashes are cached under `data/raw/extended_controls/`, so reproduction does not depend on the earlier project's paths once cached. Running `--refresh` updates the original official VA/IIP/climate inputs; it retains this explicitly frozen added-control snapshot.

The [IIP report](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/heat_economy.html) repeats the analysis using monthly manufacturing IIP and quarterly averages of monthly index levels. The broad panel has six manufacturing clusters (first monthly growth 1992-02; first quarterly growth 1992Q2). The detailed panel has 18 non-overlapping industries (1996-02/1996Q2 onward). IIP is manufacturing real production output, distinct from real value added. Source: SingStat M355381/M355382 (monthly NSA/SA, 2025=100, SSIC 2025), with published quarterly NSA M355391 used to verify aggregation.

The broad panel has 11 non-overlapping sectors from 1982Q1–2026Q2. The detailed panel has 19 sectors from 2005Q2–2026Q2. Real VA is from SingStat, and the monthly Changi temperature, rainfall and humidity series are from NEA via data.gov.sg. Exact source URLs, vintages and hashes are in [the source manifest](https://github.com/calvintancy/heat-econ-singapore-replication/releases/download/v1.1.0-audit.2/official-source-manifest.json).

The dependent variable is 100 times the quarterly log change in seasonally adjusted real VA, not annualised. Weather means are day-weighted; rainfall totals are summed. A national temperature series cannot be estimated alongside full year-quarter FE. The baseline therefore uses separate year FE and sector × quarter-of-year FE. Main common-slope panel standard errors cluster by sector (CR1, t(S−1)); S is 11/19 for VA or 6/18 for manufacturing. This permits dependence across dates within each unit but not across sectors. Calendar-aware Driscoll–Kraay results are retained for cross-sector/time dependence sensitivity. National single-series and sector-specific slopes retain calendar HAC as their baseline; the separately labelled industry-clustered sensitivity does not remove its single-industry and degeneracy limitations.

These are exploratory conditional associations. Common national weather, quarterly macro shocks, pandemic sensitivity and the absence of an exogenous identification argument limit causal interpretation. Equal sector weights do not estimate an aggregate GDP effect.

## Current revision

With the v2 replication inputs/results restored, use the locked environment:

```sh
uv sync --locked --python 3.11
uv run --locked --with playwright==1.65.0 python scripts/replicate_native_wald.py verify --with-browser
uv run --locked --with playwright==1.65.0 python scripts/replicate_native_wald.py reproduce --install-master --with-browser
```

This rebuilds native-Wald inference and the current HTML from the preserved model fits. It does not rerun the original economic regressions.

## Original model pipeline

Run the historical commands below in the original release or a separate workspace. The current native revision uses the commands above. Requires Python 3.11+ and `uv`.

```bash
uv sync --locked
uv run python scripts/run_all.py
```

To verify and rebuild the paired tables and complete HTML from the fitted comparisons:

```bash
uv run python scripts/covid_comparison.py --reports-only
```

The paired engine uses isolated native PyFixest receipts under `outputs/covid_comparison/`; it preserves the original results and estimator receipt. `verification.json` checks coefficients, covariance, exact sample masks and coverage independently; `report_verification.json` checks the native tables and every selector combination.


The supplied raw files are cached and reused. To explicitly replace them with current official vintages and rebuild:

```bash
uv run python scripts/run_all.py --refresh
```

Data APIs may be rate limited; the downloader retries with backoff and saves each source immediately. Metadata and downloadable CSVs can have different latest months. Complete observed quarters determine the weather panel. Validation rejects partial quarters, time gaps, duplicate observations, overlapping industry selections, incomplete VA downloads and rank-deficient regressions.

## Files

- `outputs/heat_economy.html`: single tabbed entry point, embedding every report, coefficient download, figure and source/method note.
- `outputs/annual.html`, `outputs/annual/`: annual NSA/SA growth, pooled/sector-FE/own-series fits, paired COVID samples, explicit unavailable requests, annual diagnostics and independent verification.
- `data/raw/annual/`, `outputs/annual/reconciliation_summary.json`: official annual-source cache and aggregation checks against annual VA/IIP publications.
- `outputs/granular_va.html`, `outputs/manufacturing_detail.html`, `outputs/granular/`: detailed/aggregate manufacturing analyses, exact samples and independent QR/HAC verification.
- `outputs/reconciliation.html`, `outputs/reconciliation/`: VA identities, taxonomy, actual IIP weights, hierarchy, annual-overlap/two-stage residuals and rounding checks.
- `outputs/sector_cluster_verification.json`, `outputs/sector_cluster_contrasts_verification.json`: independent QR and sector-score CR1 validation, cluster counts and finite-sample factors.
- `outputs/aggregate_va.html`, `outputs/aggregate_va/`: aggregate GVA regressions, exact samples, outcome/controls/COVID comparisons, mean-growth context, diagnostics and independent verification.
- `outputs/sample_context/`: exact-sample mean/SD lookup, pooled/sector growth means, magnitude comparisons and independent raw-level/calendar verification.
- `outputs/stationarity.html`, `outputs/stationarity/`: all-series unit-root evidence, exact model mappings, lag sensitivity and independent numerical/source verification.
- `outputs/granular_interactions/`: main detailed-sector FE slopes, full joint covariance, equality tests, all-pair Holm comparisons and exact samples.
- `outputs/collection/`: overview comparison CSV and a manifest of embedded report sources/hashes.
- `data/raw/`: official JSON/CSV sources, metadata and hash manifest.
- `data/processed/`: monthly/quarterly weather, sector catalog, broad/detailed panels and separately published aggregate real GVA.
- `scripts/`: download, build, estimate, plot/report and independent verification.
- `outputs/first_results.md`: specification, findings, sensitivity, figures, sources and interpretation.
- `outputs/heat_econ_results.html`: portable interactive report, regenerated by the pipeline; its template is `templates/results.html`.
- `outputs/regression_tables.html`: native ETable output for quarterly VA, monthly IIP and quarterly IIP, including robustness columns; HTML and LaTeX fragments are in `outputs/tables/`.
- `outputs/iip_results.html`, `iip_results.md`: monthly/quarterly IIP results, regression tables, charts, methods and official source links.
- `data/processed/iip/`: balanced broad/detailed IIP panels at both frequencies and source reconciliation checks.
- `outputs/iip/monthly/`, `outputs/iip/quarterly/`: regressions, complete baseline samples, cluster slopes, distributed lags, aggregate comparisons and estimation checks.
- `outputs/iip/verification.json`: independent raw-growth, official-quarter-average, period-mean OLS and calendar-aware HAC checks.
- `outputs/extended_controls.html`: wind/PM2.5 comparisons with full coefficients, sample effects, COVID/coverage checks, source notes and coverage download.
- `outputs/extended_controls/`: result CSVs and exact samples for VA, monthly IIP and quarterly IIP, a temperature summary, and independent coefficient/HAC verification.
- `outputs/outcome_comparisons.html`: clearly labelled MoM/QoQ, YoY, log-level and raw-level tables, long/full-control switches, alternative YoY weather definitions and inference sensitivities.
- `outputs/outcomes/`: new coefficient CSVs, exact samples, paired-temperature restriction checks, sample definitions and independent verification; the original growth output files are retained.
- `data/processed/outcomes/`: full processed panels with output transformations and calendar-aligned weather lags, differences and trailing-year means.
- `notes/aea_temperature_literature.md`: four primary AEA studies and the distinction between growth, level and exposure-window specifications.
- `outputs/sector_heterogeneity.html`: interaction/separate-series regression tables, coefficient plots, joint/pairwise tests, relative slopes, endpoint sensitivity and stationarity diagnostics; all data and figures are embedded for offline use.
- `outputs/heterogeneity/`: exact samples, sector slopes, full slope covariances, joint and Holm-adjusted pairwise tests, separate-series weather coefficients, native table/LaTeX fragments, standalone SVG figures and independent verification.
- `outputs/heterogeneity/diagnostics/`: ADF/KPSS tests, paired evidence, autocorrelations, chart-ready tested series, calendar audits, source hashes and independent statistic checks.
- `notes/heterogeneity_review.md`, `notes/stationarity_methods.md`: model equivalences, inference constraints and diagnostic choices/limitations.
- `data/processed/extended_controls/`: monthly and quarterly controls, coverage metrics, station diagnostics and rejected PM2.5 source rows.
- `outputs/regression_results.csv`: temperature and weather coefficients, intervals, p-values, samples and formulas.
- `outputs/all_coefficients.csv`: all estimates, including fixed effects.
- `outputs/distributed_lag_results.csv`, `distributed_lag_sum.json`: timing check.
- `outputs/sector_slopes.csv`: exploratory pooled sector-specific temperature slopes.
- `outputs/baseline_sample.csv`: exact estimation sample.
- `outputs/verification.json`: independent quarterly-mean OLS and manual HAC checks, raw-source hashes and package versions.

Source data are credited to the Singapore Department of Statistics and the National Environment Agency, under the terms linked from the official dataset pages. Chain-linked real VA components need not sum to the published aggregate.
