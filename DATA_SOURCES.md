# Data sources and frozen inputs

This package uses public official statistics and derived analytical inputs. The source manifests record the URLs, source/retrieval dates where available, sizes and SHA-256 hashes. The release manifest identifies the exact delivered bytes. A live download can reflect a different revision and is not a substitute for the frozen release input.

## Official economic series

| Data | Source IDs and links | Cached location |
|---|---|---|
| Quarterly real GDP/VA by industry, NSA and SA | SingStat [M015661](https://tablebuilder.singstat.gov.sg/table/TS/M015661) / [M015662](https://tablebuilder.singstat.gov.sg/table/TS/M015662) | `data/raw/va_{nsa,sa}_*.json` |
| Monthly manufacturing IIP, NSA and SA | SingStat/EDB [M355381](https://tablebuilder.singstat.gov.sg/table/TS/M355381) / [M355382](https://tablebuilder.singstat.gov.sg/table/TS/M355382) | `data/raw/iip_{nsa,sa}_*.json` |
| Published quarterly NSA IIP aggregation check | SingStat [M355391](https://tablebuilder.singstat.gov.sg/table/TS/M355391) | `data/raw/iip_quarterly_nsa_*.json` |
| Official annual NSA output checks | SingStat [M015721](https://tablebuilder.singstat.gov.sg/table/TS/M015721), [M355401](https://tablebuilder.singstat.gov.sg/table/TS/M355401) | `data/raw/annual/` |
| Current-price VA accounting checks | SingStat [M015651](https://tablebuilder.singstat.gov.sg/table/TS/M015651) | `data/raw/va_reconciliation/` |
| IIP weights, chain-linking and classifications | [EDB manufacturing statistics](https://www.edb.gov.sg/en/manufacturing-statistics), official releases/methods and SingStat industry/census tables | `data/raw/reconciliation/` |

VA is in chained-2015 Singapore dollars, with source unit metadata retained. IIP is a volume index with 2025=100. The official aggregate GVA series and published manufacturing total are modeled separately from equally represented sector panels. Chain-linked industry volumes need not add to their published aggregate. The reconciliation notes identify recovered weight years and unresolved vintage/historical-weight limitations.

The main VA cache's source update is 11 August 2026; the monthly IIP cache's source update is 26 October 2026. Annual publications can have different update dates. Use the recorded metadata for each file, not a single assumed common vintage.

## Weather and added controls

| Data | Provider and dataset | Construction/input boundary |
|---|---|---|
| Monthly mean air temperature | NEA, Changi Climate Station; [dataset d_755290a24afe70c8f9e8bcbf9f251573](https://data.gov.sg/datasets/d_755290a24afe70c8f9e8bcbf9f251573/view) | Official monthly mean in °C; `data/raw/temperature_monthly.csv` |
| Monthly rainfall total | NEA; [dataset d_b16d06b83473fdfcc92ed9d37b66ba58](https://data.gov.sg/datasets/d_b16d06b83473fdfcc92ed9d37b66ba58/view) | Monthly total in mm; `data/raw/rainfall_monthly.csv` |
| Monthly mean relative humidity | NEA; [dataset d_e3b1111a26b2aefef4f386c068d4ca5d](https://data.gov.sg/datasets/d_e3b1111a26b2aefef4f386c068d4ca5d/view) | Monthly mean in percent; `data/raw/humidity_monthly.csv` |
| Wind | Derived NEA station-averaged half-hour cache from the earlier electricity project; [upstream dataset d_7677738484067741bf3b56ab5d69c7e9](https://data.gov.sg/datasets/d_7677738484067741bf3b56ab5d69c7e9/view) | Canonical 2016–2025 half-hour means, active-station counts and quality report in `data/raw/extended_controls/`; input knots are converted using 1852/3600 to m/s |
| Hourly PM2.5 | NEA; [dataset d_6c0d5fc34838b12472475fdb73c0af29](https://data.gov.sg/datasets/d_6c0d5fc34838b12472475fdb73c0af29/view) | Original hourly five-region bulk CSV and metadata in `data/raw/extended_controls/`; regional mean, hour-ending-to-start alignment and cleaning are explicit |

The wind input is a **derived canonical cache**, not a fresh dump of every original station reading. Its upstream provenance, station-count diagnostics and file hashes are retained. Reproduction starts from that bundled cache and does not reharvest the earlier project's station data or require its machine paths. PM2.5 uses the official hourly bulk input; annual PM2.5 is not substituted for monthly measurements.

Quarterly temperature/humidity means weight months by calendar days; quarterly rainfall sums months. Monthly wind/PM2.5 eligibility requires 90% observed temporal coverage for each variable, with an 80% sensitivity; the quarterly rule requires every constituent month to pass. Annual controls use a separate stated rule: twelve observed months and at least 90% annual expected reading counts for both variables, with annual means weighted by retained observation counts. Neither rule imputes missing readings. See the extended-control and annual method notes for exact cleaning, coverage and units.

## Visitor arrivals and calendar control

International visitor arrivals come from Singapore Tourism Board data distributed through data.gov.sg, with SingStat metadata:

- NSA arrivals: [d_d1c33009b674dcf70b8e8c790b793f28](https://data.gov.sg/datasets/d_d1c33009b674dcf70b8e8c790b793f28/view), [M550001](https://tablebuilder.singstat.gov.sg/table/TS/M550001).
- SA arrivals: [d_c6d3f73a65d1fa3a92f1f4e70e619a66](https://data.gov.sg/datasets/d_c6d3f73a65d1fa3a92f1f4e70e619a66/view), [M550002](https://tablebuilder.singstat.gov.sg/table/TS/M550002).

The cached arrival counts and metadata are in `data/raw/visitor_arrivals/`. The analysis checks the aggregate, region and country hierarchy and keeps SA/NSA definitions and zero-count treatments explicit.

The Lunar New Year control derives from Hong Kong Observatory's official [Gregorian–Lunar calendar conversion tables](https://www.hko.gov.hk/en/gts/time/conversion.htm), with yearly URLs/hashes in `data/raw/visitor_arrivals/calendar/manifest.json`. Cached text tables cover 2008–2026. The original Python extractor verifies each complete Gregorian calendar, extracts the first day of the first lunar month and allocates a ±7-day window to months. The HKO material has its own conditions; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Provenance and audit files to retain

The relevant input manifests include:

- `data/raw/manifest.json`
- `data/raw/extended_controls/manifest.json`
- `data/raw/va_reconciliation/manifest.json`
- `data/raw/reconciliation/iip_source_manifest.json`
- `data/raw/annual/manifest.json`
- `data/raw/visitor_arrivals/manifest.json`
- `data/raw/visitor_arrivals/calendar/manifest.json`

The release includes `data/processed/`, both `data/reference_before_pyfixest/` and `data/reference_before_nsa_inputs/`, and the output-side frozen migration/paired input references needed by independent verifiers. These reference files are audit inputs, not disposable intermediates. Native fit logs and estimator manifests document which actual models were fit; independent verification receipts have a separate scope.

The sector-clustered interaction sensitivity introduces no new statistical source. It reuses the exact parent common samples, records their paths and hashes, and keeps its added covariance results and native ledger separate from the original calendar-DK analyses.

Historical provenance can name source-machine locations from the earlier electricity project. Those strings are retained where they are part of recorded history or hashed references. The analytical code uses the bundled project-relative files. A source manifest's historical path is not a requirement to access that workstation.

## Literature boundary

The release retains small citation/provenance manifests and the project's methods notes. Third-party academic full-text PDFs, extracted paper text, the AEA replication ZIP and third-party Stata do-files are excluded from the distributable archives. They are not runtime dependencies of the numerical analyses. The literature notes link to their primary sources and distinguish published articles from inspected working-paper versions; empirical results from those papers were not reproduced by this project.

Some immutable historical input-reference JSON files also record the hashes of the earlier literature inspection's `maketable2.do`, `maketable5.do` and `replication.zip`. Those references are retained as historical provenance; the three third-party contents are not redistributed. The reference checks distinguish these explicitly named provenance-only omissions from analytical inputs and report that their omitted contents were not reverified. This is not permission to omit any statistical input, reference sample, source manifest or analytical code, and the original reference JSON files are not reset.

Dataset reuse remains subject to each provider's terms and attribution requirements. These data are not placed under an invented common licence, and no official agency endorses the project's estimates or transformations. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
