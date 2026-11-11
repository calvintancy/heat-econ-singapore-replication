# Sector-to-aggregate VA reconciliation

Implemented 11 November 2026. Entry point: `scripts.va_reconciliation.run_va_reconciliation()`. This audit adds source checks and descriptive outputs. It does not change any regression, source file used by earlier regressions, or sector weight.

## Sources and category coverage

The two existing real series are [NSA M015661](https://tablebuilder.singstat.gov.sg/table/TS/M015661) and [SA M015662](https://tablebuilder.singstat.gov.sg/table/TS/M015662). The newly cached additive comparison is [current-price NSA M015651](https://tablebuilder.singstat.gov.sg/table/TS/M015651). All three use SSIC 2020, report S$ million, contain 29 series and 4,534 cells, and cover 1975Q1–2026Q2. All three source updates are **11 August 2026**, so this comparison does not mix vintages. The real tables express volumes in chained-2015 dollars; the current-price table is nominal.

Basic-price GVA is official row `1.4`. The existing 11-sector panel comprises four goods sectors (`1.1.1`–`1.1.4`) and seven broad service sectors (`1.2.1`–`1.2.7`). **Ownership of dwellings (`1.3`) is an additional component outside that panel.** Its output includes the housing service provided by owner-occupied homes, as explained in the [official quarterly GDP methods paper](https://www.singstat.gov.sg/files/3cb0efc7-28d4-43b1-ad97-3ff03907c585.pdf), paragraphs 3–4. GDP at market prices is row `1`; taxes on products are separately reported as `1.5`, outside basic-price GVA.

The script checks 11 distinct identities: the incomplete 11-sector sum; the complete 11-sector sum plus dwellings; goods plus services plus dwellings; four goods sectors against the goods parent; seven broad service sectors against the services parent; GVA plus taxes against GDP; the 19 detailed leaves plus dwellings; and the four split-parent identities. Parent and child rows are never included in the same component sum.

## What reconciles

The residual is always **sum of listed components minus published total**. No balancing item is inserted and no component is rescaled. These are the latest, 2026Q2, results in S$ million:

| Comparison | Real NSA | Real SA | Current-price NSA |
| --- | ---: | ---: | ---: |
| 11 sectors + dwellings − GVA | −449.5 | −225.6 | 0.0 |
| Goods + services + dwellings − GVA | +265.4 | +461.1 | 0.0 |
| 19 detailed sectors + dwellings − GVA | −459.2 | −241.0 | 0.0 |
| GVA + taxes − GDP | −196.4 | −248.0 | 0.0 |

The 11 sectors alone fall short of latest official GVA by S$6,566.0m NSA and S$6,342.1m SA. Adding dwellings corrects that coverage omission; it does not remove the chain-volume residual.

Every complete current-price identity reconciles within reported rounding on every available quarter. The maximum absolute quarterly residual is S$0.1m. For the complete broad real sum, the largest absolute historical residual is S$1,426.2m NSA and S$1,416.4m SA. The latest broad residual is −0.3048% of official NSA GVA and −0.1524% of official SA GVA. `va_identity_summary.csv` records all identities, extrema and dates; `va_quarterly.csv` preserves the full history.

## Why real levels do not add

The [2019 national-accounts benchmarking paper](https://www.singstat.gov.sg/files/6448c773-d7c9-4798-9ac7-68ba46e19efb.pdf), methodological appendix, printed pages 15–18, explains annual reweighting and chain-linking. Relative price weights change each year; chained component levels generally do not sum to the separately chained aggregate. The reference year sets the scale of the volume series, rather than imposing fixed 2015 weights throughout history. The paper identifies additivity in the reference year and immediately following year. The empirical check here applies that annual property at **annual NSA frequency**.

All 29 NSA real series equal their corresponding current-price values in annual 2015 sums, exactly at the published precision. The complete broad NSA annual component residual is S$0.0m in 2015 and +S$0.1m in 2016. Quarterly 2015 broad residuals are +68.1, +29.1, −31.2 and −66.0m: quarterly additivity must not be inferred from the annual reference-year identity. `va_reference_year_levels.csv` provides all quarterly and annual 2015 real/nominal comparisons.

The same 2019 paper, printed page 18, specifies **direct seasonal adjustment for all real national-account series from ESS 2019Q1**. Each real aggregate is adjusted in its own right. The older [2006 seasonal-adjustment paper](https://www.singstat.gov.sg/files/0937df16-76d1-405e-ab24-d31aceea7055.pdf), paragraphs 18–22, explains direct versus indirect adjustment: indirect adjustment constructs the adjusted aggregate from adjusted components, whereas direct adjustment estimates the aggregate separately. Its historical description of indirect GDP adjustment predates the chain-volume change. SA component sums therefore should not be forced to match. In annual 2015 the broad SA residual is +S$83.7m, despite the NSA annual identity.

The residual records the discrepancy from arithmetic summation. It is not an omitted industry, an economic loss or a measurement-error estimate. The script does not attempt to attribute each quarter’s residual quantitatively between chaining and seasonal adjustment.

## Detailed taxonomy and balanced coverage

The 19-sector variant replaces four broad service categories with their published children:

| Broad category replaced | Detailed categories |
| --- | --- |
| Wholesale & Retail Trade | Wholesale Trade; Retail Trade |
| Accommodation & Food Services | Accommodation; Food & Beverage Services |
| Real Estate, Professional Services And Administrative & Support Services | Real Estate; Professional Services; Administrative & Support Services |
| Other Services Industries | Public Administration & Defence; Education; Health & Social Services; Arts, Entertainment & Recreation; Other Services - Others |

The remaining seven broad sectors are retained unchanged: Manufacturing, Construction, Utilities, Other Goods Industries, Transportation & Storage, Information & Communications, and Finance & Insurance. The 12 split leaves begin in **2005Q1** in both SA and NSA source tables; the seven unsplit series have earlier histories. The balanced detailed panel therefore has 86 level dates from 2005Q1, 85 QoQ dates from 2005Q2, and 82 YoY dates from 2006Q1 through 2026Q2. A common comparison of levels, QoQ and YoY begins in 2006Q1. Dwellings remain outside both sector panels and must be added separately for a category-complete national comparison. `va_taxonomy.csv` supplies source-specific row IDs, names, broad-parent mapping and first valid dates.

## Shares, contributions and growth interpretation

`va_annual_current_price_shares.csv` reports each broad/detailed component’s share of annual nominal GVA, including dwellings. Only complete calendar years are used. These are economic composition shares; they are **not fixed aggregation weights for chained real levels**.

`va_annual_nominal_growth_contributions.csv` reports `100 × (nominal_VA[i,t] − nominal_VA[i,t−1]) / nominal_GVA[t−1]`. Complete components sum to official **nominal** annual growth up to source rounding. They must not be relabeled contributions to real growth. This audit does not infer exact real-growth contributions by multiplying sectoral real growth by these nominal shares.

`va_growth_comparison.csv` contrasts published aggregate quarterly log growth, growth of the diagnostic sum of real component levels, and the equal-weight 11-sector mean of log growth. Only the first measures official national GVA growth. In the latest SA quarter, growth of the component-level sum differs from official aggregate growth by +0.5913 log-growth points; the equal-sector mean differs by −0.8317 points. The earlier panel regression estimates an equally weighted sector response, while the national aggregate regression uses the published aggregate and includes dwellings. The difference is substantive and should remain explicit in labels.

## Reproduction and verification

`data/raw/va_reconciliation/manifest.json` records new-source URLs, retrieval timestamps, byte counts, hashes and the API page list. The new current-price download uses the metadata cell total and a pagination loop; one 5,000-cell page supplies all 4,534 declared cells. The original real-table hashes are checked against the existing manifest before and after the audit. All rows are checked for duplicate periods, contiguous available history, metadata row coverage, declared cell counts and matching data/metadata vintages. The primary methodological PDFs are cached and hashed too.

For independent arithmetic verification, a second pass re-reads raw JSON into Decimal values and uses explicit loops, independently of the DataFrame reconciliation sums. It checks **4,998 quarterly identities, 1,233 annual identities, 1,032 nominal share cells, 1,000 nominal contribution cells, 29 reference-year series and 410 growth-comparison rows**. All pass. The largest arithmetic difference is below 1.9×10⁻¹⁰ S$ million. Complete nominal identities and annual NSA 2015/2016 identities are checked against conservative rounding bounds: each independently rounded one-decimal input can contribute at most S$0.05m, with four quarters’ bounds added for annual sums. Gaps are preserved rather than rounded away in the saved data.

Machine-readable results are `outputs/reconciliation/va_summary.json` and `va_verification.json`. The chart is `va_reconciliation.png`. Source IDs, definitions and all output paths are included in the summary so the consolidated report can render this audit without recomputing any regression.
