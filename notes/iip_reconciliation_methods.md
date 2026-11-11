# IIP aggregation, weights and reconciliation

The national manufacturing outcome uses SingStat's **published total IIP**. The six-cluster and 18-industry panels estimate a different, equal-sector association. Adding child indices or applying one fixed weight pattern throughout the history does not reconstruct the official chain-linked total.

## Official method and available inputs

EDB introduced annual Laspeyres chain linking from reference year 2019. Its 2021 information paper explains the loss of ordinary additivity and the linking of earlier fixed-weight history. The current February 2026 memorandum documents the 2025 rebase: CMA 2024 supplies the main information; commodity weights use output and industry/division weights use value added. Published history before 2025 is linked into the new base. The formulas on memorandum pages 4–5 were checked visually because PDF text extraction omitted them. See the [2021 chain-link paper](https://www.edb.gov.sg/content/dam/edb-en/about-edb/media-releases/manufacturing-statistics/chain_linking_of_monthly_index_of_industrial_production/Info-paper-on-Chain-Link-IIP-(2019100).pdf) and [2026 rebase memorandum](https://assets.app.optical.gov.sg/edb/production/published/collections/pages/en%252Fmanufacturing-statistics/db0f05be-8254-45d7-8ff3-2b9fde4f10a6.pdf).

The current metadata identifies SSIC 2025 and a 2025=100 reference base. Its hierarchy contains six clusters with **4, 4, 2, 2, 3 and 3** leaves respectively: 18 mutually exclusive published subclusters. The additional “total excluding biomedical” series overlaps these components and is excluded from the child sets. [Monthly NSA table M355381](https://tablebuilder.singstat.gov.sg/table/TS/M355381), [monthly SA table M355382](https://tablebuilder.singstat.gov.sg/table/TS/M355382).

Official releases recovered for December 2022, January 2023, January 2024, January 2025 and September 2026 provide 125 weight rows: total, six clusters and 18 leaves in each release. Weights are percentages of total manufacturing, displayed to one decimal place. Small child/parent discrepancies of 0.1 percentage point fit the printing precision; normalization is recorded explicitly. These are observed release patterns, without interpolation or backward extension. The current weights appear on page 3 of the [September 2026 release](https://assets.app.optical.gov.sg/edb/production/published/collections/pages/en%252Fmanufacturing-statistics/6766e558-d08e-4e23-a2eb-037c7aa367b4.pdf); historical release PDFs, exact official SGPC download URLs and hashes are retained in the source manifest.

| Cluster | 2022 | 2023 | 2024 | 2025 old vintage | 2026 |
|---|---:|---:|---:|---:|---:|
| Electronics | 40.0 | 45.3 | 47.8 | 37.4 | 40.2 |
| Chemicals | 13.3 | 16.9 | 15.3 | 17.8 | 15.0 |
| Biomedical | 16.9 | 12.3 | 11.0 | 14.1 | 11.3 |
| Precision engineering | 14.8 | 13.1 | 14.1 | 15.8 | 15.2 |
| Transport engineering | 7.2 | 5.8 | 5.8 | 8.2 | 9.6 |
| General manufacturing | 7.9 | 6.6 | 6.0 | 6.6 | 8.7 |

The January 2025 weights belong to the **then-current 2019-based series**. They cannot be treated as the matching weights for the subsequently rebased 2025 observations. A basket change can also require previous-year overlap indices compiled on the new basket. The memorandum separately provides 19 SSIC division-group base-pattern weights for the 2019 and 2025 bases, transcribed into `iip_division_rebase_weights.csv`. Those division groups do not map one-to-one to the six cross-cutting clusters.

[CMA value added table M355181](https://tablebuilder.singstat.gov.sg/table/TS/M355181) supplies census shares for 2000–2025, with 2025 preliminary. They are saved for context and explicitly **not labelled IIP compilation weights**. For example, 2024 census biomedical/general-manufacturing shares are about 11.119%/8.904%, whereas the 2026 IIP release shows 11.3%/8.7%.

## Conditional annual-overlap reconstruction

Let `A_c,y−1` be the previous calendar year's mean published NSA child index. EDB's unchained relative and chaining identities imply, when the effective weights, hierarchy, overlap and vintage match:

```text
U_c,y,m = I_c,y,m / A_c,y−1
I_parent,y,m = A_parent,y−1 × Σ_c w_c,y U_c,y,m
```

Thus extra unpublished linking factors are not automatically required in an ordinary link year: the previous annual published levels supply them. Exact effective compilation weights and a valid overlap remain necessary. Our candidates use displayed weights normalized within their parent; they do not estimate weights from the total.

The signed errors below are candidate minus published total, in index points. The 18-leaf reconstruction first derives six cluster relatives from the leaves and then combines those relatives into total manufacturing.

| Effective year | Six-cluster signed error range | Six-cluster maximum absolute error | 18-leaf two-stage maximum absolute error | Six-cluster months outside rounding bounds |
|---|---:|---:|---:|---:|
| 2022 | −0.005737 to +0.013403 | 0.013403 | 0.028206 | 0/12 |
| 2023 | −0.008848 to +0.019275 | 0.019275 | 0.029167 | 0/12 |
| 2024 | −0.015663 to +0.001622 | 0.015663 | 0.019043 | 0/12 |
| 2025, old/new vintage mismatch | −0.947311 to +0.930837 | 0.947311 | 5.331469 | 12/12 |
| 2026, January–September | −0.002947 to +0.033051 | 0.033051 | 0.027496 | 0/9 |

All **315 parent/month comparisons outside 2025**—total and six cluster parents—and all 45 corresponding two-stage total comparisons are individually compatible with conservative rounding intervals. Each weight can vary by ±0.05 percentage point, subject to the displayed parent-sum bounds; each index cell can vary by ±0.0005. The script evaluates feasible weight-polytope vertices. Two-stage intervals are conservative outer bounds. Passing these per-month checks is not proof that one unique exact weight vector reproduces every month simultaneously.

The known 2025 mismatch is material: the biomedical cluster's maximum absolute error is 27.839444 index points. It remains a labelled failed-vintage diagnostic, rather than a replacement series or an attribution of error to EDB.

## Growth aggregation and fixed-weight diagnostics

Within a link year, arithmetic growth factors can be combined using **previous-period effective volume shares**:

```text
s_c,m−1 = [w_c,y × I_c,m−1/A_c,y−1] / Σ_j[w_j,y × I_j,m−1/A_j,y−1]
F_parent,m = Σ_c s_c,m−1 × (I_c,m/I_c,m−1)
log growth = 100 ln(F_parent,m)
```

This is not a weighted average of child log growth. The same identity applies to consecutive quarter means within a calendar year. January/Q1 crossings are deliberately excluded from these within-year growth checks. Saved derived shares are labelled calculated quantities from rounded inputs, not official published weights.

For the 417 common months from January 1992 to September 2026, the naive equal average of six NSA cluster indices has mean/max absolute deviations of 12.826/24.289 index points from official total. A fixed application of the current **2025-base displayed release weights** has deviations of 7.370/16.748. Both are diagnostics, with no claim that the fixed pattern is historically valid. Raw sums are also saved and are plainly inappropriate for indices individually normalized to 100. SA comparisons are retained separately; additivity is not imposed after seasonal adjustment. DOS's still-linked [2006 seasonal-adjustment paper](https://www.singstat.gov.sg/files/0937df16-76d1-405e-ab24-d31aceea7055.pdf), Appendix II(E), documents direct adjustment of overall IIP. Its date is acknowledged rather than assuming every current component uses an unchanged procedure.

## Reproducibility and remaining limits

Run `run_iip_reconciliation()` from `scripts/iip_reconciliation.py`, or `.venv/bin/python scripts/iip_reconciliation.py`. It uses cached sources; `--fetch` explicitly refreshes the listed sources. Main outputs under `outputs/reconciliation/` are the `iip_reconciliation_summary.json`, weight/hierarchy tables, complete parent comparison residuals, annual-overlap and two-stage tables, within-year growth checks, and derived volume shares. Source URLs and SHA256 hashes are in `data/raw/reconciliation/iip_source_manifest.json`.

Independent scalar calculations agree with the vectorized fixed-weight and annual-overlap arithmetic within 5.7×10⁻¹⁴ index points. Original NSA/SA cache hashes are checked against the original source manifest. The existing independently checked quarterly audit is reused: 3778 published NSA quarter cells agree with three-month means within 0.001667 index point. This verifies time aggregation, not cross-sector additivity.

The project does not have a complete matching history of exact compilation weights and base-change overlap inputs. It therefore uses official published total manufacturing for the nationally weighted regression outcome, and reports six-cluster/18-industry panels separately. No historical weights are made up, no census shares silently substituted, and no fitted weights used to claim official reconciliation.
