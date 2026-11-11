# Annual temperature and real-output growth specifications

These annual specifications add a frequency comparison alongside the retained quarterly and monthly analyses. They follow the annual log-growth/temperature-level setup discussed in Dell, Jones and Olken (2012), while adapting the outcome, controls and fixed effects to one country. They are **not a replication of that paper's cross-country identification or uncertainty calculation**. The paper uses annual real GDP-per-capita growth and country/region-time effects; this project uses Singapore real VA or industrial production, without a per-capita adjustment. See the verified source and equation notes in `aea_temperature_literature.md`.

## Annual output and exposure construction

For each official series and calendar year, annual VA is the sum of four complete quarterly flow observations. Annual IIP is the arithmetic mean of twelve complete monthly index observations. The official aggregate GVA and published IIP total remain their own series; industry panels do not replace these aggregates with an equal average or a sum of chain-linked sector indices. Sector outputs are aggregated over time within each series, not across industries.

Annual output is constructed separately from the cached official NSA and SA levels. NSA annual growth is the primary annual outcome, with the SA construction retained as a companion. Each is `100 × [log(annual output[y]) − log(annual output[y−1])]`, using the exact preceding calendar year. Both years must have all four VA quarters or twelve IIP months. Growth is calculated before weather eligibility and before the 2020–2022 endpoint exclusion. Incomplete 2026 output/weather years are not annual observations.

Annual air temperature and relative humidity weight the twelve official monthly means by calendar days; annual rainfall sums monthly totals and is expressed in 100 mm. A complete twelve-month baseline-weather year is required. Wind and PM2.5 annual means weight monthly observed-reading means by their actual retained half-hour/hour counts. Their annual coverage uses the full calendar year's expected counts, including leap years. All twelve months must have positive observed counts for both variables, and each variable must reach 90% annual coverage. This annual coverage rule is stated separately from the quarterly/monthly rule that requires every constituent month to pass its own threshold. Missing readings are not imputed.

The controlled annual window currently has nine eligible years, 2017–2025. Excluding observations dated 2020–2022 leaves six eligible years. This exclusion is applied after annual growth and weather construction; 2023 annual growth still compares with 2022 output. Actual year indices remain uncompressed in covariance calculations.

## Identification and model families

The common-slope model is

```text
annual log growth[s,y] = sector FE[s] + beta × annual temperature[y]
                        + annual weather controls[y] + linear trend[y] + error[s,y]
```

Full common year effects are omitted because every sector has the same national annual temperature in a given year: year effects would absorb its common coefficient exactly. The linear trend and weather controls are explicit modeling choices, not substitutes for all unobserved national shocks. There are no seasonal indicators in a calendar-annual specification.

The six economic contexts are official aggregate GVA, 11 broad VA sectors, 19 detailed VA sectors, official total IIP, six manufacturing clusters and 18 detailed manufacturing industries. A sector-interaction model estimates a main temperature coefficient plus nonreference interactions, with shared annual controls and trend. Its reported absolute sector slope is the main coefficient plus the interaction; inference includes their covariance cross term. Separate own-series trend/weather regressions provide a distinct robustness comparison and use their own parameter counts and covariance correction.

Each context retains baseline weather and the full wind/PM2.5 control set, both output adjustments and the full/excluded samples. Unavailable specifications remain in the inventory with their intended exact samples, means and an explicit reason. They are not silently replaced by a smaller control set.

## Inference and short-sample limits

The main pooled common-slope panel uncertainty remains sector CR1 with `t(S−1)`. National single series and industry-specific slopes use actual-calendar Bartlett HAC, with annual bandwidths one and two. Joint panel score covariance uses `T/(T−1) × (N−1)/(N−K)`; own-series covariance uses `T/(T−K)`. Every reported sector effect tests its one total linear combination. Equal-slopes omnibus tests leave the common temperature coefficient unrestricted; singular restriction covariance gives an unavailable joint p-value. Pairwise inference uses joint covariance and a separately stated Holm family.

Repeated sectors do not add distinct annual weather observations. Rank must therefore be checked on the full design and on the time-varying common regressors, not inferred from the larger stacked row count. The nine-year full-control model has only two effective time-series residual degrees of freedom. Its six-year excluded counterpart cannot identify an intercept, trend and five weather coefficients and is explicitly unavailable. Even available short-window fits are fragile; a nominal p-value is not reliable evidence of a stable causal effect.

Stationarity diagnostics require at least twenty contiguous annual observations. Gapped exclusions are not compressed into consecutive years for standard ADF/KPSS tests. Short or gapped annual histories remain unavailable for these screens. Unit-root diagnostics are marginal, finite-sample evidence and do not establish residual stationarity, cointegration, absence of breaks or causal identification.

## Independent verification

`verify_annual.py` reconstructs annual SA/NSA output and baseline weather directly from the cached official raw sources. It independently checks observed-count-weighted wind/PM2.5 aggregation and annual coverage, exact prior-year growth, sector/hierarchy coverage, all sample masks and unavailable specifications. Regression benchmarks use full-model QR and explicit CR1 or calendar score sums, independently of the production native-Pyfixest adapter. Coefficients, full joint slope/component covariance, confidence intervals, p-values, R², means, parameter ranks, joint-test availability and pairwise multiplicity are checked against saved outputs. The isolated annual native receipt is reconciled with the actual estimable model/outcome/unit inventory. Numerical validation preserves the annual model's identification and small-sample limits.

<!-- completed-annual-audit:begin -->
## Completed independent audit

All 512 requested annual specifications were accounted for: 384 native Pyfixest models and 128 explicit unavailable requests. The latter are the requested six-year full-control exclusions; controls were retained. All 98,644 coefficient covariance entries, 10,104 joint sector covariance entries and 4,728 pairwise/Holm rows passed independent checks. Of the available-model equal-slopes tests, 30 are full rank and 18 correctly have no p-value; 8 additional joint requests belong to underidentified models.

Raw annual reconstruction and official annual NSA publications agree across all 2,044 matched sector-year cells within the stated publication-precision envelopes (1,119 VA and 925 IIP cells). The published annual cache and higher-frequency source vintages are recorded separately. Annual diagnostics include 260 independently recomputed ADF/KPSS tests; 828 short, gapped or constant-series test requests correctly remain unavailable.

The annual covariance implementation uses native orthonormal-coordinate coefficient influences before summing scores. This avoids cancellation in the raw-design bread–meat–bread multiplication while preserving the fitted model, calendar lags, finite-sample multipliers and null hypotheses. Independent full-QR covariance checks pass at the original numerical tolerance.
<!-- completed-annual-audit:end -->
