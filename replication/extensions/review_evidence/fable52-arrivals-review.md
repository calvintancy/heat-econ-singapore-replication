# Referee audit: frozen visitor-arrivals numeric companions

**Verdict.** I found no arithmetic bug. Every count, sample, design, coefficient, covariance, p-value and label I rebuilt matches the saved outputs to numerical precision. The concerns are inferential and about labelling:

- Joint Wald p-values cannot be read as evidence in either direction once q reaches about 8 (the smallest I tested), and the `joint_test_fragile` flag does not say so.
- Origin-cluster rows for common slopes cannot see the error component that a common weather series competes with.
- The calendar-HAC t reference over-rejects by a design-dependent margin.

Under the primary method the grid shows no excess of pointwise rejections. So none of this overturns a null reading; it removes support for any positive one.

## 1. Scope, coverage and source generation

**Snapshot.** Worktree `dist/aea_audit_20261111/fable-arrivals`, git HEAD `e47edeefeb750fd8a64106658d9351783db6e7fa` plus the uncommitted overlay. The parent scope file records `snapshot_manifest_sha256 77b47a53ec17fa3dd01bd61093604d2b367122c5c48441b7efde0b9a888b2cd4` (2,250 files). The sources I read (full list in `audit_probes/probe01_result.json`):

| File | SHA-256 |
|---|---|
| `scripts/visitor_rigour_monthly.py` | `7798c0a47c65740b801ce530700cd014bc24fef759a87cbea6f83903ca70757b` |
| `scripts/visitor_rigour_companions.py` | `f8a5960628acb4e74f89c3ecb822665db9c040600c077dd5f813013a7d8f1712` |
| `scripts/visitor_origin_year.py` | `384eef7f3a990d8d35b006b51adc7d1d6815e265ffd14a300b8e8d60d3970aa6` |
| `scripts/literature_full_controls.py` (Wald) | `c075ad3cae37bc02aa63521ead29b8b47891a616483ce336125561609eeda8b2` |
| `scripts/literature_sensitivity.py` (calendar HAC) | `95d4d2fdd3b95a0dc0dc30875d50b314ab2d53b3774746d29bf9e8dcf5cbdca7` |
| `notes/visitor_rigour_protocol.md` | `5ebd154176a6b5343d4b3d86b38469fb55ee0b83a3ce816ab5d3334c361ba9bc` |
| `notes/visitor_origin_year_protocol.md` | `dff6a24028e160055a4d50e2f6f7a5e432914daa2df163d9871e95581d8e9d9b` |
| `data/raw/visitor_arrivals/nsa.csv` | `3d8d498dce66a70722c883a6cfa5b2552e27dfd0fac849d5c152bae9b22a4202` |

**Counts (Probe 01).** The stated scope is confirmed from the ledgers, not assumed.

| Family | Requests | Native fits | Not estimated |
|---|---|---|---|
| Monthly | 270 | 260 Feols | 10 rank-unavailable (all YoY, full controls, strict mask, distributed or dynamic; T = K_time = 44) |
| Annual | 30 | 15 Feols | 15 full-control requests (T = 5, 3, 3 eligible years; wind and PM2.5 retained) |
| PPML | 30 | 20 Fepois | 10 strict-mask requests recorded as explicit aliases of event-excluded fits |
| Origin×year | 24 | 24 Feols | none |
| Total | 354 | 319 | |

All 319 design archives bind to the native fit logs by outcome, coefficient and residual hash with no mismatch. The engine is pyfixest 0.70.0, with no dropped rows or coefficients.

**Probes.** Each runs as `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit_probes/<script>` (single BLAS thread, set in `_probe_common.py`) and writes `audit_probes/probeNN_result.json`. Below, "P04" means Probe 04.

| # | Script | Checks | Outcome |
|---|---|---|---|
| 1 | `probe01_scope_counts_source_generation.py` | counts, ledger binding, pins, hashes | 354 / 319 confirmed |
| 2 | `probe02_growth_ar_masks.py` | raw-count growth, AR history, masks for all 72 samples | exact |
| 3 | `probe03_lny_calendar_metadata.py` | HKO-2007 supplement plus anchors, 228 months | exact; Dec 2007 = 0 |
| 4 | `probe04_h1_origin_year_design.py` | H1 and origin×year designs, QR/FWL, cross-term variance, 24 parent equalities | exact |
| 5 | `probe05_time_rank_reference_law.py` | K_time, span rank, correction for 294 archives; exact null laws for thin designs | 0 mismatches |
| 6 | `probe06_calendar_hac_gaps.py` | Bartlett HAC 12/24 with zero-score gap bins; alternative scalings | ≤ 4.2e-15 |
| 7 | `probe07_annual_reconstruction.py` | annual sums, day weights, masks, 30 statuses | exact |
| 8 | `probe08_ppml_zero_retention_irls.py` | receipts, zero retention, independent IRLS, score/bread/influence | β within 6e-16 |
| 9 | `probe09_ppml_partition_identities.py` | all 8 partition identities | ≤ 2e-11 |
| 10 | `probe10_claim_layer_origin_year_reporting.py` | labels, multiplicity, cluster rows, joint tests, diagnostics | see §3 |
| 11 | `probe11_selected_reference_law_benchmarks.py` | fixed-design null laws: five joint-test designs, six YoY designs | see §3 |
| 12 | `probe12_origin_cluster_common_shock_blindness.py` | cluster identity on 110 common designs | see §3 |

Three unnumbered read-only tabulations (`_check_cluster_counts.py`, `_check_holm_families.py`, `_show_p11_p12_detail.py`) reconcile row counts. I wrote only under `audit_probes/`: no file outside it is newer than the parent scope file, and no bytecode was created. I ran no native estimation, producer, verifier grid, install or fetch.

**Benchmark laws.** Where I quote a "benchmark null rejection", I hold the saved design fixed and draw Gaussian errors that are independent across origins. They are iid over months, or MA(11) for the mechanical overlap of a 12-month log difference. The null law of the saved statistic is then exact for t-ratios (P05, P11) and simulated for joint tests (4,000–20,000 draws). These are favourable reference laws for selected designs, not size estimates for the data.

## 2. Accepted actual bugs

None. What reproduces:

- **Samples (P02).** Growth and AR terms are computed on the raw calendar before masks. All 5,400 mask rows reproduce, including strict t−12..t for own-current designs and t−4..t (MoM) or t−24..t (YoY) for matched max-history designs. There is no +1, interpolation or unit exclusion: a zero count in any market drops that month from every cut (17 MoM and 24 YoY months), leaving T = 207 and 189.
- **Calendar (P03).** All 228 cached monthly holiday values agree with the reconstruction and Dec 2007 = 0. The 42 January-2008 PPML rows are retained with `lny_lag1 = 0`. Raw metadata hashes match.
- **Designs and estimates (P04, P06).** Independent QR reproduces native coefficients and residuals (≤ 3e-12). Total origin slope equals base plus deviation with the full cross-term variance. All 24 origin×year samples and stochastic columns are bitwise equal to their parents; only the deterministic block differs. Deviation rows are contrasts against the first origin, so their individual p-values depend on that reference (the joint equality test does not), and they test a different null from the total-effect rows.
- **Time rank (P05).** K_time, span rank, time df, correction and cluster factor reproduce for all 294 monthly and origin×year archives. Native point availability and temporal-correction availability are recorded separately (`scripts/visitor_rigour_monthly.py:337`).
- **HAC (P06).** Full HAC 12/24 matrices match to 4.2e-15 with missing dates kept as zero-score bins. A compressed-gap version differs by 2.1–45.6 %, so production does not lag across compressed gaps.
- **Annual (P07).** Twelve-month sums for 2008–2025, day-weighted weather, 17 growth years (2009–2025) and all 30 statuses reproduce.
- **PPML (P08, P09).** All 516 zero cells are retained and no rows or units are dropped. An independent IRLS matches β to 6e-16 and the score, bread, influence and HAC layers to about 1e-12. The only `+1` near counts is a diagnostic denominator (`scripts/visitor_rigour_companions.py:390`). All 8 partition identities hold to 2e-11.
- **Labels (P10).** `mean_y` and `sd_y` are exact for all 354 specifications; PPML unit conversions are exact on 84 rows.

## 3. Qualified design and inference concerns

### 3A. New discrepancies (none changes a saved number)

**N1. `joint_test_fragile = False` does not mean the nominal p is usable.**

- *Source.* `scripts/literature_full_controls.py:149-151,158`; `scripts/visitor_rigour_companions.py:142-152`. Tests are built at `scripts/visitor_rigour_monthly.py:436-442`, `scripts/visitor_origin_year.py:195-199` and `scripts/visitor_rigour_companions.py:271`.
- *Intended.* `notes/visitor_rigour_protocol.md:108-110`: "high-dimensional or ill-conditioned computable tests remain visibly fragile". Every multi-restriction row is also labelled "Not established; nominal asymptotic diagnostic" (`scripts/visitor_rigour_monthly.py:287`).
- *Observed (P11, P10).* The flag fires only when q ≥ 0.8·(T−1), the condition number exceeds 1e8, or time df < 10. None of the rows below is flagged.

| Saved test | q, T | Saved F (nominal p) | Benchmark null rejection at nominal 5 % (main / wider bandwidth) | Share of null draws ≥ saved F |
|---|---|---|---|---|
| Annual region H1, slopes equal | 8, 17 | 10.96 (3.4e-5) | 0.79 / 0.88 | 0.24 |
| Region MoM H1, effects equal | 8, 207 | 1.41 (0.19) | 0.26 / 0.49 | 0.47 |
| Country MoM H1, effects equal | 41, 207 | 2.39 (3.3e-5) | 1.00 / 1.00 | 0.9995 |
| Country YoY two-channel H1, all zero | 84, 161 | 116.2 (7e-112) | 1.00 / 1.00 | iid 0.000; overlap 1.000 |
| Country YoY origin×year H1, effects equal | 41, 189 | 24.25 (1.3e-55) | 1.00 / 1.00 | iid 0.012; overlap 0.37 |

- *Ledger context.* In the monthly ledger, 2,954 of 8,008 multi-restriction rows have p < .05: 1,030 of 3,954 (26.0 %) at HAC-12, 1,842 of 3,954 (46.6 %) at HAC-24, and 82 of 100 under origin clustering. Of these, 2,848 are flagged not fragile, 1,466 of them with p < .001.
- The two monthly HAC shares are no higher than what the q = 8 test on a benign 207-month design produces with no effect at all (26 % and 49 %).
- In the origin×year ledger 61 of 88 rows reject and none is flagged. All 12 computable annual region joint tests reject, while 2 of 153 annual origin slopes do and none after Holm.
- *Consequence.* No joint p-value in these ledgers is evidence for or against heterogeneity or a mechanism. A nominal p of 3e-5 can lie below the 5th percentile of a benign null law (country MoM), and 1e-55 can lie inside the bulk of another (origin×year under overlap).
- The wider bandwidth rejects more because it leaves fewer effective degrees of freedom, not because the evidence is stronger. A rough Bartlett heuristic, 1.5·span/(L+1), gives about 26 at 12 months and 13 at 24, against the 206 used as denominator df.
- I am not proposing a q ≤ time-df guard: the failure is already large at q = 8 with 172 time df. The gap is that the flag measures rank proximity while its name invites the reading "not fragile, so usable".
- The limitation is disclosed in kind (`notes/visitor_origin_year_protocol.md:31-32`, `templates/visitor_rigour.html:6`). Its size for arrival designs and the flag's silence are new.

**N2. Origin-cluster rows for common slopes are exactly blind to common date shocks.**

- *Source.* `scripts/visitor_rigour_monthly.py:364-369,382`; `scripts/visitor_rigour_companions.py:240-251,393`.
- *Intended.* "A labelled pooled comparison under independent-origin assumptions" that "cannot create independent Singapore-weather realizations" (`notes/visitor_rigour_protocol.md:97-100`), with `main_inference = False`.
- *Observed (P12).* In all 88 static balanced common designs with more than one origin (82 monthly, 6 annual), the regression weight on date t is the same for every origin (spread ≤ 1.3e-13). The slope's estimation error therefore loads on date totals of the errors. Each origin's score contains only its residual's deviation from the date mean (≤ 3.7e-14; the rebuilt covariance equals the archive to 2.0e-15). A shock common to all origins on a date moves the estimate and leaves the cluster covariance unchanged.
- Exact refits confirm it. The injected shock is iid over dates with the SD of the saved date-mean residual, at 1× and 10×, 200 draws each.

| Design | Saved estimate | Cluster SE | Calendar SE | SD of estimate under shock (1×, 10×) | Change in cluster SE |
|---|---|---|---|---|---|
| `country__y_yoy_nsa__long__event_dates_excluded__common__own_current` | −14.71 | 1.22 | 14.40 | 6.2, 62.0 | ≤ 1.7e-14 |
| `region__y_mom_nsa__long__all_eligible__common__own_current` | 1.82 | 1.15 | 1.62 | 2.7, 26.3 | ≤ 1.6e-14 |
| `annual__region__long__all_eligible__common` | −16.02 | 12.10 | 161.9 | 146, 1,566 | ≤ 1.1e-14 |

- The first row's |t| is 12.1 under the cluster SE and 1.02 under the calendar SE.
- The saved residuals show the common component is present. The variance of the date-mean residual is 1.7 times (9 regions) and 4.3 times (42 markets) its value under origin independence in monthly designs, and 6.4 and 23 times in annual designs (medians across designs). Mean pairwise residual correlation is 0.11–0.13 monthly and 0.69–0.83 annual.
- In the 22 dynamic designs the identity is approximate (median 6 % deviation, maximum 30 %).
- Across ledgers, 243 origin-cluster rows have p < .05 where the primary method has 29 for the same rows: monthly 232 of 688 across the temperature-effect and channel ledgers, annual 3 of 6, PPML 8 of 24. The cluster/calendar SE ratio falls as low as 0.028.
- *Consequence.* For a common slope the cluster SE measures idiosyncratic origin noise only. It is not a second opinion on the same uncertainty: it omits the common component altogether, which in most designs is the larger one (median SE ratio 0.48).
- The ledger does not say so. Monthly cluster rows carry the same generic `calibration_status` as calendar rows, and annual rows carry none.
- The builder emits every covariance as a selectable starred view with "Yes" at p < .05 (`scripts/visitor_rigour_report.py:123-128,193-204`; `templates/visitor_rigour.html:21`). The builder is outside the definitive scope, but a method-specific label belongs on these ledger rows.
- I did not test the identity for PPML, where Poisson weights differ by origin.

**N3. The origin×year family cannot move the average slope.**

- *Source.* `scripts/visitor_origin_year.py:27,75,256`; `notes/visitor_origin_year_protocol.md:3-7,21-26`.
- *Intended.* Exploratory robustness against origin-specific annual demand and capacity shifts.
- *Observed (P04, P10).* The panel is balanced and every stochastic regressor is common to all origins. So the average of the origin slopes and every shared-control slope are identical to the parent's in all 24 requests (≤ 1.3e-12 and ≤ 6.8e-13). K_time, time df and corrections are identical too.
- Only deviations from the average move, and the changes sum to zero (≤ 4.9e-13). Of 612 origin slopes, 85 change sign. Nominal p < .05 occurs for 25 in the parents and 27 in the children, with 11 in common. Holm-adjusted rejections go from 0 to 1.
- *Consequence.* The family cannot corroborate or weaken any average effect. It speaks only to heterogeneity, and there the set of "significant" origins is unstable.
- Reporting is as promised: all 24 are saved, with no p-based retention, no cluster rows and unchanged inference rules. Within-month origin shocks remain, as the protocol says (`:23`).
- Neither the protocol nor the ledger states the average-slope identity, and no parent-to-child change table is stored. `audit_probes/probe10_origin_year_change_table.csv` supplies one.

**N4. Bookkeeping (minor).**

- *Pins (P01).* 85 pinned input files are absent from the snapshot: `outputs/visitor_arrivals/report_verification.json` (annual and PPML manifests) and 84 files under `outputs/visitor_rigour/monthly/preflight/` (origin×year manifest). Every pin that is present matches byte for byte, and the 24 origin×year samples equal their parents' current samples (P04).
- *Status strings.* The family manifests, the companions manifest and `diagnostics/summary.json` read "…not_yet_independently_verified". The five `verification.json` files read "passed", and the builder writes "built_from_independently_verified_results" (`scripts/visitor_rigour_report.py:142,245`). A reader of a manifest alone sees the earlier state.
- *Annual holiday term.* `annual_LNY_included = False` (`scripts/visitor_rigour_companions.py:225`), while the protocol says "current shared controls" (`notes/visitor_rigour_protocol.md:118-119`). This is numerically harmless, because the annual window total is constant and so collinear with the intercepts, but it is undocumented there.

### 3B. Known, disclosed limitations: confirmed and quantified

**K1. The calendar t(T−1) reference over-rejects.** It is declared a working convention (`scripts/visitor_rigour_monthly.py:336,382`; `notes/visitor_rigour_protocol.md:103`). Exact benchmark rejection of saved t-ratios at nominal 5 % (P05, P11; a Monte Carlo check gives 0.2166 against the exact 0.2149, SE 0.0021):

| Designs | iid errors, HAC-12 (HAC-24) | MA(11) overlap, HAC-12 (HAC-24) |
|---|---|---|
| MoM, time df ≥ 157 | 5.6–8.7 % (7.1–9.8 %) | not applicable |
| Aggregate YoY, T = 158–189 | 7.7–10.5 % (9.5–12.8 %) | 14.7–22.0 % (14.1–21.6 %) |

- Under overlap, HAC-12 recovers 49–64 % of the true variance, and the exact 5 % critical |t| is 2.74–3.38 rather than 1.97.
- Event-excluded YoY residual autocorrelations at lags 1, 2, 3, 6 and 9 (0.69, 0.39, 0.11, −0.24, −0.35) track the overlap-implied pattern (0.63, 0.37, 0.16, −0.16, −0.27), though they diverge at lags 11–13. All-eligible YoY lag 1 is 0.21–0.23, between the two laws.
- On these benchmarks, YoY intervals are too narrow by a factor between roughly 1.1 and 1.7.

**K2. Thin designs (P05).** Fifteen estimated monthly specifications have fewer than ten time df. Ten of them have 1–3, with corrections up to 61 (`scripts/visitor_rigour_monthly.py:266-269,361`).

- In the single-series design with one time df, the benchmark rejects a true null 24–44 % of the time (exact 5 % critical |t| of 10–21), and its joint tests reject 88–97 %.
- In panel cuts with 1–3 time df, the correction inflates the variance 3–21 times and rejection is below 1 %. So the convention is neither uniformly conservative nor uniformly liberal there.
- The joint rows are flagged fragile and the template warns below ten time df (`templates/visitor_rigour.html:20`). But the 1,512 single-coefficient HAC-12 rows in the ten thinnest specifications are `inference_status = available` with only the generic label.
- Holm-significant H1 lag rows occur in 11 of 78 families below ten time df, against 19 of 482 above.
- In 44 estimated dynamic specifications the full cross-origin span rank exceeds K_time, so the correction is smaller than a span-rank rule would give. The ledger declares span rank diagnostic only (`:269`).

**K3. Finite-sample scaling is a material convention (P06).** For `region__y_yoy_nsa__full__event_dates_excluded__common__two_channel`, the endpoint difference of 7.58 has p = 0.056 under the production T/(T−K_time) scaling, 0.0088 under N/(N−K) and 0.0029 uncorrected. Production is the most conservative of the three, and K1 shows that even it over-rejects in the designs examined.

**K4. No excess of pointwise rejections (P10, P11).** With the main method at nominal 5 % and time df ≥ 10:

| Rows | Share with p < .05 |
|---|---|
| MoM H1 | 6.6 % (283 of 4,284) |
| MoM common | 9.1 % (23 of 252) |
| YoY H1 | 3.2 % (337 of 10,506) |
| YoY common | 4.7 % (29 of 618) |
| Origin×year | 4.4 % (27 of 612) |
| Annual common | 0 of 9 |
| PPML | 0 of 30 |

- These are comparable to or below what K1's benchmarks give with no effect.
- Holm, applied within profile only (`scripts/visitor_rigour_monthly.py:305-314`), leaves 29 lag rows in 19 of 482 families and 1 cumulative row in 98 families.
- Rejection shares rise from all-eligible to strict samples (MoM 4.8, 6.7, 8.4 %). Average slopes change sign across crisis variants (region YoY full: +30.5, −18.2, +4.3).
- The shares are descriptive: all rows share one weather series.

**K5. Annual (P07).** Seventeen growth years, origin intercepts plus a common trend, no year effects and no holiday term. Estimates are in 100 × log points per °C.

- Aggregate long-control: −10.9 (SE 152, p = 0.94; leave-one-year-out range −102 to 144).
- Event-excluded: −9.9 (SE 42); 2010 alone holds 44 % of the identifying temperature variation.
- Strict: +26.6 (SE 15.7); residual temperature SD is 0.096 °C after the other regressors explain 93 %.
- Benchmark rejection is 9.8–11.9 %. The companion cannot discriminate between effect sizes of any plausible magnitude.

**K6. PPML (P08, P09).** For `ppml__country__long__all_eligible` the temperature slope is 2.30 (HAC-12 SE 5.38, p = 0.67) and Pearson χ²/df is 2,618.

- The 60 crisis months hold 89 % of the squared temperature influence with 12 % of the Poisson weight; excluding them gives 0.46.
- The 30 requests yield 12 distinct temperature estimates: 10 are aliases, and aggregate, complete-10 and complete-52 coincide by the partition theorem.
- That theorem is an exact reconciliation under exhaustive counts, shared regressors and the stated fixed effects. It is not corroboration, as the template says at line 7.
- The corrected covariances coincide only because every origin's local time design, and hence K_time (35, 30, 27, 24), equals the aggregate's. HC0 and origin-cluster matrices are not part of the identity (ratios 0.13–1.55 and 0.05–0.90).
- The non-exhaustive cuts do not satisfy it. The country cut's slope differs from the aggregate's by up to 0.35, and the region cut's by at most 0.01 because regions hold 99.99 % of arrivals.

**K7. Diagnostics (P10).** Of 6,623 ADF/KPSS diagnostics, 6,503 are unavailable because of calendar gaps and 55 because of length. None completed for any monthly, annual or origin×year growth series. The 65 completed ones attach only to PPML level series, and 38 of them are flagged as a unit-root concern.

## 4. Not established

- **Causal identification.** One destination and one common weather series; the protocol lists the confounders (`notes/visitor_rigour_protocol.md:153-158`).
- **Finite-sample calibration of any test.** My benchmarks are reference laws on selected designs, not size estimates for every equation and not the data's error law.
- **Any conclusion from a multi-restriction joint test or from an origin-cluster row.**
- **Stationarity of any growth series.**
- **Corroboration** from the PPML partition match, the origin×year average slope, native bookkeeping or cross-engine agreement. These are algebraic or computational reconciliations.
- **That the origin×year family was fixed before its estimates were read.** The protocol asserts it and disclaims preregistration; the snapshot cannot verify it.
- **Stata agreement and browser behaviour**, which I did not execute.

## 5. Limitations of this review

- Twelve targeted probes. Raw-input reconstruction covers selected designs (P02–P04, P07, P08). The other probes use the saved design archives, which P01 binds to the native logs by hash.
- I did not repeat the production verifier's full covariance audit and ran no native estimation. Agreement is between saved native output and independent NumPy/SciPy QR and IRLS.
- The benchmark laws hold the design fixed with homoskedastic Gaussian errors, independent across origins. The saved residuals contradict that independence (N2), and one saved statistic sits in the far lower tail of its iid law (country MoM, share 0.9995). So these laws show that the nominal reference is unreliable; they do not say what the right reference is.
- P11 and the exact identity in P12 exclude lagged-outcome designs. P05 treats lagged outcomes as fixed regressors.
- Simulation is limited to five joint-test designs (4,000–20,000 draws) and four refit demonstrations (200 draws each). Grid-wide shares are descriptive.
- The 85 absent pinned files could not be hashed.
- The session environment reports the model as `claude-fable-5-2`. The runner's served-model receipt, not this text, is the authoritative record of that.