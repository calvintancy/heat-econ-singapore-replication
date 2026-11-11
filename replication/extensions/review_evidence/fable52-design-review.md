# Adversarial econometrics review of heat_econ at `e47edee` (read-only audit)

## Verdict

**No computational mistake was found, and I give no AEA-ready certification.** Every saved estimate I recomputed independently matches to 1.2e-10 or better. All eleven findings below are design or inference limitations, not arithmetic errors.

The evidence does not support causal or "significant" temperature claims in any family:

- **Main inference is not valid for the estimand.** Sector-clustered (CR1) errors, displayed as the main uncertainty, reject a true null 20–25% of the time on average at nominal 5%, and up to 51–58% in the worst case.
- **The "default baseline" is the lockdown quarter.** The 2017–2025 full-control result disappears or changes sign when 2020Q2 alone is removed.
- **The YoY coefficient is not a current-quarter effect.** It is more than fully accounted for by growth realised before the current quarter.
- **Nothing in the full long history survives honest inference.** No long-history coefficient has p < .05 once the time-level degrees of freedom are used.
- **The requested five-window analysis is a forking path.** "All five windows + YoY-baseline rule" gives the most negative YoY coefficient of 62 window-by-rule combinations in 8 of 12 series, flips the published sign, and is contradicted in sign by the annual design on the same windows.

Corrections I would make first:

1. Make time-level HAC plus year-block randomisation the primary inference for every coefficient on a national regressor (F1, F2).
2. Stop calling the 2017–2025 full-control sample the baseline (F3).
3. Replace current-temperature-only YoY rows with exposure matched to the growth window (F4).
4. Freeze the crisis-window list and exclusion rule in writing before estimating (Section 4).
5. Report the official aggregate beside every equal-weight panel (F9).

Terms used throughout:

- **Outcomes:** G1 is SA QoQ (MoM for monthly) log growth ×100; G2 is SA YoY; G2N is NSA YoY.
- **Project DK:** Driscoll–Kraay with factor T/(T−1)×(N−1)/(N−K_panel) and a t(T−1) reference.
- **Time-level:** the same calendar Bartlett HAC with factor T/(T−K_time) and t(T−K_time), where K_time counts season, year and weather parameters in the equivalent national time-series regression.
- **Exclusion rules:** *date-only* drops t if t is in a window; *YoY-baseline* also drops t if t−1 or t−h is in a window; *full-window* drops t if any date in [t−h, t] is in a window.

## 1. Replay

Run from the worktree root:

```
.venv/bin/python audit_probes/<probe>.py [args] > audit_probes/evidence/<file>.txt 2>&1
```

- **Environment.** `audit_probes/_env.py:11-19` sets the four BLAS thread variables to 1 and puts `scripts/` on `sys.path` before NumPy loads. I ran this bare form; the explicit-prefix form (`PYTHONPATH=scripts OMP_NUM_THREADS=1 …`) is equivalent by construction but I did not execute it.
- **Determinism.** In a final pass I re-ran every probe except `p00_inventory.py` (a ledger listing) and compared with `cmp`. All outputs are byte-identical to the stored evidence. Random probes use seed 20261111.
- **Windows.** The tentative windows live in one line, `audit_probes/_lib.py:39`. The helper supports whole-calendar-year windows only.
- **Worktree.** `git status --porcelain` shows only `?? .venv` (pre-existing) and `?? audit_probes/`. Nothing is staged or committed, and no source, data, output or report file was touched. No project producer or verifier was called.

| Probe (args) | Evidence file | Content |
|---|---|---|
| `p02_validate_lib.py` | `p02_validate_lib.txt` | Independent reproduction of 36 saved fits |
| `p03_time_level_equivalence.py` | `p03_time_level_equivalence.txt` | Panel ≡ national time series; DK factor |
| `p04_crisis_windows.py samples` / `tentative5` / `covid_only` | `p04_samples.txt`, `p04_estimates_*.txt` | Window sample accounting and estimates by rule |
| `p05_influence.py long` / `full` | `p05_influence_*.txt` | Leverage, leave-one-year/quarter-out |
| `p06_dynamics.py decomp` / `persistence` / `dl` / `rules` / `placebo` / `ac` | `p06_*.txt` | YoY decomposition, lags, falsification |
| `p07_placebo_size.py 2000 none` / `base` / `endpoint` | `p07_placebo_size_*.txt` | Empirical size of each inference convention |
| `p08_saved_yoy_weather.py` | `p08_saved_yoy_weather.txt` | Saved YoY-weather and DL ledgers |
| `p09_variation.py` | `p09_variation.txt` | Identifying variation, collinearity |
| `p10_aggregation.py va` / `iip` | `p10_aggregation_*.txt` | Equal-weight vs share-weighted vs official |
| `p11_controls_selection.py` | `p11_controls_selection.txt` | Coverage screen vs controls |
| `p12_annual_inventory.py`, `p13_annual_windows.py` | `p12_*.txt`, `p13_annual_windows.txt` | Annual family |
| `p14_visitor_inventory.py main` / `crisis` / `ppml`, `p19_visitor_ppml_cr1.py` | `p14_visitor_*.txt`, `p19_*.txt` | Visitor family |
| `p15_heterogeneity_saved.py va`, `p16_heterogeneity_identification.py 2000` | `p15_*.txt`, `p16_*.txt` | Sector heterogeneity |
| `p17_cr1_identity.py` | `p17_cr1_identity.txt` | Closed form of the CR1 standard error |
| `p18_window_multiverse.py` | `p18_window_multiverse.txt` | 31 window subsets × 2 rules |
| `p20_multiplicity.py`, `p21_clustered_sensitivity.py` | `p20_*.txt`, `p21_*.txt` | Counts of saved nominal tests |
| `p22_construction_checks.py` | `p22_construction_checks.txt` | Data identities; full-control base rule |

## 2. What the existing checks do and do not establish

**They do establish:**

- **Numerical reproducibility.** Independent NumPy algebra reproduces all 36 saved long-history G1/G2/G2N fits across 12 samples (p02): max absolute differences are b 1.03e-11, DK SE 9.64e-11, CR1 SE 5.91e-11, "ALL REPRODUCED (<1e-7): True". The 64 saved annual common-slope rows match to 1.26e-12 (b) and 3.64e-11 (SE). The saved H3 equality F statistics and CR1 SEs are reproduced exactly.
- **Construction identities (p22).** Quarterly and annual weather equal day-weighted monthly aggregates (max temperature difference 3.55e-15 and 7.11e-15). Growth outcomes equal 100×log differences by calendar lookup (1947/1914/1914 rows, max difference 1.42e-14). Annual log growth matches annual levels (3978 rows, 1.70e-13).
- **Balanced panels with common regressors.** The pooled slope equals the time-series slope of the equal-weight cross-sector mean (max difference 1.96e-12, p03).
- **Calendar-gap handling.** The HAC indexes the calendar and treats missing periods as zero rows. With whole-year windows the retained segments are at least 9 quarters or 25 months apart, so bandwidths of 4/8 quarters and 12/24 months never bridge a gap.
- **SA versus NSA.** YoY results do not depend on seasonal adjustment (VA11 G2 1.255 vs G2N 1.333).
- **PPML mechanics.** The fits converged (relative score error 0.0 at five decimals, 4–11 iterations) and zero cells are retained (50 and 46).
- **Rounding.** Temperature is published to 0.1 °C; rounding noise is 0.7–3.8% of identifying variance, so it is not a concern.

**They do not establish:**

- That displayed p-values have correct size (F1, F2).
- Causal identification. There is one national weather series, no cross-sectional exposure variation, and no control for global demand, ENSO or the electronics cycle anywhere in the VA/IIP families.
- A coherent dynamic specification (F4).
- Robustness to 2020Q2, to the partial year 2026, or to the window list (F3, Section 4).
- An aggregate GDP effect from equal-weight panels (F9).
- Stationarity of gapped samples, which README.md:20 itself disclaims.
- That wind and PM2.5 are valid controls.

## 3. Proven findings

| ID | Severity | Class | One line |
|---|---|---|---|
| F1 | Critical | Design (inference) | Sector/origin CR1 shown as main uncertainty for a national regressor |
| F2 | High | Design (convention inconsistency) | DK factor and t/F reference overstate time degrees of freedom |
| F3 | High | Design (sample selection, leverage) | "Default baseline" 2017–2025 result is 2020Q2 |
| F4 | High | Design (specification) | YoY on current temperature is mostly pre-period growth |
| F5 | High (new analysis), Medium (existing) | Design (inconsistency) | Date-only exclusion in VA/IIP/annual vs base/interval in visitor |
| F6 | High | Design (forking path) | Five-window result is the extreme of 62 combinations |
| F7 | Medium–High | Design | Tiny identifying variation; annual and quarterly signs disagree |
| F8 | Medium | Design | Sector heterogeneity reflects trends and COVID exposure; F-tests oversized |
| F9 | Medium | Design | Equal-weight panel differs from official aggregate, sometimes in sign |
| F10 | Medium | Design | Visitor results flip sign; PPML does not corroborate |
| F11 | Medium | Design (multiplicity) | About 10,000 saved nominal tests, significance concentrated where inference is weakest |

### F1 — Sector/origin-clustered errors are the displayed main uncertainty for a national regressor

**Severity:** Critical. **Class:** design limitation; the CR1 arithmetic is correct.

**Where:**
- README.md:20, README.md:58
- scripts/inference_display.py:1-10
- scripts/covid_common.py:239, :281
- scripts/extended_controls.py:250
- scripts/sector_cluster.py:38-39, :45
- scripts/annual_analysis.py:318-319
- notes/sector_cluster_inference.md:11 already concedes that sector clustering "does not address common national shocks".

**Replay:** `p17_cr1_identity.py`; `p07_placebo_size.py 2000 none`; `p03_time_level_equivalence.py`; `p06_dynamics.py placebo`; `p19_visitor_ppml_cr1.py`; `p21_clustered_sensitivity.py`.

**Actual output:**

The CR1 standard error of the pooled slope equals sd(sector slopes)/√S × √((N−1)/(N−K)) exactly: "max |identity - saved CR1 SE| = 8.94e-13" (p17). It contains no term for time-series sampling variation.

| Sample | Pooled b | sd of sector slopes | Identity = saved CR1 SE | Project DK SE | Time-level SE |
|---|---|---|---|---|---|
| VA11 long G1 (S 11, T 178) | −1.4279 | 2.5793 | 0.7962 | 1.2536 | 1.4455 |
| VA11 long G2 | 1.2551 | 2.3535 | 0.7265 | 0.6193 | 0.7141 |
| VA11 full-control 29q G2 | −7.7771 | 7.4523 | 2.4755 | 4.8044 | 6.6613 |
| IIP-m6 long G2 (S 6, T 405) | 1.4342 | 2.7835 | 1.1623 | 1.5289 | 1.5902 |

Placebo size when national weather year-blocks are reassigned to other years (p07, no exclusion, mean over nine multi-sector samples × three outcomes):

```
        rej05_dk  rej05_time  rej05_cr1  rej10_dk  rej10_time  rej10_cr1
cyclic     0.130       0.083      0.250     0.199       0.147      0.331
random     0.079       0.053      0.203     0.139       0.100      0.288
```

- **Worst cells (random scheme, nominal 5%):** VA19 G1 0.514 (cyclic 0.579); IIP-q18 G2 0.356; VA11-matched G1 0.330; IIP-m18 G2 0.308; VA11 G2 0.269.
- **Not reliably sized even without crises.** With the five windows dropped the CR1 average is 6.0–6.6%, but individual samples range from 0.8% to 13%.
- **Headline rows that depend on it (p03):** VA11 full-control G2 has CR1 p .0105 against time-level p .2657. IIP-m18 long G2 has CR1 p .0209 against DK p .2167. The saved VA distributed-lag cumulative effect has CR1 p .0222 against DK p .1204 (p08).
- **CR1 "detects" impossible effects (p06 placebo).** For VA19, current temperature "explains" growth realised before the quarter with CR1 p .0147 (DK p .3422). Next-quarter temperature "explains" current growth with CR1 p .0203 (DK p .3606).
- **Visitor family (p19):** "median CR1/HAC SE ratio 0.448; range 0.077..0.920; CR1 p<.05 in 18 of 28; HAC p<.05 in 3 of 28". Country YoY is +3.83 (CR1 p .0142) on all long dates and −12.07 (CR1 p .0000) on endpoint-excluded long dates. These are opposite-signed "significant" results on overlapping samples.
- **Interaction ledger (p21):** over 1170 saved slopes, "CR1 p<.05: 0.421; DK p<.05 (same rows): 0.289".

**Consequence:** the main rows measure how much sector slopes differ from each other, conditional on the realised weather history. They do not measure whether a different weather history would give the same answer. Any significance read from them is unsupported.

**Minimum correction:** make the time-level HAC and a year-block randomisation p-value primary for every coefficient on a national regressor. Keep CR1 only as a labelled dispersion statistic. The visitor family already does this at scripts/visitor_inference.py:62.

### F2 — DK small-sample factor and reference distributions overstate time degrees of freedom

**Severity:** High. **Class:** design limitation and cross-family inconsistency. The code does what notes/sector_cluster_inference.md:18 documents.

**Where:**
- scripts/estimate.py:49-50
- scripts/outcome_analysis.py:107-114 computes the national-time residual df but does not use it; :190 uses T−1.
- scripts/heterogeneity.py:144, :166, :172, :179 against :318
- scripts/annual_analysis.py:326-328
- scripts/covid_sector.py:160
- scripts/granular_interactions.py:55
- scripts/visitor_analysis.py:288 against :292
- scripts/aggregate_va.py:199, :212, :216

**Replay:** `p03_time_level_equivalence.py`; `p11_controls_selection.py`; `p07_placebo_size.py 2000 none`.

**Actual output:** "all balanced: True | max |b_panel-b_mean|: 1.96e-12 | max rel diff raw HAC SE: 2.29e-10". The raw HAC is identical; only the factor and reference differ. Variance factors are 1.054 (panel) vs 1.402 (time-level) for VA11 long, 1.257 vs 2.417 for 29 quarters, and 1.417 vs 3.8 for 19 quarters. Ten rows with project DK p < .05 lose it:

| Sample (T / time df) | Model | b | Project DK SE, p | Time-level SE, p |
|---|---|---|---|---|
| VA11 long (178 / 127) | G2 | 1.2551 | 0.6193, .0442 | 0.7141, .0812 |
| VA11 long | G2N | 1.3333 | 0.6133, .0310 | 0.7072, .0617 |
| VA11 long excl. 2020–22 (166 / 118) | G2 | 1.3173 | 0.5874, .0263 | 0.6778, .0544 |
| VA11 full-control (29 / 12) | G1 | −27.0542 | 11.1922, .0224 | 15.5182, .1068 |
| VA19 full-control (29 / 12) | G1 | −40.5163 | 16.8734, .0232 | 23.6228, .1120 |
| IIP-q6 full-control (29 / 12) | G1 | −15.4685 | 5.2759, .0066 | 7.1718, .0520 |
| IIP-q6 full-control excl. (19 / 5) | G2 | −20.4520 | 7.3677, .0125 | 11.7622, .1426 |
| IIP-q6 full-control excl. | G2N | −19.4686 | p .0158 | p .1560 |
| IIP-q18 full-control excl. (19 / 5) | G2 | −11.3017 | 4.9463, .0347 | 8.1949, .2264 |
| IIP-q18 full-control excl. | G2N | −11.3278 | p .0259 | p .2026 |

- **Same SE, honest reference (p11):** aggregate full-control excl. G2X −3.477 (1.550) has p .0377 on t(18) and .0749 on t(5). IIP-q total excl. G2 −23.906 (9.363) has p .0200 and .0511.
- **Annual full-control models** have 9 years and 2 residual time df but use t(8).
- **Saved full-control heterogeneity tests** use F(10, 28) with 12 time df (F8).
- **Placebo size (p07):** project DK rejects 7.9–13.0% at nominal 5% in full samples and 11.4–16.5% in gapped samples. Time-level rejects 5.3–8.3% and 7.8–11.8%.

**Consequence:** the only nominally significant long-history result under DK (VA11 YoY +1.26) is not significant under the factor the project itself uses for singleton, visitor and separate-sector fits.

**Minimum correction:** use T/(T−K_time) with t(T−K_time) or F(q, T−K_time) wherever the regressor is national, in all families. The `inference_df` hook at scripts/estimate.py:50 already exists. Report "descriptive only" instead of a p-value when time df is below about 10.

### F3 — The "default baseline" full-control 2017–2025 sample is driven by 2020Q2

**Severity:** High. **Class:** design limitation (sample selection and leverage).

**Where:** README.md:50; scripts/extended_controls.py:174, :185, :239, :247, :251-252.

**Replay:** `p05_influence.py full`; `p11_controls_selection.py`; `p09_variation.py`; `p15_heterogeneity_saved.py va`; `p22_construction_checks.py`.

**Actual output:**

- **Leverage (p05).** 2020–22 holds 48.0% of temperature leverage; 2020Q2 alone holds 19.8%.

| Series (29 quarters) | Saved b | Without 2020Q2 | Without 2020–22 |
|---|---|---|---|
| VA11 G1 | −27.054 | −9.642 | −0.031 |
| VA11 G2 | −7.777 | +6.166 | −0.359 |
| Aggregate G1 | −13.841 | −4.830 | +0.043 |
| Aggregate G2 | −5.782 | +0.556 | −3.477 |

- **Sector slopes rank COVID exposure (p15).** The correlation between a sector's slope and its 2020Q2 growth is +0.976. Construction goes from −106.410 to −24.488 without 2020Q2 and to −1.009 without 2020–22.
- **Collinearity (p09).** R² of temperature on the controls given fixed effects is 0.715; correlation with rain is −0.776. VA11 G1 moves from −4.616 (temperature only) to −21.707 (adding rain), −26.722 (adding humidity) and −27.054 (adding wind and PM2.5). Wind and PM2.5 move it by less than 1.
- **Support (p09).** Residual temperature has SD 0.114 °C and maximum 0.269 °C. One degree is 8.8 SDs and 3.7 times the largest deviation in the sample.
- **Coverage screen (p11).** The 90% screen is decided entirely by wind coverage; PM2.5 coverage is at least 0.997 everywhere. It drops 2019Q1 (wind coverage 0.172; the largest temperature anomaly of the window, +1.078 °C) and 2019Q3 (+1.025 °C; the PM2.5 haze maximum). On the same controls the screen alone moves VA11 G1 from −17.483 (36 quarters) to −26.722 (29 quarters).
- **Consistent base rule (p22).** Applying the YoY-baseline rule to 2020–22 leaves 17 quarters in 5 years with 4 residual df.

**Consequence:** −14 to −40 points of quarterly growth per °C is the lockdown quarter fitted to a +0.25 °C warm deviation. It is not a heat effect and is an order of magnitude outside the literature (Section 5).

**Minimum correction:** make the long-history temperature + rain + humidity model the baseline. Report the full-control sample as descriptive, with leave-one-quarter-out ranges and the screen-versus-controls decomposition.

### F4 — YoY growth regressed on current temperature is not a current-quarter effect

**Severity:** High. **Class:** design limitation (specification).

**Where:**
- scripts/outcome_analysis.py:133-141, :173-176
- README.md:66
- notes/aea_temperature_literature.md:96 concedes the base-period confound.
- notes/temperature_horizon_review.md:77-81 recommends lags 1–4.

**Replay:** `p06_dynamics.py decomp`; `p06_dynamics.py placebo`; `p06_dynamics.py dl`; `p07_placebo_size.py 2000 none`.

**Actual output:** YoY growth equals current-period growth plus growth from t−h to t−1, so b(G2) = b(G1) + b(PRE) exactly (identity gap 0).

| Sample | b(G1), current period | b(PRE), realised before t | b(G2) |
|---|---|---|---|
| Aggregate (178 q) | −0.6761 (1.0419) | +1.6403 (1.2745) | +0.9641 |
| VA11 | −1.4279 (1.2536) | +2.6830 (1.4073) | +1.2551 |
| IIP-m total (513 m) | +0.0814 | +1.3723 (1.6249) | +1.4537 |
| IIP-m18 | −0.4239 | +2.2141 (1.5240) | +1.7902 |

- **Falsification.** After controlling for temperature lags 1 to h+3, current temperature still "predicts" already-realised growth: VA11 2.6699 (1.4558), DK p .068; aggregate 2.1054 (1.3944). That is twice the headline coefficient.
- **Inconsistent with the QoQ lag model.** In the lag 0–4 model for VA11, the current-temperature coefficient is −1.353 (1.100) for G1 and +1.119 (0.741) for G2. If YoY were the sum of four QoQ responses these would be equal.
- **Randomisation p-values** for VA11 G2 (+1.315 on complete years) are .241 (random) and .295 (cyclic), against a DK p of .039.

**Consequence:** the positive long-history YoY coefficients cannot be read as an effect of this quarter's heat. They reflect within-year timing of earlier growth.

**Minimum correction:** for YoY outcomes use exposure matched to the growth window (the project's own Y3/Y4 or lag 0–4 forms). Print the pre-period falsification coefficient beside every YoY estimate.

### F5 — Exclusion rule differs across families

**Severity:** High for the requested five-window analysis; Medium for the existing COVID-only VA/IIP rows. **Class:** design limitation and inconsistency.

**Where:**
- Date-only for 2020–22: scripts/estimate.py:90; scripts/iip_analysis.py:127; scripts/outcome_analysis.py:150; scripts/aggregate_va.py:152; scripts/extended_controls.py:185; scripts/heterogeneity.py:236; scripts/annual_analysis.py:464; scripts/covid_common.py:212.
- Base and interval rules, visitor family only: scripts/visitor_analysis.py:226-230, :268-273; scripts/visitor_crisis.py:26-28.
- README.md:68 discloses that 2023 YoY uses 2022 bases.

**Replay:** `p04_crisis_windows.py covid_only`; `p14_visitor_inventory.py main`; `p22_construction_checks.py`.

**Actual output:**

- **COVID-only, long VA11.** G1 is −1.428 (none), −0.077 (date-only), −0.194 (base). G2 is 1.255, 1.317, 1.356. The numerical difference is small here.
- **The project's own visitor ledger shows the rule can flip the sign.** Aggregate YoY long is −8.2989 (10.1589) under date-only and +3.9815 (1.6693), p .0182, under the clean base rule.

**Consequence and correction:** see Section 4, where the difference becomes material.

### F6 — The five-window result is the extreme of the window-by-rule grid

**Severity:** High. **Class:** design limitation (forking path).

**Replay:** `p18_window_multiverse.py`. Output and correction are in Section 4.6.

### F7 — Identifying variation is tiny, and annual and quarterly designs disagree in sign

**Severity:** Medium–High. **Class:** design limitation.

**Where:** scripts/estimate.py:19; scripts/aggregate_va.py:17; scripts/annual_analysis.py:262-289; README.md:15, :34.

**Replay:** `p09_variation.py`; `p13_annual_windows.py`; `p05_influence.py long`.

**Actual output:**

- **Quarterly.** Temperature SD falls from 0.694 raw to 0.480 after season, 0.272 after year effects and 0.201 after rain and humidity. Year effects remove 68.0% of deseasonalised variance.
- **Annual.** SD is 0.234 after trend, rain and humidity. The year 1998 alone holds 25.9% of leverage (residual +0.781 °C, growth −1.815).
- **Full-history signs.** Aggregate annual is +0.419 (3.451); quarterly G1 is −0.676 (1.042); quarterly G2 is +0.964 (0.967). None is distinguishable from zero.
- **Published long-history coefficients come from the crisis years.** Additive contributions to aggregate G2 +0.964 are +1.386 from the 12 in-window years, +0.554 from the 5 first post-window years, and −0.976 from the other 28 years. For VA11 G1 −1.428 the split is −1.578, +0.308 and −0.157.

**Consequence:** a "per °C" effect is a five-SD extrapolation. There is no stable sign across frequencies.

**Minimum correction:** report effects per SD of identifying variation with the support stated. Present annual and quarterly as different estimands. Name ENSO and global demand as uncontrolled confounders.

### F8 — Sector heterogeneity results reflect trends and COVID exposure

**Severity:** Medium. **Class:** design limitation.

**Where:** scripts/heterogeneity.py:60-84, :144, :166, :172, :236, :256; README.md:37, :39.

**Replay:** `p15_heterogeneity_saved.py va`; `p16_heterogeneity_identification.py 2000`.

**Actual output:**

- **H1 relative slopes use no year effects and no controls.** They equal the slopes of a deviation-from-mean regression without either (max difference 5.77e-15), so they draw on between-year and trend variation, which is 68.0% of deseasonalised temperature variance in the VA sample. For YoY, the saved H1 relative slopes correlate +0.956 with between-year-only slopes.
- **Example.** Other Goods YoY is 6.121 (p .001) under H1 and 0.160 (p .934) under separate within-year fits. Its relative slope is +4.865 under H1 and +0.185 once a sector trend is added.
- **Equality F-tests are oversized.** Under placebo weather, nominal 5% tests reject 15.1% (VA11 QoQ), 27.9% (VA11 YoY) and 24.3% (IIP-m6 YoY). The saved H3 p-values of .0182 and .0161 become randomisation p-values of .134 and .057.
- **Full-control tests.** Saved F statistics of 33.62 and 65.75 are referred to F(10, 28) with 12 time df.

**Consequence:** long-sample "heat-sensitive sector" rankings pick up differential sector trends during a warming period. Full-control rankings pick up pandemic exposure.

**Minimum correction:** use within-year relative slopes (H3-type, or H1 with sector trends). Test equality by randomisation or F(q, T−K_time).

### F9 — Equal-weight panels are not the aggregate, and can differ in sign

**Severity:** Medium. **Class:** design limitation. README.md:93 already says equal weights do not estimate a GDP effect.

**Replay:** `p10_aggregation.py va`; `p10_aggregation.py iip`.

**Actual output:**

| Sample, outcome | Equal-weight pooled | Share-weighted mean of slopes | Official aggregate (DK SE) |
|---|---|---|---|
| Long, no exclusion, G2 | +1.255 | +0.882 | +0.964 (0.967) |
| Long, five windows date-only, G2 | +0.424 | −0.652 | −0.579 (0.740) |
| Long, five windows YoY-baseline, G2 | +0.047 | −1.708 | −1.667 (0.660) |
| Full-control 29q, G1 | −27.054 | −16.778 | −13.841 (8.246) |

- **Sector driver.** Under the YoY-baseline rule Manufacturing (22% share) has slope −6.95 while Construction (5%) has +2.85.
- **Cluster driver.** Within manufacturing, Electronics is −15.35 quarterly and −10.59 monthly. The official IIP total is −6.720 (2.437) against −3.480 pooled.

**Consequence:** the panel overweights tiny sectors (Other Goods has a 0.2% share against a 9.1% equal weight). The negative manufacturing association is carried by one export cluster with an obvious unmodelled global cycle.

**Minimum correction:** print the official aggregate and share-weighted estimates beside every pooled coefficient.

### F10 — Visitor results are unstable and uncorroborated

**Severity:** Medium. **Class:** design limitation.

**Where:** scripts/visitor_analysis.py:288, :292; scripts/visitor_crisis.py:14-20.

**Replay:** `p14_visitor_inventory.py crisis`; `p19_visitor_ppml_cr1.py`.

**Actual output:**

- **Aggregate YoY, long sample:** +5.95 (16.54) with no exclusion; +3.98 (1.67), p .018, for COVID; +2.87 (1.91), p .135, for buffered windows; +3.45 (2.77), p .216, for stable pre-2020.
- **Aggregate MoM, full controls:** −4.08 (1.93), p .039, for COVID, referred to t(65) with 44 residual df; −0.55 (1.60) for stable pre-2020, with 31 against 13.
- **PPML:** all twelve level models have p > .66.

**Consequence:** the significant cells are isolated and opposite-signed across outcome and sample.

**Minimum correction:** use the t(T−K) reference; label the family exploratory; show PPML beside OLS.

### F11 — Multiplicity

**Severity:** Medium. **Class:** design limitation.

**Replay:** `p20_multiplicity.py`.

**Actual output:** saved temperature rows with nominal p < .05 are 1782 of 8057 (22.1%) under calendar HAC and 653 of 2015 (32.4%) under unit-cluster CR1. Counts overlap across ledgers and are descriptive only. Significance concentrates in the full-control samples, in CR1 rows and in unadjusted sector slopes.

**Minimum correction:** declare a small primary set and treat everything else as exploratory.

## 4. Audit of the requested crisis-window exclusion

Windows are the tentative list 1985–86, 1997–98, 2001–03, 2008–09, 2020–22. I used only that list and exhaustive subsets of it. Nothing was chosen from outcomes or significance.

### 4.1 How the three rules differ

- **Date-only** keeps the h periods after each window. Their YoY growth is measured from a crisis-period base, so the crisis is still inside the outcome with the opposite sign. That is 20 of 130 retained VA quarters and 60 of 369 IIP months.
- **YoY-baseline** removes those periods. For whole-year windows it equals date-only with every window extended by one year.
- **Full-window** is identical to YoY-baseline whenever a window is at least h periods long. p04 prints `interval_equals_base True` for all 12 samples.
- **Shorter windows.** The two rules differ only for windows shorter than h, such as a single official recession quarter. There the full-window rule is stricter and is the one to use. This is a derivation; I did not probe quarter-dated windows.
- **QoQ outcomes.** Only the first period after each window is affected (five quarters). Those five quarters move the IIP-q total QoQ coefficient from +1.564 to −0.552 and aggregate GVA from +0.231 to −0.200.

### 4.2 Sample accounting (p04 `samples`; T / years / segments / residual time df)

| Series (start) | No exclusion | Date-only | YoY-baseline = full-window | Windows that bind |
|---|---|---|---|---|
| Aggregate GVA, VA11 (1982Q1) | 178 / 45 / 1 / 127 | 130 / 33 / 6 / 91 | 110 / 28 / 6 / 76 | all five |
| VA11 matched, VA19 (2006Q1) | 82 / 21 / 1 / 55 | 62 / 16 / 3 / 40 | 54 / 14 / 3 / 34 | 2008–09 and 2020–22 only |
| IIP-q total (1984Q1) | 171 / 43 / 1 / 122 | 123 / 31 / 6 / 86 | 103 / 26 / 6 / 71 | all five; 1984 isolated |
| IIP-q6 (1993Q1) | 135 / 34 / 1 / 95 | 95 / 24 / 5 / 65 | 79 / 20 / 5 / 53 | four |
| IIP-q6 matched, IIP-q18 (1997Q1) | 119 / 30 / 1 / 83 | 79 / 20 / 4 / 53 | 63 / 16 / 4 / 41 | four; starts 1999Q1 or 2000Q1 |
| IIP-m total (1984-01) | 513 / 43 / 1 / 456 | 369 / 31 / 6 / 324 | 309 / 26 / 6 / 269 | all five |
| IIP-m6 (1993-01) | 405 / 34 / 1 / 357 | 285 / 24 / 5 / 247 | 237 / 20 / 5 / 203 | four |
| IIP-m6 matched, IIP-m18 (1997-01) | 357 / 30 / 1 / 313 | 237 / 20 / 4 / 203 | 189 / 16 / 4 / 159 | four |

Consistency of controls, fixed effects, HAC and samples:

- **Controls** are consistent: temperature, rain and humidity exist for the whole history in every family. Wind and PM2.5 cannot be used, since only 2020–22 falls inside their range.
- **Fixed effects** stay balanced with whole-year windows. No retained year is a singleton, and only the final year 2026 is partial. Quarter-dated windows would create partial years.
- **HAC gaps** are handled correctly (never bridged). The cost is fewer lag pairs: 174 at lag 4 falls to 106 and then 86 for VA. Under the YoY-baseline rule the year 2000 is an isolated one-year segment.
- **Samples are not consistent across families.** The same list removes five windows from VA11 but two from VA19. A cross-family comparison therefore mixes window incidence with sector scope unless a common-start version is also reported.

### 4.3 Estimates by rule (p04 `tentative5`)

Temperature + rain + humidity; cells are b (project DK SE); p is time-level.

| Series | Outcome | No exclusion | Date-only | YoY-baseline = full-window |
|---|---|---|---|---|
| Aggregate GVA | G1 | −0.676 (1.042) | 0.231 (0.656) p .725 | −0.577 (0.512) p .264 |
| | G2 | 0.964 (0.967) | −0.579 (0.740) p .436 | −1.667 (0.660) p .014 |
| VA11 | G1 | −1.428 (1.254) | 0.238 (0.383) p .593 | −0.153 (0.394) p .739 |
| | G2 | 1.255 (0.619) | 0.424 (0.409) p .373 | 0.047 (0.421) p .924 |
| VA11 matched | G1 | −2.271 (2.963) | 1.152 (0.422) p .026 | 0.735 (0.427) p .156 |
| | G2 | 0.143 (0.834) | 0.305 (0.326) p .431 | −0.049 (0.330) p .901 |
| VA19 | G1 | −3.780 (4.348) | 1.054 (0.446) p .053 | 0.578 (0.374) p .203 |
| | G2 | −1.432 (1.460) | −0.412 (0.423) p .416 | 0.078 (0.480) p .892 |
| IIP-q total | G1 | 1.071 (2.159) | 1.564 (2.624) p .553 | −2.291 (1.500) p .131 |
| | G2 | 1.947 (3.073) | −2.688 (2.216) p .229 | −4.968 (2.248) p .030 |
| IIP-q6 | G1 | 2.017 (1.883) | 2.672 (2.194) p .293 | −0.816 (1.919) p .714 |
| | G2 | 1.640 (2.113) | −2.109 (1.635) p .266 | −3.480 (2.151) p .167 |
| IIP-q6 matched | G1 | 2.071 (2.060) | 2.824 (2.351) p .303 | −0.741 (1.978) p .749 |
| | G2 | 2.015 (2.315) | −2.126 (1.663) p .273 | −3.794 (2.203) p .146 |
| IIP-q18 | G1 | 0.051 (1.718) | 1.090 (1.404) p .511 | −0.857 (1.662) p .664 |
| | G2 | 2.205 (2.120) | −2.121 (1.254) p .155 | −4.061 (1.323) p .013 |
| IIP-m total | G1 | 0.081 (0.944) | 0.205 (1.020) p .841 | −0.856 (0.914) p .350 |
| | G2 | 1.454 (1.702) | −1.623 (1.535) p .291 | −2.768 (1.441) p .056 |
| IIP-m6 | G1 | 0.252 (1.068) | −0.152 (1.159) p .900 | −0.889 (1.136) p .453 |
| | G2 | 1.434 (1.529) | −1.711 (1.424) p .250 | −1.858 (1.531) p .246 |
| IIP-m6 matched | G1 | 0.486 (1.113) | 0.122 (1.143) p .918 | −0.619 (0.961) p .538 |
| | G2 | 1.846 (1.666) | −1.501 (1.544) p .352 | −1.708 (1.657) p .325 |
| IIP-m18 | G1 | −0.424 (0.622) | −0.270 (0.596) p .666 | −0.536 (0.494) p .303 |
| | G2 | 1.790 (1.447) | −1.631 (1.041) p .136 | −2.146 (0.935) p .030 |

- G2N tracks G2 throughout.
- No no-exclusion coefficient has time-level p < .05.
- YoY coefficients for aggregate GVA and all manufacturing series change sign from positive to negative. The pooled VA11 panel goes to zero.

### 4.4 Inference in the gapped samples (p07 `base`, complete years only)

Mean placebo rejection at nominal 5%, multi-sector samples:

```
        rej05_dk  rej05_time  rej05_cr1
cyclic     0.165       0.118      0.064
random     0.114       0.078      0.060
```

Singleton national series reject 6.7%.

| Series, G2, YoY-baseline rule | b | Time-level p | Randomisation p (random / cyclic) |
|---|---|---|---|
| Aggregate GVA | −1.745 | .015 | .034 / .074 |
| IIP-q total | −5.732 | .016 | .031 / .080 |
| IIP-q6 | −4.820 | .056 | .096 / .158 |
| IIP-q18 | −5.227 | .000 | .026 / .200 |
| IIP-m total | −3.080 | .033 | .066 / .080 |
| IIP-m18 | −2.588 | .003 | .066 / .200 |

- **Partial year 2026.** These estimates differ from 4.3 only by dropping it. That alone moves IIP-q6 G2 from −3.480 to −4.820 and IIP-q18 from −4.061 to −5.227.
- **Caveats.** The randomisation assumes year blocks of weather are exchangeable, which multi-year ENSO episodes violate somewhat. The cyclic scheme has only 12–43 draws.

### 4.5 Annual frequency (p13)

| Series (annual NSA log growth) | No exclusion | Five windows, date-only | Five windows, baseline rule |
|---|---|---|---|
| Aggregate | 0.419 (3.451) | 5.165 (2.299) p .032 | 2.498 (1.003) p .019 |
| VA11 | −0.145 (3.114) | 3.742 (1.454) p .015 | 2.425 (0.869) p .010 |
| IIP total | 2.372 (5.211) | 12.728 (6.468) p .059 | 5.557 (3.785) p .155 |
| IIP 6 clusters | 7.021 (4.210) | 16.425 (5.650) p .008 | 8.808 (3.507) p .022 |

- **Opposite sign.** The annual sign is positive where the quarterly YoY sign is negative.
- **Leverage.** In the annual design the windows hold 54.2% of temperature leverage and remove the single most informative year (1998). In the within-year designs they hold 27–31%.
- **What exclusion does in each design.** With year effects, dropping crisis years removes high-variance within-year swings; it does not fix confounding. Without year effects (annual), it removes the regressor's main variation.

### 4.6 Sensitivity to the window list (p18)

Each cell is from 62 specifications per series: 31 non-empty subsets × two rules.

| Series, G2 | Min | Median | Max | Share negative | Time-level p < .05 |
|---|---|---|---|---|---|
| Aggregate GVA | −1.667 | 0.092 | 0.991 | 40% | 1 negative |
| VA11 | 0.040 | 0.861 | 1.424 | 0% | 3 positive |
| IIP-q total | −4.968 | −0.461 | 2.121 | 58% | 1 negative |
| IIP-q18 | −4.061 | −0.024 | 2.205 | 52% | 4 negative |

- **"All five + YoY-baseline" is the minimum** for G2 in aggregate GVA and all seven manufacturing series with an all-five cell shown (all but IIP-q6 matched share that minimum exactly).
- **No single window turns aggregate YoY negative:** +0.51, +0.47, +0.66, +0.45, +0.92 for the five single-window exclusions.
- **Removing one window from the list** moves aggregate G2 from −1.67 to −0.58 (keep 1997–98) or −0.68 (keep 2008–09). IIP-q total moves from −4.97 to −1.97 or −1.01.
- **QoQ signs** follow whether 2020–22 (VA) or 2008–09 with its 2010 rebound (IIP) is excluded.

### 4.7 Recommended protocol for the new analysis

1. **Fix the windows and rule in writing first.** Take the list from official sources with one stated treatment of rebound periods, and record it before any estimate is produced. As listed, 2020–22 includes rebound years while 1987, 1999, 2004 and 2010 fall outside their windows. In the full-history fit, 2010 is the single most influential year for IIP-q total QoQ (leave-out move −1.683).
2. **Use the full-window rule,** which equals the YoY-baseline rule for whole-year windows. Apply one retained-date set to G1, G2 and G2N. The visitor family already implements this at scripts/visitor_analysis.py:268-273.
3. **Compute growth and weather lags on the full calendar before masking** (already done, README.md:20). Keep the calendar-indexed HAC; never compress the index.
4. **Keep whole calendar years.** If official quarter-dated windows are adopted, report both versions.
5. **Use the same controls and fixed effects everywhere.** Add the lag 0–4 specification that notes/temperature_horizon_review.md:78 already recommends.
6. **Report native-history and common-start samples,** since the list binds differently by family.
7. **Inference:** time-level HAC plus year-block randomisation p-values as primary; CR1 as dispersion only.
8. **Show the official aggregate and share-weighted estimates** beside pooled ones, and the Electronics slope beside manufacturing totals.
9. **Publish the leave-one-window-out table and the date-only column** as declared sensitivities. Do not describe the retained sample as crisis-free (the wording at scripts/visitor_crisis.py:28 is appropriate).
10. **Do not run the five-window design on full-control samples.** It leaves 4 residual df.

## 5. AEA-literature comparability

These statements are from my knowledge and the repo's notes; I did not check sources online.

- **Dell–Jones–Olken (2012, AEJ: Macro)** identify from many countries with region-by-year effects and two-way clustering (notes/aea_temperature_literature.md:17). Here there is one country, so common shocks cannot be absorbed and the effective sample is T periods, not S×T. The annual design shares their transformation only; README.md:34 says as much.
- **Magnitudes.** Their headline is roughly −1.3 points of annual growth per °C in poor countries and about zero in rich ones. Long-history estimates here (about ±1–2 with SE near 1) are compatible with zero. Full-control estimates of −14 to −40 are not comparable to anything in that literature (F3).
- **Variation used.** Quarterly designs with year effects discard the between-year variation that literature uses, and identify from within-year deviations with SD 0.2 °C. Nonlinearity cannot be assessed on that support.
- **Credible sub-national and sector designs** rely on spatial exposure variation (US states, plants). A single-station national series has none.
- **Non-AEA anchors.** Burke–Hsiang–Miguel (2015) and Colacito et al. (2019) are not AEA publications, as notes/aea_temperature_literature.md:112 states.

## 6. Unverified concerns (not findings)

- **ENSO and global-cycle confounding.** No external series is in the repo. The only evidence is indirect: leverage concentrated in 1998, 2016, 2014 and 1983, and the Electronics pattern.
- **PM2.5 as an activity-driven mediator.** Second-quarter PM2.5 in 2020 (1.012) is below every pre-2020 second quarter, but 2023 (0.994) is lower still. This is suggestive only.
- **Wind missingness.** Why wind coverage is 0.172 in 2019Q1, and whether it relates to weather.
- **Official dating of the tentative windows.** From memory only, 1997 and 2002 may not be recession years. The parent's source check should decide this, not my results.
- **Quarter-dated windows.** They would create partial years; not probed.
- **Exposure measurement.** Nonlinearity, daily extremes, wet-bulb or maximum temperature, and single-station representativeness.
- **Seasonal adjustment and calendar effects.** Leakage of crisis outliers into neighbouring quarters for G1, and Lunar New Year or working-day effects in VA/IIP.
- **Mechanism.** The explanation for the Electronics-driven result and for the positive annual coefficient.

## 7. Coverage limits

- **Re-estimated:** the common-slope temperature + rain + humidity and full-control models for all 12 VA/IIP samples; annual common-slope models; H1/H3 heterogeneity for VA11, IIP-q6 and IIP-m6.
- **Read from saved ledgers only:** visitor OLS, crisis, PPML and origin-cluster results; the extended-control C0–C6 rows; the granular-interaction and COVID-comparison ledgers (counted, not re-estimated); the H2, R1 and H3X heterogeneity rows.
- **Not examined:** log-level and raw-level models beyond residual autocorrelation; visitor profile and leave-one-year-out variants; quadratic, daily and lunar visitor variants; stationarity and reconciliation ledgers; native PyFixest fit logs; HTML reports; raw-data retrieval.
- **Placebo samples** use complete calendar years, so their "actual" estimates differ slightly from p04.
- **Not done:** no network access, no check of official crisis dates, no producers or verifiers run.

Everything is in `audit_probes/`; evidence is under `audit_probes/evidence/`.