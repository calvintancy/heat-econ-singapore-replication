# Adversarial inference review — heat_econ at e47edee

Fable adversarial econometrics reviewer (inference focus) · 11 Nov 2026 · worktree `dist/aea_audit_20261111/fable-inference`

## Verdict

**No AEA-ready certification.**

1. **The saved numbers are right as arithmetic.** My independent re-implementations match the saved ledgers to within 9e-08 across the common-slope, heterogeneity, annual, visitor and sector-CR1 families. I found no computational error in any saved estimate.
2. **The inference attached to those numbers is not credible as designated.** On the project's own saved samples, the rules that produce "main" p-values, stars and heterogeneity claims reject a true null far more often than 5%: up to 67% for the sector-clustered main rule and 65–99% on average for the equal-slope omnibus tests.
3. **Proven computational mistakes are limited to two report-generation defects (F10) and one latent HAC hazard (F11).** The hazard does not touch any saved output, but it does fire on the kind of base-clean exclusion the new crisis-window analysis needs.
4. **The requested crisis-window analysis can be run consistently for temperature, rain and humidity, but its results depend on the exclusion rule, on exactly which windows are in the list and on frequency.** The rule and the window list have to be fixed before estimation and every rule shown side by side.

Everything outside F10 and F11 is a design or interpretation limitation. The README and notes already acknowledge several of them in words. What this review adds is their size on the real samples, and the places where displays and labels contradict the acknowledgements.

## How to read the evidence

- **Sector CR1**: covariance clustered by sector with a t(S−1) reference. It is the project's designated "main" rule for pooled panels.
- **Pooled DK(h)**: Driscoll–Kraay with Bartlett bandwidth h as the project computes it, with factor T/(T−1)×(N−1)/(N−K) and a t(T−1) reference.
- **Series convention**: the rule the project itself uses for single national series. It is the same Bartlett sum with factor T/(T−K_T), where K_T counts time-series parameters (weather terms, season and year dummies). I pair it with a t(T−K_T) reference.
- **Placebo**: a year-block randomization on the real saved samples. On the weather side, outcomes are real and each sample year's weather is replaced by the real weather of a different calendar year. On the outcome side, weather and design are real and each year's outcome block is moved to another year. The placebo regressor has no effect by construction, so a 5% test should reject about 5% of the time. Permutations have no fixed points; the seed is 20261111. With 1,000 draws the Monte Carlo standard error of a rate near 5% is about 0.007.
- In probe output, `HACs` means the series convention and `R0`–`R3` are the four crisis rules defined in the crisis section. Quoted output lines are verbatim apart from collapsed whitespace.

Every replay command is `.venv/bin/python audit_probes/<probe>.py [draws]`, run from the worktree root. `audit_probes/_common.py` sets single-thread BLAS and puts `scripts/` on the path internally, because my sandbox required approval for env-prefixed commands. Evidence is in `audit_probes/evidence/`. I wrote nothing outside `audit_probes/` and ran no producer or verifier.

## Findings at a glance

| # | Severity | Type | Finding |
|---|---|---|---|
| F1 | High | Design limitation, claim inflation | The "main" sector-clustered rule cannot see national shocks. It rejects a true null in up to 67% of placebo draws, and 29 of 32 main-significant pooled cells are not significant under the project's own single-series rule |
| F2 | Medium (High if pooled DK is promoted to primary unchanged) | Convention inconsistency | Pooled DK standard errors are 1.04–2.57× smaller than the project's own single-series rule gives on identical time-series information; t(T−1) is used with 2–12 residual degrees of freedom |
| F3 | High | Design limitation, claim inflation | Equal-slope omnibus F(q, T−1), Holm pair families and per-sector total-effect tests reject a true null in 65–99% of placebo draws on average |
| F4 | High | Interpretation, label | The README "default baseline" owes 82% of its coefficient to 2020–22 dates; dropping 2020 alone moves −27.05 to −1.48 |
| F5 | Medium | Interpretation | Signs and sizes are unstable across samples and frequencies |
| F6 | High | Label, claim | The sector-CR1 heterogeneity sensitivity prints 1,170 per-sector p-values from one shared standard error per table |
| F7 | Medium | Design limitation | Annual family: the main rule rejects 27–52% in VA panels; nine-year full-control fits reject 19–44% |
| F8 | Medium | Design limitation, label | Visitor family: the origin-clustered comparison and the profile omnibus are invalid for common weather; 0 of 14 nominally significant results survive randomization on both sides |
| F9 | Medium | Claim inflation | About 66,600 p-values, stars from p < 0.10, Holm only for pairs; YoY and level outcomes over-reject more than one-period growth |
| F10 | Low | Computational mistake (report generation) | Unlabelled duplicate control rows in `first_results.md`; `iip_results.md` describes one inference rule and prints another |
| F11 | Low for saved outputs, blocking for the new analysis | Latent computational hazard | statsmodels silently compresses calendar gaps when the span exceeds twice the sample size; a base-clean exclusion on the short single-series samples triggers it and understates standard errors by 7–21% |

## Proven findings

### F1. Sector CR1 is the designated main pooled uncertainty, but it cannot see national shocks

**High · design limitation with claim inflation; the arithmetic reproduces.**

- **Where.** Designation: `README.md:20,58`, `scripts/inference_display.py:5-10`, `scripts/report.py:99-101,156`, `scripts/iip_report.py:86-89,103`, `scripts/covid_common.py:239`, `scripts/annual_analysis.py:315-319`. Formula: `scripts/sector_cluster.py:38,45`. The project's own text says the opposite of the designation at `notes/sector_cluster_inference.md:11`, `README.md:91`, `notes/covid_comparison_methods.md:47` and `outputs/first_results.md:20`.
- **Replay.** `p01_cr1_dk_decomposition.py`; `p02_placebo_common.py 2000` then `p02b_placebo_tables.py`; `p04_ledger_flags.py`.
- **Output** (VA broad 11, model S3, N=1958, S=11, T=178):

```
[identity 2] mean-series beta = -1.427924 |diff| = 8.84e-11
[identity 4] sd(own-series slopes)/sqrt(S)*sqrt((N-1)/(N-K)) = 0.796218 |diff to CR1 SE| = 5.85e-11
```

Identity 5 scales the national common component of the outcome by c and leaves sector deviations alone:

| c | β | CR1 SE | CR1 p | DK(4) t |
|---|---|---|---|---|
| 0.5 | −0.71396 | 0.79622 | 0.39096 | −1.13906 |
| 1 (actual) | −1.42792 | 0.79622 | 0.10316 | −1.13906 |
| 2 | −2.85585 | 0.79622 | 0.00496 | −1.13906 |
| 5 | −7.13962 | 0.79622 | 0.00000 | −1.13906 |

The pooled coefficient is exactly the national mean-series coefficient, and the CR1 standard error is exactly the cross-sector dispersion of own-series slopes. Doubling national noise doubles the coefficient and leaves the CR1 standard error unchanged.

```
CR1    : 120 of 192 cells above 0.10 ; median 0.126 ; max 0.671 ; min 0.000
main-significant cells that are NOT significant under the series convention: 29 of 32
```

Long-history growth outcomes reject 22–23% on average. VA detailed-19 long-history QoQ rejects 49–52%. In the 19-quarter excluded samples the rule under-rejects on the weather side (1.3%), so it is miscalibrated in both directions.

- **Consequence.** A starred pooled coefficient under the main rule says the sectors' slopes are similar to each other. It does not say the national weather–output association is distinguishable from national time-series noise. IIP detailed-18 quarterly long-history YoY (NSA) is +2.479 with main p 0.0095, pooled DK p 0.258 and series p 0.332. The rule can also be too wide (IIP broad-6 quarterly full controls: CR1 p 0.093, pooled DK p 0.0066), because it measures a different quantity.
- **In fairness.** The VA headline is not a false positive: `outputs/first_results.md:3,36` reports p = 0.103 and says the result "does not establish" an effect. Its interval [−3.20, 0.35] is the CR1 one, though; the series-convention standard error is 1.4455 (p 0.325) against 0.796.
- **Minimum correction.** Make national time-series inference primary for every coefficient on a nationally common regressor, using the series convention or year-block randomization p-values. Relabel sector CR1 as a cross-sector dispersion sensitivity, and stop deriving stars, filled dots and `main_inference` flags from it.

### F2. Pooled DK uses a finite-sample factor that contradicts the project's own single-series rule

**Medium, High if pooled DK becomes primary unchanged · convention inconsistency (design). Not an arithmetic bug: statsmodels applies its documented "cluster" factor (`sandwich_covariance.py:869-872`, printed by `p05c_statsmodels_source.py`).**

- **Where.** Pooled factor: `scripts/estimate.py:49-50`, `scripts/heterogeneity.py:144`, `scripts/annual_analysis.py:326`; stated at `scripts/iip_report.py:122`. Single-series rule: `scripts/heterogeneity.py:318`, `scripts/visitor_analysis.py:287-288`, `notes/annual_methods.md:32`, `notes/covid_comparison_methods.md:21`. t(T−1) reference: `scripts/estimate.py:50`, `scripts/heterogeneity.py:176-182`, `scripts/aggregate_va.py:199,212,216`, `scripts/visitor_analysis.py:291-292`.
- **Replay.** `p01_cr1_dk_decomposition.py`; `p04_ledger_flags.py`; `p11_outcome_type_size.py`.
- **Output.**

```
[identity 3] uncorrected pooled DK(4) SE = 1.220992 ; uncorrected mean-series HAC(4) SE = 1.220992 ; |diff| = 6.31e-11
[correction] pooled T/(T-1)*(N-1)/(N-K) = 1.0541   single-series T/(T-K_time) = 1.4016   SE understatement factor = 1.153
pooled rows: ratio SE_series/SE_DK min/median/max: {'min': 1.038, '50%': 1.153, 'max': 2.565}
national cells: 102; p<0.05 with t(T-1): 9; with t(T-K_T): 3; lost: 6
```

The pooled panel and the national mean series carry identical information about a national regressor, yet the pooled factor counts N = S×T observations, so the year and season dummies cost almost nothing. Placebo rejection for pooled DK(h) is 11–16% for one-period growth and 25–30% in the 2020–22-excluded samples. The series convention rejects 5–7% and 8.5–13% in the same cells.

- **Consequence.** VA broad-11 full controls, −27.05: pooled DK p 0.0224, series p 0.107. IIP total quarterly YoY excluding 2020–22, −23.91: p 0.020 with t(T−1), 0.051 with the 5 residual degrees of freedom. Annual full-control pooled DK standard errors are 1.78–1.85× smaller than the series convention gives at T = 9.
- **Minimum correction.** One convention across families: T/(T−K_T) with a t(T−K_T) reference for any coefficient on a national regressor, pooled or not. Even that rule rejects 8.5–13% with 5 residual degrees of freedom, so those cells need an explicit warning or randomization p-values.

### F3. Equal-slope omnibus, pairwise and per-sector total-effect tests do not have their nominal size

**High · design limitation with claim inflation; the arithmetic reproduces.**

- **Where.** `scripts/heterogeneity.py:154-173` (F(q, T−1) at :172, availability by numerical rank at :159-163), `scripts/granular_interactions.py:68-82`, `scripts/annual_analysis.py:374-389,543-552` (rank bound T−2 at :380), `scripts/visitor_analysis.py:342-386`, `scripts/sector_temperature.py:9-14`. Claims: `README.md:37,77`.
- **Replay.** `p03a_h1_identity.py`; `p03b_h1_placebo.py 2000` then `p03c_tables.py`; `p08_annual.py 2000`; `p09_visitor.py 1000`; `p11_outcome_type_size.py`.
- **Output.** The total effects (temperature main effect plus sector interaction, with their joint covariance) are computed correctly, and sector contrasts do not depend on the shared controls or year effects at all:

```
specifications checked: 138
max |beta - saved|               : 2.2773747332394123e-08
max |contrast difference| when controls AND year effects are deleted from the model: 9.094947017729282e-13
```

So every equality and pairwise test is identified only from sector-difference time series regressed on temperature and season, and the average total effect is the common-slope coefficient of F1. The tests themselves:

```
H1 omnibus available in 118 of 138 spec/outcome/bandwidth cells; nominal p<0.05 in 98.
Linear calendar trend used as 'temperature': nominal p<0.05 in 97 of 118 cells; F_trend >= F_H1 in 96 cells.
62 growth-outcome H1 omnibus cells have nominal p<0.05; randomisation p<0.05 on the weather side in 3, on the outcome side in 8, on both in 3.
```

Placebo rejection of a true equal-slopes null at nominal 5%, across the two placebo sides:

| Test | Restrictions q, periods T | Placebo rejection |
|---|---|---|
| VA detailed 19, long history, QoQ, 4 lags | 18, 82 | 0.84 |
| VA detailed 19, full controls | 18, 29 | 1.00 |
| VA broad 11, long history, QoQ, 4 lags / 8 lags | 10, 178 | 0.15–0.17 / 0.28–0.33 |
| IIP detailed 18 monthly, long history, 12 lags / 24 lags | 17, 357 | 0.33–0.36 / 0.72–0.75 |
| IIP broad 6 monthly, long history | 5, 405 | 0.06–0.07 |
| Annual VA 19 (NSA), 1 lag | 18, 20 | 1.00 |
| Annual IIP 18, 1 lag | 17, 29 | 0.99 |
| Visitor country profile | 41 | 1.00 |

By outcome type the average is 65% for one-period growth, 75–79% for YoY, 77–80% for log levels and 98–99% for raw levels. Holm-adjusted pair families have an average family-wise error of 21–43%. Under the placebo, the probability that at least one sector's total effect is "significant" runs from 15% to 98%.

The annual code admits a zero-degrees-of-freedom case. Annual VA 19 (NSA) tests q = 18 restrictions from T = 20 years, exactly at the rank bound T−2. It is reported "available" with F = 232,300, p = 5.8e-47 and condition number 3.1e9, while its seasonally adjusted twin is "unavailable" at rank 17. Its randomization p is 0.034–0.063. Annual IIP 6 with full controls (T = 9, 2 residual degrees of freedom) reports F = 12.4, p = 0.0013; the placebo rejects 71–75% and the randomization p is 0.37–0.42.

- **Consequence.** Claims that sectors differ in temperature sensitivity are not supported at the stated levels. In the saved sector ledger 348 of 480 omnibus tests (72.5%) are below 5%, which is what the placebo produces with no heterogeneity at all. The three growth-outcome cells that survive randomization on both sides are all VA detailed-19 long-history YoY, and there a linear calendar trend in place of temperature gives a larger F (60.8 against 18.6): the test is picking up sector-specific trends.
- **Why.** The q×q covariance is estimated from at most T−Q−1 time blocks (the code's own comment at `scripts/granular_interactions.py:69-72`), but the reference is F(q, T−1).
- **In fairness.** With few restrictions and a long sample the test is close to nominal (IIP broad 6 monthly, 6–7%). `README.md:77` already calls short-sample omnibus p-values fragile; the placebo shows the problem extends to the long-history samples.
- **Minimum correction.** Report omnibus and pairwise results with year-block randomization p-values, or as descriptive only. At the least, mark the test unavailable unless q is small relative to the residual time degrees of freedom, and do not let numerical rank at a knife-edge decide availability.

### F4. The README "default baseline" is a 2020 coincidence

**High · interpretation and label; the estimates are computed correctly.**

- **Where.** Label at `README.md:50`. It contradicts the warning at `README.md:83` and the degrees-of-freedom statements at `README.md:27,68`.
- **Replay.** `p06_fullcontrol_influence.py`.
- **Output.** The full-control sample has 29 quarters in 2017Q1–2025Q4, 10 of them dated 2020–22. After partialling out controls and fixed effects, identifying temperature variation has a standard deviation of 0.112 °C.

| Series, outcome (full controls) | β all dates | Share of β from 2020–22 dates | β dropping 2020 only (series p) | β dropping 2020–22 |
|---|---|---|---|---|
| VA broad 11, QoQ | −27.05 | 0.817 | −1.479 (0.549) | −0.031 |
| VA detailed 19, QoQ | −40.52 | 0.820 | −2.175 (0.445) | 0.000 |
| VA aggregate, QoQ | −13.84 | 0.814 | 0.234 (0.808) | 0.043 |
| VA broad 11, YoY | −7.777 | 0.843 | 0.607 (0.766) | −0.359 |
| IIP detailed 18 quarterly, QoQ | −18.93 | 0.816 | −3.399 (0.585) | −1.593 |
| IIP total quarterly, YoY | −17.08 | 0.415 | −14.34 (0.141) | −23.91 (0.051) |

For VA broad 11, 2020Q2 alone contributes −11.09 of the −27.05 (leverage 0.1975), 2020Q3 contributes −6.46, and 2020 as a whole contributes −20.05.

- **Consequence.** The headline full-control coefficient is the 2020Q2 collapse (mean growth −20.9, temperature residual +0.27 °C) and the 2020Q3 rebound (+11.9, residual −0.22 °C). A 1 °C change is about nine standard deviations of the identifying variation, so the magnitude is an extrapolation as well. The only series that does not hinge on 2020 is IIP total quarterly YoY, and it rests on 12 residual degrees of freedom, falls to −11.12 when 2025 is removed, and has a monthly counterpart of +0.05 (p 0.98).
- **Minimum correction.** Remove the "default baseline" label from the 2017–2025 all-dates full-control sample. Show it only beside the drop-2020 and excluded results with the leverage statement. Take the long-history temperature–rain–humidity specification as primary.

### F5. Signs and sizes are unstable across samples and frequencies

**Medium · interpretation.**

- **Where.** `README.md:20,68,83`; saved ledger `outputs/covid_comparison/common/estimates.csv`.
- **Replay.** `p04_ledger_flags.py` (Part 3); `p06_fullcontrol_influence.py`.
- **Output.**

```
paired cells: 218; sign differs between samples: 46
main-inference p<0.05 in all-eligible: 30; in excluded: 11; in both with the same sign: 5
```

The median ratio of |β excluded| to |β all dates| is 0.51. Full-control IIP YoY is −17.08 quarterly against +0.05 monthly (total), −11.76 against +2.66 (broad 6) and −8.45 against +2.18 (detailed 18). VA detailed-19 long-history QoQ is −3.78 on all dates and +1.22 excluding 2020–22, and the positive value is nominally significant under all three conventions (0.023, 0.017, 0.045).

- **Consequence.** There is no stable sign to interpret. Any single cell quoted alone is selective.
- **Minimum correction.** Wherever one estimate is quoted, show its all-dates/excluded and quarterly/monthly counterparts beside it, and use no directional language.

### F6. The sector-CR1 heterogeneity sensitivity prints p-values that carry no per-sector information

**High · label and claim defect on top of a design limitation; the arithmetic reproduces. The user requested this sensitivity, so the fix is to the display, not the computation.**

- **Where.** `scripts/clustered_sector_sensitivity.py:88-101` (degenerate only when the variance is at most 1e-12 of the reference), `:306-309,320-328,361` (stars and the "Sector CR1 p / significance" column), `:288,358` (limitation acknowledged), `README.md:39`.
- **Replay.** `p12_cluster_h1.py 1000`.
- **Output.**

```
sector-CR1 inference_status: {'available': 1170}
rank of the G x G sector-CR1 covariance of the sector slopes (summary.csv): {1: 90}
within a model/outcome cell, max over sectors of SE_CR1 / min SE_CR1 - 1: max over the 90 cells = 7.99e-13
cells: 90; max |beta - saved| = 3.75e-08; max |formula/saved - 1| = 2.62e-11; K agrees in 90 cells
within a cell the CR1 p-value is a monotone function of |beta_i| alone: Spearman(p, -|beta|) min over cells = 1.000000
```

With clusters equal to sectors, each sector's own-temperature scores sum to zero inside its own cluster. The G×G covariance of the sector slopes is therefore a constant matrix of rank one, and every sector in a table has the same standard error:

`SE = sqrt( G/(G−1) × (N−1)/(N−K) × Σ_j (b_j_own − β_j_H1)² ) / G`

where b_j_own is sector j's slope from its own separate regression. The formula reproduces all 90 saved tables to 2.6e-11. That number measures how far the slopes move when shared control and year coefficients are made sector-specific. It is not the sampling variability of any sector's slope.

Of the 1,170 displayed CR1 p-values, 492 are below 5% against 338 for DK(h), and 249 are significant under CR1 only. Under the placebo, 28–32% of sectors are "significant" on average and at least one is in 77–82% of draws.

- **Consequence.** A reader sees a starred "Sector CR1" column beside the DK column and reads agreement as robustness. `README.md:39` says "Degenerate inference is explicitly flagged and has no invented p-value"; that holds only for the 90 joint tests.
- **Minimum correction.** Treat a rank-one absolute covariance as degenerate: suppress per-sector CR1 p-values, intervals and stars, or print the single shared standard error once with the limitation. Correct `README.md:39`.

### F7. Annual family: the main rule and the nine-year fits are badly sized

**Medium · design limitation; no annual common-slope claim is significant under the main rule, so no headline is a false positive.**

- **Where.** `scripts/annual_analysis.py:306-328,334-335,363,374-389,464,594-596`; `notes/annual_methods.md:24,32`; `README.md:15`.
- **Replay.** `p08_annual.py 2000`.
- **Output.**

```
requested specifications: 512 | {'estimated': 384, 'unavailable': 128}
pooled cells: 24; CR1 [main] p<0.05: 0; DK(1) t(T-1) p<0.05: 2; series HAC(1) T/(T-K_T), t(T-K_T) p<0.05: 2; main-significant but not series-significant: 0
single-series cells: 12; DK(1) t(T-1) [main] p<0.05: 0; same SE with t(T-K_T): 0
```

Anchors reproduce the saved annual ledgers to 8.9e-08. Placebo rejection for the main sector-CR1 rule on weather-only, all-dates panels is 27–28% (VA 11), 51–52% (VA 19), 32–33% (IIP 18) and 12–13% (IIP 6). In the nine-year full-control fits, DK(1) rejects 19–25% for single series and 32–44% for pooled panels; the series convention rejects 5–10%.

- **Consequence.** The rule labelled main is the worst-calibrated one in this family, so neither its rejections nor its non-rejections are informative. The annual omnibus claims are covered by F3. The 128 unavailable requests are correctly unavailable.
- **Minimum correction.** As F1 and F2. Present the nine-year full-control fits as descriptive only.

### F8. Visitor family: the origin-clustered comparison and profile omnibus are invalid for common weather

**Medium · design limitation and label. This family already uses the series convention as primary and hedges its text.**

- **Where.** `scripts/visitor_inference.py:36-48,61-65`; `scripts/visitor_analysis.py:287-292,304-306,342-386`.
- **Replay.** `p09_visitor.py 1000` (after `p09a_visitor_schema.py`).
- **Output.**

```
specifications anchored: 204 (core main 84, crisis 120)
origin CR1 vs HAC(12) on the 254 pooled models: CR1 p<0.05 in 102, HAC(12) p<0.05 in 29, CR1-only 80, HAC-only 7
main+crisis cells with nominal HAC(12) p<0.05: 14 of 204; with randomisation p(|t HAC12|)<0.05 on the weather side: 0; on the outcome side: 2; on both: 0
```

Origin CR1 rejects 19–90% under the placebo. The primary HAC(12) rule averages 9–10% for growth outcomes but 13–27% for YoY growth in the clean and pre-2020 samples, and it under-rejects (0.3–3%) in the long samples that keep the COVID collapse. The country-profile omnibus rejects 100%. The aggregate full-clean headline of −4.076 (p 0.039) is significant in 1 of 6 leave-one-year fits and has randomization p 0.07–0.11.

- **Consequence.** No visitor temperature result survives. The origin-CR1 column makes 80 extra models look significant and must not be read as corroboration.
- **Minimum correction.** Remove p-values from the origin-CR1 comparison or label it invalid for common weather. Drop the q = 41 omnibus. Flag YoY HAC p-values in gapped clean samples as fragile.
- **PPML.** I checked saved artefacts only: 50 zero cells of 9,450 and 46 of 4,200, p-values 0.66–0.93, no claim at stake.

### F9. Multiplicity and transformation-dependent size inflate what the stars suggest

**Medium · claim inflation.**

- **Where.** `scripts/significance.py:21-23`; ETable settings at `scripts/regression_tables.py:114-115` and the other report scripts; `README.md:43`.
- **Replay.** `p10_claims_multiplicity.py` (Part C); `p11_outcome_type_size.py`.
- **Output.** `total p-values counted above: 66,566; nominal p<0.05: 8,429` (ledgers overlap; this is descriptive). Stars start at p < 0.10 and only pairwise families are Holm-adjusted. Main-inference temperature rows are below 5% in 47 of 546 cells (8.6%), which is inside what the main rule produces under a true null. Under the series convention, pooled placebo rejection is 5–7% for one-period growth, 7–11% for YoY growth and 8–12% for log levels; for single series it is 7%, 11–14% and 12–14%.
- **Consequence.** The count of starred cells across the grid is not evidence. Overlapping YoY outcomes and levels are tested with the same bandwidths as one-period growth and over-reject more. `README.md:41` already states that stationarity is not established.
- **A nuance against my own case.** Per-sector total effects are below 5% in 26% of saved rows, more than the 10–18% placebo average. The trend diagnostic in F3 is the likely reason, and a year permutation cannot reproduce trends, so the placebo is if anything lenient here.
- **Minimum correction.** Declare a small primary family in advance (for example long history, one-period growth, national-series inference), adjust within it, label everything else exploratory, and drop the 10% star.

### F10. Two report-generation mistakes

**Low · computational mistakes in report text and tables; no estimate is affected.**

- **Where and output (a).** `scripts/report.py:126-129` selects control rows by model and term only, so `outputs/first_results.md:44-47` prints each control twice with no covariance label and with contradictory intervals:

```
44: | rain_100mm | -0.314 | [-0.844, 0.216] |
45: | rh_pct | -0.071 | [-0.241, 0.098] |
46: | rain_100mm | -0.314 | [-0.605, -0.023] |
47: | rh_pct | -0.071 | [-0.320, 0.177] |
```

Rows 44–45 are DK(4) and rows 46–47 are sector CR1. The rain interval that excludes zero is the CR1 one (p 0.037); under the series convention the rain p-value is 0.312.

- **Where and output (b).** `scripts/iip_report.py:122` writes into `outputs/iip_results.md:60` that "Inference uses12monthly or4quarterly Bartlett lags … P-values use t(T−1)", but the headline numbers at `outputs/iip_results.md:7,32` (from `scripts/iip_report.py:113-115`) are the sector-CR1 t(5) rows:

```
note headline: ('0.075521', '0.594038', '[−1.45, 1.60]', '0.903790.')
headline SE matches covariance row(s): ['Sector cluster'] | inference df: [5]
```

- **Replay.** `p10_claims_multiplicity.py` (Parts A and B).
- **Minimum correction.** Filter or label the control table by covariance. Make the inference sentence describe the rule actually printed.

### F11. statsmodels silently compresses calendar gaps in sparse samples

**Low for saved outputs, blocking for the new analysis · latent computational hazard.**

- **Where.** Every `cov_nw_groupsum` call site: `scripts/estimate.py:37-39,49` (whose comment says "It uses actual quarter indices, including empty periods"), `scripts/covid_common.py:160-161`, `scripts/aggregate_va.py:190-191,207-208`, `scripts/sector_temperature.py:11-12`, `scripts/outcome_analysis.py:184`, `scripts/iip_analysis.py:179`. Cause: statsmodels 0.14.6 `statsmodels/tools/grouputils.py:101-103`.
- **Replay.** `p05c_statsmodels_source.py`; `p05_group_sums.py`; `p05b_relabel_fullcontrol.py`.
- **Output.**

```
  101:         # re-label groups or bincount takes too much memory
  102:         if np.max(group) > 2 * x.shape[0]:
  103:             group = pd.factorize(group)[0]

sample files found: 1260; with a time column: 1260; without: 0
files where the relabel would be triggered: 0

fits: 48; with the relabel trigger: 8
  of these the project call equals the gap-compressed HAC in 8 and the calendar-aware HAC in 0
  range of SE(compressed)/SE(calendar) - 1 on triggered fits: -0.2104 to -0.0718
```

The trigger is (calendar span − 1) > 2 × observations. No saved sample meets it. The single-series quarterly full-control samples do once 2020–2023 are dropped (17 quarters over a span of 36). VA aggregate YoY then gets a standard error of 1.411 from the project's call against 1.582 calendar-aware (p 0.023 against 0.040), and IIP total quarterly YoY gets 7.607 against 9.489 (p 0.006 against 0.021).

- **Consequence.** None today. A base-clean crisis rule applied to short single-series samples would switch the HAC from calendar lags to observation-order lags with no warning.
- **Minimum correction.** Assert `time.max() <= 2 * len(time)` at each call site, or reuse the zero-filled calendar Bartlett sum the project already has at `scripts/heterogeneity.py:136-143`. The annual and visitor families use their own explicit calendar sums and are not exposed.

## Audit of the proposed crisis-window analysis

I ran the tentative windows exactly as given (1985–86, 1997–98, 2001–03, 2008–09, 2020–22) on the long-history specification: temperature, rain and humidity with sector×season and year fixed effects, for aggregate VA, VA broad 11, VA detailed 19 and IIP total, broad 6 and detailed 18 at both frequencies, plus the annual family. I did not tune any date. The leave-one-window results below are descriptive and must not be used to choose windows. Replay: `p07_crisis_windows.py 1000`; annual in `p08_annual.py 2000` (Part 5).

### Rules compared

| Rule | Drops | Status in the project |
|---|---|---|
| All dates (`R0`) | nothing | saved baseline |
| Date-only (`R1`) | observations dated inside a window | the existing 2020–22 rule in VA, IIP and annual code (`scripts/outcome_analysis.py:150`, `scripts/granular_analysis.py:95`, `scripts/aggregate_va.py:152-171`, `scripts/covid_common.py:212`, `scripts/annual_analysis.py:464`, `scripts/iip_analysis.py:127`) |
| Base-clean, own horizon (`R2`) | also observations whose growth base lies in a window | the visitor family's interval rule (`scripts/visitor_analysis.py:226-230,268-273`) |
| Base-clean, common (`R3`) | window years plus the following calendar year, for every outcome | not in the project; identical to `R2` for YoY and annual growth |

For log-difference outcomes, contamination enters only through the two endpoints, so an endpoint rule is sufficient. With whole-year windows the endpoint rule and the visitor "any intersecting interval" rule coincide. They would differ only for windows shorter than the growth horizon.

### Coverage and sample geometry

| Family | Sample start | Windows inside the sample | T all dates | T date-only | T base-clean QoQ / YoY |
|---|---|---|---|---|---|
| VA aggregate, VA broad 11 | 1982Q1 | all five | 178 | 130 | 125 / 110 |
| VA detailed 19 | 2006Q1 | 2008–09 and 2020–22 only | 82 | 62 | 60 / 54 |
| IIP total quarterly | 1984Q1 | all five | 171 | 123 | 118 / 103 |
| IIP broad 6 quarterly | 1993Q1 | four (no 1985–86) | 135 | 95 | 91 / 79 |
| IIP detailed 18 quarterly | 1997Q1 | four; the sample starts inside 1997–98 | 119 | 79 | 75 / 63 |

- Monthly IIP, in the same column order: total 513 / 369 / 364 / 309 months, broad 6 405 / 285 / 281 / 237, detailed 18 357 / 237 / 233 / 189.
- The retained sample breaks into up to six calendar segments. Base-clean rules leave 2000 as an isolated year between 1997–98 and 2001–03, and 1984 is isolated for IIP total.
- "Consistently across families" cannot mean the same windows: the detailed panels see only two to four of the five. Each table must state which windows bind.
- The wind and PM2.5 samples start in 2017 and contain only 2020–22, so a long-history crisis analysis is necessarily the temperature–rain–humidity specification.

### What the rules change

**Base contamination under the date-only rule.** Among retained observations, 12.9–20.3% of YoY growth rates have their base inside a window (20 of 130 for VA; 16 of 79 for IIP detailed 18), against 1.4–5.1% of one-period growth rates. These are the rebound observations. Descriptively, aggregate GVA YoY growth averages 8.08% in the first year after a window, 6.65% in other years and 2.35% in window years.

**Temperature coefficients by rule** (probe output: `Sign changes R0->R1: 18 ; R1->R2: 5 ; R1->R3: 8 of 27`):

| Series, outcome | All dates | Date-only | Base-clean own | Base-clean common |
|---|---|---|---|---|
| VA aggregate, QoQ | −0.676 | 0.231 | −0.200 | −0.577 |
| VA aggregate, YoY | 0.964 | −0.579 | −1.667 | −1.667 |
| VA broad 11, QoQ | −1.428 | 0.238 | 0.050 | −0.153 |
| VA broad 11, YoY | 1.255 | 0.424 | 0.047 | 0.047 |
| VA detailed 19, QoQ | −3.780 | 1.054 | 1.134 | 0.578 |
| VA detailed 19, YoY | −1.432 | −0.412 | 0.078 | 0.078 |
| IIP total quarterly, YoY | 1.947 | −2.688 | −4.968 | −4.968 |
| IIP detailed 18 quarterly, YoY | 2.205 | −2.121 | −4.061 | −4.061 |
| IIP total monthly, YoY | 1.454 | −1.624 | −2.768 | −2.768 |
| IIP detailed 18 monthly, YoY | 1.790 | −1.631 | −2.146 | −2.146 |

**What looks significant, and why not to trust it yet.** Under base-clean rules four seasonally adjusted YoY temperature coefficients are nominally significant by the series convention (their NSA twins have the same sign and similar size, and two of them are also below 5%): VA aggregate −1.667 (p 0.014; randomization p 0.025–0.036), IIP total quarterly −4.968 (0.030; 0.039–0.059), IIP detailed 18 quarterly −4.061 (0.013; 0.029–0.035) and monthly −2.146 (0.030; 0.083–0.093). Five things undercut them:

- They appear only under the base-clean rule and only for YoY (probe output: `p HACs(h) t(T-K_T) : cells with p<0.05 by rule -> R0: 0/27, R1: 0/27, R2: 6/27, R3: 6/27`). Date-only gives −0.579 (p 0.44) for VA aggregate, and QoQ gives −0.20 to −0.58 (p 0.26–0.74).
- The equal-weighted VA panels show nothing: 0.047 (p 0.92) for broad 11 and 0.078 (p 0.89) for detailed 19.
- They depend on the exact window list. For VA aggregate, keeping any one window in the sample loses the result: −0.581 (p 0.53) with 1997–98 kept, −0.682 (0.52) with 2008–09, −0.829 (0.35) with 1985–86, −1.247 (0.086) with 2001–03 and −1.266 (0.057) with 2020–22, against −1.667 (0.014) with all five dropped. The IIP results are lost whenever 1997–98 or 2008–09 stays in (IIP total quarterly −1.974 and −1.012; IIP detailed 18 quarterly −0.474 and −0.992).
- In the quarterly families that contain it, 1997–98 holds 4.5–6.7% of dates but 12.0–16.4% of the identifying temperature variation (9.0–12.4% monthly). Dropping it removes a disproportionately informative temperature episode, so the exclusion is not neutral with respect to the regressor.
- The annual version has the opposite sign: aggregate VA is +5.17 under date-only (series p 0.033; randomization p 0.048–0.051) and +2.50 under base-clean (0.021; 0.044–0.052), VA broad 11 is +3.74 and +2.43, and IIP broad 6 is +16.4 and +8.8. The quarterly specification identifies within years; the annual one has only a linear trend and identifies between years (`notes/annual_methods.md:24`). Opposite signs across the two are an identification warning, not a finding.

**Rain and humidity.** No rain coefficient is nominally significant under the series convention in any quarterly or monthly family under any exclusion rule. Humidity is negative and nominally significant only for IIP detailed-18 YoY (quarterly −0.966 date-only, p 0.001, randomization p 0.007; −0.791 base-clean; monthly −0.646 and −0.576). In the annual family humidity is positive and nominally significant (aggregate VA +0.78 and +0.46; IIP broad 6 +2.94 and +1.92, randomization p 0.001–0.015). The sign again flips with frequency.

**Multiplicity.** `Cells (family x outcome x rule R1-R3 x term) with nominal HACs p<0.05: 14 of 162; of these with randomisation p(|t HACs|)<0.05: 10`. Those 14 are 8 distinct results (four temperature, four humidity), because YoY cells repeat across the two base-clean rules.

### Inference on the gapped samples

Weather-side placebo rejection at nominal 5%, averaged over families:

| Outcome | Rule | Sector CR1 | Pooled DK(h) | Series convention |
|---|---|---|---|---|
| One-period growth | All dates | 0.200 | 0.056 | 0.041 |
| One-period growth | Date-only | 0.046 | 0.073 | 0.051 |
| One-period growth | Base-clean common | 0.043 | 0.076 | 0.054 |
| YoY growth | All dates | 0.243 | 0.083 | 0.064 |
| YoY growth | Date-only | 0.093 | 0.102 | 0.081 |
| YoY growth | Base-clean common | 0.069 | 0.111 | 0.084 |

- Sector CR1 looks better once crisis years are gone, because the common national shocks it cannot see are smaller there. It is still 10–16% for IIP detailed-18 YoY, and F1 applies unchanged.
- The series convention is the best-sized rule, at 4–5% for one-period growth and about 8% for YoY.
- The project's DK call is calendar-aware on every gapped sample I built (`max |project/calendar - 1| = 7.93e-11`). Treating retained observations as adjacent instead would move standard errors by −6.0% to +3.9%. The compression trap in F11 does not fire for these long-history samples, but it does for short single-series samples, so the assertion in F11 should be in place first.

### Consistency with the existing code

- The existing 2020–22 exclusion is date-only in the VA, IIP and annual families and interval-based in the visitor family. `README.md:20,68`, `notes/annual_methods.md:13` and `notes/covid_comparison_methods.md:13` admit the retained 2023 YoY bases. The new analysis should not inherit two rules.
- Growth rates and weather lags are built before dates are dropped (`README.md:20`), which is correct and should be kept.
- Windows are defined by realised output collapses. That is unavoidable for recession dating, but it changes the estimand to "the association in non-crisis periods", and it is legitimate only if the list comes from a source fixed independently of these regressions.

### Minimum pre-specification

1. One window list, taken from the official source the parent is checking, written down before any estimate is read.
2. One declared primary rule. I recommend base-clean common, because date-only leaves 13–20% of YoY observations with a crisis base and own-horizon base-clean gives QoQ and YoY different dates. Its cost is 20 more quarters and the isolated year 2000.
3. All four rules reported side by side for every family, with the base-contamination share and the list of windows that bind.
4. The same controls, fixed effects and bandwidths as the long-history specification, on actual calendar positions, with the F11 assertion.
5. National-series inference (series convention plus year-block randomization p-values) as primary; sector CR1 as a labelled sensitivity only.
6. No leave-one-window selection, a multiplicity adjustment over the family × outcome × term grid, and the 1997–98 leverage and the quarterly-versus-annual sign reversal stated beside any result.

## What the existing checks establish, and what they do not

**They do establish** (`replication/VALIDATION.md:3-15`, the 22 verifier programs and their receipts under `outputs/`):

- Numerical reproduction of coefficients, covariances and tests as specified, from a clean checkout, for 4,781 native fits. My independent anchors agree to 9e-08 or better.
- That the reproduction covers the analytical code at HEAD. It ran at commit 8bb9830, and `git diff --stat 8bb9830 e47edee -- scripts notes README.md` shows only two added README lines.
- Sample masks, source hashes and report wiring.
- That the 128 annual full-control excluded requests are unavailable for a real reason.

**They do not establish:**

- The size or validity of any inference rule: sector CR1 under common shocks, DK or HAC in short samples, F(q, T−1). The verifiers re-implement the same formulas, so they agree by construction.
- The appropriateness of the finite-sample conventions (F2).
- Causal identification, stationarity, robustness to influential periods or multiplicity control. `replication/VALIDATION.md:15` and `README.md:41,93` say so themselves.
- Correctness of narrative text. Both F10 defects passed every verifier.

## Unverified concerns

These are not findings. I have no probe evidence for them, or the evidence has a known gap.

- **PPML inference.** Checked on saved artefacts only; no placebo or size check was run.
- **Randomization p-values based on |β| in the full-control placebo.** The weather-side placebo keeps real wind and PM2.5, which breaks their collinearity with temperature and may bias |β|-based p-values. I rely on |t|-based values.
- **Kiefer–Vogelsang fixed-b text in `p01_output.txt`.** Those critical values were written from memory. Ignore those lines.
- **Official crisis dates.** I did not fetch any source; the parent is checking. All crisis results use the tentative list.
- **Seasonally adjusted outcomes near windows.** Official two-sided filters spread crisis information into adjacent periods, so SA growth just outside a window is not strictly crisis-free. Not probed.
- **Bartlett weighting of overlapping YoY outcomes.** If one-period growth were serially uncorrelated, Bartlett weights at 4 quarterly or 12 monthly lags would understate the YoY long-run variance by roughly 25–30%. That is illustrative arithmetic, not a probe result; the measured evidence is the higher YoY placebo rejection in F9.
- **Stationarity.** I did not replay the unit-root screens. The evidence here is the level-outcome over-rejection and the trend-as-temperature result.
- **Placebo exchangeability.** Year-block permutation assumes years are exchangeable and destroys across-year persistence on the permuted side. It is therefore lenient toward trend confounding.

## Coverage limits

- **Placebo-audited:** common-slope pooled and single-series models (192 pooled and 64 single-series cells over four sample windows and four outcome transformations), H1 and H3 heterogeneity (138 specifications), annual common and H1 models, visitor core and crisis models with profiles, the sector-CR1 sensitivity (90 tables) and the crisis rules.
- **Ledger-level only:** distributed-lag sums and YoY-weather variants are included in the 444-cell ledger comparison (F1, F2, F5) but have no placebo of their own.
- **Not audited beyond reproduction:** H2, R1 and H3X heterogeneity variants; PPML; rendered HTML (I read generator code and saved CSV/Markdown only; no visual inspection).
- **Scheme.** Results use the permutation scheme. The cyclic-shift scheme has only 19–43 distinct draws and is not relied on.
- **Scope file.** `scope.md` was readable in my first session only; later sessions worked from my notes of it.
- **Monte Carlo error.** Rates come from 1,000–2,000 draws; randomization p-values near 0.05 carry about ±0.007.

## Replay

Run from the worktree root. Tabulators read CSVs written by the placebo probes, so order matters where shown.

| Finding | Commands | Evidence in `audit_probes/evidence/` |
|---|---|---|
| F1, F2 | `p01_cr1_dk_decomposition.py`; `p02_placebo_common.py 2000`; `p02b_placebo_tables.py`; `p04_ledger_flags.py` | `p01_output.txt`, `p02b_output.txt`, `p04_output.txt` |
| F3 | `p03a_h1_identity.py`; `p03b_h1_placebo.py 2000`; `p03c_tables.py` | `p03a_output.txt`, `p03c_output.txt` |
| F4, F5 | `p06_fullcontrol_influence.py`; `p04_ledger_flags.py` | `p06_output.txt`, `p04_output.txt` |
| F6 | `p12_cluster_h1.py 1000` | `p12_output.txt` |
| F7 | `p08_annual.py 2000` | `p08_output.txt` |
| F8 | `p09a_visitor_schema.py`; `p09_visitor.py 1000` | `p09_output.txt` |
| F9, F10 | `p10_claims_multiplicity.py`; `p11_outcome_type_size.py` (after 02, 03b, 07, 09) | `p10_output.txt`, `p11_output.txt` |
| F11 | `p05c_statsmodels_source.py`; `p05_group_sums.py`; `p05b_relabel_fullcontrol.py` | `p05c_output.txt`, `p05_output.txt`, `p05b_output.txt` |
| Crisis audit | `p07_crisis_windows.py 1000`; `p08_annual.py 2000` | `p07_output.txt`, `p08_output.txt` |

`p01` prints a harmless `RuntimeWarning: invalid value encountered in sqrt`. Inventory probes `p00_inventory.py`, `p00b_schema.py` and `p00c_samples.py` only list files and schemas.

**No AEA-ready certification is given or implied.**

An identical copy of this review is saved at `audit_probes/REVIEW.md`. `git status --short` shows only the untracked `.venv` and `audit_probes/`; nothing was staged or committed, and no source, data, reference snapshot, saved estimate or report file was changed.