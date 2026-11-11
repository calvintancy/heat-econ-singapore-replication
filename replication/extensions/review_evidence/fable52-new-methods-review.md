**No numerical error found within the tested scope.** All 2,361 subannual and 2,428 annual fits replay from the saved designs to rounding, and the samples, histories, transforms and masks rebuild exactly from raw inputs. The confirmed defects are in the inference rules: three give badly wrong p-values in identifiable parts of the grid (findings 1–3). The share of significant rows is no higher than those rules produce under a true null, apart from some annual DJO1/DJO5 cells that I can attribute to neither effect nor artifact. This does not certify AEA readiness.

Served model: `claude-fable-5-2` (as reported by this session's environment).

## Scope

- **Audited:** `scripts/literature_sensitivity.py`, `scripts/annual_literature_sensitivity.py`, `notes/literature_specification_protocol.md`, `notes/crisis_sensitivity_policy.json`, and the saved outputs under `outputs/literature_sensitivity/` including `annual/`.
- **Written:** only `audit_probes/` (16 scripts, evidence in `audit_probes/evidence/`). `git diff --stat HEAD` is empty and nothing is committed.
- **Target integrity:** the four target files are untracked, so git cannot vouch for them; their modification times (20:25–20:31) precede my first probe (20:35).
- **Commands:** each probe runs as `.venv/bin/python audit_probes/<name>.py` from the worktree root. Evidence files carry the prefix of the probe that wrote them; `p4_view.log` is the readable summary of `p4_null_calibration.py`.
- **Not rerun:** the target scripts (they write into `outputs/`), so native PyFixest was not re-executed. The replay is independent NumPy.

## What reproduces

| Probe | Coverage | Worst discrepancy |
|---|---|---|
| `p1_calendar_history.py` | Weather aggregation, lag histories, growth transforms from raw official levels (20 contexts, 77,406 rows), 180 subannual and 504 annual samples, 12,024 mask rows | 2.3e-13; 0 mask or year-set mismatches |
| `p2_subannual_replay.py` | All 2,361 subannual fits: 89,784 effect rows, 48,090 test rows, 5,118 covariance matrices | coefficients 5.4e-10, covariance 4.2e-14, test F 3.3e-11 (relative) |
| `p5_annual_replay.py` | All 1,214 annual designs (2,428 fits): 79,800 estimate rows, 42,176 test rows, 5,376 covariance matrices | estimates, SE, CI, p 5.0e-12; test F 8.3e-08 relative |

These replays cover the H1 base-plus-interaction sum at each lag, the two-channel implied lags and scaling, cumulative sums and marginal effects with full covariance cross terms, and Bartlett weights on gapped calendars.

## Findings, most severe first

Notation: G sectors, T dates, q joint restrictions, N observations, K parameters. "Time df" is T minus the rank of the date-level part of the design (saved as `time_residual_df`). "True-null rejection" is the share of nominal-5% rejections when the saved design is kept and the outcome is replaced by Gaussian noise (22 designs, 400–2,000 draws each).

1. **Driscoll–Kraay (DK) and HAC joint tests are reported as available where their F(q, T−1) reference is invalid — computational bug (inference rule; the arithmetic reproduces as coded).**
   - The availability bound caps q near the number of dates, not by time df.
   - In five H1 designs, true-null rejection is 100% for "jointly zero" (q = 60 and 234), 36–99.5% for "cumulative responses equal", and 98–100% for the annual "response functions equal".
   - In the saved outputs, 110 subannual and 2,796 annual available tests have q above time df (1,314 of the annual ones are main inference); 100% and 75% of them show p < .05.
   - The simulated 100% cases have q at only 0.5–0.8 of time df, so those counts are the unambiguous subset, not the full extent.
   - Source: `scripts/literature_sensitivity.py:160` (also lines 172, 288, 321, 330); `scripts/annual_literature_sensitivity.py:102` (also 113).
   - Evidence: `p3_test_semantics.json`, `p6_annual_dk_tests.csv`, `p4_view.log`.

2. **Sector-cluster joint tests use F(q, G−1) without the Hotelling scaling — computational bug (reference distribution).**
   - This is the main inference for pooled common slopes.
   - Under independent Gaussian sectors, the exact size of a nominal-5% test is 30–33% (G=6, q=4), 59–62% (G=6, q=5), 68–71% (G=18, q=13) and 91% (annual G=12, q=11). Simulation gives 62%, 68% and 92.5%.
   - Multiplying the reported F by (G−q)/(G−1)·(N−1)/(N−K) and referring it to F(q, G−q) restores 5%.
   - Among saved rows with q > 1, that correction cuts p < .05 from 81 to 15 of 312 (subannual, non-dynamic) and from 171 to 112 of 460 (annual).
   - Single-restriction tests are unaffected (2.6–4.7%).
   - Source: `scripts/literature_sensitivity.py:172` (also 232–236); `scripts/annual_literature_sensitivity.py:113`; `scripts/annual_analysis.py:315`.
   - Evidence: `p3_sector_cluster_joint_tests.csv`, `p6_annual_test_semantics.json`, `p4_view.log`.

3. **The pooled DK factor T/(T−1)·(N−1)/(N−K) with reference df T−1 leaves date-level parameters uncorrected — computational bug (finite-sample correction).**
   - The pooled common coefficient equals the regression of the cross-sector mean series (to 1.6e-12). Its DK standard error is 0.69–0.96 (subannual) and 0.39–0.97 (annual) of what the code's own single-series rule, T/(T−K), gives for that same regression.
   - True-null t rejection is 8–13% with 131–345 dates, 22–24% in a 42-quarter strict sample, and 33–37% for annual DJO5/DJO10 with 20–34 years. Under the single-series rule it is 4–10%.
   - The same rule is the main inference for every H1 sector effect, about 67% of subannual and 74% of annual main-inference rows. There it rejects 6–10% per lag and 7–14% for lag sums.
   - Example: annual pooled DJO10 has 98 DK-significant rows as reported and 8 under the single-series factor.
   - Source: `scripts/literature_sensitivity.py:230` (also 237–240, 255); `scripts/annual_analysis.py:320`.
   - Evidence: `p4_identities.json`, `p4_view.log`, `p7_identification_and_significance.log` (section C).

4. **Sector-cluster standard errors cannot see national date shocks — scientific limitation of the chosen rule.**
   - With one national weather series, adding any common date shock to the outcome leaves the sector-cluster covariance unchanged to 1e-12, while DK temperature variances rise 32- to 853-fold (8 designs).
   - With noise calibrated to the saved cross-sector residual correlation, the sector-cluster t-test rejects a true null 12–41% of the time in 4 of 12 pooled designs and 0–5% in the other 8. That calibration understates national shocks, because the fit has already absorbed part of them.
   - Of pooled non-dynamic rows significant under sector-cluster, 20 of 90 (subannual) and 214 of 438 (annual) stay significant under DK with the single-series factor. These counts include the duplicated rows in finding 8.
   - Source: `scripts/literature_sensitivity.py:232` (also 255); `scripts/annual_analysis.py:315`; `notes/literature_specification_protocol.md:88`; `notes/crisis_sensitivity_policy.json:81`.
   - Evidence: `p4_identities.json`, `p4_view.log`, `p7_identification_and_significance.log`.

5. **The 179 smallest annual p-values (below .05/38,732) sit in 13 fit groups with at most 9 time df and hinge on 2010 or 2020 — scientific limitation, amplified by finding 3.**
   - 170 of them are in four overlapping 20-year (2006–2025) H1 DJO5 value-added panels with 5 time df and a t reference of 19 df.
   - On those designs the reported rule gives p < .05 for 10–12% of true-null rows under Gaussian noise and 17–20% under year-permutation and wild-bootstrap placebos. It produces p-values below that threshold roughly 150 to 10,000 times too often.
   - Dropping 2010 lifts every flagged row above the threshold in all four panels; dropping 2020 leaves at most 1 of 15 below it.
   - Against that, the saved panels sit in the upper placebo tail: only 6–12% of placebo draws match their p < .05 share and 0.4–5.5% their below-threshold share. Every flagged coefficient keeps its sign and p < .05 under every single-year drop.
   - Source: `scripts/annual_analysis.py:326`.
   - Evidence: `p7_identification_and_significance.log` (section D), `p8_annual_small_time_df.json`, `p9_annual_h1_placebo.json`, `p10_annual_h1_leave_year_out.csv`.

6. **Single-series Bartlett HAC over-rejects in short samples — scientific limitation.**
   - This is the main inference for aggregate and separate-series fits.
   - True-null t rejection is 4–5% at 174 quarters but 8–13% at 24–44 dates; joint tests give 7–8% versus 14–37%.
   - Year-on-year overlap in the noise raises the lag-sum t-test to 10% (174 quarters) and 20% (42 quarters).
   - Source: `scripts/literature_sensitivity.py:237`; `scripts/annual_analysis.py:320`.
   - Evidence: `p4_view.log`.

7. **Identifying variation is thin — scientific limitation.**
   - After fixed effects and the other weather terms, lag-sum contrasts rest on a residual temperature standard deviation equivalent to 0.05–0.09 °C (subannual) and 0.03–0.18 °C (annual).
   - The annual quadratic and its p5/p50/p95 marginal effects are identified from a p5-to-p95 spread of 0.9–1.4 °C over 13–44 years.
   - Evidence: `p7_identifying_variation_subannual.csv`, `p7_identifying_variation_annual.csv`.

8. **One restriction appears under two names — presentation ambiguity.**
   - In the two-channel model, "Cumulative direct temperature response zero" and "Exact contemporaneous static-output-level restriction" both test the trailing-sum coefficient (2,052 of 2,052 pairs identical).
   - In annual DJO1, "Temperature coefficient sum zero" and "Static contemporaneous annual level channel" coincide (3,204 of 3,204).
   - The algebra is right, but it contradicts the protocol's blanket statement that a zero lag sum is a different restriction.
   - Benchmark and DJO0 also repeat each estimate as both a "lag" and a "lag-sum" row, which inflates significance counts.
   - Source: `scripts/literature_sensitivity.py:297`; `scripts/annual_literature_sensitivity.py:225`; `notes/literature_specification_protocol.md:41`.
   - Evidence: `p3_test_semantics.json`, `p6_annual_test_semantics.json`.

9. **The protocol's "visibly flagged" claim is not met — presentation ambiguity.**
   - Affected tests keep status "available" with ordinary p-values. The only flags are numeric CSV columns (`time_residual_df`, `covariance_condition`, `structural_rank_bound`).
   - Covariance condition numbers among available tests reach 1.5e6 (subannual) and 3.4e9 (annual).
   - Nothing in this checkout renders these outputs.
   - Source: `notes/literature_specification_protocol.md:91`; `scripts/literature_sensitivity.py:162`; `scripts/annual_literature_sensitivity.py:102`.

10. **The policy file describes a different strict mask — presentation ambiguity, minor.**
    - `notes/crisis_sensitivity_policy.json:61` (also line 79) states the core rule: t−h..t with the same dates for all outcomes.
    - The literature script masks t−h−1..t for short growth and t−2h..t for YoY (`scripts/literature_sensitivity.py:76`), as the protocol documents (`notes/literature_specification_protocol.md:32`).
    - The masks themselves match the protocol (0 mismatches in 12,024 rows, `p1_calendar_history.json`). The copy at `outputs/literature_sensitivity/policy.json` is byte-identical to the policy file.

11. **Not a confirmed defect.** The label "Exact contemporaneous static-output-level restriction" is also applied in the dynamic (lagged-outcome) specifications (`scripts/literature_sensitivity.py:181`, used at line 302), where it is no longer the exact static-level implication. This is reasoned from the code, not tested numerically.

## Not tested

- End-to-end rerun of the target scripts and native PyFixest execution.
- The old full grid, earlier literature searches, and the core crisis and complete-control outputs (read for context only).
- The protocol's literature citations against the cached sources.
- Dynamic specifications beyond the arithmetic replay: no null simulation and no finite-T bias assessment.
- Stationarity, anywhere.
- True-null rates outside the 22 simulated designs and the 13 small-time-df annual fit groups. The placebo and leave-year-out probes cover seven and four annual H1 DJO5 panels.

## Defensible conclusions

1. The saved numbers are what the code computes, and the samples, calendar histories, NSA/SA transforms and strict masks are what the protocol describes.
2. Lag sums are sums of direct conditional coefficients. Nothing divides by one minus the AR sum, so no long-run multiplier is implied.
3. Subannual: 7.3% of 43,632 main-inference rows show p < .05, against 4–14% true-null rejection for the same rules in the tested designs. That is no robust evidence of an association; with 0.05–0.3 °C of identifying variation it is also not evidence of no effect.
4. Annual: 15.3% of 38,732 main-inference rows show p < .05, inside the 7–20% the DK/HAC rules produce under nulls and placebos. DJO1/DJO5 cells run higher (20–34%) and cannot be attributed to effect or artifact (findings 4 and 5).
5. H1 joint heterogeneity and equality tests are uninformative as reported (finding 1).
6. Most pooled "significant" rows are significant under only one of sector-cluster and DK. A claim needs both.
7. Everything is a post-exploration association; the original papers' fixed effects and country panels are not replicated (`notes/literature_specification_protocol.md:84`).

## Remaining threats

- Omitted national shocks: there is one weather series, and the annual models have no year effects.
- Few effective time df, HAC small-sample bias and YoY overlap, all pushing toward false positives.
- Leverage of 2010 and 2020 in the short panels.
- Multiple testing across 82,364 heavily dependent main-inference rows.
- Dynamic specifications: finite-T bias is acknowledged in the protocol (`notes/literature_specification_protocol.md:25`), uncorrected, and untested here.
- No stationarity check; the quadratic rests on about 1 °C of range; equal-weight sector panels are not national output.