# Adjudication of the Fable 5.2 design review

The selected numerical claims reproduce. The strongest general claims about invalid inference, true-null rejection rates and absence of effects require qualification. This pass does **not** change the user-requested sector-CR1 main inference, the full-control baseline, or any analytical file.

Scope is Fable's isolated commit `e47edeefeb750fd8a64106658d9351783db6e7fa`, not the later corrected literature generations. Four NumPy-only probes were replayed; no native fit, producer or large simulation grid ran. All **7,718** checked lane source/data/result/reference/original-probe-evidence files retained their bytes, and its Git status stayed unchanged. The Fable review itself has SHA-256 `34a4ac7b1a5828c4fc17737f2f388722708730c0ffef1e686469ae7b43c2e170`.

## Accepted

1. **Numerical reproducibility of the tested subset — computational check.** `p02_validate_lib.py` reproduces all 36 saved long-sample growth fits: maximum coefficient error `1.03e-11`, maximum standard-error error `1.16e-10`. This supports the review's distinction between computational correctness and design limitations; it does not cover every fit in the project.

2. **National-regressor information and panel/time equivalence — design fact.** Across 42 comparisons in `p03_time_level_equivalence.py`, pooled coefficients equal those from the equal-sector-mean time series to `1.96e-12`; uncorrected calendar Bartlett HAC standard errors match to relative `2.29e-10`. Repeating the same national weather series across sectors does not create independent weather draws. This exact equivalence relies on these balanced samples, common weather/control regressors and the stated fixed-effect design; it is not a theorem for arbitrary unbalanced panels or sector-specific weather designs.

3. **CR1 dispersion identity — design fact with an assumption boundary.** All eight cases in `p17_cr1_identity.py` verify
   `SE_CR1 = sd(sector time-series slopes)/sqrt(S) × sqrt((N−1)/(N−K_panel))`
   against saved CR1 errors to `8.94e-13`. Thus a common temporal residual component that shifts every sector slope together is not captured by cross-sector dispersion. Sector CR1 still has an interpretation under independent sector errors conditional on regressors, with additional few-cluster concerns; a national regressor alone does not prove those assumptions false. Keep the requested main inference, label those assumptions, and retain calendar inference as sensitivity.

4. **YoY coefficient decomposition — algebra, not causal attribution.** `p06_dynamics.py decomp` verifies all 12 SA cases to `1.11e-14`: on the identical design/sample, `b_YoY = b_short + b_pre`. VA11 is `1.2551 = −1.4279 + 2.6830`. The current-temperature-only YoY coefficient cannot automatically be called a current-period causal effect. The pre-period term is an association: persistence of earlier weather, omitted dynamics, common shocks or noise can produce it. Its ratio to the YoY coefficient is not a causal share and becomes unstable when that denominator is near zero.

## Qualified

5. **F2's finite-sample sensitivity is real; “honest inference” is not a unique reference law.** The replay confirms ten nominal project-DK 5% results cross that threshold under the alternative `T/(T−K_time)` factor and `t(T−K_time)` reference. For VA11 full-control short growth, p changes from `.0224` to `.1068`; for long SA YoY, `.0442` to `.0812`. These are consequential robustness diagnostics. With serial correlation/heteroskedasticity and estimated HAC covariance, neither chosen Student reference is an exact finite-sample certificate. The alternative convention is not proof that the documented project convention is an arithmetic error or that all main inference must be replaced. See `_lib.py:232–248` and the review at lines 12 and 177.

6. **F3/F7 leverage, sample and frequency sensitivity warrant disclosure, not a causal verdict.** The review's saved leave-quarter/year-out evidence supports investigating pandemic leverage and limited residual-temperature support. Those leave-out probes were not replayed in this small pass. “It is not a heat effect” (`fable52-design-review.md:211`) goes beyond those diagnostics: sensitivity does not identify whether the association is confounded, heterogeneous or causal. Annual-trend and quarterly-with-year-FE coefficients also target different variation and transformations, so opposite signs are not an algebraic contradiction. Retain the authorized full-control baseline with its short-sample qualifications and the declared long-history/crisis sensitivities.

7. **F6 exposes window sensitivity, but does not establish outcome-driven selection.** The code explicitly says the windows were externally supplied (`_lib.py:38–39`, `p18_window_multiverse.py:1–6`). The extreme rank in its grid is a useful warning, not proof that the analyst chose windows because of the result. A fresh calendar-only enumeration also finds only **7 distinct retained calendars** among the 62 nominal combinations for VA19 and matched VA11, and 31 for several IIP cuts; many windows do not bind in shorter histories. The combinations and overlapping outcomes/sector cuts are not 62 independent opportunities everywhere. Record the post-exploration protocol and show declared variants rather than present this rank as a selection probability.

## Not established by these probes

8. **“Reject a true null 20–25% of the time” is too strong.** `p07_placebo_size.py:3–11,32–50,73–84,108–117` jointly permutes temperature, rain and humidity year-blocks against fixed observed outcomes, restricts to complete years, and uses assumed exchangeability/cyclic invariance. It does not generate a known temperature-only zero-effect outcome process. A conditional temperature null can allow nonzero rain/humidity effects; jointly permuting those controls is not automatically a randomization test of that null. Warming, evolving within-year patterns and multi-year weather dependence also challenge exchangeability. The stored frequencies are **surrogate-weather permutation rejection frequencies under declared assumptions**, not certified sampling size for Singapore or proof that every nominal result is false. The original review's later exchangeability caveat does not make its headline unconditional statement valid. This simulation grid was inspected, not rerun.

9. **The two truncated lag regressions need not have equal current coefficients.** The review at line 237 treats differing current coefficients in short-growth DL0–4 and YoY DL0–4 as an inconsistency. If short growth has a genuine DL0–4, summing four consecutive short-growth equations generally produces a YoY DL0–7, plus appropriately accumulated other regressors. Truncating both equations at lag4 can give different projections without violating the exact growth identity. This comparison cannot establish a data-construction error or falsify all dynamic heat effects.

10. **“Nothing survives honest inference”/“no significant claims in any family” is not an all-project conclusion.** The narrow historical static common-slope comparisons and their convention sensitivity are supported; the headline at lines 7–13 exceeds that scope. The review itself excludes many levels, profiles, stationarity/reconciliation checks, raw retrieval and native logs, and its literature section relies on memory/repo notes rather than checked primary full texts (lines 490–508). It does not audit the later distributed/dynamic, annual, seasonal or full-control companions. Likewise, F8's shared controls/year effects cancelling in *relative* sector contrasts does not mean the absolute H1 model omitted them; and its placebo-size claims inherit item8's assumptions. Aggregation, exclusion-support and multiplicity concerns remain useful design cautions, not a certificate that all effects are zero.

## Archived diagnostic and replay

`time_level_vs_project_DK_CR1.csv` is the exact 42-row diagnostic for audit HTML/download use. Label it **historical static-model finite-sample-convention sensitivity**, not a replacement set of baseline p-values or a uniquely valid test. SHA-256: `073dff9dd87d065962ddd5f0a666681b2b25f5e565ea3979748e1be2f5f5cc61`.

`CR1_dispersion_identity.csv` SHA-256: `45378fccc9596374136b4df9f7d53844bc9b9a61f7477225deb80a47cbb752ad`.

`SA_YoY_decomposition.csv` SHA-256: `4003b49a443fe10865c59e3867df4189fca4d791d538514e2e263f3111029152`.

All three are byte-identical to the corresponding original Fable evidence. Commands from the current repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 dist/aea_audit_20261111/native_claims/fable52_adjudication/run_replays.py
PYTHONDONTWRITEBYTECODE=1 uv run --locked python dist/aea_audit_20261111/native_claims/fable52_adjudication/summarize.py
```

`run_replays.py` executes the lane's `.venv/bin/python` and its unchanged probes with evidence redirected here. `replay_receipt.json` records the exact per-probe commands, source/output hashes, successful exits and full before/after inventory proof; `summary.json` records numerical maxima and mask counts. No analytical correction or inference-policy change was made.
