# Inference review: accepted fixes and qualified scientific claims

The completed provider receipt identifies `claude-fable-5-2`, maximum effort,
authenticated bounded isolated-write mode. The original report is retained
unedited. Its scope is the original e47edee release, not the later completed GVA,
literature or arrivals extension. We read the full review and distinguish the
following claims; we did not independently rerun its complete permutation grid.

## Three concrete closures

1. **Unlabelled control intervals and inconsistent IIP prose (F10): accepted.**
   `report.py` now labels every rain/humidity row by covariance, bandwidth, SE,
   p-value and reference df. `iip_report.py` now describes the sector-CR1/t(5)
   headline rule and the separately retained calendar alternatives. Both revised
   generators were run in an isolated copy. The portable
   `scripts/corrected_legacy_notes.py` route also renders corrected notes into
   `outputs/aea_audit/`, source-checks the four control
   rows, and verifies that the original release notes, HTML, native ledgers and
   manifest remain unchanged. The corrected Markdown is linked in the audit;
   the old release artifacts remain historical snapshots.
2. **Sparse-calendar statsmodels relabelling (F11): accepted latent hazard.**
   The old grouped-HAC helper can factorize sparse integer date labels and
   compress gaps when their range exceeds twice the row count. No original saved
   output was affected. The new `calendar_hac.py` retains the original
   score/bread and finite-sample formulas while matching lags on actual dates.
   Six narrow import migrations cover the old consumers. Its verifier checks
   685 saved sample calendars, 80 distinct row layouts and eight large-gap
   fixtures: existing-layout covariances are bit-identical and direct lag-pair
   calculations confirm noncompression in the sparse fixtures. The new heat and
   arrivals families already used explicit actual-calendar sums.
3. **Unsupported per-sector CR1 interpretation (F6): accepted display defect.**
   The preserved 90-fit raw sensitivity has rank-one absolute-slope covariance
   and collapsed sector-difference covariance. The numerical nonzero shared
   standard error comes from shared nuisance estimation, not independent
   uncertainty for each sector's temperature slope. The qualified sensitivity
   report keeps 180 native calendar-DK tables, prints each of the 90 algebraic
   shared CR1 SE values once, and suppresses individual CR1 intervals, p-values
   and stars. Raw CSVs and the original HTML remain clearly labelled archived
   downloads because the user requested that numerical sensitivity. The new
   viewer's source/runtime verifier checks all 180 views and original-byte guards.

The source-backed current receipts are respectively
`outputs/aea_audit/narrative_corrections.json`,
`outputs/calendar_hac/verification.json`, and
`outputs/clustered_sector_sensitivity_qualified/verification.json`.
These closures change reporting and future sparse-calendar handling; they do not
refit or replace the original saved economic estimates.

## Scientific concerns accepted with their assumptions

National weather is shared across sectors/origins. Conditional unit clustering
does not estimate uncertainty from omitted common national shocks. It is not
automatically invalid merely because a regressor is common: its independent-unit
error assumptions, few-cluster approximation and the scientific estimand matter.
The requested pooled sector-CR1 main rows are retained with these limitations and
calendar alternatives. Agreement between two p-values is not a calibrated
combined testing rule.

The pooled/mean-series coefficient identities and finite-sample convention
differences are useful diagnostics. They do not establish a uniquely exact HAC
Student reference. The independent design adjudication already reproduces 42
original static-model convention comparisons and selected common-shock/dispersion
identities. The separate new-methods adjudication provides stronger evidence:
a fixed-design, genuinely zero-coefficient IID Gaussian counterexample for an
unchanged current distributed-lag design. That selected failure is accepted and
prominent in the results, without extrapolating its rejection rate to every model.

Year-block permutations require exchangeability and disrupt temporal dependence,
trends and weather/control relationships. Permuting the complete weather block
against real Y does not isolate a temperature-only null when rain/humidity can
have effects. These permutation outputs are diagnostic sensitivities; the review's
blanket language about exact true-null size, absence of every visitor effect,
or universally valid randomization p-values is not established. Neither an
insignificant estimate nor a failed calibration experiment proves zero heat damage.

Pandemic leverage, different annual versus subannual identification, changing
partitions, short temporal residual df, YoY overlap and extensive multiple testing
remain material. The reported full-control baseline is the user's requested
control definition, not a significance-selected preferred causal model. Long,
event-excluded and strict-support results stay beside it. Crisis blocks are fixed,
explicit analyst sensitivities with official historical context; they are not an
official exact recession-dating series or all possible shocks. Strictly excluding
raw output support also does not undo every influence of a two-sided official SA
filter. NSA YoY and annual outcomes are therefore retained as primary comparisons.

The new arrival analysis additionally checks exact zeros/count aggregation,
shared-control H1 effects, static relative-effect identities, annual source
aggregation, matched origin-year conditioning and separate Stata re-estimation.
The original inference review predates these additions; their numerical and
presentation evidence must be read separately. No review or successful numerical
replay constitutes AEA acceptance, universal stationary data or causal identification.
