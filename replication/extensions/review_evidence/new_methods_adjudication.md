# New-methods Fable 5.2 adjudication

**A demonstrated joint-inference calibration failure applies to an unchanged current design.** This is stronger than a generic small-sample warning. It does not imply an error in native coefficients/covariance arithmetic, establish that the observed effects are zero, or justify replacing every reference distribution by a new universal rule. The user-requested sector-CR1 main inference and all analytical/reference bytes remain unchanged.

The exact provider receipt is authenticated, complete, exit0 and model-verified as `claude-fable-5-2`; its stdout result matches the review text. The sandbox is the `e47edee` Git baseline plus untracked extension source snapshots, whose individual hashes are recorded. Git HEAD alone does not identify those extension versions. Four semantic/history probes were replayed, along with independent identities and one original fixed-design null experiment. No native model was fitted.

## Accepted: current-design known-null failure

The saved **VA12, NSA YoY, long distributed-lag H1** design has **174 quarterly dates**, 12 sectors and 116 residual time degrees of freedom. The all-temperature-slopes-zero hypothesis has **60 restrictions**. Holding its exact X fixed and replacing Y by iid standard Gaussian noise makes every coefficient zero under the stated conditional model. Replaying the original case with seed **20261119**, 2,000 draws per original noise process and chunk250 gives the following results, also reproduced by a **separate portable implementation using only current packaged outputs**:

| IID Gaussian null; nominal 5% | DK bandwidth | Restrictions | Available draws | Rejections |
|---|---:|---:|---:|---:|
| All sector temperature-lag responses zero | 4 | 60 | 2,000 | **2,000 (100%)** |
| All sector temperature-lag responses zero | 8 | 60 | 2,000 | **2,000 (100%)** |
| All sector cumulative direct responses equal | 4 | 11 | 2,000 | **712 (35.6%)** |
| All sector cumulative direct responses equal | 8 | 11 | 2,000 | **1,091 (54.55%)** |

This is a genuine known-null calibration counterexample for that fixed design and noise process, not a weather-exchangeability permutation claim. `mathematical_proof.json` binds **X, Y, time, periods and units** to the **current** archive by exact array equality and hashes. The current design therefore remains relevant despite later raw-history/rank corrections. `decisive_launch.json` records the executed script, original calibration/helper hashes, successful exit and evidence hashes.

**Consequence:** these computable multi-df Wald p-values cannot be used as calibrated headline significance for this design. Present them as nominal/asymptotic diagnostics with the demonstrated failure visible. Numerical covariance invertibility is not finite-sample calibration. In particular, `q=60 < time_df=116`: substituting `q≤time_df` for the valid score-rank bound would not fix this counterexample. The general calendar-score bound remains T−1, with actual numerical rank checked; it is not a coverage guarantee.

The replay does not calibrate every design or noise process, and it does not cover dynamic regressors reconstructed from the newly drawn outcomes. Observed GDP is not shown to follow the iid noise null. The result establishes a failure under a simple legitimate null, not absence of an observed association.

## Accepted under explicit assumptions: CR1/Hotelling diagnostic

For three balanced non-dynamic common-regressor designs, an independent multivariate check reproduces the weather-coefficient CR1 covariance as

`c0 × sample_cov(unit coefficient vectors)/G`, with `c0=(N−1)/(N−K)`,

to maximum absolute error **2.27e−12**. If those coefficient vectors are **iid multivariate Gaussian with homogeneous mean/covariance**, Hotelling's exact law gives

`F_alt = F_report × c0 × (G−q)/(G−1)`, referenced to `F(q,G−q)`.

| Selected joint null | G / q | Reported CR1 p | Gaussian Hotelling p | Exact rejection probability of the reported 5% rule under that Gaussian model |
|---|---:|---:|---:|---:|
| IIP6 quarterly, all DL coefficients zero | 6 / 5 | .0473 | .6089 | 61.53% |
| IIP18 monthly, all DL coefficients zero | 18 / 13 | .00158 | .3498 | 70.65% |
| Annual VA12 DJO10, all temperature coefficients zero | 12 / 11 | 1.80e−7 | .3583 | 91.45% |

These are material few-cluster/high-q reference failures under the specified Gaussian benchmark. They are **not a universal Hotelling correction** for dependent or heteroskedastic sector vectors, heterogeneous sector means, arbitrary unbalanced panels, or dynamic Y-derived X. Scalar restrictions also retain the conventional c0 correction; they are not literally numerically unaffected by removing it. The user's main CR1 rows stay in place, with this conditional finite-sample sensitivity available beside them.

## Accepted identities; finite-sample convention qualified

The original `p4_identities.py` replay confirms eight pooled/time-series identities: maximum weather-coefficient difference **1.62e−12**; adding a common date shock changes the sector-CR1 covariance by at most relative **1.18e−12**, while the median temperature DK variance rises by factors **32.43–852.71**. This demonstrates what sector clustering does not estimate when common shocks remain; it does not identify the actual size of omitted shocks in GDP.

`inference_comparison.csv` contains **12 observed DK/time-convention comparisons** plus nine CR1/Hotelling comparisons and four IID joint-null rows. Example: annual VA12 DJO10 current-temperature DK1 p is `.0151` under the reported factor/reference and `.1948` with the equivalent mean-series `T/(T−K_time), t(T−K_time)` convention. The coefficient and raw HAC are unchanged. This is a useful convention sensitivity. Unlike the demonstrated null failure above, the factor/reference comparison alone does not establish a uniquely exact HAC Student law. The review's blanket classification of every such difference as a computational bug is too broad.

## Confirmed interpretation/presentation defects

- **Dynamic static-level label:** with stable AR polynomial Φ(L), a structural instantaneous level channel generally implies `B(L)=γ[1−Φ(L)](1−L^h)`. A concrete AR1 example (`γ=2`, `φ=.4`, `h=4`) has temperature coefficients `[2,−.8,0,0,−2,.8]` at lags0–5. A β-only restriction truncated at h is therefore only a **conditional direct-weather shape**, not an exact structural static-output-level test. A display-only qualification is appropriate; no new structural test or coefficient change is implied.
- **Duplicate nulls:** independently confirmed **2,052** two-channel pairs where cumulative-zero and static-shape-zero test the same growth-sum coefficient; **3,204** annual DJO1 pairs where sum-zero and static β0+β1=0 coincide. Benchmark/DJO0 current and one-coefficient-sum rows are likewise aliases. They are valid algebra, but not independent confirmations and should be labelled as such in significance summaries.
- **Policy/mask distinction:** replayed masks match the reviewed literature protocol. Its maximum support (`t−h−1..t` for short growth; `t−2h..t` for YoY) differs deliberately from the generic core-policy description. That is a scope/documentation distinction, not evidence that growth was calculated after exclusions.

## Historical scope and an independently exposed blind spot

The review predates final rank/raw-AR correction, seasonal/full-control companions, report qualifiers and the extension stationarity appendix. Its `p1_calendar_history.py:198–243` validates AR lags through a lookup constructed from the already-cropped parent; verifying raw current growth does **not** prove full raw AR eligibility.

The current-versus-reviewed inventory comparison finds **39 context × outcome × variant sample-calendar rows** with additional dates, **all short-growth**. This count is neither native fit groups nor a count of distinct context-date observations. For VA20 at2006Q2, AR4 uses2005Q2 short growth **5.53176177055**, available from raw2005Q1/Q2 SA levels but absent from the reviewed parent. The corrected current sample retains it. This confirms the boundary issue already corrected by the parent; it does not dismiss the unchanged-YoY Gaussian counterexample above.

Other headline claims about every small p-value, effect absence, the 22-design simulation grid or all causal mechanisms are not independently established by this bounded replay. The full 22-design grid and leave-year-out grid were not rerun. “Both CR1 and DK must be significant” is not a generally calibrated decision rule; neither agreement nor reproduction certifies causal identification.

## Evidence and replay

- `mathematical_proof.json`: current-design bindings, exact aliases, raw-AR example, dynamic polynomial and selected IID rejection rows.
- `decisive_launch.json`: successful execution, source hashes, seed-bearing script and comparison/proof hashes.
- `inference_comparison.csv`: representative observed-reference sensitivities and explicit IID rejection counts, with method/bandwidth and assumptions.
- `replay_receipt.json`: provider identity/receipt validation, four original probe commands and source/output hashes.
- `generation_calendar_differences.csv`: the 39 precisely defined sample-calendar differences.

The selected known-null claim is reproducible from a clean restored replication without the ignored Fable sandbox:

```sh
uv run --locked python scripts/selected_null_calibration.py
```

This standalone route uses the exact frozen current NPZ, seed20261119 and 2,000 draws. It writes `outputs/aea_audit_calibration/{summary.json,rejections.csv,draw_statistics.csv}` and verifies input/source/ledger hashes. Its separately implemented score/HAC/restriction arithmetic reproduces all four rejection counts above; a PSD-trace bound also proves the production HC0 numerical-zero gate inactive in every draw. It performs no native fits. The packaged `new_methods_selected_fixed_design.json` is its summary.

For the original-sandbox semantic probes and independent reference comparison, from the repository root with the retained audit sandbox:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 dist/aea_audit_20261111/native_claims/new_methods_adjudication/run_replays.py
PYTHONDONTWRITEBYTECODE=1 uv run --locked python dist/aea_audit_20261111/native_claims/new_methods_adjudication/decisive_probes.py
```

The final packaged evidence uses `replication/extensions/review_evidence/new_methods_*`. No native fits, original references, analytical inputs, estimation samples or saved inference rows were altered by this adjudication.
