# Independent review of the detailed-sector FE interaction model

Read-only design review, 11 November 2026. Reviewed `scripts/granular_interactions.py`, its `heterogeneity.py` design/covariance/Wald helpers, saved summaries and report wording. Full numerical replay belongs to the separate interaction verifier; the numerical checks below are explicitly limited spot checks.

The H1 specification is coherent:

`Y_it = β_i T_t + γ′W_t + α_i,season(t) + λ_year(t) + ε_it`.

It identifies absolute sector slopes with common weather controls, common year effects and sector-specific seasonal effects. All sector-temperature columns are retained without a duplicate common-temperature column. These are not omitted-reference differences. No full-date fixed effects are present. Adding saturated date effects would make the common temperature component collinear and identify only relative slopes unless an arbitrary normalisation were imposed; that normalisation would not establish an economically absolute response.

## Balanced-design identities

Write `β_i = β_bar + δ_i` with the deviations summing to zero. On a balanced, identical-calendar panel with common regressors, the least-squares objective separates into a date-mean equation and sector-deviation equations. The mean equation uses temperature, all shared controls, seasons and years. Consequently **the arithmetic mean of the H1 temperature slopes equals the common-slope panel coefficient**. This depends on the balancing and equal observation weights; it is not a general unbalanced-panel result. The largest saved mean-versus-benchmark discrepancy is 9.35×10⁻¹¹.

This identity concerns point estimates. It does not imply identical reported uncertainty: the common-slope benchmark uses sector-CR1, while H1 uses joint calendar HAC and has more fitted parameters in its finite-sample correction. It also does not make the mean response an official national output-weighted response.

For two sectors, common year effects and common weather terms cancel:

`Y_it − Y_jt = (β_i − β_j)T_t + (α_i,season − α_j,season) + (ε_it − ε_jt)`.

Thus the H1 fitted difference is the temperature coefficient from the outcome difference regressed on temperature and seasonal indicators. Holding dates fixed, changing shared controls or common year effects cannot purge differential sector-year shocks from these contrasts; it shifts the mean/absolute response instead. This is why the fully flexible sensitivity is substantively useful. H1 heterogeneity, especially in levels, can reflect sector-specific trends or shocks correlated with national temperature rather than causal differences in heat sensitivity.

Read-only spot checks for the first sector pair in VA excluded-quarterly, IIP full-monthly and IIP long-quarterly samples reproduced the saved one-period-growth and raw-level differences using only temperature and season indicators. The largest discrepancy was below 1.2×10⁻¹¹. These checks support the algebra; they are not a replacement for the full covariance verifier.

## The contrast-rank bound is correct

Let `F` contain the `Q` observed seasonal indicators and temperature. For any sector pair the fitted residual difference is `M_F(Y_i−Y_j)`, which lies in a subspace of dimension at most `T−Q−1`. Multiplying by the season-residualised temperature to form slope influence scores cannot increase rank. Zero-filling excluded calendar dates cannot increase it either. Applying the Bartlett time kernel and forming covariance therefore gives

`rank(R V R′) ≤ min(S−1, T−Q−1)`

for the all-equal contrast matrix `R`. This is an upper bound, not a promise that the covariance attains that rank. It depends on a common temperature series, shared nuisance responses, balance and the same dates across sectors. It must not be transferred unchanged to unbalanced or fully sector-specific-control designs.

For the excluded quarterly sample, `T=19`, `Q=4`, so the bound is **14**, below the 18 VA or 17 IIP restrictions. Suppressing the omnibus p-value is correct. Individual or pairwise variance estimates can still be available; a singular high-dimensional omnibus covariance does not automatically invalidate every one-dimensional contrast.

The full quarterly sample has 29 dates and a bound of 24, so its 17/18-restriction test is algebraically available. **Available does not mean precise.** There are many restrictions relative to the number of time observations, and the restriction covariance can be ill-conditioned: the saved primary-bandwidth VA raw-level condition number is about 6.15×10⁸; VA short-growth is about 1.99×10⁵. Tiny nominal F p-values need a visible short-T/high-dimensional caution beside the joint result. The `F(r,T−1)` reference is an approximation, not an exact finite-sample distribution.

## Covariance, comparisons and robustness

Joint calendar Driscoll–Kraay is appropriate as the stated time-asymptotic covariance approach for this model. It retains cross-sector score covariance, uses actual gaps, applies Bartlett `h` and `2h` bandwidths, and counts the full panel design in the finite-sample factor. Sector-specific response components receive their direct temperature information from a single sector. Estimating shared nuisance coefficients using other sectors does not provide many independent supporting sector clusters for each fixed sector slope. Conventional sector-CR1 therefore cannot supply reliable individual-slope inference, even though a formula may produce a numerical matrix. This restriction does not prevent sector-CR1 for the genuinely common-slope benchmark.

The all-pair family is complete: 171 VA pairs and 153 IIP pairs per context/window/outcome/bandwidth. Its variance must include the saved off-diagonal covariance, `V_ii+V_jj−2V_ij`, as the code does. Holm adjustment is scoped to that family, not to all outcomes/windows in the project. Absolute-slope tests remain pointwise, and their 95% intervals are not family-wise intervals. “Significant in one sector but not another” is not a test of their difference.

Separate time-series fits are the fully flexible H3 point-estimate sensitivity: every sector has its own weather slopes, seasons and years. A block-diagonal fully interacted panel is algebraically the same collection of OLS problems. Those slopes generally differ from H1, which imposes shared controls and years. Their own-series HAC finite-sample correction can differ from a pooled H3 correction, so point equivalence must not be advertised as automatic SE equivalence.

The report appropriately keeps growth as the primary outcome, levels descriptive, exact-window stationarity flags separate from coefficient significance, and unresolved/gapped cases exploratory. The new interaction parameterisation changes restrictions on the same Y/X samples; it does not create new evidence of stationary disturbances, remove confounding or identify permanent warming damages.
