# Sector-clustered inference

The pooled panel tables report standard errors clustered by the panel unit `sector_id`. The point estimates, dependent variables, controls, fixed effects and estimation samples are unchanged. Previously reported calendar Driscoll–Kraay (DK) results remain alongside them as a sensitivity check.

For an OLS design matrix X, fitted residuals u and S sectors, define the sector score as gᵢ = Σₜ xᵢₜuᵢₜ. The reported CR1 covariance is

V = [S/(S−1)] [(N−1)/(N−K)] (X′X)⁻¹ [Σᵢ gᵢgᵢ′] (X′X)⁻¹.

N is the total number of sector-period observations, and K counts every explicitly estimated column, including the fixed effects. Standard errors are the square roots of the relevant diagonal elements. Two-sided p-values and 95% confidence intervals use the Student t distribution with S−1 degrees of freedom. This is the small-sample correction implemented by [statsmodels cov_cluster](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html); the [robust covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html) describes group-minus-one inference.

The cluster counts are 11 broad or 19 detailed VA sectors, and six broad or 18 detailed manufacturing industries. Each sector may have arbitrary serial correlation and heteroskedasticity within its time series. The method assumes sectors are independent of one another. Because Singapore weather is a national exposure repeated across sectors, sector clustering does **not** address common national shocks or cross-sector error dependence. Six or 11 clusters also make conventional cluster inference fragile; the t correction does not guarantee reliable finite-sample size. These are reasons to inspect the DK alternative and avoid treating a change in significance as proof of a causal result. [Cameron and Miller (2015)](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf) discuss these clustering assumptions and few-cluster limitations.

There are two necessary exceptions:

- Official aggregate GVA and aggregate manufacturing IIP each contain one national series. Sector clustering is undefined with S=1, so these retain calendar HAC inference.
- A coefficient for one sector's temperature interaction is supported by one sector cluster. In a fully sector-specific time-series design, the corresponding cluster score can be zero by the OLS normal equations. Reporting that as a precise sector-clustered estimate would be misleading. Individual-sector slopes and the flexible heterogeneity tables therefore retain their labelled calendar-HAC inference. Common-slope panel comparisons receive sector CR1.

The original DK estimates use four quarterly or 12 monthly Bartlett lags, with doubled bandwidths retained where previously reported. DK sums scores across sectors within each calendar period before applying HAC. It therefore allows cross-sector contemporaneous dependence and time dependence under a different asymptotic framework. Its correction and t(T−1) reference are unchanged. Distributed-lag sum and paired-weather restriction JSON files preserve their original DK fields and add a `sector_cluster` object. The main report displays that labelled CR1 contrast; DK values remain available in the downloads.

## Implementation and verification

`scripts/sector_cluster.py` exposes `sector_cluster_covariance(model, groups)`, returning a covariance matrix and metadata (`n_clusters`, `n_parameters`, `finite_sample_correction`, `inference_df`, `cluster_variable`). The groups must follow the fitted observation order. `scripts/estimate.py` appends the sector-cluster rows once, at the baseline bandwidth call; each has `covariance = "Sector cluster"` and `hac_lags = 0`. Requests for sector-specific coefficients and single-unit samples are excluded.

`scripts/verify_sector_cluster.py` does not import the estimator or statsmodels covariance functions. It reconstructs the exact saved samples and formulas, computes OLS through QR, explicitly sums sector scores and their outer products, and checks the CR1 standard errors, t(S−1) p-values, confidence intervals and correction metadata. It also verifies that all corresponding DK rows have the same coefficients and that sector-specific/aggregate annexes do not acquire invalid cluster errors. New analysis folders can supply exact inputs at `samples/{model_id}.csv`.

Run after all estimation scripts:

```sh
uv run python scripts/verify_sector_cluster.py
```

The audit is written to `outputs/sector_cluster_verification.json`.
