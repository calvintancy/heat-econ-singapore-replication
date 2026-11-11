"""Sector-clustered CR1 covariance for pooled OLS common-slope estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.sandwich_covariance import cov_cluster


def common_slope_terms(terms: list[str], cluster_col: str = "sector_id") -> list[str]:
    """Exclude coefficients that are specific to a single clustering unit.

    Sector fixed effects also contain this name, but callers pass weather terms,
    not fixed-effect coefficients. Keeping this explicit prevents the zero SEs
    that clustering would produce for fully sector-specific weather slopes.
    """
    return [term for term in terms if cluster_col not in term]


def sector_cluster_covariance(model, groups, *, cluster_col: str = "sector_id") -> tuple[np.ndarray, dict]:
    """Return CR1 covariance and its finite-sample/inference metadata.

    ``groups`` must follow the exact observation order in the fitted model.
    A national single-series regression cannot use this covariance estimator.
    All explicitly estimated fixed-effect columns count toward K.
    """
    groups = pd.Series(np.asarray(groups))
    n, k = model.model.exog.shape
    if len(groups) != n or int(model.nobs) != n:
        raise ValueError("Sector groups must match the exact fitted OLS sample")
    if groups.isna().any():
        raise ValueError("Sector cluster identifiers cannot be missing")
    codes, unique = pd.factorize(groups, sort=False)
    clusters = len(unique)
    if clusters < 2:
        raise ValueError("Sector-clustered inference requires at least two sectors")
    if n <= k:
        raise ValueError("Sector CR1 correction requires positive OLS residual degrees of freedom")
    correction = clusters/(clusters-1)*(n-1)/(n-k)
    covariance = cov_cluster(model, codes, use_correction=True)
    return covariance, {
        "cluster_variable": cluster_col,
        "n_clusters": clusters,
        "n_parameters": k,
        "finite_sample_correction": correction,
        "inference_df": clusters-1,
    }
