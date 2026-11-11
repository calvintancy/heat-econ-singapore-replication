"""Statsmodels-compatible grouped HAC retaining actual integer calendar gaps.

Statsmodels' general group_sums helper factorizes large integer group labels.
That is suitable for categorical groups but changes HAC time distances. This
wrapper preserves the existing score/bread and correction conventions while
using real period differences even when the calendar span exceeds 2*N.
"""
from __future__ import annotations

import numpy as np
from statsmodels.stats.sandwich_covariance import (
    S_hac_simple, _get_sandwich_arrays, _HCCM2, weights_bartlett,
)


def calendar_meat(scores, time, nlags=None, weights_func=weights_bartlett):
    scores, time = np.asarray(scores), np.asarray(time)
    if scores.ndim == 1: scores = scores[:, None]
    if scores.ndim != 2 or time.ndim != 1 or len(time) != len(scores) or not len(time):
        raise ValueError("HAC requires a nonempty score matrix and one time code per row")
    if not np.issubdtype(time.dtype, np.integer):
        raise TypeError("Calendar time codes must be integers representing equally spaced periods")
    offset = time - time.min();span = int(offset.max()) + 1
    if nlags is None: nlags = int(np.floor(4 * (span / 100.) ** (2. / 9.)))
    if not isinstance(nlags, (int, np.integer)) or nlags < 0: raise ValueError("nlags must be a nonnegative integer or None")
    if span <= 2 * len(scores) + 1:
        # This is the identical bincount score grid used by the old consumers
        # whenever statsmodels does not trigger categorical relabelling.
        aggregate = np.asarray([np.bincount(offset, weights=scores[:, column], minlength=span)
                                for column in range(scores.shape[1])]).T
        return S_hac_simple(aggregate, nlags=nlags, weights_func=weights_func)
    # Avoid an allocation proportional to a potentially huge missing interval.
    # Inverse codes aggregate rows only; lag matching uses original date values.
    dates, inverse = np.unique(offset, return_inverse=True)
    aggregate = np.asarray([np.bincount(inverse, weights=scores[:, column])
                            for column in range(scores.shape[1])]).T
    weights = weights_func(nlags)
    meat = weights[0] * (aggregate.T @ aggregate)
    for lag in range(1, min(nlags, span - 1) + 1):
        target = dates - lag
        previous = np.searchsorted(dates, target)
        valid = previous < len(dates)
        valid[valid] &= dates[previous[valid]] == target[valid]
        if np.any(valid):
            cross = aggregate[valid].T @ aggregate[previous[valid]]
            meat += weights[lag] * (cross + cross.T)
    return meat


def cov_nw_groupsum(results, nlags, time, weights_func=weights_bartlett, use_correction=0):
    """Drop-in covariance with unchanged statsmodels finite-sample formulas.

    Calendar origin is immaterial. Empty periods remain empty, not adjacent.
    With nlags=None the default bandwidth uses the physical calendar span.
    The cluster correction uses the number of observed dates, as before.
    """
    scores, bread = _get_sandwich_arrays(results)
    meat = calendar_meat(scores, time, nlags=nlags, weights_func=weights_func)
    covariance = _HCCM2(bread, meat)
    if use_correction:
        nobs, parameters = scores.shape
        if use_correction == "hac": covariance *= nobs / float(nobs - parameters)
        elif use_correction in ["c", "cluster"]:
            groups = len(np.unique(time))
            covariance *= groups / (groups - 1.)
            covariance *= (nobs - 1.) / float(nobs - parameters)
    return covariance
