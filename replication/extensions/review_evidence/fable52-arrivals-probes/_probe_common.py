"""Shared read-only helpers for the numbered audit probes.

Sets single BLAS threads before NumPy is imported, never imports production
modules, and only ever writes below audit_probes/.
"""
import os

for _name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"

import sys

sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "outputs/visitor_rigour"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def write(name, payload):
    target = HERE / name
    assert target.resolve().parent == HERE
    target.write_text(json.dumps(clean(payload), indent=1, ensure_ascii=False) + "\n")
    return target


def bartlett_calendar(scores, dates, bandwidth):
    """Independent actual-calendar Bartlett HAC with explicit zero-score bins.

    Written as an explicit sum over autocovariances of the zero-filled calendar
    array (not the production accumulate-preceding form).
    """
    dates = np.asarray(dates, dtype=np.int64)
    first, last = int(dates.min()), int(dates.max())
    bins = np.zeros((last - first + 1, scores.shape[1]))
    for position, date in enumerate(dates):
        bins[int(date) - first] += scores[position]
    total = bins.T @ bins
    for lag in range(1, bandwidth + 1):
        if lag >= len(bins):
            break
        gamma = bins[lag:].T @ bins[:-lag]
        total += (1.0 - lag / (bandwidth + 1.0)) * (gamma + gamma.T)
    return total


def bartlett_compressed(scores, dates, bandwidth):
    """Deliberately wrong comparator: lags counted over observed rows, gaps closed."""
    dates = np.asarray(dates, dtype=np.int64)
    unique = np.unique(dates)
    bins = np.zeros((len(unique), scores.shape[1]))
    for position, date in enumerate(dates):
        bins[int(np.searchsorted(unique, date))] += scores[position]
    total = bins.T @ bins
    for lag in range(1, bandwidth + 1):
        if lag >= len(bins):
            break
        gamma = bins[lag:].T @ bins[:-lag]
        total += (1.0 - lag / (bandwidth + 1.0)) * (gamma + gamma.T)
    return total


def scaled_rank(matrix):
    matrix = np.asarray(matrix, float)
    scales = np.linalg.norm(matrix, axis=0)
    keep = scales > 0
    if not keep.any():
        return 0
    singular = np.linalg.svd(matrix[:, keep] / scales[keep], compute_uv=False)
    return int(np.sum(singular > singular[0] * max(matrix.shape) * np.finfo(float).eps))


def maxabs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def maxrel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(float(np.max(np.abs(b))), np.finfo(float).tiny) if b.size else 1.0
    return float(np.max(np.abs(a - b)) / scale) if a.size else 0.0
