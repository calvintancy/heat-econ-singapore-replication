"""Native PyFixest OLS with the project's exact full-rank FE designs.

Patsy constructs formula designs, but never estimates coefficients.  PyFixest
estimates every coefficient and residual.  Explicit FE indicators preserve the
existing parameter count and original-X sandwich covariance conventions.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

for _thread_variable in ["OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd
import patsy
import pyfixest as pf
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs/pyfixest"
ESTIMATOR_VERSION = pf.__version__
ESTIMATOR = f"PyFixest {ESTIMATOR_VERSION}"
SOLVER = "np.linalg.lstsq"
COLLINEARITY_TOLERANCE = 1e-12
NATIVE_DESIGN_BASIS = "orthonormal SVD preconditioned full LSDV"
_FIT_LOG: list[dict] = []
_API_COUNTER = 0
_RUN_STARTED = datetime.now(timezone.utc).isoformat()


def _array_hash(array) -> str:
    values = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()


def _text_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass
class PyFixestResult:
    """Small result facade; native Feols remains accessible for provenance."""

    params: pd.Series
    resid: np.ndarray | pd.Series
    wresid: np.ndarray
    fittedvalues: np.ndarray
    normalized_cov_params: np.ndarray
    model: SimpleNamespace
    native_model: object
    coefficient_transform: np.ndarray
    native_coefficients: np.ndarray
    nobs: float
    df_resid: float
    df_model: float
    rsquared: float
    ssr: float
    scale: float
    estimator: str = ESTIMATOR
    estimator_version: str = ESTIMATOR_VERSION

    @property
    def pyfixest_model(self):
        return self.native_model

    def cov_params(self) -> np.ndarray:
        """Conventional iid covariance; reported project inference is external."""
        return self.normalized_cov_params * self.scale


def _geometry(x: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray, float]:
    """Design-only rank, scale and bread; this function never sees an outcome."""
    n, k = x.shape
    if n <= k or not np.isfinite(x).all():
        raise ValueError(f"{label}: finite X and positive residual degrees of freedom required")
    scales = np.linalg.norm(x, axis=0)
    if (scales == 0).any():
        raise ValueError(f"{label}: zero design column")
    scaled = x / scales
    with threadpool_limits(limits=1):
        u, singular, vt = np.linalg.svd(scaled, full_matrices=False)
        tolerance = singular[0] * max(n, k) * np.finfo(float).eps
        rank = int(np.sum(singular > tolerance))
        if rank != k:
            raise ValueError(f"{label}: rank-deficient design {rank}/{k}")
        # X = U S V' D, so original beta = D^-1 V S^-1 theta.
        # U and this invertible map depend only on X, never on an outcome.
        transform = (vt.T / singular) / scales[:, None]
        bread = transform @ transform.T
        one = np.ones(n)
        constant_in_span = np.max(np.abs(u @ (u.T @ one) - one)) < 1e-8
        orthonormal_error = float(np.max(np.abs(u.T @ u - np.eye(k))))
        if orthonormal_error > 1e-10:
            raise ValueError(f"{label}: native preconditioned design is not orthonormal")
    return scales, (bread + bread.T) / 2, bool(constant_in_span), u, transform, orthonormal_error


def fit_many_matrix(y_array, x_array, coef_names, label: str,
                    outcome_names=None, *, source_formula: str | None = None,
                    row_index=None) -> list[PyFixestResult]:
    """Fit all outcome columns through native feols; return input-column order.

    X must contain the complete desired intercept/FE basis.  No column is added,
    absorbed or dropped by the adapter.  Multi-outcome feols models each supply
    their own actual coefficient and residual arrays.
    """
    global _API_COUNTER
    x = np.array(x_array, dtype=float, copy=True, order="C")
    y = np.array(y_array, dtype=float, copy=True, order="C")
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or y.shape[1] < 1:
        raise ValueError(f"{label}: incompatible outcome/design shapes")
    if not np.isfinite(y).all():
        raise ValueError(f"{label}: non-finite outcome; no silent sample deletion allowed")
    names = list(coef_names)
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError(f"{label}: coefficient names must be unique and match X")
    if outcome_names is None:
        outcome_names = list(y_array.columns) if isinstance(y_array, pd.DataFrame) else [f"y{i:03d}" for i in range(y.shape[1])]
    outcome_names = [str(name) for name in outcome_names]
    if len(outcome_names) != y.shape[1]:
        raise ValueError(f"{label}: outcome-name count does not match Y")
    if row_index is not None and len(row_index) != len(y):
        raise ValueError(f"{label}: row-index length mismatch")
    scales, bread, constant_in_span, orthonormal_x, transform, orthonormal_error = _geometry(x, label)
    safe_x = [f"x{i:04d}" for i in range(x.shape[1])]
    safe_y = [f"y{i:03d}" for i in range(y.shape[1])]
    native_data = pd.DataFrame(orthonormal_x, columns=safe_x)
    for i, name in enumerate(safe_y):
        native_data[name] = y[:, i]
    lhs = safe_y[0] if len(safe_y) == 1 else "sw(" + ", ".join(safe_y) + ")"
    native_formula = lhs + " ~ 0 + " + " + ".join(safe_x)
    _API_COUNTER += 1
    api_call_id = _API_COUNTER
    with threadpool_limits(limits=1):
        native = pf.feols(native_formula, native_data, vcov="iid", solver=SOLVER,
                          collin_tol=COLLINEARITY_TOLERANCE, fixef_rm="none",
                          copy_data=True, store_data=False)
    models = [native] if len(safe_y) == 1 else [native.fetch_model(i, print_fml=False) for i in range(len(safe_y))]
    results = []
    x_hash, scaled_hash, names_hash, scales_hash = _array_hash(x), _array_hash(x / scales), _text_hash(names), _array_hash(scales)
    native_design_hash, transform_hash = _array_hash(orthonormal_x), _array_hash(transform)
    for i, fitted in enumerate(models):
        native_params = fitted.coef()
        if list(native_params.index) != safe_x:
            raise ValueError(f"{label}: PyFixest changed coefficient columns/order: {list(native_params.index)}")
        if fitted.collinearity.dropped_coef_names:
            raise ValueError(f"{label}: PyFixest dropped coefficients: {fitted.collinearity.dropped_coef_names}")
        if fitted.sample_info.n_obs != len(y) or fitted.sample_info.dropped_row_index:
            raise ValueError(f"{label}: PyFixest changed the estimation sample")
        native_response = np.asarray(fitted.within_data.response, dtype=float).reshape(-1)
        if native_response.shape != y[:, i].shape or not np.array_equal(native_response, y[:, i]):
            raise ValueError(f"{label}: PyFixest outcome/order differs from the requested outcome")
        native_design = np.asarray(fitted.within_data.design, dtype=float)
        if native_design.shape != orthonormal_x.shape or not np.array_equal(native_design, orthonormal_x):
            raise ValueError(f"{label}: PyFixest changed the orthonormal design/order")
        theta = native_params.to_numpy(dtype=float)
        with threadpool_limits(limits=1):
            coefficients = transform @ theta
            mapping_error = float(np.max(np.abs(coefficients-transform @ theta)))
        residual = np.asarray(fitted.resid(), dtype=float).reshape(-1).copy()
        if residual.shape != (len(y),) or not np.isfinite(residual).all() or not np.isfinite(coefficients).all():
            raise ValueError(f"{label}: invalid native coefficients/residuals")
        # This is a diagnostic identity only; stored residuals come from feols.
        residual_identity_error = float(np.max(np.abs(y[:, i] - x @ coefficients - residual)))
        if residual_identity_error > 5e-10 * (1 + np.max(np.abs(y[:, i]))):
            raise ValueError(f"{label}: native residuals do not match the original design")
        ssr = float(residual @ residual)
        total = float(np.square(y[:, i] - y[:, i].mean()).sum()) if constant_in_span else float(y[:, i] @ y[:, i])
        r_squared = 1 - ssr / total if total > 0 else np.nan
        model = SimpleNamespace(exog=x, wexog=x, endog=y[:, i], wendog=y[:, i],
                                exog_names=names, endog_names=outcome_names[i],
                                k_constant=int(constant_in_span), formula=source_formula,
                                data=SimpleNamespace(row_labels=row_index))
        result = PyFixestResult(
            params=pd.Series(coefficients, index=names, name="Estimate"),
            resid=residual, wresid=residual, fittedvalues=np.asarray(fitted.fitted_values.response).reshape(-1),
            normalized_cov_params=bread, model=model, native_model=fitted,
            coefficient_transform=transform, native_coefficients=theta,
            nobs=float(len(y)), df_resid=float(len(y)-len(names)),
            df_model=float(len(names)-int(constant_in_span)), rsquared=r_squared,
            ssr=ssr, scale=ssr/(len(y)-len(names)))
        results.append(result)
        _FIT_LOG.append({
            "fit_id": len(_FIT_LOG)+1, "api_call_id": api_call_id, "label": label,
            "outcome": outcome_names[i], "estimator": ESTIMATOR, "estimator_version": ESTIMATOR_VERSION,
            "source_formula": source_formula or "Explicit full-rank design matrix",
            "native_formula": str(fitted.model.parsed_formula), "native_call_formula": native_formula,
            "native_class": type(fitted).__name__, "native_module": type(fitted).__module__,
            "n_obs": len(y), "n_parameters": len(names), "df_resid": len(y)-len(names),
            "constant_in_span": constant_in_span, "solver": SOLVER,
            "design_sha256": x_hash, "scaled_design_sha256": scaled_hash,
            "native_design_basis": NATIVE_DESIGN_BASIS, "native_design_sha256": native_design_hash,
            "coefficient_transform_sha256": transform_hash,
            "coefficient_transform_rows": transform.shape[0], "coefficient_transform_columns": transform.shape[1],
            "coefficient_transform_rank": x.shape[1],
            "native_design_orthogonality_max_abs_error": orthonormal_error,
            "native_design_max_abs_error": float(np.max(np.abs(native_design-orthonormal_x))),
            "outcome_sha256": _array_hash(y[:, i]), "coefficient_names_sha256": names_hash,
            "native_outcome_sha256": _array_hash(native_response),
            "native_response_max_abs_error": float(np.max(np.abs(native_response-y[:, i]))),
            "native_outcome_max_abs_error": float(np.max(np.abs(native_response-y[:, i]))),
            "column_scales_sha256": scales_hash, "coefficients_sha256": _array_hash(coefficients),
            "native_coefficients_sha256": _array_hash(native_params.to_numpy()),
            "residuals_sha256": _array_hash(residual), "native_residuals_sha256": _array_hash(fitted.resid()),
            "row_index_sha256": _text_hash([str(v) for v in row_index]) if row_index is not None else "",
            "native_mapped_coefficient_max_abs_error": mapping_error,
            "native_residual_max_abs_error": float(np.max(np.abs(residual-np.asarray(fitted.resid()).reshape(-1)))),
            "original_design_residual_identity_max_abs_error": residual_identity_error,
            "native_dropped_coefficients": len(fitted.collinearity.dropped_coef_names),
            "native_dropped_rows": len(fitted.sample_info.dropped_row_index),
            "coefficient_names": json.dumps(names, ensure_ascii=False),
            "column_scales": json.dumps(scales.tolist()),
        })
    return results


def fit_matrix(y_array, x_array, coef_names, label: str) -> PyFixestResult:
    y = np.asarray(y_array)
    if y.ndim > 1 and y.shape[1] != 1:
        raise ValueError(f"{label}: fit_matrix expects a single outcome; use fit_many_matrix")
    name = str(y_array.name) if isinstance(y_array, pd.Series) and y_array.name is not None else "y000"
    return fit_many_matrix(y_array, x_array, coef_names, label, outcome_names=[name])[0]


def fit(df: pd.DataFrame, formula: str, label: str) -> PyFixestResult:
    """Patsy design compatibility, native PyFixest coefficient estimation."""
    response, design = patsy.dmatrices(formula, df, return_type="dataframe", NA_action="raise")
    if len(design) != len(df) or response.shape[1] != 1:
        raise ValueError(f"{label}: formula changed sample or has multiple outcomes")
    result = fit_many_matrix(response.to_numpy(), design.to_numpy(), design.columns, label,
                             outcome_names=response.columns, source_formula=formula,
                             row_index=design.index)[0]
    result.resid = pd.Series(result.wresid, index=design.index, name="residual")
    return result


def reset_fit_log() -> None:
    """Start a fresh production receipt and invalidate any stale success record."""
    global _API_COUNTER, _RUN_STARTED
    _FIT_LOG.clear()
    _API_COUNTER = 0
    _RUN_STARTED = datetime.now(timezone.utc).isoformat()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["fit_id", "api_call_id", "label", "outcome"]).to_csv(LOG_DIR / "fit_log.csv", index=False)
    (LOG_DIR / "estimator_manifest.json").write_text(json.dumps({
        "status": "running", "run_started_utc": _RUN_STARTED, "estimator": ESTIMATOR,
        "native_fits_recorded": 0, "claim": "New estimation run started; no completion claim yet"}, indent=2)+"\n")


def write_fit_log() -> dict:
    """Seal the current native-fit inventory after the caller's model stage."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(_FIT_LOG)
    temporary = LOG_DIR / f"fit_log.{os.getpid()}.tmp"
    records.to_csv(temporary, index=False)
    temporary.replace(LOG_DIR / "fit_log.csv")
    manifest = {
        "status": "model_inventory_written", "run_started_utc": _RUN_STARTED,
        "written_utc": datetime.now(timezone.utc).isoformat(), "estimator": ESTIMATOR,
        "estimator_version": ESTIMATOR_VERSION, "solver": SOLVER,
        "fixed_effect_implementation": "Explicit full-rank LSDV design passed to native PyFixest feols",
        "coefficient_source": "Native Feols.coef() on orthonormal U; original beta = D^-1 V S^-1 theta_native",
        "native_design_basis": NATIVE_DESIGN_BASIS,
        "coefficient_transform": "SVD(X D^-1)=U S V'; original X = U S V' D; transform=D^-1 V S^-1",
        "residual_source": "Native Feols.resid(); never replaced by a manual OLS fit",
        "r_squared_source": "Original outcome and native residuals; centered whenever the constant is in the design span",
        "native_fits_recorded": len(records), "native_api_calls": _API_COUNTER,
        "collinearity_tolerance": COLLINEARITY_TOLERANCE,
        "all_native_samples_and_coefficients_retained": bool(len(records) and records.native_dropped_rows.eq(0).all() and records.native_dropped_coefficients.eq(0).all()),
        "backend": "No absorption/GPU demeaner; invertible orthonormal preconditioning of full explicit design, CPU, BLAS thread limit 1",
        "native_models_retained_on_wrappers": True,
        "fit_log_sha256": hashlib.sha256((LOG_DIR / "fit_log.csv").read_bytes()).hexdigest(),
        "validation_scope": "Native estimator provenance only; independent numerical migration verification is separate",
    }
    target = LOG_DIR / f"estimator_manifest.{os.getpid()}.tmp"
    target.write_text(json.dumps(manifest, indent=2)+"\n")
    target.replace(LOG_DIR / "estimator_manifest.json")
    return manifest
