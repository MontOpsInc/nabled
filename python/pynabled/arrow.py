"""PyArrow bridge for nabled/ndarrow workflows.

Requires pynabled built with the arrow feature: ``pip install pynabled[arrow]``
and ``maturin develop --features arrow`` (or equivalent).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from .config import (
    AdamConfig,
    BFGSConfig,
    GradientDescentConfig,
    IterativeConfig,
    JacobianConfig,
    LineSearchConfig,
    MomentumConfig,
    ProjectedGradientConfig,
    RMSPropConfig,
)
from .results import (
    CholeskyResult,
    EigenResult,
    GeneralizedEigenResult,
    LogDetResult,
    LuResult,
    NonsymmetricBiEigenResult,
    NonsymmetricEigenResult,
    PcaResult,
    PolarResult,
    QrResult,
    RegressionResult,
    SchurResult,
    SvdResult,
)

try:
    import pynabled._pynabled as _raw
except ImportError as e:
    raise ImportError(
        "pynabled arrow support not available. "
        "Install with: pip install pynabled[arrow] and build with --features arrow"
    ) from e


if not hasattr(_raw, "arrow_dot"):
    raise ImportError(
        "pynabled arrow support not available. "
        "Install with: pip install pynabled[arrow] and build with --features arrow"
    )


_COMPLEX_EXTENSION_NAMES = {"ndarrow.complex64"}
_COMPLEX_STORAGE_TYPE = pa.list_(pa.field("item", pa.float64(), nullable=False), 2)


def _arrow_field(array, name):
    if isinstance(array, pa.ExtensionArray) and _extension_name(array.type) in _COMPLEX_EXTENSION_NAMES:
        return _complex_vector_field(name)
    return pa.field(name, array.type, nullable=False)


def _extension_array(field, storage):
    return pa.ExtensionArray.from_storage(field.type, storage)


def _extension_name(value):
    type_ = value.type if hasattr(value, "type") else value
    return getattr(type_, "extension_name", None)


def _is_extension_type(type_):
    return isinstance(type_, pa.ExtensionType)


def _is_complex_scalar_type(type_):
    return _is_extension_type(type_) and _extension_name(type_) in _COMPLEX_EXTENSION_NAMES


def _field_metadata_value(field, key):
    metadata = field.metadata or {}
    if key in metadata:
        return metadata[key]
    return metadata.get(key.encode())


def _complex_vector_field(name):
    return pa.field(
        name,
        _COMPLEX_STORAGE_TYPE,
        nullable=False,
        metadata={"ARROW:extension:name": "ndarrow.complex64"},
    )


def _complex_vector_storage(array):
    if isinstance(array, pa.ExtensionArray) and _extension_name(array.type) in _COMPLEX_EXTENSION_NAMES:
        return array.storage
    return None


def _complex_matrix_storage(array):
    if not isinstance(array, pa.FixedSizeListArray):
        return None
    value_field = array.type.value_field
    if (
        value_field.type == _COMPLEX_STORAGE_TYPE
        and _field_metadata_value(value_field, "ARROW:extension:name") in ("ndarrow.complex64", b"ndarrow.complex64")
    ):
        return array
    if _is_extension_type(value_field.type) and _extension_name(value_field.type) in _COMPLEX_EXTENSION_NAMES:
        return pa.FixedSizeListArray.from_arrays(
            array.values.storage,
            type=pa.list_(_complex_vector_field("item"), array.type.list_size),
        )
    return None


def _is_complex_vector(array):
    return _complex_vector_storage(array) is not None


def _is_complex_matrix(array):
    return _complex_matrix_storage(array) is not None


def _is_complex_array(array):
    return _is_complex_vector(array) or _is_complex_matrix(array)


def _require_complex(name, **arrays):
    wrong = [label for label, array in arrays.items() if not _is_complex_array(array)]
    if wrong:
        joined = ", ".join(wrong)
        raise TypeError(f"{name} requires ndarrow.complex64 Arrow carriers for: {joined}")


def _require_real(name, **arrays):
    wrong = [label for label, array in arrays.items() if _is_complex_array(array)]
    if wrong:
        joined = ", ".join(wrong)
        raise TypeError(f"{name} does not currently admit ndarrow.complex64 Arrow carriers: {joined}")


def _complex_mode(name, **arrays):
    flags = {label: _is_complex_array(array) for label, array in arrays.items()}
    if any(flags.values()) and not all(flags.values()):
        joined = ", ".join(arrays)
        raise TypeError(
            f"{name} requires {joined} to all be real Arrow carriers or all ndarrow.complex64 carriers"
        )
    return any(flags.values())


def _resolve_config(config, config_type, **kwargs):
    if config is None:
        return kwargs
    if not isinstance(config, config_type):
        raise TypeError(f"config must be {config_type.__name__} or None")
    conflicts = [name for name, value in kwargs.items() if value is not None]
    if conflicts:
        joined = ", ".join(conflicts)
        raise TypeError(
            f"pass either {config_type.__name__} via config= or explicit keyword arguments, not both: {joined}"
        )
    return {name: getattr(config, name) for name in kwargs}


def _svd_result(raw_result) -> SvdResult:
    u, singular_values, vt = raw_result
    return SvdResult(u=u, singular_values=singular_values, vt=vt)


def _qr_result(raw_result) -> QrResult:
    if len(raw_result) == 3:
        q, r, rank = raw_result
        return QrResult(q=q, r=r, rank=rank)
    q, r, p, rank = raw_result
    return QrResult(q=q, r=r, rank=rank, p=p)


def _lu_result(raw_result) -> LuResult:
    l, u = raw_result
    return LuResult(l=l, u=u)


def _cholesky_result(raw_result) -> CholeskyResult:
    return CholeskyResult(l=raw_result)


def _eigen_result(raw_result) -> EigenResult:
    eigenvalues, eigenvectors = raw_result
    return EigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def _generalized_eigen_result(raw_result) -> GeneralizedEigenResult:
    eigenvalues, eigenvectors = raw_result
    return GeneralizedEigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def _nonsymmetric_eigen_result(raw_result) -> NonsymmetricEigenResult:
    eigenvalues, schur_vectors = raw_result
    return NonsymmetricEigenResult(eigenvalues=eigenvalues, schur_vectors=schur_vectors)


def _nonsymmetric_bi_eigen_result(raw_result) -> NonsymmetricBiEigenResult:
    (
        eigenvalues,
        right_eigenvectors,
        left_eigenvectors,
        balancing_diagonal,
        balanced_matrix,
    ) = raw_result
    return NonsymmetricBiEigenResult(
        eigenvalues=eigenvalues,
        right_eigenvectors=right_eigenvectors,
        left_eigenvectors=left_eigenvectors,
        balancing_diagonal=balancing_diagonal,
        balanced_matrix=balanced_matrix,
    )


def _schur_result(raw_result) -> SchurResult:
    t, q = raw_result
    return SchurResult(q=q, t=t)


def _polar_result(raw_result) -> PolarResult:
    u, p = raw_result
    return PolarResult(u=u, p=p)


def _log_det_result(raw_result) -> LogDetResult:
    sign, ln_abs_det = raw_result
    return LogDetResult(sign=sign, ln_abs_det=float(ln_abs_det))


def _pca_result(raw_result) -> PcaResult:
    components, explained_variance, explained_variance_ratio, mean, scores = raw_result
    return PcaResult(
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean,
        scores=scores,
    )


def _regression_result(raw_result) -> RegressionResult:
    coefficients, fitted_values, residuals, r_squared = raw_result
    return RegressionResult(
        coefficients=coefficients,
        fitted_values=fitted_values,
        residuals=residuals,
        r_squared=r_squared,
    )


def arrow_dot_hermitian(left, right):
    _require_complex("arrow_dot_hermitian", left=left, right=right)
    left_storage = _complex_vector_storage(left)
    right_storage = _complex_vector_storage(right)
    return _raw.arrow_dot_hermitian(
        _complex_vector_field("left"),
        left_storage,
        _complex_vector_field("right"),
        right_storage,
    )


def arrow_l2_norm_complex(vector):
    _require_complex("arrow_l2_norm_complex", vector=vector)
    return _raw.arrow_l2_norm_complex(
        _complex_vector_field("vector"),
        _complex_vector_storage(vector),
    )


def arrow_cosine_similarity_complex(left, right):
    _require_complex("arrow_cosine_similarity_complex", left=left, right=right)
    left_storage = _complex_vector_storage(left)
    right_storage = _complex_vector_storage(right)
    field, storage = _raw.arrow_cosine_similarity_complex(
        _complex_vector_field("left"),
        left_storage,
        _complex_vector_field("right"),
        right_storage,
    )
    return _extension_array(field, storage)


def arrow_batched_dot_hermitian(left, right):
    _require_complex("arrow_batched_dot_hermitian", left=left, right=right)
    field, storage = _raw.arrow_batched_dot_hermitian(
        _complex_matrix_storage(left),
        _complex_matrix_storage(right),
    )
    return _extension_array(field, storage)


def arrow_batched_l2_norm_complex(rows):
    _require_complex("arrow_batched_l2_norm_complex", rows=rows)
    return _raw.arrow_batched_l2_norm_complex(_complex_matrix_storage(rows))


def arrow_batched_cosine_similarity_complex(left, right):
    _require_complex("arrow_batched_cosine_similarity_complex", left=left, right=right)
    field, storage = _raw.arrow_batched_cosine_similarity_complex(
        _complex_matrix_storage(left),
        _complex_matrix_storage(right),
    )
    return _extension_array(field, storage)


def arrow_batched_normalize_complex(rows):
    _require_complex("arrow_batched_normalize_complex", rows=rows)
    return _raw.arrow_batched_normalize_complex(_complex_matrix_storage(rows))


def arrow_matvec_complex(matrix, vector):
    _require_complex("arrow_matvec_complex", matrix=matrix, vector=vector)
    matrix_storage = _complex_matrix_storage(matrix)
    vector_storage = _complex_vector_storage(vector)
    field, storage = _raw.arrow_matvec_complex(
        matrix_storage,
        _complex_vector_field("vector"),
        vector_storage,
    )
    return _extension_array(field, storage)


def arrow_matmat_complex(left, right):
    _require_complex("arrow_matmat_complex", left=left, right=right)
    return _raw.arrow_matmat_complex(_complex_matrix_storage(left), _complex_matrix_storage(right))


def arrow_column_means_complex(matrix):
    _require_complex("arrow_column_means_complex", matrix=matrix)
    field, storage = _raw.arrow_column_means_complex(_complex_matrix_storage(matrix))
    return _extension_array(field, storage)


def arrow_center_columns_complex(matrix):
    _require_complex("arrow_center_columns_complex", matrix=matrix)
    return _raw.arrow_center_columns_complex(_complex_matrix_storage(matrix))


def arrow_covariance_matrix_complex(matrix):
    _require_complex("arrow_covariance_matrix_complex", matrix=matrix)
    return _raw.arrow_covariance_matrix_complex(_complex_matrix_storage(matrix))


def arrow_correlation_matrix_complex(matrix):
    _require_complex("arrow_correlation_matrix_complex", matrix=matrix)
    return _raw.arrow_correlation_matrix_complex(_complex_matrix_storage(matrix))


def arrow_gram_schmidt_complex(matrix):
    _require_complex("arrow_gram_schmidt_complex", matrix=matrix)
    return _raw.arrow_gram_schmidt_complex(_complex_matrix_storage(matrix))


def arrow_solve_lower_complex(matrix, rhs):
    _require_complex("arrow_solve_lower_complex", matrix=matrix, rhs=rhs)
    matrix_storage = _complex_matrix_storage(matrix)
    rhs_storage = _complex_vector_storage(rhs)
    field, storage = _raw.arrow_solve_lower_complex(
        matrix_storage,
        _complex_vector_field("rhs"),
        rhs_storage,
    )
    return _extension_array(field, storage)


def arrow_solve_upper_complex(matrix, rhs):
    _require_complex("arrow_solve_upper_complex", matrix=matrix, rhs=rhs)
    matrix_storage = _complex_matrix_storage(matrix)
    rhs_storage = _complex_vector_storage(rhs)
    field, storage = _raw.arrow_solve_upper_complex(
        matrix_storage,
        _complex_vector_field("rhs"),
        rhs_storage,
    )
    return _extension_array(field, storage)


def arrow_batched_matmat(left, right):
    field, storage = _raw.arrow_batched_matmat(
        _arrow_field(left, "left"),
        left,
        _arrow_field(right, "right"),
        right,
    )
    return _extension_array(field, storage)


def arrow_batched_matmat_broadcast_right(left, right):
    field, storage = _raw.arrow_batched_matmat_broadcast_right(_arrow_field(left, "left"), left, right)
    return _extension_array(field, storage)


def arrow_batched_matmat_broadcast_left(left, right):
    field, storage = _raw.arrow_batched_matmat_broadcast_left(left, _arrow_field(right, "right"), right)
    return _extension_array(field, storage)


def arrow_batched_qr(matrices, rank_tolerance=None, max_iterations=None) -> list[QrResult]:
    return [
        _qr_result(result)
        for result in _raw.arrow_batched_qr(
            _arrow_field(matrices, "matrices"),
            matrices,
            rank_tolerance=rank_tolerance,
            max_iterations=max_iterations,
        )
    ]


def arrow_batched_svd(matrices) -> list[SvdResult]:
    return [
        _svd_result(result)
        for result in _raw.arrow_batched_svd(_arrow_field(matrices, "matrices"), matrices)
    ]


def arrow_batched_lu(matrices) -> list[LuResult]:
    return [
        _lu_result(result)
        for result in _raw.arrow_batched_lu(_arrow_field(matrices, "matrices"), matrices)
    ]


def arrow_batched_cholesky(matrices) -> list[CholeskyResult]:
    return [
        _cholesky_result(result)
        for result in _raw.arrow_batched_cholesky(_arrow_field(matrices, "matrices"), matrices)
    ]


def arrow_batched_symmetric_eigen(matrices) -> list[EigenResult]:
    return [
        _eigen_result(result)
        for result in _raw.arrow_batched_symmetric_eigen(_arrow_field(matrices, "matrices"), matrices)
    ]


def arrow_svd_decompose_complex(data) -> SvdResult:
    _require_complex("arrow_svd_decompose_complex", data=data)
    return _svd_result(_raw.arrow_svd_decompose_complex(_complex_matrix_storage(data)))


def arrow_qr_decompose_complex(data) -> QrResult:
    _require_complex("arrow_qr_decompose_complex", data=data)
    return _qr_result(_raw.arrow_qr_decompose_complex(_complex_matrix_storage(data)))


def arrow_lu_solve_complex(matrix, rhs):
    _require_complex("arrow_lu_solve_complex", matrix=matrix, rhs=rhs)
    matrix_storage = _complex_matrix_storage(matrix)
    rhs_storage = _complex_vector_storage(rhs)
    field, storage = _raw.arrow_lu_solve_complex(
        matrix_storage,
        _complex_vector_field("rhs"),
        rhs_storage,
    )
    return _extension_array(field, storage)


def arrow_lu_inverse_complex(matrix):
    _require_complex("arrow_lu_inverse_complex", matrix=matrix)
    return _raw.arrow_lu_inverse_complex(_complex_matrix_storage(matrix))


def arrow_lu_determinant_complex(matrix):
    _require_complex("arrow_lu_determinant_complex", matrix=matrix)
    return _raw.arrow_lu_determinant_complex(_complex_matrix_storage(matrix))


def arrow_cholesky_decompose_complex(matrix) -> CholeskyResult:
    _require_complex("arrow_cholesky_decompose_complex", matrix=matrix)
    return _cholesky_result(_raw.arrow_cholesky_decompose_complex(_complex_matrix_storage(matrix)))


def arrow_cholesky_solve_complex(matrix, rhs):
    _require_complex("arrow_cholesky_solve_complex", matrix=matrix, rhs=rhs)
    matrix_storage = _complex_matrix_storage(matrix)
    rhs_storage = _complex_vector_storage(rhs)
    field, storage = _raw.arrow_cholesky_solve_complex(
        matrix_storage,
        _complex_vector_field("rhs"),
        rhs_storage,
    )
    return _extension_array(field, storage)


def arrow_cholesky_inverse_complex(matrix):
    _require_complex("arrow_cholesky_inverse_complex", matrix=matrix)
    return _raw.arrow_cholesky_inverse_complex(_complex_matrix_storage(matrix))


def arrow_eigen_nonsymmetric_complex(matrix) -> NonsymmetricEigenResult:
    _require_complex("arrow_eigen_nonsymmetric_complex", matrix=matrix)
    return _nonsymmetric_eigen_result(_raw.arrow_eigen_nonsymmetric_complex(_complex_matrix_storage(matrix)))


def arrow_schur_compute_complex(matrix) -> SchurResult:
    _require_complex("arrow_schur_compute_complex", matrix=matrix)
    return _schur_result(_raw.arrow_schur_compute_complex(_complex_matrix_storage(matrix)))


def arrow_polar_compute_complex(matrix) -> PolarResult:
    _require_complex("arrow_polar_compute_complex", matrix=matrix)
    return _polar_result(_raw.arrow_polar_compute_complex(_complex_matrix_storage(matrix)))


def arrow_matrix_exp_complex(matrix, max_terms=None, tolerance=None):
    _require_complex("arrow_matrix_exp_complex", matrix=matrix)
    return _raw.arrow_matrix_exp_complex(
        _complex_matrix_storage(matrix),
        max_terms=max_terms,
        tolerance=tolerance,
    )


def arrow_matrix_exp_eigen_complex(matrix):
    _require_complex("arrow_matrix_exp_eigen_complex", matrix=matrix)
    return _raw.arrow_matrix_exp_eigen_complex(_complex_matrix_storage(matrix))


def arrow_matrix_log_eigen_complex(matrix):
    _require_complex("arrow_matrix_log_eigen_complex", matrix=matrix)
    return _raw.arrow_matrix_log_eigen_complex(_complex_matrix_storage(matrix))


def arrow_matrix_log_svd_complex(matrix):
    _require_complex("arrow_matrix_log_svd_complex", matrix=matrix)
    return _raw.arrow_matrix_log_svd_complex(_complex_matrix_storage(matrix))


def arrow_matrix_power_complex(matrix, power):
    _require_complex("arrow_matrix_power_complex", matrix=matrix)
    return _raw.arrow_matrix_power_complex(_complex_matrix_storage(matrix), power)


def arrow_matrix_sign_complex(matrix):
    _require_complex("arrow_matrix_sign_complex", matrix=matrix)
    return _raw.arrow_matrix_sign_complex(_complex_matrix_storage(matrix))


def arrow_compute_pca_complex(matrix, n_components=None) -> PcaResult:
    _require_complex("arrow_compute_pca_complex", matrix=matrix)
    return _pca_result(
        _raw.arrow_compute_pca_complex(_complex_matrix_storage(matrix), n_components=n_components)
    )


def arrow_pca_transform_complex(matrix, result: PcaResult):
    _require_complex("arrow_pca_transform_complex", matrix=matrix)
    return _raw.arrow_pca_transform_complex(
        _complex_matrix_storage(matrix),
        result.components,
        result.mean,
    )


def arrow_pca_inverse_transform_complex(scores, result: PcaResult):
    _require_complex("arrow_pca_inverse_transform_complex", scores=scores)
    return _raw.arrow_pca_inverse_transform_complex(
        _complex_matrix_storage(scores),
        result.components,
        result.mean,
    )


def arrow_linear_regression_complex(x, y, add_intercept=True) -> RegressionResult:
    _require_complex("arrow_linear_regression_complex", x=x, y=y)
    return _regression_result(
        _raw.arrow_linear_regression_complex(
            _complex_matrix_storage(x),
            _complex_vector_field("y"),
            _complex_vector_storage(y),
            add_intercept=add_intercept,
        )
    )


def arrow_dot(left, right):
    if _complex_mode("arrow_dot", left=left, right=right):
        return arrow_dot_hermitian(left, right)
    return _raw.arrow_dot(left, right)


def arrow_l2_norm(vector):
    if _is_complex_vector(vector):
        return arrow_l2_norm_complex(vector)
    return _raw.arrow_l2_norm(vector)


def arrow_cosine_similarity(left, right):
    if _complex_mode("arrow_cosine_similarity", left=left, right=right):
        return arrow_cosine_similarity_complex(left, right)
    return _raw.arrow_cosine_similarity(left, right)


def arrow_cosine_distance(left, right):
    _require_real("arrow_cosine_distance", left=left, right=right)
    return _raw.arrow_cosine_distance(left, right)


def arrow_pairwise_l2_distance(left, right):
    _require_real("arrow_pairwise_l2_distance", left=left, right=right)
    return _raw.arrow_pairwise_l2_distance(left, right)


def arrow_pairwise_cosine_similarity(left, right):
    _require_real("arrow_pairwise_cosine_similarity", left=left, right=right)
    return _raw.arrow_pairwise_cosine_similarity(left, right)


def arrow_pairwise_cosine_distance(left, right):
    _require_real("arrow_pairwise_cosine_distance", left=left, right=right)
    return _raw.arrow_pairwise_cosine_distance(left, right)


def arrow_batched_dot(left, right):
    if _complex_mode("arrow_batched_dot", left=left, right=right):
        return arrow_batched_dot_hermitian(left, right)
    return _raw.arrow_batched_dot(left, right)


def arrow_batched_l2_norm(rows):
    if _is_complex_matrix(rows):
        return arrow_batched_l2_norm_complex(rows)
    return _raw.arrow_batched_l2_norm(rows)


def arrow_batched_cosine_similarity(left, right):
    if _complex_mode("arrow_batched_cosine_similarity", left=left, right=right):
        return arrow_batched_cosine_similarity_complex(left, right)
    return _raw.arrow_batched_cosine_similarity(left, right)


def arrow_batched_cosine_distance(left, right):
    _require_real("arrow_batched_cosine_distance", left=left, right=right)
    return _raw.arrow_batched_cosine_distance(left, right)


def arrow_batched_normalize(rows):
    if _is_complex_matrix(rows):
        return arrow_batched_normalize_complex(rows)
    return _raw.arrow_batched_normalize(rows)


def arrow_batched_row_matvec(batch_vectors, matrix):
    _require_real("arrow_batched_row_matvec", batch_vectors=batch_vectors, matrix=matrix)
    return _raw.arrow_batched_row_matvec(batch_vectors, matrix)


def arrow_matvec(matrix, vector):
    if _complex_mode("arrow_matvec", matrix=matrix, vector=vector):
        return arrow_matvec_complex(matrix, vector)
    return _raw.arrow_matvec(matrix, vector)


def arrow_matmat(left, right):
    if _complex_mode("arrow_matmat", left=left, right=right):
        return arrow_matmat_complex(left, right)
    return _raw.arrow_matmat(left, right)


def arrow_column_means(matrix):
    if _is_complex_matrix(matrix):
        return arrow_column_means_complex(matrix)
    return _raw.arrow_column_means(matrix)


def arrow_center_columns(matrix):
    if _is_complex_matrix(matrix):
        return arrow_center_columns_complex(matrix)
    return _raw.arrow_center_columns(matrix)


def arrow_covariance_matrix(matrix):
    if _is_complex_matrix(matrix):
        return arrow_covariance_matrix_complex(matrix)
    return _raw.arrow_covariance_matrix(matrix)


def arrow_correlation_matrix(matrix):
    if _is_complex_matrix(matrix):
        return arrow_correlation_matrix_complex(matrix)
    return _raw.arrow_correlation_matrix(matrix)


def arrow_gram_schmidt(matrix):
    if _is_complex_matrix(matrix):
        return arrow_gram_schmidt_complex(matrix)
    return _raw.arrow_gram_schmidt(matrix)


def arrow_gram_schmidt_classic(matrix):
    _require_real("arrow_gram_schmidt_classic", matrix=matrix)
    return _raw.arrow_gram_schmidt_classic(matrix)


def arrow_solve_lower(matrix, rhs):
    if _complex_mode("arrow_solve_lower", matrix=matrix, rhs=rhs):
        return arrow_solve_lower_complex(matrix, rhs)
    return _raw.arrow_solve_lower(matrix, rhs)


def arrow_solve_upper(matrix, rhs):
    if _complex_mode("arrow_solve_upper", matrix=matrix, rhs=rhs):
        return arrow_solve_upper_complex(matrix, rhs)
    return _raw.arrow_solve_upper(matrix, rhs)


def arrow_solve_lower_matrix(matrix, rhs):
    _require_real("arrow_solve_lower_matrix", matrix=matrix, rhs=rhs)
    return _raw.arrow_solve_lower_matrix(matrix, rhs)


def arrow_solve_upper_matrix(matrix, rhs):
    _require_real("arrow_solve_upper_matrix", matrix=matrix, rhs=rhs)
    return _raw.arrow_solve_upper_matrix(matrix, rhs)


def arrow_svd_decompose(data) -> SvdResult:
    if _is_complex_matrix(data):
        return arrow_svd_decompose_complex(data)
    return _svd_result(_raw.arrow_svd_decompose(data))


def arrow_svd_decompose_truncated(data, k) -> SvdResult:
    _require_real("arrow_svd_decompose_truncated", data=data)
    return _svd_result(_raw.arrow_svd_decompose_truncated(data, k))


def arrow_svd_decompose_with_tolerance(data, tolerance) -> SvdResult:
    _require_real("arrow_svd_decompose_with_tolerance", data=data)
    return _svd_result(_raw.arrow_svd_decompose_with_tolerance(data, tolerance))


def arrow_svd_pseudo_inverse(data):
    _require_real("arrow_svd_pseudo_inverse", data=data)
    return _raw.arrow_svd_pseudo_inverse(data)


def arrow_svd_null_space(data, tolerance=None):
    _require_real("arrow_svd_null_space", data=data)
    return _raw.arrow_svd_null_space(data, tolerance)


def arrow_qr_decompose(data, rank_tolerance=None, max_iterations=None) -> QrResult:
    if _is_complex_matrix(data):
        return arrow_qr_decompose_complex(data)
    return _qr_result(
        _raw.arrow_qr_decompose(
            data,
            rank_tolerance=rank_tolerance,
            max_iterations=max_iterations,
        )
    )


def arrow_qr_decompose_reduced(data, rank_tolerance=None, max_iterations=None) -> QrResult:
    _require_real("arrow_qr_decompose_reduced", data=data)
    return _qr_result(
        _raw.arrow_qr_decompose_reduced(
            data,
            rank_tolerance=rank_tolerance,
            max_iterations=max_iterations,
        )
    )


def arrow_qr_decompose_pivoted(data, rank_tolerance=None, max_iterations=None) -> QrResult:
    _require_real("arrow_qr_decompose_pivoted", data=data)
    return _qr_result(
        _raw.arrow_qr_decompose_pivoted(
            data,
            rank_tolerance=rank_tolerance,
            max_iterations=max_iterations,
        )
    )


def arrow_qr_solve_least_squares(a, b, rank_tolerance=None, max_iterations=None):
    _require_real("arrow_qr_solve_least_squares", a=a, b=b)
    return _raw.arrow_qr_solve_least_squares(
        a,
        b,
        rank_tolerance=rank_tolerance,
        max_iterations=max_iterations,
    )


def arrow_lu_decompose(matrix) -> LuResult:
    _require_real("arrow_lu_decompose", matrix=matrix)
    return _lu_result(_raw.arrow_lu_decompose(matrix))


def arrow_lu_solve(matrix, rhs):
    if _complex_mode("arrow_lu_solve", matrix=matrix, rhs=rhs):
        return arrow_lu_solve_complex(matrix, rhs)
    return _raw.arrow_lu_solve(matrix, rhs)


def arrow_lu_inverse(matrix):
    if _is_complex_matrix(matrix):
        return arrow_lu_inverse_complex(matrix)
    return _raw.arrow_lu_inverse(matrix)


def arrow_lu_determinant(matrix):
    if _is_complex_matrix(matrix):
        return arrow_lu_determinant_complex(matrix)
    return _raw.arrow_lu_determinant(matrix)


def arrow_lu_log_determinant(matrix) -> LogDetResult:
    _require_real("arrow_lu_log_determinant", matrix=matrix)
    return _log_det_result(_raw.arrow_lu_log_determinant(matrix))


def arrow_cholesky_decompose(matrix) -> CholeskyResult:
    if _is_complex_matrix(matrix):
        return arrow_cholesky_decompose_complex(matrix)
    return _cholesky_result(_raw.arrow_cholesky_decompose(matrix))


def arrow_cholesky_solve(matrix, rhs):
    if _complex_mode("arrow_cholesky_solve", matrix=matrix, rhs=rhs):
        return arrow_cholesky_solve_complex(matrix, rhs)
    return _raw.arrow_cholesky_solve(matrix, rhs)


def arrow_cholesky_inverse(matrix):
    if _is_complex_matrix(matrix):
        return arrow_cholesky_inverse_complex(matrix)
    return _raw.arrow_cholesky_inverse(matrix)


def arrow_eigen_symmetric(matrix) -> EigenResult:
    _require_real("arrow_eigen_symmetric", matrix=matrix)
    return _eigen_result(_raw.arrow_eigen_symmetric(matrix))


def arrow_eigen_generalized(matrix_a, matrix_b) -> GeneralizedEigenResult:
    _require_real("arrow_eigen_generalized", matrix_a=matrix_a, matrix_b=matrix_b)
    return _generalized_eigen_result(_raw.arrow_eigen_generalized(matrix_a, matrix_b))


def arrow_eigen_nonsymmetric(matrix) -> NonsymmetricEigenResult:
    if _is_complex_matrix(matrix):
        return arrow_eigen_nonsymmetric_complex(matrix)
    return _nonsymmetric_eigen_result(_raw.arrow_eigen_nonsymmetric(matrix))


def arrow_eigen_nonsymmetric_bi(matrix) -> NonsymmetricBiEigenResult:
    _require_real("arrow_eigen_nonsymmetric_bi", matrix=matrix)
    return _nonsymmetric_bi_eigen_result(_raw.arrow_eigen_nonsymmetric_bi(matrix))


def arrow_schur_compute(matrix) -> SchurResult:
    if _is_complex_matrix(matrix):
        return arrow_schur_compute_complex(matrix)
    return _schur_result(_raw.arrow_schur_compute(matrix))


def arrow_polar_compute(matrix) -> PolarResult:
    if _is_complex_matrix(matrix):
        return arrow_polar_compute_complex(matrix)
    return _polar_result(_raw.arrow_polar_compute(matrix))


def arrow_matrix_exp(matrix, max_terms=None, tolerance=None):
    if _is_complex_matrix(matrix):
        return arrow_matrix_exp_complex(matrix, max_terms=max_terms, tolerance=tolerance)
    return _raw.arrow_matrix_exp(matrix, max_terms=max_terms, tolerance=tolerance)


def arrow_matrix_exp_eigen(matrix):
    if _is_complex_matrix(matrix):
        return arrow_matrix_exp_eigen_complex(matrix)
    return _raw.arrow_matrix_exp_eigen(matrix)


def arrow_matrix_log_taylor(matrix, max_terms=None, tolerance=None):
    _require_real("arrow_matrix_log_taylor", matrix=matrix)
    return _raw.arrow_matrix_log_taylor(matrix, max_terms=max_terms, tolerance=tolerance)


def arrow_matrix_log_eigen(matrix):
    if _is_complex_matrix(matrix):
        return arrow_matrix_log_eigen_complex(matrix)
    return _raw.arrow_matrix_log_eigen(matrix)


def arrow_matrix_log_svd(matrix):
    if _is_complex_matrix(matrix):
        return arrow_matrix_log_svd_complex(matrix)
    return _raw.arrow_matrix_log_svd(matrix)


def arrow_matrix_power(matrix, power):
    if _is_complex_matrix(matrix):
        return arrow_matrix_power_complex(matrix, power)
    return _raw.arrow_matrix_power(matrix, power)


def arrow_matrix_sign(matrix):
    if _is_complex_matrix(matrix):
        return arrow_matrix_sign_complex(matrix)
    return _raw.arrow_matrix_sign(matrix)


def arrow_compute_pca(matrix, n_components=None) -> PcaResult:
    if _is_complex_matrix(matrix):
        return arrow_compute_pca_complex(matrix, n_components=n_components)
    return _pca_result(_raw.arrow_compute_pca(matrix, n_components=n_components))


def arrow_pca_transform(matrix, result: PcaResult):
    if np.iscomplexobj(result.components):
        return arrow_pca_transform_complex(matrix, result)
    return _raw.arrow_pca_transform(matrix, result.components, result.mean)


def arrow_pca_inverse_transform(scores, result: PcaResult):
    if np.iscomplexobj(result.components):
        return arrow_pca_inverse_transform_complex(scores, result)
    return _raw.arrow_pca_inverse_transform(scores, result.components, result.mean)


def arrow_linear_regression(x, y, add_intercept=True) -> RegressionResult:
    if _complex_mode("arrow_linear_regression", x=x, y=y):
        return arrow_linear_regression_complex(x, y, add_intercept=add_intercept)
    return _regression_result(_raw.arrow_linear_regression(x, y, add_intercept=add_intercept))


def arrow_conjugate_gradient(matrix, rhs, tolerance=None, max_iterations=None, *, config: IterativeConfig | None = None):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if _complex_mode("arrow_conjugate_gradient", matrix=matrix, rhs=rhs):
        return arrow_conjugate_gradient_complex(matrix, rhs, **kwargs)
    return _raw.arrow_conjugate_gradient(matrix, rhs, **kwargs)


def arrow_conjugate_gradient_complex(matrix, rhs, tolerance=None, max_iterations=None, *, config: IterativeConfig | None = None):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_complex("arrow_conjugate_gradient_complex", matrix=matrix, rhs=rhs)
    field, storage = _raw.arrow_conjugate_gradient_complex(
        _complex_matrix_storage(matrix),
        _complex_vector_field("rhs"),
        _complex_vector_storage(rhs),
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_gmres(matrix, rhs, tolerance=None, max_iterations=None, *, config: IterativeConfig | None = None):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if _complex_mode("arrow_gmres", matrix=matrix, rhs=rhs):
        return arrow_gmres_complex(matrix, rhs, **kwargs)
    return _raw.arrow_gmres(matrix, rhs, **kwargs)


def arrow_gmres_complex(matrix, rhs, tolerance=None, max_iterations=None, *, config: IterativeConfig | None = None):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_complex("arrow_gmres_complex", matrix=matrix, rhs=rhs)
    field, storage = _raw.arrow_gmres_complex(
        _complex_matrix_storage(matrix),
        _complex_vector_field("rhs"),
        _complex_vector_storage(rhs),
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_numerical_jacobian(function, x, step_size=None, tolerance=None, max_iterations=None, *, config: JacobianConfig | None = None):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_real("arrow_numerical_jacobian", x=x)
    return _raw.arrow_numerical_jacobian(function, x, **kwargs)


def arrow_numerical_jacobian_central(function, x, step_size=None, tolerance=None, max_iterations=None, *, config: JacobianConfig | None = None):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_real("arrow_numerical_jacobian_central", x=x)
    return _raw.arrow_numerical_jacobian_central(function, x, **kwargs)


def arrow_numerical_gradient(function, x, step_size=None, tolerance=None, max_iterations=None, *, config: JacobianConfig | None = None):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_real("arrow_numerical_gradient", x=x)
    return _raw.arrow_numerical_gradient(function, x, **kwargs)


def arrow_numerical_hessian(function, x, step_size=None, tolerance=None, max_iterations=None, *, config: JacobianConfig | None = None):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_real("arrow_numerical_hessian", x=x)
    return _raw.arrow_numerical_hessian(function, x, **kwargs)


def arrow_backtracking_line_search(
    point,
    direction,
    objective,
    gradient,
    initial_step=None,
    contraction=None,
    sufficient_decrease=None,
    max_iterations=None,
    *,
    config: LineSearchConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        LineSearchConfig,
        initial_step=initial_step,
        contraction=contraction,
        sufficient_decrease=sufficient_decrease,
        max_iterations=max_iterations,
    )
    if _complex_mode("arrow_backtracking_line_search", point=point, direction=direction):
        return arrow_backtracking_line_search_complex(point, direction, objective, gradient, **kwargs)
    return _raw.arrow_backtracking_line_search(point, direction, objective, gradient, **kwargs)


def arrow_backtracking_line_search_complex(
    point,
    direction,
    objective,
    gradient,
    initial_step=None,
    contraction=None,
    sufficient_decrease=None,
    max_iterations=None,
    *,
    config: LineSearchConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        LineSearchConfig,
        initial_step=initial_step,
        contraction=contraction,
        sufficient_decrease=sufficient_decrease,
        max_iterations=max_iterations,
    )
    _require_complex("arrow_backtracking_line_search_complex", point=point, direction=direction)
    return _raw.arrow_backtracking_line_search_complex(
        _complex_vector_field("point"),
        _complex_vector_storage(point),
        _complex_vector_field("direction"),
        _complex_vector_storage(direction),
        objective,
        gradient,
        **kwargs,
    )


def arrow_gradient_descent(initial, objective, gradient, learning_rate=None, max_iterations=None, tolerance=None, *, config: GradientDescentConfig | None = None):
    kwargs = _resolve_config(
        config,
        GradientDescentConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_gradient_descent_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_gradient_descent", initial=initial)
    return _raw.arrow_gradient_descent(initial, objective, gradient, **kwargs)


def arrow_gradient_descent_complex(initial, objective, gradient, learning_rate=None, max_iterations=None, tolerance=None, *, config: GradientDescentConfig | None = None):
    kwargs = _resolve_config(
        config,
        GradientDescentConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_gradient_descent_complex", initial=initial)
    field, storage = _raw.arrow_gradient_descent_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_adam(
    initial,
    objective,
    gradient,
    learning_rate=None,
    beta1=None,
    beta2=None,
    epsilon=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: AdamConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        AdamConfig,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_adam_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_adam", initial=initial)
    return _raw.arrow_adam(initial, objective, gradient, **kwargs)


def arrow_adam_complex(
    initial,
    objective,
    gradient,
    learning_rate=None,
    beta1=None,
    beta2=None,
    epsilon=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: AdamConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        AdamConfig,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_adam_complex", initial=initial)
    field, storage = _raw.arrow_adam_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_momentum_descent(
    initial,
    objective,
    gradient,
    learning_rate=None,
    momentum=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: MomentumConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        MomentumConfig,
        learning_rate=learning_rate,
        momentum=momentum,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_momentum_descent_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_momentum_descent", initial=initial)
    return _raw.arrow_momentum_descent(initial, objective, gradient, **kwargs)


def arrow_momentum_descent_complex(
    initial,
    objective,
    gradient,
    learning_rate=None,
    momentum=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: MomentumConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        MomentumConfig,
        learning_rate=learning_rate,
        momentum=momentum,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_momentum_descent_complex", initial=initial)
    field, storage = _raw.arrow_momentum_descent_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_rmsprop(
    initial,
    objective,
    gradient,
    learning_rate=None,
    rho=None,
    epsilon=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: RMSPropConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        RMSPropConfig,
        learning_rate=learning_rate,
        rho=rho,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_rmsprop_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_rmsprop", initial=initial)
    return _raw.arrow_rmsprop(initial, objective, gradient, **kwargs)


def arrow_rmsprop_complex(
    initial,
    objective,
    gradient,
    learning_rate=None,
    rho=None,
    epsilon=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: RMSPropConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        RMSPropConfig,
        learning_rate=learning_rate,
        rho=rho,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_rmsprop_complex", initial=initial)
    field, storage = _raw.arrow_rmsprop_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_projected_gradient_descent_box(
    initial,
    objective,
    gradient,
    lower_bounds,
    upper_bounds,
    learning_rate=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: ProjectedGradientConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        ProjectedGradientConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _complex_mode(
        "arrow_projected_gradient_descent_box",
        initial=initial,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    ):
        return arrow_projected_gradient_descent_box_complex(
            initial,
            objective,
            gradient,
            lower_bounds,
            upper_bounds,
            **kwargs,
        )
    return _raw.arrow_projected_gradient_descent_box(
        initial,
        objective,
        gradient,
        lower_bounds,
        upper_bounds,
        **kwargs,
    )


def arrow_projected_gradient_descent_box_complex(
    initial,
    objective,
    gradient,
    lower_bounds,
    upper_bounds,
    learning_rate=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: ProjectedGradientConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        ProjectedGradientConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex(
        "arrow_projected_gradient_descent_box_complex",
        initial=initial,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    field, storage = _raw.arrow_projected_gradient_descent_box_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        _complex_vector_storage(lower_bounds),
        _complex_vector_storage(upper_bounds),
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_stochastic_gradient_descent(
    initial,
    stochastic_gradient,
    learning_rate=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: GradientDescentConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        GradientDescentConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_stochastic_gradient_descent_complex(initial, stochastic_gradient, **kwargs)
    _require_real("arrow_stochastic_gradient_descent", initial=initial)
    return _raw.arrow_stochastic_gradient_descent(initial, stochastic_gradient, **kwargs)


def arrow_stochastic_gradient_descent_complex(
    initial,
    stochastic_gradient,
    learning_rate=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: GradientDescentConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        GradientDescentConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_stochastic_gradient_descent_complex", initial=initial)
    field, storage = _raw.arrow_stochastic_gradient_descent_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        stochastic_gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_bfgs(
    initial,
    objective,
    gradient,
    step_size=None,
    max_iterations=None,
    tolerance=None,
    curvature_tolerance=None,
    *,
    config: BFGSConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        BFGSConfig,
        step_size=step_size,
        max_iterations=max_iterations,
        tolerance=tolerance,
        curvature_tolerance=curvature_tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_bfgs_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_bfgs", initial=initial)
    return _raw.arrow_bfgs(initial, objective, gradient, **kwargs)


def arrow_bfgs_complex(
    initial,
    objective,
    gradient,
    step_size=None,
    max_iterations=None,
    tolerance=None,
    curvature_tolerance=None,
    *,
    config: BFGSConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        BFGSConfig,
        step_size=step_size,
        max_iterations=max_iterations,
        tolerance=tolerance,
        curvature_tolerance=curvature_tolerance,
    )
    _require_complex("arrow_bfgs_complex", initial=initial)
    field, storage = _raw.arrow_bfgs_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


__all__ = [
    "arrow_adam",
    "arrow_adam_complex",
    "arrow_backtracking_line_search",
    "arrow_backtracking_line_search_complex",
    "arrow_batched_cholesky",
    "arrow_batched_cosine_distance",
    "arrow_batched_cosine_similarity",
    "arrow_batched_cosine_similarity_complex",
    "arrow_batched_dot",
    "arrow_batched_dot_hermitian",
    "arrow_batched_l2_norm",
    "arrow_batched_l2_norm_complex",
    "arrow_batched_lu",
    "arrow_batched_matmat",
    "arrow_batched_matmat_broadcast_left",
    "arrow_batched_matmat_broadcast_right",
    "arrow_batched_normalize",
    "arrow_batched_normalize_complex",
    "arrow_batched_qr",
    "arrow_batched_row_matvec",
    "arrow_batched_svd",
    "arrow_batched_symmetric_eigen",
    "arrow_bfgs",
    "arrow_bfgs_complex",
    "arrow_center_columns",
    "arrow_center_columns_complex",
    "arrow_cholesky_decompose",
    "arrow_cholesky_decompose_complex",
    "arrow_cholesky_inverse",
    "arrow_cholesky_inverse_complex",
    "arrow_cholesky_solve",
    "arrow_cholesky_solve_complex",
    "arrow_column_means",
    "arrow_column_means_complex",
    "arrow_conjugate_gradient",
    "arrow_conjugate_gradient_complex",
    "arrow_compute_pca",
    "arrow_compute_pca_complex",
    "arrow_correlation_matrix",
    "arrow_correlation_matrix_complex",
    "arrow_cosine_distance",
    "arrow_cosine_similarity",
    "arrow_cosine_similarity_complex",
    "arrow_covariance_matrix",
    "arrow_covariance_matrix_complex",
    "arrow_dot",
    "arrow_dot_hermitian",
    "arrow_eigen_generalized",
    "arrow_eigen_nonsymmetric",
    "arrow_eigen_nonsymmetric_bi",
    "arrow_eigen_nonsymmetric_complex",
    "arrow_eigen_symmetric",
    "arrow_gram_schmidt",
    "arrow_gram_schmidt_classic",
    "arrow_gram_schmidt_complex",
    "arrow_gmres",
    "arrow_gmres_complex",
    "arrow_gradient_descent",
    "arrow_gradient_descent_complex",
    "arrow_l2_norm",
    "arrow_l2_norm_complex",
    "arrow_linear_regression",
    "arrow_linear_regression_complex",
    "arrow_lu_decompose",
    "arrow_lu_determinant",
    "arrow_lu_determinant_complex",
    "arrow_lu_inverse",
    "arrow_lu_inverse_complex",
    "arrow_lu_log_determinant",
    "arrow_lu_solve",
    "arrow_lu_solve_complex",
    "arrow_matmat",
    "arrow_matmat_complex",
    "arrow_matvec",
    "arrow_matvec_complex",
    "arrow_matrix_exp",
    "arrow_matrix_exp_complex",
    "arrow_matrix_exp_eigen",
    "arrow_matrix_exp_eigen_complex",
    "arrow_matrix_log_eigen",
    "arrow_matrix_log_eigen_complex",
    "arrow_matrix_log_svd",
    "arrow_matrix_log_svd_complex",
    "arrow_matrix_log_taylor",
    "arrow_matrix_power",
    "arrow_matrix_power_complex",
    "arrow_matrix_sign",
    "arrow_matrix_sign_complex",
    "arrow_momentum_descent",
    "arrow_momentum_descent_complex",
    "arrow_numerical_gradient",
    "arrow_numerical_hessian",
    "arrow_numerical_jacobian",
    "arrow_numerical_jacobian_central",
    "arrow_pairwise_cosine_distance",
    "arrow_pairwise_cosine_similarity",
    "arrow_pairwise_l2_distance",
    "arrow_pca_inverse_transform",
    "arrow_pca_inverse_transform_complex",
    "arrow_pca_transform",
    "arrow_pca_transform_complex",
    "arrow_polar_compute",
    "arrow_polar_compute_complex",
    "arrow_projected_gradient_descent_box",
    "arrow_projected_gradient_descent_box_complex",
    "arrow_qr_decompose",
    "arrow_qr_decompose_complex",
    "arrow_qr_decompose_pivoted",
    "arrow_qr_decompose_reduced",
    "arrow_qr_solve_least_squares",
    "arrow_rmsprop",
    "arrow_rmsprop_complex",
    "arrow_schur_compute",
    "arrow_schur_compute_complex",
    "arrow_solve_lower",
    "arrow_solve_lower_complex",
    "arrow_solve_lower_matrix",
    "arrow_solve_upper",
    "arrow_solve_upper_complex",
    "arrow_solve_upper_matrix",
    "arrow_svd_decompose",
    "arrow_svd_decompose_complex",
    "arrow_svd_decompose_truncated",
    "arrow_svd_decompose_with_tolerance",
    "arrow_svd_null_space",
    "arrow_svd_pseudo_inverse",
    "arrow_stochastic_gradient_descent",
    "arrow_stochastic_gradient_descent_complex",
]
