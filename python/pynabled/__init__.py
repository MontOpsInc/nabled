"""pynabled - Python bindings for nabled linear algebra and ML library."""

from __future__ import annotations

import pynabled._pynabled as _raw

from .results import (
    CholeskyResult,
    CpAls3Result,
    CpAlsNdResult,
    CpAlsReport,
    CpConvergenceReport,
    CpErrorMetrics,
    EigenResult,
    GeneralizedEigenResult,
    Hosvd3Result,
    HosvdNdResult,
    LuResult,
    NonsymmetricEigenResult,
    PcaResult,
    PolarResult,
    QrResult,
    RegressionResult,
    SchurResult,
    SvdResult,
    TensorTrainResult,
)
from .sparse import (
    CsrMatrix,
    sparse_jacobi_solve,
    sparse_matmat_dense,
    sparse_matvec,
    sparse_pcg_solve,
    sparse_transpose,
)


def _cp_metrics(raw_metrics) -> CpErrorMetrics:
    return CpErrorMetrics(
        signal_norm=raw_metrics[0],
        residual_norm=raw_metrics[1],
        relative_error=raw_metrics[2],
        fit=raw_metrics[3],
    )


def _cp_report(raw_report) -> CpAlsReport:
    raw_convergence, raw_metrics = raw_report
    return CpAlsReport(
        convergence=CpConvergenceReport(
            iterations_run=raw_convergence[0],
            converged=raw_convergence[1],
            final_max_factor_change=raw_convergence[2],
        ),
        metrics=_cp_metrics(raw_metrics),
    )


def _cp_als_nd_result(weights, factors) -> CpAlsNdResult:
    factors = list(factors)
    return CpAlsNdResult(
        weights=weights,
        factors=factors,
        shape=tuple(factor.shape[0] for factor in factors),
    )


def _tt_result(cores) -> TensorTrainResult:
    return TensorTrainResult(cores=list(cores))


def svd_decompose(a) -> SvdResult:
    u, singular_values, vt = _raw.svd_decompose(a)
    return SvdResult(u=u, singular_values=singular_values, vt=vt)


def svd_decompose_truncated(a, k) -> SvdResult:
    u, singular_values, vt = _raw.svd_decompose_truncated(a, k)
    return SvdResult(u=u, singular_values=singular_values, vt=vt)


def svd_pseudo_inverse(a):
    return _raw.svd_pseudo_inverse(a)


def svd_reconstruct_matrix(result: SvdResult):
    return _raw.svd_reconstruct_matrix(result.u, result.singular_values, result.vt)


def svd_condition_number(result: SvdResult):
    return _raw.svd_condition_number(result.u, result.singular_values, result.vt)


def svd_rank(result: SvdResult, tolerance=None):
    return _raw.svd_rank(result.singular_values, tolerance)


def svd_null_space(a, tolerance=None):
    return _raw.svd_null_space(a, tolerance)


def qr_decompose(a) -> QrResult:
    q, r, rank = _raw.qr_decompose(a)
    return QrResult(q=q, r=r, rank=rank)


def qr_solve_least_squares(a, b):
    return _raw.qr_solve_least_squares(a, b)


def lu_decompose(a) -> LuResult:
    l, u = _raw.lu_decompose(a)
    return LuResult(l=l, u=u)


def lu_solve(a, b):
    return _raw.lu_solve(a, b)


def lu_inverse(a):
    return _raw.lu_inverse(a)


def lu_determinant(a):
    return _raw.lu_determinant(a)


def cholesky_decompose(a) -> CholeskyResult:
    return CholeskyResult(l=_raw.cholesky_decompose(a))


def cholesky_solve(a, b):
    return _raw.cholesky_solve(a, b)


def cholesky_inverse(a):
    return _raw.cholesky_inverse(a)


def eigen_symmetric(a) -> EigenResult:
    eigenvalues, eigenvectors = _raw.eigen_symmetric(a)
    return EigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def eigen_generalized(a, b) -> GeneralizedEigenResult:
    eigenvalues, eigenvectors = _raw.eigen_generalized(a, b)
    return GeneralizedEigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def eigen_nonsymmetric(a) -> NonsymmetricEigenResult:
    eigenvalues, schur_vectors = _raw.eigen_nonsymmetric(a)
    return NonsymmetricEigenResult(eigenvalues=eigenvalues, schur_vectors=schur_vectors)


def schur_compute(a) -> SchurResult:
    t, q = _raw.schur_compute(a)
    return SchurResult(q=q, t=t)


def polar_compute(a) -> PolarResult:
    u, p = _raw.polar_compute(a)
    return PolarResult(u=u, p=p)


def sylvester_solve(a, b, c):
    return _raw.sylvester_solve(a, b, c)


def lyapunov_solve(a, q):
    return _raw.lyapunov_solve(a, q)


def batched_qr(matrices) -> list[QrResult]:
    return [QrResult(q=q, r=r, rank=rank) for q, r, rank in _raw.batched_qr(matrices)]


def batched_svd(matrices) -> list[SvdResult]:
    return [SvdResult(u=u, singular_values=s, vt=vt) for u, s, vt in _raw.batched_svd(matrices)]


def batched_lu(matrices) -> list[LuResult]:
    return [LuResult(l=l, u=u) for l, u in _raw.batched_lu(matrices)]


def batched_cholesky(matrices) -> list[CholeskyResult]:
    return [CholeskyResult(l=l) for l in _raw.batched_cholesky(matrices)]


def batched_symmetric_eigen(matrices) -> list[EigenResult]:
    return [
        EigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        for eigenvalues, eigenvectors in _raw.batched_symmetric_eigen(matrices)
    ]


def compute_pca(x, n_components=None) -> PcaResult:
    components, explained_variance, explained_variance_ratio, mean, scores = _raw.compute_pca(
        x,
        n_components=n_components,
    )
    return PcaResult(
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean,
        scores=scores,
    )


def compute_pca_complex(x, n_components=None) -> PcaResult:
    components, explained_variance, explained_variance_ratio, mean, scores = (
        _raw.compute_pca_complex(x, n_components=n_components)
    )
    return PcaResult(
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean,
        scores=scores,
    )


def pca_transform(x, result: PcaResult):
    return _raw.pca_transform(x, result.components, result.mean)


def pca_transform_complex(x, result: PcaResult):
    return _raw.pca_transform_complex(x, result.components, result.mean)


def pca_inverse_transform(scores, result: PcaResult):
    return _raw.pca_inverse_transform(scores, result.components, result.mean)


def pca_inverse_transform_complex(scores, result: PcaResult):
    return _raw.pca_inverse_transform_complex(scores, result.components, result.mean)


def linear_regression(x, y) -> RegressionResult:
    coefficients, fitted_values, residuals, r_squared = _raw.linear_regression(x, y)
    return RegressionResult(
        coefficients=coefficients,
        fitted_values=fitted_values,
        residuals=residuals,
        r_squared=r_squared,
    )


def linear_regression_complex(x, y) -> RegressionResult:
    coefficients, fitted_values, residuals, r_squared = _raw.linear_regression_complex(x, y)
    return RegressionResult(
        coefficients=coefficients,
        fitted_values=fitted_values,
        residuals=residuals,
        r_squared=r_squared,
    )


def tensor_hosvd3(cube, r0, r1, r2) -> Hosvd3Result:
    core, u0, u1, u2 = _raw.tensor_hosvd3(cube, r0, r1, r2)
    return Hosvd3Result(core=core, u0=u0, u1=u1, u2=u2)


def tensor_hosvd3_reconstruct(result: Hosvd3Result):
    return _raw.tensor_hosvd3_reconstruct(result.core, result.u0, result.u1, result.u2)


def tensor_hosvd_nd(tensor, ranks) -> HosvdNdResult:
    core, factors = _raw.tensor_hosvd_nd(tensor, ranks)
    return HosvdNdResult(core=core, factors=list(factors))


def tensor_hooi_nd(tensor, ranks, max_iterations=None, tolerance=None) -> HosvdNdResult:
    core, factors = _raw.tensor_hooi_nd(
        tensor,
        ranks,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return HosvdNdResult(core=core, factors=list(factors))


def tensor_hosvd_nd_reconstruct(result: HosvdNdResult):
    return _raw.tensor_hosvd_nd_reconstruct(result.core, result.factors)


def tensor_tucker_project(tensor, result: HosvdNdResult):
    return _raw.tensor_tucker_project(tensor, result.factors)


def tensor_tucker_expand(result: HosvdNdResult):
    return _raw.tensor_tucker_expand(result.core, result.factors)


def tensor_cp_als3(cube, rank, max_iterations=None, tolerance=None) -> CpAls3Result:
    weights, factor_0, factor_1, factor_2 = _raw.tensor_cp_als3(
        cube,
        rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return CpAls3Result(
        weights=weights,
        factor_0=factor_0,
        factor_1=factor_1,
        factor_2=factor_2,
    )


def tensor_cp_als3_with_report(cube, rank, max_iterations=None, tolerance=None):
    raw_result, raw_report = _raw.tensor_cp_als3_with_report(
        cube,
        rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    weights, factor_0, factor_1, factor_2 = raw_result
    return (
        CpAls3Result(
            weights=weights,
            factor_0=factor_0,
            factor_1=factor_1,
            factor_2=factor_2,
        ),
        _cp_report(raw_report),
    )


def tensor_cp_als3_diagnostics(cube, result: CpAls3Result) -> CpErrorMetrics:
    return _cp_metrics(
        _raw.tensor_cp_als3_diagnostics(
            cube,
            result.weights,
            result.factor_0,
            result.factor_1,
            result.factor_2,
        )
    )


def tensor_cp_als3_reconstruct(result: CpAls3Result):
    return _raw.tensor_cp_als3_reconstruct(
        result.weights,
        result.factor_0,
        result.factor_1,
        result.factor_2,
    )


def tensor_cp_als_nd(tensor, rank, max_iterations=None, tolerance=None) -> CpAlsNdResult:
    weights, factors = _raw.tensor_cp_als_nd(
        tensor,
        rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return _cp_als_nd_result(weights, factors)


def tensor_cp_als_nd_with_report(tensor, rank, max_iterations=None, tolerance=None):
    raw_result, raw_report = _raw.tensor_cp_als_nd_with_report(
        tensor,
        rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    weights, factors = raw_result
    return (_cp_als_nd_result(weights, factors), _cp_report(raw_report))


def tensor_cp_als_nd_diagnostics(tensor, result: CpAlsNdResult) -> CpErrorMetrics:
    return _cp_metrics(_raw.tensor_cp_als_nd_diagnostics(tensor, result.weights, result.factors))


def tensor_cp_als_nd_reconstruct(result: CpAlsNdResult):
    return _raw.tensor_cp_als_nd_reconstruct(result.weights, result.factors)


def tensor_tt_svd(tensor, max_rank=None, tolerance=None) -> TensorTrainResult:
    return _tt_result(_raw.tensor_tt_svd(tensor, max_rank=max_rank, tolerance=tolerance))


def tensor_tt_orthogonalize_left(result: TensorTrainResult) -> TensorTrainResult:
    return _tt_result(_raw.tensor_tt_orthogonalize_left(result.cores))


def tensor_tt_orthogonalize_right(result: TensorTrainResult) -> TensorTrainResult:
    return _tt_result(_raw.tensor_tt_orthogonalize_right(result.cores))


def tensor_tt_round(result: TensorTrainResult, max_rank=None, tolerance=None) -> TensorTrainResult:
    return _tt_result(_raw.tensor_tt_round(result.cores, max_rank=max_rank, tolerance=tolerance))


def tensor_tt_inner(left: TensorTrainResult, right: TensorTrainResult):
    return _raw.tensor_tt_inner(left.cores, right.cores)


def tensor_tt_norm(result: TensorTrainResult):
    return _raw.tensor_tt_norm(result.cores)


def tensor_tt_add(left: TensorTrainResult, right: TensorTrainResult) -> TensorTrainResult:
    return _tt_result(_raw.tensor_tt_add(left.cores, right.cores))


def tensor_tt_hadamard(left: TensorTrainResult, right: TensorTrainResult) -> TensorTrainResult:
    return _tt_result(_raw.tensor_tt_hadamard(left.cores, right.cores))


def tensor_tt_hadamard_round(
    left: TensorTrainResult,
    right: TensorTrainResult,
    max_rank=None,
    tolerance=None,
) -> TensorTrainResult:
    return _tt_result(
        _raw.tensor_tt_hadamard_round(
            left.cores,
            right.cores,
            max_rank=max_rank,
            tolerance=tolerance,
        )
    )


def tensor_tt_svd_reconstruct(result: TensorTrainResult):
    return _raw.tensor_tt_svd_reconstruct(result.cores)


adam = _raw.adam
adam_complex = _raw.adam_complex
backtracking_line_search = _raw.backtracking_line_search
backtracking_line_search_complex = _raw.backtracking_line_search_complex
batched_matmat = _raw.batched_matmat
batched_row_matvec = _raw.batched_row_matvec
bfgs = _raw.bfgs
bfgs_complex = _raw.bfgs_complex
center_columns = _raw.center_columns
center_columns_complex = _raw.center_columns_complex
column_means = _raw.column_means
column_means_complex = _raw.column_means_complex
conjugate_gradient = _raw.conjugate_gradient
conjugate_gradient_complex = _raw.conjugate_gradient_complex
correlation_matrix = _raw.correlation_matrix
correlation_matrix_complex = _raw.correlation_matrix_complex
cosine_similarity = _raw.cosine_similarity
covariance_matrix = _raw.covariance_matrix
covariance_matrix_complex = _raw.covariance_matrix_complex
dot = _raw.dot
gmres = _raw.gmres
gmres_complex = _raw.gmres_complex
gradient_descent = _raw.gradient_descent
gradient_descent_complex = _raw.gradient_descent_complex
gram_schmidt = _raw.gram_schmidt
gram_schmidt_classic = _raw.gram_schmidt_classic
l2_norm = _raw.l2_norm
matmat = _raw.matmat
matrix_exp = _raw.matrix_exp
matrix_exp_eigen = _raw.matrix_exp_eigen
matrix_log_eigen = _raw.matrix_log_eigen
matrix_log_svd = _raw.matrix_log_svd
matrix_log_taylor = _raw.matrix_log_taylor
matrix_power = _raw.matrix_power
matrix_sign = _raw.matrix_sign
matvec = _raw.matvec
momentum_descent = _raw.momentum_descent
momentum_descent_complex = _raw.momentum_descent_complex
numerical_gradient = _raw.numerical_gradient
numerical_hessian = _raw.numerical_hessian
numerical_jacobian = _raw.numerical_jacobian
numerical_jacobian_central = _raw.numerical_jacobian_central
pairwise_cosine_similarity = _raw.pairwise_cosine_similarity
pairwise_l2_distance = _raw.pairwise_l2_distance
projected_gradient_descent_box = _raw.projected_gradient_descent_box
projected_gradient_descent_box_complex = _raw.projected_gradient_descent_box_complex
rmsprop = _raw.rmsprop
rmsprop_complex = _raw.rmsprop_complex
stochastic_gradient_descent = _raw.stochastic_gradient_descent
stochastic_gradient_descent_complex = _raw.stochastic_gradient_descent_complex
tensor_batched_dot_last_axis = _raw.tensor_batched_dot_last_axis
tensor_batched_dot_last_axis_complex = _raw.tensor_batched_dot_last_axis_complex
tensor_batched_matmul_last_two = _raw.tensor_batched_matmul_last_two
tensor_batched_matmul_last_two_complex = _raw.tensor_batched_matmul_last_two_complex
tensor_contract_axes = _raw.tensor_contract_axes
tensor_contract_axes_complex = _raw.tensor_contract_axes_complex
tensor_cube_matmat = _raw.tensor_cube_matmat
tensor_cube_matmat_complex = _raw.tensor_cube_matmat_complex
tensor_cube_matvec = _raw.tensor_cube_matvec
tensor_cube_matvec_complex = _raw.tensor_cube_matvec_complex
tensor_einsum = _raw.tensor_einsum
tensor_einsum_complex = _raw.tensor_einsum_complex
tensor_l2_norm_last_axis = _raw.tensor_l2_norm_last_axis
tensor_l2_norm_last_axis_complex = _raw.tensor_l2_norm_last_axis_complex
tensor_normalize_last_axis = _raw.tensor_normalize_last_axis
tensor_normalize_last_axis_complex = _raw.tensor_normalize_last_axis_complex
tensor_permute_axes = _raw.tensor_permute_axes
tensor_permute_axes_complex = _raw.tensor_permute_axes_complex
tensor_sum_last_axis = _raw.tensor_sum_last_axis
tensor_sum_last_axis_complex = _raw.tensor_sum_last_axis_complex
triangular_solve_lower = _raw.triangular_solve_lower
triangular_solve_lower_matrix = _raw.triangular_solve_lower_matrix
triangular_solve_upper = _raw.triangular_solve_upper
triangular_solve_upper_matrix = _raw.triangular_solve_upper_matrix


__all__ = [
    "SvdResult",
    "QrResult",
    "LuResult",
    "CholeskyResult",
    "EigenResult",
    "GeneralizedEigenResult",
    "NonsymmetricEigenResult",
    "SchurResult",
    "PolarResult",
    "PcaResult",
    "RegressionResult",
    "Hosvd3Result",
    "HosvdNdResult",
    "CpAls3Result",
    "CpAlsNdResult",
    "CpErrorMetrics",
    "CpConvergenceReport",
    "CpAlsReport",
    "TensorTrainResult",
    "svd_decompose",
    "svd_decompose_truncated",
    "svd_pseudo_inverse",
    "svd_reconstruct_matrix",
    "svd_condition_number",
    "svd_rank",
    "svd_null_space",
    "qr_decompose",
    "qr_solve_least_squares",
    "lu_decompose",
    "lu_solve",
    "lu_inverse",
    "lu_determinant",
    "cholesky_decompose",
    "cholesky_solve",
    "cholesky_inverse",
    "eigen_symmetric",
    "eigen_generalized",
    "eigen_nonsymmetric",
    "schur_compute",
    "polar_compute",
    "sylvester_solve",
    "lyapunov_solve",
    "triangular_solve_lower",
    "triangular_solve_upper",
    "triangular_solve_lower_matrix",
    "triangular_solve_upper_matrix",
    "matrix_exp",
    "matrix_exp_eigen",
    "matrix_log_taylor",
    "matrix_log_eigen",
    "matrix_log_svd",
    "matrix_power",
    "matrix_sign",
    "gram_schmidt",
    "gram_schmidt_classic",
    "batched_qr",
    "batched_svd",
    "batched_lu",
    "batched_cholesky",
    "batched_symmetric_eigen",
    "matvec",
    "matmat",
    "batched_row_matvec",
    "batched_matmat",
    "dot",
    "l2_norm",
    "cosine_similarity",
    "pairwise_l2_distance",
    "pairwise_cosine_similarity",
    "linear_regression",
    "linear_regression_complex",
    "compute_pca",
    "compute_pca_complex",
    "pca_transform",
    "pca_transform_complex",
    "pca_inverse_transform",
    "pca_inverse_transform_complex",
    "column_means",
    "column_means_complex",
    "center_columns",
    "center_columns_complex",
    "covariance_matrix",
    "covariance_matrix_complex",
    "correlation_matrix",
    "correlation_matrix_complex",
    "conjugate_gradient",
    "conjugate_gradient_complex",
    "gmres",
    "gmres_complex",
    "numerical_jacobian",
    "numerical_jacobian_central",
    "numerical_gradient",
    "numerical_hessian",
    "backtracking_line_search",
    "backtracking_line_search_complex",
    "gradient_descent",
    "gradient_descent_complex",
    "adam",
    "adam_complex",
    "momentum_descent",
    "momentum_descent_complex",
    "rmsprop",
    "rmsprop_complex",
    "projected_gradient_descent_box",
    "projected_gradient_descent_box_complex",
    "stochastic_gradient_descent",
    "stochastic_gradient_descent_complex",
    "bfgs",
    "bfgs_complex",
    "CsrMatrix",
    "sparse_matvec",
    "sparse_matmat_dense",
    "sparse_transpose",
    "sparse_jacobi_solve",
    "sparse_pcg_solve",
    "tensor_cube_matvec",
    "tensor_cube_matvec_complex",
    "tensor_cube_matmat",
    "tensor_cube_matmat_complex",
    "tensor_sum_last_axis",
    "tensor_sum_last_axis_complex",
    "tensor_l2_norm_last_axis",
    "tensor_l2_norm_last_axis_complex",
    "tensor_normalize_last_axis",
    "tensor_normalize_last_axis_complex",
    "tensor_batched_dot_last_axis",
    "tensor_batched_dot_last_axis_complex",
    "tensor_permute_axes",
    "tensor_permute_axes_complex",
    "tensor_contract_axes",
    "tensor_contract_axes_complex",
    "tensor_batched_matmul_last_two",
    "tensor_batched_matmul_last_two_complex",
    "tensor_einsum",
    "tensor_einsum_complex",
    "tensor_hosvd3",
    "tensor_hosvd3_reconstruct",
    "tensor_hosvd_nd",
    "tensor_hooi_nd",
    "tensor_hosvd_nd_reconstruct",
    "tensor_tucker_project",
    "tensor_tucker_expand",
    "tensor_cp_als3",
    "tensor_cp_als3_with_report",
    "tensor_cp_als3_diagnostics",
    "tensor_cp_als3_reconstruct",
    "tensor_cp_als_nd",
    "tensor_cp_als_nd_with_report",
    "tensor_cp_als_nd_diagnostics",
    "tensor_cp_als_nd_reconstruct",
    "tensor_tt_svd",
    "tensor_tt_orthogonalize_left",
    "tensor_tt_orthogonalize_right",
    "tensor_tt_round",
    "tensor_tt_inner",
    "tensor_tt_norm",
    "tensor_tt_add",
    "tensor_tt_hadamard",
    "tensor_tt_hadamard_round",
    "tensor_tt_svd_reconstruct",
]
