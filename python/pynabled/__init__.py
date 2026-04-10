"""pynabled - Python bindings for nabled linear algebra and ML library."""

from __future__ import annotations

import numpy as np

import pynabled._pynabled as _raw

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
    BalancedNonsymmetricResult,
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
    TensorTrainResult,
)
from .sparse import (
    CooMatrix,
    CscMatrix,
    CsrMatrix,
    IC0Factorization,
    ILDL0Factorization,
    ILU0Factorization,
    ILUKConfig,
    ILUKFactorization,
    ILUTConfig,
    ILUTFactorization,
    JacobiPreconditioner,
    SparseLUFactorization,
    sparse_bicgstab_solve,
    sparse_bicgstab_ildl0_solve,
    sparse_bicgstab_ildl0_solve_multiple,
    sparse_bicgstab_ilu0_solve,
    sparse_bicgstab_ilu0_solve_multiple,
    sparse_bicgstab_iluk_solve,
    sparse_bicgstab_iluk_solve_multiple,
    sparse_bicgstab_ilut_solve,
    sparse_bicgstab_ilut_solve_multiple,
    sparse_conjugate_gradient_solve,
    sparse_coo_to_csr,
    sparse_csc_to_csr,
    sparse_csr_to_csc,
    sparse_gauss_seidel_solve,
    sparse_gmres_ildl0_solve,
    sparse_gmres_ildl0_solve_multiple,
    sparse_gmres_ilu0_solve,
    sparse_gmres_ilu0_solve_multiple,
    sparse_gmres_iluk_solve,
    sparse_gmres_iluk_solve_multiple,
    sparse_gmres_ilut_solve,
    sparse_gmres_ilut_solve_multiple,
    sparse_ic0_factor,
    sparse_ildl0_factor,
    sparse_ilu0_factor,
    sparse_iluk_factor,
    sparse_ilut_factor,
    sparse_jacobi_preconditioner,
    sparse_jacobi_solve,
    sparse_lu_factor,
    sparse_lu_solve,
    sparse_matmat_dense,
    sparse_matmat_sparse,
    sparse_matvec,
    sparse_matvec_csc,
    sparse_pcg_ic0_solve,
    sparse_pcg_solve,
    sparse_transpose,
)
from .workspaces import (
    MatrixFunctionWorkspace,
    PairwiseCosineWorkspace,
    SchurWorkspace,
    SylvesterWorkspace,
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


def _require_workspace(workspace, workspace_type):
    if not isinstance(workspace, workspace_type):
        raise TypeError(f"workspace must be {workspace_type.__name__} or None")
    return workspace


def _require_result_out(out, result_type):
    if not isinstance(out, result_type):
        raise TypeError(f"out must be {result_type.__name__} or None")
    return out


def _array_unary_out(raw, raw_into, array, *, out=None):
    if out is None:
        return raw(array)
    raw_into(array, out)
    return out


def _array_binary_out(raw, raw_into, left, right, *, out=None):
    if out is None:
        return raw(left, right)
    raw_into(left, right, out)
    return out


def _array_ternary_out(raw, raw_into, left, middle, right, *, out=None):
    if out is None:
        return raw(left, middle, right)
    raw_into(left, middle, right, out)
    return out


def _array_unary_out_kwargs(raw, raw_into, array, *, out=None, **kwargs):
    if out is None:
        return raw(array, **kwargs)
    raw_into(array, out, **kwargs)
    return out


def _array_binary_scalar_out(raw, raw_into, array, scalar, *, out=None):
    if out is None:
        return raw(array, scalar)
    raw_into(array, scalar, out)
    return out


def build_features() -> tuple[str, ...]:
    """Return the Cargo feature names compiled into the installed extension."""
    return tuple(_raw.build_features())


def svd_decompose(a) -> SvdResult:
    u, singular_values, vt = _raw.svd_decompose(a)
    return SvdResult(u=u, singular_values=singular_values, vt=vt)


def svd_decompose_truncated(a, k) -> SvdResult:
    u, singular_values, vt = _raw.svd_decompose_truncated(a, k)
    return SvdResult(u=u, singular_values=singular_values, vt=vt)


def svd_pseudo_inverse(a, *, out=None):
    return _array_unary_out(_raw.svd_pseudo_inverse, _raw.svd_pseudo_inverse_into, a, out=out)


def svd_reconstruct_matrix(result: SvdResult, *, out=None):
    if out is None:
        return _raw.svd_reconstruct_matrix(result.u, result.singular_values, result.vt)
    _raw.svd_reconstruct_matrix_into(result.u, result.singular_values, result.vt, out)
    return out


def svd_condition_number(result: SvdResult):
    return _raw.svd_condition_number(result.singular_values)


def svd_rank(result: SvdResult, tolerance=None):
    return _raw.svd_rank(result.singular_values, tolerance)


def svd_null_space(a, tolerance=None):
    return _raw.svd_null_space(a, tolerance)


def qr_decompose(a, rank_tolerance=None, max_iterations=None) -> QrResult:
    q, r, rank = _raw.qr_decompose(
        a,
        rank_tolerance=rank_tolerance,
        max_iterations=max_iterations,
    )
    return QrResult(q=q, r=r, rank=rank)


def qr_decompose_reduced(a, rank_tolerance=None, max_iterations=None) -> QrResult:
    q, r, rank = _raw.qr_decompose_reduced(
        a,
        rank_tolerance=rank_tolerance,
        max_iterations=max_iterations,
    )
    return QrResult(q=q, r=r, rank=rank)


def qr_decompose_pivoted(a, rank_tolerance=None, max_iterations=None) -> QrResult:
    q, r, p, rank = _raw.qr_decompose_pivoted(
        a,
        rank_tolerance=rank_tolerance,
        max_iterations=max_iterations,
    )
    return QrResult(q=q, r=r, rank=rank, p=p)


def qr_reconstruct_matrix(result: QrResult, *, out=None):
    if result.p is None:
        return _array_binary_out(_raw.qr_reconstruct_matrix, _raw.qr_reconstruct_matrix_into, result.q, result.r, out=out)
    if out is None:
        return _raw.qr_reconstruct_matrix_pivoted(result.q, result.r, result.p)
    _raw.qr_reconstruct_matrix_pivoted_into(result.q, result.r, result.p, out)
    return out


def qr_condition_number(result: QrResult):
    return _raw.qr_condition_number(result.r)


def qr_solve_least_squares(a, b, rank_tolerance=None, max_iterations=None):
    return _raw.qr_solve_least_squares(
        a,
        b,
        rank_tolerance=rank_tolerance,
        max_iterations=max_iterations,
    )


def lu_decompose(a) -> LuResult:
    l, u = _raw.lu_decompose(a)
    return LuResult(l=l, u=u)


def lu_solve(a, b):
    return _raw.lu_solve(a, b)


def lu_inverse(a):
    return _raw.lu_inverse(a)


def lu_determinant(a):
    return _raw.lu_determinant(a)


def lu_log_determinant(a) -> LogDetResult:
    sign, ln_abs_det = _raw.lu_log_determinant(a)
    return LogDetResult(sign=sign, ln_abs_det=ln_abs_det)


def cholesky_decompose(a) -> CholeskyResult:
    return CholeskyResult(l=_raw.cholesky_decompose(a))


def cholesky_solve(a, b, *, out=None):
    if isinstance(a, CholeskyResult):
        if out is None:
            return _raw.cholesky_solve_from_factor(a.l, b)
        _raw.cholesky_solve_from_factor_into(a.l, b, out)
        return out
    return _array_binary_out(_raw.cholesky_solve, _raw.cholesky_solve_into, a, b, out=out)


def cholesky_inverse(a, *, out=None):
    if isinstance(a, CholeskyResult):
        return _array_unary_out(_raw.cholesky_inverse_from_factor, _raw.cholesky_inverse_from_factor_into, a.l, out=out)
    return _array_unary_out(_raw.cholesky_inverse, _raw.cholesky_inverse_into, a, out=out)


def eigen_symmetric(a) -> EigenResult:
    eigenvalues, eigenvectors = _raw.eigen_symmetric(a)
    return EigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def eigen_generalized(a, b) -> GeneralizedEigenResult:
    eigenvalues, eigenvectors = _raw.eigen_generalized(a, b)
    return GeneralizedEigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def eigen_nonsymmetric(a) -> NonsymmetricEigenResult:
    eigenvalues, schur_vectors = _raw.eigen_nonsymmetric(a)
    return NonsymmetricEigenResult(eigenvalues=eigenvalues, schur_vectors=schur_vectors)


def eigen_balance_nonsymmetric(
    a,
    balance=True,
    balance_max_iterations=None,
    balance_tolerance=None,
) -> BalancedNonsymmetricResult:
    balanced_matrix, balancing_diagonal = _raw.eigen_balance_nonsymmetric(
        a,
        balance=balance,
        balance_max_iterations=balance_max_iterations,
        balance_tolerance=balance_tolerance,
    )
    return BalancedNonsymmetricResult(
        balanced_matrix=balanced_matrix,
        balancing_diagonal=balancing_diagonal,
    )


def eigen_nonsymmetric_bi(
    a,
    balance=True,
    balance_max_iterations=None,
    balance_tolerance=None,
) -> NonsymmetricBiEigenResult:
    (
        eigenvalues,
        right_eigenvectors,
        left_eigenvectors,
        balancing_diagonal,
        balanced_matrix,
    ) = _raw.eigen_nonsymmetric_bi(
        a,
        balance=balance,
        balance_max_iterations=balance_max_iterations,
        balance_tolerance=balance_tolerance,
    )
    return NonsymmetricBiEigenResult(
        eigenvalues=eigenvalues,
        right_eigenvectors=right_eigenvectors,
        left_eigenvectors=left_eigenvectors,
        balancing_diagonal=balancing_diagonal,
        balanced_matrix=balanced_matrix,
    )


def schur_compute(a, *, out: SchurResult | None = None, workspace: SchurWorkspace | None = None) -> SchurResult:
    if workspace is not None:
        return _require_workspace(workspace, SchurWorkspace).compute(a, out=out)
    if out is None:
        t, q = _raw.schur_compute(a)
        return SchurResult(q=q, t=t)
    schur_out = _require_result_out(out, SchurResult)
    _raw.schur_compute_into(a, schur_out.q, schur_out.t)
    return schur_out


def polar_compute(a) -> PolarResult:
    u, p = _raw.polar_compute(a)
    return PolarResult(u=u, p=p)


def sylvester_solve(a, b, c, *, out=None, workspace: SylvesterWorkspace | None = None):
    if workspace is None:
        return _array_ternary_out(
            _raw.sylvester_solve,
            _raw.sylvester_solve_into,
            a,
            b,
            c,
            out=out,
        )
    return _require_workspace(workspace, SylvesterWorkspace).solve(a, b, c, out=out)


def lyapunov_solve(a, q, *, out=None, workspace: SylvesterWorkspace | None = None):
    if workspace is None:
        return _array_binary_out(_raw.lyapunov_solve, _raw.lyapunov_solve_into, a, q, out=out)
    return _require_workspace(workspace, SylvesterWorkspace).lyapunov(a, q, out=out)


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


def backtracking_line_search(
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
    return _raw.backtracking_line_search(point, direction, objective, gradient, **kwargs)


def backtracking_line_search_complex(
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
    return _raw.backtracking_line_search_complex(point, direction, objective, gradient, **kwargs)


def batched_matmat(left, right, *, out=None):
    return _array_binary_out(_raw.batched_matmat, _raw.batched_matmat_into, left, right, out=out)


def batched_matmat_broadcast_left(left, right, *, out=None):
    return _array_binary_out(
        _raw.batched_matmat_broadcast_left,
        _raw.batched_matmat_broadcast_left_into,
        left,
        right,
        out=out,
    )


def batched_matmat_broadcast_right(left, right, *, out=None):
    return _array_binary_out(
        _raw.batched_matmat_broadcast_right,
        _raw.batched_matmat_broadcast_right_into,
        left,
        right,
        out=out,
    )


def batched_row_matvec(matrix, vectors, *, out=None):
    return _array_binary_out(
        _raw.batched_row_matvec,
        _raw.batched_row_matvec_into,
        matrix,
        vectors,
        out=out,
    )


def batched_cosine_distance(left, right, *, out=None):
    return _array_binary_out(
        _raw.batched_cosine_distance,
        _raw.batched_cosine_distance_into,
        left,
        right,
        out=out,
    )


def batched_cosine_similarity(left, right, *, out=None):
    return _array_binary_out(
        _raw.batched_cosine_similarity,
        _raw.batched_cosine_similarity_into,
        left,
        right,
        out=out,
    )


def batched_dot(left, right, *, out=None):
    return _array_binary_out(_raw.batched_dot, _raw.batched_dot_into, left, right, out=out)


def batched_l2_norm(rows, *, out=None):
    return _array_unary_out(_raw.batched_l2_norm, _raw.batched_l2_norm_into, rows, out=out)


def batched_normalize(rows, *, out=None):
    return _array_unary_out(_raw.batched_normalize, _raw.batched_normalize_into, rows, out=out)


def bfgs(
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
    return _raw.bfgs(initial, objective, gradient, **kwargs)


def bfgs_complex(
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
    return _raw.bfgs_complex(initial, objective, gradient, **kwargs)


center_columns = _raw.center_columns
center_columns_complex = _raw.center_columns_complex
column_means = _raw.column_means
column_means_complex = _raw.column_means_complex


def conjugate_gradient(
    matrix_a,
    matrix_b,
    tolerance=None,
    max_iterations=None,
    *,
    config: IterativeConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _raw.conjugate_gradient(matrix_a, matrix_b, **kwargs)


def conjugate_gradient_complex(
    matrix_a,
    matrix_b,
    tolerance=None,
    max_iterations=None,
    *,
    config: IterativeConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _raw.conjugate_gradient_complex(matrix_a, matrix_b, **kwargs)


correlation_matrix = _raw.correlation_matrix
correlation_matrix_complex = _raw.correlation_matrix_complex
cosine_distance = _raw.cosine_distance
cosine_similarity = _raw.cosine_similarity
covariance_matrix = _raw.covariance_matrix
covariance_matrix_complex = _raw.covariance_matrix_complex
dot = _raw.dot


def gmres(
    matrix_a,
    matrix_b,
    tolerance=None,
    max_iterations=None,
    *,
    config: IterativeConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _raw.gmres(matrix_a, matrix_b, **kwargs)


def gmres_complex(
    matrix_a,
    matrix_b,
    tolerance=None,
    max_iterations=None,
    *,
    config: IterativeConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _raw.gmres_complex(matrix_a, matrix_b, **kwargs)


def gradient_descent(
    initial,
    objective,
    gradient,
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
    return _raw.gradient_descent(initial, objective, gradient, **kwargs)


def gradient_descent_complex(
    initial,
    objective,
    gradient,
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
    return _raw.gradient_descent_complex(initial, objective, gradient, **kwargs)


gram_schmidt = _raw.gram_schmidt
gram_schmidt_classic = _raw.gram_schmidt_classic
l2_norm = _raw.l2_norm


def matmat(left, right, *, out=None):
    return _array_binary_out(_raw.matmat, _raw.matmat_into, left, right, out=out)


def matrix_exp(
    matrix,
    max_terms=None,
    tolerance=None,
    *,
    out=None,
    workspace: MatrixFunctionWorkspace | None = None,
):
    if workspace is None:
        return _array_unary_out_kwargs(
            _raw.matrix_exp,
            _raw.matrix_exp_into,
            matrix,
            out=out,
            max_terms=max_terms,
            tolerance=tolerance,
        )
    return _require_workspace(workspace, MatrixFunctionWorkspace).exp(
        matrix,
        max_terms=max_terms,
        tolerance=tolerance,
        out=out,
    )


def matrix_exp_eigen(matrix, *, out=None, workspace: MatrixFunctionWorkspace | None = None):
    if workspace is None:
        return _array_unary_out(_raw.matrix_exp_eigen, _raw.matrix_exp_eigen_into, matrix, out=out)
    return _require_workspace(workspace, MatrixFunctionWorkspace).exp_eigen(matrix, out=out)


def matrix_log_eigen(matrix, *, out=None, workspace: MatrixFunctionWorkspace | None = None):
    if workspace is None:
        return _array_unary_out(_raw.matrix_log_eigen, _raw.matrix_log_eigen_into, matrix, out=out)
    return _require_workspace(workspace, MatrixFunctionWorkspace).log_eigen(matrix, out=out)


def matrix_log_svd(matrix, *, out=None, workspace: MatrixFunctionWorkspace | None = None):
    if workspace is None:
        return _array_unary_out(_raw.matrix_log_svd, _raw.matrix_log_svd_into, matrix, out=out)
    return _require_workspace(workspace, MatrixFunctionWorkspace).log_svd(matrix, out=out)


def matrix_log_taylor(
    matrix,
    max_terms=None,
    tolerance=None,
    *,
    out=None,
    workspace: MatrixFunctionWorkspace | None = None,
):
    if workspace is None:
        return _array_unary_out_kwargs(
            _raw.matrix_log_taylor,
            _raw.matrix_log_taylor_into,
            matrix,
            out=out,
            max_terms=max_terms,
            tolerance=tolerance,
        )
    return _require_workspace(workspace, MatrixFunctionWorkspace).log_taylor(
        matrix,
        max_terms=max_terms,
        tolerance=tolerance,
        out=out,
    )


def matrix_power(
    matrix,
    power,
    *,
    out=None,
    workspace: MatrixFunctionWorkspace | None = None,
):
    if workspace is None:
        return _array_binary_scalar_out(
            _raw.matrix_power,
            _raw.matrix_power_into,
            matrix,
            power,
            out=out,
        )
    return _require_workspace(workspace, MatrixFunctionWorkspace).power(matrix, power, out=out)


def matrix_sign(matrix, *, out=None, workspace: MatrixFunctionWorkspace | None = None):
    if workspace is None:
        return _array_unary_out(_raw.matrix_sign, _raw.matrix_sign_into, matrix, out=out)
    return _require_workspace(workspace, MatrixFunctionWorkspace).sign(matrix, out=out)


def matvec(matrix, vector, *, out=None):
    return _array_binary_out(_raw.matvec, _raw.matvec_into, matrix, vector, out=out)


def momentum_descent(
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
    return _raw.momentum_descent(initial, objective, gradient, **kwargs)


def momentum_descent_complex(
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
    return _raw.momentum_descent_complex(initial, objective, gradient, **kwargs)


def numerical_gradient(
    function,
    x,
    step_size=None,
    tolerance=None,
    max_iterations=None,
    *,
    config: JacobianConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _raw.numerical_gradient(function, x, **kwargs)


def numerical_hessian(
    function,
    x,
    step_size=None,
    tolerance=None,
    max_iterations=None,
    *,
    config: JacobianConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _raw.numerical_hessian(function, x, **kwargs)


def numerical_jacobian(
    function,
    x,
    step_size=None,
    tolerance=None,
    max_iterations=None,
    *,
    config: JacobianConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _raw.numerical_jacobian(function, x, **kwargs)


def numerical_jacobian_central(
    function,
    x,
    step_size=None,
    tolerance=None,
    max_iterations=None,
    *,
    config: JacobianConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _raw.numerical_jacobian_central(function, x, **kwargs)


def pairwise_cosine_similarity(
    left,
    right,
    *,
    out=None,
    workspace: PairwiseCosineWorkspace | None = None,
):
    if workspace is None:
        return _array_binary_out(
            _raw.pairwise_cosine_similarity,
            _raw.pairwise_cosine_similarity_into,
            left,
            right,
            out=out,
        )
    return _require_workspace(workspace, PairwiseCosineWorkspace).similarity(left, right, out=out)


def pairwise_cosine_distance(
    left,
    right,
    *,
    out=None,
    workspace: PairwiseCosineWorkspace | None = None,
):
    if workspace is None:
        return _array_binary_out(
            _raw.pairwise_cosine_distance,
            _raw.pairwise_cosine_distance_into,
            left,
            right,
            out=out,
        )
    return _require_workspace(workspace, PairwiseCosineWorkspace).distance(left, right, out=out)


def pairwise_l2_distance(left, right, *, out=None):
    return _array_binary_out(
        _raw.pairwise_l2_distance,
        _raw.pairwise_l2_distance_into,
        left,
        right,
        out=out,
    )


def projected_gradient_descent_box(
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
    return _raw.projected_gradient_descent_box(
        initial,
        objective,
        gradient,
        lower_bounds,
        upper_bounds,
        **kwargs,
    )


def projected_gradient_descent_box_complex(
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
    return _raw.projected_gradient_descent_box_complex(
        initial,
        objective,
        gradient,
        lower_bounds,
        upper_bounds,
        **kwargs,
    )


def rmsprop(
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
    return _raw.rmsprop(initial, objective, gradient, **kwargs)


def rmsprop_complex(
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
    return _raw.rmsprop_complex(initial, objective, gradient, **kwargs)


def stochastic_gradient_descent(
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
    return _raw.stochastic_gradient_descent(initial, stochastic_gradient, **kwargs)


def stochastic_gradient_descent_complex(
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
    return _raw.stochastic_gradient_descent_complex(initial, stochastic_gradient, **kwargs)


def adam(
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
    return _raw.adam(initial, objective, gradient, **kwargs)


def adam_complex(
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
    return _raw.adam_complex(initial, objective, gradient, **kwargs)


def _tensor_unary_out(raw, raw_into, tensor, *, out=None):
    if out is None:
        return raw(tensor)
    raw_into(tensor, out)
    return out


def _tensor_binary_out(raw, raw_into, left, right, *, out=None):
    if out is None:
        return raw(left, right)
    raw_into(left, right, out)
    return out


def _tensor_permute_out(raw, raw_into, tensor, permutation, *, out=None):
    if out is None:
        return raw(tensor, permutation)
    raw_into(tensor, permutation, out)
    return out


def _tensor_contract_out(raw, raw_into, left, right, left_axes, right_axes, *, out=None):
    if out is None:
        return raw(left, right, left_axes, right_axes)
    raw_into(left, right, left_axes, right_axes, out)
    return out


def tensor_cube_matvec(cube, vectors, *, out=None):
    return _tensor_binary_out(_raw.tensor_cube_matvec, _raw.tensor_cube_matvec_into, cube, vectors, out=out)


def tensor_cube_matvec_complex(cube, vectors, *, out=None):
    return _tensor_binary_out(
        _raw.tensor_cube_matvec_complex,
        _raw.tensor_cube_matvec_complex_into,
        cube,
        vectors,
        out=out,
    )


def tensor_cube_matmat(left, right, *, out=None):
    return _tensor_binary_out(_raw.tensor_cube_matmat, _raw.tensor_cube_matmat_into, left, right, out=out)


def tensor_cube_matmat_complex(left, right, *, out=None):
    return _tensor_binary_out(
        _raw.tensor_cube_matmat_complex,
        _raw.tensor_cube_matmat_complex_into,
        left,
        right,
        out=out,
    )


tensor_einsum = _raw.tensor_einsum
tensor_einsum_complex = _raw.tensor_einsum_complex


def tensor_sum_last_axis(tensor, *, out=None):
    return _tensor_unary_out(_raw.tensor_sum_last_axis, _raw.tensor_sum_last_axis_into, tensor, out=out)


def tensor_sum_last_axis_complex(tensor, *, out=None):
    return _tensor_unary_out(
        _raw.tensor_sum_last_axis_complex,
        _raw.tensor_sum_last_axis_complex_into,
        tensor,
        out=out,
    )


def tensor_l2_norm_last_axis(tensor, *, out=None):
    return _tensor_unary_out(
        _raw.tensor_l2_norm_last_axis,
        _raw.tensor_l2_norm_last_axis_into,
        tensor,
        out=out,
    )


def tensor_l2_norm_last_axis_complex(tensor, *, out=None):
    return _tensor_unary_out(
        _raw.tensor_l2_norm_last_axis_complex,
        _raw.tensor_l2_norm_last_axis_complex_into,
        tensor,
        out=out,
    )


def tensor_normalize_last_axis(tensor, *, out=None):
    return _tensor_unary_out(
        _raw.tensor_normalize_last_axis,
        _raw.tensor_normalize_last_axis_into,
        tensor,
        out=out,
    )


def tensor_normalize_last_axis_complex(tensor, *, out=None):
    return _tensor_unary_out(
        _raw.tensor_normalize_last_axis_complex,
        _raw.tensor_normalize_last_axis_complex_into,
        tensor,
        out=out,
    )


def tensor_batched_dot_last_axis(left, right, *, out=None):
    return _tensor_binary_out(
        _raw.tensor_batched_dot_last_axis,
        _raw.tensor_batched_dot_last_axis_into,
        left,
        right,
        out=out,
    )


def tensor_batched_dot_last_axis_complex(left, right, *, out=None):
    return _tensor_binary_out(
        _raw.tensor_batched_dot_last_axis_complex,
        _raw.tensor_batched_dot_last_axis_complex_into,
        left,
        right,
        out=out,
    )


def tensor_permute_axes(tensor, permutation, *, out=None):
    return _tensor_permute_out(_raw.tensor_permute_axes, _raw.tensor_permute_axes_into, tensor, permutation, out=out)


def tensor_permute_axes_complex(tensor, permutation, *, out=None):
    return _tensor_permute_out(
        _raw.tensor_permute_axes_complex,
        _raw.tensor_permute_axes_complex_into,
        tensor,
        permutation,
        out=out,
    )


def tensor_contract_axes(left, right, left_axes, right_axes, *, out=None):
    return _tensor_contract_out(
        _raw.tensor_contract_axes,
        _raw.tensor_contract_axes_into,
        left,
        right,
        left_axes,
        right_axes,
        out=out,
    )


def tensor_contract_axes_complex(left, right, left_axes, right_axes, *, out=None):
    return _tensor_contract_out(
        _raw.tensor_contract_axes_complex,
        _raw.tensor_contract_axes_complex_into,
        left,
        right,
        left_axes,
        right_axes,
        out=out,
    )


def tensor_batched_matmul_last_two(left, right, *, out=None):
    return _tensor_binary_out(
        _raw.tensor_batched_matmul_last_two,
        _raw.tensor_batched_matmul_last_two_into,
        left,
        right,
        out=out,
    )


def tensor_batched_matmul_last_two_complex(left, right, *, out=None):
    return _tensor_binary_out(
        _raw.tensor_batched_matmul_last_two_complex,
        _raw.tensor_batched_matmul_last_two_complex_into,
        left,
        right,
        out=out,
    )


triangular_solve_lower = _raw.triangular_solve_lower
triangular_solve_lower_matrix = _raw.triangular_solve_lower_matrix
triangular_solve_upper = _raw.triangular_solve_upper
triangular_solve_upper_matrix = _raw.triangular_solve_upper_matrix


__all__ = [
    "SvdResult",
    "BalancedNonsymmetricResult",
    "QrResult",
    "LuResult",
    "LogDetResult",
    "CholeskyResult",
    "EigenResult",
    "GeneralizedEigenResult",
    "NonsymmetricBiEigenResult",
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
    "AdamConfig",
    "BFGSConfig",
    "GradientDescentConfig",
    "IterativeConfig",
    "JacobianConfig",
    "LineSearchConfig",
    "MomentumConfig",
    "ProjectedGradientConfig",
    "RMSPropConfig",
    "MatrixFunctionWorkspace",
    "PairwiseCosineWorkspace",
    "SchurWorkspace",
    "SylvesterWorkspace",
    "build_features",
    "svd_decompose",
    "svd_decompose_truncated",
    "svd_pseudo_inverse",
    "svd_reconstruct_matrix",
    "svd_condition_number",
    "svd_rank",
    "svd_null_space",
    "qr_decompose",
    "qr_decompose_reduced",
    "qr_decompose_pivoted",
    "qr_reconstruct_matrix",
    "qr_condition_number",
    "qr_solve_least_squares",
    "lu_decompose",
    "lu_solve",
    "lu_inverse",
    "lu_determinant",
    "lu_log_determinant",
    "cholesky_decompose",
    "cholesky_solve",
    "cholesky_inverse",
    "eigen_symmetric",
    "eigen_generalized",
    "eigen_nonsymmetric",
    "eigen_balance_nonsymmetric",
    "eigen_nonsymmetric_bi",
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
    "batched_matmat_broadcast_right",
    "batched_matmat_broadcast_left",
    "dot",
    "l2_norm",
    "cosine_similarity",
    "cosine_distance",
    "pairwise_l2_distance",
    "pairwise_cosine_similarity",
    "pairwise_cosine_distance",
    "batched_dot",
    "batched_l2_norm",
    "batched_cosine_similarity",
    "batched_cosine_distance",
    "batched_normalize",
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
    "CooMatrix",
    "CscMatrix",
    "CsrMatrix",
    "IC0Factorization",
    "ILDL0Factorization",
    "ILU0Factorization",
    "ILUKConfig",
    "ILUKFactorization",
    "ILUTConfig",
    "ILUTFactorization",
    "JacobiPreconditioner",
    "SparseLUFactorization",
    "sparse_bicgstab_solve",
    "sparse_bicgstab_ildl0_solve",
    "sparse_bicgstab_ildl0_solve_multiple",
    "sparse_bicgstab_ilu0_solve",
    "sparse_bicgstab_ilu0_solve_multiple",
    "sparse_bicgstab_iluk_solve",
    "sparse_bicgstab_iluk_solve_multiple",
    "sparse_bicgstab_ilut_solve",
    "sparse_bicgstab_ilut_solve_multiple",
    "sparse_conjugate_gradient_solve",
    "sparse_gmres_ildl0_solve",
    "sparse_gmres_ildl0_solve_multiple",
    "sparse_gmres_ilu0_solve",
    "sparse_gmres_ilu0_solve_multiple",
    "sparse_gmres_iluk_solve",
    "sparse_gmres_iluk_solve_multiple",
    "sparse_gmres_ilut_solve",
    "sparse_gmres_ilut_solve_multiple",
    "sparse_ic0_factor",
    "sparse_ildl0_factor",
    "sparse_ilu0_factor",
    "sparse_iluk_factor",
    "sparse_ilut_factor",
    "sparse_coo_to_csr",
    "sparse_csc_to_csr",
    "sparse_csr_to_csc",
    "sparse_jacobi_preconditioner",
    "sparse_lu_factor",
    "sparse_lu_solve",
    "sparse_matvec",
    "sparse_matvec_csc",
    "sparse_matmat_dense",
    "sparse_matmat_sparse",
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
