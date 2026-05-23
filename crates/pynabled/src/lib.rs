//! Python bindings for nabled via PyO3.

#![expect(
    unreachable_pub,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_value
)]
#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

mod error;
mod linalg;
mod ml;
mod physical_ai;
mod sparse;
mod utils;

#[cfg(feature = "arrow")]
mod arrow;

use pyo3::prelude::*;

#[pyfunction]
fn build_features() -> Vec<String> {
    let mut features: Vec<String> = [
        ("accelerator-rayon", cfg!(feature = "accelerator-rayon")),
        ("accelerator-wgpu", cfg!(feature = "accelerator-wgpu")),
        ("arrow", cfg!(feature = "arrow")),
        ("magma-system", cfg!(feature = "magma-system")),
        ("netlib-static", cfg!(feature = "netlib-static")),
        ("netlib-system", cfg!(feature = "netlib-system")),
        ("openblas-static", cfg!(feature = "openblas-static")),
        ("openblas-system", cfg!(feature = "openblas-system")),
        ("signal", cfg!(feature = "signal")),
    ]
    .into_iter()
    .filter(|&(_, enabled)| enabled)
    .map(|(name, _)| name.to_owned())
    .collect();
    features.sort_unstable();
    features
}

#[pymodule]
#[pyo3(name = "_pynabled")]
#[expect(clippy::too_many_lines)]
fn pynabled(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(build_features, m)?)?;
    m.add_class::<linalg::vector::PyPairwiseCosineWorkspace>()?;
    m.add_class::<linalg::matrix_functions::PyMatrixFunctionWorkspace>()?;
    m.add_class::<linalg::sylvester::PySylvesterWorkspace>()?;
    m.add_class::<linalg::schur::PySchurWorkspace>()?;
    m.add_class::<sparse::csr::PyJacobiPreconditioner>()?;
    m.add_class::<sparse::csr::PyIlu0Factorization>()?;
    m.add_class::<sparse::csr::PyIlutFactorization>()?;
    m.add_class::<sparse::csr::PyIlukFactorization>()?;
    m.add_class::<sparse::csr::PyIc0Factorization>()?;
    m.add_class::<sparse::csr::PyIldl0Factorization>()?;
    m.add_class::<sparse::csr::PySparseLuFactorization>()?;

    // SVD
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::decompose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::decompose_truncated, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::pseudo_inverse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::pseudo_inverse_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::pseudo_inverse_from_factors, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::pseudo_inverse_from_factors_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::reconstruct_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::reconstruct_matrix_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::condition_number, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::rank, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::null_space, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::null_space_from_factors, m)?)?;

    // QR
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::decompose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::decompose_reduced, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::decompose_pivoted, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::reconstruct_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::reconstruct_matrix_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::reconstruct_matrix_pivoted, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::reconstruct_matrix_pivoted_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::condition_number, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::solve_least_squares, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::solve_least_squares_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::solve_least_squares_from_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::solve_least_squares_from_factor_into, m)?)?;

    // LU
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::decompose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::solve_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::solve_from_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::solve_from_factor_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::solve_mixed, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::inverse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::inverse_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::inverse_from_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::inverse_from_factor_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::determinant, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::determinant_from_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::log_determinant, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::log_determinant_from_factor, m)?)?;

    // Cholesky
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::decompose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::solve_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::solve_from_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::solve_from_factor_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::inverse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::inverse_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::inverse_from_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::inverse_from_factor_into, m)?)?;

    // Eigen
    m.add_function(pyo3::wrap_pyfunction!(linalg::eigen::symmetric, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::eigen::generalized, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::eigen::nonsymmetric, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::eigen::balance_nonsymmetric, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::eigen::nonsymmetric_bi, m)?)?;

    // Schur
    m.add_function(pyo3::wrap_pyfunction!(linalg::schur::compute_schur, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::schur::compute_schur_into, m)?)?;

    // Polar
    m.add_function(pyo3::wrap_pyfunction!(linalg::polar::compute_polar, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::polar::compute_polar_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::polar::compute_polar_from_factors, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::polar::compute_polar_from_factors_into, m)?)?;

    // Sylvester
    m.add_function(pyo3::wrap_pyfunction!(linalg::sylvester::solve_sylvester, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::sylvester::solve_sylvester_mixed, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::sylvester::solve_sylvester_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::sylvester::solve_lyapunov, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::sylvester::solve_lyapunov_mixed, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::sylvester::solve_lyapunov_into, m)?)?;

    // Triangular
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_lower, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_lower_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_upper, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_upper_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_lower_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_lower_matrix_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_upper_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_upper_matrix_into, m)?)?;

    // Matrix functions
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_exp, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_exp_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_exp_eigen, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_exp_eigen_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_exp_eigen_from_factors,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_exp_eigen_from_factors_into,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_taylor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_taylor_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_eigen, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_eigen_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_log_eigen_from_factors,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_log_eigen_from_factors_into,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_svd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_svd_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_log_svd_from_factors,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_log_svd_from_factors_into,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_power, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_power_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_power_from_factors,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_power_from_factors_into,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_sign, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_sign_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_sign_from_factors, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix_functions::matrix_sign_from_factors_into,
        m
    )?)?;

    // Orthogonalization
    m.add_function(pyo3::wrap_pyfunction!(linalg::orthogonalization::gram_schmidt, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::orthogonalization::gram_schmidt_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::orthogonalization::gram_schmidt_classic, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::orthogonalization::gram_schmidt_classic_into,
        m
    )?)?;

    // Tensor
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matvec, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matvec_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matvec_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matvec_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matmat, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matmat_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matmat_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matmat_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::sum_last_axis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::sum_last_axis_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::sum_last_axis_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::sum_last_axis_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::l2_norm_last_axis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::l2_norm_last_axis_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::l2_norm_last_axis_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::l2_norm_last_axis_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::normalize_last_axis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::normalize_last_axis_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::normalize_last_axis_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::normalize_last_axis_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_dot_last_axis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_dot_last_axis_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_dot_last_axis_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_dot_last_axis_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::permute_axes, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::permute_axes_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::permute_axes_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::permute_axes_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::contract_axes, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::contract_axes_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::contract_axes_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::contract_axes_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_matmul_last_two, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_matmul_last_two_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_matmul_last_two_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::tensor::batched_matmul_last_two_complex_into,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::einsum, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::einsum_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::einsum_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::einsum_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd3, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd3_reconstruct, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd3_reconstruct_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd_nd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hooi_nd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd_nd_reconstruct, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd_nd_reconstruct_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tucker_project, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tucker_project_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tucker_expand, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tucker_expand_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3_with_report, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3_diagnostics, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3_reconstruct, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3_reconstruct_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd_with_report, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd_diagnostics, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd_reconstruct, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd_reconstruct_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_svd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_orthogonalize_left, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_orthogonalize_right, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_round, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_inner, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_norm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_add, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_hadamard, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_hadamard_round, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_svd_reconstruct, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tt_svd_reconstruct_into, m)?)?;

    // Batched
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::qr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::svd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::lu, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::cholesky, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::symmetric_eigen, m)?)?;

    // Matrix
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::matvec, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::matvec_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::matmat, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::matmat_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_row_matvec, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_row_matvec_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_matmat, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_matmat_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_matmat_broadcast_right, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        linalg::matrix::batched_matmat_broadcast_right_into,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_matmat_broadcast_left, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_matmat_broadcast_left_into, m)?)?;

    // Vector
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::dot, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::l2_norm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::cosine_similarity, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::cosine_distance, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::pairwise_l2_distance, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::pairwise_l2_distance_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::pairwise_cosine_similarity, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::pairwise_cosine_similarity_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::pairwise_cosine_distance, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::pairwise_cosine_distance_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_dot, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_dot_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_l2_norm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_l2_norm_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_cosine_similarity, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_cosine_similarity_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_cosine_distance, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_cosine_distance_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_normalize, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::batched_normalize_into, m)?)?;

    // ML
    m.add_function(pyo3::wrap_pyfunction!(ml::regression::linear_regression, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::regression::linear_regression_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::regression::linear_regression_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::regression::linear_regression_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::compute_pca, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::compute_pca_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::compute_pca_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::compute_pca_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_transform, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_transform_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_transform_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_transform_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_inverse_transform, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_inverse_transform_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_inverse_transform_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_inverse_transform_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::column_means, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::column_means_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::center_columns, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::center_columns_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::covariance_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::covariance_matrix_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::correlation_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::correlation_matrix_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::column_means_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::column_means_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::center_columns_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::center_columns_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::covariance_matrix_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::covariance_matrix_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::correlation_matrix_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::correlation_matrix_complex_into, m)?)?;

    // Rolling stats
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::rolling_covariance, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::rolling_covariance_into, m)?)?;

    // Iterative (dense CG, GMRES)
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::conjugate_gradient, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::conjugate_gradient_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::gmres, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::gmres_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::conjugate_gradient_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::conjugate_gradient_complex_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::gmres_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::gmres_complex_into, m)?)?;

    // Jacobian / derivatives
    m.add_function(pyo3::wrap_pyfunction!(ml::jacobian::numerical_jacobian, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::jacobian::numerical_jacobian_central, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::jacobian::numerical_gradient, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::jacobian::numerical_hessian, m)?)?;

    // Optimization
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::backtracking_line_search, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::backtracking_line_search_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::gradient_descent, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::gradient_descent_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::adam, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::adam_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::momentum_descent, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::momentum_descent_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::rmsprop, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::rmsprop_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::projected_gradient_descent_box, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        ml::optimization::projected_gradient_descent_box_complex,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::stochastic_gradient_descent, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        ml::optimization::stochastic_gradient_descent_complex,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::bfgs, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::optimization::bfgs_complex, m)?)?;

    // Sparse
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::matvec, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::matvec_csc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::matmat_dense, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::matmat_sparse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::transpose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::csr_to_csc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::csc_to_csr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::coo_to_csr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::jacobi_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::pcg_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::gauss_seidel_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::conjugate_gradient_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::pcg_ic0_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::bicgstab_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::jacobi_preconditioner, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::ilu0_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::ilut_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::iluk_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::ic0_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::ildl0_factor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::sparse_lu_factor, m)?)?;

    // Physical AI — geometry
    m.add_class::<physical_ai::geometry::PyTransform3>()?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::quat_from_axis_angle, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::quat_to_rotation_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::se3_compose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::se3_log, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::se3_log_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::se3_compose_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::se3_exp, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::transform3_from_parts, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::transform3_to_parts, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::geometry::so3_compose, m)?)?;

    // Physical AI — kinematics
    m.add_class::<physical_ai::kinematics::PyChainSpec>()?;
    m.add_class::<physical_ai::kinematics::PyIkConfig>()?;
    m.add_class::<physical_ai::kinematics::PyIkWorkspace>()?;
    m.add_class::<physical_ai::kinematics::PyIkResult>()?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::end_effector_pose_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::fk, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::jacobian_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::jacobian_translation_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::pose_error_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::pose_error_into_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::end_effector_pose_tree_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::jacobian_tree_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::link_transforms_tree_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::kinematics::inverse_kinematics_dls_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        physical_ai::kinematics::inverse_kinematics_dls_into_py,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        physical_ai::kinematics::inverse_kinematics_tree_dls_py,
        m
    )?)?;

    // Physical AI — model
    m.add_class::<physical_ai::model::PyRobotModel>()?;
    m.add_class::<physical_ai::model::PyPlanar2rFixture>()?;
    m.add_class::<physical_ai::model::PySixDofDhFixture>()?;
    m.add_class::<physical_ai::model::PyYBranchFixture>()?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::model::from_urdf_file_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::model::from_urdf_str_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::model::to_chain_spec_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::model::extract_chain_spec_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::model::load_planar2r_fixture, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::model::load_six_dof_dh_fixture, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::model::load_y_branch_fixture, m)?)?;

    // Physical AI — dynamics
    m.add_class::<physical_ai::dynamics::PyDynamicsConfig>()?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::dynamics::rnea, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::dynamics::mass_matrix_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::dynamics::forward_dynamics, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::dynamics::rnea_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::dynamics::forward_dynamics_into, m)?)?;

    // Physical AI — control
    m.add_class::<physical_ai::control::PyLqrResult>()?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::control::discrete_lqr_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::control::dare_solve_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::control::dare_residual_norm_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::control::controllability_gramian_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::control::place_poles_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::control::luenberger_gain_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::control::discrete_lqr_into, m)?)?;

    // Physical AI — sensor
    m.add_class::<physical_ai::sensor::PyKalmanState>()?;
    m.add_class::<physical_ai::sensor::PyPinholeIntrinsics>()?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::kalman_predict, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::kalman_update, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::kalman_predict_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::kalman_update_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::ekf_predict, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::ekf_update, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::pinhole_project_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::strapdown_predict_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(physical_ai::sensor::strapdown_predict_into, m)?)?;

    #[cfg(feature = "signal")]
    {
        m.add_function(pyo3::wrap_pyfunction!(physical_ai::signal::rfft_py, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(physical_ai::signal::irfft_py, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(physical_ai::signal::dominant_frequency_py, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(physical_ai::signal::bin_to_hz_py, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(physical_ai::signal::autocorrelation_full_py, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(physical_ai::signal::rfft_into_py, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(physical_ai::signal::irfft_into_py, m)?)?;
    }

    #[cfg(feature = "arrow")]
    {
        m.add_function(pyo3::wrap_pyfunction!(arrow::dot, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::dot_hermitian, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cosine_similarity, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cosine_similarity_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::l2_norm, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::l2_norm_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cosine_distance, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::pairwise_l2_distance, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::pairwise_cosine_similarity, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::pairwise_cosine_distance, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_dot, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_dot_hermitian, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_l2_norm, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_l2_norm_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_cosine_similarity, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_cosine_similarity_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_cosine_distance, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_normalize, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_normalize_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matvec, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matvec_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matmat, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matmat_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_row_matvec, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_matmat, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_matmat_broadcast_right, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_matmat_broadcast_left, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_qr, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_svd, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_lu, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_cholesky, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::batched_symmetric_eigen, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_sum_last_axis_fixed, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_sum_last_axis_variable, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_l2_norm_last_axis_fixed, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_l2_norm_last_axis_variable, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_normalize_last_axis_fixed, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_normalize_last_axis_variable, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_batched_dot_last_axis_fixed, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_batched_dot_last_axis_variable, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_sum_last_axis_fixed_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_sum_last_axis_variable_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_sum_last_axis_variable_complex_storage,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_l2_norm_last_axis_fixed_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_l2_norm_last_axis_variable_complex,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_l2_norm_last_axis_variable_complex_storage,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_normalize_last_axis_fixed_complex,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_normalize_last_axis_variable_complex,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_normalize_last_axis_variable_complex_storage,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_batched_dot_last_axis_fixed_complex,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_batched_dot_last_axis_variable_complex,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::tensor_batched_dot_last_axis_variable_complex_storage,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_permute_axes, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_permute_axes_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_contract_axes, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_contract_axes_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_batched_matmul_last_two, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_batched_matmul_last_two_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cube_matvec, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cube_matvec_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cube_matmat, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cube_matmat_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_flatten_cubes, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_einsum, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_einsum_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cp_als3, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cp_als3_with_report, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cp_als3_diagnostics, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cp_als3_reconstruct, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cp_als_nd, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cp_als_nd_with_report, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cp_als_nd_diagnostics, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_cp_als_nd_reconstruct, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_hosvd_nd, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_hooi_nd, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_hosvd_nd_reconstruct, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tucker_project, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tucker_expand, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_svd, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_orthogonalize_left, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_orthogonalize_right, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_round, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_inner, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_norm, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_add, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_hadamard, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_hadamard_round, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::tensor_tt_svd_reconstruct, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::column_means, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::column_means_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::center_columns, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::center_columns_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::covariance_matrix, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::covariance_matrix_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::correlation_matrix, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::correlation_matrix_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::gram_schmidt, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::gram_schmidt_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::gram_schmidt_classic, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::solve_lower, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::solve_lower_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::solve_upper, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::solve_upper_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::solve_lower_matrix, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::solve_upper_matrix, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::svd_decompose, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::svd_decompose_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::svd_decompose_truncated, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::svd_decompose_with_tolerance, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::svd_pseudo_inverse, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::svd_null_space, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::qr_decompose, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::qr_decompose_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::qr_decompose_reduced, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::qr_decompose_pivoted, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::qr_solve_least_squares, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::lu_decompose, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::lu_solve, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::lu_solve_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::lu_inverse, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::lu_inverse_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::lu_determinant, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::lu_determinant_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::lu_log_determinant, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cholesky_decompose, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cholesky_decompose_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cholesky_solve, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cholesky_solve_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cholesky_inverse, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::cholesky_inverse_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::eigen_symmetric, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::eigen_generalized, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::eigen_nonsymmetric, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::eigen_nonsymmetric_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::eigen_nonsymmetric_bi, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::schur_compute, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::schur_compute_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::polar_compute, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::polar_compute_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_exp, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_exp_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_exp_eigen, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_exp_eigen_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_log_taylor, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_log_eigen, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_log_eigen_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_log_svd, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_log_svd_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_power, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_power_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_sign, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::matrix_sign_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::compute_pca, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::compute_pca_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::pca_transform, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::pca_transform_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::pca_inverse_transform, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::pca_inverse_transform_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::linear_regression, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::linear_regression_complex, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_matvec_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_matmat_dense_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_lu_solve_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_jacobi_solve_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_gauss_seidel_solve_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_conjugate_gradient_solve_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_pcg_solve_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_batched_matvec_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_transpose_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_csr_to_csc_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_matmat_sparse_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_jacobi_preconditioner_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::sparse_apply_jacobi_preconditioner_arrow,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_ilu0_factor_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_apply_ilu0_preconditioner_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_ilut_factor_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_apply_ilut_preconditioner_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_iluk_factor_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_apply_iluk_preconditioner_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_ic0_factor_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_apply_ic0_preconditioner_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_ildl0_factor_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_apply_ildl0_preconditioner_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_lu_factor_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::sparse_lu_solve_with_factorization_arrow,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::sparse_lu_solve_multiple_with_factorization_arrow,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_batch_matvec_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_batch_matmat_dense_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_batch_transpose_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::sparse_batch_matmat_sparse_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::conjugate_gradient, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::conjugate_gradient_complex_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::gmres_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::gmres_complex_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::numerical_jacobian_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::numerical_jacobian_central_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::numerical_gradient_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::numerical_hessian_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::backtracking_line_search_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::backtracking_line_search_complex_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::gradient_descent_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::gradient_descent_complex_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::adam_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::adam_complex_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::momentum_descent_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::momentum_descent_complex_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::rmsprop_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::rmsprop_complex_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::projected_gradient_descent_box_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::projected_gradient_descent_box_complex_arrow,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::stochastic_gradient_descent_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            arrow::stochastic_gradient_descent_complex_arrow,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::bfgs_arrow, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::bfgs_complex_arrow, m)?)?;
    }

    Ok(())
}
