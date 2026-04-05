//! Python bindings for nabled via PyO3.

#![expect(
    unreachable_pub,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_value
)]

mod error;
mod linalg;
mod ml;
mod sparse;
mod utils;

#[cfg(feature = "arrow")]
mod arrow;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_pynabled")]
#[expect(clippy::too_many_lines)]
fn pynabled(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // SVD
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::decompose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::decompose_truncated, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::pseudo_inverse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::reconstruct_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::condition_number, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::rank, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::svd::null_space, m)?)?;

    // QR
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::decompose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::qr::solve_least_squares, m)?)?;

    // LU
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::decompose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::inverse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::lu::determinant, m)?)?;

    // Cholesky
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::decompose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::cholesky::inverse, m)?)?;

    // Eigen
    m.add_function(pyo3::wrap_pyfunction!(linalg::eigen::symmetric, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::eigen::generalized, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::eigen::nonsymmetric, m)?)?;

    // Schur
    m.add_function(pyo3::wrap_pyfunction!(linalg::schur::compute_schur, m)?)?;

    // Polar
    m.add_function(pyo3::wrap_pyfunction!(linalg::polar::compute_polar, m)?)?;

    // Sylvester
    m.add_function(pyo3::wrap_pyfunction!(linalg::sylvester::solve_sylvester, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::sylvester::solve_lyapunov, m)?)?;

    // Triangular
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_lower, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_upper, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_lower_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::triangular::solve_upper_matrix, m)?)?;

    // Matrix functions
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_exp, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_exp_eigen, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_taylor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_eigen, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_log_svd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_power, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix_functions::matrix_sign, m)?)?;

    // Orthogonalization
    m.add_function(pyo3::wrap_pyfunction!(linalg::orthogonalization::gram_schmidt, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::orthogonalization::gram_schmidt_classic, m)?)?;

    // Tensor
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matvec, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matvec_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matmat, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cube_matmat_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::sum_last_axis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::sum_last_axis_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::l2_norm_last_axis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::l2_norm_last_axis_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::normalize_last_axis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::normalize_last_axis_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_dot_last_axis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_dot_last_axis_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::permute_axes, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::permute_axes_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::contract_axes, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::contract_axes_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_matmul_last_two, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::batched_matmul_last_two_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::einsum, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::einsum_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd3, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd3_reconstruct, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd_nd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hooi_nd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::hosvd_nd_reconstruct, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tucker_project, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::tucker_expand, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3_with_report, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3_diagnostics, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als3_reconstruct, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd_with_report, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd_diagnostics, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::tensor::cp_als_nd_reconstruct, m)?)?;
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

    // Batched
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::qr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::svd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::lu, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::cholesky, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::batched::symmetric_eigen, m)?)?;

    // Matrix
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::matvec, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::matmat, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_row_matvec, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::matrix::batched_matmat, m)?)?;

    // Vector
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::dot, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::l2_norm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::cosine_similarity, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::pairwise_l2_distance, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::vector::pairwise_cosine_similarity, m)?)?;

    // ML
    m.add_function(pyo3::wrap_pyfunction!(ml::regression::linear_regression, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::regression::linear_regression_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::compute_pca, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::compute_pca_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_transform, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_transform_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_inverse_transform, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::pca::pca_inverse_transform_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::column_means, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::center_columns, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::covariance_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::correlation_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::column_means_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::center_columns_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::covariance_matrix_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::stats::correlation_matrix_complex, m)?)?;

    // Iterative (dense CG, GMRES)
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::conjugate_gradient, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::gmres, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::conjugate_gradient_complex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ml::iterative::gmres_complex, m)?)?;

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
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::matmat_dense, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::transpose, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::jacobi_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sparse::csr::pcg_solve, m)?)?;

    #[cfg(feature = "arrow")]
    {
        m.add_function(pyo3::wrap_pyfunction!(arrow::dot, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::l2_norm, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(arrow::svd_decompose, m)?)?;
    }

    Ok(())
}
