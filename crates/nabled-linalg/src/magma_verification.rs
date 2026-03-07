#![cfg(feature = "magma-system")]

use ndarray::{Array1, Array2, Axis, stack};
use num_complex::Complex64;

use crate::internal::DenseKernelPolicy;
use crate::provider::{magma, magma_runtime, magma_sparse};
use crate::qr::{self, QRConfig};
use crate::sparse::{self, CsrMatrixView};
use crate::{batched, cholesky, eigen, lu, matrix_functions, polar, schur, svd, sylvester};

struct VerifyForceGuard;

impl VerifyForceGuard {
    fn new() -> Self {
        DenseKernelPolicy::set_magma_verify_force_override(Some(true));
        Self
    }
}

impl Drop for VerifyForceGuard {
    fn drop(&mut self) { DenseKernelPolicy::set_magma_verify_force_override(None); }
}

fn assert_dense_provider_used(context: &str) {
    let provider_calls = magma::magma_provider_call_count();
    let runtime_calls = magma_runtime::magma_runtime_call_count();
    assert!(
        provider_calls > 0 || runtime_calls > 0,
        "{context}: expected MAGMA dense provider/runtime call to be observed"
    );
}

fn assert_sparse_provider_used(context: &str) {
    let calls = magma_sparse::magma_sparse_provider_call_count();
    assert!(calls > 0, "{context}: expected MAGMA sparse provider to be called");
}

fn run_dense_case<R>(context: &str, operation: impl FnOnce() -> R) {
    magma::reset_magma_provider_call_count();
    magma_runtime::reset_magma_runtime_call_count();
    if std::env::var_os("NABLED_MAGMA_VERIFY_TRACE").is_some() {
        eprintln!("[magma-verify] dense begin: {context}");
    }
    let _result = operation();
    if std::env::var_os("NABLED_MAGMA_VERIFY_TRACE").is_some() {
        eprintln!("[magma-verify] dense end: {context}");
    }
    assert_dense_provider_used(context);
}

fn run_sparse_case<R>(context: &str, operation: impl FnOnce() -> Result<R, sparse::SparseError>) {
    magma_sparse::reset_magma_sparse_provider_call_count();
    let result = operation();
    assert_sparse_provider_used(context);
    if let Err(error) = result {
        assert!(
            !DenseKernelPolicy::magma_fail_fast_mode(),
            "{context}: unexpected sparse provider failure in fail-fast mode: {error:?}"
        );
    }
}

fn make_well_conditioned(n: usize) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros((n, n));
    let mut diagonal = 4.0_f64;
    for i in 0..n {
        matrix[[i, i]] = diagonal;
        diagonal += 0.01;
        if i + 1 < n {
            matrix[[i, i + 1]] = 0.2;
            matrix[[i + 1, i]] = 0.1;
        }
    }
    matrix
}

fn make_spd(n: usize) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros((n, n));
    let mut diagonal = 3.0_f64;
    for i in 0..n {
        matrix[[i, i]] = diagonal;
        diagonal += 0.01;
        if i + 1 < n {
            matrix[[i, i + 1]] = 0.15;
            matrix[[i + 1, i]] = 0.15;
        }
    }
    matrix
}

fn make_complex(matrix: &Array2<f64>) -> Array2<Complex64> {
    let mut output = Array2::<Complex64>::zeros(matrix.dim());
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            output[[row, col]] = Complex64::new(matrix[[row, col]], 0.0);
        }
    }
    output
}

fn make_rectangular(rows: usize, cols: usize) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros((rows, cols));

    // Seed an explicit diagonal in the first `cols` rows to guarantee full
    // column rank for tall matrices used by least-squares verification paths.
    let mut diagonal = 2.0_f64;
    for diag in 0..cols.min(rows) {
        matrix[[diag, diag]] = diagonal;
        diagonal += 0.01;
    }

    let mut row_term = 0.001_f64;
    for row in 0..rows {
        let mut term = row_term;
        for col in 0..cols {
            matrix[[row, col]] += term;
            term += 0.007;
        }
        row_term += 0.001;
    }

    matrix
}

fn make_upper_triangular(n: usize) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros((n, n));
    let mut diagonal = 2.0_f64;
    for row in 0..n {
        matrix[[row, row]] = diagonal;
        diagonal += 0.01;
        let mut denominator = 2.0_f64;
        for col in (row + 1)..n {
            matrix[[row, col]] = 0.05 / denominator;
            denominator += 1.0;
        }
    }
    matrix
}

fn make_positive_diagonal(n: usize) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros((n, n));
    let mut diagonal = 1.0_f64;
    for i in 0..n {
        matrix[[i, i]] = diagonal;
        diagonal += 0.05;
    }
    matrix
}

#[test]
#[ignore = "requires MAGMA-capable CUDA host"]
#[allow(clippy::too_many_lines)]
#[allow(clippy::similar_names)]
fn magma_dense_provider_execution_matrix() {
    let _guard = VerifyForceGuard::new();

    let n = 32;
    let square = make_well_conditioned(n);
    let spd = make_spd(n);
    let rhs = Array1::from_elem(n, 1.0_f64);
    let square_c = make_complex(&square);
    let spd_c = make_complex(&spd);
    let schur_matrix = make_upper_triangular(n);
    let schur_matrix_c = make_complex(&schur_matrix);
    let hermitian_diagonal = make_positive_diagonal(n);
    let hermitian_diagonal_c = make_complex(&hermitian_diagonal);
    let rhs_c = Array1::from_elem(n, Complex64::new(1.0, 0.0));
    let rect = make_rectangular(64, 32);
    let rect_c = make_complex(&rect);
    let rhs_ls = Array1::from_elem(rect.nrows(), 0.5_f64);
    let qr_config = QRConfig::<f64>::default();
    let spd_b = make_spd(n);
    let sylvester_a = make_well_conditioned(2);
    let sylvester_b = make_spd(2);
    let sylvester_c = Array2::from_elem((2, 2), 0.2_f64);
    let sylvester_a_complex = make_complex(&sylvester_a);
    let sylvester_b_complex = make_complex(&sylvester_b);
    let sylvester_c_complex = make_complex(&sylvester_c);

    run_dense_case("lu::solve", || lu::solve(&square, &rhs).unwrap());
    run_dense_case("lu::inverse", || lu::inverse(&square).unwrap());
    run_dense_case("lu::determinant", || lu::determinant(&square).unwrap());
    run_dense_case("lu::solve_complex", || lu::solve_complex(&square_c, &rhs_c).unwrap());
    run_dense_case("lu::inverse_complex", || lu::inverse_complex(&square_c).unwrap());
    run_dense_case("lu::determinant_complex", || lu::determinant_complex(&square_c).unwrap());
    run_dense_case("lu::solve_mixed_f64", || lu::solve_mixed_f64(&square, &rhs).unwrap());
    run_dense_case("lu::solve_mixed_complex", || {
        lu::solve_mixed_complex(&square_c, &rhs_c).unwrap()
    });

    run_dense_case("cholesky::decompose", || cholesky::decompose(&spd).unwrap());
    run_dense_case("cholesky::solve", || cholesky::solve(&spd, &rhs).unwrap());
    run_dense_case("cholesky::inverse", || cholesky::inverse(&spd).unwrap());
    run_dense_case("cholesky::decompose_complex", || cholesky::decompose_complex(&spd_c).unwrap());
    run_dense_case("cholesky::solve_complex", || cholesky::solve_complex(&spd_c, &rhs_c).unwrap());
    run_dense_case("cholesky::inverse_complex", || cholesky::inverse_complex(&spd_c).unwrap());

    run_dense_case("qr::decompose", || qr::decompose(&rect, &qr_config).unwrap());
    run_dense_case("qr::decompose_complex", || qr::decompose_complex(&rect_c, &qr_config).unwrap());
    run_dense_case("qr::solve_least_squares", || {
        qr::solve_least_squares(&rect, &rhs_ls, &qr_config).unwrap()
    });
    run_dense_case("svd::decompose", || svd::decompose(&rect).unwrap());
    run_dense_case("svd::decompose_complex", || svd::decompose_complex(&rect_c).unwrap());
    run_dense_case("eigen::symmetric", || eigen::symmetric(&spd).unwrap());
    run_dense_case("eigen::generalized", || eigen::generalized(&spd, &spd_b).unwrap());
    run_dense_case("eigen::nonsymmetric_complex", || {
        eigen::nonsymmetric_complex(&square_c).unwrap()
    });
    run_dense_case("schur::compute_schur", || schur::compute_schur(&schur_matrix).unwrap());
    run_dense_case("schur::compute_schur_complex", || {
        schur::compute_schur_complex(&schur_matrix_c).unwrap()
    });
    run_dense_case("polar::compute_polar", || polar::compute_polar(&spd).unwrap());
    run_dense_case("polar::compute_polar_complex", || {
        polar::compute_polar_complex(&spd_c).unwrap()
    });
    run_dense_case("matrix_functions::matrix_exp_eigen", || {
        matrix_functions::matrix_exp_eigen(&spd).unwrap()
    });
    run_dense_case("matrix_functions::matrix_exp_eigen_complex", || {
        matrix_functions::matrix_exp_eigen_complex(&hermitian_diagonal_c).unwrap()
    });
    run_dense_case("matrix_functions::matrix_log_eigen", || {
        matrix_functions::matrix_log_eigen(&spd).unwrap()
    });
    run_dense_case("matrix_functions::matrix_log_eigen_complex", || {
        matrix_functions::matrix_log_eigen_complex(&hermitian_diagonal_c).unwrap()
    });
    run_dense_case("matrix_functions::matrix_log_svd", || {
        matrix_functions::matrix_log_svd(&spd).unwrap()
    });
    run_dense_case("matrix_functions::matrix_log_svd_complex", || {
        matrix_functions::matrix_log_svd_complex(&hermitian_diagonal_c).unwrap()
    });
    run_dense_case("matrix_functions::matrix_power", || {
        matrix_functions::matrix_power(&spd, 0.5_f64).unwrap()
    });
    run_dense_case("matrix_functions::matrix_power_complex", || {
        matrix_functions::matrix_power_complex(&hermitian_diagonal_c, 0.5_f64).unwrap()
    });
    run_dense_case("matrix_functions::matrix_sign", || {
        matrix_functions::matrix_sign(&spd).unwrap()
    });
    run_dense_case("matrix_functions::matrix_sign_complex", || {
        matrix_functions::matrix_sign_complex(&hermitian_diagonal_c).unwrap()
    });

    let batch_square = make_well_conditioned(2);
    let batch_spd = make_spd(2);
    let batch_rect = make_rectangular(4, 2);
    let batched_square = stack(Axis(0), &[batch_square.view(), batch_square.view()]).unwrap();
    let batched_spd = stack(Axis(0), &[batch_spd.view(), batch_spd.view()]).unwrap();
    let batched_rect = stack(Axis(0), &[batch_rect.view(), batch_rect.view()]).unwrap();

    run_dense_case("batched::lu", || batched::lu(&batched_square).unwrap());
    run_dense_case("batched::cholesky", || batched::cholesky(&batched_spd).unwrap());
    run_dense_case("batched::qr", || batched::qr(&batched_rect, &qr_config).unwrap());
    run_dense_case("batched::svd", || batched::svd(&batched_rect).unwrap());
    run_dense_case("batched::symmetric_eigen", || batched::symmetric_eigen(&batched_spd).unwrap());

    run_dense_case("sylvester::solve_sylvester_mixed_f64", || {
        sylvester::solve_sylvester_mixed_f64(&sylvester_a, &sylvester_b, &sylvester_c).unwrap()
    });
    run_dense_case("sylvester::solve_lyapunov_mixed_f64", || {
        sylvester::solve_lyapunov_mixed_f64(&sylvester_a, &sylvester_c).unwrap()
    });
    run_dense_case("sylvester::solve_sylvester_mixed_complex", || {
        sylvester::solve_sylvester_mixed_complex(
            &sylvester_a_complex,
            &sylvester_b_complex,
            &sylvester_c_complex,
        )
        .unwrap()
    });
    run_dense_case("sylvester::solve_lyapunov_mixed_complex", || {
        sylvester::solve_lyapunov_mixed_complex(&sylvester_a_complex, &sylvester_c_complex).unwrap()
    });
}

#[test]
#[ignore = "requires MAGMA-capable CUDA host"]
fn magma_sparse_provider_execution_matrix() {
    let _guard = VerifyForceGuard::new();

    let spd_row_ptrs = vec![0_i32, 2, 5, 7];
    let spd_col_indices = vec![0_i32, 1, 0, 1, 2, 1, 2];
    let spd_values_f64 = vec![4.0_f64, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
    let spd_values_f32 = vec![4.0_f32, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
    let nonsym_row_ptrs = vec![0_i32, 2, 5, 7];
    let nonsym_col_indices = vec![0_i32, 1, 0, 1, 2, 1, 2];
    let nonsym_values_f64 = vec![4.0_f64, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0];
    let nonsym_values_f32 = vec![4.0_f32, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0];

    let spd_f64 =
        CsrMatrixView::<i32, f64, i32>::new(3, 3, &spd_row_ptrs, &spd_col_indices, &spd_values_f64)
            .unwrap();
    let spd_f32 =
        CsrMatrixView::<i32, f32, i32>::new(3, 3, &spd_row_ptrs, &spd_col_indices, &spd_values_f32)
            .unwrap();
    let nonsym_f64 = CsrMatrixView::<i32, f64, i32>::new(
        3,
        3,
        &nonsym_row_ptrs,
        &nonsym_col_indices,
        &nonsym_values_f64,
    )
    .unwrap();
    let nonsym_f32 = CsrMatrixView::<i32, f32, i32>::new(
        3,
        3,
        &nonsym_row_ptrs,
        &nonsym_col_indices,
        &nonsym_values_f32,
    )
    .unwrap();

    let x_f64 = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let x_f32 = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]);
    let dense_f64 = Array2::from_shape_vec((3, 8), vec![0.25_f64; 24]).unwrap();
    let dense_f32 = Array2::from_shape_vec((3, 8), vec![0.25_f32; 24]).unwrap();
    let rhs_spd_f64 = sparse::matvec_view(&spd_f64, &x_f64).unwrap();
    let rhs_spd_f32 = sparse::matvec_view(&spd_f32, &x_f32).unwrap();
    let rhs_nonsym_f64 = sparse::matvec_view(&nonsym_f64, &x_f64).unwrap();
    let rhs_nonsym_f32 = sparse::matvec_view(&nonsym_f32, &x_f32).unwrap();

    run_sparse_case("sparse::matvec_magma_f64_view", || {
        sparse::matvec_magma_f64_view(&spd_f64, &x_f64)
    });
    run_sparse_case("sparse::matvec_magma_f32_view", || {
        sparse::matvec_magma_f32_view(&spd_f32, &x_f32)
    });
    run_sparse_case("sparse::matmat_dense_magma_f64_view", || {
        sparse::matmat_dense_magma_f64_view(&spd_f64, &dense_f64)
    });
    run_sparse_case("sparse::matmat_dense_magma_f32_view", || {
        sparse::matmat_dense_magma_f32_view(&spd_f32, &dense_f32)
    });
    run_sparse_case("sparse::conjugate_gradient_magma_f64_view", || {
        sparse::conjugate_gradient_magma_f64_view(&spd_f64, &rhs_spd_f64, 1e-10_f64, 256)
    });
    run_sparse_case("sparse::conjugate_gradient_magma_f32_view", || {
        sparse::conjugate_gradient_magma_f32_view(&spd_f32, &rhs_spd_f32, 1e-6_f32, 256)
    });
    run_sparse_case("sparse::pcg_jacobi_magma_f64_view", || {
        sparse::pcg_jacobi_magma_f64_view(&spd_f64, &rhs_spd_f64, 1e-10_f64, 256)
    });
    run_sparse_case("sparse::pcg_jacobi_magma_f32_view", || {
        sparse::pcg_jacobi_magma_f32_view(&spd_f32, &rhs_spd_f32, 1e-6_f32, 256)
    });
    run_sparse_case("sparse::gmres_magma_f64_view", || {
        sparse::gmres_magma_f64_view(&nonsym_f64, &rhs_nonsym_f64, 1e-10_f64, 64)
    });
    run_sparse_case("sparse::gmres_magma_f32_view", || {
        sparse::gmres_magma_f32_view(&nonsym_f32, &rhs_nonsym_f32, 1e-3_f32, 128)
    });
    run_sparse_case("sparse::gmres_ilu0_magma_f64_view", || {
        sparse::gmres_ilu0_magma_f64_view(&nonsym_f64, &rhs_nonsym_f64, 1e-10_f64, 16)
    });
    run_sparse_case("sparse::gmres_ilu0_magma_f32_view", || {
        sparse::gmres_ilu0_magma_f32_view(&nonsym_f32, &rhs_nonsym_f32, 1e-3_f32, 128)
    });
    run_sparse_case("sparse::bicgstab_magma_f64_view", || {
        sparse::bicgstab_magma_f64_view(&nonsym_f64, &rhs_nonsym_f64, 1e-10_f64, 256)
    });
    run_sparse_case("sparse::bicgstab_magma_f32_view", || {
        sparse::bicgstab_magma_f32_view(&nonsym_f32, &rhs_nonsym_f32, 1e-3_f32, 512)
    });
    run_sparse_case("sparse::bicgstab_ilu0_magma_f64_view", || {
        sparse::bicgstab_ilu0_magma_f64_view(&nonsym_f64, &rhs_nonsym_f64, 1e-10_f64, 256)
    });
    run_sparse_case("sparse::bicgstab_ilu0_magma_f32_view", || {
        sparse::bicgstab_ilu0_magma_f32_view(&nonsym_f32, &rhs_nonsym_f32, 1e-3_f32, 512)
    });
}
