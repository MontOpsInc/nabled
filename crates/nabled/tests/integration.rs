//! Integration tests for ndarray-first APIs.

use approx::assert_relative_eq;
use nabled::svd::{self as svd, SVDError};
use nabled::vector::{self as vector, PairwiseCosineWorkspace};
use nabled::{
    CpuBackend, CsrMatrix, CudaBackend, IntoNabledError, IterativeConfig, NabledError, accelerator,
    cholesky, eigen, iterative, lu, matrix, matrix_functions, orthogonalization, pca, polar,
    regression, schur, sparse, stats, sylvester, tensor, triangular,
};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

fn conjugate_transpose(matrix: &Array2<Complex64>) -> Array2<Complex64> {
    matrix.t().mapv(|value| value.conj())
}

#[test]
fn test_svd_identity_matrix() {
    let identity = Array2::<f64>::eye(3);
    let svd = svd::decompose(&identity).unwrap();

    for &sv in &svd.singular_values {
        assert_relative_eq!(sv, 1.0, epsilon = 1e-10);
    }
    assert_relative_eq!(svd::condition_number(&svd), 1.0, epsilon = 1e-10);
    assert_eq!(svd::rank(&svd, None), 3);
}

#[test]
fn test_svd_reconstruction() {
    let matrix =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    let svd = svd::decompose(&matrix).unwrap();
    let reconstructed = svd::reconstruct_matrix(&svd);

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            assert_relative_eq!(matrix[[i, j]], reconstructed[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_truncated_svd_and_errors() {
    let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let truncated = svd::decompose_truncated(&matrix, 1).unwrap();
    assert_eq!(truncated.singular_values.len(), 1);

    let invalid = svd::decompose_truncated(&matrix, 0);
    assert!(matches!(invalid, Err(SVDError::InvalidInput(_))));
}

#[test]
fn test_triangular_residual() {
    let lower =
        Array2::from_shape_vec((3, 3), vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = Array1::from_vec(vec![2.0, 5.0, 32.0]);
    let x = triangular::solve_lower(&lower, &rhs).unwrap();
    let reconstructed = lower.dot(&x);

    for i in 0..3 {
        assert_relative_eq!(reconstructed[i], rhs[i], epsilon = 1e-10);
    }
}

#[test]
fn test_iterative_cg_matches_direct_system() {
    let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let b = Array1::from_vec(vec![1.0, 2.0]);
    let x = iterative::conjugate_gradient(&a, &b, &IterativeConfig::default()).unwrap();
    let reconstructed = a.dot(&x);
    assert_relative_eq!(reconstructed[0], b[0], epsilon = 1e-8);
    assert_relative_eq!(reconstructed[1], b[1], epsilon = 1e-8);
}

#[test]
fn test_stats_covariance() {
    let matrix =
        Array2::from_shape_vec((4, 2), vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]).unwrap();
    let cov = stats::covariance_matrix(&matrix).unwrap();
    assert_eq!(cov.dim(), (2, 2));
}

#[test]
fn test_matrix_function_roundtrip() {
    let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let log_matrix = matrix_functions::matrix_log_eigen(&matrix).unwrap();
    let roundtrip = matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(roundtrip[[i, j]], matrix[[i, j]], epsilon = 1e-6);
        }
    }
}

#[test]
#[allow(clippy::many_single_char_names)]
fn test_regression_pca_orthogonalization_and_sylvester() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let regression = regression::linear_regression(&x, &y, true).unwrap();
    assert_relative_eq!(regression.coefficients[0], 1.0, epsilon = 1e-8);
    assert_relative_eq!(regression.coefficients[1], 2.0, epsilon = 1e-8);

    let pca_input = Array2::from_shape_vec((5, 3), vec![
        1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0,
    ])
    .unwrap();
    let pca = pca::compute_pca(&pca_input, Some(3)).unwrap();
    let transformed = pca::transform(&pca_input, &pca);
    let reconstructed = pca::inverse_transform(&transformed, &pca);
    assert_eq!(reconstructed.dim(), pca_input.dim());

    let q = orthogonalization::gram_schmidt_classic(&pca_input).unwrap();
    assert_eq!(q.nrows(), pca_input.nrows());

    let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
    let b = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 0.0, 4.0]).unwrap();
    let c = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let x = sylvester::solve_sylvester(&a, &b, &c).unwrap();
    assert_eq!(x.dim(), (2, 2));
}

#[test]
fn test_vector_primitives_and_workspace_paths() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
    let dot = vector::dot(&a, &b).unwrap();
    assert_relative_eq!(dot, 32.0, epsilon = 1e-10);

    let left = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let right = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

    let mut cosine = Array2::<f64>::zeros((left.nrows(), right.nrows()));
    let mut workspace = PairwiseCosineWorkspace::default();
    vector::pairwise_cosine_similarity_with_workspace_into(
        &left,
        &right,
        &mut cosine,
        &mut workspace,
    )
    .unwrap();
    assert_relative_eq!(cosine[[0, 0]], 1.0, epsilon = 1e-10);
    assert_relative_eq!(cosine[[0, 1]], 0.0, epsilon = 1e-10);

    let mut l2 = Array2::<f64>::zeros((left.nrows(), right.nrows()));
    vector::pairwise_l2_distance_into(&left, &right, &mut l2).unwrap();
    assert_relative_eq!(l2[[0, 0]], 0.0, epsilon = 1e-10);
}

#[test]
fn test_complex_dense_parity_pipeline() {
    let hpd = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(5.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(4.0, 0.0),
    ])
    .unwrap();
    let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]);

    let chol = cholesky::decompose_complex(&hpd).unwrap();
    let chol_view = cholesky::decompose_complex_view(&hpd.view()).unwrap();
    assert_eq!(chol.l.dim(), chol_view.l.dim());

    let chol_solution = cholesky::solve_complex(&hpd, &rhs).unwrap();
    let chol_solution_view = cholesky::solve_complex_view(&hpd.view(), &rhs.view()).unwrap();
    for i in 0..rhs.len() {
        assert!((chol_solution[i] - chol_solution_view[i]).norm() < 1e-10);
    }

    let lu_solution = lu::solve_complex(&hpd, &rhs).unwrap();
    let lu_solution_view = lu::solve_complex_view(&hpd.view(), &rhs.view()).unwrap();
    for i in 0..rhs.len() {
        assert!((lu_solution[i] - lu_solution_view[i]).norm() < 1e-10);
    }

    let svd_decomp = svd::decompose_complex(&hpd).unwrap();
    let reconstructed = svd::reconstruct_matrix_complex(&svd_decomp);
    for i in 0..2 {
        for j in 0..2 {
            assert!((reconstructed[[i, j]] - hpd[[i, j]]).norm() < 1e-8);
        }
    }

    let log_h = matrix_functions::matrix_log_eigen_complex(&hpd).unwrap();
    let roundtrip_h = matrix_functions::matrix_exp_eigen_complex(&log_h).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!((roundtrip_h[[i, j]] - hpd[[i, j]]).norm() < 1e-6);
        }
    }

    let signed_h = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(-4.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(9.0, 0.0),
    ])
    .unwrap();
    let sign_h = matrix_functions::matrix_sign_complex(&signed_h).unwrap();
    assert!((sign_h[[0, 0]] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
    assert!((sign_h[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);

    let polar_result = polar::compute_polar_complex(&hpd).unwrap();
    let polar_reconstructed = polar_result.u.dot(&polar_result.p);
    for i in 0..2 {
        for j in 0..2 {
            assert!((polar_reconstructed[[i, j]] - hpd[[i, j]]).norm() < 1e-6);
        }
    }

    let schur_result = schur::compute_schur_complex(&hpd).unwrap();
    let schur_reconstructed =
        schur_result.q.dot(&schur_result.t).dot(&conjugate_transpose(&schur_result.q));
    for i in 0..2 {
        for j in 0..2 {
            assert!((schur_reconstructed[[i, j]] - hpd[[i, j]]).norm() < 1e-6);
        }
    }

    let a = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 0.0),
    ])
    .unwrap();
    let b = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(4.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(5.0, 0.0),
    ])
    .unwrap();
    let c = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ])
    .unwrap();
    let x = sylvester::solve_sylvester_complex(&a, &b, &c).unwrap();
    let residual = a.dot(&x) + x.dot(&b) - c;
    for value in &residual {
        assert!(value.norm() < 1e-8);
    }
}

#[test]
fn test_complex_error_mapping_paths() {
    let non_hermitian = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ])
    .unwrap();
    let matrix_error = matrix_functions::matrix_log_eigen_complex(&non_hermitian)
        .expect_err("non-Hermitian input should error")
        .into_nabled_error();
    assert!(matches!(matrix_error, NabledError::NotSymmetric));

    let non_hpd = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ])
    .unwrap();
    let cholesky_error = cholesky::decompose_complex(&non_hpd)
        .expect_err("indefinite Hermitian input should error")
        .into_nabled_error();
    assert!(matches!(cholesky_error, NabledError::NotPositiveDefinite));

    let singular = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ])
    .unwrap();
    let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
    let lu_error = lu::solve_complex(&singular, &rhs)
        .expect_err("singular solve should error")
        .into_nabled_error();
    assert!(matches!(lu_error, NabledError::SingularMatrix));
}

#[test]
fn test_matrix_sparse_and_nonsymmetric_eigen_paths() {
    let dense = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 1.0]).unwrap();
    let input_vector = Array1::from_vec(vec![1.0, 0.0, 2.0]);
    let dense_y = matrix::matvec(&dense, &input_vector).unwrap();
    assert_relative_eq!(dense_y[0], 7.0, epsilon = 1e-10);
    assert_relative_eq!(dense_y[1], 2.0, epsilon = 1e-10);

    let dense_rhs = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let dense_mm = matrix::matmat(&dense, &dense_rhs).unwrap();
    assert_eq!(dense_mm.dim(), (2, 2));

    let batch_vectors =
        Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, 0.5, -1.0, 1.0]).unwrap();
    let batch_out = matrix::batched_row_matvec(&batch_vectors, &dense).unwrap();
    assert_eq!(batch_out.dim(), (2, 2));

    let left_batches = Array3::from_shape_vec((2, 2, 3), vec![
        1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 3.0, 2.0,
    ])
    .unwrap();
    let right_batches = Array3::from_shape_vec((2, 3, 2), vec![
        1.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0,
    ])
    .unwrap();
    let batched_mm = matrix::batched_matmat(&left_batches, &right_batches).unwrap();
    assert_eq!(batched_mm.dim(), (2, 2, 2));

    assert_sparse_tensor_and_accelerator_paths(&dense, &dense_rhs);

    let rotation = Array2::from_shape_vec((2, 2), vec![0.0, -1.0, 1.0, 0.0]).unwrap();
    let nonsymmetric = eigen::nonsymmetric(&rotation).unwrap();
    assert_eq!(nonsymmetric.eigenvalues.len(), 2);

    let bad_vector = Array1::from_vec(vec![1.0]);
    let matrix_error = matrix::matvec(&dense, &bad_vector).unwrap_err().into_nabled_error();
    assert!(matches!(matrix_error, NabledError::Shape(nabled::ShapeError::DimensionMismatch)));
}

fn assert_sparse_tensor_and_accelerator_paths(dense: &Array2<f64>, dense_rhs: &Array2<f64>) {
    let sparse_matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
        4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0,
    ])
    .unwrap();
    let transposed = sparse::transpose(&sparse_matrix).unwrap();
    assert_eq!(transposed.nrows, 3);
    assert_eq!(transposed.ncols, 3);

    let csc = sparse::csr_to_csc(&sparse_matrix).unwrap();
    let csc_rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let csc_matvec = sparse::matvec_csc(&csc, &csc_rhs).unwrap();
    assert_eq!(csc_matvec.len(), 3);

    let rhs_dense = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let sparse_mm = sparse::matmat_dense(&sparse_matrix, &rhs_dense).unwrap();
    assert_eq!(sparse_mm.dim(), (3, 2));

    let sparse_sparse = sparse::matmat_sparse(&sparse_matrix, &sparse_matrix).unwrap();
    assert_eq!(sparse_sparse.nrows, 3);
    assert_eq!(sparse_sparse.ncols, 3);

    let sparse_batch = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0]).unwrap();
    let sparse_batch_out = sparse::batched_matvec(&sparse_matrix, &sparse_batch).unwrap();
    assert_eq!(sparse_batch_out.dim(), (2, 3));

    let rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let cg = sparse::conjugate_gradient_solve(&sparse_matrix, &rhs, 1e-10, 2000).unwrap();
    let reconstructed = sparse::matvec(&sparse_matrix, &cg).unwrap();
    for i in 0..rhs.len() {
        assert_relative_eq!(reconstructed[i], rhs[i], epsilon = 1e-6);
    }

    let bicgstab = sparse::bicgstab_solve(&sparse_matrix, &rhs, 1e-10, 5000).unwrap();
    let reconstructed_bicgstab = sparse::matvec(&sparse_matrix, &bicgstab).unwrap();
    for i in 0..rhs.len() {
        assert_relative_eq!(reconstructed_bicgstab[i], rhs[i], epsilon = 1e-5);
    }

    let jacobi = sparse::jacobi_preconditioner(&sparse_matrix).unwrap();
    assert_eq!(jacobi.inverse_diagonal.len(), rhs.len());
    let pcg = sparse::pcg_solve(&sparse_matrix, &rhs, 1e-10, 2000).unwrap();
    let reconstructed_pcg = sparse::matvec(&sparse_matrix, &pcg).unwrap();
    for i in 0..rhs.len() {
        assert_relative_eq!(reconstructed_pcg[i], rhs[i], epsilon = 1e-6);
    }

    let cube = Array3::from_shape_vec((2, 2, 3), vec![
        1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 2.0, -1.0, 0.5, 3.0, 0.0, 2.0,
    ])
    .unwrap();
    let cube_vectors = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, 0.5, -1.0, 1.0]).unwrap();
    let cube_out = tensor::cube_matvec(&cube, &cube_vectors).unwrap();
    assert_eq!(cube_out.dim(), (2, 2));
    let flat = tensor::flatten_cubes(&cube).unwrap();
    assert_eq!(flat.dim(), (2, 6));

    let tensor_dyn = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2, 3]), vec![
        1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 2.0,
    ])
    .unwrap();
    let tensor_sum = tensor::sum_last_axis(&tensor_dyn).unwrap();
    assert_eq!(tensor_sum.shape(), &[2, 2]);
    let tensor_norm = tensor::l2_norm_last_axis(&tensor_dyn).unwrap();
    assert_eq!(tensor_norm.shape(), &[2, 2]);
    let normalized = tensor::normalize_last_axis(&tensor_dyn).unwrap();
    let normalized_norm = tensor::l2_norm_last_axis(&normalized).unwrap();
    for value in &normalized_norm {
        assert_relative_eq!(*value, 1.0, epsilon = 1e-10);
    }

    assert_accelerator_paths(dense, dense_rhs);
}

fn assert_accelerator_paths(dense: &Array2<f64>, dense_rhs: &Array2<f64>) {
    let cpu_result = accelerator::backends::execute::<CpuBackend, _, _>(|| 21 + 21).unwrap();
    assert_eq!(cpu_result, 42);
    let cuda_result = accelerator::backends::execute::<CudaBackend, _, _>(|| 1);
    assert!(cuda_result.is_err());

    let serial = accelerator::cpu::matmat_serial(dense, dense_rhs).unwrap();
    let backend_cuda =
        accelerator::dispatch::matmat_with_backend::<CudaBackend>(dense, dense_rhs).unwrap();
    for row in 0..serial.nrows() {
        for col in 0..serial.ncols() {
            assert_relative_eq!(serial[[row, col]], backend_cuda[[row, col]], epsilon = 1e-12);
        }
    }

    let accelerated = accelerator::cpu::matmat_accelerated(dense, dense_rhs);
    #[cfg(feature = "accelerator-rayon")]
    {
        assert!(accelerated.is_ok());
    }
    #[cfg(not(feature = "accelerator-rayon"))]
    {
        assert!(matches!(accelerated, Err(nabled::AcceleratorError::FeatureNotEnabled)));
    }
}
