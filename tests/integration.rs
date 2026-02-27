//! Integration tests for the linear algebra library

use approx::assert_relative_eq;
use nabled::IterativeConfig;
use nabled::iterative::{nalgebra_iterative, ndarray_iterative};
use nabled::matrix_functions::{nalgebra_matrix_functions, ndarray_matrix_functions};
use nabled::orthogonalization::ndarray_orthogonalization;
use nabled::pca::ndarray_pca;
use nabled::regression::ndarray_regression;
use nabled::stats::{nalgebra_stats, ndarray_stats};
use nabled::svd::{SVDError, nalgebra_svd, ndarray_svd};
use nabled::sylvester::ndarray_sylvester;
use nabled::triangular::{nalgebra_triangular, ndarray_triangular};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};

#[test]
fn test_nalgebra_svd_identity_matrix() {
    let identity = DMatrix::<f64>::identity(3, 3);
    let svd = nalgebra_svd::compute_svd(&identity).unwrap();

    // For identity matrix, singular values should all be 1.0
    for &sv in &svd.singular_values {
        assert_relative_eq!(sv, 1.0, epsilon = 1e-10);
    }

    // Condition number should be 1.0
    assert_relative_eq!(nalgebra_svd::condition_number(&svd), 1.0, epsilon = 1e-10);

    // Rank should be 3
    assert_eq!(nalgebra_svd::matrix_rank(&svd, None), 3);
}

#[test]
fn test_ndarray_svd_identity_matrix() {
    let identity = Array2::<f64>::eye(3);
    let svd = ndarray_svd::compute_svd(&identity).unwrap();

    // For identity matrix, singular values should all be 1.0
    for &sv in &svd.singular_values {
        assert_relative_eq!(sv, 1.0, epsilon = 1e-10);
    }

    // Condition number should be 1.0
    assert_relative_eq!(ndarray_svd::condition_number(&svd), 1.0, epsilon = 1e-10);

    // Rank should be 3
    assert_eq!(ndarray_svd::matrix_rank(&svd, None), 3);
}

#[test]
fn test_nalgebra_svd_reconstruction() {
    let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    let svd = nalgebra_svd::compute_svd(&matrix).unwrap();
    let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);

    // Check that reconstruction is accurate
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            assert_relative_eq!(matrix[(i, j)], reconstructed[(i, j)], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_ndarray_svd_reconstruction() {
    let matrix =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    let svd = ndarray_svd::compute_svd(&matrix).unwrap();
    let reconstructed = ndarray_svd::reconstruct_matrix(&svd);

    // Check that reconstruction is accurate
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            assert_relative_eq!(matrix[[i, j]], reconstructed[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_nalgebra_truncated_svd() {
    let matrix = DMatrix::from_row_slice(4, 4, &[
        1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0,
    ]);

    let full_svd = nalgebra_svd::compute_svd(&matrix).unwrap();
    let truncated_svd = nalgebra_svd::compute_truncated_svd(&matrix, 2).unwrap();

    // Truncated SVD should have only 2 singular values
    assert_eq!(truncated_svd.singular_values.len(), 2);
    assert_eq!(truncated_svd.u.ncols(), 2);
    assert_eq!(truncated_svd.vt.nrows(), 2);

    // The first two singular values should match
    assert_relative_eq!(
        truncated_svd.singular_values[0],
        full_svd.singular_values[0],
        epsilon = 1e-10
    );
    assert_relative_eq!(
        truncated_svd.singular_values[1],
        full_svd.singular_values[1],
        epsilon = 1e-10
    );
}

#[test]
fn test_ndarray_truncated_svd() {
    let matrix = Array2::from_shape_vec((4, 4), vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0,
    ])
    .unwrap();

    let full_svd = ndarray_svd::compute_svd(&matrix).unwrap();
    let truncated_svd = ndarray_svd::compute_truncated_svd(&matrix, 2).unwrap();

    // Truncated SVD should have only 2 singular values
    assert_eq!(truncated_svd.singular_values.len(), 2);
    assert_eq!(truncated_svd.u.ncols(), 2);
    assert_eq!(truncated_svd.vt.nrows(), 2);

    // The first two singular values should match
    assert_relative_eq!(
        truncated_svd.singular_values[0],
        full_svd.singular_values[0],
        epsilon = 1e-10
    );
    assert_relative_eq!(
        truncated_svd.singular_values[1],
        full_svd.singular_values[1],
        epsilon = 1e-10
    );
}

#[test]
fn test_nalgebra_svd_empty_matrix() {
    let empty_matrix = DMatrix::<f64>::zeros(0, 0);
    let result = nalgebra_svd::compute_svd(&empty_matrix);

    assert!(matches!(result, Err(SVDError::EmptyMatrix)));
}

#[test]
fn test_ndarray_svd_empty_matrix() {
    let empty_matrix = Array2::<f64>::zeros((0, 0));
    let result = ndarray_svd::compute_svd(&empty_matrix);

    assert!(matches!(result, Err(SVDError::EmptyMatrix)));
}

#[test]
fn test_nalgebra_svd_invalid_truncation() {
    let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let result = nalgebra_svd::compute_truncated_svd(&matrix, 0);

    assert!(matches!(result, Err(SVDError::InvalidInput(_))));
}

#[test]
fn test_ndarray_svd_invalid_truncation() {
    let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = ndarray_svd::compute_truncated_svd(&matrix, 0);

    assert!(matches!(result, Err(SVDError::InvalidInput(_))));
}

#[test]
fn test_nalgebra_svd_rank_deficient_matrix() {
    // Create a rank-deficient matrix
    let matrix = DMatrix::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0, 2.0, 4.0, 6.0, // This row is 2 * first row
        1.0, 2.0, 3.0, // This row is same as first row
    ]);

    let svd = nalgebra_svd::compute_svd(&matrix).unwrap();
    let rank = nalgebra_svd::matrix_rank(&svd, None);

    // Matrix should have rank 1 (only one linearly independent row)
    assert_eq!(rank, 1);
}

#[test]
fn test_ndarray_svd_rank_deficient_matrix() {
    // Create a rank-deficient matrix
    let matrix = Array2::from_shape_vec((3, 3), vec![
        1.0, 2.0, 3.0, 2.0, 4.0, 6.0, // This row is 2 * first row
        1.0, 2.0, 3.0, // This row is same as first row
    ])
    .unwrap();

    let svd = ndarray_svd::compute_svd(&matrix).unwrap();
    let rank = ndarray_svd::matrix_rank(&svd, None);

    // Matrix should have rank 1 (only one linearly independent row)
    assert_eq!(rank, 1);
}

#[test]
fn test_integration_triangular_cross_backend() {
    let l_nalg = DMatrix::from_row_slice(3, 3, &[2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0]);
    let b_nalg = DVector::from_vec(vec![2.0, 5.0, 32.0]);
    let x_nalg = nalgebra_triangular::solve_lower(&l_nalg, &b_nalg).unwrap();

    let l_nd =
        Array2::from_shape_vec((3, 3), vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0]).unwrap();
    let b_nd = Array1::from_vec(vec![2.0, 5.0, 32.0]);
    let x_nd = ndarray_triangular::solve_lower(&l_nd, &b_nd).unwrap();

    for i in 0..3 {
        assert_relative_eq!(x_nalg[i], x_nd[i], epsilon = 1e-10);
    }
}

#[test]
fn test_integration_iterative_cross_backend() {
    let a_nalg = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    let b_nalg = DVector::from_vec(vec![1.0, 2.0]);
    let x_nalg =
        nalgebra_iterative::conjugate_gradient(&a_nalg, &b_nalg, &IterativeConfig::default())
            .unwrap();

    let a_nd = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let b_nd = Array1::from_vec(vec![1.0, 2.0]);
    let x_nd =
        ndarray_iterative::conjugate_gradient(&a_nd, &b_nd, &IterativeConfig::default()).unwrap();

    for i in 0..2 {
        assert_relative_eq!(x_nalg[i], x_nd[i], epsilon = 1e-8);
    }
}

#[test]
fn test_integration_stats_cross_backend() {
    let nalg_matrix = DMatrix::from_row_slice(4, 2, &[1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]);
    let nd_matrix =
        Array2::from_shape_vec((4, 2), vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]).unwrap();

    let cov_nalg = nalgebra_stats::covariance_matrix(&nalg_matrix).unwrap();
    let cov_nd = ndarray_stats::covariance_matrix(&nd_matrix).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(cov_nalg[(i, j)], cov_nd[[i, j]], epsilon = 1e-12);
        }
    }
}

#[test]
fn test_integration_matrix_function_cross_backend() {
    let nalg_matrix = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    let log_nalg = nalgebra_matrix_functions::matrix_log_eigen(&nalg_matrix).unwrap();
    let roundtrip_nalg = nalgebra_matrix_functions::matrix_exp_eigen(&log_nalg).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(roundtrip_nalg[(i, j)], nalg_matrix[(i, j)], epsilon = 1e-8);
        }
    }

    let nd_matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let log_nd = ndarray_matrix_functions::matrix_log_eigen(&nd_matrix).unwrap();
    let roundtrip_nd = ndarray_matrix_functions::matrix_exp_eigen(&log_nd).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(roundtrip_nd[[i, j]], nd_matrix[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_integration_ndarray_regression_pca_and_orthogonalization() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let regression = ndarray_regression::linear_regression(&x, &y, true).unwrap();
    assert_relative_eq!(regression.coefficients[0], 1.0, epsilon = 0.1);
    assert_relative_eq!(regression.coefficients[1], 2.0, epsilon = 0.1);
    assert_relative_eq!(regression.r_squared, 1.0, epsilon = 1e-10);

    let pca_input = Array2::from_shape_vec((5, 3), vec![
        1.0, 2.0, 1.5, 2.0, 1.0, 3.5, 3.0, 4.0, 2.0, 4.0, 3.0, 5.0, 5.0, 5.0, 4.5,
    ])
    .unwrap();
    let pca = ndarray_pca::compute_pca(&pca_input, Some(3)).unwrap();
    let transformed = ndarray_pca::transform(&pca_input, &pca);
    let reconstructed = ndarray_pca::inverse_transform(&transformed, &pca);
    for i in 0..pca_input.nrows() {
        for j in 0..pca_input.ncols() {
            assert_relative_eq!(reconstructed[[i, j]], pca_input[[i, j]], epsilon = 1e-8);
        }
    }

    let q = ndarray_orthogonalization::gram_schmidt_classic(&pca_input).unwrap();
    assert_eq!(q.nrows(), pca_input.nrows());
    assert_eq!(q.ncols(), pca_input.ncols());
}

#[test]
fn test_integration_ndarray_sylvester_residual() {
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
    let b = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 0.0, 4.0]).unwrap();
    let c = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let x = ndarray_sylvester::solve_sylvester(&a, &b, &c).unwrap();

    let ax = a.dot(&x);
    let xb = x.dot(&b);
    let residual = &ax + &xb;
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(residual[[i, j]], c[[i, j]], epsilon = 1e-8);
        }
    }
}
