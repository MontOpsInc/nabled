//! Integration tests for the linear algebra library

use approx::assert_relative_eq;
use nalgebra::DMatrix;
use ndarray::Array2;
use rust_linalg::svd::{SVDError, nalgebra_svd, ndarray_svd};

#[test]
fn test_nalgebra_svd_identity_matrix() {
    let identity = DMatrix::<f64>::identity(3, 3);
    let svd = nalgebra_svd::compute_svd(&identity).unwrap();

    // For identity matrix, singular values should all be 1.0
    for &sv in svd.singular_values.iter() {
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
    for &sv in svd.singular_values.iter() {
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
