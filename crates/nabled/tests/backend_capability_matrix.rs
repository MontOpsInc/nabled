//! Backend capability matrix tests for ndarray-first APIs.

use nabled::cholesky::ndarray_cholesky;
use nabled::eigen::ndarray_eigen;
use nabled::lu::ndarray_lu;
#[cfg(feature = "openblas-system")]
use nabled::matrix_functions::ndarray_matrix_functions;
use nabled::qr::{QRConfig, ndarray_qr};
use nabled::schur::ndarray_schur;
use nabled::svd::ndarray_svd;
use nabled::triangular::ndarray_triangular;
use ndarray::{Array1, Array2};

fn symmetric_matrix() -> Array2<f64> {
    Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap()
}

#[test]
fn test_tier_a_svd_baseline() {
    assert_eq!(ndarray_svd::compute_svd(&symmetric_matrix()).unwrap().singular_values.len(), 2);
}

#[test]
fn test_tier_a_qr_baseline() {
    let config = QRConfig::default();
    assert_eq!(ndarray_qr::compute_qr(&symmetric_matrix(), &config).unwrap().rank, 2);
}

#[test]
fn test_tier_a_lu_baseline() {
    assert_eq!(ndarray_lu::compute_lu(&symmetric_matrix()).unwrap().l.dim(), (2, 2));
}

#[test]
fn test_tier_a_cholesky_baseline() {
    assert_eq!(ndarray_cholesky::compute_cholesky(&symmetric_matrix()).unwrap().l.dim(), (2, 2));
}

#[test]
fn test_tier_a_eigen_baseline() {
    assert_eq!(
        ndarray_eigen::compute_symmetric_eigen(&symmetric_matrix()).unwrap().eigenvalues.len(),
        2
    );
}

#[test]
fn test_tier_a_schur_baseline() {
    assert_eq!(ndarray_schur::compute_schur(&symmetric_matrix()).unwrap().t.dim(), (2, 2));
}

#[test]
fn test_tier_a_triangular_baseline() {
    let lower = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
    let rhs = Array1::from_vec(vec![4.0, 8.0]);
    assert_eq!(ndarray_triangular::solve_lower(&lower, &rhs).unwrap().len(), 2);
}

#[cfg(feature = "openblas-system")]
#[test]
fn test_tier_a_svd_provider() {
    assert_eq!(ndarray_svd::compute_svd(&symmetric_matrix()).unwrap().singular_values.len(), 2);
}

#[cfg(feature = "openblas-system")]
#[test]
fn test_tier_a_qr_provider() {
    let config = QRConfig::default();
    assert_eq!(ndarray_qr::compute_qr(&symmetric_matrix(), &config).unwrap().rank, 2);
}

#[cfg(feature = "openblas-system")]
#[test]
fn test_tier_a_lu_provider() {
    assert_eq!(ndarray_lu::compute_lu(&symmetric_matrix()).unwrap().l.dim(), (2, 2));
}

#[cfg(feature = "openblas-system")]
#[test]
fn test_tier_a_cholesky_provider() {
    assert_eq!(ndarray_cholesky::compute_cholesky(&symmetric_matrix()).unwrap().l.dim(), (2, 2));
}

#[cfg(feature = "openblas-system")]
#[test]
fn test_tier_a_eigen_provider() {
    assert_eq!(
        ndarray_eigen::compute_symmetric_eigen(&symmetric_matrix()).unwrap().eigenvalues.len(),
        2
    );
}

#[cfg(feature = "openblas-system")]
#[test]
fn test_tier_a_schur_provider() {
    assert_eq!(ndarray_schur::compute_schur(&symmetric_matrix()).unwrap().t.dim(), (2, 2));
}

#[cfg(feature = "openblas-system")]
#[test]
fn test_tier_a_triangular_provider() {
    let lower = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
    let rhs = Array1::from_vec(vec![4.0, 8.0]);
    assert_eq!(ndarray_triangular::solve_lower(&lower, &rhs).unwrap().len(), 2);
}

#[cfg(feature = "openblas-system")]
#[test]
fn test_tier_b_matrix_functions_provider() {
    let exp_matrix = ndarray_matrix_functions::matrix_exp_eigen(&symmetric_matrix()).unwrap();
    assert_eq!(exp_matrix.dim(), (2, 2));
}
