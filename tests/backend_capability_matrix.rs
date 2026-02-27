//! Backend capability matrix tests.

use nabled::cholesky::{nalgebra_cholesky, ndarray_cholesky};
use nabled::eigen::{nalgebra_eigen, ndarray_eigen};
use nabled::lu::{nalgebra_lu, ndarray_lu};
use nabled::qr::{QRConfig, nalgebra_qr, ndarray_qr};
use nabled::schur::{nalgebra_schur, ndarray_schur};
use nabled::svd::{nalgebra_svd, ndarray_svd};
use nabled::triangular::{nalgebra_triangular, ndarray_triangular};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};

fn symmetric_matrix_nalg() -> DMatrix<f64> { DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]) }

fn symmetric_matrix_nd() -> Array2<f64> {
    Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap()
}

#[test]
fn test_tier_a_svd_baseline() {
    assert_eq!(
        nalgebra_svd::compute_svd(&symmetric_matrix_nalg()).unwrap().singular_values.len(),
        2
    );
    assert_eq!(ndarray_svd::compute_svd(&symmetric_matrix_nd()).unwrap().singular_values.len(), 2);
}

#[test]
fn test_tier_a_qr_baseline() {
    let config = QRConfig::default();
    assert_eq!(nalgebra_qr::compute_qr(&symmetric_matrix_nalg(), &config).unwrap().rank, 2);
    assert_eq!(ndarray_qr::compute_qr(&symmetric_matrix_nd(), &config).unwrap().rank, 2);
}

#[test]
fn test_tier_a_lu_baseline() {
    assert_eq!(nalgebra_lu::compute_lu(&symmetric_matrix_nalg()).unwrap().l.shape(), (2, 2));
    assert_eq!(ndarray_lu::compute_lu(&symmetric_matrix_nd()).unwrap().l.dim(), (2, 2));
}

#[test]
fn test_tier_a_cholesky_baseline() {
    assert_eq!(
        nalgebra_cholesky::compute_cholesky(&symmetric_matrix_nalg()).unwrap().l.shape(),
        (2, 2)
    );
    assert_eq!(ndarray_cholesky::compute_cholesky(&symmetric_matrix_nd()).unwrap().l.dim(), (2, 2));
}

#[test]
fn test_tier_a_eigen_baseline() {
    assert_eq!(
        nalgebra_eigen::compute_symmetric_eigen(&symmetric_matrix_nalg())
            .unwrap()
            .eigenvalues
            .len(),
        2
    );
    assert_eq!(
        ndarray_eigen::compute_symmetric_eigen(&symmetric_matrix_nd()).unwrap().eigenvalues.len(),
        2
    );
}

#[test]
fn test_tier_a_schur_baseline() {
    assert_eq!(nalgebra_schur::compute_schur(&symmetric_matrix_nalg()).unwrap().t.shape(), (2, 2));
    assert_eq!(ndarray_schur::compute_schur(&symmetric_matrix_nd()).unwrap().t.dim(), (2, 2));
}

#[test]
fn test_tier_a_triangular_baseline() {
    let lower_nalg = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
    let rhs_nalg = DVector::from_vec(vec![4.0, 8.0]);
    assert_eq!(nalgebra_triangular::solve_lower(&lower_nalg, &rhs_nalg).unwrap().len(), 2);

    let lower_nd = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
    let rhs_nd = Array1::from_vec(vec![4.0, 8.0]);
    assert_eq!(ndarray_triangular::solve_lower(&lower_nd, &rhs_nd).unwrap().len(), 2);
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[test]
fn test_tier_a_svd_lapack() {
    assert_eq!(
        nalgebra_svd::compute_svd_lapack(&symmetric_matrix_nalg()).unwrap().singular_values.len(),
        2
    );
    assert_eq!(
        ndarray_svd::compute_svd_lapack(&symmetric_matrix_nd()).unwrap().singular_values.len(),
        2
    );
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[test]
fn test_tier_a_qr_lapack() {
    let config = QRConfig::default();
    assert_eq!(nalgebra_qr::compute_qr_lapack(&symmetric_matrix_nalg(), &config).unwrap().rank, 2);
    assert_eq!(ndarray_qr::compute_qr_lapack(&symmetric_matrix_nd(), &config).unwrap().rank, 2);
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[test]
fn test_tier_a_lu_lapack() {
    assert_eq!(nalgebra_lu::compute_lu_lapack(&symmetric_matrix_nalg()).unwrap().l.shape(), (2, 2));
    assert_eq!(ndarray_lu::compute_lu_lapack(&symmetric_matrix_nd()).unwrap().l.dim(), (2, 2));
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[test]
fn test_tier_a_cholesky_lapack() {
    assert_eq!(
        nalgebra_cholesky::compute_cholesky_lapack(&symmetric_matrix_nalg()).unwrap().l.shape(),
        (2, 2)
    );
    assert_eq!(
        ndarray_cholesky::compute_cholesky_lapack(&symmetric_matrix_nd()).unwrap().l.dim(),
        (2, 2)
    );
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[test]
fn test_tier_a_eigen_lapack() {
    assert_eq!(
        nalgebra_eigen::compute_symmetric_eigen_lapack(&symmetric_matrix_nalg())
            .unwrap()
            .eigenvalues
            .len(),
        2
    );
    assert_eq!(
        ndarray_eigen::compute_symmetric_eigen_lapack(&symmetric_matrix_nd())
            .unwrap()
            .eigenvalues
            .len(),
        2
    );
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[test]
fn test_tier_a_schur_lapack() {
    assert_eq!(
        nalgebra_schur::compute_schur_lapack(&symmetric_matrix_nalg()).unwrap().t.shape(),
        (2, 2)
    );
    assert_eq!(
        ndarray_schur::compute_schur_lapack(&symmetric_matrix_nd()).unwrap().t.dim(),
        (2, 2)
    );
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[test]
fn test_tier_a_triangular_lapack_build() {
    // Triangular currently routes through baseline kernels under lapack-enabled builds.
    let lower_nalg = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
    let rhs_nalg = DVector::from_vec(vec![4.0, 8.0]);
    assert_eq!(nalgebra_triangular::solve_lower(&lower_nalg, &rhs_nalg).unwrap().len(), 2);

    let lower_nd = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
    let rhs_nd = Array1::from_vec(vec![4.0, 8.0]);
    assert_eq!(ndarray_triangular::solve_lower(&lower_nd, &rhs_nd).unwrap().len(), 2);
}
