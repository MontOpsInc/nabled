//! # Utility Functions
//!
//! Common utility functions for linear algebra operations.

use nalgebra::{DMatrix, RealField};
use ndarray::Array;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num_traits::Float;

/// Convert nalgebra DMatrix to ndarray Array2
pub fn nalgebra_to_ndarray<T: Float>(matrix: &DMatrix<T>) -> Array2<T> {
    let (rows, cols) = matrix.shape();
    let mut data = Vec::with_capacity(rows * cols);

    // nalgebra is column-major, but we need row-major for ndarray
    for i in 0..rows {
        for j in 0..cols {
            data.push(matrix[(i, j)]);
        }
    }

    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// Convert ndarray Array2 to nalgebra DMatrix
pub fn ndarray_to_nalgebra<T: RealField>(array: &Array2<T>) -> DMatrix<T> {
    let (rows, cols) = array.dim();
    let mut matrix = DMatrix::zeros(rows, cols);

    // Copy element by element to ensure correct layout
    for i in 0..rows {
        for j in 0..cols {
            matrix[(i, j)] = array[[i, j]].clone();
        }
    }

    matrix
}

/// Generate a random matrix with specified dimensions.
/// Elements are drawn from a uniform distribution in [0, 1).
pub fn random_matrix<T>(rows: usize, cols: usize) -> Array2<T>
where
    T: Float + rand::distributions::uniform::SampleUniform + Copy,
{
    Array::random((rows, cols), Uniform::new(T::zero(), T::one()))
}

/// Check if a matrix is approximately equal to another
pub fn matrix_approx_eq<T: Float>(a: &Array2<T>, b: &Array2<T>, epsilon: T) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    a.iter()
        .zip(b.iter())
        .all(|(&x, &y)| (x - y).abs() < epsilon)
}

/// Compute the Frobenius norm of a matrix
pub fn frobenius_norm<T: Float + std::iter::Sum>(matrix: &Array2<T>) -> T {
    matrix.iter().map(|&x| x * x).sum::<T>().sqrt()
}

/// Compute the spectral norm (largest singular value) of a matrix
/// Note: This uses conversion to nalgebra for computation
pub fn spectral_norm<T: Float + RealField>(matrix: &Array2<T>) -> T {
    // Convert to nalgebra and compute SVD
    let nalgebra_matrix = ndarray_to_nalgebra(matrix);
    match crate::svd::nalgebra_svd::compute_svd(&nalgebra_matrix) {
        Ok(svd) => svd.singular_values.max(),
        Err(_) => T::zero(),
    }
}

/// Nalgebra-backed matrix norms and operations
pub mod nalgebra_utils {
    use nalgebra::{DMatrix, RealField};

    /// Compute the trace (sum of diagonal elements) of a matrix
    #[must_use]
    pub fn trace<T: RealField + Copy>(matrix: &DMatrix<T>) -> T {
        let n = matrix.nrows().min(matrix.ncols());
        (0..n).map(|i| matrix[(i, i)]).fold(T::zero(), |a, b| a + b)
    }

    /// Compute the 1-norm (max column sum) of a matrix
    #[must_use]
    pub fn norm_1<T: RealField + Copy + num_traits::Float>(matrix: &DMatrix<T>) -> T {
        if matrix.is_empty() {
            return T::zero();
        }
        let (_, cols) = matrix.shape();
        (0..cols)
            .map(|j| {
                matrix
                    .column(j)
                    .iter()
                    .map(|x| x.abs())
                    .fold(T::zero(), |a, b| a + b)
            })
            .fold(T::zero(), num_traits::Float::max)
    }

    /// Compute the infinity norm (max row sum) of a matrix
    #[must_use]
    pub fn norm_inf<T: RealField + Copy + num_traits::Float>(matrix: &DMatrix<T>) -> T {
        if matrix.is_empty() {
            return T::zero();
        }
        let (rows, _) = matrix.shape();
        (0..rows)
            .map(|i| {
                matrix
                    .row(i)
                    .iter()
                    .map(|x| x.abs())
                    .fold(T::zero(), |a, b| a + b)
            })
            .fold(T::zero(), num_traits::Float::max)
    }

    /// Compute the nuclear norm (sum of singular values) of a matrix
    pub fn nuclear_norm<T: RealField + Copy + num_traits::Float>(
        matrix: &DMatrix<T>,
    ) -> Result<T, crate::svd::SVDError> {
        let svd = crate::svd::nalgebra_svd::compute_svd(matrix)?;
        Ok(svd.singular_values.iter().fold(T::zero(), |a, b| a + *b))
    }

    /// Compute the Kronecker product A ⊗ B of two matrices
    #[must_use]
    pub fn kronecker_product<T: RealField + Copy>(a: &DMatrix<T>, b: &DMatrix<T>) -> DMatrix<T> {
        let (ra, ca) = a.shape();
        let (rb, cb) = b.shape();
        let mut result = DMatrix::zeros(ra * rb, ca * cb);
        for i in 0..ra {
            for j in 0..ca {
                let aij = a[(i, j)];
                for p in 0..rb {
                    for q in 0..cb {
                        result[(i * rb + p, j * cb + q)] = aij * b[(p, q)];
                    }
                }
            }
        }
        result
    }
}

/// Ndarray-backed matrix norms and operations
pub mod ndarray_utils {
    use nalgebra::RealField;
    use ndarray::{Array2, Axis};
    use num_traits::Float;

    use crate::utils::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute the trace (sum of diagonal elements) of a matrix
    #[must_use]
    pub fn trace<T: Float>(matrix: &Array2<T>) -> T {
        let n = matrix.nrows().min(matrix.ncols());
        (0..n).map(|i| matrix[[i, i]]).fold(T::zero(), |a, b| a + b)
    }

    /// Compute the 1-norm (max column sum) of a matrix
    #[must_use]
    pub fn norm_1<T: Float>(matrix: &Array2<T>) -> T {
        if matrix.is_empty() {
            return T::zero();
        }
        matrix
            .axis_iter(Axis(1))
            .map(|col| col.mapv(|x| x.abs()).sum())
            .fold(T::zero(), T::max)
    }

    /// Compute the infinity norm (max row sum) of a matrix
    #[must_use]
    pub fn norm_inf<T: Float>(matrix: &Array2<T>) -> T {
        if matrix.is_empty() {
            return T::zero();
        }
        matrix
            .axis_iter(Axis(0))
            .map(|row| row.mapv(|x| x.abs()).sum())
            .fold(T::zero(), T::max)
    }

    /// Compute the nuclear norm (sum of singular values) of a matrix
    pub fn nuclear_norm<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<T, crate::svd::SVDError> {
        let nalg = ndarray_to_nalgebra(matrix);
        super::nalgebra_utils::nuclear_norm(&nalg)
    }

    /// Compute the Kronecker product A ⊗ B of two matrices
    #[must_use]
    pub fn kronecker_product<T: Float + RealField>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_b = ndarray_to_nalgebra(b);
        nalgebra_to_ndarray(&super::nalgebra_utils::kronecker_product(&nalg_a, &nalg_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use ndarray::Array2;

    #[test]
    fn test_conversion_roundtrip() {
        let original = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let converted = nalgebra_to_ndarray(&original);
        let back = ndarray_to_nalgebra(&converted);

        for i in 0..original.nrows() {
            for j in 0..original.ncols() {
                assert!(approx::relative_eq!(
                    original[(i, j)],
                    back[(i, j)],
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let norm = frobenius_norm(&matrix);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_norm() {
        let identity = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(approx::relative_eq!(
            spectral_norm(&identity),
            1.0,
            epsilon = 1e-10
        ));

        let diagonal = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 3.0]).unwrap();
        assert!(approx::relative_eq!(
            spectral_norm(&diagonal),
            3.0,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_matrix_approx_eq() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(matrix_approx_eq(&a, &b, 1e-10));

        let c = Array2::from_shape_vec((2, 2), vec![1.0 + 1e-5, 2.0, 3.0, 4.0]).unwrap();
        assert!(!matrix_approx_eq(&a, &c, 1e-10));
        assert!(matrix_approx_eq(&a, &c, 1e-4));

        let d = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(!matrix_approx_eq(&a, &d, 1e-10));
    }

    #[test]
    fn test_random_matrix() {
        let m = random_matrix::<f64>(3, 4);
        assert_eq!(m.shape(), &[3, 4]);
    }

    #[test]
    fn test_nalgebra_trace() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(approx::relative_eq!(
            nalgebra_utils::trace(&m),
            5.0,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_ndarray_trace() {
        let m = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(approx::relative_eq!(
            ndarray_utils::trace(&m),
            5.0,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_nalgebra_norm_1() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(approx::relative_eq!(
            nalgebra_utils::norm_1(&m),
            6.0,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_nalgebra_norm_inf() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(approx::relative_eq!(
            nalgebra_utils::norm_inf(&m),
            7.0,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_nalgebra_nuclear_norm() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let nuc = nalgebra_utils::nuclear_norm(&m).unwrap();
        assert!(approx::relative_eq!(nuc, 2.0, epsilon = 1e-10));
    }

    #[test]
    fn test_nalgebra_kronecker() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DMatrix::from_row_slice(1, 1, &[10.0]);
        let k = nalgebra_utils::kronecker_product(&a, &b);
        assert_eq!(k.nrows(), 2);
        assert_eq!(k.ncols(), 2);
        assert!(approx::relative_eq!(k[(0, 0)], 10.0, epsilon = 1e-10));
        assert!(approx::relative_eq!(k[(1, 0)], 30.0, epsilon = 1e-10));
    }

    #[test]
    fn test_ndarray_kronecker() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = ndarray_utils::kronecker_product(&a, &b);
        assert_eq!(k.shape(), &[4, 4]);
    }
}
