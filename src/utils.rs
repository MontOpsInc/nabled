//! # Utility Functions
//! 
//! Common utility functions for linear algebra operations.

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
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

/// Generate a random matrix with specified dimensions
/// Note: This function is simplified for now. In a full implementation,
/// you would use ndarray-rand with proper trait bounds.
pub fn random_matrix<T: Float>(rows: usize, cols: usize) -> Array2<T> {
    // For now, return a matrix of zeros
    // In a full implementation, you would use ndarray-rand
    Array2::zeros((rows, cols))
}

/// Check if a matrix is approximately equal to another
pub fn matrix_approx_eq<T: Float>(a: &Array2<T>, b: &Array2<T>, epsilon: T) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    
    a.iter().zip(b.iter()).all(|(&x, &y)| (x - y).abs() < epsilon)
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
                assert!(approx::relative_eq!(original[(i, j)], back[(i, j)], epsilon = 1e-10));
            }
        }
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let norm = frobenius_norm(&matrix);
        assert!((norm - 5.0).abs() < 1e-10);
    }
}
