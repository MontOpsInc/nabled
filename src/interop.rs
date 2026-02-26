//! # Interop Helpers
//!
//! Conversion helpers between nalgebra and ndarray matrix representations.

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;

/// Convert a nalgebra matrix into an ndarray matrix.
#[must_use]
pub(crate) fn nalgebra_to_ndarray<T: Float>(matrix: &DMatrix<T>) -> Array2<T> {
    let (rows, cols) = matrix.shape();
    let mut data = Vec::with_capacity(rows * cols);

    // nalgebra is column-major, but ndarray expects row-major data here.
    for i in 0..rows {
        for j in 0..cols {
            data.push(matrix[(i, j)]);
        }
    }

    Array2::from_shape_vec((rows, cols), data)
        .expect("matrix shape and data length should always match")
}

/// Convert an ndarray matrix into a nalgebra matrix.
#[must_use]
pub(crate) fn ndarray_to_nalgebra<T: RealField>(array: &Array2<T>) -> DMatrix<T> {
    let (rows, cols) = array.dim();
    let mut matrix = DMatrix::zeros(rows, cols);

    for i in 0..rows {
        for j in 0..cols {
            matrix[(i, j)] = array[[i, j]].clone();
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use super::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    #[test]
    fn conversion_roundtrip_preserves_entries() {
        let original = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let converted = nalgebra_to_ndarray(&original);
        let back = ndarray_to_nalgebra(&converted);

        for i in 0..original.nrows() {
            for j in 0..original.ncols() {
                assert!(approx::relative_eq!(original[(i, j)], back[(i, j)], epsilon = 1e-10));
            }
        }
    }
}
