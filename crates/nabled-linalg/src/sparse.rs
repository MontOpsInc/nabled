//! Sparse matrix primitives and iterative solves over CSR matrices.

use ndarray::Array1;
use thiserror::Error;

const DEFAULT_TOLERANCE: f64 = 1.0e-12;

/// Error type for sparse matrix operations.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SparseError {
    /// Matrix or vector input is empty.
    #[error("input cannot be empty")]
    EmptyInput,
    /// CSR structure is invalid.
    #[error("invalid CSR structure")]
    InvalidStructure,
    /// Dimensions are incompatible.
    #[error("input dimensions are incompatible")]
    DimensionMismatch,
    /// Matrix is singular.
    #[error("matrix is singular")]
    SingularMatrix,
    /// Iterative solve exceeded iteration budget.
    #[error("maximum iterations exceeded")]
    MaxIterationsExceeded,
}

/// Compressed sparse row (CSR) matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    /// Number of rows.
    pub nrows:   usize,
    /// Number of columns.
    pub ncols:   usize,
    /// Row pointer offsets (`len = nrows + 1`).
    pub indptr:  Vec<usize>,
    /// Column index for each non-zero value.
    pub indices: Vec<usize>,
    /// Non-zero values.
    pub data:    Vec<f64>,
}

impl CsrMatrix {
    /// Construct a CSR matrix after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or CSR arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Result<Self, SparseError> {
        if nrows == 0 || ncols == 0 {
            return Err(SparseError::EmptyInput);
        }
        if indptr.len() != nrows + 1 {
            return Err(SparseError::InvalidStructure);
        }
        if indptr.first().copied().unwrap_or(usize::MAX) != 0 {
            return Err(SparseError::InvalidStructure);
        }
        if indices.len() != data.len() {
            return Err(SparseError::InvalidStructure);
        }
        if indptr[nrows] != indices.len() {
            return Err(SparseError::InvalidStructure);
        }
        for row in 0..nrows {
            if indptr[row] > indptr[row + 1] {
                return Err(SparseError::InvalidStructure);
            }
        }
        if indices.iter().any(|&index| index >= ncols) {
            return Err(SparseError::InvalidStructure);
        }

        Ok(Self { nrows, ncols, indptr, indices, data })
    }
}

/// Coordinate list (COO) sparse matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix {
    /// Number of rows.
    pub nrows:       usize,
    /// Number of columns.
    pub ncols:       usize,
    /// Row index for each non-zero entry.
    pub row_indices: Vec<usize>,
    /// Column index for each non-zero entry.
    pub col_indices: Vec<usize>,
    /// Non-zero values.
    pub data:        Vec<f64>,
}

impl CooMatrix {
    /// Construct a COO matrix after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or COO arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Result<Self, SparseError> {
        if nrows == 0 || ncols == 0 {
            return Err(SparseError::EmptyInput);
        }
        if row_indices.len() != col_indices.len() || row_indices.len() != data.len() {
            return Err(SparseError::InvalidStructure);
        }
        if row_indices.iter().any(|&row| row >= nrows)
            || col_indices.iter().any(|&col| col >= ncols)
        {
            return Err(SparseError::InvalidStructure);
        }

        Ok(Self { nrows, ncols, row_indices, col_indices, data })
    }

    /// Convert COO to CSR. Duplicate coordinates are summed.
    ///
    /// # Errors
    /// Returns an error if COO structure is invalid.
    pub fn to_csr(&self) -> Result<CsrMatrix, SparseError> {
        let mut entries = self
            .row_indices
            .iter()
            .copied()
            .zip(self.col_indices.iter().copied())
            .zip(self.data.iter().copied())
            .map(|((row, col), value)| (row, col, value))
            .collect::<Vec<_>>();
        entries.sort_by_key(|&(row, col, _)| (row, col));

        let mut collapsed = Vec::<(usize, usize, f64)>::new();
        for (row, col, value) in entries {
            if let Some((last_row, last_col, last_value)) = collapsed.last_mut()
                && *last_row == row
                && *last_col == col
            {
                *last_value += value;
            } else {
                collapsed.push((row, col, value));
            }
        }

        let mut indptr = vec![0_usize; self.nrows + 1];
        for &(row, _, _) in &collapsed {
            indptr[row + 1] += 1;
        }
        for row in 0..self.nrows {
            indptr[row + 1] += indptr[row];
        }

        let indices = collapsed.iter().map(|&(_, col, _)| col).collect::<Vec<_>>();
        let data = collapsed.iter().map(|&(_, _, value)| value).collect::<Vec<_>>();
        CsrMatrix::new(self.nrows, self.ncols, indptr, indices, data)
    }
}

/// Ndarray-backed sparse primitives.
pub mod ndarray_sparse {
    use super::*;

    /// Compute sparse matrix-vector product `y = A x`.
    ///
    /// # Errors
    /// Returns an error if vector length mismatches matrix columns.
    pub fn matvec(matrix: &CsrMatrix, vector: &Array1<f64>) -> Result<Array1<f64>, SparseError> {
        let mut output = Array1::<f64>::zeros(matrix.nrows);
        matvec_into(matrix, vector, &mut output)?;
        Ok(output)
    }

    /// Compute sparse matrix-vector product `y = A x` into `output`.
    ///
    /// # Errors
    /// Returns an error if input/output dimensions are incompatible.
    pub fn matvec_into(
        matrix: &CsrMatrix,
        vector: &Array1<f64>,
        output: &mut Array1<f64>,
    ) -> Result<(), SparseError> {
        if vector.len() != matrix.ncols || output.len() != matrix.nrows {
            return Err(SparseError::DimensionMismatch);
        }

        for row in 0..matrix.nrows {
            let start = matrix.indptr[row];
            let end = matrix.indptr[row + 1];
            let mut sum = 0.0_f64;
            for entry in start..end {
                sum += matrix.data[entry] * vector[matrix.indices[entry]];
            }
            output[row] = sum;
        }

        Ok(())
    }

    /// Solve sparse linear system `A x = b` with Jacobi iteration.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions, singular diagonals, or non-convergence.
    pub fn jacobi_solve(
        matrix: &CsrMatrix,
        rhs: &Array1<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Array1<f64>, SparseError> {
        if matrix.nrows != matrix.ncols {
            return Err(SparseError::DimensionMismatch);
        }
        if rhs.len() != matrix.nrows {
            return Err(SparseError::DimensionMismatch);
        }
        if rhs.is_empty() {
            return Err(SparseError::EmptyInput);
        }

        let n = matrix.nrows;
        let tolerance = tolerance.max(DEFAULT_TOLERANCE);

        let mut diagonal = Array1::<f64>::zeros(n);
        for row in 0..n {
            let start = matrix.indptr[row];
            let end = matrix.indptr[row + 1];
            for entry in start..end {
                if matrix.indices[entry] == row {
                    diagonal[row] = matrix.data[entry];
                    break;
                }
            }
            if diagonal[row].abs() <= DEFAULT_TOLERANCE {
                return Err(SparseError::SingularMatrix);
            }
        }

        let mut x = Array1::<f64>::zeros(n);
        let mut x_next = Array1::<f64>::zeros(n);

        for _ in 0..max_iterations.max(1) {
            for row in 0..n {
                let start = matrix.indptr[row];
                let end = matrix.indptr[row + 1];
                let mut off_diagonal = 0.0_f64;
                for entry in start..end {
                    let col = matrix.indices[entry];
                    if col != row {
                        off_diagonal += matrix.data[entry] * x[col];
                    }
                }
                x_next[row] = (rhs[row] - off_diagonal) / diagonal[row];
            }

            let mut delta_inf = 0.0_f64;
            for i in 0..n {
                delta_inf = delta_inf.max((x_next[i] - x[i]).abs());
                x[i] = x_next[i];
            }

            if delta_inf <= tolerance {
                return Ok(x);
            }
        }

        Err(SparseError::MaxIterationsExceeded)
    }

    /// Solve sparse linear system `A x = b` with Gauss-Seidel iteration.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions, singular diagonals, or non-convergence.
    pub fn gauss_seidel_solve(
        matrix: &CsrMatrix,
        rhs: &Array1<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Array1<f64>, SparseError> {
        if matrix.nrows != matrix.ncols {
            return Err(SparseError::DimensionMismatch);
        }
        if rhs.len() != matrix.nrows {
            return Err(SparseError::DimensionMismatch);
        }
        if rhs.is_empty() {
            return Err(SparseError::EmptyInput);
        }

        let n = matrix.nrows;
        let tolerance = tolerance.max(DEFAULT_TOLERANCE);
        let mut x = Array1::<f64>::zeros(n);

        for _ in 0..max_iterations.max(1) {
            let previous = x.clone();
            for row in 0..n {
                let start = matrix.indptr[row];
                let end = matrix.indptr[row + 1];

                let mut diagonal = 0.0_f64;
                let mut sum = 0.0_f64;
                for entry in start..end {
                    let col = matrix.indices[entry];
                    let value = matrix.data[entry];
                    if col == row {
                        diagonal = value;
                    } else {
                        sum += value * x[col];
                    }
                }

                if diagonal.abs() <= DEFAULT_TOLERANCE {
                    return Err(SparseError::SingularMatrix);
                }

                x[row] = (rhs[row] - sum) / diagonal;
            }

            let mut delta_inf = 0.0_f64;
            for i in 0..n {
                delta_inf = delta_inf.max((x[i] - previous[i]).abs());
            }
            if delta_inf <= tolerance {
                return Ok(x);
            }
        }

        Err(SparseError::MaxIterationsExceeded)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::{CooMatrix, CsrMatrix, SparseError, ndarray_sparse};

    fn toy_matrix() -> CsrMatrix {
        // [4 1 0]
        // [1 3 1]
        // [0 1 2]
        CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap()
    }

    #[test]
    fn matvec_matches_expected() {
        let matrix = toy_matrix();
        let vector = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = ndarray_sparse::matvec(&matrix, &vector).unwrap();
        assert!((y[0] - 6.0).abs() < 1e-12);
        assert!((y[1] - 10.0).abs() < 1e-12);
        assert!((y[2] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn jacobi_solves_diagonally_dominant_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0, 3.0]);
        let solution = ndarray_sparse::jacobi_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reconstructed = ndarray_sparse::matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_invalid_structure() {
        let result = CsrMatrix::new(2, 2, vec![0, 1], vec![0], vec![1.0]);
        assert!(matches!(result, Err(SparseError::InvalidStructure)));
    }

    #[test]
    fn coo_to_csr_roundtrip_matvec() {
        let coo = CooMatrix::new(3, 3, vec![0, 0, 1, 1, 1, 2, 2], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let csr = coo.to_csr().unwrap();
        let vector = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = ndarray_sparse::matvec(&csr, &vector).unwrap();
        assert!((y[0] - 6.0).abs() < 1e-12);
        assert!((y[1] - 10.0).abs() < 1e-12);
        assert!((y[2] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn gauss_seidel_solves_diagonally_dominant_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0, 3.0]);
        let solution = ndarray_sparse::gauss_seidel_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reconstructed = ndarray_sparse::matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }
}
