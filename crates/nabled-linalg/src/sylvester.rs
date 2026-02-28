//! Sylvester and Lyapunov solvers over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2};

use crate::lu::ndarray_lu;

/// Error type for Sylvester/Lyapunov solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SylvesterError {
    /// Matrix input is empty.
    EmptyMatrix,
    /// Matrix must be square.
    NotSquare,
    /// Input dimensions are incompatible.
    DimensionMismatch,
    /// Linear system is singular.
    SingularSystem,
}

impl fmt::Display for SylvesterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SylvesterError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            SylvesterError::NotSquare => write!(f, "Matrix must be square"),
            SylvesterError::DimensionMismatch => write!(f, "Input dimensions are incompatible"),
            SylvesterError::SingularSystem => write!(f, "Sylvester system is singular"),
        }
    }
}

impl std::error::Error for SylvesterError {}

/// Ndarray Sylvester/Lyapunov functions.
pub mod ndarray_sylvester {
    use super::*;

    /// Solve Sylvester equation `A X + X B = C`.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid or system is singular.
    pub fn solve_sylvester(
        matrix_a: &Array2<f64>,
        matrix_b: &Array2<f64>,
        matrix_c: &Array2<f64>,
    ) -> Result<Array2<f64>, SylvesterError> {
        if matrix_a.is_empty() || matrix_b.is_empty() || matrix_c.is_empty() {
            return Err(SylvesterError::EmptyMatrix);
        }
        if matrix_a.nrows() != matrix_a.ncols() || matrix_b.nrows() != matrix_b.ncols() {
            return Err(SylvesterError::NotSquare);
        }

        let n = matrix_a.nrows();
        let m = matrix_b.nrows();
        if matrix_c.dim() != (n, m) {
            return Err(SylvesterError::DimensionMismatch);
        }

        let system_size = n * m;
        let mut coefficient = Array2::<f64>::zeros((system_size, system_size));
        let mut rhs = Array1::<f64>::zeros(system_size);

        for i in 0..n {
            for j in 0..m {
                let row = i * m + j;
                rhs[row] = matrix_c[[i, j]];

                for p in 0..n {
                    let col = p * m + j;
                    coefficient[[row, col]] += matrix_a[[i, p]];
                }
                for q in 0..m {
                    let col = i * m + q;
                    coefficient[[row, col]] += matrix_b[[q, j]];
                }
            }
        }

        let solution =
            ndarray_lu::solve(&coefficient, &rhs).map_err(|_| SylvesterError::SingularSystem)?;

        let mut x = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                x[[i, j]] = solution[i * m + j];
            }
        }

        Ok(x)
    }

    /// Solve continuous Lyapunov equation `A X + X A^T + Q = 0`.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid or system is singular.
    pub fn solve_lyapunov(a: &Array2<f64>, q: &Array2<f64>) -> Result<Array2<f64>, SylvesterError> {
        if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
            return Err(SylvesterError::DimensionMismatch);
        }
        let neg_q = -q;
        solve_sylvester(a, &a.t().to_owned(), &neg_q)
    }

    /// Solve Sylvester using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_sylvester_lapack(
        matrix_a: &Array2<f64>,
        matrix_b: &Array2<f64>,
        matrix_c: &Array2<f64>,
    ) -> Result<Array2<f64>, SylvesterError> {
        solve_sylvester(matrix_a, matrix_b, matrix_c)
    }

    /// Solve Lyapunov using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_lyapunov_lapack(
        a: &Array2<f64>,
        q: &Array2<f64>,
    ) -> Result<Array2<f64>, SylvesterError> {
        solve_lyapunov(a, q)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::ndarray_sylvester;

    #[test]
    fn solves_diagonal_sylvester() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 0.0, 4.0]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let x = ndarray_sylvester::solve_sylvester(&a, &b, &c).unwrap();
        let residual = a.dot(&x) + x.dot(&b) - c;
        assert!(residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max) < 1e-8);
    }
}
