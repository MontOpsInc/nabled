//! Sylvester and Lyapunov solvers over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView2};

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

/// Reusable workspace for Sylvester/Lyapunov solves.
#[derive(Debug, Clone, Default)]
pub struct SylvesterWorkspace {
    coefficient: Array2<f64>,
    rhs:         Array1<f64>,
    solution:    Array1<f64>,
}

impl SylvesterWorkspace {
    fn ensure_dims(&mut self, rows: usize, cols: usize) {
        let system_size = rows * cols;
        if self.coefficient.dim() == (system_size, system_size) {
            self.coefficient.fill(0.0);
        } else {
            self.coefficient = Array2::<f64>::zeros((system_size, system_size));
        }
        if self.rhs.len() == system_size {
            self.rhs.fill(0.0);
        } else {
            self.rhs = Array1::<f64>::zeros(system_size);
        }
        if self.solution.len() != system_size {
            self.solution = Array1::<f64>::zeros(system_size);
        }
    }
}

/// Ndarray Sylvester/Lyapunov functions.
pub mod ndarray_sylvester {
    use super::*;

    fn validate_sylvester_dims(
        matrix_a: &Array2<f64>,
        matrix_b: &Array2<f64>,
        matrix_c: &Array2<f64>,
    ) -> Result<(usize, usize), SylvesterError> {
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
        Ok((n, m))
    }

    /// Solve Sylvester equation `A X + X B = C`.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid or system is singular.
    pub fn solve_sylvester(
        matrix_a: &Array2<f64>,
        matrix_b: &Array2<f64>,
        matrix_c: &Array2<f64>,
    ) -> Result<Array2<f64>, SylvesterError> {
        let (n, m) = validate_sylvester_dims(matrix_a, matrix_b, matrix_c)?;
        let mut workspace = SylvesterWorkspace::default();
        let mut output = Array2::<f64>::zeros((n, m));
        solve_sylvester_with_workspace_into(
            matrix_a,
            matrix_b,
            matrix_c,
            &mut output,
            &mut workspace,
        )?;
        Ok(output)
    }

    /// Solve Sylvester equation `A X + X B = C` from matrix views.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid or system is singular.
    pub fn solve_sylvester_view(
        matrix_a: &ArrayView2<'_, f64>,
        matrix_b: &ArrayView2<'_, f64>,
        matrix_c: &ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, SylvesterError> {
        solve_sylvester(&matrix_a.to_owned(), &matrix_b.to_owned(), &matrix_c.to_owned())
    }

    /// Solve Sylvester equation `A X + X B = C` into `output`.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid or system is singular.
    pub fn solve_sylvester_into(
        matrix_a: &Array2<f64>,
        matrix_b: &Array2<f64>,
        matrix_c: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> Result<(), SylvesterError> {
        let mut workspace = SylvesterWorkspace::default();
        solve_sylvester_with_workspace_into(matrix_a, matrix_b, matrix_c, output, &mut workspace)
    }

    /// Solve Sylvester equation `A X + X B = C` into `output` with reusable `workspace`.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
    pub fn solve_sylvester_with_workspace_into(
        matrix_a: &Array2<f64>,
        matrix_b: &Array2<f64>,
        matrix_c: &Array2<f64>,
        output: &mut Array2<f64>,
        workspace: &mut SylvesterWorkspace,
    ) -> Result<(), SylvesterError> {
        let (n, m) = validate_sylvester_dims(matrix_a, matrix_b, matrix_c)?;
        if output.dim() != (n, m) {
            return Err(SylvesterError::DimensionMismatch);
        }

        workspace.ensure_dims(n, m);

        for i in 0..n {
            for j in 0..m {
                let row = i * m + j;
                workspace.rhs[row] = matrix_c[[i, j]];

                for p in 0..n {
                    let col = p * m + j;
                    workspace.coefficient[[row, col]] += matrix_a[[i, p]];
                }
                for q in 0..m {
                    let col = i * m + q;
                    workspace.coefficient[[row, col]] += matrix_b[[q, j]];
                }
            }
        }

        workspace.solution = ndarray_lu::solve(&workspace.coefficient, &workspace.rhs)
            .map_err(|_| SylvesterError::SingularSystem)?;

        for i in 0..n {
            for j in 0..m {
                output[[i, j]] = workspace.solution[i * m + j];
            }
        }
        Ok(())
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

    /// Solve continuous Lyapunov equation `A X + X A^T + Q = 0` from matrix views.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid or system is singular.
    pub fn solve_lyapunov_view(
        a: &ArrayView2<'_, f64>,
        q: &ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, SylvesterError> {
        solve_lyapunov(&a.to_owned(), &q.to_owned())
    }

    /// Solve continuous Lyapunov equation into `output`.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
    pub fn solve_lyapunov_into(
        a: &Array2<f64>,
        q: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> Result<(), SylvesterError> {
        if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
            return Err(SylvesterError::DimensionMismatch);
        }
        let neg_q = -q;
        solve_sylvester_into(a, &a.t().to_owned(), &neg_q, output)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{SylvesterWorkspace, ndarray_sylvester};

    #[test]
    fn solves_diagonal_sylvester() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 0.0, 4.0]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let x = ndarray_sylvester::solve_sylvester(&a, &b, &c).unwrap();
        let residual = a.dot(&x) + x.dot(&b) - c;
        assert!(residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max) < 1e-8);
    }

    #[test]
    fn solves_diagonal_sylvester_into_with_workspace() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![4.0, 0.0, 0.0, 5.0]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 6.0, 4.0]).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        let mut workspace = SylvesterWorkspace::default();
        ndarray_sylvester::solve_sylvester_with_workspace_into(
            &a,
            &b,
            &c,
            &mut output,
            &mut workspace,
        )
        .unwrap();

        let residual = a.dot(&output) + output.dot(&b) - c;
        assert!(residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max) < 1e-8);
    }

    #[test]
    fn solves_lyapunov_equation() {
        let a = Array2::from_shape_vec((2, 2), vec![-2.0, 0.0, 0.0, -3.0]).unwrap();
        let q = Array2::eye(2);
        let x = ndarray_sylvester::solve_lyapunov(&a, &q).unwrap();
        let residual = a.dot(&x) + x.dot(&a.t().to_owned()) + q;
        assert!(residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max) < 1e-8);
    }

    #[test]
    fn lyapunov_into_rejects_bad_output_shape() {
        let a = Array2::eye(2);
        let q = Array2::eye(2);
        let mut output = Array2::<f64>::zeros((1, 1));
        let result = ndarray_sylvester::solve_lyapunov_into(&a, &q, &mut output);
        assert!(matches!(result, Err(super::SylvesterError::DimensionMismatch)));
    }

    #[test]
    fn view_variants_match_owned() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 4.0]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let owned = ndarray_sylvester::solve_sylvester(&a, &b, &c).unwrap();
        let viewed =
            ndarray_sylvester::solve_sylvester_view(&a.view(), &b.view(), &c.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned[[i, j]] - viewed[[i, j]]).abs() < 1e-12);
            }
        }

        let q = Array2::eye(2);
        let lyapunov_owned = ndarray_sylvester::solve_lyapunov(&a, &q).unwrap();
        let lyapunov_viewed = ndarray_sylvester::solve_lyapunov_view(&a.view(), &q.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((lyapunov_owned[[i, j]] - lyapunov_viewed[[i, j]]).abs() < 1e-12);
            }
        }
    }
}
