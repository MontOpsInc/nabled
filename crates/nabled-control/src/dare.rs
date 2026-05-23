//! Discrete algebraic Riccati equation (DARE) solver.

use nabled_linalg::lu::{self, LuProviderScalar};
use ndarray::{Array2, ArrayView2};

use crate::ControlError;

const MAX_ITERATIONS: usize = 500;

/// Solve discrete-time DARE via fixed-point iteration.
pub fn dare_solve<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    q: &Array2<T>,
    r: &Array2<T>,
) -> Result<Array2<T>, ControlError> {
    dare_solve_view(&a.view(), &b.view(), &q.view(), &r.view())
}

/// Solve DARE from matrix views.
pub fn dare_solve_view<T: LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
) -> Result<Array2<T>, ControlError> {
    if a.is_empty() {
        return Err(ControlError::EmptyMatrix);
    }
    let n = a.nrows();
    if a.ncols() != n || q.dim() != (n, n) {
        return Err(ControlError::DimensionMismatch);
    }
    let mut p = q.to_owned();
    for _ in 0..MAX_ITERATIONS {
        let bpb = b.t().dot(&p.dot(b)) + r;
        let bpb_inv = lu::inverse(&bpb).map_err(|_| ControlError::SingularSystem)?;
        let gain_term = a.t().dot(&p.dot(b)).dot(&bpb_inv).dot(&b.t()).dot(&p).dot(a);
        let p_next = a.t().dot(&p.dot(a)) - gain_term + q;
        let diff = (&p_next - &p).mapv(|v| (v * v).to_f64().unwrap_or(0.0)).sum();
        p = p_next;
        if diff.sqrt() < 1e-10 {
            return Ok(p);
        }
    }
    Err(ControlError::ConvergenceFailed)
}

/// Solve DARE into caller buffer.
pub fn dare_solve_into<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    q: &Array2<T>,
    r: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), ControlError> {
    let p = dare_solve(a, b, q, r)?;
    if output.dim() != p.dim() {
        return Err(ControlError::DimensionMismatch);
    }
    output.assign(&p);
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn dare_double_integrator_converges() {
        let dt = 0.1_f64;
        let a = arr2(&[[1.0, dt], [0.0, 1.0]]);
        let b = arr2(&[[0.0], [dt]]);
        let q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let r = arr2(&[[1.0]]);
        let p = dare_solve(&a, &b, &q, &r).unwrap();
        assert!(p[[0, 0]] > 0.0);
    }
}
