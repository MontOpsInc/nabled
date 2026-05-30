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

/// Algebraic residual `P - A' P A + A' P B (R + B' P B)⁻¹ B' P A - Q` for S9 validation.
pub fn dare_residual<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    q: &Array2<T>,
    r: &Array2<T>,
    p: &Array2<T>,
) -> Result<Array2<T>, ControlError> {
    dare_residual_view(&a.view(), &b.view(), &q.view(), &r.view(), &p.view())
}

/// Algebraic DARE residual from matrix views.
pub fn dare_residual_view<T: LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
    p: &ArrayView2<'_, T>,
) -> Result<Array2<T>, ControlError> {
    if a.is_empty() {
        return Err(ControlError::EmptyMatrix);
    }
    let n = a.nrows();
    if a.ncols() != n
        || q.dim() != (n, n)
        || p.dim() != (n, n)
        || b.nrows() != n
        || r.nrows() != b.ncols()
        || r.ncols() != b.ncols()
    {
        return Err(ControlError::DimensionMismatch);
    }
    let bpb = b.t().dot(&p.dot(b)) + r;
    let bpb_inv = lu::inverse(&bpb).map_err(|_| ControlError::SingularSystem)?;
    let gain_term = a.t().dot(&p.dot(b)).dot(&bpb_inv).dot(&b.t()).dot(p).dot(a);
    Ok(p - &a.t().dot(&p.dot(a)) + gain_term - q)
}

/// Frobenius norm of the DARE algebraic residual.
pub fn dare_residual_norm<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    q: &Array2<T>,
    r: &Array2<T>,
    p: &Array2<T>,
) -> Result<f64, ControlError> {
    dare_residual_norm_view(&a.view(), &b.view(), &q.view(), &r.view(), &p.view())
}

/// Frobenius norm of the DARE algebraic residual from matrix views.
pub fn dare_residual_norm_view<T: LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
    p: &ArrayView2<'_, T>,
) -> Result<f64, ControlError> {
    let residual = dare_residual_view(a, b, q, r, p)?;
    Ok(residual.mapv(|value| (value * value).to_f64().unwrap_or(0.0)).sum().sqrt())
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

    #[test]
    fn dare_residual_near_zero_for_solution() {
        let dt = 0.1_f64;
        let a = arr2(&[[1.0, dt], [0.0, 1.0]]);
        let b = arr2(&[[0.0], [dt]]);
        let q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let r = arr2(&[[1.0]]);
        let p = dare_solve(&a, &b, &q, &r).unwrap();
        let norm = dare_residual_norm(&a, &b, &q, &r, &p).unwrap();
        assert!(norm < 1e-8, "residual norm {norm}");
    }

    #[test]
    fn dare_rejects_empty_and_mismatched_dimensions() {
        let empty = arr2(&[[]]);
        let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let b = arr2(&[[0.0], [1.0]]);
        let q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let r = arr2(&[[1.0]]);
        assert!(matches!(dare_solve(&empty, &b, &q, &r), Err(ControlError::EmptyMatrix)));
        assert!(matches!(
            dare_solve(&a, &b, &arr2(&[[1.0]]), &r),
            Err(ControlError::DimensionMismatch)
        ));
        let p = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert!(matches!(
            dare_residual(&a, &b, &q, &r, &arr2(&[[1.0]])),
            Err(ControlError::DimensionMismatch)
        ));
        assert!(dare_residual_norm(&a, &b, &q, &r, &p).unwrap() >= 0.0);
    }
}
