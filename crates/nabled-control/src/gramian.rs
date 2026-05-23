//! Controllability and observability gramians.

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu::LuProviderScalar;
use nabled_linalg::sylvester;
use ndarray::{Array2, ArrayView2};

use crate::ControlError;

const MAX_STEIN_ITERATIONS: usize = 500;

fn validate_state_input<T: NabledReal>(
    a: &ArrayView2<'_, T>,
    b_rows: usize,
    b_cols: usize,
) -> Result<usize, ControlError> {
    if a.is_empty() {
        return Err(ControlError::EmptyMatrix);
    }
    let n = a.nrows();
    if a.ncols() != n || b_rows != n {
        return Err(ControlError::DimensionMismatch);
    }
    if b_cols == 0 {
        return Err(ControlError::DimensionMismatch);
    }
    Ok(n)
}

fn validate_output_input<T: NabledReal>(
    a: &ArrayView2<'_, T>,
    c: &ArrayView2<'_, T>,
) -> Result<usize, ControlError> {
    if a.is_empty() {
        return Err(ControlError::EmptyMatrix);
    }
    let n = a.nrows();
    if a.ncols() != n || c.ncols() != n || c.nrows() == 0 {
        return Err(ControlError::DimensionMismatch);
    }
    Ok(n)
}

fn stein_solve<T: NabledReal + LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
) -> Result<Array2<T>, ControlError> {
    let mut w = q.to_owned();
    for _ in 0..MAX_STEIN_ITERATIONS {
        let next = a.dot(&w).dot(&a.t()) + q;
        let diff = (&next - &w)
            .mapv(|value| (value * value).to_f64().unwrap_or(0.0))
            .sum()
            .sqrt();
        w = next;
        if diff < 1e-12 {
            return Ok(w);
        }
    }
    Err(ControlError::ConvergenceFailed)
}


#[cfg(test)]
fn stein_residual_norm<T: NabledReal + LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    w: &ArrayView2<'_, T>,
    rhs: &ArrayView2<'_, T>,
) -> f64 {
    let residual = w - &a.dot(w).dot(&a.t()) - rhs;
    residual.mapv(|value| (value * value).to_f64().unwrap_or(0.0)).sum().sqrt()
}

pub fn controllability_gramian<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
) -> Result<Array2<T>, ControlError> {
    controllability_gramian_view(&a.view(), &b.view())
}

pub fn controllability_gramian_view<T: LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
) -> Result<Array2<T>, ControlError> {
    let _n = validate_state_input(a, b.nrows(), b.ncols())?;
    let bb = b.dot(&b.t());
    sylvester::solve_lyapunov(&a.to_owned(), &bb).map_err(|_| ControlError::SingularSystem)
}

pub fn controllability_gramian_into<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), ControlError> {
    let w = controllability_gramian(a, b)?;
    if output.dim() != w.dim() {
        return Err(ControlError::DimensionMismatch);
    }
    output.assign(&w);
    Ok(())
}

pub fn observability_gramian<T: LuProviderScalar>(
    a: &Array2<T>,
    c: &Array2<T>,
) -> Result<Array2<T>, ControlError> {
    observability_gramian_view(&a.view(), &c.view())
}

pub fn observability_gramian_view<T: LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    c: &ArrayView2<'_, T>,
) -> Result<Array2<T>, ControlError> {
    let _n = validate_output_input(a, c)?;
    let cc = c.t().dot(c);
    sylvester::solve_lyapunov(&a.t().to_owned(), &cc).map_err(|_| ControlError::SingularSystem)
}

pub fn observability_gramian_into<T: LuProviderScalar>(
    a: &Array2<T>,
    c: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), ControlError> {
    let w = observability_gramian(a, c)?;
    if output.dim() != w.dim() {
        return Err(ControlError::DimensionMismatch);
    }
    output.assign(&w);
    Ok(())
}

/// Discrete-time controllability gramian solving `W - A W A' = B B'`.
pub fn discrete_controllability_gramian<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
) -> Result<Array2<T>, ControlError> {
    discrete_controllability_gramian_view(&a.view(), &b.view())
}

/// Discrete-time controllability gramian from matrix views.
pub fn discrete_controllability_gramian_view<T: LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
) -> Result<Array2<T>, ControlError> {
    let _n = validate_state_input(a, b.nrows(), b.ncols())?;
    let bb = b.dot(&b.t());
    stein_solve(a, &bb.view())
}

/// Discrete-time observability gramian solving `W - A' W A = C' C`.
pub fn discrete_observability_gramian<T: LuProviderScalar>(
    a: &Array2<T>,
    c: &Array2<T>,
) -> Result<Array2<T>, ControlError> {
    discrete_observability_gramian_view(&a.view(), &c.view())
}

/// Discrete-time observability gramian from matrix views.
pub fn discrete_observability_gramian_view<T: LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    c: &ArrayView2<'_, T>,
) -> Result<Array2<T>, ControlError> {
    let _n = validate_output_input(a, c)?;
    let cc = c.t().dot(c);
    stein_solve(&a.t(), &cc.view())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::arr2;

    use super::*;

    #[test]
    fn controllability_gramian_satisfies_lyapunov() {
        let a = arr2(&[[-1.0, 0.0], [0.0, -2.0]]);
        let b = arr2(&[[1.0], [0.0]]);
        let w = controllability_gramian(&a, &b).expect("gramian");
        let bb = b.dot(&b.t());
        let residual = a.dot(&w) + w.dot(&a.t()) + bb;
        let norm = residual.mapv(f64::from).mapv(|v| v * v).sum().sqrt();
        assert!(norm < 1e-8, "residual {norm}");
    }

    #[test]
    fn observability_gramian_view_matches_allocating() {
        let a = arr2(&[[-1.0, 0.0], [0.0, -2.0]]);
        let c = arr2(&[[1.0, 0.0]]);
        let w = observability_gramian_view(&a.view(), &c.view()).expect("view");
        let w_alloc = observability_gramian(&a, &c).expect("alloc");
        assert_relative_eq!(w, w_alloc, epsilon = 1e-12);
    }

    #[test]
    fn discrete_controllability_gramian_satisfies_stein() {
        let a = arr2(&[[0.8, 0.1], [0.0, 0.7]]);
        let b = arr2(&[[1.0], [0.0]]);
        let w = discrete_controllability_gramian(&a, &b).expect("discrete gramian");
        let bb = b.dot(&b.t());
        let residual = stein_residual_norm(&a.view(), &w.view(), &bb.view());
        assert!(residual < 1e-8, "residual {residual}");
    }

    #[test]
    fn discrete_observability_gramian_satisfies_stein() {
        let a = arr2(&[[0.8, 0.1], [0.0, 0.7]]);
        let c = arr2(&[[1.0, 0.0]]);
        let w = discrete_observability_gramian(&a, &c).expect("discrete gramian");
        let cc = c.t().dot(&c);
        let residual = stein_residual_norm(&a.t(), &w.view(), &cc.view());
        assert!(residual < 1e-8, "residual {residual}");
    }

    #[test]
    fn gramian_rejects_dimension_mismatch() {
        let a = arr2(&[[-1.0, 0.0], [0.0, -2.0]]);
        let b = arr2(&[[1.0, 0.0]]);
        assert!(matches!(
            controllability_gramian(&a, &b),
            Err(ControlError::DimensionMismatch)
        ));
    }
}
