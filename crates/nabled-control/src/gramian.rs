//! Controllability and observability gramians (stubs).

use nabled_linalg::lu::LuProviderScalar;
use nabled_linalg::sylvester;
use ndarray::{Array2, ArrayView2};

use crate::ControlError;

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
