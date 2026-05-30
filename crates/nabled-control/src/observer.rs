//! Luenberger observer gain via dual pole placement.

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu::LuProviderScalar;
use ndarray::{Array2, ArrayView2};

use crate::ControlError;
use crate::pole::place_poles_view;

fn validate_observer_inputs<T: NabledReal>(
    a: &ArrayView2<'_, T>,
    c: &ArrayView2<'_, T>,
    pole_count: usize,
) -> Result<usize, ControlError> {
    if a.is_empty() {
        return Err(ControlError::EmptyMatrix);
    }
    let n = a.nrows();
    if a.ncols() != n || c.ncols() != n || c.nrows() != 1 || pole_count != n {
        return Err(ControlError::DimensionMismatch);
    }
    Ok(n)
}

/// Compute Luenberger observer gain `L` for SISO `(A, C)` via dual pole placement.
pub fn luenberger_gain<T: NabledReal + LuProviderScalar>(
    a: &Array2<T>,
    c: &Array2<T>,
    poles: &[T],
) -> Result<Array2<T>, ControlError> {
    luenberger_gain_view(&a.view(), &c.view(), poles)
}

/// Compute Luenberger observer gain from matrix views.
pub fn luenberger_gain_view<T: NabledReal + LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    c: &ArrayView2<'_, T>,
    poles: &[T],
) -> Result<Array2<T>, ControlError> {
    let _n = validate_observer_inputs(a, c, poles.len())?;
    let dual_gain = place_poles_view(&a.t(), &c.t(), poles)?;
    Ok(dual_gain.t().to_owned())
}

/// Compute Luenberger observer gain into caller-provided buffer.
pub fn luenberger_gain_into<T: NabledReal + LuProviderScalar>(
    a: &Array2<T>,
    c: &Array2<T>,
    poles: &[T],
    output: &mut Array2<T>,
) -> Result<(), ControlError> {
    let gain = luenberger_gain(a, c, poles)?;
    if output.dim() != gain.dim() {
        return Err(ControlError::DimensionMismatch);
    }
    output.assign(&gain);
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_linalg::eigen;
    use ndarray::arr2;
    use num_complex::Complex;

    use super::*;

    fn eigenvalue_near(
        requested: f64,
        values: impl IntoIterator<Item = Complex<f64>>,
        tol: f64,
    ) -> bool {
        values
            .into_iter()
            .any(|lambda| (lambda.re - requested).abs() < tol && lambda.im.abs() < tol)
    }

    #[test]
    fn luenberger_places_observer_poles() {
        let a = arr2(&[[0.0, 1.0], [0.0, 0.0]]);
        let c = arr2(&[[1.0, 0.0]]);
        let poles = [-4.0, -5.0];
        let l = luenberger_gain(&a, &c, &poles).expect("observer gain");
        assert_eq!(l.dim(), (2, 1));
        let closed = &a - &l.dot(&c);
        let eig = eigen::nonsymmetric(&closed).expect("eigen");
        for &pole in &poles {
            assert!(eigenvalue_near(pole, eig.eigenvalues.iter().copied(), 1e-6));
        }
    }

    #[test]
    fn luenberger_gain_into_matches_allocating() {
        let a = arr2(&[[0.0, 1.0], [0.0, 0.0]]);
        let c = arr2(&[[1.0, 0.0]]);
        let poles = [-2.0, -3.0];
        let gain = luenberger_gain_view(&a.view(), &c.view(), &poles).expect("view");
        let mut into_buf = arr2(&[[0.0], [0.0]]);
        luenberger_gain_into(&a, &c, &poles, &mut into_buf).expect("into");
        assert_relative_eq!(gain, into_buf, epsilon = 1e-12);
    }
}
