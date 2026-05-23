//! SISO pole placement via Ackermann's formula.

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu::{self, LuProviderScalar};
use ndarray::{Array1, Array2, ArrayView2};

use crate::ControlError;

fn validate_siso_state_input<T: NabledReal>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
    pole_count: usize,
) -> Result<usize, ControlError> {
    if a.is_empty() {
        return Err(ControlError::EmptyMatrix);
    }
    let n = a.nrows();
    if a.ncols() != n || b.nrows() != n || b.ncols() != 1 || pole_count != n {
        return Err(ControlError::DimensionMismatch);
    }
    Ok(n)
}

fn controllability_matrix<T: NabledReal + LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
) -> Result<Array2<T>, ControlError> {
    let n = validate_siso_state_input(a, b, a.nrows())?;
    let mut ctl = Array2::<T>::zeros((n, n));
    let mut power = b.to_owned();
    for col in 0..n {
        for row in 0..n {
            ctl[[row, col]] = power[[row, 0]];
        }
        if col + 1 < n {
            power = a.dot(&power);
        }
    }
    Ok(ctl)
}

/// Monic polynomial coefficients `[1, c_{n-1}, ..., c_0]` from real pole locations.
fn monic_coefficients_from_poles<T: NabledReal>(poles: &[T]) -> Vec<T> {
    let mut coeffs =
        vec![num_traits::FromPrimitive::from_f64(1.0).unwrap_or_else(num_traits::One::one)];
    for &pole in poles {
        let mut next = vec![num_traits::Zero::zero(); coeffs.len() + 1];
        for (index, &coef) in coeffs.iter().enumerate() {
            next[index] += coef;
            next[index + 1] -= coef * pole;
        }
        coeffs = next;
    }
    coeffs
}

fn matrix_monic_polynomial<T: NabledReal + LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    coeffs: &[T],
) -> Array2<T> {
    let n = a.nrows();
    let degree = coeffs.len() - 1;
    let mut powers = Vec::with_capacity(degree + 1);
    let mut power = Array2::<T>::eye(n);
    for _ in 0..=degree {
        powers.push(power.clone());
        power = a.dot(&power);
    }
    let mut result = Array2::<T>::zeros((n, n));
    for (index, &coef) in coeffs.iter().enumerate() {
        let exponent = degree - index;
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] += coef * powers[exponent][[i, j]];
            }
        }
    }
    result
}

fn ackermann_gain<T: NabledReal + LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
    poles: &[T],
) -> Result<Array2<T>, ControlError> {
    let _n = validate_siso_state_input(a, b, poles.len())?;
    let n = a.nrows();
    let ctl = controllability_matrix(a, b)?;
    let det = lu::determinant(&ctl).map_err(|_| ControlError::SingularSystem)?;
    if num_traits::Float::abs(det)
        <= num_traits::FromPrimitive::from_f64(1e-12).unwrap_or(T::epsilon())
    {
        return Err(ControlError::InvalidInput("system is not controllable".to_string()));
    }
    let ctl_inv = lu::inverse(&ctl).map_err(|_| ControlError::SingularSystem)?;
    let coeffs = monic_coefficients_from_poles(poles);
    let alpha_a = matrix_monic_polynomial(a, &coeffs);
    let mut selector = Array1::<T>::zeros(n);
    selector[n - 1] = num_traits::FromPrimitive::from_f64(1.0).unwrap_or_else(num_traits::One::one);
    let gain_row = selector.dot(&ctl_inv.dot(&alpha_a));
    Ok(Array2::from_shape_vec((1, n), gain_row.to_vec()).expect("valid gain shape"))
}

/// Place closed-loop poles for SISO `(A, B)` via Ackermann's formula.
pub fn place_poles<T: NabledReal + LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    poles: &[T],
) -> Result<Array2<T>, ControlError> {
    place_poles_view(&a.view(), &b.view(), poles)
}

/// Place closed-loop poles from matrix views.
pub fn place_poles_view<T: NabledReal + LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
    poles: &[T],
) -> Result<Array2<T>, ControlError> {
    ackermann_gain(a, b, poles)
}

/// Place closed-loop poles into caller-provided buffer.
pub fn place_poles_into<T: NabledReal + LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    poles: &[T],
    output: &mut Array2<T>,
) -> Result<(), ControlError> {
    let gain = place_poles(a, b, poles)?;
    if output.dim() != gain.dim() {
        return Err(ControlError::DimensionMismatch);
    }
    output.assign(&gain);
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::arr2;
    use num_complex::Complex;

    use nabled_linalg::eigen;

    use super::*;

    fn eigenvalue_near(
        requested: f64,
        values: impl IntoIterator<Item = Complex<f64>>,
        tol: f64,
    ) -> bool {
        values.into_iter().any(|lambda| {
            let re_diff = (lambda.re - requested).abs();
            let im = lambda.im.abs();
            re_diff < tol && im < tol
        })
    }

    #[test]
    fn place_poles_double_integrator() {
        let a = arr2(&[[0.0, 1.0], [0.0, 0.0]]);
        let b = arr2(&[[0.0], [1.0]]);
        let poles = [-1.0, -2.0];
        let k = place_poles(&a, &b, &poles).expect("gain");
        assert_eq!(k.dim(), (1, 2));
        let closed = &a - &b.dot(&k);
        let eig = eigen::nonsymmetric(&closed).expect("eigen");
        for &pole in &poles {
            assert!(
                eigenvalue_near(pole, eig.eigenvalues.iter().copied(), 1e-6),
                "missing pole {pole}"
            );
        }
    }

    #[test]
    fn place_poles_view_into_matches_allocating() {
        let a = arr2(&[[0.0, 1.0], [0.0, 0.0]]);
        let b = arr2(&[[0.0], [1.0]]);
        let poles = [-3.0, -4.0];
        let gain = place_poles_view(&a.view(), &b.view(), &poles).expect("view");
        let mut into_buf = arr2(&[[0.0, 0.0]]);
        place_poles_into(&a, &b, &poles, &mut into_buf).expect("into");
        assert_relative_eq!(gain, into_buf, epsilon = 1e-12);
    }

    #[test]
    fn place_poles_rejects_uncontrollable() {
        let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
        let b = arr2(&[[0.0], [1.0]]);
        let poles = [-1.0, -2.0];
        let err = place_poles(&a, &b, &poles).unwrap_err();
        assert!(matches!(err, ControlError::InvalidInput(_) | ControlError::SingularSystem));
    }
}
