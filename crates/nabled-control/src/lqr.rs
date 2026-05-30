//! Discrete LQR gain computation.

use nabled_linalg::lu::LuProviderScalar;
use ndarray::{Array2, ArrayView2};

use crate::ControlError;
use crate::dare::dare_solve_view;

/// LQR solution containing gain and Riccati matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct LqrResult<T> {
    pub gain:    Array2<T>,
    pub riccati: Array2<T>,
}

/// Compute discrete-time LQR gain.
pub fn discrete_lqr<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    q: &Array2<T>,
    r: &Array2<T>,
) -> Result<LqrResult<T>, ControlError> {
    discrete_lqr_view(&a.view(), &b.view(), &q.view(), &r.view())
}

/// Compute discrete LQR from views.
pub fn discrete_lqr_view<T: LuProviderScalar>(
    a: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
) -> Result<LqrResult<T>, ControlError> {
    let p = dare_solve_view(a, b, q, r)?;
    let bpb = b.t().dot(&p.dot(b)) + r;
    let bpb_inv = nabled_linalg::lu::inverse(&bpb).map_err(|_| ControlError::SingularSystem)?;
    let gain = bpb_inv.dot(&b.t()).dot(&p).dot(a);
    Ok(LqrResult { gain, riccati: p })
}

/// Compute LQR into caller-provided gain and riccati buffers.
pub fn discrete_lqr_into<T: LuProviderScalar>(
    a: &Array2<T>,
    b: &Array2<T>,
    q: &Array2<T>,
    r: &Array2<T>,
    gain: &mut Array2<T>,
    riccati: &mut Array2<T>,
) -> Result<(), ControlError> {
    let result = discrete_lqr(a, b, q, r)?;
    if gain.dim() != result.gain.dim() || riccati.dim() != result.riccati.dim() {
        return Err(ControlError::DimensionMismatch);
    }
    gain.assign(&result.gain);
    riccati.assign(&result.riccati);
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn lqr_stabilizes_double_integrator() {
        let dt = 0.1_f64;
        let a = arr2(&[[1.0, dt], [0.0, 1.0]]);
        let b = arr2(&[[0.0], [dt]]);
        let q = arr2(&[[10.0, 0.0], [0.0, 1.0]]);
        let r = arr2(&[[0.1]]);
        let result = discrete_lqr(&a, &b, &q, &r).unwrap();
        let closed = &a - &b.dot(&result.gain);
        let eig = nabled_linalg::eigen::nonsymmetric(&closed).unwrap();
        for lambda in &eig.eigenvalues {
            let mag = (lambda.re * lambda.re + lambda.im * lambda.im).sqrt();
            assert!(mag < 1.0, "eigenvalue magnitude {mag} >= 1");
        }
    }

    #[test]
    fn lqr_view_and_into_match_allocating() {
        let a = arr2(&[[1.0_f64, 0.1], [0.0, 1.0]]);
        let b = arr2(&[[0.0], [0.1]]);
        let q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let r = arr2(&[[0.05]]);
        let result = discrete_lqr_view(&a.view(), &b.view(), &q.view(), &r.view()).unwrap();
        let mut gain = arr2(&[[0.0, 0.0]]);
        let mut riccati = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
        discrete_lqr_into(&a, &b, &q, &r, &mut gain, &mut riccati).unwrap();
        assert_eq!(result.gain, gain);
        assert_eq!(result.riccati, riccati);
    }

    #[test]
    fn lqr_into_rejects_buffer_dimension_mismatch() {
        let a = arr2(&[[1.0_f64, 0.1], [0.0, 1.0]]);
        let b = arr2(&[[0.0], [0.1]]);
        let q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let r = arr2(&[[0.05]]);
        let mut gain = arr2(&[[0.0, 0.0]]);
        let mut riccati = arr2(&[[0.0]]);
        assert!(matches!(
            discrete_lqr_into(&a, &b, &q, &r, &mut gain, &mut riccati),
            Err(ControlError::DimensionMismatch)
        ));
    }
}
