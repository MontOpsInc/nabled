//! Twist cross products and adjoint operations.

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, ArrayView1, Axis};

use crate::geometry::so3;

/// Motion cross product for 6-vectors `[omega; v]`.
#[must_use]
pub fn motion_cross<T: NabledReal>(a: &ArrayView1<'_, T>, b: &ArrayView1<'_, T>) -> Array1<T> {
    let omega_a = a.slice(ndarray::s![0..3]);
    let v_a = a.slice(ndarray::s![3..6]);
    let omega_b = b.slice(ndarray::s![0..3]);
    let v_b = b.slice(ndarray::s![3..6]);
    let omega_cross = cross3(&omega_a, &omega_b);
    let v_part = cross3(&omega_a, &v_b) + cross3(&v_a, &omega_b);
    ndarray::concatenate![Axis(0), omega_cross, v_part]
}

/// Force cross product for 6-vectors `[tau; f]`.
#[must_use]
pub fn force_cross<T: NabledReal>(a: &ArrayView1<'_, T>, b: &ArrayView1<'_, T>) -> Array1<T> {
    let tau_a = a.slice(ndarray::s![0..3]);
    let f_a = a.slice(ndarray::s![3..6]);
    let tau_b = b.slice(ndarray::s![0..3]);
    let f_b = b.slice(ndarray::s![3..6]);
    let tau_part = cross3(&tau_a, &tau_b) + cross3(&f_a, &f_b);
    let f_part = cross3(&tau_a, &f_b) + cross3(&f_a, &tau_b);
    ndarray::concatenate![Axis(0), tau_part, f_part]
}

/// Adjoint action on motion twist.
#[must_use]
pub fn adjoint_motion<T: NabledReal>(
    adjoint: &ndarray::Array2<T>,
    twist: &ArrayView1<'_, T>,
) -> Array1<T> {
    adjoint.dot(twist)
}

/// Adjoint action on force wrench.
#[must_use]
pub fn adjoint_force<T: NabledReal>(
    adjoint: &ndarray::Array2<T>,
    wrench: &ArrayView1<'_, T>,
) -> Array1<T> {
    adjoint.t().dot(wrench)
}

fn cross3<T: NabledReal>(a: &ArrayView1<'_, T>, b: &ArrayView1<'_, T>) -> Array1<T> {
    so3::hat(a).dot(b)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array2;

    use super::*;

    #[test]
    fn motion_and_force_cross_products_have_expected_shape() {
        let a = ndarray::arr1(&[1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let b = ndarray::arr1(&[0.0_f64, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let motion = motion_cross(&a.view(), &b.view());
        let force = force_cross(&a.view(), &b.view());
        assert_eq!(motion.len(), 6);
        assert_eq!(force.len(), 6);
        assert_relative_eq!(motion[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn adjoint_motion_and_force_preserve_identity_action() {
        let adj = Array2::<f64>::eye(6);
        let twist = ndarray::arr1(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let wrench = ndarray::arr1(&[0.5_f64, -1.0, 2.0, 3.0, 4.0, -2.0]);
        assert_eq!(adjoint_motion(&adj, &twist.view()), twist);
        assert_eq!(adjoint_force(&adj, &wrench.view()), wrench);
    }
}
