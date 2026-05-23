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
