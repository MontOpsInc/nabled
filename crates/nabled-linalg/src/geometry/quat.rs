//! Quaternion operations (scalar-first `[w, x, y, z]` convention).

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, ArrayView1};

use super::{AxisAngle, GeometryError, Quat, Rotation3, scalar_half, scalar_two};

/// Identity quaternion.
#[must_use]
pub fn identity<T: NabledReal>() -> Quat<T> {
    Quat { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
}

/// Build a quaternion from axis-angle representation.
#[must_use]
pub fn from_axis_angle<T: NabledReal>(axis_angle: &AxisAngle<T>) -> Quat<T> {
    let half = axis_angle.angle * scalar_half::<T>();
    let s = half.sin();
    Quat {
        w: half.cos(),
        x: axis_angle.axis[0] * s,
        y: axis_angle.axis[1] * s,
        z: axis_angle.axis[2] * s,
    }
}

/// Build a quaternion from a 3×3 rotation matrix.
pub fn from_rotation_matrix<T: NabledReal>(
    rotation: &Rotation3<T>,
) -> Result<Quat<T>, GeometryError> {
    let m = &rotation.matrix;
    let trace = m[[0, 0]] + m[[1, 1]] + m[[2, 2]];
    if trace > T::zero() {
        let s = ((trace + T::one()) * scalar_two::<T>()).sqrt();
        Ok(Quat {
            w: s * scalar_half::<T>(),
            x: (m[[2, 1]] - m[[1, 2]]) / s,
            y: (m[[0, 2]] - m[[2, 0]]) / s,
            z: (m[[1, 0]] - m[[0, 1]]) / s,
        })
    } else if m[[0, 0]] >= m[[1, 1]] && m[[0, 0]] >= m[[2, 2]] {
        let s = ((T::one() + m[[0, 0]] - m[[1, 1]] - m[[2, 2]]) * scalar_two::<T>()).sqrt();
        Ok(Quat {
            w: (m[[2, 1]] - m[[1, 2]]) / s,
            x: s * scalar_half::<T>(),
            y: (m[[0, 1]] + m[[1, 0]]) / s,
            z: (m[[0, 2]] + m[[2, 0]]) / s,
        })
    } else if m[[1, 1]] >= m[[2, 2]] {
        let s = ((T::one() + m[[1, 1]] - m[[0, 0]] - m[[2, 2]]) * scalar_two::<T>()).sqrt();
        Ok(Quat {
            w: (m[[0, 2]] - m[[2, 0]]) / s,
            x: (m[[0, 1]] + m[[1, 0]]) / s,
            y: s * scalar_half::<T>(),
            z: (m[[1, 2]] + m[[2, 1]]) / s,
        })
    } else {
        let s = ((T::one() + m[[2, 2]] - m[[0, 0]] - m[[1, 1]]) * scalar_two::<T>()).sqrt();
        Ok(Quat {
            w: (m[[1, 0]] - m[[0, 1]]) / s,
            x: (m[[0, 2]] + m[[2, 0]]) / s,
            y: (m[[1, 2]] + m[[2, 1]]) / s,
            z: s * scalar_half::<T>(),
        })
    }
}

/// Convert quaternion to 3×3 rotation matrix.
#[must_use]
pub fn to_rotation_matrix<T: NabledReal>(q: &Quat<T>) -> Rotation3<T> {
    let qn = normalize(q);
    let (w, x, y, z) = (qn.w, qn.x, qn.y, qn.z);
    let two = scalar_two::<T>();
    let mut matrix = ndarray::Array2::<T>::zeros((3, 3));
    matrix[[0, 0]] = T::one() - two * (y * y + z * z);
    matrix[[0, 1]] = two * (x * y - w * z);
    matrix[[0, 2]] = two * (x * z + w * y);
    matrix[[1, 0]] = two * (x * y + w * z);
    matrix[[1, 1]] = T::one() - two * (x * x + z * z);
    matrix[[1, 2]] = two * (y * z - w * x);
    matrix[[2, 0]] = two * (x * z - w * y);
    matrix[[2, 1]] = two * (y * z + w * x);
    matrix[[2, 2]] = T::one() - two * (x * x + y * y);
    Rotation3 { matrix }
}

/// Convert quaternion to axis-angle.
#[must_use]
pub fn to_axis_angle<T: NabledReal>(q: &Quat<T>) -> AxisAngle<T> {
    let qn = normalize(q);
    let angle = scalar_two::<T>() * qn.w.acos();
    let s = (T::one() - qn.w * qn.w).sqrt();
    if s < T::from_f64(1e-12).unwrap_or(T::zero()) {
        return AxisAngle { axis: [T::one(), T::zero(), T::zero()], angle: T::zero() };
    }
    AxisAngle { axis: [qn.x / s, qn.y / s, qn.z / s], angle }
}

#[must_use]
pub fn mul<T: NabledReal>(a: &Quat<T>, b: &Quat<T>) -> Quat<T> {
    Quat {
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    }
}

#[must_use]
pub fn conjugate<T: NabledReal>(q: &Quat<T>) -> Quat<T> {
    Quat { w: q.w, x: -q.x, y: -q.y, z: -q.z }
}

#[must_use]
pub fn inverse<T: NabledReal>(q: &Quat<T>) -> Quat<T> {
    let n2 = norm(q) * norm(q);
    let c = conjugate(q);
    Quat { w: c.w / n2, x: c.x / n2, y: c.y / n2, z: c.z / n2 }
}

#[must_use]
pub fn normalize<T: NabledReal>(q: &Quat<T>) -> Quat<T> {
    let n = norm(q);
    if n <= T::from_f64(1e-15).unwrap_or(T::zero()) {
        return identity();
    }
    Quat { w: q.w / n, x: q.x / n, y: q.y / n, z: q.z / n }
}

pub fn normalize_into<T: NabledReal>(q: &mut Quat<T>) -> Result<(), GeometryError> {
    let n = norm(q);
    if n <= T::from_f64(1e-15).unwrap_or(T::zero()) {
        return Err(GeometryError::ZeroNorm);
    }
    q.w /= n;
    q.x /= n;
    q.y /= n;
    q.z /= n;
    Ok(())
}

#[must_use]
pub fn norm<T: NabledReal>(q: &Quat<T>) -> T {
    (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt()
}

#[must_use]
pub fn dot<T: NabledReal>(a: &Quat<T>, b: &Quat<T>) -> T {
    a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z
}

#[must_use]
pub fn slerp<T: NabledReal>(a: &Quat<T>, b: &Quat<T>, t: T) -> Quat<T> {
    let mut b_adj = *b;
    let mut cos_theta = dot(a, &b_adj);
    if cos_theta < T::zero() {
        b_adj = Quat { w: -b.w, x: -b.x, y: -b.y, z: -b.z };
        cos_theta = -cos_theta;
    }
    if cos_theta > T::one() - T::from_f64(1e-6).unwrap_or(T::zero()) {
        return nlerp(a, &b_adj, t);
    }
    let theta = cos_theta.acos();
    let sin_theta = theta.sin();
    let wa = ((T::one() - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;
    Quat {
        w: wa * a.w + wb * b_adj.w,
        x: wa * a.x + wb * b_adj.x,
        y: wa * a.y + wb * b_adj.y,
        z: wa * a.z + wb * b_adj.z,
    }
}

#[must_use]
pub fn nlerp<T: NabledReal>(a: &Quat<T>, b: &Quat<T>, t: T) -> Quat<T> {
    normalize(&Quat {
        w: (T::one() - t) * a.w + t * b.w,
        x: (T::one() - t) * a.x + t * b.x,
        y: (T::one() - t) * a.y + t * b.y,
        z: (T::one() - t) * a.z + t * b.z,
    })
}

#[must_use]
pub fn exp<T: NabledReal>(omega: &ArrayView1<'_, T>) -> Quat<T> {
    let angle = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    if angle <= T::from_f64(1e-12).unwrap_or(T::zero()) {
        return identity();
    }
    let half = angle * scalar_half::<T>();
    let s = half.sin() / angle;
    Quat { w: half.cos(), x: omega[0] * s, y: omega[1] * s, z: omega[2] * s }
}

#[must_use]
pub fn log<T: NabledReal>(q: &Quat<T>) -> Array1<T> {
    let qn = normalize(q);
    let cos_half = qn.w.abs().min(T::one());
    let half_angle = cos_half.acos();
    let sin_half = half_angle.sin();
    if sin_half.abs() <= T::from_f64(1e-12).unwrap_or(T::zero()) {
        return Array1::<T>::zeros(3);
    }
    let scale = scalar_two::<T>() * half_angle / sin_half;
    ndarray::arr1(&[qn.x * scale, qn.y * scale, qn.z * scale])
}

pub fn mul_into<T: NabledReal>(a: &Quat<T>, b: &Quat<T>, out: &mut Quat<T>) { *out = mul(a, b); }

pub fn slerp_into<T: NabledReal>(a: &Quat<T>, b: &Quat<T>, t: T, out: &mut Quat<T>) {
    *out = slerp(a, b, t);
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::geometry::{AxisAngle, GeometryError};

    #[test]
    fn identity_is_unit_and_yields_identity_matrix() {
        let q = identity::<f64>();
        assert_relative_eq!(norm(&q), 1.0, epsilon = 1e-12);
        let rot = to_rotation_matrix(&q);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(rot.matrix[[i, j]], expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn exp_log_round_trip() {
        let omega = ndarray::arr1(&[0.1_f64, 0.2, -0.15]);
        let q = exp(&omega.view());
        let recovered = log(&q);
        for i in 0..3 {
            assert_relative_eq!(recovered[i], omega[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn composition_matches_rotation_matrix_product() {
        let q1 = from_axis_angle(&AxisAngle { axis: [0.0, 0.0, 1.0], angle: 0.3 });
        let q2 = from_axis_angle(&AxisAngle { axis: [0.0, 1.0, 0.0], angle: 0.5 });
        let composed = mul(&q1, &q2);
        let r_composed = to_rotation_matrix(&composed).matrix;
        let expected = to_rotation_matrix(&q1).matrix.dot(&to_rotation_matrix(&q2).matrix);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r_composed[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn from_rotation_matrix_round_trip() {
        let q = from_axis_angle(&AxisAngle { axis: [1.0, 2.0, 3.0], angle: 0.4 });
        let qn = normalize(&q);
        let rot = to_rotation_matrix(&qn);
        let recovered = from_rotation_matrix(&rot).unwrap();
        let recovered_rot = to_rotation_matrix(&recovered).matrix;
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(rot.matrix[[i, j]], recovered_rot[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn normalize_into_zero_quaternion_errors() {
        let mut q = Quat { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(normalize_into(&mut q), Err(GeometryError::ZeroNorm));
    }

    #[test]
    fn inverse_times_original_is_identity() {
        let q = normalize(&from_axis_angle(&AxisAngle { axis: [0.3, 0.4, 0.5], angle: 0.7 }));
        let product = mul(&q, &inverse(&q));
        assert_relative_eq!(product.w, 1.0, epsilon = 1e-10);
        assert_relative_eq!(product.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(product.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(product.z, 0.0, epsilon = 1e-10);
    }
}
