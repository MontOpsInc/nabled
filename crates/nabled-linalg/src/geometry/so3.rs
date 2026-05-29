//! SO(3) rotation operations.

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::{GeometryError, Rotation3, scalar_two};
use crate::geometry::quat;

#[must_use]
pub fn hat<T: NabledReal>(omega: &ArrayView1<'_, T>) -> Array2<T> {
    let mut m = Array2::<T>::zeros((3, 3));
    m[[0, 1]] = -omega[2];
    m[[0, 2]] = omega[1];
    m[[1, 0]] = omega[2];
    m[[1, 2]] = -omega[0];
    m[[2, 0]] = -omega[1];
    m[[2, 1]] = omega[0];
    m
}

#[must_use]
pub fn vee<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array1<T> {
    ndarray::arr1(&[matrix[[2, 1]], matrix[[0, 2]], matrix[[1, 0]]])
}

pub fn exp<T: NabledReal>(omega: &ArrayView1<'_, T>) -> Result<Rotation3<T>, GeometryError> {
    if omega.iter().any(|v| !v.is_finite()) {
        return Err(GeometryError::NumericalInstability);
    }
    Ok(quat::to_rotation_matrix(&quat::exp(omega)))
}

pub fn log<T: NabledReal>(rotation: &Rotation3<T>) -> Result<Array1<T>, GeometryError> {
    let q = quat::from_rotation_matrix(rotation)?;
    let aa = quat::to_axis_angle(&q);
    Ok(ndarray::arr1(&[aa.axis[0] * aa.angle, aa.axis[1] * aa.angle, aa.axis[2] * aa.angle]))
}

pub fn compose<T: NabledReal>(
    r1: &Rotation3<T>,
    r2: &Rotation3<T>,
) -> Result<Rotation3<T>, GeometryError> {
    if r1.matrix.nrows() != 3 || r2.matrix.ncols() != 3 {
        return Err(GeometryError::DimensionMismatch);
    }
    Ok(Rotation3 { matrix: r1.matrix.dot(&r2.matrix) })
}

#[must_use]
pub fn inverse<T: NabledReal>(rotation: &Rotation3<T>) -> Rotation3<T> {
    Rotation3 { matrix: rotation.matrix.t().to_owned() }
}

#[must_use]
pub fn rotate_vector<T: NabledReal>(
    rotation: &Rotation3<T>,
    vector: &ArrayView1<'_, T>,
) -> Array1<T> {
    rotation.matrix.dot(vector)
}

pub fn rotate_vector_into<T: NabledReal>(
    rotation: &Rotation3<T>,
    vector: &ArrayView1<'_, T>,
    output: &mut Array1<T>,
) -> Result<(), GeometryError> {
    if output.len() != 3 || vector.len() != 3 {
        return Err(GeometryError::DimensionMismatch);
    }
    output.assign(&rotation.matrix.dot(vector));
    Ok(())
}

#[must_use]
pub fn relative_rotation<T: NabledReal>(r1: &Rotation3<T>, r2: &Rotation3<T>) -> Rotation3<T> {
    Rotation3 { matrix: r1.matrix.t().dot(&r2.matrix) }
}

#[must_use]
pub fn angle_between<T: NabledReal>(r1: &Rotation3<T>, r2: &Rotation3<T>) -> T {
    let rel = relative_rotation(r1, r2);
    let trace = rel.matrix[[0, 0]] + rel.matrix[[1, 1]] + rel.matrix[[2, 2]];
    let cos_angle = ((trace - T::one()) / scalar_two::<T>()).clamp(-T::one(), T::one());
    cos_angle.acos()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array2;

    use super::*;

    fn identity_rotation() -> Rotation3<f64> { Rotation3 { matrix: Array2::eye(3) } }

    #[test]
    fn compose_with_identity_preserves_rotation() {
        let id = identity_rotation();
        let omega = ndarray::arr1(&[0.0_f64, 0.0, 0.5]);
        let r = exp(&omega.view()).unwrap();
        let composed = compose(&id, &r).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(composed.matrix[[i, j]], r.matrix[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn exp_log_round_trip() {
        let omega = ndarray::arr1(&[0.05_f64, -0.1, 0.2]);
        let r = exp(&omega.view()).unwrap();
        let recovered = log(&r).unwrap();
        for i in 0..3 {
            assert_relative_eq!(recovered[i], omega[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse_composition_is_identity() {
        let omega = ndarray::arr1(&[0.1_f64, 0.2, 0.3]);
        let r = exp(&omega.view()).unwrap();
        let composed = compose(&r, &inverse(&r)).unwrap();
        let id = identity_rotation();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(composed.matrix[[i, j]], id.matrix[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn rotate_vector_identity_leaves_vector_unchanged() {
        let id = identity_rotation();
        let vector = ndarray::arr1(&[1.0_f64, -2.0, 3.5]);
        let rotated = rotate_vector(&id, &vector.view());
        for i in 0..3 {
            assert_relative_eq!(rotated[i], vector[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn hat_vee_round_trip_and_relative_rotation_metrics() {
        let omega = ndarray::arr1(&[0.1_f64, -0.2, 0.3]);
        let skew = hat(&omega.view());
        let recovered = vee(&skew.view());
        for i in 0..3 {
            assert_relative_eq!(recovered[i], omega[i], epsilon = 1e-12);
        }

        let id = identity_rotation();
        let r = exp(&omega.view()).unwrap();
        let rel = relative_rotation(&id, &r);
        assert_relative_eq!(rel.matrix[[0, 0]], r.matrix[[0, 0]], epsilon = 1e-10);
        assert_relative_eq!(angle_between(&id, &id), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn compose_rotate_into_and_exp_reject_invalid_inputs() {
        let id = identity_rotation();
        let bad = Rotation3 { matrix: Array2::zeros((3, 2)) };
        assert_eq!(compose(&id, &bad), Err(GeometryError::DimensionMismatch));

        let vector = ndarray::arr1(&[1.0_f64, 2.0, 3.0]);
        let mut out = Array1::zeros(2);
        assert_eq!(
            rotate_vector_into(&id, &vector.view(), &mut out),
            Err(GeometryError::DimensionMismatch)
        );

        let mut valid_out = Array1::zeros(3);
        rotate_vector_into(&id, &vector.view(), &mut valid_out).unwrap();
        assert_relative_eq!(valid_out[0], vector[0], epsilon = 1e-12);

        let bad_omega = ndarray::arr1(&[f64::NAN, 0.0, 0.0]);
        assert_eq!(exp(&bad_omega.view()), Err(GeometryError::NumericalInstability));
    }
}
