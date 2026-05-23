//! Joint origin transforms from URDF-style parameters.

use nabled_core::scalar::NabledReal;
use nabled_linalg::geometry::{Rotation3, Transform3, se3, so3};
use ndarray::{Array1, Array2, arr1};

use crate::ModelError;

/// Build a rigid transform from URDF `<origin xyz="..." rpy="..."/>`.
///
/// Rotation uses fixed-axis roll (X), pitch (Y), yaw (Z): `R = Rz * Ry * Rx`.
pub fn transform_from_xyz_rpy<T: NabledReal>(
    xyz: [T; 3],
    rpy: [T; 3],
) -> Result<Transform3<T>, ModelError> {
    let rotation = rotation_from_urdf_rpy(rpy)?;
    let translation = arr1(&[xyz[0], xyz[1], xyz[2]]);
    Ok(se3::from_rotation_translation(&rotation, &translation))
}

/// Map legacy DH scalar fields stored on [`BodySpec`](crate::robot::BodySpec) to a static origin.
pub fn joint_origin_from_dh_scalars<T: NabledReal>(
    a: T,
    alpha: T,
    d: T,
    theta: T,
) -> Result<Transform3<T>, ModelError> {
    transform_from_xyz_rpy([a, T::zero(), d], [alpha, theta, T::zero()])
}

/// Identity transform (parent and child frames coincident).
#[must_use]
pub fn identity_transform<T: NabledReal>() -> Transform3<T> {
    se3::from_rotation_translation(
        &Rotation3 { matrix: Array2::<T>::eye(3) },
        &Array1::<T>::zeros(3),
    )
}

fn rotation_from_urdf_rpy<T: NabledReal>(rpy: [T; 3]) -> Result<Rotation3<T>, ModelError> {
    let rx = so3::exp(&arr1(&[rpy[0], T::zero(), T::zero()]).view())
        .map_err(|_| ModelError::InvalidInput("invalid roll angle".to_string()))?;
    let ry = so3::exp(&arr1(&[T::zero(), rpy[1], T::zero()]).view())
        .map_err(|_| ModelError::InvalidInput("invalid pitch angle".to_string()))?;
    let rz = so3::exp(&arr1(&[T::zero(), T::zero(), rpy[2]]).view())
        .map_err(|_| ModelError::InvalidInput("invalid yaw angle".to_string()))?;
    let ry_rx = so3::compose(&ry, &rx)
        .map_err(|_| ModelError::InvalidInput("invalid origin rotation".to_string()))?;
    so3::compose(&rz, &ry_rx)
        .map_err(|_| ModelError::InvalidInput("invalid origin rotation".to_string()))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn pure_translation_origin() {
        let tf = transform_from_xyz_rpy([1.0_f64, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
        assert_relative_eq!(tf.translation[0], 1.0, epsilon = 1e-12);
    }
}
