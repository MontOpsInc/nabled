//! SE(3) rigid transform operations.

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1};

use super::{GeometryError, Rotation3, Transform3};
use crate::geometry::so3;

#[must_use]
pub fn from_rotation_translation<T: NabledReal>(
    rotation: &Rotation3<T>,
    translation: &Array1<T>,
) -> Transform3<T> {
    Transform3 { rotation: rotation.clone(), translation: translation.clone() }
}

#[must_use]
pub fn rotation<T: NabledReal>(transform: &Transform3<T>) -> &Rotation3<T> { &transform.rotation }

#[must_use]
pub fn translation<T: NabledReal>(transform: &Transform3<T>) -> &Array1<T> {
    &transform.translation
}

pub fn compose<T: NabledReal>(
    t1: &Transform3<T>,
    t2: &Transform3<T>,
) -> Result<Transform3<T>, GeometryError> {
    let rotation = so3::compose(&t1.rotation, &t2.rotation)?;
    let translated = so3::rotate_vector(&t1.rotation, &t2.translation.view());
    let mut out_t = t1.translation.clone();
    out_t[0] += translated[0];
    out_t[1] += translated[1];
    out_t[2] += translated[2];
    Ok(Transform3 { rotation, translation: out_t })
}

#[must_use]
pub fn inverse<T: NabledReal>(transform: &Transform3<T>) -> Transform3<T> {
    let inv_r = so3::inverse(&transform.rotation);
    let mut neg_translation = transform.translation.clone();
    neg_translation.mapv_inplace(|value| -value);
    let neg_t = so3::rotate_vector(&inv_r, &neg_translation.view());
    Transform3 { rotation: inv_r, translation: neg_t }
}

#[must_use]
pub fn transform_point<T: NabledReal>(
    transform: &Transform3<T>,
    point: &ArrayView1<'_, T>,
) -> Array1<T> {
    let mut out = so3::rotate_vector(&transform.rotation, point);
    out[0] += transform.translation[0];
    out[1] += transform.translation[1];
    out[2] += transform.translation[2];
    out
}

pub fn transform_point_into<T: NabledReal>(
    transform: &Transform3<T>,
    point: &ArrayView1<'_, T>,
    output: &mut Array1<T>,
) -> Result<(), GeometryError> {
    if point.len() != 3 || output.len() != 3 {
        return Err(GeometryError::DimensionMismatch);
    }
    output.assign(&transform_point(transform, point));
    Ok(())
}

#[must_use]
pub fn transform_vector<T: NabledReal>(
    transform: &Transform3<T>,
    vector: &ArrayView1<'_, T>,
) -> Array1<T> {
    so3::rotate_vector(&transform.rotation, vector)
}

#[must_use]
pub fn adjoint<T: NabledReal>(transform: &Transform3<T>) -> Array2<T> {
    let r = &transform.rotation.matrix;
    let t = &transform.translation;
    let tx = so3::hat(&ndarray::arr1(&[t[0], t[1], t[2]]).view());
    let mut adj = Array2::<T>::zeros((6, 6));
    for i in 0..3 {
        for j in 0..3 {
            adj[[i, j]] = r[[i, j]];
            adj[[i + 3, j + 3]] = r[[i, j]];
        }
    }
    let t_hat_r = tx.dot(r);
    for i in 0..3 {
        for j in 0..3 {
            adj[[i, j + 3]] = t_hat_r[[i, j]];
        }
    }
    adj
}

#[must_use]
pub fn to_homogeneous<T: NabledReal>(transform: &Transform3<T>) -> Array2<T> {
    let mut h = Array2::<T>::zeros((4, 4));
    for i in 0..3 {
        for j in 0..3 {
            h[[i, j]] = transform.rotation.matrix[[i, j]];
        }
        h[[i, 3]] = transform.translation[i];
    }
    h[[3, 3]] = T::one();
    h
}

pub fn from_homogeneous<T: NabledReal>(matrix: &Array2<T>) -> Result<Transform3<T>, GeometryError> {
    if matrix.nrows() != 4 || matrix.ncols() != 4 {
        return Err(GeometryError::DimensionMismatch);
    }
    let mut rotation = Array2::<T>::zeros((3, 3));
    let mut translation = Array1::<T>::zeros(3);
    for i in 0..3 {
        for j in 0..3 {
            rotation[[i, j]] = matrix[[i, j]];
        }
        translation[i] = matrix[[i, 3]];
    }
    Ok(Transform3 { rotation: Rotation3 { matrix: rotation }, translation })
}

pub fn log<T: NabledReal>(transform: &Transform3<T>) -> Result<Array1<T>, GeometryError> {
    let omega = so3::log(&transform.rotation)?;
    let mut twist = Array1::<T>::zeros(6);
    twist.slice_mut(ndarray::s![0..3]).assign(&omega);
    twist.slice_mut(ndarray::s![3..6]).assign(&transform.translation);
    Ok(twist)
}

pub fn exp<T: NabledReal>(twist: &ArrayView1<'_, T>) -> Result<Transform3<T>, GeometryError> {
    if twist.len() != 6 {
        return Err(GeometryError::DimensionMismatch);
    }
    let rotation = so3::exp(&twist.slice(ndarray::s![0..3]).view())?;
    let translation = twist.slice(ndarray::s![3..6]).to_owned();
    Ok(from_rotation_translation(&rotation, &translation))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2};

    use super::*;
    use crate::geometry::{GeometryError, Rotation3};

    fn identity_transform() -> Transform3<f64> {
        Transform3 {
            rotation:    Rotation3 { matrix: Array2::eye(3) },
            translation: Array1::zeros(3),
        }
    }

    #[test]
    fn compose_with_identity_preserves_transform() {
        let id = identity_transform();
        let omega = ndarray::arr1(&[0.0_f64, 0.0, 0.3]);
        let rotation = so3::exp(&omega.view()).unwrap();
        let translation = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let transform = from_rotation_translation(&rotation, &translation);
        let composed = compose(&id, &transform).unwrap();
        for i in 0..3 {
            assert_relative_eq!(composed.translation[i], translation[i], epsilon = 1e-10);
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    composed.rotation.matrix[[i, j]],
                    rotation.matrix[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn exp_log_round_trip() {
        let twist = ndarray::arr1(&[0.1_f64, 0.0, 0.0, 0.5, -0.2, 1.0]);
        let transform = exp(&twist.view()).unwrap();
        let recovered = log(&transform).unwrap();
        for i in 0..6 {
            assert_relative_eq!(recovered[i], twist[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn inverse_composition_is_identity_transform() {
        let twist = ndarray::arr1(&[0.0_f64, 0.2, 0.0, 1.0, -0.5, 2.0]);
        let transform = exp(&twist.view()).unwrap();
        let composed = compose(&transform, &inverse(&transform)).unwrap();
        let id = identity_transform();
        for i in 0..3 {
            assert_relative_eq!(composed.translation[i], id.translation[i], epsilon = 1e-10);
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    composed.rotation.matrix[[i, j]],
                    id.rotation.matrix[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn transform_point_identity_leaves_point_unchanged() {
        let id = identity_transform();
        let point = ndarray::arr1(&[1.0_f64, -2.0, 3.5]);
        let transformed = transform_point(&id, &point.view());
        for i in 0..3 {
            assert_relative_eq!(transformed[i], point[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn exp_dimension_mismatch_errors() {
        let bad_twist = ndarray::arr1(&[0.0_f64; 5]);
        assert_eq!(exp(&bad_twist.view()), Err(GeometryError::DimensionMismatch));
    }

    #[test]
    fn accessors_homogeneous_adjoint_and_into_paths() {
        let twist = ndarray::arr1(&[0.0_f64, 0.2, 0.0, 1.0, -0.5, 2.0]);
        let transform = exp(&twist.view()).unwrap();
        assert_eq!(rotation(&transform).matrix.dim(), (3, 3));
        assert_eq!(translation(&transform).len(), 3);

        let homogeneous = to_homogeneous(&transform);
        let recovered = from_homogeneous(&homogeneous).unwrap();
        for i in 0..3 {
            assert_relative_eq!(
                recovered.translation[i],
                transform.translation[i],
                epsilon = 1e-10
            );
        }

        let adj = adjoint(&transform);
        assert_eq!(adj.dim(), (6, 6));

        let point = ndarray::arr1(&[1.0_f64, -2.0, 3.5]);
        let mut out = Array1::zeros(3);
        transform_point_into(&transform, &point.view(), &mut out).unwrap();
        assert_relative_eq!(out[0], transform_point(&transform, &point.view())[0], epsilon = 1e-12);

        let vector = transform_vector(&transform, &point.view());
        assert_eq!(vector.len(), 3);
    }

    #[test]
    fn homogeneous_and_transform_point_into_reject_bad_shapes() {
        let transform = identity_transform();
        let bad_homogeneous = Array2::<f64>::zeros((3, 4));
        assert_eq!(from_homogeneous(&bad_homogeneous), Err(GeometryError::DimensionMismatch));

        let bad_point = ndarray::arr1(&[1.0_f64, 2.0]);
        let mut out = Array1::zeros(3);
        assert_eq!(
            transform_point_into(&transform, &bad_point.view(), &mut out),
            Err(GeometryError::DimensionMismatch)
        );
    }
}
