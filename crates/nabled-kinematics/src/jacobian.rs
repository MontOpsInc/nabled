//! Geometric Jacobian computation.

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1};

use crate::chain::{ChainSpec, JointType};
use crate::error::KinematicsError;
use crate::fk::link_transforms_view;

fn z_axis<T: NabledReal>(transform: &nabled_linalg::geometry::Transform3<T>) -> Array1<T> {
    let r = &transform.rotation.matrix;
    ndarray::arr1(&[r[[0, 2]], r[[1, 2]], r[[2, 2]]])
}

fn origin<T: NabledReal>(transform: &nabled_linalg::geometry::Transform3<T>) -> Array1<T> {
    transform.translation.clone()
}

/// Full 6×n geometric Jacobian.
pub fn jacobian<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
) -> Result<Array2<T>, KinematicsError> {
    jacobian_view(chain, &q.view())
}

/// Jacobian from joint view.
pub fn jacobian_view<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
) -> Result<Array2<T>, KinematicsError> {
    chain.validate()?;
    if q.len() != chain.num_joints() {
        return Err(KinematicsError::DimensionMismatch);
    }
    let transforms = link_transforms_view(chain, q)?;
    let ee_joint = chain.ee_joint_index();
    let ee = &transforms[ee_joint + 1];
    let p_e = origin(ee);
    let n = chain.num_joints();
    let mut j = Array2::<T>::zeros((6, n));
    for i in 0..n {
        if i > ee_joint {
            continue;
        }
        let frame = &transforms[i];
        let z_i = z_axis(frame);
        let p_i = origin(frame);
        let r = &mut j.column_mut(i);
        match chain.joint_types[i] {
            JointType::Revolute => {
                let diff = ndarray::arr1(&[p_e[0] - p_i[0], p_e[1] - p_i[1], p_e[2] - p_i[2]]);
                let linear = cross3(&z_i.view(), &diff.view());
                r[0] = linear[0];
                r[1] = linear[1];
                r[2] = linear[2];
                r[3] = z_i[0];
                r[4] = z_i[1];
                r[5] = z_i[2];
            }
            JointType::Prismatic => {
                r[0] = z_i[0];
                r[1] = z_i[1];
                r[2] = z_i[2];
                r[3] = T::zero();
                r[4] = T::zero();
                r[5] = T::zero();
            }
        }
    }
    Ok(j)
}

/// Jacobian into caller buffer.
pub fn jacobian_into<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    output: &mut Array2<T>,
) -> Result<(), KinematicsError> {
    let j = jacobian(chain, q)?;
    if output.dim() != j.dim() {
        return Err(KinematicsError::DimensionMismatch);
    }
    output.assign(&j);
    Ok(())
}

/// Translation-only 3×n Jacobian block.
pub fn jacobian_translation<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
) -> Result<Array2<T>, KinematicsError> {
    jacobian_translation_view(chain, &q.view())
}

/// Translation-only 3×n Jacobian block from joint view.
pub fn jacobian_translation_view<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
) -> Result<Array2<T>, KinematicsError> {
    let j = jacobian_view(chain, q)?;
    Ok(j.slice(ndarray::s![0..3, ..]).to_owned())
}

/// Rotation-only 3×n Jacobian block.
pub fn jacobian_rotation<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
) -> Result<Array2<T>, KinematicsError> {
    jacobian_rotation_view(chain, &q.view())
}

/// Rotation-only 3×n Jacobian block from joint view.
pub fn jacobian_rotation_view<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
) -> Result<Array2<T>, KinematicsError> {
    let j = jacobian_view(chain, q)?;
    Ok(j.slice(ndarray::s![3..6, ..]).to_owned())
}

fn cross3<T: NabledReal>(a: &ArrayView1<'_, T>, b: &ArrayView1<'_, T>) -> Array1<T> {
    ndarray::arr1(&[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::arr1;

    use super::*;
    use crate::chain::{ChainSpec, DhConvention, JointType};

    #[test]
    fn planar_2r_jacobian_at_zero() {
        let chain = ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Revolute, JointType::Revolute],
            arr1(&[1.0_f64, 1.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        let q = arr1(&[0.0_f64, 0.0]);
        let j = jacobian_translation(&chain, &q).unwrap();
        assert_relative_eq!(j[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(j[[1, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(j[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(j[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn prismatic_jacobian_linear_block_is_joint_axis() {
        let chain = ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Prismatic],
            arr1(&[0.0_f64]),
            arr1(&[0.0]),
            arr1(&[0.0]),
            arr1(&[0.0]),
        )
        .unwrap();
        let q = arr1(&[0.25_f64]);
        let j = jacobian(&chain, &q).unwrap();
        assert_relative_eq!(j[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(j[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(j[[2, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(j[[3, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let chain = ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Revolute, JointType::Revolute],
            arr1(&[1.0_f64, 1.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        let err = jacobian(&chain, &arr1(&[0.0])).unwrap_err();
        assert_eq!(err, KinematicsError::DimensionMismatch);
    }

    #[test]
    fn jacobian_view_matches_allocating() {
        let chain = ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Revolute, JointType::Revolute],
            arr1(&[1.0_f64, 1.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        let q = arr1(&[0.2_f64, 0.1]);
        let owned = jacobian(&chain, &q).unwrap();
        let viewed = jacobian_view(&chain, &q.view()).unwrap();
        assert_relative_eq!(owned, viewed, epsilon = 1e-12);
    }
}
