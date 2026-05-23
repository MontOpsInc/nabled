//! Forward kinematics.

use nabled_core::scalar::NabledReal;
use nabled_linalg::geometry::{Rotation3, Transform3, se3};
use ndarray::{Array1, Array2, ArrayView1};

use crate::chain::{ChainSpec, DhConvention, JointType};
use crate::error::KinematicsError;

fn dh_transform<T: NabledReal>(
    convention: DhConvention,
    joint_type: JointType,
    a: T,
    alpha: T,
    d: T,
    theta: T,
) -> Transform3<T> {
    let (ct, st, ca, sa) = (theta.cos(), theta.sin(), alpha.cos(), alpha.sin());
    let matrix = match convention {
        DhConvention::Standard => {
            let mut m = Array2::<T>::zeros((3, 3));
            m[[0, 0]] = ct;
            m[[0, 1]] = -st * ca;
            m[[0, 2]] = st * sa;
            m[[1, 0]] = st;
            m[[1, 1]] = ct * ca;
            m[[1, 2]] = -ct * sa;
            m[[2, 0]] = T::zero();
            m[[2, 1]] = sa;
            m[[2, 2]] = ca;
            let tx = a * ct;
            let ty = a * st;
            let tz = d;
            (m, tx, ty, tz)
        }
        DhConvention::Modified => {
            let mut m = Array2::<T>::zeros((3, 3));
            m[[0, 0]] = ct;
            m[[0, 1]] = -st;
            m[[0, 2]] = T::zero();
            m[[1, 0]] = st * ca;
            m[[1, 1]] = ct * ca;
            m[[1, 2]] = -sa;
            m[[2, 0]] = st * sa;
            m[[2, 1]] = ct * sa;
            m[[2, 2]] = ca;
            let tx = a;
            let ty = -sa * d;
            let tz = ca * d;
            (m, tx, ty, tz)
        }
    };
    let (rot, tx, ty, tz) = matrix;
    let mut translation = Array1::<T>::zeros(3);
    translation[0] = tx;
    translation[1] = ty;
    translation[2] = tz;
    let _ = joint_type;
    se3::from_rotation_translation(&Rotation3 { matrix: rot }, &translation)
}

fn joint_transform<T: NabledReal>(chain: &ChainSpec<T>, q: T, index: usize) -> Transform3<T> {
    let joint_type = chain.joint_types[index];
    let a = chain.a[index];
    let alpha = chain.alpha[index];
    let d = chain.d[index];
    let theta_offset = chain.theta_offset[index];
    let (link_d, theta) = match joint_type {
        JointType::Revolute => (d, q + theta_offset),
        JointType::Prismatic => (q + d, theta_offset),
    };
    dh_transform(chain.convention, joint_type, a, alpha, link_d, theta)
}

/// Forward kinematics end-effector transform.
pub fn fk<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
) -> Result<Transform3<T>, KinematicsError> {
    fk_view(chain, &q.view())
}

/// Forward kinematics from joint view.
pub fn fk_view<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
) -> Result<Transform3<T>, KinematicsError> {
    chain.validate()?;
    if q.len() != chain.num_joints() {
        return Err(KinematicsError::DimensionMismatch);
    }
    let mut transform = se3::from_rotation_translation(
        &Rotation3 { matrix: Array2::<T>::eye(3) },
        &Array1::<T>::zeros(3),
    );
    for (i, &qi) in q.iter().enumerate() {
        let link = joint_transform(chain, qi, i);
        transform =
            se3::compose(&transform, &link).map_err(|_| KinematicsError::NumericalInstability)?;
    }
    Ok(transform)
}

/// Forward kinematics into existing transform (overwrites).
pub fn fk_into<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    output: &mut Transform3<T>,
) -> Result<(), KinematicsError> {
    *output = fk(chain, q)?;
    Ok(())
}

/// Per-link cumulative transforms including base frame.
pub fn link_transforms<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
) -> Result<Vec<Transform3<T>>, KinematicsError> {
    link_transforms_view(chain, &q.view())
}

/// Per-link cumulative transforms from joint view.
pub fn link_transforms_view<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
) -> Result<Vec<Transform3<T>>, KinematicsError> {
    chain.validate()?;
    if q.len() != chain.num_joints() {
        return Err(KinematicsError::DimensionMismatch);
    }
    let mut transforms = Vec::with_capacity(chain.num_joints() + 1);
    transforms.push(se3::from_rotation_translation(
        &Rotation3 { matrix: Array2::<T>::eye(3) },
        &Array1::<T>::zeros(3),
    ));
    for (i, &qi) in q.iter().enumerate() {
        let link = joint_transform(chain, qi, i);
        let next = se3::compose(transforms.last().unwrap(), &link)
            .map_err(|_| KinematicsError::NumericalInstability)?;
        transforms.push(next);
    }
    Ok(transforms)
}

/// Per-link transforms into caller buffer.
pub fn link_transforms_into<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    output: &mut [Transform3<T>],
) -> Result<(), KinematicsError> {
    let transforms = link_transforms(chain, q)?;
    if output.len() != transforms.len() {
        return Err(KinematicsError::DimensionMismatch);
    }
    output.clone_from_slice(&transforms);
    Ok(())
}

/// End-effector pose (alias for FK).
pub fn end_effector_pose<T: NabledReal>(
    chain: &ChainSpec<T>,
    q: &Array1<T>,
) -> Result<Transform3<T>, KinematicsError> {
    fk(chain, q)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::arr1;

    use super::*;
    use crate::chain::{ChainSpec, DhConvention, JointType};

    #[test]
    fn planar_2r_fk() {
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
        let pose = fk(&chain, &q).unwrap();
        assert_relative_eq!(pose.translation[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(pose.translation[1], 0.0, epsilon = 1e-10);
    }
}
