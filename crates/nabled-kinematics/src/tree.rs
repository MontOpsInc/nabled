//! Tree-structured forward kinematics and Jacobians from URDF joint origins.

use std::collections::{HashMap, HashSet};

use nabled_core::scalar::NabledReal;
use nabled_linalg::geometry::{Rotation3, Transform3, se3, so3};
use ndarray::{Array1, Array2, ArrayView1, arr1};

use crate::error::KinematicsError;

/// Joint classification for tree kinematics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeJointType {
    Revolute,
    Prismatic,
    Fixed,
}

/// Minimal tree-model interface implemented by [`nabled_model::robot::RobotModel`].
pub trait KinematicTreeModel<T: NabledReal> {
    fn validate_tree(&self) -> Result<(), KinematicsError>;
    fn dof(&self) -> usize;
    fn actuated_indices(&self) -> Vec<usize>;
    fn topological_order(&self) -> Vec<usize>;
    fn body_index_for_link(&self, link_name: &str) -> Option<usize>;
    fn parent_link(&self, body_index: usize) -> &str;
    fn child_link(&self, body_index: usize) -> &str;
    fn joint_type(&self, body_index: usize) -> TreeJointType;
    fn joint_origin(&self, body_index: usize) -> &Transform3<T>;
    fn joint_axis(&self, body_index: usize) -> [T; 3];
    fn chain_indices(&self, base_link: &str, ee_link: &str) -> Result<Vec<usize>, KinematicsError>;
}

/// World-frame link transforms keyed by link name (base link at identity).
pub fn link_transforms_tree<M, T>(
    model: &M,
    q: &Array1<T>,
) -> Result<HashMap<String, Transform3<T>>, KinematicsError>
where
    M: KinematicTreeModel<T>,
    T: NabledReal,
{
    link_transforms_tree_view(model, &q.view())
}

/// World-frame link transforms from a joint view.
pub fn link_transforms_tree_view<M, T>(
    model: &M,
    q: &ArrayView1<'_, T>,
) -> Result<HashMap<String, Transform3<T>>, KinematicsError>
where
    M: KinematicTreeModel<T>,
    T: NabledReal,
{
    model.validate_tree()?;
    if q.len() != model.dof() {
        return Err(KinematicsError::DimensionMismatch);
    }

    let mut transforms = HashMap::new();
    let identity = se3::from_rotation_translation(
        &Rotation3 { matrix: Array2::<T>::eye(3) },
        &Array1::<T>::zeros(3),
    );

    let actuated = model.actuated_indices();
    let mut actuated_map = HashMap::new();
    for (joint_index, &body_index) in actuated.iter().enumerate() {
        let _ = actuated_map.insert(body_index, joint_index);
    }

    for body_index in model.topological_order() {
        let parent_link = model.parent_link(body_index);
        let child_link = model.child_link(body_index);
        let parent_tf = transforms.get(parent_link).cloned().unwrap_or_else(|| identity.clone());
        let origin = model.joint_origin(body_index);
        let joint_tf = match model.joint_type(body_index) {
            TreeJointType::Fixed => identity.clone(),
            TreeJointType::Revolute | TreeJointType::Prismatic => {
                let joint_index = actuated_map.get(&body_index).copied().ok_or_else(|| {
                    KinematicsError::InvalidInput(format!(
                        "missing actuation index for body {body_index}"
                    ))
                })?;
                joint_motion(
                    model.joint_type(body_index),
                    model.joint_axis(body_index),
                    q[joint_index],
                )?
            }
        };
        let composed =
            se3::compose(&parent_tf, origin).map_err(|_| KinematicsError::NumericalInstability)?;
        let child_tf = se3::compose(&composed, &joint_tf)
            .map_err(|_| KinematicsError::NumericalInstability)?;
        drop(transforms.insert(child_link.to_string(), child_tf));
    }

    Ok(transforms)
}

/// End-effector pose of `ee_link` relative to `base_link`.
pub fn end_effector_pose_tree<M, T>(
    model: &M,
    base_link: &str,
    ee_link: &str,
    q: &Array1<T>,
) -> Result<Transform3<T>, KinematicsError>
where
    M: KinematicTreeModel<T>,
    T: NabledReal,
{
    let transforms = link_transforms_tree(model, q)?;
    let base = transforms.get(base_link).cloned().unwrap_or_else(|| {
        se3::from_rotation_translation(
            &Rotation3 { matrix: Array2::<T>::eye(3) },
            &Array1::<T>::zeros(3),
        )
    });
    let ee = transforms
        .get(ee_link)
        .ok_or_else(|| KinematicsError::InvalidInput(format!("unknown link {ee_link}")))?;
    let base_inv = se3::inverse(&base);
    se3::compose(&base_inv, ee).map_err(|_| KinematicsError::NumericalInstability)
}

/// Geometric Jacobian (6×`dof`) for the end-effector twist in the base frame.
pub fn jacobian_tree<M, T>(
    model: &M,
    base_link: &str,
    ee_link: &str,
    q: &Array1<T>,
) -> Result<Array2<T>, KinematicsError>
where
    M: KinematicTreeModel<T>,
    T: NabledReal,
{
    jacobian_tree_view(model, base_link, ee_link, &q.view())
}

/// Geometric Jacobian from a joint view.
pub fn jacobian_tree_view<M, T>(
    model: &M,
    base_link: &str,
    ee_link: &str,
    q: &ArrayView1<'_, T>,
) -> Result<Array2<T>, KinematicsError>
where
    M: KinematicTreeModel<T>,
    T: NabledReal,
{
    model.validate_tree()?;
    if q.len() != model.dof() {
        return Err(KinematicsError::DimensionMismatch);
    }

    let chain = model.chain_indices(base_link, ee_link)?;
    let chain_set: HashSet<usize> = chain.iter().copied().collect();
    let transforms = link_transforms_tree_view(model, q)?;
    let ee_pose = end_effector_pose_tree(model, base_link, ee_link, &q.to_owned())?;
    let p_e = ee_pose.translation.clone();

    let actuated = model.actuated_indices();
    let mut actuated_map = HashMap::new();
    for (joint_index, &body_index) in actuated.iter().enumerate() {
        let _ = actuated_map.insert(body_index, joint_index);
    }

    let base = transforms.get(base_link).cloned().unwrap_or_else(|| {
        se3::from_rotation_translation(
            &Rotation3 { matrix: Array2::<T>::eye(3) },
            &Array1::<T>::zeros(3),
        )
    });
    let base_inv = se3::inverse(&base);

    let mut j = Array2::<T>::zeros((6, model.dof()));
    for &body_index in &chain {
        if !chain_set.contains(&body_index) {
            continue;
        }
        if matches!(model.joint_type(body_index), TreeJointType::Fixed) {
            continue;
        }
        let joint_index = actuated_map.get(&body_index).copied().ok_or_else(|| {
            KinematicsError::InvalidInput(format!("missing actuation index for body {body_index}"))
        })?;

        let parent_link = model.parent_link(body_index);
        let parent_tf = transforms.get(parent_link).cloned().unwrap_or_else(|| {
            se3::from_rotation_translation(
                &Rotation3 { matrix: Array2::<T>::eye(3) },
                &Array1::<T>::zeros(3),
            )
        });
        let origin = model.joint_origin(body_index);
        let joint_frame =
            se3::compose(&parent_tf, origin).map_err(|_| KinematicsError::NumericalInstability)?;
        let joint_in_base = se3::compose(&base_inv, &joint_frame)
            .map_err(|_| KinematicsError::NumericalInstability)?;

        let axis_local = model.joint_axis(body_index);
        let z_i = rotate_axis(&joint_in_base.rotation, axis_local);
        let p_i = joint_in_base.translation;
        let mut col = j.column_mut(joint_index);
        match model.joint_type(body_index) {
            TreeJointType::Revolute => {
                let diff = arr1(&[p_e[0] - p_i[0], p_e[1] - p_i[1], p_e[2] - p_i[2]]);
                let linear = cross3(&z_i.view(), &diff.view());
                col[0] = linear[0];
                col[1] = linear[1];
                col[2] = linear[2];
                col[3] = z_i[0];
                col[4] = z_i[1];
                col[5] = z_i[2];
            }
            TreeJointType::Prismatic => {
                col[0] = z_i[0];
                col[1] = z_i[1];
                col[2] = z_i[2];
                col[3] = T::zero();
                col[4] = T::zero();
                col[5] = T::zero();
            }
            TreeJointType::Fixed => {}
        }
    }
    Ok(j)
}

fn joint_motion<T: NabledReal>(
    joint_type: TreeJointType,
    axis: [T; 3],
    q: T,
) -> Result<Transform3<T>, KinematicsError> {
    match joint_type {
        TreeJointType::Fixed => Ok(se3::from_rotation_translation(
            &Rotation3 { matrix: Array2::<T>::eye(3) },
            &Array1::<T>::zeros(3),
        )),
        TreeJointType::Revolute => {
            let omega = Array1::from_vec(vec![axis[0] * q, axis[1] * q, axis[2] * q]);
            let rotation =
                so3::exp(&omega.view()).map_err(|_| KinematicsError::NumericalInstability)?;
            Ok(se3::from_rotation_translation(&rotation, &Array1::<T>::zeros(3)))
        }
        TreeJointType::Prismatic => Ok(se3::from_rotation_translation(
            &Rotation3 { matrix: Array2::<T>::eye(3) },
            &Array1::from_vec(vec![axis[0] * q, axis[1] * q, axis[2] * q]),
        )),
    }
}

fn rotate_axis<T: NabledReal>(rotation: &Rotation3<T>, axis: [T; 3]) -> Array1<T> {
    let local = Array1::from_vec(vec![axis[0], axis[1], axis[2]]);
    rotation.matrix.dot(&local)
}

fn cross3<T: NabledReal>(a: &ArrayView1<'_, T>, b: &ArrayView1<'_, T>) -> Array1<T> {
    Array1::from_vec(vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}
