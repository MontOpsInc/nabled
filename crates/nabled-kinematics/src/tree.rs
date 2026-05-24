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
    /// Lower/upper limits for actuated joint `joint_index` (0-based among actuated DOF).
    fn joint_limits(&self, joint_index: usize) -> Option<(T, T)> {
        let _ = joint_index;
        None
    }
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

#[cfg(test)]
pub(crate) mod y_branch_fixture {
    //! Minimal Y-branch tree model mirroring `Y_branch.urdf` for in-crate tests.

    use nabled_linalg::geometry::{Rotation3, Transform3, se3};
    use ndarray::{Array1, Array2};

    use super::*;

    pub(crate) struct YBranchCase {
        pub q:                    [f64; 3],
        pub left_ee_translation:  [f64; 3],
        pub right_ee_translation: [f64; 3],
    }

    pub(crate) fn fixture_cases() -> [YBranchCase; 2] {
        [
            YBranchCase {
                q:                    [0.0, 0.0, 0.0],
                left_ee_translation:  [2.0, 0.0, 0.0],
                right_ee_translation: [1.0, 1.0, 0.0],
            },
            YBranchCase {
                q:                    [0.0, std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2],
                left_ee_translation:  [2.0, 0.0, 0.0],
                right_ee_translation: [1.0, 1.0, 0.0],
            },
        ]
    }

    pub(crate) struct YBranchModel;

    impl YBranchModel {
        fn joint_origin(x: f64, y: f64, z: f64) -> Transform3<f64> {
            se3::from_rotation_translation(
                &Rotation3 { matrix: Array2::<f64>::eye(3) },
                &Array1::from_vec(vec![x, y, z]),
            )
        }
    }

    impl KinematicTreeModel<f64> for YBranchModel {
        fn validate_tree(&self) -> Result<(), KinematicsError> { Ok(()) }

        fn dof(&self) -> usize { 3 }

        fn actuated_indices(&self) -> Vec<usize> { vec![0, 1, 2] }

        fn topological_order(&self) -> Vec<usize> { vec![0, 1, 2] }

        fn body_index_for_link(&self, link_name: &str) -> Option<usize> {
            match link_name {
                "trunk" => Some(0),
                "left_ee" => Some(1),
                "right_ee" => Some(2),
                _ => None,
            }
        }

        fn parent_link(&self, body_index: usize) -> &str {
            match body_index {
                0 => "base",
                1 | 2 => "trunk",
                _ => panic!("invalid body index {body_index}"),
            }
        }

        fn child_link(&self, body_index: usize) -> &str {
            match body_index {
                0 => "trunk",
                1 => "left_ee",
                2 => "right_ee",
                _ => panic!("invalid body index {body_index}"),
            }
        }

        fn joint_type(&self, body_index: usize) -> TreeJointType {
            match body_index {
                0..=2 => TreeJointType::Revolute,
                _ => panic!("invalid body index {body_index}"),
            }
        }

        fn joint_origin(&self, body_index: usize) -> &Transform3<f64> {
            static ORIGIN_TRUNK: std::sync::OnceLock<Transform3<f64>> = std::sync::OnceLock::new();
            static ORIGIN_LEFT: std::sync::OnceLock<Transform3<f64>> = std::sync::OnceLock::new();
            static ORIGIN_RIGHT: std::sync::OnceLock<Transform3<f64>> = std::sync::OnceLock::new();
            match body_index {
                0 => ORIGIN_TRUNK.get_or_init(|| YBranchModel::joint_origin(1.0, 0.0, 0.0)),
                1 => ORIGIN_LEFT.get_or_init(|| YBranchModel::joint_origin(1.0, 0.0, 0.0)),
                2 => ORIGIN_RIGHT.get_or_init(|| YBranchModel::joint_origin(0.0, 1.0, 0.0)),
                _ => panic!("invalid body index {body_index}"),
            }
        }

        fn joint_axis(&self, _body_index: usize) -> [f64; 3] { [0.0, 0.0, 1.0] }

        fn chain_indices(
            &self,
            base_link: &str,
            ee_link: &str,
        ) -> Result<Vec<usize>, KinematicsError> {
            if base_link != "base" {
                return Err(KinematicsError::InvalidInput(format!("unknown base link {base_link}")));
            }
            match ee_link {
                "left_ee" => Ok(vec![0, 1]),
                "right_ee" => Ok(vec![0, 2]),
                other => Err(KinematicsError::InvalidInput(format!("unknown link {other}"))),
            }
        }

        fn joint_limits(&self, joint_index: usize) -> Option<(f64, f64)> {
            if joint_index < 3 {
                Some((-std::f64::consts::PI, std::f64::consts::PI))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::arr1;

    use super::y_branch_fixture::{YBranchModel, fixture_cases};
    use super::*;

    #[test]
    fn y_branch_fk_matches_fixture() {
        let model = YBranchModel;
        for case in fixture_cases() {
            let q = arr1(&case.q);
            let left = end_effector_pose_tree(&model, "base", "left_ee", &q).expect("left fk");
            let right = end_effector_pose_tree(&model, "base", "right_ee", &q).expect("right fk");
            assert_relative_eq!(left.translation[0], case.left_ee_translation[0], epsilon = 1e-10);
            assert_relative_eq!(left.translation[1], case.left_ee_translation[1], epsilon = 1e-10);
            assert_relative_eq!(left.translation[2], case.left_ee_translation[2], epsilon = 1e-10);
            assert_relative_eq!(right.translation[0], case.right_ee_translation[0], epsilon = 1e-10);
            assert_relative_eq!(right.translation[1], case.right_ee_translation[1], epsilon = 1e-10);
            assert_relative_eq!(right.translation[2], case.right_ee_translation[2], epsilon = 1e-10);
        }
    }

    #[test]
    fn y_branch_jacobian_matches_finite_difference() {
        let model = YBranchModel;
        let q = arr1(&fixture_cases()[0].q);
        let j_left = jacobian_tree(&model, "base", "left_ee", &q).expect("jacobian");
        assert_eq!(j_left.nrows(), 6);
        assert_eq!(j_left.ncols(), 3);

        let h = 1e-6;
        for col in 0..3 {
            let mut q_plus = q.clone();
            q_plus[col] += h;
            let pose_plus = end_effector_pose_tree(&model, "base", "left_ee", &q_plus).unwrap();
            let mut q_minus = q.clone();
            q_minus[col] -= h;
            let pose_minus = end_effector_pose_tree(&model, "base", "left_ee", &q_minus).unwrap();
            let deriv_x = (pose_plus.translation[0] - pose_minus.translation[0]) / (2.0 * h);
            let deriv_y = (pose_plus.translation[1] - pose_minus.translation[1]) / (2.0 * h);
            assert_relative_eq!(j_left[[0, col]], deriv_x, epsilon = 1e-5);
            assert_relative_eq!(j_left[[1, col]], deriv_y, epsilon = 1e-5);
        }
    }

    #[test]
    fn rejects_wrong_q_dimension() {
        let model = YBranchModel;
        let err = link_transforms_tree(&model, &arr1(&[0.0, 0.0])).unwrap_err();
        assert_eq!(err, KinematicsError::DimensionMismatch);
    }

    #[test]
    fn rejects_unknown_end_effector_link() {
        let model = YBranchModel;
        let q = arr1(&[0.0_f64, 0.0, 0.0]);
        let err = end_effector_pose_tree(&model, "base", "missing_link", &q).unwrap_err();
        assert!(matches!(err, KinematicsError::InvalidInput(_)));
    }

    #[test]
    fn rejects_disconnected_chain_indices() {
        let model = YBranchModel;
        let err = jacobian_tree(&model, "left_ee", "right_ee", &arr1(&[0.0, 0.0, 0.0])).unwrap_err();
        assert!(matches!(err, KinematicsError::InvalidInput(_)));
    }

    #[test]
    fn validate_tree_fails_on_invalid_model() {
        struct InvalidModel;
        impl KinematicTreeModel<f64> for InvalidModel {
            fn validate_tree(&self) -> Result<(), KinematicsError> {
                Err(KinematicsError::InvalidInput("bad tree".into()))
            }
            fn dof(&self) -> usize { 0 }
            fn actuated_indices(&self) -> Vec<usize> { vec![] }
            fn topological_order(&self) -> Vec<usize> { vec![] }
            fn body_index_for_link(&self, _: &str) -> Option<usize> { None }
            fn parent_link(&self, _: usize) -> &str { "base" }
            fn child_link(&self, _: usize) -> &str { "base" }
            fn joint_type(&self, _: usize) -> TreeJointType { TreeJointType::Fixed }
            fn joint_origin(&self, _: usize) -> &Transform3<f64> {
                static ID: std::sync::OnceLock<Transform3<f64>> = std::sync::OnceLock::new();
                ID.get_or_init(|| {
                    se3::from_rotation_translation(
                        &Rotation3 { matrix: Array2::<f64>::eye(3) },
                        &Array1::zeros(3),
                    )
                })
            }
            fn joint_axis(&self, _: usize) -> [f64; 3] { [0.0, 0.0, 1.0] }
            fn chain_indices(&self, _: &str, _: &str) -> Result<Vec<usize>, KinematicsError> {
                Ok(vec![])
            }
        }
        let err = link_transforms_tree(&InvalidModel, &arr1(&[])).unwrap_err();
        assert!(matches!(err, KinematicsError::InvalidInput(_)));
    }

    #[test]
    fn prismatic_joint_advances_translation() {
        struct PrismaticModel;
        impl KinematicTreeModel<f64> for PrismaticModel {
            fn validate_tree(&self) -> Result<(), KinematicsError> { Ok(()) }
            fn dof(&self) -> usize { 1 }
            fn actuated_indices(&self) -> Vec<usize> { vec![0] }
            fn topological_order(&self) -> Vec<usize> { vec![0] }
            fn body_index_for_link(&self, link: &str) -> Option<usize> {
                if link == "ee" { Some(0) } else { None }
            }
            fn parent_link(&self, _: usize) -> &str { "base" }
            fn child_link(&self, _: usize) -> &str { "ee" }
            fn joint_type(&self, _: usize) -> TreeJointType { TreeJointType::Prismatic }
            fn joint_origin(&self, _: usize) -> &Transform3<f64> {
                static ORIGIN: std::sync::OnceLock<Transform3<f64>> = std::sync::OnceLock::new();
                ORIGIN.get_or_init(|| {
                    se3::from_rotation_translation(
                        &Rotation3 { matrix: Array2::<f64>::eye(3) },
                        &Array1::zeros(3),
                    )
                })
            }
            fn joint_axis(&self, _: usize) -> [f64; 3] { [0.0, 0.0, 1.0] }
            fn chain_indices(&self, _: &str, _: &str) -> Result<Vec<usize>, KinematicsError> {
                Ok(vec![0])
            }
        }

        let model = PrismaticModel;
        let q = arr1(&[0.4_f64]);
        let pose = end_effector_pose_tree(&model, "base", "ee", &q).expect("fk");
        assert_relative_eq!(pose.translation[2], 0.4, epsilon = 1e-10);
        let j = jacobian_tree(&model, "base", "ee", &q).expect("jacobian");
        assert_relative_eq!(j[[2, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(j[[3, 0]], 0.0, epsilon = 1e-10);
    }
}
