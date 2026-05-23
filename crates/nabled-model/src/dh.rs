//! DH conversion to kinematic chain.

use std::collections::HashMap;

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::{ChainSpec, DhConvention, JointType as KinJointType};
use ndarray::Array1;

use crate::ModelError;
use crate::joint::JointType;
use crate::robot::{RobotModel, extract_chain};

fn chain_spec_from_indices<T: NabledReal + Default>(
    model: &RobotModel<T>,
    indices: &[usize],
) -> Result<ChainSpec<T>, ModelError> {
    model.validate()?;
    let mut joint_types = Vec::new();
    let mut a = Vec::new();
    let mut alpha = Vec::new();
    let mut d = Vec::new();
    let mut theta_offset = Vec::new();
    for &index in indices {
        let body = model.joint(index).ok_or(ModelError::EmptyModel)?;
        if matches!(body.joint_type, JointType::Fixed) {
            continue;
        }
        joint_types.push(match body.joint_type {
            JointType::Revolute => KinJointType::Revolute,
            JointType::Prismatic => KinJointType::Prismatic,
            JointType::Fixed => unreachable!(),
        });
        a.push(body.dh_a);
        alpha.push(body.dh_alpha);
        d.push(body.dh_d);
        theta_offset.push(body.dh_theta);
    }
    ChainSpec::from_dh(
        DhConvention::Standard,
        joint_types,
        Array1::from(a),
        Array1::from(alpha),
        Array1::from(d),
        Array1::from(theta_offset),
    )
    .map_err(|_| ModelError::DimensionMismatch)
}

/// Convert serial robot model to `ChainSpec` using full topological order.
pub fn to_chain_spec<T: NabledReal + Default>(
    model: &RobotModel<T>,
) -> Result<ChainSpec<T>, ModelError> {
    let order = model.topological_order();
    chain_spec_from_indices(model, &order)
}

/// Extract a serial `ChainSpec` between `base_link` and `ee_link`.
pub fn extract_chain_spec<T: NabledReal + Default>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
) -> Result<ChainSpec<T>, ModelError> {
    let indices = extract_chain(model, base_link, ee_link)?;
    chain_spec_from_indices(model, &indices)
}

/// Serial branch extracted for RNEA/forward dynamics on a tree model.
///
/// Whole-tree RNEA remains out of scope; extract a branch and slice `q` with
/// [`DynamicsBranchSpec::branch_q`].
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicsBranchSpec<T> {
    pub chain:     ChainSpec<T>,
    /// Indices into full model `q` (actuated ordering) for each joint of `chain`.
    pub q_indices: Vec<usize>,
}

impl<T: Clone> DynamicsBranchSpec<T> {
    /// Slice full model coordinates to branch serial coordinates.
    ///
    /// # Errors
    /// Returns [`ModelError::DimensionMismatch`] when `q.len()` differs from model DOF
    /// or the sliced vector length differs from `chain.num_joints()`.
    pub fn branch_q(&self, model: &RobotModel<T>, q: &Array1<T>) -> Result<Array1<T>, ModelError>
    where
        T: NabledReal,
    {
        if q.len() != model.dof() {
            return Err(ModelError::DimensionMismatch);
        }
        if self.q_indices.len() != self.chain.num_joints() {
            return Err(ModelError::DimensionMismatch);
        }
        Ok(Array1::from(self.q_indices.iter().map(|&index| q[index]).collect::<Vec<_>>()))
    }
}

/// Extract a serial branch between `base_link` and `ee_link` for dynamics.
///
/// Joint coordinates follow full-model actuated ordering; use [`DynamicsBranchSpec::branch_q`]
/// to obtain the serial `q` expected by RNEA/FD.
pub fn extract_chain_spec_for_dynamics<T: NabledReal + Default>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
) -> Result<DynamicsBranchSpec<T>, ModelError> {
    let indices = extract_chain(model, base_link, ee_link)?;
    let chain = chain_spec_from_indices(model, &indices)?;

    let actuated = model.actuated_indices();
    let actuated_map: HashMap<usize, usize> = actuated
        .iter()
        .enumerate()
        .map(|(joint_index, &body_index)| (body_index, joint_index))
        .collect();

    let mut q_indices = Vec::new();
    for &body_index in &indices {
        let body = model.joint(body_index).ok_or(ModelError::EmptyModel)?;
        if matches!(body.joint_type, JointType::Fixed) {
            continue;
        }
        let joint_index =
            actuated_map.get(&body_index).copied().ok_or(ModelError::DimensionMismatch)?;
        q_indices.push(joint_index);
    }

    if q_indices.len() != chain.num_joints() {
        return Err(ModelError::DimensionMismatch);
    }

    Ok(DynamicsBranchSpec { chain, q_indices })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::arr1;

    use super::*;
    use crate::joint::JointAxis;
    use crate::link::LinkSpec;
    use crate::origin::joint_origin_from_dh_scalars;
    use crate::robot::BodySpec;

    fn sample_body(name: &str, parent_link: &str) -> BodySpec<f64> {
        BodySpec {
            link:         LinkSpec { name: name.to_string() },
            parent_link:  parent_link.to_string(),
            joint_type:   JointType::Revolute,
            axis:         JointAxis::Z,
            limits:       None,
            inertial:     None,
            joint_origin: joint_origin_from_dh_scalars(1.0, 0.0, 0.0, 0.0).unwrap(),
            dh_a:         1.0,
            dh_alpha:     0.0,
            dh_d:         0.0,
            dh_theta:     0.0,
        }
    }

    #[test]
    fn extract_chain_matches_full_serial_model() {
        let mut model = RobotModel::new();
        let root = model.add_body(None, sample_body("link1", "base"));
        let _ = model.add_body(Some(root), sample_body("link2", "link1"));
        let full = to_chain_spec(&model).unwrap();
        let extracted = extract_chain_spec(&model, "base", "link2").unwrap();
        assert_eq!(full, extracted);
        assert_eq!(full.a, arr1(&[1.0, 1.0]));
    }

    #[test]
    fn dynamics_branch_slices_full_q() {
        let mut model = RobotModel::new();
        let root = model.add_body(None, sample_body("link1", "base"));
        let _ = model.add_body(Some(root), sample_body("link2", "link1"));
        let branch = extract_chain_spec_for_dynamics(&model, "base", "link2").unwrap();
        assert_eq!(branch.chain.num_joints(), 2);
        assert_eq!(branch.q_indices, vec![0, 1]);
        let q = arr1(&[0.2, 0.4]);
        let sliced = branch.branch_q(&model, &q).unwrap();
        assert_relative_eq!(sliced, q, epsilon = 1e-12);
    }

    #[test]
    fn dynamics_branch_rejects_dof_mismatch() {
        let mut model = RobotModel::new();
        let root = model.add_body(None, sample_body("link1", "base"));
        let _ = model.add_body(Some(root), sample_body("link2", "link1"));
        let branch = extract_chain_spec_for_dynamics(&model, "base", "link2").unwrap();
        let q = arr1(&[0.2]);
        assert!(branch.branch_q(&model, &q).is_err());
    }
}
