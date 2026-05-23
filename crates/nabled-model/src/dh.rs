//! DH conversion to kinematic chain.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::{ChainSpec, DhConvention, JointType as KinJointType};
use ndarray::Array1;

use crate::ModelError;
use crate::joint::JointType;
use crate::robot::RobotModel;

/// Convert serial robot model to `ChainSpec`.
pub fn to_chain_spec<T: NabledReal + Default>(
    model: &RobotModel<T>,
) -> Result<ChainSpec<T>, ModelError> {
    model.validate()?;
    let order = model.topological_order();
    let mut joint_types = Vec::new();
    let mut a = Vec::new();
    let mut alpha = Vec::new();
    let mut d = Vec::new();
    let mut theta_offset = Vec::new();
    for index in order {
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
