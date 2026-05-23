//! [`RobotModel`] adapter for tree kinematics.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::error::KinematicsError;
use nabled_kinematics::tree::{KinematicTreeModel, TreeJointType};
use nabled_linalg::geometry::Transform3;

use crate::joint::JointType;
use crate::robot::{RobotModel, extract_chain};

impl<T: NabledReal + Clone> KinematicTreeModel<T> for RobotModel<T> {
    fn validate_tree(&self) -> Result<(), KinematicsError> {
        self.validate().map_err(|err| KinematicsError::InvalidInput(err.to_string()))
    }

    fn dof(&self) -> usize {
        RobotModel::dof(self)
    }

    fn actuated_indices(&self) -> Vec<usize> {
        RobotModel::actuated_indices(self)
    }

    fn topological_order(&self) -> Vec<usize> {
        RobotModel::topological_order(self)
    }

    fn body_index_for_link(&self, link_name: &str) -> Option<usize> {
        RobotModel::body_index_for_link(self, link_name)
    }

    fn parent_link(&self, body_index: usize) -> &str {
        &self.joint(body_index).expect("valid body index").parent_link
    }

    fn child_link(&self, body_index: usize) -> &str {
        &self.joint(body_index).expect("valid body index").link.name
    }

    fn joint_type(&self, body_index: usize) -> TreeJointType {
        match self.joint(body_index).expect("valid body index").joint_type {
            JointType::Revolute => TreeJointType::Revolute,
            JointType::Prismatic => TreeJointType::Prismatic,
            JointType::Fixed => TreeJointType::Fixed,
        }
    }

    fn joint_origin(&self, body_index: usize) -> &Transform3<T> {
        &self.joint(body_index).expect("valid body index").joint_origin
    }

    fn joint_axis(&self, body_index: usize) -> [T; 3] {
        self.joint(body_index).expect("valid body index").axis.unit_vector()
    }

    fn chain_indices(&self, base_link: &str, ee_link: &str) -> Result<Vec<usize>, KinematicsError> {
        extract_chain(self, base_link, ee_link)
            .map_err(|err| KinematicsError::InvalidInput(err.to_string()))
    }
}
