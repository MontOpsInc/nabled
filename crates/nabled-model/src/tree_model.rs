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

    fn dof(&self) -> usize { RobotModel::dof(self) }

    fn actuated_indices(&self) -> Vec<usize> { RobotModel::actuated_indices(self) }

    fn topological_order(&self) -> Vec<usize> { RobotModel::topological_order(self) }

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

    fn joint_limits(&self, joint_index: usize) -> Option<(T, T)> {
        self.limits_for_joint(joint_index).map(|limits| (limits.lower, limits.upper))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_kinematics::tree::{KinematicTreeModel, TreeJointType};

    use super::*;
    use crate::urdf::from_urdf_file;

    fn y_branch_model() -> RobotModel<f64> {
        let urdf_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nabled/tests/fixtures/physical_ai/Y_branch.urdf"
        );
        from_urdf_file(urdf_path).expect("Y-branch URDF")
    }

    #[test]
    fn y_branch_tree_model_metadata() {
        let model = y_branch_model();
        model.validate_tree().expect("valid tree");
        assert_eq!(model.dof(), 3);
        assert_eq!(model.actuated_indices(), vec![0, 1, 2]);
        assert_eq!(model.topological_order(), vec![0, 1, 2]);
        assert_eq!(model.body_index_for_link("left_ee"), Some(1));
        assert_eq!(model.body_index_for_link("right_ee"), Some(2));
    }

    #[test]
    fn y_branch_chain_indices_for_branches() {
        let model = y_branch_model();
        let left = model.chain_indices("base", "left_ee").expect("left chain");
        let right = model.chain_indices("base", "right_ee").expect("right chain");
        assert_eq!(left, vec![0, 1]);
        assert_eq!(right, vec![0, 2]);
    }

    #[test]
    fn y_branch_joint_limits_from_urdf() {
        let model = y_branch_model();
        for joint_index in 0..model.dof() {
            let (lower, upper) = model.joint_limits(joint_index).expect("limits");
            assert!(lower < upper);
            assert_relative_eq!(lower, -std::f64::consts::PI, epsilon = 1e-4);
            assert_relative_eq!(upper, std::f64::consts::PI, epsilon = 1e-4);
        }
    }

    #[test]
    fn rejects_unknown_link_in_chain_indices() {
        let model = y_branch_model();
        let err = model.chain_indices("base", "missing").unwrap_err();
        assert!(matches!(err, KinematicsError::InvalidInput(_)));
    }

    #[test]
    fn trait_accessors_exercise_all_body_fields() {
        let model = y_branch_model();
        for &body_index in &model.actuated_indices() {
            assert!(!model.parent_link(body_index).is_empty());
            assert!(!model.child_link(body_index).is_empty());
            assert!(matches!(
                model.joint_type(body_index),
                TreeJointType::Revolute | TreeJointType::Prismatic | TreeJointType::Fixed
            ));
            let origin = model.joint_origin(body_index);
            assert!(origin.translation.iter().all(|v| v.is_finite()));
            let axis = model.joint_axis(body_index);
            let axis_len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            assert_relative_eq!(axis_len, 1.0, epsilon = 1e-6);
        }
    }
}
