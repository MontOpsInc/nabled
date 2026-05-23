//! Validated robot context for serial-chain orchestration.

use nabled_core::scalar::NabledReal;
use nabled_dynamics::DynamicsConfig;
use nabled_kinematics::chain::ChainSpec;
use nabled_model::dh::{DynamicsBranchSpec, extract_chain_spec_for_dynamics};
use nabled_model::robot::RobotModel;

use crate::SimError;

/// Shared context binding model, serial chain, and dynamics configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RobotContext<T> {
    pub model:           RobotModel<T>,
    pub chain:           ChainSpec<T>,
    pub dynamics_config: DynamicsConfig<T>,
}

impl<T: NabledReal + Default> RobotContext<T> {
    /// Build context without validation; call [`Self::validate`] before use.
    #[must_use]
    pub fn new(
        model: RobotModel<T>,
        chain: ChainSpec<T>,
        dynamics_config: DynamicsConfig<T>,
    ) -> Self {
        Self { model, chain, dynamics_config }
    }

    /// Ensure model DOF matches serial chain joint count.
    pub fn validate(&self) -> Result<(), SimError> {
        if self.model.dof() != self.chain.num_joints() {
            return Err(SimError::DimensionMismatch);
        }
        self.chain.validate()?;
        self.model.validate()?;
        Ok(())
    }

    /// Extract a serial branch for branch dynamics (compose-down to `nabled-model`).
    pub fn extract_chain_for_dynamics(
        &self,
        base_link: &str,
        ee_link: &str,
    ) -> Result<DynamicsBranchSpec<T>, SimError> {
        extract_chain_spec_for_dynamics(&self.model, base_link, ee_link).map_err(SimError::from)
    }
}

#[cfg(test)]
mod tests {
    use nabled_kinematics::chain::ChainSpec;
    use nabled_model::fixture::load_planar2r_json;
    use ndarray::arr1;

    use super::*;

    #[test]
    fn validates_matching_dof() {
        let fixture = load_planar2r_json().expect("fixture");
        let model = fixture.to_robot_model::<f64>().expect("model");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let config = DynamicsConfig {
            gravity: fixture.gravity.unwrap_or([0.0, -9.81, 0.0]),
            ..DynamicsConfig::default()
        };
        let ctx = RobotContext::new(model, chain, config);
        ctx.validate().expect("valid context");
    }

    #[test]
    fn rejects_dof_mismatch() {
        let fixture = load_planar2r_json().expect("fixture");
        let model = fixture.to_robot_model::<f64>().expect("model");
        let chain = ChainSpec::from_dh(
            nabled_kinematics::chain::DhConvention::Standard,
            vec![
                nabled_kinematics::chain::JointType::Revolute,
                nabled_kinematics::chain::JointType::Revolute,
                nabled_kinematics::chain::JointType::Revolute,
            ],
            arr1(&[1.0, 1.0, 1.0]),
            arr1(&[0.0, 0.0, 0.0]),
            arr1(&[0.0, 0.0, 0.0]),
            arr1(&[0.0, 0.0, 0.0]),
        )
        .expect("3-dof chain");
        let config = DynamicsConfig { gravity: [0.0, -9.81, 0.0], ..DynamicsConfig::default() };
        let ctx = RobotContext::new(model, chain, config);
        assert_eq!(ctx.validate(), Err(SimError::DimensionMismatch));
    }
}
