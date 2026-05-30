//! Semi-implicit rigid-body simulation step.

use nabled_core::scalar::NabledReal;
use nabled_dynamics::fd::forward_dynamics_with_config;
use nabled_kinematics::fk::end_effector_pose;
use nabled_linalg::geometry::Transform3;
use nabled_linalg::lu::LuProviderScalar;
use ndarray::{Array1, ArrayView1};

use crate::SimError;
use crate::context::RobotContext;

/// Joint position and velocity state.
#[derive(Debug, Clone, PartialEq)]
pub struct SimState<T> {
    pub q:  Array1<T>,
    pub qd: Array1<T>,
}

impl<T: NabledReal> SimState<T> {
    #[must_use]
    pub fn new(q: Array1<T>, qd: Array1<T>) -> Self { Self { q, qd } }
}

/// Integration and logging options.
#[derive(Debug, Clone, PartialEq)]
pub struct SimConfig<T> {
    pub dt:          T,
    pub log_ee_pose: bool,
}

impl<T: NabledReal> SimConfig<T> {
    #[must_use]
    pub fn new(dt: T) -> Self { Self { dt, log_ee_pose: false } }
}

/// Output of one semi-implicit Euler step.
#[derive(Debug, Clone, PartialEq)]
pub struct SimStepResult<T> {
    pub state:   SimState<T>,
    pub qdd:     Array1<T>,
    pub ee_pose: Option<Transform3<T>>,
}

/// Advance `(q, qd)` by one semi-implicit Euler step using forward dynamics.
pub fn semi_implicit_step<T: NabledReal + Default + LuProviderScalar>(
    ctx: &RobotContext<T>,
    state: &SimState<T>,
    tau: &ArrayView1<'_, T>,
    config: &SimConfig<T>,
) -> Result<SimStepResult<T>, SimError> {
    ctx.validate()?;
    if tau.len() != ctx.chain.num_joints() {
        return Err(SimError::DimensionMismatch);
    }
    if state.q.len() != ctx.chain.num_joints() || state.qd.len() != ctx.chain.num_joints() {
        return Err(SimError::DimensionMismatch);
    }

    let qdd = forward_dynamics_with_config(
        &ctx.model,
        &ctx.chain,
        &state.q.view(),
        &state.qd.view(),
        tau,
        &ctx.dynamics_config,
    )?;

    let mut qd = state.qd.clone();
    qd += &(&qdd * config.dt);
    let mut q = state.q.clone();
    q += &(&qd * config.dt);

    let ee_pose = if config.log_ee_pose { Some(end_effector_pose(&ctx.chain, &q)?) } else { None };

    Ok(SimStepResult { state: SimState { q, qd }, qdd, ee_pose })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_dynamics::DynamicsConfig;
    use nabled_model::fixture::load_planar2r_json;
    use ndarray::{Array1, arr1};

    use super::*;
    use crate::context::RobotContext;

    #[test]
    fn semi_implicit_step_advances_state() {
        let fixture = load_planar2r_json().expect("fixture");
        let ctx = RobotContext::new(
            fixture.to_robot_model::<f64>().expect("model"),
            fixture.to_chain_spec::<f64>().expect("chain"),
            DynamicsConfig {
                gravity: fixture.gravity.unwrap_or([0.0, -9.81, 0.0]),
                ..DynamicsConfig::default()
            },
        );
        let state = SimState::new(arr1(&[0.2, 0.4]), Array1::zeros(2));
        let tau = Array1::zeros(2);
        let config = SimConfig::new(0.01);
        let result = semi_implicit_step(&ctx, &state, &tau.view(), &config).expect("step");
        assert!(result.state.q[0].is_finite());
        assert!(result.state.q[1].is_finite());
        assert_relative_eq!(result.state.qd[0], result.qdd[0] * 0.01, epsilon = 1e-12);
    }
}
