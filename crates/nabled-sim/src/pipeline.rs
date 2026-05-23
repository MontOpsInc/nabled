//! Opt-in multi-stage Physical AI pipelines.

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu::LuProviderScalar;
use nabled_ml::stats::rolling::rolling_covariance;
use ndarray::{Array1, Array2};

use crate::SimError;
use crate::context::RobotContext;
use crate::sim::{SimConfig, SimState, semi_implicit_step};

/// Logged torque rows from a simulation run.
#[derive(Debug, Clone, PartialEq)]
pub struct TorqueLog<T> {
    pub samples: Array2<T>,
}

impl<T: NabledReal> TorqueLog<T> {
    #[must_use]
    pub fn len(&self) -> usize { self.samples.nrows() }

    #[must_use]
    pub fn is_empty(&self) -> bool { self.samples.nrows() == 0 }
}

/// Builder for cross-crate workflows (sim → stats, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalAiPipeline<T> {
    pub ctx:               RobotContext<T>,
    pub sim_config:        SimConfig<T>,
    pub torque_log_window: usize,
}

impl<T: NabledReal + Default + LuProviderScalar> PhysicalAiPipeline<T> {
    #[must_use]
    pub fn new(ctx: RobotContext<T>, sim_config: SimConfig<T>, torque_log_window: usize) -> Self {
        Self { ctx, sim_config, torque_log_window }
    }

    /// Run `steps` semi-implicit steps, logging applied joint torques each step.
    pub fn run_sim_with_torque_log(
        &self,
        initial: &SimState<T>,
        tau_fn: impl Fn(usize) -> Array1<T>,
        steps: usize,
    ) -> Result<(SimState<T>, TorqueLog<T>), SimError> {
        self.ctx.validate()?;
        let dof = self.ctx.chain.num_joints();
        let mut state = initial.clone();
        let mut log = Array2::zeros((steps, dof));
        for step in 0..steps {
            let tau = tau_fn(step);
            if tau.len() != dof {
                return Err(SimError::DimensionMismatch);
            }
            log.row_mut(step).assign(&tau);
            let result = semi_implicit_step(&self.ctx, &state, &tau.view(), &self.sim_config)?;
            state = result.state;
        }
        Ok((state, TorqueLog { samples: log }))
    }

    /// Rolling covariance over logged torques (compose-down to `nabled-ml::stats`).
    pub fn rolling_torque_covariance(log: &TorqueLog<T>, window: usize) -> Array2<T> {
        rolling_covariance(&log.samples.view(), window)
    }

    /// Convenience: simulate, log torques, and return rolling covariance.
    pub fn sim_torque_rolling_covariance(
        &self,
        initial: &SimState<T>,
        tau_fn: impl Fn(usize) -> Array1<T>,
        steps: usize,
    ) -> Result<(SimState<T>, Array2<T>), SimError> {
        let (state, log) = self.run_sim_with_torque_log(initial, tau_fn, steps)?;
        let cov = Self::rolling_torque_covariance(&log, self.torque_log_window);
        Ok((state, cov))
    }
}

#[cfg(test)]
mod tests {
    use nabled_dynamics::DynamicsConfig;
    use nabled_model::fixture::load_planar2r_json;
    use ndarray::{Array1, arr1};

    use super::*;
    use crate::context::RobotContext;

    #[test]
    #[expect(clippy::cast_possible_truncation)]
    fn sim_torque_pipeline_produces_bounded_covariance() {
        let fixture = load_planar2r_json().expect("fixture");
        let ctx = RobotContext::new(
            fixture.to_robot_model::<f64>().expect("model"),
            fixture.to_chain_spec::<f64>().expect("chain"),
            DynamicsConfig {
                gravity: fixture.gravity.unwrap_or([0.0, -9.81, 0.0]),
                ..DynamicsConfig::default()
            },
        );
        let pipeline = PhysicalAiPipeline::new(ctx, SimConfig::new(0.01), 5);
        let initial = SimState::new(arr1(&[0.2, 0.4]), Array1::zeros(2));
        let (_, cov) = pipeline
            .sim_torque_rolling_covariance(
                &initial,
                |step| {
                    let t = f64::from(step as u32) * 0.01;
                    arr1(&[0.5 * t.sin(), 0.2 * t.cos()])
                },
                20,
            )
            .expect("pipeline");
        assert_eq!(cov.nrows(), 20);
        let last = cov.row(19);
        assert!(last.iter().all(|v| v.is_finite()));
        assert!(last.iter().any(|v| v.abs() > 0.0) || cov[[19, 3]].abs() > 0.0);
    }
}
