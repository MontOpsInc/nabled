//! Batch damped least-squares IK over a time grid.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::{ChainSpec, JointLimits};
use nabled_kinematics::fk::fk_view;
use nabled_kinematics::ik::{
    IkConfig, IkResult, inverse_kinematics_dls_with_limits,
    inverse_kinematics_tree_dls_with_limits, pose_error,
};
use nabled_kinematics::tree::KinematicTreeModel;
use nabled_linalg::lu::LuProviderScalar;
use ndarray::{Array1, ArrayView1};

use crate::SimError;

fn error_norm<T: NabledReal>(error: &Array1<T>) -> T {
    error.iter().map(|v| *v * *v).fold(T::zero(), |acc, v| acc + v).sqrt()
}

/// IK solver options for trajectory batching.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryIkConfig<T> {
    pub ik_config: IkConfig<T>,
    pub limits:    Option<Vec<JointLimits<T>>>,
}

impl<T: NabledReal> Default for TrajectoryIkConfig<T> {
    fn default() -> Self { Self { ik_config: IkConfig::default(), limits: None } }
}

/// One time-grid IK solve with pose verification.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryIkStep<T> {
    pub t:               T,
    pub result:          IkResult<T>,
    pub pose_error_norm: T,
}

/// Aggregated trajectory IK output.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryIkResult<T> {
    pub steps:           Vec<TrajectoryIkStep<T>>,
    pub max_error:       T,
    pub converged_count: usize,
}

/// Interpolated joint-space targets over a time grid, solved with serial DLS IK.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryIk<T> {
    pub times:   Vec<T>,
    pub q_start: Array1<T>,
    pub q_end:   Array1<T>,
}

impl<T: NabledReal + LuProviderScalar> TrajectoryIk<T> {
    /// Solve IK at each time sample by linearly blending `q_start` → `q_end` FK targets.
    pub fn solve(
        &self,
        chain: &ChainSpec<T>,
        config: &TrajectoryIkConfig<T>,
    ) -> Result<TrajectoryIkResult<T>, SimError> {
        if self.times.is_empty() {
            return Err(SimError::InvalidInput("time grid cannot be empty".to_string()));
        }
        if self.q_start.len() != chain.num_joints() || self.q_end.len() != chain.num_joints() {
            return Err(SimError::DimensionMismatch);
        }

        let duration = self.times[self.times.len() - 1];
        if duration.is_zero() {
            return Err(SimError::InvalidInput("time grid duration must be positive".to_string()));
        }

        let limits = config.limits.as_deref();
        let mut q_seed = self.q_start.clone();
        let mut max_error = T::zero();
        let mut converged_count = 0_usize;
        let mut steps = Vec::with_capacity(self.times.len());

        for &t in &self.times {
            let blend = t / duration;
            let q_target = &self.q_start + &((&self.q_end - &self.q_start) * blend);
            let target = fk_view(chain, &q_target.view())?;
            let result = inverse_kinematics_dls_with_limits(
                chain,
                &q_seed,
                &target,
                &config.ik_config,
                limits,
            )?;
            let achieved = fk_view(chain, &result.q.view())?;
            let err = pose_error(&achieved, &target)?;
            let err_norm = error_norm(&err);
            if err_norm > max_error {
                max_error = err_norm;
            }
            if result.converged {
                converged_count += 1;
            }
            q_seed.clone_from(&result.q);
            steps.push(TrajectoryIkStep { t, result, pose_error_norm: err_norm });
        }

        Ok(TrajectoryIkResult { steps, max_error, converged_count })
    }

    /// Solve IK from explicit SE(3) targets (one per time sample).
    pub fn solve_targets(
        times: &[T],
        chain: &ChainSpec<T>,
        q_init: &ArrayView1<'_, T>,
        targets: &[nabled_linalg::geometry::Transform3<T>],
        config: &TrajectoryIkConfig<T>,
    ) -> Result<TrajectoryIkResult<T>, SimError> {
        if times.len() != targets.len() {
            return Err(SimError::DimensionMismatch);
        }
        if times.is_empty() {
            return Err(SimError::InvalidInput("time grid cannot be empty".to_string()));
        }
        if q_init.len() != chain.num_joints() {
            return Err(SimError::DimensionMismatch);
        }

        let limits = config.limits.as_deref();
        let mut q_seed = q_init.to_owned();
        let mut max_error = T::zero();
        let mut converged_count = 0_usize;
        let mut steps = Vec::with_capacity(times.len());

        for (&t, target) in times.iter().zip(targets.iter()) {
            let result = inverse_kinematics_dls_with_limits(
                chain,
                &q_seed,
                target,
                &config.ik_config,
                limits,
            )?;
            let achieved = fk_view(chain, &result.q.view())?;
            let err = pose_error(&achieved, target)?;
            let err_norm = error_norm(&err);
            if err_norm > max_error {
                max_error = err_norm;
            }
            if result.converged {
                converged_count += 1;
            }
            q_seed.clone_from(&result.q);
            steps.push(TrajectoryIkStep { t, result, pose_error_norm: err_norm });
        }

        Ok(TrajectoryIkResult { steps, max_error, converged_count })
    }
}

/// Tree-model trajectory IK over a time grid (full-model `q` in actuated order).
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryTreeIk<T> {
    pub times:     Vec<T>,
    pub q_start:   Array1<T>,
    pub q_end:     Array1<T>,
    pub base_link: String,
    pub ee_link:   String,
}

impl<T: NabledReal + LuProviderScalar> TrajectoryTreeIk<T> {
    /// Solve tree DLS IK at each time sample by blending joint targets then FK-ing to SE(3).
    pub fn solve<M: KinematicTreeModel<T>>(
        &self,
        model: &M,
        config: &TrajectoryIkConfig<T>,
    ) -> Result<TrajectoryIkResult<T>, SimError> {
        if self.times.is_empty() {
            return Err(SimError::InvalidInput("time grid cannot be empty".to_string()));
        }
        if self.q_start.len() != model.dof() || self.q_end.len() != model.dof() {
            return Err(SimError::DimensionMismatch);
        }

        let duration = self.times[self.times.len() - 1];
        if duration.is_zero() {
            return Err(SimError::InvalidInput("time grid duration must be positive".to_string()));
        }

        let limits = config.limits.as_deref();
        let mut q_seed = self.q_start.clone();
        let mut max_error = T::zero();
        let mut converged_count = 0_usize;
        let mut steps = Vec::with_capacity(self.times.len());

        for &t in &self.times {
            let blend = t / duration;
            let q_target = &self.q_start + &((&self.q_end - &self.q_start) * blend);
            let target = nabled_kinematics::tree::end_effector_pose_tree(
                model,
                &self.base_link,
                &self.ee_link,
                &q_target,
            )?;
            let result = inverse_kinematics_tree_dls_with_limits(
                model,
                &self.base_link,
                &self.ee_link,
                &q_seed,
                &target,
                &config.ik_config,
                limits,
            )?;
            let achieved = nabled_kinematics::tree::end_effector_pose_tree(
                model,
                &self.base_link,
                &self.ee_link,
                &result.q,
            )?;
            let err = pose_error(&achieved, &target)?;
            let err_norm = error_norm(&err);
            if err_norm > max_error {
                max_error = err_norm;
            }
            if result.converged {
                converged_count += 1;
            }
            q_seed.clone_from(&result.q);
            steps.push(TrajectoryIkStep { t, result, pose_error_norm: err_norm });
        }

        Ok(TrajectoryIkResult { steps, max_error, converged_count })
    }
}

#[cfg(test)]
mod tests {
    use nabled_kinematics::fk::{end_effector_pose, fk_view};
    use nabled_model::fixture::{load_planar2r_json, load_six_dof_dh_json, load_y_branch_json};
    use nabled_model::urdf::from_urdf_file;
    use ndarray::arr1;

    use super::*;

    #[test]
    fn trajectory_ik_converges_on_grid() {
        let fixture = load_six_dof_dh_json().expect("fixture");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let trajectory = TrajectoryIk {
            times:   (0..=5).map(|i| f64::from(i) * 0.1).collect(),
            q_start: arr1(&[0.0, -0.3, 0.5, 0.2, -0.1, 0.4]),
            q_end:   arr1(&[0.3, -0.2, 0.4, 0.1, 0.0, 0.5]),
        };
        let config = TrajectoryIkConfig {
            ik_config: IkConfig { max_iterations: 200, tolerance: 1e-3, ..IkConfig::default() },
            limits:    None,
        };
        let result = trajectory.solve(&chain, &config).expect("solve");
        assert_eq!(result.steps.len(), 6);
        assert!(result.max_error < 1e-2);
    }

    #[test]
    fn trajectory_ik_rejects_empty_time_grid() {
        let fixture = load_six_dof_dh_json().expect("fixture");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let trajectory =
            TrajectoryIk { times: vec![], q_start: arr1(&[0.0; 6]), q_end: arr1(&[0.1; 6]) };
        let config = TrajectoryIkConfig::default();
        assert_eq!(
            trajectory.solve(&chain, &config),
            Err(SimError::InvalidInput("time grid cannot be empty".to_string()))
        );
    }

    #[test]
    fn trajectory_ik_rejects_zero_duration() {
        let fixture = load_six_dof_dh_json().expect("fixture");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let trajectory = TrajectoryIk {
            times:   vec![0.0, 0.0],
            q_start: arr1(&[0.0; 6]),
            q_end:   arr1(&[0.1; 6]),
        };
        let config = TrajectoryIkConfig::default();
        assert_eq!(
            trajectory.solve(&chain, &config),
            Err(SimError::InvalidInput("time grid duration must be positive".to_string()))
        );
    }

    #[test]
    fn trajectory_ik_rejects_dof_mismatch() {
        let fixture = load_six_dof_dh_json().expect("fixture");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let trajectory = TrajectoryIk {
            times:   vec![0.0, 1.0],
            q_start: arr1(&[0.0; 5]),
            q_end:   arr1(&[0.1; 6]),
        };
        let config = TrajectoryIkConfig::default();
        assert_eq!(trajectory.solve(&chain, &config), Err(SimError::DimensionMismatch));
    }

    #[test]
    fn trajectory_ik_solve_targets_on_planar2r() {
        let fixture = load_planar2r_json().expect("fixture");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let q_a = arr1(&[0.0_f64, 0.0]);
        let q_b = arr1(&[0.5, -0.3]);
        let targets = [&q_a, &q_b]
            .iter()
            .map(|q| end_effector_pose(&chain, *q).expect("fk"))
            .collect::<Vec<_>>();
        let times = [0.0_f64, 1.0];
        let config = TrajectoryIkConfig {
            ik_config: IkConfig { max_iterations: 200, tolerance: 1e-3, ..IkConfig::default() },
            limits:    None,
        };
        let result = TrajectoryIk::solve_targets(&times, &chain, &q_a.view(), &targets, &config)
            .expect("solve targets");
        assert_eq!(result.steps.len(), 2);
        assert!(result.max_error < 1e-2);
    }

    #[test]
    fn trajectory_ik_solve_targets_on_six_dof() {
        let fixture = load_six_dof_dh_json().expect("fixture");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let q_init = arr1(&[0.0, -0.3, 0.5, 0.2, -0.1, 0.4]);
        let q_goal = arr1(&[0.3, -0.2, 0.4, 0.1, 0.0, 0.5]);
        let targets = [q_init.view(), q_goal.view()]
            .iter()
            .map(|q| fk_view(&chain, q).expect("fk"))
            .collect::<Vec<_>>();
        let times = [0.0_f64, 1.0];
        let config = TrajectoryIkConfig {
            ik_config: IkConfig { max_iterations: 200, tolerance: 1e-3, ..IkConfig::default() },
            limits:    None,
        };
        let result = TrajectoryIk::solve_targets(&times, &chain, &q_init.view(), &targets, &config)
            .expect("solve targets");
        assert_eq!(result.steps.len(), 2);
        assert!(result.max_error < 1e-2);
    }

    #[test]
    fn trajectory_ik_solve_targets_rejects_mismatched_grid() {
        let fixture = load_planar2r_json().expect("fixture");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let q = arr1(&[0.0_f64, 0.0]);
        let target = end_effector_pose(&chain, &q).expect("fk");
        let config = TrajectoryIkConfig::default();
        assert_eq!(
            TrajectoryIk::solve_targets(&[0.0, 1.0], &chain, &q.view(), &[target], &config),
            Err(SimError::DimensionMismatch)
        );
    }

    #[test]
    fn trajectory_tree_ik_on_y_branch() {
        let fixture = load_y_branch_json().expect("fixture");
        let urdf_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nabled/tests/fixtures/physical_ai/Y_branch.urdf"
        );
        let model = from_urdf_file::<f64>(urdf_path).expect("urdf");
        let case = &fixture.cases[0];
        let trajectory = TrajectoryTreeIk {
            times:     vec![0.0_f64, 0.5, 1.0],
            q_start:   arr1(&case.q),
            q_end:     arr1(&[0.3, 0.5, -0.2]),
            base_link: "base".to_string(),
            ee_link:   "left_ee".to_string(),
        };
        let config = TrajectoryIkConfig {
            ik_config: IkConfig { max_iterations: 300, tolerance: 1e-3, ..IkConfig::default() },
            limits:    None,
        };
        let result = trajectory.solve(&model, &config).expect("tree ik");
        assert_eq!(result.steps.len(), 3);
        assert!(result.max_error < 1e-2);
    }

    #[test]
    fn trajectory_tree_ik_rejects_empty_time_grid() {
        let urdf_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nabled/tests/fixtures/physical_ai/Y_branch.urdf"
        );
        let model = from_urdf_file::<f64>(urdf_path).expect("urdf");
        let trajectory = TrajectoryTreeIk {
            times:     vec![],
            q_start:   arr1(&[0.0; 3]),
            q_end:     arr1(&[0.1; 3]),
            base_link: "base".to_string(),
            ee_link:   "right_ee".to_string(),
        };
        let config = TrajectoryIkConfig::default();
        assert_eq!(
            trajectory.solve(&model, &config),
            Err(SimError::InvalidInput("time grid cannot be empty".to_string()))
        );
    }
}
