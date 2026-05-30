//! Inverse kinematics (DLS) and pose error.

use nabled_core::scalar::NabledReal;
use nabled_linalg::geometry::{Transform3, se3};
use nabled_linalg::lu::{self, LuProviderScalar};
use ndarray::{Array1, Array2, ArrayView1};

use crate::chain::{ChainSpec, JointLimits};
use crate::error::KinematicsError;
use crate::fk::fk_view;
use crate::jacobian::jacobian_view;
use crate::tree::{KinematicTreeModel, end_effector_pose_tree, jacobian_tree_view};

/// IK solver configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct IkConfig<T> {
    pub max_iterations: usize,
    pub tolerance:      T,
    pub damping:        T,
    pub step_scale:     T,
}

impl<T: NabledReal> Default for IkConfig<T> {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            tolerance:      T::from_f64(1e-4).unwrap_or(T::zero()),
            damping:        T::from_f64(0.01).unwrap_or(T::zero()),
            step_scale:     T::one(),
        }
    }
}

/// Result of damped least-squares IK.
#[derive(Debug, Clone, PartialEq)]
pub struct IkResult<T> {
    pub q:           Array1<T>,
    pub iterations:  usize,
    pub final_error: T,
    pub converged:   bool,
}

/// Reusable workspace for DLS IK hot paths.
#[derive(Debug, Clone, PartialEq)]
pub struct IkWorkspace<T> {
    jacobian:   Array2<T>,
    error:      Array1<T>,
    task_error: Array1<T>,
    jtj:        Array2<T>,
    jte:        Array1<T>,
    dq:         Array1<T>,
    q:          Array1<T>,
}

impl<T: NabledReal> IkWorkspace<T> {
    /// Allocate workspace buffers for a chain with `num_joints` DOF.
    #[must_use]
    pub fn new(num_joints: usize) -> Self {
        Self {
            jacobian:   Array2::zeros((6, num_joints)),
            error:      Array1::zeros(6),
            task_error: Array1::zeros(6),
            jtj:        Array2::zeros((num_joints, num_joints)),
            jte:        Array1::zeros(num_joints),
            dq:         Array1::zeros(num_joints),
            q:          Array1::zeros(num_joints),
        }
    }
}

/// Pose error twist via `geometry::se3::log` (spatial frame).
pub fn pose_error<T: NabledReal>(
    current: &Transform3<T>,
    target: &Transform3<T>,
) -> Result<Array1<T>, KinematicsError> {
    let relative = se3::compose(target, &se3::inverse(current))
        .map_err(|_| KinematicsError::NumericalInstability)?;
    se3::log(&relative).map_err(|_| KinematicsError::NumericalInstability)
}

/// Pose error into caller buffer.
pub fn pose_error_into<T: NabledReal>(
    current: &Transform3<T>,
    target: &Transform3<T>,
    output: &mut Array1<T>,
) -> Result<(), KinematicsError> {
    let err = pose_error(current, target)?;
    if output.len() != 6 {
        return Err(KinematicsError::DimensionMismatch);
    }
    output.assign(&err);
    Ok(())
}

fn error_norm<T: NabledReal>(error: &Array1<T>) -> T {
    error.iter().map(|v| *v * *v).fold(T::zero(), |acc, v| acc + v).sqrt()
}

/// Map `se3::log` twist `[angular; translation]` to Jacobian task order `[linear; angular]`.
fn task_error_for_jacobian<T: NabledReal>(error: &Array1<T>, output: &mut Array1<T>) {
    for i in 0..3 {
        output[i] = error[i + 3];
        output[i + 3] = error[i];
    }
}

fn validate_limits<T: PartialOrd>(
    q: &ArrayView1<'_, T>,
    limits: &[JointLimits<T>],
) -> Result<(), KinematicsError> {
    let n = q.len().min(limits.len());
    for i in 0..n {
        if q[i] < limits[i].lower || q[i] > limits[i].upper {
            return Err(KinematicsError::JointLimitViolation(i));
        }
    }
    Ok(())
}

fn clip_to_limits<T: NabledReal>(q: &mut Array1<T>, limits: &[JointLimits<T>]) {
    let n = q.len().min(limits.len());
    for i in 0..n {
        q[i] = q[i].max(limits[i].lower).min(limits[i].upper);
    }
}

fn dls_step<T: NabledReal + LuProviderScalar>(
    jacobian: &Array2<T>,
    error: &Array1<T>,
    damping: T,
    jtj: &mut Array2<T>,
    jte: &mut Array1<T>,
    dq: &mut Array1<T>,
) -> Result<(), KinematicsError> {
    let n = jtj.nrows();
    let jt = jacobian.t();
    jtj.assign(&jt.dot(jacobian));
    let lambda_sq = damping * damping;
    for i in 0..n {
        jtj[[i, i]] += lambda_sq;
    }
    jte.assign(&jt.dot(error));
    *dq = lu::solve(jtj, jte).map_err(|_| KinematicsError::NumericalInstability)?;
    Ok(())
}

/// Damped least-squares IK with optional joint limits.
pub fn inverse_kinematics_dls_with_limits<T: NabledReal + LuProviderScalar>(
    chain: &ChainSpec<T>,
    q_init: &Array1<T>,
    target: &Transform3<T>,
    config: &IkConfig<T>,
    limits: Option<&[JointLimits<T>]>,
) -> Result<IkResult<T>, KinematicsError> {
    let mut workspace = IkWorkspace::new(chain.num_joints());
    let mut output = Array1::zeros(chain.num_joints());
    inverse_kinematics_dls_into(chain, q_init, target, config, limits, &mut workspace, &mut output)
}

/// Damped least-squares IK returning joint configuration.
pub fn inverse_kinematics_dls<T: NabledReal + LuProviderScalar>(
    chain: &ChainSpec<T>,
    q_init: &Array1<T>,
    target: &Transform3<T>,
    config: &IkConfig<T>,
) -> Result<Array1<T>, KinematicsError> {
    inverse_kinematics_dls_with_limits(chain, q_init, target, config, None).map(|result| result.q)
}

/// DLS IK into caller buffers with reusable workspace.
pub fn inverse_kinematics_dls_into<T: NabledReal + LuProviderScalar>(
    chain: &ChainSpec<T>,
    q_init: &Array1<T>,
    target: &Transform3<T>,
    config: &IkConfig<T>,
    limits: Option<&[JointLimits<T>]>,
    workspace: &mut IkWorkspace<T>,
    output: &mut Array1<T>,
) -> Result<IkResult<T>, KinematicsError> {
    chain.validate()?;
    let n = chain.num_joints();
    if q_init.len() != n || output.len() != n {
        return Err(KinematicsError::DimensionMismatch);
    }
    if workspace.jacobian.ncols() != n || workspace.jtj.nrows() != n {
        *workspace = IkWorkspace::new(n);
    }
    if let Some(limits) = limits {
        if limits.len() != n {
            return Err(KinematicsError::DimensionMismatch);
        }
        validate_limits(&q_init.view(), limits)?;
    }

    workspace.q.assign(q_init);
    let mut iterations = 0_usize;
    let mut final_error = T::zero();
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let current = fk_view(chain, &workspace.q.view())?;
        pose_error_into(&current, target, &mut workspace.error)?;
        final_error = error_norm(&workspace.error);
        if final_error <= config.tolerance {
            converged = true;
            break;
        }

        let j = jacobian_view(chain, &workspace.q.view())?;
        workspace.jacobian.assign(&j);
        task_error_for_jacobian(&workspace.error, &mut workspace.task_error);
        dls_step(
            &workspace.jacobian,
            &workspace.task_error,
            config.damping,
            &mut workspace.jtj,
            &mut workspace.jte,
            &mut workspace.dq,
        )?;

        for i in 0..n {
            workspace.q[i] += config.step_scale * workspace.dq[i];
        }
        if let Some(limits) = limits {
            clip_to_limits(&mut workspace.q, limits);
        }
    }

    if !converged {
        return Err(KinematicsError::ConvergenceFailed);
    }

    output.assign(&workspace.q);
    Ok(IkResult { q: output.clone(), iterations, final_error, converged })
}

fn tree_limits_from_model<M, T>(model: &M, dof: usize) -> Option<Vec<JointLimits<T>>>
where
    M: KinematicTreeModel<T>,
    T: NabledReal,
{
    let mut limits = Vec::with_capacity(dof);
    for joint_index in 0..dof {
        let (lower, upper) = model.joint_limits(joint_index)?;
        limits.push(JointLimits { lower, upper });
    }
    Some(limits)
}

/// Damped least-squares tree IK returning full actuated `q` (`model.actuated_indices()` order).
pub fn inverse_kinematics_tree_dls<M, T>(
    model: &M,
    base_link: &str,
    ee_link: &str,
    q_init: &Array1<T>,
    target: &Transform3<T>,
    config: &IkConfig<T>,
) -> Result<Array1<T>, KinematicsError>
where
    M: KinematicTreeModel<T>,
    T: NabledReal + LuProviderScalar,
{
    inverse_kinematics_tree_dls_with_limits(model, base_link, ee_link, q_init, target, config, None)
        .map(|result| result.q)
}

/// Tree DLS IK with optional joint limits; defaults to [`KinematicTreeModel::joint_limits`].
pub fn inverse_kinematics_tree_dls_with_limits<M, T>(
    model: &M,
    base_link: &str,
    ee_link: &str,
    q_init: &Array1<T>,
    target: &Transform3<T>,
    config: &IkConfig<T>,
    limits: Option<&[JointLimits<T>]>,
) -> Result<IkResult<T>, KinematicsError>
where
    M: KinematicTreeModel<T>,
    T: NabledReal + LuProviderScalar,
{
    model.validate_tree()?;
    let dof = model.dof();
    if q_init.len() != dof {
        return Err(KinematicsError::DimensionMismatch);
    }

    let model_limits = tree_limits_from_model(model, dof);
    let effective_limits = limits.or(model_limits.as_deref());
    if let Some(limits) = effective_limits {
        if limits.len() != dof {
            return Err(KinematicsError::DimensionMismatch);
        }
        validate_limits(&q_init.view(), limits)?;
    }

    let mut workspace = IkWorkspace::new(dof);
    let mut q = q_init.clone();
    let mut iterations = 0_usize;
    let mut final_error = T::zero();
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let current = end_effector_pose_tree(model, base_link, ee_link, &q)?;
        pose_error_into(&current, target, &mut workspace.error)?;
        final_error = error_norm(&workspace.error);
        if final_error <= config.tolerance {
            converged = true;
            break;
        }

        let j = jacobian_tree_view(model, base_link, ee_link, &q.view())?;
        workspace.jacobian.assign(&j);
        task_error_for_jacobian(&workspace.error, &mut workspace.task_error);
        dls_step(
            &workspace.jacobian,
            &workspace.task_error,
            config.damping,
            &mut workspace.jtj,
            &mut workspace.jte,
            &mut workspace.dq,
        )?;

        for i in 0..dof {
            q[i] += config.step_scale * workspace.dq[i];
        }
        if let Some(limits) = effective_limits {
            clip_to_limits(&mut q, limits);
        }
    }

    if !converged {
        return Err(KinematicsError::ConvergenceFailed);
    }

    Ok(IkResult { q, iterations, final_error, converged })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_linalg::geometry::Rotation3;
    use ndarray::{Array2, arr1};

    use super::*;
    use crate::chain::{ChainSpec, DhConvention, JointType};
    use crate::tree::end_effector_pose_tree;
    use crate::tree::y_branch_fixture::YBranchModel;

    fn six_dof_chain() -> ChainSpec<f64> {
        ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Revolute; 6],
            arr1(&[0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0]),
            arr1(&[
                std::f64::consts::FRAC_PI_2,
                0.0,
                std::f64::consts::FRAC_PI_2,
                -std::f64::consts::FRAC_PI_2,
                std::f64::consts::FRAC_PI_2,
                0.0,
            ]),
            arr1(&[0.089159, 0.0, 0.0, 0.43307, 0.0, 0.0]),
            arr1(&[0.0; 6]),
        )
        .unwrap()
    }

    fn planar_2r_chain() -> ChainSpec<f64> {
        ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Revolute, JointType::Revolute],
            arr1(&[1.0_f64, 1.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
        )
        .unwrap()
    }

    #[test]
    fn pose_error_zero_at_same_pose() {
        let t = se3::from_rotation_translation(
            &Rotation3 { matrix: Array2::<f64>::eye(3) },
            &arr1(&[1.0, 2.0, 3.0]),
        );
        let err = pose_error(&t, &t).unwrap();
        assert!(err.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn dls_cold_start_six_dof() {
        let chain = six_dof_chain();
        let q_target = arr1(&[0.2_f64, -0.3, 0.5, 0.1, -0.2, 0.4]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let q_init = arr1(&[0.0; 6]);
        let config = IkConfig { max_iterations: 200, tolerance: 1e-3, ..IkConfig::default() };
        let result =
            inverse_kinematics_dls_with_limits(&chain, &q_init, &target, &config, None).unwrap();
        let achieved = fk_view(&chain, &result.q.view()).unwrap();
        let err = pose_error(&achieved, &target).unwrap();
        assert!(result.converged);
        assert!(error_norm(&err) < 1e-3);
    }

    #[test]
    fn dls_recovers_six_dof_warm_start() {
        let chain = six_dof_chain();
        let q_target = arr1(&[0.2_f64, -0.3, 0.5, 0.1, -0.2, 0.4]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let q = inverse_kinematics_dls(&chain, &q_target, &target, &IkConfig::default()).unwrap();
        let achieved = fk_view(&chain, &q.view()).unwrap();
        let err = pose_error(&achieved, &target).unwrap();
        assert!(error_norm(&err) < 1e-2);
    }

    #[test]
    fn dls_step_aligns_with_pose_error_gradient() {
        let chain = planar_2r_chain();
        let q = arr1(&[0.3_f64, -0.4]);
        let q_target = arr1(&[0.8_f64, 0.2]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let current = fk_view(&chain, &q.view()).unwrap();
        let err = pose_error(&current, &target).unwrap();
        let j = jacobian_view(&chain, &q.view()).unwrap();
        let mut task_err = Array1::zeros(6);
        task_error_for_jacobian(&err, &mut task_err);
        let mut jtj = Array2::zeros((2, 2));
        let mut jte = Array1::zeros(2);
        let mut dq = Array1::zeros(2);
        dls_step(&j, &task_err, 0.05, &mut jtj, &mut jte, &mut dq).unwrap();
        let two = 2.0;
        let grad = j.t().dot(&task_err).mapv(|v| v * two);
        assert!(dq.dot(&grad) > 0.0);
    }

    #[test]
    fn planar_2r_reaches_target_with_limits() {
        let chain = planar_2r_chain();
        let q_target = arr1(&[0.5_f64, -0.3]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let limits =
            vec![JointLimits { lower: -3.0, upper: 3.0 }, JointLimits { lower: -3.0, upper: 3.0 }];
        let result = inverse_kinematics_dls_with_limits(
            &chain,
            &arr1(&[0.0, 0.0]),
            &target,
            &IkConfig::default(),
            Some(&limits),
        )
        .unwrap();
        let achieved = fk_view(&chain, &result.q.view()).unwrap();
        let err = pose_error(&achieved, &target).unwrap();
        assert!(error_norm(&err) < 1e-3);
    }

    #[test]
    fn rejects_initial_joint_limit_violation() {
        let chain = planar_2r_chain();
        let q_target = arr1(&[0.5_f64, -0.3]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let limits =
            vec![JointLimits { lower: -1.0, upper: 1.0 }, JointLimits { lower: -1.0, upper: 1.0 }];
        let err = inverse_kinematics_dls_with_limits(
            &chain,
            &arr1(&[2.0, 0.0]),
            &target,
            &IkConfig::default(),
            Some(&limits),
        )
        .unwrap_err();
        assert_eq!(err, KinematicsError::JointLimitViolation(0));
    }

    #[test]
    fn f32_smoke_on_planar_2r() {
        let chain = ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Revolute, JointType::Revolute],
            arr1(&[1.0_f32, 1.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        let q_target = arr1(&[0.4_f32, 0.2]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let result = inverse_kinematics_dls_with_limits(
            &chain,
            &arr1(&[0.0, 0.0]),
            &target,
            &IkConfig::default(),
            None,
        )
        .unwrap();
        let achieved = fk_view(&chain, &result.q.view()).unwrap();
        let err = pose_error(&achieved, &target).unwrap();
        let err_norm = err.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt();
        assert!(err_norm < 1e-2);
    }

    #[test]
    fn dls_into_reuses_workspace() {
        let chain = planar_2r_chain();
        let q_target = arr1(&[0.6_f64, 0.1]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let mut workspace = IkWorkspace::new(2);
        let mut output = arr1(&[0.0, 0.0]);
        let result = inverse_kinematics_dls_into(
            &chain,
            &arr1(&[0.0, 0.0]),
            &target,
            &IkConfig::default(),
            None,
            &mut workspace,
            &mut output,
        )
        .unwrap();
        assert_relative_eq!(
            result.q.as_slice().unwrap(),
            output.as_slice().unwrap(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn prismatic_chain_reaches_z_target() {
        let chain = ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Prismatic],
            arr1(&[0.0_f64]),
            arr1(&[0.0]),
            arr1(&[0.0]),
            arr1(&[0.0]),
        )
        .unwrap();
        let q_target = arr1(&[0.75_f64]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let result = inverse_kinematics_dls_with_limits(
            &chain,
            &arr1(&[0.0]),
            &target,
            &IkConfig::default(),
            None,
        )
        .unwrap();
        assert_relative_eq!(result.q[0], 0.75, epsilon = 1e-3);
    }

    #[test]
    fn y_branch_tree_ik_reaches_target() {
        let model = YBranchModel;
        let q_target = arr1(&[0.3_f64, 0.5, -0.2]);
        let target =
            end_effector_pose_tree(&model, "base", "left_ee", &q_target).expect("target fk");
        let q_init = arr1(&[0.0_f64, 0.0, 0.0]);
        let config = IkConfig { max_iterations: 300, tolerance: 1e-3, ..IkConfig::default() };
        let q = inverse_kinematics_tree_dls(&model, "base", "left_ee", &q_init, &target, &config)
            .expect("tree ik");
        assert_eq!(q.len(), model.dof());
        let achieved = end_effector_pose_tree(&model, "base", "left_ee", &q).expect("achieved fk");
        let err = pose_error(&achieved, &target).unwrap();
        assert!(error_norm(&err) < 1e-3);
    }

    #[test]
    fn tree_ik_with_model_limits_rejects_initial_violation() {
        let model = YBranchModel;
        let q_target = arr1(&[0.3_f64, 0.5, -0.2]);
        let target = end_effector_pose_tree(&model, "base", "left_ee", &q_target).unwrap();
        let err = inverse_kinematics_tree_dls_with_limits(
            &model,
            "base",
            "left_ee",
            &arr1(&[4.0, 0.0, 0.0]),
            &target,
            &IkConfig::default(),
            None,
        )
        .unwrap_err();
        assert_eq!(err, KinematicsError::JointLimitViolation(0));
    }

    #[test]
    fn rejects_q_init_dimension_mismatch() {
        let chain = planar_2r_chain();
        let target = fk_view(&chain, &arr1(&[0.0, 0.0]).view()).unwrap();
        let err = inverse_kinematics_dls_with_limits(
            &chain,
            &arr1(&[0.0]),
            &target,
            &IkConfig::default(),
            None,
        )
        .unwrap_err();
        assert_eq!(err, KinematicsError::DimensionMismatch);
    }

    #[test]
    fn rejects_limits_length_mismatch() {
        let chain = planar_2r_chain();
        let target = fk_view(&chain, &arr1(&[0.0, 0.0]).view()).unwrap();
        let limits = vec![JointLimits { lower: -1.0, upper: 1.0 }];
        let err = inverse_kinematics_dls_with_limits(
            &chain,
            &arr1(&[0.0, 0.0]),
            &target,
            &IkConfig::default(),
            Some(&limits),
        )
        .unwrap_err();
        assert_eq!(err, KinematicsError::DimensionMismatch);
    }

    #[test]
    fn fails_when_not_converged_within_iteration_budget() {
        let chain = planar_2r_chain();
        let q_target = arr1(&[1.5_f64, -1.0]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let config = IkConfig { max_iterations: 1, tolerance: 1e-12, ..IkConfig::default() };
        let err =
            inverse_kinematics_dls_with_limits(&chain, &arr1(&[0.0, 0.0]), &target, &config, None)
                .unwrap_err();
        assert_eq!(err, KinematicsError::ConvergenceFailed);
    }

    #[test]
    fn tree_ik_rejects_wrong_q_dimension() {
        let model = YBranchModel;
        let target =
            end_effector_pose_tree(&model, "base", "left_ee", &arr1(&[0.0, 0.0, 0.0])).unwrap();
        let err = inverse_kinematics_tree_dls(
            &model,
            "base",
            "left_ee",
            &arr1(&[0.0, 0.0]),
            &target,
            &IkConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, KinematicsError::DimensionMismatch);
    }
}
