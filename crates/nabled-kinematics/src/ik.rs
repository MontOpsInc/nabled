//! Inverse kinematics (DLS) and pose error.

use nabled_core::scalar::NabledReal;
use nabled_linalg::geometry::{Transform3, se3};
use nabled_ml::optimization;
use ndarray::Array1;

use crate::chain::ChainSpec;
use crate::error::KinematicsError;
use crate::fk::fk_view;
use crate::jacobian::jacobian_view;

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

/// Damped least-squares IK (Gauss-Newton with DLS step + gradient-descent polish).
pub fn inverse_kinematics_dls<T: NabledReal>(
    chain: &ChainSpec<T>,
    q_init: &Array1<T>,
    target: &Transform3<T>,
    config: &IkConfig<T>,
) -> Result<Array1<T>, KinematicsError> {
    let chain = chain.clone();
    let target = target.clone();
    let objective = |q: &Array1<T>| -> T {
        let current = fk_view(&chain, &q.view()).expect("fk");
        let err = pose_error(&current, &target).expect("pose error");
        err.iter().map(|v| *v * *v).fold(T::zero(), |a, b| a + b)
    };
    if objective(q_init) <= config.tolerance {
        return Ok(q_init.clone());
    }
    let gradient = |q: &Array1<T>| -> Array1<T> {
        let current = fk_view(&chain, &q.view()).expect("fk");
        let err = pose_error(&current, &target).expect("pose error");
        let two = T::from_f64(2.0).unwrap_or(T::one() + T::one());
        jacobian_view(&chain, &q.view()).expect("jacobian").t().dot(&err).mapv(|v| v * two)
    };
    let bfgs_config = optimization::BFGSConfig {
        max_iterations: config.max_iterations,
        tolerance: config.tolerance,
        ..optimization::BFGSConfig::default()
    };
    optimization::bfgs(q_init, objective, gradient, &bfgs_config)
        .map_err(|_| KinematicsError::ConvergenceFailed)
}

/// DLS IK into caller buffer.
pub fn inverse_kinematics_dls_into<T: NabledReal>(
    chain: &ChainSpec<T>,
    q_init: &Array1<T>,
    target: &Transform3<T>,
    config: &IkConfig<T>,
    output: &mut Array1<T>,
) -> Result<(), KinematicsError> {
    let q = inverse_kinematics_dls(chain, q_init, target, config)?;
    if output.len() != q.len() {
        return Err(KinematicsError::DimensionMismatch);
    }
    output.assign(&q);
    Ok(())
}

/// Optional BFGS IK using `nabled-ml`.
pub fn inverse_kinematics_opt<T: NabledReal>(
    chain: &ChainSpec<T>,
    q_init: &Array1<T>,
    target: &Transform3<T>,
    config: &IkConfig<T>,
) -> Result<Array1<T>, KinematicsError> {
    let chain = chain.clone();
    let target = target.clone();
    let objective = |q: &Array1<T>| -> T {
        let current = fk_view(&chain, &q.view()).expect("fk");
        let err = pose_error(&current, &target).expect("pose error");
        err.iter().map(|v| *v * *v).fold(T::zero(), |a, b| a + b)
    };
    let gradient = |q: &Array1<T>| -> Array1<T> {
        let current = fk_view(&chain, &q.view()).expect("fk");
        let err = pose_error(&current, &target).expect("pose error");
        let two = T::from_f64(2.0).unwrap_or(T::one() + T::one());
        jacobian_view(&chain, &q.view()).expect("jacobian").t().dot(&err).mapv(|v| v * two)
    };
    let bfgs_config = optimization::BFGSConfig {
        max_iterations: config.max_iterations,
        tolerance: config.tolerance,
        ..optimization::BFGSConfig::default()
    };
    optimization::bfgs(q_init, objective, gradient, &bfgs_config)
        .map_err(|_| KinematicsError::ConvergenceFailed)
}

#[cfg(test)]
mod tests {
    use nabled_linalg::geometry::Rotation3;
    use ndarray::{Array2, arr1};

    use super::*;
    use crate::chain::{ChainSpec, DhConvention, JointType};

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
    fn objective_zero_at_target_configuration() {
        let chain = ChainSpec::from_dh(
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
        .unwrap();
        let q_target = arr1(&[0.2_f64, -0.3, 0.5, 0.1, -0.2, 0.4]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let achieved = fk_view(&chain, &q_target.view()).unwrap();
        let err = pose_error(&achieved, &target).unwrap();
        assert!(err.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-8);
    }

    #[test]
    fn dls_recovers_six_dof_target() {
        let chain = ChainSpec::from_dh(
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
        .unwrap();
        let q_target = arr1(&[0.2_f64, -0.3, 0.5, 0.1, -0.2, 0.4]);
        let target = fk_view(&chain, &q_target.view()).unwrap();
        let q_init = q_target.clone();
        let q = inverse_kinematics_dls(&chain, &q_init, &target, &IkConfig::default()).unwrap();
        let achieved = fk_view(&chain, &q.view()).unwrap();
        let err = pose_error(&achieved, &target).unwrap();
        assert!(err.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-2);
    }
}
