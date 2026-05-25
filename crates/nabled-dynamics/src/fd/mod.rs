//! Forward dynamics (ABA default, CRBA+LU fallback).

mod aba;
mod crba_lu;

pub use aba::aba_forward_dynamics;
pub use crba_lu::crba_lu_forward_dynamics;
use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_linalg::lu;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, ArrayView1};

use crate::DynamicsError;
use crate::config::{DynamicsConfig, ForwardDynamicsMethod};

pub fn forward_dynamics<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    forward_dynamics_with_config(model, chain, q, qd, tau, &DynamicsConfig::default())
}

pub fn forward_dynamics_with_config<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array1<T>, DynamicsError> {
    match config.forward_dynamics {
        ForwardDynamicsMethod::Aba => aba_forward_dynamics(model, chain, q, qd, tau, config),
        ForwardDynamicsMethod::CrbaLu => crba_lu_forward_dynamics(model, chain, q, qd, tau, config),
    }
}

pub fn forward_dynamics_view<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    forward_dynamics(model, chain, q, qd, tau)
}

pub fn forward_dynamics_into<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
    output: &mut Array1<T>,
) -> Result<(), DynamicsError> {
    let qdd = forward_dynamics(model, chain, q, qd, tau)?;
    if output.len() != qdd.len() {
        return Err(DynamicsError::DimensionMismatch);
    }
    output.assign(&qdd);
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_model::fixture::load_planar2r_json;
    use ndarray::arr1;

    use super::*;
    use crate::config::DynamicsConfig;
    use crate::rnea::rnea_with_config;

    #[test]
    fn forward_dynamics_round_trip() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model().unwrap();
        let chain = fixture.to_chain_spec().unwrap();
        let config = DynamicsConfig::default();
        let q = arr1(&[0.2_f64, 0.3]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let qdd = arr1(&[0.5_f64, 0.25]);
        let tau =
            rnea_with_config(&model, &chain, &q.view(), &qd.view(), &qdd.view(), &config).unwrap();
        let recovered = forward_dynamics_with_config(
            &model,
            &chain,
            &q.view(),
            &qd.view(),
            &tau.view(),
            &config,
        )
        .unwrap();
        assert_relative_eq!(recovered, qdd, epsilon = 1e-6);
    }

    #[test]
    fn aba_matches_crba_lu_on_planar2r() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model().unwrap();
        let chain = fixture.to_chain_spec().unwrap();
        let gravity: [f64; 3] = fixture.gravity.unwrap_or([0.0, -9.81, 0.0]);
        let aba_config = DynamicsConfig { gravity, forward_dynamics: ForwardDynamicsMethod::Aba };
        let lu_config = DynamicsConfig { gravity, forward_dynamics: ForwardDynamicsMethod::CrbaLu };
        let q = arr1(&[0.2_f64, 0.3]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let qdd = arr1(&[0.5_f64, 0.25]);
        let tau = rnea_with_config(&model, &chain, &q.view(), &qd.view(), &qdd.view(), &aba_config)
            .unwrap();
        let aba = forward_dynamics_with_config(
            &model,
            &chain,
            &q.view(),
            &qd.view(),
            &tau.view(),
            &aba_config,
        )
        .unwrap();
        let lu = forward_dynamics_with_config(
            &model,
            &chain,
            &q.view(),
            &qd.view(),
            &tau.view(),
            &lu_config,
        )
        .unwrap();
        assert_relative_eq!(aba, lu, epsilon = 1e-6);
        assert_relative_eq!(aba, qdd, epsilon = 1e-6);
    }

    #[test]
    fn forward_dynamics_view_and_into_match_allocating() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model().unwrap();
        let chain = fixture.to_chain_spec().unwrap();
        let q = arr1(&[0.2_f64, 0.3]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let tau = arr1(&[0.5_f64, 0.25]);
        let qdd =
            forward_dynamics_view(&model, &chain, &q.view(), &qd.view(), &tau.view()).unwrap();
        let mut buf = arr1(&[0.0, 0.0]);
        forward_dynamics_into(&model, &chain, &q.view(), &qd.view(), &tau.view(), &mut buf)
            .unwrap();
        assert_relative_eq!(qdd, buf, epsilon = 1e-12);
    }

    #[test]
    fn forward_dynamics_into_rejects_output_dimension_mismatch() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model().unwrap();
        let chain = fixture.to_chain_spec().unwrap();
        let q = arr1(&[0.2_f64, 0.3]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let tau = arr1(&[0.5_f64, 0.25]);
        let mut buf = arr1(&[0.0]);
        assert!(matches!(
            forward_dynamics_into(&model, &chain, &q.view(), &qd.view(), &tau.view(), &mut buf),
            Err(DynamicsError::DimensionMismatch)
        ));
    }
}
