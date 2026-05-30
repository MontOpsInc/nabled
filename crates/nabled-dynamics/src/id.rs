//! Inverse dynamics aliases and partial torques.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, ArrayView1};

use crate::DynamicsError;
use crate::rnea::{rnea, rnea_view};

pub fn inverse_dynamics<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    qdd: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    rnea(model, chain, q, qd, qdd)
}

pub fn gravity_torques<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    let zeros = Array1::<T>::zeros(q.len());
    let qdd = Array1::<T>::zeros(q.len());
    rnea(model, chain, q, &zeros.view(), &qdd.view())
}

pub fn coriolis_torques<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    let zeros = Array1::<T>::zeros(q.len());
    let qdd = Array1::<T>::zeros(q.len());
    let tau_qd = rnea_view(model, chain, &q.view(), &qd.view(), &qdd.view())?;
    let tau_zero = rnea_view(model, chain, &q.view(), &zeros.view(), &qdd.view())?;
    Ok(tau_qd - tau_zero)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_model::fixture::load_planar2r_json;
    use ndarray::arr1;

    use super::*;
    use crate::config::DynamicsConfig;
    use crate::crba::mass_matrix;
    use crate::rnea::rnea;

    #[test]
    fn gravity_torques_match_fixture_when_present() {
        let fixture = load_planar2r_json().expect("fixture");
        let model = fixture.to_robot_model::<f64>().expect("model");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let gravity: [f64; 3] = fixture.gravity.unwrap_or([0.0, -9.81, 0.0]);
        let config = DynamicsConfig { gravity, ..DynamicsConfig::default() };
        for case in &fixture.cases {
            if let Some(tau_gravity) = &case.tau_gravity {
                let q = arr1(&case.q);
                let tau = gravity_torques(&model, &chain, &q.view()).expect("gravity torques");
                for (computed, expected) in tau.iter().zip(tau_gravity.iter()) {
                    assert_relative_eq!(computed, expected, epsilon = 1e-6);
                }
                let via_rnea = crate::rnea::rnea_with_config(
                    &model,
                    &chain,
                    &q.view(),
                    &Array1::<f64>::zeros(q.len()).view(),
                    &Array1::<f64>::zeros(q.len()).view(),
                    &config,
                )
                .expect("rnea gravity");
                assert_relative_eq!(tau, via_rnea, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn coriolis_decomposes_inverse_dynamics_on_planar2r() {
        let fixture = load_planar2r_json().expect("fixture");
        let model = fixture.to_robot_model::<f64>().expect("model");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let gravity: [f64; 3] = fixture.gravity.unwrap_or([0.0, -9.81, 0.0]);
        let config = DynamicsConfig { gravity, ..DynamicsConfig::default() };
        let case = fixture
            .cases
            .iter()
            .find(|c| c.qd.is_some() && c.qdd.is_some())
            .expect("dynamics case");
        let q = arr1(&case.q);
        let qd = arr1(case.qd.as_ref().expect("qd"));
        let qdd = arr1(case.qdd.as_ref().expect("qdd"));
        let tau = inverse_dynamics(&model, &chain, &q.view(), &qd.view(), &qdd.view())
            .expect("inverse dynamics");
        let tau_g = gravity_torques(&model, &chain, &q.view()).expect("gravity torques");
        let tau_c = coriolis_torques(&model, &chain, &q.view(), &qd.view()).expect("coriolis");
        let mass = mass_matrix(&model, &chain, &q.view(), &config).expect("mass matrix");
        let tau_inertial = mass.dot(&qdd);
        let reconstructed = &tau_g + &tau_c + &tau_inertial;
        assert_relative_eq!(tau, reconstructed, epsilon = 1e-6);
    }

    #[test]
    fn inverse_dynamics_matches_rnea_alias() {
        let fixture = load_planar2r_json().expect("fixture");
        let model = fixture.to_robot_model::<f64>().expect("model");
        let chain = fixture.to_chain_spec::<f64>().expect("chain");
        let case = fixture.cases.iter().find(|c| c.qd.is_some()).expect("case");
        let q = arr1(&case.q);
        let qd = arr1(case.qd.as_ref().expect("qd"));
        let qdd = arr1(case.qdd.as_ref().unwrap_or(&vec![0.0; q.len()]));
        let tau_id = inverse_dynamics(&model, &chain, &q.view(), &qd.view(), &qdd.view()).unwrap();
        let tau_rnea = rnea(&model, &chain, &q.view(), &qd.view(), &qdd.view()).unwrap();
        assert_relative_eq!(tau_id, tau_rnea, epsilon = 1e-12);
    }
}
