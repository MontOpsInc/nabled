//! Forward dynamics via mass matrix and RNEA bias.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_linalg::lu;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, ArrayView1};

use crate::DynamicsError;
use crate::config::DynamicsConfig;
use crate::crba::mass_matrix;
use crate::rnea::rnea_with_config;

pub fn forward_dynamics<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    tau: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    forward_dynamics_with_config(model, chain, q, qd, tau, &DynamicsConfig::default())
}

pub fn forward_dynamics_with_config<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    tau: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array1<T>, DynamicsError> {
    if tau.len() != chain.num_joints() {
        return Err(DynamicsError::DimensionMismatch);
    }
    let zero_qdd = Array1::<T>::zeros(tau.len());
    let bias = rnea_with_config(model, chain, q, qd, &zero_qdd.view(), config)?;
    let m = mass_matrix(model, chain, q, config)?;
    let rhs = tau - bias;
    lu::solve(&m, &rhs)
        .map_err(|_| DynamicsError::InvalidInput("mass matrix solve failed".to_string()))
}

pub fn forward_dynamics_view<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    forward_dynamics(model, chain, &q.to_owned(), &qd.to_owned(), tau)
}

pub fn forward_dynamics_into<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
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
    use nabled_model::fixture::Planar2rFixture;
    use ndarray::arr1;

    use super::*;
    use crate::config::DynamicsConfig;
    use crate::rnea::rnea_with_config;

    #[test]
    fn forward_dynamics_round_trip() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nabled/tests/fixtures/physical_ai/2r_planar.json"
        );
        let fixture = Planar2rFixture::from_file(path).unwrap();
        let model = fixture.to_robot_model().unwrap();
        let chain = fixture.to_chain_spec().unwrap();
        let config = DynamicsConfig::default();
        let q = arr1(&[0.2_f64, 0.3]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let qdd = arr1(&[0.5_f64, 0.25]);
        let tau = rnea_with_config(&model, &chain, &q, &qd, &qdd.view(), &config).unwrap();
        let recovered =
            forward_dynamics_with_config(&model, &chain, &q, &qd, &tau.view(), &config).unwrap();
        assert_relative_eq!(recovered, qdd, epsilon = 1e-6);
    }
}
