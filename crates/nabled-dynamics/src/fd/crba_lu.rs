//! Forward dynamics via composite rigid-body mass matrix and LU solve.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_linalg::lu;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, ArrayView1};

use crate::DynamicsError;
use crate::config::DynamicsConfig;
use crate::crba::mass_matrix;
use crate::rnea::rnea_with_config;

pub fn crba_lu_forward_dynamics<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array1<T>, DynamicsError> {
    if tau.len() != chain.num_joints() {
        return Err(DynamicsError::DimensionMismatch);
    }
    let zero_qdd = Array1::<T>::zeros(tau.len());
    let bias = rnea_with_config(model, chain, q, qd, &zero_qdd.view(), config)?;
    let m = mass_matrix(model, chain, q, config)?;
    let rhs = tau.to_owned() - bias;
    lu::solve(&m, &rhs)
        .map_err(|_| DynamicsError::InvalidInput("mass matrix solve failed".to_string()))
}
