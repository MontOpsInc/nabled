//! Forward dynamics (ABA stub).

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, ArrayView1};

use crate::DynamicsError;
use crate::crba::mass_matrix;

pub fn forward_dynamics<T: NabledReal>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    tau: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    let m = mass_matrix(model, chain, q)?;
    Ok(m.dot(tau) - qd.clone())
}

pub fn forward_dynamics_view<T: NabledReal>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    forward_dynamics(model, chain, &q.to_owned(), &qd.to_owned(), tau)
}

pub fn forward_dynamics_into<T: NabledReal>(
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
