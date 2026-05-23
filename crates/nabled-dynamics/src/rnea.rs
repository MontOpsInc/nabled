//! Recursive Newton-Euler algorithm.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, ArrayView1};

use crate::DynamicsError;

pub fn rnea<T: NabledReal>(
    _model: &RobotModel<T>,
    _chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    qdd: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    if q.len() != qd.len() || q.len() != qdd.len() {
        return Err(DynamicsError::DimensionMismatch);
    }
    // Minimal 1-DOF / serial stub: return qdd as torque placeholder for scaffold
    Ok(qdd.to_owned())
}

pub fn rnea_view<T: NabledReal>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    qdd: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    rnea(model, chain, &q.to_owned(), &qd.to_owned(), qdd)
}

pub fn rnea_into<T: NabledReal>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    qdd: &ArrayView1<'_, T>,
    output: &mut Array1<T>,
) -> Result<(), DynamicsError> {
    let tau = rnea(model, chain, q, qd, qdd)?;
    if output.len() != tau.len() {
        return Err(DynamicsError::DimensionMismatch);
    }
    output.assign(&tau);
    Ok(())
}
