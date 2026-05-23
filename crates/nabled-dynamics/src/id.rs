//! Inverse dynamics aliases and partial torques.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, ArrayView1};

use crate::DynamicsError;
use crate::rnea::{rnea, rnea_view};

pub fn inverse_dynamics<T: NabledReal>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    qdd: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    rnea(model, chain, q, qd, qdd)
}

pub fn gravity_torques<T: NabledReal>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
) -> Result<Array1<T>, DynamicsError> {
    let zeros = Array1::<T>::zeros(q.len());
    let qdd = Array1::<T>::zeros(q.len());
    rnea(model, chain, q, &zeros, &qdd.view())
}

pub fn coriolis_torques<T: NabledReal>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
) -> Result<Array1<T>, DynamicsError> {
    let qdd = Array1::<T>::zeros(q.len());
    rnea_view(model, chain, &q.view(), &qd.view(), &qdd.view())
}
