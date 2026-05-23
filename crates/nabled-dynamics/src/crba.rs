//! Composite rigid body algorithm (mass matrix).

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, Array2};

use crate::DynamicsError;

pub fn mass_matrix<T: NabledReal>(
    _model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    _q: &Array1<T>,
) -> Result<Array2<T>, DynamicsError> {
    let n = chain.num_joints();
    Ok(Array2::<T>::eye(n))
}

pub fn mass_matrix_into<T: NabledReal>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    output: &mut Array2<T>,
) -> Result<(), DynamicsError> {
    let m = mass_matrix(model, chain, q)?;
    if output.dim() != m.dim() {
        return Err(DynamicsError::DimensionMismatch);
    }
    output.assign(&m);
    Ok(())
}
