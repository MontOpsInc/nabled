//! Composite rigid body algorithm (mass matrix via inverse dynamics identity).

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, Array2, ArrayView1};

use crate::DynamicsError;
use crate::config::DynamicsConfig;
use crate::rnea::rnea_with_config;

/// Compute joint-space mass matrix `M(q)` using the inverse-dynamics column identity.
pub fn mass_matrix<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array2<T>, DynamicsError> {
    let n = chain.num_joints();
    let zeros = Array1::<T>::zeros(n);
    let bias = rnea_with_config(model, chain, q, &zeros.view(), &zeros.view(), config)?;
    let mut m = Array2::<T>::zeros((n, n));
    for i in 0..n {
        let mut qdd = Array1::<T>::zeros(n);
        qdd[i] = T::one();
        let tau = rnea_with_config(model, chain, q, &zeros.view(), &qdd.view(), config)?;
        for row in 0..n {
            m[[row, i]] = tau[row] - bias[row];
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (m[[i, j]] + m[[j, i]]) * T::from_f64(0.5).unwrap_or(T::zero());
            m[[i, j]] = avg;
            m[[j, i]] = avg;
        }
    }
    Ok(m)
}

pub fn mass_matrix_into<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    config: &DynamicsConfig<T>,
    output: &mut Array2<T>,
) -> Result<(), DynamicsError> {
    let m = mass_matrix(model, chain, &q.view(), config)?;
    if output.dim() != m.dim() {
        return Err(DynamicsError::DimensionMismatch);
    }
    output.assign(&m);
    Ok(())
}

#[cfg(test)]
mod tests {
    use nabled_model::fixture::Planar2rFixture;
    use ndarray::arr1;

    use super::*;
    use crate::DynamicsError;
    use crate::config::DynamicsConfig;

    #[test]
    fn mass_matrix_is_symmetric_positive_diagonal() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nabled/tests/fixtures/physical_ai/2r_planar.json"
        );
        let fixture = Planar2rFixture::from_file(path).unwrap();
        let model = fixture.to_robot_model().unwrap();
        let chain = fixture.to_chain_spec().unwrap();
        let q = arr1(&[0.3_f64, 0.2]);
        let config = DynamicsConfig::default();
        let m = mass_matrix(&model, &chain, &q.view(), &config).unwrap();
        assert!(m[[0, 0]] > 0.0);
        assert!(m[[1, 1]] > 0.0);
        assert!((m[[0, 1]] - m[[1, 0]]).abs() < 1e-6);
    }

    #[test]
    fn mass_matrix_rejects_dimension_mismatch() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nabled/tests/fixtures/physical_ai/2r_planar.json"
        );
        let fixture = Planar2rFixture::from_file(path).unwrap();
        let model = fixture.to_robot_model().unwrap();
        let chain = fixture.to_chain_spec().unwrap();
        let q = arr1(&[0.3_f64]);
        let config = DynamicsConfig::default();
        assert!(matches!(
            mass_matrix(&model, &chain, &q.view(), &config),
            Err(DynamicsError::DimensionMismatch)
        ));
    }
}
