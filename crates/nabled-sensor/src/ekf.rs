//! Extended Kalman filter.

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu::{self, LuProviderScalar};
use ndarray::{Array1, Array2, ArrayView1};

use crate::SensorError;
use crate::kalman::KalmanState;

/// Nonlinear EKF model callbacks.
#[derive(Clone)]
pub struct EkModel<T> {
    pub predict_state:     fn(&ArrayView1<'_, T>) -> Array1<T>,
    pub predict_jacobian:  fn(&ArrayView1<'_, T>) -> Array2<T>,
    pub measure:           fn(&ArrayView1<'_, T>) -> Array1<T>,
    pub measure_jacobian:  fn(&ArrayView1<'_, T>) -> Array2<T>,
}

#[derive(Debug, Clone)]
pub struct EkConfig<T> {
    pub process_noise:     Array2<T>,
    pub measurement_noise: Array2<T>,
}

pub fn ekf_predict<T: NabledReal>(
    state: &KalmanState<T>,
    model: &EkModel<T>,
    config: &EkConfig<T>,
) -> Result<KalmanState<T>, SensorError> {
    let mut mean = Array1::zeros(state.mean.len());
    let mut covariance = Array2::zeros(state.covariance.dim());
    ekf_predict_into(state, model, config, &mut mean, &mut covariance)?;
    Ok(KalmanState { mean, covariance })
}

pub fn ekf_predict_into<T: NabledReal>(
    state: &KalmanState<T>,
    model: &EkModel<T>,
    config: &EkConfig<T>,
    mean_out: &mut Array1<T>,
    covariance_out: &mut Array2<T>,
) -> Result<(), SensorError> {
    let f = (model.predict_jacobian)(&state.mean.view());
    let predicted_mean = (model.predict_state)(&state.mean.view());
    if mean_out.len() != predicted_mean.len() || covariance_out.dim() != state.covariance.dim() {
        return Err(SensorError::DimensionMismatch);
    }
    mean_out.assign(&predicted_mean);
    *covariance_out = f.dot(&state.covariance).dot(&f.t()) + &config.process_noise;
    Ok(())
}

pub fn ekf_update<T: NabledReal + LuProviderScalar>(
    state: &KalmanState<T>,
    measurement: &ArrayView1<'_, T>,
    model: &EkModel<T>,
    config: &EkConfig<T>,
) -> Result<KalmanState<T>, SensorError> {
    let mut mean = state.mean.clone();
    let mut covariance = state.covariance.clone();
    ekf_update_into(state, measurement, model, config, &mut mean, &mut covariance)?;
    Ok(KalmanState { mean, covariance })
}

pub fn ekf_update_into<T: NabledReal + LuProviderScalar>(
    state: &KalmanState<T>,
    measurement: &ArrayView1<'_, T>,
    model: &EkModel<T>,
    config: &EkConfig<T>,
    mean_out: &mut Array1<T>,
    covariance_out: &mut Array2<T>,
) -> Result<(), SensorError> {
    let h = (model.measure_jacobian)(&state.mean.view());
    let predicted = (model.measure)(&state.mean.view());
    let innovation = measurement - &predicted;
    let s = h.dot(&state.covariance).dot(&h.t()) + &config.measurement_noise;
    let s_inv = lu::inverse(&s).map_err(|_| SensorError::NumericalInstability)?;
    let k = state.covariance.dot(&h.t()).dot(&s_inv);
    mean_out.assign(&(state.mean.clone() + k.dot(&innovation)));
    let n = state.mean.len();
    let identity = Array2::<T>::eye(n);
    *covariance_out = (identity - k.dot(&h)).dot(&state.covariance);
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;

    fn scalar_model() -> EkModel<f64> {
        EkModel {
            predict_state: |x| arr1(&[x[0].sin()]),
            predict_jacobian: |x| ndarray::arr2(&[[x[0].cos()]]),
            measure: |x| arr1(&[x[0]]),
            measure_jacobian: |_| ndarray::arr2(&[[1.0]]),
        }
    }

    #[test]
    fn ekf_update_moves_toward_measurement() {
        let state = KalmanState { mean: arr1(&[0.2_f64]), covariance: ndarray::arr2(&[[1.0]]) };
        let model = scalar_model();
        let config = EkConfig {
            process_noise:     ndarray::arr2(&[[0.01]]),
            measurement_noise: ndarray::arr2(&[[0.05]]),
        };
        let updated = ekf_update(&state, &arr1(&[1.0]).view(), &model, &config).unwrap();
        assert!(updated.mean[0] > state.mean[0]);
    }
}
