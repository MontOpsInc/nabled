//! Extended Kalman filter (signatures).

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2};

use crate::SensorError;
use crate::kalman::KalmanState;

#[derive(Debug, Clone)]
pub struct EkConfig<T> {
    pub process_noise:     Array2<T>,
    pub measurement_noise: Array2<T>,
}

pub fn ekf_predict<T: NabledReal>(
    _state: &KalmanState<T>,
    _config: &EkConfig<T>,
) -> Result<KalmanState<T>, SensorError> {
    Err(SensorError::InvalidInput("EKF predict not yet implemented".to_string()))
}

pub fn ekf_update<T: NabledReal>(
    _state: &KalmanState<T>,
    _measurement: &Array1<T>,
    _config: &EkConfig<T>,
) -> Result<KalmanState<T>, SensorError> {
    Err(SensorError::InvalidInput("EKF update not yet implemented".to_string()))
}
