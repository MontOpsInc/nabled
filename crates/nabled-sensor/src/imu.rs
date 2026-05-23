//! IMU strapdown integration (stub).

use nabled_core::scalar::NabledReal;
use ndarray::Array1;

use crate::SensorError;

pub fn strapdown_predict<T: NabledReal>(
    _orientation: &Array1<T>,
    _gyro: &Array1<T>,
    _dt: T,
) -> Result<Array1<T>, SensorError> {
    Err(SensorError::InvalidInput("strapdown predict not yet implemented".to_string()))
}
