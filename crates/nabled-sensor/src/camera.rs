//! Pinhole camera model (stubs).

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1};

use crate::SensorError;

pub fn pinhole_project<T: NabledReal>(
    _point: &ArrayView1<'_, T>,
    _intrinsics: &Array2<T>,
) -> Result<Array1<T>, SensorError> {
    Err(SensorError::InvalidInput("pinhole projection not yet implemented".to_string()))
}

pub fn pinhole_jacobian<T: NabledReal>(
    _point: &ArrayView1<'_, T>,
    _intrinsics: &Array2<T>,
) -> Result<Array2<T>, SensorError> {
    Err(SensorError::InvalidInput("pinhole jacobian not yet implemented".to_string()))
}
