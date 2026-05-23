//! Luenberger observer gain (stub).

use nabled_core::scalar::NabledReal;
use ndarray::Array2;

use crate::ControlError;

pub fn luenberger_gain<T: NabledReal>(
    _a: &Array2<T>,
    _c: &Array2<T>,
    _poles: &[T],
) -> Result<Array2<T>, ControlError> {
    Err(ControlError::InvalidInput("Luenberger gain not yet implemented".to_string()))
}
