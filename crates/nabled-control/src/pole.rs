//! Pole placement (SISO stub).

use nabled_core::scalar::NabledReal;
use ndarray::Array2;

use crate::ControlError;

pub fn place_poles<T: NabledReal>(
    _a: &Array2<T>,
    _b: &Array2<T>,
    _poles: &[T],
) -> Result<Array2<T>, ControlError> {
    Err(ControlError::InvalidInput("pole placement not yet implemented".to_string()))
}
