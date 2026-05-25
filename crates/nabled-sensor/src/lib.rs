//! Sensor fusion and estimation.

#![allow(clippy::missing_errors_doc, clippy::many_single_char_names)]

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

pub mod camera;
pub mod ekf;
pub mod imu;
pub mod kalman;

#[derive(Debug, Clone, PartialEq)]
pub enum SensorError {
    EmptyInput,
    DimensionMismatch,
    InvalidInput(String),
    NumericalInstability,
}

impl std::fmt::Display for SensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensorError::EmptyInput => write!(f, "input cannot be empty"),
            SensorError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
            SensorError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            SensorError::NumericalInstability => write!(f, "numerical instability detected"),
        }
    }
}

impl std::error::Error for SensorError {}

impl IntoNabledError for SensorError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            SensorError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            SensorError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            SensorError::InvalidInput(message) => NabledError::InvalidInput(message),
            SensorError::NumericalInstability => NabledError::NumericalInstability,
        }
    }
}

#[cfg(test)]
mod tests {
    use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

    use super::*;

    #[test]
    fn sensor_errors_display_and_map_to_shared_taxonomy() {
        assert_eq!(SensorError::EmptyInput.to_string(), "input cannot be empty");
        assert_eq!(SensorError::DimensionMismatch.to_string(), "input dimensions are incompatible");
        assert_eq!(
            SensorError::InvalidInput("bad z".to_string()).to_string(),
            "invalid input: bad z"
        );
        assert_eq!(SensorError::NumericalInstability.to_string(), "numerical instability detected");

        assert!(matches!(
            SensorError::EmptyInput.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            SensorError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            SensorError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            SensorError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));
    }
}
