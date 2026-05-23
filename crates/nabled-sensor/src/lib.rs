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
