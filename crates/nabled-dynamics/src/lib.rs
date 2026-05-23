//! Robot dynamics.

#![allow(clippy::missing_errors_doc)]

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

pub mod config;
pub mod crba;
pub mod fd;
pub mod id;
pub mod rnea;
pub mod spatial;

pub use config::{DynamicsConfig, ForwardDynamicsMethod};

#[derive(Debug, Clone, PartialEq)]
pub enum DynamicsError {
    EmptyModel,
    DimensionMismatch,
    InvalidInput(String),
    NotImplemented,
}

impl std::fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynamicsError::EmptyModel => write!(f, "dynamics model cannot be empty"),
            DynamicsError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
            DynamicsError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            DynamicsError::NotImplemented => write!(f, "dynamics routine not yet implemented"),
        }
    }
}

impl std::error::Error for DynamicsError {}

impl IntoNabledError for DynamicsError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            DynamicsError::EmptyModel => NabledError::Shape(ShapeError::EmptyInput),
            DynamicsError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            DynamicsError::InvalidInput(message) => NabledError::InvalidInput(message),
            DynamicsError::NotImplemented => {
                NabledError::Other("dynamics routine not implemented".to_string())
            }
        }
    }
}
