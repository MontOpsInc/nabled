//! Robot dynamics.

#![allow(clippy::missing_errors_doc, clippy::too_many_arguments, clippy::similar_names)]

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

pub mod config;
pub mod crba;
pub mod fd;
pub mod id;
pub mod rnea;
pub mod spatial;
pub mod tree;

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

#[cfg(test)]
mod tests {
    use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

    use super::*;

    #[test]
    fn dynamics_errors_display_and_map_to_shared_taxonomy() {
        assert_eq!(DynamicsError::EmptyModel.to_string(), "dynamics model cannot be empty");
        assert_eq!(
            DynamicsError::DimensionMismatch.to_string(),
            "input dimensions are incompatible"
        );
        assert_eq!(
            DynamicsError::InvalidInput("bad q".to_string()).to_string(),
            "invalid input: bad q"
        );
        assert_eq!(
            DynamicsError::NotImplemented.to_string(),
            "dynamics routine not yet implemented"
        );

        assert!(matches!(
            DynamicsError::EmptyModel.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            DynamicsError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            DynamicsError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(DynamicsError::NotImplemented.into_nabled_error(), NabledError::Other(_)));
    }
}
