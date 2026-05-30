//! Robot model representation.

#![allow(clippy::missing_errors_doc, let_underscore_drop, clippy::missing_panics_doc)]

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

pub mod dh;
pub mod fixture;
pub mod joint;
pub mod link;
pub mod origin;
pub mod robot;
pub mod tree_model;
pub mod urdf;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    EmptyModel,
    DimensionMismatch,
    InvalidInput(String),
    ParseError(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::EmptyModel => write!(f, "robot model cannot be empty"),
            ModelError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
            ModelError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            ModelError::ParseError(message) => write!(f, "parse error: {message}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl IntoNabledError for ModelError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            ModelError::EmptyModel => NabledError::Shape(ShapeError::EmptyInput),
            ModelError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            ModelError::InvalidInput(message) | ModelError::ParseError(message) => {
                NabledError::InvalidInput(message)
            }
        }
    }
}

#[cfg(test)]
mod error_mapping {
    use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

    use crate::ModelError;

    #[test]
    fn into_nabled_error_covers_all_variants() {
        assert!(matches!(
            ModelError::EmptyModel.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            ModelError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            ModelError::InvalidInput("bad".into()).into_nabled_error(),
            NabledError::InvalidInput(message) if message == "bad"
        ));
        assert!(matches!(
            ModelError::ParseError("xml".into()).into_nabled_error(),
            NabledError::InvalidInput(message) if message == "xml"
        ));
    }
}
