//! Control theory for LTI systems.

#![allow(clippy::missing_errors_doc, clippy::many_single_char_names)]

use std::fmt;

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

pub mod dare;
pub mod gramian;
pub mod lqr;
pub mod observer;
pub mod pole;

#[derive(Debug, Clone, PartialEq)]
pub enum ControlError {
    EmptyMatrix,
    DimensionMismatch,
    InvalidInput(String),
    ConvergenceFailed,
    SingularSystem,
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlError::EmptyMatrix => write!(f, "matrix cannot be empty"),
            ControlError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
            ControlError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            ControlError::ConvergenceFailed => write!(f, "control iteration did not converge"),
            ControlError::SingularSystem => write!(f, "singular control system"),
        }
    }
}

impl std::error::Error for ControlError {}

impl IntoNabledError for ControlError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            ControlError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            ControlError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            ControlError::InvalidInput(message) => NabledError::InvalidInput(message),
            ControlError::ConvergenceFailed => NabledError::ConvergenceFailed,
            ControlError::SingularSystem => NabledError::SingularMatrix,
        }
    }
}

#[cfg(test)]
mod tests {
    use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

    use super::*;

    #[test]
    fn control_errors_display_and_map_to_shared_taxonomy() {
        assert_eq!(ControlError::EmptyMatrix.to_string(), "matrix cannot be empty");
        assert_eq!(
            ControlError::DimensionMismatch.to_string(),
            "input dimensions are incompatible"
        );
        assert_eq!(
            ControlError::InvalidInput("bad poles".to_string()).to_string(),
            "invalid input: bad poles"
        );
        assert_eq!(
            ControlError::ConvergenceFailed.to_string(),
            "control iteration did not converge"
        );
        assert_eq!(ControlError::SingularSystem.to_string(), "singular control system");

        assert!(matches!(
            ControlError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            ControlError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            ControlError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            ControlError::ConvergenceFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            ControlError::SingularSystem.into_nabled_error(),
            NabledError::SingularMatrix
        ));
    }
}
