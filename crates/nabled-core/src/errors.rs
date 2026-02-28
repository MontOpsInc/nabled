//! Shared error types for ndarray-native kernels.

use thiserror::Error;

/// Common shape and dimensional validation errors.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    /// Matrix or vector input is empty.
    #[error("input cannot be empty")]
    EmptyInput,
    /// Matrix is expected to be square.
    #[error("matrix must be square")]
    NotSquare,
    /// Matrix and vector dimensions do not match.
    #[error("dimension mismatch")]
    DimensionMismatch,
}
