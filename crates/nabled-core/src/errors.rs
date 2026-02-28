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

/// Cross-domain top-level error taxonomy for `nabled` crates.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum NabledError {
    /// Shape or dimensional validation failure.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// Matrix/vector is singular.
    #[error("matrix is singular")]
    SingularMatrix,
    /// Matrix symmetry requirement failed.
    #[error("matrix must be symmetric")]
    NotSymmetric,
    /// Matrix positive-definiteness requirement failed.
    #[error("matrix must be positive definite")]
    NotPositiveDefinite,
    /// Numerical method failed to converge.
    #[error("algorithm failed to converge")]
    ConvergenceFailed,
    /// Non-finite or unstable numerical state observed.
    #[error("numerical instability detected")]
    NumericalInstability,
    /// Invalid user input.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Domain-specific catch-all.
    #[error("{0}")]
    Other(String),
}

/// Trait for domain errors that can be normalized into [`NabledError`].
pub trait IntoNabledError {
    /// Convert domain-specific error into shared taxonomy.
    fn into_nabled_error(self) -> NabledError;
}
