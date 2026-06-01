//! Error taxonomy for embedding retrieval compute.

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};
use nabled_linalg::matrix::MatrixError;
use nabled_linalg::vector::VectorError;
use nabled_ml::pca::PCAError;
use thiserror::Error;

/// Errors produced by `nabled-embeddings` retrieval compute.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EmbeddingError {
    /// Input matrix or vector is empty.
    #[error("input cannot be empty")]
    EmptyInput,
    /// Query and corpus feature dimensions are incompatible, or an output buffer is mis-sized.
    #[error("input dimensions are incompatible")]
    DimensionMismatch,
    /// A metric that requires non-zero norms encountered a zero-norm row.
    #[error("operation is undefined for zero-norm vectors")]
    ZeroNorm,
    /// Caller-provided argument is invalid (for example a non-positive component count).
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// The underlying matrix decomposition failed to converge.
    #[error("decomposition failed")]
    DecompositionFailed,
}

impl From<VectorError> for EmbeddingError {
    fn from(value: VectorError) -> Self {
        match value {
            VectorError::EmptyInput => EmbeddingError::EmptyInput,
            VectorError::DimensionMismatch => EmbeddingError::DimensionMismatch,
            VectorError::ZeroNorm => EmbeddingError::ZeroNorm,
        }
    }
}

impl From<MatrixError> for EmbeddingError {
    fn from(value: MatrixError) -> Self {
        match value {
            MatrixError::EmptyInput => EmbeddingError::EmptyInput,
            MatrixError::DimensionMismatch => EmbeddingError::DimensionMismatch,
        }
    }
}

impl From<PCAError> for EmbeddingError {
    fn from(value: PCAError) -> Self {
        match value {
            PCAError::EmptyMatrix => EmbeddingError::EmptyInput,
            PCAError::InvalidInput(message) => EmbeddingError::InvalidInput(message),
            PCAError::DecompositionFailed => EmbeddingError::DecompositionFailed,
        }
    }
}

impl IntoNabledError for EmbeddingError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            EmbeddingError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            EmbeddingError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            EmbeddingError::ZeroNorm => NabledError::InvalidInput(
                "operation is undefined for zero-norm vectors".to_string(),
            ),
            EmbeddingError::InvalidInput(message) => NabledError::InvalidInput(message),
            EmbeddingError::DecompositionFailed => NabledError::ConvergenceFailed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_error_maps_to_embedding_error() {
        assert_eq!(EmbeddingError::from(VectorError::EmptyInput), EmbeddingError::EmptyInput);
        assert_eq!(
            EmbeddingError::from(VectorError::DimensionMismatch),
            EmbeddingError::DimensionMismatch
        );
        assert_eq!(EmbeddingError::from(VectorError::ZeroNorm), EmbeddingError::ZeroNorm);
    }

    #[test]
    fn matrix_error_maps_to_embedding_error() {
        assert_eq!(EmbeddingError::from(MatrixError::EmptyInput), EmbeddingError::EmptyInput);
        assert_eq!(
            EmbeddingError::from(MatrixError::DimensionMismatch),
            EmbeddingError::DimensionMismatch
        );
    }

    #[test]
    fn pca_error_maps_to_embedding_error() {
        assert_eq!(EmbeddingError::from(PCAError::EmptyMatrix), EmbeddingError::EmptyInput);
        assert_eq!(
            EmbeddingError::from(PCAError::InvalidInput("x".to_string())),
            EmbeddingError::InvalidInput("x".to_string())
        );
        assert_eq!(
            EmbeddingError::from(PCAError::DecompositionFailed),
            EmbeddingError::DecompositionFailed
        );
    }

    #[test]
    fn embedding_error_maps_to_shared_taxonomy() {
        assert!(matches!(
            EmbeddingError::EmptyInput.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            EmbeddingError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            EmbeddingError::ZeroNorm.into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            EmbeddingError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            EmbeddingError::DecompositionFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
    }

    #[test]
    fn embedding_error_displays_messages() {
        assert_eq!(EmbeddingError::EmptyInput.to_string(), "input cannot be empty");
        assert_eq!(
            EmbeddingError::DimensionMismatch.to_string(),
            "input dimensions are incompatible"
        );
        assert_eq!(
            EmbeddingError::ZeroNorm.to_string(),
            "operation is undefined for zero-norm vectors"
        );
        assert_eq!(EmbeddingError::InvalidInput("oops".to_string()).to_string(), "invalid input: oops");
        assert_eq!(EmbeddingError::DecompositionFailed.to_string(), "decomposition failed");
    }
}
