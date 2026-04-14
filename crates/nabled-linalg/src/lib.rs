//! Ndarray-native linear algebra domains for the `nabled` workspace.
//!
//! `nabled-linalg` provides decomposition routines, dense/sparse kernels,
//! vector/tensor primitives, and matrix-function algorithms over ndarray data.
//!
//! # Feature Flags
//!
//! 1. `blas`: enables BLAS acceleration through `ndarray/blas`.
//! 2. `openblas-system`: enables provider-backed `LAPACK` paths via system `OpenBLAS`.
//! 3. `openblas-static`: enables provider-backed `LAPACK` paths via statically linked `OpenBLAS`.
//! 4. `netlib-system`: enables provider-backed `LAPACK` paths via system `Netlib` `LAPACK`.
//! 5. `netlib-static`: enables provider-backed `LAPACK` paths via statically linked `Netlib`
//!    `LAPACK`.
//! 6. `magma-system`: enables CUDA-backed MAGMA provider paths for supported decomposition domains.
//! 7. `accelerator-rayon`: enables selected parallel CPU kernels.
//! 8. `accelerator-wgpu`: enables bounded GPU (`f32`/`f64`) kernel paths.
//!
//! # Execution Model
//!
//! 1. `Provider`: decomposition implementation source (internal or selected LAPACK provider).
//! 2. `Backend`: operation-kernel execution target (`CpuBackend`, `GpuBackend`).
//! 3. `Kernel`: operation-family contract (for example matmat, sparse matvec, tensor contraction).
//!
//! Provider selection and backend selection are orthogonal, compile-time concerns.
//! Public APIs remain ndarray-native and backend/provider agnostic.
//!
//! # Example
//!
//! ```rust
//! use ndarray::arr2;
//! use nabled_linalg::svd;
//!
//! let a = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
//! let decomposition = svd::decompose(&a)?;
//! assert_eq!(decomposition.singular_values.len(), 2);
//! # Ok::<(), nabled_linalg::svd::SVDError>(())
//! ```

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

use crate::accelerator::backends::AcceleratorError;
use crate::cholesky::CholeskyError;
use crate::eigen::EigenError;
use crate::lu::LUError;
use crate::matrix::MatrixError;
use crate::matrix_functions::MatrixFunctionError;
use crate::orthogonalization::OrthogonalizationError;
use crate::polar::PolarError;
use crate::qr::QRError;
use crate::schur::SchurError;
use crate::sparse::SparseError;
use crate::svd::SVDError;
use crate::sylvester::SylvesterError;
use crate::tensor::TensorError;
use crate::triangular::TriangularError;
use crate::vector::VectorError;

pub mod accelerator;
pub mod batched;
mod internal;
#[cfg(all(test, feature = "magma-system"))]
mod magma_verification;
mod provider;

pub mod cholesky;
pub mod eigen;
pub mod lu;
pub mod matrix;
pub mod matrix_functions;
pub mod orthogonalization;
pub mod polar;
pub mod qr;
pub mod schur;
pub mod sparse;
pub mod svd;
pub mod sylvester;
pub mod tensor;
pub mod triangular;
pub mod vector;

impl IntoNabledError for AcceleratorError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            AcceleratorError::UnsupportedBackend(kind) => {
                NabledError::Other(format!("backend {kind:?} is not currently available"))
            }
            AcceleratorError::InvalidChunkSize => {
                NabledError::InvalidInput("chunk size must be greater than zero".to_string())
            }
            AcceleratorError::DimensionMismatch => {
                NabledError::Shape(ShapeError::DimensionMismatch)
            }
            AcceleratorError::FeatureNotEnabled => {
                NabledError::Other("requested accelerator feature is not enabled".to_string())
            }
            AcceleratorError::DeviceUnavailable => {
                NabledError::Other("no suitable GPU device is available".to_string())
            }
            AcceleratorError::KernelExecutionFailed => {
                NabledError::Other("GPU kernel execution failed".to_string())
            }
        }
    }
}

impl IntoNabledError for CholeskyError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            CholeskyError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            CholeskyError::NotSquare => NabledError::Shape(ShapeError::NotSquare),
            CholeskyError::NotPositiveDefinite => NabledError::NotPositiveDefinite,
            CholeskyError::InvalidInput(message) => NabledError::InvalidInput(message),
            CholeskyError::NumericalInstability => NabledError::NumericalInstability,
        }
    }
}

impl IntoNabledError for EigenError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            EigenError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            EigenError::NotSquare => NabledError::Shape(ShapeError::NotSquare),
            EigenError::NotSymmetric => NabledError::NotSymmetric,
            EigenError::InvalidDimensions => NabledError::Shape(ShapeError::DimensionMismatch),
            EigenError::NotPositiveDefinite => NabledError::NotPositiveDefinite,
            EigenError::ConvergenceFailed => NabledError::ConvergenceFailed,
            EigenError::NumericalInstability => NabledError::NumericalInstability,
        }
    }
}

impl IntoNabledError for LUError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            LUError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            LUError::NotSquare => NabledError::Shape(ShapeError::NotSquare),
            LUError::SingularMatrix => NabledError::SingularMatrix,
            LUError::ConvergenceFailed => NabledError::ConvergenceFailed,
            LUError::InvalidInput(message) => NabledError::InvalidInput(message),
            LUError::NumericalInstability => NabledError::NumericalInstability,
        }
    }
}

impl IntoNabledError for MatrixError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            MatrixError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            MatrixError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
        }
    }
}

impl IntoNabledError for MatrixFunctionError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            MatrixFunctionError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            MatrixFunctionError::NotSquare => NabledError::Shape(ShapeError::NotSquare),
            MatrixFunctionError::NotSymmetric => NabledError::NotSymmetric,
            MatrixFunctionError::NotPositiveDefinite => NabledError::NotPositiveDefinite,
            MatrixFunctionError::ConvergenceFailed => NabledError::ConvergenceFailed,
            MatrixFunctionError::InvalidInput(message) => NabledError::InvalidInput(message),
        }
    }
}

impl IntoNabledError for OrthogonalizationError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            OrthogonalizationError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            OrthogonalizationError::InvalidInput(message) => NabledError::InvalidInput(message),
            OrthogonalizationError::NumericalInstability => NabledError::NumericalInstability,
        }
    }
}

impl IntoNabledError for PolarError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            PolarError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            PolarError::NotSquare => NabledError::Shape(ShapeError::NotSquare),
            PolarError::DecompositionFailed => NabledError::ConvergenceFailed,
            PolarError::NumericalInstability => NabledError::NumericalInstability,
        }
    }
}

impl IntoNabledError for QRError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            QRError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            QRError::SingularMatrix => NabledError::SingularMatrix,
            QRError::ConvergenceFailed => NabledError::ConvergenceFailed,
            QRError::InvalidDimensions(message) | QRError::InvalidInput(message) => {
                NabledError::InvalidInput(message)
            }
            QRError::NumericalInstability => NabledError::NumericalInstability,
        }
    }
}

impl IntoNabledError for SchurError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            SchurError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            SchurError::NotSquare => NabledError::Shape(ShapeError::NotSquare),
            SchurError::ConvergenceFailed => NabledError::ConvergenceFailed,
            SchurError::NumericalInstability => NabledError::NumericalInstability,
            SchurError::InvalidInput(message) => NabledError::InvalidInput(message),
        }
    }
}

impl IntoNabledError for SparseError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            SparseError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            SparseError::InvalidStructure => {
                NabledError::InvalidInput("invalid sparse structure".to_string())
            }
            SparseError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            SparseError::SingularMatrix => NabledError::SingularMatrix,
            SparseError::MaxIterationsExceeded => NabledError::ConvergenceFailed,
        }
    }
}

impl IntoNabledError for SVDError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            SVDError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            SVDError::NotSquare => NabledError::Shape(ShapeError::NotSquare),
            SVDError::ConvergenceFailed => NabledError::ConvergenceFailed,
            SVDError::InvalidInput(message) => NabledError::InvalidInput(message),
        }
    }
}

impl IntoNabledError for SylvesterError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            SylvesterError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            SylvesterError::NotSquare => NabledError::Shape(ShapeError::NotSquare),
            SylvesterError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            SylvesterError::SingularSystem => NabledError::SingularMatrix,
            SylvesterError::InvalidInput(message) => NabledError::InvalidInput(message),
        }
    }
}

impl IntoNabledError for TriangularError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            TriangularError::Shape(error) => NabledError::Shape(error),
            TriangularError::Singular => NabledError::SingularMatrix,
        }
    }
}

impl IntoNabledError for TensorError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            TensorError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            TensorError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
        }
    }
}

impl IntoNabledError for VectorError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            VectorError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            VectorError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            VectorError::ZeroNorm => NabledError::InvalidInput(
                "cosine similarity is undefined for zero-norm vectors".to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

    use super::*;
    use crate::accelerator::backends::BackendKind;

    #[test]
    fn linalg_errors_map_to_shared_taxonomy() {
        assert!(matches!(
            CholeskyError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            CholeskyError::NotSquare.into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));
        assert!(matches!(
            CholeskyError::NotPositiveDefinite.into_nabled_error(),
            NabledError::NotPositiveDefinite
        ));
        assert!(matches!(
            CholeskyError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));
        assert!(matches!(
            CholeskyError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));

        assert!(matches!(EigenError::NotSymmetric.into_nabled_error(), NabledError::NotSymmetric));
        assert!(matches!(
            EigenError::InvalidDimensions.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));

        assert!(matches!(LUError::SingularMatrix.into_nabled_error(), NabledError::SingularMatrix));
        assert!(matches!(
            MatrixError::EmptyInput.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            MatrixError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));

        assert!(matches!(
            MatrixFunctionError::ConvergenceFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));

        assert!(matches!(
            OrthogonalizationError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));

        assert!(matches!(
            PolarError::DecompositionFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));

        assert!(matches!(QRError::SingularMatrix.into_nabled_error(), NabledError::SingularMatrix));
        assert!(matches!(
            QRError::InvalidDimensions("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            QRError::InvalidInput("y".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
    }

    #[test]
    fn linalg_errors_map_to_shared_taxonomy_additional_domains() {
        assert!(matches!(
            SchurError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));

        assert!(matches!(
            SparseError::InvalidStructure.into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            SparseError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            SparseError::SingularMatrix.into_nabled_error(),
            NabledError::SingularMatrix
        ));
        assert!(matches!(
            SparseError::MaxIterationsExceeded.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));

        assert!(matches!(
            SVDError::ConvergenceFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));

        assert!(matches!(
            SylvesterError::SingularSystem.into_nabled_error(),
            NabledError::SingularMatrix
        ));

        assert!(matches!(
            TriangularError::Singular.into_nabled_error(),
            NabledError::SingularMatrix
        ));

        assert!(matches!(VectorError::ZeroNorm.into_nabled_error(), NabledError::InvalidInput(_)));
        assert!(matches!(
            TensorError::EmptyInput.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            TensorError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            AcceleratorError::UnsupportedBackend(BackendKind::Gpu).into_nabled_error(),
            NabledError::Other(_)
        ));
        assert!(matches!(
            AcceleratorError::InvalidChunkSize.into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            AcceleratorError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            AcceleratorError::FeatureNotEnabled.into_nabled_error(),
            NabledError::Other(_)
        ));
    }

    #[test]
    fn linalg_error_mapping_covers_remaining_variants_part_1() {
        assert!(matches!(
            EigenError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            EigenError::NotSquare.into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));
        assert!(matches!(
            EigenError::NotPositiveDefinite.into_nabled_error(),
            NabledError::NotPositiveDefinite
        ));
        assert!(matches!(
            EigenError::ConvergenceFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            EigenError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));

        assert!(matches!(
            LUError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            LUError::NotSquare.into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));
        assert!(matches!(
            LUError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            LUError::ConvergenceFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            LUError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));

        assert!(matches!(
            QRError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            QRError::ConvergenceFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            QRError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));
    }

    #[test]
    fn linalg_error_mapping_covers_remaining_variants_part_2() {
        assert!(matches!(
            MatrixFunctionError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            MatrixFunctionError::NotSquare.into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));
        assert!(matches!(
            MatrixFunctionError::NotSymmetric.into_nabled_error(),
            NabledError::NotSymmetric
        ));
        assert!(matches!(
            MatrixFunctionError::NotPositiveDefinite.into_nabled_error(),
            NabledError::NotPositiveDefinite
        ));
        assert!(matches!(
            MatrixFunctionError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));

        assert!(matches!(
            OrthogonalizationError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));

        assert!(matches!(
            PolarError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            PolarError::NotSquare.into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));
        assert!(matches!(
            PolarError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));

        assert!(matches!(
            SchurError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            SchurError::NotSquare.into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));
        assert!(matches!(
            SchurError::ConvergenceFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            SchurError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));

        assert!(matches!(
            SparseError::EmptyInput.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
    }

    #[test]
    fn linalg_error_mapping_covers_remaining_variants_part_3() {
        assert!(matches!(
            SVDError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            SVDError::NotSquare.into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));
        assert!(matches!(
            SVDError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));

        assert!(matches!(
            SylvesterError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            SylvesterError::NotSquare.into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));
        assert!(matches!(
            SylvesterError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            SylvesterError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));

        assert!(matches!(
            TriangularError::Shape(ShapeError::NotSquare).into_nabled_error(),
            NabledError::Shape(ShapeError::NotSquare)
        ));

        assert!(matches!(
            VectorError::EmptyInput.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            VectorError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
    }
}
