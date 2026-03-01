//! Ndarray-native linear algebra domain crate.

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};
pub use nabled_core::prelude;

mod internal;

pub mod cholesky;
pub mod eigen;
pub mod lu;
pub mod matrix_functions;
pub mod orthogonalization;
pub mod polar;
pub mod qr;
pub mod schur;
pub mod sparse;
pub mod svd;
pub mod sylvester;
pub mod triangular;
pub mod vector;

pub use cholesky::{CholeskyError, NdarrayCholeskyResult};
pub use eigen::{EigenError, NdarrayEigenResult, NdarrayGeneralizedEigenResult};
pub use lu::{LUError, LogDetResult, NdarrayLUResult};
pub use matrix_functions::{MatrixFunctionError, MatrixFunctionWorkspace};
pub use orthogonalization::OrthogonalizationError;
pub use polar::{NdarrayComplexPolarResult, NdarrayPolarResult, PolarError};
pub use qr::{QRConfig, QRError, QRResult};
pub use schur::{NdarraySchurResult, SchurError, SchurWorkspace};
pub use sparse::{CooMatrix, CsrMatrix, SparseError};
pub use svd::{NdarrayComplexSVD, NdarraySVD, PseudoInverseConfig, SVDError};
pub use sylvester::{SylvesterError, SylvesterWorkspace};
pub use triangular::TriangularError;
pub use vector::{PairwiseCosineWorkspace, VectorError};

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
            LUError::InvalidInput(message) => NabledError::InvalidInput(message),
            LUError::NumericalInstability => NabledError::NumericalInstability,
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
