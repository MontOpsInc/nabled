use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};
use nabled_linalg::{
    CholeskyError, EigenError, LUError, MatrixFunctionError, OrthogonalizationError, PolarError,
    QRError, SVDError, SchurError, SparseError, SylvesterError, TriangularError, VectorError,
};

#[test]
fn maps_cholesky_eigen_lu_matrix_functions() {
    assert_eq!(
        CholeskyError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        CholeskyError::NotSquare.into_nabled_error(),
        NabledError::Shape(ShapeError::NotSquare)
    );
    assert_eq!(
        CholeskyError::NotPositiveDefinite.into_nabled_error(),
        NabledError::NotPositiveDefinite
    );
    assert_eq!(
        CholeskyError::InvalidInput("bad".to_string()).into_nabled_error(),
        NabledError::InvalidInput("bad".to_string())
    );
    assert_eq!(
        CholeskyError::NumericalInstability.into_nabled_error(),
        NabledError::NumericalInstability
    );

    assert_eq!(
        EigenError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        EigenError::NotSquare.into_nabled_error(),
        NabledError::Shape(ShapeError::NotSquare)
    );
    assert_eq!(EigenError::NotSymmetric.into_nabled_error(), NabledError::NotSymmetric);
    assert_eq!(
        EigenError::InvalidDimensions.into_nabled_error(),
        NabledError::Shape(ShapeError::DimensionMismatch)
    );
    assert_eq!(
        EigenError::NotPositiveDefinite.into_nabled_error(),
        NabledError::NotPositiveDefinite
    );
    assert_eq!(EigenError::ConvergenceFailed.into_nabled_error(), NabledError::ConvergenceFailed);
    assert_eq!(
        EigenError::NumericalInstability.into_nabled_error(),
        NabledError::NumericalInstability
    );

    assert_eq!(
        LUError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(LUError::NotSquare.into_nabled_error(), NabledError::Shape(ShapeError::NotSquare));
    assert_eq!(LUError::SingularMatrix.into_nabled_error(), NabledError::SingularMatrix);
    assert_eq!(
        LUError::InvalidInput("lu".to_string()).into_nabled_error(),
        NabledError::InvalidInput("lu".to_string())
    );
    assert_eq!(
        LUError::NumericalInstability.into_nabled_error(),
        NabledError::NumericalInstability
    );

    assert_eq!(
        MatrixFunctionError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        MatrixFunctionError::NotSquare.into_nabled_error(),
        NabledError::Shape(ShapeError::NotSquare)
    );
    assert_eq!(MatrixFunctionError::NotSymmetric.into_nabled_error(), NabledError::NotSymmetric);
    assert_eq!(
        MatrixFunctionError::NotPositiveDefinite.into_nabled_error(),
        NabledError::NotPositiveDefinite
    );
    assert_eq!(
        MatrixFunctionError::ConvergenceFailed.into_nabled_error(),
        NabledError::ConvergenceFailed
    );
    assert_eq!(
        MatrixFunctionError::InvalidInput("mf".to_string()).into_nabled_error(),
        NabledError::InvalidInput("mf".to_string())
    );
}

#[test]
fn maps_orthogonalization_polar_qr_schur() {
    assert_eq!(
        OrthogonalizationError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        OrthogonalizationError::NumericalInstability.into_nabled_error(),
        NabledError::NumericalInstability
    );

    assert_eq!(
        PolarError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        PolarError::NotSquare.into_nabled_error(),
        NabledError::Shape(ShapeError::NotSquare)
    );
    assert_eq!(PolarError::DecompositionFailed.into_nabled_error(), NabledError::ConvergenceFailed);
    assert_eq!(
        PolarError::NumericalInstability.into_nabled_error(),
        NabledError::NumericalInstability
    );

    assert_eq!(
        QRError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(QRError::SingularMatrix.into_nabled_error(), NabledError::SingularMatrix);
    assert_eq!(QRError::ConvergenceFailed.into_nabled_error(), NabledError::ConvergenceFailed);
    assert_eq!(
        QRError::InvalidDimensions("dim".to_string()).into_nabled_error(),
        NabledError::InvalidInput("dim".to_string())
    );
    assert_eq!(
        QRError::NumericalInstability.into_nabled_error(),
        NabledError::NumericalInstability
    );
    assert_eq!(
        QRError::InvalidInput("qr".to_string()).into_nabled_error(),
        NabledError::InvalidInput("qr".to_string())
    );

    assert_eq!(
        SchurError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        SchurError::NotSquare.into_nabled_error(),
        NabledError::Shape(ShapeError::NotSquare)
    );
    assert_eq!(SchurError::ConvergenceFailed.into_nabled_error(), NabledError::ConvergenceFailed);
    assert_eq!(
        SchurError::NumericalInstability.into_nabled_error(),
        NabledError::NumericalInstability
    );
    assert_eq!(
        SchurError::InvalidInput("schur".to_string()).into_nabled_error(),
        NabledError::InvalidInput("schur".to_string())
    );
}

#[test]
fn maps_sparse_svd_sylvester_triangular() {
    assert_eq!(
        SparseError::EmptyInput.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        SparseError::InvalidStructure.into_nabled_error(),
        NabledError::InvalidInput("invalid sparse structure".to_string())
    );
    assert_eq!(
        SparseError::DimensionMismatch.into_nabled_error(),
        NabledError::Shape(ShapeError::DimensionMismatch)
    );
    assert_eq!(SparseError::SingularMatrix.into_nabled_error(), NabledError::SingularMatrix);
    assert_eq!(
        SparseError::MaxIterationsExceeded.into_nabled_error(),
        NabledError::ConvergenceFailed
    );

    assert_eq!(
        SVDError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(SVDError::NotSquare.into_nabled_error(), NabledError::Shape(ShapeError::NotSquare));
    assert_eq!(SVDError::ConvergenceFailed.into_nabled_error(), NabledError::ConvergenceFailed);
    assert_eq!(
        SVDError::InvalidInput("svd".to_string()).into_nabled_error(),
        NabledError::InvalidInput("svd".to_string())
    );

    assert_eq!(
        SylvesterError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        SylvesterError::NotSquare.into_nabled_error(),
        NabledError::Shape(ShapeError::NotSquare)
    );
    assert_eq!(
        SylvesterError::DimensionMismatch.into_nabled_error(),
        NabledError::Shape(ShapeError::DimensionMismatch)
    );
    assert_eq!(SylvesterError::SingularSystem.into_nabled_error(), NabledError::SingularMatrix);

    assert_eq!(
        TriangularError::Shape(ShapeError::NotSquare).into_nabled_error(),
        NabledError::Shape(ShapeError::NotSquare)
    );
    assert_eq!(TriangularError::Singular.into_nabled_error(), NabledError::SingularMatrix);
}

#[test]
fn maps_vector_errors() {
    assert_eq!(
        VectorError::EmptyInput.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        VectorError::DimensionMismatch.into_nabled_error(),
        NabledError::Shape(ShapeError::DimensionMismatch)
    );
    assert_eq!(
        VectorError::ZeroNorm.into_nabled_error(),
        NabledError::InvalidInput(
            "cosine similarity is undefined for zero-norm vectors".to_string()
        )
    );
}
