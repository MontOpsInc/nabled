//! Error types for Arrow linear algebra operations

use crate::jacobian::JacobianError;
use crate::matrix_functions::MatrixFunctionError;
use crate::qr::QRError;
use crate::svd::SVDError;
use std::fmt;

/// Errors that can occur when converting between Arrow and matrix types
#[derive(Debug, Clone)]
pub enum ArrowConversionError {
    /// Invalid matrix/array shape
    InvalidShape(String),
    /// Unsupported Arrow data type
    UnsupportedType(String),
    /// Array or batch is empty
    EmptyArray,
    /// Null values not supported in linear algebra
    NullValues,
    /// Dimension mismatch between arrays
    DimensionMismatch(String),
}

impl fmt::Display for ArrowConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrowConversionError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
            ArrowConversionError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
            ArrowConversionError::EmptyArray => write!(f, "Array is empty"),
            ArrowConversionError::NullValues => write!(f, "Null values not supported"),
            ArrowConversionError::DimensionMismatch(msg) => {
                write!(f, "Dimension mismatch: {}", msg)
            }
        }
    }
}

impl std::error::Error for ArrowConversionError {}

/// Unified error type for Arrow linear algebra operations
#[derive(Debug)]
pub enum ArrowLinalgError {
    /// Conversion error
    Conversion(ArrowConversionError),
    /// SVD computation error
    SVD(SVDError),
    /// QR computation error
    QR(QRError),
    /// Matrix function error
    MatrixFunction(MatrixFunctionError),
    /// Jacobian computation error
    Jacobian(JacobianError),
}

impl fmt::Display for ArrowLinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrowLinalgError::Conversion(e) => write!(f, "Conversion error: {}", e),
            ArrowLinalgError::SVD(e) => write!(f, "SVD error: {}", e),
            ArrowLinalgError::QR(e) => write!(f, "QR error: {}", e),
            ArrowLinalgError::MatrixFunction(e) => write!(f, "Matrix function error: {}", e),
            ArrowLinalgError::Jacobian(e) => write!(f, "Jacobian error: {}", e),
        }
    }
}

impl std::error::Error for ArrowLinalgError {}

impl From<ArrowConversionError> for ArrowLinalgError {
    fn from(e: ArrowConversionError) -> Self {
        ArrowLinalgError::Conversion(e)
    }
}

impl From<SVDError> for ArrowLinalgError {
    fn from(e: SVDError) -> Self {
        ArrowLinalgError::SVD(e)
    }
}

impl From<QRError> for ArrowLinalgError {
    fn from(e: QRError) -> Self {
        ArrowLinalgError::QR(e)
    }
}

impl From<MatrixFunctionError> for ArrowLinalgError {
    fn from(e: MatrixFunctionError) -> Self {
        ArrowLinalgError::MatrixFunction(e)
    }
}

impl From<JacobianError> for ArrowLinalgError {
    fn from(e: JacobianError) -> Self {
        ArrowLinalgError::Jacobian(e)
    }
}
