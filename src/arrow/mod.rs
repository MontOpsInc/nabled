//! # Arrow Linear Algebra
//!
//! This module provides Apache Arrow as the primary interface for linear algebra
//! operations. All functions accept RecordBatch or Float64Array as input and return
//! Arrow types as output.

pub mod conversions;
pub mod error;
pub mod jacobian;
pub mod matrix_functions;
pub mod qr;
pub mod svd;

pub use conversions::*;
pub use error::{ArrowConversionError, ArrowLinalgError};
pub use jacobian::*;
pub use matrix_functions::*;
// SVD and QR have overlapping names (reconstruct_matrix, condition_number) - use arrow::svd::* and arrow::qr::* for full API
