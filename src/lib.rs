//! # Rust Linear Algebra Library
//! 
//! Advanced linear algebra functions built on top of nalgebra and ndarray.
//! This library provides enhanced implementations of common linear algebra operations
//! with focus on performance and numerical stability.
//!
//! ## Quick Start
//!
//! Add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rust-linalg = "0.3.0"
//! nalgebra = "0.32"
//! ndarray = "0.15"
//! ```
//!
//! ## Basic Usage
//!
//! ### SVD with Nalgebra
//! ```rust
//! use rust_linalg::svd::nalgebra_svd;
//! use nalgebra::DMatrix;
//!
//! let matrix = DMatrix::from_row_slice(3, 3, &[
//!     1.0, 2.0, 3.0,
//!     4.0, 5.0, 6.0,
//!     7.0, 8.0, 9.0
//! ]);
//!
//! let svd = nalgebra_svd::compute_svd(&matrix)?;
//! println!("Singular values: {:?}", svd.singular_values);
//! println!("Condition number: {}", nalgebra_svd::condition_number(&svd));
//! println!("Matrix rank: {}", nalgebra_svd::matrix_rank(&svd, None));
//! # Ok::<(), rust_linalg::svd::SVDError>(())
//! ```
//!
//! ### SVD with Ndarray
//! ```rust
//! use rust_linalg::svd::ndarray_svd;
//! use ndarray::Array2;
//!
//! let matrix = Array2::from_shape_vec((3, 3), vec![
//!     1.0, 2.0, 3.0,
//!     4.0, 5.0, 6.0,
//!     7.0, 8.0, 9.0
//! ])?;
//!
//! let svd = ndarray_svd::compute_svd(&matrix)?;
//! println!("Singular values: {:?}", svd.singular_values);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Truncated SVD for Dimensionality Reduction
//! ```rust
//! use rust_linalg::svd::nalgebra_svd;
//! use nalgebra::DMatrix;
//!
//! let matrix = DMatrix::from_row_slice(4, 4, &[
//!     1.0, 0.0, 0.0, 0.0,
//!     0.0, 2.0, 0.0, 0.0,
//!     0.0, 0.0, 3.0, 0.0,
//!     0.0, 0.0, 0.0, 4.0
//! ]);
//!
//! // Keep only the 2 largest singular values
//! let truncated_svd = nalgebra_svd::compute_truncated_svd(&matrix, 2)?;
//! println!("Truncated singular values: {:?}", truncated_svd.singular_values);
//! # Ok::<(), rust_linalg::svd::SVDError>(())
//! ```
//!
//! ### Matrix Reconstruction
//! ```rust
//! use rust_linalg::svd::nalgebra_svd;
//! use nalgebra::DMatrix;
//!
//! let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
//! let svd = nalgebra_svd::compute_svd(&matrix)?;
//! let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);
//! 
//! // reconstructed should be approximately equal to the original matrix
//! println!("Original: {}", matrix);
//! println!("Reconstructed: {}", reconstructed);
//! # Ok::<(), rust_linalg::svd::SVDError>(())
//! ```
//!
//! ### Matrix Exponential
//! ```rust
//! use rust_linalg::matrix_functions::nalgebra_matrix_functions;
//! use nalgebra::DMatrix;
//!
//! let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
//! let exp_matrix = nalgebra_matrix_functions::matrix_exp_eigen(&matrix)?;
//! println!("exp(A): {}", exp_matrix);
//! # Ok::<(), rust_linalg::matrix_functions::MatrixFunctionError>(())
//! ```
//!
//! ### Matrix Logarithm
//! ```rust
//! use rust_linalg::matrix_functions::nalgebra_matrix_functions;
//! use nalgebra::DMatrix;
//!
//! let matrix = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
//! let log_matrix = nalgebra_matrix_functions::matrix_log_eigen(&matrix)?;
//! println!("log(A): {}", log_matrix);
//! # Ok::<(), rust_linalg::matrix_functions::MatrixFunctionError>(())
//! ```
//!
//! ### Matrix Power
//! ```rust
//! use rust_linalg::matrix_functions::nalgebra_matrix_functions;
//! use nalgebra::DMatrix;
//!
//! let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 4.0]);
//! let power_matrix = nalgebra_matrix_functions::matrix_power(&matrix, 0.5)?;
//! println!("A^0.5: {}", power_matrix);
//! # Ok::<(), rust_linalg::matrix_functions::MatrixFunctionError>(())
//! ```
//!
//! ### Jacobian Computation
//! ```rust
//! use rust_linalg::jacobian::nalgebra_jacobian;
//! use nalgebra::DVector;
//!
//! let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
//!     Ok(DVector::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
//! };
//!
//! let x = DVector::from_vec(vec![2.0, 3.0]);
//! let jacobian = nalgebra_jacobian::numerical_jacobian(&f, &x, &Default::default())?;
//! println!("Jacobian: {}", jacobian);
//! # Ok::<(), rust_linalg::jacobian::JacobianError>(())
//! ```
//!
//! ### Gradient Computation
//! ```rust
//! use rust_linalg::jacobian::nalgebra_jacobian;
//! use nalgebra::DVector;
//!
//! let f = |x: &DVector<f64>| -> Result<f64, String> {
//!     Ok(x[0] * x[0] + x[1] * x[1])
//! };
//!
//! let x = DVector::from_vec(vec![3.0, 4.0]);
//! let gradient = nalgebra_jacobian::numerical_gradient(&f, &x, &Default::default())?;
//! println!("Gradient: {}", gradient);
//! # Ok::<(), rust_linalg::jacobian::JacobianError>(())
//! ```
//!
//! ### Complex Derivatives
//! ```rust
//! use rust_linalg::jacobian::complex_jacobian;
//! use nalgebra::DVector;
//! use num_complex::Complex;
//! 
//! let f = |x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> {
//!     let mut result = DVector::zeros(x.len());
//!     for i in 0..x.len() {
//!         result[i] = x[i] * x[i]; // f(z) = z²
//!     }
//!     Ok(result)
//! };
//! 
//! let x = DVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
//! let jacobian = complex_jacobian::numerical_jacobian(&f, &x, &Default::default())?;
//! println!("Complex Jacobian: {}", jacobian);
//! # Ok::<(), rust_linalg::jacobian::JacobianError>(())
//! ```
//!
//! ### QR Decomposition
//! ```rust
//! use rust_linalg::qr::nalgebra_qr;
//! use nalgebra::DMatrix;
//!
//! let matrix = DMatrix::from_row_slice(3, 3, &[
//!     1.0, 2.0, 3.0,
//!     4.0, 5.0, 6.0,
//!     7.0, 8.0, 10.0
//! ]);
//!
//! let qr = nalgebra_qr::compute_qr(&matrix, &Default::default())?;
//! println!("Q: {}", qr.q);
//! println!("R: {}", qr.r);
//! println!("Rank: {}", qr.rank);
//! # Ok::<(), rust_linalg::qr::QRError>(())
//! ```
//!
//! ### Least Squares Solving
//! ```rust
//! use rust_linalg::qr::nalgebra_qr;
//! use nalgebra::{DMatrix, DVector};
//!
//! let a = DMatrix::from_row_slice(4, 2, &[
//!     1.0, 1.0,
//!     1.0, 2.0,
//!     1.0, 3.0,
//!     1.0, 4.0
//! ]);
//! let b = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
//!
//! let x = nalgebra_qr::solve_least_squares(&a, &b, &Default::default())?;
//! println!("Solution: {}", x);
//! # Ok::<(), rust_linalg::qr::QRError>(())
//! ```

#[cfg(feature = "arrow")]
pub mod arrow;
pub mod cholesky;
pub mod eigen;
pub mod lu;
pub mod pca;
pub mod regression;
pub mod stats;
pub mod svd;
pub mod matrix_functions;
pub mod jacobian;
pub mod qr;
pub mod utils;

/// Re-exports for convenience
pub use svd::*;
pub use matrix_functions::*;
pub use jacobian::*;
pub use jacobian::{JacobianConfig, JacobianError};
pub use qr::*;
pub use qr::{QRConfig, QRError, QRResult};
pub use lu::{LUError, NalgebraLUResult, NdarrayLUResult};
pub use cholesky::{CholeskyError, NalgebraCholeskyResult, NdarrayCholeskyResult};
pub use eigen::{EigenError, NalgebraEigenResult, NdarrayEigenResult};
pub use stats::StatsError;
pub use pca::{PCAError, NalgebraPCAResult, NdarrayPCAResult};
pub use regression::{RegressionError, NalgebraRegressionResult, NdarrayRegressionResult};
