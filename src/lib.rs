//! # rust-linalg
//!
//! A linear algebra library that provides numerically focused routines on top of
//! [`nalgebra`](https://crates.io/crates/nalgebra) and [`ndarray`](https://crates.io/crates/ndarray).
//!
//! ## Backends
//!
//! Most domains expose both backends through module pairs:
//!
//! - `nalgebra_*` modules operate on `nalgebra::DMatrix`/`DVector`
//! - `ndarray_*` modules operate on `ndarray::Array2`/`Array1`
//!
//! ## Core Areas
//!
//! - Decompositions: SVD, QR, LU, Cholesky, Schur, Polar, Eigen
//! - Solvers: least squares, triangular solve, iterative methods, Sylvester/Lyapunov
//! - Higher-level methods: PCA, linear regression
//! - Numerical differentiation: Jacobian, gradient, Hessian
//! - Matrix functions: exp/log/power
//! - Statistics: covariance/correlation and centering helpers
//!
//! ## Example
//!
//! ```rust
//! use nalgebra::DMatrix;
//! use rust_linalg::svd::nalgebra_svd;
//!
//! let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
//! let svd = nalgebra_svd::compute_svd(&matrix)?;
//!
//! assert_eq!(svd.singular_values.len(), 2);
//! # Ok::<(), rust_linalg::svd::SVDError>(())
//! ```

pub mod cholesky;
pub mod eigen;
pub(crate) mod interop;
pub mod iterative;
pub mod jacobian;
pub mod lu;
pub mod matrix_functions;
pub mod orthogonalization;
pub mod pca;
pub mod polar;
pub mod qr;
pub mod regression;
pub mod schur;
pub mod stats;
pub mod svd;
pub mod sylvester;
pub mod triangular;

pub use cholesky::{CholeskyError, NalgebraCholeskyResult, NdarrayCholeskyResult};
pub use eigen::{
    EigenError, NalgebraEigenResult, NalgebraGeneralizedEigenResult, NdarrayEigenResult,
    NdarrayGeneralizedEigenResult,
};
pub use iterative::{IterativeConfig, IterativeError};
pub use jacobian::{JacobianConfig, JacobianError, *};
pub use lu::{LUError, LogDetResult, NalgebraLUResult, NdarrayLUResult};
pub use matrix_functions::*;
pub use orthogonalization::OrthogonalizationError;
pub use pca::{NalgebraPCAResult, NdarrayPCAResult, PCAError};
pub use polar::{NalgebraPolarResult, NdarrayPolarResult, PolarError};
pub use qr::{QRConfig, QRError, QRResult, *};
pub use regression::{NalgebraRegressionResult, NdarrayRegressionResult, RegressionError};
pub use schur::{NalgebraSchurResult, NdarraySchurResult, SchurError};
pub use stats::StatsError;
/// Re-exports for convenience
pub use svd::*;
pub use sylvester::SylvesterError;
pub use triangular::TriangularError;
