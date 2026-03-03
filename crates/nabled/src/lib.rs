//! # nabled
//!
//! `nabled` is an ndarray-native numerical library focused on production-grade
//! linear algebra and ML-oriented primitives.
//!
//! ## Crate Layout
//!
//! 1. [`core`] for shared error taxonomy, validation, and ndarray prelude exports.
//! 2. [`linalg`] for linear algebra and decomposition domains.
//! 3. [`ml`] for ML-oriented numerical routines.
//!
//! ## Feature Flags
//!
//! 1. `blas`: enables `ndarray/blas` in lower crates.
//! 2. `openblas-system`: enables provider-backed LAPACK paths via `OpenBLAS`.
//! 3. `accelerator-rayon`: enables parallel CPU kernels where implemented.
//! 4. `accelerator-wgpu`: enables WGPU-backed kernel paths where implemented.
//!
//! ## Execution Semantics
//!
//! 1. `Provider`: decomposition implementation source (internal vs OpenBLAS-backed paths).
//! 2. `Backend`: primitive-kernel execution target (CPU/WGPU).
//! 3. `Kernel`: operation-family backend contract (`matmat`, `matvec`, sparse ops, tensor ops).
//!
//! ## Quick Start
//!
//! ```rust
//! use ndarray::arr2;
//! use nabled::linalg::svd;
//!
//! let a = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
//! let decomposition = svd::decompose(&a)?;
//! assert_eq!(decomposition.singular_values.len(), 2);
//! # Ok::<(), nabled::linalg::svd::SVDError>(())
//! ```
//!
//! ## Optional Provider Build
//!
//! ```text
//! cargo test -p nabled --features openblas-system
//! ```
//!
//! ## Optional Accelerator Build
//!
//! ```text
//! cargo test -p nabled --features accelerator-wgpu
//! ```

/// Shared core types, error taxonomy, and validation primitives.
pub mod core {
    pub use nabled_core::{errors, prelude, validation};
}

/// Ndarray-native linear algebra domains.
pub mod linalg {
    pub use nabled_linalg::*;
}

/// ML-oriented numerical domains built on ndarray-native primitives.
pub mod ml {
    pub use nabled_ml::*;
}

/// Common ndarray and complex-number prelude exports.
pub mod prelude {
    pub use crate::core::prelude::*;
}
