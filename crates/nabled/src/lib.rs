//! # nabled
//!
//! `nabled` is an ndarray-native numerical library focused on production-grade
//! linear algebra and ML-oriented primitives.
//!
//! ## Crate Layout
//!
//! 1. [`core`] for shared error taxonomy, validation, and ndarray prelude exports.
//! 2. [`linalg`] for linear algebra and decomposition domains (including `geometry` and `signal`).
//! 3. [`ml`] for ML-oriented numerical routines.
//! 4. [`kinematics`] for chain specification, FK, Jacobian, and IK.
//! 5. [`model`] for robot model representation and URDF ingestion.
//! 6. [`dynamics`] for rigid-body dynamics algorithms.
//! 7. [`control`] for LTI control synthesis (DARE/LQR and related tools).
//! 8. [`sensor`] for sensor fusion (Kalman/EKF and related models).
//!
//! ## Feature Flags
//!
//! 1. `blas`: enables `ndarray/blas` in lower crates.
//! 2. `openblas-system`: enables provider-backed `LAPACK` paths via system `OpenBLAS`.
//! 3. `openblas-static`: enables provider-backed `LAPACK` paths via statically linked `OpenBLAS`.
//! 4. `netlib-system`: enables provider-backed `LAPACK` paths via system `Netlib` `LAPACK`.
//! 5. `netlib-static`: enables provider-backed `LAPACK` paths via statically linked `Netlib`
//!    `LAPACK`.
//! 6. `magma-system`: enables NVIDIA MAGMA provider-backed decomposition paths.
//! 7. `accelerator-rayon`: enables parallel CPU kernels where implemented.
//! 8. `accelerator-wgpu`: enables WGPU-backed kernel paths where implemented.
//! 9. `signal`: enables `nabled-linalg::signal` FFT-backed routines.
//! 10. `arrow`: enables facade-only Arrow/ndarray interop adapters backed by `ndarrow`.
//!
//! ## Execution Semantics
//!
//! 1. `Provider`: decomposition implementation source (internal vs selected LAPACK provider).
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

/// Kinematics algorithms for Physical AI workloads.
pub mod kinematics {
    pub use nabled_kinematics::*;
}

/// Robot model representation for Physical AI workloads.
pub mod model {
    pub use nabled_model::*;
}

/// Rigid-body dynamics for Physical AI workloads.
pub mod dynamics {
    pub use nabled_dynamics::*;
}

/// Control algorithms for Physical AI workloads.
pub mod control {
    pub use nabled_control::*;
}

/// Sensor fusion for Physical AI workloads.
pub mod sensor {
    pub use nabled_sensor::*;
}

/// Optional Arrow/ndarray interop adapters that delegate into ndarray-native `nabled` domains.
///
/// This module exists only behind feature `arrow`. Core numerical domains remain ndarray-native;
/// Arrow knowledge is isolated to this facade-level integration surface.
#[cfg(feature = "arrow")]
pub mod arrow;

/// Re-exported Arrow/ndarray bridge used by `nabled::arrow`.
///
/// This keeps downstream Arrow consumers on the same `ndarrow` contract version as `nabled`
/// without requiring a second explicit dependency.
#[cfg(feature = "arrow")]
pub use ndarrow;

/// Common ndarray and complex-number prelude exports.
pub mod prelude {
    pub use crate::core::prelude::*;
}
