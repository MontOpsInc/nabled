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
//! 4. [`embeddings`] for embedding retrieval compute (normalize, score, rerank, kNN, compress).
//! 5. [`kinematics`] for chain specification, FK, Jacobian, and IK.
//! 6. [`model`] for robot model representation and URDF ingestion.
//! 7. [`dynamics`] for rigid-body dynamics algorithms.
//! 8. [`control`] for LTI control synthesis (DARE/LQR and related tools).
//! 9. [`sensor`] for sensor fusion (Kalman/EKF and related models).
//! 10. [`sim`] for cross-crate Physical AI orchestration (simulation, IK trajectories, control
//!     loops, estimation pipelines).
//!
//! ## Feature Flags
//!
//! Per-domain opt-in features gate which sub-crate facades are re-exported. The
//! default is `["linalg"]`; enable additional features as needed. See
//! `docs/FEATURE_MATRIX.md` for the full matrix.
//!
//! Domain features:
//!
//! 1. `linalg`: re-exports [`crate::linalg`] (`nabled-linalg`). Default.
//! 2. `geometry`: re-exports [`crate::linalg::geometry`] (implied by `linalg`).
//! 3. `signal`: enables `nabled-linalg::signal` FFT-backed routines and propagates to `nabled-sim`
//!    when enabled.
//! 4. `ml`: re-exports [`crate::ml`] (`nabled-ml`).
//! 5. `embeddings`: re-exports [`crate::embeddings`] (`nabled-embeddings`); implies `linalg` + `ml`.
//! 6. `model`: re-exports [`crate::model`] (`nabled-model`).
//! 7. `kinematics`: re-exports [`crate::kinematics`] (implies `model`).
//! 8. `dynamics`: re-exports [`crate::dynamics`] (implies `kinematics` + `model`).
//! 9. `control`: re-exports [`crate::control`] (`nabled-control`).
//! 10. `sensor`: re-exports [`crate::sensor`] (`nabled-sensor`).
//! 11. `sim`: re-exports [`crate::sim`] orchestration (implies `kinematics + dynamics + control +
//!     sensor + ml + model + geometry`).
//! 12. `physical-ai`: umbrella enabling every Physical AI domain feature.
//!
//! BLAS/LAPACK provider features:
//!
//! 1. `blas`: enables `ndarray/blas` in lower crates.
//! 2. `openblas-system`/`openblas-static`/`netlib-system`/`netlib-static`: provider-backed `LAPACK`
//!    paths.
//! 3. `magma-system`: NVIDIA MAGMA provider-backed decomposition paths.
//! 4. `accelerator-rayon`: enables parallel CPU kernels where implemented.
//! 5. `accelerator-wgpu`: enables WGPU-backed kernel paths where implemented.
//! 6. `arrow`: facade-only Arrow/ndarray interop adapters backed by `ndarrow`.
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
#[cfg(feature = "linalg")]
pub mod linalg {
    pub use nabled_linalg::*;
}

/// ML-oriented numerical domains built on ndarray-native primitives.
#[cfg(feature = "ml")]
pub mod ml {
    pub use nabled_ml::*;
}

/// Embedding retrieval compute: normalize, score, rerank, exact kNN, and PCA compression.
#[cfg(feature = "embeddings")]
pub mod embeddings {
    pub use nabled_embeddings::*;
}

/// Kinematics algorithms for Physical AI workloads.
#[cfg(feature = "kinematics")]
pub mod kinematics {
    pub use nabled_kinematics::*;
}

/// Robot model representation for Physical AI workloads.
#[cfg(feature = "model")]
pub mod model {
    pub use nabled_model::*;
}

/// Rigid-body dynamics for Physical AI workloads.
#[cfg(feature = "dynamics")]
pub mod dynamics {
    pub use nabled_dynamics::*;
}

/// Control algorithms for Physical AI workloads.
#[cfg(feature = "control")]
pub mod control {
    pub use nabled_control::*;
}

/// Sensor fusion for Physical AI workloads.
#[cfg(feature = "sensor")]
pub mod sensor {
    pub use nabled_sensor::*;
}

/// Physical AI orchestration (simulation, manipulation, control, estimation pipelines).
#[cfg(feature = "sim")]
pub mod sim {
    pub use nabled_sim::*;
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
