//! Core shared types and utilities for the `nabled` workspace.
//!
//! `nabled-core` is the common substrate used by `nabled-linalg` and
//! `nabled-ml`. It intentionally stays lightweight and focused on:
//!
//! 1. Shared error taxonomy (`ShapeError`, `NabledError`, `IntoNabledError`).
//! 2. Reusable shape/validation helpers over ndarray inputs.
//! 3. Prelude exports for ndarray and complex number types.
//!
//! # Main Modules
//!
//! 1. [`errors`]: shared error contracts used across workspace crates.
//! 2. [`validation`]: reusable ndarray validation helpers.
//! 3. [`prelude`]: common array/scalar and complex-number type exports.
//! 4. [`scalar`]: shared real-scalar trait bounds (`f32`/`f64`).
//!
//! # Example
//!
//! ```rust
//! use ndarray::{arr1, arr2};
//! use nabled_core::validation::validate_square_system;
//!
//! let a = arr2(&[[2.0_f64, 1.0], [1.0, 2.0]]);
//! let b = arr1(&[1.0_f64, 1.0]);
//! validate_square_system(&a, &b)?;
//! # Ok::<(), nabled_core::errors::ShapeError>(())
//! ```

pub mod errors;
pub mod prelude {
    pub use ndarray::{
        Array, Array1, Array2, Array3, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1,
        ArrayViewMut2, ArrayViewMut3,
    };
    pub use num_complex::{Complex32, Complex64};

    pub use crate::scalar::NabledReal;
}
pub mod scalar;
pub mod validation;
