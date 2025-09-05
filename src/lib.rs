//! # Rust Linear Algebra Library
//! 
//! Advanced linear algebra functions built on top of nalgebra and ndarray.
//! This library provides enhanced implementations of common linear algebra operations
//! with focus on performance and numerical stability.

pub mod svd;
pub mod utils;

/// Re-exports for convenience
pub use svd::*;
