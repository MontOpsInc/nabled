//! Core ndarray-native types and shared utilities for nabled crates.

pub mod errors;
pub mod prelude {
    pub use ndarray::{
        Array, Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2,
    };
    pub use num_complex::{Complex32, Complex64};
}
pub mod validation;
