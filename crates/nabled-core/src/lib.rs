//! Core ndarray-native types and shared utilities for nabled crates.

pub mod errors;
pub mod prelude {
    pub use ndarray::{
        Array, Array1, Array2, Array3, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1,
        ArrayViewMut2, ArrayViewMut3,
    };
    pub use num_complex::{Complex32, Complex64};
}
pub mod validation;
