//! Shared scalar trait bounds.

use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};

/// Canonical real scalar bound for ndarray-native public APIs.
///
/// This trait intentionally maps to the first-class real scalar set supported
/// by nabled: `f32` and `f64`.
pub trait NabledReal:
    Float
    + FromPrimitive
    + ScalarOperand
    + Copy
    + Send
    + Sync
    + std::fmt::Debug
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + 'static
{
}

impl NabledReal for f32 {}
impl NabledReal for f64 {}
