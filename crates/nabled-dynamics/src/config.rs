//! Dynamics configuration (gravity and defaults).

use nabled_core::scalar::NabledReal;

/// Configuration for inverse/forward dynamics routines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicsConfig<T> {
    /// World-frame gravity vector `[gx, gy, gz]`.
    pub gravity: [T; 3],
}

impl<T: NabledReal> Default for DynamicsConfig<T> {
    fn default() -> Self {
        Self { gravity: [T::zero(), T::from_f64(-9.81).unwrap_or(T::zero()), T::zero()] }
    }
}
