//! Dynamics configuration (gravity and defaults).

use nabled_core::scalar::NabledReal;

/// Forward dynamics implementation selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForwardDynamicsMethod {
    /// Articulated Body Algorithm (O(n), default).
    #[default]
    Aba,
    /// Composite rigid-body mass matrix + LU solve (cross-check / fallback).
    CrbaLu,
}

/// Configuration for inverse/forward dynamics routines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicsConfig<T> {
    /// World-frame gravity vector `[gx, gy, gz]`.
    pub gravity:          [T; 3],
    /// Forward dynamics algorithm (ABA default; CRBA+LU fallback).
    pub forward_dynamics: ForwardDynamicsMethod,
}

impl<T: NabledReal> Default for DynamicsConfig<T> {
    fn default() -> Self {
        Self {
            gravity:          [T::zero(), T::from_f64(-9.81).unwrap_or(T::zero()), T::zero()],
            forward_dynamics: ForwardDynamicsMethod::default(),
        }
    }
}
