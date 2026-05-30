//! Rigid-body geometry: quaternions, SO(3), SE(3), and twists.

#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::many_single_char_names)]

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2};
use thiserror::Error;

pub mod quat;
pub mod se3;
pub mod so3;
pub mod twist;

/// Unit quaternion stored as `[w, x, y, z]` (scalar-first / Hamilton convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat<T> {
    /// Scalar part.
    pub w: T,
    /// Vector part x.
    pub x: T,
    /// Vector part y.
    pub y: T,
    /// Vector part z.
    pub z: T,
}

/// Axis-angle representation with axis unit vector and angle in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisAngle<T> {
    /// Rotation axis.
    pub axis:  [T; 3],
    /// Rotation angle in radians.
    pub angle: T,
}

/// 3×3 rotation matrix wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotation3<T> {
    /// Row-major rotation matrix.
    pub matrix: Array2<T>,
}

/// Rigid transform: rotation + translation.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform3<T> {
    /// Orientation.
    pub rotation:    Rotation3<T>,
    /// Translation vector.
    pub translation: Array1<T>,
}

/// Errors for geometry operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GeometryError {
    #[error("input dimensions are incompatible")]
    DimensionMismatch,
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("zero norm quaternion or vector")]
    ZeroNorm,
    #[error("numerical instability detected")]
    NumericalInstability,
}

pub(crate) fn scalar_two<T: NabledReal>() -> T { T::from_f64(2.0).unwrap_or(T::one() + T::one()) }

pub(crate) fn scalar_half<T: NabledReal>() -> T { T::from_f64(0.5).unwrap_or(T::zero()) }
