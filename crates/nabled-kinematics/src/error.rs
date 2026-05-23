//! Kinematics domain errors.

use std::fmt;

/// Error type for kinematics operations.
#[derive(Debug, Clone, PartialEq)]
pub enum KinematicsError {
    EmptyChain,
    DimensionMismatch,
    InvalidInput(String),
    ConvergenceFailed,
    NumericalInstability,
}

impl fmt::Display for KinematicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KinematicsError::EmptyChain => write!(f, "kinematic chain cannot be empty"),
            KinematicsError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
            KinematicsError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            KinematicsError::ConvergenceFailed => write!(f, "IK iteration did not converge"),
            KinematicsError::NumericalInstability => write!(f, "numerical instability detected"),
        }
    }
}

impl std::error::Error for KinematicsError {}
