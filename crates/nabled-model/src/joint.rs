//! Joint types and limits.

use crate::ModelError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointType {
    Revolute,
    Prismatic,
    Fixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointAxis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointLimits<T> {
    pub lower:    T,
    pub upper:    T,
    pub velocity: T,
    pub effort:   T,
}

/// Validate joint limits ordering.
pub fn validate_limits<T: PartialOrd>(limits: &JointLimits<T>) -> Result<(), ModelError> {
    if limits.lower > limits.upper {
        return Err(ModelError::InvalidInput("lower limit exceeds upper limit".to_string()));
    }
    Ok(())
}
