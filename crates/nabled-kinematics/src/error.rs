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
    /// Joint index that violates configured limits.
    JointLimitViolation(usize),
}

impl fmt::Display for KinematicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KinematicsError::EmptyChain => write!(f, "kinematic chain cannot be empty"),
            KinematicsError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
            KinematicsError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            KinematicsError::ConvergenceFailed => write!(f, "IK iteration did not converge"),
            KinematicsError::NumericalInstability => write!(f, "numerical instability detected"),
            KinematicsError::JointLimitViolation(joint) => {
                write!(f, "joint limit violated at joint {joint}")
            }
        }
    }
}

impl std::error::Error for KinematicsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_stable() {
        assert_eq!(
            KinematicsError::EmptyChain.to_string(),
            "kinematic chain cannot be empty"
        );
        assert_eq!(
            KinematicsError::DimensionMismatch.to_string(),
            "input dimensions are incompatible"
        );
        assert_eq!(
            KinematicsError::InvalidInput("bad link".into()).to_string(),
            "invalid input: bad link"
        );
        assert_eq!(
            KinematicsError::ConvergenceFailed.to_string(),
            "IK iteration did not converge"
        );
        assert_eq!(
            KinematicsError::NumericalInstability.to_string(),
            "numerical instability detected"
        );
        assert_eq!(
            KinematicsError::JointLimitViolation(2).to_string(),
            "joint limit violated at joint 2"
        );
    }

    #[test]
    fn variants_are_clone_and_eq() {
        let err = KinematicsError::JointLimitViolation(1);
        assert_eq!(err.clone(), err);
    }
}
