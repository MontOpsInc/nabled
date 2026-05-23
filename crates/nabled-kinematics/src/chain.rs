//! Kinematic chain specification (DH parameters).

use nabled_core::scalar::NabledReal;
use ndarray::Array1;

use crate::error::KinematicsError;

/// Denavit-Hartenberg convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DhConvention {
    Standard,
    Modified,
}

/// Joint motion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointType {
    Revolute,
    Prismatic,
}

/// Serial chain DH specification.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainSpec<T> {
    pub convention:   DhConvention,
    pub joint_types:  Vec<JointType>,
    pub a:            Array1<T>,
    pub alpha:        Array1<T>,
    pub d:            Array1<T>,
    pub theta_offset: Array1<T>,
}

impl<T: NabledReal> ChainSpec<T> {
    /// Build a chain from per-joint DH parameter vectors.
    ///
    /// # Errors
    /// Returns an error when parameter lengths mismatch.
    pub fn from_dh(
        convention: DhConvention,
        joint_types: Vec<JointType>,
        a: Array1<T>,
        alpha: Array1<T>,
        d: Array1<T>,
        theta_offset: Array1<T>,
    ) -> Result<Self, KinematicsError> {
        let n = joint_types.len();
        if a.len() != n || alpha.len() != n || d.len() != n || theta_offset.len() != n {
            return Err(KinematicsError::DimensionMismatch);
        }
        if n == 0 {
            return Err(KinematicsError::EmptyChain);
        }
        Ok(Self { convention, joint_types, a, alpha, d, theta_offset })
    }

    /// Number of joints / DOF.
    #[must_use]
    pub fn num_joints(&self) -> usize { self.joint_types.len() }

    /// Validate chain parameters.
    ///
    /// # Errors
    /// Returns an error when the chain is empty or malformed.
    pub fn validate(&self) -> Result<(), KinematicsError> {
        if self.joint_types.is_empty() {
            return Err(KinematicsError::EmptyChain);
        }
        let n = self.num_joints();
        if self.a.len() != n
            || self.alpha.len() != n
            || self.d.len() != n
            || self.theta_offset.len() != n
        {
            return Err(KinematicsError::DimensionMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;

    #[test]
    fn from_dh_validates_lengths() {
        let chain = ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Revolute, JointType::Revolute],
            arr1(&[1.0_f64, 1.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        assert_eq!(chain.num_joints(), 2);
    }
}
