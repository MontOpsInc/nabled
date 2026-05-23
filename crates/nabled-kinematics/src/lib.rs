//! Robot kinematics: forward kinematics, Jacobians, and inverse kinematics.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::needless_range_loop
)]

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

pub mod chain;
pub mod error;
pub mod fk;
pub mod ik;
pub mod jacobian;
pub mod tree;

pub use chain::{ChainSpec, DhConvention, JointLimits, JointType};
pub use error::KinematicsError;
pub use ik::{
    IkConfig, IkResult, IkWorkspace, inverse_kinematics_dls, inverse_kinematics_dls_into,
    inverse_kinematics_dls_with_limits, inverse_kinematics_tree_dls,
    inverse_kinematics_tree_dls_with_limits, pose_error, pose_error_into,
};

impl IntoNabledError for KinematicsError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            KinematicsError::EmptyChain => NabledError::Shape(ShapeError::EmptyInput),
            KinematicsError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            KinematicsError::InvalidInput(message) => NabledError::InvalidInput(message),
            KinematicsError::ConvergenceFailed => NabledError::ConvergenceFailed,
            KinematicsError::NumericalInstability => NabledError::NumericalInstability,
            KinematicsError::JointLimitViolation(joint) => {
                NabledError::InvalidInput(format!("joint limit violated at joint {joint}"))
            }
        }
    }
}
