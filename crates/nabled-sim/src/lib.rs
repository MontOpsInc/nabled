//! Physical AI orchestration — compose-down only.
//!
//! Domain algorithms remain in their owner crates; this crate wires validated context,
//! simulation steps, batch IK, closed-loop control, and estimation pipelines.

#![allow(clippy::missing_errors_doc)]

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

pub mod context;
pub mod control_loop;
pub mod estimation;
pub mod manipulation;
pub mod pipeline;
pub mod sim;

#[derive(Debug, Clone, PartialEq)]
pub enum SimError {
    DimensionMismatch,
    InvalidInput(String),
    Model(nabled_model::ModelError),
    Kinematics(nabled_kinematics::KinematicsError),
    Dynamics(nabled_dynamics::DynamicsError),
    Control(nabled_control::ControlError),
    Sensor(nabled_sensor::SensorError),
    Stats(nabled_ml::stats::StatsError),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
            SimError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            SimError::Model(err) => write!(f, "{err}"),
            SimError::Kinematics(err) => write!(f, "{err}"),
            SimError::Dynamics(err) => write!(f, "{err}"),
            SimError::Control(err) => write!(f, "{err}"),
            SimError::Sensor(err) => write!(f, "{err}"),
            SimError::Stats(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for SimError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SimError::Model(err) => Some(err),
            SimError::Kinematics(err) => Some(err),
            SimError::Dynamics(err) => Some(err),
            SimError::Control(err) => Some(err),
            SimError::Sensor(err) => Some(err),
            SimError::Stats(err) => Some(err),
            SimError::DimensionMismatch | SimError::InvalidInput(_) => None,
        }
    }
}

impl IntoNabledError for SimError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            SimError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            SimError::InvalidInput(message) => NabledError::InvalidInput(message),
            SimError::Model(err) => err.into_nabled_error(),
            SimError::Kinematics(err) => err.into_nabled_error(),
            SimError::Dynamics(err) => err.into_nabled_error(),
            SimError::Control(err) => err.into_nabled_error(),
            SimError::Sensor(err) => err.into_nabled_error(),
            SimError::Stats(err) => err.into_nabled_error(),
        }
    }
}

impl From<nabled_model::ModelError> for SimError {
    fn from(value: nabled_model::ModelError) -> Self { Self::Model(value) }
}

impl From<nabled_kinematics::KinematicsError> for SimError {
    fn from(value: nabled_kinematics::KinematicsError) -> Self { Self::Kinematics(value) }
}

impl From<nabled_dynamics::DynamicsError> for SimError {
    fn from(value: nabled_dynamics::DynamicsError) -> Self { Self::Dynamics(value) }
}

impl From<nabled_control::ControlError> for SimError {
    fn from(value: nabled_control::ControlError) -> Self { Self::Control(value) }
}

impl From<nabled_sensor::SensorError> for SimError {
    fn from(value: nabled_sensor::SensorError) -> Self { Self::Sensor(value) }
}

impl From<nabled_ml::stats::StatsError> for SimError {
    fn from(value: nabled_ml::stats::StatsError) -> Self { Self::Stats(value) }
}
